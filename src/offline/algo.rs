//! Offline diarization orchestrator.

use std::sync::Arc;

use crate::{
  cluster::centroid::SP_ALIVE_THRESHOLD,
  embed::EMBEDDING_DIM,
  ops::spill::SpillOptions,
  pipeline::{AssignEmbeddingsInput, ChunkAssignment, assign_embeddings},
  plda::{PldaTransform, RawEmbedding},
  reconstruct::{ReconstructInput, RttmSpan, SlidingWindow, reconstruct, try_discrete_to_spans},
};
use nalgebra::DVector;

/// Diarizer error type (re-exports the pipeline error since that's
/// where most failures surface in offline mode).
#[derive(Debug, thiserror::Error)]
pub enum Error {
  /// Input shape / configuration is invalid — see `ShapeError`.
  #[error("offline: {0}")]
  Shape(#[from] ShapeError),
  /// Propagated from [`crate::pipeline::assign_embeddings`].
  #[error("offline: pipeline: {0}")]
  Pipeline(#[from] crate::pipeline::Error),
  /// Propagated from [`crate::reconstruct::reconstruct`] / RTTM
  /// emission.
  #[error("offline: reconstruct: {0}")]
  Reconstruct(#[from] crate::reconstruct::Error),
  /// Propagated from [`crate::plda::PldaTransform`].
  #[error("offline: plda: {0}")]
  Plda(#[from] crate::plda::Error),
  /// Propagated from segmentation ONNX inference inside the
  /// `OwnedDiarizationPipeline` audio entrypoint.
  #[cfg(feature = "ort")]
  #[cfg_attr(docsrs, doc(cfg(feature = "ort")))]
  #[error("offline: segment: {0}")]
  Segment(#[from] crate::segment::Error),
  /// Propagated from embedding ONNX inference inside the
  /// `OwnedDiarizationPipeline` audio entrypoint.
  #[cfg(feature = "ort")]
  #[cfg_attr(docsrs, doc(cfg(feature = "ort")))]
  #[error("offline: embed: {0}")]
  Embed(#[from] crate::embed::Error),
  /// Propagated from `aggregate::try_count_pyannote` when the count
  /// tensor cannot be computed (e.g. invalid `onset` configuration).
  /// This replaces a panic path through the infallible
  /// `count_pyannote` wrapper used by the audio entrypoint.
  #[error("offline: aggregate: {0}")]
  Aggregate(#[from] crate::aggregate::Error),
  /// Propagated from `crate::ops::spill::SpillBytesMut::zeros` when the
  /// per-call segmentation / embedding scratch buffers cannot be
  /// allocated (mmap failure on the spill backend, tempfile creation
  /// failure, size overflow). At multi-hour scale these buffers
  /// cross the 64 MiB default threshold and route through the
  /// file-backed mmap path; surfacing the failure here keeps a
  /// `Result`-returning API from OOM-aborting.
  #[error("offline: spill: {0}")]
  Spill(#[from] crate::ops::spill::SpillError),
}

/// Specific shape-violation reasons for [`Error::Shape`].
#[derive(Debug, thiserror::Error, Clone, Copy, PartialEq)]
pub enum ShapeError {
  #[error("num_chunks must be at least 1")]
  ZeroNumChunks,
  #[error("num_speakers must be at least 1")]
  ZeroNumSpeakers,
  #[error("num_frames_per_chunk must be at least 1")]
  ZeroNumFramesPerChunk,
  #[error("raw_embeddings size overflow")]
  RawEmbeddingsOverflow,
  #[error("raw_embeddings.len() must equal num_chunks * num_speakers * EMBEDDING_DIM")]
  RawEmbeddingsLenMismatch,
  #[error("segmentations size overflow")]
  SegmentationsOverflow,
  #[error("segmentations.len() must equal num_chunks * num_frames_per_chunk * num_speakers")]
  SegmentationsLenMismatch,
  #[error("samples is empty")]
  EmptySamples,
  #[error("step_samples must be > 0")]
  ZeroStepSamples,
  /// `step_samples` exceeds `WINDOW_SAMPLES`. The owned/streaming
  /// chunk planners use `start = c * step` and stop after
  /// `(samples.len() - win).div_ceil(step) + 1` chunks; with `step >
  /// win`, samples in `[win .. step)` per chunk are never segmented
  /// or embedded — silent data loss returning `Ok(_)` with missing
  /// speech. Reject at validation rather than letting it propagate.
  #[error("step_samples ({step}) must not exceed WINDOW_SAMPLES ({window})")]
  StepSamplesExceedsWindow { step: u32, window: u32 },
  /// `onset` is outside the documented `(0.0, 1.0]` range. Hard
  /// segmentations are 0/1; the per-frame mask `seg >= onset`
  /// degenerates: with `onset > 1.0` no frame is active (empty
  /// diarization), with `onset <= 0.0` even zero cells are active
  /// (corrupted frame masks, embeddings, and counts). NaN turns
  /// every comparison false and behaves like `onset > 1.0`.
  #[error("onset ({onset}) must be finite in (0.0, 1.0]")]
  OnsetOutOfRange { onset: f32 },
  /// `min_duration_off` is NaN/±inf or negative. RTTM span-merge
  /// reads this as a non-negative seconds quantity; `+inf` merges
  /// every same-cluster gap, `NaN` silently disables the merge
  /// (every comparison becomes false), and negative values are
  /// nonsensical. Catches serde-bypassed configs.
  #[error("min_duration_off ({value}) must be finite and >= 0")]
  MinDurationOffOutOfRange { value: f64 },
  /// `smoothing_epsilon` is `Some(NaN/±inf)` or `Some(< 0)`. The
  /// smoothing step compares activation differences against this
  /// epsilon; `Some(+inf)` collapses top-k onto stable index order,
  /// `Some(NaN)` makes every comparison false. `None` is the
  /// pyannote-argmax bit-exact path and is always valid.
  #[error("smoothing_epsilon ({value:?}) must be None or Some(finite >= 0)")]
  SmoothingEpsilonOutOfRange { value: Option<f32> },
}

// ── Memory budget for `diarize_offline` ───────────────────────────
//
// The matrices that scale with input length are now all spill-backed
// through [`crate::ops::spill::SpillBytesMut`], so multi-hour inputs
// no longer allocate hundreds of MB of contiguous heap:
//   * `embeddings`           — `(num_chunks * num_speakers, embed_dim)`
//     f64, row-major flat → `SpillBytes<f64>` (built below)
//   * `post_plda`            — `(num_train, plda_dim)` f64, row-major
//     flat → `SpillBytes<f64>` (built below). The pipeline transposes
//     into a column-major spill region internally for VBx's GEMM.
//   * `train_embeddings`     — `(num_train, embed_dim)` f64, row-major
//     flat → `SpillBytes<f64>` (built inside `assign_embeddings`)
//   * AHC pdist condensed    — `n*(n-1)/2` f64 → `SpillBytesMut`
//   * `discrete_diarization` — `(num_output_frames, num_alive)` f32 →
//     `SpillBytes<f32>` (built inside `reconstruct`)
//
// VBx internal working matrices (`rho`, `gamma`, `log_p`, `new_gamma`,
// `inv_l`, `alpha`, `rho_alpha_t`) remain heap-allocated `nalgebra::
// DMatrix` values. These are bounded by `pipeline::MAX_AHC_TRAIN` and
// `pipeline::MAX_QINIT_CELLS`: at the cap the working set is
// `O(num_train * plda_dim) + O(num_train * num_init)` ≤ ~50 MB, which
// is independent of input length and well below any sane heap budget.
// They sit on the EM hot path with iteration-level reads + writes and
// would lose 20-50× performance if backed by paged mmap, so spilling
// them is intentionally not done.
//
// `qinit` is also heap-allocated but gated by the same `MAX_QINIT_CELLS`
// check in `pipeline::algo` before VBx is invoked.

/// `const fn` predicate: `v` is finite and `>= 0` (f64). Used for
/// `min_duration_off`, a non-negative seconds quantity passed
/// unchanged into RTTM span post-processing. Hand-coded with `v == v`
/// (NaN check) and an `!= INFINITY` clause so it can be `const`
/// (`f64::is_finite` is not yet `const`).
#[inline]
pub(crate) const fn check_min_duration_off(v: f64) -> bool {
  #[allow(clippy::eq_op)] // intentional NaN check: NaN != NaN by IEEE 754.
  let not_nan = !(v != v);
  not_nan && v >= 0.0 && v != f64::INFINITY
}

/// `const fn` predicate: `v` is `None` or `Some(finite >= 0)` (f32).
/// Used for the optional smoothing epsilon; `None` disables smoothing
/// (bit-exact pyannote argmax) and is always valid.
#[inline]
pub(crate) const fn check_smoothing_epsilon(v: Option<f32>) -> bool {
  match v {
    None => true,
    Some(x) => {
      #[allow(clippy::eq_op)] // intentional NaN check: NaN != NaN by IEEE 754.
      let not_nan = !(x != x);
      not_nan && x >= 0.0 && x != f32::INFINITY
    }
  }
}

/// Inputs to [`diarize_offline`].
///
/// Caller has already produced segmentation + raw-embedding tensors
/// via their own ONNX inference. Tensors must follow the pyannote
/// community-1 layout.
pub struct OfflineInput<'a> {
  raw_embeddings: &'a [f32],
  num_chunks: usize,
  num_speakers: usize,
  segmentations: &'a [f64],
  num_frames_per_chunk: usize,
  count: &'a [u8],
  num_output_frames: usize,
  chunks_sw: SlidingWindow,
  frames_sw: SlidingWindow,
  plda: &'a PldaTransform,
  threshold: f64,
  fa: f64,
  fb: f64,
  max_iters: usize,
  min_duration_off: f64,
  smoothing_epsilon: Option<f32>,
  /// Spill backend configuration. [`diarize_offline`] forwards it to
  /// the inner [`AssignEmbeddingsInput`] / [`ReconstructInput`], so
  /// every transitive [`crate::ops::spill::SpillBytesMut::zeros`] reached
  /// from this call sees the same options. Defaults to
  /// [`SpillOptions::default`].
  spill_options: SpillOptions,
}

impl<'a> OfflineInput<'a> {
  /// Construct with `community-1` hyperparameter defaults
  /// (`threshold = 0.6`, `fa = 0.07`, `fb = 0.8`, `max_iters = 20`,
  /// `min_duration_off = 0.0`, `smoothing_epsilon = None`). Override
  /// individual hyperparameters via the `with_*` builders.
  ///
  /// Required data inputs:
  /// - `raw_embeddings`: pre-PLDA WeSpeaker raw embeddings, flattened
  ///   `[c][s][d]`. Length `num_chunks * num_speakers * EMBEDDING_DIM`.
  /// - `segmentations`: per-`(chunk, frame, speaker)` activity flattened
  ///   `[c][f][s]`. Length `num_chunks * num_frames_per_chunk * num_speakers`.
  /// - `count`: per-output-frame instantaneous speaker count.
  ///   Length `num_output_frames`.
  /// - `chunks_sw` / `frames_sw`: sliding-window timing.
  /// - `plda`: PLDA model.
  #[allow(clippy::too_many_arguments)]
  pub const fn new(
    raw_embeddings: &'a [f32],
    num_chunks: usize,
    num_speakers: usize,
    segmentations: &'a [f64],
    num_frames_per_chunk: usize,
    count: &'a [u8],
    num_output_frames: usize,
    chunks_sw: SlidingWindow,
    frames_sw: SlidingWindow,
    plda: &'a PldaTransform,
  ) -> Self {
    Self {
      raw_embeddings,
      num_chunks,
      num_speakers,
      segmentations,
      num_frames_per_chunk,
      count,
      num_output_frames,
      chunks_sw,
      frames_sw,
      plda,
      // Community-1 defaults.
      threshold: 0.6,
      fa: 0.07,
      fb: 0.8,
      max_iters: 20,
      min_duration_off: 0.0,
      smoothing_epsilon: None,
      spill_options: SpillOptions::new(),
    }
  }

  /// Set the AHC linkage threshold (builder).
  #[must_use]
  pub const fn with_threshold(mut self, threshold: f64) -> Self {
    self.threshold = threshold;
    self
  }

  /// Set the VBx Fa hyperparameter (builder).
  #[must_use]
  pub const fn with_fa(mut self, fa: f64) -> Self {
    self.fa = fa;
    self
  }

  /// Set the VBx Fb hyperparameter (builder).
  #[must_use]
  pub const fn with_fb(mut self, fb: f64) -> Self {
    self.fb = fb;
    self
  }

  /// Set the VBx max-iterations cap (builder).
  #[must_use]
  pub const fn with_max_iters(mut self, max_iters: usize) -> Self {
    self.max_iters = max_iters;
    self
  }

  /// Set the gap-merging threshold for span post-processing (builder).
  ///
  /// # Panics
  /// Panics if `min_duration_off` is NaN/±inf or negative. RTTM span-
  /// merge consumes this as a non-negative seconds quantity; `+inf`
  /// merges every same-cluster gap and `NaN` silently disables the
  /// merge (every comparison becomes false).
  #[must_use]
  pub const fn with_min_duration_off(mut self, min_duration_off: f64) -> Self {
    assert!(
      check_min_duration_off(min_duration_off),
      "min_duration_off must be finite and >= 0"
    );
    self.min_duration_off = min_duration_off;
    self
  }

  /// Set the temporal-smoothing epsilon for reconstruct (builder).
  /// `None` = bit-exact pyannote argmax. `Some(0.1)` recommended for
  /// `OwnedDiarizationPipeline`.
  ///
  /// # Panics
  /// Panics if `smoothing_epsilon` is `Some(NaN/±inf)` or `Some(< 0)`.
  /// `Some(+inf)` collapses top-k onto stable index order, `Some(NaN)`
  /// makes every smoothing comparison false.
  #[must_use]
  pub const fn with_smoothing_epsilon(mut self, smoothing_epsilon: Option<f32>) -> Self {
    assert!(
      check_smoothing_epsilon(smoothing_epsilon),
      "smoothing_epsilon must be None or Some(finite >= 0)"
    );
    self.smoothing_epsilon = smoothing_epsilon;
    self
  }

  /// Set the spill backend configuration (builder).
  ///
  /// Not `const fn`: `SpillOptions` has a non-const destructor
  /// (`Option<PathBuf>`).
  #[must_use]
  pub fn with_spill_options(mut self, spill_options: SpillOptions) -> Self {
    self.spill_options = spill_options;
    self
  }

  /// Pre-PLDA WeSpeaker raw embeddings.
  pub const fn raw_embeddings(&self) -> &'a [f32] {
    self.raw_embeddings
  }
  /// Number of chunks.
  pub const fn num_chunks(&self) -> usize {
    self.num_chunks
  }
  /// Speaker slots per chunk.
  pub const fn num_speakers(&self) -> usize {
    self.num_speakers
  }
  /// Per-`(chunk, frame, speaker)` segmentation activity.
  pub const fn segmentations(&self) -> &'a [f64] {
    self.segmentations
  }
  /// Frames per chunk.
  pub const fn num_frames_per_chunk(&self) -> usize {
    self.num_frames_per_chunk
  }
  /// Per-output-frame speaker count.
  pub const fn count(&self) -> &'a [u8] {
    self.count
  }
  /// Output-frame grid length.
  pub const fn num_output_frames(&self) -> usize {
    self.num_output_frames
  }
  /// Outer (chunk-level) sliding window.
  pub const fn chunks_sw(&self) -> SlidingWindow {
    self.chunks_sw
  }
  /// Inner (frame-level) sliding window.
  pub const fn frames_sw(&self) -> SlidingWindow {
    self.frames_sw
  }
  /// PLDA model.
  pub const fn plda(&self) -> &'a PldaTransform {
    self.plda
  }
  /// AHC linkage threshold.
  pub const fn threshold(&self) -> f64 {
    self.threshold
  }
  /// VBx Fa.
  pub const fn fa(&self) -> f64 {
    self.fa
  }
  /// VBx Fb.
  pub const fn fb(&self) -> f64 {
    self.fb
  }
  /// VBx max iterations.
  pub const fn max_iters(&self) -> usize {
    self.max_iters
  }
  /// Gap merging threshold for span post-processing.
  pub const fn min_duration_off(&self) -> f64 {
    self.min_duration_off
  }
  /// Optional smoothing epsilon for reconstruct.
  pub const fn smoothing_epsilon(&self) -> Option<f32> {
    self.smoothing_epsilon
  }
  /// Spill backend configuration forwarded into the inner
  /// [`AssignEmbeddingsInput`] / [`ReconstructInput`] by
  /// [`diarize_offline`].
  pub const fn spill_options(&self) -> &SpillOptions {
    &self.spill_options
  }
}

/// Output of [`diarize_offline`].
///
/// Owned slices are `Arc<[T]>` so multiple downstream consumers
/// (RTTM emission, metric computation, visualization, etc.) can share
/// the same buffer with cheap `Arc::clone` rather than re-allocating.
#[derive(Debug, Clone)]
pub struct OfflineOutput {
  hard_clusters: Arc<[ChunkAssignment]>,
  /// Spill-backed (heap-or-mmap), cheap-clone via the inner
  /// `Arc`. Cloning the `OfflineOutput` clones this without
  /// copying the underlying buffer; large multi-hour grids that
  /// crossed `SpillOptions::threshold_bytes` during reconstruction
  /// remain mmap-backed without extra memory pressure here.
  discrete_diarization: crate::ops::spill::SpillBytes<f32>,
  num_clusters: usize,
  spans: Arc<[RttmSpan]>,
}

impl OfflineOutput {
  /// Construct.
  pub fn new(
    hard_clusters: Arc<[ChunkAssignment]>,
    discrete_diarization: crate::ops::spill::SpillBytes<f32>,
    num_clusters: usize,
    spans: Arc<[RttmSpan]>,
  ) -> Self {
    Self {
      hard_clusters,
      discrete_diarization,
      num_clusters,
      spans,
    }
  }

  /// Cheap-clone handle to the per-chunk hard speaker assignment.
  /// Each row is `[i32; MAX_SPEAKER_SLOTS]` (= 3) with `-2` for
  /// unmatched slots. Length = `num_chunks`.
  pub fn hard_clusters(&self) -> Arc<[ChunkAssignment]> {
    Arc::clone(&self.hard_clusters)
  }

  /// Borrow the per-chunk hard speaker assignment without cloning the
  /// `Arc`.
  pub fn hard_clusters_slice(&self) -> &[ChunkAssignment] {
    &self.hard_clusters
  }

  /// Cheap-clone handle to the frame-level binary diarization grid
  /// `(num_output_frames, num_clusters)`, flattened row-major
  /// `[t][k]`.
  ///
  /// Returns [`crate::ops::spill::SpillBytes<f32>`]: heap-backed
  /// for grids under `SpillOptions::threshold_bytes`, mmap-backed
  /// above. Cloning is `Arc::clone`-cheap on either backend; both
  /// `Send` and `Sync`.
  pub fn discrete_diarization(&self) -> crate::ops::spill::SpillBytes<f32> {
    self.discrete_diarization.clone()
  }

  /// Borrow the frame-level binary diarization grid without cloning
  /// the underlying `SpillBytes` handle.
  pub fn discrete_diarization_slice(&self) -> &[f32] {
    self.discrete_diarization.as_slice()
  }

  /// Number of clusters in the output diarization grid.
  pub const fn num_clusters(&self) -> usize {
    self.num_clusters
  }

  /// Cheap-clone handle to the RTTM spans (uri-agnostic). Caller
  /// wraps with file id to format.
  pub fn spans(&self) -> Arc<[RttmSpan]> {
    Arc::clone(&self.spans)
  }

  /// Borrow the RTTM spans without cloning the `Arc`.
  pub fn spans_slice(&self) -> &[RttmSpan] {
    &self.spans
  }
}

/// Run the offline pyannote-equivalent diarization pipeline.
///
/// Mirrors `pyannote.audio.pipelines.clustering.VBxClustering.__call__`
/// plus `pyannote/audio/pipelines/speaker_diarization.SpeakerDiarization.apply`'s
/// reconstruction step. Pyannote-equivalent output on the captured
/// fixtures (parity-tested in `crate::offline::parity_tests`).
///
/// # Errors
///
/// - [`Error::Shape`] if any tensor dimension mismatches.
/// - [`Error::Plda`] if a (chunk, speaker) raw embedding is degenerate
///   (zero-norm / NaN — caught by the `RawEmbedding` constructor in
///   `crate::plda`).
/// - [`Error::Pipeline`] if `assign_embeddings` rejects a non-finite
///   intermediate or hits a shape gate.
/// - [`Error::Reconstruct`] for non-finite segmentations or invalid
///   sliding-window timing.
pub fn diarize_offline(input: &OfflineInput<'_>) -> Result<OfflineOutput, Error> {
  // `..` skips `spill_options`: it is non-Copy, so destructuring it
  // by value would not compile. The inner `AssignEmbeddingsInput` /
  // `ReconstructInput` carry their own clones (set below), and any
  // direct allocation in this function reads `&input.spill_options`.
  let &OfflineInput {
    raw_embeddings,
    num_chunks,
    num_speakers,
    segmentations,
    num_frames_per_chunk,
    count,
    num_output_frames,
    chunks_sw,
    frames_sw,
    plda,
    threshold,
    fa,
    fb,
    max_iters,
    min_duration_off,
    smoothing_epsilon,
    ..
  } = input;

  // ── Boundary checks ────────────────────────────────────────────
  if num_chunks == 0 {
    return Err(ShapeError::ZeroNumChunks.into());
  }
  if num_speakers == 0 {
    return Err(ShapeError::ZeroNumSpeakers.into());
  }
  if num_frames_per_chunk == 0 {
    return Err(ShapeError::ZeroNumFramesPerChunk.into());
  }
  // Defense-in-depth on the reconstruction knobs. The `OfflineInput`
  // setters panic on out-of-range values, but a `pub const fn new()`
  // call followed by direct field-by-field construction (or any
  // future serde wrapper around `OfflineInput`) bypasses them. Both
  // values flow unchanged into reconstruct/RTTM span emission;
  // `+inf` smoothing collapses top-k onto stable index order and
  // `+inf` min_duration_off merges every same-cluster gap, returning
  // `Ok(_)` with corrupted spans. Surface the misconfiguration here.
  if !check_min_duration_off(min_duration_off) {
    return Err(
      ShapeError::MinDurationOffOutOfRange {
        value: min_duration_off,
      }
      .into(),
    );
  }
  if !check_smoothing_epsilon(smoothing_epsilon) {
    return Err(
      ShapeError::SmoothingEpsilonOutOfRange {
        value: smoothing_epsilon,
      }
      .into(),
    );
  }
  let expected_emb_len = num_chunks
    .checked_mul(num_speakers)
    .and_then(|n| n.checked_mul(EMBEDDING_DIM))
    .ok_or(ShapeError::RawEmbeddingsOverflow)?;
  if raw_embeddings.len() != expected_emb_len {
    return Err(ShapeError::RawEmbeddingsLenMismatch.into());
  }
  let expected_seg_len = num_chunks
    .checked_mul(num_frames_per_chunk)
    .and_then(|n| n.checked_mul(num_speakers))
    .ok_or(ShapeError::SegmentationsOverflow)?;
  if segmentations.len() != expected_seg_len {
    return Err(ShapeError::SegmentationsLenMismatch.into());
  }
  // Mirror `reconstruct`'s count boundary checks at the offline
  // entrypoint so a malformed count tensor (length mismatch, zero
  // `num_output_frames`, or `count[t] > MAX_COUNT_PER_FRAME`
  // sentinel/overflow) fails before stage 1 burns the
  // `train_chunk_idx`/`train_speaker_idx` filter pass, the
  // spill-backed `embeddings` and `post_plda` builds, and the entire
  // `assign_embeddings` (AHC + VBx + Hungarian) chain. `reconstruct`
  // itself already rejects these cheaply at the back end of the
  // pipeline; without this early gate, a bad count alongside otherwise
  // valid large tensors burns PLDA projection, AHC distance work, and
  // spill disk space before surfacing the same typed error. Errors
  // are routed through `Error::Reconstruct(reconstruct::Error::Shape)`
  // so the surfaced variant is identical to the late path.
  if num_output_frames == 0 {
    return Err(
      crate::reconstruct::Error::Shape(crate::reconstruct::ShapeError::ZeroNumOutputFrames).into(),
    );
  }
  if count.len() != num_output_frames {
    return Err(
      crate::reconstruct::Error::Shape(crate::reconstruct::ShapeError::CountLenMismatch).into(),
    );
  }
  for &c in count {
    if c > crate::reconstruct::MAX_COUNT_PER_FRAME {
      return Err(
        crate::reconstruct::Error::Shape(crate::reconstruct::ShapeError::CountAboveMax).into(),
      );
    }
  }

  // ── Stage 1: filter active (chunk, speaker) pairs ──────────────
  //
  // Bit-exact port of `pyannote.audio.pipelines.clustering.
  // VBxClustering.filter_embeddings` (community-1):
  //
  //   single_active = sum(seg, axis=speaker) == 1     # per (c, f)
  //   clean[c, s] = sum_f (seg[c, f, s] * single_active[c, f])
  //   active[c, s] = clean[c, s] >= 0.2 * num_frames  # MIN_ACTIVE_RATIO
  //   chunk_idx, speaker_idx = where(active)
  //
  // The clean-frame criterion drops (chunk, speaker) pairs that are
  // ONLY active during overlap regions — where pyannote's powerset
  // segmentation has multiple slots active simultaneously. Their
  // embeddings are noisy mixtures and tend to corrupt AHC + VBx,
  // most catastrophically on 04_three_speaker (heavy 3-way overlap):
  // including them gave 38% DER, dropping them brings it to ~0%.
  //
  // The previous comment claimed pyannote uses a simple `sum > 0`
  // rule; that was wrong — `pyannote/audio/pipelines/clustering.py:
  // filter_embeddings:106-125` is unambiguous. The captured
  // `train_chunk_idx`/`train_speaker_idx` arrays in our fixtures
  // happened to match `sum > 0` for the easier fixtures
  // (01/02/03/05/06) because nearly every (c, s) with non-zero
  // activity also met the 20% clean-frame bar. 04 is the outlier.
  const MIN_ACTIVE_RATIO: f64 = 0.2;
  let min_clean_frames = MIN_ACTIVE_RATIO * num_frames_per_chunk as f64;
  let mut train_chunk_idx: Vec<usize> = Vec::new();
  let mut train_speaker_idx: Vec<usize> = Vec::new();
  for c in 0..num_chunks {
    // Per-frame: how many speakers active at this (c, f)?
    let mut single_active = vec![false; num_frames_per_chunk];
    for f in 0..num_frames_per_chunk {
      let mut active_count = 0u32;
      for s in 0..num_speakers {
        // Pyannote uses BINARIZED segmentations here. The
        // `_speaker_count` and `filter_embeddings` paths both
        // interpret nonzero seg values as active. We've already
        // run binarize upstream (via `>= onset` in the segmentation
        // step that produces the captured/streamed segmentations
        // tensor), so any nonzero entry here is binary-active.
        if segmentations[(c * num_frames_per_chunk + f) * num_speakers + s] > 0.0 {
          active_count += 1;
        }
      }
      single_active[f] = active_count == 1;
    }
    for s in 0..num_speakers {
      let mut clean_frames = 0.0_f64;
      for f in 0..num_frames_per_chunk {
        if single_active[f] {
          clean_frames += segmentations[(c * num_frames_per_chunk + f) * num_speakers + s];
        }
      }
      if clean_frames >= min_clean_frames {
        train_chunk_idx.push(c);
        train_speaker_idx.push(s);
      }
    }
  }
  let num_train = train_chunk_idx.len();

  // ── Stage 2: build full f64 embeddings buffer ──────────────────
  // shape `(num_chunks * num_speakers, EMBEDDING_DIM)`, row-major
  // flat layout. Spill-backed via `SpillBytesMut` so multi-hour
  // inputs cross the heap threshold cleanly into mmap rather than
  // OOM-aborting on the previous `DMatrix::zeros` heap allocation.
  // After fill, the buffer is frozen into `SpillBytes<f64>` and
  // passed by slice into `assign_embeddings` (no DMatrix needed —
  // the consumer accesses by manual row indexing). See the
  // heap-bound matrix-cluster note above `ShapeError` for the
  // remaining heap matrices.
  let embeddings_len = num_chunks
    .checked_mul(num_speakers)
    .and_then(|n| n.checked_mul(EMBEDDING_DIM))
    .ok_or(ShapeError::RawEmbeddingsOverflow)?;
  let mut embeddings_buf =
    crate::ops::spill::SpillBytesMut::<f64>::zeros(embeddings_len, &input.spill_options)?;
  {
    let dst = embeddings_buf.as_mut_slice();
    for c in 0..num_chunks {
      for s in 0..num_speakers {
        let row = c * num_speakers + s;
        let base = row * EMBEDDING_DIM;
        let src = &raw_embeddings[base..base + EMBEDDING_DIM];
        let row_dst = &mut dst[base..base + EMBEDDING_DIM];
        for (d, &v) in src.iter().enumerate() {
          row_dst[d] = v as f64;
        }
      }
    }
  }
  let embeddings = embeddings_buf.freeze();

  // ── Stage 3: PLDA project active embeddings ────────────────────
  //
  // Spill-backed, **row-major** layout (`data[i * plda_dim + d]`) —
  // numpy/pyannote's natural C-order convention and the contract of
  // [`AssignEmbeddingsInput::post_plda`]. The pipeline transposes
  // into a column-major spill region internally for the VBx GEMM
  // call site; the row-major boundary keeps the layout intent
  // unambiguous from any producer (numpy, row-wise Rust code, this
  // module) without an untyped layout footgun.
  let plda_dim = plda.phi().len();
  let post_plda_len = num_train
    .checked_mul(plda_dim)
    .ok_or(ShapeError::RawEmbeddingsOverflow)?;
  let mut post_plda_buf =
    crate::ops::spill::SpillBytesMut::<f64>::zeros(post_plda_len, &input.spill_options)?;
  {
    let storage = post_plda_buf.as_mut_slice();
    for (i, (&c, &s)) in train_chunk_idx
      .iter()
      .zip(train_speaker_idx.iter())
      .enumerate()
    {
      let base = (c * num_speakers + s) * EMBEDDING_DIM;
      let mut arr = [0.0_f32; EMBEDDING_DIM];
      arr.copy_from_slice(&raw_embeddings[base..base + EMBEDDING_DIM]);
      let raw = RawEmbedding::from_raw_array(arr)?;
      let projected = plda.project(&raw)?;
      let row_dst = &mut storage[i * plda_dim..(i + 1) * plda_dim];
      for (d, v) in projected.iter().enumerate() {
        // Row-major write: row `i`, column `d`.
        row_dst[d] = *v;
      }
    }
  }
  let post_plda = post_plda_buf.freeze();
  let phi = DVector::<f64>::from_iterator(plda_dim, plda.phi().iter().copied());

  // ── Stage 4: assign_embeddings (AHC + VBx + centroid + Hungarian) ─
  let pipeline_input = AssignEmbeddingsInput::new(
    embeddings.as_slice(),
    EMBEDDING_DIM,
    num_chunks,
    num_speakers,
    segmentations,
    num_frames_per_chunk,
    post_plda.as_slice(),
    plda_dim,
    &phi,
    &train_chunk_idx,
    &train_speaker_idx,
  )
  .with_threshold(threshold)
  .with_fa(fa)
  .with_fb(fb)
  .with_max_iters(max_iters)
  .with_spill_options(input.spill_options.clone());
  let hard_clusters = assign_embeddings(&pipeline_input)?;
  let _ = SP_ALIVE_THRESHOLD; // doc reference

  // ── Stage 5: reconstruct → frame-level diarization ──────────────
  //
  // Match `reconstruct`'s internal `num_clusters` computation
  // exactly: it pads up to `max(count)` so the top-K binarization
  // has enough cluster slots. If we under-count here, the
  // `discrete_to_spans` assertion `grid.len() == num_frames *
  // num_clusters` panics for fixtures where `count` peaks higher
  // than the number of distinct hard-cluster ids.
  let mut max_cluster_id = -1i32;
  for row in hard_clusters.iter() {
    for &k in row {
      if k > max_cluster_id {
        max_cluster_id = k;
      }
    }
  }
  let num_clusters_from_hard = if max_cluster_id < 0 {
    0
  } else {
    (max_cluster_id + 1) as usize
  };
  let max_count = count.iter().copied().max().unwrap_or(0) as usize;
  let num_clusters = num_clusters_from_hard.max(max_count.max(1));
  let recon_input = ReconstructInput::new(
    segmentations,
    num_chunks,
    num_frames_per_chunk,
    num_speakers,
    &hard_clusters,
    count,
    num_output_frames,
    chunks_sw,
    frames_sw,
  )
  .with_smoothing_epsilon(smoothing_epsilon)
  .with_spill_options(input.spill_options.clone());
  let discrete_diarization = reconstruct(&recon_input)?;

  // ── Stage 6: discrete diarization → RTTM spans ─────────────────
  // Use the FALLIBLE variant: `try_discrete_to_spans` returns typed
  // errors (`MinDurationOffOutOfRange`, `InvalidFramesTiming`,
  // `GridNonBinaryCell`) on bad inputs; the infallible
  // `discrete_to_spans` panics on those preconditions, which would
  // unwind across this `Result`-returning public API.
  let spans = try_discrete_to_spans(
    discrete_diarization.as_slice(),
    num_output_frames,
    num_clusters,
    frames_sw,
    min_duration_off,
  )
  .map_err(crate::reconstruct::Error::from)?;

  // `try_discrete_to_spans` builds via `Vec::push` because span
  // count is unknown a-priori; convert to `Arc<[RttmSpan]>` once at
  // the boundary. This is a one-time O(num_spans) copy (typically
  // <1000 elements) — small price for the fan-out savings on every
  // downstream `Arc::clone`.
  let spans: Arc<[RttmSpan]> = Arc::from(spans);
  Ok(OfflineOutput::new(
    hard_clusters,
    discrete_diarization,
    num_clusters,
    spans,
  ))
}

#[cfg(test)]
mod reconstruction_knob_validation_tests {
  //! `diarize_offline` must reject NaN/±inf/negative
  //! `min_duration_off` and `Some(NaN/±inf)`/`Some(<0)`
  //! `smoothing_epsilon`. The setters panic on these, but a caller
  //! can field-construct (or future-serde-bypass) an `OfflineInput`
  //! with bad values; the runtime check at `diarize_offline` entry
  //! surfaces a typed error before reconstruction silently corrupts
  //! span boundaries / top-k smoothing.

  use super::*;
  use crate::reconstruct::SlidingWindow;

  /// Build a minimal valid `OfflineInput` skeleton for predicate
  /// tests. Tensors are sized to the smallest configuration that
  /// passes the shape checks; their content does not matter because
  /// the reconstruction-knob validation runs before any tensor work.
  /// Field-by-field construction bypasses the `with_*` setter
  /// panics, which is exactly what we are exercising.
  #[allow(clippy::too_many_arguments)]
  fn build_input<'a>(
    raw: &'a [f32],
    seg: &'a [f64],
    count: &'a [u8],
    plda: &'a crate::plda::PldaTransform,
    chunks_sw: SlidingWindow,
    frames_sw: SlidingWindow,
    min_duration_off: f64,
    smoothing_epsilon: Option<f32>,
  ) -> OfflineInput<'a> {
    OfflineInput {
      raw_embeddings: raw,
      num_chunks: 1,
      num_speakers: 3,
      segmentations: seg,
      num_frames_per_chunk: 4,
      count,
      num_output_frames: 4,
      chunks_sw,
      frames_sw,
      plda,
      threshold: 0.6,
      fa: 0.07,
      fb: 0.8,
      max_iters: 20,
      min_duration_off,
      smoothing_epsilon,
      spill_options: SpillOptions::new(),
    }
  }

  #[test]
  fn diarize_offline_rejects_nan_min_duration_off() {
    let plda = crate::plda::PldaTransform::new().expect("plda");
    let raw = vec![0.0_f32; 1 * 3 * EMBEDDING_DIM];
    let seg = vec![0.0_f64; 1 * 4 * 3];
    let count = vec![0_u8; 4];
    let chunks_sw = SlidingWindow::new(0.0, 10.0, 1.0);
    let frames_sw = SlidingWindow::new(0.0, 0.0619375, 0.016875);
    let input = build_input(
      &raw,
      &seg,
      &count,
      &plda,
      chunks_sw,
      frames_sw,
      f64::NAN,
      None,
    );
    let r = diarize_offline(&input);
    assert!(
      matches!(
        r,
        Err(Error::Shape(ShapeError::MinDurationOffOutOfRange { .. }))
      ),
      "got {r:?}"
    );
  }

  #[test]
  fn diarize_offline_rejects_inf_min_duration_off() {
    let plda = crate::plda::PldaTransform::new().expect("plda");
    let raw = vec![0.0_f32; 1 * 3 * EMBEDDING_DIM];
    let seg = vec![0.0_f64; 1 * 4 * 3];
    let count = vec![0_u8; 4];
    let chunks_sw = SlidingWindow::new(0.0, 10.0, 1.0);
    let frames_sw = SlidingWindow::new(0.0, 0.0619375, 0.016875);
    let input = build_input(
      &raw,
      &seg,
      &count,
      &plda,
      chunks_sw,
      frames_sw,
      f64::INFINITY,
      None,
    );
    let r = diarize_offline(&input);
    assert!(
      matches!(
        r,
        Err(Error::Shape(ShapeError::MinDurationOffOutOfRange { .. }))
      ),
      "got {r:?}"
    );
  }

  #[test]
  fn diarize_offline_rejects_negative_min_duration_off() {
    let plda = crate::plda::PldaTransform::new().expect("plda");
    let raw = vec![0.0_f32; 1 * 3 * EMBEDDING_DIM];
    let seg = vec![0.0_f64; 1 * 4 * 3];
    let count = vec![0_u8; 4];
    let chunks_sw = SlidingWindow::new(0.0, 10.0, 1.0);
    let frames_sw = SlidingWindow::new(0.0, 0.0619375, 0.016875);
    let input = build_input(&raw, &seg, &count, &plda, chunks_sw, frames_sw, -0.5, None);
    let r = diarize_offline(&input);
    assert!(
      matches!(
        r,
        Err(Error::Shape(ShapeError::MinDurationOffOutOfRange { .. }))
      ),
      "got {r:?}"
    );
  }

  #[test]
  fn diarize_offline_rejects_nan_smoothing_epsilon() {
    let plda = crate::plda::PldaTransform::new().expect("plda");
    let raw = vec![0.0_f32; 1 * 3 * EMBEDDING_DIM];
    let seg = vec![0.0_f64; 1 * 4 * 3];
    let count = vec![0_u8; 4];
    let chunks_sw = SlidingWindow::new(0.0, 10.0, 1.0);
    let frames_sw = SlidingWindow::new(0.0, 0.0619375, 0.016875);
    let input = build_input(
      &raw,
      &seg,
      &count,
      &plda,
      chunks_sw,
      frames_sw,
      0.0,
      Some(f32::NAN),
    );
    let r = diarize_offline(&input);
    assert!(
      matches!(
        r,
        Err(Error::Shape(ShapeError::SmoothingEpsilonOutOfRange { .. }))
      ),
      "got {r:?}"
    );
  }

  #[test]
  fn diarize_offline_rejects_inf_smoothing_epsilon() {
    let plda = crate::plda::PldaTransform::new().expect("plda");
    let raw = vec![0.0_f32; 1 * 3 * EMBEDDING_DIM];
    let seg = vec![0.0_f64; 1 * 4 * 3];
    let count = vec![0_u8; 4];
    let chunks_sw = SlidingWindow::new(0.0, 10.0, 1.0);
    let frames_sw = SlidingWindow::new(0.0, 0.0619375, 0.016875);
    let input = build_input(
      &raw,
      &seg,
      &count,
      &plda,
      chunks_sw,
      frames_sw,
      0.0,
      Some(f32::INFINITY),
    );
    let r = diarize_offline(&input);
    assert!(
      matches!(
        r,
        Err(Error::Shape(ShapeError::SmoothingEpsilonOutOfRange { .. }))
      ),
      "got {r:?}"
    );
  }

  #[test]
  fn diarize_offline_rejects_negative_smoothing_epsilon() {
    let plda = crate::plda::PldaTransform::new().expect("plda");
    let raw = vec![0.0_f32; 1 * 3 * EMBEDDING_DIM];
    let seg = vec![0.0_f64; 1 * 4 * 3];
    let count = vec![0_u8; 4];
    let chunks_sw = SlidingWindow::new(0.0, 10.0, 1.0);
    let frames_sw = SlidingWindow::new(0.0, 0.0619375, 0.016875);
    let input = build_input(
      &raw,
      &seg,
      &count,
      &plda,
      chunks_sw,
      frames_sw,
      0.0,
      Some(-0.001),
    );
    let r = diarize_offline(&input);
    assert!(
      matches!(
        r,
        Err(Error::Shape(ShapeError::SmoothingEpsilonOutOfRange { .. }))
      ),
      "got {r:?}"
    );
  }

  /// `with_min_duration_off` and `with_smoothing_epsilon` setters
  /// panic-validate (parity with `OwnedPipelineOptions`).
  #[test]
  #[should_panic(expected = "min_duration_off must be finite and >= 0")]
  fn with_min_duration_off_setter_panics_on_inf() {
    let plda = crate::plda::PldaTransform::new().expect("plda");
    let raw = vec![0.0_f32; 1 * 3 * EMBEDDING_DIM];
    let seg = vec![0.0_f64; 1 * 4 * 3];
    let count = vec![0_u8; 4];
    let chunks_sw = SlidingWindow::new(0.0, 10.0, 1.0);
    let frames_sw = SlidingWindow::new(0.0, 0.0619375, 0.016875);
    let _ = OfflineInput::new(&raw, 1, 3, &seg, 4, &count, 4, chunks_sw, frames_sw, &plda)
      .with_min_duration_off(f64::INFINITY);
  }

  #[test]
  #[should_panic(expected = "smoothing_epsilon must be None or Some(finite >= 0)")]
  fn with_smoothing_epsilon_setter_panics_on_nan() {
    let plda = crate::plda::PldaTransform::new().expect("plda");
    let raw = vec![0.0_f32; 1 * 3 * EMBEDDING_DIM];
    let seg = vec![0.0_f64; 1 * 4 * 3];
    let count = vec![0_u8; 4];
    let chunks_sw = SlidingWindow::new(0.0, 10.0, 1.0);
    let frames_sw = SlidingWindow::new(0.0, 0.0619375, 0.016875);
    let _ = OfflineInput::new(&raw, 1, 3, &seg, 4, &count, 4, chunks_sw, frames_sw, &plda)
      .with_smoothing_epsilon(Some(f32::NAN));
  }
}
