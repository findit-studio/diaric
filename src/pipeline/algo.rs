//! Pyannote `cluster_vbx` flow stages 2–7 wired end-to-end.

use std::sync::Arc;

use crate::{
  cluster::{
    ahc::ahc_init,
    centroid::{SP_ALIVE_THRESHOLD, weighted_centroids},
    hungarian::{ChunkAssignment, UNMATCHED, constrained_argmax},
    vbx::{StopReason, vbx_iterate},
  },
  ops::spill::SpillOptions,
  pipeline::error::Error,
  segment::options::MAX_SPEAKER_SLOTS,
};
use nalgebra::{DMatrix, DVector};

/// Pyannote's `qinit` smoothing factor: each AHC label becomes a
/// `softmax(7.0 * one_hot)` row over `num_init` columns. Hardcoded in
/// pyannote (`utils/vbx.py:cluster_vbx`).
const QINIT_SMOOTHING: f64 = 7.0;

/// Hard upper bound on the `num_init * num_train` cell count of the
/// dense `qinit` matrix that feeds VBx EM. Pyannote realistically
/// converges on `num_init ∈ {1..15}` after AHC, and `num_train` is
/// bounded by the pipeline's intended scale (~10_000 active pairs
/// for a 1-hour stream). At those scales `qinit` is at most
/// `15 * 10_000 = 150_000` cells (~1 MB).
///
/// A pathologically tiny `threshold` can isolate every training row
/// (`num_init == num_train`); the resulting `num_train²` matrix
/// allocation could hit hundreds of MB and OOM-abort the
/// `Result`-returning pipeline. `MAX_QINIT_CELLS = 5_000_000`
/// (~40 MB at f64) is well above realistic loads but well below the
/// `vec!` capacity-overflow / OOM cliff. Surfaces as
/// [`crate::pipeline::error::ShapeError::QinitAllocationTooLarge`].
pub const MAX_QINIT_CELLS: usize = 5_000_000;

/// Hard upper bound on `num_train` — the pre-AHC active-pair count.
/// AHC's hot path is `pdist_euclidean`, which builds a condensed
/// distance vector of `num_train * (num_train - 1) / 2` f64 cells
/// (`O(N²)` memory) and runs `O(N² · embed_dim)` distance work.
///
/// At pyannote community-1's documented scale (~10_000 active pairs
/// for a 1-hour stream), that's ~50M pair distances (~400 MB) —
/// well under typical production memory budgets.
///
/// `MAX_AHC_TRAIN = 32_000` (~512M pair cells = ~4 GB) caps the
/// pdist allocation at the bound where AHC's `O(N² · embed_dim)`
/// distance work itself becomes user-perceptible (multi-second on
/// modern CPUs at `embed_dim = 256`). The 4 GB allocation is safe
/// because the pdist condensed buffer routes through
/// `crate::ops::spill::SpillBytesMut`, which falls back to file-backed
/// mmap above `SpillOptions::threshold_bytes` (default 64 MiB).
/// The kernel pages cold rows out via the mmap'd tempfile rather
/// than RAM+swap.
///
/// The realistic post-AHC `num_init` after VBx convergence is
/// small (typically `≤ 15`), so the post-AHC `num_init * num_train`
/// check against `MAX_QINIT_CELLS` is the precise allocation guard
/// for VBx; `MAX_AHC_TRAIN` is the broader AHC pdist guard.
///
/// Surfaces as
/// [`crate::pipeline::error::ShapeError::AhcTrainSizeAboveMax`].
pub const MAX_AHC_TRAIN: usize = 32_000;

/// Inputs to [`assign_embeddings`]. Grouped to keep the function
/// signature manageable.
#[derive(Debug, Clone)]
pub struct AssignEmbeddingsInput<'a> {
  /// Pre-PLDA per-`(chunk, speaker)` f64 embeddings, **row-major**
  /// flat layout `[c][s][d]`. Length must equal
  /// `num_chunks * num_speakers * embed_dim`. The slice is the
  /// authoritative shape — use [`Self::embed_dim`] to reconstruct
  /// the matrix dimensions.
  ///
  /// This used to be a `&DMatrix<f64>` (column-major) but was
  /// changed so the caller can back the storage with anything that
  /// can hand out a `&[f64]` — e.g. a heap `Vec<f64>` or a
  /// spill-backed [`crate::ops::spill::SpillBytes<f64>`]. All
  /// internal access here is by manual row indexing
  /// (`row * embed_dim + d`); no nalgebra ops are applied to
  /// `embeddings` itself.
  embeddings: &'a [f64],
  /// Per-row dimensionality of [`Self::embeddings`]. Must equal
  /// `embed_dim` (the speaker-embedding dimension produced by the
  /// upstream embedder, e.g. `EMBEDDING_DIM = 256` for community-1).
  embed_dim: usize,
  num_chunks: usize,
  num_speakers: usize,
  segmentations: &'a [f64],
  num_frames: usize,
  /// Post-PLDA features for the active training subset, **row-major**
  /// flat layout `[i][d]` (numpy/pyannote default `C`-order):
  /// `data[i * plda_dim + d]` for entry at row `i`, column `d`.
  /// Length must equal `num_train * plda_dim`.
  ///
  /// Backed by anything that exposes `&[f64]` — heap `Vec<f64>` or
  /// spill-backed `SpillBytes<f64>`. The pipeline transposes this
  /// into a separate column-major spill region before constructing
  /// the `nalgebra::DMatrixView` that VBx's GEMM call site consumes.
  /// The boundary is row-major (matching numpy / row-wise Rust
  /// code's natural convention) to avoid the silent-wrong-output
  /// failure mode of an untyped column-major handoff; the transpose
  /// is paid once per call inside `assign_embeddings`.
  post_plda: &'a [f64],
  /// Per-row dimensionality of [`Self::post_plda`] (i.e. PLDA
  /// projected feature width).
  plda_dim: usize,
  phi: &'a DVector<f64>,
  train_chunk_idx: &'a [usize],
  train_speaker_idx: &'a [usize],
  threshold: f64,
  fa: f64,
  fb: f64,
  max_iters: usize,
  /// Spill backend configuration. [`assign_embeddings`] passes this
  /// by reference to [`crate::cluster::ahc::ahc_init`], whose pdist
  /// [`crate::ops::spill::SpillBytesMut::zeros`] call honors it. Defaults
  /// to [`SpillOptions::default`].
  spill_options: SpillOptions,
}

impl<'a> AssignEmbeddingsInput<'a> {
  /// Construct with `community-1` AHC + VBx hyperparameter defaults
  /// (`threshold = 0.6`, `fa = 0.07`, `fb = 0.8`, `max_iters = 20`).
  /// Override individual hyperparameters via the `with_*` builders.
  ///
  /// Required data inputs:
  /// - `embeddings`: raw per-(chunk, speaker) embeddings, **row-major**
  ///   flat layout `[c][s][d]`. Length `num_chunks * num_speakers *
  ///   embed_dim`. Backed by anything that exposes `&[f64]` — a heap
  ///   `Vec<f64>`, a spill-backed `SpillBytes<f64>`, or any other
  ///   `Deref<Target=[f64]>` storage.
  /// - `embed_dim`: per-row dimensionality of `embeddings`.
  /// - `segmentations`: per-`(chunk, frame, speaker)` activity flattened
  ///   `[c][f][s]`. Length `num_chunks * num_frames * num_speakers`.
  /// - `post_plda`: post-PLDA features for the active training subset,
  ///   shape `(num_train, plda_dim)`, **row-major** flat layout
  ///   (`data[i * plda_dim + d]`).
  /// - `phi`: eigenvalue diagonal (length `plda_dim`).
  /// - `train_chunk_idx` / `train_speaker_idx`: row-major active
  ///   indices, length `num_train`.
  #[allow(clippy::too_many_arguments)]
  pub const fn new(
    embeddings: &'a [f64],
    embed_dim: usize,
    num_chunks: usize,
    num_speakers: usize,
    segmentations: &'a [f64],
    num_frames: usize,
    post_plda: &'a [f64],
    plda_dim: usize,
    phi: &'a DVector<f64>,
    train_chunk_idx: &'a [usize],
    train_speaker_idx: &'a [usize],
  ) -> Self {
    Self {
      embeddings,
      embed_dim,
      num_chunks,
      num_speakers,
      segmentations,
      num_frames,
      post_plda,
      plda_dim,
      phi,
      train_chunk_idx,
      train_speaker_idx,
      // Community-1 defaults.
      threshold: 0.6,
      fa: 0.07,
      fb: 0.8,
      max_iters: 20,
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

  /// Set the spill backend configuration (builder).
  ///
  /// Not `const fn`: `SpillOptions` has a non-const destructor
  /// (`Option<PathBuf>`).
  #[must_use]
  pub fn with_spill_options(mut self, spill_options: SpillOptions) -> Self {
    self.spill_options = spill_options;
    self
  }

  /// Raw per-`(chunk, speaker)` embeddings (row-major flat slice;
  /// length `num_chunks * num_speakers * embed_dim`).
  pub const fn embeddings(&self) -> &'a [f64] {
    self.embeddings
  }
  /// Per-row dimensionality of [`Self::embeddings`].
  pub const fn embed_dim(&self) -> usize {
    self.embed_dim
  }
  /// Number of chunks.
  pub const fn num_chunks(&self) -> usize {
    self.num_chunks
  }
  /// Speaker slots per chunk.
  pub const fn num_speakers(&self) -> usize {
    self.num_speakers
  }
  /// Per-`(chunk, frame, speaker)` activity.
  pub const fn segmentations(&self) -> &'a [f64] {
    self.segmentations
  }
  /// Frames per chunk.
  pub const fn num_frames(&self) -> usize {
    self.num_frames
  }
  /// Post-PLDA features for the active training subset, **row-major**
  /// flat slice (`data[i * plda_dim + d]`; length
  /// `num_train * plda_dim`). The pipeline transposes into a
  /// column-major spill region internally for the VBx
  /// `nalgebra::DMatrixView` handoff — see the field-level docs on
  /// [`AssignEmbeddingsInput::post_plda`].
  pub const fn post_plda(&self) -> &'a [f64] {
    self.post_plda
  }
  /// Per-row dimensionality of [`Self::post_plda`].
  pub const fn plda_dim(&self) -> usize {
    self.plda_dim
  }
  /// PLDA eigenvalue diagonal.
  pub const fn phi(&self) -> &'a DVector<f64> {
    self.phi
  }
  /// Active chunk indices (length `num_train`).
  pub const fn train_chunk_idx(&self) -> &'a [usize] {
    self.train_chunk_idx
  }
  /// Active speaker indices (length `num_train`).
  pub const fn train_speaker_idx(&self) -> &'a [usize] {
    self.train_speaker_idx
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
  /// Spill backend configuration passed by reference to
  /// [`crate::cluster::ahc::ahc_init`] from [`assign_embeddings`].
  pub const fn spill_options(&self) -> &SpillOptions {
    &self.spill_options
  }
}

/// Run pyannote's `cluster_vbx` flow (stages 2–7).
///
/// Returns `Vec<Vec<i32>>` of length `num_chunks`; each inner vector is
/// length `num_speakers`. Entries are alive-cluster indices in the
/// reduced (`sp > SP_ALIVE_THRESHOLD`) cluster space, or
/// [`crate::cluster::hungarian::UNMATCHED`] = `-2` for speakers with no
/// surviving cluster.
///
/// # Speaker-count constraints (currently unsupported)
///
/// Pyannote's `cluster_vbx` (`clustering.py:617-633`) supports
/// `num_clusters` / `min_clusters` / `max_clusters` constraints by
/// running a KMeans fallback over the L2-normalized training
/// embeddings *after* VBx, when auto-VBx's cluster count violates
/// the constraints. This Rust port currently only exposes the
/// auto-VBx path — there is no `num_clusters` field in
/// [`AssignEmbeddingsInput`]. All five captured fixtures used the
/// auto path, so existing parity tests are unaffected, but any
/// caller that needs a forced speaker count must either
/// post-process VBx output or wait for this feature to land.
///
/// **TODO**: add
/// `num_clusters: Option<usize>`, `min_clusters: Option<usize>`,
/// `max_clusters: Option<usize>` to the input struct and port
/// pyannote's KMeans branch when an auto-VBx count violates the
/// constraints. Adding it will require:
///   1. A k-means++ implementation (or a `linfa-clustering` dep) on
///      L2-normalized embeddings — pyannote uses sklearn's KMeans
///      with `n_init=3, random_state=42`.
///   2. Centroid recomputation from the KMeans cluster assignment.
///   3. Disabling `constrained_assignment` in this branch (pyannote
///      does this to avoid artificial cluster inflation).
///   4. A new fixture captured with `num_clusters` forcing != auto.
pub fn assign_embeddings(
  input: &AssignEmbeddingsInput<'_>,
) -> Result<Arc<[ChunkAssignment]>, Error> {
  // `..` skips `spill_options`: it is non-Copy, so destructuring it
  // by value would not compile. The AHC call below reads it via
  // `&input.spill_options` instead.
  let &AssignEmbeddingsInput {
    embeddings,
    embed_dim,
    num_chunks,
    num_speakers,
    segmentations,
    num_frames,
    post_plda,
    plda_dim,
    phi,
    train_chunk_idx,
    train_speaker_idx,
    threshold,
    fa,
    fb,
    max_iters,
    ..
  } = input;

  use crate::pipeline::error::{NonFiniteField, ShapeError};
  // ── Boundary checks ────────────────────────────────────────────
  if num_chunks == 0 {
    return Err(ShapeError::ZeroNumChunks.into());
  }
  if num_speakers != MAX_SPEAKER_SLOTS as usize {
    return Err(ShapeError::WrongNumSpeakers.into());
  }
  if embed_dim == 0 {
    return Err(ShapeError::ZeroEmbeddingDim.into());
  }
  // Use checked arithmetic at the public boundary: enormous dimension
  // products would otherwise wrap silently in release builds, letting
  // a malformed caller match the equality check with a tiny buffer
  // and reach allocation/index code with bogus shape metadata. Mirrors
  // `offline::algo`.
  let expected_emb_rows = num_chunks
    .checked_mul(num_speakers)
    .ok_or(ShapeError::EmbeddingsRowsOverflow)?;
  let expected_emb_len = expected_emb_rows
    .checked_mul(embed_dim)
    .ok_or(ShapeError::EmbeddingsLenOverflow)?;
  if embeddings.len() != expected_emb_len {
    return Err(ShapeError::EmbeddingsRowMismatch.into());
  }
  if num_frames == 0 {
    return Err(ShapeError::ZeroNumFrames.into());
  }
  let expected_seg_len = num_chunks
    .checked_mul(num_frames)
    .and_then(|n| n.checked_mul(num_speakers))
    .ok_or(ShapeError::SegmentationsOverflow)?;
  if segmentations.len() != expected_seg_len {
    return Err(ShapeError::SegmentationsLenMismatch.into());
  }
  if train_chunk_idx.len() != train_speaker_idx.len() {
    return Err(ShapeError::TrainIndexLenMismatch.into());
  }
  let num_train = train_chunk_idx.len();
  if plda_dim == 0 {
    // Zero-column post_plda would let VBx iterate on no PLDA evidence
    // — `inv_l`, `alpha`, `log_p` all degenerate to empty/zero. The
    // resulting posterior is independent of the input embeddings,
    // producing plausible hard_clusters from pure prior. A schema
    // drift in PLDA capture or downstream feeding the wrong array
    // would silently yield wrong diarization.
    return Err(ShapeError::ZeroPldaDim.into());
  }
  let expected_post_plda_len = num_train
    .checked_mul(plda_dim)
    .ok_or(ShapeError::PostPldaRowMismatch)?;
  if post_plda.len() != expected_post_plda_len {
    return Err(ShapeError::PostPldaRowMismatch.into());
  }
  if phi.len() != plda_dim {
    return Err(ShapeError::PhiPldaDimMismatch.into());
  }
  // Validate `post_plda` is entirely finite *before* any expensive
  // allocation. `vbx_iterate` itself rejects non-finite `x`, but only
  // after `assign_embeddings` has built `train_embeddings`, the
  // L2-normalized AHC matrix, the O(num_train²) condensed pdist, and
  // run linkage. A single NaN/`±inf` in `post_plda` near the train
  // cap would burn substantial spill disk + CPU before surfacing a
  // typed input error. Pull the check forward to fail fast with the
  // same `NonFiniteField::PostPlda` error regardless of input scale.
  for &v in post_plda {
    if !v.is_finite() {
      return Err(NonFiniteField::PostPlda.into());
    }
  }
  // Validate train indices stay within bounds — out-of-range silently
  // poisons centroid math by reading garbage embeddings.
  for i in 0..num_train {
    let c = train_chunk_idx[i];
    let s = train_speaker_idx[i];
    if c >= num_chunks {
      return Err(ShapeError::TrainChunkIdxOutOfRange.into());
    }
    if s >= num_speakers {
      return Err(ShapeError::TrainSpeakerIdxOutOfRange.into());
    }
  }
  // Validate that *every* row of `embeddings` and *every* entry of
  // `segmentations` is finite. AHC and centroid only validate the
  // train subset (rows indexed by `train_chunk_idx`/`train_speaker_idx`),
  // but stage 6 reads ALL embedding rows for cosine scoring and stage
  // 7 reads ALL segmentations for the inactive-speaker mask. A NaN in
  // a non-train row would silently become a soft cost that
  // `constrained_argmax` rewrites to the global `nanmin` — yielding
  // a plausible-looking but wrong assignment with no surfaced error.
  //
  // We also accumulate the per-row squared L2 norm and reject if it
  // overflows to `+inf`. A row of finite-but-very-large values
  // (`|v| > ~1e154` for `D=256`) silently produces `Σ v² = +inf`,
  // and the per-element `is_finite()` check above will not catch it.
  // Stage 6 then computes `dot(embedding, centroid)` per row via
  // `ops::scalar::dot`; an overflowing row poisons cosine scoring with
  // `inf` (rejected by Hungarian's `±inf` guard) or `NaN`
  // (silently rewritten by `nan_to_num` to global `nanmin`, returning
  // a plausible but wrong assignment). Mirrors `cluster::ahc`'s
  // `RowNormOverflow` defense for the train subset, extended to the
  // full matrix.
  for r in 0..expected_emb_rows {
    let row = &embeddings[r * embed_dim..(r + 1) * embed_dim];
    let mut sq = 0.0f64;
    for &v in row {
      if !v.is_finite() {
        return Err(NonFiniteField::Embeddings.into());
      }
      sq += v * v;
    }
    if !sq.is_finite() {
      return Err(ShapeError::RowNormOverflow { row: r }.into());
    }
  }
  for v in segmentations.iter() {
    if !v.is_finite() {
      return Err(NonFiniteField::Segmentations.into());
    }
  }
  // Validate ALL clustering hyperparameters BEFORE the
  // `num_train < 2` fast path. The fast path skips AHC + VBx
  // entirely, so any validation deferred to those modules is
  // data-dependent: an invalid `threshold`/`fa`/`fb`/`max_iters`
  // returns `Ok(_)` on sparse / silent input and fails only once
  // enough speech accumulates. Pulling the checks forward makes
  // option-validation deterministic regardless of input data.
  if !threshold.is_finite() || threshold <= 0.0 {
    return Err(ShapeError::InvalidThreshold.into());
  }
  if !fa.is_finite() || fa <= 0.0 {
    return Err(ShapeError::InvalidFa.into());
  }
  if !fb.is_finite() || fb <= 0.0 {
    return Err(ShapeError::InvalidFb.into());
  }
  if max_iters == 0 {
    return Err(ShapeError::ZeroMaxIters.into());
  }
  if max_iters > crate::cluster::vbx::MAX_ITERS_CAP {
    return Err(
      ShapeError::MaxItersExceedsCap {
        got: max_iters,
        cap: crate::cluster::vbx::MAX_ITERS_CAP,
      }
      .into(),
    );
  }
  // Pyannote one-cluster fast path (`clustering.py:588-594`): when
  // fewer than 2 active embeddings survive `filter_embeddings`,
  // pyannote skips AHC/VBx entirely and returns
  // `hard_clusters = np.zeros((num_chunks, num_speakers))` —
  // i.e. every speaker in every chunk gets cluster 0. This handles
  // short clips, sparse speech, or single-usable-speaker recordings
  // without erroring.
  if num_train < 2 {
    // Build directly via TrustedLen iterator collect — no
    // `Vec`-then-`Arc` round-trip.
    return Ok(
      (0..num_chunks)
        .map(|_| [0_i32; MAX_SPEAKER_SLOTS as usize])
        .collect(),
    );
  }

  // Pre-AHC work cap. AHC's hot path is `pdist_euclidean` with
  // `O(num_train² · embed_dim)` time + `O(num_train²)` memory.
  // Reject `num_train > MAX_AHC_TRAIN` upfront so a pathological
  // input cannot burn unbounded distance work before any clustering
  // decision. This is a SEPARATE concern from the qinit allocation
  // cap (`MAX_QINIT_CELLS`): qinit is `num_init * num_train` post-
  // AHC, and realistic `num_init ≤ 15`, so `num_train²` would be
  // far too tight. The post-AHC check below catches the actual
  // qinit allocation; this pre-AHC check just bounds the AHC work
  // itself.
  if num_train > MAX_AHC_TRAIN {
    return Err(
      ShapeError::AhcTrainSizeAboveMax {
        got: num_train,
        max: MAX_AHC_TRAIN,
      }
      .into(),
    );
  }

  // ── Stage 2: AHC on active embeddings ──────────────────────────
  // Project the rows of `embeddings` selected by `(chunk_idx,
  // speaker_idx)` into a contiguous `(num_train, embed_dim)` flat
  // buffer, **row-major** (matching the `embeddings` layout). The
  // buffer is spill-backed via `SpillBytesMut<f64>` so multi-hour
  // / large-`num_train` inputs don't OOM-abort here even though
  // the previous nalgebra `DMatrix` allocation was heap-only.
  // `ahc_init` and `weighted_centroids` consume the row-major
  // `&[f64]` directly — no `DMatrix` materialization.
  let train_emb_len = num_train
    .checked_mul(embed_dim)
    .ok_or(ShapeError::EmbeddingsLenOverflow)?;
  let mut train_embeddings_buf =
    crate::ops::spill::SpillBytesMut::<f64>::zeros(train_emb_len, &input.spill_options)?;
  {
    let dst = train_embeddings_buf.as_mut_slice();
    for i in 0..num_train {
      let c = train_chunk_idx[i];
      let s = train_speaker_idx[i];
      let row = c * num_speakers + s;
      let src = &embeddings[row * embed_dim..(row + 1) * embed_dim];
      let row_dst = &mut dst[i * embed_dim..(i + 1) * embed_dim];
      row_dst.copy_from_slice(src);
    }
  }
  let train_embeddings = train_embeddings_buf.freeze();
  let ahc_clusters = ahc_init(
    train_embeddings.as_slice(),
    num_train,
    embed_dim,
    threshold,
    &input.spill_options,
  )?;

  // ── Stage 3 (caller-supplied): post_plda is the VBx feature matrix.
  // ── Stage 4: VBx ──────────────────────────────────────────────
  let num_init = ahc_clusters.iter().copied().max().expect("num_train >= 2") + 1;
  // Resource gate before the dense `num_train * num_init` qinit
  // allocation. AHC with a pathologically tiny threshold can produce
  // `num_init == num_train`, so the worst-case allocation scales
  // quadratically with `num_train`. Surface as a typed error before
  // hitting the `vec!` allocation panic / OOM-abort.
  let qinit_cells = num_train
    .checked_mul(num_init)
    .ok_or(ShapeError::QinitAllocationTooLarge {
      got: usize::MAX,
      max: MAX_QINIT_CELLS,
    })?;
  if qinit_cells > MAX_QINIT_CELLS {
    return Err(
      ShapeError::QinitAllocationTooLarge {
        got: qinit_cells,
        max: MAX_QINIT_CELLS,
      }
      .into(),
    );
  }
  let qinit = build_qinit(&ahc_clusters, num_init);
  // Transpose the row-major caller buffer into a column-major spill
  // region so we can hand a `nalgebra::DMatrixView::from_slice` to
  // `vbx_iterate`. nalgebra's `DMatrix` is column-major, so the view
  // expects `data[d * num_train + i]`; the caller-facing API takes
  // row-major (`data[i * plda_dim + d]`) to match numpy/pyannote's
  // C-order convention. A previous revision reinterpreted the raw
  // slice directly without a layout marker — a row-major caller
  // silently produced wrong responsibilities.
  //
  // The transpose is a single O(num_train · plda_dim) pass; at the
  // production cap (`num_train ≤ MAX_AHC_TRAIN = 32_000`,
  // `plda_dim = 128`) that is ~32 MB of spill-backed write, sub-ms
  // wall time. We allocate the column-major buffer through
  // `SpillBytesMut` so a multi-hour stream that crosses the
  // `SpillOptions::threshold_bytes` boundary keeps the typed
  // `SpillError` path instead of OOM-aborting on the heap.
  let post_plda_col_len = num_train
    .checked_mul(plda_dim)
    .ok_or(ShapeError::PostPldaRowMismatch)?;
  let mut post_plda_col_buf =
    crate::ops::spill::SpillBytesMut::<f64>::zeros(post_plda_col_len, &input.spill_options)?;
  {
    let dst = post_plda_col_buf.as_mut_slice();
    for i in 0..num_train {
      let src_row = &post_plda[i * plda_dim..(i + 1) * plda_dim];
      for (d, &v) in src_row.iter().enumerate() {
        dst[d * num_train + i] = v;
      }
    }
  }
  let post_plda_col = post_plda_col_buf.freeze();
  let post_plda_view =
    nalgebra::DMatrixView::from_slice(post_plda_col.as_slice(), num_train, plda_dim);
  let vbx_out = vbx_iterate(post_plda_view, phi, &qinit, fa, fb, max_iters)?;
  if vbx_out.stop_reason() == StopReason::MaxIterationsReached {
    // Pyannote silently accepts max_iters reached — it's the common
    // case in real data (16 of 20 captured iters converged but pyannote
    // doesn't check). The Rust port follows suit; downstream consumers
    // can inspect VbxOutput separately if they need the convergence flag.
  }

  // ── Stage 5: drop sp-squashed clusters, compute centroids ───────
  let centroids = weighted_centroids(
    vbx_out.gamma(),
    vbx_out.pi(),
    train_embeddings.as_slice(),
    num_train,
    embed_dim,
    SP_ALIVE_THRESHOLD,
  )?;
  let num_alive = centroids.nrows();

  // ── Stage 6: cdist(embeddings, centroids, metric="cosine") ─────
  // Then `soft_clusters = 2 - e2k_distance`. Per pyannote.
  //
  // SIMD dot — bit-identical to scalar on aarch64 (see
  // `ops::scalar::dot` module docs). The cosine costs feed
  // `constrained_argmax` (Hungarian) at stage 7; cross-architecture
  // determinism on aarch64 is guaranteed by the scalar/NEON
  // bit-identical contract.
  //
  // nalgebra is column-major so `embeddings.row(r)` and
  // `centroids.row(k)` are strided. We pack all centroid rows into
  // one flat row-major buffer (`centroid_buf`, length
  // `num_alive * embed_dim`, single heap alloc) and reuse one
  // `emb_row` scratch buffer across the inner k-loop. `norm_sq`
  // factors are hoisted: `centroid_norm_sq[k]` is a stage-6
  // constant, `emb_norm_sq` is constant across the inner k-loop.
  let mut soft = vec![DMatrix::<f64>::zeros(num_speakers, num_alive); num_chunks];
  let mut centroid_buf: Vec<f64> = Vec::with_capacity(num_alive * embed_dim);
  for k in 0..num_alive {
    for d in 0..embed_dim {
      centroid_buf.push(centroids[(k, d)]);
    }
  }
  // Scalar dot for the Hungarian-feeding cosine: stage 6's soft
  // scores are consumed by `constrained_argmax` (Hungarian), which is
  // a hard discrete decision. AVX2/AVX-512 vs scalar/NEON ulp drift
  // could flip a near-tie centroid argmax across CPU families. NEON
  // matches scalar bit-exact, but x86 does not — using the scalar
  // primitives here keeps Hungarian assignments deterministic across
  // every supported architecture.
  let centroid_norm_sq: Vec<f64> = centroid_buf
    .chunks_exact(embed_dim)
    .map(|row| crate::ops::scalar::dot(row, row))
    .collect();
  for (c, soft_c) in soft.iter_mut().enumerate() {
    for s in 0..num_speakers {
      let row = c * num_speakers + s;
      // `embeddings` is row-major flat: rows are already contiguous.
      // No need for an `emb_row` scratch copy — pass the slice
      // directly to `dot` / `cosine_distance_pre_norm`.
      let emb_row = &embeddings[row * embed_dim..(row + 1) * embed_dim];
      let emb_norm_sq = crate::ops::scalar::dot(emb_row, emb_row);
      for (k, c_row) in centroid_buf.chunks_exact(embed_dim).enumerate() {
        let dist = cosine_distance_pre_norm(emb_row, emb_norm_sq, c_row, centroid_norm_sq[k]);
        soft_c[(s, k)] = 2.0 - dist;
      }
    }
  }

  // ── Stage 7: constrained_assignment masking + Hungarian ────────
  // Pyannote: const = soft.min() - 1.0; soft[seg.sum(1) == 0] = const.
  // The mask is over (chunk, speaker) where every frame had zero
  // activity — equivalently, the speaker is "off" in this chunk.
  let mut soft_min = f64::INFINITY;
  for chunk in &soft {
    for v in chunk.iter() {
      if *v < soft_min {
        soft_min = *v;
      }
    }
  }
  let inactive_const = soft_min - 1.0;
  for c in 0..num_chunks {
    for s in 0..num_speakers {
      // sum over frames of seg[c, f, s].
      let mut sum_activity = 0.0;
      for f in 0..num_frames {
        sum_activity += segmentations[(c * num_frames + f) * num_speakers + s];
      }
      if sum_activity == 0.0 {
        for k in 0..num_alive {
          soft[c][(s, k)] = inactive_const;
        }
      }
    }
  }
  let hard = constrained_argmax(&soft)?;

  // Sanity: hard.len() == num_chunks; each row has length num_speakers.
  debug_assert_eq!(hard.len(), num_chunks);
  for row in &hard {
    debug_assert_eq!(row.len(), num_speakers);
  }

  // Build `Arc<[ChunkAssignment]>` directly from the trusted-len
  // iterator. `Vec::into_iter()` is `TrustedLen`, so std's specialized
  // `<Arc<[T]> as FromIterator<T>>::from_iter` writes each `[i32; 3]`
  // straight into the `Arc<[..]>` allocation — no intermediate `Vec`
  // round-trip.
  let hard_arc: Arc<[ChunkAssignment]> = hard
    .into_iter()
    .map(|row| {
      let mut arr = [UNMATCHED; MAX_SPEAKER_SLOTS as usize];
      arr.copy_from_slice(&row);
      arr
    })
    .collect();

  Ok(hard_arc)
}

/// Build pyannote's `qinit = scipy_softmax(one_hot(ahc_clusters) * 7.0)`
/// matrix. Shape `(num_train, num_init)` with each row a softmax of a
/// one-hot vector at column `ahc_clusters[i]`. Smoothing factor 7.0 is
/// hardcoded in `pyannote.audio.utils.vbx.cluster_vbx`.
fn build_qinit(ahc_clusters: &[usize], num_init: usize) -> DMatrix<f64> {
  let n = ahc_clusters.len();
  let on_logit = QINIT_SMOOTHING;
  // softmax over (one_hot * 7.0): row r has logits [0, …, 7 (at hot col), …, 0].
  // Numerator: exp(7.0) at hot col, exp(0) = 1 elsewhere.
  // Denominator: exp(7.0) + (num_init - 1).
  let on_exp = on_logit.exp();
  let denom = on_exp + (num_init - 1) as f64;
  let on_mass = on_exp / denom;
  let off_mass = 1.0 / denom;
  let mut q = DMatrix::<f64>::from_element(n, num_init, off_mass);
  for (r, &k) in ahc_clusters.iter().enumerate() {
    q[(r, k)] = on_mass;
  }
  q
}

/// Cosine distance between two rows of two matrices: `1 - dot / (|a| *
/// |b|)`. Matches `scipy.spatial.distance.cdist(metric="cosine")` for
/// finite vectors.
///
/// Zero-norm rows return `NaN` (matching scipy's 0/0 behavior). Stage
/// 7's `diarization::cluster::hungarian::constrained_argmax` rewrites NaN to the global
/// nanmin via `np.nan_to_num`, so a zero-norm active row gets the
/// worst possible cost and is NOT preferred over genuinely-similar
/// embeddings. Returning `1.0` (mid-similarity) instead — as the
/// previous version did — would have let a corrupt zero-vector
/// embedding tie or beat a real low-similarity match.
///
/// Cosine distance variant that takes pre-computed `||row||²` for
/// both inputs. Used by stage 6's hot inner loop where `norm_sq_b` is
/// constant across the k-iteration and `norm_sq_a` is constant across
/// the cluster loop — so the caller hoists both out and only pays for
/// one dot per (c, s, k).
///
/// Uses scalar dot specifically — see stage 6 comment block. The
/// score feeds Hungarian argmax, where ulp drift could flip a
/// discrete decision across CPU families.
fn cosine_distance_pre_norm(row_a: &[f64], norm_sq_a: f64, row_b: &[f64], norm_sq_b: f64) -> f64 {
  debug_assert_eq!(row_a.len(), row_b.len());
  let dot = crate::ops::scalar::dot(row_a, row_b);
  let denom = norm_sq_a.sqrt() * norm_sq_b.sqrt();
  if denom == 0.0 {
    return f64::NAN;
  }
  1.0 - dot / denom
}
