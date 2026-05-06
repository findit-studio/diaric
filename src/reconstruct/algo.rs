//! Reconstruction math: clustered_segmentations + overlap-add aggregate
//! + top-K binarize.

use crate::{
  cluster::hungarian::{ChunkAssignment, UNMATCHED},
  reconstruct::error::Error,
};

/// Hard upper bound on the cluster-id range accepted in `hard_clusters`.
/// Pyannote's diarization pipeline emits ids bounded by the alive
/// cluster count after VBx (typically 1–4). `1024` is ~256× any
/// realistic speaker count; it stops a corrupt or malicious caller
/// from driving the `num_clusters * num_chunks * num_frames_per_chunk`
/// allocation into the multi-GB range.
pub const MAX_CLUSTER_ID: i32 = 1023;

/// Hard upper bound on `count[t]` (instantaneous active speaker count
/// per output frame). Pyannote derives `count` from
/// `aggregate(sum(binarized_seg, axis=-1))`, so the theoretical max is
/// `overlap_factor * num_speakers` ≈ 30 for the community-1 config
/// (10s chunk, 1s step, 3 speakers). Real fixtures observe max=2.
/// Capping at `64` allows comfortable headroom over realistic values
/// while catching `u8::MAX = 255`-style sentinel corruption that would
/// drive `num_clusters` and the top-K binarize past the actual
/// speaker space.
pub const MAX_COUNT_PER_FRAME: u8 = 64;

/// Hard upper bound on `num_output_frames * num_clusters` accepted by
/// [`reconstruct`].
///
/// All four large allocations along the reconstruct path —
/// `aggregated`, `agg_mask`, `clustered`, `clustered_mask`, and the
/// returned discrete diarization grid — route through
/// [`crate::ops::spill::SpillBytesMut`] / [`crate::ops::spill::SpillBytes`]
/// and spill to file-backed mmap above
/// [`crate::ops::spill::SpillOptions::threshold_bytes`] (default
/// 64 MiB). This cap is therefore a soft upper bound on disk
/// space, not an OOM cliff: at `4e8` cells the scratch state
/// approaches `1.6 GB` of `f32`/`f64` plus `400 MB` of `u8` mask,
/// well above the realistic production envelope but bounded by
/// the configured `spill_dir` filesystem rather than RAM.
pub const MAX_RECONSTRUCT_GRID_CELLS: usize = 400_000_000;

/// Pyannote `SlidingWindow` (start, duration, step), all in seconds.
#[derive(Debug, Clone, Copy)]
pub struct SlidingWindow {
  start: f64,
  duration: f64,
  step: f64,
}

impl SlidingWindow {
  /// Construct a sliding window. All values in seconds.
  pub const fn new(start: f64, duration: f64, step: f64) -> Self {
    Self {
      start,
      duration,
      step,
    }
  }

  /// First-frame center offset (seconds).
  pub const fn start(&self) -> f64 {
    self.start
  }

  /// Per-frame receptive-field length (seconds).
  pub const fn duration(&self) -> f64 {
    self.duration
  }

  /// Stride between consecutive frame centers (seconds).
  pub const fn step(&self) -> f64 {
    self.step
  }

  /// Builder: replace `start`.
  #[must_use]
  pub const fn with_start(mut self, start: f64) -> Self {
    self.start = start;
    self
  }

  /// Builder: replace `duration`.
  #[must_use]
  pub const fn with_duration(mut self, duration: f64) -> Self {
    self.duration = duration;
    self
  }

  /// Builder: replace `step`.
  #[must_use]
  pub const fn with_step(mut self, step: f64) -> Self {
    self.step = step;
    self
  }

  /// `pyannote.core.SlidingWindow.closest_frame(t)` — round to the
  /// nearest frame index whose center is at `t`. Frame `i`'s center
  /// is at `start + duration / 2 + i * step`.
  ///
  /// Uses `round_ties_even` (banker's rounding) so the rounding
  /// contract matches `crate::aggregate::count`'s
  /// `(c * chunk_step / frame_step).round_ties_even() as i64`. With
  /// plain `f64::round` (half-away-from-zero), exact `k + 0.5`
  /// inputs would shift the chunk start by one frame relative to
  /// the aggregate code, producing version/parity-dependent
  /// boundaries on tie inputs. The captured fixtures don't hit
  /// exact ties, so the parity tests can't catch this drift.
  fn closest_frame(&self, t: f64) -> i64 {
    ((t - self.start - self.duration / 2.0) / self.step).round_ties_even() as i64
  }
}

/// `const fn` predicate: `v` is `None` or `Some(finite >= 0)` (f32).
/// Mirrors `crate::offline::algo::check_smoothing_epsilon` — duplicated
/// rather than imported because `reconstruct` does not depend on
/// `offline` (it is the lower-level layer the offline orchestrator
/// calls into). Hand-coded with `v == v` (NaN check) and explicit
/// `!= INFINITY` so it remains `const` (`f32::is_finite` is not yet
/// `const`).
#[inline]
const fn check_smoothing_epsilon(v: Option<f32>) -> bool {
  match v {
    None => true,
    Some(x) => {
      #[allow(clippy::eq_op)] // intentional NaN check: NaN != NaN by IEEE 754.
      let not_nan = !(x != x);
      not_nan && x >= 0.0 && x != f32::INFINITY
    }
  }
}

/// `const fn` predicate: `v` is finite and `>= 0` (f64). Mirrors
/// `crate::offline::algo::check_min_duration_off`. See above for why
/// it is duplicated rather than imported.
///
/// Exposed `pub(crate)` so [`crate::reconstruct::rttm::try_discrete_to_spans`]
/// can apply the same check at its public boundary.
#[inline]
pub(crate) const fn check_min_duration_off(v: f64) -> bool {
  #[allow(clippy::eq_op)] // intentional NaN check: NaN != NaN by IEEE 754.
  let not_nan = !(v != v);
  not_nan && v >= 0.0 && v != f64::INFINITY
}

/// Inputs to [`reconstruct`].
#[derive(Debug, Clone)]
pub struct ReconstructInput<'a> {
  segmentations: &'a [f64],
  num_chunks: usize,
  num_frames_per_chunk: usize,
  num_speakers: usize,
  hard_clusters: &'a [ChunkAssignment],
  count: &'a [u8],
  num_output_frames: usize,
  chunks_sw: SlidingWindow,
  frames_sw: SlidingWindow,
  smoothing_epsilon: Option<f32>,
  /// Spill backend configuration. [`reconstruct`] passes this by
  /// reference to every per-cluster grid / mask
  /// [`crate::ops::spill::SpillBytesMut::zeros`] in its body. Defaults to
  /// [`crate::ops::spill::SpillOptions::default`].
  spill_options: crate::ops::spill::SpillOptions,
}

impl<'a> ReconstructInput<'a> {
  /// Construct with `smoothing_epsilon = None` (bit-exact pyannote
  /// argmax). Pass `Some(eps)` via [`Self::with_smoothing_epsilon`]
  /// to prefer the previous frame's selection when two clusters are
  /// within `eps` activation.
  ///
  /// All shape preconditions are re-verified by [`reconstruct`] —
  /// see its doc-comment for the validation rules.
  ///
  /// Required data inputs:
  /// - `segmentations`: per-`(chunk, frame, speaker)` activity flattened
  ///   `[c][f][s]`. Length `num_chunks * num_frames_per_chunk * num_speakers`.
  /// - `hard_clusters`: per-chunk hard cluster assignment (output of
  ///   `diarization::pipeline`). Length `num_chunks`; each inner vec has
  ///   length `num_speakers` with `-2` indicating an unmatched speaker.
  /// - `count`: per-output-frame instantaneous speaker count.
  ///   Length `num_output_frames`.
  /// - `chunks_sw` / `frames_sw`: outer / inner sliding windows.
  #[allow(clippy::too_many_arguments)]
  pub const fn new(
    segmentations: &'a [f64],
    num_chunks: usize,
    num_frames_per_chunk: usize,
    num_speakers: usize,
    hard_clusters: &'a [ChunkAssignment],
    count: &'a [u8],
    num_output_frames: usize,
    chunks_sw: SlidingWindow,
    frames_sw: SlidingWindow,
  ) -> Self {
    Self {
      segmentations,
      num_chunks,
      num_frames_per_chunk,
      num_speakers,
      hard_clusters,
      count,
      num_output_frames,
      chunks_sw,
      frames_sw,
      smoothing_epsilon: None,
      spill_options: crate::ops::spill::SpillOptions::new(),
    }
  }

  /// Set the temporal-smoothing epsilon for top-k selection (builder).
  /// `None` = strict descending-activation argmax. `Some(eps)` =
  /// prefer the previous frame's selection when two clusters are
  /// within `eps` activation.
  ///
  /// # Panics
  /// Panics if `smoothing_epsilon` is `Some(NaN/±inf)` or `Some(< 0)`.
  /// `Some(+inf)` makes every activation pair "within epsilon" and
  /// collapses top-k onto stable cluster index order; `Some(NaN)`
  /// makes every comparison false. Mirrors the offline-entrypoint
  /// contract checked by `crate::offline::diarize_offline`.
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
  pub fn with_spill_options(mut self, spill_options: crate::ops::spill::SpillOptions) -> Self {
    self.spill_options = spill_options;
    self
  }

  /// Per-`(chunk, frame, speaker)` activity, flattened `[c][f][s]`.
  pub const fn segmentations(&self) -> &'a [f64] {
    self.segmentations
  }
  /// Number of chunks.
  pub const fn num_chunks(&self) -> usize {
    self.num_chunks
  }
  /// Frames per chunk (segmentation model output).
  pub const fn num_frames_per_chunk(&self) -> usize {
    self.num_frames_per_chunk
  }
  /// Speaker slots per chunk.
  pub const fn num_speakers(&self) -> usize {
    self.num_speakers
  }
  /// Per-chunk hard cluster assignment.
  pub const fn hard_clusters(&self) -> &'a [ChunkAssignment] {
    self.hard_clusters
  }
  /// Per-output-frame instantaneous speaker count.
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
  /// Optional smoothing epsilon for top-k selection.
  pub const fn smoothing_epsilon(&self) -> Option<f32> {
    self.smoothing_epsilon
  }
  /// Spill backend configuration passed by reference to every
  /// [`crate::ops::spill::SpillBytesMut::zeros`] call inside
  /// [`reconstruct`].
  pub const fn spill_options(&self) -> &crate::ops::spill::SpillOptions {
    &self.spill_options
  }
}

/// Run pyannote's reconstruction.
///
/// Returns a binary `(num_output_frames * num_clusters)` flat vector
/// where row `t` has `1.0` at the top-`count[t]` cluster indices by
/// aggregated activation, `0.0` elsewhere.
///
/// `num_clusters` is derived as `max(hard_clusters) + 1`. If all
/// clusters are `UNMATCHED` (`-2`), returns an all-zero grid (no
/// clusters to assign).
///
/// # Errors
///
/// - [`Error::Shape`] for any dimension mismatch.
/// - [`Error::NonFinite`] if `segmentations` contains a non-finite
///   value (NaN handling is supported via pyannote's
///   `Inference.aggregate` mask path; arbitrary `±inf` is rejected).
/// - [`Error::Timing`] for non-finite or non-positive sliding-window
///   parameters.
pub fn reconstruct(
  input: &ReconstructInput<'_>,
) -> Result<crate::ops::spill::SpillBytes<f32>, Error> {
  // `..` skips `spill_options`: it is non-Copy, so destructuring it
  // by value would not compile. The buffer-allocation sites below
  // read it via `&input.spill_options` instead.
  let &ReconstructInput {
    segmentations,
    num_chunks,
    num_frames_per_chunk,
    num_speakers,
    hard_clusters,
    count,
    num_output_frames,
    chunks_sw,
    frames_sw,
    smoothing_epsilon,
    ..
  } = input;

  use crate::reconstruct::error::{NonFiniteField, ShapeError, TimingError};
  // ── Boundary checks ────────────────────────────────────────────
  // Defense-in-depth: `with_smoothing_epsilon` panics on out-of-range
  // values, but a `ReconstructInput` constructed via direct field
  // assignment (or any future serde wrapper) bypasses the setter.
  // `+inf` collapses every "within epsilon" comparison and forces
  // top-k onto stable cluster index order; `NaN` makes every
  // comparison false. Surface a typed error before the sort.
  if !check_smoothing_epsilon(smoothing_epsilon) {
    return Err(
      ShapeError::SmoothingEpsilonOutOfRange {
        value: smoothing_epsilon,
      }
      .into(),
    );
  }
  if num_chunks == 0 {
    return Err(ShapeError::ZeroNumChunks.into());
  }
  if num_frames_per_chunk == 0 {
    return Err(ShapeError::ZeroNumFramesPerChunk.into());
  }
  if num_speakers == 0 {
    return Err(ShapeError::ZeroNumSpeakers.into());
  }
  // Use checked arithmetic at the public boundary: a malformed caller
  // could pick dimensions whose product wraps in release (e.g.
  // `num_frames_per_chunk = usize::MAX/2 + 1`, `num_speakers = 2`,
  // wrapping to a small value), match the wrapped count with a tiny
  // segmentations slice, and reach allocation/index code with bogus
  // shape metadata. Reject overflow before the equality check.
  let expected_seg_len = num_chunks
    .checked_mul(num_frames_per_chunk)
    .and_then(|n| n.checked_mul(num_speakers))
    .ok_or(ShapeError::SegmentationsSizeOverflow)?;
  if segmentations.len() != expected_seg_len {
    return Err(ShapeError::SegmentationsLenMismatch.into());
  }
  if hard_clusters.len() != num_chunks {
    return Err(ShapeError::HardClustersLenMismatch.into());
  }
  // Each `hard_clusters[c]` is `[i32; MAX_SPEAKER_SLOTS]` by type, so
  // its length is statically equal to `MAX_SPEAKER_SLOTS = 3`. We
  // require `num_speakers <= MAX_SPEAKER_SLOTS` so the body's
  // `0..num_speakers` indexing stays in-bounds.
  if num_speakers > crate::segment::options::MAX_SPEAKER_SLOTS as usize {
    return Err(ShapeError::TooManySpeakers.into());
  }
  if num_output_frames == 0 {
    // Zero output frames with nonempty chunks/segmentations is a
    // schema/timing drift signal, not a valid input. Returning an
    // empty grid would make a downstream caller computing
    // `grid.len() / num_output_frames` divide by zero.
    return Err(ShapeError::ZeroNumOutputFrames.into());
  }
  if count.len() != num_output_frames {
    return Err(ShapeError::CountLenMismatch.into());
  }
  // count[t] = instantaneous active speaker count at output frame t.
  // Pyannote derives this from `aggregate(sum(binarized_seg, axis=-1))`
  // which sums per-chunk active counts over overlapping chunks. Real
  // fixtures observe max=2; theoretical max for community-1 is
  // overlap_factor * num_speakers ≈ 30. `MAX_COUNT_PER_FRAME = 64`
  // allows headroom while catching u8::MAX=255 sentinel corruption that
  // would expand `num_clusters` past the actual speaker space and
  // fabricate dummy speakers in the top-K binarize.
  for &c in count {
    if c > MAX_COUNT_PER_FRAME {
      return Err(ShapeError::CountAboveMax.into());
    }
  }
  for w in [chunks_sw, frames_sw] {
    if !w.duration.is_finite() || !w.step.is_finite() || !w.start.is_finite() {
      return Err(TimingError::NonFiniteParameter.into());
    }
    if w.duration <= 0.0 || w.step <= 0.0 {
      return Err(TimingError::NonPositiveDurationOrStep.into());
    }
  }
  // Validate the DERIVED timing values produced by the inner loop:
  //   chunk_start_time = chunks_sw.start + (c as f64) * chunks_sw.step
  //   center_offset    = 0.5 * frames_sw.duration
  //   t                = chunk_start_time + center_offset
  //   normalized       = (t - frames_sw.start - frames_sw.duration/2) / frames_sw.step
  //   start_frame      = normalized.round() as i64
  //   out_f            = start_frame + f (f in 0..num_frames_per_chunk)
  //
  // Adversarial-but-finite raw fields can drive any of these to
  // `±inf` or out of `i64` range, after which `as i64` is
  // unspecified behavior (saturates on most archs but unspecified
  // by the Rust Reference) and `start_frame + f as i64` overflows
  // i64 in debug. Both endpoints (first and last chunk) need
  // validation: with positive `chunks_sw.step` the largest `c`
  // dominates the upper bound, but the FIRST chunk (`c = 0`) also
  // pulls in `chunks_sw.start` directly. A finite very-negative
  // `chunks_sw.start` paired with a large `step` makes the first
  // normalized coord far below `i64::MIN/2` while the last is
  // comfortably in range — so a single-endpoint check would miss
  // the leading chunks and silently clip them to garbage indices.
  // Bound the normalized frame index well within `i64` so adding
  // `num_frames_per_chunk - 1` cannot overflow. Generous safety
  // margin: `[i64::MIN/2, i64::MAX/2]`. Production timings produce
  // values O(num_chunks) — never close to this bound.
  if num_chunks > 0 {
    let frames_center_offset = 0.5 * frames_sw.duration;
    let safe_lo = -(i64::MAX / 2) as f64;
    let safe_hi = (i64::MAX / 2) as f64;
    let normalize =
      |t: f64| -> f64 { (t - frames_sw.start - frames_sw.duration / 2.0) / frames_sw.step };

    // First chunk (c = 0). chunks_sw.start was already finite-checked
    // by the per-window guard above, so first_t is safe to add.
    let first_t = chunks_sw.start + frames_center_offset;
    if !first_t.is_finite() {
      return Err(TimingError::NonFiniteParameter.into());
    }
    let normalized_first = normalize(first_t);
    if !normalized_first.is_finite() || !(safe_lo..=safe_hi).contains(&normalized_first) {
      return Err(TimingError::NonFiniteParameter.into());
    }

    // Last chunk (c = num_chunks - 1). The `(num_chunks - 1) * step`
    // multiply can itself overflow before the add.
    let last_chunk_offset = (num_chunks as f64 - 1.0) * chunks_sw.step;
    let last_chunk_start = chunks_sw.start + last_chunk_offset;
    if !last_chunk_start.is_finite() {
      return Err(TimingError::NonFiniteParameter.into());
    }
    let last_t = last_chunk_start + frames_center_offset;
    if !last_t.is_finite() {
      return Err(TimingError::NonFiniteParameter.into());
    }
    let normalized_last = normalize(last_t);
    if !normalized_last.is_finite() || !(safe_lo..=safe_hi).contains(&normalized_last) {
      return Err(TimingError::NonFiniteParameter.into());
    }
    // `num_output_frames` must cover the last chunk's last frame.
    // Otherwise the inner loop's `out_f >= num_output_frames` skip
    // silently truncates trailing chunk contributions, returning
    // `Ok(_)` with the tail of the diarization dropped. Same shape
    // as the `try_hamming_aggregate` undersized-frames guard.
    //
    // Use `usize::try_from` rather than `as usize`: on 32-bit
    // targets a positive `i64` past `u32::MAX` wraps via `as`, so
    // the cast could produce a small valid usize and pass the
    // following `<` check, then write into a low-numbered output
    // frame. `try_from` returns `Err` for out-of-range values,
    // which we surface as `InvalidFramesTiming` (the same path
    // adversarial-but-finite raw timing already takes).
    let last_start_frame = normalized_last.round_ties_even() as i64;
    if last_start_frame >= 0 {
      let last_start_usize = usize::try_from(last_start_frame).map_err(|_| {
        ShapeError::InvalidFramesTiming(
          "derived last_start_frame exceeds usize::MAX on this target",
        )
      })?;
      let last_required = last_start_usize.saturating_add(num_frames_per_chunk);
      if num_output_frames < last_required {
        return Err(
          ShapeError::OutputFrameCountTooSmall {
            got: num_output_frames,
            required: last_required,
          }
          .into(),
        );
      }
    }
  }
  // Reject all non-finite segmentation values (NaN and ±inf). Pyannote's
  // `Inference.aggregate` does `np.nan_to_num(score, nan=0.0)` and tracks
  // missingness via a parallel mask, but the realistic source of NaN is
  // upstream model corruption (torch nan-prop), and a silent fallback
  // here lets a degraded inference dependency produce plausible-but-
  // wrong RTTM output. Surfacing it as a clear typed error matches
  // `diarization::cluster::hungarian`'s ±inf rejection at the solver boundary.
  for &v in segmentations {
    if !v.is_finite() {
      return Err(NonFiniteField::Segmentations.into());
    }
  }

  // Validate cluster ids: `UNMATCHED` (-2) is allowed; non-negative
  // values must be in `[0, MAX_CLUSTER_ID]`.
  // round 4: a stray negative id (e.g. -1) silently dropped active
  // speech under the previous code (skipped by the speakers_in_k
  // filter), and a corrupt large positive id could drive the
  // num_clusters allocation into multi-GB range.
  //
  // We restrict id-range validation to the first `num_speakers`
  // slots (the active range). Trailing slots in `[num_speakers,
  // MAX_SPEAKER_SLOTS)` MUST be UNMATCHED — without that constraint,
  // a non-UNMATCHED trailing slot would survive validation and the
  // downstream `speakers_in_k` filter would index `segmentations`
  // with `s >= num_speakers`, OOB-reading the next frame's data.
  for row in hard_clusters {
    for &k in row.iter().take(num_speakers) {
      if k == UNMATCHED {
        continue;
      }
      if k < 0 {
        return Err(ShapeError::HardClustersNegativeId.into());
      }
      if k > MAX_CLUSTER_ID {
        return Err(ShapeError::HardClustersIdAboveMax.into());
      }
    }
    for &k in row.iter().skip(num_speakers) {
      if k != UNMATCHED {
        return Err(ShapeError::HardClustersTrailingSlotNotUnmatched.into());
      }
    }
  }

  // Determine num_clusters from hard_clusters. Only consult the active
  // `num_speakers` slots — trailing slots are guaranteed UNMATCHED by
  // the validation above.
  let mut max_cluster = -1i32;
  for row in hard_clusters {
    for &k in row.iter().take(num_speakers) {
      if k > max_cluster {
        max_cluster = k;
      }
    }
  }
  if max_cluster < 0 {
    // No assigned clusters anywhere — return an all-zero grid via
    // a fresh `SpillBytesMut::zeros` (which honors the per-call
    // spill threshold) frozen for cheap-clone fan-out. The
    // zero-init is intrinsic to `zeros`; no fill loop needed.
    let buf =
      crate::ops::spill::SpillBytesMut::<f32>::zeros(num_output_frames, &input.spill_options)?;
    return Ok(buf.freeze());
  }
  let num_clusters_from_hard = (max_cluster + 1) as usize;

  // Pyannote pads num_clusters up to `max(count)` if needed (so the
  // top-K binarization can pull at least `count[t]` cluster slots).
  let max_count = count.iter().copied().max().unwrap_or(0) as usize;
  let num_clusters = num_clusters_from_hard.max(max_count.max(1));

  // ── Stage 1: clustered_segmentations ────────────────────────────
  // Initialized to NaN sentinel. We track NaN-ness via a parallel
  // bool mask to avoid f64::is_nan overhead in the aggregation loop.
  // Per-chunk: for each cluster k present in hard_clusters[c],
  // clustered[c, f, k] = max over speakers s where hard_clusters[c, s] == k
  //                       of segmentations[c, f, s].
  // Checked product: `num_clusters` derives from `max_cluster + 1`
  // which is bounded by MAX_CLUSTER_ID validation above, but the
  // multi-axis product can still overflow on adversarial dimensions
  // even if each axis individually is sane.
  let cs_size = num_chunks
    .checked_mul(num_frames_per_chunk)
    .and_then(|n| n.checked_mul(num_clusters))
    .ok_or(ShapeError::ClusteredSizeOverflow)?;
  // Cap the clustered allocation against the same budget as the
  // output grid. `clustered` is `f64` (8 B/cell) and `clustered_mask`
  // is `bool` (1 B), so `cs_size > MAX_RECONSTRUCT_GRID_CELLS` would
  // allocate >800 MB + 100 MB before the post-aggregation
  // `output_grid_size` cap fires. Reject upfront to prevent the
  // OOM-abort path.
  if cs_size > MAX_RECONSTRUCT_GRID_CELLS {
    return Err(
      ShapeError::OutputGridTooLarge {
        got: cs_size,
        max: MAX_RECONSTRUCT_GRID_CELLS,
      }
      .into(),
    );
  }
  // Spill-aware: `cs_size` reaches `MAX_RECONSTRUCT_GRID_CELLS = 1e8`
  // (~800 MB f64 + 100 MB u8 mask) at the cap. Routing through
  // `SpillBytesMut` lets the allocation fall back to file-backed mmap
  // above `SpillOptions::threshold_bytes` (default 64 MiB) instead
  // of OOM-aborting. The mask migrates from `Vec<bool>` to
  // `SpillBytesMut<u8>` because `bool` is not `bytemuck::Pod`; we use
  // `0u8` / `1u8` as the active flag (treated identically by the
  // downstream `mask[idx] == 1` check).
  let mut clustered =
    crate::ops::spill::SpillBytesMut::<f64>::zeros(cs_size, &input.spill_options)?;
  let mut clustered_mask =
    crate::ops::spill::SpillBytesMut::<u8>::zeros(cs_size, &input.spill_options)?;
  let clustered = clustered.as_mut_slice();
  let clustered_mask = clustered_mask.as_mut_slice();

  for c in 0..num_chunks {
    for k_iter in 0..num_clusters_from_hard {
      let k = k_iter as i32;
      // Find speakers in this chunk assigned to cluster k. Iterate
      // only the active `num_speakers` slots — slots beyond that are
      // guaranteed UNMATCHED by the validation above, but capping
      // explicitly is the load-bearing guarantee that `s` stays in
      // `0..num_speakers` so the segmentation index below cannot OOB.
      let speakers_in_k: Vec<usize> = hard_clusters[c]
        .iter()
        .take(num_speakers)
        .enumerate()
        .filter_map(|(s, &kk)| (kk == k).then_some(s))
        .collect();
      if speakers_in_k.is_empty() {
        continue;
      }
      for f in 0..num_frames_per_chunk {
        let mut max_act = f64::NEG_INFINITY;
        for &s in &speakers_in_k {
          let v = segmentations[(c * num_frames_per_chunk + f) * num_speakers + s];
          if v > max_act {
            max_act = v;
          }
        }
        let cs_idx = (c * num_frames_per_chunk + f) * num_clusters + k_iter;
        clustered[cs_idx] = max_act;
        clustered_mask[cs_idx] = 1;
      }
    }
  }
  // UNMATCHED speakers (k == -2) skipped — clustered_mask stays false
  // for those (cluster, frame) cells and aggregate treats them as NaN
  // (skipped contribution).

  // ── Stage 2: aggregate(skip_average=True) ──────────────────────
  // Pyannote's overlap-add: for each chunk c, find start_frame =
  // closest_frame(chunk_start_time + 0.5 * frame_duration), then
  //   aggregated[start_frame .. start_frame + npc, k] += clustered * mask
  // hamming + warm_up are all-ones in cluster_vbx's call path.
  //
  // Checked product: `num_output_frames * num_clusters` is independent
  // from the `cs_size` axes guarded above. On 32-bit targets, a feasible
  // `count.len()` near `usize::MAX / 1024` combined with a valid
  // MAX_CLUSTER_ID = 1023 would wrap silently and let the allocations
  // below get a tiny buffer that later indexing OOBs into.
  let output_grid_size = num_output_frames
    .checked_mul(num_clusters)
    .ok_or(ShapeError::OutputGridSizeOverflow)?;
  // Cap the grid allocation at `MAX_RECONSTRUCT_GRID_CELLS` so the
  // `Result`-returning API never reaches an OOM-aborting `vec!`
  // even from valid-shape inputs. A multi-million-frame +
  // ~1024-cluster grid would allocate multiple GB; production
  // realistic loads stay well within the cap.
  if output_grid_size > MAX_RECONSTRUCT_GRID_CELLS {
    return Err(
      ShapeError::OutputGridTooLarge {
        got: output_grid_size,
        max: MAX_RECONSTRUCT_GRID_CELLS,
      }
      .into(),
    );
  }
  // Same spill rationale as `clustered`/`clustered_mask` above:
  // `output_grid_size` reaches `MAX_RECONSTRUCT_GRID_CELLS` at the
  // cap. `agg_mask` migrates from `Vec<bool>` to `SpillBytesMut<u8>`
  // (0/1 sentinel; `bytemuck::Pod` requirement).
  let mut aggregated =
    crate::ops::spill::SpillBytesMut::<f32>::zeros(output_grid_size, &input.spill_options)?;
  let mut agg_mask =
    crate::ops::spill::SpillBytesMut::<u8>::zeros(output_grid_size, &input.spill_options)?;
  let aggregated = aggregated.as_mut_slice();
  let agg_mask = agg_mask.as_mut_slice();

  for c in 0..num_chunks {
    let chunk_start_time = chunks_sw.start + (c as f64) * chunks_sw.step;
    let center_offset = 0.5 * frames_sw.duration;
    let start_frame = frames_sw.closest_frame(chunk_start_time + center_offset);
    if start_frame < 0 {
      // Pyannote produces frames at non-negative indices; if a chunk
      // starts before the first output frame, clip its leading
      // frames out. start_frame_clamp = 0; clip leading
      // (-start_frame) of the chunk.
    }
    for f in 0..num_frames_per_chunk {
      let out_f = start_frame + f as i64;
      if out_f < 0 || out_f as usize >= num_output_frames {
        continue;
      }
      let out_f = out_f as usize;
      for k in 0..num_clusters_from_hard {
        let cs_idx = (c * num_frames_per_chunk + f) * num_clusters + k;
        if clustered_mask[cs_idx] == 0 {
          continue;
        }
        let v = clustered[cs_idx] as f32;
        let agg_idx = out_f * num_clusters + k;
        aggregated[agg_idx] += v;
        agg_mask[agg_idx] = 1;
      }
    }
  }
  // Cells that never received a contribution → leave as 0.0
  // (pyannote uses `missing=0.0` for to_diarization).
  for (i, &m) in agg_mask.iter().enumerate() {
    if m == 0 {
      aggregated[i] = 0.0;
    }
  }

  // ── Stage 3: top-`count[t]` binarize per output frame ──────────
  //
  // Build the output through `SpillBytesMut<f32>` so the final grid
  // honors the same heap-or-mmap budget as the scratch buffers.
  // After the fill loop the buffer is frozen into a cheap-clone
  // `SpillBytes<f32>` — read-phase, `Send + Sync`, and shareable
  // across consumers without copying. `SpillBytesMut::zeros`
  // pre-zeros the cells, so the body only needs to overwrite cells
  // that get a `1.0` selection.
  let mut out_buf =
    crate::ops::spill::SpillBytesMut::<f32>::zeros(output_grid_size, &input.spill_options)?;
  let out = out_buf.as_mut_slice();
  let mut prev_selected: Vec<usize> = Vec::new();
  for (t, &c_byte) in count.iter().enumerate().take(num_output_frames) {
    let c_count = c_byte as usize;
    if c_count == 0 {
      prev_selected.clear();
      continue;
    }
    // Sort cluster indices by descending activation at frame t.
    let row_start = t * num_clusters;
    let mut sorted: Vec<usize> = (0..num_clusters).collect();
    if let Some(eps) = smoothing_epsilon {
      // Speakrs-style tie-breaking, expressed as an additive key to
      // guarantee a strict weak order (Rust's `sort_by` requires
      // transitivity; non-transitive comparators give implementation-
      // and input-dependent output).
      //
      // Per-cluster effective activation:
      //   eff(c) = aggregated[c] + (prev_selected.contains(&c) ? eps : 0)
      //
      // Equivalence to the original "if |a-b| < eps prefer previously-
      // selected; else strict descending activation" rule:
      //
      // Case (A) was_a == was_b: eff differences equal raw differences,
      //   so descending eff = descending raw. Same as old "raw fallback".
      //
      // Case (B) was_a true, was_b false: a wins iff
      //     eff(a) > eff(b) iff va + eps > vb iff vb - va < eps.
      //   - vb > va by ≥ eps  → b wins (matches old: |va-vb| ≥ eps → raw vb wins).
      //   - vb > va by < eps  → a wins (matches old: |va-vb| < eps → bias to a).
      //   - va ≥ vb           → a wins (matches old: bias OR raw).
      //
      // Case (C) symmetric to (B).
      //
      // Counterexample fixed: with eps=0.1, activations [0.0, 0.06,
      // 0.12], no prev_selected → old comparator was non-transitive
      // (0<1, 2<0, 1==2). New: eff = [0.0, 0.06, 0.12], descending
      // sort gives [2, 1, 0] (deterministic, activation-respecting).
      //
      // `total_cmp` defends against NaN even though we already
      // validated `aggregated` finiteness (segmentations were
      // finite-checked at the pipeline boundary, and `aggregated` is
      // a finite linear combination of those).
      sorted.sort_by(|&a, &b| {
        let va_raw = aggregated[row_start + a];
        let vb_raw = aggregated[row_start + b];
        let bias_a = if prev_selected.contains(&a) { eps } else { 0.0 };
        let bias_b = if prev_selected.contains(&b) { eps } else { 0.0 };
        let va_eff = va_raw + bias_a;
        let vb_eff = vb_raw + bias_b;
        // Lexicographic key: (eff desc, raw desc, index asc).
        // Secondary `raw desc` resolves the exact-eps boundary (e.g.
        // prev cluster 0 = 0.0, cluster 1 = 1.0, eps = 1.0): both
        // effs equal 1.0, so eff alone falls back to stable index
        // order and the previously-selected cluster wins — but the
        // documented strict rule says gaps `>= eps` use raw
        // activation, where cluster 1 (higher raw) should win. The
        // secondary `raw desc` enforces that. Stable sort + index
        // tie-break only fires when raw activations are also tied.
        match vb_eff.total_cmp(&va_eff) {
          std::cmp::Ordering::Equal => vb_raw.total_cmp(&va_raw),
          other => other,
        }
      });
    } else {
      sorted.sort_by(|&a, &b| {
        let va = aggregated[row_start + a];
        let vb = aggregated[row_start + b];
        // Descending; stable tie-break by index (sort_by is stable).
        vb.total_cmp(&va)
      });
    }
    let now_selected: Vec<usize> = sorted.iter().take(c_count).copied().collect();
    for &k in &now_selected {
      out[row_start + k] = 1.0;
    }
    prev_selected = now_selected;
  }

  // Drop the `&mut [f32]` borrow so `freeze` can move out of
  // `out_buf`. NLL would also let the implicit drop happen at the
  // end of scope, but the explicit name-rebind makes the
  // ordering clear.
  let _ = out;
  // Reference UNMATCHED so the import isn't dead code.
  let _ = UNMATCHED;
  Ok(out_buf.freeze())
}
