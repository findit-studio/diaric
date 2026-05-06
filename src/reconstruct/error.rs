//! Errors for `diarization::reconstruct`.

use thiserror::Error;

/// Errors returned by [`crate::reconstruct::reconstruct`] and the
/// fallible RTTM-emission helpers.
#[derive(Debug, Error)]
pub enum Error {
  /// Input shape is invalid — see [`ShapeError`] for the specific
  /// reason.
  #[error("reconstruct: shape error: {0}")]
  Shape(#[from] ShapeError),
  /// A NaN/`±inf` was found in a field that requires finite values.
  #[error("reconstruct: non-finite value in {0}")]
  NonFinite(#[from] NonFiniteField),
  /// `chunks_sw` / `frames_sw` sliding-window parameters are invalid.
  #[error("reconstruct: invalid sliding-window timing: {0}")]
  Timing(#[from] TimingError),
  /// Failed to allocate a scratch buffer (`clustered`, `clustered_mask`,
  /// `aggregated`, `agg_mask`). The buffers route through
  /// `crate::ops::spill::SpillBytesMut`; on inputs whose grid sizes
  /// exceed `SpillOptions::threshold_bytes` (default 64 MiB), the
  /// allocation falls through to file-backed mmap, and tempfile /
  /// mmap failures surface here.
  ///
  /// [`SpillOptions`]: crate::ops::spill::SpillOptions
  #[error("reconstruct: failed to allocate scratch buffer: {0}")]
  Spill(#[from] crate::ops::spill::SpillError),
}

/// Specific shape-violation reasons for [`Error::Shape`].
#[derive(Debug, Error, Clone, Copy, PartialEq)]
pub enum ShapeError {
  /// `num_chunks == 0`.
  #[error("num_chunks must be at least 1")]
  ZeroNumChunks,
  /// `num_frames_per_chunk == 0`.
  #[error("num_frames_per_chunk must be at least 1")]
  ZeroNumFramesPerChunk,
  /// `num_speakers == 0`.
  #[error("num_speakers must be at least 1")]
  ZeroNumSpeakers,
  /// `num_speakers > MAX_SPEAKER_SLOTS`.
  #[error("num_speakers must be <= MAX_SPEAKER_SLOTS (3)")]
  TooManySpeakers,
  /// `segmentations.len() != num_chunks * num_frames_per_chunk *
  /// num_speakers`.
  #[error("segmentations.len() != num_chunks * num_frames_per_chunk * num_speakers")]
  SegmentationsLenMismatch,
  /// `hard_clusters.len() != num_chunks`.
  #[error("hard_clusters.len() != num_chunks")]
  HardClustersLenMismatch,
  /// `num_output_frames == 0`.
  #[error("num_output_frames must be at least 1")]
  ZeroNumOutputFrames,
  /// `count.len() != num_output_frames`.
  #[error("count.len() != num_output_frames")]
  CountLenMismatch,
  /// A `count[t]` entry exceeds [`crate::reconstruct::MAX_COUNT_PER_FRAME`].
  #[error("count entry exceeds MAX_COUNT_PER_FRAME (64)")]
  CountAboveMax,
  /// `hard_clusters` contains a negative id other than the reserved
  /// `UNMATCHED` sentinel.
  #[error("hard_clusters contains a negative id other than UNMATCHED")]
  HardClustersNegativeId,
  /// A `hard_clusters[c][s]` value exceeds
  /// [`crate::reconstruct::MAX_CLUSTER_ID`].
  #[error("hard_clusters id exceeds MAX_CLUSTER_ID (1023)")]
  HardClustersIdAboveMax,
  /// `num_chunks * num_frames_per_chunk * num_speakers` overflows
  /// `usize`.
  #[error("num_chunks * num_frames_per_chunk * num_speakers overflows usize")]
  SegmentationsSizeOverflow,
  /// `num_chunks * num_frames_per_chunk * num_clusters` overflows
  /// `usize`.
  #[error("num_chunks * num_frames_per_chunk * num_clusters overflows usize")]
  ClusteredSizeOverflow,
  /// `num_output_frames * num_clusters` overflows `usize`.
  #[error("num_output_frames * num_clusters overflows usize")]
  OutputGridSizeOverflow,
  /// `hard_clusters[c]` has a non-UNMATCHED id in a trailing slot
  /// (beyond `num_speakers`).
  #[error(
    "hard_clusters[c] has a non-UNMATCHED id in a slot beyond num_speakers; \
     slots num_speakers..MAX_SPEAKER_SLOTS must all be UNMATCHED"
  )]
  HardClustersTrailingSlotNotUnmatched,
  /// `grid.len() != num_frames * num_clusters`.
  #[error("grid.len() must equal num_frames * num_clusters")]
  GridLenMismatch,
  /// `num_frames * num_clusters` overflows `usize`.
  #[error("num_frames * num_clusters overflows usize")]
  GridSizeOverflow,
  /// `smoothing_epsilon` is `Some(NaN/±inf)` or `Some(< 0)`. The
  /// per-frame top-k pass compares activation differences against
  /// this epsilon; `Some(+inf)` collapses every comparison
  /// (every pair is "within epsilon"), making selection fall back
  /// to stable cluster index order rather than activation order.
  /// `Some(NaN)` makes every comparison false. `None` is the bit-
  /// exact pyannote argmax path and is always valid.
  ///
  /// Mirrors the same predicate the offline / streaming entrypoints
  /// enforce, but checked at the lower-level `reconstruct` boundary
  /// so direct callers cannot bypass it.
  #[error("smoothing_epsilon ({value:?}) must be None or Some(finite >= 0)")]
  SmoothingEpsilonOutOfRange {
    /// The offending `smoothing_epsilon` value.
    value: Option<f32>,
  },
  /// `min_duration_off` is NaN/±inf or negative. RTTM span-merge
  /// reads this as a non-negative seconds quantity; `+inf` merges
  /// every same-cluster gap, `NaN` silently disables merging
  /// (every comparison becomes false), and negative values are
  /// nonsensical. Catches direct callers of [`try_discrete_to_spans`]
  /// that bypass the offline-entrypoint validation.
  ///
  /// [`try_discrete_to_spans`]: crate::reconstruct::try_discrete_to_spans
  #[error("min_duration_off ({value}) must be finite and >= 0")]
  MinDurationOffOutOfRange {
    /// The offending `min_duration_off` value.
    value: f64,
  },
  /// `frames_sw` (the frame-level [`SlidingWindow`]) has a non-finite
  /// `start`/`duration`/`step` or non-positive `duration`/`step`. RTTM
  /// span emission computes `start + s * step + duration/2` per
  /// active run; non-finite or zero/negative timing produces NaN or
  /// non-monotonic span boundaries with `Ok(_)`. Direct callers of
  /// [`try_discrete_to_spans`] would otherwise silently emit invalid
  /// timestamps; the offline entrypoints construct `frames_sw` from
  /// validated pyannote constants and never trigger this.
  ///
  /// [`SlidingWindow`]: crate::reconstruct::SlidingWindow
  /// [`try_discrete_to_spans`]: crate::reconstruct::try_discrete_to_spans
  #[error("frames_sw timing invalid: {0}")]
  InvalidFramesTiming(&'static str),
  /// A grid cell is non-finite (`NaN`/`±inf`) or finite but not in
  /// `{0.0, 1.0}`. The walk treats `cell != 0.0` as "active", so a
  /// `NaN` (NaN != 0.0 is true), `±inf`, or arbitrary finite value
  /// silently becomes an active frame and contaminates emitted span
  /// boundaries. The reconstruction stage that produces grids only
  /// emits {0, 1}, so any non-binary cell here indicates upstream
  /// corruption rather than a legitimate input to RTTM emission.
  #[error("grid contains non-binary value at index {index}: {value}")]
  GridNonBinaryCell {
    /// 0-based flat index of the offending cell.
    index: usize,
    /// The non-binary cell value.
    value: f32,
  },
  /// `try_discrete_to_spans` was called with `num_frames == 0`. The
  /// `num_frames * num_clusters` product is zero in that case, so
  /// the empty-grid length check passes for any `num_clusters` —
  /// the per-cluster loop would then burn CPU running over a huge
  /// `num_clusters` while producing no spans. Reject upfront.
  ///
  /// The full-pipeline `reconstruct` boundary already enforces
  /// `num_output_frames > 0`; this variant is the lower-level
  /// fallible RTTM API's equivalent.
  #[error("num_frames must be at least 1 for try_discrete_to_spans")]
  ZeroNumFrames,
  /// `try_discrete_to_spans` was called with `num_clusters == 0`.
  /// Equivalent precondition to `ZeroNumFrames`. Strict cluster-id
  /// indexing in the per-cluster loop relies on `num_clusters >= 1`.
  #[error("num_clusters must be at least 1 for try_discrete_to_spans")]
  ZeroNumClusters,
  /// `num_clusters` exceeds the documented cap of `MAX_CLUSTER_ID + 1
  /// = 1024`. Pyannote's diarization pipeline emits ids bounded by
  /// the alive cluster count after VBx (typically 1–4). Any value
  /// past the cap is upstream corruption rather than a legitimate
  /// input — and lets a caller of the public RTTM API drive an
  /// unbounded per-cluster loop.
  #[error("num_clusters ({got}) exceeds cap ({max} = MAX_CLUSTER_ID + 1)")]
  TooManyClusters {
    /// Actual `num_clusters` value provided.
    got: usize,
    /// Cap (`MAX_CLUSTER_ID + 1 = 1024`).
    max: usize,
  },
  /// `num_output_frames * num_clusters` exceeds
  /// [`MAX_RECONSTRUCT_GRID_CELLS`]. The `reconstruct` function
  /// allocates `aggregated` and `agg_mask` of that size; without
  /// this cap a caller could pass valid-shape but pathologically
  /// large dimensions (millions of frames × ~1024 clusters) and
  /// trigger OOM-abort from a `Result`-returning API. Surface a
  /// typed error before the `vec!` allocation.
  ///
  /// [`MAX_RECONSTRUCT_GRID_CELLS`]: crate::reconstruct::MAX_RECONSTRUCT_GRID_CELLS
  #[error("num_output_frames * num_clusters ({got}) exceeds MAX_RECONSTRUCT_GRID_CELLS ({max})")]
  OutputGridTooLarge {
    /// `num_output_frames * num_clusters` cells requested.
    got: usize,
    /// Cap (`MAX_RECONSTRUCT_GRID_CELLS`).
    max: usize,
  },
  /// `num_output_frames` is positive but too small to cover the
  /// last chunk's frames. The `reconstruct` aggregation loop
  /// silently skips `out_f >= num_output_frames` contributions via
  /// the `continue` path, returning `Ok(_)` with a truncated
  /// diarization grid (trailing speech dropped). Same shape as the
  /// `try_hamming_aggregate` undersized-frames guard. Required
  /// minimum is `last_start_frame + num_frames_per_chunk`.
  #[error(
    "num_output_frames ({got}) is positive but smaller than the required \
     minimum ({required} = last_start_frame + num_frames_per_chunk); \
     trailing chunk contributions would be silently truncated"
  )]
  OutputFrameCountTooSmall {
    /// Actual `num_output_frames` value.
    got: usize,
    /// Minimum required (`last_start_frame + num_frames_per_chunk`).
    required: usize,
  },
}

/// Field that contained a non-finite value.
#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum NonFiniteField {
  /// `segmentations` contained a NaN/`±inf` entry.
  #[error("segmentations")]
  Segmentations,
}

/// Specific timing-validation failures for [`Error::Timing`].
#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum TimingError {
  /// A sliding-window `start` / `duration` / `step` is non-finite.
  #[error("non-finite sliding-window parameter")]
  NonFiniteParameter,
  /// A sliding-window `duration` or `step` is `<= 0`.
  #[error("non-positive duration or step")]
  NonPositiveDurationOrStep,
}
