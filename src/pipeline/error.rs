//! Errors for `diarization::pipeline`.

use thiserror::Error;

/// Errors returned by [`crate::pipeline::assign_embeddings`].
#[derive(Debug, Error)]
pub enum Error {
  /// Input shape is invalid (e.g., zero chunks, mismatched dims, etc.).
  #[error("pipeline: shape error: {0}")]
  Shape(#[from] ShapeError),
  /// A NaN/`±inf` entry was found where finite values are required.
  #[error("pipeline: non-finite value in {0}")]
  NonFinite(#[from] NonFiniteField),
  /// `min_active_ratio` falls outside `(0.0, 1.0]`.
  #[error("pipeline: invalid min_active_ratio (must be in (0, 1]): {0}")]
  InvalidActiveRatio(f64),
  /// Propagated from `diarization::cluster::ahc`.
  #[error("pipeline: ahc: {0}")]
  Ahc(#[from] crate::cluster::ahc::Error),
  /// Propagated from `diarization::cluster::vbx`.
  #[error("pipeline: vbx: {0}")]
  Vbx(#[from] crate::cluster::vbx::Error),
  /// Propagated from `diarization::cluster::centroid`.
  #[error("pipeline: centroid: {0}")]
  Centroid(#[from] crate::cluster::centroid::Error),
  /// Propagated from `diarization::cluster::hungarian`.
  #[error("pipeline: hungarian: {0}")]
  Hungarian(#[from] crate::cluster::hungarian::Error),
  /// Propagated from `diarization::plda`.
  #[error("pipeline: plda: {0}")]
  Plda(#[from] crate::plda::Error),
  /// Propagated from `crate::ops::spill::SpillBytesMut::zeros` when
  /// a spill-backed scratch buffer cannot be allocated. The
  /// `train_embeddings` row-major buffer in `assign_embeddings` and
  /// any future spill-backed matrices route through here.
  #[error("pipeline: spill: {0}")]
  Spill(#[from] crate::ops::spill::SpillError),
}

/// Specific shape-violation reasons for [`Error::Shape`].
#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum ShapeError {
  /// `num_chunks == 0`.
  #[error("num_chunks must be at least 1")]
  ZeroNumChunks,
  /// `num_speakers != MAX_SPEAKER_SLOTS` (community-1 expects 3).
  #[error("num_speakers must equal MAX_SPEAKER_SLOTS (segmentation-3.0 / community-1 = 3)")]
  WrongNumSpeakers,
  /// `embed_dim == 0`.
  #[error("embeddings must have at least one column")]
  ZeroEmbeddingDim,
  /// `num_chunks * num_speakers` overflows `usize`.
  #[error("num_chunks * num_speakers overflows usize")]
  EmbeddingsRowsOverflow,
  /// `num_chunks * num_speakers * embed_dim` overflows `usize`.
  #[error("num_chunks * num_speakers * embed_dim overflows usize")]
  EmbeddingsLenOverflow,
  /// `embeddings.len() != num_chunks * num_speakers * embed_dim`.
  #[error("embeddings.len() must equal num_chunks * num_speakers * embed_dim")]
  EmbeddingsRowMismatch,
  /// `num_frames == 0`.
  #[error("num_frames must be at least 1")]
  ZeroNumFrames,
  /// `num_chunks * num_frames * num_speakers` overflows `usize`.
  #[error("num_chunks * num_frames * num_speakers overflows usize")]
  SegmentationsOverflow,
  /// `segmentations.len()` does not equal
  /// `num_chunks * num_frames * num_speakers`.
  #[error("segmentations.len() must equal num_chunks * num_frames * num_speakers")]
  SegmentationsLenMismatch,
  /// `train_chunk_idx.len() != train_speaker_idx.len()`.
  #[error("train_chunk_idx and train_speaker_idx must have the same length")]
  TrainIndexLenMismatch,
  /// `post_plda.len() != num_train * plda_dim`.
  #[error("post_plda.len() must equal num_train * plda_dim")]
  PostPldaRowMismatch,
  /// `plda_dim == 0`.
  #[error("post_plda must have at least one column (PLDA dimension)")]
  ZeroPldaDim,
  /// `phi.len() != plda_dim`.
  #[error("phi.len() must equal plda_dim")]
  PhiPldaDimMismatch,
  /// `train_chunk_idx[i] >= num_chunks`.
  #[error("train_chunk_idx[i] out of range")]
  TrainChunkIdxOutOfRange,
  /// `train_speaker_idx[i] >= num_speakers`.
  #[error("train_speaker_idx[i] out of range")]
  TrainSpeakerIdxOutOfRange,
  /// `threshold` is non-finite or non-positive.
  #[error("threshold must be a positive finite scalar")]
  InvalidThreshold,
  /// `max_iters == 0`.
  #[error("max_iters must be at least 1")]
  ZeroMaxIters,
  /// Per-row squared-L2-norm of `embeddings` overflowed to `+inf`. The
  /// per-element finite check rejects NaN/`±inf` entries, but a row of
  /// finite-but-very-large values (`|v| > ~1e154` for `D=256`) still
  /// produces `Σ v² = +inf`. Stage 6 reads every row for cosine
  /// scoring; an overflowing non-train row turns
  /// `dot(embedding, centroid)` into `inf`, which `constrained_argmax`
  /// then either rejects (`±inf` guard) or rewrites silently
  /// (`NaN → nanmin`) — the latter would yield a plausible but wrong
  /// assignment. Reject upfront, mirroring `cluster::ahc`'s
  /// `RowNormOverflow` defense for the train subset.
  #[error("embeddings row {row} squared-norm overflow (sum of v*v exceeded f64::MAX)")]
  RowNormOverflow {
    /// 0-based row index that overflowed.
    row: usize,
  },
  /// VBx EM `fa` is non-finite or non-positive. Mirrors
  /// `crate::cluster::vbx::error::ShapeError::InvalidFa` but reported
  /// from `assign_embeddings` so the pipeline rejects bad config
  /// before the `num_train < 2` fast path skips VBx entirely. Without
  /// this check, an invalid config can return `Ok` on sparse / silent
  /// inputs and only fail once enough speech accumulates — making
  /// option-validation data-dependent.
  #[error("VBx fa must be a positive finite scalar")]
  InvalidFa,
  /// VBx EM `fb` is non-finite or non-positive. See `InvalidFa` for
  /// the rationale (validate before the fast path).
  #[error("VBx fb must be a positive finite scalar")]
  InvalidFb,
  /// `max_iters` exceeds the documented cap. Mirrors
  /// `crate::cluster::vbx::error::ShapeError::MaxItersExceedsCap`,
  /// pulled forward to the pipeline boundary.
  #[error("VBx max_iters ({got}) exceeds cap ({cap})")]
  MaxItersExceedsCap {
    /// Configured max_iters.
    got: usize,
    /// Cap (`MAX_ITERS_CAP = 1_000`).
    cap: usize,
  },
  /// `num_train` exceeds [`MAX_AHC_TRAIN`]. Bounds AHC's
  /// `O(num_train² · embed_dim)` distance work upfront so a
  /// pathological caller cannot burn unbounded compute before any
  /// clustering decision is made. Realistic production loads
  /// (~10_000 active pairs for a 1-hour stream) are well within
  /// the cap; rejection here means the input scale exceeds the
  /// documented intended use.
  ///
  /// [`MAX_AHC_TRAIN`]: crate::pipeline::MAX_AHC_TRAIN
  #[error("num_train ({got}) exceeds MAX_AHC_TRAIN ({max})")]
  AhcTrainSizeAboveMax {
    /// Actual `num_train` (active-pair count).
    got: usize,
    /// Cap (`MAX_AHC_TRAIN`).
    max: usize,
  },
  /// AHC initialization produced a `num_init × num_train` qinit
  /// allocation whose cell count exceeds [`MAX_QINIT_CELLS`]. Pyannote
  /// realistically converges on `num_init ≈ {1..15}`, so an
  /// induced `num_init` matching `num_train` indicates either a
  /// pathological tiny `threshold` or `num_train` past the
  /// pipeline's intended scale (`~10_000` for a 1-hour stream).
  /// The dense `qinit` plus VBx's `gamma`/posterior matrices would
  /// allocate hundreds of MB before `vbx_iterate` runs — surface as
  /// a typed error instead of OOM-aborting.
  ///
  /// [`MAX_QINIT_CELLS`]: crate::pipeline::MAX_QINIT_CELLS
  #[error(
    "AHC produced num_init * num_train ({got}) cells in qinit, exceeds MAX_QINIT_CELLS ({max}); \
     reduce input size or raise threshold"
  )]
  QinitAllocationTooLarge {
    /// `num_init * num_train`.
    got: usize,
    /// Cap (`MAX_QINIT_CELLS`).
    max: usize,
  },
}

/// Field that contained a non-finite value.
#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum NonFiniteField {
  /// `embeddings` contained a NaN/`±inf` entry.
  #[error("embeddings")]
  Embeddings,
  /// `segmentations` contained a NaN/`±inf` entry.
  #[error("segmentations")]
  Segmentations,
  /// `post_plda` had a NaN/`±inf` entry. Validated upfront in
  /// `assign_embeddings` so the failure surfaces before the
  /// `train_embeddings` / AHC / pdist allocations.
  #[error("post_plda")]
  PostPlda,
}
