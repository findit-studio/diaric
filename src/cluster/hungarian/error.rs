//! Errors for `diarization::cluster::hungarian`.

use thiserror::Error;

/// Errors returned by [`crate::cluster::hungarian::constrained_argmax`].
#[derive(Debug, Error)]
pub enum Error {
  /// Input shape is invalid (e.g., 0 speakers or 0 clusters).
  #[error("hungarian: shape error: {0}")]
  Shape(#[from] ShapeError),
  /// A NaN/`±inf` entry was found in the cost matrix.
  #[error("hungarian: non-finite value: {0}")]
  NonFinite(#[from] NonFiniteError),
}

/// Specific shape-violation reasons for [`Error::Shape`].
#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum ShapeError {
  /// `chunks.len() == 0`.
  #[error("chunks must contain at least one chunk")]
  EmptyChunks,
  /// `num_speakers == 0`.
  #[error("num_speakers must be at least 1")]
  ZeroSpeakers,
  /// `num_clusters == 0`.
  #[error("num_clusters must be at least 1")]
  ZeroClusters,
  /// Chunks have differing `(num_speakers, num_clusters)` shapes.
  #[error("all chunks must share the same shape")]
  InconsistentChunkShape,
}

/// Specific non-finite reasons for [`Error::NonFinite`].
#[derive(Debug, Error, Clone, Copy, PartialEq)]
pub enum NonFiniteError {
  /// `soft_clusters` contains a non-finite value (`+inf`, `-inf`, or
  /// `NaN`). The Hungarian boundary in `constrained_argmax` only ever
  /// emits this variant on `±inf`, but the LSAP layer underneath
  /// rejects any non-finite input — `+inf` overflows the dual-update
  /// arithmetic and `NaN` poisons the running min comparisons. The
  /// variant name is preserved for backward compatibility with the
  /// public enum shape; the renamed message reflects the wider check.
  #[error("soft_clusters contains a non-finite value (+inf, -inf, or NaN)")]
  InfInSoftClusters,
  /// `soft_clusters` is entirely NaN — no finite value is available
  /// as the `nanmin` replacement that pyannote uses.
  #[error("soft_clusters has no finite entries; cannot compute nanmin replacement")]
  NoFiniteEntries,
  /// A finite cost magnitude exceeds [`MAX_COST_MAGNITUDE`]. The
  /// `kuhn_munkres` solver internally accumulates `lx[i] + ly[j] -
  /// weight[i,j]` and label sums; values approaching `f64::MAX`
  /// overflow to `±inf` after one or two additions, which can wedge
  /// the solver per the crate's own docs and reintroduce the failure
  /// mode the upstream `±inf` guard exists to prevent.
  ///
  /// `MAX_COST_MAGNITUDE = 1e15` is the documented safe range:
  /// production cosine distances and PLDA log-likelihoods are bounded
  /// by O(1)–O(100), so any value beyond `1e15` indicates upstream
  /// corruption rather than a legitimate cost matrix.
  ///
  /// [`MAX_COST_MAGNITUDE`]: crate::cluster::hungarian::MAX_COST_MAGNITUDE
  #[error(
    "soft_clusters contains finite value {value:e} with |value| > MAX_COST_MAGNITUDE ({max:e})"
  )]
  WeightOutOfBounds {
    /// The offending finite value.
    value: f64,
    /// The configured `MAX_COST_MAGNITUDE` cap.
    max: f64,
  },
}
