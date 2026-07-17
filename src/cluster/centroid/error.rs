//! Errors for `diarization::cluster::centroid`.

use thiserror::Error;

/// Errors returned by [`crate::cluster::centroid::weighted_centroids`].
#[derive(Debug, Error)]
pub enum Error {
  /// Input shape is invalid (mismatched dims, no surviving clusters,
  /// non-positive `sp_threshold`, etc.).
  #[error("centroid: shape error: {0}")]
  Shape(#[from] ShapeError),
  /// A NaN/`±inf` entry was found in `q`, `sp`, or `embeddings`.
  #[error("centroid: non-finite value in {0}")]
  NonFinite(#[from] NonFiniteField),
  /// A `sp[k]` value lands inside the SIMD-vs-scalar guard band around
  /// `sp_threshold`. The discrete alive/squashed decision could differ
  /// across CPU backends (NEON ↔ AVX2 ↔ AVX-512 reductions diverge by
  /// O(1e-15) relative). Caller must rerun on a deterministic path or
  /// surface the input as ambiguous. See `weighted_centroids` for
  /// the band definition.
  #[error(
    "centroid: sp[{cluster}] = {value:.3e} lands within the SIMD guard band \
     [{lo:.0e}, {hi:.0e}] around sp_threshold = {threshold:.0e}; \
     alive/squashed decision is non-deterministic across CPU backends"
  )]
  AmbiguousAliveCluster {
    /// The cluster index whose `sp` lands in the guard band.
    cluster: usize,
    /// The actual `sp[cluster]` value.
    value: f64,
    /// The configured `sp_threshold`.
    threshold: f64,
    /// Lower bound of the guard band (exclusive).
    lo: f64,
    /// Upper bound of the guard band (exclusive).
    hi: f64,
  },
}

/// Specific shape-violation reasons for [`Error::Shape`].
#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum ShapeError {
  /// `q.nrows() == 0`.
  #[error("q must have at least one row")]
  EmptyQ,
  /// `q.ncols() == 0`.
  #[error("q must have at least one column")]
  ZeroQClusters,
  /// `sp.len() != q.ncols()`.
  #[error("sp.len() must equal q.ncols()")]
  SpQClusterMismatch,
  /// `num_train_embeddings != q.nrows()`.
  #[error("num_train_embeddings must equal q.nrows()")]
  EmbeddingsQRowMismatch,
  /// `embed_dim == 0`.
  #[error("embeddings must have at least one column")]
  ZeroEmbeddingDim,
  /// `num_train_embeddings * embed_dim` overflows `usize`.
  #[error("num_train_embeddings * embed_dim overflows usize")]
  EmbeddingsLenOverflow,
  /// `embeddings.len() != num_train_embeddings * embed_dim`.
  #[error("embeddings.len() must equal num_train_embeddings * embed_dim")]
  EmbeddingsLenMismatch,
  /// `sp_threshold` is NaN or `±inf`.
  #[error("sp_threshold must be finite")]
  NonFiniteSpThreshold,
  /// No surviving cluster after the sp-threshold filter.
  #[error("no clusters survive the sp threshold (would produce empty centroid set)")]
  NoSurvivingClusters,
  /// A surviving cluster's total `q`-column weight is `<= 0`.
  /// Normalizing by it would yield NaN.
  #[error(
    "surviving cluster has non-positive total weight; \
     cannot normalize without producing NaN"
  )]
  NonPositiveTotalWeight,
}

/// Field that contained a non-finite value.
#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum NonFiniteField {
  /// A NaN/`±inf` entry in the `q` posterior.
  #[error("q")]
  Q,
  /// A NaN/`±inf` entry in the `sp` speaker priors.
  #[error("sp")]
  Sp,
  /// A NaN/`±inf` entry in the `embeddings` slice.
  #[error("embeddings")]
  Embeddings,
}
