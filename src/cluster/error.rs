//! Error type for `diarization::cluster`. Matches spec §4.3.

/// Errors returned by [`crate::cluster`] entrypoints.
#[derive(Debug, thiserror::Error)]
pub enum Error {
  /// `cluster_offline` was passed an empty embeddings list.
  #[error("input embeddings list is empty")]
  EmptyInput,

  /// `target_speakers` strictly greater than the embedding count.
  #[error("target_speakers ({target}) > input embeddings count ({n})")]
  TargetExceedsInput {
    /// The requested target speaker count.
    target: u32,
    /// The number of input embeddings.
    n: usize,
  },

  /// `target_speakers = Some(0)`.
  #[error("target_speakers must be >= 1")]
  TargetTooSmall,

  /// Input contains NaN/inf — see also `DegenerateEmbedding`.
  #[error("input contains NaN or non-finite values")]
  NonFiniteInput,

  /// Input contains a zero-norm or near-zero-norm embedding
  /// (`||e|| < NORM_EPSILON`). Distinct from `NonFiniteInput`.
  #[error("input contains a zero-norm or degenerate embedding")]
  DegenerateEmbedding,

  /// All pairwise similarities ≤ 0 OR at least one node is isolated
  /// (`D_ii < NORM_EPSILON`) → spectral clustering's normalized
  /// Laplacian is undefined. Spec §5.5 step 2.
  #[error(
    "affinity graph has an isolated node or all-zero similarities; spectral clustering undefined"
  )]
  AllDissimilar,

  /// Eigendecomposition failed (matrix likely singular or pathological).
  #[error("eigendecomposition failed")]
  EigendecompositionFailed,

  /// `OfflineClusterOptions::similarity_threshold` is NaN/±inf or
  /// outside `[-1.0, 1.0]`. The setters enforce this on the builder
  /// path; this variant catches serde-bypassed configs that read
  /// directly into the field. The N==2 fast path uses the threshold
  /// as `sim >= threshold`, and agglomerative uses it as `1 -
  /// threshold` for the merge stop distance — out-of-range values
  /// flip both decisions silently and produce plausible-but-wrong
  /// clusterings.
  #[error("similarity_threshold ({0}) must be finite in [-1.0, 1.0]")]
  InvalidSimilarityThreshold(f32),

  /// Offline clustering input exceeds the dense-method size cap.
  ///
  /// Spectral and full-pairwise agglomerative clustering allocate dense
  /// `N × N` matrices and compute O(N³) eigendecomposition / linkage,
  /// which can OOM or stall the process before returning. The size
  /// limit ([`crate::cluster::MAX_OFFLINE_INPUT`]) is a defense-in-depth
  /// guard — callers who really need to recluster huge corpora should
  /// down-sample, batch, or use an external sparse method.
  #[error(
    "input size ({n}) exceeds the offline clustering cap ({limit}); \
     dense methods would allocate an {n}×{n} matrix"
  )]
  InputTooLarge {
    /// Actual number of input embeddings.
    n: usize,
    /// Configured cap.
    limit: usize,
  },
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn target_exceeds_input_message() {
    let e = Error::TargetExceedsInput { target: 10, n: 3 };
    let s = format!("{e}");
    assert!(s.contains("10"));
    assert!(s.contains("3"));
  }
}
