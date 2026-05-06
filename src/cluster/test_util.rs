//! Shared test helpers for `diarization::cluster` test modules.
//!
//! Test-only (not visible in non-`cfg(test)` builds).

use crate::embed::{EMBEDDING_DIM, Embedding};

/// Construct a unit-direction embedding `e_i` with a small leak into
/// dimension `(i+1) % EMBEDDING_DIM`. Norm-1 by `Embedding::normalize_from`.
///
/// `scale = 0.0` produces a pure unit basis vector (orthogonal to all
/// other `perturbed_unit(j, _)` for j ≠ i — these will trigger
/// `Error::AllDissimilar` in spectral clustering). Use a small non-zero
/// scale (e.g., 0.05) to give the affinity graph minimal connectivity.
pub(crate) fn perturbed_unit(i: usize, scale: f32) -> Embedding {
  let mut v = [0.0f32; EMBEDDING_DIM];
  v[i] = 1.0;
  v[(i + 1) % EMBEDDING_DIM] = scale;
  Embedding::normalize_from(v).unwrap()
}

/// Pure unit basis vector: `e_i` along dimension `i`, zero elsewhere.
/// L2-normalized (already unit norm).
pub(crate) fn unit(i: usize) -> Embedding {
  perturbed_unit(i, 0.0)
}
