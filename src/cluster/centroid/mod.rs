//! Post-VBx weighted centroid computation.
//!
//! Ports the centroid step in pyannote's clustering pipeline
//! (`pyannote/audio/pipelines/clustering.py:618-621`):
//!
//! ```python
//! W = q[:, sp > 1e-7]   # responsibilities of speakers VBx kept alive
//! centroids = W.T @ train_embeddings.reshape(-1, dimension) / W.sum(0, keepdims=True).T
//! ```
//!
//! The result is a `(num_alive_clusters, embed_dim)` matrix used as the
//! reference set for downstream e2k distance / Hungarian assignment
//! inside the diarization pipeline.

#[cfg(test)]
pub(crate) mod algo;
#[cfg(not(test))]
mod algo;
mod error;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod parity_tests;

pub use algo::{SP_ALIVE_THRESHOLD, weighted_centroids};
pub use error::Error;
