//! Agglomerative hierarchical clustering — initialization for VBx.
//!
//! Ports pyannote's AHC step
//! (`pyannote.audio.pipelines.clustering.SpeakerEmbedding.assign_embeddings`,
//! `clustering.py:597-604` in pyannote.audio 4.0.4) to Rust:
//!
//! ```python
//! train_embeddings_normed = train_embeddings / np.linalg.norm(
//!     train_embeddings, axis=1, keepdims=True
//! )
//! dendrogram = linkage(train_embeddings_normed, method="centroid", metric="euclidean")
//! ahc_clusters = fcluster(dendrogram, self.threshold, criterion="distance") - 1
//! _, ahc_clusters = np.unique(ahc_clusters, return_inverse=True)
//! ```
//!
//! Output: contiguous labels `0..k` of length `num_train`, ready to feed
//! VBx's softmax-of-one-hot `qinit` construction.

#[cfg(test)]
pub(crate) mod algo;
#[cfg(not(test))]
mod algo;
mod error;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod parity_tests;

pub use algo::ahc_init;
pub use error::Error;
