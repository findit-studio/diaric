//! Full pyannote-equivalent batch clustering pipeline.
//!
//! Ports the per-chunk diarization assignment in
//! `pyannote.audio.pipelines.clustering.SpeakerEmbedding.__call__`
//! (`clustering.py:570-625` in pyannote.audio 4.0.4):
//!
//! 1. Filter active embeddings (currently caller-supplied).
//! 2. AHC initialization on the active subset (`diarization::cluster::ahc`).
//! 3. PLDA project (`diarization::plda::PldaTransform::project` — currently caller-supplied).
//! 4. VBx EM iterations (`diarization::cluster::vbx::vbx_iterate`).
//! 5. Drop sp-squashed clusters and compute weighted centroids (`diarization::cluster::centroid`).
//! 6. Per-chunk per-speaker centroid distances (cdist with cosine metric).
//! 7. `constrained_argmax` over masked soft clusters (`diarization::cluster::hungarian`).
//!
//! Output: per-chunk hard-cluster assignments `Arc<[ChunkAssignment]>`,
//! where each [`ChunkAssignment`] is `[i32; MAX_SPEAKER_SLOTS]` (= 3)
//! and `UNMATCHED = -2` marks speakers with no surviving cluster (only
//! possible when `num_speakers > num_alive_clusters`).
//!
//! Stage 8 (per-frame discrete diarization) is handled by
//! [`crate::reconstruct`]. Callers usually reach this pipeline
//! transitively via [`crate::offline::diarize_offline`] or
//! [`crate::streaming::StreamingOfflineDiarizer`].

mod algo;
pub mod error;

#[cfg(test)]
mod parity_tests;

#[cfg(test)]
mod tests;

pub use crate::cluster::hungarian::ChunkAssignment;
pub use algo::{AssignEmbeddingsInput, MAX_AHC_TRAIN, MAX_QINIT_CELLS, assign_embeddings};
pub use error::Error;
