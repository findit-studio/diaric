//! Constrained Hungarian assignment — per-chunk speaker → cluster matching.
//!
//! Ports `pyannote.audio.pipelines.clustering.SpeakerEmbedding.constrained_argmax`
//! (`clustering.py:127-140` in pyannote.audio 4.0.4). Given a per-chunk
//! `(num_speakers, num_clusters)` cost matrix (typically
//! `2 - cosine_distance(embedding, centroid)`), returns the maximum-weight
//! bipartite matching as `Vec<i32>` of length `num_speakers`. Unmatched
//! speakers (possible when `num_speakers > num_clusters`) carry the sentinel
//! [`UNMATCHED`] (`-2`).

mod algo;
mod error;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod parity_tests;

pub use algo::{
  ChunkAssignment, ChunkLayout, DefaultLayout, MAX_COST_MAGNITUDE, Segmentation3, UNMATCHED,
  constrained_argmax,
};
pub use error::Error;
