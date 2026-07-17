//! Pyannote reconstruction stage: hard_clusters + segmentations + count
//! → per-output-frame discrete diarization (binary `(frames, clusters)`
//! grid).
//!
//! Ports two pyannote functions:
//! - `pyannote.audio.pipelines.speaker_diarization.reconstruct` builds
//!   `clustered_segmentations` by maxing per-cluster speaker activity
//!   per frame.
//! - `pyannote.audio.pipelines.utils.diarization.to_diarization` runs
//!   `Inference.aggregate(skip_average=True)` overlap-add on the
//!   clustered segmentations, then top-`count[t]` binarizes per frame.
//!
mod algo;
mod error;

#[cfg(test)]
mod parity_tests;

#[cfg(test)]
mod rttm_parity_tests;

#[cfg(test)]
mod tests;

pub use algo::{
  MAX_CLUSTER_ID, MAX_COUNT_PER_FRAME, MAX_RECONSTRUCT_GRID_CELLS, ReconstructInput, SlidingWindow,
  reconstruct,
};
pub use error::{Error, ShapeError};
pub use rttm::{RttmSpan, discrete_to_spans, spans_to_rttm_lines, try_discrete_to_spans};

mod rttm;
