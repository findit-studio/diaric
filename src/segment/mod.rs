//! Speaker segmentation: Sans-I/O state machine + powerset post-processing.
//!
//! The backend-free segmentation surface: the [`Segmenter`] sans-I/O
//! windowing/hysteresis state machine, powerset decoding, and the option
//! constants. The ONNX segmentation model runner (`SegmentModel`) lives in
//! the `diarization` crate.
//!
//! See the crate-level docs and `docs/superpowers/specs/` for the design.

mod error;
mod hysteresis;
pub(crate) mod options;
pub mod powerset;
mod segmenter;
pub(crate) mod stitch;
mod types;
mod window;

pub use error::Error;
pub use options::{
  FRAMES_PER_WINDOW, MAX_SPEAKER_SLOTS, POWERSET_CLASSES, PYANNOTE_FRAME_DURATION_S,
  PYANNOTE_FRAME_STEP_S, SAMPLE_RATE_HZ, SAMPLE_RATE_TB, SegmentOptions, WINDOW_SAMPLES,
};
pub use segmenter::Segmenter;
pub use types::{Action, Event, SpeakerActivity, WindowId};

// Compile-time trait assertions (spec §9). Catch a future field-type
// change that would silently regress Send/Sync auto-derive.
const _: fn() = || {
  fn assert_send_sync<T: Send + Sync>() {}
  assert_send_sync::<Segmenter>();
};
