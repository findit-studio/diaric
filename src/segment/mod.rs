//! Speaker segmentation: Sans-I/O state machine + optional ort driver.
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

#[cfg(feature = "ort")]
#[cfg_attr(docsrs, doc(cfg(feature = "ort")))]
mod model;

pub use error::Error;
pub use options::{
  FRAMES_PER_WINDOW, MAX_SPEAKER_SLOTS, POWERSET_CLASSES, PYANNOTE_FRAME_DURATION_S,
  PYANNOTE_FRAME_STEP_S, SAMPLE_RATE_HZ, SAMPLE_RATE_TB, SegmentOptions, WINDOW_SAMPLES,
};
pub use segmenter::Segmenter;
pub use types::{Action, Event, SpeakerActivity, WindowId};

#[cfg(feature = "ort")]
#[cfg_attr(docsrs, doc(cfg(feature = "ort")))]
pub use model::{SegmentModel, SegmentModelOptions};

#[cfg(feature = "ort")]
#[cfg_attr(docsrs, doc(cfg(feature = "ort")))]
pub use ort::ep::ExecutionProviderDispatch;
/// Re-exported ort types used by [`SegmentModelOptions`] builders.
///
/// We re-export so callers can compose provider/optimization configurations
/// without importing `ort` directly. `GraphOptimizationLevel` mirrors what
/// silero exposes; `ExecutionProviderDispatch` is dia's deliberate
/// divergence — silero hard-codes provider selection, but dia exposes a
/// `with_providers` builder so we have to re-export the type it takes.
#[cfg(feature = "ort")]
#[cfg_attr(docsrs, doc(cfg(feature = "ort")))]
pub use ort::session::builder::GraphOptimizationLevel;

// Compile-time trait assertions (spec §9). Catch a future field-type
// change that would silently regress Send/Sync auto-derive.
const _: fn() = || {
  fn assert_send_sync<T: Send + Sync>() {}
  assert_send_sync::<Segmenter>();

  #[cfg(feature = "ort")]
  fn assert_send<T: Send>() {}
  // SegmentModel: Send (auto-derived). The !Sync property rides on
  // ort::Session and is not asserted here without static_assertions.
  #[cfg(feature = "ort")]
  assert_send::<SegmentModel>();
};
