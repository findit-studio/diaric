//! Error type for the segmentation module.

use thiserror::Error;

use crate::segment::types::WindowId;

/// All errors produced by `crate::segment`.
#[derive(Debug, Error)]
pub enum Error {
  /// Construction-time validation failure for [`SegmentOptions`].
  ///
  /// [`SegmentOptions`]: crate::segment::SegmentOptions
  #[error("invalid segment options: {0}")]
  InvalidOptions(#[from] InvalidOptionsReason),

  /// `push_inference` received a `scores` slice of the wrong length.
  ///
  /// Expected length is [`FRAMES_PER_WINDOW`] × [`POWERSET_CLASSES`] = 4123.
  ///
  /// [`FRAMES_PER_WINDOW`]: crate::segment::FRAMES_PER_WINDOW
  /// [`POWERSET_CLASSES`]: crate::segment::POWERSET_CLASSES
  #[error("inference scores length {got}, expected {expected}")]
  InferenceShapeMismatch {
    /// Expected element count.
    expected: usize,
    /// Actual length received.
    got: usize,
  },

  /// `push_inference` was called with a [`WindowId`] that is not in the
  /// pending set.
  ///
  /// See [`Segmenter::push_inference`] rustdoc for the four scenarios this
  /// covers (never-yielded, already-consumed, stale-after-`clear`,
  /// cross-segmenter-collision).
  ///
  /// [`Segmenter::push_inference`]: crate::segment::Segmenter::push_inference
  #[error("inference scores received for unknown WindowId {id:?}")]
  UnknownWindow {
    /// The unknown id.
    id: WindowId,
  },

  /// `push_inference` received a `scores` slice containing one or more
  /// non-finite values (`NaN`, `+inf`, or `-inf`).
  ///
  /// The [`WindowId`] is left in the pending set so the caller can
  /// re-run inference (e.g. retry on a transient backend failure that
  /// produced bad logits) without losing the window.
  #[error("inference scores for WindowId {id:?} contain non-finite values")]
  NonFiniteScores {
    /// The window whose scores were rejected. Still pending; safe to
    /// retry `push_inference` after producing valid logits.
    id: WindowId,
  },
}

/// Specific reasons for [`Error::InvalidOptions`].
#[derive(Debug, Error, Clone, Copy, PartialEq)]
pub enum InvalidOptionsReason {
  #[error("step_samples must be > 0")]
  ZeroStepSamples,
  /// `step_samples` exceeds [`crate::segment::WINDOW_SAMPLES`]. The
  /// `plan_starts` window scheduler advances `s += step` between
  /// regular windows; with `step > window`, samples in
  /// `[window..step)` per chunk are never scheduled, leaving the
  /// final tail anchor as the only post-gap window. Reject at
  /// construction so this cannot reach the planner via a serde-
  /// deserialized config that bypassed
  /// [`crate::segment::SegmentOptions::with_step_samples`].
  #[error("step_samples ({step}) must not exceed WINDOW_SAMPLES ({window})")]
  StepSamplesExceedsWindow { step: u32, window: u32 },
  /// A hysteresis threshold (`onset_threshold` or `offset_threshold`)
  /// is NaN/±inf or outside `[0.0, 1.0]`. The setters already enforce
  /// this on the builder path; this variant catches serde-bypassed
  /// configs that read the field directly. Without it,
  /// `Hysteresis::new(NaN, _)` would build a sticky-silent state
  /// machine and `Hysteresis::new(_, > 1.0)` would prevent a started
  /// voice run from ever closing.
  #[error("{which}_threshold ({value}) must be finite in [0.0, 1.0]")]
  HysteresisThresholdOutOfRange {
    /// Which threshold violated the bound: `"onset"` or `"offset"`.
    which: &'static str,
    /// The offending value (NaN/±inf is shown verbatim by `Display`).
    value: f32,
  },
  /// `offset_threshold > onset_threshold`. The hysteresis state
  /// machine requires the falling-edge threshold to be no stricter
  /// than the rising-edge threshold, otherwise a started voice run
  /// can never close. The setters enforce this; the variant exists
  /// so serde-bypassed configs are also rejected at construction.
  #[error("offset_threshold ({offset}) must be <= onset_threshold ({onset})")]
  OffsetAboveOnset {
    /// The configured offset threshold.
    offset: f32,
    /// The configured onset threshold.
    onset: f32,
  },
}
