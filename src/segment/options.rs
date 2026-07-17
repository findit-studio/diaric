//! Configuration constants and tunables for `diarization::segment`.

use core::{num::NonZeroU32, time::Duration};

use mediatime::Timebase;

/// Audio sample rate this module supports — 16 kHz.
///
/// pyannote/segmentation-3.0 was trained at 16 kHz only. Callers must
/// resample upstream.
pub const SAMPLE_RATE_HZ: u32 = 16_000;

/// `mediatime` timebase for every sample-indexed `Timestamp` and `TimeRange`
/// emitted by this module: `1 / 16_000` seconds.
pub const SAMPLE_RATE_TB: Timebase = Timebase::new(1, NonZeroU32::new(SAMPLE_RATE_HZ).unwrap());

/// Sample count of one model window — 160 000 samples (10 s at 16 kHz).
pub const WINDOW_SAMPLES: u32 = 160_000;

/// Output frames produced per window by the segmentation model.
pub const FRAMES_PER_WINDOW: usize = 589;

/// Output-frame stride in seconds for pyannote community-1's
/// segmentation model — the time between successive frame *centers*
/// in the model's output sliding window. This is **NOT** the same as
/// `WINDOW_SAMPLES / FRAMES_PER_WINDOW` (which is the *naive* per-
/// chunk frame spacing); pyannote sets it to a model-specific value
/// captured from `Inference.aggregate(frames=...)`.
///
/// `0.016875 = 270 / 16_000`. Drives the `count` tensor and
/// `discrete_diarization` output sliding-window grid.
pub const PYANNOTE_FRAME_STEP_S: f64 = 0.016875;

/// Output-frame receptive-field duration in seconds for pyannote
/// community-1's segmentation model. Used by `closest_frame` and the
/// reconstruction-side aggregation. `0.0619375 ≈ 991 / 16_000`.
pub const PYANNOTE_FRAME_DURATION_S: f64 = 0.0619375;

/// Powerset class count: silence, A, B, C, A+B, A+C, B+C.
pub const POWERSET_CLASSES: usize = 7;

/// Maximum simultaneous speakers per window.
pub const MAX_SPEAKER_SLOTS: u8 = 3;

// Hysteresis-threshold validation predicates ().
//
// Setters previously stored arbitrary `f32`. NaN turns every `p >=
// threshold` comparison false (segmenter goes permanently silent);
// values outside `[0,1]` similarly invert the hysteresis. We also
// require `offset <= onset` so a started voice run can actually close
// (the falling-edge threshold cannot be stricter than the rising-edge
// threshold). All four setters call these in `const fn` context;
// `assert!` works there if the condition is `const`, but `is_finite`
// is not `const fn` until the unstable `const_float_classify`
// feature stabilizes — so we do the equivalent check by hand:
// `v == v` (rejects NaN) and `v.is_infinite()` via comparison.
//
// `f32::INFINITY > 1.0` and `f32::NEG_INFINITY < 0.0`, so the range
// check `(0.0..=1.0).contains` would reject both — but `Range::contains`
// also isn't `const`. We do it by hand with `>=`/`<=`.
#[inline]
const fn check_hysteresis_threshold(v: f32) -> bool {
  // NaN: `!v.is_nan()` is false (here we use the `v != v` idiom phrased
  // in a clippy-clean way; `v != v` is true iff v is NaN). ±inf: out of
  // [0,1]. Finite & in-range: true.
  #[allow(clippy::eq_op)] // intentional NaN check: NaN != NaN by IEEE 754.
  let not_nan = !(v != v);
  not_nan && v >= 0.0 && v <= 1.0
}

/// Tunables for the segmenter. Defaults match the upstream pyannote pipeline.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SegmentOptions {
  #[cfg_attr(feature = "serde", serde(default = "default_onset_threshold"))]
  onset_threshold: f32,
  #[cfg_attr(feature = "serde", serde(default = "default_offset_threshold"))]
  offset_threshold: f32,
  #[cfg_attr(feature = "serde", serde(default = "default_step_samples"))]
  step_samples: u32,
  #[cfg_attr(feature = "serde", serde(default, with = "humantime_serde"))]
  min_voice_duration: Duration,
  #[cfg_attr(feature = "serde", serde(default, with = "humantime_serde"))]
  min_activity_duration: Duration,
  #[cfg_attr(feature = "serde", serde(default, with = "humantime_serde"))]
  voice_merge_gap: Duration,
}

#[cfg(feature = "serde")]
const fn default_onset_threshold() -> f32 {
  0.5
}
#[cfg(feature = "serde")]
const fn default_offset_threshold() -> f32 {
  0.357
}
#[cfg(feature = "serde")]
const fn default_step_samples() -> u32 {
  40_000
}

impl Default for SegmentOptions {
  fn default() -> Self {
    Self::new()
  }
}

impl SegmentOptions {
  /// Construct with pyannote defaults: onset 0.5, offset 0.357,
  /// step 40 000 samples (2.5 s), all duration filters disabled.
  pub const fn new() -> Self {
    Self {
      onset_threshold: 0.5,
      offset_threshold: 0.357,
      step_samples: 40_000,
      min_voice_duration: Duration::ZERO,
      min_activity_duration: Duration::ZERO,
      voice_merge_gap: Duration::ZERO,
    }
  }

  /// Onset (rising-edge) threshold for hysteresis binarization.
  pub const fn onset_threshold(&self) -> f32 {
    self.onset_threshold
  }
  /// Offset (falling-edge) threshold for hysteresis binarization.
  pub const fn offset_threshold(&self) -> f32 {
    self.offset_threshold
  }
  /// Sliding-window step in samples (default 40 000 = 2.5 s).
  pub const fn step_samples(&self) -> u32 {
    self.step_samples
  }
  /// Minimum voice-span duration; shorter spans are dropped (default 0).
  pub const fn min_voice_duration(&self) -> Duration {
    self.min_voice_duration
  }
  /// Minimum speaker-activity duration (default 0).
  pub const fn min_activity_duration(&self) -> Duration {
    self.min_activity_duration
  }
  /// Merge adjacent voice spans separated by at most this gap (default 0).
  pub const fn voice_merge_gap(&self) -> Duration {
    self.voice_merge_gap
  }

  /// Builder: set the onset threshold.
  ///
  /// # Panics
  /// Panics if `v` is NaN/±inf or outside `[0.0, 1.0]`, or if the
  /// resulting pair would violate `offset <= onset`.
  pub const fn with_onset_threshold(mut self, v: f32) -> Self {
    assert!(
      check_hysteresis_threshold(v),
      "onset_threshold must be finite in [0.0, 1.0]"
    );
    assert!(
      self.offset_threshold <= v,
      "offset_threshold must remain <= onset_threshold; lower offset first"
    );
    self.onset_threshold = v;
    self
  }
  /// Builder: set the offset threshold.
  ///
  /// # Panics
  /// Panics if `v` is NaN/±inf or outside `[0.0, 1.0]`, or if the
  /// resulting pair would violate `offset <= onset`.
  pub const fn with_offset_threshold(mut self, v: f32) -> Self {
    assert!(
      check_hysteresis_threshold(v),
      "offset_threshold must be finite in [0.0, 1.0]"
    );
    assert!(
      v <= self.onset_threshold,
      "offset_threshold must be <= onset_threshold; raise onset first"
    );
    self.offset_threshold = v;
    self
  }
  /// Builder: set the sliding-window step in samples.
  ///
  /// # Panics
  /// Panics if `v == 0` or `v > WINDOW_SAMPLES`. Zero step would hang
  /// the streaming pump (`schedule_ready_windows` would emit windows
  /// starting at 0 forever); `step > window` causes silent audio gaps
  /// of `step - window` samples between consecutive chunks where no
  /// segmentation is ever produced.
  pub const fn with_step_samples(mut self, v: u32) -> Self {
    assert!(v > 0, "step_samples must be > 0");
    assert!(
      v <= WINDOW_SAMPLES,
      "step_samples must be <= WINDOW_SAMPLES (160_000)"
    );
    self.step_samples = v;
    self
  }
  /// Builder: set the minimum voice-span duration.
  pub const fn with_min_voice_duration(mut self, v: Duration) -> Self {
    self.min_voice_duration = v;
    self
  }
  /// Builder: set the minimum speaker-activity duration.
  pub const fn with_min_activity_duration(mut self, v: Duration) -> Self {
    self.min_activity_duration = v;
    self
  }
  /// Builder: set the voice-span merge gap.
  pub const fn with_voice_merge_gap(mut self, v: Duration) -> Self {
    self.voice_merge_gap = v;
    self
  }

  /// Mutating: set the onset threshold.
  ///
  /// # Panics
  /// Panics if `v` is NaN/±inf or outside `[0.0, 1.0]`, or if the
  /// resulting pair would violate `offset <= onset`.
  pub fn set_onset_threshold(&mut self, v: f32) -> &mut Self {
    assert!(
      check_hysteresis_threshold(v),
      "onset_threshold must be finite in [0.0, 1.0]; got {v}"
    );
    assert!(
      self.offset_threshold <= v,
      "offset_threshold ({offset}) must remain <= onset_threshold ({v}); lower offset first",
      offset = self.offset_threshold
    );
    self.onset_threshold = v;
    self
  }
  /// Mutating: set the offset threshold.
  ///
  /// # Panics
  /// Panics if `v` is NaN/±inf or outside `[0.0, 1.0]`, or if the
  /// resulting pair would violate `offset <= onset`.
  pub fn set_offset_threshold(&mut self, v: f32) -> &mut Self {
    assert!(
      check_hysteresis_threshold(v),
      "offset_threshold must be finite in [0.0, 1.0]; got {v}"
    );
    assert!(
      v <= self.onset_threshold,
      "offset_threshold ({v}) must be <= onset_threshold ({onset}); raise onset first",
      onset = self.onset_threshold
    );
    self.offset_threshold = v;
    self
  }
  /// Mutating: set the sliding-window step in samples.
  ///
  /// # Panics
  /// Panics if `v == 0` or `v > WINDOW_SAMPLES`.
  /// See [`with_step_samples`](Self::with_step_samples).
  pub fn set_step_samples(&mut self, v: u32) -> &mut Self {
    assert!(v > 0, "step_samples must be > 0");
    assert!(
      v <= WINDOW_SAMPLES,
      "step_samples must be <= WINDOW_SAMPLES (160_000)"
    );
    self.step_samples = v;
    self
  }
  /// Mutating: set the minimum voice-span duration.
  pub fn set_min_voice_duration(&mut self, v: Duration) -> &mut Self {
    self.min_voice_duration = v;
    self
  }
  /// Mutating: set the minimum speaker-activity duration.
  pub fn set_min_activity_duration(&mut self, v: Duration) -> &mut Self {
    self.min_activity_duration = v;
    self
  }
  /// Mutating: set the voice-span merge gap.
  pub fn set_voice_merge_gap(&mut self, v: Duration) -> &mut Self {
    self.voice_merge_gap = v;
    self
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn defaults_match_pyannote() {
    let o = SegmentOptions::default();
    assert_eq!(o.onset_threshold(), 0.5);
    assert!((o.offset_threshold() - 0.357).abs() < 1e-6);
    assert_eq!(o.step_samples(), 40_000);
    assert_eq!(o.min_voice_duration(), Duration::ZERO);
  }

  #[test]
  fn builder_round_trip() {
    let o = SegmentOptions::new()
      .with_onset_threshold(0.6)
      .with_offset_threshold(0.4)
      .with_step_samples(20_000)
      .with_min_voice_duration(Duration::from_millis(100))
      .with_min_activity_duration(Duration::from_millis(50))
      .with_voice_merge_gap(Duration::from_millis(30));

    assert_eq!(o.onset_threshold(), 0.6);
    assert_eq!(o.offset_threshold(), 0.4);
    assert_eq!(o.step_samples(), 20_000);
    assert_eq!(o.min_voice_duration(), Duration::from_millis(100));
    assert_eq!(o.min_activity_duration(), Duration::from_millis(50));
    assert_eq!(o.voice_merge_gap(), Duration::from_millis(30));
  }

  #[test]
  fn sample_rate_tb_matches_constant() {
    assert_eq!(SAMPLE_RATE_TB.den().get(), SAMPLE_RATE_HZ);
    assert_eq!(SAMPLE_RATE_TB.num(), 1);
  }

  #[test]
  #[should_panic(expected = "step_samples must be > 0")]
  fn with_step_samples_zero_panics() {
    let _ = SegmentOptions::default().with_step_samples(0);
  }

  #[test]
  #[should_panic(expected = "step_samples must be > 0")]
  fn set_step_samples_zero_panics() {
    let mut o = SegmentOptions::default();
    o.set_step_samples(0);
  }

  /// `step > WINDOW_SAMPLES` causes silent gaps of `step - window`
  /// samples between consecutive chunks (the regular-grid loop skips
  /// past audio that the tail-anchor step does not re-cover). Reject
  /// at the option boundary so this cannot reach the planner.
  #[test]
  #[should_panic(expected = "step_samples must be <= WINDOW_SAMPLES")]
  fn with_step_samples_above_window_panics() {
    let _ = SegmentOptions::default().with_step_samples(WINDOW_SAMPLES + 1);
  }

  #[test]
  #[should_panic(expected = "step_samples must be <= WINDOW_SAMPLES")]
  fn set_step_samples_above_window_panics() {
    let mut o = SegmentOptions::default();
    o.set_step_samples(WINDOW_SAMPLES + 1);
  }

  /// Boundary: step == WINDOW_SAMPLES is a valid no-overlap config.
  #[test]
  fn step_equal_to_window_ok() {
    let o = SegmentOptions::default().with_step_samples(WINDOW_SAMPLES);
    assert_eq!(o.step_samples(), WINDOW_SAMPLES);
  }

  //: hysteresis threshold setters reject invalid
  // values. With NaN, every `p >= threshold` is false → segmenter
  // permanently silent. With > 1.0, same effect. With < 0.0, every
  // probability is "active" → segmenter permanently active. With
  // offset > onset, no run can ever close.

  #[test]
  #[should_panic(expected = "onset_threshold must be finite in [0.0, 1.0]")]
  fn onset_threshold_nan_panics() {
    let _ = SegmentOptions::default().with_onset_threshold(f32::NAN);
  }

  #[test]
  #[should_panic(expected = "onset_threshold must be finite in [0.0, 1.0]")]
  fn onset_threshold_inf_panics() {
    let _ = SegmentOptions::default().with_onset_threshold(f32::INFINITY);
  }

  #[test]
  #[should_panic(expected = "onset_threshold must be finite in [0.0, 1.0]")]
  fn onset_threshold_above_one_panics() {
    let _ = SegmentOptions::default().with_onset_threshold(1.01);
  }

  #[test]
  #[should_panic(expected = "onset_threshold must be finite in [0.0, 1.0]")]
  fn onset_threshold_below_zero_panics() {
    let _ = SegmentOptions::default().with_onset_threshold(-0.01);
  }

  #[test]
  #[should_panic(expected = "offset_threshold must be finite in [0.0, 1.0]")]
  fn offset_threshold_nan_panics() {
    let _ = SegmentOptions::default().with_offset_threshold(f32::NAN);
  }

  #[test]
  #[should_panic(expected = "offset_threshold must be finite in [0.0, 1.0]")]
  fn offset_threshold_neg_inf_panics() {
    let _ = SegmentOptions::default().with_offset_threshold(f32::NEG_INFINITY);
  }

  /// `with_offset_threshold(0.6)` after the default onset of 0.5
  /// should reject the invariant violation `offset (0.6) > onset (0.5)`.
  #[test]
  #[should_panic(expected = "offset_threshold must be <= onset_threshold")]
  fn offset_above_onset_panics() {
    let _ = SegmentOptions::default().with_offset_threshold(0.6);
  }

  /// Lowering the onset below the current offset must also be rejected.
  #[test]
  #[should_panic(expected = "offset_threshold must remain <= onset_threshold")]
  fn lowering_onset_below_offset_panics() {
    // Default: onset=0.5, offset=0.357. Lowering onset to 0.3 puts
    // it below the current offset.
    let _ = SegmentOptions::default().with_onset_threshold(0.3);
  }

  /// Boundary 0.0 = 0.0 = 0.0 is degenerate but valid (everything always active).
  #[test]
  fn onset_offset_zero_zero_ok() {
    let o = SegmentOptions::new()
      .with_offset_threshold(0.0)
      .with_onset_threshold(0.0);
    assert_eq!(o.onset_threshold(), 0.0);
    assert_eq!(o.offset_threshold(), 0.0);
  }

  /// Equal onset = offset (degenerate but valid).
  #[test]
  fn onset_equals_offset_ok() {
    let o = SegmentOptions::new()
      .with_onset_threshold(0.7)
      .with_offset_threshold(0.7);
    assert_eq!(o.onset_threshold(), 0.7);
    assert_eq!(o.offset_threshold(), 0.7);
  }

  #[test]
  #[should_panic(expected = "onset_threshold must be finite in [0.0, 1.0]")]
  fn set_onset_threshold_validates() {
    let mut o = SegmentOptions::default();
    o.set_onset_threshold(f32::NAN);
  }

  #[test]
  #[should_panic(expected = "offset_threshold must be finite in [0.0, 1.0]")]
  fn set_offset_threshold_validates() {
    let mut o = SegmentOptions::default();
    o.set_offset_threshold(f32::INFINITY);
  }
}
