//! Error type for `crate::score_norm`.

use thiserror::Error;

/// Errors produced while building a side's cohort statistics or applying
/// them to a trial score.
///
/// Every variant is a refusal, never a degraded result: AS-Norm exists so
/// one threshold means the same thing for every speaker, and a side whose
/// statistics are silently wrong moves that threshold for exactly one
/// speaker — the failure mode hardest to notice downstream.
#[derive(Debug, Error)]
pub enum Error {
  /// No usable cohort scores for this side.
  ///
  /// Either the cohort was empty, or identity exclusion
  /// ([`Cohort::stats_excluding`](crate::score_norm::Cohort::stats_excluding))
  /// removed every member — which happens when the cohort holds only the
  /// speaker being normalized.
  #[error("score_norm: no usable cohort scores (empty cohort, or self-exclusion removed all)")]
  EmptyCohort,

  /// Fewer usable cohort scores than
  /// [`MIN_COHORT_SCORES`](crate::score_norm::MIN_COHORT_SCORES).
  #[error("score_norm: {0}")]
  CohortTooSmall(CohortTooSmall),

  /// A cohort score, or the trial score being normalized, was `NaN` or
  /// `±inf`.
  ///
  /// Carries the offending value. A non-finite score poisons the mean,
  /// the standard deviation and every normalized score derived from
  /// them, so it is rejected at the boundary rather than propagated.
  #[error("score_norm: score is not finite ({0})")]
  NonFiniteScore(f64),

  /// The selected top-N cohort scores are identical to within the
  /// configured floor, so `(s - mean) / deviation` has no usable
  /// denominator.
  #[error("score_norm: {0}")]
  DegenerateDeviation(DegenerateDeviation),

  /// [`AsNormOptions::min_deviation`](crate::score_norm::AsNormOptions::min_deviation)
  /// is not finite and strictly positive.
  ///
  /// [`with_min_deviation`](crate::score_norm::AsNormOptions::with_min_deviation)
  /// asserts this on the builder path, but a `#[serde(default)]`
  /// deserialize reads straight into the field and bypasses it. The value
  /// matters: the degenerate-standard-deviation guard is a `deviation <
  /// floor` comparison, and a floor of `0` makes it false for every
  /// non-negative deviation while a `NaN` floor makes it false for *all*
  /// of them. Either one silently disables the guard and lets a
  /// zero-deviation cohort divide by zero — producing exactly the
  /// poisoned score the guard exists to refuse. Checked before any
  /// statistics are computed.
  #[error("score_norm: min_deviation ({0}) must be finite and > 0.0")]
  InvalidMinDeviation(f64),
}

/// Payload of [`Error::CohortTooSmall`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CohortTooSmall {
  available: usize,
  required: usize,
}

impl CohortTooSmall {
  pub(crate) const fn new(available: usize, required: usize) -> Self {
    Self {
      available,
      required,
    }
  }

  /// Usable cohort scores this side actually had.
  pub const fn available(&self) -> usize {
    self.available
  }

  /// Fewest this side needed — [`MIN_COHORT_SCORES`](crate::score_norm::MIN_COHORT_SCORES).
  pub const fn required(&self) -> usize {
    self.required
  }
}

impl core::fmt::Display for CohortTooSmall {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    write!(
      f,
      "cohort has {} usable score(s), needs at least {}",
      self.available, self.required
    )
  }
}

/// Payload of [`Error::DegenerateDeviation`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DegenerateDeviation {
  deviation: f64,
  minimum: f64,
  selected: usize,
}

impl DegenerateDeviation {
  pub(crate) const fn new(deviation: f64, minimum: f64, selected: usize) -> Self {
    Self {
      deviation,
      minimum,
      selected,
    }
  }

  /// The standard deviation actually computed over the selected scores.
  pub const fn deviation(&self) -> f64 {
    self.deviation
  }

  /// The configured floor — [`AsNormOptions::min_deviation`](crate::score_norm::AsNormOptions::min_deviation).
  pub const fn minimum(&self) -> f64 {
    self.minimum
  }

  /// How many cohort scores the deviation was computed over.
  pub const fn selected(&self) -> usize {
    self.selected
  }
}

impl core::fmt::Display for DegenerateDeviation {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    write!(
      f,
      "standard deviation {:.3e} over {} selected score(s) is below the floor {:.3e}; \
       the cohort does not discriminate",
      self.deviation, self.selected, self.minimum
    )
  }
}
