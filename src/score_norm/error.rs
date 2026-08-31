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
  ///
  /// The payload's deviation is always a real, finite spread. An
  /// arithmetic blow-up is [`Self::NonFiniteResult`] instead — a variant
  /// whose name reports what happened, rather than this one claiming a
  /// degenerate cohort for a cohort that is merely large.
  #[error("score_norm: {0}")]
  DegenerateDeviation(DegenerateDeviation),

  /// A value this module *computed* came out non-finite even though every
  /// input to it was finite: a side's standard deviation, or a normalized
  /// trial score.
  ///
  /// Carries the offending value. Distinct from [`Self::NonFiniteScore`],
  /// which rejects a non-finite *input* — here the inputs were all sound
  /// and the arithmetic left f64's range anyway, which a caller may want
  /// to act on differently (rescale the scoring function, widen the
  /// deviation floor).
  ///
  /// # Why this is a refusal and not an `Ok`
  ///
  /// A normalized score is consumed by comparing it against a **fixed
  /// absolute threshold** — that is the entire reason AS-Norm keeps its
  /// `0.5` (see the [module docs](crate::score_norm)). `+inf` clears
  /// every such threshold, so returning it as `Ok` is an unconditional
  /// match: the one failure this module exists to prevent, arriving
  /// silently. `NaN` fails every comparison instead, which is an
  /// unconditional *non*-match — quieter still.
  #[error("score_norm: computed value is not finite ({0})")]
  NonFiniteResult(f64),

  /// The two sides' z-scores cancel so completely that the average would
  /// be made of their own rounding rather than of the data.
  ///
  /// `(raw − μ) / σ` costs two rounded operations, so each z-score is
  /// carried with a relative error of about `2^-52`. Averaging two of them
  /// keeps that error *absolutely* while the sum shrinks: for z-scores
  /// near `2^50` the discarded bits are worth `0.25`, and if the two sides
  /// land one ulp apart with opposite signs, `0.25` is the entire result.
  /// The returned value can then have the wrong sign — and every guard
  /// above it passes, because every intermediate is finite and every
  /// deviation is real.
  ///
  /// Refused rather than returned once
  /// [`ZScoreCancellation::error_bound`] exceeds
  /// [`ZScoreCancellation::tolerance`]; see
  /// [`MAX_NORMALIZED_ERROR`](crate::score_norm::MAX_NORMALIZED_ERROR) for
  /// what a successful normalization carries instead, and the
  /// [module docs](crate::score_norm#accuracy) for why the bound is sound.
  ///
  /// Reaching this needs a score source spanning ~`1e9` *and* a cohort
  /// deviation at the floor: cosine similarities live in `[-1, 1]` and
  /// PLDA log-likelihood ratios in the tens, so at the default floor the
  /// widest trial either can construct is still 2147 times short of it.
  #[error("score_norm: {0}")]
  ZScoreCancellation(ZScoreCancellation),

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

/// Payload of [`Error::ZScoreCancellation`].
///
/// Carries both standardized scores and the average they would have
/// produced, so a caller can see the cancellation itself rather than only
/// its verdict.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZScoreCancellation {
  z_self: f64,
  z_other: f64,
  normalized: f64,
}

impl ZScoreCancellation {
  pub(crate) const fn new(z_self: f64, z_other: f64, normalized: f64) -> Self {
    Self {
      z_self,
      z_other,
      normalized,
    }
  }

  /// The receiver's standardized trial score, `(raw − μ) / σ`.
  ///
  /// "Self" and "other" are
  /// [`CohortStats::normalize`](crate::score_norm::CohortStats::normalize)'s
  /// two sides; through [`as_norm`](crate::score_norm::as_norm) they are
  /// the enrollment and test sides in that order. The operation is
  /// symmetric, so which is which changes nothing but the label.
  pub const fn z_self(&self) -> f64 {
    self.z_self
  }

  /// The other side's standardized trial score.
  pub const fn z_other(&self) -> f64 {
    self.z_other
  }

  /// The average that was computed and refused.
  ///
  /// Kept because it is the thing under suspicion: a caller comparing it
  /// against [`Self::error_bound`] can see how much of it is data.
  pub const fn normalized(&self) -> f64 {
    self.normalized
  }

  /// How far [`Self::normalized`] may sit from the exact average of the
  /// two z-scores — the quantity that exceeded [`Self::tolerance`].
  pub fn error_bound(&self) -> f64 {
    super::stats::z_score_error_bound(self.z_self, self.z_other)
  }

  /// The largest error a *successful* normalization carries:
  /// [`MAX_NORMALIZED_ERROR`](crate::score_norm::MAX_NORMALIZED_ERROR)
  /// standard deviations, or that much relative once the result exceeds
  /// one.
  pub fn tolerance(&self) -> f64 {
    super::stats::permitted_error(self.normalized)
  }
}

impl core::fmt::Display for ZScoreCancellation {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    write!(
      f,
      "z-scores {:.6e} and {:.6e} cancel to {:.6e}, which their own rounding could move by \
       up to {:.3e}; a successful normalization carries at most {:.3e}",
      self.z_self,
      self.z_other,
      self.normalized,
      self.error_bound(),
      self.tolerance()
    )
  }
}
