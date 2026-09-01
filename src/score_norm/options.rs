//! Options for [`CohortStats`](super::CohortStats) construction.

use core::num::NonZeroUsize;

/// Default cohort-truncation size — the `--top_n` WeSpeaker's
/// `wespeaker/bin/score_norm.py` recipe passes for AS-Norm.
///
/// The "adaptive" in AS-Norm *is* this truncation: the cohort is not used
/// whole, only its `top_n` highest-scoring members against the side being
/// normalized.
pub const DEFAULT_TOP_N: usize = 300;

/// Default lower bound on a side's cohort standard deviation.
///
/// Cohort scores in this crate are cosine similarities in `[-1, 1]`
/// ([`Embedding::similarity`](crate::embed::Embedding::similarity)), so a
/// spread below `1e-6` means the selected cohort scores are identical to
/// within f32 rounding and the `(s - mean) / deviation` division has no
/// meaningful denominator. Callers scoring on a wider scale (PLDA
/// log-likelihood ratios range over tens) should raise this to match.
pub const DEFAULT_MIN_DEVIATION: f64 = 1e-6;

/// Fewest usable cohort scores a side needs before its statistics are
/// meaningful.
///
/// A single score has no dispersion at all — its population standard
/// deviation is exactly `0` — so one usable cohort member can never
/// produce a usable side. Refusing it up front gives
/// [`Error::CohortTooSmall`](super::Error::CohortTooSmall), which names
/// the real problem, instead of
/// [`Error::DegenerateDeviation`](super::Error::DegenerateDeviation),
/// which would suggest a pathological cohort rather than an absent one.
pub const MIN_COHORT_SCORES: usize = 2;

/// Tunables for the per-side cohort statistics AS-Norm is built on.
///
/// Both knobs are properties of the *score distribution*, not of whatever
/// produced the scores — which is why nothing here mentions embeddings,
/// their dimension, or the scoring function.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AsNormOptions {
  #[cfg_attr(feature = "serde", serde(default = "default_top_n"))]
  top_n: NonZeroUsize,
  #[cfg_attr(feature = "serde", serde(default = "default_min_deviation"))]
  min_deviation: f64,
}

#[cfg(feature = "serde")]
fn default_top_n() -> NonZeroUsize {
  // `DEFAULT_TOP_N` is a non-zero literal; the expect is unreachable.
  NonZeroUsize::new(DEFAULT_TOP_N).expect("DEFAULT_TOP_N is non-zero")
}

#[cfg(feature = "serde")]
const fn default_min_deviation() -> f64 {
  DEFAULT_MIN_DEVIATION
}

impl Default for AsNormOptions {
  /// [`DEFAULT_TOP_N`] / [`DEFAULT_MIN_DEVIATION`].
  fn default() -> Self {
    Self {
      top_n: NonZeroUsize::new(DEFAULT_TOP_N).expect("DEFAULT_TOP_N is non-zero"),
      min_deviation: DEFAULT_MIN_DEVIATION,
    }
  }
}

impl AsNormOptions {
  /// Construct with the defaults (see [`Default`]).
  pub fn new() -> Self {
    Self::default()
  }

  // ── Accessors ────────────────────────────────────────────────────────

  /// How many of the highest cohort scores feed a side's mean and
  /// standard deviation.
  ///
  /// [`NonZeroUsize`] because a zero-member selection has no mean; the
  /// type removes the check rather than deferring it to a runtime error.
  /// A value *larger* than the cohort is not an error — see
  /// [`CohortStats::selected`](super::CohortStats::selected).
  pub fn top_n(&self) -> NonZeroUsize {
    self.top_n
  }

  /// Smallest standard deviation accepted for a side.
  ///
  /// A side whose selected scores spread less than this is refused with
  /// [`Error::DegenerateDeviation`](super::Error::DegenerateDeviation).
  pub fn min_deviation(&self) -> f64 {
    self.min_deviation
  }

  // ── Builders (consuming with_*) ──────────────────────────────────────

  /// Set the cohort truncation size (builder).
  pub fn with_top_n(mut self, top_n: NonZeroUsize) -> Self {
    self.top_n = top_n;
    self
  }

  /// Set the standard-deviation floor (builder).
  ///
  /// # Panics
  /// Panics if `min_deviation` is not finite and strictly positive. A
  /// floor of `0` would re-admit the division-by-zero this guard exists
  /// to refuse, and a `NaN` floor makes every comparison against it
  /// false, silently disabling the guard.
  pub fn with_min_deviation(mut self, min_deviation: f64) -> Self {
    assert!(
      min_deviation.is_finite() && min_deviation > 0.0,
      "min_deviation must be finite and > 0.0; got {min_deviation}"
    );
    self.min_deviation = min_deviation;
    self
  }
}
