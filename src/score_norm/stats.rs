//! Per-side cohort statistics ([`CohortStats`]) and the AS-Norm
//! combination ([`as_norm`]).

use crate::{
  ops::kahan_sum,
  score_norm::{
    AsNormOptions, Error,
    error::{CohortTooSmall, DegenerateDeviation},
    options::MIN_COHORT_SCORES,
  },
};

/// One side of a trial: the mean and standard deviation of that side's
/// top-N cohort scores.
///
/// This is the value AS-Norm is built on, and the unit of reuse. A side
/// depends only on *itself* and the cohort — never on the other side of
/// the trial — so it is computed once per speaker and reused across every
/// trial that speaker takes part in. See the [module
/// docs](crate::score_norm#cost) for what that saves.
///
/// `Copy` and four machine words wide, so caching one per speaker in a
/// `HashMap` costs nothing worth measuring.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CohortStats {
  mean: f64,
  deviation: f64,
  selected: usize,
  considered: usize,
}

impl CohortStats {
  /// Build a side from its scores against the cohort.
  ///
  /// The scores are this side's `s(e, ε)` for every cohort member `ε` it
  /// was compared against — nothing else about `e` or `ε` reaches this
  /// function, which is what keeps the whole surface free of any
  /// embedding dimension.
  ///
  /// The `top_n` highest scores are selected (Matějka et al. 2017,
  /// eq. 6), then reduced to a mean and a **population** standard
  /// deviation — divisor `N`, matching WeSpeaker's unparameterised
  /// `np.std`, whose `ddof` defaults to `0`.
  ///
  /// # Selection
  ///
  /// `top_n` larger than the number of scores is **not** an error: the
  /// whole set is used, which degrades AS-Norm to plain S-Norm. This
  /// matches the reference implementation, where `[:, :top_n]` is a
  /// NumPy slice and silently truncates. [`Self::selected`] and
  /// [`Self::considered`] report whether that happened, so a caller who
  /// cares can detect an under-sized cohort instead of guessing.
  ///
  /// Selection is a linear-time partial partition ([`slice::select_nth_unstable_by`]),
  /// not a sort: only the *membership* of the top-N affects the mean and
  /// deviation, never its order.
  ///
  /// # Numerics
  ///
  /// The mean and the sum of squared deviations are both accumulated
  /// with the crate's Neumaier-compensated [`ops::kahan_sum`](crate::ops::kahan_sum),
  /// and the variance is computed in **two passes** — mean first, then
  /// `Σ(x − mean)²`. The one-pass `E[x²] − E[x]²` form cancels
  /// catastrophically for cohorts that are tightly clustered far from
  /// zero, which is not an exotic case but the *normal* one: AS-Norm
  /// selects the top of the distribution, so the selected scores are
  /// tightly clustered by construction.
  ///
  /// # Errors
  ///
  /// - [`Error::NonFiniteScore`] — any score is `NaN` or `±inf`. Checked
  ///   before selection, so a `NaN` cannot silently win or lose the
  ///   top-N comparison.
  /// - [`Error::EmptyCohort`] — no scores at all.
  /// - [`Error::CohortTooSmall`] — fewer than [`MIN_COHORT_SCORES`](crate::score_norm::MIN_COHORT_SCORES)
  ///   scores survive selection. Reachable two ways: a cohort that small,
  ///   or a `top_n` that small.
  /// - [`Error::DegenerateDeviation`] — the selected scores spread less
  ///   than [`AsNormOptions::min_deviation`].
  /// - [`Error::InvalidMinDeviation`] — that floor is itself `0`,
  ///   negative or non-finite, which would disable the guard above.
  ///   Reachable only by deserializing an options value, which bypasses
  ///   [`AsNormOptions::with_min_deviation`]'s assertion.
  pub fn from_scores<I>(scores: I, options: &AsNormOptions) -> Result<Self, Error>
  where
    I: IntoIterator<Item = f64>,
  {
    // Validate the floor before anything else. `with_min_deviation`
    // asserts it, but a serde-deserialized config reads straight into the
    // field; a `0`, negative or `NaN` floor makes the `deviation < floor`
    // comparison below false for every deviation, silently disabling the
    // one guard standing between a zero-spread cohort and a division by
    // zero.
    let min_deviation = options.min_deviation();
    if !min_deviation.is_finite() || min_deviation <= 0.0 {
      return Err(Error::InvalidMinDeviation(min_deviation));
    }

    let iter = scores.into_iter();
    let mut buf: Vec<f64> = Vec::with_capacity(iter.size_hint().0);
    for score in iter {
      // Reject before selection: `total_cmp` gives NaN a well-defined
      // sort position rather than propagating it, so a NaN would be
      // silently ordered into (or out of) the top-N and only surface as
      // a NaN mean much later.
      if !score.is_finite() {
        return Err(Error::NonFiniteScore(score));
      }
      buf.push(score);
    }

    let considered = buf.len();
    if considered == 0 {
      return Err(Error::EmptyCohort);
    }

    // Clamp rather than reject — see the `# Selection` note above.
    let selected = options.top_n().get().min(considered);
    if selected < MIN_COHORT_SCORES {
      return Err(Error::CohortTooSmall(CohortTooSmall::new(
        selected,
        MIN_COHORT_SCORES,
      )));
    }
    if selected < considered {
      // Partition so `buf[..selected]` holds the `selected` largest.
      // `total_cmp` is a total order on f64 and every element is already
      // known finite, so the reversed comparison is a plain descending
      // sort key.
      buf.select_nth_unstable_by(selected, |a, b| b.total_cmp(a));
      buf.truncate(selected);
    }

    let n = selected as f64;
    let mean = kahan_sum(&buf) / n;
    // Two-pass: square the mean-shifted values in place, so the second
    // compensated sum needs no extra allocation.
    for v in &mut buf {
      let d = *v - mean;
      *v = d * d;
    }
    // Non-negative by construction — a sum of squares — so `sqrt` cannot
    // produce NaN here the way the one-pass form can.
    let deviation = (kahan_sum(&buf) / n).sqrt();

    if !deviation.is_finite() || deviation < min_deviation {
      return Err(Error::DegenerateDeviation(DegenerateDeviation::new(
        deviation,
        min_deviation,
        selected,
      )));
    }

    Ok(Self {
      mean,
      deviation,
      selected,
      considered,
    })
  }

  // ── Accessors ────────────────────────────────────────────────────────

  /// Mean of the selected top-N cohort scores — `μ` in the AS-Norm
  /// equation.
  pub const fn mean(&self) -> f64 {
    self.mean
  }

  /// Population standard deviation of the selected top-N cohort scores —
  /// `σ` in the AS-Norm equation. Guaranteed finite and at least
  /// [`AsNormOptions::min_deviation`], so the division in
  /// [`Self::normalize`] cannot blow up.
  pub const fn deviation(&self) -> f64 {
    self.deviation
  }

  /// How many scores the statistics were actually computed over.
  ///
  /// `min(top_n, considered)`. Compare against [`Self::considered`] to
  /// detect a cohort smaller than `top_n`, where AS-Norm has silently
  /// degraded to S-Norm.
  pub const fn selected(&self) -> usize {
    self.selected
  }

  /// How many scores were offered, before top-N selection.
  ///
  /// With [`Cohort::stats_excluding`](crate::score_norm::Cohort::stats_excluding)
  /// this is the cohort size *after* self-exclusion, so it is also how a
  /// caller confirms that exclusion removed what it expected to.
  pub const fn considered(&self) -> usize {
    self.considered
  }

  // ── The normalization itself ─────────────────────────────────────────

  /// AS-Norm the trial score `raw` between this side and `other`.
  ///
  /// ```text
  /// 0.5 * ( (raw - μ_self) / σ_self  +  (raw - μ_other) / σ_other )
  /// ```
  ///
  /// Symmetric in the two sides, so which one is "enrollment" and which
  /// is "test" does not matter.
  ///
  /// Each term is computed shift-then-divide — the numerator subtraction
  /// happens first — rather than as `raw/σ − μ/σ`, which would form two
  /// large quantities and subtract them.
  ///
  /// # Errors
  ///
  /// [`Error::NonFiniteScore`] if `raw` is `NaN` or `±inf`. Both sides
  /// are already validated at construction, so this is the only way the
  /// result could be non-finite.
  pub fn normalize(&self, raw: f64, other: &CohortStats) -> Result<f64, Error> {
    if !raw.is_finite() {
      return Err(Error::NonFiniteScore(raw));
    }
    Ok(0.5 * ((raw - self.mean) / self.deviation + (raw - other.mean) / other.deviation))
  }
}

/// Free-function form of [`CohortStats::normalize`] for callers who
/// prefer it, mirroring [`cosine_similarity`](crate::embed::cosine_similarity)
/// beside [`Embedding::similarity`](crate::embed::Embedding::similarity).
/// **Bit-exactly equivalent** to the method — it delegates.
///
/// The argument names follow the literature; the operation is symmetric,
/// so swapping them cannot change the result.
///
/// # Errors
///
/// See [`CohortStats::normalize`].
pub fn as_norm(raw: f64, enrollment: &CohortStats, test: &CohortStats) -> Result<f64, Error> {
  enrollment.normalize(raw, test)
}
