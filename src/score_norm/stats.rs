//! Per-side cohort statistics ([`CohortStats`]) and the AS-Norm
//! combination ([`as_norm`]).

use crate::{
  ops::kahan_sum,
  score_norm::{
    AsNormOptions, Error,
    error::{CohortTooSmall, DegenerateDeviation, ZScoreCancellation},
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
  /// and the variance is computed in **two passes** — never the one-pass
  /// `E[x²] − E[x]²`, which cancels catastrophically for cohorts that are
  /// tightly clustered far from zero. That is not an exotic case but the
  /// *normal* one: AS-Norm selects the top of the distribution, so the
  /// selected scores are tightly clustered by construction.
  ///
  /// ## Dispersion is measured about a score the cohort contains
  ///
  /// The passes shift by a selected score before centring on the mean of
  /// the shifted values, rather than centring on `μ` directly. Dispersion
  /// is translation-invariant, so this changes nothing mathematically. It
  /// changes the degenerate case.
  ///
  /// `Σx / n` is a *rounded* mean, and for a cohort whose scores are all
  /// the same number it need not round back to that number. Five copies
  /// of `0x1.fffffffffffffp+33` sum to a value needing 55 significant
  /// bits, so the mean lands one ulp low and every score then "deviates"
  /// from it by the same `2^-19`. That is dispersion the cohort does not
  /// have; it clears the default `1e-6` floor, so the side is *accepted*,
  /// and normalizing that same score against it divides the fabricated
  /// deviation by itself and returns a perfect `1.0`.
  ///
  /// Shifting by a member first makes the constant case exactly zero **by
  /// construction**: `x − x` is `0` for every finite `x`, so the shifted
  /// set is all zeros, its mean is zero, and every squared deviation is
  /// zero. [`AsNormOptions::min_deviation`] is then a policy knob for
  /// cohorts that genuinely barely discriminate, rather than the last line
  /// of defence against arithmetic that invents a spread — which is a load
  /// it had already failed to carry.
  ///
  /// It is also more accurate in the regime AS-Norm actually runs in.
  /// Measured against an exact-rational reference over 400 cohorts per
  /// regime, worst-case error in `σ`: cosine scores of `1e-9` spread
  /// 137.8 → 1.07 ulp, this crate's own tight-cluster-at-`1e8` case
  /// 3.72 → 0.96 ulp, near-constant cohorts 1.6e16 → 1.28 ulp. Every
  /// other regime measured — looser cosine spreads, full-range cosine,
  /// PLDA-scale log-likelihood ratios — moves by at most 0.2 ulp in
  /// either direction. The tight end is the one that matters: top-N
  /// selection produces tightly clustered scores by construction, so it
  /// is where every real cohort lands.
  ///
  /// `μ` is untouched — still the direct compensated mean, bit for bit.
  /// Recovering it from the shifted domain as `anchor + shifted mean`
  /// instead would cancel the shift back out and cost it hundreds of ulps
  /// wherever the mean sits near zero, which is why the shift is confined
  /// to the dispersion pass that needs it.
  ///
  /// ## Range
  ///
  /// Every pass runs over scores divided by an exact power of two, and
  /// the scale is restored afterwards: the module's one [range-safety
  /// rule](crate::score_norm#range-safety), whose factor here is
  /// [`rescale_factor`]. That costs nothing — scaling by a power of two
  /// commutes with the rounding of every operation involved, so the
  /// statistics are bit-identical either way — and it makes the function
  /// **total over finite scores**. Unscaled, `Σ(x − mean)²` overflows for
  /// inputs whose deviation is representable, and `Σx` for inputs whose
  /// mean is; compensated summation cannot recover from either, because
  /// once a term is `inf` the compensation becomes `NaN`.
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
  /// - [`Error::NonFiniteResult`] — the computed deviation is not finite.
  ///   A postcondition, not a case: the rescale described above leaves no
  ///   finite input that can reach it.
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

    // Work in a rescaled domain where nothing can leave f64's range, then
    // restore the scale at the end. Bit-neutral, and it is what keeps the
    // arithmetic total — see `rescale_factor`.
    let scale = rescale_factor(&buf);
    for v in &mut buf {
      *v /= scale;
    }

    let mean_scaled = kahan_sum(&buf) / n;

    // Dispersion is translation-invariant, so measure it about a score the
    // cohort actually contains rather than about the rounded mean. `x - x`
    // is `0` for every finite `x`, so a cohort of one repeated score
    // spreads by exactly zero *by construction* — see `# Numerics` above
    // for the case that made this necessary. `selected` is at least
    // `MIN_COHORT_SCORES`, so index 0 exists.
    let anchor = buf[0];
    for v in &mut buf {
      *v -= anchor;
    }
    let anchored_mean = kahan_sum(&buf) / n;
    // Two-pass: square the mean-shifted values in place, so the second
    // compensated sum needs no extra allocation.
    for v in &mut buf {
      let d = *v - anchored_mean;
      *v = d * d;
    }
    // Non-negative by construction — a sum of squares — so `sqrt` cannot
    // produce NaN here the way the one-pass form can.
    let deviation_scaled = (kahan_sum(&buf) / n).sqrt();

    let mean = mean_scaled * scale;
    let deviation = deviation_scaled * scale;

    // Postcondition, not a case: no finite score set reaches it. Held
    // anyway because a non-finite `σ` escaping as `Ok` would divide into
    // `normalize` and produce a finite-looking `0.0` — a plausible wrong
    // answer, which is the one outcome this module refuses. A non-finite
    // `μ` needs no separate guard: it poisons every `*v - anchored_mean`,
    // and so the deviation with it.
    if !deviation.is_finite() {
      return Err(Error::NonFiniteResult(deviation));
    }
    if deviation < min_deviation {
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
  /// Two of the three steps can leave f64's range for values this
  /// module's own constructor accepts, and both are handled by the single
  /// [range-safety rule](crate::score_norm#range-safety) — a power-of-two
  /// factor chosen from the operands, `½` exactly when `1` will not do.
  /// [`Self::z_score`] carries the shift, [`half_sum`] the average; each
  /// documents its own case.
  ///
  /// Neither factor is applied unconditionally, and that is the whole
  /// point. Halving `raw` and `μ` on *every* call also keeps the shift in
  /// range — and rounds a subnormal z-score to zero on the way. For
  /// `μ = -u` and `σ = u`, with `u` the smallest positive subnormal, each
  /// side's exact z-score for `raw = u` is `2`; but `0.5 * u` and
  /// `0.5 * -u` are both a signed zero, so the numerator collapses and the
  /// answer comes back as `0.0`. Finite, and wrong — which the
  /// postcondition below cannot see, because being finite is all it
  /// checks.
  ///
  /// # Accuracy, and the second postcondition
  ///
  /// Being finite is not the same as being right. Each z-score costs two
  /// rounded operations, so it is carried with a relative error of about
  /// `2^-52` — negligible against itself, and *not* negligible against
  /// their average once the two cancel. Averaging keeps that error at its
  /// absolute size while the sum shrinks, so for z-scores large enough the
  /// discarded bits are the whole answer, sign included. A successful
  /// return therefore also satisfies
  ///
  /// ```text
  /// |returned − exact| ≤ MAX_NORMALIZED_ERROR * max(|returned|, 1)
  /// ```
  ///
  /// where `exact` is the exact average of the two z-scores the stored
  /// statistics define. Anything looser is [`Error::ZScoreCancellation`].
  /// Nothing here changes a value, so every answer this returns is the one
  /// it always returned.
  ///
  /// The test is in two tiers, and only the first is on the common path.
  /// [`z_score_error_bound`] costs one comparison and asks what the
  /// operands *permit*; clearing it settles the question, and every trial
  /// any score source here produces clears it. Exceeding it settles
  /// nothing — a trial whose two divisions come out exact permits an error
  /// it did not make — so those trials go to [`refined_error_bound`],
  /// which recovers the roundings that actually happened and refuses only
  /// what they convict. See the [module docs](crate::score_norm#accuracy)
  /// for the identity behind the second tier and the measured separation
  /// between the two regimes.
  ///
  /// # Errors
  ///
  /// - [`Error::NonFiniteScore`] — `raw` is `NaN` or `±inf`.
  /// - [`Error::NonFiniteResult`] — the normalized score is not finite
  ///   although `raw` and both sides were: a quotient that genuinely
  ///   exceeds [`f64::MAX`], which a deviation floor small enough to admit
  ///   a near-zero `σ` makes reachable. A successful return is therefore
  ///   always finite, which is what a consumer comparing against a fixed
  ///   absolute threshold needs: `inf` clears every threshold, so an
  ///   `Ok(inf)` would be an unconditional match.
  /// - [`Error::ZScoreCancellation`] — the two z-scores cancel past the
  ///   accuracy above, as their own recovered residuals confirm. At the
  ///   default deviation floor no bounded score source reaches even the
  ///   first tier: a cosine similarity or a PLDA log-likelihood ratio
  ///   falls short by three orders of magnitude.
  pub fn normalize(&self, raw: f64, other: &CohortStats) -> Result<f64, Error> {
    if !raw.is_finite() {
      return Err(Error::NonFiniteScore(raw));
    }
    let z_self = self.z_score(raw);
    let z_other = other.z_score(raw);
    let normalized = half_sum(z_self, z_other);
    if !normalized.is_finite() {
      return Err(Error::NonFiniteResult(normalized));
    }
    let tolerance = permitted_error(normalized);
    // Tier one is a filter, not a verdict. `z_score_error_bound` is an
    // *upper* bound, so clearing it proves the answer sound — but
    // exceeding it proves only that the answer *might* be unsound, which
    // is a reason to look closer and not a reason to refuse. Every trial
    // any score source here produces clears it, and clearing it costs the
    // one comparison it always cost.
    let filtered = z_score_error_bound(z_self, z_other);
    if filtered > tolerance {
      // Tier two, reached only by the trials tier one could not clear.
      // The residuals turn the bound into the error itself, so what is
      // refused here is an answer that really is made of its own
      // rounding rather than one whose operands merely allow it to be.
      let refined = refined_error_bound(raw, self, other, z_self, z_other, filtered);
      if refined > tolerance {
        return Err(Error::ZScoreCancellation(ZScoreCancellation::new(
          z_self, z_other, normalized, refined,
        )));
      }
    }
    Ok(normalized)
  }

  /// This side's standardized trial score, `(raw − μ) / σ`.
  ///
  /// The shift is formed first — not `raw/σ − μ/σ`, which builds two large
  /// quantities and subtracts them. Where that shift leaves f64's range,
  /// the [range-safety rule](crate::score_norm#range-safety) applies with
  /// a factor of `½`, and that branch is reachable only where the halving
  /// is exact: an overflowing difference leaves no room for a small
  /// operand — neither can fall below `2^970`, or the subtraction would
  /// have fitted — so both are large normals and `0.5 *` is exact on each.
  /// Folding the same `½` into `σ` accounts for the factor: `(x/2) / (y/2)`
  /// has the exact quotient `x / y` and rounds identically, so the halved
  /// form returns precisely what the direct one would have returned had it
  /// fitted.
  ///
  /// `0.5 * self.deviation` can itself round — `σ` may be subnormal under
  /// a lowered floor — but only on a path whose answer does not exist:
  /// a numerator past `f64::MAX / 2` over a subnormal `σ` exceeds
  /// `f64::MAX` by hundreds of orders of magnitude, so the branch returns
  /// an infinity that [`Self::normalize`] refuses, which is the correct
  /// outcome however precisely the divisor was formed.
  fn z_score(&self, raw: f64) -> f64 {
    self.z_terms(raw).quotient()
  }

  /// The operands [`Self::z_score`] rounds, kept apart so the roundings
  /// can be recovered afterwards.
  ///
  /// The branch is the one above and the arithmetic is the same
  /// arithmetic — `a - b` and `a + (-b)` round identically — so
  /// [`ZTerms::quotient`] returns `z_score`'s value bit for bit. What the
  /// split adds is that `minuend` and `subtrahend` are *exact*: the
  /// rescaled branch's halvings land on large normals, so both are exact
  /// there too, and a two-sum over the pair therefore recovers the
  /// subtraction's residual rather than a residual of an already-rounded
  /// shift.
  fn z_terms(&self, raw: f64) -> ZTerms {
    let shift = raw - self.mean;
    if shift.is_finite() {
      return ZTerms {
        shift,
        minuend: raw,
        subtrahend: -self.mean,
        deviation: self.deviation,
      };
    }
    let (minuend, subtrahend) = (0.5 * raw, -(0.5 * self.mean));
    ZTerms {
      shift: minuend + subtrahend,
      minuend,
      subtrahend,
      deviation: 0.5 * self.deviation,
    }
  }
}

/// One side's z-score as the exact operands behind it: the quotient is
/// `(minuend + subtrahend) / deviation`, and every field is a float that
/// carries no error of its own.
///
/// [`CohortStats::z_score`] rounds twice — once forming `shift`, once
/// dividing — and the whole of [`refined_error_bound`] is recovering
/// those two roundings. Neither is recoverable from `shift` alone, which
/// is why the pair that produced it travels with it.
#[derive(Debug, Clone, Copy)]
struct ZTerms {
  /// `fl(minuend + subtrahend)` — the rounded numerator.
  shift: f64,
  /// The trial score, halved on the rescaled branch.
  minuend: f64,
  /// The negated cohort mean, halved on the rescaled branch.
  subtrahend: f64,
  /// The cohort deviation, halved on the rescaled branch.
  deviation: f64,
}

impl ZTerms {
  /// The rounded quotient — [`CohortStats::z_score`]'s value.
  fn quotient(self) -> f64 {
    self.shift / self.deviation
  }

  /// `z − q`: everything the two roundings threw away, for the `q` this
  /// side produced.
  ///
  /// Both residuals are recovered **exactly**. Knuth's two-sum gives
  /// `(minuend + subtrahend) − shift`, and `mul_add` gives
  /// `shift − q·σ` — the classical result that a division's residual is
  /// representable, so the fused form rounds nothing. The exact quotient
  /// is then `q + (r + ρ)/σ`, and this returns the second term.
  ///
  /// Also returned is the floor beneath it: the one underflow in the
  /// refinement that does not stay negligible, since it is the one that
  /// gets divided by `σ`. See [`RESIDUAL_UNDERFLOW`].
  fn correction(self, quotient: f64) -> (f64, f64) {
    let shift_residual = two_sum_residual(self.minuend, self.subtrahend);
    let quotient_residual = (-quotient).mul_add(self.deviation, self.shift);
    (
      (shift_residual + quotient_residual) / self.deviation,
      RESIDUAL_UNDERFLOW / self.deviation,
    )
  }
}

/// Knuth's two-sum residual: the part of `a + b` that `fl(a + b)` could
/// not keep.
///
/// `a + b == fl(a + b) + two_sum_residual(a, b)` exactly, for finite
/// operands whose sum is finite — including through gradual underflow,
/// and with no ordering requirement on the two magnitudes. Where the sum
/// is *not* finite the intermediates below produce a `NaN`, which
/// [`refined_error_bound`] turns back into the bound that selected it.
fn two_sum_residual(a: f64, b: f64) -> f64 {
  let sum = a + b;
  let b_part = sum - a;
  (a - (sum - b_part)) + (b - b_part)
}

/// The average of two standardized scores — AS-Norm1's `1/2`.
///
/// The [range-safety rule](crate::score_norm#range-safety) again, with the
/// same `½`: two terms of `1e308` average to a representable `1e308`, but
/// their sum is `inf` and half of `inf` is `inf`. As with the shift, an
/// overflowing sum leaves no room for a small term — neither can fall
/// below `2^970` — so both are large normals and halving each first is
/// exact.
///
/// Applying the factor unconditionally is what must not happen. `0.5 * a`
/// rounds a *subnormal* term to a signed zero, so two z-scores of `u`, the
/// smallest positive subnormal, would average to `0.0` instead of `u`.
/// Where the sum fits, `0.5 * (a + b)` is used instead — the sum of two
/// subnormals is exact, and it is also the naive form the literature
/// writes, so a threshold taken from a published AS-Norm number transfers
/// bit for bit.
fn half_sum(a: f64, b: f64) -> f64 {
  let sum = a + b;
  if sum.is_finite() {
    return 0.5 * sum;
  }
  0.5 * a + 0.5 * b
}

/// The largest error a successful [`CohortStats::normalize`] carries, in
/// the unit its result is measured in: **one cohort standard deviation**.
///
/// `2^-20`. A successful normalization lands within
/// `MAX_NORMALIZED_ERROR * max(|result|, 1)` of the exact average of the
/// two z-scores the stored statistics define — absolute while the result
/// is at most one standard deviation, relative beyond. Anything looser is
/// [`Error::ZScoreCancellation`] instead of an `Ok`; see the [module
/// docs](crate::score_norm#accuracy) for the bound behind it.
///
/// # Why this size
///
/// Three things pin it, and they pin it from both directions.
///
/// - **No threshold can see it.** AS-Norm exists so that a *fixed
///   absolute* threshold means the same thing for every speaker, and such
///   thresholds are `O(1)` in these units. `2^-20` is `9.5e-7` standard
///   deviations: six orders of magnitude below the smallest threshold
///   anyone sets.
/// - **The statistics are not that sharp themselves.** `μ` and `σ` are
///   sample statistics over [`CohortStats::selected`] scores, so `μ`
///   alone carries a standard error of `σ/√N` — `1/√N` standard
///   deviations, which is `0.058` at the recommended `top_n` of 300 and
///   `0.71` at [`MIN_COHORT_SCORES`](crate::score_norm::MIN_COHORT_SCORES).
///   The arithmetic tolerance sits some 60 000 times below the
///   statistical one. Tightening it would buy nothing a caller could
///   measure and would start refusing answers whose real uncertainty is
///   somewhere else entirely.
/// - **It leaves the reachable range alone.** The first tier's bound is
///   `2^-51` per unit of z-score magnitude, so this tolerance answers
///   *every* trial whose two z-scores average below `2^31 ≈ 2.1e9`
///   outright, however completely they cancel — an exactly zero result
///   included — and answers the ones above it whenever their residuals
///   say the answer is real. Cosine similarities span `[-1, 1]`, so at the
///   [`DEFAULT_MIN_DEVIATION`](crate::score_norm::DEFAULT_MIN_DEVIATION)
///   floor they cannot standardize past `2e6`: two thousandfold clear of
///   even the first tier in the widest trial that floor admits at all, and
///   six to nine orders of magnitude clear at any realistic cohort spread.
///   The [module docs](crate::score_norm#accuracy) carry the measured
///   figures in both directions.
pub const MAX_NORMALIZED_ERROR: f64 = 1.0 / (1u64 << 20) as f64;

/// Bound on the relative error a single z-score is carried with.
///
/// `(raw − μ) / σ` is two rounded operations, so the computed `q` sits
/// within `2u(1 + 2u)` of the exact quotient with `u = 2^-53`. This
/// constant is `2^-51`, twice that, and the doubling is what makes the
/// bound usable as written: the second-order terms, the average's own
/// rounding, and a quotient that lands in the subnormals all fit inside
/// it rather than needing to be carried beside it.
const Z_SCORE_RELATIVE_ERROR: f64 = 2.0 * f64::EPSILON;

/// How far [`CohortStats::normalize`]'s returned value can sit from the
/// exact average of the two z-scores it stands for, from the **operands
/// alone** — the guard's first tier.
///
/// A bound, not an estimate — see the [module
/// docs](crate::score_norm#accuracy). It is worth being explicit about
/// what it does *not* depend on: not on how badly the two sides cancelled,
/// not on the returned value, and **not on whether the roundings it
/// budgets for actually happened**. That last one is why it can only
/// filter. It answers "how wrong could this be", so clearing it settles
/// the question; exceeding it does not, because a trial whose two
/// divisions are exact carries none of the error this charges it for.
/// [`refined_error_bound`] is what settles the other direction.
pub(super) fn z_score_error_bound(z_self: f64, z_other: f64) -> f64 {
  // `(|a| + |b|) / 2`, halved term by term. The average of two z-scores
  // can be representable where their sum is not — that is the case
  // `half_sum` exists for — so the magnitude must be formed the same way.
  // Halving before the sum is exact for normals; for subnormals it can
  // round, and cannot matter: a magnitude that small is `2^-1021` away
  // from a tolerance whose floor is `MAX_NORMALIZED_ERROR`.
  Z_SCORE_RELATIVE_ERROR * (0.5 * z_self.abs() + 0.5 * z_other.abs())
}

/// Slack on the refined figure, covering the six roundings that combine
/// the recovered residuals into it.
///
/// `2^-50`, eight times `u = 2^-53`, against a needed `4.01`. Writing
/// `Σ = |s| + |c₁| + |c₂|` for the quantity this multiplies (before the
/// halving both it and the figure share):
///
/// - each side's correction costs an addition and a division, both
///   relative to that side's own `|cᵢ|`, so the two sides together
///   contribute at most `2.01u(|c₁| + |c₂|)`;
/// - the two additions that sum the three residuals are relative to
///   partial sums no larger than `Σ`, contributing at most `2uΣ`.
///
/// `4.01uΣ` in total, and the halving is exact. The near-doubling to `8u`
/// leaves the same room for second-order terms that
/// [`Z_SCORE_RELATIVE_ERROR`]'s doubling does, and it is what keeps the
/// figure above the error even when `Σ` is itself computed a rounding
/// low.
const REFINED_RESIDUAL_SLACK: f64 = 4.0 * f64::EPSILON;

/// The largest absolute error `mul_add` can leave in a division residual
/// it cannot represent: the smallest positive subnormal, `2^-1074`.
///
/// `shift − q·σ` is exactly representable *while it stays normal*, which
/// is the classical result [`ZTerms::correction`] rests on. Below that the
/// fused multiply-add still rounds only once — but into the subnormals,
/// and the correction then divides by `σ`, which is what stops this being
/// one more `η` to wave at. Carried explicitly instead: it contributes
/// `2^-1054` at [`DEFAULT_MIN_DEVIATION`](crate::score_norm::DEFAULT_MIN_DEVIATION),
/// and `1` for a `σ` at the very bottom of the range — where it refuses,
/// correctly, because nothing about such a trial can be certified.
///
/// Every other underflow in the refinement stays where it is made: the
/// two-sum is exact through gradual underflow, adding two subnormals is
/// exact, and a quotient or a halving that lands among them is off by at
/// most `2^-1075` *absolutely* — some thousand binades below the
/// tolerance's own floor of `2^-20`.
const RESIDUAL_UNDERFLOW: f64 = f64::MIN_POSITIVE * f64::EPSILON;

/// The error [`CohortStats::normalize`] **actually** carries — the
/// guard's second tier, and the one that decides.
///
/// [`z_score_error_bound`] charges each z-score for two roundings it may
/// not have made. This recovers them instead. Writing `q₁, q₂` for the
/// computed z-scores, `c₁, c₂` for their exact corrections
/// ([`ZTerms::correction`]) and `s` for the residual of `q₁ + q₂`, the
/// exact average `A` and the returned `G` satisfy an *identity*, not an
/// inequality:
///
/// ```text
/// A − G  =  ½(s + c₁ + c₂) − δ,        |δ| ≤ 2^-1075
/// ```
///
/// `δ` is the halving's own rounding, which is zero unless the sum lands
/// among the subnormals. So the figure returned here is the error, plus
/// [`REFINED_RESIDUAL_SLACK`] for the four roundings that combined the
/// residuals and [`RESIDUAL_UNDERFLOW`] for the one that cannot be
/// combined at all.
///
/// # Why the sign is kept
///
/// The three residuals are summed **signed** and the absolute value taken
/// once, at the end. Taking `|s| + |c₁| + |c₂|` instead would be sound and
/// would repeat the first tier's mistake one level down: a trial whose own
/// error terms cancel would then be charged for an error it does not have.
/// The identity above holds for the signed sum, so there is nothing to
/// give up by using it.
///
/// # Cost
///
/// Two two-sums, two `mul_add`s, two divisions and five additions, on a
/// path reached only when the first tier could not clear the trial — so
/// never for a cosine similarity or a PLDA log-likelihood ratio at any
/// cohort spread this module has been measured on. `mul_add` is a
/// software call on a target without a hardware `fma`; that cost was the
/// reason not to make this the *primary* mechanism, and it is not a reason
/// against a fallback.
///
/// # It is never looser than the tier that selected it
///
/// The closing `min` says so, and it is also what makes the refinement
/// total. `two_sum_residual` yields `NaN` where a sum overflows — a branch
/// [`half_sum`] exists for, and one the first tier cannot select, since
/// two z-scores that overflow their sum share a sign and cancel by
/// nothing — and `f64::min` returns the operand that is not `NaN`. A
/// refinement that cannot be computed therefore refines nothing, rather
/// than deciding by comparing against a `NaN`.
pub(super) fn refined_error_bound(
  raw: f64,
  side: &CohortStats,
  other: &CohortStats,
  z_self: f64,
  z_other: f64,
  filtered: f64,
) -> f64 {
  let (correction_self, underflow_self) = side.z_terms(raw).correction(z_self);
  let (correction_other, underflow_other) = other.z_terms(raw).correction(z_other);
  let sum_residual = two_sum_residual(z_self, z_other);

  let error = 0.5 * ((sum_residual + correction_self) + correction_other);
  let residuals = 0.5 * ((sum_residual.abs() + correction_self.abs()) + correction_other.abs());
  let underflow = 0.5 * (underflow_self + underflow_other);

  let refined = error.abs() + (REFINED_RESIDUAL_SLACK * residuals + underflow);
  refined.min(filtered)
}

/// The error a successful normalization is allowed to carry:
/// [`MAX_NORMALIZED_ERROR`], with an absolute floor of one standard
/// deviation.
///
/// The floor is the load-bearing half. A purely relative criterion would
/// refuse every result that legitimately cancels to near zero — which is
/// precisely the region a match threshold sits in, and the region where a
/// z-score's own absolute error is most obviously harmless.
pub(super) fn permitted_error(normalized: f64) -> f64 {
  MAX_NORMALIZED_ERROR * normalized.abs().max(1.0)
}

/// The exact power of two a side's scores are divided by before its mean
/// and deviation are accumulated: `2^⌊log₂ max|score|⌋`.
///
/// This is the accumulator row of the module's [range-safety
/// rule](crate::score_norm#range-safety), and the one whose factor varies
/// over the whole exponent range rather than between `1` and `½`.
///
/// # Why a power of two, and why it is free
///
/// Masking off the significand leaves the exponent, so the factor is a
/// power of two *by construction*. Division by one is exact in IEEE-754,
/// and scaling by one commutes with the rounding of `+`, `-`, `/` and
/// `sqrt`. Every intermediate of the compensated two-pass — including
/// every Neumaier compensation term — is therefore exactly the unscaled
/// intermediate times the same factor, and the closing multiplication
/// restores it exactly. Measured over 136 000 cohorts spanning cosine
/// scores from `1e-9` spread to full `[-1, 1]`, the crate's own
/// tight-cluster-at-`1e8` case and PLDA-scale log-likelihood ratios, the
/// rescaled mean and deviation are bit-identical to the unscaled ones in
/// every case. Nothing the compensated summation preserves is spent here.
///
/// # What it buys
///
/// Totality. After the division every value is at most `2` in magnitude,
/// so neither `Σx` nor `Σ(x − mean)²` can leave f64's range for any
/// finite input; and since a population deviation never exceeds the
/// largest absolute value it was taken over (Popoviciu's inequality),
/// neither can the restored `μ` or `σ`. It also recovers the other
/// direction: a set whose deviation is `1e-200` used to square to zero
/// and be reported as an exactly-degenerate `0`, and now reports the
/// deviation it actually has.
fn rescale_factor(scores: &[f64]) -> f64 {
  /// f64's exponent field. The sign bit is already clear on an absolute
  /// value, and masking the significand away leaves `2^exponent`.
  const EXPONENT_MASK: u64 = 0x7ff0_0000_0000_0000;

  let max_abs = scores.iter().fold(0.0f64, |m, v| m.max(v.abs()));
  let factor = f64::from_bits(max_abs.to_bits() & EXPONENT_MASK);
  if factor == 0.0 {
    // `max_abs` is zero or subnormal — a subnormal's exponent field is
    // all zeros, so the mask yields `0.0` and dividing by it would be
    // worse than useless. The smallest positive normal scales a subnormal
    // set up into the normal range instead, exactly: a subnormal carries
    // fewer than 53 significant bits, so nothing is lost on the way up.
    // An all-zero set stays at zero and falls to `DegenerateDeviation`,
    // which is what it is.
    f64::MIN_POSITIVE
  } else {
    factor
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
