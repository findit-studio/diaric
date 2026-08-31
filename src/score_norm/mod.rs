//! Adaptive score normalization (AS-Norm) over speaker-trial scores.
//!
//! Raw similarity scores are not comparable across speakers. One speaker
//! sits in a crowded region of the embedding space and scores highly
//! against everyone; another sits alone and scores lower against
//! everyone. A single global threshold therefore over-merges the first
//! and under-merges the second. AS-Norm rescales each trial score against
//! the *cohort distribution of the two speakers involved*, so one
//! threshold means the same thing everywhere.
//!
//! # The variant implemented here
//!
//! **AS-Norm1**, as defined by Matějka, Novotný, Plchot, Burget, Diez
//! Sánchez and Černocký, *"Analysis of Score Normalization in
//! Multilingual Speaker Recognition"*, Interspeech 2017, eq. (7). For a
//! trial between `e` and `t` scoring `s(e, t)`:
//!
//! ```text
//!                1   ⎛ s(e,t) − μ(Sₑ(Eₑᵗᵒᵖ))     s(e,t) − μ(Sₜ(Eₜᵗᵒᵖ)) ⎞
//! as-norm1  =   ─── ⎜ ───────────────────── + ───────────────────── ⎟
//!                2   ⎝      σ(Sₑ(Eₑᵗᵒᵖ))             σ(Sₜ(Eₜᵗᵒᵖ))       ⎠
//! ```
//!
//! Each side selects its own top-N of the shared cohort (`Eₑᵗᵒᵖ`,
//! `Eₜᵗᵒᵖ`) and takes the mean and standard deviation of exactly those
//! selected scores. That per-side independence is the whole reason this
//! variant is the practical one — see [Cost](#cost).
//!
//! The name AS-Norm is due to Cumani, Batzu, Colibro, Vair, Laface and
//! Vasilakakis, *"Comparison of Speaker Recognition Approaches for Real
//! Applications"*, Interspeech 2011, eq. (3), whose original formulation
//! is the **crossed** variant that Matějka later labels AS-Norm2 (eq. 8):
//! there each side's statistics are taken over the subset selected by the
//! *other* side. AS-Norm2 is not implemented here. Matějka's own
//! measurement is that it *"performs about the same as adaptive
//! S-norm1"*, and it cannot be precomputed per speaker — its statistics
//! depend on the pairing, so every trial would re-derive both sides.
//!
//! ## The `0.5` is load-bearing here
//!
//! Both primary sources carry the `1/2`, and so does WeSpeaker
//! (`wespeaker/bin/score_norm.py`). Because it is a global positive
//! constant it cannot change any *ranking*, so EER, AUC and minDCF are
//! identical with or without it. But a clustering stage compares the
//! normalized score against a **fixed absolute threshold**, and there a
//! factor of two is the difference between a tuned threshold and a
//! meaningless one. Keeping it is what makes a threshold derived from
//! published AS-Norm numbers directly usable.
//!
//! ## Population, not sample, standard deviation
//!
//! Divisor `N`, not `N − 1`. This matches the reference implementation,
//! which calls `np.std` without `ddof` — and NumPy's default is `ddof=0`.
//! With `top_n` in the recommended 200–500 band the two differ by well
//! under a percent, but matching the reference exactly is what makes a
//! published threshold transferable.
//!
//! # Nothing here knows what an embedding is
//!
//! AS-Norm is arithmetic on **scores**. Accordingly no type or function
//! in this module mentions an embedding, a vector, or a dimension:
//!
//! - [`CohortStats::from_scores`] takes an iterator of `f64` scores.
//! - [`Cohort<K, T>`] is generic over the item type `T`, and reaches it
//!   only through a caller-supplied `FnMut(&S, &T) -> f64`.
//!
//! So the scoring function is a parameter too — cosine
//! ([`cosine_similarity`](crate::embed::cosine_similarity)) today, PLDA
//! ([`PldaTransform`](crate::plda::PldaTransform)) later, both already in
//! this crate — and a future model emitting 192-d (ECAPA, CAM++) or 512-d
//! (XVEC) vectors instead of today's 256-d needs no change here at all.
//!
//! ```
//! use diaric::{
//!   embed::{Embedding, cosine_similarity},
//!   score_norm::{AsNormOptions, Cohort, as_norm},
//! };
//!
//! # // Stand-in for real inference output: spread-out unit vectors, so
//! # // the cohort scores actually vary. Real callers pass whatever their
//! # // embedder produced — the dimension is theirs, not this module's.
//! # fn speaker(seed: usize) -> Embedding {
//! #   let mut v = [0.0f32; diaric::embed::EMBEDDING_DIM];
//! #   for (i, x) in v.iter_mut().enumerate() {
//! #     *x = (((i * 37 + seed * 101) % 97) as f32) / 97.0 - 0.5;
//! #   }
//! #   Embedding::normalize_from(v).unwrap()
//! # }
//! let options = AsNormOptions::new();
//! let (alice, bob) = (speaker(200), speaker(201));
//!
//! // A cohort sampled from the speaker library itself — so it contains
//! // Alice and Bob, exactly the situation `stats_excluding` is for.
//! let mut cohort: Cohort<u32, Embedding> = Cohort::new();
//! for i in 0..64 {
//!   cohort.push(1000 + i as u32, speaker(i));
//! }
//! cohort.push(1, alice);
//! cohort.push(2, bob);
//!
//! // The dimension appears nowhere below: only the scoring closure,
//! // which is the single place the embedding type is named at all.
//! let score = |a: &Embedding, b: &Embedding| f64::from(cosine_similarity(a, b));
//!
//! // Computed once per speaker, then reused for every trial. Each side
//! // excludes its own entry: without that, Alice's self-match scores a
//! // perfect 1.0 and top-N selection is guaranteed to pick it up.
//! let alice_side = cohort.stats_excluding(&1, &alice, score, &options)?;
//! let bob_side = cohort.stats_excluding(&2, &bob, score, &options)?;
//! assert_eq!(alice_side.considered(), 65); // 66 members, less Alice's own
//!
//! let raw = score(&alice, &bob);
//! let normalized = as_norm(raw, &alice_side, &bob_side)?;
//! # assert!(normalized.is_finite());
//! # Ok::<(), diaric::score_norm::Error>(())
//! ```
//!
//! # Cohort self-contamination
//!
//! The failure this module works hardest to prevent. A cohort is meant to
//! be impostors; if the speaker being normalized is in it, top-N
//! selection is *guaranteed* to pick up its own material, because a
//! self-match is the highest score obtainable. Choose deliberately
//! between the two entrypoints:
//!
//! - [`Cohort::stats_excluding`] — drops every entry whose speaker key
//!   matches the one being normalized. Use this whenever the cohort is
//!   drawn from the same library being scored.
//! - [`Cohort::stats_assuming_disjoint`] — scores the whole cohort. Only
//!   correct for a held-out cohort, which is what the primary sources
//!   assume.
//!
//! # Cost
//!
//! A [`CohortStats`] depends only on its own side and the cohort, so it
//! is computed **once per speaker** and reused. Scoring `N` speakers
//! against each other with a cohort of `C`:
//!
//! | | cohort scores | trial scores |
//! |---|---|---|
//! | recomputing per trial | `N(N−1)·C` | `N(N−1)/2` |
//! | reusing [`CohortStats`] | `N·C` | `N(N−1)/2` |
//!
//! For `N = 1000`, `C = 300` that is 300 000 cohort scores instead of
//! 300 million. Selection itself is linear in the cohort, not
//! `O(C log C)`, since only the membership of the top-N matters.
//!
//! # Range safety
//!
//! Cohort scores are `f64` and nothing constrains their magnitude, so
//! several steps here can leave f64's range for inputs whose *answer* is
//! perfectly representable. One rule covers all of them:
//!
//! > Perform the operation in a domain rescaled by an exact power of two,
//! > chosen **from the operands**, and account for that factor exactly.
//!
//! Scaling by a power of two is exact, and it commutes with the rounding
//! of `+`, `-`, `*`, `/` and `sqrt`. The rescaled computation therefore
//! returns the number the unscaled one would have returned had it fitted:
//! range safety costs no precision, which is why it needs no case analysis
//! of when it is worth paying for.
//!
//! **Chosen from the operands** is the load-bearing half. A *fixed* factor
//! moves the failure instead of removing it — halving unconditionally
//! keeps a shift near [`f64::MAX`] in range and, in the same stroke,
//! rounds a subnormal z-score to a signed zero. Three sites apply the
//! rule, each taking its factor from the values in hand:
//!
//! | site | factor | what would leave the range |
//! |---|---|---|
//! | [`CohortStats::from_scores`] accumulators | `2^⌊log₂ max score⌋` | `Σx`, and `Σ(x − μ)²` for a cohort whose deviation is itself representable |
//! | [`CohortStats::normalize`] shift | `1`, or `½` when `raw − μ` does not fit | a cohort mean near `-1.7e308`, against a trial score at the other end |
//! | [`CohortStats::normalize`] average | `1`, or `½` when the two terms' sum does not fit | two terms of `1e308`, whose average is an ordinary `1e308` |
//!
//! What survives the rule is a genuine overflow — a quotient that really
//! does exceed [`f64::MAX`] — and that is refused as
//! [`Error::NonFiniteResult`], never returned.
//!
//! # Accuracy
//!
//! Range safety keeps a value inside f64; it says nothing about that
//! value being right. [`CohortStats::normalize`] returns the *average* of
//! two z-scores, and an average is where a small relative error becomes a
//! large absolute one.
//!
//! `(raw − μ) / σ` is two rounded operations, so each computed z-score `q`
//! sits close to the exact quotient `z` it stands for — but only
//! *relatively*:
//!
//! ```text
//! |z − q| ≤ 2u|q|(1 + 2u) + 3η,      u = 2^-53,  η = 2^-1075
//! ```
//!
//! one rounding from the subtraction, one from the division, with `η`
//! covering a quotient that lands among the subnormals. The rescaled
//! branch of the shift is the same two roundings of the same exact
//! quantity — the `½` is exact on both operands there, or the quotient
//! is an infinity the finiteness postcondition takes first — so the
//! bound covers it without a case of its own. Averaging two of
//! them carries both errors across at their **absolute** size while the
//! sum itself is free to cancel to nothing. Writing
//! `mag = (|q₁| + |q₂|) / 2`, the returned `G` and the exact average `A`
//! satisfy
//!
//! ```text
//! |A − G| ≤ ½(|z₁ − q₁| + |z₂ − q₂|) + u|G| + η   ≤   2^-51 · mag + 2^-1072
//! ```
//!
//! — the middle term being the sum's own rounding and the halving, which
//! is relative to the answer and therefore always harmless. The point is
//! the first term: it is set by the *operands*, so it does not shrink when
//! they cancel. Two z-scores near `2^50` that land one ulp apart with
//! opposite signs sum to `0.25`, and `0.25` is exactly the size of the
//! bits that were thrown away — the result is then rounding noise wearing
//! the answer's clothes, sign included.
//!
//! So a successful normalization asserts an accuracy, not just a
//! finiteness:
//!
//! ```text
//! refuse when   2^-51 · mag   >   MAX_NORMALIZED_ERROR · max(|G|, 1)
//! ```
//!
//! Two properties make this a guard rather than a heuristic. It is a
//! **bound**, so it never refuses a result that was in fact sound; and its
//! tolerance has an **absolute floor of one standard deviation**, so the
//! predicate reduces to `mag > 2^31` whenever the answer is small. That is
//! a statement about the inputs and not about how badly they cancelled:
//! every trial whose z-scores average below `2^31 ≈ 2.1e9` is answered,
//! however completely the two sides cancel, an exactly zero result
//! included. A purely relative criterion would have refused exactly those
//! — the near-zero results a match threshold lives among.
//!
//! Nothing here alters a returned value: the guard only decides whether
//! the value is returned, so every answer this module gave before it is
//! bit-identical to the one it gives now.
//!
//! ## How far the guard is from real data
//!
//! A guard that fires on real scores would be worse than the defect it
//! prevents, so the distance is measured rather than asserted. Each row
//! is the worst `error bound / tolerance` over the regimes this module
//! has been measured on — `1.0` is where it fires:
//!
//! | regime | worst ratio | margin |
//! |---|---|---|
//! | cosine, spread `1e-9`, floor `1e-15` | `1.58e-7` | 6.3e6× |
//! | cosine, spread `1e-6` … `5e-1` | `8.5e-8` … `1.4e-9` | 1.2e7× … 7.0e8× |
//! | tight cluster at `1e8` | `4.66e-10` | 2.1e9× |
//! | PLDA log-likelihood ratios | `3.43e-9` | 2.9e8× |
//! | widest trial the default floor admits | `4.66e-4` | 2147× |
//!
//! The last row is constructed rather than sampled: two cosine sides with
//! `σ` exactly on [`DEFAULT_MIN_DEVIATION`] and means placed
//! symmetrically about the trial score, so the z-scores are as large as
//! `[-1, 1]` allows and the cancellation is total. It is the closest any
//! bounded score source gets, and it is still three orders of magnitude
//! away. The case that does fire — z-scores near `2^50` one ulp apart —
//! exceeds the tolerance by `6.05e5×`, so the two regimes are separated
//! by nine orders of magnitude with nothing in between that a caller can
//! construct. `the_guard_stays_orders_of_magnitude_clear_of_every_measured_regime`
//! pins every row.
//!
//! ## Why the residuals are not recovered instead
//!
//! They could be. `f64::mul_add` yields `d − q·σ` exactly, a two-sum
//! yields the subtraction's residual, and adding both back would make the
//! cancelling case correct rather than refused. It is deliberately not
//! done. The machinery pays only in a regime no score source here can
//! reach — the cancellation needs `|raw − μ| ≥ 1.1e9` at a deviation on
//! the floor, while cosine similarities span `[-1, 1]` and PLDA
//! log-likelihood ratios the tens — and `mul_add` is a software `fma`
//! call on every target without a hardware one, on a function called once
//! per trial. What it would buy is measured too: the residual bound is a
//! median 3.5 times tighter than the one above across the cosine sweeps
//! and 3.0 times tighter on the case that does fire, where the two exceed
//! the tolerance by `2.0e5×` and `6.1e5×` respectively. Three times
//! tighter never moves a verdict when the nearest real data is a million
//! times away. Refusing is also what the rest of this module does with an
//! answer it cannot stand behind.
//!
//! # Not implemented
//!
//! Matějka §4.1 also advises rejecting cohort scores outside ±4–5σ of the
//! cohort mean before taking statistics. No reference implementation does
//! this, and it is a second, independent policy question; it is left to
//! the caller, who can filter the score iterator before it reaches
//! [`CohortStats::from_scores`].

mod cohort;
mod error;
mod options;
mod stats;

#[cfg(test)]
mod tests;

pub use cohort::{Cohort, CohortEntry};
pub use error::{CohortTooSmall, DegenerateDeviation, Error, ZScoreCancellation};
pub use options::{AsNormOptions, DEFAULT_MIN_DEVIATION, DEFAULT_TOP_N, MIN_COHORT_SCORES};
pub use stats::{CohortStats, MAX_NORMALIZED_ERROR, as_norm};

// Compile-time trait assertions. Catches a future field-type change that
// would silently regress Send/Sync auto-derive on the public types.
const _: fn() = || {
  fn assert_send_sync<T: Send + Sync>() {}
  assert_send_sync::<AsNormOptions>();
  assert_send_sync::<CohortStats>();
  assert_send_sync::<Error>();
  assert_send_sync::<CohortTooSmall>();
  assert_send_sync::<DegenerateDeviation>();
  assert_send_sync::<ZScoreCancellation>();
  assert_send_sync::<Cohort<u32, crate::embed::Embedding>>();
  assert_send_sync::<CohortEntry<u32, crate::embed::Embedding>>();
};
