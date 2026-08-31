use core::num::NonZeroUsize;

use crate::{
  embed::{EMBEDDING_DIM, Embedding, cosine_similarity},
  score_norm::{
    AsNormOptions, Cohort, CohortEntry, CohortStats, DEFAULT_TOP_N, Error, MIN_COHORT_SCORES,
    as_norm,
  },
};

fn top_n(n: usize) -> NonZeroUsize {
  NonZeroUsize::new(n).expect("test top_n is non-zero")
}

fn opts(n: usize) -> AsNormOptions {
  AsNormOptions::new().with_top_n(top_n(n))
}

/// Scoring closure for the identity cohorts below: the "item" *is* the
/// score, so the tests exercise selection and statistics without any
/// embedding arithmetic in the way.
fn item_score(_side: &(), item: &f64) -> f64 {
  *item
}

fn cohort_of(pairs: &[(u32, f64)]) -> Cohort<u32, f64> {
  Cohort::from_entries(
    pairs
      .iter()
      .map(|&(k, v)| CohortEntry::new(k, v))
      .collect::<Vec<_>>(),
  )
}

fn basis_embedding(i: usize, sign: f32) -> Embedding {
  let mut v = [0.0f32; EMBEDDING_DIM];
  v[i] = sign;
  Embedding::normalize_from(v).expect("basis vector is normalizable")
}

// ── Trap 1: cohort self-contamination ──────────────────────────────────

/// `stats_excluding` must drop every cohort entry belonging to the
/// speaker being normalized.
///
/// Speaker `0`'s own four entries score 0.95–0.98 against its own side,
/// far above the six impostors at 0.10–0.20. Top-N selection takes the
/// *largest* scores, so an unexcluded self-entry is not merely present in
/// the selected set — it is guaranteed to dominate it. The impostor mean
/// here is 0.17; leaving speaker 0 in produces 0.965.
#[test]
fn stats_excluding_drops_the_normalized_speakers_own_entries() {
  let cohort = cohort_of(&[
    (1, 0.10),
    (2, 0.12),
    (3, 0.14),
    (4, 0.16),
    (5, 0.18),
    (6, 0.20),
    (0, 0.95),
    (0, 0.96),
    (0, 0.97),
    (0, 0.98),
  ]);

  let stats = cohort
    .stats_excluding(&0, &(), item_score, &opts(4))
    .expect("six impostors are enough for a usable side");

  // Only the six impostors were scored at all.
  assert_eq!(stats.considered(), 6, "self entries must not be scored");
  assert_eq!(stats.selected(), 4);
  // Top-4 impostors: 0.20, 0.18, 0.16, 0.14 → mean 0.17.
  assert!(
    (stats.mean() - 0.17).abs() < 1e-12,
    "mean {} is contaminated by speaker 0's own entries",
    stats.mean()
  );
}

/// The unguarded counterpart: `stats_assuming_disjoint` deliberately scores every
/// member, so it *does* pick up speaker 0. This pins the difference the
/// exclusion makes, so a future refactor cannot quietly collapse the two
/// entrypoints into one.
#[test]
fn stats_assuming_disjoint_keeps_every_entry_including_self() {
  let cohort = cohort_of(&[(1, 0.10), (2, 0.12), (0, 0.97), (0, 0.98)]);

  let stats = cohort
    .stats_assuming_disjoint(&(), item_score, &opts(2))
    .expect("four members are enough");

  assert_eq!(stats.considered(), 4);
  assert!(
    (stats.mean() - 0.975).abs() < 1e-12,
    "mean {} should be the contaminated 0.975",
    stats.mean()
  );
}

/// Excluding a speaker that owns *every* entry leaves nothing to score.
#[test]
fn stats_excluding_everything_is_an_empty_cohort() {
  let cohort = cohort_of(&[(7, 0.1), (7, 0.2), (7, 0.3)]);
  let err = cohort
    .stats_excluding(&7, &(), item_score, &opts(2))
    .expect_err("self-exclusion removed every member");
  assert!(matches!(err, Error::EmptyCohort), "got {err:?}");
}

// ── Trap 2: degenerate sigma ───────────────────────────────────────────

/// Identical top-N scores give `sigma == 0`; the division must be refused,
/// not performed.
#[test]
fn identical_cohort_scores_are_refused() {
  let err = CohortStats::from_scores([0.5; 8], &opts(4))
    .expect_err("zero-spread cohort must not produce a score");
  let Error::DegenerateDeviation(d) = err else {
    panic!("expected DegenerateDeviation, got {err:?}");
  };
  assert_eq!(d.selected(), 4);
  assert!(d.deviation() < d.minimum());
}

/// A spread below the configured floor is refused just like an exactly
/// zero one — the floor, not `== 0.0`, is the test.
#[test]
fn spread_below_the_floor_is_refused() {
  let scores = [0.5, 0.5 + 1e-12, 0.5 + 2e-12, 0.5 + 3e-12];
  let err =
    CohortStats::from_scores(scores, &opts(4)).expect_err("1e-12 spread is below the 1e-6 floor");
  assert!(matches!(err, Error::DegenerateDeviation(_)), "got {err:?}");
}

/// The floor is configurable, so a caller on a wider score scale can
/// accept a spread this crate's cosine default would refuse.
#[test]
fn lowering_the_floor_admits_a_tight_cohort() {
  let scores = [0.5, 0.5 + 1e-12, 0.5 + 2e-12, 0.5 + 3e-12];
  let options = opts(4).with_min_deviation(1e-15);
  let stats = CohortStats::from_scores(scores, &options).expect("floor lowered below the spread");
  assert!(stats.deviation() > 0.0);
}

// ── Trap 3: top_n vs cohort size ───────────────────────────────────────

/// `top_n` larger than the cohort is not an error: the whole cohort is
/// used, which is exactly S-Norm. `selected()` reports the truncation so
/// the caller can see it happened.
#[test]
fn top_n_larger_than_cohort_uses_the_whole_cohort() {
  let scores = [0.1, 0.4, 0.9];
  let stats = CohortStats::from_scores(scores, &AsNormOptions::new())
    .expect("a 3-member cohort with top_n=300 is usable");
  assert_eq!(stats.selected(), 3, "whole cohort used");
  assert_eq!(stats.considered(), 3);
  let mean = (0.1 + 0.4 + 0.9) / 3.0;
  assert!((stats.mean() - mean).abs() < 1e-12);
}

/// `top_n` exactly equal to the cohort size is the boundary of the clamp.
#[test]
fn top_n_equal_to_cohort_size_uses_the_whole_cohort() {
  let stats = CohortStats::from_scores([0.1, 0.4, 0.9], &opts(3)).expect("boundary case is usable");
  assert_eq!(stats.selected(), 3);
}

/// An empty cohort has no distribution at all.
#[test]
fn empty_cohort_is_refused() {
  let err = CohortStats::from_scores([], &AsNormOptions::new())
    .expect_err("an empty cohort has no statistics");
  assert!(matches!(err, Error::EmptyCohort), "got {err:?}");
}

/// One usable score has no dispersion; refuse it by count rather than
/// letting it fall through to a zero standard deviation.
#[test]
fn single_score_cohort_is_refused_by_count() {
  let err =
    CohortStats::from_scores([0.42], &AsNormOptions::new()).expect_err("one score is not a spread");
  let Error::CohortTooSmall(c) = err else {
    panic!("expected CohortTooSmall, got {err:?}");
  };
  assert_eq!(c.available(), 1);
  assert_eq!(c.required(), MIN_COHORT_SCORES);
}

/// `top_n == 1` selects a single score whose population standard
/// deviation is exactly zero by definition — the sharpest edge of the
/// adaptive selection, and one the reference implementation divides by.
/// Refused by count, so the message names the real cause (the selection
/// is too narrow) rather than blaming the cohort.
#[test]
fn top_n_of_one_is_refused_by_count() {
  let err = CohortStats::from_scores([0.1, 0.2, 0.3, 0.4], &opts(1))
    .expect_err("a one-score selection has no spread");
  let Error::CohortTooSmall(c) = err else {
    panic!("expected CohortTooSmall, got {err:?}");
  };
  assert_eq!(c.available(), 1);
  assert_eq!(c.required(), MIN_COHORT_SCORES);
}

// ── Trap 4: numerical care ─────────────────────────────────────────────

/// Tightly-clustered scores far from zero: the textbook case where
/// `E[x^2] - E[x]^2` cancels catastrophically.
///
/// The selected top-4 are `1e8 + {1,2,3,4}`, whose exact population
/// variance is 1.25 (sigma = 1.118033988749895). Both `sum_sq/n` and
/// `mean^2` land near 1e16, where the f64 ulp is 2 — so the one-pass form
/// can only return a multiple of ~2, and often returns a *negative*
/// variance whose square root is NaN. A two-pass computation over the
/// mean-shifted values keeps full precision.
#[test]
fn tight_cluster_far_from_zero_keeps_precision() {
  let base = 1.0e8;
  // Two low scores keep top-N selection meaningful (and away from the
  // `top_n == len` boundary that Trap 3 covers).
  let scores = [base + 1.0, base + 2.0, base + 3.0, base + 4.0, 0.0, 1.0];
  let stats = CohortStats::from_scores(scores, &opts(4)).expect("well-formed cohort");

  assert_eq!(stats.selected(), 4);
  assert!(
    (stats.mean() - (base + 2.5)).abs() < 1e-9,
    "mean {} drifted",
    stats.mean()
  );
  let expected = 1.25f64.sqrt();
  assert!(
    (stats.deviation() - expected).abs() < 1e-9,
    "deviation {} != {expected} — one-pass cancellation",
    stats.deviation()
  );
}

/// A non-finite cohort score is a broken scorer, not a data point.
#[test]
fn non_finite_cohort_score_is_refused() {
  let err = CohortStats::from_scores([0.1, f64::NAN, 0.3, 0.4], &opts(2))
    .expect_err("NaN cohort score must be refused");
  assert!(matches!(err, Error::NonFiniteScore(_)), "got {err:?}");

  let err = CohortStats::from_scores([0.1, 0.2, f64::INFINITY, 0.4], &opts(2))
    .expect_err("inf cohort score must be refused");
  assert!(matches!(err, Error::NonFiniteScore(_)), "got {err:?}");
}

/// A non-finite *trial* score is refused at the same boundary.
#[test]
fn non_finite_trial_score_is_refused() {
  let side = CohortStats::from_scores([0.1, 0.2, 0.3, 0.4], &opts(4)).expect("usable side");
  for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
    let err = as_norm(bad, &side, &side).expect_err("non-finite trial score must be refused");
    assert!(matches!(err, Error::NonFiniteScore(_)), "got {err:?}");
  }
}

// ── The AS-Norm combination itself ─────────────────────────────────────

/// Hand-computed golden. Enrollment side top-2 of `[0.0, 0.2, 0.4]` is
/// `{0.4, 0.2}` → mean 0.3, population sigma 0.1. Test side top-2 of
/// `[0.1, 0.5, 0.9]` is `{0.9, 0.5}` → mean 0.7, sigma 0.2. For a raw
/// trial score of 0.8:
///
/// ```text
/// 0.5 * ((0.8 - 0.3)/0.1 + (0.8 - 0.7)/0.2) = 0.5 * (5.0 + 0.5) = 2.75
/// ```
#[test]
fn as_norm_matches_the_hand_computed_value() {
  let e = CohortStats::from_scores([0.0, 0.2, 0.4], &opts(2)).expect("enrollment side");
  let t = CohortStats::from_scores([0.1, 0.5, 0.9], &opts(2)).expect("test side");

  assert!((e.mean() - 0.3).abs() < 1e-12, "e.mean = {}", e.mean());
  assert!(
    (e.deviation() - 0.1).abs() < 1e-12,
    "e.dev = {}",
    e.deviation()
  );
  assert!((t.mean() - 0.7).abs() < 1e-12, "t.mean = {}", t.mean());
  assert!(
    (t.deviation() - 0.2).abs() < 1e-12,
    "t.dev = {}",
    t.deviation()
  );

  let got = as_norm(0.8, &e, &t).expect("finite trial score");
  assert!((got - 2.75).abs() < 1e-12, "as_norm = {got}");
}

/// The two sides are averaged, so swapping them cannot change the trial's
/// normalized score. A trial is unordered; if this ever became
/// asymmetric, `score(a, b)` and `score(b, a)` would disagree.
#[test]
fn as_norm_is_symmetric_in_its_two_sides() {
  let e = CohortStats::from_scores([0.0, 0.2, 0.4], &opts(2)).expect("enrollment side");
  let t = CohortStats::from_scores([0.1, 0.5, 0.9], &opts(2)).expect("test side");
  let ab = as_norm(0.8, &e, &t).expect("finite");
  let ba = as_norm(0.8, &t, &e).expect("finite");
  assert_eq!(ab, ba);
}

/// The free function and the method are the same computation.
#[test]
fn free_function_matches_method() {
  let e = CohortStats::from_scores([0.0, 0.2, 0.4], &opts(2)).expect("enrollment side");
  let t = CohortStats::from_scores([0.1, 0.5, 0.9], &opts(2)).expect("test side");
  assert_eq!(
    as_norm(0.8, &e, &t).expect("finite"),
    e.normalize(0.8, &t).expect("finite")
  );
}

/// AS-Norm's whole purpose: a raw score that sits at the same place in
/// each speaker's own cohort distribution normalizes to the same value,
/// even though the raw scores differ. A single global threshold on the
/// normalized score therefore means the same thing for both speakers.
#[test]
fn equal_z_positions_normalize_equally_across_speakers() {
  // Speaker A: tight, low cohort. Speaker B: wide, high cohort.
  let a = CohortStats::from_scores([0.00, 0.10, 0.20], &opts(3)).expect("side a");
  let b = CohortStats::from_scores([0.40, 0.60, 0.80], &opts(3)).expect("side b");

  // Raw scores two standard deviations above each side's own mean.
  let raw_a = a.mean() + 2.0 * a.deviation();
  let raw_b = b.mean() + 2.0 * b.deviation();
  assert!(raw_a < raw_b, "the raw scores differ: {raw_a} vs {raw_b}");

  let na = as_norm(raw_a, &a, &a).expect("finite");
  let nb = as_norm(raw_b, &b, &b).expect("finite");
  assert!((na - nb).abs() < 1e-12, "{na} != {nb}");
  assert!((na - 2.0).abs() < 1e-12, "two sigmas above the mean is 2.0");
}

// ── Precomputation ─────────────────────────────────────────────────────

/// A side's statistics depend only on that side and the cohort, never on
/// the other side of the trial. That independence is what makes it sound
/// to compute a speaker's side once and reuse it across every trial the
/// speaker takes part in — the property the whole precomputation surface
/// rests on.
#[test]
fn a_side_is_reusable_across_trials() {
  let cohort = cohort_of(&[(1, 0.10), (2, 0.30), (3, 0.50), (4, 0.70)]);
  let options = opts(3);

  let a = cohort
    .stats_excluding(&10, &(), item_score, &options)
    .expect("side a");
  let b = cohort
    .stats_excluding(&11, &(), item_score, &options)
    .expect("side b");
  let c = cohort
    .stats_excluding(&12, &(), item_score, &options)
    .expect("side c");

  // Recomputing a side yields a bit-identical side: nothing about the
  // partner leaks into it.
  let a_again = cohort
    .stats_excluding(&10, &(), item_score, &options)
    .expect("recomputed side a");
  assert_eq!(a, a_again);

  // So a cached side is interchangeable with a freshly computed one in
  // every trial it appears in.
  assert_eq!(
    as_norm(0.6, &a, &b).expect("finite"),
    as_norm(0.6, &a_again, &b).expect("finite")
  );
  assert_eq!(
    as_norm(0.6, &a, &c).expect("finite"),
    as_norm(0.6, &a_again, &c).expect("finite")
  );

  // Sides differing only in an exclusion key that matches no entry are
  // the same side.
  assert_eq!(b, c);
}

// ── Genericity ─────────────────────────────────────────────────────────

/// The whole point of the score-source parameterisation: the crate's own
/// 256-d `Embedding` plugs in through `cosine_similarity` without the
/// dimension appearing anywhere in the AS-Norm surface.
#[test]
fn cosine_over_embeddings_plugs_in_unchanged() {
  let side = basis_embedding(0, 1.0);
  let cohort = Cohort::from_entries(vec![
    CohortEntry::new(1u32, basis_embedding(1, 1.0)),
    CohortEntry::new(2u32, basis_embedding(2, 1.0)),
    CohortEntry::new(3u32, basis_embedding(3, 1.0)),
    // Speaker 0's own entry: cosine 1.0 against `side`, the maximum a
    // cosine score can take, so top-N selection is guaranteed to pick it
    // if it is not excluded.
    CohortEntry::new(0u32, basis_embedding(0, 1.0)),
  ]);

  let score = |a: &Embedding, b: &Embedding| f64::from(cosine_similarity(a, b));
  let options = opts(3).with_min_deviation(1e-12);

  let excluded = cohort
    .stats_excluding(&0, &side, score, &options)
    .expect_err("three orthogonal impostors all score exactly 0.0");
  assert!(
    matches!(excluded, Error::DegenerateDeviation(_)),
    "orthogonal impostors have zero spread; got {excluded:?}"
  );

  // Without exclusion the self-entry's 1.0 enters the selection and the
  // spread becomes non-zero — a "healthy-looking" side built entirely on
  // the contamination.
  let contaminated = cohort
    .stats_assuming_disjoint(&side, score, &options)
    .expect("the self-match manufactures a spread");
  assert!(
    contaminated.mean() > 0.0,
    "mean {} is lifted by the self-match",
    contaminated.mean()
  );
}

/// Nothing in the AS-Norm surface is tied to a vector at all: a cohort of
/// plain `f32` scalars scored by absolute difference works identically.
#[test]
fn an_arbitrary_non_vector_score_source_works() {
  let cohort = Cohort::from_entries(vec![
    CohortEntry::new("bob", 3.0f32),
    CohortEntry::new("carol", 7.0f32),
    CohortEntry::new("dave", 11.0f32),
    CohortEntry::new("alice", 1.0f32),
  ]);
  let stats = cohort
    .stats_excluding(
      &"alice",
      &1.0f32,
      |s: &f32, t: &f32| -f64::from((s - t).abs()),
      &opts(2),
    )
    .expect("three impostors");
  assert_eq!(stats.considered(), 3, "alice's own entry excluded");
  assert_eq!(stats.selected(), 2);
  // Top-2 of {-2, -6, -10} is {-2, -6} → mean -4.
  assert!((stats.mean() + 4.0).abs() < 1e-12, "mean {}", stats.mean());
}

// ── Options ────────────────────────────────────────────────────────────

#[test]
fn default_options_match_the_published_constants() {
  let o = AsNormOptions::default();
  assert_eq!(o.top_n().get(), DEFAULT_TOP_N);
  assert_eq!(o.top_n().get(), 300);
  assert_eq!(o.min_deviation(), crate::score_norm::DEFAULT_MIN_DEVIATION);
  assert_eq!(AsNormOptions::new(), o);
}

#[test]
#[should_panic(expected = "min_deviation must be finite and > 0.0")]
fn zero_min_deviation_panics() {
  let _ = AsNormOptions::new().with_min_deviation(0.0);
}

#[test]
#[should_panic(expected = "min_deviation must be finite and > 0.0")]
fn nan_min_deviation_panics() {
  let _ = AsNormOptions::new().with_min_deviation(f64::NAN);
}

#[test]
fn cohort_construction_and_accessors() {
  let mut c: Cohort<u32, f64> = Cohort::new();
  assert!(c.is_empty());
  assert_eq!(c.len(), 0);
  c.push(1, 0.5);
  assert_eq!(c.len(), 1);
  assert!(!c.is_empty());
  assert_eq!(c.entries()[0].speaker(), &1);
  assert_eq!(c.entries()[0].item(), &0.5);
  assert_eq!(Cohort::<u32, f64>::default().len(), 0);
}

/// `with_min_deviation` asserts, but serde reads straight into the field.
/// A zero floor makes the `deviation < floor` guard false for every
/// non-negative deviation, silently re-admitting the division by zero.
#[cfg(feature = "serde")]
#[test]
fn serde_bypassed_zero_min_deviation_is_refused() {
  let o: AsNormOptions =
    serde_json::from_str(r#"{"top_n":4,"min_deviation":0.0}"#).expect("deserialize");
  assert_eq!(o.min_deviation(), 0.0, "the builder assert was bypassed");
  // A cohort that the guard must reject.
  let err = CohortStats::from_scores([0.5; 8], &o).expect_err("a disabled floor must be refused");
  assert!(matches!(err, Error::InvalidMinDeviation(_)), "got {err:?}");
}

/// The guard's predicate is "finite and > 0", not just "!= 0". JSON
/// cannot carry `NaN`, so this exercises the same rejection path through
/// a negative floor — which JSON *can* carry, and which is equally
/// nonsensical as a standard-deviation bound.
#[cfg(feature = "serde")]
#[test]
fn serde_bypassed_negative_min_deviation_is_refused() {
  let o: AsNormOptions =
    serde_json::from_str(r#"{"top_n":4,"min_deviation":-1.0}"#).expect("deserialize");
  assert_eq!(o.min_deviation(), -1.0, "the builder assert was bypassed");
  let err = CohortStats::from_scores([0.1, 0.2, 0.3, 0.4], &o)
    .expect_err("a negative floor must be refused");
  let Error::InvalidMinDeviation(v) = err else {
    panic!("expected InvalidMinDeviation, got {err:?}");
  };
  assert_eq!(v, -1.0);
}

#[cfg(feature = "serde")]
#[test]
fn options_serde_roundtrip() {
  let o = AsNormOptions::new()
    .with_top_n(top_n(64))
    .with_min_deviation(1e-4);
  let json = serde_json::to_string(&o).expect("serialize");
  let back: AsNormOptions = serde_json::from_str(&json).expect("deserialize");
  assert_eq!(o, back);
  // Absent fields fall back to the same source of truth as `Default`.
  let empty: AsNormOptions = serde_json::from_str("{}").expect("deserialize empty");
  assert_eq!(empty, AsNormOptions::default());
}

// ── Trap 5: identity keys must be reflexive ────────────────────────────

// `stats_excluding` requires `K: Eq`, so a non-reflexive key — `f64`, via
// `NAN != NAN` — cannot reach it at all. That is a *compile-time* guard,
// and it is pinned by the `compile_fail` doctest on `stats_excluding`
// itself rather than here; a runtime test cannot express "this does not
// build". The tests below pin the other half: that ordinary `Eq` keys are
// unaffected.

/// A `String` key — owned, non-`Copy`, `Eq` — excludes exactly its own
/// entries. `Eq` costs no realistic caller anything.
#[test]
fn an_owned_eq_key_still_excludes_its_own_entries() {
  let cohort = Cohort::from_entries(vec![
    CohortEntry::new(String::from("alice"), 0.99),
    CohortEntry::new(String::from("bob"), 0.20),
    CohortEntry::new(String::from("carol"), 0.40),
    CohortEntry::new(String::from("alice"), 0.98),
  ]);
  let stats = cohort
    .stats_excluding(&String::from("alice"), &(), item_score, &opts(2))
    .expect("two impostors remain");
  assert_eq!(stats.considered(), 2, "both alice entries dropped");
  assert!((stats.mean() - 0.30).abs() < 1e-12, "mean {}", stats.mean());
}

// ── Trap 6: representable statistics must be computed, not refused ─────

/// The squares overflow where the deviation does not. Scores `[-1e155,
/// 1e155]` have mean `0` and population deviation `1e155` — both
/// perfectly representable — but each `d * d` is `1e310`. Compensated
/// summation cannot recover from that: the infinities cancel into a
/// `NaN`, and the side was refused as "degenerate" when it is nothing of
/// the sort.
#[test]
fn a_representable_deviation_far_from_zero_is_computed() {
  let stats = CohortStats::from_scores([-1e155, 1e155], &opts(2))
    .expect("mean 0 and deviation 1e155 are both representable");
  assert_eq!(stats.mean(), 0.0);
  assert_eq!(stats.deviation(), 1e155);
}

/// The same defect one pass earlier: `Σx` overflows for scores whose
/// *mean* is representable, and the compensated sum turns the overflow
/// into a `NaN` that poisons everything downstream of it.
#[test]
fn a_representable_mean_of_huge_scores_is_computed() {
  let big = 1.7e308;
  let stats =
    CohortStats::from_scores([big, big, -big], &opts(3)).expect("mean and deviation are finite");
  let expected_mean = big / 3.0;
  assert!(
    (stats.mean() - expected_mean).abs() <= expected_mean * 1e-15,
    "mean {} != {expected_mean}",
    stats.mean()
  );
  // Population deviation of {b, b, -b} is b * sqrt(8)/3.
  let expected_dev = big * 8f64.sqrt() / 3.0;
  assert!(
    (stats.deviation() - expected_dev).abs() <= expected_dev * 1e-15,
    "deviation {} != {expected_dev}",
    stats.deviation()
  );
}

/// The underflow direction of the same defect, and the one that shows in
/// an error payload: `[-1e-200, 1e-200]` spreads by `1e-200`, but each
/// `d * d` is `1e-400` and flushes to zero, so the side was refused with a
/// deviation of exactly `0` — a number it does not have. The refusal is
/// still correct (`1e-200` is far below the floor); the *reported reason*
/// was not.
#[test]
fn a_tiny_deviation_is_reported_at_its_real_magnitude() {
  let err =
    CohortStats::from_scores([-1e-200, 1e-200], &opts(2)).expect_err("1e-200 is below the floor");
  let Error::DegenerateDeviation(d) = err else {
    panic!("expected DegenerateDeviation, got {err:?}");
  };
  assert_eq!(d.deviation(), 1e-200, "the payload must not report 0");
  assert_eq!(d.minimum(), crate::score_norm::DEFAULT_MIN_DEVIATION);
}

/// Totality, swept end to end across f64's exponent range: a set of
/// finite scores either yields finite statistics or is refused for a
/// reason that is about the *cohort*. A non-finite statistic must never
/// be handed back as `Ok`, and `NonFiniteResult` — the postcondition that
/// would catch one — must never fire.
#[test]
fn statistics_are_total_over_finite_scores() {
  let options = opts(4).with_min_deviation(f64::MIN_POSITIVE);
  let mut checked = 0usize;
  for exp in -320i32..=308 {
    let m = 10f64.powi(exp);
    if m == 0.0 {
      continue;
    }
    for scores in [
      vec![-m, m, -m, m],
      vec![m, m, m, -m],
      vec![m, -m, m / 3.0, -m / 7.0],
      vec![f64::MAX, f64::MAX, -f64::MAX, m],
      vec![-f64::MAX, f64::MIN_POSITIVE, f64::MAX, m],
    ] {
      checked += 1;
      match CohortStats::from_scores(scores.iter().copied(), &options) {
        Ok(s) => {
          assert!(s.mean().is_finite(), "mean {} for {scores:?}", s.mean());
          assert!(
            s.deviation().is_finite(),
            "deviation {} for {scores:?}",
            s.deviation()
          );
        }
        Err(Error::NonFiniteResult(v)) => {
          panic!("non-finite statistic {v} for {scores:?}")
        }
        Err(_) => {}
      }
    }
  }
  assert!(checked > 3_000, "the sweep must actually run: {checked}");
}

/// The rescale is a power of two precisely so it is *free*: the
/// compensated two-pass must produce bit-identical statistics with and
/// without it. Recomputes the unscaled form inline and demands exact
/// equality over the only regime that occurs — cosine scores, tightly
/// clustered, which is what top-N selection produces by construction.
#[test]
fn the_rescale_is_bit_neutral_in_the_cosine_regime() {
  fn unscaled(scores: &[f64]) -> (f64, f64) {
    let n = scores.len() as f64;
    let mut buf = scores.to_vec();
    let mean = crate::ops::kahan_sum(&buf) / n;
    for v in &mut buf {
      let d = *v - mean;
      *v = d * d;
    }
    (mean, (crate::ops::kahan_sum(&buf) / n).sqrt())
  }

  // Deterministic xorshift64*, uniform in [-1, 1).
  let mut state: u64 = 0x2545_F491_4F6C_DD1D;
  let mut next = move || {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    let v = state.wrapping_mul(0x2545_F491_4F6C_DD1D);
    ((v >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
  };

  let options = opts(300).with_min_deviation(1e-15);
  let mut cohorts = 0usize;
  for spread in [1e-9f64, 1e-6, 1e-3, 1e-1, 5e-1] {
    for _ in 0..200 {
      // `base` and `spread` are bounded so every score lands in [-1, 1]
      // without clamping, which would manufacture ties.
      let base = next() * 0.5;
      let scores: Vec<f64> = (0..300).map(|_| base + spread * next()).collect();
      let stats = CohortStats::from_scores(scores.iter().copied(), &options).expect("usable side");
      let (mean, deviation) = unscaled(&scores);
      assert_eq!(
        stats.mean().to_bits(),
        mean.to_bits(),
        "rescaled mean {} != unscaled {mean} (spread {spread})",
        stats.mean()
      );
      assert_eq!(
        stats.deviation().to_bits(),
        deviation.to_bits(),
        "rescaled deviation {} != unscaled {deviation} (spread {spread})",
        stats.deviation()
      );
      cohorts += 1;
    }
  }
  assert_eq!(cohorts, 1_000, "the comparison must actually run");
}

// ── Trap 7: a successful normalization is finite ───────────────────────

/// The `0.5` must reach each term before they are added. Both sides here
/// standardize `1e208` to `1e308`, whose true average is `1e308` — but
/// summing first overflows and then halves an infinity.
#[test]
fn the_average_is_halved_before_the_terms_are_summed() {
  let options = opts(2).with_min_deviation(1e-101);
  let side = CohortStats::from_scores([-1e-100, 1e-100], &options).expect("usable side");
  assert_eq!(side.deviation(), 1e-100, "sigma");

  let got = as_norm(1e208, &side, &side).expect("a representable normalized score");
  assert!(got.is_finite(), "the 0.5 was applied too late: got {got}");
  assert!(
    (got - 1e308).abs() <= 1e308 * 1e-15,
    "normalized {got} != 1e308"
  );
}

/// Halving early widens the representable range but does not make it
/// infinite. When the result genuinely will not fit, it must be refused —
/// never returned as `Ok(inf)`, which a consumer comparing against a
/// fixed absolute threshold would read as an unconditional match.
#[test]
fn a_non_finite_normalized_score_is_refused() {
  let options = opts(2).with_min_deviation(1e-31);
  let side = CohortStats::from_scores([-1e-30, 1e-30], &options).expect("usable side");

  let err = as_norm(1e290, &side, &side).expect_err("1e290 / 1e-30 runs past f64::MAX");
  let Error::NonFiniteResult(v) = err else {
    panic!("expected NonFiniteResult, got {err:?}");
  };
  assert!(v.is_infinite(), "the offending value is carried: {v}");
  assert!(err.to_string().contains("not finite"), "message: {err}");
}

/// A successful normalization is finite for every side and every trial
/// score the constructor admits — the postcondition stated as a sweep,
/// not a single case.
#[test]
fn a_successful_normalization_is_always_finite() {
  let options = opts(2).with_min_deviation(f64::MIN_POSITIVE);
  let mut checked = 0usize;
  for exp in -300i32..=300 {
    let m = 10f64.powi(exp);
    let Ok(side) = CohortStats::from_scores([-m, m], &options) else {
      continue;
    };
    for raw in [0.0, 1.0, -1.0, m, -m, f64::MAX, -f64::MAX] {
      checked += 1;
      if let Ok(got) = as_norm(raw, &side, &side) {
        assert!(got.is_finite(), "Ok({got}) for raw {raw}, sigma {m}");
      }
    }
  }
  assert!(checked > 3_000, "the sweep must actually run: {checked}");
}

/// The shifted numerator must not overflow either. `from_scores` accepts
/// a cohort whose mean sits near `-1.7e308`, so `raw - mean` runs past
/// `f64::MAX` for a trial score at the other end of the range — while the
/// answer it stands for is an ordinary small number. Halving `raw` and
/// the mean separately keeps it representable, bit-neutrally.
#[test]
fn the_shifted_numerator_is_halved_before_it_can_overflow() {
  let side = CohortStats::from_scores([-1.7e308, -1.6e308], &opts(2)).expect("usable side");
  assert!((side.mean() + 1.65e308).abs() <= 1.65e308 * 1e-12, "mu");
  assert!((side.deviation() - 5e306).abs() <= 5e306 * 1e-12, "sigma");

  // (1.7e308 + 1.65e308) / 5e306 = 67, both sides alike.
  let got = as_norm(1.7e308, &side, &side).expect("an ordinary normalized score");
  assert!(
    (got - 67.0).abs() <= 67.0 * 1e-12,
    "normalized {got} != 67 — the shift overflowed"
  );
}

/// The `0.5` migration must be free as well: the naive
/// `0.5 * (t_self + t_other)` and the pushed-in form have to agree bit
/// for bit wherever the naive one is representable at all. Only the cases
/// where it is not are allowed to differ.
#[test]
fn the_pushed_in_half_is_bit_neutral_in_the_cosine_regime() {
  // Deterministic xorshift64*, uniform in [-1, 1).
  let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
  let mut next = move || {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    let v = state.wrapping_mul(0x2545_F491_4F6C_DD1D);
    ((v >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
  };

  let options = opts(8).with_min_deviation(1e-15);
  let mut trials = 0usize;
  for spread in [1e-9f64, 1e-6, 1e-3, 1e-1, 5e-1] {
    for _ in 0..400 {
      let base_e = next() * 0.5;
      let scores_e: Vec<f64> = (0..8).map(|_| base_e + spread * next()).collect();
      let base_t = next() * 0.5;
      let scores_t: Vec<f64> = (0..8).map(|_| base_t + spread * next()).collect();
      let e = CohortStats::from_scores(scores_e, &options).expect("side e");
      let t = CohortStats::from_scores(scores_t, &options).expect("side t");
      let raw = next();
      let naive = 0.5 * ((raw - e.mean()) / e.deviation() + (raw - t.mean()) / t.deviation());
      let got = as_norm(raw, &e, &t).expect("finite");
      assert_eq!(
        got.to_bits(),
        naive.to_bits(),
        "reordered {got} != naive {naive} (spread {spread}, raw {raw})"
      );
      trials += 1;
    }
  }
  assert_eq!(trials, 2_000, "the comparison must actually run");
}
