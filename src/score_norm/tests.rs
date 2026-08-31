use core::num::NonZeroUsize;

use crate::{
  embed::{EMBEDDING_DIM, Embedding, cosine_similarity},
  score_norm::{
    AsNormOptions, Cohort, CohortEntry, CohortStats, DEFAULT_MIN_DEVIATION, DEFAULT_TOP_N, Error,
    MAX_NORMALIZED_ERROR, MIN_COHORT_SCORES, as_norm,
  },
};

fn top_n(n: usize) -> NonZeroUsize {
  NonZeroUsize::new(n).expect("test top_n is non-zero")
}

fn opts(n: usize) -> AsNormOptions {
  AsNormOptions::new().with_top_n(top_n(n))
}

/// `2^e`, assembled from the bit pattern so the construction itself
/// cannot round: a normal for `e >= -1022`, a subnormal below that, and
/// the full exponent range in one expression. The exact-reference sweeps
/// build both their inputs and their expected answers out of these, so
/// nothing rounded stands between an assertion and the arithmetic it is
/// checking.
fn pow2(e: i32) -> f64 {
  assert!((-1074..=1023).contains(&e), "2^{e} is not a finite f64");
  if e >= -1022 {
    f64::from_bits(((e + 1023) as u64) << 52)
  } else {
    f64::from_bits(1u64 << (e + 1074))
  }
}

/// The smallest positive subnormal — the sharpest floor
/// [`AsNormOptions::with_min_deviation`] accepts, and the one the
/// full-range sweeps need so a `σ` at the bottom of the range is not
/// refused before it can be checked.
fn tiniest_floor() -> f64 {
  f64::from_bits(1)
}

/// A deterministic xorshift64*, uniform in `[-1, 1)`. The sweeps want
/// reproducible inputs, not good ones; the seed is the only thing that
/// distinguishes one sweep's stream from another's.
fn uniform_stream(seed: u64) -> impl FnMut() -> f64 {
  let mut state = seed;
  move || {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    let v = state.wrapping_mul(0x2545_F491_4F6C_DD1D);
    ((v >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
  }
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
  /// `from_scores` with the power-of-two rescale removed and **nothing
  /// else changed** — the same anchored two-pass over the same
  /// compensated sums. Any difference this test sees is therefore the
  /// rescale's, which is the only thing it is entitled to measure.
  fn unscaled(scores: &[f64]) -> (f64, f64) {
    let n = scores.len() as f64;
    let mut buf = scores.to_vec();
    let mean = crate::ops::kahan_sum(&buf) / n;
    let anchor = buf[0];
    for v in &mut buf {
      *v -= anchor;
    }
    let anchored_mean = crate::ops::kahan_sum(&buf) / n;
    for v in &mut buf {
      let d = *v - anchored_mean;
      *v = d * d;
    }
    (mean, (crate::ops::kahan_sum(&buf) / n).sqrt())
  }

  let mut next = uniform_stream(0x2545_F491_4F6C_DD1D);

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

/// Where nothing leaves f64's range, `normalize` must agree **bit for
/// bit** with the naive `0.5 * (t_self + t_other)` the literature writes.
/// That is what makes a threshold taken from a published AS-Norm number
/// transfer exactly, and it is the claim the range-safety rule is
/// obliged to keep.
///
/// Read honestly: in this regime both branch predicates are false, so the
/// module evaluates that very expression and the agreement is by
/// construction. What the test still catches is a *scale* creeping into
/// the unscaled path — dividing `raw` and `μ` by a data-derived
/// power of two and folding it into `σ` (the obvious unification of this
/// module's two range mechanisms, and the one that was rejected) drifts by
/// ~1e-15 relative here and would fail. The branches themselves are
/// pinned by value in
/// [`the_rescaled_branches_return_the_exact_value`] and by the two
/// exact-reference sweeps, not here.
#[test]
fn normalize_matches_the_naive_formula_bit_for_bit_in_the_cosine_regime() {
  let mut next = uniform_stream(0x9E37_79B9_7F4A_7C15);

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

// ── Trap 8: dispersion the cohort does not have ────────────────────────

/// The headline case, at the default floor: five copies of
/// `0x1.fffffffffffffp+33`.
///
/// Their sum needs 55 significant bits, so `Σx / n` cannot round back to
/// the common value and lands one ulp below it. Every element then
/// "deviates" from that rounded mean by the same `2^-19`, and `σ` is
/// reported as `2^-19 = 1.907e-6` — **above** the default `1e-6` floor,
/// so the side is accepted. Normalizing `raw = x` against it then divides
/// the fabricated deviation by itself and returns `Ok(1.0)`: a perfect
/// z-score for a cohort that does not discriminate at all, and one that
/// clears any fixed match threshold below 1.
#[test]
fn a_constant_cohort_is_not_accepted_with_a_fabricated_deviation() {
  let x = f64::from_bits(0x420f_ffff_ffff_ffff);
  let err = CohortStats::from_scores([x; 5], &AsNormOptions::new())
    .expect_err("five copies of one score do not discriminate");
  let Error::DegenerateDeviation(d) = err else {
    panic!("expected DegenerateDeviation, got {err:?}");
  };
  assert_eq!(
    d.deviation(),
    0.0,
    "a cohort of one repeated score spreads by exactly nothing"
  );
  // The floor is a policy knob, not the guard that catches this: the
  // fabricated deviation cleared it.
  assert!(
    2f64.powi(-19) > d.minimum(),
    "the fabricated 2^-19 cleared the default floor {:e}",
    d.minimum()
  );
}

/// The same defect as a property rather than a case: a cohort whose
/// selected scores are all the *same* number has a population standard
/// deviation of exactly zero. Nothing about that depends on how the mean
/// is computed — but a deviation pass anchored on `Σx / n` makes it
/// depend on it anyway.
#[test]
fn an_exactly_constant_cohort_has_exactly_zero_dispersion() {
  // Values whose repeated sum is not representable, so the rounded mean
  // is not the common value. The last is the report's case.
  let sharp = [
    0x3ff5_5555_5555_5555u64,
    0x3fef_ffff_ffff_ffff,
    0x4069_9999_9999_999a,
    0x420f_ffff_ffff_ffff,
  ];
  let mut checked = 0usize;
  for bits in sharp {
    let x = f64::from_bits(bits);
    for n in [2usize, 3, 5, 7, 11] {
      let err = CohortStats::from_scores(vec![x; n], &opts(n))
        .expect_err("a cohort of identical scores does not discriminate");
      let Error::DegenerateDeviation(d) = err else {
        panic!("expected DegenerateDeviation for {n} copies of {x:e}, got {err:?}");
      };
      assert_eq!(
        d.deviation(),
        0.0,
        "{n} copies of {x:e} spread by {:e}, but every score is the same number",
        d.deviation()
      );
      checked += 1;
    }
  }
  assert_eq!(checked, 20, "the sweep must actually run");
}

// ── Trap 9: finite is not the same as correct ──────────────────────────

/// The finiteness postcondition cannot see a wrong answer. `0.0` is
/// finite; here it is simply not the number AS-Norm defines.
///
/// With `u` the smallest positive subnormal, the cohort `[-2u, 0]` has
/// `μ = -u` and `σ = u` exactly, so for `raw = u` each side's z-score is
/// `(u - (-u)) / u = 2` and the average of two `2`s is `2`. Halving `raw`
/// and `μ` *before* the shift rounds each to a signed zero — `u/2` is not
/// representable — so the numerator collapses and the whole thing returns
/// `Ok(0.0)`. Every guard in the module passes: the inputs are finite,
/// `σ` clears its floor, and the result is finite.
#[test]
fn a_subnormal_z_score_is_not_erased_by_the_averaging() {
  let u = f64::from_bits(1);
  let options = opts(2).with_min_deviation(u);
  let side = CohortStats::from_scores([-2.0 * u, 0.0], &options).expect("usable side");
  assert_eq!(side.mean(), -u, "mu");
  assert_eq!(side.deviation(), u, "sigma");

  let got = as_norm(u, &side, &side).expect("2.0 is representable");
  assert_eq!(
    got, 2.0,
    "z-score erased: (u - -u)/u = 2 on both sides, so the average is 2"
  );
}

// ── Trap 10: the whole exponent range, checked by value ────────────────

/// `from_scores` against an **exact** reference, at every representable
/// power of two from the smallest subnormal to `2^1023`.
///
/// Each cohort is a small integer pattern times `2^e`. Both the pattern's
/// population mean and its population deviation are themselves integers,
/// so the exact answer is `(m, s) * 2^e` — an exact product, since
/// multiplying a small integer by a power of two is exact everywhere it
/// is representable, subnormals included. The assertion is therefore
/// equality, not a tolerance, and no floating-point reference computation
/// sits between it and the arithmetic under test.
///
/// This is the check the finiteness postcondition cannot make. `Ok` of a
/// finite but wrong statistic passes every guard in the module; only
/// comparing the value catches it.
#[test]
fn from_scores_matches_an_exact_reference_across_the_exponent_range() {
  // (pattern, population mean, population deviation) — all integral.
  const PATTERNS: [(&[i32], i32, i32); 6] = [
    (&[-1, 1], 0, 1),
    (&[1, 3], 2, 1),
    (&[-1, -1, 1, 1], 0, 1),
    (&[3, 3, 5, 5], 4, 1),
    (&[-4, 0, 0, 0, 0, 0, 0, 4], 0, 2),
    (&[-3, 1, 1, 1, 1, 1, 1, 5], 1, 2),
  ];

  let options = opts(8).with_min_deviation(tiniest_floor());
  let mut checked = 0usize;
  for (pattern, mean_i, deviation_i) in PATTERNS {
    let peak = f64::from(
      pattern
        .iter()
        .map(|k| k.abs())
        .max()
        .expect("non-empty pattern"),
    );
    for e in -1074i32..=1023 {
      let unit = pow2(e);
      // Skip only where an *input* would not be representable.
      if !(peak * unit).is_finite() {
        continue;
      }
      let scores: Vec<f64> = pattern.iter().map(|&k| f64::from(k) * unit).collect();
      let stats = CohortStats::from_scores(scores.iter().copied(), &options)
        .unwrap_or_else(|err| panic!("{pattern:?} * 2^{e} is a usable cohort, got {err:?}"));
      assert_eq!(
        stats.mean(),
        f64::from(mean_i) * unit,
        "mean of {pattern:?} * 2^{e}"
      );
      assert_eq!(
        stats.deviation(),
        f64::from(deviation_i) * unit,
        "deviation of {pattern:?} * 2^{e}"
      );
      checked += 1;
    }
  }
  assert!(checked > 12_000, "the sweep must actually run: {checked}");
}

/// `normalize` against an exact reference, over the same range.
///
/// Two sides are built at each scale: `[-2^e, 2^e]`, whose mean is `0` and
/// whose deviation is `2^e`, and `[0, 2^(e+1)]`, whose mean and deviation
/// are both `2^e`. For a trial score `k * 2^e` the first side's z-score is
/// exactly `k` and the second's is exactly `k - 1`, so the AS-Norm average
/// is exactly `k` for a matched pair and exactly `k - 0.5` for a mixed
/// one. Every one of those is representable, at every scale — including
/// the bottom, where `σ` is the smallest subnormal and the round-1
/// halving returned `0.0` for an answer of `2`.
#[test]
fn normalize_matches_an_exact_reference_across_the_exponent_range() {
  let options = opts(2).with_min_deviation(tiniest_floor());
  let mut checked = 0usize;
  let mut mixed = 0usize;
  for e in -1074i32..=1023 {
    let unit = pow2(e);
    let centred =
      CohortStats::from_scores([-unit, unit], &options).expect("a symmetric side at every scale");
    assert_eq!(centred.mean(), 0.0, "mu at 2^{e}");
    assert_eq!(centred.deviation(), unit, "sigma at 2^{e}");

    // Mean and deviation both `2^e`, so this side's z-score is `k - 1`.
    let offset = (2.0 * unit)
      .is_finite()
      .then(|| CohortStats::from_scores([0.0, 2.0 * unit], &options).expect("an offset side"));

    for k in [-3i32, -2, -1, 0, 1, 2, 3] {
      let raw = f64::from(k) * unit;
      if !raw.is_finite() {
        continue;
      }
      let matched = as_norm(raw, &centred, &centred).expect("an exactly representable z-score");
      assert_eq!(matched, f64::from(k), "z-score of {k} * 2^{e}");
      checked += 1;

      if let Some(offset) = offset {
        assert_eq!(offset.mean(), unit, "offset mu at 2^{e}");
        assert_eq!(offset.deviation(), unit, "offset sigma at 2^{e}");
        let got = as_norm(raw, &centred, &offset).expect("an exactly representable average");
        assert_eq!(got, f64::from(k) - 0.5, "average of {k} and {} ", k - 1);
        mixed += 1;
      }
    }
  }
  assert!(checked > 14_000, "the sweep must actually run: {checked}");
  assert!(mixed > 14_000, "the mixed pair must actually run: {mixed}");
}

/// The two rescaled branches, checked by value rather than by tolerance.
///
/// Both are reachable only at the very top of the range, so neither sweep
/// above reaches them; both are exactly representable, so neither needs a
/// tolerance.
#[test]
fn the_rescaled_branches_return_the_exact_value() {
  let options = opts(2).with_min_deviation(tiniest_floor());

  // The shift. mu = -2^1023 and sigma = 2^1022, so `raw - mu` is 2^1024 —
  // an overflow — while the answer it stands for is exactly 4.
  let side = CohortStats::from_scores([-3.0 * pow2(1022), -pow2(1022)], &options)
    .expect("a side whose mean sits at -2^1023");
  assert_eq!(side.mean(), -pow2(1023), "mu");
  assert_eq!(side.deviation(), pow2(1022), "sigma");
  assert!(
    !(pow2(1023) - side.mean()).is_finite(),
    "the unscaled shift must really overflow, or this proves nothing"
  );
  assert_eq!(
    as_norm(pow2(1023), &side, &side).expect("4 is representable"),
    4.0
  );

  // The average. Each side standardizes `raw` to exactly f64::MAX, whose
  // sum overflows and whose average is f64::MAX.
  let side = CohortStats::from_scores([-pow2(-100), pow2(-100)], &options).expect("a tiny sigma");
  assert_eq!(side.deviation(), pow2(-100), "sigma");
  let raw = f64::MAX * pow2(-100);
  assert_eq!(raw / side.deviation(), f64::MAX, "each term is f64::MAX");
  assert!(
    !(f64::MAX + f64::MAX).is_finite(),
    "the unscaled sum must really overflow"
  );
  assert_eq!(
    as_norm(raw, &side, &side).expect("f64::MAX is representable"),
    f64::MAX
  );
}

/// A one-ulp spread, measured exactly, at every scale in the normal range.
///
/// The cohort is `[x, x + ulp]` for `x = 1.5 * 2^e`. Its exact mean is
/// `x + ulp/2`, which needs 54 significant bits and so is **not**
/// representable — it rounds to `x` — while its exact deviation is
/// `ulp/2`, which is a power of two and is representable everywhere.
///
/// That gap is the whole point. Centring the second pass on the rounded
/// mean gives deviations of `0` and `ulp`, hence `σ = ulp/√2` — 41% high,
/// at every scale, for a cohort AS-Norm's top-N selection produces
/// routinely. Centring on a member gives `∓ulp/2` and the exact answer.
/// This is the same defect as the constant-cohort case one step away from
/// degenerate, where it is no longer caught by any floor because the side
/// is legitimately usable.
#[test]
fn a_one_ulp_spread_is_measured_exactly_across_the_exponent_range() {
  let options = opts(2).with_min_deviation(tiniest_floor());
  let mut checked = 0usize;
  // `ulp/2` is `2^(e-53)`, so the smallest `e` whose deviation is still
  // representable is `-1021`.
  for e in -1021i32..=1023 {
    let x = pow2(e) + pow2(e - 1); // 1.5 * 2^e, exact
    let ulp = pow2(e - 52);
    let stats = CohortStats::from_scores([x, x + ulp], &options)
      .unwrap_or_else(|err| panic!("[x, x+ulp] at 2^{e} is usable, got {err:?}"));
    assert_eq!(stats.mean(), x, "mean at 2^{e} (exact mean rounds to x)");
    assert_eq!(stats.deviation(), pow2(e - 53), "deviation at 2^{e}");
    checked += 1;
  }
  assert_eq!(checked, 2_045, "the sweep must actually run: {checked}");
}

// ── Trap 11: an answer made entirely of rounding error ─────────────────

/// `raw` for the cancellation case below. Nothing about it is special —
/// it is the trial score the two cohorts were solved around.
const CANCELLING_RAW: f64 = 0.3;

/// A cohort whose selected pair is `[lo, hi]`, given as bit patterns so
/// the inputs cannot be perturbed by a decimal literal's rounding.
fn side_from_bits(lo: u64, hi: u64, options: &AsNormOptions) -> CohortStats {
  CohortStats::from_scores([f64::from_bits(lo), f64::from_bits(hi)], options)
    .expect("both cohorts clear the default deviation floor")
}

/// `x` as an exact `m * 2^e` with `m` odd (or zero).
fn dyadic(x: f64) -> (i128, i32) {
  assert!(x.is_finite(), "{x} is not finite");
  if x == 0.0 {
    return (0, 0);
  }
  let bits = x.to_bits();
  let biased = ((bits >> 52) & 0x7ff) as i32;
  let frac = i128::from(bits & 0x000f_ffff_ffff_ffff);
  let (mut m, mut e) = if biased == 0 {
    (frac, -1074)
  } else {
    (frac | (1i128 << 52), biased - 1075)
  };
  while m % 2 == 0 {
    m /= 2;
    e += 1;
  }
  if bits >> 63 == 1 { (-m, e) } else { (m, e) }
}

fn gcd(a: i128, b: i128) -> i128 {
  let (mut a, mut b) = (a.abs(), b.abs());
  while b != 0 {
    let t = a % b;
    a = b;
    b = t;
  }
  a.max(1)
}

/// `(raw - mu) / sigma` as an exact reduced rational, denominator > 0.
///
/// Every step is integer arithmetic on the dyadic parts of the stored
/// statistics, so nothing between the assertion and the module's own
/// inputs is rounded. Panics rather than wrapping if a case needs more
/// than `i128`, so an input this cannot represent exactly fails loudly.
fn exact_z(raw: f64, mean: f64, deviation: f64) -> (i128, i128) {
  let (mr, er) = dyadic(raw);
  let (mm, em) = dyadic(mean);
  let (ms, es) = dyadic(deviation);
  assert!(ms > 0, "a stored deviation is positive");
  let e = er.min(em);
  assert!(er - e < 120 && em - e < 120, "exponent span exceeds i128");
  let shift = |m: i128, k: i32| {
    m.checked_mul(1i128 << k)
      .unwrap_or_else(|| panic!("dyadic shift overflows i128"))
  };
  let numerator = shift(mr, er - e)
    .checked_sub(shift(mm, em - e))
    .expect("shifted difference fits i128");
  // numerator * 2^e / (ms * 2^es)
  let (mut num, mut den) = if e >= es {
    (shift(numerator, e - es), ms)
  } else {
    (numerator, shift(ms, es - e))
  };
  let g = gcd(num, den);
  num /= g;
  den /= g;
  (num, den)
}

/// The exact AS-Norm average of two sides, as a reduced rational with a
/// positive denominator.
fn exact_as_norm(raw: f64, a: &CohortStats, b: &CohortStats) -> (i128, i128) {
  let (n1, d1) = exact_z(raw, a.mean(), a.deviation());
  let (n2, d2) = exact_z(raw, b.mean(), b.deviation());
  let num = n1
    .checked_mul(d2)
    .and_then(|l| n2.checked_mul(d1).and_then(|r| l.checked_add(r)))
    .expect("cross-multiplied sum fits i128");
  let den = d1
    .checked_mul(d2)
    .and_then(|d| d.checked_mul(2))
    .expect("halved denominator fits i128");
  let g = gcd(num, den);
  (num / g, den / g)
}

/// `|num/den − value|` as an `f64`, with the cancellation between the two
/// taken in exact integers before anything is converted.
///
/// Everywhere this is used, `value` is what the module returned and
/// `num/den` is what it should have returned, so the two agree to within
/// the error being measured. Converting either on its own would round that
/// error away completely. Only the small quotient that survives the exact
/// subtraction is converted, which costs two roundings **of the
/// difference** — a relative `2^-52` against a bound whose own slack is
/// `2^-50`, so the comparison has room to spare in the direction that
/// matters.
fn exact_gap(num: i128, den: i128, value: f64) -> f64 {
  let (vm, ve) = dyadic(value);
  assert!(ve.abs() < 120, "2^{ve} does not fit an i128 shift");
  let shift = |m: i128, k: i32| {
    m.checked_mul(1i128 << k)
      .unwrap_or_else(|| panic!("dyadic shift overflows i128"))
  };
  let (dn, dd) = if ve >= 0 {
    (num - shift(vm, ve) * den, den)
  } else {
    (shift(num, -ve) - vm * den, shift(den, -ve))
  };
  ((dn as f64) / (dd as f64)).abs()
}

/// Two sides whose z-scores are **adjacent floats of opposite sign** at
/// `2^50`: their sum is exactly one ulp, so the entire returned value is
/// a single rounding step of the two divisions — and its sign is not the
/// answer's.
///
/// Both sides are ordinary: `from_scores` accepts them at the **default**
/// `1e-6` floor, with deviations of `7 * 2^-22` and `11 * 2^-22`. Every
/// intermediate stays finite, so neither the range-safety branches nor
/// the finiteness postcondition can see anything wrong. The module
/// returns `0.125`; the exact average of the two stored z-scores is
/// `-6.75e-2`. A threshold at zero reads a non-match as a match.
///
/// Distributing the halving does not help — the error was made in the
/// divisions, before any addition — so `0.5 * a + 0.5 * b` is `0.125`
/// too. The cure has to be a refusal, not a reordering.
#[test]
fn cancelling_z_scores_are_refused_rather_than_returned_as_rounding_noise() {
  let options = AsNormOptions::new();
  let e = side_from_bits(0x41e0_2a33_6dc3_f818, 0x41e0_2a33_6dc3_f81f, &options);
  let t = side_from_bits(0xc1e9_66e3_1a1b_4414, 0xc1e9_66e3_1a1b_4409, &options);
  assert_eq!(e.deviation(), 7.0 * pow2(-22), "sigma_e");
  assert_eq!(t.deviation(), 11.0 * pow2(-22), "sigma_t");
  assert!(
    e.deviation() > options.min_deviation() && t.deviation() > options.min_deviation(),
    "both sides clear the default floor, so nothing earlier refuses them"
  );

  // The exact answer, from the stored statistics, by integer arithmetic.
  let (num, den) = exact_as_norm(CANCELLING_RAW, &e, &t);
  assert!(den > 0, "denominator normalised positive");
  assert!(num < 0, "the exact average is negative: {num}/{den}");
  // -0.0676 < exact < -0.0675, checked by cross-multiplication.
  assert!(
    num * 10_000 > -676 * den && num * 10_000 < -675 * den,
    "exact average {num}/{den} is not -6.75e-2"
  );

  // Reordering cannot rescue it: the error was made in the divisions,
  // before either term reached an addition, so distributing the halving
  // returns the same wrong number the fused form does.
  let z_e = (CANCELLING_RAW - e.mean()) / e.deviation();
  let z_t = (CANCELLING_RAW - t.mean()) / t.deviation();
  assert_eq!(
    z_e.to_bits() ^ (1u64 << 63),
    z_t.to_bits() - 1,
    "adjacent ulps"
  );
  assert_eq!(z_e + z_t, 0.25, "the sum is exactly one ulp at 2^50");
  assert_eq!(0.5 * (z_e + z_t), 0.125);
  assert_eq!(
    0.5 * z_e + 0.5 * z_t,
    0.125,
    "distributing the halving is no cure"
  );

  let err = as_norm(CANCELLING_RAW, &e, &t).expect_err("an answer made of rounding");
  let Error::ZScoreCancellation(c) = err else {
    panic!("expected ZScoreCancellation, got {err:?}");
  };
  assert_eq!(c.z_self(), z_e, "the offending z-scores are carried");
  assert_eq!(c.z_other(), z_t);
  assert_eq!(c.normalized(), 0.125, "and the value that was refused");
  assert!(
    c.error_bound() > c.tolerance(),
    "the refusal is the bound exceeding the tolerance: {:e} vs {:e}",
    c.error_bound(),
    c.tolerance()
  );

  // Round 4: the refusal is the *refined* figure's, and the payload
  // reports the number the predicate actually compared. Recomputing it
  // from the same function is the anti-drift guard — a `Display` that
  // formatted anything else would have to disagree with this.
  let cheap = crate::score_norm::stats::z_score_error_bound(z_e, z_t);
  let refined =
    crate::score_norm::stats::refined_error_bound(CANCELLING_RAW, &e, &t, z_e, z_t, cheap);
  assert_eq!(
    c.error_bound(),
    refined,
    "the payload must report the refined bound, not the cheap one ({cheap:e})"
  );

  // And the refined figure is the error itself, checked against the exact
  // rational above rather than against the machinery that produced it.
  let gap = exact_gap(num, den, 0.125);
  assert!(
    gap <= refined,
    "the refinement must still be a bound: gap {gap:e} exceeds {refined:e}"
  );
  assert!(
    refined <= gap * (1.0 + 1e-14),
    "and it must be the error, not a bound around it: {refined:e} vs {gap:e}"
  );

  // Three times tighter than the filter that selected the trial, and
  // still five orders of magnitude past the tolerance — so the verdict is
  // exactly the one round 3 reached, reached for a reason that holds.
  assert!(
    (2.9..3.1).contains(&(cheap / refined)),
    "the refinement is {}x tighter, not 3x",
    cheap / refined
  );
  assert!(
    c.error_bound() > 2.0e5 * c.tolerance() && c.error_bound() < 2.1e5 * c.tolerance(),
    "bound {:e} is {}x the tolerance",
    c.error_bound(),
    c.error_bound() / c.tolerance()
  );
  assert!(err.to_string().contains("cancel to"), "message: {err}");
  assert!(
    err.to_string().contains("1.925e-1"),
    "the message must carry the refined bound: {err}"
  );
}

/// The exact average of the two z-scores, recovered with error-free
/// transformations rather than asserted.
///
/// `f64::mul_add` gives `d − q·σ` exactly, Knuth's two-sum gives the
/// subtraction's residual exactly, and the crate's compensated sum adds
/// the four terms without losing them again. The result is good to about
/// `2^-104` relative — some fifty binades finer than the `2^-51` bound it
/// exists to check, so it can serve as the reference for it.
///
/// This is the machinery [`CohortStats::normalize`] deliberately does not
/// carry; here it costs nothing, because a test is not on anyone's hot
/// path.
fn oracle_as_norm(raw: f64, a: &CohortStats, b: &CohortStats) -> f64 {
  fn exact_terms(raw: f64, mean: f64, sigma: f64) -> (f64, f64) {
    // Two-sum of `raw + (-mean)`: `residual` is exact.
    let d = raw - mean;
    let split = d - raw;
    let residual = (raw - (d - split)) + (-mean - split);
    let q = d / sigma;
    // `d - q * sigma`, exact under a single rounding.
    let quotient_residual = (-q).mul_add(sigma, d);
    (q, (residual + quotient_residual) / sigma)
  }
  let (q_a, e_a) = exact_terms(raw, a.mean(), a.deviation());
  let (q_b, e_b) = exact_terms(raw, b.mean(), b.deviation());
  0.5 * crate::ops::kahan_sum(&[q_a, q_b, e_a, e_b])
}

/// The accuracy claim, checked against the error-free reference rather
/// than restated: whenever `normalize` returns, the returned value really
/// is within [`z_score_error_bound`] of the exact average.
///
/// The sweep is built to press on the bound, not to sit comfortably
/// inside it — deviations that are odd multiples of a power of two, so
/// both divisions round, at scales from `2^-40` to `2^40`, with trial
/// scores placed both far from the two means (no cancellation, where the
/// answer's own rounding adds to the inherited error) and between them
/// (cancellation, where only the inherited error survives). The second
/// assertion is what keeps the first honest: if no case came near the
/// bound, a bound half the size would pass too.
#[test]
fn the_error_bound_really_bounds_the_error() {
  let options = opts(2).with_min_deviation(tiniest_floor());
  let mut next = uniform_stream(0x9E37_79B9_7F4A_7C15);
  let mut worst = 0.0f64;
  let mut checked = 0usize;
  for e in (-40i32..=40).step_by(1) {
    let unit = pow2(e);
    for &k in &[3.0f64, 5.0, 7.0, 11.0, 13.0] {
      let sigma = k * unit;
      // A side centred on `centre` with deviation `sigma`.
      let side = |centre: f64| CohortStats::from_scores([centre - sigma, centre + sigma], &options);
      for &offset in &[0.0f64, 1.0, 7.0, 1e5, 1e9] {
        let Ok(a) = side(offset * unit) else { continue };
        let Ok(b) = side(-offset * unit * 1.5 + unit) else {
          continue;
        };
        for _ in 0..6 {
          // Both regimes: a raw score near the two means, and one far
          // enough out that the average does not cancel at all.
          for raw in [
            next() * offset.max(1.0) * unit,
            next() * offset.max(1.0) * unit * 1e6,
          ] {
            let Ok(got) = as_norm(raw, &a, &b) else {
              continue;
            };
            // The direct form is `z_score`'s answer only where the shift
            // does not overflow; assert that rather than assume it, or a
            // case that took the rescaled branch would be measured
            // against an infinite bound and pass for the wrong reason.
            assert!(
              (raw - a.mean()).is_finite() && (raw - b.mean()).is_finite(),
              "the sweep must stay on the direct branch (raw {raw:e})"
            );
            let bound = crate::score_norm::stats::z_score_error_bound(
              (raw - a.mean()) / a.deviation(),
              (raw - b.mean()) / b.deviation(),
            );
            let exact = oracle_as_norm(raw, &a, &b);
            let error = (exact - got).abs();
            assert!(
              error <= bound,
              "error {error:e} exceeds the bound {bound:e} (raw {raw:e}, sigma {sigma:e})"
            );
            if bound > 0.0 {
              worst = worst.max(error / bound);
            }
            checked += 1;
          }
        }
      }
    }
  }
  assert!(checked > 5_000, "the sweep must actually run: {checked}");
  assert!(
    worst > 0.45,
    "no case came near the bound (worst {worst:.4}), so it is not being tested"
  );
}

/// The property the absolute floor in [`permitted_error`] exists for, and
/// the one round 4 had to add to it: a trial whose two sides cancel
/// **exactly** is answered at *every* magnitude — including the ones the
/// cheap filter cannot clear.
///
/// Side `a` is centred on zero with `σ = 1`, so its z-score is `raw`;
/// side `b` is centred on `2·raw`, so its z-score is `-raw`. Both are
/// exact for as long as `2·raw ± 1` is representable, which reaches
/// `2^51`. The average is exactly `0` — the value a match threshold sits
/// closest to, and the one a purely relative accuracy criterion would
/// refuse for having no significant digits left. It has no digits because
/// it is zero.
///
/// The closing assertion is the dimension the round-3 margin table was
/// missing. That table measured how far *below* the firing point real
/// data sits; this measures how far *above* it a result can sit while
/// still being exactly right. At `2^51` the cheap bound is `2^20` times
/// the tolerance — a million-fold past the point where round 3 began
/// refusing — and the answer is still an exact `0`.
#[test]
fn an_exactly_cancelling_pair_is_answered_however_far_above_the_filter() {
  let options = opts(2);
  let mut checked = 0usize;
  let mut worst_filter_ratio = 0.0f64;
  for e in 0i32..=51 {
    let raw = pow2(e);
    let a = CohortStats::from_scores([-1.0, 1.0], &options).expect("centred side");
    let b = CohortStats::from_scores([2.0 * raw - 1.0, 2.0 * raw + 1.0], &options)
      .expect("side centred on 2*raw");
    assert_eq!(a.mean(), 0.0);
    assert_eq!(a.deviation(), 1.0);
    assert_eq!(b.mean(), 2.0 * raw, "mu_b at 2^{e}");
    assert_eq!(b.deviation(), 1.0, "sigma_b at 2^{e}");
    assert_eq!(
      as_norm(raw, &a, &b).unwrap_or_else(|err| panic!("refused 2^{e}: {err}")),
      0.0,
      "z-scores of +/-2^{e} average to exactly zero"
    );
    worst_filter_ratio = worst_filter_ratio.max(
      crate::score_norm::stats::z_score_error_bound(raw, -raw)
        / crate::score_norm::stats::permitted_error(0.0),
    );
    checked += 1;
  }
  assert_eq!(checked, 52, "the sweep must actually run");
  assert_eq!(
    worst_filter_ratio,
    pow2(20),
    "an exact answer must survive a million-fold past the filter's firing point"
  );
}

/// Round 4's falsifier. An **exactly correct** result, refused.
///
/// The cheap bound is an upper bound, so clearing it proves a result
/// sound — but exceeding it proves only that the result *might* be
/// unsound, and the round-3 guard read the second as the first. The
/// construction above, carried one binade past the point where the cheap
/// bound overtakes the tolerance, is the counterexample: at `M = 2^32`
/// both z-scores are exactly `±M`, so their average is exactly `0` with
/// no arithmetic error whatsoever, and the guard refuses it.
///
/// `0` is the most threshold-relevant value this module produces, which
/// is the whole reason [`permitted_error`] carries an absolute floor —
/// and the floor did not save this one, because the predicate never asked
/// what the error *was*.
#[test]
fn an_exactly_cancelling_pair_above_the_cheap_bound_is_still_answered() {
  let options = opts(2);
  let m = pow2(32);
  let a = CohortStats::from_scores([-1.0, 1.0], &options).expect("centred side");
  let b = CohortStats::from_scores([2.0 * m - 1.0, 2.0 * m + 1.0], &options)
    .expect("side centred on 2*M");
  assert_eq!(a.mean(), 0.0, "mu_a");
  assert_eq!(a.deviation(), 1.0, "sigma_a");
  assert_eq!(b.mean(), 2.0 * m, "mu_b");
  assert_eq!(b.deviation(), 1.0, "sigma_b");

  // Both z-scores are exact: the shift is a difference of two powers of
  // two and the divisor is one, so neither operation rounds.
  let z_a = (m - a.mean()) / a.deviation();
  let z_b = (m - b.mean()) / b.deviation();
  assert_eq!(z_a, m, "z_a is exactly M");
  assert_eq!(z_b, -m, "z_b is exactly -M");
  assert_eq!(z_a + z_b, 0.0, "the exact average is exactly zero");

  // The cheap bound does fire — 2^-19 against a tolerance of 2^-20 — so
  // this is the tier-2 path and not an accident of the tier-1 predicate.
  let cheap = crate::score_norm::stats::z_score_error_bound(z_a, z_b);
  let tolerance = crate::score_norm::stats::permitted_error(0.0);
  assert_eq!(cheap, pow2(-19), "the cheap bound at mag 2^32");
  assert_eq!(tolerance, MAX_NORMALIZED_ERROR, "the absolute floor");
  assert!(
    cheap > tolerance,
    "the cheap filter must fire, or nothing is proved"
  );

  // The second tier finds nothing to charge it for. What is left is the
  // floor beneath the refinement and nothing else: `RESIDUAL_UNDERFLOW`
  // divided by each side's `σ`, which is `1` on both, averaged.
  assert_eq!(
    crate::score_norm::stats::refined_error_bound(m, &a, &b, z_a, z_b, cheap),
    f64::MIN_POSITIVE * f64::EPSILON,
    "every residual here is zero, so only the underflow floor survives"
  );

  assert_eq!(
    as_norm(m, &a, &b).unwrap_or_else(|err| panic!("refused an exact zero: {err}")),
    0.0,
    "an exactly computed zero is an answer, not a cancellation"
  );
}

/// The **filter's** boundary is exactly `2^31`, checked one ulp either
/// side of it — and crossing it decides nothing on its own.
///
/// With the construction above the two z-scores are `±raw`, so
/// `mag == raw` and the cheap bound is `2^-51 · raw` against a tolerance
/// of `MAX_NORMALIZED_ERROR`. At `raw = 2^31` the two are exactly equal
/// and the filter does not fire; one ulp higher it does. That is the
/// predicate's boundary rather than its middle, so a `>` quietly becoming
/// a `>=` has somewhere to fail.
///
/// What round 4 changed is what happens on the far side. Firing hands the
/// trial to the second tier, which finds no error to refuse it for, so
/// **both** magnitudes are answered and with the same exact `0`. Round 3
/// returned `false` for the second row: that was the defect, and this is
/// the row that pins its absence.
#[test]
fn the_cheap_filter_fires_exactly_above_two_to_the_thirty_first() {
  let options = opts(2);
  let a = CohortStats::from_scores([-1.0, 1.0], &options).expect("centred side");
  let admissible = pow2(31);
  let over = admissible + pow2(-21); // one ulp at 2^31

  for (raw, fires) in [(admissible, false), (over, true)] {
    let b = CohortStats::from_scores([2.0 * raw - 1.0, 2.0 * raw + 1.0], &options)
      .expect("side centred on 2*raw");
    assert_eq!(b.mean(), 2.0 * raw, "mu_b");
    assert_eq!(b.deviation(), 1.0, "sigma_b");
    assert_eq!(
      crate::score_norm::stats::z_score_error_bound(raw, -raw)
        > crate::score_norm::stats::permitted_error(0.0),
      fires,
      "raw {raw:e} should {} fire the filter",
      if fires { "" } else { "not" }
    );
    assert_eq!(
      as_norm(raw, &a, &b).unwrap_or_else(|err| panic!("refused {raw:e}: {err}")),
      0.0,
      "the filter is not the verdict: raw {raw:e} is answered either way"
    );
  }
  assert_eq!(
    crate::score_norm::stats::z_score_error_bound(admissible, -admissible),
    MAX_NORMALIZED_ERROR,
    "2^-51 * 2^31 is exactly the tolerance floor"
  );
}

/// The number that decides whether this guard is worth having: how close
/// the regimes this module was *measured* on come to firing it.
///
/// A guard that trips on real cosine or PLDA scores would be worse than
/// the defect it prevents, so the margin is asserted rather than
/// described. Each regime reports `error_bound / permitted_error`; the
/// pinned ceilings are just above the measured worst case, so a change
/// that moves the guard toward real data fails here first.
#[test]
fn the_guard_stays_orders_of_magnitude_clear_of_every_measured_regime() {
  fn worst_ratio(sides: &[(CohortStats, CohortStats, f64)]) -> f64 {
    let mut worst = 0.0f64;
    for (a, b, raw) in sides {
      let z_a = (raw - a.mean()) / a.deviation();
      let z_b = (raw - b.mean()) / b.deviation();
      let got = as_norm(*raw, a, b).expect("every regime here is answered");
      worst = worst.max(
        crate::score_norm::stats::z_score_error_bound(z_a, z_b)
          / crate::score_norm::stats::permitted_error(got),
      );
    }
    worst
  }

  // Cosine, across the spreads the bit-neutrality sweep uses, at a floor
  // low enough to admit the tightest of them.
  let options = opts(8).with_min_deviation(1e-15);
  let mut next = uniform_stream(0x9E37_79B9_7F4A_7C15);
  for spread in [1e-9f64, 1e-6, 1e-3, 1e-1, 5e-1] {
    let mut cases = Vec::new();
    for _ in 0..400 {
      let base_e = next() * 0.5;
      let e: Vec<f64> = (0..8).map(|_| base_e + spread * next()).collect();
      let base_t = next() * 0.5;
      let t: Vec<f64> = (0..8).map(|_| base_t + spread * next()).collect();
      let raw = next();
      if let (Ok(e), Ok(t)) = (
        CohortStats::from_scores(e, &options),
        CohortStats::from_scores(t, &options),
      ) {
        cases.push((e, t, raw));
      }
    }
    assert!(
      cases.len() > 350,
      "spread {spread:e} produced {}",
      cases.len()
    );
    let worst = worst_ratio(&cases);
    assert!(
      worst < 1e-6,
      "cosine spread {spread:e} reaches {worst:e} of the guard"
    );
  }

  // The crate's own tight-cluster-at-1e8 cohort, and PLDA-scale
  // log-likelihood ratios.
  let wide = opts(4);
  let cluster_a = CohortStats::from_scores([1e8 + 1.0, 1e8 + 2.0, 1e8 + 3.0, 1e8 + 4.0], &wide)
    .expect("tight cluster");
  let cluster_b = CohortStats::from_scores([1e8 + 1.0, 1e8 + 2.0, 1e8 + 3.5, 1e8 + 4.0], &wide)
    .expect("tight cluster");
  let cluster: Vec<_> = [0.8f64, 1e8, 1e8 + 2.5, -1e8]
    .into_iter()
    .map(|raw| (cluster_a, cluster_b, raw))
    .collect();
  assert!(
    worst_ratio(&cluster) < 1e-8,
    "the tight cluster reaches {:e} of the guard",
    worst_ratio(&cluster)
  );

  let mut plda = Vec::new();
  for _ in 0..400 {
    let base_e = next() * 30.0;
    let e: Vec<f64> = (0..8).map(|_| base_e + 8.0 * next()).collect();
    let base_t = next() * 30.0;
    let t: Vec<f64> = (0..8).map(|_| base_t + 8.0 * next()).collect();
    let raw = next() * 40.0;
    if let (Ok(e), Ok(t)) = (
      CohortStats::from_scores(e, &opts(8)),
      CohortStats::from_scores(t, &opts(8)),
    ) {
      plda.push((e, t, raw));
    }
  }
  assert!(plda.len() > 350, "PLDA regime produced {}", plda.len());
  assert!(
    worst_ratio(&plda) < 1e-7,
    "the PLDA regime reaches {:e} of the guard",
    worst_ratio(&plda)
  );

  // The worst case the *default* floor admits at all, constructed rather
  // than sampled: two cosine sides with sigma exactly on the floor and
  // means placed symmetrically about the trial score, so the cancellation
  // is total and the z-scores are as large as `[-1, 1]` permits.
  let floor = DEFAULT_MIN_DEVIATION;
  let pinned = opts(2);
  // A hair over `2 * floor`, because `-1.0 + 2e-6` rounds the span a few
  // ulps short of it and the side would be refused as degenerate before
  // it could be measured.
  let span = 2.0 * floor * (1.0 + 1e-9);
  let a = CohortStats::from_scores([-1.0, -1.0 + span], &pinned).expect("side at the floor");
  let b = CohortStats::from_scores([1.0 - span, 1.0], &pinned).expect("side at the floor");
  assert!(
    a.deviation() < 1.01 * floor && b.deviation() < 1.01 * floor,
    "sigma on the floor"
  );
  let worst = worst_ratio(&[(a, b, 0.0)]);
  assert!(
    worst < 1e-3,
    "the widest cosine trial the default floor admits reaches {worst:e} of the guard"
  );
}
/// What the second tier buys, measured rather than described: how much
/// tighter the recovered error is than the operand bound that would send
/// a trial to it.
///
/// Both figures are computed for every case over the same cosine regimes
/// the margin table uses. The second tier is never *reached* there — that
/// is the first tier's whole job — but the comparison is what says the
/// fallback would be worth having if it were, and it is the number the
/// module docs quote.
///
/// The ratio can never fall below one: [`refined_error_bound`] closes with
/// a `min` against the bound that selected the trial, so the two-tier
/// predicate cannot refuse anything the one-tier predicate answered.
/// Measured worst-case minimum across these regimes is `1.7`, median `8.8`
/// to `9.6`; the pins sit well inside that so a NEON-versus-scalar
/// difference in the cohort statistics cannot flake the test.
#[test]
fn the_refinement_is_measurably_tighter_than_the_filter_that_selects_it() {
  let options = opts(8).with_min_deviation(1e-15);
  let mut next = uniform_stream(0x9E37_79B9_7F4A_7C15);
  for spread in [1e-9f64, 1e-6, 1e-3, 1e-1, 5e-1] {
    let mut ratios = Vec::new();
    for _ in 0..400 {
      let base_e = next() * 0.5;
      let e: Vec<f64> = (0..8).map(|_| base_e + spread * next()).collect();
      let base_t = next() * 0.5;
      let t: Vec<f64> = (0..8).map(|_| base_t + spread * next()).collect();
      let raw = next();
      let (Ok(a), Ok(b)) = (
        CohortStats::from_scores(e, &options),
        CohortStats::from_scores(t, &options),
      ) else {
        continue;
      };
      let z_a = (raw - a.mean()) / a.deviation();
      let z_b = (raw - b.mean()) / b.deviation();
      let cheap = crate::score_norm::stats::z_score_error_bound(z_a, z_b);
      let refined = crate::score_norm::stats::refined_error_bound(raw, &a, &b, z_a, z_b, cheap);
      assert!(
        refined > 0.0 && refined <= cheap,
        "refined {refined:e} is not a tightening of {cheap:e}"
      );
      ratios.push(cheap / refined);
    }
    assert!(
      ratios.len() > 350,
      "spread {spread:e} produced {}",
      ratios.len()
    );
    ratios.sort_by(f64::total_cmp);
    let median = ratios[ratios.len() / 2];
    assert!(
      ratios[0] > 1.5,
      "spread {spread:e}: the refinement is only {}x tighter at worst",
      ratios[0]
    );
    assert!(
      median > 5.0,
      "spread {spread:e}: the refinement is only {median}x tighter at the median"
    );
  }
}

/// The second tier on the shift's **rescaled** branch, where the numerator
/// it has to recover is not the one `raw − μ` produced.
///
/// `μ_a = -2^1023` against `raw = 2^1023` overflows the shift, so
/// [`CohortStats::z_score`] takes the halved form — and a two-sum over
/// `raw` and `-μ_a` would recover the residual of an infinity. `ZTerms`
/// carries the halved operands instead, which are exact here because both
/// are large normals.
///
/// Every quantity is a power of two, so both z-scores are exactly
/// `±2^34` and the exact average is exactly `0`. The cheap bound is eight
/// times the tolerance, so the trial does reach the second tier; the
/// second tier finds no residual at all and answers it.
#[test]
fn the_refinement_recovers_the_rescaled_branchs_residuals() {
  let options = AsNormOptions::new();
  let a = CohortStats::from_scores([-pow2(1023) - pow2(990), -pow2(1023) + pow2(990)], &options)
    .expect("a side centred on -2^1023");
  let mu_b = pow2(1023) + pow2(1022); // 1.5 * 2^1023
  let b = CohortStats::from_scores([mu_b - pow2(988), mu_b + pow2(988)], &options)
    .expect("a side centred on 1.5 * 2^1023");
  assert_eq!(a.mean(), -pow2(1023), "mu_a");
  assert_eq!(a.deviation(), pow2(990), "sigma_a");
  assert_eq!(b.mean(), mu_b, "mu_b");
  assert_eq!(b.deviation(), pow2(988), "sigma_b");

  let raw = pow2(1023);
  assert!(
    !(raw - a.mean()).is_finite(),
    "side a must take the rescaled branch, or the test proves nothing"
  );
  let z_a = (0.5 * raw - 0.5 * a.mean()) / (0.5 * a.deviation());
  let z_b = (raw - b.mean()) / b.deviation();
  assert_eq!(z_a, pow2(34), "z_a is exactly 2^34");
  assert_eq!(z_b, -pow2(34), "z_b is exactly -2^34");

  let cheap = crate::score_norm::stats::z_score_error_bound(z_a, z_b);
  assert_eq!(
    cheap / crate::score_norm::stats::permitted_error(0.0),
    8.0,
    "the filter must fire, or the second tier is not exercised"
  );
  assert_eq!(
    crate::score_norm::stats::refined_error_bound(raw, &a, &b, z_a, z_b, cheap),
    0.0,
    "the halved operands are exact, so there is no residual to find"
  );
  assert_eq!(
    as_norm(raw, &a, &b).unwrap_or_else(|err| panic!("refused an exact zero: {err}")),
    0.0
  );
}

/// Where the two z-scores' *sum* overflows, the two-sum the refinement
/// needs cannot be taken — and the refinement says so by falling back to
/// the bound that selected it, rather than by deciding a comparison
/// against a `NaN`.
///
/// The predicate cannot reach this: an overflowing sum needs two
/// same-signed terms, which cancel by nothing, so the first tier clears
/// them by `2^-31` and never asks. Asserted directly all the same, because
/// "unreachable" is a property of today's first tier and not of this
/// function — and because a `NaN` here would compare `false` against the
/// tolerance and *accept*, which is the one direction a guard must not
/// fail in.
#[test]
fn a_refinement_that_cannot_be_taken_falls_back_to_the_filter() {
  let options = opts(2).with_min_deviation(1e-101);
  let side = CohortStats::from_scores([-1e-100, 1e-100], &options).expect("usable side");
  let raw = 1e208;
  let z = (raw - side.mean()) / side.deviation();
  assert_eq!(z, 1e308, "each side standardizes to 1e308");
  assert!(!(z + z).is_finite(), "and their sum overflows");

  let cheap = crate::score_norm::stats::z_score_error_bound(z, z);
  let refined = crate::score_norm::stats::refined_error_bound(raw, &side, &side, z, z, cheap);
  assert_eq!(refined, cheap, "an untakeable refinement refines nothing");
  assert!(refined.is_finite(), "and it is never a NaN: {refined}");
  assert!(
    cheap < crate::score_norm::stats::permitted_error(1e308),
    "the filter clears this regime anyway, by 2^-31"
  );
  assert_eq!(as_norm(raw, &side, &side).expect("answered"), 1e308);
}
