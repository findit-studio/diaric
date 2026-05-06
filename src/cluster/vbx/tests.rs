//! Model-free unit tests for `diarization::cluster::vbx`.
//!
//! Heavy parity tests against pyannote's captured outputs live in
//! `src/vbx/parity_tests.rs`. This module covers smaller, model-free
//! invariants — the kind of thing that should hold for any input,
//! and that catches regressions long before the parity tests fail.

use super::algo::logsumexp_rows;
use nalgebra::DMatrix;

/// scipy.special.logsumexp on a 2x3 matrix along axis=-1 returns a
/// length-2 vector. Reference values computed in Python:
///
/// ```python
/// >>> import math
/// >>> vals = [-100.0, -101.0, -102.0]; mx = max(vals)
/// >>> math.log(sum(math.exp(v - mx) for v in vals)) + mx
/// -99.59239403555561
/// ```
///
/// Row0: logsumexp([1, 2, 3]) = log(e^1 + e^2 + e^3) ≈ 3.40760596
/// Row1: logsumexp([-100, -101, -102]) ≈ -99.59239403555561
#[test]
fn logsumexp_rows_matches_scipy_reference() {
  let m = DMatrix::<f64>::from_row_slice(2, 3, &[1.0, 2.0, 3.0, -100.0, -101.0, -102.0]);
  let lse = logsumexp_rows(&m);
  assert!((lse[0] - 3.40760596).abs() < 1e-8, "row0: {}", lse[0]);
  assert!(
    (lse[1] - (-99.592_394_035_555_61)).abs() < 1e-10,
    "row1: {}",
    lse[1]
  );
}

/// All -inf row → -inf result (matches scipy behavior).
#[test]
fn logsumexp_rows_all_neg_inf_returns_neg_inf() {
  let m = DMatrix::<f64>::from_row_slice(
    1,
    3,
    &[f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY],
  );
  let lse = logsumexp_rows(&m);
  assert!(lse[0].is_infinite() && lse[0] < 0.0, "got {}", lse[0]);
}

use crate::cluster::vbx::{Error, vbx_iterate};
use nalgebra::DVector;

/// Deterministic non-uniform qinit for tests. Each row `tt` is peaked
/// on speaker `tt % s` with mass 0.95; the remaining 0.05 mass is
/// split evenly across the other speakers.
fn deterministic_qinit(t: usize, s: usize) -> DMatrix<f64> {
  assert!(s > 1, "deterministic_qinit requires S > 1");
  let off = 0.05 / (s - 1) as f64;
  DMatrix::<f64>::from_fn(t, s, |tt, sj| if sj == tt % s { 0.95 } else { off })
}

#[test]
fn vbx_rejects_phi_with_non_positive_entry() {
  let x = DMatrix::<f64>::zeros(5, 4);
  let mut phi = DVector::<f64>::from_element(4, 1.0);
  phi[2] = -0.5;
  let qinit = DMatrix::<f64>::from_element(5, 2, 0.5);
  let result = vbx_iterate(x.as_view(), &phi, &qinit, 0.07, 0.8, 20);
  assert!(
    matches!(result, Err(Error::NonPositivePhi(_, 2))),
    "got {result:?}"
  );
}

#[test]
fn vbx_rejects_shape_mismatch_x_vs_qinit() {
  let x = DMatrix::<f64>::zeros(5, 4);
  let phi = DVector::<f64>::from_element(4, 1.0);
  let qinit = DMatrix::<f64>::from_element(6, 2, 0.5); // T=6 ≠ 5
  let result = vbx_iterate(x.as_view(), &phi, &qinit, 0.07, 0.8, 20);
  assert!(matches!(result, Err(Error::Shape(_))), "got {result:?}");
}

#[test]
fn vbx_rejects_shape_mismatch_phi_vs_x() {
  let x = DMatrix::<f64>::zeros(5, 4); // D=4
  let phi = DVector::<f64>::from_element(3, 1.0); // D=3 ≠ 4
  let qinit = DMatrix::<f64>::from_element(5, 2, 0.5);
  let result = vbx_iterate(x.as_view(), &phi, &qinit, 0.07, 0.8, 20);
  assert!(matches!(result, Err(Error::Shape(_))), "got {result:?}");
}

#[test]
fn vbx_rejects_qinit_with_zero_clusters() {
  let x = DMatrix::<f64>::zeros(5, 4);
  let phi = DVector::<f64>::from_element(4, 1.0);
  let qinit = DMatrix::<f64>::zeros(5, 0);
  let result = vbx_iterate(x.as_view(), &phi, &qinit, 0.07, 0.8, 20);
  assert!(matches!(result, Err(Error::Shape(_))), "got {result:?}");
}

/// VBx must produce a monotonically non-decreasing ELBO (modulo a tiny
/// epsilon-band at convergence). A regression that, e.g., reuses the
/// previous iteration's gamma in the alpha update would break this.
#[test]
fn vbx_elbo_is_monotonically_non_decreasing() {
  // 50 frames × 8 dim × 3 speakers, deterministic non-pathological input.
  let t = 50;
  let d = 8;
  let s = 3;
  let mut x = DMatrix::<f64>::zeros(t, d);
  for i in 0..t {
    for j in 0..d {
      x[(i, j)] = ((i * 7 + j * 13) as f64 % 11.0) - 5.0;
    }
  }
  let phi = DVector::<f64>::from_element(d, 2.0);
  let qinit = deterministic_qinit(t, s);
  let out = vbx_iterate(x.as_view(), &phi, &qinit, 0.07, 0.8, 20).expect("vbx_iterate");
  for w in out.elbo_trajectory().windows(2) {
    // Allow tiny float wobble at convergence (≤ 1e-6) before the
    // epsilon-based stop fires.
    assert!(
      w[1] - w[0] > -1.0e-6,
      "ELBO must not decrease: {} → {}",
      w[0],
      w[1]
    );
  }
}

/// At every iteration, `gamma[t, :]` is a discrete probability over
/// speakers, so each row must sum to 1 (within float roundoff).
#[test]
fn vbx_gamma_rows_sum_to_one() {
  let t = 30;
  let d = 4;
  let s = 4;
  let mut x = DMatrix::<f64>::zeros(t, d);
  for i in 0..t {
    for j in 0..d {
      x[(i, j)] = ((i + j) as f64).sin();
    }
  }
  let phi = DVector::<f64>::from_element(d, 1.5);
  let qinit = deterministic_qinit(t, s);
  let out = vbx_iterate(x.as_view(), &phi, &qinit, 0.1, 0.5, 10).expect("vbx_iterate");
  for r in 0..t {
    let row_sum: f64 = (0..s).map(|c| out.gamma()[(r, c)]).sum();
    assert!(
      (row_sum - 1.0).abs() < 1e-12,
      "gamma row {r} sums to {row_sum}"
    );
  }
}

/// `pi` is a discrete probability over speakers; it must sum to 1.
#[test]
fn vbx_pi_sums_to_one() {
  let t = 20;
  let d = 4;
  let s = 5;
  let x = DMatrix::<f64>::from_fn(t, d, |i, j| ((i * 3 + j) as f64).cos());
  let phi = DVector::<f64>::from_element(d, 1.0);
  let qinit = deterministic_qinit(t, s);
  let out = vbx_iterate(x.as_view(), &phi, &qinit, 0.07, 0.8, 20).expect("vbx_iterate");
  let pi_sum: f64 = out.pi().iter().sum();
  assert!((pi_sum - 1.0).abs() < 1e-12, "pi sums to {pi_sum}");
}

/// The algorithm has no RNG anywhere, so two calls with the same input
/// must return bit-identical outputs. Catches regressions where, e.g.,
/// Zero feature columns must error at the boundary rather than
/// running the EM loop with no PLDA evidence (which produces
/// finite-looking but meaningless gamma/pi).
#[test]
fn vbx_rejects_zero_feature_dim() {
  let t = 5;
  let s = 2;
  let x = DMatrix::<f64>::zeros(t, 0);
  let phi = DVector::<f64>::zeros(0);
  let qinit = deterministic_qinit(t, s);
  let r = vbx_iterate(x.as_view(), &phi, &qinit, 0.07, 0.8, 5);
  assert!(
    matches!(
      r,
      Err(Error::Shape(
        crate::cluster::vbx::error::ShapeError::ZeroXFeatureDim
      ))
    ),
    "expected Shape(ZeroXFeatureDim) for d=0 input, got {r:?}"
  );
}

/// `HashMap` ordering or `f64::partial_cmp` tiebreaks leak into the
/// algorithm.
#[test]
fn vbx_is_deterministic() {
  let t = 15;
  let d = 4;
  let s = 3;
  let x = DMatrix::<f64>::from_fn(t, d, |i, j| (i + 2 * j) as f64 * 0.1);
  let phi = DVector::<f64>::from_element(d, 2.0);
  let qinit = deterministic_qinit(t, s);
  let a = vbx_iterate(x.as_view(), &phi, &qinit, 0.07, 0.8, 10).expect("a");
  let b = vbx_iterate(x.as_view(), &phi, &qinit, 0.07, 0.8, 10).expect("b");
  assert_eq!(a.elbo_trajectory(), b.elbo_trajectory());
  for r in 0..t {
    for c in 0..s {
      assert_eq!(a.gamma()[(r, c)], b.gamma()[(r, c)]);
    }
  }
  for c in 0..s {
    assert_eq!(a.pi()[c], b.pi()[c]);
  }
}

// ── Input-value validation ─
//
// Boundary validation for `qinit` (finite, nonnegative, row-sum ≈ 1)
// and for `Fa`/`Fb` (positive, finite). Without these, a malformed
// initializer or hyperparameter silently biases the first speaker-
// model update and propagates garbage through the rest of the run;
// pyannote does not validate these, so this is a deliberate divergence
// to fail-fast at the boundary instead of producing fabricated speaker
// evidence.

#[test]
fn vbx_rejects_qinit_with_nan_entry() {
  let t = 5;
  let s = 2;
  let x = DMatrix::<f64>::zeros(t, 4);
  let phi = DVector::<f64>::from_element(4, 1.0);
  let mut qinit = DMatrix::<f64>::from_element(t, s, 0.5);
  qinit[(2, 1)] = f64::NAN;
  let result = vbx_iterate(x.as_view(), &phi, &qinit, 0.07, 0.8, 20);
  assert!(
    matches!(
      result,
      Err(Error::NonFinite(
        crate::cluster::vbx::error::NonFiniteField::Qinit
      ))
    ),
    "got {result:?}"
  );
}

#[test]
fn vbx_rejects_qinit_with_inf_entry() {
  let t = 5;
  let s = 2;
  let x = DMatrix::<f64>::zeros(t, 4);
  let phi = DVector::<f64>::from_element(4, 1.0);
  let mut qinit = DMatrix::<f64>::from_element(t, s, 0.5);
  qinit[(0, 0)] = f64::INFINITY;
  let result = vbx_iterate(x.as_view(), &phi, &qinit, 0.07, 0.8, 20);
  assert!(
    matches!(
      result,
      Err(Error::NonFinite(
        crate::cluster::vbx::error::NonFiniteField::Qinit
      ))
    ),
    "got {result:?}"
  );
}

#[test]
fn vbx_rejects_qinit_with_negative_entry() {
  let t = 5;
  let s = 2;
  let x = DMatrix::<f64>::zeros(t, 4);
  let phi = DVector::<f64>::from_element(4, 1.0);
  // Per-row sum still 1.0 (0.6 + 0.4) so we exercise the negative-
  // value path, not the row-sum path. Set one entry to -0.1 and
  // bump its sibling to 1.1 so the row sums to 1.0.
  let mut qinit = DMatrix::<f64>::from_element(t, s, 0.5);
  qinit[(0, 0)] = -0.1;
  qinit[(0, 1)] = 1.1;
  let result = vbx_iterate(x.as_view(), &phi, &qinit, 0.07, 0.8, 20);
  assert!(matches!(result, Err(Error::Shape(_))), "got {result:?}");
}

#[test]
fn vbx_rejects_qinit_with_unnormalized_row() {
  let t = 5;
  let s = 2;
  let x = DMatrix::<f64>::zeros(t, 4);
  let phi = DVector::<f64>::from_element(4, 1.0);
  // Row 3 has entries [0.5, 0.4] — sum = 0.9, fails the 1e-9 tolerance.
  let mut qinit = DMatrix::<f64>::from_element(t, s, 0.5);
  qinit[(3, 1)] = 0.4;
  let result = vbx_iterate(x.as_view(), &phi, &qinit, 0.07, 0.8, 20);
  assert!(matches!(result, Err(Error::Shape(_))), "got {result:?}");
}

#[test]
fn vbx_rejects_zero_fa() {
  let t = 5;
  let s = 2;
  let x = DMatrix::<f64>::zeros(t, 4);
  let phi = DVector::<f64>::from_element(4, 1.0);
  let qinit = DMatrix::<f64>::from_element(t, s, 0.5);
  let result = vbx_iterate(x.as_view(), &phi, &qinit, 0.0, 0.8, 20);
  assert!(matches!(result, Err(Error::Shape(_))), "got {result:?}");
}

#[test]
fn vbx_rejects_negative_fa() {
  let t = 5;
  let s = 2;
  let x = DMatrix::<f64>::zeros(t, 4);
  let phi = DVector::<f64>::from_element(4, 1.0);
  let qinit = DMatrix::<f64>::from_element(t, s, 0.5);
  let result = vbx_iterate(x.as_view(), &phi, &qinit, -0.1, 0.8, 20);
  assert!(matches!(result, Err(Error::Shape(_))), "got {result:?}");
}

#[test]
fn vbx_rejects_nan_fa() {
  let t = 5;
  let s = 2;
  let x = DMatrix::<f64>::zeros(t, 4);
  let phi = DVector::<f64>::from_element(4, 1.0);
  let qinit = DMatrix::<f64>::from_element(t, s, 0.5);
  let result = vbx_iterate(x.as_view(), &phi, &qinit, f64::NAN, 0.8, 20);
  assert!(matches!(result, Err(Error::Shape(_))), "got {result:?}");
}

#[test]
fn vbx_rejects_zero_fb() {
  let t = 5;
  let s = 2;
  let x = DMatrix::<f64>::zeros(t, 4);
  let phi = DVector::<f64>::from_element(4, 1.0);
  let qinit = DMatrix::<f64>::from_element(t, s, 0.5);
  let result = vbx_iterate(x.as_view(), &phi, &qinit, 0.07, 0.0, 20);
  assert!(matches!(result, Err(Error::Shape(_))), "got {result:?}");
}

#[test]
fn vbx_rejects_inf_fb() {
  let t = 5;
  let s = 2;
  let x = DMatrix::<f64>::zeros(t, 4);
  let phi = DVector::<f64>::from_element(4, 1.0);
  let qinit = DMatrix::<f64>::from_element(t, s, 0.5);
  let result = vbx_iterate(x.as_view(), &phi, &qinit, 0.07, f64::INFINITY, 20);
  assert!(matches!(result, Err(Error::Shape(_))), "got {result:?}");
}

/// `max_iters == 0` is rejected at the boundary. Skipping the EM
/// loop returns gamma=qinit and pi=1/S, which is internally
/// inconsistent for any non-uniform qinit (pi should equal
/// `gamma.column_sum() / T`) but indistinguishable from a completed
/// VBx run by the type system.
#[test]
fn vbx_rejects_max_iters_zero() {
  let t = 6;
  let s = 3;
  let x = DMatrix::<f64>::zeros(t, 4);
  let phi = DVector::<f64>::from_element(4, 1.0);
  let qinit = deterministic_qinit(t, s);
  let result = vbx_iterate(x.as_view(), &phi, &qinit, 0.07, 0.8, 0);
  assert!(matches!(result, Err(Error::Shape(_))), "got {result:?}");
}

/// `max_iters > MAX_ITERS_CAP` is rejected at the boundary so a
/// malformed config cannot turn one diarization call into hours of
/// unbounded matmul work. Pyannote's default is 20; the cap is 1_000.
#[test]
fn vbx_rejects_max_iters_above_cap() {
  use crate::cluster::vbx::MAX_ITERS_CAP;
  let t = 6;
  let s = 3;
  let x = DMatrix::<f64>::zeros(t, 4);
  let phi = DVector::<f64>::from_element(4, 1.0);
  let qinit = deterministic_qinit(t, s);
  let result = vbx_iterate(x.as_view(), &phi, &qinit, 0.07, 0.8, MAX_ITERS_CAP + 1);
  assert!(matches!(result, Err(Error::Shape(_))), "got {result:?}");
}

/// `max_iters == MAX_ITERS_CAP` is allowed (boundary inclusive).
#[test]
fn vbx_accepts_max_iters_at_cap() {
  use crate::cluster::vbx::MAX_ITERS_CAP;
  let t = 4;
  let s = 2;
  let x = DMatrix::<f64>::zeros(t, 4);
  let phi = DVector::<f64>::from_element(4, 1.0);
  let qinit = deterministic_qinit(t, s);
  // The actual loop will converge well before MAX_ITERS_CAP; we only
  // verify the boundary check accepts it.
  let result = vbx_iterate(x.as_view(), &phi, &qinit, 0.07, 0.8, MAX_ITERS_CAP);
  assert!(result.is_ok(), "got {result:?}");
}

/// Strongly non-uniform qinit (each row peaked on a different speaker)
/// with `max_iters = 0` would return `gamma = qinit` and `pi = 1/S` —
/// inconsistent (`pi` should equal `gamma.col_sum() / T`). Now blocked
/// at the boundary by the max_iters check.
#[test]
fn vbx_rejects_max_iters_zero_with_non_uniform_qinit() {
  let t = 10;
  let s = 2;
  let d = 4;
  let x = DMatrix::<f64>::from_fn(t, d, |i, j| ((i + j) as f64) * 0.3);
  let phi = DVector::<f64>::from_element(d, 1.0);
  let qinit = deterministic_qinit(t, s);
  let result = vbx_iterate(x.as_view(), &phi, &qinit, 0.07, 0.8, 0);
  assert!(
    matches!(result, Err(Error::Shape(_))),
    "non-uniform qinit + max_iters=0 must reject (would otherwise \
     return gamma=qinit + pi=1/S inconsistent state); got {result:?}"
  );
}

/// Realistic per-frame assignment: even rows favor speaker 0,
/// odd rows favor speaker 1. End-to-end smoke test that VBx accepts
/// a valid pyannote-style softmax(7) one-hot initializer.
#[test]
fn vbx_accepts_qinit_with_alternating_column_assignment() {
  let t = 10;
  let s = 2;
  let d = 4;
  let x = DMatrix::<f64>::from_fn(t, d, |i, j| ((i + j) as f64) * 0.3);
  let phi = DVector::<f64>::from_element(d, 1.0);
  let mut qinit = DMatrix::<f64>::zeros(t, s);
  for tt in 0..t {
    if tt % 2 == 0 {
      qinit[(tt, 0)] = 0.95;
      qinit[(tt, 1)] = 0.05;
    } else {
      qinit[(tt, 0)] = 0.05;
      qinit[(tt, 1)] = 0.95;
    }
  }
  let _out = vbx_iterate(x.as_view(), &phi, &qinit, 0.07, 0.8, 10)
    .expect("alternating real columns must pass");
}

/// S=1 is a degenerate case (single speaker) — `qinit` is forced to
/// be all 1.0 by the row-sum invariant, but VBx still runs.
#[test]
fn vbx_accepts_single_speaker_qinit() {
  let t = 5;
  let s = 1;
  let d = 4;
  let x = DMatrix::<f64>::from_fn(t, d, |i, j| ((i + j) as f64) * 0.1);
  let phi = DVector::<f64>::from_element(d, 1.0);
  let qinit = DMatrix::<f64>::from_element(t, s, 1.0);
  let out = vbx_iterate(x.as_view(), &phi, &qinit, 0.07, 0.8, 10)
    .expect("S=1 single-speaker qinit must pass");
  // With S=1 there is only one cluster; pi[0] should be 1.0.
  assert!((out.pi()[0] - 1.0).abs() < 1e-12, "pi[0] = {}", out.pi()[0]);
}

// ── X / Phi non-finite hardening ─
//
// The previous boundary accepted `+inf` Phi (the check used
// `is_nan()` only) and didn't validate X at all. Either case
// poisons G/rho silently — caught downstream as a generic
// `NonFinite("ELBO")` if max_iters > 0, or returned as Ok with the
// unvalidated qinit at max_iters = 0. Tightening to `is_finite()`
// + a leading X scan rejects upstream-corrupted PLDA inputs at the
// boundary with a clear typed error.

#[test]
fn vbx_rejects_phi_with_pos_inf() {
  let x = DMatrix::<f64>::zeros(5, 4);
  let mut phi = DVector::<f64>::from_element(4, 1.0);
  phi[1] = f64::INFINITY;
  let qinit = DMatrix::<f64>::from_element(5, 2, 0.5);
  let result = vbx_iterate(x.as_view(), &phi, &qinit, 0.07, 0.8, 20);
  assert!(
    matches!(result, Err(Error::NonPositivePhi(p, 1)) if p.is_infinite() && p > 0.0),
    "got {result:?}"
  );
}

#[test]
fn vbx_rejects_phi_with_nan() {
  let x = DMatrix::<f64>::zeros(5, 4);
  let mut phi = DVector::<f64>::from_element(4, 1.0);
  phi[3] = f64::NAN;
  let qinit = DMatrix::<f64>::from_element(5, 2, 0.5);
  let result = vbx_iterate(x.as_view(), &phi, &qinit, 0.07, 0.8, 20);
  assert!(
    matches!(result, Err(Error::NonPositivePhi(p, 3)) if p.is_nan()),
    "got {result:?}"
  );
}

#[test]
fn vbx_rejects_x_with_nan() {
  let mut x = DMatrix::<f64>::zeros(5, 4);
  x[(2, 1)] = f64::NAN;
  let phi = DVector::<f64>::from_element(4, 1.0);
  let qinit = DMatrix::<f64>::from_element(5, 2, 0.5);
  let result = vbx_iterate(x.as_view(), &phi, &qinit, 0.07, 0.8, 20);
  assert!(
    matches!(
      result,
      Err(Error::NonFinite(
        crate::cluster::vbx::error::NonFiniteField::X
      ))
    ),
    "got {result:?}"
  );
}

#[test]
fn vbx_rejects_x_with_pos_inf() {
  let mut x = DMatrix::<f64>::zeros(5, 4);
  x[(0, 0)] = f64::INFINITY;
  let phi = DVector::<f64>::from_element(4, 1.0);
  let qinit = DMatrix::<f64>::from_element(5, 2, 0.5);
  let result = vbx_iterate(x.as_view(), &phi, &qinit, 0.07, 0.8, 20);
  assert!(
    matches!(
      result,
      Err(Error::NonFinite(
        crate::cluster::vbx::error::NonFiniteField::X
      ))
    ),
    "got {result:?}"
  );
}

#[test]
fn vbx_rejects_x_with_neg_inf() {
  let mut x = DMatrix::<f64>::zeros(5, 4);
  x[(4, 3)] = f64::NEG_INFINITY;
  let phi = DVector::<f64>::from_element(4, 1.0);
  let qinit = DMatrix::<f64>::from_element(5, 2, 0.5);
  let result = vbx_iterate(x.as_view(), &phi, &qinit, 0.07, 0.8, 20);
  assert!(
    matches!(
      result,
      Err(Error::NonFinite(
        crate::cluster::vbx::error::NonFiniteField::X
      ))
    ),
    "got {result:?}"
  );
}

/// At `max_iters = 0` the loop never runs, so the generic NaN-
/// intermediate guard never fires. Boundary validation must catch
/// invalid inputs even when no iterations run.
#[test]
fn vbx_rejects_invalid_x_even_with_max_iters_zero() {
  let mut x = DMatrix::<f64>::zeros(5, 4);
  x[(2, 2)] = f64::NAN;
  let phi = DVector::<f64>::from_element(4, 1.0);
  let qinit = DMatrix::<f64>::from_element(5, 2, 0.5);
  let result = vbx_iterate(x.as_view(), &phi, &qinit, 0.07, 0.8, 0);
  assert!(
    matches!(
      result,
      Err(Error::NonFinite(
        crate::cluster::vbx::error::NonFiniteField::X
      ))
    ),
    "boundary validation must run even at max_iters=0; got {result:?}"
  );
}

#[test]
fn vbx_rejects_invalid_phi_even_with_max_iters_zero() {
  let x = DMatrix::<f64>::zeros(5, 4);
  let mut phi = DVector::<f64>::from_element(4, 1.0);
  phi[2] = f64::INFINITY;
  let qinit = DMatrix::<f64>::from_element(5, 2, 0.5);
  let result = vbx_iterate(x.as_view(), &phi, &qinit, 0.07, 0.8, 0);
  assert!(
    matches!(result, Err(Error::NonPositivePhi(p, 2)) if p.is_infinite()),
    "boundary validation must run even at max_iters=0; got {result:?}"
  );
}

// ── ELBO step classification () ─
//
// VB EM's monotonicity is a fundamental invariant. The previous
// `delta < epsilon` convergence branch fired for both small-positive
// improvements (intended) and negative deltas (a regression — bug
// or numerical instability). The new `classify_elbo_step` helper
// separates the three regimes, and `vbx_iterate` propagates a
// regression as `Error::ElboRegression` rather than silently
// returning the regressed posterior.

use super::algo::{ElboStep, classify_elbo_step};

// Most classifier tests use small-magnitude `prev`/`elbo` so the
// scale-aware regression band collapses to ~atol (~1e-9). Two tests
// near the bottom exercise the band at large magnitude ().

#[test]
fn classify_elbo_step_continues_on_large_positive_delta() {
  assert_eq!(
    classify_elbo_step(0.5, -1.5, -1.0, 1.0e-4),
    ElboStep::Continue
  );
}

#[test]
fn classify_elbo_step_converges_on_small_positive_delta() {
  assert_eq!(
    classify_elbo_step(1.0e-5, -1.00001, -1.0, 1.0e-4),
    ElboStep::Converged
  );
}

#[test]
fn classify_elbo_step_converges_on_tiny_negative_delta_within_tolerance() {
  // Delta in float-roundoff regime — well inside the band.
  assert_eq!(
    classify_elbo_step(-1.0e-12, -1.0, -1.0 - 1.0e-12, 1.0e-4),
    ElboStep::Converged
  );
}

#[test]
fn classify_elbo_step_regresses_on_large_negative_delta() {
  match classify_elbo_step(-1.0e-4, -1.0, -1.0001, 1.0e-4) {
    ElboStep::Regressed(d) => assert_eq!(d, -1.0e-4),
    other => panic!("expected Regressed, got {other:?}"),
  }
}

#[test]
fn classify_elbo_step_regresses_just_outside_tolerance() {
  // |elbo|=1.0 → tol = 1e-9 + 1e-9*1 = 2e-9. delta=-1e-8 is 5x outside.
  match classify_elbo_step(-1.0e-8, -1.0, -1.00000001, 1.0e-4) {
    ElboStep::Regressed(d) => assert_eq!(d, -1.0e-8),
    other => panic!("expected Regressed, got {other:?}"),
  }
}

#[test]
fn classify_elbo_step_zero_delta_is_converged() {
  // Exactly zero — flat ELBO, treat as converged.
  assert_eq!(
    classify_elbo_step(0.0, -1.0, -1.0, 1.0e-4),
    ElboStep::Converged
  );
}

// ── Scale-aware regression band () ─
//
// ELBO is an accumulated sum over T * S * D matrix entries plus T
// per-frame terms; float roundoff scales with the magnitude of the
// ELBO itself. The previous absolute `-1e-9` regression tolerance
// (calibrated against the |ELBO|≈2700 captured fixture) errored out
// on numerically awkward but otherwise valid inputs. The
// `atol + rtol * max(|prev|, |elbo|)` band absorbs that.

/// Regression case: final delta of `-2.47e-8` at |ELBO| ≈ 2700 —
/// outside an absolute `1e-9` band but well inside the scale-aware
/// band (1e-9 + 1e-9 * 2700 ≈ 2.7e-6).
#[test]
fn classify_elbo_step_absorbs_relative_float_roundoff_at_large_magnitude() {
  let prev = -2700.0_f64;
  let delta = -2.47e-8_f64;
  let elbo = prev + delta;
  assert_eq!(
    classify_elbo_step(delta, prev, elbo, 1.0e-4),
    ElboStep::Converged,
    "scale-aware band must absorb a delta the absolute tolerance \
     would have rejected"
  );
}

/// Even at large magnitude, materially-large negative drops still
/// surface as `Regressed`. Tests the upper edge of the scale-aware band.
#[test]
fn classify_elbo_step_still_rejects_material_regression_at_large_magnitude() {
  let prev = -2700.0_f64;
  // Band at this magnitude is ~2.7e-6; a -1e-3 drop is ~370× outside.
  let delta = -1.0e-3_f64;
  let elbo = prev + delta;
  match classify_elbo_step(delta, prev, elbo, 1.0e-4) {
    ElboStep::Regressed(d) => assert_eq!(d, delta),
    other => panic!("expected Regressed at large magnitude, got {other:?}"),
  }
}

// ── Stop reason: converged vs max-iters-reached () ─
//
// pointed out that `vbx_iterate` returned the same
// shape of `Ok` for two semantically distinct cases:
//   - Converged within max_iters (early break on ElboStep::Converged)
//   - max_iters reached without ever converging (loop falls through)
// Both could have `elbo_trajectory.len() == max_iters` (when
// convergence happens on the very last allowed iteration). Callers
// could not reliably distinguish the two, so an unconverged
// posterior would silently flow into downstream centroid/label
// assignment. `VbxOutput::stop_reason` makes the distinction
// observable at the type level.

use crate::cluster::vbx::StopReason;

/// `max_iters = 1`: the convergence check requires `ii > 0`, so a
/// 1-iter run can never fire the `Converged` branch. The loop ends
/// naturally and `stop_reason == MaxIterationsReached`.
#[test]
fn vbx_reports_max_iterations_reached_when_cap_is_one() {
  let t = 6;
  let s = 2;
  let d = 4;
  let mut x = DMatrix::<f64>::zeros(t, d);
  for i in 0..t {
    for j in 0..d {
      x[(i, j)] = ((i + j) as f64) * 0.5;
    }
  }
  let phi = DVector::<f64>::from_element(d, 1.0);
  let qinit = deterministic_qinit(t, s);
  let out = vbx_iterate(x.as_view(), &phi, &qinit, 0.07, 0.8, 1).expect("vbx_iterate");
  assert_eq!(
    out.stop_reason(),
    StopReason::MaxIterationsReached,
    "max_iters=1 cannot fire convergence (check requires ii > 0)"
  );
  assert_eq!(out.elbo_trajectory().len(), 1, "ran exactly 1 iteration");
}

/// On a small input that converges quickly, the same call with a
/// generous `max_iters` should report `Converged`. Together with
/// the previous test this proves callers can distinguish the two
/// stop reasons.
#[test]
fn vbx_reports_converged_on_easy_input() {
  let t = 6;
  let s = 2;
  let d = 4;
  let mut x = DMatrix::<f64>::zeros(t, d);
  for i in 0..t {
    for j in 0..d {
      x[(i, j)] = ((i + j) as f64) * 0.5;
    }
  }
  let phi = DVector::<f64>::from_element(d, 1.0);
  let qinit = deterministic_qinit(t, s);
  let out = vbx_iterate(x.as_view(), &phi, &qinit, 0.07, 0.8, 50).expect("vbx_iterate");
  assert_eq!(
    out.stop_reason(),
    StopReason::Converged,
    "easy input with generous cap must converge before exhaustion; \
     ran {} iterations",
    out.elbo_trajectory().len()
  );
  // Convergence on a trivial input is fast (well below the cap).
  assert!(
    out.elbo_trajectory().len() < 50,
    "expected early convergence, ran {} iters",
    out.elbo_trajectory().len()
  );
}
