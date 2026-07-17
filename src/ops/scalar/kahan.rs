//! Compensated-sum f64 dot product (Neumaier variant).
//!
//! Plain f64 summation accumulates roundoff bounded by `O(n * ε)` per
//! reduction. For the `(S, T) × (T, D)` and `(T, D) × (D, S)` GEMMs in
//! `cluster::vbx::vbx_iterate`, T grows with audio length (≈1000 chunks
//! for a 17-min recording), so plain GEMM ULP drift across reduction
//! orderings (matrixmultiply's cache-blocked microkernel vs numpy/BLAS)
//! is enough to flip a discrete `pi[s] > SP_ALIVE_THRESHOLD = 1e-7`
//! decision after 20 EM iterations — the exact failure mode that the
//! audit tagged as "GEMM roundoff drift on long recordings"
//! (pipeline I-P1) and that surfaces as the
//! `06_long_recording` strict parity test failure.
//!
//! Neumaier compensation drops the error bound to `O(ε)` regardless of
//! summation order, which makes the reduction effectively
//! order-independent across BLAS backends. The EM-iteration-after-iteration
//! drift accumulation goes away. This is significantly more accurate
//! than plain Kahan on adversarial inputs (cancellation when an incoming
//! summand exceeds the running sum).
//!
//! ## Cost
//!
//! Each compensated summand is two `f64` additions + one branch + the
//! original product. ≈ 4× the FMA-tree dot. At VBx scale (T ≈ 1000,
//! S ≈ 10, D = 128) that's a few million extra f64 adds per EM iter
//! — negligible against the ResNet inference and PLDA transform that
//! precede VBx.
//!
//! ## Why Neumaier vs plain Kahan
//!
//! Plain Kahan loses the compensation when `|x| > |sum|` because the
//! `t - sum` step computes the lower-magnitude operand of the addition,
//! which is `sum`, not `x`. Neumaier branches on `|sum| ≥ |x|` and
//! recovers the high bits of whichever summand was canceled. For the
//! VBx products `gamma[t,s] * rho[t,d]` the magnitudes vary across the
//! sum (gamma is in [0,1] and decays rapidly toward singletons; rho has
//! mixed sign), so the cancellation case fires often enough that the
//! Kahan/Neumaier distinction matters.

/// Compensated dot product: `Σ a[i] * b[i]` with Neumaier summation.
///
/// Result is independent of summation order to `O(ε)`, modulo the
/// f64 mul rounding of each `a[i] * b[i]` term.
///
/// # Panics
///
/// Asserts `a.len() == b.len()` unconditionally (release + debug).
/// The loop indexes `b[i]` for `i in 0..a.len()`, so a length
/// mismatch would panic on bounds-check in release anyway —
/// surfacing the contract violation early with a descriptive
/// message keeps it consistent with [`crate::ops::dispatch::dot`].
#[inline]
pub fn kahan_dot(a: &[f64], b: &[f64]) -> f64 {
  assert_eq!(
    a.len(),
    b.len(),
    "kahan_dot: a.len() ({}) must equal b.len() ({})",
    a.len(),
    b.len()
  );
  let n = a.len();
  let mut sum = 0.0_f64;
  let mut comp = 0.0_f64; // running compensation
  for i in 0..n {
    let x = a[i] * b[i];
    let t = sum + x;
    if sum.abs() >= x.abs() {
      // High bits of `sum` survive in `t`; the lost low bits of `x`
      // are recovered as `(sum - t) + x`.
      comp += (sum - t) + x;
    } else {
      // High bits of `x` survive; lost low bits of `sum` are
      // `(x - t) + sum`. The asymmetric branch is what makes this
      // Neumaier rather than plain Kahan.
      comp += (x - t) + sum;
    }
    sum = t;
  }
  sum + comp
}

/// Compensated sum: `Σ xs[i]` with Neumaier summation. Companion to
/// [`kahan_dot`] for plain reductions (column sums, slice totals).
#[inline]
pub fn kahan_sum(xs: &[f64]) -> f64 {
  let mut sum = 0.0_f64;
  let mut comp = 0.0_f64;
  for &x in xs {
    let t = sum + x;
    if sum.abs() >= x.abs() {
      comp += (sum - t) + x;
    } else {
      comp += (x - t) + sum;
    }
    sum = t;
  }
  sum + comp
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn matches_naive_for_well_conditioned_input() {
    let a: Vec<f64> = (0..100).map(|i| (i as f64) * 0.01).collect();
    let b: Vec<f64> = (0..100).map(|i| (i as f64).sin()).collect();
    let kahan = kahan_dot(&a, &b);
    let naive: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    // For well-conditioned inputs, the difference is sub-ULP.
    assert!(
      (kahan - naive).abs() < 1e-12,
      "kahan={kahan}, naive={naive}, diff={}",
      (kahan - naive).abs()
    );
  }

  #[test]
  fn handles_catastrophic_cancellation() {
    // Adversarial input: large + small + -large + small. Naive
    // summation drops the small terms entirely; Neumaier recovers them.
    let a = vec![1e16_f64, 1.0, -1e16_f64, 1.0];
    let b = vec![1.0_f64; 4];
    let kahan = kahan_dot(&a, &b);
    let naive: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    // True value is 2.0. Naive often returns 0.0; Kahan returns 2.0.
    assert_eq!(kahan, 2.0, "kahan should recover the small terms");
    let _ = naive; // not asserted — its result depends on FP optimization
  }

  #[test]
  fn order_invariant() {
    let a: Vec<f64> = (0..200).map(|i| ((i as f64) * 0.31).sin()).collect();
    let b: Vec<f64> = (0..200).map(|i| ((i as f64) * 0.71).cos()).collect();
    let forward = kahan_dot(&a, &b);
    // Reverse the input order — the f64 product values feed the
    // accumulator in reverse, so any reduction-order divergence would
    // surface here.
    let mut a_rev = a.clone();
    a_rev.reverse();
    let mut b_rev = b.clone();
    b_rev.reverse();
    let backward = kahan_dot(&a_rev, &b_rev);
    // For Neumaier summation, forward == backward up to a single ULP
    // (the order of f64 mul still matters, but the Σ part is
    // order-independent).
    let diff = (forward - backward).abs();
    assert!(
      diff < 1e-13,
      "order-dependent: forward={forward} backward={backward} diff={diff}"
    );
  }

  #[test]
  fn empty_input_returns_zero() {
    let a: Vec<f64> = vec![];
    let b: Vec<f64> = vec![];
    assert_eq!(kahan_dot(&a, &b), 0.0);
  }

  #[test]
  fn single_element() {
    assert_eq!(kahan_dot(&[3.0], &[4.0]), 12.0);
  }
}
