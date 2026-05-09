//! NEON f64 Neumaier-compensated dot product and sum.
//!
//! 2-lane `float64x2_t` parallel accumulators with per-lane
//! Neumaier compensation. The conditional that distinguishes Neumaier
//! from plain Kahan (`if |sum| >= |x|`) is implemented per-lane with
//! `vbslq_f64` (bitwise select) over the `vcgeq_f64` mask, so each
//! lane independently picks the right compensation branch.
//!
//! ## Numerical contract
//!
//! Per-lane summation is order-independent to `O(ε)` (Neumaier bound).
//! The 2 → 1 lane reduction adds one more Neumaier step, so the final
//! result is also `O(ε)` order-independent. This is **not** bit-
//! identical to [`crate::ops::scalar::kahan_dot`] — the scalar path
//! sees all `n` products in serial order, while NEON sees them split
//! across 2 lanes plus a final cross-lane combine. Both paths agree
//! to within a few ULPs, and both produce the same answer modulo the
//! Neumaier error bound regardless of summation order; that's the
//! whole point of using a compensated reduction in VBx (where the
//! BLAS-vs-matrixmultiply order divergence on long recordings was
//! flipping discrete `pi[s] > SP_ALIVE_THRESHOLD` decisions).

use core::arch::aarch64::{
  float64x2_t, uint64x2_t, vabsq_f64, vaddq_f64, vbslq_f64, vcgeq_f64, vdupq_n_f64, vgetq_lane_f64,
  vld1q_f64, vmulq_f64, vsubq_f64,
};

/// Compensated dot product `Σ a[i] * b[i]` (Neumaier), 2-lane NEON.
///
/// # Safety
///
/// 1. NEON must be available on the executing CPU (caller's
///    obligation; see [`crate::ops::neon_available`]).
/// 2. `a.len() == b.len()` (debug-asserted).
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn kahan_dot(a: &[f64], b: &[f64]) -> f64 {
  debug_assert_eq!(a.len(), b.len(), "neon::kahan_dot: length mismatch");
  let n = a.len();
  unsafe {
    let mut sum_v: float64x2_t = vdupq_n_f64(0.0);
    let mut comp_v: float64x2_t = vdupq_n_f64(0.0);
    let mut i = 0usize;
    while i + 2 <= n {
      let av = vld1q_f64(a.as_ptr().add(i));
      let bv = vld1q_f64(b.as_ptr().add(i));
      let xv = vmulq_f64(av, bv);
      let abs_sum = vabsq_f64(sum_v);
      let abs_x = vabsq_f64(xv);
      // Per-lane: cond[lane] = |sum[lane]| >= |x[lane]| (all-1s
      // mask if true, all-0s if false).
      let cond: uint64x2_t = vcgeq_f64(abs_sum, abs_x);
      let tv = vaddq_f64(sum_v, xv);
      // case A (|sum| >= |x|): comp += (sum - t) + x.
      let case_a = vaddq_f64(vsubq_f64(sum_v, tv), xv);
      // case B (|x| > |sum|): comp += (x - t) + sum.
      let case_b = vaddq_f64(vsubq_f64(xv, tv), sum_v);
      // vbslq_f64(mask, a, b): bits from a where mask is 1, b where 0.
      let delta = vbslq_f64(cond, case_a, case_b);
      comp_v = vaddq_f64(comp_v, delta);
      sum_v = tv;
      i += 2;
    }
    // Reduce 2 lanes → scalar with one more Neumaier step. Drop
    // lane 0's `comp` into scalar `comp`, fold lane 1's `sum` into
    // scalar `sum` via Neumaier, accumulate lane 1's `comp`.
    let mut sum = vgetq_lane_f64(sum_v, 0);
    let mut comp = vgetq_lane_f64(comp_v, 0);
    let s1 = vgetq_lane_f64(sum_v, 1);
    let c1 = vgetq_lane_f64(comp_v, 1);
    let t1 = sum + s1;
    if sum.abs() >= s1.abs() {
      comp += (sum - t1) + s1;
    } else {
      comp += (s1 - t1) + sum;
    }
    sum = t1;
    comp += c1;
    // Scalar tail (length-mod-2 leftover).
    while i < n {
      let x = *a.get_unchecked(i) * *b.get_unchecked(i);
      let t = sum + x;
      if sum.abs() >= x.abs() {
        comp += (sum - t) + x;
      } else {
        comp += (x - t) + sum;
      }
      sum = t;
      i += 1;
    }
    sum + comp
  }
}

/// Compensated sum `Σ xs[i]` (Neumaier), 2-lane NEON. Companion to
/// [`kahan_dot`] for plain reductions (column sums, slice totals).
///
/// # Safety
///
/// NEON must be available on the executing CPU.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn kahan_sum(xs: &[f64]) -> f64 {
  let n = xs.len();
  unsafe {
    let mut sum_v: float64x2_t = vdupq_n_f64(0.0);
    let mut comp_v: float64x2_t = vdupq_n_f64(0.0);
    let mut i = 0usize;
    while i + 2 <= n {
      let xv = vld1q_f64(xs.as_ptr().add(i));
      let abs_sum = vabsq_f64(sum_v);
      let abs_x = vabsq_f64(xv);
      let cond: uint64x2_t = vcgeq_f64(abs_sum, abs_x);
      let tv = vaddq_f64(sum_v, xv);
      let case_a = vaddq_f64(vsubq_f64(sum_v, tv), xv);
      let case_b = vaddq_f64(vsubq_f64(xv, tv), sum_v);
      let delta = vbslq_f64(cond, case_a, case_b);
      comp_v = vaddq_f64(comp_v, delta);
      sum_v = tv;
      i += 2;
    }
    let mut sum = vgetq_lane_f64(sum_v, 0);
    let mut comp = vgetq_lane_f64(comp_v, 0);
    let s1 = vgetq_lane_f64(sum_v, 1);
    let c1 = vgetq_lane_f64(comp_v, 1);
    let t1 = sum + s1;
    if sum.abs() >= s1.abs() {
      comp += (sum - t1) + s1;
    } else {
      comp += (s1 - t1) + sum;
    }
    sum = t1;
    comp += c1;
    while i < n {
      let x = *xs.get_unchecked(i);
      let t = sum + x;
      if sum.abs() >= x.abs() {
        comp += (sum - t) + x;
      } else {
        comp += (x - t) + sum;
      }
      sum = t;
      i += 1;
    }
    sum + comp
  }
}
