//! NEON f64 dot product.
//!
//! 2-lane FMA over `float64x2_t`. Two parallel accumulators hide FMA
//! latency on cores where dependent FMAs serialize — the common case
//! on Apple silicon and Cortex-A series. The PLDA / embedding dims
//! shipped today (D = 192 / 256) are both multiples of 4, so the
//! scalar tail only runs for odd-dim test inputs.

use core::arch::aarch64::{float64x2_t, vaddq_f64, vaddvq_f64, vdupq_n_f64, vfmaq_f64, vld1q_f64};

/// `Σ a[i] * b[i]`. NEON 2-lane f64.
///
/// # Safety
///
/// 1. NEON must be available on the executing CPU (caller's
///    obligation; see [`crate::ops::neon_available`]).
/// 2. `a.len() == b.len()` (debug-asserted).
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn dot(a: &[f64], b: &[f64]) -> f64 {
  debug_assert_eq!(a.len(), b.len(), "neon::dot: length mismatch");
  let n = a.len();

  // SAFETY: pointer adds are bounded by the loop conditions and the
  // caller-promised `a.len() == b.len()`.
  unsafe {
    let mut acc0: float64x2_t = vdupq_n_f64(0.0);
    let mut acc1: float64x2_t = vdupq_n_f64(0.0);
    let mut i = 0usize;
    // 4-wide unroll (2 NEON regs × 2 lanes).
    while i + 4 <= n {
      let a0 = vld1q_f64(a.as_ptr().add(i));
      let b0 = vld1q_f64(b.as_ptr().add(i));
      let a1 = vld1q_f64(a.as_ptr().add(i + 2));
      let b1 = vld1q_f64(b.as_ptr().add(i + 2));
      acc0 = vfmaq_f64(acc0, a0, b0);
      acc1 = vfmaq_f64(acc1, a1, b1);
      i += 4;
    }
    // 2-wide tail.
    if i + 2 <= n {
      let a0 = vld1q_f64(a.as_ptr().add(i));
      let b0 = vld1q_f64(b.as_ptr().add(i));
      acc0 = vfmaq_f64(acc0, a0, b0);
      i += 2;
    }
    let acc = vaddq_f64(acc0, acc1);
    let mut sum = vaddvq_f64(acc);
    // Scalar tail must FMA each element directly into `sum` —
    // matches `ops::scalar::dot`'s `sum = f64::mul_add(a[i], b[i],
    // sum)` final loop. Routing through a recursive `scalar::dot`
    // call would compute its own per-tail sum (one rounding) and
    // then `sum += that` (a second rounding), drifting by ½ ulp on
    // odd `n` and breaking the bit-identical contract that AHC /
    // VBx / centroid / Hungarian rely on.    // HIGH (round 4).
    while i < n {
      sum = f64::mul_add(*a.get_unchecked(i), *b.get_unchecked(i), sum);
      i += 1;
    }
    sum
  }
}
