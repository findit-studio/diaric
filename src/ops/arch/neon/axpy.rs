//! NEON f64 AXPY: `y[i] += alpha * x[i]`.
//!
//! 2-lane FMA, two-accumulator unroll for ILP. Falls back to scalar
//! tail for the trailing 0–3 odd elements.

use core::arch::aarch64::{vdupq_n_f64, vfmaq_f64, vld1q_f64, vst1q_f64};

use crate::ops::scalar;

/// `y[i] += alpha * x[i]`.
///
/// # Safety
///
/// 1. NEON must be available (caller's obligation).
/// 2. `y.len() == x.len()` (debug-asserted).
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn axpy(y: &mut [f64], alpha: f64, x: &[f64]) {
  debug_assert_eq!(y.len(), x.len(), "neon::axpy: length mismatch");
  let n = y.len();

  // SAFETY: pointer adds bounded by loop conditions; caller-promised
  // length parity.
  unsafe {
    let av = vdupq_n_f64(alpha);
    let mut i = 0usize;
    while i + 4 <= n {
      let y0 = vld1q_f64(y.as_ptr().add(i));
      let x0 = vld1q_f64(x.as_ptr().add(i));
      let y1 = vld1q_f64(y.as_ptr().add(i + 2));
      let x1 = vld1q_f64(x.as_ptr().add(i + 2));
      let r0 = vfmaq_f64(y0, av, x0);
      let r1 = vfmaq_f64(y1, av, x1);
      vst1q_f64(y.as_mut_ptr().add(i), r0);
      vst1q_f64(y.as_mut_ptr().add(i + 2), r1);
      i += 4;
    }
    if i + 2 <= n {
      let y0 = vld1q_f64(y.as_ptr().add(i));
      let x0 = vld1q_f64(x.as_ptr().add(i));
      let r0 = vfmaq_f64(y0, av, x0);
      vst1q_f64(y.as_mut_ptr().add(i), r0);
      i += 2;
    }
    if i < n {
      scalar::axpy(&mut y[i..], alpha, &x[i..]);
    }
  }
}
