//! AVX-512F f64 AXPY: `y[i] += alpha * x[i]`.

use core::arch::x86_64::{_mm512_fmadd_pd, _mm512_loadu_pd, _mm512_set1_pd, _mm512_storeu_pd};

use crate::ops::scalar;

/// `y[i] += alpha * x[i]`.
///
/// # Safety
///
/// 1. Caller must verify AVX-512F.
/// 2. `y.len() == x.len()` (debug-asserted).
#[inline]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn axpy(y: &mut [f64], alpha: f64, x: &[f64]) {
  debug_assert_eq!(y.len(), x.len(), "x86_avx512::axpy: length mismatch");
  let n = y.len();

  // SAFETY: pointer adds bounded; AVX-512F verified at dispatcher.
  unsafe {
    let av = _mm512_set1_pd(alpha);
    let mut i = 0usize;
    while i + 16 <= n {
      let y0 = _mm512_loadu_pd(y.as_ptr().add(i));
      let x0 = _mm512_loadu_pd(x.as_ptr().add(i));
      let y1 = _mm512_loadu_pd(y.as_ptr().add(i + 8));
      let x1 = _mm512_loadu_pd(x.as_ptr().add(i + 8));
      let r0 = _mm512_fmadd_pd(av, x0, y0);
      let r1 = _mm512_fmadd_pd(av, x1, y1);
      _mm512_storeu_pd(y.as_mut_ptr().add(i), r0);
      _mm512_storeu_pd(y.as_mut_ptr().add(i + 8), r1);
      i += 16;
    }
    if i + 8 <= n {
      let y0 = _mm512_loadu_pd(y.as_ptr().add(i));
      let x0 = _mm512_loadu_pd(x.as_ptr().add(i));
      let r0 = _mm512_fmadd_pd(av, x0, y0);
      _mm512_storeu_pd(y.as_mut_ptr().add(i), r0);
      i += 8;
    }
    if i < n {
      scalar::axpy(&mut y[i..], alpha, &x[i..]);
    }
  }
}
