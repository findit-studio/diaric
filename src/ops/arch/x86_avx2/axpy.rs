//! AVX2 + FMA f64 AXPY: `y[i] += alpha * x[i]`.

use core::arch::x86_64::{_mm256_fmadd_pd, _mm256_loadu_pd, _mm256_set1_pd, _mm256_storeu_pd};

use crate::ops::scalar;

/// `y[i] += alpha * x[i]`.
///
/// # Safety
///
/// 1. Caller must verify AVX2 + FMA.
/// 2. `y.len() == x.len()` (debug-asserted).
#[inline]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn axpy(y: &mut [f64], alpha: f64, x: &[f64]) {
  debug_assert_eq!(y.len(), x.len(), "x86_avx2::axpy: length mismatch");
  let n = y.len();

  // SAFETY: pointer adds bounded; AVX2 + FMA verified at dispatcher.
  unsafe {
    let av = _mm256_set1_pd(alpha);
    let mut i = 0usize;
    while i + 8 <= n {
      let y0 = _mm256_loadu_pd(y.as_ptr().add(i));
      let x0 = _mm256_loadu_pd(x.as_ptr().add(i));
      let y1 = _mm256_loadu_pd(y.as_ptr().add(i + 4));
      let x1 = _mm256_loadu_pd(x.as_ptr().add(i + 4));
      let r0 = _mm256_fmadd_pd(av, x0, y0);
      let r1 = _mm256_fmadd_pd(av, x1, y1);
      _mm256_storeu_pd(y.as_mut_ptr().add(i), r0);
      _mm256_storeu_pd(y.as_mut_ptr().add(i + 4), r1);
      i += 8;
    }
    if i + 4 <= n {
      let y0 = _mm256_loadu_pd(y.as_ptr().add(i));
      let x0 = _mm256_loadu_pd(x.as_ptr().add(i));
      let r0 = _mm256_fmadd_pd(av, x0, y0);
      _mm256_storeu_pd(y.as_mut_ptr().add(i), r0);
      i += 4;
    }
    if i < n {
      scalar::axpy(&mut y[i..], alpha, &x[i..]);
    }
  }
}
