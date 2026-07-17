//! AVX-512F f64 dot product. 8-lane FMA, two parallel accumulators.

use core::arch::x86_64::{
  __m512d, _mm512_add_pd, _mm512_fmadd_pd, _mm512_loadu_pd, _mm512_reduce_add_pd, _mm512_setzero_pd,
};

/// `Σ a[i] * b[i]`. AVX-512F 8-lane f64 + FMA.
///
/// # Safety
///
/// 1. Caller must verify AVX-512F via [`crate::ops::avx512_available`].
/// 2. `a.len() == b.len()` (debug-asserted).
#[inline]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn dot(a: &[f64], b: &[f64]) -> f64 {
  debug_assert_eq!(a.len(), b.len(), "x86_avx512::dot: length mismatch");
  let n = a.len();

  // SAFETY: pointer adds bounded by loop conditions; AVX-512F verified
  // at dispatcher.
  unsafe {
    let mut acc0: __m512d = _mm512_setzero_pd();
    let mut acc1: __m512d = _mm512_setzero_pd();
    let mut i = 0usize;
    while i + 16 <= n {
      let a0 = _mm512_loadu_pd(a.as_ptr().add(i));
      let b0 = _mm512_loadu_pd(b.as_ptr().add(i));
      let a1 = _mm512_loadu_pd(a.as_ptr().add(i + 8));
      let b1 = _mm512_loadu_pd(b.as_ptr().add(i + 8));
      acc0 = _mm512_fmadd_pd(a0, b0, acc0);
      acc1 = _mm512_fmadd_pd(a1, b1, acc1);
      i += 16;
    }
    if i + 8 <= n {
      let a0 = _mm512_loadu_pd(a.as_ptr().add(i));
      let b0 = _mm512_loadu_pd(b.as_ptr().add(i));
      acc0 = _mm512_fmadd_pd(a0, b0, acc0);
      i += 8;
    }
    let acc = _mm512_add_pd(acc0, acc1);
    let mut sum = _mm512_reduce_add_pd(acc);
    // Scalar tail must FMA each element directly into `sum` —
    // routing through `scalar::dot` rounds twice.
    while i < n {
      sum = f64::mul_add(*a.get_unchecked(i), *b.get_unchecked(i), sum);
      i += 1;
    }
    sum
  }
}
