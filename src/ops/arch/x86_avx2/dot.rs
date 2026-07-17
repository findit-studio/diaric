//! AVX2 + FMA f64 dot product.
//!
//! 4-lane FMA over `__m256d`, two parallel accumulators (8-wide
//! unroll). PLDA / embedding D = 192 / 256 are both multiples of 8.

use core::arch::x86_64::{
  __m256d, _mm_add_pd, _mm_cvtsd_f64, _mm_unpackhi_pd, _mm256_add_pd, _mm256_castpd256_pd128,
  _mm256_extractf128_pd, _mm256_fmadd_pd, _mm256_loadu_pd, _mm256_setzero_pd,
};

/// `Σ a[i] * b[i]`. AVX2 4-lane f64 + FMA.
///
/// # Safety
///
/// 1. Caller must verify AVX2 + FMA via [`crate::ops::avx2_available`].
/// 2. `a.len() == b.len()` (debug-asserted).
#[inline]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn dot(a: &[f64], b: &[f64]) -> f64 {
  debug_assert_eq!(a.len(), b.len(), "x86_avx2::dot: length mismatch");
  let n = a.len();

  // SAFETY: pointer adds bounded by loop conditions; caller-promised
  // length parity. AVX2 + FMA verified at the dispatcher.
  unsafe {
    let mut acc0: __m256d = _mm256_setzero_pd();
    let mut acc1: __m256d = _mm256_setzero_pd();
    let mut i = 0usize;
    while i + 8 <= n {
      let a0 = _mm256_loadu_pd(a.as_ptr().add(i));
      let b0 = _mm256_loadu_pd(b.as_ptr().add(i));
      let a1 = _mm256_loadu_pd(a.as_ptr().add(i + 4));
      let b1 = _mm256_loadu_pd(b.as_ptr().add(i + 4));
      acc0 = _mm256_fmadd_pd(a0, b0, acc0);
      acc1 = _mm256_fmadd_pd(a1, b1, acc1);
      i += 8;
    }
    if i + 4 <= n {
      let a0 = _mm256_loadu_pd(a.as_ptr().add(i));
      let b0 = _mm256_loadu_pd(b.as_ptr().add(i));
      acc0 = _mm256_fmadd_pd(a0, b0, acc0);
      i += 4;
    }
    let acc = _mm256_add_pd(acc0, acc1);
    // Horizontal sum of 4 f64 lanes.
    let lo = _mm256_castpd256_pd128(acc);
    let hi = _mm256_extractf128_pd::<1>(acc);
    let sum2 = _mm_add_pd(lo, hi);
    // sum2 = [s0, s1]; horizontal add via unpackhi.
    let sum = _mm_cvtsd_f64(_mm_add_pd(sum2, _mm_unpackhi_pd(sum2, sum2)));
    let mut total = sum;
    // Scalar tail must FMA each element directly into `total` —
    // routing through `scalar::dot(&a[i..], &b[i..])` rounds twice
    // (per-tail sum, then add into `total`), drifting by ½ ulp on
    // odd `n`.
    while i < n {
      total = f64::mul_add(*a.get_unchecked(i), *b.get_unchecked(i), total);
      i += 1;
    }
    total
  }
}
