//! AVX-512F f64 pairwise Euclidean distance.

use core::arch::x86_64::{
  __m512d, _mm512_add_pd, _mm512_fmadd_pd, _mm512_loadu_pd, _mm512_reduce_add_pd,
  _mm512_setzero_pd, _mm512_sub_pd,
};

/// Pairwise Euclidean distance, condensed `pdist` ordering. See
/// [`crate::ops::scalar::pdist_euclidean`] for the contract.
///
/// # Safety
///
/// 1. Caller must verify AVX-512F.
/// 2. `rows.len() == n * d` (debug-asserted).
#[inline]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn pdist_euclidean(rows: &[f64], n: usize, d: usize) -> Vec<f64> {
  debug_assert_eq!(
    rows.len(),
    n * d,
    "x86_avx512::pdist_euclidean: shape mismatch"
  );
  let pair_count = if n >= 2 {
    n.checked_mul(n - 1)
      .expect("x86_avx512::pdist_euclidean: n * (n - 1) overflows usize")
      / 2
  } else {
    0
  };
  let mut out = Vec::with_capacity(pair_count);

  // SAFETY: row indices in `0..n`, pointer adds bounded by `i*d + d <=
  // rows.len()`. AVX-512F verified at the dispatcher.
  unsafe {
    for i in 0..n {
      let row_i_ptr = rows.as_ptr().add(i * d);
      for j in (i + 1)..n {
        let row_j_ptr = rows.as_ptr().add(j * d);
        let mut acc0: __m512d = _mm512_setzero_pd();
        let mut acc1: __m512d = _mm512_setzero_pd();
        let mut k = 0usize;
        while k + 16 <= d {
          let a0 = _mm512_loadu_pd(row_i_ptr.add(k));
          let b0 = _mm512_loadu_pd(row_j_ptr.add(k));
          let a1 = _mm512_loadu_pd(row_i_ptr.add(k + 8));
          let b1 = _mm512_loadu_pd(row_j_ptr.add(k + 8));
          let d0 = _mm512_sub_pd(a0, b0);
          let d1 = _mm512_sub_pd(a1, b1);
          acc0 = _mm512_fmadd_pd(d0, d0, acc0);
          acc1 = _mm512_fmadd_pd(d1, d1, acc1);
          k += 16;
        }
        if k + 8 <= d {
          let a0 = _mm512_loadu_pd(row_i_ptr.add(k));
          let b0 = _mm512_loadu_pd(row_j_ptr.add(k));
          let d0 = _mm512_sub_pd(a0, b0);
          acc0 = _mm512_fmadd_pd(d0, d0, acc0);
          k += 8;
        }
        let acc = _mm512_add_pd(acc0, acc1);
        let mut sq = _mm512_reduce_add_pd(acc);
        // Scalar tail must use `f64::mul_add` to match the scalar
        // reference.
        while k < d {
          let diff = *row_i_ptr.add(k) - *row_j_ptr.add(k);
          sq = f64::mul_add(diff, diff, sq);
          k += 1;
        }
        out.push(sq.sqrt());
      }
    }
  }

  out
}
