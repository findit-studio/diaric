//! Scalar f64 dot product.
//!
//! Implementation matches the NEON kernel's reduction tree exactly:
//! - Per-element FMA via `f64::mul_add` (one IEEE 754 rounding, same
//!   as `vfmaq_f64`).
//! - Four partial accumulators over the modulo-4 residue classes,
//!   mirroring NEON's two 2-lane registers (`acc0[0]`, `acc0[1]`,
//!   `acc1[0]`, `acc1[1]`).
//! - Final reduction tree `((s00 + s10) + (s01 + s11))`, identical
//!   to NEON's `vaddq_f64 + vaddvq_f64` sequence.
//!
//! Result is bit-identical to [`crate::ops::arch::neon::dot`] for
//! every input. The AVX2/AVX-512 backends use their native lane
//! widths (4 / 8) and *do* diverge from this reduction tree —
//! cross-architecture bit-identity is not claimed.

/// Inner product of two equal-length f64 slices: `Σ a[i] * b[i]`.
///
/// # Panics (debug only)
///
/// Debug asserts on `a.len() == b.len()`. Release builds trust the
/// caller — SIMD backends in `arch::*` rely on the same precondition.
#[inline]
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
  debug_assert_eq!(a.len(), b.len(), "dot: length mismatch");
  let n = a.len();
  let mut s00 = 0.0_f64; // accumulates positions ≡ 0 mod 4
  let mut s01 = 0.0_f64; // ≡ 1 mod 4
  let mut s10 = 0.0_f64; // ≡ 2 mod 4
  let mut s11 = 0.0_f64; // ≡ 3 mod 4
  let mut i = 0usize;
  while i + 4 <= n {
    s00 = f64::mul_add(a[i], b[i], s00);
    s01 = f64::mul_add(a[i + 1], b[i + 1], s01);
    s10 = f64::mul_add(a[i + 2], b[i + 2], s10);
    s11 = f64::mul_add(a[i + 3], b[i + 3], s11);
    i += 4;
  }
  // 2-wide tail: NEON also FMAs into acc0 only.
  if i + 2 <= n {
    s00 = f64::mul_add(a[i], b[i], s00);
    s01 = f64::mul_add(a[i + 1], b[i + 1], s01);
    i += 2;
  }
  // Reduction tree matches NEON's `vaddq_f64(acc0, acc1)` then
  // `vaddvq_f64(acc) = acc[0] + acc[1]`.
  let mut sum = (s00 + s10) + (s01 + s11);
  // Final scalar tail for odd lengths.
  while i < n {
    sum = f64::mul_add(a[i], b[i], sum);
    i += 1;
  }
  sum
}
