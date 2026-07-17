//! Kahan/Neumaier-compensated dot + sum dispatcher.
//!
//! Routes to the best-available SIMD backend at runtime, with a fall-
//! back to [`crate::ops::scalar`]. Used by `cluster::vbx::vbx_iterate`
//! for the EM-iteration GEMMs that need order-independent
//! reductions on long recordings.

#[cfg(target_arch = "aarch64")]
use crate::ops::arch;
#[cfg(target_arch = "aarch64")]
use crate::ops::neon_available;
use crate::ops::scalar;

/// Compensated dot product `Σ a[i] * b[i]`.
///
/// Routes to NEON when available on aarch64, else scalar. AVX2/AVX-512
/// SIMD backends are not yet wired (would mirror the existing dot/axpy
/// pattern); x86 callers fall through to the scalar reference.
///
/// # Panics
///
/// If `a.len() != b.len()`. Mirrors [`crate::ops::dot`]'s contract —
/// the unsafe SIMD kernel reads raw pointers bounded by `a.len()` and
/// would otherwise OOB-read `b` in release builds.
#[inline]
pub fn kahan_dot(a: &[f64], b: &[f64]) -> f64 {
  assert_eq!(
    a.len(),
    b.len(),
    "ops::kahan_dot: a.len() ({}) must equal b.len() ({})",
    a.len(),
    b.len()
  );
  #[cfg(target_arch = "aarch64")]
  {
    if neon_available() {
      // SAFETY: `neon_available()` confirmed NEON is on this CPU.
      // `a.len() == b.len()` is enforced unconditionally above.
      return unsafe { arch::neon::kahan_dot(a, b) };
    }
  }
  scalar::kahan_dot(a, b)
}

/// Compensated sum `Σ xs[i]`.
#[inline]
pub fn kahan_sum(xs: &[f64]) -> f64 {
  #[cfg(target_arch = "aarch64")]
  {
    if neon_available() {
      // SAFETY: NEON availability checked.
      return unsafe { arch::neon::kahan_sum(xs) };
    }
  }
  scalar::kahan_sum(xs)
}
