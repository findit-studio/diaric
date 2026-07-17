//! Dot product dispatcher.

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
use crate::ops::arch;
#[cfg(target_arch = "aarch64")]
use crate::ops::neon_available;
use crate::ops::scalar;
#[cfg(target_arch = "x86_64")]
use crate::ops::{avx2_available, avx512_available};

/// Inner product of two equal-length f64 slices.
///
/// Routes to the best available SIMD backend on this `target_arch`
/// after runtime CPU-feature detection. Callers needing byte-identical
/// scalar output across CPU families (e.g. for threshold-sensitive
/// discrete decisions) call [`crate::ops::scalar::dot`] directly.
///
/// # Panics
///
/// If `a.len() != b.len()`. This is enforced *unconditionally* — the
/// arch SIMD kernels read raw pointers bounded only by `a.len()` and
/// would otherwise load past `b` end in release builds, where their
/// `debug_assert!` is a no-op.
#[inline]
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
  assert_eq!(
    a.len(),
    b.len(),
    "ops::dot: a.len() ({}) must equal b.len() ({})",
    a.len(),
    b.len()
  );
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        // SAFETY: `neon_available()` confirmed NEON is on this CPU.
        // `a.len() == b.len()` is the documented dispatcher
        // precondition (debug-asserted in the kernel).
        return unsafe { arch::neon::dot(a, b) };
      }
    },
    target_arch = "x86_64" => {
      if avx512_available() {
        // SAFETY: `avx512_available()` confirmed AVX-512F.
        return unsafe { arch::x86_avx512::dot(a, b) };
      }
      if avx2_available() {
        // SAFETY: `avx2_available()` confirmed AVX2 + FMA.
        return unsafe { arch::x86_avx2::dot(a, b) };
      }
    },
    _ => {}
  }
  scalar::dot(a, b)
}
