//! AXPY dispatcher.

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
use crate::ops::arch;
#[cfg(target_arch = "aarch64")]
use crate::ops::neon_available;
use crate::ops::scalar;
#[cfg(target_arch = "x86_64")]
use crate::ops::{avx2_available, avx512_available};

/// `y[i] += alpha * x[i]`.
///
/// Routes to the best available SIMD backend per arch + runtime
/// detection. Callers needing scalar output explicitly call
/// [`crate::ops::scalar::axpy`].
///
/// # Panics
///
/// If `y.len() != x.len()`. Enforced unconditionally so a release-mode
/// safe-Rust caller cannot bypass the precondition into the unsafe
/// SIMD kernel (which only `debug_assert!`s and would OOB-read `x`
/// otherwise).
#[inline]
pub fn axpy(y: &mut [f64], alpha: f64, x: &[f64]) {
  assert_eq!(
    y.len(),
    x.len(),
    "ops::axpy: y.len() ({}) must equal x.len() ({})",
    y.len(),
    x.len()
  );
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        // SAFETY: `neon_available()` confirmed NEON is on this CPU.
        unsafe { arch::neon::axpy(y, alpha, x); }
        return;
      }
    },
    target_arch = "x86_64" => {
      if avx512_available() {
        // SAFETY: `avx512_available()` confirmed AVX-512F.
        unsafe { arch::x86_avx512::axpy(y, alpha, x); }
        return;
      }
      if avx2_available() {
        // SAFETY: `avx2_available()` confirmed AVX2 + FMA.
        unsafe { arch::x86_avx2::axpy(y, alpha, x); }
        return;
      }
    },
    _ => {}
  }
  scalar::axpy(y, alpha, x);
}

/// f32 AXPY: `y[i] += alpha * x[i]`.
///
/// Used by [`crate::embed::embedder`] to accumulate per-window
/// WeSpeaker embeddings into a 256-d aggregator. No arch-specific
/// kernel yet — the scalar `f32::mul_add` loop autovectorizes to
/// `vfmaq_f32` (NEON) / `_mm256_fmadd_ps` (AVX2 + FMA) with
/// `--release`. Plug in explicit SIMD kernels later without touching
/// call sites.
///
/// # Panics
///
/// If `y.len() != x.len()`.
#[inline]
// `axpy_f32`'s only callers (in `crate::embed::embedder`) are gated
// behind `any(feature = "ort", feature = "tch")`. Under
// `--no-default-features` the function is unused but must stay
// reachable so SDE / miri jobs that build without either backend can
// still verify the SIMD-policy doesn't regress. `RUSTFLAGS=-Dwarnings`
// would otherwise turn the dead-code warning into a hard error and
// skip backend coverage entirely.
#[allow(dead_code)]
pub fn axpy_f32(y: &mut [f32], alpha: f32, x: &[f32]) {
  assert_eq!(
    y.len(),
    x.len(),
    "ops::axpy_f32: y.len() ({}) must equal x.len() ({})",
    y.len(),
    x.len()
  );
  scalar::axpy_f32(y, alpha, x);
}
