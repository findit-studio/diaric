//! Scalar AXPY: `y += alpha * x`.
//!
//! Uses `f64::mul_add` for per-element FMA — bit-identical to the
//! NEON / AVX2 / AVX-512 backends, which use `vfmaq_f64` /
//! `_mm256_fmadd_pd` / `_mm512_fmadd_pd`. AXPY has no inter-element
//! reduction, so cross-architecture bit-identity holds for AXPY
//! everywhere FMA is available (mandatory in ARMv8 baseline; gated
//! behind the AVX2 dispatcher's `fma` runtime check on x86_64).

/// In-place fused multiply-add over a slice: `y[i] = alpha * x[i] +
/// y[i]` for each `i`, with one IEEE 754 rounding per element.
///
/// Used by `centroid::weighted_centroids`'s
/// `centroids[k, d] += w * embeddings[t, d]` accumulator. The
/// k-by-d-by-t triple-nested loop reduces to repeated AXPY calls
/// (one per `(k, t)` pair, sized by `d = embed_dim`).
///
/// # Panics (debug only)
///
/// Debug asserts on `y.len() == x.len()`.
#[inline]
pub fn axpy(y: &mut [f64], alpha: f64, x: &[f64]) {
  debug_assert_eq!(y.len(), x.len(), "axpy: length mismatch");
  for i in 0..y.len() {
    y[i] = f64::mul_add(alpha, x[i], y[i]);
  }
}

/// f32 variant of [`axpy`]. Used by the embedding aggregation path
/// (`embed::embedder::embed_unweighted` / `embed_weighted_inner`) to
/// sum per-window WeSpeaker outputs into a 256-d accumulator.
///
/// Implemented in scalar form with `f32::mul_add`; the Rust compiler
/// emits NEON `vfmaq_f32` / AVX2 `_mm256_fmadd_ps` for this loop in
/// release mode (verified on 1.95 nightly with `cargo asm`). We keep
/// it as a named primitive so callers route through the SIMD-aware
/// [`crate::ops::axpy_f32`] dispatcher; arch-specific overrides can
/// be added later without touching call sites.
#[inline]
#[allow(dead_code)]
pub fn axpy_f32(y: &mut [f32], alpha: f32, x: &[f32]) {
  debug_assert_eq!(y.len(), x.len(), "axpy_f32: length mismatch");
  for i in 0..y.len() {
    y[i] = f32::mul_add(alpha, x[i], y[i]);
  }
}
