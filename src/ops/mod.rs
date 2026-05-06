//! Numerical primitives shared across the diarization algorithms.
//!
//! Four primitives cover the production hot paths:
//!
//! - [`dot`] — f64 dot product. Used by VBx (`gamma.column_sum`,
//!   `rho_alpha_t` row), AHC (per-row L2 norm), pipeline (cosine
//!   distance), centroid (weighted sum check).
//! - [`axpy`] — `y += alpha * x`. Used by centroid
//!   (`centroids[k] += w * embeddings[t]`).
//! - [`pdist_euclidean`] — pairwise condensed Euclidean distance.
//!   Used by AHC (the dominant N²·D inner loop).
//! - [`logsumexp_row`] — numerically-stable `ln(Σ exp(row))`. Used by
//!   VBx's responsibility update.
//!
//! ## Backends
//!
//! Following the colconv pattern (the sister crate at
//! `findit-studio/colconv`):
//!
//! - [`scalar`] — always-compiled reference implementation. The math
//!   contract is anchored here.
//! - [`arch::neon`] — aarch64 NEON.
//! - [`arch::x86_avx2`], [`arch::x86_avx512`] — x86_64 tiers.
//! - wasm32 falls through to scalar (no SIMD backend wired).
//!
//! Public dispatchers in [`self`] (`dot`, `axpy`, `logsumexp_row`)
//! always select the best-available SIMD backend at runtime. Callers
//! needing scalar output explicitly call [`scalar::dot`],
//! [`scalar::axpy`], etc.
//!
//! ## SIMD selection per call site
//!
//! - **AHC pdist** ([`crate::cluster::ahc::ahc_init`]): scalar via
//!   [`scalar::pdist_euclidean`]. The dendrogram cut at `<= threshold`
//!   is a hard discrete decision; AVX2/AVX-512 ulp drift could flip
//!   a partition.
//! - **Hungarian-feeding cosine** ([`crate::pipeline::assign_embeddings`]
//!   stage 6): scalar via [`scalar::dot`]. Soft scores feed
//!   `constrained_argmax`, which is also discrete.
//! - **VBx EM** ([`crate::cluster::vbx::vbx_iterate`]) and centroid
//!   sums ([`crate::cluster::centroid::weighted_centroids`]): SIMD via
//!   [`dot`]/[`axpy`]. These stages are continuous/iterative; ulp
//!   drift smooths instead of flipping discrete decisions.
//! - **Embed aggregation** ([`crate::embed::embedder`]): SIMD via
//!   [`axpy_f32`]. Continuous f32 sum.
//!
//! ## Cross-architecture determinism
//!
//! - **NEON ≡ scalar bit-exact** on aarch64 (`f64::mul_add` 4-acc
//!   tree on both). Verified by [`differential_tests`].
//! - **AVX2/AVX-512 diverge from scalar** by O(1e-15) relative on
//!   well-conditioned inputs (different reduction trees).
//! - **`nalgebra`/matrixmultiply GEMMs** in VBx have their own
//!   uncontrolled SIMD dispatch — cross-arch bit-equality
//!   end-to-end is therefore not deliverable. Algorithm robustness
//!   against ulp drift is validated empirically by `parity_tests`
//!   modules (DER ≤ 0.4% on all 6 captured fixtures, every arch).

pub(crate) mod arch;
mod dispatch;
pub mod scalar;
pub mod spill;

#[cfg(any(feature = "ort", feature = "tch"))]
pub use dispatch::axpy_f32;
#[cfg(feature = "_bench")]
pub use dispatch::pdist_euclidean;
pub use dispatch::{axpy, dot, logsumexp_row};

// ─── runtime CPU-feature detection ───────────────────────────────────
//
// Runtime atomic-cached CPU-feature detection. The crate uses std
// throughout, so we always have access to `std::sync::atomic`;
// detection is computed once and cached.
// `diarization_force_scalar` overrides everything for testing — set
// it via `RUSTFLAGS="--cfg diarization_force_scalar"` to bypass any
// SIMD backend.

#[cfg(target_arch = "aarch64")]
pub(crate) fn neon_available() -> bool {
  if cfg!(diarization_force_scalar) {
    return false;
  }
  std::arch::is_aarch64_feature_detected!("neon")
}

#[cfg(target_arch = "x86_64")]
pub(crate) fn avx2_available() -> bool {
  if cfg!(diarization_force_scalar) || cfg!(diarization_disable_avx2) {
    return false;
  }
  // FMA must be present too. The arch::x86_avx2 kernels are compiled
  // with `#[target_feature(enable = "avx2,fma")]` and use
  // `_mm256_fmadd_pd` directly — Intel mandated AVX2 ⇒ FMA on Haswell
  // (2013), but VIA's Eden X4, hypervisor-masked guests, and a few
  // Pentium/Celeron parts ship AVX2 without FMA. Without this guard
  // those CPUs would hit `#UD` on the first FMA instruction instead
  // of falling through to scalar.
  std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
}

#[cfg(target_arch = "x86_64")]
pub(crate) fn avx512_available() -> bool {
  if cfg!(diarization_force_scalar) || cfg!(diarization_disable_avx512) {
    return false;
  }
  // AVX-512F covers `_mm512_*pd` (8-lane f64) which is what we'd use
  // for dot/axpy/pdist. Other extensions (BW, VL) aren't required.
  std::arch::is_x86_feature_detected!("avx512f")
}

/// Backend-selection assertion tests. The SDE CI jobs run cargo test with
/// `--cfg diarization_assert_avx512` (or `_avx2`) so a feature-detection or
/// emulator regression that silently falls the dispatcher back to scalar
/// fails the build instead of producing a green "scalar matches scalar"
/// differential check. Without this, an SDE/CPUID/XCR0 misconfig could
/// leave the unsafe SIMD load + reduction paths untested in CI.
#[cfg(test)]
mod backend_selection_tests {
  /// Only fires under the AVX-512 SDE job. Asserts the dispatcher would
  /// pick the AVX-512 path. Mirrors `ci/sde_avx512.sh`'s emulation
  /// expectation.
  #[test]
  #[cfg(all(target_arch = "x86_64", diarization_assert_avx512))]
  fn dispatch_selects_avx512_under_sde() {
    assert!(
      super::avx512_available(),
      "diarization_assert_avx512 set but avx512_available() == false; \
       SDE/CPUID regression would silently route SIMD tests through scalar"
    );
  }

  /// Only fires under the AVX2 SDE job. Asserts AVX2+FMA is selected and
  /// AVX-512 is disabled (so the AVX2 backend is actually exercised, not
  /// AVX-512). Mirrors `ci/sde_avx2.sh`'s `-hsw` Haswell emulation.
  #[test]
  #[cfg(all(target_arch = "x86_64", diarization_assert_avx2))]
  fn dispatch_selects_avx2_under_sde() {
    assert!(
      super::avx2_available(),
      "diarization_assert_avx2 set but avx2_available() == false; \
       SDE/CPUID regression would silently route SIMD tests through scalar"
    );
    assert!(
      !super::avx512_available(),
      "diarization_assert_avx2 set but avx512_available() == true; \
       dispatcher would pick AVX-512 instead of the AVX2 backend we want \
       to exercise — check `--cfg diarization_disable_avx512` is in RUSTFLAGS"
    );
  }
}

#[cfg(test)]
mod differential_tests {
  //! Scalar vs SIMD differential tests.
  //!
  //! Contract:
  //! - On `aarch64` (the deployment target), scalar and the NEON
  //!   backend produce **bit-identical** results for all five
  //!   primitives. Achieved by:
  //!   1. scalar uses `f64::mul_add` for per-element FMA (one IEEE
  //!      754 rounding, identical to `vfmaq_f64`);
  //!   2. scalar's reduction tree mirrors NEON's (4 partial sums
  //!      over modulo-4 indices, then `((s00+s10) + (s01+s11))`).
  //! - On `x86_64`, AVX2 (4-lane) and AVX-512 (8-lane) use their
  //!   native lane widths — different reduction trees from NEON.
  //!   Per-element FMA is still bit-identical, but the lane-width
  //!   reduction may diverge from scalar by O(1e-15) relative on
  //!   well-conditioned inputs. Cross-architecture bit-identity is
  //!   not claimed.
  //! - On both architectures, catastrophic-cancellation inputs
  //!   (`[1e16, 1, -1e16, 1]`) legitimately diverge between scalar
  //!   and SIMD due to the documented reduction-order difference.

  use rand::{SeedableRng, prelude::*};
  use rand_chacha::ChaCha20Rng;

  /// On aarch64 scalar matches NEON bit-for-bit; elsewhere the
  /// well-conditioned inputs hold a tighter bound than the previous
  /// 1e-12 contract.
  #[test]
  fn dot_well_conditioned_inputs_match() {
    for d in [4usize, 16, 64, 128, 192, 256] {
      let mut rng = ChaCha20Rng::seed_from_u64(0xab + d as u64);
      let a: Vec<f64> = (0..d).map(|_| rng.random::<f64>() * 2.0 - 1.0).collect();
      let b: Vec<f64> = (0..d).map(|_| rng.random::<f64>() * 2.0 - 1.0).collect();
      let s = super::scalar::dot(&a, &b);
      let v = super::dispatch::dot(&a, &b);
      #[cfg(target_arch = "aarch64")]
      assert_eq!(
        s.to_bits(),
        v.to_bits(),
        "dot d={d} scalar/NEON not bit-identical (s={s}, v={v})"
      );
      #[cfg(not(target_arch = "aarch64"))]
      {
        let rel = ((s - v) / s.abs().max(1.0)).abs();
        assert!(
          rel < 1.0e-14,
          "dot d={d} scalar/SIMD divergence {rel:e} exceeds 1e-14 (s={s}, v={v})"
        );
      }
    }
  }

  /// Odd / non-vector-aligned dimensions exercise the scalar-tail
  /// FMA contract. Without per-tail `f64::mul_add` into the running
  /// sum, the SIMD kernels would drift by ½ ulp from the scalar
  /// reference and break VBx + cosine-distance threshold-sensitive
  /// decisions on odd embedding/PLDA dimensions.
  #[test]
  fn dot_odd_dim_match() {
    for d in [1usize, 3, 5, 7, 9, 17, 33, 65, 129] {
      let mut rng = ChaCha20Rng::seed_from_u64(0xb00 + d as u64);
      let a: Vec<f64> = (0..d).map(|_| rng.random::<f64>() * 2.0 - 1.0).collect();
      let b: Vec<f64> = (0..d).map(|_| rng.random::<f64>() * 2.0 - 1.0).collect();
      let s = super::scalar::dot(&a, &b);
      let v = super::dispatch::dot(&a, &b);
      #[cfg(target_arch = "aarch64")]
      assert_eq!(
        s.to_bits(),
        v.to_bits(),
        "dot d={d} (odd) scalar/NEON not bit-identical (s={s}, v={v})"
      );
      #[cfg(not(target_arch = "aarch64"))]
      {
        let rel = ((s - v) / s.abs().max(1.0)).abs();
        assert!(
          rel < 1.0e-14,
          "dot d={d} (odd) scalar/SIMD divergence {rel:e}"
        );
      }
    }
  }

  /// Realistic embedding-dim L2-norm-squared (the AHC + cosine
  /// normalization pattern).
  #[test]
  fn dot_self_l2_norm_match() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x101);
    let a: Vec<f64> = (0..256).map(|_| rng.random::<f64>() * 2.0 - 1.0).collect();
    let s = super::scalar::dot(&a, &a);
    let v = super::dispatch::dot(&a, &a);
    #[cfg(target_arch = "aarch64")]
    assert_eq!(
      s.to_bits(),
      v.to_bits(),
      "‖a‖² scalar/NEON not bit-identical"
    );
    #[cfg(not(target_arch = "aarch64"))]
    {
      let rel = ((s - v) / s.abs()).abs();
      assert!(rel < 1.0e-14, "‖a‖² scalar/SIMD divergence {rel:e}");
    }
  }

  /// Catastrophic-cancellation inputs *do* diverge across reduction
  /// orders. Scalar uses 4-acc pair reduction; AVX2 uses 4-lane;
  /// AVX-512 uses 8-lane. Test captures the magnitude so any future
  /// kernel rewrite that widens it surfaces here.
  #[test]
  fn dot_catastrophic_cancellation_within_known_band() {
    let a: [f64; 4] = [1e16, 1.0, -1e16, 1.0];
    let b: [f64; 4] = [1.0; 4];
    let s = super::scalar::dot(&a, &b);
    let v = super::dispatch::dot(&a, &b);
    let abs_gap = (s - v).abs();
    assert!(
      abs_gap < 10.0,
      "catastrophic-cancellation gap blew up: {abs_gap}"
    );
  }

  /// `pdist_euclidean` differential.
  #[test]
  fn pdist_euclidean_well_conditioned_match() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x202);
    let n = 32usize;
    let d = 192usize;
    let rows: Vec<f64> = (0..n * d)
      .map(|_| rng.random::<f64>() * 2.0 - 1.0)
      .collect();
    let s = super::scalar::pdist_euclidean(&rows, n, d);
    let v = super::dispatch::pdist_euclidean(&rows, n, d);
    assert_eq!(s.len(), v.len(), "pdist length mismatch");
    for (idx, (sv, vv)) in s.iter().zip(v.iter()).enumerate() {
      #[cfg(target_arch = "aarch64")]
      assert_eq!(
        sv.to_bits(),
        vv.to_bits(),
        "pdist[{idx}] scalar/NEON not bit-identical (s={sv}, v={vv})"
      );
      #[cfg(not(target_arch = "aarch64"))]
      {
        let rel = ((sv - vv) / sv.abs().max(1.0)).abs();
        assert!(rel < 1.0e-14, "pdist[{idx}] divergence {rel:e}");
        let _ = idx;
      }
    }
  }

  /// `pdist_euclidean` differential at odd / non-vector-aligned
  /// dimensions. Locks the scalar-tail FMA contract: every backend's
  /// scalar tail must use `f64::mul_add`. Without this, an odd-d
  /// run drifts by ½ ulp per tail step on the SIMD path, which
  /// can flip AHC merges around the threshold for embeddings whose
  /// dim isn't a multiple of the vector width.
  #[test]
  fn pdist_euclidean_odd_dim_match() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x2031);
    // Pick a few non-power-of-2 dims that exercise the tail loop in
    // each backend (NEON: 2-wide; AVX2: 4-wide; AVX-512: 8-wide).
    for &d in &[1, 3, 5, 7, 9, 17, 33, 65, 129] {
      let n = 8usize;
      let rows: Vec<f64> = (0..n * d)
        .map(|_| rng.random::<f64>() * 2.0 - 1.0)
        .collect();
      let s = super::scalar::pdist_euclidean(&rows, n, d);
      let v = super::dispatch::pdist_euclidean(&rows, n, d);
      assert_eq!(s.len(), v.len(), "pdist length mismatch (d={d})");
      for (idx, (sv, vv)) in s.iter().zip(v.iter()).enumerate() {
        #[cfg(target_arch = "aarch64")]
        assert_eq!(
          sv.to_bits(),
          vv.to_bits(),
          "pdist[{idx}] (d={d}) scalar/NEON not bit-identical (s={sv}, v={vv})"
        );
        #[cfg(not(target_arch = "aarch64"))]
        {
          let rel = ((sv - vv) / sv.abs().max(1.0)).abs();
          assert!(rel < 1.0e-14, "pdist[{idx}] (d={d}) divergence {rel:e}");
          let _ = idx;
        }
      }
    }
  }

  /// Mismatched `dot` lengths must `panic!` (not UB). The dispatcher
  /// enforces `a.len() == b.len()` unconditionally before routing to
  /// the unsafe SIMD kernel — this test would silently OOB-read `b`
  /// if that guard were debug-only.
  #[test]
  #[should_panic(expected = "ops::dot")]
  fn dot_dispatch_panics_on_length_mismatch() {
    let a = vec![1.0_f64; 8];
    let b = vec![1.0_f64; 4];
    let _ = super::dispatch::dot(&a, &b);
  }

  /// Mismatched `axpy` lengths must `panic!` not UB.
  #[test]
  #[should_panic(expected = "ops::axpy")]
  fn axpy_dispatch_panics_on_length_mismatch_under_simd() {
    let mut y = vec![0.0_f64; 8];
    let x = vec![1.0_f64; 4];
    super::dispatch::axpy(&mut y, 0.5, &x);
  }

  /// `pdist_euclidean` rejects shape mismatch with a panic.
  #[test]
  #[should_panic(expected = "ops::pdist_euclidean")]
  fn pdist_dispatch_panics_on_shape_mismatch_under_simd() {
    let rows = vec![1.0_f64; 100]; // 5 * 20 worth of data
    // claim 10 rows × 20 cols (200 entries) — doesn't match 100.
    let _ = super::dispatch::pdist_euclidean(&rows, 10, 20);
  }

  /// `pdist_euclidean` rejects `n * d` overflow before hitting the
  /// unsafe path.
  #[test]
  #[should_panic(expected = "ops::pdist_euclidean")]
  fn pdist_dispatch_panics_on_dim_overflow() {
    let rows: Vec<f64> = vec![];
    let _ = super::dispatch::pdist_euclidean(&rows, usize::MAX, 2);
  }

  /// `axpy` is per-element FMA with no reduction. With scalar using
  /// `f64::mul_add` it must match SIMD's `vfmaq_f64` /
  /// `_mm256_fmadd_pd` / `_mm512_fmadd_pd` bit-for-bit on every
  /// architecture.
  #[test]
  fn axpy_byte_identical() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x303);
    let d = 256usize;
    let alpha = 0.7_f64;
    let x: Vec<f64> = (0..d).map(|_| rng.random::<f64>() * 2.0 - 1.0).collect();
    let y_init: Vec<f64> = (0..d).map(|_| rng.random::<f64>() * 2.0 - 1.0).collect();
    let mut y_scalar = y_init.clone();
    let mut y_simd = y_init.clone();
    super::scalar::axpy(&mut y_scalar, alpha, &x);
    super::dispatch::axpy(&mut y_simd, alpha, &x);
    for (i, (s, v)) in y_scalar.iter().zip(y_simd.iter()).enumerate() {
      assert_eq!(
        s.to_bits(),
        v.to_bits(),
        "axpy[{i}] scalar/SIMD not bit-identical (s={s}, v={v})"
      );
    }
  }
}
