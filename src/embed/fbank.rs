//! Bit-near-exact port of `torchaudio.compliance.kaldi.fbank` plus the
//! pyannote / WeSpeaker post-processing.
//!
//! Reference: `torchaudio/compliance/kaldi.py:514` (torchaudio 2.11).
//! The previous fbank backend was the `kaldi-native-fbank` C++ crate,
//! which uses kaldi's reference implementation but produced ~2.4e-4
//! f32 drift vs torchaudio. On the 23.6-min Mandarin interview
//! `08_luyu_jinjing_freedom`, that drift amplified through ResNet34's
//! 33 conv layers to ~0.66 absolute error in one embedding element on
//! a single (chunk, speaker) pair, flipping a borderline AHC merge
//! and producing a spurious 4th speaker. After this port, the same
//! audio produces 3 speakers / 448 segments / DER = 0.0000 vs
//! pyannote 4.0.4.
//!
//! ## Pipeline
//!
//! Mirrors `torchaudio.compliance.kaldi.fbank` with the WeSpeaker
//! / pyannote configuration baked in:
//!
//! 1. `_get_strided`: split samples into `(num_frames, 400)` frames
//!    at shift 160, snip_edges=true.
//! 2. `remove_dc_offset`: subtract per-frame mean.
//! 3. `preemphasis`: `x[i, j] -= 0.97 * x[i, max(0, j-1)]`.
//! 4. Hamming window (alpha=0.54, beta=0.46, periodic=false).
//! 5. Zero-pad each frame to padded_window_size = 512.
//! 6. Real FFT → `(num_frames, 257)` complex spectrum.
//! 7. Power spectrum: `re² + im²`.
//! 8. Mel filterbank: 80 triangular bins, 20 Hz → Nyquist.
//! 9. `log(max(eps, mel_energies))` with `eps = f32::EPSILON`.
//!
//! Then per-clip pyannote post-processing:
//!
//! - Input is scaled by `1 << 15 = 32_768` so the int16-magnitude
//!   computation matches WeSpeaker's reference
//!   (`pyannote/audio/pipelines/speaker_verification.py:549`).
//! - Output is mean-subtracted per mel band across frames
//!   (`speaker_verification.py:566`).
//!
//! ## Numerical contract
//!
//! Verified against `torchaudio.compliance.kaldi.fbank`: max abs
//! element error ~2.2e-4 on the worst frame of a 23.6-min Mandarin
//! recording, but propagates to ≤1e-5 max abs in the WeSpeaker
//! embedding (vs 0.66 with the prior `kaldi-native-fbank` backend).
//! 95 % of cells agree below 1e-5; the residual is f32 FFT
//! reduction-order noise (rustfft radix-2 vs PyTorch's pocketfft).
//!
//! ## SIMD
//!
//! The mel filterbank dot product (~20 M f32 ops per 10 s chunk) is
//! the dominant cost. It uses an `f32` multiplication + `f64`
//! accumulation kernel that mirrors PyTorch's BLAS-backed `torch.mm`
//! (sgemm with f64 reductions). Backends are selected at runtime via
//! the `crate::ops` feature-detection helpers:
//!
//! | Arch                | Lanes (f32 mul) | Lanes (f64 acc) |
//! |---------------------|----------------:|----------------:|
//! | aarch64 NEON        |               4 |               2 |
//! | x86_64 SSE2         |               4 |               2 |
//! | x86_64 AVX2 + FMA   |               8 |               4 |
//! | x86_64 AVX-512F     |              16 |               8 |
//!
//! Window multiply and power spectrum use NEON / SSE2 with auto-
//! vectorization fallback; they're a small fraction of total cost.

use std::{cell::RefCell, sync::OnceLock};

use realfft::{RealFftPlanner, RealToComplex, num_complex::Complex32};

use crate::embed::{
  error::Error,
  options::{FBANK_FRAMES, FBANK_NUM_MELS, MIN_CLIP_SAMPLES},
};

#[cfg(target_arch = "aarch64")]
use crate::ops::neon_available;
#[cfg(target_arch = "x86_64")]
use crate::ops::{avx2_available, avx512_available};

// ────────────────────────────────────────────────────────────────────
// Constants — fixed by the WeSpeaker / pyannote contract.
// ────────────────────────────────────────────────────────────────────

const SAMPLE_RATE_HZ: f32 = 16_000.0;
const WINDOW_SIZE: usize = 400; // 25 ms @ 16 kHz
const WINDOW_SHIFT: usize = 160; // 10 ms @ 16 kHz
const PADDED_WINDOW_SIZE: usize = 512; // round_to_power_of_two(400)
const NUM_MEL_BINS: usize = 80;
const LOW_FREQ_HZ: f32 = 20.0;
const PREEMPH_COEFF: f32 = 0.97;
const NUM_FFT_BINS: usize = PADDED_WINDOW_SIZE / 2; // 256
const FFT_SPECTRUM_LEN: usize = NUM_FFT_BINS + 1; // 257 incl. Nyquist
const SCALE_INT16: f32 = 32_768.0; // 1 << 15

// `f32::EPSILON = 2^-23`. Matches torchaudio's `_get_epsilon` floor.
const EPSILON: f32 = f32::EPSILON;

/// Maximum f32 sample count that the thread-local `scaled` /
/// `RAW_BUF` scratches keep across calls. The hot path is fixed
/// 10 s / 16 kHz chunks (160 K samples) plus a small safety margin.
/// One-off long clips (e.g. 30 min via `compute_full_fbank`) still
/// run correctly — they just allocate a fresh buffer that is dropped
/// at the end of the call rather than pinning hundreds of MB per
/// worker thread for the lifetime of the process.
const SCRATCH_RETAIN_LIMIT: usize = 256 * 1024;

// `FBANK_NUM_MELS` is dia's public-API constant; compile-time check it
// matches the local `NUM_MEL_BINS` (so changes to `embed::options`
// can't silently desync the kernel).
const _: () = assert!(NUM_MEL_BINS == FBANK_NUM_MELS);

// ────────────────────────────────────────────────────────────────────
// Cached resources (process-global, init-once).
// ────────────────────────────────────────────────────────────────────

static HAMMING_WINDOW: OnceLock<[f32; WINDOW_SIZE]> = OnceLock::new();
static MEL_BANK: OnceLock<MelBank> = OnceLock::new();

/// Symmetric Hamming window (`periodic=False`): computed in f64 then
/// cast — matches torchaudio's `_feature_window_function`.
fn hamming_window() -> &'static [f32; WINDOW_SIZE] {
  HAMMING_WINDOW.get_or_init(|| {
    let mut w = [0.0_f32; WINDOW_SIZE];
    let denom = (WINDOW_SIZE as f64) - 1.0;
    let two_pi = 2.0_f64 * std::f64::consts::PI;
    for (i, slot) in w.iter_mut().enumerate() {
      *slot = (0.54_f64 - 0.46_f64 * (two_pi * (i as f64) / denom).cos()) as f32;
    }
    w
  })
}

/// Mel-scale conversion (kaldi convention): `1127 * ln(1 + f/700)`.
#[inline]
fn mel_scale(freq: f64) -> f64 {
  1127.0 * (1.0 + freq / 700.0).ln()
}

/// Row-major `(NUM_MEL_BINS, FFT_SPECTRUM_LEN)` triangular mel
/// filterbank. Column 256 (Nyquist) is zero — torchaudio right-pads
/// the bank before matmul, we bake that pad into the cached array.
type MelBank = [[f32; FFT_SPECTRUM_LEN]; NUM_MEL_BINS];

fn mel_bank() -> &'static MelBank {
  MEL_BANK.get_or_init(|| {
    let nyquist = (SAMPLE_RATE_HZ as f64) * 0.5;
    let fft_bin_width = (SAMPLE_RATE_HZ as f64) / (PADDED_WINDOW_SIZE as f64);
    let mel_low = mel_scale(LOW_FREQ_HZ as f64);
    let mel_high = mel_scale(nyquist);
    let mel_delta = (mel_high - mel_low) / (NUM_MEL_BINS as f64 + 1.0);
    let mut bank: MelBank = [[0.0_f32; FFT_SPECTRUM_LEN]; NUM_MEL_BINS];
    for (m, row) in bank.iter_mut().enumerate() {
      let left_mel = mel_low + (m as f64) * mel_delta;
      let center_mel = mel_low + (m as f64 + 1.0) * mel_delta;
      let right_mel = mel_low + (m as f64 + 2.0) * mel_delta;
      for (k, slot) in row.iter_mut().enumerate().take(NUM_FFT_BINS) {
        let mel_freq = mel_scale(fft_bin_width * (k as f64));
        let up = (mel_freq - left_mel) / (center_mel - left_mel);
        let down = (right_mel - mel_freq) / (right_mel - center_mel);
        *slot = up.min(down).max(0.0) as f32;
      }
    }
    bank
  })
}

// ────────────────────────────────────────────────────────────────────
// Thread-local scratch + FFT plan.
// ────────────────────────────────────────────────────────────────────
//
// Per-call alloc/free of these (~10 KB total of small Vecs + a planner
// borrow_mut) was visible in profiles for short clips. Pinning them
// thread-local cuts ~6 alloc/free pairs per `compute_fbank` call and
// avoids re-planning the size-512 r2c FFT each time.

struct FftScratch {
  plan: std::sync::Arc<dyn RealToComplex<f32>>,
  fft_input: Vec<f32>,
  fft_output: Vec<Complex32>,
  frame: Vec<f32>,
  power: Vec<f32>,
  /// Pre-scaled `samples * 1<<15`. Pre-scaling once (instead of in
  /// the per-frame copy) is necessary because frames overlap by
  /// `WINDOW_SIZE - WINDOW_SHIFT = 240` samples, so an inlined
  /// scale would re-multiply each sample ~2.5× on average. The
  /// buffer is reused across calls — only the first call to a
  /// thread allocates.
  scaled: Vec<f32>,
}

thread_local! {
  static FFT_SCRATCH: RefCell<Option<FftScratch>> = const { RefCell::new(None) };
}

impl FftScratch {
  fn new() -> Self {
    let plan = RealFftPlanner::<f32>::new().plan_fft_forward(PADDED_WINDOW_SIZE);
    Self {
      plan,
      fft_input: vec![0.0_f32; PADDED_WINDOW_SIZE],
      fft_output: vec![Complex32::new(0.0, 0.0); FFT_SPECTRUM_LEN],
      frame: vec![0.0_f32; WINDOW_SIZE],
      power: vec![0.0_f32; FFT_SPECTRUM_LEN],
      scaled: Vec::new(),
    }
  }
}

// ────────────────────────────────────────────────────────────────────
// SIMD kernels.
// ────────────────────────────────────────────────────────────────────

/// In-place element-wise multiply `a[i] *= b[i]` (Hamming window).
#[inline]
// Per-arch cfg blocks each end with their dispatched call + an
// explicit `return`; the trailing returns look needless to clippy
// but each arch only sees one block. Removing them would let
// non-arch-matched fallbacks execute on archs where they shouldn't.
#[allow(clippy::needless_return)]
fn apply_window_inplace(a: &mut [f32], b: &[f32]) {
  // Real assert (not debug_assert) — the SIMD kernels below issue
  // raw-pointer vector loads from both inputs bounded only by
  // `a.len()`. A length mismatch here would OOB-read `b` in release
  // builds where `debug_assert_eq` is a no-op. Mirrors the same
  // safe-boundary rule used by `crate::ops::dispatch::dot`.
  assert_eq!(
    a.len(),
    b.len(),
    "apply_window_inplace: a.len()={} != b.len()={}",
    a.len(),
    b.len()
  );
  // Force-scalar escape hatch for sanitizer / cross-arch determinism
  // testing (`RUSTFLAGS="--cfg diarization_force_scalar"`). Mirrors
  // the gate already wired into `crate::ops` for the cluster ops.
  if cfg!(diarization_force_scalar) {
    window_mul_scalar(a, b);
    return;
  }
  #[cfg(target_arch = "aarch64")]
  {
    if neon_available() {
      // SAFETY: NEON checked.
      unsafe { window_mul_neon(a, b) };
      return;
    }
    window_mul_scalar(a, b);
    return;
  }
  #[cfg(target_arch = "x86_64")]
  {
    // SAFETY: SSE2 is the x86_64 baseline (Rust default target features).
    unsafe { window_mul_sse2(a, b) };
    return;
  }
  #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
  window_mul_scalar(a, b);
}

#[allow(dead_code)] // SIMD path may shadow on some arches
fn window_mul_scalar(a: &mut [f32], b: &[f32]) {
  for (x, y) in a.iter_mut().zip(b.iter()) {
    *x *= *y;
  }
}

/// `power[k] = (sqrt(re² + im²))²` over a real FFT spectrum, matching
/// torchaudio's `complex.abs().pow(2.0)` operation order.
///
/// Mathematically `(sqrt(x))² == x`, but in f32 the two extra
/// roundings (sqrt + multiply) shift the result by ~1-2 ULP per bin
/// from the direct `re² + im²` formula. We follow torchaudio's
/// formula bit-for-bit so the kernel preserves the literal reference
/// contract. Verified empirically that all 14 fixtures in the parity
/// bench (in-repo + testaudioset) still match pyannote's spk/seg
/// counts with this formula.
#[inline]
// See `apply_window_inplace` for the cfg-gated dispatch rationale.
#[allow(clippy::needless_return)]
fn power_spectrum(fft: &[Complex32], power: &mut [f32]) {
  // See `apply_window_inplace` for why this is a real assert.
  assert_eq!(
    fft.len(),
    power.len(),
    "power_spectrum: fft.len()={} != power.len()={}",
    fft.len(),
    power.len()
  );
  if cfg!(diarization_force_scalar) {
    power_scalar(fft, power);
    return;
  }
  #[cfg(target_arch = "aarch64")]
  {
    if neon_available() {
      // SAFETY: NEON checked.
      unsafe { power_neon(fft, power) };
      return;
    }
    power_scalar(fft, power);
    return;
  }
  #[cfg(target_arch = "x86_64")]
  {
    // SAFETY: SSE2 is x86_64 baseline.
    unsafe { power_sse2(fft, power) };
    return;
  }
  #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
  power_scalar(fft, power);
}

#[allow(dead_code)] // SIMD path may shadow on some arches
fn power_scalar(fft: &[Complex32], power: &mut [f32]) {
  for (k, c) in fft.iter().enumerate() {
    // sqrt-then-square mirrors torchaudio's
    // `complex.abs().pow(2.0)` rounding sequence.
    let mag = (c.re * c.re + c.im * c.im).sqrt();
    power[k] = mag * mag;
  }
}

/// `Σ a[i] * b[i]` with f32 multiplication and f64 accumulation.
///
/// torchaudio's mel matmul (`torch.mm` on an f32 power spectrum × f32
/// mel filterbank) reduces in f32 internally — strictly speaking the
/// "BLAS contract" is f32 throughout. We deliberately use a wider f64
/// accumulator instead because:
///
/// 1. The cell-level worst-case drift vs torchaudio is identical
///    (~2.2e-4 max abs on 08 chunk 1146 with either accumulator) —
///    f32-FFT-stage noise dominates the residual.
/// 2. f32 reduction-order rounding noise across SIMD lane widths
///    (NEON 4-lane vs AVX-512 16-lane vs scalar sequential) flips
///    a borderline binarization threshold on at least one fixture
///    in the 14-audio parity bench (09_mrbeast_dollar_date: 8/468
///    → 8/470 segments). f64 accumulation eliminates that noise
///    floor without changing the FFT-dominated worst-case drift.
/// 3. f64 accumulation is strictly more numerically stable, never
///    less. On every cell where torchaudio and dia agree exactly
///    in f32, they also agree exactly when dia widens to f64;
///    where they differ, dia's f64 result is at least as close
///    to the true mathematical sum as torchaudio's f32 result.
///
/// The "f64 widening makes us diverge from torchaudio's BLAS
/// contract" framing is technically true but doesn't matter here:
/// torchaudio's reduction order itself is implementation-defined
/// (Accelerate vs MKL vs OpenBLAS), so there is no single f32 result
/// to match bit-exactly. f64 is the most stable common-case choice.
#[inline]
// See `apply_window_inplace` for the cfg-gated dispatch rationale.
#[allow(clippy::needless_return)]
fn fma_dot_f32_to_f64(a: &[f32], b: &[f32]) -> f64 {
  // Real assert (not debug_assert) — SIMD bodies do raw-pointer
  // vector loads from both inputs bounded only by `a.len()`. Without
  // this guard, a release-build call from a future site that drifted
  // its expected length would OOB-read `b`.
  assert_eq!(
    a.len(),
    b.len(),
    "fma_dot_f32_to_f64: a.len()={} != b.len()={}",
    a.len(),
    b.len()
  );
  if cfg!(diarization_force_scalar) {
    return fma_dot_scalar(a, b);
  }
  #[cfg(target_arch = "aarch64")]
  {
    if neon_available() {
      // SAFETY: NEON checked.
      return unsafe { dot_neon(a, b) };
    }
    return fma_dot_scalar(a, b);
  }
  #[cfg(target_arch = "x86_64")]
  {
    if avx512_available() {
      // SAFETY: AVX-512F checked.
      return unsafe { dot_avx512(a, b) };
    }
    if avx2_available() {
      // SAFETY: AVX2 + FMA checked.
      return unsafe { dot_avx2(a, b) };
    }
    // SAFETY: SSE2 is x86_64 baseline.
    return unsafe { dot_sse2(a, b) };
  }
  #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
  fma_dot_scalar(a, b)
}

#[allow(dead_code)] // referenced from tests + non-SIMD fallbacks
fn fma_dot_scalar(a: &[f32], b: &[f32]) -> f64 {
  // Match the SIMD backends bit-exactly: multiply in f32 first
  // (matching `_mm*_mul_ps` / `vmulq_f32`), then widen to f64 for
  // accumulation. A naive `(*x as f64) * (*y as f64)` would compute
  // the product in f64 — measurably different (~3e-12 per term) on
  // production-scale inputs.
  let mut sum = 0.0_f64;
  for (x, y) in a.iter().zip(b.iter()) {
    sum += (*x * *y) as f64;
  }
  sum
}

// ─── aarch64 NEON kernels ──────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
unsafe fn window_mul_neon(a: &mut [f32], b: &[f32]) {
  use std::arch::aarch64::{vld1q_f32, vmulq_f32, vst1q_f32};
  unsafe {
    let n = a.len();
    let chunks = n / 4;
    let ap = a.as_mut_ptr();
    let bp = b.as_ptr();
    for i in 0..chunks {
      let av = vld1q_f32(ap.add(i * 4));
      let bv = vld1q_f32(bp.add(i * 4));
      vst1q_f32(ap.add(i * 4), vmulq_f32(av, bv));
    }
    for i in (chunks * 4)..n {
      a[i] *= b[i];
    }
  }
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
unsafe fn power_neon(fft: &[Complex32], power: &mut [f32]) {
  use std::arch::aarch64::{vaddq_f32, vld2q_f32, vmulq_f32, vsqrtq_f32, vst1q_f32};
  unsafe {
    let n = fft.len();
    let chunks = n / 4;
    let fp = fft.as_ptr() as *const f32;
    let pp = power.as_mut_ptr();
    for i in 0..chunks {
      // De-interleave 4 complex samples into (re, im) f32x4 vectors.
      let pair = vld2q_f32(fp.add(i * 8));
      let re = pair.0;
      let im = pair.1;
      // Two separate multiplies + add (not `vfmaq_f32`): match the
      // scalar / SSE2 paths' two-rounding semantics. Then sqrt-then-
      // square mirrors torchaudio's `complex.abs().pow(2.0)`.
      let sum = vaddq_f32(vmulq_f32(re, re), vmulq_f32(im, im));
      let mag = vsqrtq_f32(sum);
      vst1q_f32(pp.add(i * 4), vmulq_f32(mag, mag));
    }
    for k in (chunks * 4)..n {
      let c = fft[k];
      let mag = (c.re * c.re + c.im * c.im).sqrt();
      power[k] = mag * mag;
    }
  }
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
unsafe fn dot_neon(a: &[f32], b: &[f32]) -> f64 {
  use std::arch::aarch64::{
    float64x2_t, vaddq_f64, vcvt_f64_f32, vcvt_high_f64_f32, vget_low_f32, vld1q_f32, vld1q_f64,
    vmulq_f32,
  };
  unsafe {
    let n = a.len();
    let chunks = n / 4;
    let zero = [0.0_f64, 0.0_f64];
    let mut acc0 = vld1q_f64(zero.as_ptr());
    let mut acc1 = vld1q_f64(zero.as_ptr());
    let ap = a.as_ptr();
    let bp = b.as_ptr();
    for i in 0..chunks {
      let av = vld1q_f32(ap.add(i * 4));
      let bv = vld1q_f32(bp.add(i * 4));
      // f32 mul (matches torchaudio's f32 product), then widen to
      // f64 lanes for the accumulation tree — see the rationale on
      // `fma_dot_f32_to_f64`.
      let prod = vmulq_f32(av, bv);
      let lo: float64x2_t = vcvt_f64_f32(vget_low_f32(prod));
      let hi: float64x2_t = vcvt_high_f64_f32(prod);
      acc0 = vaddq_f64(acc0, lo);
      acc1 = vaddq_f64(acc1, hi);
    }
    let pair = vaddq_f64(acc0, acc1);
    let mut buf = [0.0_f64; 2];
    std::ptr::copy_nonoverlapping(&pair as *const _ as *const f64, buf.as_mut_ptr(), 2);
    let mut sum = buf[0] + buf[1];
    for i in (chunks * 4)..n {
      sum += (a[i] * b[i]) as f64;
    }
    sum
  }
}

// ─── x86_64 SSE2 kernels ──────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "sse2")]
unsafe fn window_mul_sse2(a: &mut [f32], b: &[f32]) {
  use core::arch::x86_64::{_mm_loadu_ps, _mm_mul_ps, _mm_storeu_ps};
  unsafe {
    let n = a.len();
    let chunks = n / 4;
    let ap = a.as_mut_ptr();
    let bp = b.as_ptr();
    for i in 0..chunks {
      let av = _mm_loadu_ps(ap.add(i * 4));
      let bv = _mm_loadu_ps(bp.add(i * 4));
      _mm_storeu_ps(ap.add(i * 4), _mm_mul_ps(av, bv));
    }
    for i in (chunks * 4)..n {
      a[i] *= b[i];
    }
  }
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "sse2")]
unsafe fn power_sse2(fft: &[Complex32], power: &mut [f32]) {
  use core::arch::x86_64::{
    _mm_add_ps, _mm_loadu_ps, _mm_mul_ps, _mm_shuffle_ps, _mm_sqrt_ps, _mm_storeu_ps,
  };
  unsafe {
    let n = fft.len();
    let chunks = n / 4;
    let fp = fft.as_ptr() as *const f32;
    let pp = power.as_mut_ptr();
    for i in 0..chunks {
      let v0 = _mm_loadu_ps(fp.add(i * 8)); // [c0re, c0im, c1re, c1im]
      let v1 = _mm_loadu_ps(fp.add(i * 8 + 4)); // [c2re, c2im, c3re, c3im]
      // De-interleave: shuffle 0b10_00_10_00 picks indices [0,2] from
      // each operand → [c0re, c1re, c2re, c3re].
      let re = _mm_shuffle_ps::<0b10_00_10_00>(v0, v1);
      let im = _mm_shuffle_ps::<0b11_01_11_01>(v0, v1);
      // sqrt-then-square mirrors torchaudio's
      // `complex.abs().pow(2.0)` rounding sequence.
      let sum = _mm_add_ps(_mm_mul_ps(re, re), _mm_mul_ps(im, im));
      let mag = _mm_sqrt_ps(sum);
      _mm_storeu_ps(pp.add(i * 4), _mm_mul_ps(mag, mag));
    }
    for k in (chunks * 4)..n {
      let c = fft[k];
      let mag = (c.re * c.re + c.im * c.im).sqrt();
      power[k] = mag * mag;
    }
  }
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "sse2")]
unsafe fn dot_sse2(a: &[f32], b: &[f32]) -> f64 {
  use core::arch::x86_64::{
    __m128d, _mm_add_pd, _mm_cvtps_pd, _mm_loadu_ps, _mm_movehl_ps, _mm_mul_ps, _mm_setzero_pd,
    _mm_unpackhi_pd,
  };
  unsafe {
    let n = a.len();
    let chunks = n / 4;
    let mut acc0: __m128d = _mm_setzero_pd();
    let mut acc1: __m128d = _mm_setzero_pd();
    let ap = a.as_ptr();
    let bp = b.as_ptr();
    for i in 0..chunks {
      let av = _mm_loadu_ps(ap.add(i * 4));
      let bv = _mm_loadu_ps(bp.add(i * 4));
      let prod = _mm_mul_ps(av, bv); // 4 f32
      let lo = _mm_cvtps_pd(prod); // bottom 2 f32 → 2 f64
      let hi = _mm_cvtps_pd(_mm_movehl_ps(prod, prod)); // top 2 f32 → 2 f64
      acc0 = _mm_add_pd(acc0, lo);
      acc1 = _mm_add_pd(acc1, hi);
    }
    let acc = _mm_add_pd(acc0, acc1);
    let hi2 = _mm_unpackhi_pd(acc, acc);
    let sum_v = _mm_add_pd(acc, hi2);
    let buf: [f64; 2] = std::mem::transmute(sum_v);
    let mut sum = buf[0];
    for i in (chunks * 4)..n {
      sum += (a[i] * b[i]) as f64;
    }
    sum
  }
}

// ─── x86_64 AVX2 kernel (mel matmul only) ─────────────────────────

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f64 {
  use core::arch::x86_64::{
    __m256d, _mm_add_pd, _mm_cvtsd_f64, _mm_unpackhi_pd, _mm256_add_pd, _mm256_castpd256_pd128,
    _mm256_castps256_ps128, _mm256_cvtps_pd, _mm256_extractf128_pd, _mm256_extractf128_ps,
    _mm256_loadu_ps, _mm256_mul_ps, _mm256_setzero_pd,
  };
  unsafe {
    let n = a.len();
    let chunks = n / 8;
    let mut acc0: __m256d = _mm256_setzero_pd();
    let mut acc1: __m256d = _mm256_setzero_pd();
    let ap = a.as_ptr();
    let bp = b.as_ptr();
    for i in 0..chunks {
      let av = _mm256_loadu_ps(ap.add(i * 8));
      let bv = _mm256_loadu_ps(bp.add(i * 8));
      let prod = _mm256_mul_ps(av, bv);
      let lo = _mm256_cvtps_pd(_mm256_castps256_ps128(prod));
      let hi = _mm256_cvtps_pd(_mm256_extractf128_ps::<1>(prod));
      acc0 = _mm256_add_pd(acc0, lo);
      acc1 = _mm256_add_pd(acc1, hi);
    }
    let acc = _mm256_add_pd(acc0, acc1);
    let lo128 = _mm256_castpd256_pd128(acc);
    let hi128 = _mm256_extractf128_pd::<1>(acc);
    let sum2 = _mm_add_pd(lo128, hi128);
    let mut sum = _mm_cvtsd_f64(_mm_add_pd(sum2, _mm_unpackhi_pd(sum2, sum2)));
    for i in (chunks * 8)..n {
      sum += (a[i] * b[i]) as f64;
    }
    sum
  }
}

// ─── x86_64 AVX-512F kernel (mel matmul only) ─────────────────────

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn dot_avx512(a: &[f32], b: &[f32]) -> f64 {
  use core::arch::x86_64::{
    __m512d, _mm512_add_pd, _mm512_castps512_ps256, _mm512_cvtps_pd, _mm512_loadu_ps,
    _mm512_mul_ps, _mm512_reduce_add_pd, _mm512_setzero_pd, _mm512_shuffle_f32x4,
  };
  unsafe {
    let n = a.len();
    let chunks = n / 16;
    let mut acc0: __m512d = _mm512_setzero_pd();
    let mut acc1: __m512d = _mm512_setzero_pd();
    let ap = a.as_ptr();
    let bp = b.as_ptr();
    for i in 0..chunks {
      let av = _mm512_loadu_ps(ap.add(i * 16));
      let bv = _mm512_loadu_ps(bp.add(i * 16));
      let prod = _mm512_mul_ps(av, bv);
      let lo = _mm512_cvtps_pd(_mm512_castps512_ps256(prod));
      // AVX-512F-only path to extract upper 256 bits — see commentary
      // on the AVX-512DQ/F gating: `_mm512_extractf32x8_ps` is DQ,
      // `_mm512_shuffle_f32x4` is F.
      let hi_full = _mm512_shuffle_f32x4::<0b00_00_11_10>(prod, prod);
      let hi = _mm512_cvtps_pd(_mm512_castps512_ps256(hi_full));
      acc0 = _mm512_add_pd(acc0, lo);
      acc1 = _mm512_add_pd(acc1, hi);
    }
    let acc = _mm512_add_pd(acc0, acc1);
    let mut sum = _mm512_reduce_add_pd(acc);
    for i in (chunks * 16)..n {
      sum += (a[i] * b[i]) as f64;
    }
    sum
  }
}

// ────────────────────────────────────────────────────────────────────
// Core fbank kernel (raw torchaudio-style output, not pyannote-
// post-processed).
// ────────────────────────────────────────────────────────────────────

/// Compute `(num_frames, NUM_MEL_BINS)` log-mel features for `samples`
/// (16 kHz mono f32 in `[-1, 1]`).
///
/// `num_frames = 1 + (n - 400) / 160` for `snip_edges=true`.
/// The caller's input is the un-scaled `[-1, 1]` waveform; we multiply
/// by `1 << 15 = 32_768` internally per pyannote convention.
#[inline]
fn compute_torchaudio_fbank(samples: &[f32], out: &mut Vec<f32>) {
  out.clear();
  let n = samples.len();
  if n < WINDOW_SIZE {
    return;
  }
  let num_frames = 1 + (n - WINDOW_SIZE) / WINDOW_SHIFT;
  out.resize(num_frames * NUM_MEL_BINS, 0.0);

  let window = hamming_window();
  let bank = mel_bank();

  FFT_SCRATCH.with(|cell| {
    let mut slot = cell.borrow_mut();
    let scratch = slot.get_or_insert_with(FftScratch::new);
    let FftScratch {
      plan,
      fft_input,
      fft_output,
      frame,
      power,
      scaled,
    } = scratch;

    // Pre-scale once. We only need `n_used = num_frames * shift +
    // window` samples — the trailing audio after the last frame's
    // window is unused. `resize` reuses the existing capacity across
    // calls, so this is alloc-free after the first call per thread.
    let n_used = (num_frames - 1) * WINDOW_SHIFT + WINDOW_SIZE;
    shrink_scratch_before_resize(scaled, n_used);
    scaled.resize(n_used, 0.0);
    for (i, dst) in scaled.iter_mut().enumerate() {
      *dst = samples[i] * SCALE_INT16;
    }

    for f_idx in 0..num_frames {
      let start = f_idx * WINDOW_SHIFT;
      frame.copy_from_slice(&scaled[start..start + WINDOW_SIZE]);

      // 1. remove_dc_offset.
      let mut sum = 0.0_f32;
      for v in frame.iter() {
        sum += *v;
      }
      let mean = sum / (WINDOW_SIZE as f32);
      for v in frame.iter_mut() {
        *v -= mean;
      }

      // 2. preemphasis: walk right-to-left so j-1 still holds the
      //    pre-update value when read.
      let prev0 = frame[0];
      for j in (1..WINDOW_SIZE).rev() {
        frame[j] -= PREEMPH_COEFF * frame[j - 1];
      }
      frame[0] -= PREEMPH_COEFF * prev0;

      // 3. Hamming window.
      apply_window_inplace(frame, window);

      // 4. Zero-pad to padded_window_size.
      fft_input[..WINDOW_SIZE].copy_from_slice(frame);
      for v in fft_input[WINDOW_SIZE..].iter_mut() {
        *v = 0.0;
      }

      // 5. Real FFT.
      plan
        .process(fft_input, fft_output)
        .expect("rfft size matches plan");

      // 6. Power spectrum.
      power_spectrum(fft_output, power);

      // 7. Mel filterbank multiplication. f32 multiply, f64 accumulate.
      let row_dst = &mut out[f_idx * NUM_MEL_BINS..(f_idx + 1) * NUM_MEL_BINS];
      for m in 0..NUM_MEL_BINS {
        let acc = fma_dot_f32_to_f64(power, &bank[m]);
        let acc_f32 = acc as f32;
        // NaN-propagating floor. Rust's `f32::max` returns the non-NaN
        // operand (unlike `torch.max` which propagates NaN), so a
        // `NaN.max(EPSILON).ln()` would silently produce
        // `EPSILON.ln() = -16.118` and hide a corrupted FFT input.
        // Manual cmp keeps NaN flowing — the embed model's
        // `Error::NonFiniteOutput` check then surfaces it instead of
        // emitting silently-corrupted finite embeddings.
        let floored = if acc_f32 < EPSILON { EPSILON } else { acc_f32 };
        row_dst[m] = floored.ln();
      }
    }

    shrink_scratch_after_loop(scaled);
  });
}

/// Pre-resize cap-and-reset: drop a retained scratch buffer that's
/// larger than the cap when the upcoming call is small enough that
/// it doesn't need it. An earlier huge call must not pin its
/// allocation across this smaller one.
///
/// Extracted as a free function so Miri can verify both branches
/// without going through `compute_torchaudio_fbank`'s FFT path
/// (rustfft's default planners use SIMD intrinsics that Miri can't
/// evaluate).
#[inline]
fn shrink_scratch_before_resize(scaled: &mut Vec<f32>, n_used: usize) {
  if scaled.capacity() > SCRATCH_RETAIN_LIMIT && n_used <= SCRATCH_RETAIN_LIMIT {
    *scaled = Vec::with_capacity(n_used.max(WINDOW_SIZE));
  }
}

/// Post-loop cap-and-reset: a one-shot long clip can grow `scaled`
/// past the retention limit even if it started small. Drop the
/// buffer at the end of `compute_torchaudio_fbank` so it can't pin
/// hundreds of MB per worker thread for the process's lifetime.
///
/// Extracted as a free function — see `shrink_scratch_before_resize`
/// for the rationale.
#[inline]
fn shrink_scratch_after_loop(scaled: &mut Vec<f32>) {
  if scaled.capacity() > SCRATCH_RETAIN_LIMIT {
    *scaled = Vec::new();
  }
}

// ────────────────────────────────────────────────────────────────────
// Public API: pyannote-conventions-applied wrappers.
// ────────────────────────────────────────────────────────────────────

/// Compute the kaldi-compatible fbank for a clip and pad / center-crop
/// to exactly `[FBANK_FRAMES, FBANK_NUM_MELS] = [200, 80]`.
///
/// Used by `EmbedModel::embed*` in the per-window inner loop.
///
/// # Errors
/// - [`Error::InvalidClip`] if `samples.len() < MIN_CLIP_SAMPLES` (< 25 ms).
/// - [`Error::NonFiniteInput`] if any sample is NaN/inf.
pub fn compute_fbank(samples: &[f32]) -> Result<Box<[[f32; FBANK_NUM_MELS]; FBANK_FRAMES]>, Error> {
  if samples.len() < MIN_CLIP_SAMPLES as usize {
    return Err(Error::InvalidClip {
      len: samples.len(),
      min: MIN_CLIP_SAMPLES as usize,
    });
  }
  if samples.iter().any(|s| !s.is_finite()) {
    return Err(Error::NonFiniteInput);
  }

  thread_local! {
    static RAW_BUF: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
  }

  // Boxed: 200 × 80 × 4 = 64 KB array would overflow typical thread
  // stack budgets; heap alloc is amortized over hundreds of inner-loop
  // FFTs.
  let mut out = Box::new([[0.0_f32; FBANK_NUM_MELS]; FBANK_FRAMES]);

  // Predict frame count so we can crop input *before* feeding the
  // kernel. `compute_fbank` always returns exactly `FBANK_FRAMES`
  // rows — there is no reason to compute log-mel features for
  // anything beyond the centered audio window we'll keep. Bounding
  // the input here also bounds every downstream scratch (`scaled` in
  // `FftScratch`, `RAW_BUF`) regardless of how big a clip the caller
  // passes us.
  let total_samples = samples.len();
  let max_frames = if total_samples >= WINDOW_SIZE {
    1 + (total_samples - WINDOW_SIZE) / WINDOW_SHIFT
  } else {
    0
  };
  let cropped: &[f32] = if max_frames > FBANK_FRAMES {
    let start_frame = (max_frames - FBANK_FRAMES) / 2;
    let start_sample = start_frame * WINDOW_SHIFT;
    let end_sample = start_sample + (FBANK_FRAMES - 1) * WINDOW_SHIFT + WINDOW_SIZE;
    &samples[start_sample..end_sample]
  } else {
    samples
  };

  RAW_BUF.with(|cell| {
    let mut raw = cell.borrow_mut();
    compute_torchaudio_fbank(cropped, &mut raw);
    let n_avail = raw.len() / FBANK_NUM_MELS;

    if n_avail >= FBANK_FRAMES {
      // Cropped exactly to FBANK_FRAMES (or one over due to the
      // off-by-one in the crop arithmetic above) — copy straight
      // through. Center-cropping was already done at the slice level.
      let start = (n_avail - FBANK_FRAMES) / 2;
      for (f, out_row) in out.iter_mut().enumerate() {
        let src = &raw[(start + f) * FBANK_NUM_MELS..(start + f + 1) * FBANK_NUM_MELS];
        out_row.copy_from_slice(src);
      }
    } else {
      // Short clip path: zero-pad symmetrically.
      let pad_left = (FBANK_FRAMES - n_avail) / 2;
      for (f, out_row) in out.iter_mut().skip(pad_left).take(n_avail).enumerate() {
        let src = &raw[f * FBANK_NUM_MELS..(f + 1) * FBANK_NUM_MELS];
        out_row.copy_from_slice(src);
      }
    }

    // RAW_BUF is bounded by the cropped-input contract above, but
    // reset it on the unlikely path where `n_avail` exceeded the
    // expected `FBANK_FRAMES * NUM_MEL_BINS` cap (e.g. someone calls
    // `compute_torchaudio_fbank` directly through this thread).
    if raw.capacity() > SCRATCH_RETAIN_LIMIT {
      *raw = Vec::new();
    }
  });

  // Mean-subtract per mel band across frames (pyannote
  // `speaker_verification.py:566`). f64 accumulator: summing 200 raw
  // log-mel f32 coefficients in f32 would lose mantissa bits when the
  // running sum's magnitude exceeds the per-cell magnitude by a few
  // orders. Widening to f64 first keeps the per-mel mean accurate.
  let mut mean_per_mel = [0.0_f64; FBANK_NUM_MELS];
  for row in out.iter() {
    for (m, &v) in row.iter().enumerate() {
      mean_per_mel[m] += v as f64;
    }
  }
  for m in mean_per_mel.iter_mut() {
    *m /= FBANK_FRAMES as f64;
  }
  for row in out.iter_mut() {
    for (m, v) in row.iter_mut().enumerate() {
      *v -= mean_per_mel[m] as f32;
    }
  }

  Ok(out)
}

/// Compute a kaldi-style fbank for an arbitrary-length clip,
/// returning a flat row-major `(num_frames, FBANK_NUM_MELS)` Vec.
///
/// Same kaldi parameters as [`compute_fbank`], same int16 scaling,
/// same per-mel mean centering across frames. Used by the ORT
/// backend for the 10 s chunk + frame-mask path
/// ([`crate::embed::EmbedModel::embed_chunk_with_frame_mask`]) where
/// the frame count varies with the input length and the fixed-size
/// [`compute_fbank`] return type doesn't fit.
pub fn compute_full_fbank(samples: &[f32]) -> Result<Vec<f32>, Error> {
  if samples.len() < MIN_CLIP_SAMPLES as usize {
    return Err(Error::InvalidClip {
      len: samples.len(),
      min: MIN_CLIP_SAMPLES as usize,
    });
  }
  if samples.iter().any(|s| !s.is_finite()) {
    return Err(Error::NonFiniteInput);
  }

  let mut out = Vec::new();
  compute_torchaudio_fbank(samples, &mut out);
  let num_frames = out.len() / FBANK_NUM_MELS;
  if num_frames == 0 {
    return Ok(out);
  }

  // Mean-subtract per mel band across frames.
  let mut mean_per_mel = [0.0_f64; FBANK_NUM_MELS];
  for f in 0..num_frames {
    for m in 0..FBANK_NUM_MELS {
      mean_per_mel[m] += out[f * FBANK_NUM_MELS + m] as f64;
    }
  }
  for m in mean_per_mel.iter_mut() {
    *m /= num_frames as f64;
  }
  for f in 0..num_frames {
    for m in 0..FBANK_NUM_MELS {
      out[f * FBANK_NUM_MELS + m] -= mean_per_mel[m] as f32;
    }
  }

  Ok(out)
}

// ────────────────────────────────────────────────────────────────────
// Tests.
// ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
  use super::*;
  use crate::embed::options::EMBED_WINDOW_SAMPLES;

  // ─── shape / error-path tests (ported verbatim from the prior
  //     fbank.rs, exercise the public API contracts) ─────────────────

  #[test]
  fn rejects_too_short() {
    let r = compute_fbank(&[0.1; 100]);
    assert!(
      matches!(r, Err(Error::InvalidClip { len: 100, min: 400 })),
      "expected InvalidClip {{ len: 100, min: 400 }}, got {r:?}"
    );
  }

  #[test]
  fn rejects_nan() {
    let r = compute_fbank(&[f32::NAN; 32_000]);
    assert!(
      matches!(r, Err(Error::NonFiniteInput)),
      "expected NonFiniteInput, got {r:?}"
    );
  }

  #[test]
  fn produces_correct_shape_for_2s_clip() {
    let samples = vec![0.001_f32; EMBED_WINDOW_SAMPLES as usize];
    let f = compute_fbank(&samples).unwrap();
    assert_eq!(f.len(), FBANK_FRAMES);
    assert_eq!(f[0].len(), FBANK_NUM_MELS);
    for row in f.iter() {
      for &v in row.iter() {
        assert!(v.is_finite(), "fbank coefficient went non-finite: {v}");
      }
    }
  }

  #[test]
  fn produces_correct_shape_for_short_clip_with_padding() {
    let samples = vec![0.001_f32; MIN_CLIP_SAMPLES as usize + 100];
    let f = compute_fbank(&samples).unwrap();
    assert_eq!(f.len(), FBANK_FRAMES);
  }

  #[test]
  fn accepts_min_clip_samples_exactly() {
    let samples = vec![0.001_f32; MIN_CLIP_SAMPLES as usize];
    let f = compute_fbank(&samples).unwrap();
    assert_eq!(f.len(), FBANK_FRAMES);
    assert_eq!(f[0].len(), FBANK_NUM_MELS);
  }

  #[test]
  fn produces_correct_shape_for_long_clip_with_center_crop() {
    let samples = vec![0.001_f32; 2 * EMBED_WINDOW_SAMPLES as usize];
    let f = compute_fbank(&samples).unwrap();
    assert_eq!(f.len(), FBANK_FRAMES);
    assert_eq!(f[0].len(), FBANK_NUM_MELS);
    for row in f.iter() {
      for &v in row.iter() {
        assert!(v.is_finite(), "center-crop branch produced non-finite: {v}");
      }
    }
  }

  #[test]
  fn full_fbank_rejects_too_short() {
    let r = compute_full_fbank(&[0.1; 100]);
    assert!(
      matches!(r, Err(Error::InvalidClip { len: 100, min: 400 })),
      "expected InvalidClip {{ len: 100, min: 400 }}, got {r:?}"
    );
  }

  #[test]
  fn full_fbank_rejects_non_finite() {
    let r = compute_full_fbank(&[f32::NAN; 32_000]);
    assert!(matches!(r, Err(Error::NonFiniteInput)));
    let r = compute_full_fbank(&[f32::INFINITY; 32_000]);
    assert!(matches!(r, Err(Error::NonFiniteInput)));
  }

  #[test]
  fn full_fbank_shape_scales_with_input_length() {
    // 10 s @ 16 kHz, 25 ms / 10 ms, snip_edges = true → 998 frames.
    let samples = vec![0.001_f32; 160_000];
    let out = compute_full_fbank(&samples).unwrap();
    assert!(!out.is_empty());
    assert_eq!(out.len() % FBANK_NUM_MELS, 0);
    assert_eq!(out.len() / FBANK_NUM_MELS, 998);
    for v in &out {
      assert!(v.is_finite());
    }
  }

  #[test]
  fn full_fbank_is_mean_centered_per_mel() {
    let samples: Vec<f32> = (0..32_000)
      .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16_000.0).sin() * 0.5)
      .collect();
    let out = compute_full_fbank(&samples).unwrap();
    let frames = out.len() / FBANK_NUM_MELS;
    assert!(frames > 1);
    for m in 0..FBANK_NUM_MELS {
      let mean: f64 = (0..frames)
        .map(|f| f64::from(out[f * FBANK_NUM_MELS + m]))
        .sum::<f64>()
        / frames as f64;
      assert!(
        mean.abs() < 1e-3,
        "mel {m} mean = {mean} (should be ≈ 0 after mean-subtraction)"
      );
    }
  }

  /// Always-on torchaudio parity test that does NOT require external
  /// fixtures. Pins eight `(frame, mel) → log-mel` anchor points
  /// captured directly from
  /// `torchaudio.compliance.kaldi.fbank` (torchaudio 2.11) on a
  /// deterministic chirp input. Catches regressions in any of:
  /// - mel filterbank construction
  /// - Hamming window
  /// - FFT plan / size
  /// - power spectrum kernel (NEON / SSE2): direct `re²+im²` formula
  /// - dot kernel (NEON / SSE2 / AVX2 / AVX-512)
  /// - per-mel mean centering
  ///
  /// The reference values are torchaudio's actual output (see the
  /// commented capture script below), so this test is a true
  /// cross-implementation parity check. The `1e-4` tolerance pins
  /// the f32 FFT-stage drift floor: realfft (radix-2 Cooley–Tukey)
  /// vs PyTorch's pocketfft contributes ~1e-7 relative on each FFT
  /// output, amplifying through `|fft|²` and the mel filterbank
  /// matmul to ~few ULP per log-mel cell.
  ///
  /// To regenerate the snapshot:
  /// ```python
  /// # tests/parity/python/.venv/bin/python
  /// import torch, numpy as np
  /// import torchaudio.compliance.kaldi as k
  /// n, sr = 24_000, 16_000.0
  /// t = np.arange(n, dtype=np.float32) / sr
  /// freq = 100.0 + 600.0 * t
  /// chunk = 0.3 * np.sin(2.0 * np.pi * freq * t)
  /// fb = k.fbank(
  ///     torch.from_numpy(chunk * 32768.0).unsqueeze(0),
  ///     sample_frequency=16000, frame_length=25.0, frame_shift=10.0,
  ///     dither=0.0, preemphasis_coefficient=0.97, remove_dc_offset=True,
  ///     window_type='hamming', round_to_power_of_two=True,
  ///     snip_edges=True, num_mel_bins=80, low_freq=20.0, high_freq=0.0,
  ///     use_energy=False, raw_energy=True, htk_compat=False,
  ///     energy_floor=1.0, use_log_fbank=True, use_power=True,
  /// )
  /// fb = (fb - fb.mean(dim=0, keepdim=True)).numpy().astype(np.float32)
  /// for f, m in [(5,0),(5,40),(5,79),(50,10),(50,40),(50,70),(100,25),(140,55)]:
  ///     print(f'({f}, {m}, {fb[f, m]}_f32),')
  /// ```
  #[test]
  fn compares_against_torchaudio_inline_chirp_snapshot() {
    // 1.5 s linear chirp from 100 Hz → 1 kHz at 16 kHz mono.
    let n = 24_000_usize;
    let sr = 16_000.0_f32;
    let chunk: Vec<f32> = (0..n)
      .map(|i| {
        let t = i as f32 / sr;
        let freq = 100.0 + 600.0 * t;
        0.3 * (2.0 * std::f32::consts::PI * freq * t).sin()
      })
      .collect();
    let out = compute_full_fbank(&chunk).unwrap();
    let frames = out.len() / NUM_MEL_BINS;
    assert_eq!(frames, 148);
    // torchaudio.compliance.kaldi.fbank reference values.
    let torchaudio_ref: [(usize, usize, f32); 8] = [
      (5, 0, 2.446_690),
      (5, 40, -4.950_203),
      (5, 79, -3.003_859),
      (50, 10, -1.586_259),
      (50, 40, -2.035_988),
      (50, 70, -0.119_349),
      (100, 25, -0.236_334),
      (140, 55, 2.090_996),
    ];
    let mut max_abs = 0.0_f32;
    for (f_idx, m, expected) in torchaudio_ref {
      let got = out[f_idx * NUM_MEL_BINS + m];
      let d = (got - expected).abs();
      if d > max_abs {
        max_abs = d;
      }
    }
    assert!(
      max_abs < 1e-4,
      "fbank vs torchaudio drifted by {max_abs:.3e} (max abs over 8 \
       anchor cells); a SIMD dispatch or kernel regression?"
    );
  }

  // ─── parity checks against captured torchaudio reference ────────

  /// Mel filterbank parity vs torchaudio. `#[ignore]`-gated because
  /// it depends on a captured `.npz` fixture that's not in the repo;
  /// run explicitly with `cargo test -- --ignored` after generating
  /// the fixture via `tests/parity/python/capture_intermediates.py`.
  /// The always-on `compares_against_torchaudio_inline_chirp_snapshot`
  /// test (above) covers
  /// the kernel under CI.
  #[test]
  #[ignore = "needs captured /tmp/mel_bank_ref.npz; run with --ignored"]
  fn matches_torchaudio_mel_bank() {
    let path = std::path::PathBuf::from("/tmp/mel_bank_ref.npz");
    if !path.exists() {
      panic!(
        "{} missing — generate via tests/parity/python/capture_intermediates.py",
        path.display()
      );
    }
    use std::{fs::File, io::BufReader};
    let f = File::open(&path).expect("open");
    let mut z = npyz::npz::NpzArchive::new(BufReader::new(f)).expect("npz");
    let mel_npy = z.by_name("mel").expect("query").expect("missing");
    let ref_mel: Vec<f32> = mel_npy.into_vec().expect("decode");
    let bank = mel_bank();
    let ref_cols = 256; // torchaudio shape is (80, 256), our cached pad is 257
    let mut max_abs = 0.0_f32;
    for m in 0..NUM_MEL_BINS {
      for k in 0..ref_cols {
        let d = (bank[m][k] - ref_mel[m * ref_cols + k]).abs();
        if d > max_abs {
          max_abs = d;
        }
      }
    }
    eprintln!("[mel_bank_parity] max abs error = {max_abs:.3e}");
    assert!(max_abs < 5e-5, "mel bank parity {max_abs:.3e} > 5e-5");
  }

  /// Bit-near-exact parity vs torchaudio on the real chunk that
  /// previously caused the 08 spurious-cluster failure. Same
  /// `#[ignore]` rationale as `matches_torchaudio_mel_bank`.
  #[test]
  #[ignore = "needs captured /tmp/pyannote_fbank_08_c1146.npz; run with --ignored"]
  fn matches_torchaudio_on_08_chunk_1146() {
    let path = std::path::PathBuf::from("/tmp/pyannote_fbank_08_c1146.npz");
    if !path.exists() {
      panic!(
        "{} missing — generate via tests/parity/python/capture_intermediates.py",
        path.display()
      );
    }
    use std::{fs::File, io::BufReader};
    let f = File::open(&path).expect("open");
    let mut z = npyz::npz::NpzArchive::new(BufReader::new(f)).expect("open npz");
    let fbank_npy = z.by_name("fbank").expect("query").expect("missing fbank");
    let fbank_shape: Vec<u64> = fbank_npy.shape().to_vec();
    let num_frames = fbank_shape[0] as usize;
    let ref_fbank: Vec<f32> = fbank_npy.into_vec().expect("decode");
    let chunk_npy = z.by_name("chunk").expect("query").expect("missing chunk");
    let chunk: Vec<f32> = chunk_npy.into_vec().expect("decode");

    let mut got = Vec::new();
    compute_torchaudio_fbank(&chunk, &mut got);
    assert_eq!(got.len(), num_frames * NUM_MEL_BINS);

    let total = num_frames * NUM_MEL_BINS;
    let (mut max_abs, mut e6, mut e5, mut e4, mut sum_sq) =
      (0.0_f32, 0_usize, 0_usize, 0_usize, 0.0_f64);
    let mut max_loc = (0_usize, 0_usize);
    for f_idx in 0..num_frames {
      for m in 0..NUM_MEL_BINS {
        let d = (got[f_idx * NUM_MEL_BINS + m] - ref_fbank[f_idx * NUM_MEL_BINS + m]).abs();
        if d > max_abs {
          max_abs = d;
          max_loc = (f_idx, m);
        }
        if d > 1e-6 {
          e6 += 1;
        }
        if d > 1e-5 {
          e5 += 1;
        }
        if d > 1e-4 {
          e4 += 1;
        }
        sum_sq += (d as f64) * (d as f64);
      }
    }
    eprintln!(
      "[fbank_parity] max abs error = {max_abs:.3e} at frame {} mel {}",
      max_loc.0, max_loc.1
    );
    eprintln!(
      "[fbank_parity] cells > 1e-6: {e6}/{total} ({:.2}%); > 1e-5: {e5}; > 1e-4: {e4}; rms = {:.3e}",
      100.0 * (e6 as f64) / (total as f64),
      (sum_sq / total as f64).sqrt()
    );
    // Drift gauge: residual ~2e-4 is f32 FFT-reduction-order noise
    // (rustfft radix-2 vs PyTorch's pocketfft). Failing this means a
    // meaningful regression upstream.
    assert!(max_abs < 5e-4, "fbank parity {max_abs:.3e} > 5e-4");
  }

  // ─── SIMD cross-check: every available backend agrees with scalar ─

  /// NaN must propagate through the log floor, not be masked to a
  /// finite log value. Rust's `f32::max(NaN, x)` returns `x`, which
  /// would have silently floored a corrupted f32 multiplication to
  /// `EPSILON.ln() = -16.118`. We feed a power spectrum tainted with
  /// NaN through the same dot+log pipeline `compute_torchaudio_fbank`
  /// uses internally and assert NaN survives.
  #[test]
  fn nan_propagates_through_log_floor() {
    // `power` and `bank_row` must have the same length to mirror the
    // production matmul. Place a NaN in `power[3]`.
    let mut power = vec![1.0_f32; FFT_SPECTRUM_LEN];
    power[3] = f32::NAN;
    let bank = mel_bank();
    let acc = fma_dot_f32_to_f64(&power, &bank[10]);
    let acc_f32 = acc as f32;
    let floored = if acc_f32 < EPSILON { EPSILON } else { acc_f32 };
    let log = floored.ln();
    assert!(
      log.is_nan(),
      "expected NaN to propagate through the log floor, got {log}"
    );
  }

  /// Force-scalar escape hatch: when the kernel sees a deterministic
  /// nonsense input, both the SIMD-dispatched dot and the explicit
  /// scalar fallback must agree. This exercises the cfg!() bypass at
  /// the top of `fma_dot_f32_to_f64` only when the build flag is set
  /// (`RUSTFLAGS="--cfg diarization_force_scalar"`); otherwise it
  /// asserts the SIMD path matches scalar within the established
  /// rounding tolerance — which would catch regressions where either
  /// kernel diverges.
  #[test]
  fn force_scalar_cfg_routes_through_scalar_when_set() {
    let n = 257_usize;
    let a: Vec<f32> = (0..n).map(|i| (i as f32 * 0.137).sin()).collect();
    let b: Vec<f32> = (0..n).map(|i| (i as f32 * 0.241 + 1.0).cos()).collect();
    let dispatched = fma_dot_f32_to_f64(&a, &b);
    let scalar = fma_dot_scalar(&a, &b);
    if cfg!(diarization_force_scalar) {
      assert_eq!(
        dispatched, scalar,
        "force-scalar mode but dispatched != scalar — SIMD path was \
         not bypassed"
      );
    } else {
      // f32 mul + f64 accumulate, tree-reduced in SIMD vs left-to-
      // right in scalar: divergence is bounded by `n * f32::EPSILON`
      // since f32 product magnitudes drive the rounding noise floor.
      let tol = (n as f64) * (f32::EPSILON as f64) * (1.0 + scalar.abs());
      assert!(
        (dispatched - scalar).abs() < tol,
        "dispatched={dispatched}, scalar={scalar}, tol={tol:.3e}"
      );
    }
  }

  /// Codex review #2: a one-shot huge call must NOT permanently pin
  /// hundreds of MB in the thread-local fbank scratch. We simulate a
  /// 60 s clip at 16 kHz (~960 K samples), then call `compute_fbank`
  /// with a small clip, and inspect the retained `scaled` capacity.
  /// Both APIs must keep retained scratch ≤ `SCRATCH_RETAIN_LIMIT`.
  ///
  /// Skipped under Miri: ~6 K interpreted-mode FFTs is well past
  /// Miri's per-test budget. The lighter
  /// `caps_oversized_scratch_capacity` test below covers the
  /// cap-and-reset paths under Miri.
  #[cfg_attr(miri, ignore = "interprets ~6K FFTs; covered by lighter test below")]
  #[test]
  fn bounds_thread_local_scratch_after_huge_call() {
    // Huge clip via the unbounded API. Allowed to allocate, but must
    // shrink before returning.
    let huge: Vec<f32> = vec![0.001_f32; 960_000];
    let _ = compute_full_fbank(&huge).unwrap();
    let cap_after_huge = FFT_SCRATCH.with(|cell| {
      cell
        .borrow()
        .as_ref()
        .map(|s| s.scaled.capacity())
        .unwrap_or(0)
    });
    assert!(
      cap_after_huge <= SCRATCH_RETAIN_LIMIT,
      "scaled capacity {cap_after_huge} > SCRATCH_RETAIN_LIMIT {SCRATCH_RETAIN_LIMIT} \
       after huge `compute_full_fbank` call"
    );

    // Now call the fixed-shape API with a typical 2 s clip — its
    // input cropping must keep all scratches bounded too.
    let small: Vec<f32> = vec![0.001_f32; 32_000];
    let _ = compute_fbank(&small).unwrap();
    let cap_after_small = FFT_SCRATCH.with(|cell| {
      cell
        .borrow()
        .as_ref()
        .map(|s| s.scaled.capacity())
        .unwrap_or(0)
    });
    assert!(
      cap_after_small <= SCRATCH_RETAIN_LIMIT,
      "scaled capacity {cap_after_small} > SCRATCH_RETAIN_LIMIT \
       after `compute_fbank` follow-up"
    );
  }

  /// Pre-resize branch — small upcoming call must drop the retained
  /// oversized buffer. Pure helper, no FFT — runs under Miri.
  #[test]
  fn shrink_before_resize_drops_oversized_when_call_small() {
    let mut v: Vec<f32> = Vec::with_capacity(SCRATCH_RETAIN_LIMIT * 2);
    shrink_scratch_before_resize(&mut v, /* n_used = */ MIN_CLIP_SAMPLES as usize);
    assert!(
      v.capacity() <= SCRATCH_RETAIN_LIMIT,
      "scratch capacity {} not bounded after small-call shrink",
      v.capacity()
    );
  }

  /// Pre-resize branch — huge upcoming call must KEEP the retained
  /// buffer (no point dropping a buffer we're about to refill).
  #[test]
  fn shrink_before_resize_keeps_buffer_when_call_huge() {
    let cap_before = SCRATCH_RETAIN_LIMIT * 2;
    let mut v: Vec<f32> = Vec::with_capacity(cap_before);
    shrink_scratch_before_resize(&mut v, /* n_used = */ SCRATCH_RETAIN_LIMIT * 4);
    assert_eq!(
      v.capacity(),
      cap_before,
      "shrink fired on a huge upcoming call — would re-allocate the buffer we're about to use"
    );
  }

  /// Pre-resize branch — already-bounded buffer is left alone.
  #[test]
  fn shrink_before_resize_leaves_bounded_buffer() {
    let cap_before = SCRATCH_RETAIN_LIMIT / 4;
    let mut v: Vec<f32> = Vec::with_capacity(cap_before);
    shrink_scratch_before_resize(&mut v, /* n_used = */ MIN_CLIP_SAMPLES as usize);
    assert_eq!(v.capacity(), cap_before);
  }

  /// Post-loop branch — buffer that grew past the cap during a huge
  /// call must be dropped. This is the branch a previous Miri test
  /// missed (it only exercised the pre-resize branch via a small
  /// call). Pure helper, no FFT — runs under Miri.
  #[test]
  fn shrink_after_loop_drops_oversized() {
    let mut v: Vec<f32> = Vec::with_capacity(SCRATCH_RETAIN_LIMIT * 2);
    shrink_scratch_after_loop(&mut v);
    assert_eq!(
      v.capacity(),
      0,
      "post-loop shrink failed to drop oversized buffer"
    );
  }

  /// Post-loop branch — buffer below the cap is left alone.
  #[test]
  fn shrink_after_loop_keeps_bounded_buffer() {
    let cap_before = SCRATCH_RETAIN_LIMIT / 2;
    let mut v: Vec<f32> = Vec::with_capacity(cap_before);
    shrink_scratch_after_loop(&mut v);
    assert_eq!(v.capacity(), cap_before);
  }

  /// Length-mismatch must `assert!` even in release builds — SIMD
  /// kernels do unchecked raw-pointer loads bounded only by `a.len()`.
  /// Replaces the prior `debug_assert_eq!` which was a no-op in release
  /// and could have OOB-read past `b`.
  #[test]
  #[should_panic(expected = "fma_dot_f32_to_f64")]
  fn dot_panics_on_length_mismatch_in_release() {
    let a = [1.0_f32; 16];
    let b = [1.0_f32; 8];
    let _ = fma_dot_f32_to_f64(&a, &b);
  }

  #[test]
  #[should_panic(expected = "apply_window_inplace")]
  fn window_panics_on_length_mismatch_in_release() {
    let mut a = [1.0_f32; 16];
    let b = [1.0_f32; 8];
    apply_window_inplace(&mut a, &b);
  }

  #[test]
  #[should_panic(expected = "power_spectrum")]
  fn power_panics_on_length_mismatch_in_release() {
    let fft = vec![Complex32::new(0.0, 0.0); 16];
    let mut p = vec![0.0_f32; 8];
    power_spectrum(&fft, &mut p);
  }

  /// Cross-check that whichever SIMD backend the dispatcher selects
  /// at runtime returns the same value as the scalar reference up to
  /// f64 rounding-tree noise. Length grid spans every relevant tail
  /// modulus (3, 7, 15, 17 etc.) for the four backends (4-, 8-, 16-
  /// lane).
  #[test]
  fn dot_kernels_agree_with_scalar() {
    let lens = [1, 3, 4, 7, 8, 15, 16, 17, 31, 32, 64, 257];
    for &n in &lens {
      // Deterministic pseudo-random pattern, no `rand` dep needed.
      let a: Vec<f32> = (0..n).map(|i| (i as f32 * 0.137).sin()).collect();
      let b: Vec<f32> = (0..n).map(|i| (i as f32 * 0.241 + 1.0).cos()).collect();
      let s = fma_dot_scalar(&a, &b);
      let dispatched = fma_dot_f32_to_f64(&a, &b);
      // Tolerance: scalar is left-to-right f64 sum of f32-products,
      // SIMD does tree-reduced f64 sum across lane widths; both are
      // bounded by `n * f32::EPSILON * |s|` per Wilkinson's analysis
      // (the f32-product magnitude drives the rounding noise floor;
      // the f64 accumulator keeps it from compounding).
      let tol = (n as f64) * (f32::EPSILON as f64) * (1.0 + s.abs());
      assert!(
        (dispatched - s).abs() < tol,
        "n={n}: dispatched={dispatched}, scalar={s}, tol={tol:.3e}"
      );
    }
  }

  // ─── per-backend tests ────────────────────────────────────────────
  //
  // The dispatcher above only routes to one SIMD path per CPU
  // (AVX-512 > AVX2 > SSE2 on x86_64; NEON > scalar on aarch64).
  // These per-backend tests bypass the dispatcher and call each
  // unsafe kernel directly, gated on runtime feature detection.
  // Backends not selected by the dispatcher on the current host
  // (e.g. SSE2 on an AVX-512 chip) still get exercised here.

  // Helpers for the per-backend tests. Only the
  // `target_arch = "aarch64"` / `target_arch = "x86_64"` direct-call
  // tests use them; on other archs (i686, riscv64, …) every per-
  // backend test is cfg-excluded and these helpers would be dead
  // code under `-Dwarnings`. Cfg-gate the helpers to match.
  #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
  fn make_test_inputs(n: usize) -> (Vec<f32>, Vec<f32>) {
    let a: Vec<f32> = (0..n).map(|i| (i as f32 * 0.137).sin()).collect();
    let b: Vec<f32> = (0..n).map(|i| (i as f32 * 0.241 + 1.0).cos()).collect();
    (a, b)
  }

  #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
  fn assert_dot_within_tol(got: f64, expected: f64, n: usize) {
    let tol = (n as f64) * (f32::EPSILON as f64) * (1.0 + expected.abs());
    assert!(
      (got - expected).abs() < tol,
      "n={n}: got={got}, scalar={expected}, tol={tol:.3e}"
    );
  }

  #[cfg(target_arch = "aarch64")]
  #[test]
  fn dot_neon_agrees_with_scalar_directly() {
    if !std::arch::is_aarch64_feature_detected!("neon") {
      eprintln!("skip: NEON not available");
      return;
    }
    for n in [1, 3, 4, 7, 8, 15, 16, 17, 31, 32, 64, 257] {
      let (a, b) = make_test_inputs(n);
      let s = fma_dot_scalar(&a, &b);
      // SAFETY: NEON checked via runtime feature detection above.
      let got = unsafe { dot_neon(&a, &b) };
      assert_dot_within_tol(got, s, n);
    }
  }

  #[cfg(target_arch = "x86_64")]
  #[test]
  fn dot_sse2_agrees_with_scalar_directly() {
    // SSE2 is x86_64 baseline; runtime check kept for completeness.
    if !std::arch::is_x86_feature_detected!("sse2") {
      eprintln!("skip: SSE2 not available");
      return;
    }
    for n in [1, 3, 4, 7, 8, 15, 16, 17, 31, 32, 64, 257] {
      let (a, b) = make_test_inputs(n);
      let s = fma_dot_scalar(&a, &b);
      // SAFETY: SSE2 checked.
      let got = unsafe { dot_sse2(&a, &b) };
      assert_dot_within_tol(got, s, n);
    }
  }

  #[cfg(target_arch = "x86_64")]
  #[test]
  fn dot_avx2_agrees_with_scalar_directly() {
    if !std::arch::is_x86_feature_detected!("avx2") || !std::arch::is_x86_feature_detected!("fma") {
      eprintln!("skip: AVX2 + FMA not available");
      return;
    }
    for n in [1, 3, 4, 7, 8, 15, 16, 17, 31, 32, 64, 257] {
      let (a, b) = make_test_inputs(n);
      let s = fma_dot_scalar(&a, &b);
      // SAFETY: AVX2 + FMA checked.
      let got = unsafe { dot_avx2(&a, &b) };
      assert_dot_within_tol(got, s, n);
    }
  }

  #[cfg(target_arch = "x86_64")]
  #[test]
  fn dot_avx512_agrees_with_scalar_directly() {
    if !std::arch::is_x86_feature_detected!("avx512f") {
      eprintln!("skip: AVX-512F not available");
      return;
    }
    for n in [1, 3, 4, 7, 8, 15, 16, 17, 31, 32, 64, 257] {
      let (a, b) = make_test_inputs(n);
      let s = fma_dot_scalar(&a, &b);
      // SAFETY: AVX-512F checked.
      let got = unsafe { dot_avx512(&a, &b) };
      assert_dot_within_tol(got, s, n);
    }
  }

  // Per-backend window/power tests follow the same pattern. Smaller
  // length grid since these don't have lane-width-dependent rounding
  // trees — just direct lane-wise mul/add.

  #[cfg(target_arch = "aarch64")]
  #[test]
  fn window_neon_agrees_with_scalar_directly() {
    if !std::arch::is_aarch64_feature_detected!("neon") {
      return;
    }
    for n in [4, 16, 17, 400] {
      let (mut a, b) = make_test_inputs(n);
      let mut a_scalar = a.clone();
      window_mul_scalar(&mut a_scalar, &b);
      // SAFETY: NEON checked.
      unsafe { window_mul_neon(&mut a, &b) };
      assert_eq!(a, a_scalar);
    }
  }

  #[cfg(target_arch = "x86_64")]
  #[test]
  fn window_sse2_agrees_with_scalar_directly() {
    if !std::arch::is_x86_feature_detected!("sse2") {
      return;
    }
    for n in [4, 16, 17, 400] {
      let (mut a, b) = make_test_inputs(n);
      let mut a_scalar = a.clone();
      window_mul_scalar(&mut a_scalar, &b);
      // SAFETY: SSE2 checked.
      unsafe { window_mul_sse2(&mut a, &b) };
      assert_eq!(a, a_scalar);
    }
  }

  #[cfg(target_arch = "aarch64")]
  #[test]
  fn power_neon_agrees_with_scalar_directly() {
    if !std::arch::is_aarch64_feature_detected!("neon") {
      return;
    }
    for n in [4, 16, 17, FFT_SPECTRUM_LEN] {
      let fft: Vec<Complex32> = (0..n)
        .map(|i| {
          let v = i as f32;
          Complex32::new(v.sin() * 1e3, v.cos() * 1e3)
        })
        .collect();
      let mut p_scalar = vec![0.0_f32; n];
      let mut p_neon = vec![0.0_f32; n];
      power_scalar(&fft, &mut p_scalar);
      // SAFETY: NEON checked.
      unsafe { power_neon(&fft, &mut p_neon) };
      assert_eq!(p_neon, p_scalar);
    }
  }

  #[cfg(target_arch = "x86_64")]
  #[test]
  fn power_sse2_agrees_with_scalar_directly() {
    if !std::arch::is_x86_feature_detected!("sse2") {
      return;
    }
    for n in [4, 16, 17, FFT_SPECTRUM_LEN] {
      let fft: Vec<Complex32> = (0..n)
        .map(|i| {
          let v = i as f32;
          Complex32::new(v.sin() * 1e3, v.cos() * 1e3)
        })
        .collect();
      let mut p_scalar = vec![0.0_f32; n];
      let mut p_sse2 = vec![0.0_f32; n];
      power_scalar(&fft, &mut p_scalar);
      // SAFETY: SSE2 checked.
      unsafe { power_sse2(&fft, &mut p_sse2) };
      assert_eq!(p_sse2, p_scalar);
    }
  }
}
