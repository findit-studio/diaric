//! Module-level tests for `diarization::plda`.
//!
//! Heavy parity tests against pyannote's captured outputs live in
//! `tests/parity_plda.rs`. This module covers smaller, model-free
//! invariants — the kind of thing that should hold for any input,
//! and that catches regressions long before the parity tests fail.

use crate::plda::{
  EMBEDDING_DIMENSION, Error, PLDA_DIMENSION, PldaTransform, PostXvecEmbedding, RawEmbedding,
};

fn raw(arr: [f32; EMBEDDING_DIMENSION]) -> RawEmbedding {
  RawEmbedding::from_raw_array(arr).expect("test input must be finite")
}

/// `xvec_transform` output norm is `sqrt(PLDA_DIMENSION) ≈ 11.31` —
/// see `pyannote/audio/utils/vbx.py:211-213`. Catches silent
/// regressions where the outer `sqrt(D_out)` factor is dropped.
#[test]
fn xvec_transform_output_norm_is_sqrt_d_out() {
  let plda = PldaTransform::new().expect("load PLDA");
  // Constant input — non-trivial after centering by mean1.
  let input = raw([0.1f32; EMBEDDING_DIMENSION]);
  let out = plda.xvec_transform(&input).expect("non-degenerate input");
  let norm = out.as_array().iter().map(|v| v * v).sum::<f64>().sqrt();
  let expected = (PLDA_DIMENSION as f64).sqrt();
  assert!(
    (norm - expected).abs() < 1e-6,
    "xvec output norm = {norm}, expected sqrt({PLDA_DIMENSION}) = {expected}"
  );
}

/// `phi` (eigenvalues consumed by VBx) must be sorted descending. The
/// Cholesky-reduced eigh in `transform.rs::generalized_eigh_descending`
/// must produce the same ordering as scipy's `eigh(...)[::-1]`.
#[test]
fn phi_is_sorted_descending() {
  let plda = PldaTransform::new().expect("load PLDA");
  let phi = plda.phi();
  assert_eq!(phi.len(), PLDA_DIMENSION);
  for w in phi.windows(2) {
    assert!(
      w[0] >= w[1],
      "phi must be descending; saw {} < {}",
      w[0],
      w[1]
    );
  }
  // `phi` should also be strictly positive — the generalized eigh
  // of two positive-definite matrices has positive eigenvalues.
  assert!(phi.iter().all(|v| *v > 0.0), "phi must be positive");
}

/// `project()` is `plda_transform(xvec_transform(input))`. Cheap
/// algebraic property: shape-preserving + finite outputs.
#[test]
fn project_chain_is_finite() {
  let plda = PldaTransform::new().expect("load PLDA");
  let input = raw([0.5f32; EMBEDDING_DIMENSION]);
  let projected = plda.project(&input).expect("non-degenerate input");
  assert_eq!(projected.len(), PLDA_DIMENSION);
  assert!(
    projected.iter().all(|v| v.is_finite()),
    "project produced non-finite values: {projected:?}"
  );
}

/// PLDA construction is deterministic — no RNG anywhere in the load
/// path, so two `new()` calls must return bit-identical state.
#[test]
fn new_is_deterministic() {
  let a = PldaTransform::new().expect("load PLDA");
  let b = PldaTransform::new().expect("load PLDA");
  let phi_a = a.phi();
  let phi_b = b.phi();
  for (x, y) in phi_a.iter().zip(phi_b.iter()) {
    assert_eq!(x, y, "phi differs between two PldaTransform::new() calls");
  }
  // Same projection input → same output, byte-identical. The
  // input must have non-trivial norm (the boundary check now
  // rejects all-zero raw vectors as a degraded-embedder failure
  // mode), so use a constant 0.5 here rather than zeros.
  let input = raw([0.5f32; EMBEDDING_DIMENSION]);
  let pa = a.project(&input).expect("non-degenerate");
  let pb = b.project(&input).expect("non-degenerate");
  assert_eq!(pa, pb);
}

// ── Validation tests ( + HIGH) ──────────────────
//
// Input finite-ness is now enforced at `RawEmbedding::from_raw_array`
// construction — `xvec_transform` cannot receive a non-finite input
// at all. Tests that previously fed NaN/Inf directly to
// `xvec_transform` therefore moved to the constructor.

/// NaN input must be rejected at the `RawEmbedding` boundary so it
/// cannot reach any math. Without this check, NaN propagates silently
/// into VBx / clustering with no observability for the caller.
#[test]
fn raw_embedding_rejects_nan() {
  let mut arr = [0.5f32; EMBEDDING_DIMENSION];
  arr[42] = f32::NAN;
  let result = RawEmbedding::from_raw_array(arr);
  assert!(
    matches!(result, Err(Error::NonFiniteInput)),
    "got {result:?}"
  );
}

#[test]
fn raw_embedding_rejects_pos_inf() {
  let mut arr = [0.5f32; EMBEDDING_DIMENSION];
  arr[7] = f32::INFINITY;
  let result = RawEmbedding::from_raw_array(arr);
  assert!(
    matches!(result, Err(Error::NonFiniteInput)),
    "got {result:?}"
  );
}

#[test]
fn raw_embedding_rejects_neg_inf() {
  let mut arr = [0.5f32; EMBEDDING_DIMENSION];
  arr[42] = f32::NEG_INFINITY;
  let result = RawEmbedding::from_raw_array(arr);
  assert!(
    matches!(result, Err(Error::NonFiniteInput)),
    "got {result:?}"
  );
}

// ── Degenerate-input rejection ───
//
// `from_raw_array` only checking finiteness was insufficient: an
// all-zero ONNX output reached xvec_transform, and a `‖arr‖ <
// NORM_EPSILON` floor with `NORM_EPSILON = 1e-12` is below the
// literal floating-point noise floor of f32, so a degraded embedder
// returning `[1e-13; 256]` (norm 1.6e-12) passed the boundary,
// then `x - mean1 ≈ -mean1` produced a centered norm of `‖mean1‖`
// well above XVEC_CENTERED_MIN_NORM, and the L2-normalize
// amplified a fixed `-mean1`-direction into a finite PLDA output.
// The data-calibrated RAW_EMBEDDING_MIN_NORM = 0.01 (50× below
// the smallest real raw norm of 0.536) closes that class.

/// All-zero raw input is the canonical degraded-embedder failure mode
/// (e.g. an ONNX inference that returned zeros without raising). It
/// must be rejected at the boundary, not silently transformed into
/// fabricated speaker evidence downstream.
#[test]
fn raw_embedding_rejects_zero_vector() {
  let arr = [0.0f32; EMBEDDING_DIMENSION];
  let result = RawEmbedding::from_raw_array(arr);
  assert!(
    matches!(result, Err(Error::DegenerateInput)),
    "all-zero raw input must be rejected, got {result:?}"
  );
}

/// Near-zero raw input — per-element `1e-15`, total norm `1.6e-14`.
/// Always rejected: well below any reasonable raw-norm floor.
#[test]
fn raw_embedding_rejects_near_zero_vector() {
  let arr = [1.0e-15f32; EMBEDDING_DIMENSION];
  let result = RawEmbedding::from_raw_array(arr);
  assert!(
    matches!(result, Err(Error::DegenerateInput)),
    "near-zero raw input must be rejected, got {result:?}"
  );
}

/// Tiny-but-nonzero attack (). Per-element `1e-13`,
/// total norm `1.6e-12` — sits *just above* `NORM_EPSILON = 1e-12`,
/// 9 orders of magnitude below the smallest real raw norm of 0.536.
/// With the previous `NORM_EPSILON`-based floor this would have
/// passed, and `xvec_transform` would have produced fabricated
/// speaker evidence (centered norm `‖mean1‖ ≈ 1.42`, way above
/// `XVEC_CENTERED_MIN_NORM`). Must now be rejected.
#[test]
fn raw_embedding_rejects_tiny_nonzero_just_above_norm_epsilon() {
  let arr = [1.0e-13f32; EMBEDDING_DIMENSION];

  // Sanity: the attack input was specifically constructed to slip
  // through a NORM_EPSILON floor. If raw norm ever drops below
  // NORM_EPSILON the test stops being meaningful.
  let raw_norm: f64 = arr
    .iter()
    .map(|v| f64::from(*v) * f64::from(*v))
    .sum::<f64>()
    .sqrt();
  assert!(
    raw_norm > 1.0e-12,
    "test setup invariant: raw_norm = {raw_norm:.3e} must sit \
     above NORM_EPSILON for this regression to verify the fix"
  );

  let result = RawEmbedding::from_raw_array(arr);
  assert!(
    matches!(result, Err(Error::DegenerateInput)),
    "tiny-but-nonzero raw input (norm {raw_norm:.3e}) must be \
     rejected — would otherwise produce fixed-direction speaker \
     evidence after `x - mean1` centering. Got {result:?}"
  );
}

/// Sanity: a normal raw input passes the gate. WeSpeaker outputs are
/// O(units)-magnitude; this test guards against an over-tight
/// threshold that would silently kill real signal.
#[test]
fn raw_embedding_accepts_normal_magnitude_input() {
  let arr = [0.5f32; EMBEDDING_DIMENSION];
  let _ok = RawEmbedding::from_raw_array(arr).expect("normal-magnitude input must pass");
}

// ── Centered-norm degeneracy: collapse-to-mean attack family ─────
//
// The from_raw_array boundary catches all-zero / near-zero inputs.
// More sophisticated variants of the same threat target the inner
// centered-norm guard:
//
// (a) input = mean1.astype(f32) — passes the boundary (raw norm =
//     ‖mean1‖ ≈ 1.42), centered norm is mean1's f32 roundtrip noise
//     (~3.5e-8 for the committed weights). Caught.
//
// (b) input = mean1.astype(f32) + jitter where ‖jitter‖ is small but
//     non-trivial. An earlier f32-noise-calibrated threshold (mean1
//     roundtrip noise × 1000 ≈ 3.5e-5) admitted any jitter above that
//     floor, letting the L2-normalize amplify the attacker-chosen
//     jitter direction into a fabricated speaker-evidence vector.
//     The current threshold XVEC_CENTERED_MIN_NORM = 0.1 (data-
//     calibrated against real centered-norm minimum of 1.36) closes
//     the window. (round 6).

/// Regression for the (a) collapse-to-mean attack. Input is
/// `mean1.astype(f32)` exactly; centered f64 vector is pure
/// quantization noise.
#[test]
fn xvec_transform_rejects_input_equal_to_mean1_as_f32() {
  use super::loader::load_xvec;

  let plda = PldaTransform::new().expect("load PLDA");

  let mean1 = load_xvec().mean1;
  let mut arr = [0.0f32; EMBEDDING_DIMENSION];
  for (slot, value) in arr.iter_mut().zip(mean1.iter()) {
    *slot = *value as f32;
  }

  let raw = RawEmbedding::from_raw_array(arr).expect("input has nontrivial raw norm");
  let result = plda.xvec_transform(&raw);
  assert!(
    matches!(result, Err(Error::DegenerateInput)),
    "mean1.astype(f32) must be rejected, got {result:?}"
  );
}

/// Regression for the (b) `mean1 + jitter` attack. Input is
/// `mean1.astype(f32)` plus a constant offset, sized so its
/// centered f64 norm sits at `1e-3` — well above the previous
/// noise-floor-based threshold (3.5e-5) and well below the new
/// data-calibrated threshold (0.1) and the smallest real centered
/// norm (1.36). With the previous threshold this would have passed
/// and the L2-normalize would have amplified the constant-direction
/// jitter into a unit-norm vector, which the rest of the pipeline
/// would then whiten into a finite `sqrt(128)`-normed PLDA output.
#[test]
fn xvec_transform_rejects_mean1_plus_small_jitter() {
  use super::loader::load_xvec;

  let plda = PldaTransform::new().expect("load PLDA");

  // Build mean1 as f32 + a constant per-element offset whose
  // resulting centered f64 norm is `1e-3`. A constant offset of
  // magnitude `c` across `D` elements gives centered norm
  // `c * sqrt(D)`, so `c = 1e-3 / sqrt(256) ≈ 6.25e-5`.
  let target_centered_norm = 1.0e-3_f64;
  let offset = (target_centered_norm / (EMBEDDING_DIMENSION as f64).sqrt()) as f32;

  let mean1 = load_xvec().mean1;
  let mut arr = [0.0f32; EMBEDDING_DIMENSION];
  for (slot, value) in arr.iter_mut().zip(mean1.iter()) {
    *slot = (*value as f32) + offset;
  }

  // Boundary accepts (raw norm ≈ ‖mean1‖, well above NORM_EPSILON).
  let raw = RawEmbedding::from_raw_array(arr).expect("input has nontrivial raw norm");

  // Sanity: the actual centered f64 norm here is in the danger band
  // `(prev_threshold, new_threshold) = (3.5e-5, 0.1)`.
  let centered_norm: f64 = arr
    .iter()
    .zip(mean1.iter())
    .map(|(v, m)| {
      let d = f64::from(*v) - *m;
      d * d
    })
    .sum::<f64>()
    .sqrt();
  assert!(
    (1.0e-4..1.0e-2).contains(&centered_norm),
    "test setup invariant: centered_norm = {centered_norm:.3e} must \
     sit in the previous-threshold-bypass window for the test to be meaningful"
  );

  let result = plda.xvec_transform(&raw);
  assert!(
    matches!(result, Err(Error::DegenerateInput)),
    "mean1 + small jitter (centered norm {centered_norm:.3e}) must be \
     rejected — attacker controls the jitter direction, the \
     L2-normalize would amplify it into fabricated speaker evidence; \
     got {result:?}"
  );
}

// ── PostXvecEmbedding boundary ( stage 2) ─────────
//
// `plda_transform` no longer accepts a bare `[f64; 128]` — its input
// is now `&PostXvecEmbedding`, a newtype that enforces the post-`xvec_tf`
// distribution invariant. NaN/Inf rejection moved to the constructor.

#[test]
fn post_xvec_capture_rejects_nan() {
  let mut arr = [0.0f64; PLDA_DIMENSION];
  arr[3] = f64::NAN;
  let result = PostXvecEmbedding::from_pyannote_capture(arr);
  assert!(
    matches!(result, Err(Error::NonFiniteInput)),
    "got {result:?}"
  );
}

#[test]
fn post_xvec_capture_rejects_inf() {
  let mut arr = [0.0f64; PLDA_DIMENSION];
  arr[100] = f64::INFINITY;
  let result = PostXvecEmbedding::from_pyannote_capture(arr);
  assert!(
    matches!(result, Err(Error::NonFiniteInput)),
    "got {result:?}"
  );
}

/// L2-normalized 128-d vector (norm = 1.0) is the most likely
/// stage-2 misuse. The `from_pyannote_capture` norm check rejects it.
#[test]
fn post_xvec_capture_rejects_l2_normalized_vector() {
  let mut arr = [0.0f64; PLDA_DIMENSION];
  arr[0] = 1.0; // unit vector along axis 0 — norm = 1.0
  let result = PostXvecEmbedding::from_pyannote_capture(arr);
  assert!(
    matches!(result, Err(Error::WrongPostXvecNorm { actual, expected, .. })
        if (actual - 1.0).abs() < 1e-12 && (expected - (PLDA_DIMENSION as f64).sqrt()).abs() < 1e-9),
    "got {result:?}"
  );
}

/// Random / hand-constructed input with arbitrary norm is also
/// rejected. Catches accidental zero-vectors, mis-scaled inputs, etc.
#[test]
fn post_xvec_capture_rejects_zero_vector() {
  let arr = [0.0f64; PLDA_DIMENSION];
  let result = PostXvecEmbedding::from_pyannote_capture(arr);
  assert!(
    matches!(result, Err(Error::WrongPostXvecNorm { actual: 0.0, .. })),
    "got {result:?}"
  );
}

/// Sanity: a synthetic vector with the right norm passes the gate.
#[test]
fn post_xvec_capture_accepts_correctly_scaled_vector() {
  let expected_norm = (PLDA_DIMENSION as f64).sqrt();
  let per_elem = expected_norm / (PLDA_DIMENSION as f64).sqrt();
  // each element = 1.0; sum of squares = 128; norm = sqrt(128) ✓
  assert!((per_elem - 1.0).abs() < 1e-12);
  let arr = [per_elem; PLDA_DIMENSION];
  let post = PostXvecEmbedding::from_pyannote_capture(arr).expect("right norm");
  assert_eq!(post.as_array().len(), PLDA_DIMENSION);
}

/// Round-trip: `xvec_transform`'s output goes straight into
/// `plda_transform` via the type system — no extra validation needed.
#[test]
fn xvec_to_plda_round_trip_uses_post_xvec_type() {
  let plda = PldaTransform::new().expect("load PLDA");
  let input = raw([0.5f32; EMBEDDING_DIMENSION]);
  let post = plda.xvec_transform(&input).expect("non-degenerate");
  let _ = plda.plda_transform(&post); // infallible — no Result on stage 2
}

// ── RawEmbedding domain enforcement () ────────────

/// Feeding an L2-normalized vector (the wrong distribution for PLDA)
/// produces a materially-different output than feeding the
/// corresponding raw vector. The test is observable evidence that
/// the API distinction matters — if a future refactor accidentally
/// loses the `RawEmbedding` wrapper, this test stays as proof of
/// what's at stake.
///
/// We construct the same vector in both forms (`raw_arr` vs
/// `raw_arr / ‖raw_arr‖`), wrap each as `RawEmbedding`, and assert
/// that `xvec_transform`'s outputs differ by far more than float
/// roundoff.
#[test]
fn normalized_vs_raw_input_produce_materially_different_output() {
  let plda = PldaTransform::new().expect("load PLDA");

  // Use a noticeably-non-unit input vector.
  let mut raw_arr = [0.0f32; EMBEDDING_DIMENSION];
  for (i, slot) in raw_arr.iter_mut().enumerate() {
    *slot = ((i as f32) - 128.0) * 0.01;
  }
  let raw_norm: f32 = raw_arr.iter().map(|v| v * v).sum::<f32>().sqrt();
  assert!(
    (raw_norm - 1.0).abs() > 0.5,
    "test input must be far from unit norm: norm = {raw_norm}"
  );
  let mut normed_arr = raw_arr;
  for slot in normed_arr.iter_mut() {
    *slot /= raw_norm;
  }

  let raw_in = raw(raw_arr);
  let normed_in = raw(normed_arr);
  let raw_out = plda.xvec_transform(&raw_in).expect("raw out");
  let normed_out = plda.xvec_transform(&normed_in).expect("normed out");

  let l1_diff: f64 = raw_out
    .as_array()
    .iter()
    .zip(normed_out.as_array().iter())
    .map(|(a, b)| (a - b).abs())
    .sum();
  // The PLDA transform is non-linear (centering + L2-norm + sqrt(D)
  // scaling at two different stages); identical inputs always
  // produce identical outputs, but materially different inputs
  // (raw vs L2-normalized) produce materially different outputs.
  // This bound (>1.0 sum-abs-difference over 128 dims) is loose
  // enough to be robust to tiny test-input changes but tight
  // enough to catch a regression where the type system stops
  // distinguishing raw from normalized.
  assert!(
    l1_diff > 1.0,
    "normalized vs raw produced near-identical output (sum-abs diff = \
     {l1_diff:.3e}); the API contract is broken"
  );
}
