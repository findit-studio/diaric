//! Parity tests for `diarization::plda` against the captured artifacts.
//!
//! Loads `tests/parity/fixtures/01_dialogue/{raw_embeddings, plda_embeddings}.npz`
//! and asserts that the Rust transforms reproduce the captured pyannote
//! outputs within float-cast tolerance.
//!
//! **Hard-fails** when fixtures are absent. The fixtures are committed
//! to the repo and shipped via `cargo publish`; a missing fixture is a
//! packaging or sparse-checkout error, never an opt-out. An earlier
//! silent `eprintln` skip let the high-risk algorithm port silently
//! stop being parity-checked when `cargo test` reported all green.
//!
//! Lives **inside** the crate (under `#[cfg(test)]`) rather than as an
//! integration test in `tests/`. The reason is that
//! [`RawEmbedding::from_raw_array`] and
//! [`PostXvecEmbedding::from_pyannote_capture`] are
//! `#[cfg(test)] pub(crate)` — neither downstream crates nor a
//! separate integration-test crate can construct them, by design.
//! Integration tests live in `tests/` only when they exercise the
//! *external* public API; this test is a parity check against the
//! algorithm's internals.

use std::{fs::File, io::BufReader, path::PathBuf};

use nalgebra::DMatrix;
use npyz::npz::NpzArchive;

use crate::plda::{
  EMBEDDING_DIMENSION, PLDA_DIMENSION, PldaTransform, PostXvecEmbedding, RawEmbedding,
};

fn repo_root() -> PathBuf {
  PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn fixture(rel: &str) -> PathBuf {
  repo_root().join(rel)
}

/// Hard-fail if the captured fixtures are absent. The fixtures are
/// checked into the repo (KB-sized) and shipped via `cargo publish`,
/// so a missing fixture is a packaging or sparse-checkout error,
/// never a normal-flow case.
fn require_fixtures() {
  let required = [
    "tests/parity/fixtures/01_dialogue/raw_embeddings.npz",
    "tests/parity/fixtures/01_dialogue/plda_embeddings.npz",
  ];
  let missing: Vec<&str> = required
    .iter()
    .copied()
    .filter(|p| !repo_root().join(p).exists())
    .collect();
  assert!(
    missing.is_empty(),
    "PLDA parity fixtures missing: {missing:?}. \
     These ship with the crate via `cargo publish`; a missing \
     fixture is a packaging error, not an opt-out. Re-run \
     `tests/parity/python/capture_intermediates.py` against the \
     reference clip to regenerate, or restore the files from a \
     full checkout."
  );
}

/// Open an `.npz` archive and pull out one named array. Returns the
/// decoded data plus its shape (matches numpy's `.shape`).
fn read_npz_array<T>(path: &PathBuf, key: &str) -> (Vec<T>, Vec<u64>)
where
  T: npyz::Deserialize,
{
  let f = File::open(path).expect("open npz");
  let mut z = NpzArchive::new(BufReader::new(f)).expect("read npz");
  let npy = z
    .by_name(key)
    .expect("query archive")
    .unwrap_or_else(|| panic!("array `{key}` not in {}", path.display()));
  let shape: Vec<u64> = npy.shape().to_vec();
  let data: Vec<T> = npy.into_vec().expect("decode array");
  (data, shape)
}

#[test]
fn xvec_transform_matches_pyannote_on_train_embeddings() {
  crate::parity_fixtures_or_skip!();
  require_fixtures();

  let plda = PldaTransform::new().expect("PldaTransform::new");

  // (218, 3, 256) f32 raw WeSpeaker embeddings.
  let raw_path = fixture("tests/parity/fixtures/01_dialogue/raw_embeddings.npz");
  let (raw_flat, raw_shape) = read_npz_array::<f32>(&raw_path, "embeddings");
  assert_eq!(raw_shape.len(), 3);
  let chunks = raw_shape[0] as usize;
  let slots = raw_shape[1] as usize;
  let dim = raw_shape[2] as usize;
  assert_eq!(dim, EMBEDDING_DIMENSION);

  // Train-subset post-PLDA-stage-1 reference + indices.
  let plda_emb_path = fixture("tests/parity/fixtures/01_dialogue/plda_embeddings.npz");
  let (post_xvec_flat, post_xvec_shape) = read_npz_array::<f64>(&plda_emb_path, "post_xvec");
  assert_eq!(post_xvec_shape.len(), 2);
  let n_train = post_xvec_shape[0] as usize;
  let post_dim = post_xvec_shape[1] as usize;
  assert_eq!(post_dim, PLDA_DIMENSION);
  let post_xvec_expected = DMatrix::<f64>::from_row_slice(n_train, post_dim, &post_xvec_flat);

  let (train_chunk_idx, _) = read_npz_array::<i64>(&plda_emb_path, "train_chunk_idx");
  let (train_speaker_idx, _) = read_npz_array::<i64>(&plda_emb_path, "train_speaker_idx");
  assert_eq!(train_chunk_idx.len(), n_train);
  assert_eq!(train_speaker_idx.len(), n_train);

  // Run xvec_transform on each (chunk, slot) and accumulate error stats.
  let mut max_abs_err = 0.0f64;
  let mut max_abs_err_idx = 0usize;
  let mut sum_abs_err = 0.0f64;
  let mut count = 0usize;

  for i in 0..n_train {
    let c = train_chunk_idx[i] as usize;
    let s = train_speaker_idx[i] as usize;
    assert!(c < chunks, "chunk idx {c} out of range {chunks}");
    assert!(s < slots, "slot idx {s} out of range {slots}");

    let off = (c * slots + s) * dim;
    let mut input = [0.0f32; EMBEDDING_DIMENSION];
    input.copy_from_slice(&raw_flat[off..off + EMBEDDING_DIMENSION]);

    // Captured pyannote outputs are RAW (un-L2-normed); wrap them
    // explicitly to match the type-safe API.
    let raw = RawEmbedding::from_raw_array(input).expect("captured WeSpeaker outputs are finite");
    let actual_pe = plda
      .xvec_transform(&raw)
      .expect("captured raw embedding is non-degenerate");
    let actual = actual_pe.as_array();

    for d in 0..PLDA_DIMENSION {
      let want = post_xvec_expected[(i, d)];
      let got = actual[d];
      let err = (want - got).abs();
      sum_abs_err += err;
      count += 1;
      if err > max_abs_err {
        max_abs_err = err;
        max_abs_err_idx = i;
      }
    }
  }

  let mean_abs_err = sum_abs_err / count as f64;
  eprintln!(
    "[parity_plda] xvec_transform: n_train={n_train}, \
         max_abs_err={max_abs_err:.3e} (at row {max_abs_err_idx}), \
         mean_abs_err={mean_abs_err:.3e}"
  );

  // Tolerance rationale: pyannote runs the entire xvec_tf in f64, but
  // the WeSpeaker embedding inputs are f32 from ONNX. Our Rust port
  // matches the algorithm but promotes f32 → f64 at the input
  // boundary, identically to numpy's implicit promotion. Any residual
  // error is float-cast roundoff in the L2 normalization (~1e-7
  // floor). 1e-5 is comfortably above that. Empirically the actual
  // error is ~6e-14 — essentially machine epsilon.
  assert!(
    max_abs_err < 1e-5,
    "xvec_transform parity failed: max_abs_err = {max_abs_err:.3e}"
  );
}

#[test]
fn plda_transform_matches_pyannote_modulo_eigenvector_signs() {
  crate::parity_fixtures_or_skip!();
  require_fixtures();

  let plda = PldaTransform::new().expect("PldaTransform::new");

  // Use the captured `post_xvec` as input — that way this test
  // isolates `plda_transform`. Drift in `xvec_transform` is already
  // covered by the previous test; here we only stress the Cholesky-
  // reduced generalized-eigh + projection.
  let plda_emb_path = fixture("tests/parity/fixtures/01_dialogue/plda_embeddings.npz");
  let (post_xvec_in_flat, post_xvec_in_shape) = read_npz_array::<f64>(&plda_emb_path, "post_xvec");
  let n_train = post_xvec_in_shape[0] as usize;
  let post_dim = post_xvec_in_shape[1] as usize;
  assert_eq!(post_dim, PLDA_DIMENSION);
  let post_xvec_in = DMatrix::<f64>::from_row_slice(n_train, post_dim, &post_xvec_in_flat);

  let (post_plda_flat, _) = read_npz_array::<f64>(&plda_emb_path, "post_plda");
  let post_plda_expected = DMatrix::<f64>::from_row_slice(n_train, post_dim, &post_plda_flat);

  // Run plda_transform on each captured post_xvec row.
  let mut rust_post_plda = DMatrix::<f64>::zeros(n_train, PLDA_DIMENSION);
  let mut per_elem_abs_max_err = 0.0f64;
  for i in 0..n_train {
    let mut input = [0.0f64; PLDA_DIMENSION];
    for d in 0..PLDA_DIMENSION {
      input[d] = post_xvec_in[(i, d)];
    }
    // The captured post_xvec values come from a verified pyannote
    // run; wrap explicitly via the from_pyannote_capture constructor
    // (which validates norm ≈ sqrt(D_out)).
    let post = PostXvecEmbedding::from_pyannote_capture(input)
      .expect("captured post_xvec is in-distribution");
    let actual = plda.plda_transform(&post);
    for d in 0..PLDA_DIMENSION {
      rust_post_plda[(i, d)] = actual[d];
      // Sign-invariant element comparison: |abs(want) - abs(got)|.
      // Generalized-eigh eigenvectors are unique only up to sign,
      // so any single column of plda_transform's output may flip
      // sign vs pyannote depending on LAPACK ordering tiebreaks.
      let want = post_plda_expected[(i, d)].abs();
      let got = actual[d].abs();
      let err = (want - got).abs();
      if err > per_elem_abs_max_err {
        per_elem_abs_max_err = err;
      }
    }
  }
  eprintln!("[parity_plda] plda_transform |abs| max_err = {per_elem_abs_max_err:.3e}");

  // Gram-matrix comparison — fully sign-invariant: any column-wise
  // sign flips in the eigenvector matrix cancel in `X X^T`.
  let g_rust = &rust_post_plda * rust_post_plda.transpose();
  let g_py = &post_plda_expected * post_plda_expected.transpose();
  let mut gram_max_err = 0.0f64;
  for i in 0..n_train {
    for j in 0..n_train {
      let err = (g_rust[(i, j)] - g_py[(i, j)]).abs();
      if err > gram_max_err {
        gram_max_err = err;
      }
    }
  }
  eprintln!("[parity_plda] plda_transform Gram max_err = {gram_max_err:.3e}");

  // Tolerances: per-element |abs| < 1e-4 is loose enough to absorb
  // multi-step float roundoff in the eigh + matmul chain. Gram entries
  // sum n_train * 128 products, so float-error scales accordingly;
  // 1e-3 is comfortable for n_train ≈ 200.
  assert!(
    per_elem_abs_max_err < 1e-4,
    "plda_transform |abs| parity failed: max err = {per_elem_abs_max_err:.3e}"
  );
  assert!(
    gram_max_err < 1e-3,
    "plda_transform Gram parity failed: max err = {gram_max_err:.3e}"
  );
}

#[test]
fn phi_matches_pyannote_descending_eigenvalues() {
  crate::parity_fixtures_or_skip!();
  require_fixtures();
  let plda = PldaTransform::new().expect("PldaTransform::new");
  let phi = plda.phi();
  assert_eq!(phi.len(), PLDA_DIMENSION);

  // Structural: descending order. (Sign-of-eigenvalue is positive
  // by virtue of B and W both being positive-definite, so we don't
  // need a separate >0 check — the numerical comparison below
  // would catch any sign flip.)
  for w in phi.windows(2) {
    assert!(
      w[0] >= w[1],
      "phi must be descending; saw {} < {}",
      w[0],
      w[1]
    );
  }

  // Numerical: byte-equal-ish to pyannote's `pipeline._plda.phi`,
  // captured into `plda_embeddings.npz` via
  // `tests/parity/python/capture_intermediates.py`. VBx consumes
  // phi independently of the projected feature matrix, so a
  // regression that returned raw `psi` or mis-scaled eigenvalues
  // would slip through xvec/plda projection parity (the previous
  // structural-only test) but break VBx posterior updates.
  // (round 8a).
  let plda_emb_path = fixture("tests/parity/fixtures/01_dialogue/plda_embeddings.npz");
  let (phi_expected_flat, phi_expected_shape) = read_npz_array::<f64>(&plda_emb_path, "phi");
  assert_eq!(phi_expected_shape, vec![PLDA_DIMENSION as u64]);
  let mut max_abs_err = 0.0f64;
  for (i, (got, want)) in phi.iter().zip(phi_expected_flat.iter()).enumerate() {
    let err = (got - want).abs();
    if err > max_abs_err {
      max_abs_err = err;
    }
    assert!(
      err < 1.0e-9,
      "phi[{i}] = {got} disagrees with pyannote {want} by {err:.3e}"
    );
  }
  eprintln!("[parity_plda] phi max_abs_err = {max_abs_err:.3e}");

  // Tolerance rationale: phi is a single eigh of two
  // 128×128 positive-definite matrices, computed identically in
  // scipy.linalg.eigh and nalgebra's Cholesky-reduced ordinary
  // eigh. The expected residual is float-cast roundoff (~1e-13);
  // 1e-9 is comfortably above that.
}
