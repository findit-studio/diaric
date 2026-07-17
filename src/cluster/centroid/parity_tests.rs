//! Parity test for `diarization::cluster::centroid::weighted_centroids` against
//! pyannote's captured `clustering.npz['centroids']`.
//!
//! Loads:
//! - `vbx_state.npz` for `q_final` and `sp_final` (VBx posterior).
//! - `raw_embeddings.npz` for the raw 256-dim x-vectors.
//! - `plda_embeddings.npz` for the active-frame (chunk_idx, speaker_idx)
//!   pairs used to reshape `raw_embeddings` into the `train_embeddings`
//!   pyannote averages over.
//! - `clustering.npz['centroids']` for the ground-truth centroid matrix.
//!
//! Asserts max element-wise diff ≤ 1e-9. **Hard-fails** on missing fixtures.

use std::{fs::File, io::BufReader, path::PathBuf};

use nalgebra::{DMatrix, DVector};
use npyz::npz::NpzArchive;

use crate::cluster::centroid::{SP_ALIVE_THRESHOLD, weighted_centroids};

fn repo_root() -> PathBuf {
  PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn fixture(rel: &str) -> PathBuf {
  repo_root().join(rel)
}

fn require_fixtures() {
  let required = [
    "tests/parity/fixtures/01_dialogue/raw_embeddings.npz",
    "tests/parity/fixtures/01_dialogue/plda_embeddings.npz",
    "tests/parity/fixtures/01_dialogue/vbx_state.npz",
    "tests/parity/fixtures/01_dialogue/clustering.npz",
  ];
  let missing: Vec<&str> = required
    .iter()
    .copied()
    .filter(|p| !repo_root().join(p).exists())
    .collect();
  assert!(
    missing.is_empty(),
    "centroid parity fixtures missing: {missing:?}. \
     Re-run `tests/parity/python/capture_intermediates.py` to regenerate."
  );
}

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
fn weighted_centroids_match_pyannote_clustering_centroids() {
  crate::parity_fixtures_or_skip!();
  require_fixtures();

  // Load q_final, sp_final from VBx capture.
  let vbx_path = fixture("tests/parity/fixtures/01_dialogue/vbx_state.npz");
  let (q_flat, q_shape) = read_npz_array::<f64>(&vbx_path, "q_final");
  let (sp_flat, sp_shape) = read_npz_array::<f64>(&vbx_path, "sp_final");
  assert_eq!(q_shape.len(), 2);
  let num_train = q_shape[0] as usize;
  let num_init = q_shape[1] as usize;
  assert_eq!(sp_shape, vec![num_init as u64]);
  let q = DMatrix::<f64>::from_row_slice(num_train, num_init, &q_flat);
  let sp = DVector::<f64>::from_vec(sp_flat);

  // Load raw embeddings, project to active-frame (num_train, 256).
  let raw_path = fixture("tests/parity/fixtures/01_dialogue/raw_embeddings.npz");
  let (raw_flat, raw_shape) = read_npz_array::<f32>(&raw_path, "embeddings");
  assert_eq!(raw_shape.len(), 3, "raw embeddings must be 3D");
  let num_chunks = raw_shape[0] as usize;
  let num_speakers = raw_shape[1] as usize;
  let embed_dim = raw_shape[2] as usize;

  let plda_path = fixture("tests/parity/fixtures/01_dialogue/plda_embeddings.npz");
  let (chunk_idx, _) = read_npz_array::<i64>(&plda_path, "train_chunk_idx");
  let (speaker_idx, _) = read_npz_array::<i64>(&plda_path, "train_speaker_idx");
  assert_eq!(chunk_idx.len(), num_train);
  assert_eq!(speaker_idx.len(), num_train);

  // Build a row-major `(num_train, embed_dim)` flat buffer matching
  // `weighted_centroids`'s `&[f64]` contract.
  let mut train: Vec<f64> = Vec::with_capacity(num_train * embed_dim);
  for i in 0..num_train {
    let c = chunk_idx[i] as usize;
    let s = speaker_idx[i] as usize;
    assert!(c < num_chunks && s < num_speakers);
    let base = (c * num_speakers + s) * embed_dim;
    for d in 0..embed_dim {
      train.push(raw_flat[base + d] as f64);
    }
  }

  // Run + compare to clustering.npz['centroids'].
  let got = weighted_centroids(&q, &sp, &train, num_train, embed_dim, SP_ALIVE_THRESHOLD)
    .expect("weighted_centroids");

  let cluster_path = fixture("tests/parity/fixtures/01_dialogue/clustering.npz");
  let (want_flat, want_shape) = read_npz_array::<f64>(&cluster_path, "centroids");
  assert_eq!(want_shape.len(), 2);
  let want_alive = want_shape[0] as usize;
  let want_dim = want_shape[1] as usize;
  assert_eq!(want_dim, embed_dim);
  assert_eq!(
    got.shape(),
    (want_alive, want_dim),
    "centroid shape mismatch: got {:?}, want ({want_alive}, {want_dim})",
    got.shape()
  );
  let want = DMatrix::<f64>::from_row_slice(want_alive, want_dim, &want_flat);

  let mut max_err = 0.0f64;
  let mut max_err_loc = (0usize, 0usize);
  let mut max_err_got = 0.0f64;
  let mut max_err_want = 0.0f64;
  for r in 0..want_alive {
    for c in 0..want_dim {
      let err = (got[(r, c)] - want[(r, c)]).abs();
      if err > max_err {
        max_err = err;
        max_err_loc = (r, c);
        max_err_got = got[(r, c)];
        max_err_want = want[(r, c)];
      }
    }
  }
  eprintln!(
    "[parity_centroid] max_abs_err = {max_err:.3e} at ({}, {}) got={max_err_got:.6e} want={max_err_want:.6e}",
    max_err_loc.0, max_err_loc.1
  );
  assert!(
    max_err < 1.0e-9,
    "centroid parity failed: max_abs_err = {max_err:.3e} at {max_err_loc:?} got={max_err_got:.6e} want={max_err_want:.6e}"
  );
}
