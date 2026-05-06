//! Parity test for `diarization::cluster::hungarian::constrained_argmax` against pyannote's
//! captured `hard_clusters`.
//!
//! Loads `tests/parity/fixtures/01_dialogue/clustering.npz` and asserts that
//! running `constrained_argmax` on each captured `soft_clusters[c]` chunk
//! reproduces the captured `hard_clusters[c]` exactly. **Hard-fails** on
//! missing fixtures (same convention as `src/plda/parity_tests.rs` and
//! `src/vbx/parity_tests.rs`).

use std::{fs::File, io::BufReader, path::PathBuf};

use nalgebra::DMatrix;
use npyz::npz::NpzArchive;

use crate::cluster::hungarian::constrained_argmax;

fn repo_root() -> PathBuf {
  PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn fixture(rel: &str) -> PathBuf {
  repo_root().join(rel)
}

fn require_fixtures() {
  let required = ["tests/parity/fixtures/01_dialogue/clustering.npz"];
  let missing: Vec<&str> = required
    .iter()
    .copied()
    .filter(|p| !repo_root().join(p).exists())
    .collect();
  assert!(
    missing.is_empty(),
    "Hungarian parity fixture missing: {missing:?}. \
     Ships with the crate via `cargo publish`; a missing fixture is a \
     packaging error, not an opt-out. Re-run \
     `tests/parity/python/capture_intermediates.py` to regenerate."
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
fn constrained_argmax_matches_pyannote_hard_clusters() {
  crate::parity_fixtures_or_skip!();
  require_fixtures();

  let path = fixture("tests/parity/fixtures/01_dialogue/clustering.npz");
  let (soft_flat, soft_shape) = read_npz_array::<f64>(&path, "soft_clusters");
  let (hard_flat, hard_shape) = read_npz_array::<i8>(&path, "hard_clusters");

  assert_eq!(soft_shape.len(), 3, "soft_clusters must be 3D");
  let num_chunks = soft_shape[0] as usize;
  let num_speakers = soft_shape[1] as usize;
  let num_clusters = soft_shape[2] as usize;

  assert_eq!(hard_shape.len(), 2, "hard_clusters must be 2D");
  assert_eq!(hard_shape[0] as usize, num_chunks);
  assert_eq!(hard_shape[1] as usize, num_speakers);

  let chunk_stride = num_speakers * num_clusters;
  let chunks: Vec<DMatrix<f64>> = (0..num_chunks)
    .map(|c| {
      let slice = &soft_flat[c * chunk_stride..(c + 1) * chunk_stride];
      DMatrix::<f64>::from_row_slice(num_speakers, num_clusters, slice)
    })
    .collect();

  let assignments = constrained_argmax(&chunks).expect("constrained_argmax");
  assert_eq!(assignments.len(), num_chunks);

  let mut mismatches: Vec<(usize, Vec<i32>, Vec<i32>)> = Vec::new();
  for c in 0..num_chunks {
    let got = &assignments[c];
    let want: Vec<i32> = (0..num_speakers)
      .map(|s| hard_flat[c * num_speakers + s] as i32)
      .collect();
    if *got != want {
      mismatches.push((c, got.clone(), want));
    }
  }

  if !mismatches.is_empty() {
    let preview: String = mismatches
      .iter()
      .take(5)
      .map(|(c, got, want)| format!("  chunk {c}: got {got:?}, want {want:?}"))
      .collect::<Vec<_>>()
      .join("\n");
    panic!(
      "constrained_argmax parity failed on {}/{} chunks:\n{preview}",
      mismatches.len(),
      num_chunks
    );
  }

  eprintln!(
    "[parity_hungarian] all {num_chunks} chunks match (shape {num_speakers}x{num_clusters})"
  );
}
