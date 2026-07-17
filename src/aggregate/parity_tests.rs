//! Bit-exact parity: `count_pyannote(captured_segmentations)` ==
//! `captured_count` for all 6 captured fixtures.

use crate::{aggregate::count_pyannote, reconstruct::SlidingWindow};
use npyz::npz::NpzArchive;
use std::{fs::File, io::BufReader, path::PathBuf};

fn fixture(rel: &str) -> PathBuf {
  PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(rel)
}

fn read_npz_array<T: npyz::Deserialize>(path: &PathBuf, key: &str) -> (Vec<T>, Vec<u64>) {
  let f = File::open(path).expect("open npz");
  let mut z = NpzArchive::new(BufReader::new(f)).expect("read npz");
  let npy = z
    .by_name(key)
    .expect("query archive")
    .unwrap_or_else(|| panic!("array `{key}` not in {}", path.display()));
  let shape = npy.shape().to_vec();
  let data: Vec<T> = npy.into_vec().expect("decode array");
  (data, shape)
}

fn run_count_parity(fixture_dir: &str) {
  crate::parity_fixtures_or_skip!();
  let base = format!("tests/parity/fixtures/{fixture_dir}");
  let (seg_flat_f32, seg_shape) = read_npz_array::<f32>(
    &fixture(&format!("{base}/segmentations.npz")),
    "segmentations",
  );
  let num_chunks = seg_shape[0] as usize;
  let num_frames_per_chunk = seg_shape[1] as usize;
  let num_speakers = seg_shape[2] as usize;
  let segmentations: Vec<f64> = seg_flat_f32.iter().map(|&v| v as f64).collect();

  let (captured_count, _count_shape) =
    read_npz_array::<u8>(&fixture(&format!("{base}/reconstruction.npz")), "count");

  let recon = fixture(&format!("{base}/reconstruction.npz"));
  let (chunk_step_arr, _) = read_npz_array::<f64>(&recon, "chunk_step");
  let (chunk_dur_arr, _) = read_npz_array::<f64>(&recon, "chunk_duration");
  let (frame_step_arr, _) = read_npz_array::<f64>(&recon, "frame_step");
  let (frame_dur_arr, _) = read_npz_array::<f64>(&recon, "frame_duration");

  let tensor = count_pyannote(
    &segmentations,
    num_chunks,
    num_frames_per_chunk,
    num_speakers,
    0.5, // pyannote community-1 onset
    SlidingWindow::new(0.0, chunk_dur_arr[0], chunk_step_arr[0]),
    SlidingWindow::new(0.0, frame_dur_arr[0], frame_step_arr[0]),
    &crate::ops::spill::SpillOptions::default(),
  );
  let computed = tensor.count();

  // Bit-exact: length and every frame must match.
  assert_eq!(
    computed.len(),
    captured_count.len(),
    "{fixture_dir}: count tensor length differs (got {}, want {})",
    computed.len(),
    captured_count.len()
  );
  let mut mismatched = 0usize;
  let mut first_mismatch: Option<(usize, u8, u8)> = None;
  for i in 0..computed.len() {
    if captured_count[i] != computed[i] {
      mismatched += 1;
      if first_mismatch.is_none() {
        first_mismatch = Some((i, captured_count[i], computed[i]));
      }
    }
  }
  assert_eq!(
    mismatched,
    0,
    "{fixture_dir}: {mismatched}/{n} count entries diverge from captured pyannote — \
     bit-exact match expected. First mismatch at index {first:?}",
    n = computed.len(),
    first = first_mismatch
  );
}

#[test]
fn count_matches_pyannote_01_dialogue() {
  run_count_parity("01_dialogue");
}

#[test]
fn count_matches_pyannote_02_pyannote_sample() {
  run_count_parity("02_pyannote_sample");
}

#[test]
fn count_matches_pyannote_03_dual_speaker() {
  run_count_parity("03_dual_speaker");
}

#[test]
fn count_matches_pyannote_04_three_speaker() {
  run_count_parity("04_three_speaker");
}

#[test]
fn count_matches_pyannote_05_four_speaker() {
  run_count_parity("05_four_speaker");
}

#[test]
fn count_matches_pyannote_06_long_recording() {
  run_count_parity("06_long_recording");
}
