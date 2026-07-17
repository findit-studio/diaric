//! End-to-end parity test: `diarization::reconstruct::reconstruct`
//! against pyannote's captured `discrete_diarization`.

use std::{fs::File, io::BufReader, path::PathBuf};

use nalgebra::DVector;
use npyz::npz::NpzArchive;

use crate::{
  pipeline::{AssignEmbeddingsInput, assign_embeddings},
  reconstruct::{ReconstructInput, SlidingWindow, reconstruct},
};

fn repo_root() -> PathBuf {
  PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn fixture(rel: &str) -> PathBuf {
  repo_root().join(rel)
}

fn require_fixtures(fixture_dir: &str) {
  let required: Vec<String> = [
    "raw_embeddings.npz",
    "segmentations.npz",
    "plda_embeddings.npz",
    "ahc_state.npz",
    "vbx_state.npz",
    "clustering.npz",
    "reconstruction.npz",
  ]
  .iter()
  .map(|f| format!("tests/parity/fixtures/{fixture_dir}/{f}"))
  .collect();
  let missing: Vec<&str> = required
    .iter()
    .map(String::as_str)
    .filter(|p| !repo_root().join(p).exists())
    .collect();
  assert!(
    missing.is_empty(),
    "reconstruct parity fixtures missing: {missing:?}. \
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
fn reconstruct_matches_pyannote_discrete_diarization_01_dialogue() {
  run_reconstruct_parity("01_dialogue");
}

#[test]
fn reconstruct_matches_pyannote_discrete_diarization_02_pyannote_sample() {
  run_reconstruct_parity("02_pyannote_sample");
}

#[test]
fn reconstruct_matches_pyannote_discrete_diarization_03_dual_speaker() {
  run_reconstruct_parity("03_dual_speaker");
}

#[test]
fn reconstruct_matches_pyannote_discrete_diarization_04_three_speaker() {
  run_reconstruct_parity("04_three_speaker");
}

#[test]
fn reconstruct_matches_pyannote_discrete_diarization_05_four_speaker() {
  run_reconstruct_parity("05_four_speaker");
}

/// 06_long_recording (T=1004) — bit-exact discrete-grid parity.
/// Restored by Kahan-summed VBx + `np.unique`-equivalent AHC
/// canonicalization (see
/// `pipeline::parity_tests::assign_embeddings_matches_pyannote_hard_clusters_06_long_recording`).
#[test]
fn reconstruct_matches_pyannote_discrete_diarization_06_long_recording() {
  run_reconstruct_parity("06_long_recording");
}

/// CI-enforced per-frame parity for 06_long_recording.
///
/// Runs the full pipeline (`assign_embeddings → reconstruct`),
/// builds a `(num_clusters × num_clusters)` confusion matrix between
/// our discrete grid and pyannote's captured grid, finds the
/// max-trace cluster permutation by brute-force enumeration (small
/// N, typically ≤ 5), and asserts the post-permutation per-cell
/// mismatch fraction is below a small bound. Catches catastrophic
/// regressions while permitting cluster-id relabeling and the
/// documented O(1e-15) GEMM-roundoff drift.
///
/// Bound chosen with headroom over the observed mismatch rate
/// (streaming-offline DER on this fixture is 0.19 % — per-frame
/// label confusion is typically slightly higher because DER applies
/// a 0.5 s collar; 5 % is a comfortable bound).
#[test]
fn reconstruct_within_tolerance_06_long_recording() {
  run_reconstruct_parity_with_tolerance("06_long_recording", 0.05);
}

fn run_reconstruct_parity(fixture_dir: &str) {
  crate::parity_fixtures_or_skip!();
  require_fixtures(fixture_dir);
  let base = format!("tests/parity/fixtures/{fixture_dir}");

  // ── Stage 5a: produce hard_clusters via the assign_embeddings port ──
  let raw_path = fixture(&format!("{base}/raw_embeddings.npz"));
  let (raw_flat, raw_shape) = read_npz_array::<f32>(&raw_path, "embeddings");
  let num_chunks = raw_shape[0] as usize;
  let num_speakers = raw_shape[1] as usize;
  let embed_dim = raw_shape[2] as usize;
  // Row-major flat `[c][s][d]`, matching the new
  // `AssignEmbeddingsInput::embeddings: &[f64]` contract.
  let embeddings: Vec<f64> = raw_flat.iter().map(|&v| v as f64).collect();

  let seg_path = fixture(&format!("{base}/segmentations.npz"));
  let (seg_flat_f32, seg_shape) = read_npz_array::<f32>(&seg_path, "segmentations");
  let num_frames_per_chunk = seg_shape[1] as usize;
  let segmentations: Vec<f64> = seg_flat_f32.iter().map(|&v| v as f64).collect();

  let plda_path = fixture(&format!("{base}/plda_embeddings.npz"));
  let (post_plda_flat, post_plda_shape) = read_npz_array::<f64>(&plda_path, "post_plda");
  let _num_train = post_plda_shape[0] as usize;
  let plda_dim = post_plda_shape[1] as usize;
  let post_plda: &[f64] = &post_plda_flat;
  let (phi_flat, _) = read_npz_array::<f64>(&plda_path, "phi");
  let phi = DVector::<f64>::from_vec(phi_flat);
  let (chunk_idx_i64, _) = read_npz_array::<i64>(&plda_path, "train_chunk_idx");
  let (speaker_idx_i64, _) = read_npz_array::<i64>(&plda_path, "train_speaker_idx");
  let train_chunk_idx: Vec<usize> = chunk_idx_i64.iter().map(|&v| v as usize).collect();
  let train_speaker_idx: Vec<usize> = speaker_idx_i64.iter().map(|&v| v as usize).collect();

  let ahc_path = fixture(&format!("{base}/ahc_state.npz"));
  let (threshold_data, _) = read_npz_array::<f64>(&ahc_path, "threshold");
  let threshold = threshold_data[0];
  let vbx_path = fixture(&format!("{base}/vbx_state.npz"));
  let (fa_arr, _) = read_npz_array::<f64>(&vbx_path, "fa");
  let (fb_arr, _) = read_npz_array::<f64>(&vbx_path, "fb");
  let (max_iters_arr, _) = read_npz_array::<i64>(&vbx_path, "max_iters");

  let pipeline_input = AssignEmbeddingsInput::new(
    &embeddings,
    embed_dim,
    num_chunks,
    num_speakers,
    &segmentations,
    num_frames_per_chunk,
    post_plda,
    plda_dim,
    &phi,
    &train_chunk_idx,
    &train_speaker_idx,
  )
  .with_threshold(threshold)
  .with_fa(fa_arr[0])
  .with_fb(fb_arr[0])
  .with_max_iters(max_iters_arr[0] as usize);
  let hard_clusters = assign_embeddings(&pipeline_input).expect("assign_embeddings");

  // ── Stage 5b: reconstruct ──────────────────────────────────────
  let recon_path = fixture(&format!("{base}/reconstruction.npz"));
  let (count_u8, count_shape) = read_npz_array::<u8>(&recon_path, "count");
  assert_eq!(count_shape.len(), 2);
  let num_output_frames = count_shape[0] as usize;
  // count is (num_output_frames, 1) → flatten.
  assert_eq!(count_shape[1], 1);
  let (chunk_start_arr, _) = read_npz_array::<f64>(&recon_path, "chunk_start");
  let (chunk_dur_arr, _) = read_npz_array::<f64>(&recon_path, "chunk_duration");
  let (chunk_step_arr, _) = read_npz_array::<f64>(&recon_path, "chunk_step");
  let (frame_start_arr, _) = read_npz_array::<f64>(&recon_path, "frame_start");
  let (frame_dur_arr, _) = read_npz_array::<f64>(&recon_path, "frame_duration");
  let (frame_step_arr, _) = read_npz_array::<f64>(&recon_path, "frame_step");
  let chunks_sw = SlidingWindow::new(chunk_start_arr[0], chunk_dur_arr[0], chunk_step_arr[0]);
  let frames_sw = SlidingWindow::new(frame_start_arr[0], frame_dur_arr[0], frame_step_arr[0]);

  let recon_input = ReconstructInput::new(
    &segmentations,
    num_chunks,
    num_frames_per_chunk,
    num_speakers,
    &hard_clusters,
    &count_u8,
    num_output_frames,
    chunks_sw,
    frames_sw,
  );
  let got = reconstruct(&recon_input).expect("reconstruct");

  // ── Compare to captured discrete_diarization ────────────────────
  let (want_f32, want_shape) = read_npz_array::<f32>(&recon_path, "discrete_diarization");
  assert_eq!(want_shape.len(), 2);
  let want_frames = want_shape[0] as usize;
  let want_clusters = want_shape[1] as usize;
  assert_eq!(want_frames, num_output_frames);

  // Our `got` has num_clusters columns (= max(hard_clusters)+1, padded
  // up to max(count) if needed). Pyannote's `want` has `want_clusters`
  // columns. They should match.
  let got_clusters = got.len() / num_output_frames;
  assert_eq!(
    got_clusters, want_clusters,
    "cluster count mismatch: got {got_clusters}, want {want_clusters}"
  );

  // Element-wise: count mismatched cells. For pyannote-equivalent
  // behavior we expect ZERO mismatches (both binary outputs).
  let mut mismatch = 0usize;
  let mut first_mismatch = None;
  for t in 0..num_output_frames {
    for k in 0..want_clusters {
      let g = got[t * got_clusters + k];
      let w = want_f32[t * want_clusters + k];
      if g != w {
        mismatch += 1;
        if first_mismatch.is_none() {
          first_mismatch = Some((t, k, g, w));
        }
      }
    }
  }
  let total_cells = num_output_frames * want_clusters;
  let mismatch_pct = mismatch as f64 / total_cells as f64 * 100.0;
  eprintln!(
    "[parity_reconstruct] mismatches: {mismatch}/{total_cells} ({mismatch_pct:.4}%); first: {first_mismatch:?}"
  );
  assert!(
    mismatch == 0,
    "discrete_diarization parity failed: {mismatch}/{total_cells} cells diverge ({mismatch_pct:.4}%); \
     first: {first_mismatch:?}"
  );
}

/// Same as [`run_reconstruct_parity`] but compares under a
/// max-trace cluster-id permutation and asserts a bounded per-cell
/// mismatch fraction instead of bit-exact. For long fixtures where
/// chunk-level cluster ids diverge from pyannote's by GEMM-roundoff
/// drift but the per-frame label content is still essentially
/// equivalent.
fn run_reconstruct_parity_with_tolerance(fixture_dir: &str, max_mismatch_frac: f64) {
  crate::parity_fixtures_or_skip!();
  require_fixtures(fixture_dir);
  let base = format!("tests/parity/fixtures/{fixture_dir}");

  // Reuse the data-loading + pipeline run from `run_reconstruct_parity`.
  // We can't share via a helper without a wide return tuple, so the
  // load is inlined here. Any update to the strict variant must mirror.

  let raw_path = fixture(&format!("{base}/raw_embeddings.npz"));
  let (raw_flat, raw_shape) = read_npz_array::<f32>(&raw_path, "embeddings");
  let num_chunks = raw_shape[0] as usize;
  let num_speakers = raw_shape[1] as usize;
  let embed_dim = raw_shape[2] as usize;
  // Row-major flat `[c][s][d]`.
  let embeddings: Vec<f64> = raw_flat.iter().map(|&v| v as f64).collect();

  let seg_path = fixture(&format!("{base}/segmentations.npz"));
  let (seg_flat_f32, seg_shape) = read_npz_array::<f32>(&seg_path, "segmentations");
  let num_frames_per_chunk = seg_shape[1] as usize;
  let segmentations: Vec<f64> = seg_flat_f32.iter().map(|&v| v as f64).collect();

  let plda_path = fixture(&format!("{base}/plda_embeddings.npz"));
  let (post_plda_flat, post_plda_shape) = read_npz_array::<f64>(&plda_path, "post_plda");
  let _num_train = post_plda_shape[0] as usize;
  let plda_dim = post_plda_shape[1] as usize;
  let post_plda: &[f64] = &post_plda_flat;
  let (phi_flat, _) = read_npz_array::<f64>(&plda_path, "phi");
  let phi = DVector::<f64>::from_vec(phi_flat);
  let (chunk_idx_i64, _) = read_npz_array::<i64>(&plda_path, "train_chunk_idx");
  let (speaker_idx_i64, _) = read_npz_array::<i64>(&plda_path, "train_speaker_idx");
  let train_chunk_idx: Vec<usize> = chunk_idx_i64.iter().map(|&v| v as usize).collect();
  let train_speaker_idx: Vec<usize> = speaker_idx_i64.iter().map(|&v| v as usize).collect();

  let ahc_path = fixture(&format!("{base}/ahc_state.npz"));
  let (threshold_data, _) = read_npz_array::<f64>(&ahc_path, "threshold");
  let vbx_path = fixture(&format!("{base}/vbx_state.npz"));
  let (fa_arr, _) = read_npz_array::<f64>(&vbx_path, "fa");
  let (fb_arr, _) = read_npz_array::<f64>(&vbx_path, "fb");
  let (max_iters_arr, _) = read_npz_array::<i64>(&vbx_path, "max_iters");

  let pipeline_input = AssignEmbeddingsInput::new(
    &embeddings,
    embed_dim,
    num_chunks,
    num_speakers,
    &segmentations,
    num_frames_per_chunk,
    post_plda,
    plda_dim,
    &phi,
    &train_chunk_idx,
    &train_speaker_idx,
  )
  .with_threshold(threshold_data[0])
  .with_fa(fa_arr[0])
  .with_fb(fb_arr[0])
  .with_max_iters(max_iters_arr[0] as usize);
  let hard_clusters = assign_embeddings(&pipeline_input).expect("assign_embeddings");

  let recon_path = fixture(&format!("{base}/reconstruction.npz"));
  let (count_u8, count_shape) = read_npz_array::<u8>(&recon_path, "count");
  let num_output_frames = count_shape[0] as usize;
  let (chunk_start_arr, _) = read_npz_array::<f64>(&recon_path, "chunk_start");
  let (chunk_dur_arr, _) = read_npz_array::<f64>(&recon_path, "chunk_duration");
  let (chunk_step_arr, _) = read_npz_array::<f64>(&recon_path, "chunk_step");
  let (frame_start_arr, _) = read_npz_array::<f64>(&recon_path, "frame_start");
  let (frame_dur_arr, _) = read_npz_array::<f64>(&recon_path, "frame_duration");
  let (frame_step_arr, _) = read_npz_array::<f64>(&recon_path, "frame_step");
  let chunks_sw = SlidingWindow::new(chunk_start_arr[0], chunk_dur_arr[0], chunk_step_arr[0]);
  let frames_sw = SlidingWindow::new(frame_start_arr[0], frame_dur_arr[0], frame_step_arr[0]);

  let recon_input = ReconstructInput::new(
    &segmentations,
    num_chunks,
    num_frames_per_chunk,
    num_speakers,
    &hard_clusters,
    &count_u8,
    num_output_frames,
    chunks_sw,
    frames_sw,
  );
  let got = reconstruct(&recon_input).expect("reconstruct");

  let (want_f32, want_shape) = read_npz_array::<f32>(&recon_path, "discrete_diarization");
  assert_eq!(want_shape.len(), 2);
  let want_frames = want_shape[0] as usize;
  let want_clusters = want_shape[1] as usize;
  assert_eq!(want_frames, num_output_frames);
  let got_clusters = got.len() / num_output_frames;
  assert_eq!(
    got_clusters, want_clusters,
    "cluster count mismatch: got {got_clusters}, want {want_clusters}"
  );

  // Confusion matrix: confusion[i][j] = number of frames where got
  // column i is active AND want column j is active. Per-frame both
  // grids are 0/1 (binarized).
  let k = want_clusters;
  let mut confusion = vec![vec![0usize; k]; k];
  for t in 0..num_output_frames {
    for i in 0..k {
      let gi = got[t * k + i] != 0.0;
      if !gi {
        continue;
      }
      for j in 0..k {
        let wj = want_f32[t * k + j] != 0.0;
        if wj {
          confusion[i][j] += 1;
        }
      }
    }
  }

  // Brute-force max-trace permutation. K is small (≤ 5 in our
  // fixtures); enumeration is fine. Heap's algorithm — generates all
  // K! permutations of [0..K).
  let mut perm: Vec<usize> = (0..k).collect();
  let mut best_perm: Vec<usize> = perm.clone();
  let mut best_score: usize = perm.iter().enumerate().map(|(i, &p)| confusion[i][p]).sum();
  let mut counters = vec![0usize; k];
  let mut idx = 0usize;
  while idx < k {
    if counters[idx] < idx {
      if idx.is_multiple_of(2) {
        perm.swap(0, idx);
      } else {
        perm.swap(counters[idx], idx);
      }
      let score: usize = (0..k).map(|i| confusion[i][perm[i]]).sum();
      if score > best_score {
        best_score = score;
        best_perm.clone_from(&perm);
      }
      counters[idx] += 1;
      idx = 0;
    } else {
      counters[idx] = 0;
      idx += 1;
    }
  }

  // Mismatch count under best permutation.
  let mut mismatch = 0usize;
  for t in 0..num_output_frames {
    for i in 0..k {
      let g = got[t * k + i];
      let w = want_f32[t * k + best_perm[i]];
      if g != w {
        mismatch += 1;
      }
    }
  }
  let total = num_output_frames * k;
  let frac = mismatch as f64 / total as f64;
  assert!(
    frac <= max_mismatch_frac,
    "[parity_reconstruct_tolerant] {fixture_dir}: {mismatch}/{total} ({:.3}%) cells diverge \
     under best permutation {best_perm:?}; bound = {:.3}%",
    frac * 100.0,
    max_mismatch_frac * 100.0,
  );
  eprintln!(
    "[parity_reconstruct_tolerant] {fixture_dir}: {mismatch}/{total} ({:.4}%) mismatches \
     under permutation {best_perm:?} (bound {:.3}%)",
    frac * 100.0,
    max_mismatch_frac * 100.0,
  );
}
