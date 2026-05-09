//! End-to-end parity test: `diarization::pipeline::assign_embeddings` against
//! pyannote's captured `clustering.npz['hard_clusters']`.
//!
//! Inputs (all from the captured fixtures):
//! - `raw_embeddings.npz['embeddings']` — 3D (chunks × speakers × dim) raw
//!   x-vectors (f32 → f64).
//! - `segmentations.npz['segmentations']` — 3D (chunks × frames × speakers)
//!   per-frame speaker probabilities.
//! - `plda_embeddings.npz['post_plda', 'phi', 'train_chunk_idx',
//!   'train_speaker_idx']` — pre-PLDA outputs that `cluster_vbx` would
//!   compute internally; we accept them pre-computed because PLDA parity
//!   is already validated on these exact arrays.
//! - `ahc_state.npz['threshold']` — AHC linkage cutoff (0.6).
//! - `vbx_state.npz['fa', 'fb', 'max_iters']` — VBx hyperparameters.
//!
//! Expected: `clustering.npz['hard_clusters']` (chunks × speakers, int8).
//! Comparison is **partition-equivalent** (canonicalized via
//! encounter-order on each chunk) — same trade-off documented in the
//! AHC parity test (scipy fcluster's traversal-order labels permute the
//! cluster ids; partition is the actual contract).

use std::{fs::File, io::BufReader, path::PathBuf};

use nalgebra::DVector;
use npyz::npz::NpzArchive;

use crate::{
  cluster::hungarian::UNMATCHED,
  pipeline::{AssignEmbeddingsInput, assign_embeddings},
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
    "pipeline parity fixtures missing: {missing:?}. \
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
fn assign_embeddings_matches_pyannote_hard_clusters_01_dialogue() {
  run_pipeline_parity("01_dialogue");
}

#[test]
fn assign_embeddings_matches_pyannote_hard_clusters_02_pyannote_sample() {
  run_pipeline_parity("02_pyannote_sample");
}

#[test]
fn assign_embeddings_matches_pyannote_hard_clusters_03_dual_speaker() {
  run_pipeline_parity("03_dual_speaker");
}

#[test]
fn assign_embeddings_matches_pyannote_hard_clusters_04_three_speaker() {
  run_pipeline_parity("04_three_speaker");
}

#[test]
fn assign_embeddings_matches_pyannote_hard_clusters_05_four_speaker() {
  run_pipeline_parity("05_four_speaker");
}

#[test]
/// 06_long_recording (T=1004) — bit-exact pipeline parity vs pyannote.
///
/// Previously `#[ignore]`d due to GEMM roundoff drift accumulating
/// across more EM iterations on long inputs. Two changes restored
/// strict parity:
///
/// 1. **Kahan-summed VBx GEMM** (`ops::scalar::kahan_dot`,
///    `kahan_sum`): replaces nalgebra's matrixmultiply-backed
///    `gamma.transpose() * rho` and `rho * alpha.T` with
///    Neumaier-compensated reductions. Bound is `O(ε)` regardless of
///    summation order, so the EM trajectory is identical to numpy's
///    BLAS-backed reference.
///
/// 2. **`np.unique`-equivalent AHC label canonicalization**
///    (`ahc/algo.rs::fcluster_distance_remap`): pyannote feeds
///    scipy's `fcluster - 1` through `np.unique(..., return_inverse=
///    True)` (sort distinct labels ascending, remap by rank). The
///    previous leaf-scan encounter-order canonicalization preserved
///    partition equivalence but produced a column-permuted qinit,
///    which on long inputs converged VBx to a different fixed point.
///    Sorting by the DFS-pass label aligns dia's qinit columns with
///    pyannote's bit-for-bit.
fn assign_embeddings_matches_pyannote_hard_clusters_06_long_recording() {
  run_pipeline_parity("06_long_recording");
}

#[test]
#[ignore = "ad-hoc capture from testaudioset; investigates pyannote parity on 10_mrbeast_clean_water (611 chunks)"]
fn assign_embeddings_matches_pyannote_hard_clusters_10_mrbeast_clean_water() {
  run_pipeline_parity("10_mrbeast_clean_water");
}

#[test]
#[ignore = "ad-hoc capture from testaudioset; localizes 08_luyu_jinjing_freedom +1 spk"]
fn assign_embeddings_matches_pyannote_hard_clusters_08_luyu_jinjing_freedom() {
  run_pipeline_parity("08_luyu_jinjing_freedom");
}

/// Dump dia's ahc_init labels (run on captured raw_embeddings) and
/// compare to pyannote's captured ahc_init_labels.npy. Per-row
/// alignment vs partition-equivalence with relabeling will tell us
/// whether the mismatch in pipeline parity comes from label-value
/// differences (permutation OK) or genuine partition divergence.
#[test]
#[ignore = "diagnostic; compares dia's raw AHC labels to pyannote's captured labels on 10"]
fn diagnose_ahc_labels_10_mrbeast() {
  use crate::{cluster::ahc::ahc_init, ops::spill::SpillOptions};
  let dir = "10_mrbeast_clean_water";
  let raw_path = fixture(&format!("tests/parity/fixtures/{dir}/raw_embeddings.npz"));
  let (raw_f32, raw_shape) = read_npz_array::<f32>(&raw_path, "embeddings");
  let _nc = raw_shape[0] as usize;
  let nsp = raw_shape[1] as usize;
  let dim = raw_shape[2] as usize;

  let plda_path = fixture(&format!("tests/parity/fixtures/{dir}/plda_embeddings.npz"));
  let (chunk_idx, _) = read_npz_array::<i64>(&plda_path, "train_chunk_idx");
  let (speaker_idx, _) = read_npz_array::<i64>(&plda_path, "train_speaker_idx");
  let num_train = chunk_idx.len();
  let mut train = Vec::with_capacity(num_train * dim);
  for i in 0..num_train {
    let c = chunk_idx[i] as usize;
    let s = speaker_idx[i] as usize;
    let base = (c * nsp + s) * dim;
    for d in 0..dim {
      train.push(raw_f32[base + d] as f64);
    }
  }

  let ahc_path = fixture(&format!("tests/parity/fixtures/{dir}/ahc_state.npz"));
  let (thr, _) = read_npz_array::<f64>(&ahc_path, "threshold");
  let dia_labels = ahc_init(&train, num_train, dim, thr[0], &SpillOptions::default()).expect("ahc");

  // Read NPY directly: ahc_init_labels.npy is plain .npy (not npz).
  use npyz::{NpyFile, npz::NpzArchive};
  use std::{fs::File, io::BufReader};
  let labels_path = fixture(&format!("tests/parity/fixtures/{dir}/ahc_init_labels.npy"));
  // capture_intermediates also stores ahc_init_labels in clustering.npz / ahc_state.npz?
  // Try direct .npy first.
  let py_labels: Vec<i64> = if labels_path.exists() {
    let f = File::open(&labels_path).expect("open ahc labels");
    let npy = NpyFile::new(BufReader::new(f)).expect("npy parse");
    npy.into_vec().expect("decode")
  } else {
    panic!("ahc_init_labels.npy not found at {}", labels_path.display());
  };
  let py_labels: Vec<usize> = py_labels.iter().map(|&v| v as usize).collect();
  let _ = NpzArchive::<BufReader<File>>::new; // silence unused-import warning

  // Build co-occurrence: dia label x → pyannote label y.
  let max_dia = *dia_labels.iter().max().unwrap_or(&0);
  let max_py = *py_labels.iter().max().unwrap_or(&0);
  let nd = max_dia + 1;
  let np = max_py + 1;
  let mut cooc = vec![vec![0u64; np]; nd];
  for (d, p) in dia_labels.iter().zip(py_labels.iter()) {
    cooc[*d][*p] += 1;
  }
  // Per dia label, count distinct pyannote labels it co-occurs with.
  // If all rows have exactly one nonzero entry, dia's labels are a
  // permutation of pyannote's. If any row has ≥2 nonzero, partition
  // disagreement.
  let mut split_rows = 0usize;
  let mut max_split = 0usize;
  for row in &cooc {
    let nz = row.iter().filter(|&&v| v > 0).count();
    if nz > 1 {
      split_rows += 1;
      if nz > max_split {
        max_split = nz;
      }
    }
  }
  eprintln!(
    "[diag_ahc] dia={nd} clusters, pyannote={np} clusters; rows that span multiple pyannote labels: {split_rows} (max-split={max_split})"
  );
  let mut total = 0u64;
  for row in &cooc {
    for v in row {
      total += v;
    }
  }
  eprintln!("[diag_ahc] total assignments: {total}");
  if split_rows > 0 {
    // Show first few problematic dia labels with their pyannote
    // co-occurrence breakdown.
    let mut shown = 0usize;
    for (d, row) in cooc.iter().enumerate() {
      let nz: Vec<(usize, u64)> = row
        .iter()
        .enumerate()
        .filter(|&(_, &v)| v > 0)
        .map(|(i, &v)| (i, v))
        .collect();
      if nz.len() > 1 {
        eprintln!("  dia label {d} ↔ pyannote labels: {nz:?}");
        shown += 1;
        if shown >= 5 {
          break;
        }
      }
    }
  }
}

/// Verify dia's full assign_embeddings on 10 against captured
/// pyannote hard_clusters, dumping per-chunk discrepancies.
#[test]
#[ignore = "diagnostic; localizes per-chunk pipeline divergence on 10_mrbeast_clean_water"]
fn diagnose_pipeline_per_chunk_10_mrbeast() {
  use crate::{
    cluster::hungarian::UNMATCHED,
    pipeline::{AssignEmbeddingsInput, assign_embeddings},
  };
  use nalgebra::DVector;

  let dir = "10_mrbeast_clean_water";
  let raw_path = fixture(&format!("tests/parity/fixtures/{dir}/raw_embeddings.npz"));
  let (raw_flat_f32, raw_shape) = read_npz_array::<f32>(&raw_path, "embeddings");
  let num_chunks = raw_shape[0] as usize;
  let num_speakers = raw_shape[1] as usize;
  let embed_dim = raw_shape[2] as usize;
  let raw_flat: Vec<f64> = raw_flat_f32.iter().map(|&v| v as f64).collect();

  let seg_path = fixture(&format!("tests/parity/fixtures/{dir}/segmentations.npz"));
  let (seg_f32, seg_shape) = read_npz_array::<f32>(&seg_path, "segmentations");
  let num_frames = seg_shape[1] as usize;
  let seg_flat: Vec<f64> = seg_f32.iter().map(|&v| v as f64).collect();

  let plda_path = fixture(&format!("tests/parity/fixtures/{dir}/plda_embeddings.npz"));
  let (post_plda, post_plda_shape) = read_npz_array::<f64>(&plda_path, "post_plda");
  let plda_dim = post_plda_shape[1] as usize;
  let (phi_flat, _) = read_npz_array::<f64>(&plda_path, "phi");
  let phi = DVector::<f64>::from_vec(phi_flat);
  let (chunk_i64, _) = read_npz_array::<i64>(&plda_path, "train_chunk_idx");
  let (speaker_i64, _) = read_npz_array::<i64>(&plda_path, "train_speaker_idx");
  let train_chunk_idx: Vec<usize> = chunk_i64.iter().map(|&v| v as usize).collect();
  let train_speaker_idx: Vec<usize> = speaker_i64.iter().map(|&v| v as usize).collect();

  let ahc_path = fixture(&format!("tests/parity/fixtures/{dir}/ahc_state.npz"));
  let (thr_flat, _) = read_npz_array::<f64>(&ahc_path, "threshold");
  let vbx_path = fixture(&format!("tests/parity/fixtures/{dir}/vbx_state.npz"));
  let (fa, _) = read_npz_array::<f64>(&vbx_path, "fa");
  let (fb, _) = read_npz_array::<f64>(&vbx_path, "fb");
  let (mi, _) = read_npz_array::<i64>(&vbx_path, "max_iters");

  let input = AssignEmbeddingsInput::new(
    &raw_flat,
    embed_dim,
    num_chunks,
    num_speakers,
    &seg_flat,
    num_frames,
    &post_plda,
    plda_dim,
    &phi,
    &train_chunk_idx,
    &train_speaker_idx,
  )
  .with_threshold(thr_flat[0])
  .with_fa(fa[0])
  .with_fb(fb[0])
  .with_max_iters(mi[0] as usize);
  let dia_hard = assign_embeddings(&input).expect("assign_embeddings");

  let cluster_path = fixture(&format!("tests/parity/fixtures/{dir}/clustering.npz"));
  let (py_hard, _) = read_npz_array::<i8>(&cluster_path, "hard_clusters");

  // Find the FIRST partition disagreement, ignoring label permutation.
  let mut got_to_want: std::collections::HashMap<i32, i32> = Default::default();
  let mut want_to_got: std::collections::HashMap<i32, i32> = Default::default();
  let mut shown = 0usize;
  // First pass: build provisional permutation from chunks 0..num_chunks.
  // Use co-occurrence counting (Hungarian-equivalent on cluster
  // labels) to find the best label mapping, then count exact mismatches.
  let mut cooc = vec![vec![0i64; 8]; 8]; // cooc[got][want]
  for c in 0..num_chunks {
    for sp in 0..num_speakers {
      let g = dia_hard[c][sp];
      let w = py_hard[c * num_speakers + sp] as i32;
      if g == UNMATCHED || w < 0 {
        continue;
      }
      cooc[g as usize][w as usize] += 1;
    }
  }
  eprintln!("[diag_chunk] co-occurrence matrix (got→want):");
  for g in 0..8usize {
    let mut s = format!("  got={g}: ");
    let mut empty = true;
    for w in 0..8usize {
      if cooc[g][w] > 0 {
        s.push_str(&format!("[{w}={}]", cooc[g][w]));
        empty = false;
      }
    }
    if !empty {
      eprintln!("{s}");
    }
  }
  for c in 0..num_chunks {
    for sp in 0..num_speakers {
      let g = dia_hard[c][sp];
      let w = py_hard[c * num_speakers + sp] as i32;
      if g == UNMATCHED || w < 0 {
        continue;
      }
      let g_ok = match got_to_want.get(&g).copied() {
        Some(existing) => existing == w,
        None => {
          got_to_want.insert(g, w);
          true
        }
      };
      let w_ok = match want_to_got.get(&w).copied() {
        Some(existing) => existing == g,
        None => {
          want_to_got.insert(w, g);
          true
        }
      };
      if !(g_ok && w_ok) {
        shown += 1;
        if shown <= 10 {
          let dia_chunk: Vec<i32> = (0..num_speakers).map(|x| dia_hard[c][x]).collect();
          let py_chunk: Vec<i32> = (0..num_speakers)
            .map(|x| py_hard[c * num_speakers + x] as i32)
            .collect();
          eprintln!(
            "[diag_chunk] mismatch chunk {c} speaker {sp}: dia={dia_chunk:?} pyannote={py_chunk:?}"
          );
        }
      }
    }
  }
  eprintln!("[diag_chunk] total partition disagreements: {shown}");
}

/// Tight test: feed pyannote's captured soft_clusters (already
/// inactive-masked) directly into dia's `constrained_argmax` and
/// compare to pyannote's captured `hard_clusters`. Earlier stages
/// (centroids, soft_clusters on active pairs) match bit-exactly per
/// the diagnostic test below — so a mismatch here isolates dia's
/// Hungarian assignment (`crate::cluster::hungarian::lsap`, an
/// in-tree port of SciPy's `rectangular_lsap.cpp`) against scipy's
/// `scipy.optimize.linear_sum_assignment` reference. With the LSAP
/// port replacing the prior `pathfinding::kuhn_munkres` adapter, the
/// tie-breaking is matched bit-for-bit, so this test pins the
/// integration boundary rather than just the optimal-weight contract.
#[test]
#[ignore = "isolates Hungarian tie-breaking divergence using captured 10_mrbeast_clean_water soft_clusters"]
fn hungarian_only_parity_10_mrbeast() {
  use crate::cluster::hungarian::{UNMATCHED, constrained_argmax};
  use nalgebra::DMatrix;

  let dir = "10_mrbeast_clean_water";
  let cluster_path = fixture(&format!("tests/parity/fixtures/{dir}/clustering.npz"));
  let (soft_flat, soft_shape) = read_npz_array::<f64>(&cluster_path, "soft_clusters");
  assert_eq!(soft_shape.len(), 3);
  let num_chunks = soft_shape[0] as usize;
  let num_speakers = soft_shape[1] as usize;
  let num_clusters = soft_shape[2] as usize;
  let (py_hard, _) = read_npz_array::<i8>(&cluster_path, "hard_clusters");

  // Pack chunks as (num_speakers, num_clusters) DMatrix per
  // `constrained_argmax`'s contract.
  let chunks: Vec<DMatrix<f64>> = (0..num_chunks)
    .map(|c| {
      let mut m = DMatrix::<f64>::zeros(num_speakers, num_clusters);
      for sp in 0..num_speakers {
        for k in 0..num_clusters {
          m[(sp, k)] = soft_flat[((c * num_speakers) + sp) * num_clusters + k];
        }
      }
      m
    })
    .collect();
  let dia_hard = constrained_argmax(&chunks).expect("constrained_argmax");

  // Per pyannote: inactive-(chunk, speaker) pairs are pre-masked with
  // `soft.min() - 1.0`, so Hungarian assigns them too — but pyannote
  // then overwrites them with -2 (UNMATCHED). dia's
  // `constrained_argmax` doesn't apply that overwrite (the pipeline
  // does it at stage 7). For an apples-to-apples Hungarian-only
  // comparison, accept dia's `dia_hard[c][sp] != UNMATCHED` paired
  // with `py_hard[c][sp] >= 0`, even when py_hard has the -2 mark
  // applied (those are inactive pairs we don't need to compare).
  let mut got_to_want: std::collections::HashMap<i32, i32> = Default::default();
  let mut want_to_got: std::collections::HashMap<i32, i32> = Default::default();
  let mut mismatches = 0usize;
  for c in 0..num_chunks {
    for sp in 0..num_speakers {
      let g = dia_hard[c][sp];
      let w = py_hard[c * num_speakers + sp] as i32;
      if w < 0 || g == UNMATCHED {
        continue;
      }
      // Build the partition mapping; report how many chunks would
      // violate the one-to-one mapping if we asserted strictly.
      let g_ok = match got_to_want.get(&g).copied() {
        Some(existing) => existing == w,
        None => {
          got_to_want.insert(g, w);
          true
        }
      };
      let w_ok = match want_to_got.get(&w).copied() {
        Some(existing) => existing == g,
        None => {
          want_to_got.insert(w, g);
          true
        }
      };
      if !(g_ok && w_ok) {
        mismatches += 1;
        if mismatches <= 3 {
          eprintln!("[hung_diag] mismatch at chunk {c} speaker {sp}: dia={g} pyannote={w}");
        }
      }
    }
  }
  eprintln!(
    "[hung_diag] {dir}: {num_chunks} chunks × {num_speakers} speakers, partition mismatches = {mismatches}"
  );
  assert_eq!(
    mismatches, 0,
    "Hungarian tie-breaking diverged from scipy in {mismatches} chunks — \
     `crate::cluster::hungarian::lsap` is meant to be a bit-for-bit port \
     of `scipy.optimize.linear_sum_assignment` (Crouse / LAPJV). A \
     mismatch here points to a regression in the LSAP traversal/augment \
     order, not to the documented historical `pathfinding::kuhn_munkres` \
     tie-break gap (that solver was retired)."
  );
}

/// Walk through assign_embeddings stage-by-stage on the
/// `10_mrbeast_clean_water` capture and report where dia first
/// diverges from pyannote. Stages compared: centroids (after
/// weighted_centroids), soft_clusters (after cosine cdist), and
/// final hard_clusters (after Hungarian + masking). VBx parity is
/// verified separately in `cluster::vbx::parity_tests`.
#[test]
#[ignore = "diagnostic; requires the 10_mrbeast_clean_water capture under tests/parity/fixtures/"]
fn diagnose_pipeline_divergence_10_mrbeast() {
  use crate::cluster::{
    centroid::{SP_ALIVE_THRESHOLD, weighted_centroids},
    vbx::vbx_iterate,
  };
  use nalgebra::{DMatrix, DMatrixView, DVector};

  let dir = "10_mrbeast_clean_water";
  // Inputs.
  let plda_path = fixture(&format!("tests/parity/fixtures/{dir}/plda_embeddings.npz"));
  let (post_plda_flat, post_plda_shape) = read_npz_array::<f64>(&plda_path, "post_plda");
  let num_train = post_plda_shape[0] as usize;
  let plda_dim = post_plda_shape[1] as usize;
  let (phi_flat, _) = read_npz_array::<f64>(&plda_path, "phi");
  let phi = DVector::<f64>::from_vec(phi_flat);

  // VBx: re-run with captured qinit + hyperparameters.
  let vbx_path = fixture(&format!("tests/parity/fixtures/{dir}/vbx_state.npz"));
  let (qinit_flat, qinit_shape) = read_npz_array::<f64>(&vbx_path, "qinit");
  let s = qinit_shape[1] as usize;
  let qinit = DMatrix::<f64>::from_row_slice(num_train, s, &qinit_flat);
  let (fa, _) = read_npz_array::<f64>(&vbx_path, "fa");
  let (fb, _) = read_npz_array::<f64>(&vbx_path, "fb");
  let (mi, _) = read_npz_array::<i64>(&vbx_path, "max_iters");
  // post_plda needs column-major layout for vbx_iterate's DMatrixView.
  let post_plda_rm = DMatrix::<f64>::from_row_slice(num_train, plda_dim, &post_plda_flat);
  let post_plda_cm = post_plda_rm.clone();
  let post_plda_view = DMatrixView::from(&post_plda_cm);
  let vbx_out =
    vbx_iterate(post_plda_view, &phi, &qinit, fa[0], fb[0], mi[0] as usize).expect("vbx");

  // train_embeddings extraction (raw 256-d xvec).
  let raw_path = fixture(&format!("tests/parity/fixtures/{dir}/raw_embeddings.npz"));
  let (raw_flat_f32, raw_shape) = read_npz_array::<f32>(&raw_path, "embeddings");
  let num_chunks = raw_shape[0] as usize;
  let num_speakers = raw_shape[1] as usize;
  let embed_dim = raw_shape[2] as usize;
  let raw_flat: Vec<f64> = raw_flat_f32.iter().map(|&v| v as f64).collect();

  let (chunk_idx, _) = read_npz_array::<i64>(&plda_path, "train_chunk_idx");
  let (speaker_idx, _) = read_npz_array::<i64>(&plda_path, "train_speaker_idx");
  assert_eq!(chunk_idx.len(), num_train);
  let mut train_emb = vec![0.0_f64; num_train * embed_dim];
  for i in 0..num_train {
    let c = chunk_idx[i] as usize;
    let sp_idx = speaker_idx[i] as usize;
    let src = (c * num_speakers + sp_idx) * embed_dim;
    let dst = i * embed_dim;
    train_emb[dst..dst + embed_dim].copy_from_slice(&raw_flat[src..src + embed_dim]);
  }

  // Stage 5: dia's centroids via weighted_centroids.
  let dia_centroids = weighted_centroids(
    vbx_out.gamma(),
    vbx_out.pi(),
    &train_emb,
    num_train,
    embed_dim,
    SP_ALIVE_THRESHOLD,
  )
  .expect("centroids");
  let num_alive = dia_centroids.nrows();

  // Pyannote's captured centroids.
  let cluster_path = fixture(&format!("tests/parity/fixtures/{dir}/clustering.npz"));
  let (py_centroids_flat, py_centroids_shape) = read_npz_array::<f64>(&cluster_path, "centroids");
  assert_eq!(py_centroids_shape[1] as usize, embed_dim);
  let py_num_clusters = py_centroids_shape[0] as usize;
  eprintln!("[diag] num_alive: dia={num_alive} pyannote={py_num_clusters}");

  if num_alive == py_num_clusters {
    // Try to find a 1-to-1 row matching by min-distance per row, then
    // report max element-wise error.
    let mut best_perm = vec![usize::MAX; num_alive];
    let mut used = vec![false; py_num_clusters];
    for k in 0..num_alive {
      let mut best = (f64::INFINITY, usize::MAX);
      for j in 0..py_num_clusters {
        if used[j] {
          continue;
        }
        let mut dsq = 0.0;
        for d in 0..embed_dim {
          let diff = dia_centroids[(k, d)] - py_centroids_flat[j * embed_dim + d];
          dsq += diff * diff;
        }
        if dsq < best.0 {
          best = (dsq, j);
        }
      }
      best_perm[k] = best.1;
      used[best.1] = true;
    }
    let mut max_err: f64 = 0.0;
    for k in 0..num_alive {
      let j = best_perm[k];
      for d in 0..embed_dim {
        let err = (dia_centroids[(k, d)] - py_centroids_flat[j * embed_dim + d]).abs();
        if err > max_err {
          max_err = err;
        }
      }
    }
    eprintln!("[diag] centroid max_abs_err (best perm) = {max_err:.3e}");
    // Also report the perm itself and the *identity* (no-perm) error.
    eprintln!("[diag] best_perm: dia[k] -> pyannote[best_perm[k]] = {best_perm:?}");
    let mut id_max_err: f64 = 0.0;
    for k in 0..num_alive {
      for d in 0..embed_dim {
        let err = (dia_centroids[(k, d)] - py_centroids_flat[k * embed_dim + d]).abs();
        if err > id_max_err {
          id_max_err = err;
        }
      }
    }
    eprintln!("[diag] centroid max_abs_err (identity, no perm) = {id_max_err:.3e}");
  }

  // Pyannote captured soft_clusters and hard_clusters.
  let (py_soft, py_soft_shape) = read_npz_array::<f64>(&cluster_path, "soft_clusters");
  let (py_hard, _) = read_npz_array::<i8>(&cluster_path, "hard_clusters");
  eprintln!("[diag] soft_clusters shape: {:?}", py_soft_shape);
  // Compute dia's soft_clusters [num_chunks][num_speakers, num_alive] like
  // stage 6 of assign_embeddings, then summarize element-wise error.
  let mut dia_soft = vec![vec![0.0_f64; num_speakers * num_alive]; num_chunks];
  for c in 0..num_chunks {
    for sp in 0..num_speakers {
      let row = c * num_speakers + sp;
      let emb_row = &raw_flat[row * embed_dim..(row + 1) * embed_dim];
      let emb_norm_sq = crate::ops::scalar::dot(emb_row, emb_row);
      for k in 0..num_alive {
        let mut centroid_row = vec![0.0_f64; embed_dim];
        for d in 0..embed_dim {
          centroid_row[d] = dia_centroids[(k, d)];
        }
        let cn_norm_sq = crate::ops::scalar::dot(&centroid_row, &centroid_row);
        // Replicate `crate::pipeline::algo::cosine_distance_pre_norm`
        // **exactly**: `sqrt(a) * sqrt(b)` denom, no clamp on the
        // ratio. Earlier versions of this diagnostic used
        // `sqrt(a*b)` + clamp — both are mathematically the cosine
        // distance but the f64 results round at different bit
        // boundaries, and the diagnostic must match dia's pipeline
        // bit-for-bit for the comparison to be meaningful.
        let dot = crate::ops::scalar::dot(emb_row, &centroid_row);
        let denom = emb_norm_sq.sqrt() * cn_norm_sq.sqrt();
        let dist = if denom == 0.0 {
          f64::NAN
        } else {
          1.0 - dot / denom
        };
        dia_soft[c][sp * num_alive + k] = 2.0 - dist;
      }
    }
  }
  // Compare to pyannote's soft_clusters via best-row-permutation.
  if num_alive == py_num_clusters {
    let mut best_perm = vec![0usize; num_alive];
    let mut used = vec![false; py_num_clusters];
    for k in 0..num_alive {
      let mut best = (f64::INFINITY, 0usize);
      for j in 0..py_num_clusters {
        if used[j] {
          continue;
        }
        let mut dsq = 0.0;
        for d in 0..embed_dim {
          let diff = dia_centroids[(k, d)] - py_centroids_flat[j * embed_dim + d];
          dsq += diff * diff;
        }
        if dsq < best.0 {
          best = (dsq, j);
        }
      }
      best_perm[k] = best.1;
      used[best.1] = true;
    }
    // Pyannote's captured soft_clusters has the inactive-(chunk,
    // speaker) mask applied (`soft[seg.sum(1)==0] = soft.min()-1.0`),
    // so any pair whose segmentation column sums to 0 in the captured
    // segmentations is replaced by the constant. dia's pre-mask soft
    // values would diverge there by design. Restrict the comparison
    // to active pairs (sum > 0) to expose only real centroid/cdist
    // numerical drift.
    let seg_path = fixture(&format!("tests/parity/fixtures/{dir}/segmentations.npz"));
    let (seg_flat_f32, seg_shape) = read_npz_array::<f32>(&seg_path, "segmentations");
    let seg_chunks = seg_shape[0] as usize;
    let seg_frames = seg_shape[1] as usize;
    let seg_speakers = seg_shape[2] as usize;
    assert_eq!(seg_chunks, num_chunks);
    assert_eq!(seg_speakers, num_speakers);
    let mut max_soft_err: f64 = 0.0;
    let mut max_loc = (0, 0, 0);
    let mut compared_pairs = 0usize;
    let mut total_pairs = 0usize;
    for c in 0..num_chunks {
      for sp in 0..num_speakers {
        total_pairs += 1;
        // sum_activity for (c, sp).
        let mut sum_a = 0.0_f64;
        for f in 0..seg_frames {
          sum_a += seg_flat_f32[(c * seg_frames + f) * seg_speakers + sp] as f64;
        }
        if sum_a == 0.0 {
          continue;
        }
        compared_pairs += 1;
        for k in 0..num_alive {
          let py_k = best_perm[k];
          let dia_v = dia_soft[c][sp * num_alive + k];
          let py_v = py_soft[((c * num_speakers) + sp) * py_num_clusters + py_k];
          let err = (dia_v - py_v).abs();
          if err > max_soft_err {
            max_soft_err = err;
            max_loc = (c, sp, k);
          }
        }
      }
    }
    eprintln!(
      "[diag] soft_clusters max_abs_err on ACTIVE pairs ({compared_pairs}/{total_pairs}) = \
       {max_soft_err:.3e} at (c={}, sp={}, k={})",
      max_loc.0, max_loc.1, max_loc.2
    );
  }
  // Always emit pyannote-side counts so we know whether speaker counts
  // are aligned even when partitioning differs.
  let mut py_unique = std::collections::BTreeSet::new();
  for v in &py_hard {
    if *v >= 0 {
      py_unique.insert(*v);
    }
  }
  eprintln!("[diag] pyannote: hard_clusters unique = {:?}", py_unique);

  // Final stage: emulate dia's full stage 7 (mask + Hungarian) on the
  // diagnostic-computed dia_soft, and compare hard_clusters to
  // pyannote's. This catches a divergence in soft_min / inactive_const
  // computation or the mask application (vs the Hungarian-only test
  // which fed pyannote's already-masked soft).
  if num_alive == py_num_clusters {
    use crate::cluster::hungarian::{UNMATCHED, constrained_argmax};
    use nalgebra::DMatrix;
    // Compute dia's soft_min over all dia_soft entries.
    let mut soft_min = f64::INFINITY;
    for c in 0..num_chunks {
      for sp in 0..num_speakers {
        for k in 0..num_alive {
          let v = dia_soft[c][sp * num_alive + k];
          if v < soft_min {
            soft_min = v;
          }
        }
      }
    }
    let inactive_const = soft_min - 1.0;
    eprintln!("[diag] dia soft_min = {soft_min:.10} inactive_const = {inactive_const:.10}");

    // Apply mask (per dia stage 7).
    let seg_path = fixture(&format!("tests/parity/fixtures/{dir}/segmentations.npz"));
    let (seg_flat_f32, seg_shape) = read_npz_array::<f32>(&seg_path, "segmentations");
    let seg_frames = seg_shape[1] as usize;
    for c in 0..num_chunks {
      for sp in 0..num_speakers {
        let mut sum_a = 0.0_f64;
        for f in 0..seg_frames {
          sum_a += seg_flat_f32[(c * seg_frames + f) * num_speakers + sp] as f64;
        }
        if sum_a == 0.0 {
          for k in 0..num_alive {
            dia_soft[c][sp * num_alive + k] = inactive_const;
          }
        }
      }
    }

    // Build chunks as DMatrix(num_speakers, num_alive) and call dia's Hungarian.
    let chunks: Vec<DMatrix<f64>> = (0..num_chunks)
      .map(|c| {
        let mut m = DMatrix::<f64>::zeros(num_speakers, num_alive);
        for sp in 0..num_speakers {
          for k in 0..num_alive {
            m[(sp, k)] = dia_soft[c][sp * num_alive + k];
          }
        }
        m
      })
      .collect();
    let dia_hard = constrained_argmax(&chunks).expect("constrained_argmax");

    // Compare to pyannote's hard_clusters.
    let mut got_to_want: std::collections::HashMap<i32, i32> = Default::default();
    let mut want_to_got: std::collections::HashMap<i32, i32> = Default::default();
    let mut shown = 0usize;
    for c in 0..num_chunks {
      for sp in 0..num_speakers {
        let g = dia_hard[c][sp];
        let w = py_hard[c * num_speakers + sp] as i32;
        if g == UNMATCHED || w < 0 {
          continue;
        }
        let g_ok = match got_to_want.get(&g).copied() {
          Some(existing) => existing == w,
          None => {
            got_to_want.insert(g, w);
            true
          }
        };
        let w_ok = match want_to_got.get(&w).copied() {
          Some(existing) => existing == g,
          None => {
            want_to_got.insert(w, g);
            true
          }
        };
        if !(g_ok && w_ok) {
          shown += 1;
          if shown <= 3 {
            eprintln!("[diag] full-flow mismatch chunk {c} speaker {sp}: dia={g} pyannote={w}");
          }
        }
      }
    }
    eprintln!("[diag] full-flow partition mismatches: {shown}");
  }
}

fn run_pipeline_parity(fixture_dir: &str) {
  crate::parity_fixtures_or_skip!();
  require_fixtures(fixture_dir);

  let base = format!("tests/parity/fixtures/{fixture_dir}");
  // Raw embeddings (chunks, speakers, embed_dim).
  let raw_path = fixture(&format!("{base}/raw_embeddings.npz"));
  let (raw_flat, raw_shape) = read_npz_array::<f32>(&raw_path, "embeddings");
  assert_eq!(raw_shape.len(), 3);
  let num_chunks = raw_shape[0] as usize;
  let num_speakers = raw_shape[1] as usize;
  let embed_dim = raw_shape[2] as usize;
  // Row-major flat `[c][s][d]`, matching the new
  // `AssignEmbeddingsInput::embeddings: &[f64]` contract.
  let embeddings: Vec<f64> = raw_flat.iter().map(|&v| v as f64).collect();

  // Segmentations (chunks, frames, speakers).
  let seg_path = fixture(&format!("{base}/segmentations.npz"));
  let (seg_flat_f32, seg_shape) = read_npz_array::<f32>(&seg_path, "segmentations");
  assert_eq!(seg_shape.len(), 3);
  let num_frames = seg_shape[1] as usize;
  assert_eq!(seg_shape[0] as usize, num_chunks);
  assert_eq!(seg_shape[2] as usize, num_speakers);
  let segmentations: Vec<f64> = seg_flat_f32.iter().map(|&v| v as f64).collect();

  // post_plda + phi + train_*idx (pre-filtered, pre-projected).
  // The .npz array is row-major (numpy C-order by default), which
  // matches the `AssignEmbeddingsInput::post_plda: &[f64]` row-major
  // contract directly — no layout adapter needed. The pipeline
  // transposes into column-major for VBx's GEMM internally.
  let plda_path = fixture(&format!("{base}/plda_embeddings.npz"));
  let (post_plda_flat, post_plda_shape) = read_npz_array::<f64>(&plda_path, "post_plda");
  assert_eq!(post_plda_shape.len(), 2);
  let num_train = post_plda_shape[0] as usize;
  let plda_dim = post_plda_shape[1] as usize;
  let post_plda: &[f64] = &post_plda_flat;

  let (phi_flat, phi_shape) = read_npz_array::<f64>(&plda_path, "phi");
  assert_eq!(phi_shape, vec![plda_dim as u64]);
  let phi = DVector::<f64>::from_vec(phi_flat);

  let (chunk_idx_i64, _) = read_npz_array::<i64>(&plda_path, "train_chunk_idx");
  let (speaker_idx_i64, _) = read_npz_array::<i64>(&plda_path, "train_speaker_idx");
  assert_eq!(chunk_idx_i64.len(), num_train);
  assert_eq!(speaker_idx_i64.len(), num_train);
  let train_chunk_idx: Vec<usize> = chunk_idx_i64.iter().map(|&v| v as usize).collect();
  let train_speaker_idx: Vec<usize> = speaker_idx_i64.iter().map(|&v| v as usize).collect();

  // Hyperparameters.
  let ahc_path = fixture(&format!("{base}/ahc_state.npz"));
  let (threshold_data, _) = read_npz_array::<f64>(&ahc_path, "threshold");
  let threshold = threshold_data[0];

  let vbx_path = fixture(&format!("{base}/vbx_state.npz"));
  let (fa_arr, _) = read_npz_array::<f64>(&vbx_path, "fa");
  let (fb_arr, _) = read_npz_array::<f64>(&vbx_path, "fb");
  let (max_iters_arr, _) = read_npz_array::<i64>(&vbx_path, "max_iters");
  let fa = fa_arr[0];
  let fb = fb_arr[0];
  let max_iters = max_iters_arr[0] as usize;

  // Run the port.
  let input = AssignEmbeddingsInput::new(
    &embeddings,
    embed_dim,
    num_chunks,
    num_speakers,
    &segmentations,
    num_frames,
    post_plda,
    plda_dim,
    &phi,
    &train_chunk_idx,
    &train_speaker_idx,
  )
  .with_threshold(threshold)
  .with_fa(fa)
  .with_fb(fb)
  .with_max_iters(max_iters);
  let got = assign_embeddings(&input).expect("assign_embeddings");

  // Captured ground truth.
  let cluster_path = fixture(&format!("{base}/clustering.npz"));
  let (hard_flat_i8, hard_shape) = read_npz_array::<i8>(&cluster_path, "hard_clusters");
  assert_eq!(hard_shape, vec![num_chunks as u64, num_speakers as u64]);

  // Build the captured per-chunk vectors.
  let want: Vec<Vec<i32>> = (0..num_chunks)
    .map(|c| {
      (0..num_speakers)
        .map(|s| hard_flat_i8[c * num_speakers + s] as i32)
        .collect()
    })
    .collect();

  // Compare: partition-equivalent per chunk. The captured labels use
  // scipy's fcluster traversal order; ours use kodama's order remapped
  // through encounter sort. Both produce valid clusterings of the same
  // partition; the integer labels themselves are arbitrary names. We
  // build a global cluster-id permutation by walking chunks and
  // accumulating "got_label X co-occurs with want_label Y" (and vice
  // versa); a consistent partition equivalence requires both maps to
  // be one-to-one across all chunks.
  use std::collections::HashMap;
  let mut got_to_want: HashMap<i32, i32> = HashMap::new();
  let mut want_to_got: HashMap<i32, i32> = HashMap::new();
  for c in 0..num_chunks {
    for s in 0..num_speakers {
      let g = got[c][s];
      let w = want[c][s];
      // UNMATCHED on both sides is consistent.
      if g == UNMATCHED && w == UNMATCHED {
        continue;
      }
      // UNMATCHED only on one side → partition mismatch.
      if g == UNMATCHED || w == UNMATCHED {
        panic!("UNMATCHED mismatch at chunk {c}, speaker {s}: got {g}, want {w}");
      }
      // Establish or verify the consistent permutation.
      match got_to_want.get(&g).copied() {
        Some(existing) => assert_eq!(
          existing, w,
          "partition mismatch at chunk {c}, speaker {s}: got {g} previously mapped to {existing}, now {w}"
        ),
        None => {
          got_to_want.insert(g, w);
        }
      }
      match want_to_got.get(&w).copied() {
        Some(existing) => assert_eq!(
          existing, g,
          "partition mismatch at chunk {c}, speaker {s}: want {w} previously mapped from {existing}, now {g}"
        ),
        None => {
          want_to_got.insert(w, g);
        }
      }
    }
  }
  eprintln!(
    "[parity_pipeline] {} chunks × {} speakers — partition matches pyannote (cluster mapping: {:?})",
    num_chunks, num_speakers, got_to_want
  );
}
