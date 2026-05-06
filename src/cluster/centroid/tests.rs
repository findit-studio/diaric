//! Model-free unit tests for `diarization::cluster::centroid`.
//!
//! Heavy parity against pyannote's captured `centroids` lives in
//! `src/centroid/parity_tests.rs`.

use crate::cluster::centroid::{Error, SP_ALIVE_THRESHOLD, weighted_centroids};
use nalgebra::{DMatrix, DVector};

/// Test helper: convert a column-major `DMatrix<f64>` to a row-major
/// `(Vec<f64>, num_rows, num_cols)` triple matching the new
/// `weighted_centroids` signature. Old tests that constructed `DMatrix`
/// for convenience can use this adapter rather than being rewritten in
/// row-major flat form.
fn dm_to_row_major(m: &DMatrix<f64>) -> (Vec<f64>, usize, usize) {
  let (n, d) = m.shape();
  let mut out = Vec::with_capacity(n * d);
  for r in 0..n {
    for c in 0..d {
      out.push(m[(r, c)]);
    }
  }
  (out, n, d)
}

/// Convenience wrapper: `weighted_centroids` from a `&DMatrix<f64>`
/// embedding matrix for tests.
fn weighted_centroids_dm(
  q: &DMatrix<f64>,
  sp: &DVector<f64>,
  emb: &DMatrix<f64>,
  sp_threshold: f64,
) -> Result<DMatrix<f64>, Error> {
  let (data, n, d) = dm_to_row_major(emb);
  weighted_centroids(q, sp, &data, n, d, sp_threshold)
}

#[test]
fn rejects_empty_q() {
  let q = DMatrix::<f64>::zeros(0, 2);
  let sp = DVector::<f64>::from_vec(vec![1.0, 0.0]);
  let emb = DMatrix::<f64>::zeros(0, 4);
  assert!(matches!(
    weighted_centroids_dm(&q, &sp, &emb, SP_ALIVE_THRESHOLD),
    Err(Error::Shape(_))
  ));
}

#[test]
fn rejects_sp_q_dim_mismatch() {
  let q = DMatrix::<f64>::from_element(3, 2, 0.5);
  let sp = DVector::<f64>::from_vec(vec![1.0]); // length 1, not 2
  let emb = DMatrix::<f64>::from_element(3, 4, 1.0);
  assert!(matches!(
    weighted_centroids_dm(&q, &sp, &emb, SP_ALIVE_THRESHOLD),
    Err(Error::Shape(_))
  ));
}

#[test]
fn rejects_q_emb_row_mismatch() {
  let q = DMatrix::<f64>::from_element(3, 2, 0.5);
  let sp = DVector::<f64>::from_vec(vec![1.0, 1.0]);
  let emb = DMatrix::<f64>::from_element(4, 4, 1.0); // 4 rows, q has 3
  assert!(matches!(
    weighted_centroids_dm(&q, &sp, &emb, SP_ALIVE_THRESHOLD),
    Err(Error::Shape(_))
  ));
}

#[test]
fn rejects_no_surviving_clusters() {
  // Both sp values are well below the guard band lower bound
  // (`SP_ALIVE_THRESHOLD * 0.5 = 5e-8`), so the function reaches the
  // "no surviving clusters" path rather than firing the guard.
  let q = DMatrix::<f64>::from_element(3, 2, 0.5);
  let sp = DVector::<f64>::from_vec(vec![1.0e-12, 1.0e-13]);
  let emb = DMatrix::<f64>::from_element(3, 4, 1.0);
  assert!(matches!(
    weighted_centroids_dm(&q, &sp, &emb, SP_ALIVE_THRESHOLD),
    Err(Error::Shape(_))
  ));
}

/// VBx reductions are SIMD on x86, so a
/// `sp` value landing within ulp drift of `SP_ALIVE_THRESHOLD` would
/// flip the alive/squashed decision across CPU backends. The guard
/// band `(threshold * 0.5, threshold * 2)` rejects those inputs
/// with `Error::AmbiguousAliveCluster`. This test constructs `sp`
/// values inside the band on each side of the threshold and verifies
/// the error fires before any SIMD-dependent decision is made.
#[test]
fn rejects_sp_in_simd_guard_band_above_threshold() {
  let q = DMatrix::<f64>::from_element(3, 2, 0.5);
  // 1.5e-7 is between threshold (1e-7) and the upper guard bound (2e-7).
  let sp = DVector::<f64>::from_vec(vec![1.5e-7, 0.99]);
  let emb = DMatrix::<f64>::from_element(3, 4, 1.0);
  let err =
    weighted_centroids_dm(&q, &sp, &emb, SP_ALIVE_THRESHOLD).expect_err("guard band must reject");
  assert!(
    matches!(err, Error::AmbiguousAliveCluster { cluster: 0, .. }),
    "got unexpected error: {err:?}"
  );
}

#[test]
fn rejects_sp_in_simd_guard_band_below_threshold() {
  let q = DMatrix::<f64>::from_element(3, 2, 0.5);
  // 7e-8 is between the lower guard bound (5e-8) and threshold (1e-7).
  let sp = DVector::<f64>::from_vec(vec![0.99, 7.0e-8]);
  let emb = DMatrix::<f64>::from_element(3, 4, 1.0);
  let err =
    weighted_centroids_dm(&q, &sp, &emb, SP_ALIVE_THRESHOLD).expect_err("guard band must reject");
  assert!(
    matches!(err, Error::AmbiguousAliveCluster { cluster: 1, .. }),
    "got unexpected error: {err:?}"
  );
}

/// Pyannote-valid priors that are clearly alive (≥ 2× threshold) or
/// clearly squashed (≤ 0.5× threshold) must NOT trigger the guard.
/// The previous 100× band rejected legitimate sub-O(1) priors like
/// 5e-7, breaking diarization for short-lived but real speakers.
#[test]
fn accepts_sp_clearly_alive_above_2x_threshold() {
  // 5e-7 is 5× threshold — clearly alive in pyannote, must pass.
  let q = DMatrix::<f64>::from_row_slice(3, 2, &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
  let sp = DVector::<f64>::from_vec(vec![5.0e-7, 1.0e-14]);
  let emb = DMatrix::<f64>::from_row_slice(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
  weighted_centroids_dm(&q, &sp, &emb, SP_ALIVE_THRESHOLD)
    .expect("clearly-alive 5e-7 must not fire the guard");
}

#[test]
fn accepts_sp_clearly_squashed_below_half_threshold() {
  // 1e-8 is 0.1× threshold — clearly squashed, must pass (other cluster
  // alive so we still produce centroids).
  let q = DMatrix::<f64>::from_row_slice(3, 2, &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
  let sp = DVector::<f64>::from_vec(vec![0.99, 1.0e-8]);
  let emb = DMatrix::<f64>::from_row_slice(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
  weighted_centroids_dm(&q, &sp, &emb, SP_ALIVE_THRESHOLD)
    .expect("clearly-squashed 1e-8 must not fire the guard");
}

#[test]
fn accepts_sp_at_band_boundary_2x_threshold() {
  // 2e-7 is exactly at the upper bound (exclusive); must pass.
  let q = DMatrix::<f64>::from_row_slice(3, 2, &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
  let sp = DVector::<f64>::from_vec(vec![2.0e-7, 1.0e-14]);
  let emb = DMatrix::<f64>::from_row_slice(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
  weighted_centroids_dm(&q, &sp, &emb, SP_ALIVE_THRESHOLD)
    .expect("boundary 2e-7 must not fire the guard");
}

#[test]
fn accepts_sp_at_band_boundary_half_threshold() {
  // 5e-8 is exactly at the lower bound (exclusive); must pass.
  let q = DMatrix::<f64>::from_row_slice(3, 2, &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
  let sp = DVector::<f64>::from_vec(vec![0.99, 5.0e-8]);
  let emb = DMatrix::<f64>::from_row_slice(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
  weighted_centroids_dm(&q, &sp, &emb, SP_ALIVE_THRESHOLD)
    .expect("boundary 5e-8 must not fire the guard");
}

/// `sp` exactly at threshold lands inside the band — guarded.
#[test]
fn rejects_sp_exactly_at_threshold() {
  let q = DMatrix::<f64>::from_element(3, 2, 0.5);
  let sp = DVector::<f64>::from_vec(vec![SP_ALIVE_THRESHOLD, 0.99]);
  let emb = DMatrix::<f64>::from_element(3, 4, 1.0);
  let err =
    weighted_centroids_dm(&q, &sp, &emb, SP_ALIVE_THRESHOLD).expect_err("guard band must reject");
  assert!(
    matches!(err, Error::AmbiguousAliveCluster { cluster: 0, .. }),
    "got unexpected error: {err:?}"
  );
}

/// Captured-fixture-shaped `sp`: alive ≈ 0.85, squashed ≈ 1.76e-14.
/// Both values are far outside the guard band; the function must
/// proceed normally.
#[test]
fn accepts_sp_well_outside_guard_band() {
  let q = DMatrix::<f64>::from_row_slice(3, 2, &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
  let sp = DVector::<f64>::from_vec(vec![0.85, 1.76e-14]);
  let emb = DMatrix::<f64>::from_row_slice(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
  let c = weighted_centroids_dm(&q, &sp, &emb, SP_ALIVE_THRESHOLD)
    .expect("realistic captured-fixture sp must pass");
  assert_eq!(c.shape(), (1, 2));
}

#[test]
fn rejects_non_finite_q() {
  let mut q = DMatrix::<f64>::from_element(3, 2, 0.5);
  q[(0, 0)] = f64::NAN;
  let sp = DVector::<f64>::from_vec(vec![1.0, 0.0]);
  let emb = DMatrix::<f64>::from_element(3, 4, 1.0);
  assert!(matches!(
    weighted_centroids_dm(&q, &sp, &emb, SP_ALIVE_THRESHOLD),
    Err(Error::NonFinite(_))
  ));
}

#[test]
fn rejects_non_finite_sp() {
  let q = DMatrix::<f64>::from_element(3, 2, 0.5);
  let sp = DVector::<f64>::from_vec(vec![1.0, f64::INFINITY]);
  let emb = DMatrix::<f64>::from_element(3, 4, 1.0);
  assert!(matches!(
    weighted_centroids_dm(&q, &sp, &emb, SP_ALIVE_THRESHOLD),
    Err(Error::NonFinite(_))
  ));
}

#[test]
fn rejects_non_finite_embeddings() {
  let q = DMatrix::<f64>::from_element(3, 2, 0.5);
  let sp = DVector::<f64>::from_vec(vec![1.0, 1.0]);
  let mut emb = DMatrix::<f64>::from_element(3, 4, 1.0);
  emb[(2, 1)] = f64::NEG_INFINITY;
  assert!(matches!(
    weighted_centroids_dm(&q, &sp, &emb, SP_ALIVE_THRESHOLD),
    Err(Error::NonFinite(_))
  ));
}

/// Single alive cluster, uniform q → centroid is the simple mean of all
/// embeddings, equal to the column means of `embeddings`.
#[test]
fn single_alive_cluster_uniform_q_returns_mean() {
  let q = DMatrix::<f64>::from_element(4, 1, 0.25);
  let sp = DVector::<f64>::from_vec(vec![1.0]);
  let emb = DMatrix::<f64>::from_row_slice(
    4,
    3,
    &[
      1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ],
  );
  let c = weighted_centroids_dm(&q, &sp, &emb, SP_ALIVE_THRESHOLD).expect("ok");
  // Expected mean of each column: (1+4+7+10)/4=5.5, (2+5+8+11)/4=6.5, (3+6+9+12)/4=7.5
  assert_eq!(c.shape(), (1, 3));
  assert!((c[(0, 0)] - 5.5).abs() < 1e-12);
  assert!((c[(0, 1)] - 6.5).abs() < 1e-12);
  assert!((c[(0, 2)] - 7.5).abs() < 1e-12);
}

/// Filter drops dead columns: q has 3 columns but only one survives;
/// the centroid should match what computing the centroid on just that
/// column would produce.
#[test]
fn filter_drops_dead_clusters() {
  // q: column 0 puts all weight on row 0; column 1 has zero everywhere
  // (sp will be filtered); column 2 puts all weight on row 2.
  let q = DMatrix::<f64>::from_row_slice(3, 3, &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
  let sp = DVector::<f64>::from_vec(vec![0.6, 1.0e-10, 0.4]);
  let emb = DMatrix::<f64>::from_row_slice(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
  let c = weighted_centroids_dm(&q, &sp, &emb, SP_ALIVE_THRESHOLD).expect("ok");
  assert_eq!(c.shape(), (2, 2));
  // Surviving cluster 0 (alive_idx 0) → row 0 of emb.
  assert!((c[(0, 0)] - 1.0).abs() < 1e-12);
  assert!((c[(0, 1)] - 2.0).abs() < 1e-12);
  // Surviving cluster 2 (alive_idx 1) → row 2 of emb.
  assert!((c[(1, 0)] - 5.0).abs() < 1e-12);
  assert!((c[(1, 1)] - 6.0).abs() < 1e-12);
}

/// Weighted average: column 0 has q values [0.6, 0.3, 0.1] for emb rows
/// [a, b, c]. Centroid = (0.6*a + 0.3*b + 0.1*c) / 1.0 = a*0.6 + b*0.3 + c*0.1.
#[test]
fn weighted_mean_normalizes_by_total_weight() {
  let q = DMatrix::<f64>::from_row_slice(3, 1, &[0.6, 0.3, 0.1]);
  let sp = DVector::<f64>::from_vec(vec![1.0]);
  let emb = DMatrix::<f64>::from_row_slice(3, 2, &[10.0, 20.0, 100.0, 200.0, 1000.0, 2000.0]);
  let c = weighted_centroids_dm(&q, &sp, &emb, SP_ALIVE_THRESHOLD).expect("ok");
  // weighted sum: 0.6*10 + 0.3*100 + 0.1*1000 = 6 + 30 + 100 = 136
  // weight sum = 1.0, so centroid[0] = 136
  assert!((c[(0, 0)] - 136.0).abs() < 1e-12);
  assert!((c[(0, 1)] - 272.0).abs() < 1e-12);
}

/// Surviving cluster with zero total weight (all-zero q column) →
/// `Error::Shape` rather than NaN-producing division.
#[test]
fn zero_total_weight_in_alive_cluster_errors() {
  // sp says cluster 0 is alive, but q's column 0 is all zeros.
  let q = DMatrix::<f64>::zeros(3, 1);
  let sp = DVector::<f64>::from_vec(vec![0.5]);
  let emb = DMatrix::<f64>::from_element(3, 2, 1.0);
  assert!(matches!(
    weighted_centroids_dm(&q, &sp, &emb, SP_ALIVE_THRESHOLD),
    Err(Error::Shape(_))
  ));
}

#[test]
fn deterministic_on_repeated_calls() {
  let q = DMatrix::<f64>::from_fn(8, 3, |i, j| {
    ((i * 7 + j * 13) as f64 * 0.05).sin().abs() + 0.01
  });
  let sp = DVector::<f64>::from_vec(vec![0.4, 0.4, 0.2]);
  let emb = DMatrix::<f64>::from_fn(8, 5, |i, j| ((i + 2 * j) as f64 * 0.1).cos());
  let a = weighted_centroids_dm(&q, &sp, &emb, SP_ALIVE_THRESHOLD).expect("a");
  let b = weighted_centroids_dm(&q, &sp, &emb, SP_ALIVE_THRESHOLD).expect("b");
  for r in 0..a.nrows() {
    for c in 0..a.ncols() {
      assert_eq!(a[(r, c)], b[(r, c)]);
    }
  }
}
