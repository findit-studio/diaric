//! Model-free unit tests for `diarization::cluster::ahc`.
//!
//! Heavy parity against pyannote's captured `ahc_init_labels.npy` lives
//! in `src/ahc/parity_tests.rs`. This module covers smaller invariants
//! that should hold for any input.

use crate::cluster::ahc::{Error, ahc_init};
use nalgebra::DMatrix;

/// Test helper: convert a column-major `DMatrix<f64>` to a row-major
/// `(Vec<f64>, n, d)` triple matching the new `ahc_init` signature.
/// Old tests that constructed `DMatrix` for convenience can use this
/// adapter instead of being rewritten in row-major flat form.
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

/// Convenience wrapper: `ahc_init` from a `&DMatrix<f64>` for tests.
fn ahc_init_dm(
  m: &DMatrix<f64>,
  threshold: f64,
  spill_options: &crate::ops::spill::SpillOptions,
) -> Result<Vec<usize>, Error> {
  let (data, n, d) = dm_to_row_major(m);
  ahc_init(&data, n, d, threshold, spill_options)
}

#[test]
fn rejects_empty_embeddings() {
  let m = DMatrix::<f64>::zeros(0, 4);
  assert!(matches!(
    ahc_init_dm(&m, 0.5, &crate::ops::spill::SpillOptions::default()),
    Err(Error::Shape(_))
  ));
}

#[test]
fn rejects_zero_dimension() {
  let m = DMatrix::<f64>::zeros(3, 0);
  assert!(matches!(
    ahc_init_dm(&m, 0.5, &crate::ops::spill::SpillOptions::default()),
    Err(Error::Shape(_))
  ));
}

#[test]
fn rejects_non_positive_threshold() {
  let m = DMatrix::<f64>::from_element(3, 4, 1.0);
  assert!(matches!(
    ahc_init_dm(&m, 0.0, &crate::ops::spill::SpillOptions::default()),
    Err(Error::Shape(_))
  ));
  assert!(matches!(
    ahc_init_dm(&m, -0.1, &crate::ops::spill::SpillOptions::default()),
    Err(Error::Shape(_))
  ));
}

#[test]
fn rejects_non_finite_threshold() {
  let m = DMatrix::<f64>::from_element(3, 4, 1.0);
  assert!(matches!(
    ahc_init_dm(&m, f64::NAN, &crate::ops::spill::SpillOptions::default()),
    Err(Error::Shape(_))
  ));
  assert!(matches!(
    ahc_init_dm(
      &m,
      f64::INFINITY,
      &crate::ops::spill::SpillOptions::default()
    ),
    Err(Error::Shape(_))
  ));
}

#[test]
fn rejects_nan_in_embedding() {
  let mut m = DMatrix::<f64>::from_element(3, 4, 1.0);
  m[(1, 2)] = f64::NAN;
  assert!(matches!(
    ahc_init_dm(&m, 0.5, &crate::ops::spill::SpillOptions::default()),
    Err(Error::NonFinite(_))
  ));
}

#[test]
fn rejects_inf_in_embedding() {
  let mut m = DMatrix::<f64>::from_element(3, 4, 1.0);
  m[(0, 0)] = f64::INFINITY;
  assert!(matches!(
    ahc_init_dm(&m, 0.5, &crate::ops::spill::SpillOptions::default()),
    Err(Error::NonFinite(_))
  ));
}

#[test]
fn rejects_zero_norm_row() {
  let mut m = DMatrix::<f64>::from_element(3, 4, 1.0);
  for c in 0..4 {
    m[(1, c)] = 0.0;
  }
  assert!(matches!(
    ahc_init_dm(&m, 0.5, &crate::ops::spill::SpillOptions::default()),
    Err(Error::Shape(_))
  ));
}

/// Adversarial: every element is finite but `v * v` accumulates to
/// `+inf`. Without the overflow guard, the normalize step would
/// collapse the row to all zeros and AHC would silently merge every
/// row into one cluster while returning `Ok(_)`. We must surface a
/// typed error instead.
#[test]
fn rejects_finite_row_with_overflowing_norm() {
  use crate::cluster::ahc::error::ShapeError;
  // |v| > sqrt(f64::MAX / d) → v*v sums overflow. For d=4,
  // threshold ~= sqrt(f64::MAX/4) ≈ 6.7e153. Pick a value safely above.
  let big = 1.0e154_f64;
  let mut m = DMatrix::<f64>::from_element(3, 4, 1.0);
  for c in 0..4 {
    m[(1, c)] = big;
  }
  let r = ahc_init_dm(&m, 0.5, &crate::ops::spill::SpillOptions::default());
  assert!(
    matches!(r, Err(Error::Shape(ShapeError::RowNormOverflow))),
    "got {r:?}"
  );
}

/// Single row → single cluster (matches pyannote's `< 2` short-circuit).
#[test]
fn single_row_returns_single_cluster() {
  let m = DMatrix::<f64>::from_row_slice(1, 3, &[1.0, 0.0, 0.0]);
  let labels = ahc_init_dm(&m, 0.5, &crate::ops::spill::SpillOptions::default()).expect("ahc_init");
  assert_eq!(labels, vec![0]);
}

/// Two near-identical rows + a far row → two clusters when threshold
/// admits the close pair but not the far one. The test mirrors scipy's
/// behavior that we hand-verified during development.
///
/// Rows (after L2 normalization):
/// - Row 0 ≈ (1, 0, 0)
/// - Row 1 ≈ (0.99, 0.01, 0)  → close to Row 0
/// - Row 2 ≈ (0, 1, 0)         → orthogonal
///
/// Distances after L2 norm: d(0,1) ≈ 0.014, d(0,2) ≈ 1.414, d(1,2) ≈ 1.404.
/// At threshold = 0.5: only the (0,1) pair merges. Asserts partition
/// equivalence: rows 0 and 1 share a label, row 2 has a distinct
/// label. Specific label *values* are determined by
/// `np.unique`-style canonicalization (sort distinct DFS labels
/// ascending) and depend on dendrogram traversal.
#[test]
fn merges_close_pair_separates_far_row() {
  let m = DMatrix::<f64>::from_row_slice(3, 3, &[1.0, 0.0, 0.0, 100.0, 1.0, 0.0, 0.0, 1.0, 0.0]);
  let labels = ahc_init_dm(&m, 0.5, &crate::ops::spill::SpillOptions::default()).expect("ahc_init");
  assert_eq!(
    labels[0], labels[1],
    "rows 0 and 1 should share a cluster (got {labels:?})"
  );
  assert_ne!(
    labels[0], labels[2],
    "row 2 should be its own cluster (got {labels:?})"
  );
}

/// All identical rows (after normalization) → single cluster regardless
/// of threshold. Distances are zero, so any positive threshold merges all.
#[test]
fn all_identical_normed_rows_collapse_to_one_cluster() {
  let m = DMatrix::<f64>::from_row_slice(
    4,
    2,
    &[
      1.0, 0.0, 2.0, 0.0, // same direction → same after L2 norm
      3.0, 0.0, 0.5, 0.0,
    ],
  );
  let labels =
    ahc_init_dm(&m, 0.001, &crate::ops::spill::SpillOptions::default()).expect("ahc_init");
  assert_eq!(labels, vec![0, 0, 0, 0]);
}

/// Threshold below all merge distances → every row is its own cluster.
#[test]
fn tiny_threshold_keeps_every_row_isolated() {
  // Three orthogonal directions; pairwise distance after L2 norm ≈ √2 ≈ 1.414.
  let m = DMatrix::<f64>::from_row_slice(3, 3, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
  let labels = ahc_init_dm(&m, 0.1, &crate::ops::spill::SpillOptions::default()).expect("ahc_init");
  // Each leaf is its own cluster: 3 distinct labels, all from {0, 1, 2}.
  let mut sorted = labels.clone();
  sorted.sort_unstable();
  sorted.dedup();
  assert_eq!(
    sorted,
    vec![0, 1, 2],
    "expected 3 distinct singleton clusters, got {labels:?}"
  );
}

/// Labels must be contiguous `0..k` after `np.unique`-style
/// canonicalization (sort distinct DFS labels ascending). The specific
/// label values depend on the dendrogram traversal; only partition
/// equivalence is asserted here.
#[test]
fn labels_are_contiguous_after_canonicalization() {
  // Six rows: two pairs that should merge, plus two singletons that
  // shouldn't. Specific arrangement: pair A (rows 0, 3), pair B (rows
  // 1, 4), singleton (row 2), singleton (row 5).
  let m = DMatrix::<f64>::from_row_slice(
    6,
    3,
    &[
      1.0, 0.0, 0.0, // row 0: pair A
      0.0, 1.0, 0.0, // row 1: pair B
      0.0, 0.0, 1.0, // row 2: singleton
      1.001, 0.0, 0.0, // row 3: pair A (close to row 0 after norm)
      0.0, 1.001, 0.0, // row 4: pair B (close to row 1 after norm)
      1.0, 1.0, 1.0, // row 5: singleton
    ],
  );
  let labels = ahc_init_dm(&m, 0.1, &crate::ops::spill::SpillOptions::default()).expect("ahc_init");
  // Partition equivalence: rows 0 and 3 share a cluster, rows 1 and 4
  // share, rows 2 and 5 are their own clusters.
  assert_eq!(
    labels[0], labels[3],
    "rows 0,3 should share (got {labels:?})"
  );
  assert_eq!(
    labels[1], labels[4],
    "rows 1,4 should share (got {labels:?})"
  );
  assert_ne!(labels[0], labels[1]);
  assert_ne!(labels[0], labels[2]);
  assert_ne!(labels[0], labels[5]);
  assert_ne!(labels[1], labels[2]);
  assert_ne!(labels[1], labels[5]);
  assert_ne!(labels[2], labels[5]);
  // Labels are contiguous 0..k.
  let max = *labels.iter().max().unwrap();
  let mut seen = vec![false; max + 1];
  for &l in &labels {
    seen[l] = true;
  }
  assert!(seen.iter().all(|&s| s), "labels {labels:?} not contiguous");
}

// ── Centroid-linkage inversion ─
//
// Centroid linkage (the method pyannote uses) does not produce
// monotonic dendrograms in general — a parent merge can have a
// *lower* dissimilarity than one of its children. Scipy's
// `fcluster(criterion="distance")` handles this by computing the
// max merge dissimilarity in each subtree before cutting, so a
// flat cluster's pairwise cophenetic distances are all `≤ t`.
//
// The regression test below uses a 4-point unit-vector configuration
// where:
// - d(0, 1) = 0.65          (above threshold 0.6 → step 0)
// - d(2, {0, 1}) = 0.574    (BELOW threshold via Lance-Williams)
// - d(3, *) ≈ 1.89          (far above)
//
// The dendrogram has an inversion at step 1 (lower than step 0).
// A naive bottom-up "union when step.dist ≤ t" walk would merge
// {0, 1, 2} into one cluster (matching root step 1's low dist), but
// scipy splits all three because the {0, 1} subtree's max internal
// merge (0.65) is still above threshold. The Rust port must agree
// with scipy.

/// Pyannote's centroid-linkage flow can produce a non-monotonic
/// dendrogram. The fcluster cut must use the *max* dissimilarity in
/// each subtree, not just the root's `step.dissimilarity`. This test
/// constructs a deterministic 4-point input that triggers the
/// inversion at threshold 0.6 — same partition as scipy.
#[test]
fn centroid_linkage_inversion_matches_scipy() {
  // 4 unit vectors in 3D. d(0,1)=0.65 above threshold, but step 1
  // (merging point 2 with {0,1}) inverts to dist=0.574, BELOW threshold.
  let alpha = 2.0_f64 * (0.65_f64 / 2.0).asin();
  let p0 = (1.0_f64, 0.0_f64, 0.0_f64);
  let p1 = (alpha.cos(), alpha.sin(), 0.0_f64);
  // p2 chosen so |p2-p0| = |p2-p1| = 0.66, |p2| = 1.
  let cdota = 1.0 - 0.66_f64.powi(2) / 2.0;
  let cy = (cdota - p1.0 * cdota) / p1.1;
  let cz = (1.0_f64 - cdota * cdota - cy * cy).sqrt();
  let p2 = (cdota, cy, cz);
  let p3 = (-1.0_f64, 0.0_f64, 0.0_f64);

  let m = DMatrix::<f64>::from_row_slice(
    4,
    3,
    &[
      p0.0, p0.1, p0.2, p1.0, p1.1, p1.2, p2.0, p2.1, p2.2, p3.0, p3.1, p3.2,
    ],
  );

  let labels = ahc_init_dm(&m, 0.6, &crate::ops::spill::SpillOptions::default()).expect("ahc_init");

  // Scipy on this dendrogram:
  //   step 0 (merge 0, 1): d=0.65 > 0.6
  //   step 1 (merge 2, {0,1}): d=0.574 ≤ 0.6 BUT subtree's max = 0.65 > 0.6
  //   step 2 (merge 3, ...): d=1.89 > 0.6
  // → no merges accepted; each leaf is its own cluster.
  // Each of the 4 leaves is its own cluster: 4 distinct labels.
  let mut sorted = labels.clone();
  sorted.sort_unstable();
  sorted.dedup();
  assert_eq!(
    sorted,
    vec![0, 1, 2, 3],
    "inversion case must match scipy: subtree max > threshold means split (got {labels:?})"
  );
}

/// Determinism: same input → identical output.
#[test]
fn deterministic_on_repeated_calls() {
  let m = DMatrix::<f64>::from_fn(8, 4, |i, j| ((i * 7 + j * 13) as f64 * 0.1).sin() + 1.0);
  let a = ahc_init_dm(&m, 0.5, &crate::ops::spill::SpillOptions::default()).expect("a");
  let b = ahc_init_dm(&m, 0.5, &crate::ops::spill::SpillOptions::default()).expect("b");
  assert_eq!(a, b);
}

// Removed in round 8: `ahc_init_with_simd` is gone.
//
// Production AHC now calls `ops::scalar::pdist_euclidean` directly,
// so the "AHC produces the same partition under SIMD vs scalar pdist"
// contract is satisfied trivially. Backend-differential coverage of
// pdist itself moved to `ops::differential_tests::pdist_euclidean_*`.
