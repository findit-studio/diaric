//! Model-free unit tests for `diarization::cluster::hungarian`.
//!
//! Heavy parity against pyannote's captured `hard_clusters` lives in
//! `src/hungarian/parity_tests.rs`. This module covers smaller invariants
//! that should hold for any input.

use crate::cluster::hungarian::{
  Error, MAX_COST_MAGNITUDE, UNMATCHED, constrained_argmax, error::NonFiniteError,
};
use nalgebra::DMatrix;

/// Run a single chunk through the batched API. Most unit tests work on
/// one chunk at a time; this wrapper avoids repeating the slice + index
/// boilerplate.
fn one(cost: DMatrix<f64>) -> Result<Vec<i32>, Error> {
  constrained_argmax(&[cost]).map(|mut v| v.remove(0))
}

#[test]
fn rejects_empty_chunks() {
  let result = constrained_argmax(&[]);
  assert!(matches!(result, Err(Error::Shape(_))), "got {result:?}");
}

#[test]
fn rejects_empty_speakers() {
  let cost = DMatrix::<f64>::zeros(0, 3);
  let result = one(cost);
  assert!(matches!(result, Err(Error::Shape(_))), "got {result:?}");
}

#[test]
fn rejects_empty_clusters() {
  let cost = DMatrix::<f64>::zeros(3, 0);
  let result = one(cost);
  assert!(matches!(result, Err(Error::Shape(_))), "got {result:?}");
}

#[test]
fn rejects_chunks_with_different_shapes() {
  let a = DMatrix::<f64>::from_element(2, 2, 0.5);
  let b = DMatrix::<f64>::from_element(3, 2, 0.5);
  let result = constrained_argmax(&[a, b]);
  assert!(matches!(result, Err(Error::Shape(_))), "got {result:?}");
}

/// Square 2x2 — direct kuhn_munkres path. Diagonal dominates.
#[test]
fn square_2x2_picks_diagonal_when_diagonal_dominates() {
  let cost = DMatrix::<f64>::from_row_slice(2, 2, &[0.9, 0.1, 0.2, 0.8]);
  let assign = one(cost).expect("constrained_argmax");
  assert_eq!(assign, vec![0, 1]);
}

/// Square 2x2 — anti-diagonal dominates. Catches a greedy "row max" bug.
#[test]
fn square_2x2_picks_anti_diagonal_when_off_diagonal_dominates() {
  let cost = DMatrix::<f64>::from_row_slice(2, 2, &[0.2, 0.9, 0.8, 0.1]);
  let assign = one(cost).expect("constrained_argmax");
  assert_eq!(assign, vec![1, 0]);
}

/// Tall (S < K): 2 speakers, 3 clusters. Both speakers must be matched
/// to distinct clusters; the unused cluster index is just dropped.
#[test]
fn tall_2x3_assigns_both_speakers_to_distinct_clusters() {
  let cost = DMatrix::<f64>::from_row_slice(2, 3, &[0.1, 0.5, 1.0, 0.9, 0.4, 0.3]);
  let assign = one(cost).expect("constrained_argmax");
  assert_eq!(assign, vec![2, 0]);
  assert!(!assign.contains(&UNMATCHED));
}

/// Wide (S > K): 3 speakers, 2 clusters — captured-fixture shape.
/// Exercises the transpose path. Two speakers matched, one UNMATCHED.
#[test]
fn wide_3x2_leaves_one_speaker_unmatched() {
  let cost = DMatrix::<f64>::from_row_slice(3, 2, &[0.95, 0.05, 0.05, 0.95, 0.10, 0.10]);
  let assign = one(cost).expect("constrained_argmax");
  assert_eq!(assign, vec![0, 1, UNMATCHED]);
}

/// Wide (S > K) where the optimal assignment leaves a *non-weakest*
/// speaker unmatched. Speaker 0 has cell 0.95 in cluster 0, but assigning
/// {2→0 (0.99), 1→1 (0.95)} sums to 1.94 > {0→0 (0.95), 1→1 (0.95)} = 1.90.
/// Catches a "leave the lowest-row speaker unmatched" greedy bug.
#[test]
fn wide_3x2_optimal_unmatches_non_weakest_speaker() {
  let cost = DMatrix::<f64>::from_row_slice(3, 2, &[0.95, 0.10, 0.05, 0.95, 0.99, 0.10]);
  let assign = one(cost).expect("constrained_argmax");
  assert_eq!(assign, vec![UNMATCHED, 1, 0]);
}

/// Distinct-cluster invariant: every matched assignment uses a different
/// cluster index. Holds for square, tall, and wide shapes.
#[test]
fn matched_speakers_are_assigned_distinct_clusters() {
  let cost = DMatrix::<f64>::from_fn(4, 4, |i, j| ((i * 7 + j * 13) % 17) as f64 * 0.1);
  let assign = one(cost).expect("constrained_argmax");
  let mut used = std::collections::HashSet::new();
  for &k in &assign {
    if k != UNMATCHED {
      assert!(used.insert(k), "cluster {k} assigned twice in {assign:?}");
    }
  }
  assert!(!assign.contains(&UNMATCHED));
}

#[test]
fn single_speaker_single_cluster() {
  let cost = DMatrix::<f64>::from_element(1, 1, 0.42);
  let assign = one(cost).expect("constrained_argmax");
  assert_eq!(assign, vec![0]);
}

#[test]
fn single_speaker_multiple_clusters_picks_max() {
  let cost = DMatrix::<f64>::from_row_slice(1, 4, &[0.1, 0.5, 0.9, 0.3]);
  let assign = one(cost).expect("constrained_argmax");
  assert_eq!(assign, vec![2]);
}

#[test]
fn single_cluster_multiple_speakers_matches_max_speaker() {
  let cost = DMatrix::<f64>::from_row_slice(3, 1, &[0.1, 0.9, 0.5]);
  let assign = one(cost).expect("constrained_argmax");
  assert_eq!(assign, vec![UNMATCHED, 0, UNMATCHED]);
}

#[test]
fn deterministic_on_repeated_calls() {
  let cost = DMatrix::<f64>::from_fn(5, 4, |i, j| ((i + 2 * j) as f64 * 0.13).cos());
  let a = one(cost.clone()).expect("a");
  let b = one(cost).expect("b");
  assert_eq!(a, b);
}

// ── nan_to_num semantics ─
//
// Pyannote runs `np.nan_to_num(soft_clusters, nan=np.nanmin(soft_clusters))`
// before per-chunk matching. The Rust port replicates this:
// NaN → global nanmin across all chunks, +inf → f64::MAX, -inf → f64::MIN.

/// NaN entries in a single chunk are replaced with the chunk's own min.
/// The replacement must produce a valid optimal matching, not error out.
#[test]
fn nan_in_single_chunk_replaced_with_min() {
  // 2x2 with NaN in (1, 0). Other entries: 0.9, 0.5, NaN, 0.8.
  // nanmin = 0.5. After replacement: 0.9, 0.5, 0.5, 0.8.
  // Optimal: speaker 0 → cluster 0 (0.9), speaker 1 → cluster 1 (0.8).
  let mut cost = DMatrix::<f64>::from_row_slice(2, 2, &[0.9, 0.5, 0.0, 0.8]);
  cost[(1, 0)] = f64::NAN;
  let assign = one(cost).expect("constrained_argmax with NaN must replace, not error");
  assert_eq!(assign, vec![0, 1]);
}

/// NaN replacement uses the *global* min across all chunks, not the per-
/// chunk min — this matches pyannote's contract.
///
/// Setup: chunk 0 = [[0.9, 0.5], [0.7, NaN]]. Chunk 1 contains -5.0.
/// - Local nanmin (0.5) replacement of chunk 0's NaN:
///   {s0→c0 (0.9), s1→c1 (0.5)} = 1.4 vs {s0→c1 (0.5), s1→c0 (0.7)} = 1.2
///   → optimal pairs s0→c0, s1→c1 (assignment vec![0, 1]).
/// - Global nanmin (-5.0) replacement of chunk 0's NaN:
///   {s0→c0 (0.9), s1→c1 (-5.0)} = -4.1 vs {s0→c1 (0.5), s1→c0 (0.7)} = 1.2
///   → optimal pairs s0→c1, s1→c0 (assignment vec![1, 0]).
///
/// Different assignments confirm global vs local replacement behavior.
#[test]
fn nan_replacement_uses_global_nanmin_across_chunks() {
  let mut chunk_a = DMatrix::<f64>::from_row_slice(2, 2, &[0.9, 0.5, 0.7, 0.0]);
  chunk_a[(1, 1)] = f64::NAN;
  let chunk_b = DMatrix::<f64>::from_row_slice(2, 2, &[0.0, 0.0, 0.0, -5.0]);

  let assigns = constrained_argmax(&[chunk_a, chunk_b]).expect("constrained_argmax");
  assert_eq!(assigns.len(), 2);
  // Global-min replacement (-5.0) drives the chunk-0 optimal to anti-
  // diagonal: speaker 0 → cluster 1, speaker 1 → cluster 0.
  assert_eq!(assigns[0], vec![1, 0]);
}

/// `±inf` is **rejected** at the boundary rather than substituted with
/// numpy's `f64::MAX`/`f64::MIN` defaults. Two reasons: production
/// cosine distances are always finite, so `±inf` is upstream corruption,
/// not a well-defined edge case; and feeding `f64::MAX` into
/// `kuhn_munkres`'s slack arithmetic risks overflow per the crate's own
/// docs.
#[test]
fn rejects_pos_inf_entry() {
  let mut cost = DMatrix::<f64>::from_element(2, 2, 0.5);
  cost[(0, 1)] = f64::INFINITY;
  let result = one(cost);
  assert!(matches!(result, Err(Error::NonFinite(_))), "got {result:?}");
}

#[test]
fn rejects_neg_inf_entry() {
  let mut cost = DMatrix::<f64>::from_element(2, 2, 0.5);
  cost[(1, 0)] = f64::NEG_INFINITY;
  let result = one(cost);
  assert!(matches!(result, Err(Error::NonFinite(_))), "got {result:?}");
}

/// Mixed: a chunk with both NaN and `±inf` rejects rather than
/// half-handling it.
#[test]
fn rejects_inf_even_when_nan_also_present() {
  let mut cost = DMatrix::<f64>::from_row_slice(2, 2, &[0.0, 0.5, 0.7, 0.0]);
  cost[(0, 0)] = f64::NAN;
  cost[(1, 1)] = f64::NEG_INFINITY;
  let result = one(cost);
  assert!(matches!(result, Err(Error::NonFinite(_))), "got {result:?}");
}

/// All entries non-finite → there's no value to use as the nanmin
/// replacement. Pyannote degenerates here too (`np.nanmin` of an
/// all-NaN array returns NaN, and `nan_to_num(x, nan=NaN)` is a no-op).
/// The Rust port surfaces this as `Error::NonFinite` rather than
/// silently producing a NaN-poisoned assignment.
#[test]
fn rejects_when_all_entries_non_finite() {
  let cost = DMatrix::<f64>::from_element(2, 2, f64::NAN);
  let result = one(cost);
  assert!(matches!(result, Err(Error::NonFinite(_))), "got {result:?}");
}

/// Finite-but-huge cost magnitudes overflow the solver's internal
/// `lx + ly - weight` accumulator after one or two additions. Values
/// like `f64::MAX` (which numpy's `nan_to_num` substitutes for `±inf`)
/// reintroduce the exact failure mode the upstream `±inf` guard
/// prevents. Reject at the boundary with a typed error instead of
/// letting the solver wedge.
#[test]
fn rejects_finite_value_above_max_cost_magnitude() {
  let mut cost = DMatrix::<f64>::from_element(2, 2, 0.5);
  cost[(0, 1)] = f64::MAX;
  let result = one(cost);
  assert!(
    matches!(
      result,
      Err(Error::NonFinite(NonFiniteError::WeightOutOfBounds { .. })),
    ),
    "got {result:?}"
  );
}

#[test]
fn rejects_negative_finite_below_neg_max_cost_magnitude() {
  let mut cost = DMatrix::<f64>::from_element(2, 2, 0.5);
  cost[(1, 0)] = f64::MIN;
  let result = one(cost);
  assert!(
    matches!(
      result,
      Err(Error::NonFinite(NonFiniteError::WeightOutOfBounds { .. })),
    ),
    "got {result:?}"
  );
}

/// At the boundary: |MAX_COST_MAGNITUDE| accepted; just over rejected.
#[test]
fn accepts_value_at_max_cost_magnitude() {
  let mut cost = DMatrix::<f64>::from_element(2, 2, 0.5);
  cost[(0, 0)] = MAX_COST_MAGNITUDE;
  cost[(1, 1)] = -MAX_COST_MAGNITUDE;
  let result = one(cost);
  assert!(result.is_ok(), "got {result:?}");
}

#[test]
fn rejects_value_just_above_max_cost_magnitude() {
  let mut cost = DMatrix::<f64>::from_element(2, 2, 0.5);
  // Smallest f64 strictly greater than MAX_COST_MAGNITUDE.
  cost[(0, 1)] = f64::from_bits(MAX_COST_MAGNITUDE.to_bits() + 1);
  let result = one(cost);
  assert!(
    matches!(
      result,
      Err(Error::NonFinite(NonFiniteError::WeightOutOfBounds { .. })),
    ),
    "got {result:?}"
  );
}

// ── Tie-breaking invariants ─
//
// `pathfinding::kuhn_munkres` and `scipy.optimize.linear_sum_assignment`
// can return different label permutations on tied optima. See the
// algo.rs module-level docstring for the rationale. These tests check
// the *invariants* the algorithm must satisfy under ties — total
// optimal weight, distinct cluster ids, max matching size — without
// locking in a specific label permutation.

/// Compute the maximum-weight matching's total cost by brute force on a
/// 2D matrix. Used as an oracle for tie tests where the algorithm's
/// chosen permutation is implementation-defined.
fn brute_force_max_total(cost: &DMatrix<f64>) -> f64 {
  let (rows, cols) = cost.shape();
  let n = rows.min(cols);
  // Enumerate all subsets of `n` columns, then for each, all
  // permutations of which row gets which column. Tractable for small
  // matrices used in tests (rows, cols ≤ 4).
  fn subsets(k: usize, n: usize) -> Vec<Vec<usize>> {
    if k == 0 {
      return vec![vec![]];
    }
    let mut out = Vec::new();
    for end in k..=n {
      for mut sub in subsets(k - 1, end - 1) {
        sub.push(end - 1);
        out.push(sub);
      }
    }
    out
  }
  fn permutations(items: &[usize]) -> Vec<Vec<usize>> {
    if items.is_empty() {
      return vec![vec![]];
    }
    let mut out = Vec::new();
    for i in 0..items.len() {
      let mut rest: Vec<usize> = items.to_vec();
      let head = rest.remove(i);
      for mut perm in permutations(&rest) {
        perm.insert(0, head);
        out.push(perm);
      }
    }
    out
  }

  // Choose `n` rows from `rows`, `n` cols from `cols`, then the
  // assignment is a permutation between them. Maximize over all.
  let mut best = f64::NEG_INFINITY;
  for row_subset in subsets(n, rows) {
    for col_subset in subsets(n, cols) {
      for perm in permutations(&col_subset) {
        let total: f64 = row_subset
          .iter()
          .zip(perm.iter())
          .map(|(&r, &c)| cost[(r, c)])
          .sum();
        if total > best {
          best = total;
        }
      }
    }
  }
  best
}

/// Compute the achieved total cost from an assignment vector for a
/// chunk. UNMATCHED entries contribute zero (they're not part of the
/// matching).
fn achieved_total(cost: &DMatrix<f64>, assign: &[i32]) -> f64 {
  let mut total = 0.0;
  for (s, &k) in assign.iter().enumerate() {
    if k != UNMATCHED {
      total += cost[(s, k as usize)];
    }
  }
  total
}

/// 3x2 with two equal zero rows: both `[1, -2, 0]` (pathfinding) and
/// `[-2, 1, 0]` (scipy) are valid optima with total = 1.0. The
/// invariants we enforce: total cost equals the brute-force max, exactly
/// one speaker is UNMATCHED, and matched cluster ids are distinct.
#[test]
fn tied_3x2_returns_some_optimal_matching() {
  let cost = DMatrix::<f64>::from_row_slice(3, 2, &[0.0, 0.0, 0.0, 0.0, 1.0, 1.0]);
  let assign = one(cost.clone()).expect("constrained_argmax");

  let max = brute_force_max_total(&cost);
  let achieved = achieved_total(&cost, &assign);
  assert!(
    (achieved - max).abs() < 1e-12,
    "tied input must still hit max total ({max:.6}); got {achieved:.6} from {assign:?}"
  );

  // Exactly min(3, 2) = 2 speakers matched, 1 unmatched.
  let unmatched_count = assign.iter().filter(|&&k| k == UNMATCHED).count();
  assert_eq!(unmatched_count, 1, "expected 1 unmatched, got {assign:?}");

  // Matched cluster ids are distinct (hungarian-bipartite invariant).
  let mut used = std::collections::HashSet::new();
  for &k in &assign {
    if k != UNMATCHED {
      assert!(used.insert(k), "duplicate cluster {k} in {assign:?}");
    }
  }
}

/// All-tied square matrix: every cell equal. Total = n * cell_value.
/// Every speaker must be matched, every matched cluster distinct —
/// the *which* cluster each speaker gets is implementation-defined.
#[test]
fn tied_3x3_all_equal_returns_some_optimal_matching() {
  let cost = DMatrix::<f64>::from_element(3, 3, 0.5);
  let assign = one(cost.clone()).expect("constrained_argmax");

  let max = brute_force_max_total(&cost);
  let achieved = achieved_total(&cost, &assign);
  assert!(
    (achieved - max).abs() < 1e-12,
    "all-tied square must still hit max ({max:.6}); got {achieved:.6}"
  );
  assert!(
    !assign.contains(&UNMATCHED),
    "square: all matched; got {assign:?}"
  );

  let mut used = std::collections::HashSet::new();
  for &k in &assign {
    assert!(used.insert(k), "duplicate cluster {k} in {assign:?}");
  }
}

/// Tall (S < K) with tied rows: every speaker matched, each to a
/// distinct cluster, total at the brute-force max.
#[test]
fn tied_2x3_returns_some_optimal_matching() {
  // Both rows tied within their own row (0.5 across all clusters);
  // the matching can pair any speaker with any cluster.
  let cost = DMatrix::<f64>::from_element(2, 3, 0.5);
  let assign = one(cost.clone()).expect("constrained_argmax");

  let max = brute_force_max_total(&cost);
  let achieved = achieved_total(&cost, &assign);
  assert!(
    (achieved - max).abs() < 1e-12,
    "tall tied: must hit max ({max:.6}); got {achieved:.6}"
  );
  assert!(
    !assign.contains(&UNMATCHED),
    "tall: all speakers matched; got {assign:?}"
  );

  let mut used = std::collections::HashSet::new();
  for &k in &assign {
    assert!(used.insert(k), "duplicate cluster {k} in {assign:?}");
  }
}

/// Inactive-speaker shape from pyannote's flow: one strong row + two
/// equal-constant rows (mimicking `soft_clusters[inactive] = const`).
/// The strong speaker must be matched to one of its preferred clusters;
/// the inactive speaker that gets matched can be either of the two
/// (implementation-defined). Either way, total weight is optimal.
#[test]
fn pyannote_inactive_speaker_pattern_hits_optimal_total() {
  // const = soft.min() - 1.0 = -1.5. Real speaker 0 has cells (0.9, 0.5);
  // speakers 1 and 2 are inactive (rows of -1.5).
  let cost = DMatrix::<f64>::from_row_slice(3, 2, &[0.9, 0.5, -1.5, -1.5, -1.5, -1.5]);
  let assign = one(cost.clone()).expect("constrained_argmax");

  let max = brute_force_max_total(&cost);
  let achieved = achieved_total(&cost, &assign);
  assert!(
    (achieved - max).abs() < 1e-12,
    "inactive-speaker pattern must hit max ({max:.6}); got {achieved:.6} from {assign:?}"
  );

  // Speaker 0 (the only active one) must get cluster 0 (its 0.9 peak
  // dominates 0.5 plus any inactive-row contribution at -1.5).
  assert_eq!(
    assign[0], 0,
    "active speaker 0 must be matched to its peak cluster; got {assign:?}"
  );

  // Exactly one inactive speaker is matched, one unmatched.
  let inactive_matched = (assign[1] != UNMATCHED) as usize + (assign[2] != UNMATCHED) as usize;
  assert_eq!(
    inactive_matched, 1,
    "exactly one inactive speaker should be matched; got {assign:?}"
  );
}

// ── Sealed-trait pattern regression checks ──────────────────────────────

/// `ChunkAssignment` resolves to the default layout's row type.
/// Locks in the v0.1.x community-1 commitment without hard-coding
/// `[i32; 3]` at every call site.
#[test]
fn chunk_assignment_resolves_to_default_layout_row() {
  use crate::cluster::hungarian::{ChunkAssignment, ChunkLayout, DefaultLayout, Segmentation3};

  // Compile-time identity: the public alias is the default layout's row.
  let _: ChunkAssignment = [0_i32; 3];
  // The default IS Segmentation3.
  let _: <DefaultLayout as ChunkLayout>::Row = [0_i32; Segmentation3::SLOTS];
  assert_eq!(<DefaultLayout as ChunkLayout>::SLOTS, 3);
}

/// Documents the sealed-trait contract: external crates cannot
/// `impl ChunkLayout for MyLayout` because the supertrait
/// `sealed::Sealed` is private to `cluster::hungarian::algo`. This
/// test checks the runtime side (we can construct + use the marker);
/// the seal itself is enforced by Rust's module privacy at the trait
/// definition site, not by a runtime assertion.
#[test]
fn chunk_layout_seal_smoke() {
  use crate::cluster::hungarian::{ChunkLayout, Segmentation3};
  let _layout = Segmentation3;
  assert_eq!(<Segmentation3 as ChunkLayout>::SLOTS, 3);
}
