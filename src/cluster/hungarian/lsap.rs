//! `scipy.optimize.linear_sum_assignment`-compatible rectangular LSAP.
//!
//! Direct Rust port of scipy's `rectangular_lsap.cpp` (BSD-3, Crouse's
//! shortest augmenting path; PM Larsen). The implementation is based
//! on:
//!
//!   DF Crouse, "On implementing 2D rectangular assignment algorithms,"
//!   IEEE Transactions on Aerospace and Electronic Systems
//!   52(4):1679–1696, 2016. doi:10.1109/TAES.2016.140952
//!
//! ## Why a port instead of `pathfinding::kuhn_munkres`
//!
//! Pyannote's `constrained_argmax` calls
//! `scipy.optimize.linear_sum_assignment(cost, maximize=True)` per
//! chunk. Both Kuhn-Munkres (pathfinding) and LAPJV/Crouse (scipy) are
//! exact maximum-weight matching algorithms, but on tied inputs
//! they return different optimal matchings — a documented divergence
//! in the audit (`hungarian/algo.rs`). For long recordings with
//! many sub-100ms overlap regions the inactive-(chunk, speaker) mask
//! produces fully tied rows; pyannote's choice is then implementation-
//! defined by scipy's traversal, and matching it is the only way to
//! get bit-exact `hard_clusters` (the testaudioset bench surfaced 37
//! tied-row mismatches across 611 chunks of `10_mrbeast_clean_water`).
//!
//! Two tie-breaking quirks of scipy's algorithm matter for parity:
//! 1. The `remaining` worklist is filled in reverse (`nc - it - 1`),
//!    so the first column considered is the highest-index column.
//! 2. When `shortest_path_costs[j]` ties the running minimum, scipy
//!    prefers a column whose `row4col[j] == -1` (i.e. an unassigned
//!    sink), short-circuiting the augmenting search.
//!
//! Both are reproduced exactly here.

use crate::cluster::hungarian::error::{Error, ShapeError};

/// scipy-compatible solution to the rectangular linear sum assignment
/// problem.
///
/// `cost` is row-major: `cost[i * nc + j]` is the cost of assigning
/// row `i` to column `j`. Returns `(row_ind, col_ind)` such that
/// each pair `(row_ind[k], col_ind[k])` is one assignment, and the
/// optimal cost equals `Σ cost[row_ind[k], col_ind[k]]`. Row indices
/// are sorted ascending — same contract as scipy's
/// `linear_sum_assignment`.
///
/// ## Errors
///
/// - `Error::Shape::EmptyChunks` if `nr == 0` or `nc == 0` (scipy's
///   trivial-input branch).
/// - `Error::NonFinite` if any cost cell is non-finite (`NaN`,
///   `+inf`, or `-inf`). `+inf` and `NaN` are rejected here even
///   though the in-tree caller `constrained_argmax` already filters
///   them at its own boundary, so a future caller that bypasses
///   `constrained_argmax` (or passes `maximize=false` where negation
///   wouldn't convert `+inf` to `-inf`) still gets a clear error
///   instead of an opaque `EmptyChunks` infeasibility report.
/// - `Error::Shape::EmptyChunks` (re-used) if the cost matrix is
///   "infeasible" — every augmenting path lookup hit `+inf`. With
///   finite inputs (which the check above guarantees) this branch
///   is unreachable.
///
/// `maximize=true` is handled the same way as scipy: negate the cost
/// matrix in a working copy. Caller's input slice is not mutated.
pub(crate) fn linear_sum_assignment(
  nr: usize,
  nc: usize,
  cost: &[f64],
  maximize: bool,
) -> Result<(Vec<usize>, Vec<usize>), Error> {
  if nr == 0 || nc == 0 {
    return Err(ShapeError::EmptyChunks.into());
  }
  if cost.len() != nr * nc {
    return Err(ShapeError::InconsistentChunkShape.into());
  }
  // scipy transposes when `nc < nr` so the augmenting path always
  // covers the longer dimension. Track the orientation so we can
  // un-transpose the output.
  let transpose = nc < nr;
  // Working copy: transpose and/or negate as scipy does. The caller's
  // input slice is left untouched.
  let mut working: Vec<f64> = if transpose {
    let mut t = vec![0.0_f64; nr * nc];
    for i in 0..nr {
      for j in 0..nc {
        t[j * nr + i] = cost[i * nc + j];
      }
    }
    t
  } else {
    cost.to_vec()
  };
  let (work_nr, work_nc) = if transpose { (nc, nr) } else { (nr, nc) };
  if maximize {
    for v in working.iter_mut() {
      *v = -*v;
    }
  }
  // Validate after transpose/negate so the rejection mirrors scipy
  // (which also checks the working copy). `!is_finite()` catches NaN
  // and both infinities — important because under `maximize=false`
  // a `+inf` would otherwise survive into the dual-update arithmetic
  // (the previous narrower `is_nan() || == NEG_INFINITY` check missed
  // that case for non-`constrained_argmax` callers).
  for &v in working.iter() {
    if !v.is_finite() {
      return Err(crate::cluster::hungarian::error::NonFiniteError::InfInSoftClusters.into());
    }
  }

  let mut u = vec![0.0_f64; work_nr];
  let mut v = vec![0.0_f64; work_nc];
  let mut shortest_path_costs = vec![0.0_f64; work_nc];
  let mut path = vec![-1isize; work_nc];
  let mut col4row = vec![-1isize; work_nr];
  let mut row4col = vec![-1isize; work_nc];
  let mut sr = vec![false; work_nr];
  let mut sc = vec![false; work_nc];
  let mut remaining = vec![0usize; work_nc];

  for cur_row in 0..work_nr {
    let mut min_val = 0.0_f64;
    let sink = augmenting_path(
      work_nc,
      &working,
      &u,
      &v,
      &mut path,
      &row4col,
      &mut shortest_path_costs,
      cur_row,
      &mut sr,
      &mut sc,
      &mut remaining,
      &mut min_val,
    );
    if sink < 0 {
      // Infeasible cost matrix (every augmenting path closed at +inf).
      // With finite costs this branch is unreachable; we re-use
      // EmptyChunks rather than introduce a new variant.
      return Err(ShapeError::EmptyChunks.into());
    }

    // Update dual variables.
    u[cur_row] += min_val;
    for i in 0..work_nr {
      if sr[i] && i != cur_row {
        let j_prev = col4row[i];
        // col4row[i] is set by the augmentation below for i != cur_row.
        // It cannot be -1 here because sr[i] = true means row i was
        // visited in the augmenting path, and the search only visits
        // i = row4col[j] when row4col[j] != -1.
        debug_assert!(j_prev >= 0);
        u[i] += min_val - shortest_path_costs[j_prev as usize];
      }
    }
    for j in 0..work_nc {
      if sc[j] {
        v[j] -= min_val - shortest_path_costs[j];
      }
    }

    // Augment previous solution.
    let mut j = sink as usize;
    loop {
      let i = path[j];
      row4col[j] = i;
      let prev = col4row[i as usize];
      col4row[i as usize] = j as isize;
      if i as usize == cur_row {
        break;
      }
      j = prev as usize;
    }
  }

  // Build (row_ind, col_ind). For the un-transposed case, row_ind is
  // 0..nr and col_ind is col4row. For the transposed case, scipy
  // sorts by col4row to recover row-major order — `argsort` here.
  let (row_ind, col_ind) = if transpose {
    let order = argsort_isize(&col4row);
    let mut a = Vec::with_capacity(work_nr);
    let mut b = Vec::with_capacity(work_nr);
    for v_idx in order {
      a.push(col4row[v_idx] as usize);
      b.push(v_idx);
    }
    (a, b)
  } else {
    let mut a = Vec::with_capacity(work_nr);
    let mut b = Vec::with_capacity(work_nr);
    for (i, &c) in col4row.iter().enumerate().take(work_nr) {
      a.push(i);
      b.push(c as usize);
    }
    (a, b)
  };
  Ok((row_ind, col_ind))
}

#[allow(clippy::too_many_arguments)]
fn augmenting_path(
  nc: usize,
  cost: &[f64],
  u: &[f64],
  v: &[f64],
  path: &mut [isize],
  row4col: &[isize],
  shortest_path_costs: &mut [f64],
  i_init: usize,
  sr: &mut [bool],
  sc: &mut [bool],
  remaining: &mut [usize],
  p_min_val: &mut f64,
) -> isize {
  let mut min_val = 0.0_f64;

  // Crouse's pseudocode tracks the remaining set via complement; the
  // C++ source uses an explicit Vec for efficiency. **Quirk #1 for
  // scipy parity**: fill in *reverse* order so the first column
  // considered is the highest-index column. This determines the
  // tie-break direction on fully-tied rows (e.g. inactive-mask rows
  // where every column has the `inactive_const`).
  let mut num_remaining = nc;
  for (it, slot) in remaining.iter_mut().enumerate().take(nc) {
    *slot = nc - it - 1;
  }
  for x in sr.iter_mut() {
    *x = false;
  }
  for x in sc.iter_mut() {
    *x = false;
  }
  for x in shortest_path_costs.iter_mut() {
    *x = f64::INFINITY;
  }

  let mut sink: isize = -1;
  let mut i = i_init;
  while sink == -1 {
    let mut index: isize = -1;
    let mut lowest = f64::INFINITY;
    sr[i] = true;

    for (it, &j) in remaining[..num_remaining].iter().enumerate() {
      let r = min_val + cost[i * nc + j] - u[i] - v[j];
      if r < shortest_path_costs[j] {
        path[j] = i as isize;
        shortest_path_costs[j] = r;
      }
      // **Quirk #2 for scipy parity**: among columns whose reduced
      // cost ties the running minimum, prefer one with a fresh sink
      // (`row4col[j] == -1`). This short-circuits the augmenting
      // search by handing back an unassigned column rather than
      // recursing into another row's match. Critical for tied
      // inactive-mask rows in our pipeline.
      if shortest_path_costs[j] < lowest || (shortest_path_costs[j] == lowest && row4col[j] == -1) {
        lowest = shortest_path_costs[j];
        index = it as isize;
      }
    }

    min_val = lowest;
    if min_val == f64::INFINITY {
      return -1;
    }

    let j = remaining[index as usize];
    if row4col[j] == -1 {
      sink = j as isize;
    } else {
      i = row4col[j] as usize;
    }

    sc[j] = true;
    num_remaining -= 1;
    remaining[index as usize] = remaining[num_remaining];
  }

  *p_min_val = min_val;
  sink
}

fn argsort_isize(v: &[isize]) -> Vec<usize> {
  let mut idx: Vec<usize> = (0..v.len()).collect();
  idx.sort_by(|&a, &b| v[a].cmp(&v[b]));
  idx
}

#[cfg(test)]
mod tests {
  use super::*;

  /// Audit's counterexample (hungarian/algo.rs:13-14): scipy returns
  /// (row_ind=[1,2], col_ind=[1,0]) on the unique-max row 2. Cost
  /// matrix is 3×2 maximize=True; `pathfinding::kuhn_munkres`
  /// returned `[1, -2, 0]` instead.
  #[test]
  fn matches_scipy_counterexample() {
    // Cost: [[0,0],[0,0],[1,1]], maximize=True.
    let cost = [0.0_f64, 0.0, 0.0, 0.0, 1.0, 1.0];
    let (row_ind, col_ind) = linear_sum_assignment(3, 2, &cost, true).unwrap();
    // scipy: row=[1, 2], col=[1, 0]
    assert_eq!(row_ind, vec![1, 2]);
    assert_eq!(col_ind, vec![1, 0]);
  }

  /// Identity case: scipy guarantees row_ind = 0..nr and a valid
  /// matching for square inputs. With all-zero cost, the diagonal is
  /// the canonical assignment (#11602).
  #[test]
  fn all_zero_square_returns_identity() {
    let cost = vec![0.0_f64; 4];
    let (row_ind, col_ind) = linear_sum_assignment(2, 2, &cost, false).unwrap();
    assert_eq!(row_ind, vec![0, 1]);
    assert_eq!(col_ind, vec![0, 1]);
  }

  /// Probe: 3×7 with row-0 fully tied (inactive-mask row), row-1 max
  /// at col 6, row-2 max at col 0. scipy assigns 0→2, 1→6, 2→0 (per
  /// our diagnostic). Pin this exact behavior.
  #[test]
  fn matches_scipy_inactive_mask_row() {
    let cost = vec![
      // row 0: all -0.2 (tied)
      -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, // row 1: ascending; max at col 6
      0.96, 0.95, 1.03, 1.25, 1.29, 1.47, 1.86, // row 2: max at col 0
      1.24, 1.09, 1.21, 1.21, 1.18, 1.20, 1.41,
    ];
    let (row_ind, col_ind) = linear_sum_assignment(3, 7, &cost, true).unwrap();
    assert_eq!(row_ind, vec![0, 1, 2]);
    assert_eq!(col_ind, vec![2, 6, 0]);
  }

  /// 2×4 tied row-0 → scipy picks col 1.
  #[test]
  fn matches_scipy_2x4_tied_row() {
    let cost = vec![
      0.0, 0.0, 0.0, 0.0, // row 0 tied
      1.0, 0.5, 0.3, 0.7, // row 1 max at 0
    ];
    let (row_ind, col_ind) = linear_sum_assignment(2, 4, &cost, true).unwrap();
    assert_eq!(row_ind, vec![0, 1]);
    assert_eq!(col_ind, vec![1, 0]);
  }

  /// Empty inputs surface a typed error.
  #[test]
  fn rejects_empty_dim() {
    let cost: Vec<f64> = vec![];
    assert!(linear_sum_assignment(0, 5, &cost, false).is_err());
    assert!(linear_sum_assignment(5, 0, &cost, false).is_err());
  }

  /// NaN entries are rejected (matches scipy's
  /// `RECTANGULAR_LSAP_INVALID`).
  #[test]
  fn rejects_nan_cost() {
    let cost = vec![1.0, f64::NAN, 0.0, 0.0];
    assert!(linear_sum_assignment(2, 2, &cost, false).is_err());
  }
}
