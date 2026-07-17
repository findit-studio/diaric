//! Constrained Hungarian assignment (per-chunk maximum-weight matching).
//!
//! Ports `pyannote.audio.pipelines.clustering.SpeakerEmbedding.constrained_argmax`
//! (`clustering.py:127-140` in pyannote.audio 4.0.4). Pyannote takes the
//! full `(num_chunks, num_speakers, num_clusters)` cost tensor, replaces
//! NaN entries with the *global* `np.nanmin(soft_clusters)`, and runs
//! `scipy.optimize.linear_sum_assignment(cost, maximize=True)` per chunk.
//!
//! ## Solver
//!
//! Uses the in-tree scipy-compatible LSAP port at
//! [`crate::cluster::hungarian::lsap::linear_sum_assignment`] (a direct
//! port of SciPy's `rectangular_lsap.cpp` — Crouse / LAPJV with the
//! same row-traversal and column-augmentation order). On tied optima
//! the tie-break therefore matches scipy bit-for-bit; the previous
//! `pathfinding::kuhn_munkres` adapter that left ties as a non-contract
//! has been retired.
//!
//! Pre-port counterexample (`pathfinding::kuhn_munkres` vs scipy):
//! cost `[[0,0],[0,0],[1,1]]` → scipy returns `[-2, 1, 0]`,
//! pathfinding returned `[1, -2, 0]`. The new solver returns
//! `[-2, 1, 0]` — same labels and unmatched-row choice as scipy.
//!
//! ## Shape contract
//!
//! `linear_sum_assignment` accepts any rectangular cost matrix; the
//! adapter here transposes when `num_speakers > num_clusters` to match
//! pyannote's per-chunk maximize-over-cluster orientation, then maps
//! the LSAP output back to a `[i32; SLOTS]` row. The previous
//! "rows ≤ columns" constraint that the `pathfinding` adapter required
//! no longer applies.
//!
//! Captured 218-chunk fixture passes parity exactly. The
//! invariant-based tie tests in `src/cluster/hungarian/tests.rs`
//! ("tie-breaking" section) additionally pin that the same labels
//! scipy would return are produced.

use crate::cluster::hungarian::error::Error;
use nalgebra::DMatrix;

/// Sentinel value for an unmatched speaker. Matches pyannote's
/// `-2 * np.ones((num_chunks, num_speakers), dtype=np.int8)` initializer.
pub const UNMATCHED: i32 = -2;

/// Maximum allowed magnitude for any finite entry in a cost matrix
/// passed to [`constrained_argmax`]. The LSAP solver
/// (`crate::cluster::hungarian::lsap::linear_sum_assignment`)
/// accumulates dual-variable updates that touch every cell; values
/// approaching `f64::MAX` overflow to `±inf` after one or two
/// additions and wedge the augmenting-path search. The same bound
/// applied under the previous `pathfinding::kuhn_munkres` adapter for
/// the same reason — kept here so the upstream `±inf` guard catches
/// caller-side corruption before the solver does.
///
/// `1e15` is a documented safe range with O(150) decimal orders of
/// headroom from `f64::MAX ≈ 1.8e308`. Production cosine distances are
/// bounded by 2 and PLDA log-likelihoods by O(100), so any value
/// beyond `1e15` indicates upstream corruption (decoder NaN-flooding,
/// memory bit-flips, mis-loaded float32→float64 reinterpretation)
/// rather than a legitimate cost matrix.
pub const MAX_COST_MAGNITUDE: f64 = 1e15;

// ── Sealed `ChunkLayout` trait + per-architecture marker types ───────────

mod sealed {
  pub trait Sealed {}
}

/// Sealed marker trait describing a segmentation-model output layout.
///
/// Each implementor pins the number of speaker slots a particular
/// upstream model architecture emits per chunk. The trait is
/// **sealed** (the supertrait `sealed::Sealed` is private) — external
/// crates cannot add their own layouts. New layouts must land in
/// `dia` itself, paired with:
/// 1. A captured fixture from the upstream model's reference
///    Python pipeline.
/// 2. A parity test in `cluster::hungarian::parity_tests` (or the
///    relevant downstream module) validating the new `SLOTS` count
///    against the captured tensor shapes.
///
/// The `Row` associated type is the per-chunk hard-cluster assignment
/// array (`[i32; SLOTS]`); using an associated type instead of a
/// hard-coded alias means downstream public APIs (`assign_embeddings`,
/// `OfflineOutput`, `reconstruct`) don't have to change shape if a
/// future v0.x minor adds a second layout — they switch to a
/// `<L: ChunkLayout>` generic parameter and the existing
/// [`DefaultLayout`] alias keeps current callers working.
pub trait ChunkLayout: sealed::Sealed + Copy + Default + 'static {
  /// Number of speaker slots per chunk for this layout.
  const SLOTS: usize;
  /// Per-chunk hard-cluster assignment row type — conventionally
  /// `[i32; SLOTS]`.
  type Row: Copy + 'static;
}

/// pyannote/segmentation-3.0 layout (community-1 model architecture):
/// 3 speaker slots per chunk. The only layout `dia` v0.1.x supports;
/// new pyannote model releases would add their own marker types
/// alongside this one.
#[derive(Debug, Clone, Copy, Default)]
pub struct Segmentation3;
impl sealed::Sealed for Segmentation3 {}
impl ChunkLayout for Segmentation3 {
  const SLOTS: usize = crate::segment::options::MAX_SPEAKER_SLOTS as usize;
  type Row = [i32; crate::segment::options::MAX_SPEAKER_SLOTS as usize];
}

/// Default segmentation layout for `dia` v0.1.x. Type-aliased to
/// [`Segmentation3`] so public APIs that today commit to community-1's
/// architecture don't need a `<L: ChunkLayout>` generic. When a
/// future release adds a second layout, this alias stays pinned to
/// `Segmentation3` for backward compatibility — callers wanting the
/// new layout opt in via the explicit marker type.
pub type DefaultLayout = Segmentation3;

/// Per-chunk hard-cluster assignment row for the [`DefaultLayout`]
/// (`[i32; 3]` under segmentation-3.0). `[s]` is the cluster id, or
/// [`UNMATCHED`] (`-2`) for speakers with no surviving cluster.
///
/// Resolved through the [`ChunkLayout`] associated type (rather than
/// a direct `[i32; 3]` alias) so future expansion to other model
/// architectures is a non-breaking addition rather than a public-API
/// type churn.
pub type ChunkAssignment = <DefaultLayout as ChunkLayout>::Row;

/// Batched constrained Hungarian assignment over a stack of per-chunk
/// `(num_speakers, num_clusters)` cost matrices.
///
/// Returns one `Vec<i32>` of length `num_speakers` per chunk. Each entry is
/// the cluster index assigned to that speaker, or [`UNMATCHED`] (`-2`) if
/// the speaker had no cluster left (only possible when
/// `num_speakers > num_clusters`).
///
/// # Pyannote parity: `np.nan_to_num` semantics (NaN only)
///
/// Pyannote's `constrained_argmax` runs `np.nan_to_num(soft_clusters,
/// nan=np.nanmin(soft_clusters))` before per-chunk matching. The realistic
/// NaN source is an empty AHC cluster whose centroid is `NaN/NaN` after
/// averaging zero embeddings; the Rust port replicates that:
///
/// - **NaN** → global `nanmin` across all finite entries
///   (`np.nanmin`-equivalent on the production path where `±inf` cannot
///   appear).
///
/// `±inf` is **rejected** rather than substituted with `f64::MAX/MIN`
/// (numpy's `nan_to_num` defaults). Two reasons:
///
/// 1. Production cosine distances over finite embeddings are always
///    finite, so `±inf` indicates upstream corruption rather than a
///    well-defined edge case the algorithm should silently handle.
/// 2. The LSAP solver
///    ([`crate::cluster::hungarian::lsap::linear_sum_assignment`])
///    does dual-variable updates on the costs; feeding `f64::MAX`
///    risks overflow into `±inf`/`NaN` and wedges the augmenting-path
///    search. Rejecting at the boundary keeps the solver inside its
///    safe operating envelope (and also bounded the prior
///    `pathfinding::kuhn_munkres` adapter for the same reason).
///
/// # Errors
///
/// - [`Error::Shape`] if `chunks` is empty, any chunk has zero rows or
///   zero columns, or chunks differ in shape.
/// - [`Error::NonFinite`] if any chunk contains `+inf` or `-inf`, or if
///   *every* entry across all chunks is NaN (no finite value to use as
///   the `nanmin` replacement). Pyannote degenerates in the all-NaN case
///   too (`np.nanmin` returns NaN, and the resulting assignment is
///   undefined).
///
/// # Algorithm
///
/// [`crate::cluster::hungarian::lsap::linear_sum_assignment`] is
/// shape-agnostic, but pyannote runs the assignment in
/// `(num_speakers, num_clusters)` orientation with `maximize=True`.
/// We negate to convert maximize → minimize and pass the matrix
/// directly. The output `(row_ind, col_ind)` is then mapped back to a
/// `[i32; SLOTS]` row indexed by speaker.
pub fn constrained_argmax(chunks: &[DMatrix<f64>]) -> Result<Vec<Vec<i32>>, Error> {
  use crate::cluster::hungarian::error::ShapeError;
  if chunks.is_empty() {
    return Err(ShapeError::EmptyChunks.into());
  }
  let (num_speakers, num_clusters) = chunks[0].shape();
  if num_speakers == 0 {
    return Err(ShapeError::ZeroSpeakers.into());
  }
  if num_clusters == 0 {
    return Err(ShapeError::ZeroClusters.into());
  }
  for chunk in chunks {
    if chunk.shape() != (num_speakers, num_clusters) {
      return Err(ShapeError::InconsistentChunkShape.into());
    }
  }

  // Reject ±inf upfront, then bound the magnitude of finite entries
  // so they cannot drive the LSAP solver's dual-variable updates
  // into overflow.
  //
  // Numpy's `np.nan_to_num` substitutes ±inf with `f64::MAX/MIN`, but
  // feeding those values into the solver's `lx + ly - weight` and
  // label-update sums overflows to `±inf`/`NaN` after a single
  // addition and can wedge the solver per the crate's own docs. The
  // `MAX_COST_MAGNITUDE` bound (1e15) catches `f64::MAX`-class
  // corruption while leaving O(150) decimal orders of headroom for
  // any realistic cost matrix.
  //
  // Production cosine distances and PLDA log-likelihoods are always
  // finite and bounded by O(100), so `±inf` or `|v| > 1e15` here
  // indicates upstream corruption — surface a clear typed error
  // rather than silently proceed with values that may wedge the
  // solver.
  for chunk in chunks {
    for &v in chunk.iter() {
      if v.is_infinite() {
        return Err(crate::cluster::hungarian::error::NonFiniteError::InfInSoftClusters.into());
      }
      if v.is_finite() && v.abs() > MAX_COST_MAGNITUDE {
        return Err(
          crate::cluster::hungarian::error::NonFiniteError::WeightOutOfBounds {
            value: v,
            max: MAX_COST_MAGNITUDE,
          }
          .into(),
        );
      }
    }
  }

  // Compute the global nanmin across all chunks for the NaN replacement.
  // After the `±inf` rejection above, `is_finite()` partitions entries
  // into {finite, NaN}, matching numpy's `nanmin` semantics on the
  // production path.
  let mut nanmin = f64::INFINITY;
  let mut any_finite = false;
  for chunk in chunks {
    for &v in chunk.iter() {
      if v.is_finite() {
        any_finite = true;
        if v < nanmin {
          nanmin = v;
        }
      }
    }
  }
  if !any_finite {
    return Err(crate::cluster::hungarian::error::NonFiniteError::NoFiniteEntries.into());
  }

  let mut out = Vec::with_capacity(chunks.len());
  for chunk in chunks {
    out.push(assign_one(chunk, num_speakers, num_clusters, nanmin)?);
  }
  Ok(out)
}

/// NaN-only `np.nan_to_num` cleanup: replace `NaN` with `nanmin`. The
/// `±inf` cases are rejected upstream by `constrained_argmax`, so this
/// function is only ever called on `{finite, NaN}` inputs and always
/// returns a finite value.
#[inline]
fn clean(v: f64, nanmin: f64) -> f64 {
  if v.is_nan() { nanmin } else { v }
}

fn assign_one(
  chunk: &DMatrix<f64>,
  num_speakers: usize,
  num_clusters: usize,
  nanmin: f64,
) -> Result<Vec<i32>, Error> {
  // scipy-compatible rectangular LSAP. Required for bit-exact pyannote
  // parity on tied costs (inactive-(chunk, speaker) mask rows).
  // `pathfinding::kuhn_munkres` returns the same maximum weight but
  // diverges from scipy on tie-breaking, surfacing as
  // `partition mismatch at chunk N` failures on long recordings (06,
  // testaudioset 09/10/11/12/13/14/08). The Crouse-LAPJV port in
  // `lsap` mirrors scipy's traversal order verbatim.
  let mut assignment = vec![UNMATCHED; num_speakers];
  let mut row_major = Vec::with_capacity(num_speakers * num_clusters);
  for s in 0..num_speakers {
    for k in 0..num_clusters {
      row_major.push(clean(chunk[(s, k)], nanmin));
    }
  }
  let (row_ind, col_ind) = crate::cluster::hungarian::lsap::linear_sum_assignment(
    num_speakers,
    num_clusters,
    &row_major,
    true,
  )?;
  for (r, c) in row_ind.into_iter().zip(col_ind) {
    assignment[r] = i32::try_from(c).expect("cluster idx fits in i32");
  }
  Ok(assignment)
}
