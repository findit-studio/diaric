//! AHC initialization: L2-normalize → centroid linkage → fcluster + remap.
//!
//! ## Determinism contract w.r.t. `pdist_euclidean`
//!
//! Production [`ahc_init`] calls [`crate::ops::scalar::pdist_euclidean`]
//! directly, on every architecture. AHC's `<= threshold` dendrogram
//! cut is the one threshold-sensitive discrete decision in the
//! cluster_vbx pipeline; using scalar pdist makes the AHC partition
//! bit-equal across NEON / AVX2 / AVX-512 / scalar hosts. AVX2/AVX-512
//! reductions diverge from scalar by O(1e-15) ulps and any pair
//! landing in that drift band would merge on one CPU family and split
//! on another — the scalar-by-default policy here removes the risk
//! without affecting downstream stages.
//!
//! Differential tests at the primitive level live in
//! [`crate::ops::differential_tests`]; they compare
//! [`crate::ops::pdist_euclidean`] (best-available SIMD) against
//! [`crate::ops::scalar::pdist_euclidean`].

use std::collections::HashMap;

use crate::cluster::ahc::error::Error;
use kodama::{Method, Step, linkage};

/// Run pyannote's AHC initialization.
///
/// Mirrors `pyannote/audio/pipelines/clustering.py:597-604`:
///
/// 1. L2-normalize each row of `embeddings` (shape `(N, D)`).
/// 2. Compute pairwise euclidean distances (the condensed `pdist`-style
///    upper-triangular vector scipy expects).
/// 3. Centroid-method hierarchical linkage via `kodama` (matches scipy's
///    `linkage(..., method="centroid")` Lance-Williams formula).
/// 4. `fcluster` with `criterion="distance"` and the given `threshold`:
///    union pairs whose merge dissimilarity is `≤ threshold`.
/// 5. Remap the resulting partition to encounter-order contiguous labels
///    `0..k`, equivalent to `np.unique(_, return_inverse=True)[1]`.
///
/// # Errors
///
/// - [`Error::Shape`] if `embeddings` is empty, has zero-length rows, has
///   any zero-L2-norm row, or `threshold` is non-positive / non-finite.
/// - [`Error::NonFinite`] if `embeddings` contains a NaN/`±inf`.
///
/// # Single-row degenerate case
///
/// Pyannote short-circuits AHC entirely when `train_embeddings.shape[0]
/// < 2` (`clustering.py:588-594`). This module-level boundary allows
/// `N=1` and returns `vec![0]` (one cluster, one member) so callers can
/// drive `diarization::cluster::ahc::ahc_init` uniformly without the special case
/// leaking into them.
pub fn ahc_init(
  embeddings: &[f64],
  n: usize,
  d: usize,
  threshold: f64,
  spill_options: &crate::ops::spill::SpillOptions,
) -> Result<Vec<usize>, Error> {
  use crate::cluster::ahc::error::{NonFiniteField, ShapeError};
  // Row-major flat layout: `embeddings[r * d + c]`. Caller (the
  // pipeline) builds this directly from a spill-backed
  // `SpillBytesMut<f64>` so the input is `&[f64]` rather than
  // `&DMatrix<f64>` (which would require a heap-only nalgebra
  // allocation).
  if n == 0 {
    return Err(ShapeError::EmptyEmbeddings.into());
  }
  if d == 0 {
    return Err(ShapeError::ZeroEmbeddingDim.into());
  }
  let expected_len = n.checked_mul(d).ok_or(ShapeError::EmbeddingsSizeOverflow)?;
  if embeddings.len() != expected_len {
    return Err(ShapeError::EmbeddingsLenMismatch.into());
  }
  if !threshold.is_finite() || threshold <= 0.0 {
    return Err(ShapeError::InvalidThreshold.into());
  }
  // Validate finite + nonzero L2 norm per row.
  //
  // The `!sq.is_finite()` check matters even when every individual
  // element is finite: a row with very large finite values (|v| beyond
  // ~1e152 for D=256) makes `v*v` overflow `sq` to `+inf`. Without
  // catching it, `l2_normalize_to_row_major` computes `inv_norm =
  // 1/sqrt(inf) = 0`, every output row collapses to zeros, pdist sees
  // zero-distance pairs everywhere, and AHC silently merges everything
  // into one cluster while returning `Ok(_)` — wrong clustering with
  // no error. Same threat shape as the SegmentModel/EmbedModel
  // non-finite-output guards.
  for r in 0..n {
    let row = &embeddings[r * d..(r + 1) * d];
    let mut sq = 0.0;
    for &v in row {
      if !v.is_finite() {
        return Err(NonFiniteField::Embeddings.into());
      }
      sq += v * v;
    }
    if !sq.is_finite() {
      return Err(ShapeError::RowNormOverflow.into());
    }
    if sq == 0.0 {
      return Err(ShapeError::ZeroNormRow.into());
    }
  }

  if n == 1 {
    return Ok(vec![0]);
  }

  // L2-normalize → row-major flat buffer, spill-backed via
  // `SpillBytesMut`. At the documented `MAX_AHC_TRAIN = 32_000`
  // cap with `embed_dim = 256`, the normalized matrix is
  // `32_000 * 256 * 8 ≈ 65 MB` — same data-bearing scale as
  // `train_embeddings` and worth the spill route so a multi-hour
  // input crossing `SpillOptions::threshold_bytes` keeps the
  // typed `SpillError` instead of OOM-aborting on the heap path.
  let normed_row_major = l2_normalize_to_row_major(embeddings, n, d, spill_options)?;
  // Scalar pdist on every architecture. AHC is the one place in the
  // cluster_vbx pipeline where SIMD determinism actually matters:
  // the dendrogram cut is a hard `<= threshold` decision, so a pair
  // landing inside the AVX2/AVX-512-vs-scalar ulp drift band could
  // merge on one CPU family and split on another, giving
  // CPU-dependent speaker counts that are nearly impossible to
  // reproduce. NEON matches scalar bit-exact (verified by
  // `ops::differential_tests`), but AVX2/AVX-512 use wider-lane
  // reductions and diverge by O(1e-15) relative.
  //
  // Why this is OK to "give up" SIMD here specifically: AHC's hot
  // path is exactly one `pdist_euclidean` (O(N² × D)), then scalar
  // `kodama::linkage` + scalar fcluster. There is no nalgebra GEMM
  // anywhere in this function — unlike `vbx::vbx_iterate`, where
  // `matrixmultiply`'s own SIMD dispatch is uncontrolled. So
  // forcing scalar here actually delivers cross-arch bit-equal AHC
  // partitions, with a one-shot cost on the order of a few ms on
  // the largest captured fixture (T=1004) — not user-perceptible.
  //
  // The condensed buffer can hit ~1 GB at the documented production
  // scale (`MAX_AHC_TRAIN = 16_000` → 128M f64 cells). Route through
  // `SpillBytesMut` so the allocation falls back to file-backed mmap
  // above `SpillOptions::threshold_bytes` (default 64 MiB) instead
  // of OOM-aborting from the heap path. `kodama::linkage` consumes
  // the buffer as `&mut [f64]`, which `SpillBytesMut::as_mut_slice`
  // hands out for both backends without copying.
  let pair_count = crate::ops::scalar::pair_count(n);
  let mut cond = crate::ops::spill::SpillBytesMut::<f64>::zeros(pair_count, spill_options)?;
  crate::ops::scalar::pdist_euclidean_into(normed_row_major.as_slice(), n, d, cond.as_mut_slice());
  let dend = linkage(cond.as_mut_slice(), n, Method::Centroid);

  Ok(fcluster_distance_remap(dend.steps(), n, threshold))
}

/// Pack the row-wise L2-normalized embeddings into a spill-backed
/// row-major flat buffer in a single pass. The output is the same
/// data-bearing scale as the input `embeddings` slice (`n * d` f64),
/// so production-scale inputs route through `SpillBytesMut` here too
/// — a heap `Vec` would defeat the spill plumbing the caller paid
/// for in `train_embeddings`.
///
/// [`crate::ops::pdist_euclidean`] consumes the result via the read-
/// only `&[f64]` returned by `SpillBytes::as_slice()`.
///
/// Caller has already rejected zero-norm rows AND non-finite squared
/// norms (overflow). Both invariants are debug-asserted here as a
/// defense-in-depth check; production passes through unchanged.
fn l2_normalize_to_row_major(
  m: &[f64],
  n: usize,
  d: usize,
  spill_options: &crate::ops::spill::SpillOptions,
) -> Result<crate::ops::spill::SpillBytes<f64>, crate::ops::spill::SpillError> {
  let mut out = crate::ops::spill::SpillBytesMut::<f64>::zeros(n * d, spill_options)?;
  {
    let dst = out.as_mut_slice();
    for r in 0..n {
      let row = &m[r * d..(r + 1) * d];
      let mut sq = 0.0;
      for &v in row {
        sq += v * v;
      }
      debug_assert!(
        sq.is_finite() && sq > 0.0,
        "l2_normalize_to_row_major: caller must reject non-finite/zero \
         squared norms (row {r}: sq = {sq})"
      );
      let inv_norm = sq.sqrt().recip();
      let row_dst = &mut dst[r * d..(r + 1) * d];
      for (i, &v) in row.iter().enumerate() {
        row_dst[i] = v * inv_norm;
      }
    }
  }
  Ok(out.freeze())
}

/// `fcluster(criterion="distance", t=threshold)` followed by
/// `np.unique(return_inverse=True)`. Mirrors `scipy._hierarchy.cluster_dist`:
/// (1) precompute the *maximum* merge dissimilarity in each subtree,
/// (2) walk top-down, cutting wherever that max exceeds the threshold.
///
/// Why max-per-subtree rather than the root's own dissimilarity:
/// centroid linkage can produce *inversions* (a parent merge has lower
/// dissimilarity than one of its children). A walk that only checks
/// the root's `step.dissimilarity`
/// would merge an entire subtree based on a low-dist parent even when
/// an internal child merge is above the threshold. Scipy's fcluster
/// (`scipy/cluster/_hierarchy.pyx::cluster_dist`) propagates the max
/// dissimilarity up the tree first, then uses that as the cut criterion
/// — i.e. a flat cluster contains pairs whose cophenetic distance is
/// `≤ threshold`, which is the documented contract.
///
/// # Label assignment: leaf-scan encounter order, not scipy's traversal
///
/// The second pass canonicalizes labels via *leaf-scan encounter order*
/// (the first cluster seen while scanning leaves `0..n` becomes label 0).
/// This is the np.unique-on-contiguous-labels formula but assumes scipy
/// already produced canonical scan-order labels — which **scipy does
/// not do**. Scipy's `fcluster` numbers clusters by tree-traversal
/// order; the captured `ahc_init_labels.npy` starts with label `4` for
/// row 0, not `0`.
///
/// The captured AHC parity test compares partitions, not exact
/// label assignments — partition equivalence is sufficient for
/// downstream clustering correctness (the labels are arbitrary
/// integers naming the buckets; DER is invariant to relabeling).
///
/// # Element-wise q_final parity (not enforced)
///
/// Switching the parity oracle from partition-equivalence to element-wise
/// `q_final` would expose this label-permutation gap (qinit columns would
/// not align). The realistic input distribution and downstream DER are
/// invariant to relabeling, so this is intentionally not enforced. If a
/// future test pins element-wise `q_final`, three remediation paths are
/// available: (1) port scipy's tree-traversal DFS push order verbatim;
/// (2) compare modulo column permutation recoverable from
/// `(our_labels, scipy_labels)`; (3) return the permutation alongside
/// labels and let the caller build a column-permuted qinit.
fn fcluster_distance_remap(steps: &[Step<f64>], n: usize, threshold: f64) -> Vec<usize> {
  // Single leaf — no merges; one cluster.
  if n == 1 {
    return vec![0];
  }

  // Precompute the maximum dissimilarity in each subtree. Leaves have 0
  // (they contain no merges); compound id `n + i` has max of its own
  // merge plus the max of its two children.
  let total_nodes = n + steps.len();
  let mut subtree_max = vec![0.0_f64; total_nodes];
  for (i, step) in steps.iter().enumerate() {
    let m1 = subtree_max[step.cluster1];
    let m2 = subtree_max[step.cluster2];
    subtree_max[n + i] = step.dissimilarity.max(m1).max(m2);
  }

  // First pass: top-down DFS labels leaves by partition class.
  let mut raw = vec![usize::MAX; n];
  let mut next_dfs_label = 0usize;
  let root = total_nodes - 1;
  let mut stack: Vec<usize> = vec![root];
  while let Some(node) = stack.pop() {
    if node < n {
      // Bare leaf surfaced via a split — its own cluster.
      raw[node] = next_dfs_label;
      next_dfs_label += 1;
    } else if subtree_max[node] <= threshold {
      // Whole subtree fits within the threshold — one cluster.
      let l = next_dfs_label;
      next_dfs_label += 1;
      paint_leaves(node, n, steps, l, &mut raw);
    } else {
      // Subtree contains a merge above threshold; split into children.
      let step = &steps[node - n];
      stack.push(step.cluster2);
      stack.push(step.cluster1);
    }
  }

  // Second pass: `np.unique(raw, return_inverse=True)`-equivalent
  // canonicalization. Pyannote feeds scipy's `fcluster - 1` through
  // `np.unique(..., return_inverse=True)` (clustering.py:603-604), which
  // sorts the distinct DFS-pass labels ascending and remaps each row's
  // label to its rank in that sorted unique set. The previous
  // leaf-scan encounter-order canonicalization preserved partition
  // equivalence but not the label *values*; a downstream caller
  // (pipeline `assign_embeddings`) builds qinit columns indexed by
  // these labels, so a value mismatch here produced a column-permuted
  // qinit, which cascaded into VBx convergence to a different fixed
  // point on long fixtures (06_long_recording, testaudioset 09/10
  // and friends). Sorting by raw DFS value matches `np.unique` and
  // restores bit-exact qinit, q_final, centroid, soft, and
  // hard_clusters parity downstream.
  let mut unique_sorted: Vec<usize> = raw.clone();
  unique_sorted.sort_unstable();
  unique_sorted.dedup();
  let value_to_new: HashMap<usize, usize> = unique_sorted
    .iter()
    .enumerate()
    .map(|(i, &v)| (v, i))
    .collect();
  raw.iter().map(|v| value_to_new[v]).collect()
}

/// Recursively assign `label` to every leaf reachable from `node`.
/// Uses iterative traversal to avoid stack-depth concerns on deep
/// dendrograms.
fn paint_leaves(node: usize, n: usize, steps: &[Step<f64>], label: usize, labels: &mut [usize]) {
  let mut stack = vec![node];
  while let Some(cur) = stack.pop() {
    if cur < n {
      labels[cur] = label;
    } else {
      let step = &steps[cur - n];
      stack.push(step.cluster1);
      stack.push(step.cluster2);
    }
  }
}
