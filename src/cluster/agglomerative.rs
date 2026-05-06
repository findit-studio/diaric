//! Hierarchical agglomerative clustering. Spec §5.6.
//!
//! Builds a pairwise cosine-distance matrix `D[i][j] = 1 - max(0, e_i · e_j)`,
//! then iteratively merges the closest two clusters under the chosen
//! [`Linkage`] until either the target speaker count is reached or the
//! closest pair is farther than `1 - similarity_threshold`.
//!
//! Distance is ReLU-clamped (`max(0, sim)`) to match spectral clustering's
//! affinity convention (spec §5.5 / §5.6 rev-3).

use crate::{
  cluster::{
    Error,
    options::{Linkage, OfflineClusterOptions},
  },
  embed::Embedding,
};

/// Cluster `embeddings` agglomeratively. Returns labels in `[0..k)` assigned
/// in merge-order, parallel to the input slice.
///
/// Caller guarantees `embeddings.len() >= 3` (the N<=2 fast path lives in
/// `cluster_offline`). Runs in O(N^3) time, O(N^2) space — Lance-Williams
/// caching could amortize to O(N^2 · log N) but the current scale (≈100s of
/// embeddings per session) doesn't justify the complexity.
pub(crate) fn cluster(
  embeddings: &[Embedding],
  linkage: Linkage,
  opts: &OfflineClusterOptions,
) -> Result<Vec<u64>, Error> {
  let n = embeddings.len();
  debug_assert!(n >= 3, "fast path covers N <= 2");

  // Step 1: pairwise distance matrix `D[i][j] = 1 - max(0, e_i · e_j)`.
  // Symmetric; diagonal stays 0.0. Range [0, 1]. ReLU clamp matches
  // spectral's affinity convention (spec §5.5 / §5.6 rev-3).
  let mut d = vec![vec![0.0f32; n]; n];
  for (i, ei) in embeddings.iter().enumerate() {
    for (offset, ej) in embeddings.iter().skip(i + 1).enumerate() {
      let j = i + 1 + offset;
      let sim = ei.similarity(ej).max(0.0);
      let dist = 1.0 - sim;
      d[i][j] = dist;
      d[j][i] = dist;
    }
  }

  // Step 2: initialize each input as its own cluster.
  let mut clusters: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
  let stop_dist = 1.0 - opts.similarity_threshold();

  // Step 3-4: agglomerative merge loop. O(N) iterations × O(K^2) argmin
  // = O(N^3) total. Acceptable at v0.1.0 scale; Lance-Williams update
  // would amortize to O(N^2 · log N) for a future revision.
  loop {
    if clusters.len() == 1 {
      break;
    }
    if let Some(target) = opts.target_speakers()
      && clusters.len() == target as usize
    {
      break;
    }

    // Find the two closest active clusters.
    let mut best = (0usize, 1usize);
    let mut best_dist = f32::INFINITY;
    for (a, ca) in clusters.iter().enumerate() {
      for (offset, cb) in clusters.iter().skip(a + 1).enumerate() {
        let b = a + 1 + offset;
        let dist = pair_distance(ca, cb, &d, linkage);
        if dist < best_dist {
          best_dist = dist;
          best = (a, b);
        }
      }
    }

    // Stop if best pair is past threshold AND target is not fixed.
    // (Target-mode keeps merging until cluster count == target.)
    if opts.target_speakers().is_none() && best_dist >= stop_dist {
      break;
    }

    // Merge clusters[best.1] into clusters[best.0].
    let merged = clusters.remove(best.1);
    clusters[best.0].extend(merged);
  }

  // Step 5: assign labels parallel to input.
  let mut labels = vec![0u64; n];
  for (cluster_id, members) in clusters.iter().enumerate() {
    for &m in members {
      labels[m] = cluster_id as u64;
    }
  }
  Ok(labels)
}

/// Pairwise distance between two clusters under the given linkage.
fn pair_distance(a: &[usize], b: &[usize], d: &[Vec<f32>], linkage: Linkage) -> f32 {
  debug_assert!(
    !a.is_empty() && !b.is_empty(),
    "pair_distance requires non-empty clusters"
  );
  match linkage {
    Linkage::Single => {
      // Min over a × b.
      let mut best = f32::INFINITY;
      for &i in a {
        for &j in b {
          if d[i][j] < best {
            best = d[i][j];
          }
        }
      }
      best
    }
    Linkage::Complete => {
      // Max over a × b.
      let mut worst = f32::NEG_INFINITY;
      for &i in a {
        for &j in b {
          if d[i][j] > worst {
            worst = d[i][j];
          }
        }
      }
      worst
    }
    Linkage::Average => {
      // Arithmetic mean over a × b. f64 accumulator for stability —
      // mirrors online.rs::update_speaker rationale (sum of many f32s
      // can lose mantissa bits in f32).
      let mut sum = 0.0f64;
      for &i in a {
        for &j in b {
          sum += d[i][j] as f64;
        }
      }
      (sum / (a.len() * b.len()) as f64) as f32
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{
    cluster::{OfflineMethod, test_util::unit},
    embed::EMBEDDING_DIM,
  };

  fn opt_agg(linkage: Linkage) -> OfflineClusterOptions {
    OfflineClusterOptions::default().with_method(OfflineMethod::Agglomerative { linkage })
  }

  #[test]
  fn three_identical_one_cluster() {
    let e = vec![unit(0), unit(0), unit(0)];
    let r = cluster(&e, Linkage::Single, &opt_agg(Linkage::Single)).unwrap();
    assert_eq!(r, vec![0, 0, 0]);
  }

  #[test]
  fn three_orthogonal_three_clusters() {
    // All pairwise sim = 0 → dist = 1 = stop_dist (threshold = 0.5).
    // Stop condition `best_dist >= stop_dist` is met → no merges.
    let e = vec![unit(0), unit(1), unit(2)];
    let r = cluster(&e, Linkage::Single, &opt_agg(Linkage::Single)).unwrap();
    let mut sorted = r.clone();
    sorted.sort();
    assert_eq!(sorted, vec![0, 1, 2]);
  }

  #[test]
  fn two_groups_separated() {
    // Three near-unit-x + three near-unit-y → 2 clusters under Average.
    let mut samples = Vec::new();
    for delta in [0.0, 0.05, 0.1] {
      let mut v = [0.0f32; EMBEDDING_DIM];
      v[0] = 1.0;
      v[1] = delta;
      samples.push(Embedding::normalize_from(v).unwrap());
    }
    for delta in [0.0, 0.05, 0.1] {
      let mut v = [0.0f32; EMBEDDING_DIM];
      v[1] = 1.0;
      v[0] = delta;
      samples.push(Embedding::normalize_from(v).unwrap());
    }
    let r = cluster(&samples, Linkage::Average, &opt_agg(Linkage::Average)).unwrap();
    assert_eq!(r[0], r[1]);
    assert_eq!(r[1], r[2]);
    assert_eq!(r[3], r[4]);
    assert_eq!(r[4], r[5]);
    assert_ne!(r[0], r[3]);
  }

  #[test]
  fn target_speakers_forces_count() {
    let e: Vec<_> = (0..5).map(unit).collect(); // 5 orthogonal
    let r = cluster(
      &e,
      Linkage::Average,
      &opt_agg(Linkage::Average).with_target_speakers(2),
    )
    .unwrap();
    let unique: std::collections::HashSet<_> = r.iter().copied().collect();
    assert_eq!(unique.len(), 2);
  }
}
