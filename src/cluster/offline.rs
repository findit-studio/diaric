//! Offline batch clustering entry point + shared helpers.
//! Spec §5.5 / §5.6.

use crate::{
  cluster::{
    Error, agglomerative,
    options::{Linkage, MAX_OFFLINE_INPUT, OfflineClusterOptions, OfflineMethod},
    spectral,
  },
  embed::{Embedding, NORM_EPSILON},
};

/// Validate inputs to [`cluster_offline`]. Returns the input length on
/// success. Shared between spectral (§5.5 step 0) and agglomerative
/// (§5.6 step 0) — same checks, same error variants, same order.
pub(crate) fn validate_offline_input(
  embeddings: &[Embedding],
  target_speakers: Option<u32>,
) -> Result<usize, Error> {
  if embeddings.is_empty() {
    return Err(Error::EmptyInput);
  }
  // Cap input size before any per-element work — both supported offline
  // methods allocate dense N×N matrices and the eigen / linkage paths
  // are O(N³). Without this guard, a long session's
  // `collected_embeddings` could OOM the process or block for minutes
  // before returning.
  if embeddings.len() > MAX_OFFLINE_INPUT {
    return Err(Error::InputTooLarge {
      n: embeddings.len(),
      limit: MAX_OFFLINE_INPUT,
    });
  }
  for e in embeddings {
    // f64 accumulator: 256 squared-f32 terms can lose ~8 bits of mantissa
    // in f32 (sum of values ~1.0). Promote for stability, demote at the
    // end. Mirrors online.rs::update_speaker. Not perf-critical — runs
    // once per embedding at validation time.
    let mut sq = 0.0f64;
    for &x in e.as_array() {
      if !x.is_finite() {
        return Err(Error::NonFiniteInput);
      }
      sq += (x as f64) * (x as f64);
    }
    if (sq.sqrt() as f32) < NORM_EPSILON {
      return Err(Error::DegenerateEmbedding);
    }
  }
  let n = embeddings.len();
  if let Some(k) = target_speakers {
    if k < 1 {
      return Err(Error::TargetTooSmall);
    }
    if (k as usize) > n {
      return Err(Error::TargetExceedsInput { target: k, n });
    }
  }
  Ok(n)
}

/// Cluster a batch of embeddings; returns one global speaker id per
/// input, parallel to the input slice.
///
/// Validates input first (empty list, non-finite values, zero-norm
/// embeddings, invalid `target_speakers`), then short-circuits the
/// `N==1` and `N==2` cases (spec §5.5 step 0.1, §5.6 step 0.1), then
/// dispatches to the configured [`OfflineMethod`].
pub fn cluster_offline(
  embeddings: &[Embedding],
  opts: &OfflineClusterOptions,
) -> Result<Vec<u64>, Error> {
  let n = validate_offline_input(embeddings, opts.target_speakers())?;
  // Defense-in-depth: `OfflineClusterOptions::with_similarity_threshold`
  // / `set_similarity_threshold` panic on out-of-range values, but a
  // `#[serde(default)]` deserialize bypasses those entry points and
  // can construct an `OfflineClusterOptions` whose `similarity_threshold`
  // reads NaN/±inf or > 1.0 / < -1.0 directly from JSON. Both the N==2
  // fast path (`sim >= threshold`) and agglomerative's stop distance
  // (`1 - threshold`) silently produce wrong clusterings under such
  // values — surface a typed error here before the algorithm runs.
  let t = opts.similarity_threshold();
  if !t.is_finite() || !(-1.0..=1.0).contains(&t) {
    return Err(Error::InvalidSimilarityThreshold(t));
  }

  // Fast paths (spec §5.5 step 0.1 / §5.6 step 0.1).
  if n == 1 {
    return Ok(vec![0]);
  }
  if n == 2 {
    let sim = embeddings[0].similarity(&embeddings[1]).max(0.0);
    return Ok(match opts.target_speakers() {
      Some(2) => vec![0, 1],
      Some(1) => vec![0, 0],
      _ => {
        if sim >= opts.similarity_threshold() {
          vec![0, 0]
        } else {
          vec![0, 1]
        }
      }
    });
  }

  // Dispatch.
  match opts.method() {
    OfflineMethod::Agglomerative { linkage } => agglomerative::cluster(embeddings, linkage, opts),
    OfflineMethod::Spectral => match spectral::cluster(embeddings, opts) {
      //: spectral's normalized Laplacian is
      // undefined when any node has zero positive affinity (an
      // orthogonal/antipodal outlier). Failing the whole batch on
      // a single outlier is hostile to post-hoc reclustering — fall
      // back to Agglomerative with Average linkage so the outlier
      // becomes its own speaker and the rest cluster normally.
      // `similarity_threshold` is honored by agglomerative, so the
      // user's threshold tuning still applies in this fallback.
      Err(Error::AllDissimilar) => agglomerative::cluster(embeddings, Linkage::Average, opts),
      other => other,
    },
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::embed::EMBEDDING_DIM;

  fn unit(i: usize) -> Embedding {
    let mut v = [0.0f32; EMBEDDING_DIM];
    v[i] = 1.0;
    Embedding::normalize_from(v).unwrap()
  }

  #[test]
  fn empty_input_errors() {
    let r = cluster_offline(&[], &OfflineClusterOptions::default());
    assert!(matches!(r, Err(Error::EmptyInput)));
  }

  #[test]
  fn target_speakers_zero_errors() {
    let r = cluster_offline(
      &[unit(0)],
      &OfflineClusterOptions::default().with_target_speakers(0),
    );
    assert!(matches!(r, Err(Error::TargetTooSmall)));
  }

  #[test]
  fn target_speakers_exceeds_input_errors() {
    let r = cluster_offline(
      &[unit(0), unit(1)],
      &OfflineClusterOptions::default().with_target_speakers(5),
    );
    assert!(matches!(
      r,
      Err(Error::TargetExceedsInput { target: 5, n: 2 })
    ));
  }

  #[test]
  fn fast_path_n_eq_1() {
    let r = cluster_offline(&[unit(0)], &OfflineClusterOptions::default()).unwrap();
    assert_eq!(r, vec![0]);
  }

  #[test]
  fn fast_path_n_eq_2_similar() {
    // Both identical → cosine = 1.0 >= 0.5 threshold → one cluster.
    let mut v = [0.0f32; EMBEDDING_DIM];
    v[0] = 1.0;
    let e = Embedding::normalize_from(v).unwrap();
    let r = cluster_offline(&[e, e], &OfflineClusterOptions::default()).unwrap();
    assert_eq!(r, vec![0, 0]);
  }

  #[test]
  fn fast_path_n_eq_2_dissimilar() {
    // Orthogonal → cosine = 0 < 0.5 → two clusters.
    let r = cluster_offline(&[unit(0), unit(1)], &OfflineClusterOptions::default()).unwrap();
    assert_eq!(r, vec![0, 1]);
  }

  #[test]
  fn fast_path_n_eq_2_target_forces() {
    let r1 = cluster_offline(
      &[unit(0), unit(0)],
      &OfflineClusterOptions::default().with_target_speakers(2),
    )
    .unwrap();
    assert_eq!(
      r1,
      vec![0, 1],
      "target=2 forces 2 clusters even when identical"
    );
    let r2 = cluster_offline(
      &[unit(0), unit(1)],
      &OfflineClusterOptions::default().with_target_speakers(1),
    )
    .unwrap();
    assert_eq!(
      r2,
      vec![0, 0],
      "target=1 forces 1 cluster even when orthogonal"
    );
  }

  #[test]
  fn nan_input_errors() {
    let mut v = [0.0f32; EMBEDDING_DIM];
    v[0] = f32::NAN;
    // Bypass the public Embedding constructor which would reject NaN.
    let e = Embedding(v);
    let r = cluster_offline(&[e, unit(0)], &OfflineClusterOptions::default());
    assert!(matches!(r, Err(Error::NonFiniteInput)));
  }

  #[test]
  fn zero_norm_input_errors() {
    let e = Embedding([0.0f32; EMBEDDING_DIM]);
    let r = cluster_offline(&[e, unit(0)], &OfflineClusterOptions::default());
    assert!(matches!(r, Err(Error::DegenerateEmbedding)));
  }

  #[test]
  fn validate_returns_n_on_valid_no_target() {
    let n = validate_offline_input(&[unit(0), unit(1), unit(2)], None).unwrap();
    assert_eq!(n, 3);
  }

  #[test]
  fn validate_returns_n_on_valid_with_target() {
    let n = validate_offline_input(&[unit(0), unit(1), unit(2)], Some(2)).unwrap();
    assert_eq!(n, 3);
  }

  /// regression: three orthogonal embeddings
  /// would trip spectral's `AllDissimilar` (each node has zero
  /// affinity to the others). The `OfflineMethod::Spectral` path now
  /// falls back to Agglomerative + Average so the outliers become
  /// distinct speakers rather than failing the whole batch.
  #[test]
  fn spectral_falls_back_on_all_dissimilar_no_target() {
    let inputs = vec![unit(0), unit(1), unit(2)];
    let opts = OfflineClusterOptions::default().with_method(OfflineMethod::Spectral);
    let labels = cluster_offline(&inputs, &opts).expect("fallback to agglomerative");
    // All three orthogonal → distinct labels.
    assert_eq!(labels.len(), 3);
    let unique: std::collections::HashSet<u64> = labels.iter().copied().collect();
    assert_eq!(
      unique.len(),
      3,
      "expected 3 distinct speakers, got {labels:?}"
    );
  }

  /// Same input but with `target_speakers = 2` — agglomerative's
  /// fallback respects the target by collapsing to two clusters.
  #[test]
  fn spectral_falls_back_on_all_dissimilar_with_target() {
    let inputs = vec![unit(0), unit(1), unit(2)];
    let opts = OfflineClusterOptions::default()
      .with_method(OfflineMethod::Spectral)
      .with_target_speakers(2);
    let labels = cluster_offline(&inputs, &opts).expect("fallback to agglomerative");
    let unique: std::collections::HashSet<u64> = labels.iter().copied().collect();
    assert_eq!(
      unique.len(),
      2,
      "target_speakers=2 must yield 2 clusters; got {labels:?}"
    );
  }

  #[test]
  fn input_too_large_errors() {
    //: dense offline methods must reject inputs
    // beyond MAX_OFFLINE_INPUT before allocating an N×N matrix. We
    // construct N+1 embeddings via repeating a known-good unit vector
    // — they all have identical contents, but the cap fires before
    // any per-element work runs (validation order matters).
    let one = unit(0);
    let inputs = vec![one; MAX_OFFLINE_INPUT + 1];
    let r = cluster_offline(&inputs, &OfflineClusterOptions::default());
    match r {
      Err(Error::InputTooLarge { n, limit }) => {
        assert_eq!(n, MAX_OFFLINE_INPUT + 1);
        assert_eq!(limit, MAX_OFFLINE_INPUT);
      }
      other => panic!("expected InputTooLarge, got {other:?}"),
    }
  }

  #[test]
  fn validate_target_equals_n_ok() {
    // target == n is allowed (every embedding can be its own cluster).
    let n = validate_offline_input(&[unit(0), unit(1)], Some(2)).unwrap();
    assert_eq!(n, 2);
  }

  ///: documents that `similarity_threshold` is
  /// IGNORED by `OfflineMethod::Spectral` for `N >= 3`. Two extreme
  /// thresholds must produce the same outcome (Ok labels OR Err);
  /// any drift would mean the docs lie. If a future revision wires
  /// the threshold into spectral (affinity pruning, K selection),
  /// this test should be updated rather than deleted.
  #[test]
  fn spectral_ignores_similarity_threshold_for_n_ge_3() {
    // Build 5 inputs with non-trivial affinity (mixing two basis
    // vectors per embedding) so spectral has a connected graph and
    // produces Ok labels. Pure orthogonal unit vectors would trip
    // the AllDissimilar guard for both runs and silently make this
    // test trivially pass.
    fn mix(a: usize, b: usize, w: f32) -> Embedding {
      let mut v = [0.0f32; EMBEDDING_DIM];
      v[a] = w;
      v[b] = (1.0 - w * w).sqrt();
      Embedding::normalize_from(v).unwrap()
    }
    let inputs = vec![
      mix(0, 1, 0.95),
      mix(0, 1, 0.93),
      mix(2, 3, 0.95),
      mix(2, 3, 0.91),
      mix(0, 2, 0.6),
    ];

    let opts_strict = OfflineClusterOptions::default()
      .with_method(OfflineMethod::Spectral)
      .with_similarity_threshold(0.99)
      .with_seed(42);
    let opts_loose = OfflineClusterOptions::default()
      .with_method(OfflineMethod::Spectral)
      .with_similarity_threshold(0.01)
      .with_seed(42);

    let labels_strict = cluster_offline(&inputs, &opts_strict).expect("strict ok");
    let labels_loose = cluster_offline(&inputs, &opts_loose).expect("loose ok");

    assert_eq!(
      labels_strict, labels_loose,
      "OfflineMethod::Spectral must produce identical labels regardless of \
       similarity_threshold for N >= 3 — the threshold is currently a no-op \
       for this method (see OfflineMethod docs). If this assertion fails, \
       the threshold has been wired into spectral and the docs need updating."
    );
  }

  /// `OfflineClusterOptions::with_similarity_threshold` /
  /// `set_similarity_threshold` panic on out-of-range values, but a
  /// serde-deserialized `OfflineClusterOptions` can carry a NaN/inf or
  /// outside-`[-1,1]` threshold directly. `cluster_offline` must
  /// reject this at the boundary, before the N==2 fast path or
  /// agglomerative stop-distance arithmetic silently produce wrong
  /// clusterings.
  #[cfg(feature = "serde")]
  #[test]
  fn cluster_offline_rejects_serde_bypassed_out_of_range_threshold() {
    let opts: OfflineClusterOptions =
      serde_json::from_str(r#"{"similarity_threshold": 2.0}"#).expect("deserialize");
    let r = cluster_offline(&[unit(0), unit(1)], &opts);
    assert!(
      matches!(r, Err(Error::InvalidSimilarityThreshold(t)) if t == 2.0),
      "got {r:?}"
    );

    let opts: OfflineClusterOptions =
      serde_json::from_str(r#"{"similarity_threshold": -1.5}"#).expect("deserialize");
    let r = cluster_offline(&[unit(0), unit(1)], &opts);
    assert!(
      matches!(r, Err(Error::InvalidSimilarityThreshold(t)) if t == -1.5),
      "got {r:?}"
    );
  }

  /// At the boundaries: `similarity_threshold == -1.0` and `== 1.0`
  /// are accepted (degenerate but well-defined).
  #[test]
  fn cluster_offline_accepts_boundary_thresholds() {
    let opts = OfflineClusterOptions::default().with_similarity_threshold(-1.0);
    let _ = cluster_offline(&[unit(0), unit(1)], &opts).expect("threshold = -1.0 ok");
    let opts = OfflineClusterOptions::default().with_similarity_threshold(1.0);
    let _ = cluster_offline(&[unit(0), unit(1)], &opts).expect("threshold = 1.0 ok");
  }
}
