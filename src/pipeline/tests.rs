//! Model-free unit tests for `diarization::pipeline`.

use crate::pipeline::{AssignEmbeddingsInput, assign_embeddings};
use nalgebra::DVector;

/// Pyannote one-cluster fast path (`clustering.py:588-594`): when
/// fewer than 2 active training embeddings survive `filter_embeddings`,
/// pyannote returns `hard_clusters = np.zeros((num_chunks,
/// num_speakers))`. The Rust port must do the same instead of
/// erroring — short clips, sparse speech, and single-usable-speaker
/// recordings all hit this path.
#[test]
fn assign_embeddings_returns_one_cluster_when_num_train_lt_2() {
  let num_chunks = 3;
  let num_speakers = 3;
  let embed_dim = 4;
  let plda_dim = 4;
  let num_frames = 8;
  let embeddings: Vec<f64> = vec![0.5; num_chunks * num_speakers * embed_dim];
  let segmentations = vec![0.5; num_chunks * num_frames * num_speakers];

  // num_train = 1: only one active embedding survives filter_embeddings.
  let post_plda: Vec<f64> = vec![0.1; 1 * plda_dim];
  let phi = DVector::<f64>::from_element(plda_dim, 1.0);
  let train_chunk_idx = vec![0usize];
  let train_speaker_idx = vec![0usize];
  let input = AssignEmbeddingsInput::new(
    &embeddings,
    embed_dim,
    num_chunks,
    num_speakers,
    &segmentations,
    num_frames,
    &post_plda,
    plda_dim,
    &phi,
    &train_chunk_idx,
    &train_speaker_idx,
  );
  let got = assign_embeddings(&input).expect("fast path must succeed, not error");
  assert_eq!(got.len(), num_chunks);
  for chunk_row in got.iter() {
    assert_eq!(chunk_row.len(), num_speakers);
    for &k in chunk_row {
      assert_eq!(k, 0, "every speaker in every chunk must be cluster 0");
    }
  }
}

/// Dimension products at the public boundary use `checked_mul` —
/// otherwise `num_chunks * num_speakers` (or
/// `num_chunks * num_frames * num_speakers`) would wrap silently in
/// release builds, letting a malformed caller match the equality
/// checks with a tiny buffer and reach the `num_train < 2` fast
/// path with bogus shape metadata.
#[test]
fn rejects_overflowing_chunks_times_speakers() {
  let num_chunks = usize::MAX / 2 + 2;
  let num_speakers = 3;
  let embed_dim = 4;
  let plda_dim = 4;
  let num_frames = 8;
  // We never actually allocate `num_chunks * num_speakers` rows —
  // we expect the boundary check to fail first. nalgebra DMatrix
  // construction must succeed with some small shape so we can hand
  // it to the validator; the validator rejects on
  // `embeddings.nrows() != checked_mul(...)?`.
  let embeddings: Vec<f64> = vec![0.5; 4 * embed_dim];
  let segmentations = vec![0.5; 4 * num_frames];
  let post_plda: Vec<f64> = vec![0.1; 2 * plda_dim];
  let phi = DVector::<f64>::from_element(plda_dim, 1.0);
  let train_chunk_idx = vec![0usize, 1];
  let train_speaker_idx = vec![0usize, 1];
  let input = AssignEmbeddingsInput::new(
    &embeddings,
    embed_dim,
    num_chunks,
    num_speakers,
    &segmentations,
    num_frames,
    &post_plda,
    plda_dim,
    &phi,
    &train_chunk_idx,
    &train_speaker_idx,
  );
  let result = assign_embeddings(&input);
  assert!(
    matches!(result, Err(crate::pipeline::Error::Shape(_))),
    "got {result:?}"
  );
}

#[test]
fn rejects_overflowing_chunks_times_frames_times_speakers() {
  let num_chunks = 1 << 30;
  let num_frames = 1 << 30;
  let num_speakers = 1 << 30; // product overflows usize on 64-bit
  let embed_dim = 4;
  let plda_dim = 4;
  let embeddings: Vec<f64> = vec![0.5; 4 * embed_dim];
  let segmentations = vec![0.5; 4]; // tiny; never matches the overflowed product
  let post_plda: Vec<f64> = vec![0.1; 2 * plda_dim];
  let phi = DVector::<f64>::from_element(plda_dim, 1.0);
  let train_chunk_idx = vec![0usize, 1];
  let train_speaker_idx = vec![0usize, 1];
  let input = AssignEmbeddingsInput::new(
    &embeddings,
    embed_dim,
    num_chunks,
    num_speakers,
    &segmentations,
    num_frames,
    &post_plda,
    plda_dim,
    &phi,
    &train_chunk_idx,
    &train_speaker_idx,
  );
  let result = assign_embeddings(&input);
  assert!(
    matches!(result, Err(crate::pipeline::Error::Shape(_))),
    "got {result:?}"
  );
}

/// Zero-column `post_plda` is rejected at the boundary — a schema drift
/// or wrong array fed to the pipeline would otherwise let VBx iterate
/// on no PLDA evidence and produce plausible hard_clusters from prior
/// alone.
#[test]
fn rejects_zero_column_post_plda() {
  let num_chunks = 3;
  let num_speakers = 3;
  let embed_dim = 4;
  let plda_dim = 0;
  let num_frames = 8;
  let embeddings: Vec<f64> = vec![0.5; num_chunks * num_speakers * embed_dim];
  let segmentations = vec![0.5; num_chunks * num_frames * num_speakers];
  // post_plda has zero columns (PLDA dim = 0). Length = 2 * 0 = 0.
  let post_plda: Vec<f64> = Vec::new();
  let phi = DVector::<f64>::zeros(0);
  let train_chunk_idx = vec![0usize, 1];
  let train_speaker_idx = vec![0usize, 1];
  let input = AssignEmbeddingsInput::new(
    &embeddings,
    embed_dim,
    num_chunks,
    num_speakers,
    &segmentations,
    num_frames,
    &post_plda,
    plda_dim,
    &phi,
    &train_chunk_idx,
    &train_speaker_idx,
  );
  let result = assign_embeddings(&input);
  assert!(
    matches!(result, Err(crate::pipeline::Error::Shape(_))),
    "got {result:?}"
  );
}

/// Zero active embeddings (`num_train == 0`) also takes the fast path —
/// pyannote's check is `< 2`, not `== 1`. Skipping AHC/VBx entirely
/// avoids the empty-mean NaN that would otherwise propagate from
/// `np.mean(empty, axis=0)`.
#[test]
fn assign_embeddings_returns_one_cluster_when_num_train_zero() {
  let num_chunks = 2;
  let num_speakers = 3;
  let embed_dim = 4;
  let plda_dim = 4;
  let num_frames = 8;
  let embeddings: Vec<f64> = vec![0.5; num_chunks * num_speakers * embed_dim];
  let segmentations = vec![0.5; num_chunks * num_frames * num_speakers];
  // num_train = 0 ⇒ post_plda length = 0 * plda_dim = 0.
  let post_plda: Vec<f64> = Vec::new();
  let phi = DVector::<f64>::from_element(plda_dim, 1.0);
  let input = AssignEmbeddingsInput::new(
    &embeddings,
    embed_dim,
    num_chunks,
    num_speakers,
    &segmentations,
    num_frames,
    &post_plda,
    plda_dim,
    &phi,
    &[],
    &[],
  );
  let got = assign_embeddings(&input).expect("zero-train fast path must succeed");
  for chunk_row in got.iter() {
    for &k in chunk_row {
      assert_eq!(k, 0);
    }
  }
}

/// NaN/inf in the FULL embeddings matrix — including rows outside the
/// train subset — must surface `Error::NonFinite("embeddings")` at the
/// boundary, not silently flow into stage-6 cosine scoring where
/// Hungarian's `nan_to_num` would rewrite the resulting NaN cost to
/// global `nanmin` and produce a plausible-looking but wrong assignment.
#[test]
fn rejects_nan_in_non_train_embedding_row() {
  let num_chunks = 4;
  let num_speakers = 3;
  let embed_dim = 4;
  let plda_dim = 4;
  let num_frames = 8;
  let mut embeddings: Vec<f64> = vec![0.5; num_chunks * num_speakers * embed_dim];
  // Train subset is just the first 2 rows; corrupt a non-train row.
  // Row-major: row 7, col 1 → flat index `7 * embed_dim + 1`.
  embeddings[7 * embed_dim + 1] = f64::NAN;
  let segmentations = vec![0.5; num_chunks * num_frames * num_speakers];
  let post_plda: Vec<f64> = vec![0.1; 2 * plda_dim];
  let phi = DVector::<f64>::from_element(plda_dim, 1.0);
  let input = AssignEmbeddingsInput::new(
    &embeddings,
    embed_dim,
    num_chunks,
    num_speakers,
    &segmentations,
    num_frames,
    &post_plda,
    plda_dim,
    &phi,
    &[0usize, 1],
    &[0usize, 1],
  );
  let result = assign_embeddings(&input);
  assert!(
    matches!(
      result,
      Err(crate::pipeline::Error::NonFinite(
        crate::pipeline::error::NonFiniteField::Embeddings
      ))
    ),
    "expected NonFinite(Embeddings), got {result:?}"
  );
}

/// A row of finite-but-very-large values can overflow the squared-norm
/// accumulator (Σ v² → +∞) without any individual entry being non-
/// finite. Stage 6 reads every row for cosine scoring; an overflowing
/// non-train row would turn `dot(embedding, centroid)` into ±inf or
/// NaN, after which Hungarian's nan_to_num substitution silently
/// rewrites NaN to global nanmin and returns a plausible but wrong
/// assignment. Reject with a typed `RowNormOverflow` error.
#[test]
fn rejects_finite_row_with_overflowing_norm() {
  let num_chunks = 4;
  let num_speakers = 3;
  let embed_dim = 4;
  let plda_dim = 4;
  let num_frames = 8;
  // |v|² > f64::MAX/4 → sum of 4 such values overflows to +inf.
  let huge = 1e154_f64;
  let mut embeddings: Vec<f64> = vec![0.5; num_chunks * num_speakers * embed_dim];
  // Corrupt a non-train row (train subset is first 2 rows).
  // Row-major: row 8, all cols → flat indices `8 * embed_dim + c`.
  for c in 0..embed_dim {
    embeddings[8 * embed_dim + c] = huge;
  }
  let segmentations = vec![0.5; num_chunks * num_frames * num_speakers];
  let post_plda: Vec<f64> = vec![0.1; 2 * plda_dim];
  let phi = DVector::<f64>::from_element(plda_dim, 1.0);
  let input = AssignEmbeddingsInput::new(
    &embeddings,
    embed_dim,
    num_chunks,
    num_speakers,
    &segmentations,
    num_frames,
    &post_plda,
    plda_dim,
    &phi,
    &[0usize, 1],
    &[0usize, 1],
  );
  let result = assign_embeddings(&input);
  assert!(
    matches!(
      result,
      Err(crate::pipeline::Error::Shape(
        crate::pipeline::error::ShapeError::RowNormOverflow { row: 8 }
      ))
    ),
    "expected Shape(RowNormOverflow {{ row: 8 }}), got {result:?}"
  );
}

// Removed in round 8: `assign_embeddings_with_simd` is gone. The
// `use_simd` plumbing was deleted because:
// - AHC pdist is scalar in production (`ahc_init` calls
//   `ops::scalar::pdist_euclidean` directly) — threshold-sensitive.
// - Hungarian-feeding cosine is scalar in production
//   (`assign_embeddings` calls `ops::scalar::dot`) — argmax-sensitive.
// - VBx + centroid use SIMD (`ops::dot` / `ops::axpy`) but operate
//   continuously / iteratively, so ulp drift is non-discrete.
//
// Backend-differential coverage moved to `ops::differential_tests`
// at the primitive level.

/// Same precondition for `segmentations`: stage 7 sums all entries
/// for the inactive-speaker mask. A NaN in segmentations would make
/// `sum_activity` non-zero (NaN ≠ 0) for every speaker, defeating the
/// inactive-speaker override.(this
/// commit).
#[test]
fn rejects_nan_in_segmentations() {
  let num_chunks = 3;
  let num_speakers = 3;
  let embed_dim = 4;
  let plda_dim = 4;
  let num_frames = 8;
  let embeddings: Vec<f64> = vec![0.5; num_chunks * num_speakers * embed_dim];
  let mut segmentations = vec![0.5; num_chunks * num_frames * num_speakers];
  segmentations[10] = f64::INFINITY;
  let post_plda: Vec<f64> = vec![0.1; 2 * plda_dim];
  let phi = DVector::<f64>::from_element(plda_dim, 1.0);
  let input = AssignEmbeddingsInput::new(
    &embeddings,
    embed_dim,
    num_chunks,
    num_speakers,
    &segmentations,
    num_frames,
    &post_plda,
    plda_dim,
    &phi,
    &[0usize, 1],
    &[0usize, 1],
  );
  let result = assign_embeddings(&input);
  assert!(
    matches!(
      result,
      Err(crate::pipeline::Error::NonFinite(
        crate::pipeline::error::NonFiniteField::Segmentations
      ))
    ),
    "expected NonFinite(Segmentations), got {result:?}"
  );
}

/// Hyperparameter validation must run BEFORE the `num_train < 2`
/// fast path. Otherwise an invalid `threshold` / `fa` / `fb` /
/// `max_iters` returns `Ok(_)` on sparse / silent input and only
/// fails once enough speech accumulates — making option validation
/// data-dependent.
mod hyperparameter_validation_before_fast_path {
  use super::*;
  use crate::pipeline::error::ShapeError;

  fn input_with_zero_train<'a>(
    embeddings: &'a [f64],
    segmentations: &'a [f64],
    post_plda: &'a [f64],
    phi: &'a DVector<f64>,
  ) -> AssignEmbeddingsInput<'a> {
    // Zero-length train indices => num_train == 0 => fast path active.
    AssignEmbeddingsInput::new(
      embeddings,
      4, // embed_dim
      4, // num_chunks
      3, // num_speakers
      segmentations,
      8, // num_frames
      post_plda,
      4, // plda_dim
      phi,
      &[],
      &[],
    )
  }

  #[test]
  fn rejects_inf_threshold_even_on_fast_path() {
    let embeddings: Vec<f64> = vec![0.5; 4 * 3 * 4];
    let segmentations = vec![0.5; 4 * 8 * 3];
    let post_plda: Vec<f64> = Vec::new();
    let phi = DVector::<f64>::from_element(4, 1.0);
    let input = input_with_zero_train(&embeddings, &segmentations, &post_plda, &phi)
      .with_threshold(f64::INFINITY);
    let r = assign_embeddings(&input);
    assert!(
      matches!(
        r,
        Err(crate::pipeline::Error::Shape(ShapeError::InvalidThreshold))
      ),
      "got {r:?}"
    );
  }

  #[test]
  fn rejects_zero_threshold_even_on_fast_path() {
    let embeddings: Vec<f64> = vec![0.5; 4 * 3 * 4];
    let segmentations = vec![0.5; 4 * 8 * 3];
    let post_plda: Vec<f64> = Vec::new();
    let phi = DVector::<f64>::from_element(4, 1.0);
    let input =
      input_with_zero_train(&embeddings, &segmentations, &post_plda, &phi).with_threshold(0.0);
    let r = assign_embeddings(&input);
    assert!(
      matches!(
        r,
        Err(crate::pipeline::Error::Shape(ShapeError::InvalidThreshold))
      ),
      "got {r:?}"
    );
  }

  #[test]
  fn rejects_nan_fa_even_on_fast_path() {
    let embeddings: Vec<f64> = vec![0.5; 4 * 3 * 4];
    let segmentations = vec![0.5; 4 * 8 * 3];
    let post_plda: Vec<f64> = Vec::new();
    let phi = DVector::<f64>::from_element(4, 1.0);
    let input =
      input_with_zero_train(&embeddings, &segmentations, &post_plda, &phi).with_fa(f64::NAN);
    let r = assign_embeddings(&input);
    assert!(
      matches!(r, Err(crate::pipeline::Error::Shape(ShapeError::InvalidFa))),
      "got {r:?}"
    );
  }

  #[test]
  fn rejects_negative_fb_even_on_fast_path() {
    let embeddings: Vec<f64> = vec![0.5; 4 * 3 * 4];
    let segmentations = vec![0.5; 4 * 8 * 3];
    let post_plda: Vec<f64> = Vec::new();
    let phi = DVector::<f64>::from_element(4, 1.0);
    let input = input_with_zero_train(&embeddings, &segmentations, &post_plda, &phi).with_fb(-0.5);
    let r = assign_embeddings(&input);
    assert!(
      matches!(r, Err(crate::pipeline::Error::Shape(ShapeError::InvalidFb))),
      "got {r:?}"
    );
  }

  #[test]
  fn rejects_zero_max_iters_even_on_fast_path() {
    let embeddings: Vec<f64> = vec![0.5; 4 * 3 * 4];
    let segmentations = vec![0.5; 4 * 8 * 3];
    let post_plda: Vec<f64> = Vec::new();
    let phi = DVector::<f64>::from_element(4, 1.0);
    let input =
      input_with_zero_train(&embeddings, &segmentations, &post_plda, &phi).with_max_iters(0);
    let r = assign_embeddings(&input);
    assert!(
      matches!(
        r,
        Err(crate::pipeline::Error::Shape(ShapeError::ZeroMaxIters))
      ),
      "got {r:?}"
    );
  }

  #[test]
  fn rejects_max_iters_above_cap_even_on_fast_path() {
    let embeddings: Vec<f64> = vec![0.5; 4 * 3 * 4];
    let segmentations = vec![0.5; 4 * 8 * 3];
    let post_plda: Vec<f64> = Vec::new();
    let phi = DVector::<f64>::from_element(4, 1.0);
    let input = input_with_zero_train(&embeddings, &segmentations, &post_plda, &phi)
      .with_max_iters(crate::cluster::vbx::MAX_ITERS_CAP + 1);
    let r = assign_embeddings(&input);
    assert!(
      matches!(
        r,
        Err(crate::pipeline::Error::Shape(
          ShapeError::MaxItersExceedsCap { .. }
        ))
      ),
      "got {r:?}"
    );
  }

  /// Sanity: with valid hyperparameters, the `num_train < 2` fast
  /// path still returns `Ok` (cluster 0 for every (chunk, speaker)).
  #[test]
  fn fast_path_succeeds_with_valid_options() {
    let embeddings: Vec<f64> = vec![0.5; 4 * 3 * 4];
    let segmentations = vec![0.5; 4 * 8 * 3];
    let post_plda: Vec<f64> = Vec::new();
    let phi = DVector::<f64>::from_element(4, 1.0);
    let input = input_with_zero_train(&embeddings, &segmentations, &post_plda, &phi);
    let r = assign_embeddings(&input).expect("fast path with defaults must succeed");
    assert_eq!(r.len(), 4);
    for row in r.iter() {
      assert_eq!(*row, [0_i32; 3]);
    }
  }
}

/// Pre-AHC `num_train` cap rejects pathologically large inputs
/// upfront so AHC's `O(num_train² · embed_dim)` distance work cannot
/// run unbounded. The cap is sized at `MAX_AHC_TRAIN = 32_000`
/// (~3× the documented 1-hour intended scale of ~10k active pairs);
/// production loads pass through, but inputs an order of magnitude
/// past intended scale are rejected with a typed error.
#[cfg(test)]
mod ahc_train_cap_tests {
  use super::*;
  use crate::pipeline::{MAX_AHC_TRAIN, error::ShapeError};

  #[test]
  fn rejects_num_train_above_max_ahc_train() {
    // num_train = MAX_AHC_TRAIN + 1 = 32_001. Use small embed_dim so
    // the test allocates tiny buffers; the cap fires before any
    // pdist work.
    let num_train = MAX_AHC_TRAIN + 1;
    let num_speakers = 3;
    let num_chunks = num_train.div_ceil(num_speakers);
    let embed_dim = 4;
    let plda_dim = 4;
    let num_frames = 1;

    let emb: Vec<f64> = vec![0.5; num_chunks * num_speakers * embed_dim];
    let segmentations = vec![0.5; num_chunks * num_frames * num_speakers];
    let post_plda: Vec<f64> = vec![0.1; num_train * plda_dim];
    let phi = DVector::<f64>::from_element(plda_dim, 1.0);
    let mut train_chunk_idx = Vec::with_capacity(num_train);
    let mut train_speaker_idx = Vec::with_capacity(num_train);
    'outer: for c in 0..num_chunks {
      for s in 0..num_speakers {
        if train_chunk_idx.len() >= num_train {
          break 'outer;
        }
        train_chunk_idx.push(c);
        train_speaker_idx.push(s);
      }
    }
    let input = AssignEmbeddingsInput::new(
      &emb,
      embed_dim,
      num_chunks,
      num_speakers,
      &segmentations,
      num_frames,
      &post_plda,
      plda_dim,
      &phi,
      &train_chunk_idx,
      &train_speaker_idx,
    );
    let r = assign_embeddings(&input);
    assert!(
      matches!(
        r,
        Err(crate::pipeline::Error::Shape(ShapeError::AhcTrainSizeAboveMax { got, max }))
          if got == MAX_AHC_TRAIN + 1 && max == MAX_AHC_TRAIN
      ),
      "got {r:?}"
    );
  }
}

/// A NaN/`±inf` in `post_plda` must surface as
/// `NonFiniteField::PostPlda` *before* `assign_embeddings` allocates
/// `train_embeddings`, builds the L2-normalized AHC matrix, computes
/// the O(num_train²) condensed pdist, or runs linkage — the early
/// gate keeps the failure mode bounded regardless of input scale.
/// Without it, `vbx_iterate` would catch the non-finite value only
/// after all of that work.
#[cfg(test)]
mod post_plda_finiteness_early_gate {
  use super::*;
  use crate::pipeline::error::NonFiniteField;

  /// Build the smallest valid `assign_embeddings` input that drives
  /// AHC (`num_train >= 2`) and has well-formed shapes/finiteness on
  /// every other field. The only non-finite value sits in `post_plda`
  /// — the gate must reject before AHC runs.
  fn input_with_nonfinite_post_plda(
    post_plda: &[f64],
  ) -> (Vec<f64>, Vec<f64>, DVector<f64>, Vec<usize>, Vec<usize>) {
    let num_chunks = 1;
    let num_speakers = 3; // MAX_SPEAKER_SLOTS = 3
    let num_frames = 4;
    let embed_dim = 2;
    // Distinct embeddings so the train rows have non-zero L2 norm.
    let embeddings: Vec<f64> = (0..num_chunks * num_speakers * embed_dim)
      .map(|i| (i + 1) as f64)
      .collect();
    let segmentations: Vec<f64> = vec![0.5; num_chunks * num_frames * num_speakers];
    let phi = DVector::<f64>::from_element(post_plda.len() / 2, 1.0);
    let train_chunk_idx = vec![0_usize, 0_usize];
    let train_speaker_idx = vec![0_usize, 1_usize];
    (
      embeddings,
      segmentations,
      phi,
      train_chunk_idx,
      train_speaker_idx,
    )
  }

  #[test]
  fn rejects_nan_in_post_plda_before_ahc() {
    let plda_dim = 2;
    let num_train = 2;
    let mut post_plda = vec![0.1; num_train * plda_dim];
    post_plda[1] = f64::NAN; // single poison cell
    let (embeddings, segmentations, phi, train_chunk_idx, train_speaker_idx) =
      input_with_nonfinite_post_plda(&post_plda);
    let input = AssignEmbeddingsInput::new(
      &embeddings,
      2,
      1,
      3,
      &segmentations,
      4,
      &post_plda,
      plda_dim,
      &phi,
      &train_chunk_idx,
      &train_speaker_idx,
    );
    let r = assign_embeddings(&input);
    assert!(
      matches!(
        r,
        Err(crate::pipeline::Error::NonFinite(NonFiniteField::PostPlda))
      ),
      "expected NonFinite(PostPlda), got {r:?}"
    );
  }

  #[test]
  fn rejects_pos_inf_in_post_plda_before_ahc() {
    let plda_dim = 2;
    let num_train = 2;
    let mut post_plda = vec![0.1; num_train * plda_dim];
    post_plda[3] = f64::INFINITY;
    let (embeddings, segmentations, phi, train_chunk_idx, train_speaker_idx) =
      input_with_nonfinite_post_plda(&post_plda);
    let input = AssignEmbeddingsInput::new(
      &embeddings,
      2,
      1,
      3,
      &segmentations,
      4,
      &post_plda,
      plda_dim,
      &phi,
      &train_chunk_idx,
      &train_speaker_idx,
    );
    let r = assign_embeddings(&input);
    assert!(
      matches!(
        r,
        Err(crate::pipeline::Error::NonFinite(NonFiniteField::PostPlda))
      ),
      "expected NonFinite(PostPlda), got {r:?}"
    );
  }
}
