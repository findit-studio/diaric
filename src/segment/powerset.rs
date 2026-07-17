//! Powerset → per-speaker probability decoding.
//!
//! pyannote/segmentation-3.0 outputs 7 logits per output frame, encoding
//! every subset of up to 3 simultaneous speakers:
//!
//! | class | meaning |
//! |-------|---------|
//! | 0     | silence |
//! | 1     | speaker A only |
//! | 2     | speaker B only |
//! | 3     | speaker C only |
//! | 4     | A + B   |
//! | 5     | A + C   |
//! | 6     | B + C   |
//!
//! Per-speaker probability is the marginal: speaker A is active iff class
//! 1, 4, or 5 fired. Voice (any speaker) probability is `1 - p(silence)`.

use crate::segment::options::POWERSET_CLASSES;

/// Numerically stable softmax over one row of [`POWERSET_CLASSES`] logits.
pub fn softmax_row(logits: &[f32; POWERSET_CLASSES]) -> [f32; POWERSET_CLASSES] {
  let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
  let mut out = [0f32; POWERSET_CLASSES];
  let mut sum = 0f32;
  for (i, &l) in logits.iter().enumerate() {
    let e = (l - max).exp();
    out[i] = e;
    sum += e;
  }
  debug_assert!(sum > 0.0);
  for v in out.iter_mut() {
    *v /= sum;
  }
  out
}

/// Per-speaker probabilities `[p(A), p(B), p(C)]` from a softmaxed
/// [`POWERSET_CLASSES`] row.
pub fn powerset_to_speakers(probs: &[f32; POWERSET_CLASSES]) -> [f32; 3] {
  [
    probs[1] + probs[4] + probs[5],
    probs[2] + probs[4] + probs[6],
    probs[3] + probs[5] + probs[6],
  ]
}

/// Pyannote's `to_multilabel(powerset, soft=False)`: argmax over the
/// 7 powerset classes, then look up each speaker's hard 0/1
/// activation. Mirrors `pyannote/audio/utils/powerset.py:115-140`.
///
/// Class index → speaker mask:
///   0 (silence) → (0, 0, 0)
///   1 (A)       → (1, 0, 0)
///   2 (B)       → (0, 1, 0)
///   3 (C)       → (0, 0, 1)
///   4 (A+B)     → (1, 1, 0)
///   5 (A+C)     → (1, 0, 1)
///   6 (B+C)     → (0, 1, 1)
///
/// Output is *hard* — every entry is exactly 0.0 or 1.0. Use this
/// in the segmentation aggregation path; pyannote's downstream
/// `filter_embeddings` / `count` / `reconstruct` all assume binary
/// values, and the soft marginals from
/// [`powerset_to_speakers`] disagree with hard argmax near 3-way
/// overlaps where the marginal sum-then-threshold flags a speaker
/// active when argmax would pick a different class entirely.
pub fn powerset_to_speakers_hard(probs: &[f32; POWERSET_CLASSES]) -> [f32; 3] {
  let mut argmax = 0usize;
  let mut max = probs[0];
  for (k, &p) in probs.iter().enumerate().skip(1) {
    if p > max {
      max = p;
      argmax = k;
    }
  }
  const TABLE: [[f32; 3]; POWERSET_CLASSES] = [
    [0.0, 0.0, 0.0], // silence
    [1.0, 0.0, 0.0], // A
    [0.0, 1.0, 0.0], // B
    [0.0, 0.0, 1.0], // C
    [1.0, 1.0, 0.0], // A+B
    [1.0, 0.0, 1.0], // A+C
    [0.0, 1.0, 1.0], // B+C
  ];
  TABLE[argmax]
}

/// Voice probability (= `1 - p(silence)`) for one softmaxed row.
pub(crate) fn voice_prob(probs: &[f32; POWERSET_CLASSES]) -> f32 {
  1.0 - probs[0]
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn softmax_row_sums_to_one() {
    let logits = [-1.0, 2.0, 0.5, 1.5, -0.3, 0.0, 0.7];
    let p = softmax_row(&logits);
    let s: f32 = p.iter().sum();
    assert!((s - 1.0).abs() < 1e-6);
    for &v in &p {
      assert!((0.0..=1.0).contains(&v));
    }
  }

  #[test]
  fn softmax_row_stable_with_extreme_logits() {
    let logits = [1000.0, 1001.0, 999.0, 1000.5, 998.0, 1000.2, 999.8];
    let p = softmax_row(&logits);
    let s: f32 = p.iter().sum();
    assert!((s - 1.0).abs() < 1e-5);
    assert!(p.iter().all(|v| v.is_finite()));
  }

  #[test]
  fn powerset_pure_silence() {
    let probs = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let s = powerset_to_speakers(&probs);
    assert_eq!(s, [0.0, 0.0, 0.0]);
    assert_eq!(voice_prob(&probs), 0.0);
  }

  #[test]
  fn powerset_pure_speaker_a() {
    let probs = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let s = powerset_to_speakers(&probs);
    assert_eq!(s, [1.0, 0.0, 0.0]);
    assert_eq!(voice_prob(&probs), 1.0);
  }

  #[test]
  fn powerset_a_and_b_overlap() {
    // 50% A+B, 50% silence
    let probs = [0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0];
    let s = powerset_to_speakers(&probs);
    assert!((s[0] - 0.5).abs() < 1e-6);
    assert!((s[1] - 0.5).abs() < 1e-6);
    assert_eq!(s[2], 0.0);
    assert!((voice_prob(&probs) - 0.5).abs() < 1e-6);
  }

  #[test]
  fn powerset_marginals_sum_correctly() {
    // 0.1 silence, 0.2 A, 0.1 B, 0.05 C, 0.3 A+B, 0.15 A+C, 0.1 B+C
    let probs = [0.1, 0.2, 0.1, 0.05, 0.3, 0.15, 0.1];
    let s = powerset_to_speakers(&probs);
    // p(A) = 0.2 + 0.3 + 0.15 = 0.65
    // p(B) = 0.1 + 0.3 + 0.10 = 0.50
    // p(C) = 0.05 + 0.15 + 0.10 = 0.30
    assert!((s[0] - 0.65).abs() < 1e-6);
    assert!((s[1] - 0.50).abs() < 1e-6);
    assert!((s[2] - 0.30).abs() < 1e-6);
    assert!((voice_prob(&probs) - 0.9).abs() < 1e-6);
  }
}
