//! Error type for `diarization::embed`.

use thiserror::Error;

/// Errors returned by `diarization::embed` APIs.
///
/// Marked `#[non_exhaustive]` so callers must include a `_ =>` arm in
/// any `match`. Variants in this enum represent low-level numerical /
/// boundary conditions (NaN/inf inputs, shape drift, …) and the set
/// evolves as new failure modes are surfaced or as internal kernels
/// stop being able to produce a given variant. The attribute lets us
/// add or retire variants without it being a semver-breaking change for
/// downstream exhaustive matchers.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Error {
  /// Input clip too short. Either `samples.len() < MIN_CLIP_SAMPLES`
  /// (for `embed`/`embed_weighted`) or the gathered length after
  /// applying a keep_mask in `embed_masked` was below the threshold.
  #[error("clip too short: {len} samples (need at least {min})")]
  InvalidClip {
    /// Actual sample count provided by the caller.
    len: usize,
    /// Minimum sample count required by the model
    /// (`MIN_CLIP_SAMPLES`).
    min: usize,
  },

  /// `voice_probs.len() != samples.len()` for `embed_weighted`.
  #[error("voice_probs.len() = {weights_len} must equal samples.len() = {samples_len}")]
  WeightShapeMismatch {
    /// Length of the audio sample slice.
    samples_len: usize,
    /// Length of the voice-probability slice the caller passed.
    weights_len: usize,
  },

  /// `voice_probs` contains a NaN, ±inf, negative value, or value
  /// `> 1.0`. Voice probabilities by contract live in `[0.0, 1.0]`
  /// and must be finite. NaN entries bypass the `total_weight <
  /// NORM_EPSILON` "all-silent" guard (every comparison with NaN is
  /// false) and contaminate the per-window mul_add. Out-of-range
  /// finite weights produce a signed-mixture aggregate that no longer
  /// represents a probability-weighted mean.
  #[error("voice_probs contains NaN/±inf/<0/>1; voice probabilities must be finite in [0.0, 1.0]")]
  InvalidVoiceProbs,

  /// `keep_mask.len() != samples.len()` for `embed_masked`.
  #[error("keep_mask.len() = {mask_len} must equal samples.len() = {samples_len}")]
  MaskShapeMismatch {
    /// Length of the audio sample slice.
    samples_len: usize,
    /// Length of the keep-mask slice.
    mask_len: usize,
  },

  /// All windows had near-zero voice-probability weight; the weighted
  /// average is undefined. Almost always caller error.
  #[error("all windows had effectively zero voice-activity weight")]
  AllSilent,

  /// `frame_mask` passed to `EmbedModel::embed_chunk_with_frame_mask`
  /// is empty or has no active frames. The embedding backend would feed
  /// all-zero pooling weights into statistics pooling and produce
  /// NaN from the division — surface it as a typed boundary error
  /// instead of letting NaN flow into PLDA/clustering.
  #[error("frame_mask is empty or has no active frames")]
  EmptyOrInactiveMask,

  /// `chunk_samples.len()` passed to
  /// `EmbedModel::embed_chunk_with_frame_mask` doesn't match the
  /// pyannote-style 10s chunk size (`segment::WINDOW_SAMPLES`).
  /// The embedding backend computes fbank from the whole chunk and
  /// feeds it to a pooling layer expecting fixed geometry; a non-
  /// pyannote-sized chunk produces a finite-but-wrong embedding
  /// that silently corrupts downstream PLDA/clustering.
  #[error(
    "chunk_samples.len() = {got}, expected {expected} (pyannote 10s @ 16 kHz = WINDOW_SAMPLES)"
  )]
  ChunkSamplesShapeMismatch {
    /// Expected sample count (`WINDOW_SAMPLES`).
    expected: usize,
    /// Actual sample count provided.
    got: usize,
  },

  /// `frame_mask.len()` passed to
  /// `EmbedModel::embed_chunk_with_frame_mask` doesn't match the
  /// pyannote-style 589-frame segmentation grid
  /// (`segment::FRAMES_PER_WINDOW`). The embedding backend passes
  /// `frame_mask` directly as the pooling-layer weights dimension; an
  /// off-by-one or sample-level mask changes the integration window and
  /// produces a finite-but-wrong embedding.
  #[error(
    "frame_mask.len() = {got}, expected {expected} (pyannote segmentation = FRAMES_PER_WINDOW)"
  )]
  FrameMaskShapeMismatch {
    /// Expected mask length (`FRAMES_PER_WINDOW`).
    expected: usize,
    /// Actual mask length provided.
    got: usize,
  },

  /// Input contains NaN or infinity.
  #[error("input contains non-finite values (NaN or infinity)")]
  NonFiniteInput,

  /// Input contains a zero-norm (or near-zero-norm, `< NORM_EPSILON`)
  /// embedding. Zero IS finite — kept distinct from `NonFiniteInput`
  /// so callers debugging real NaN/inf cases aren't misled.
  #[error("input contains a zero-norm or degenerate embedding")]
  DegenerateEmbedding,

  /// Inference output had an unexpected element count.
  #[error("inference scores length {got}, expected {expected}")]
  InferenceShapeMismatch {
    /// Element count the contract expects (`n * EMBEDDING_DIM`).
    expected: usize,
    /// Element count actually returned by the model.
    got: usize,
  },

  /// Inference output contained a NaN/`±inf` value. Realistic upstream
  /// causes are degraded inference providers, model corruption, or
  /// non-finite input that flows through ResNet without saturation.
  /// Owned/streaming offline diarization paths previously treated
  /// non-finite-norm embeddings as "inactive speaker" silently —
  /// this variant lets them surface the corruption instead.
  #[error("inference output contains non-finite values (NaN / +inf / -inf)")]
  NonFiniteOutput,
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn invalid_clip_message() {
    let e = Error::InvalidClip { len: 100, min: 400 };
    let s = format!("{e}");
    assert!(s.contains("100"));
    assert!(s.contains("400"));
  }

  #[test]
  fn mask_shape_mismatch_message() {
    let e = Error::MaskShapeMismatch {
      samples_len: 1000,
      mask_len: 999,
    };
    let s = format!("{e}");
    assert!(s.contains("1000"));
    assert!(s.contains("999"));
  }
}
