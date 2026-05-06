//! Error type for `diarization::embed`.

#[cfg(feature = "ort")]
use std::path::PathBuf;

use thiserror::Error;

/// Errors returned by `diarization::embed` APIs.
#[derive(Debug, Error)]
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
  /// is empty or has no active frames. Both backends would feed
  /// all-zero pooling weights into statistics pooling and produce
  /// NaN from the division — surface it as a typed boundary error
  /// instead of letting NaN flow into PLDA/clustering.
  #[error("frame_mask is empty or has no active frames")]
  EmptyOrInactiveMask,

  /// `chunk_samples.len()` passed to
  /// `EmbedModel::embed_chunk_with_frame_mask` doesn't match the
  /// pyannote-style 10s chunk size (`segment::WINDOW_SAMPLES`).
  /// The ORT/tch backends compute fbank from the whole chunk and
  /// feed it to a pooling layer expecting fixed geometry; a non-
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
  /// (`segment::FRAMES_PER_WINDOW`). The backends pass `frame_mask`
  /// directly as the pooling-layer weights dimension; an off-by-one
  /// or sample-level mask changes the integration window and produces
  /// a finite-but-wrong embedding.
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

  /// `kaldi-native-fbank` initialization failed with this message.
  /// `FbankComputer::new` returns `Result<Self, String>`; we wrap
  /// the message verbatim. This is effectively unreachable with our
  /// fixed configuration but kept as a fallible escape hatch in case
  /// a future kaldi-native-fbank version starts validating fields we
  /// currently rely on as no-ops.
  #[error("fbank computer initialization failed: {0}")]
  Fbank(String),

  /// ONNX inference output had an unexpected element count.
  #[error("inference scores length {got}, expected {expected}")]
  InferenceShapeMismatch {
    /// Element count the contract expects (`n * EMBEDDING_DIM`).
    expected: usize,
    /// Element count actually returned by the model.
    got: usize,
  },

  /// ONNX `session.run()` returned a zero-output `SessionOutputs`.
  /// Realistic causes are a malformed model export (no graph outputs)
  /// or ABI drift in `ort` itself. Without this typed error,
  /// `outputs[0]` would panic at the FFI boundary instead of
  /// surfacing as a recoverable error to library callers.
  #[cfg(feature = "ort")]
  #[cfg_attr(docsrs, doc(cfg(feature = "ort")))]
  #[error("inference returned no outputs (malformed model graph or ORT ABI drift)")]
  MissingInferenceOutput,

  /// ONNX inference output contained a NaN/`±inf` value. Realistic
  /// upstream causes are degraded ONNX providers, model corruption,
  /// or non-finite input that flows through ResNet without saturation.
  /// Owned/streaming offline diarization paths previously treated
  /// non-finite-norm embeddings as "inactive speaker" silently —
  /// this variant lets them surface the corruption instead.
  #[error("inference output contains non-finite values (NaN / +inf / -inf)")]
  NonFiniteOutput,

  /// ONNX inference output had an unexpected tensor shape (rank or per-axis size),
  /// even when the total element count would otherwise have matched. Catches
  /// silently corrupting layout drift like `[EMBEDDING_DIM, n]` or
  /// `[1, n * EMBEDDING_DIM]` from a custom/exporter-drifted model.
  #[cfg(feature = "ort")]
  #[cfg_attr(docsrs, doc(cfg(feature = "ort")))]
  #[error("inference output shape {got:?}, expected [{n}, {embedding_dim}]")]
  InferenceOutputShape {
    /// Actual shape from the ORT tensor.
    got: Vec<i64>,
    /// Batch dimension (clip count) the dispatcher passed in.
    n: usize,
    /// Per-row width the model is contracted to emit.
    embedding_dim: usize,
  },

  /// Load-time model shape verification failed.
  #[cfg(feature = "ort")]
  #[cfg_attr(docsrs, doc(cfg(feature = "ort")))]
  #[error("model {tensor} dims {got:?}, expected {expected:?}")]
  IncompatibleModel {
    /// Name of the tensor whose shape is wrong (e.g. `"input"` /
    /// `"output"`).
    tensor: &'static str,
    /// Shape the dia contract expects.
    expected: &'static [i64],
    /// Shape the loaded ONNX file actually declares.
    got: Vec<i64>,
  },

  /// Failed to load the ONNX model from disk.
  #[cfg(feature = "ort")]
  #[cfg_attr(docsrs, doc(cfg(feature = "ort")))]
  #[error("failed to load model from {path}: {source}", path = path.display())]
  LoadModel {
    /// Path to the ONNX file the loader attempted.
    path: PathBuf,
    /// Underlying error from `ort`.
    #[source]
    source: ort::Error,
  },

  /// Wrap an `ort::Error` from session/inference.
  #[cfg(feature = "ort")]
  #[cfg_attr(docsrs, doc(cfg(feature = "ort")))]
  #[error(transparent)]
  Ort(#[from] ort::Error),

  /// Failed to load a TorchScript module from disk.
  #[cfg(feature = "tch")]
  #[cfg_attr(docsrs, doc(cfg(feature = "tch")))]
  #[error("failed to load TorchScript model from {path}: {source}", path = path.display())]
  LoadTorchScript {
    /// Path to the TorchScript module the loader attempted.
    path: std::path::PathBuf,
    /// Underlying error from `tch`.
    #[source]
    source: tch::TchError,
  },

  /// Wrap a `tch::TchError` from inference.
  #[cfg(feature = "tch")]
  #[cfg_attr(docsrs, doc(cfg(feature = "tch")))]
  #[error(transparent)]
  Tch(#[from] tch::TchError),
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

  #[test]
  fn fbank_message() {
    let e = Error::Fbank("bad mel config".to_string());
    let s = format!("{e}");
    assert!(s.contains("fbank computer initialization failed"));
    assert!(s.contains("bad mel config"));
  }
}
