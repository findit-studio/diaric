//! Speaker fingerprint generation: WeSpeaker ResNet34 ONNX wrapper +
//! kaldi-compatible fbank + sliding-window mean for variable-length clips.
//!
//! See the crate-level docs and `docs/superpowers/specs/` for the design.
//! Layered API:
//! - High-level: `EmbedModel::embed`, `embed_weighted`, `embed_masked`
//! - Low-level: `compute_fbank`, `EmbedModel::embed_features`,
//!   `EmbedModel::embed_features_batch`

// `embedder` and `model` need to compile under either backend feature.
// `EmbedModel::from_torchscript_file` lives inside `model.rs` gated on
// `feature = "tch"`; if `model` is gated only on `ort`, a downstream
// build with `--no-default-features --features tch` cannot reach the
// TorchScript constructor at all.
#[cfg(any(feature = "ort", feature = "tch"))]
mod embedder;
mod error;
mod fbank;
#[cfg(any(feature = "ort", feature = "tch"))]
mod model;
mod options;
mod types;

pub use error::Error;
pub use fbank::{compute_fbank, compute_full_fbank};
#[cfg(any(feature = "ort", feature = "tch"))]
#[cfg_attr(docsrs, doc(cfg(any(feature = "ort", feature = "tch"))))]
pub use model::EmbedModel;
// `EmbedModelOptions` wraps `ort::SessionBuilder` knobs; it has no
// counterpart on the tch backend, so it stays ORT-only.
#[cfg(feature = "ort")]
#[cfg_attr(docsrs, doc(cfg(feature = "ort")))]
pub use options::EmbedModelOptions;
pub use options::{
  EMBED_WINDOW_SAMPLES, EMBEDDING_DIM, FBANK_FRAMES, FBANK_NUM_MELS, HOP_SAMPLES, MIN_CLIP_SAMPLES,
  NORM_EPSILON, SAMPLE_RATE_HZ,
};
pub use types::{Embedding, EmbeddingMeta, EmbeddingResult, cosine_similarity};

// Compile-time trait assertions. Catches a future field-type change that
// would silently regress Send/Sync auto-derive on the public types.
const _: fn() = || {
  fn assert_send_sync<T: Send + Sync>() {}
  assert_send_sync::<Embedding>();
  assert_send_sync::<EmbeddingMeta>();
  assert_send_sync::<EmbeddingResult>();
  assert_send_sync::<Error>();
};
