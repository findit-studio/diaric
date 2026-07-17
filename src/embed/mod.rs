//! Speaker-embedding value types + the kaldi-compatible fbank DSP.
//!
//! The backend-free half of the embedding pipeline: the [`Embedding`]
//! output type and its aggregation metadata, plus the bit-exact
//! torchaudio kaldi-fbank feature extractor ([`compute_fbank`],
//! [`compute_full_fbank`]). The ONNX/Torch model runner that turns fbank
//! features into raw WeSpeaker embeddings (`EmbedModel`) lives in the
//! `diarization` crate.

mod error;
mod fbank;
mod options;
mod types;

pub use error::Error;
pub use fbank::{compute_fbank, compute_full_fbank};
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
