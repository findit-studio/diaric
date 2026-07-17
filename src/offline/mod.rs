//! Offline (non-streaming) diarization.
//!
//! Wraps the full pyannote `cluster_vbx` flow: PLDA projection on
//! active embeddings → AHC initial clustering → VBx EM → centroid
//! computation → cosine cdist + constrained Hungarian assignment →
//! frame-level reconstruction → RTTM emission. Bit-exact pyannote
//! parity on the 5 short captured fixtures.
//!
//! ## Where this fits
//!
//! - This module runs the full pyannote `community-1` clustering
//!   flow as a *batch* operation on already-computed segmentation +
//!   raw-embedding tensors. DER ≈ 0% on the 5 short captured
//!   fixtures (length-dependent divergence at T=1004; tracked
//!   separately).
//! - For audio-in / RTTM-out, pair with the `OwnedDiarizationPipeline`
//!   in the `diarization` crate, which calls the segmentation +
//!   embedding models for you and forwards into [`diarize_offline`].
//!
//! ## What this module accepts
//!
//! [`OfflineInput`] takes pre-computed (segmentation, raw embedding)
//! tensors. The caller is responsible for running segmentation +
//! embedding inference — e.g. via the `SegmentModel` / `EmbedModel`
//! runners in the `diarization` crate, or a custom CoreML/CUDA path.
//! Two production sources:
//!
//! 1. The captured pyannote fixtures (`tests/parity/fixtures/*/`)
//!    — used by the parity tests in this module.
//! 2. Custom inference producing the same tensor layout.
//!
//! ## Why this is backend-free
//!
//! The offline pipeline math is pure compute over [`f64`]/[`f32`]
//! tensors — no model inference inside this function. It compiles and
//! runs with no ONNX/Torch backend, so downstream consumers with their
//! own inference path (e.g. CoreML, custom CUDA) can drive it directly.

mod algo;

#[cfg(test)]
mod parity_tests;

#[cfg(test)]
mod tests;

pub use algo::{Error, OfflineInput, OfflineOutput, diarize_offline};
