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
//! - For audio-in / RTTM-out, pair with [`OwnedDiarizationPipeline`]
//!   (under `feature = "ort"`), which calls the segmentation +
//!   embedding ONNX models for you and forwards into
//!   [`diarize_offline`].
//! - For an *incremental* push-style entrypoint (good for VAD-driven
//!   streaming where you produce voice ranges over time but only need
//!   one final RTTM), see
//!   [`crate::streaming::StreamingOfflineDiarizer`].
//!
//! ## What this module accepts
//!
//! [`OfflineInput`] takes pre-computed (segmentation, raw embedding)
//! tensors. The caller is responsible for running segmentation +
//! embedding ONNX inference. Two production sources:
//!
//! 1. The captured pyannote fixtures (`tests/parity/fixtures/*/`)
//!    — used by the parity tests in this module.
//! 2. Custom ONNX inference using [`crate::segment::SegmentModel`] +
//!    [`crate::embed::EmbedModel`].
//!
//! ## Why not feature-gate this behind `ort`
//!
//! The offline pipeline math is pure compute over [`f64`]/[`f32`]
//! tensors — no ONNX inference inside this function. It compiles and
//! runs without the `ort` feature. Useful for downstream consumers
//! that have their own inference path (e.g. CoreML, custom CUDA).

mod algo;

#[cfg(feature = "ort")]
mod owned;

#[cfg(test)]
mod parity_tests;

#[cfg(test)]
mod tests;

#[cfg(all(test, feature = "ort"))]
mod owned_smoke_tests;

pub use algo::{Error, OfflineInput, OfflineOutput, diarize_offline};

#[cfg(feature = "ort")]
#[cfg_attr(docsrs, doc(cfg(feature = "ort")))]
pub use owned::{OwnedDiarizationPipeline, OwnedPipelineOptions, SLOTS_PER_CHUNK};

/// Reused by [`crate::streaming::offline_diarizer`] for the same
/// onset / min_duration_off / smoothing_epsilon validation it
/// performs on its [`OwnedPipelineOptions`]-derived config. The two
/// reconstruction-knob predicates live in `algo` (always-on, not
/// ort-gated) because `diarize_offline` itself enforces them on the
/// pure tensor path; `check_onset` lives in `owned` because the
/// onset knob only flows through the audio entrypoints.
#[cfg(feature = "ort")]
pub(crate) use algo::{check_min_duration_off, check_smoothing_epsilon};
#[cfg(feature = "ort")]
pub(crate) use owned::check_onset;
