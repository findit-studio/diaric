//! Variational Bayes HMM speaker clustering (VBx).
//!
//! Ports `pyannote.audio.utils.vbx.VBx` (`utils/vbx.py:27-137` in
//! pyannote.audio 4.0.4) to Rust. Consumes the post-PLDA features
//! produced by `diarization::plda::PldaTransform::project()` plus the
//! eigenvalue diagonal `diarization::plda::PldaTransform::phi()`, runs
//! variational EM iterations, and returns final speaker
//! responsibilities + priors + ELBO trajectory.

#[cfg(test)]
pub(crate) mod algo;
#[cfg(not(test))]
mod algo;
mod error;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod parity_tests;

pub use algo::{MAX_ITERS_CAP, StopReason, VbxOutput, vbx_iterate};
pub use error::Error;
