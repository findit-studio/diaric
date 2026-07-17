//! PLDA (Probabilistic Linear Discriminant Analysis) transform.
//!
//! Ports `pyannote.audio.utils.vbx.vbx_setup` plus its inner `xvec_tf`
//! / `plda_tf` lambdas (`utils/vbx.py:181-218` in pyannote.audio
//! 4.0.4) to Rust. Loads two `.npz` weight files (shipped with
//! `pyannote/speaker-diarization-community-1`, redistributed under
//! `models/plda/`) and exposes a deterministic two-stage projection:
//!
//! ```text
//! 256-d WeSpeaker embedding (f32)
//!         │
//!         ▼  xvec_transform
//! 128-d PLDA stage 1 (f64, sqrt(128)-scaled L2-norm; ‖·‖ ≈ 11.31)
//!         │
//!         ▼  plda_transform
//! 128-d PLDA stage 2 (f64, whitened — input to VBx)
//! ```
//!
//! ## Pinning
//!
//! The implementation tracks pyannote.audio 4.0.4 byte-for-byte via the
//! parity tests in `src/plda/parity_tests.rs` (a `#[cfg(test)]` module),
//! which validate against the captured artifacts under
//! `tests/parity/fixtures/01_dialogue/plda_embeddings.npz`. Bumping
//! pyannote requires re-running the capture and re-validating these
//! tests. Run with `cargo test plda::parity_tests`.

mod error;
mod loader;
mod transform;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod parity_tests;

pub use error::Error;
pub use transform::{PldaTransform, PostXvecEmbedding, RawEmbedding};

/// PLDA stage-1 / stage-2 dimension. Pyannote's
/// `pyannote/speaker-diarization-community-1` always uses 128.
pub const PLDA_DIMENSION: usize = 128;

/// WeSpeaker embedding dimension (input to `xvec_transform`).
pub const EMBEDDING_DIMENSION: usize = 256;
