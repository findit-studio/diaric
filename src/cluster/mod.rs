//! Speaker clustering — generic offline batch [`cluster_offline`] plus
//! the pyannote `cluster_vbx`-pipeline primitives ([`ahc`], [`vbx`],
//! [`centroid`], [`hungarian`]).
//!
//! # Generic offline path
//! [`cluster_offline`] takes a slice of embeddings and returns a
//! `Vec<u64>` of speaker labels (one per embedding). Dispatches to
//! [`agglomerative`](OfflineMethod::Agglomerative) (Single / Complete /
//! Average linkage) or [`spectral`](OfflineMethod::Spectral) (default;
//! eigengap-K detection + K-means++ + Lloyd refinement, byte-deterministic
//! via [`ChaCha8Rng`](rand_chacha::ChaCha8Rng)).
//!
//! # Pyannote `cluster_vbx` primitives
//! The [`ahc`], [`vbx`], [`centroid`], and [`hungarian`] submodules are
//! the algorithm-level building blocks of the pyannote
//! `clustering.VBxClustering` pipeline. They're orchestrated by
//! [`crate::pipeline::assign_embeddings`] and
//! [`crate::offline::diarize_offline`]. Direct use is uncommon — the
//! pipeline / offline entrypoints are the supported API surface.

pub mod ahc;
pub mod centroid;
pub mod hungarian;
pub mod vbx;

mod error;
mod options;

pub use crate::embed::Embedding;
pub use error::Error;
pub use offline::cluster_offline;
pub use options::{
  DEFAULT_SIMILARITY_THRESHOLD, Linkage, MAX_AUTO_SPEAKERS, MAX_OFFLINE_INPUT,
  OfflineClusterOptions, OfflineMethod,
};

mod agglomerative;
mod offline;
mod spectral;

#[cfg(test)]
mod test_util;
#[cfg(test)]
mod tests;

// Compile-time trait assertions. Catches a future field-type change that
// would silently regress Send/Sync auto-derive on the public types.
//
// The submodule error types and `vbx::VbxOutput` (which wraps
// nalgebra's `DMatrix<f64>`) are also asserted here so a future
// refactor that adds a non-Send/Sync field (e.g. `Rc`, raw pointer)
// fails compilation at the type definition rather than only at the
// downstream `async`/`thread::spawn` call sites.
const _: fn() = || {
  fn assert_send_sync<T: Send + Sync>() {}
  assert_send_sync::<OfflineClusterOptions>();
  assert_send_sync::<Error>();
  assert_send_sync::<ahc::Error>();
  assert_send_sync::<vbx::Error>();
  assert_send_sync::<hungarian::Error>();
  assert_send_sync::<centroid::Error>();
  assert_send_sync::<vbx::VbxOutput>();
  assert_send_sync::<vbx::StopReason>();
};
