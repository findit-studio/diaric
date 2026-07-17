#![doc = include_str!("../README.md")]
#![deny(missing_docs)]
// Several public algorithm entrypoints document their private
// implementation helpers via intra-doc links (e.g. `constrained_argmax`
// → the `lsap` port, the online matcher's `assign` → its internal
// centroid-update step). That is a deliberate documentation choice for a
// crate whose public surface is thin over a large private kernel layer.
#![allow(rustdoc::private_intra_doc_links)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(docsrs, allow(unused_attributes))]

pub mod cluster;
pub mod embed;
pub mod segment;

#[cfg(test)]
pub(crate) mod test_util;

// Numerical primitives shared across the algorithm modules. Three-tier
// backend layout (scalar/arch/dispatch) modeled on the colconv crate.
// Crate-private — algorithm modules call into `ops::*`; downstream
// callers don't see this layer. `_bench` flips it to `pub` so external
// benches in `benches/ops.rs` can A/B scalar vs SIMD on the primitives
// directly.
#[cfg_attr(feature = "_bench", doc(hidden))]
#[cfg(feature = "_bench")]
pub mod ops;
#[cfg(not(feature = "_bench"))]
pub(crate) mod ops;

/// f32 fused multiply-add primitive: `y[i] += alpha * x[i]` for each `i`.
///
/// The single numeric primitive `diaric` publishes across the crate
/// boundary. The WeSpeaker embedding aggregator in the `diarization`
/// crate sums per-window embeddings into a 256-d accumulator through this
/// exact function, so the two crates share one implementation and cannot
/// drift by re-deriving the aggregation arithmetic.
///
/// It is a scalar `f32::mul_add` loop — vectorization is left to the
/// compiler (build-dependent autovectorization), with no architecture-
/// specific kernel. It is therefore *distinct* from the SIMD-dispatched
/// f64 [`ops::axpy`](crate::ops::axpy) that the internal algorithm modules
/// use (e.g. centroid accumulation). See
/// [`ops::axpy_f32`](crate::ops::axpy_f32) for the (scalar) implementation
/// and panic contract.
pub use ops::axpy_f32;

/// Spill-buffer configuration types reachable from public API surfaces
/// (e.g. [`OfflineInput::with_spill_options`](crate::offline::OfflineInput::with_spill_options),
/// [`AssignEmbeddingsInput`](crate::pipeline::AssignEmbeddingsInput) and
/// [`ReconstructInput`](crate::reconstruct::ReconstructInput)).
///
/// The implementation lives in the crate-private `ops::spill` module;
/// this module is the public re-export so downstream callers can name
/// and construct the types they need.
///
/// Production deployments where `/tmp` is `tmpfs` (Docker default)
/// **must** override [`SpillOptions::with_spill_dir`](crate::spill::SpillOptions::with_spill_dir)
/// to a real-disk path — without it, "spill to disk" reduces to "spill
/// to RAM" and the OOM concern that motivates this whole subsystem is
/// unaddressed. That override is only possible because these types are
/// exposed here.
pub mod spill {
  pub use crate::ops::spill::{SpillBytes, SpillBytesMut, SpillError, SpillOptions};
}

pub mod plda;

pub mod provenance;

pub mod pipeline;

pub mod reconstruct;

pub mod aggregate;

pub mod offline;
