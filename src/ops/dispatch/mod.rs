//! Public dispatchers for [`crate::ops`] primitives.
//!
//! Each dispatcher always selects the best-available SIMD backend
//! at runtime via `cfg_select!` arms guarded by `*_available()`
//! checks against [`crate::ops::arch`]. Callers needing scalar
//! output explicitly call [`crate::ops::scalar`].

mod axpy;
mod dot;
mod kahan;
mod lse;
mod pdist_euclidean;

pub use axpy::axpy;
#[cfg(any(feature = "ort", feature = "tch"))]
pub use axpy::axpy_f32;
pub use dot::dot;
pub use kahan::{kahan_dot, kahan_sum};
pub use lse::logsumexp_row;
#[cfg(any(test, feature = "_bench"))]
pub use pdist_euclidean::pdist_euclidean;
