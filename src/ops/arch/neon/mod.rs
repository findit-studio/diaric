//! aarch64 NEON kernels for the [`crate::ops`] primitives.
//!
//! Each `pub(crate) unsafe fn` is annotated `#[target_feature(enable
//! = "neon")]` and assumes the caller has verified NEON availability
//! via [`crate::ops::neon_available`]. NEON is part of AArch64
//! baseline so this is essentially always-on, but the explicit gate
//! keeps the dispatcher pattern symmetric with x86 (where AVX2/AVX512
//! detection is mandatory).

mod axpy;
mod dot;
mod kahan;
mod pdist_euclidean;

pub(crate) use axpy::axpy;
pub(crate) use dot::dot;
pub(crate) use kahan::{kahan_dot, kahan_sum};
pub(crate) use pdist_euclidean::pdist_euclidean;
