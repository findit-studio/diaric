//! x86_64 AVX-512F kernels for the [`crate::ops`] primitives.
//!
//! 8-lane f64 (`__m512d`), FMA via `_mm512_fmadd_pd`, horizontal sum
//! via `_mm512_reduce_add_pd`. Dispatcher verifies AVX-512F at runtime
//! via [`crate::ops::avx512_available`]; pre-Skylake-X / pre-Zen 4
//! CPUs fall through to AVX2.
//!
//! AVX-512F is gated behind a nightly feature on stable Rust until
//! 1.89 (stabilized as of 1.89, May 2025). The crate's MSRV is 1.95
//! (Cargo.toml), so the intrinsics are available unconditionally on
//! the supported toolchain.

mod axpy;
mod dot;
mod pdist_euclidean;

pub(crate) use axpy::axpy;
pub(crate) use dot::dot;
pub(crate) use pdist_euclidean::pdist_euclidean;
