//! x86_64 AVX2 + FMA kernels for the [`crate::ops`] primitives.
//!
//! 4-lane f64 (`__m256d`), FMA via `_mm256_fmadd_pd`. The dispatcher
//! verifies AVX2 + FMA at runtime via [`crate::ops::avx2_available`]
//! before calling these kernels. CPUs that pre-date AVX2 (Haswell,
//! 2013-) fall through to scalar.
//!
//! This crate compiles on darwin/aarch64 dev machines via the
//! `target_arch = "x86_64"` cfg gate; the kernels are exercised in CI
//! (or any x86_64 host).

mod axpy;
mod dot;
mod pdist_euclidean;

pub(crate) use axpy::axpy;
pub(crate) use dot::dot;
pub(crate) use pdist_euclidean::pdist_euclidean;
