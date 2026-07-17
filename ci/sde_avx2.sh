#!/bin/bash
set -ex

# AVX2 + FMA correctness via Intel SDE (Software Development Emulator).
#
# Free GitHub runners are AMD EPYC (which has AVX2 + FMA natively) or
# Intel Xeon (varies). Even when the runner has AVX2 natively, our
# dispatcher prefers AVX-512F if `is_x86_feature_detected!("avx512f")`
# returns true — which on a modern Xeon WILL skip the AVX2 backend
# entirely. Without this job, a reduction-or-load mistake in the unsafe
# AVX2 path would only surface on AVX2-but-not-AVX-512 hosts (Haswell,
# Broadwell, Zen 1/2/3 — still common in the field). SDE pinned to a
# Haswell CPU model emulates AVX2 + FMA without AVX-512, forcing the
# dispatcher into the AVX2 branch under emulation.
#
# Slowdown vs native: ~10-50× depending on workload. The `ops::` test
# filter scopes to ~12 differential / panic / boundary tests with
# total runtime well under a minute even under emulation.
#
# Pattern mirrors siglip2's `avx512-sde` CI job.

TARGET="x86_64-unknown-linux-gnu"

# Pinned tarball from the public Intel mirror. Update the URL when
# bumping SDE — newer versions add coverage for newer CPU families.
SDE_URL="https://downloadmirror.intel.com/843185/sde-external-9.48.0-2024-11-25-lin.tar.xz"
wget -q "$SDE_URL" -O /tmp/sde.tar.xz
mkdir -p /tmp/sde
tar -xf /tmp/sde.tar.xz -C /tmp/sde --strip-components=1
export PATH="/tmp/sde:$PATH"
sde64 -version

# Run AVX2 SIMD tests under SDE-emulated Haswell (the first Intel CPU
# with AVX2 + FMA, no AVX-512). `-hsw` selects this CPU model.
#
# `--cfg diarization_disable_avx512` is a belt-and-suspenders: even on
# Haswell-emulation, the runtime feature detector should already return
# false for AVX-512F, but if SDE leaks any feature flag we still want
# the AVX2 branch exercised, not AVX-512. The cfg short-circuits
# `avx512_available()` to `false`.
#
# CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUNNER wraps each test binary
# invocation through `sde64 -hsw --` so the dispatcher's runtime
# `is_x86_feature_detected!("avx2")` and `is_x86_feature_detected!
# ("fma")` return true, while `is_x86_feature_detected!("avx512f")`
# returns false.
# Mirrors `ci/sde_avx512.sh`'s expanded test scope. Pyannote-parity
# tests run under SDE-emulated Haswell (AVX2 + FMA, no AVX-512) so
# AVX2-induced ulp drift on threshold-sensitive decisions surfaces
# in CI.
# `--cfg diarization_assert_avx2` enables the
# `dispatch_selects_avx2_under_sde` test in `ops::backend_selection_tests`,
# which fails the build if AVX2+FMA isn't selected (or if AVX-512 leaks
# through and the dispatcher picks AVX-512 instead of the AVX2 backend
# we want emulated).
RUSTFLAGS="-Dwarnings --cfg diarization_disable_avx512 --cfg diarization_assert_avx2" \
CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUNNER="sde64 -hsw --" \
cargo test \
  --lib \
  --target "$TARGET" \
  --no-default-features \
  -- \
  ops:: \
  embed::fbank::tests \
  pipeline::parity_tests \
  cluster::ahc::parity_tests \
  cluster::vbx::parity_tests \
  cluster::centroid::parity_tests \
  offline::parity_tests \
  reconstruct::parity_tests \
  aggregate::parity_tests \
  plda::parity_tests
