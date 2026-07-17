#!/bin/bash
set -ex

# AVX-512F correctness via Intel SDE (Software Development Emulator).
#
# Free GitHub runners are AMD EPYC Milan or older Intel Xeons — neither
# reliably has AVX-512. Without this job, a reduction-or-load mistake
# in the unsafe AVX-512 path (`src/ops/arch/x86_avx512/`) would only
# surface on production AVX-512 hosts (Sapphire Rapids, Zen 4, etc.).
# SDE emulates the AVX-512 ISA in software so the dispatcher's runtime
# feature check picks the AVX-512 path under emulation and the
# differential tests in `ops::` exercise it.
#
# Slowdown vs native: ~10-50× depending on workload. The `ops::` test
# filter scopes to ~12 differential / panic / boundary tests with total
# runtime well under a minute even under emulation.
#
# Pattern mirrors siglip2's `avx512-sde` CI job.

TARGET="x86_64-unknown-linux-gnu"

SDE_URL="https://downloadmirror.intel.com/843185/sde-external-9.48.0-2024-11-25-lin.tar.xz"
wget -q "$SDE_URL" -O /tmp/sde.tar.xz
mkdir -p /tmp/sde
tar -xf /tmp/sde.tar.xz -C /tmp/sde --strip-components=1
export PATH="/tmp/sde:$PATH"
sde64 -version

# `-future` selects the widest emulated CPU (currently Sierra Forest /
# Granite Rapids — covers AVX-512F + BW + VL + DQ, which more than
# covers our `avx512f`-only kernels). The dispatcher's
# `is_x86_feature_detected!("avx512f")` will return true under
# emulation, and `cargo test` invocations get wrapped through
# `sde64 -future --` so each test process runs under the emulator.
#
# Test scope: `ops::` differential tests catch primitive-level ulp
# drift, but pyannote-parity tests under `pipeline::parity_tests`,
# `cluster::ahc::parity_tests`, `cluster::vbx::parity_tests`,
# `cluster::centroid::parity_tests`, `offline::parity_tests`, and
# `reconstruct::parity_tests` exercise the threshold-sensitive
# decisions (AHC `<= threshold` cuts, VBx alive-cluster gates,
# centroid argmax) that ulp drift could flip. We run all of them
# under SDE so an AVX-512-induced cluster decision flip is caught
# in CI rather than at runtime on AVX-512 hosts.
#
# `aggregate::parity_tests` is also included (count-tensor exact
# match) since the count loop is on the SIMD path under
# `aggregate::count`.
# `--cfg diarization_assert_avx512` enables the
# `dispatch_selects_avx512_under_sde` test in `ops::backend_selection_tests`,
# which fails the build if `avx512_available()` returns false under
# emulation. Without it, an SDE/CPUID regression would silently route the
# differential tests through the scalar fallback and report green.
RUSTFLAGS="-Dwarnings --cfg diarization_assert_avx512" \
CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUNNER="sde64 -future --" \
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
