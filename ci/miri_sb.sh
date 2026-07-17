#!/bin/bash
set -e

if [ -z "$1" ]; then
  echo "Error: TARGET is not provided"
  exit 1
fi

TARGET="$1"

# Install cross-compilation toolchain on Linux
if [ "$(uname)" = "Linux" ]; then
  case "$TARGET" in
    aarch64-unknown-linux-gnu)
      sudo apt-get update && sudo apt-get install -y gcc-aarch64-linux-gnu
      ;;
    i686-unknown-linux-gnu)
      sudo apt-get update && sudo apt-get install -y gcc-multilib
      ;;
    powerpc64-unknown-linux-gnu)
      sudo apt-get update && sudo apt-get install -y gcc-powerpc64-linux-gnu
      ;;
    s390x-unknown-linux-gnu)
      sudo apt-get update && sudo apt-get install -y gcc-s390x-linux-gnu
      ;;
    riscv64gc-unknown-linux-gnu)
      sudo apt-get update && sudo apt-get install -y gcc-riscv64-linux-gnu
      ;;
  esac
fi

rustup toolchain install nightly --component miri
rustup override set nightly
cargo miri setup

export MIRIFLAGS="-Zmiri-strict-provenance -Zmiri-disable-isolation -Zmiri-symbolic-alignment-check"

# Same scope and configuration as `miri_tb.sh` under stacked-borrows:
# SIMD-only test filter (`ops::` + the `embed::fbank::tests` allowlist),
# scalar dispatcher forced via `--cfg diarization_force_scalar` (miri
# can't evaluate intrinsics), and per-backend direct unsafe-call tests
# skipped because they call NEON/SSE2/AVX2/AVX-512F kernels directly.
# See `miri_tb.sh` for the full rationale.
export RUSTFLAGS="${RUSTFLAGS:-} --cfg diarization_force_scalar"
cargo miri test \
  --lib --target "$TARGET" \
  -- \
  ops:: \
  embed::fbank::tests::dot_panics_on_length_mismatch_in_release \
  embed::fbank::tests::window_panics_on_length_mismatch_in_release \
  embed::fbank::tests::power_panics_on_length_mismatch_in_release \
  embed::fbank::tests::dot_kernels_agree_with_scalar \
  embed::fbank::tests::nan_propagates_through_log_floor \
  embed::fbank::tests::force_scalar_cfg_routes_through_scalar_when_set \
  embed::fbank::tests::shrink_before_resize_drops_oversized_when_call_small \
  embed::fbank::tests::shrink_before_resize_keeps_buffer_when_call_huge \
  embed::fbank::tests::shrink_before_resize_leaves_bounded_buffer \
  embed::fbank::tests::shrink_after_loop_drops_oversized \
  embed::fbank::tests::shrink_after_loop_keeps_bounded_buffer
