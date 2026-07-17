#!/usr/bin/env bash
# Extract PLDA weight arrays from `.npz` files into raw little-endian
# f64 binary blobs. The blobs are committed under `models/plda/` and
# embedded into the `diaric` binary via `include_bytes!` in
# `src/plda/loader.rs`.
#
# Run after `scripts/export-plda-weights.py` has refreshed
# `models/plda/*.npz` (or any other time the source `.npz` files change).
# The scipy-derived eigenvector blobs (`eigenvectors_desc.bin`,
# `phi_desc.bin`) come from the companion `scripts/extract-plda-eigenvectors.py`.
#
# Outputs (all little-endian f64, no headers):
#   models/plda/mean1.bin   (256,)        2 048 B
#   models/plda/mean2.bin   (128,)        1 024 B
#   models/plda/lda.bin     (256, 128)  262 144 B   (row-major)
#   models/plda/mu.bin      (128,)        1 024 B
#   models/plda/tr.bin      (128, 128)  131 072 B   (row-major)
#   models/plda/psi.bin     (128,)        1 024 B

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR/.."
PLDA="$ROOT/models/plda"

XVEC_NPZ="$PLDA/xvec_transform.npz"
PLDA_NPZ="$PLDA/plda.npz"

for f in "$XVEC_NPZ" "$PLDA_NPZ"; do
  if [ ! -f "$f" ]; then
    echo "[extract-plda-blobs] missing $f" >&2
    echo "[extract-plda-blobs] run scripts/export-plda-weights.py first" >&2
    exit 1
  fi
done

# Ephemeral numpy env via uv; no project checkout or venv required.
uv run --with numpy python - "$XVEC_NPZ" "$PLDA_NPZ" "$PLDA" <<'PY'
import sys
from pathlib import Path
import numpy as np

xvec_path, plda_path, out_dir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
out_dir.mkdir(parents=True, exist_ok=True)

EXPECTED = {
    "mean1": (xvec_path, (256,)),
    "mean2": (xvec_path, (128,)),
    "lda":   (xvec_path, (256, 128)),
    "mu":    (plda_path, (128,)),
    "tr":    (plda_path, (128, 128)),
    "psi":   (plda_path, (128,)),
}

for name, (path, expected_shape) in EXPECTED.items():
    arr = np.load(path)[name]
    assert arr.shape == expected_shape, (
        f"{name}: shape={arr.shape}, expected {expected_shape}"
    )
    # Coerce to little-endian f64 row-major contiguous, no copy if already there.
    out = np.ascontiguousarray(arr, dtype=np.dtype("<f8"))
    out_path = out_dir / f"{name}.bin"
    out_path.write_bytes(out.tobytes(order="C"))
    expected_bytes = int(np.prod(expected_shape)) * 8
    print(
        f"[extract-plda-blobs] {name:>5s}: shape={expected_shape} "
        f"bytes={out_path.stat().st_size} (expected {expected_bytes})"
    )
    assert out_path.stat().st_size == expected_bytes
print("[extract-plda-blobs] done")
PY
