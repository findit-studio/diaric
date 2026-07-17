"""Derive PLDA eigenvectors_desc + phi_desc from `models/plda/plda.npz`
using scipy's `eigh` exactly the way pyannote.audio does in
`pyannote/audio/utils/vbx.py:vbx_setup`. Saves to:

  models/plda/eigenvectors_desc.bin   (128 * 128 * 8 bytes, row-major f64)
  models/plda/phi_desc.bin            (128 * 8 bytes, f64)

Run from anywhere (paths resolve relative to the repo root):

  uv run --with numpy --with scipy python scripts/extract-plda-eigenvectors.py

Why we precompute these instead of running scipy/nalgebra at runtime:
LAPACK eigenvector signs are implementation-defined and nalgebra's
SymmetricEigen disagrees with scipy on 67 of 128 column signs for the
community-1 weights. A flipped sign in `plda_eigenvectors_desc[:, d]`
gives a sign-flipped `post_plda[:, d]`, which feeds VBx asymmetrically
(the `Lambda` ridge regression term is sign-sensitive in our
implementation), causing 38% DER divergence on fixture 04.
Hard-pinning scipy's exact eigenvectors removes the LAPACK-version
dependency entirely.
"""
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

PLDA = Path(__file__).resolve().parents[1] / "models" / "plda"

z = np.load(PLDA / "plda.npz")
plda_tr = z['tr']
plda_psi = z['psi']

# pyannote's exact setup (vbx.py:202-208, verbatim).
W = np.linalg.inv(plda_tr.T.dot(plda_tr))
B = np.linalg.inv((plda_tr.T / plda_psi).dot(plda_tr))
acvar, wccn = eigh(B, W)

# Reverse to descending. wccn columns are eigenvectors.
eigvecs_desc = wccn[:, ::-1].copy()
phi_desc = acvar[::-1].copy()

assert eigvecs_desc.shape == (128, 128), eigvecs_desc.shape
assert phi_desc.shape == (128,), phi_desc.shape

# Save row-major (numpy C-order). Rust `bytes_to_row_major_matrix`
# reads `m[i, j] = bytes[i * 128 + j]`, matching this.
eigvecs_desc.astype(np.float64, order='C').tofile(PLDA / "eigenvectors_desc.bin")
phi_desc.astype(np.float64).tofile(PLDA / "phi_desc.bin")

print(f"phi_desc[:5] = {phi_desc[:5]}")
print(f"eigvecs_desc[:5, 0] = {eigvecs_desc[:5, 0]}")
print(f"wrote eigenvectors_desc.bin ({eigvecs_desc.nbytes} bytes)")
print(f"wrote phi_desc.bin ({phi_desc.nbytes} bytes)")
