# PLDA weights — pyannote/speaker-diarization-community-1

`xvec_transform.npz` and `plda.npz` are copied from the HuggingFace
snapshot of [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1).

- **License:** CC-BY-4.0. Attribution per upstream `plda/README.md`:
  PLDA model trained by [BUT Speech@FIT](https://speech.fit.vut.cz/);
  integration of VBx in pyannote.audio by Jiangyu Han and Petr Pálka.
- **Snapshot revision:** `3533c8cf8e369892e6b79ff1bf80f7b0286a54ee` (HF
  cache directory name on the machine where this snapshot was made).
- **Original layout in the HF repo:** `plda/xvec_transform.npz`,
  `plda/plda.npz`.

## File contents

`xvec_transform.npz` keys: `mean1` (256), `mean2` (128), `lda` (256×128).
Used by `xvec_tf` for centering + LDA + L2-norm + scale-by-sqrt(D_out).

`plda.npz` keys: `mu` (128), `tr` (128×128), `psi` (128).
Used by `plda_tf` for centering and whitening into the PLDA latent
space. `psi` (eigenvalues of the between-class covariance) is exposed
as `PLDA.phi` and consumed by VBx as the `Phi` parameter.

These two files together drive `pyannote.audio.utils.vbx.vbx_setup`,
which is invoked by `pyannote.audio.core.plda.PLDA.__init__` to build
the `_xvec_tf` / `_plda_tf` lambdas. The Rust port (Phase 1+) reads
the same files and must reproduce the same transformation; the
captured `post_xvec` / `post_plda` artifacts under
`tests/parity/fixtures/01_dialogue/plda_embeddings.npz` are the
reference output.

## Companion `.bin` files

The runtime data is a set of raw little-endian f64 blobs alongside the
`.npz` files. `diaric::plda` (`src/plda/loader.rs`) embeds them via
`include_bytes!`, so the production Rust path needs no `.npz` reader and
no file I/O.

Six are extracted from the two `.npz` sources by
`scripts/extract-plda-blobs.sh`:

| blob | shape | size (bytes) |
|------|-------|--------------|
| `mean1.bin` | (256,) | 2 048 |
| `mean2.bin` | (128,) | 1 024 |
| `lda.bin` | (256, 128) row-major | 262 144 |
| `mu.bin` | (128,) | 1 024 |
| `tr.bin` | (128, 128) row-major | 131 072 |
| `psi.bin` | (128,) | 1 024 |

Two more are the scipy-derived PLDA eigenvectors, precomputed offline by
`scripts/extract-plda-eigenvectors.py` (scipy's `eigh` sign convention is
pinned to remove the LAPACK-version dependency — see the rationale in
`src/plda/loader.rs`):

| blob | shape | size (bytes) |
|------|-------|--------------|
| `eigenvectors_desc.bin` | (128, 128) row-major | 131 072 |
| `phi_desc.bin` | (128,) | 1 024 |

The `.npz` files remain checked in as the build-time source for the six
extracted blobs (excluded from the published crate; regenerated via
`scripts/export-plda-weights.py`). The embedded blobs are cross-checked
against the captured pyannote reference (`plda_embeddings.npz`) by
`src/plda/parity_tests.rs`, using the dev-only `npyz` dependency.

## Refresh

Two-step refresh:

1. Run `scripts/export-plda-weights.py` (needs `huggingface_hub`) to
   re-fetch the HuggingFace snapshot and overwrite the `.npz` files in
   this directory.
2. Regenerate the `.bin` files from the refreshed `.npz`:
   `scripts/extract-plda-blobs.sh` (the six whitening blobs) and
   `scripts/extract-plda-eigenvectors.py` (the two eigenvector blobs).
   Re-run `cargo test` to confirm `diaric`'s PLDA parity tests
   (`src/plda/parity_tests.rs`) still pass against the captured
   references.
