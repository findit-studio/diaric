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

The six raw little-endian f64 blobs alongside the `.npz` files
(`mean1.bin`, `mean2.bin`, `lda.bin`, `mu.bin`, `tr.bin`, `psi.bin`)
are extracted by `scripts/extract-plda-blobs.sh`. They are the actual
runtime data — `diarization::plda` embeds them via `include_bytes!`, so the
production Rust path needs no `.npz` reader and no file I/O. Total
size on disk ~390 KB; binary delta the same.

| blob | shape | size (bytes) |
|------|-------|--------------|
| `mean1.bin` | (256,) | 2 048 |
| `mean2.bin` | (128,) | 1 024 |
| `lda.bin` | (256, 128) row-major | 262 144 |
| `mu.bin` | (128,) | 1 024 |
| `tr.bin` | (128, 128) row-major | 131 072 |
| `psi.bin` | (128,) | 1 024 |

The `.npz` files remain checked in — `tests/parity_plda.rs` loads
them via `npyz` (a dev-only dependency) to cross-check the embedded
blobs against the upstream-numpy reference.

## Refresh

Two-step refresh:

1. Re-run `tests/parity/python/capture_intermediates.py` against any
   clip under `tests/parity/fixtures/`. The `_export_plda_weights`
   step re-fetches the HuggingFace snapshot and overwrites the
   `.npz` files in this directory.
2. Run `scripts/extract-plda-blobs.sh` to regenerate the six `.bin`
   files from the new `.npz` files. Re-run `cargo test` to confirm
   `diarization::plda`'s parity tests still pass against the refreshed
   captures.
