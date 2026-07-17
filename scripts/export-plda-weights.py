"""Refresh the source PLDA `.npz` files from the upstream HuggingFace snapshot.

Copies `plda/xvec_transform.npz` and `plda/plda.npz` out of the
`pyannote/speaker-diarization-community-1` snapshot into `models/plda/`.
These two `.npz` files are the build-time source for the runtime `.bin`
blobs that `diaric` embeds via `include_bytes!`; regenerate the blobs
afterwards with:

  uv run --with numpy python scripts/extract-plda-blobs.sh          # 6 blobs
  uv run --with numpy --with scipy \
      python scripts/extract-plda-eigenvectors.py                   # 2 blobs

Run from anywhere (paths resolve relative to the repo root):

  uv run --with huggingface_hub python scripts/export-plda-weights.py

This is the PLDA-export leg lifted out of the `diarization` crate's
`tests/parity/python/capture_intermediates.py` (`_export_plda_weights`);
it lives with the `models/plda/` assets it maintains. It only needs
`huggingface_hub` — no torch/pyannote inference.
"""
from pathlib import Path

from huggingface_hub import snapshot_download

PIPELINE_NAME = "pyannote/speaker-diarization-community-1"


def export_plda_weights(repo_root: Path) -> None:
    """Copy plda/xvec_transform.npz + plda/plda.npz from the HF snapshot."""
    snap = Path(snapshot_download(PIPELINE_NAME))
    dst = repo_root / "models" / "plda"
    dst.mkdir(parents=True, exist_ok=True)
    for fname in ("xvec_transform.npz", "plda.npz"):
        src = snap / "plda" / fname
        if not src.exists():
            raise SystemExit(f"could not find {src} in HF snapshot")
        target = dst / fname
        target.write_bytes(src.read_bytes())
        print(f"[export-plda-weights] exported {fname} -> {target.relative_to(repo_root)}")


if __name__ == "__main__":
    export_plda_weights(Path(__file__).resolve().parents[1])
