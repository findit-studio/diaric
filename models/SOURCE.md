# Bundled model data

`diaric` embeds exactly one third-party model artifact into the compiled
binary via `include_bytes!`: the PLDA whitening weights under `plda/`. It
ships **no** ONNX/Torch model files.

The segmentation network (`pyannote/segmentation-3.0`), the WeSpeaker
ResNet34-LM embedding export, and the ONNX/Torch runners that load them
live in the `diarization` crate that depends on `diaric` — along with
their own download/refresh scripts and `NOTICE` entries. `diaric` neither
bundles those models nor exposes a `SegmentModel` / `EmbedModel` type.

## `plda/`

PLDA whitening + eigenvector weights from
`pyannote/speaker-diarization-community-1`, embedded by
`crate::plda::loader`. See [`plda/SOURCE.md`](plda/SOURCE.md) for the full
provenance, the array layout, and the refresh procedure.

- **License:** CC-BY-4.0 (BUT Speech@FIT; pyannote integration by
  Jiangyu Han and Petr Pálka).

Attribution is **required** in any redistributed binary: downstream
redistributors of any binary linking `diaric` must reproduce the CC-BY-4.0
PLDA attribution in [`NOTICE`](../NOTICE).
