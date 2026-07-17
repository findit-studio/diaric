<div align="center">
<h1>diaric</h1>
</div>
<div align="center">

The backend-free speaker-diarization core: clustering, PLDA, pipeline
assembly, reconstruction, fbank DSP and SIMD/mmap numeric ops — with **no
ONNX/Torch dependency**.

</div>

## What this is

`diaric` is the sans-I/O, model-runtime-free half of the
[`diarization`](https://github.com/findit-studio/diarization) pipeline. It
holds everything that computes over already-produced tensors and audio —
speaker clustering, PLDA projection, the offline diarization assembly,
frame reconstruction / RTTM emission, the kaldi-compatible fbank feature
extractor, and the shared numeric kernels — but **none** of the ONNX
Runtime (`ort`) or LibTorch (`tch`) model runners.

The segmentation and embedding **model runners** (`SegmentModel`,
`EmbedModel`) and the streaming service layer live in the `diarization`
crate, which depends on `diaric`. Consumers with their own inference path
(CoreML, custom CUDA, captured tensors) can depend on `diaric` directly and
never pull in a native ML runtime.

## Modules

| Module | What it does |
|---|---|
| `cluster` | Offline AHC → VBx clustering, spectral fallback, Hungarian assignment, and the online greedy centroid matcher (FluidAudio `SpeakerManager` port). |
| `plda` | PLDA projection over WeSpeaker embeddings. Weights are embedded via `include_bytes!` (`models/plda/*.bin`). |
| `pipeline` | `assign_embeddings` — the full AHC + VBx + centroid + constrained-Hungarian assignment stage. |
| `offline` | `diarize_offline` — batch pyannote-equivalent diarization over pre-computed (segmentation, raw-embedding) tensors. |
| `aggregate` / `reconstruct` | Speaker counting and frame-level reconstruction → RTTM spans. |
| `segment` | The sans-I/O `Segmenter` windowing/hysteresis state machine, powerset decoding, and option constants (the ONNX `SegmentModel` runner stays in `diarization`). |
| `embed` | The `Embedding` value types and the bit-exact torchaudio kaldi-fbank DSP (the ONNX `EmbedModel` runner stays in `diarization`). |
| `ops` / `spill` | Three-tier (scalar / NEON / AVX2 / AVX-512) numeric primitives and the file-backed mmap spill backend for pathological-size inputs. |
| `provenance` | Model/PLDA identity metadata. |

## Usage

Until published to crates.io, depend on a pinned git revision:

```toml
[dependencies]
diaric = { git = "https://github.com/findit-studio/diaric", rev = "..." }
```

Enable `serde` for `Serialize`/`Deserialize` on the public `*Options`
types.

## Cargo features

| Feature | Default | What it enables |
|---------|---------|-----------------|
| `serde` | no | `Serialize`/`Deserialize` impls for the public `*Options` types. `Duration` fields serialize as humantime strings ("250ms", "1.5s"). |
| `_bench` | no | Internal — exposes `pub(crate)` kernel modules to the `benches/*.rs` harnesses. Not part of the public API. |

There is deliberately **no** `ort` / `tch` / execution-provider feature: the
model runners that need them are in `diarization`.

## License

`diaric` is offered under the composite SPDX expression

```text
(MIT OR Apache-2.0) AND Apache-2.0 AND MIT AND CC-BY-4.0 AND BSD-2-Clause AND BSD-3-Clause
```

The `(MIT OR Apache-2.0)` branch is the original `diaric` code (caller's
choice); the remaining terms are mandatory obligations from the third-party
components this core vendors as source ports or embedded data. See the
`LICENSE-APACHE`, `LICENSE-MIT`, and `NOTICE` files for the full third-party
attribution record.

### LICENSE-MAPPING — which term covers what

| SPDX term | Component in `diaric` | Provenance |
|---|---|---|
| **MIT OR Apache-2.0** | The original `diaric` / `diarization` Rust code (caller's choice). | — |
| **Apache-2.0** | `cluster::online` — the greedy online centroid matcher, a source port of FluidAudio's `SpeakerManager`. A **mandatory** obligation, not the OR branch above: choosing MIT for the original code does not discharge it. | [FluidAudio](https://github.com/FluidInference/FluidAudio) (Apache-2.0), FluidInference Team. |
| **MIT** | The offline `cluster` flow (AHC/VBx/spectral/centroid), `pipeline`, `reconstruct`, `aggregate`, the `segment` post-processing, and the PLDA math — algorithm ports of `pyannote.audio`. | [pyannote/pyannote-audio](https://github.com/pyannote/pyannote-audio) (MIT) |
| **CC-BY-4.0** | `models/plda/*.bin` — PLDA weights **embedded into the compiled binary** via `include_bytes!` (`src/plda/loader.rs`). Attribution is **required** in any redistributed binary. | [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1); trained by BUT Speech@FIT. See [models/plda/SOURCE.md](models/plda/SOURCE.md). |
| **BSD-2-Clause** | `src/embed/fbank.rs` — a port of `torchaudio.compliance.kaldi.fbank` (torchaudio 2.11). torchaudio is BSD-2-Clause. | PyTorch/torchaudio (BSD-2-Clause), © 2017 Facebook Inc. (Soumith Chintala). |
| **BSD-3-Clause** | `src/cluster/hungarian/lsap.rs` (a port of SciPy's `rectangular_lsap.cpp`) and the scipy-derived PLDA eigenvector blobs (`eigenvectors_desc`, `phi_desc`). | SciPy `scipy.optimize` (BSD-3-Clause). |

> **Note on the adapted NOTICE.** The `NOTICE` file is adapted from
> `diarization`'s — materially, not carried verbatim: its preamble is
> rewritten for `diaric`'s footprint and it adds the pyannote.audio
> source-port attribution (section 7). Its `pyannote/segmentation-3.0`
> (bundled ONNX) and WeSpeaker entries describe **models the `diarization`
> crate bundles or loads — `diaric` ships no ONNX/Torch model files**. The
> only model artifact `diaric` embeds is the CC-BY-4.0 PLDA weight set
> above; a downstream binary linking `diaric` must reproduce that PLDA
> attribution and the source-port notices (pyannote, SciPy, torchaudio,
> FluidAudio).

Copyright (c) 2026 FinDIT studio authors.
