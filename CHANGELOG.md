# UNRELEASED

# 0.1.0

Initial release: the backend-free diarization core extracted (history-preserving)
from the `diarization` crate — clustering (offline AHC→VBx, online), PLDA,
pipeline assembly, reconstruction/RTTM, kaldi-fbank DSP and embedding types, and
the SIMD/mmap numeric-ops layer. Carries no ONNX/Torch dependency; the model
runners remain in `diarization`.
