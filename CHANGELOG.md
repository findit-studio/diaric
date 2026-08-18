# UNRELEASED

## 0.2.0

CHANGED

- **`mediatime` `0.1` → `0.3`.** mediatime is a public dependency —
  `segment::SAMPLE_RATE_TB` is a `Timebase`, and `WindowId`,
  `SpeakerActivity` and `SegmentEvent::VoiceSpan` hand back `Timestamp`
  and `TimeRange` — so its breakage is diaric's breakage, and the crate
  version goes to `0.2.0` for it.
  - **`Timebase` is signed.** `num: u32 → i32` and
    `den: NonZeroU32 → NonZeroI32`, matching ffmpeg's `AVRational`, so
    `Timebase::new`, `num`, `den` and the `with_*` / `set_*` setters all
    change signature. `SAMPLE_RATE_TB` moves its denominator literal to
    `NonZeroI32`; `SAMPLE_RATE_HZ` stays `u32`, because it counts samples
    rather than dividing them. `Timebase::new` also panics now on a
    negative numerator or denominator — unreachable here, the one
    construction site being the `1 / 16_000` constant.
  - **No value in this crate moves.** 0.3's other two breaking changes
    are the `checked_`/`saturating_` ladder replacing the bare-name
    arithmetic, and rounding corrected from truncation to
    nearest-with-ties-away-from-zero (`AV_ROUND_NEAR_INF`). diaric never
    rescales, so the first has no call site here at all. The second
    reaches exactly one conversion — `WindowId::range().duration()`,
    which turns a tick count into a `Duration` — and at `1 / 16_000` a
    tick is exactly 62 500 ns, so every count converts exactly under
    either rounding. Checked over the first 200 000 ticks: no value
    moves. Window and voice-span boundaries stay exact sample indices,
    and the segment suites are unchanged.

# 0.1.0

Initial release: the backend-free diarization core extracted (history-preserving)
from the `diarization` crate — clustering (offline AHC→VBx, online), PLDA,
pipeline assembly, reconstruction/RTTM, kaldi-fbank DSP and embedding types, and
the SIMD/mmap numeric-ops layer. Carries no ONNX/Torch dependency; the model
runners remain in `diarization`.
