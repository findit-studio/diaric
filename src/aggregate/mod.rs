//! Pyannote-equivalent `Inference.aggregate` primitives.
//!
//! The **count tensor** computation drives
//! [`crate::reconstruct::reconstruct`]'s top-K binarization.
//!
//! ## What pyannote does
//!
//! `pyannote/audio/pipelines/speaker_diarization.py:_speaker_count`
//! (pyannote.audio 4.0.4):
//!
//! ```python
//! binarized = (segmentations >= onset)  # (chunks, frames_per_chunk, speakers) bool
//!
//! # 1) per-output-frame fraction of covering chunks where ANY speaker is active
//! activity = aggregate(any(binarized, axis=speaker), hamming=True, skip_average=True)
//!
//! # 2) per-output-frame hamming-weighted average of per-chunk active-speaker count
//! speaker_count_raw = aggregate(sum(binarized, axis=speaker), hamming=True, skip_average=True)
//!
//! # 3) normalize by activity (NOT by total weight) and round
//! count = round(speaker_count_raw / activity)
//! ```
//!
//! ## Why this matters for DER
//!
//! Earlier `OwnedDiarizationPipeline` divided by *total* hamming
//! weight rather than *activity-weighted* hamming aggregate. In
//! regions where some covering chunks see silence, dividing by total
//! weight systematically undercounts active speakers.
//!
//! Example: 2 covering chunks, A has 2 active speakers, B has 0.
//! - Wrong (total-weight): `(2·w_A + 0·w_B) / (w_A + w_B) ≈ 1` →
//!   count = 1, reconstruction emits only the most-active speaker.
//! - Pyannote (activity-weighted): `(2·w_A) / w_A = 2` → count = 2,
//!   both speakers emitted as expected.
//!
//! This was the dominant DER contributor on dia 5d (1.77–6.71% on
//! 5/6 captured fixtures vs pyannote's 0%). The pyannote-correct
//! formula closes the gap to bit-exact `count` on the captured
//! fixtures (verified by `aggregate::parity_tests`).
//!
//! No PIT alignment is needed for the count tensor — collapsing
//! speakers within each chunk via `sum`/`any` is permutation-
//! invariant. PIT is only required for per-speaker outputs (which
//! dia's pipeline doesn't use; it goes straight to AHC + VBx +
//! reconstruct on the speaker-permutation-arbitrary segmentations).

mod count;

#[cfg(test)]
mod parity_tests;

pub use count::{
  CountTensor, Error, MAX_OUTPUT_FRAMES, count_pyannote, hamming_aggregate,
  num_output_frames_pyannote, try_count_pyannote, try_hamming_aggregate,
  try_num_output_frames_pyannote,
};
