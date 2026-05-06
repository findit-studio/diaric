//! Overlap-add stitching of per-window voice probabilities.
//!
//! Storage and computation happen at **frame rate** (one entry per
//! `FRAMES_PER_WINDOW`-th of `WINDOW_SAMPLES`), not sample rate.
//! Per-hour-of-audio storage is ~1.7 MB at frame rate vs ~460 MB if we
//! expanded each frame to its sample range. See spec §5.4.
//!
//! Each window contributes `FRAMES_PER_WINDOW` voice probabilities to a
//! stream-indexed `(sum, count)` accumulator, anchored at
//! `frame_index_of(window.start_sample)`. Overlapping windows are averaged
//! sample-by-frame (yes, frame, not sample) on drain.

extern crate alloc;

use alloc::{collections::VecDeque, vec::Vec};

use crate::segment::options::{FRAMES_PER_WINDOW, WINDOW_SAMPLES};

/// Convert a frame index in `0..=FRAMES_PER_WINDOW` to a sample offset in
/// `0..=WINDOW_SAMPLES` using rounded integer arithmetic. Bit-for-bit
/// equivalent to `round(frame_idx * 160000 / 589)` for any integer
/// `frame_idx` (see spec §5.2.1).
///
/// **Use only for window-local offsets.** Absolute frame indices grow
/// with stream length and will exceed `u32::MAX` after ~74 hours at
/// 16 kHz; for those, use [`frame_to_sample_u64`] instead.
#[inline]
pub(crate) const fn frame_to_sample(frame_idx: u32) -> u32 {
  let n = frame_idx as u64 * WINDOW_SAMPLES as u64;
  let half = (FRAMES_PER_WINDOW as u64) / 2;
  ((n + half) / FRAMES_PER_WINDOW as u64) as u32
}

/// `frame_idx (u64) → sample (u64)` — same formula as
/// [`frame_to_sample`] but operates in `u64` end-to-end so it is safe
/// for absolute frame positions on long streams. Use this everywhere
/// the input frame index is an absolute (stream-wide) position; the
/// `u32` helper above truncates after ~74 h of audio at 16 kHz and
/// would silently wrap public timestamps.
///
/// Spec §15 #54.
#[inline]
pub(crate) const fn frame_to_sample_u64(frame_idx: u64) -> u64 {
  let n = frame_idx * WINDOW_SAMPLES as u64;
  let half = (FRAMES_PER_WINDOW as u64) / 2;
  (n + half) / FRAMES_PER_WINDOW as u64
}

/// Convert an absolute sample index to an absolute frame index using
/// **floor** rounding. The boundary in §5.4.1 demands floor: a frame is
/// "below boundary" only if NO future window can contribute to it, and any
/// rounding mode other than floor either over-finalizes (admits a frame a
/// future window will still touch) or under-finalizes (delays drain).
///
/// At step boundaries the conversion can land exactly on a half-integer
/// (e.g. sample 80_000 → 80_000 × 589 / 160_000 = 294.5). Floor returns 294.
#[inline]
pub(crate) const fn frame_index_of(sample_idx: u64) -> u64 {
  sample_idx * (FRAMES_PER_WINDOW as u64) / (WINDOW_SAMPLES as u64)
}

/// Stream-indexed per-frame voice-probability accumulator. Windows
/// contribute via [`Self::add_window`]; finalized frames are drained via
/// [`Self::take_finalized`].
pub(crate) struct VoiceStitcher {
  /// First absolute frame index represented in `sum` / `count`.
  base_frame: u64,
  /// Per-frame contribution sum.
  sum: VecDeque<f32>,
  /// Per-frame contribution count.
  count: VecDeque<u32>,
}

impl VoiceStitcher {
  pub(crate) fn new() -> Self {
    Self {
      base_frame: 0,
      sum: VecDeque::new(),
      count: VecDeque::new(),
    }
  }

  pub(crate) fn clear(&mut self) {
    self.base_frame = 0;
    self.sum.clear();
    self.count.clear();
  }

  /// Add one window of per-frame voice probabilities (length
  /// [`FRAMES_PER_WINDOW`]) anchored at absolute `start_frame`.
  ///
  /// If the window's frame range overlaps the already-finalized region
  /// (i.e. `start_frame < base_frame`, possible for an end-of-stream
  /// tail-anchor window), the prefix in the finalized region is silently
  /// dropped — only the suffix contributes.
  pub(crate) fn add_window(&mut self, start_frame: u64, voice_per_frame: &[f32]) {
    debug_assert_eq!(voice_per_frame.len(), FRAMES_PER_WINDOW);

    let end_frame = start_frame + FRAMES_PER_WINDOW as u64;
    if end_frame <= self.base_frame {
      return; // entirely in finalized region
    }

    // Ensure the buffer covers [base_frame, end_frame).
    let needed_len = (end_frame - self.base_frame) as usize;
    while self.sum.len() < needed_len {
      self.sum.push_back(0.0);
      self.count.push_back(0);
    }

    for (f, &p) in voice_per_frame.iter().enumerate() {
      let abs = start_frame + f as u64;
      if abs < self.base_frame {
        continue;
      }
      let idx = (abs - self.base_frame) as usize;
      self.sum[idx] += p;
      self.count[idx] += 1;
    }
  }

  /// Drain finalized frames in `[base_frame, up_to_frame)` and return their
  /// averaged voice probabilities. Advances `base_frame`.
  pub(crate) fn take_finalized(&mut self, up_to_frame: u64) -> Vec<f32> {
    debug_assert!(up_to_frame >= self.base_frame);
    let n = (up_to_frame.saturating_sub(self.base_frame)) as usize;
    let n = n.min(self.sum.len());
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
      let s = self.sum.pop_front().unwrap();
      let c = self.count.pop_front().unwrap();
      out.push(if c == 0 { 0.0 } else { s / c as f32 });
    }
    self.base_frame += n as u64;
    out
  }

  pub(crate) fn base_frame(&self) -> u64 {
    self.base_frame
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn ones_window() -> Vec<f32> {
    vec![1.0; FRAMES_PER_WINDOW]
  }
  fn zeros_window() -> Vec<f32> {
    vec![0.0; FRAMES_PER_WINDOW]
  }

  #[test]
  fn frame_to_sample_endpoints() {
    assert_eq!(frame_to_sample(0), 0);
    assert_eq!(frame_to_sample(FRAMES_PER_WINDOW as u32), WINDOW_SAMPLES);
  }

  #[test]
  fn frame_to_sample_monotonic() {
    let mut prev = 0u32;
    for f in 1..=FRAMES_PER_WINDOW as u32 {
      let s = frame_to_sample(f);
      assert!(s >= prev);
      prev = s;
    }
  }

  /// regression: absolute frame indices on long
  /// streams routinely exceed `u32::MAX`. The old u32-only helper
  /// truncated, silently wrapping `Action::VoiceSpan` ranges past
  /// ~74 h. The u64 helper must (a) agree with the u32 helper for
  /// frame indices whose sample-result still fits in u32, and (b) not
  /// wrap above it.
  #[test]
  fn frame_to_sample_u64_agrees_with_u32_in_safe_range() {
    // The u32 helper internally promotes to u64 for the multiplication
    // but casts back to u32 at the end, so it is only correct when
    // the *output* fits in u32. WINDOW_SAMPLES / FRAMES_PER_WINDOW ≈
    // 271.65, so frame_idx ≲ u32::MAX / 272 ≈ 15.78 M is safe.
    let safe_max = (u32::MAX as u64 / WINDOW_SAMPLES as u64 * FRAMES_PER_WINDOW as u64) as u32;
    for f in [0u32, 1, FRAMES_PER_WINDOW as u32, safe_max / 2, safe_max] {
      assert_eq!(
        frame_to_sample(f) as u64,
        frame_to_sample_u64(f as u64),
        "u32/u64 helpers must agree at frame_idx = {f}"
      );
    }
  }

  #[test]
  fn frame_to_sample_u64_does_not_wrap_past_u32_max() {
    // 16 kHz × 74.6 h ≈ u32::MAX samples → u32::MAX / WINDOW_SAMPLES *
    // FRAMES_PER_WINDOW frames is in u32 range, but absolute frames
    // beyond that must still produce monotonically increasing samples.
    let f_below = u32::MAX as u64;
    let f_above = f_below + 10_000;
    let s_below = frame_to_sample_u64(f_below);
    let s_above = frame_to_sample_u64(f_above);
    assert!(
      s_above > s_below,
      "frame_to_sample_u64 must not wrap past u32 boundary: \
       f_below={f_below} → s_below={s_below}, f_above={f_above} → s_above={s_above}"
    );
    // And at least one of these should be > u32::MAX (proves we left the
    // u32 range, not just stayed inside).
    assert!(
      s_above > u32::MAX as u64,
      "expected sample index to exceed u32::MAX; got {s_above}"
    );
  }

  #[test]
  fn frame_index_of_endpoints() {
    assert_eq!(frame_index_of(0), 0);
    assert_eq!(
      frame_index_of(WINDOW_SAMPLES as u64),
      FRAMES_PER_WINDOW as u64
    );
  }

  /// Half-integer collision case from spec §5.2.2: sample 80_000 lands
  /// exactly between frames 294 and 295. Floor must give 294.
  #[test]
  fn frame_index_of_floor_at_half_integer() {
    // 80_000 * 589 / 160_000 = 47_120_000 / 160_000 = 294.5 → floor = 294
    assert_eq!(frame_index_of(80_000), 294);
    // 40_000 * 589 / 160_000 = 23_560_000 / 160_000 = 147.25 → floor = 147
    assert_eq!(frame_index_of(40_000), 147);
    // 120_000 * 589 / 160_000 = 70_680_000 / 160_000 = 441.75 → floor = 441
    assert_eq!(frame_index_of(120_000), 441);
    // 160_000 * 589 / 160_000 = 589.0 → 589
    assert_eq!(frame_index_of(160_000), 589);
  }

  #[test]
  fn single_window_finalize_all() {
    let mut s = VoiceStitcher::new();
    s.add_window(0, &ones_window());
    let out = s.take_finalized(FRAMES_PER_WINDOW as u64);
    assert_eq!(out.len(), FRAMES_PER_WINDOW);
    for v in out {
      assert!((v - 1.0).abs() < 1e-6);
    }
    assert_eq!(s.base_frame(), FRAMES_PER_WINDOW as u64);
  }

  #[test]
  fn two_overlapping_windows_average() {
    // Window 0 starts at frame 0 (= sample 0). Window 1 starts at sample
    // 40_000, which is frame_index_of(40_000) = 147.
    let mut s = VoiceStitcher::new();
    s.add_window(0, &ones_window()); // covers frames [0, 589)
    s.add_window(147, &zeros_window()); // covers frames [147, 736)
    let out = s.take_finalized(736);
    // [0, 147): only window 0 → 1.0
    // [147, 589): overlap → 0.5
    // [589, 736): only window 1 → 0.0
    assert!((out[0] - 1.0).abs() < 1e-6);
    assert!((out[146] - 1.0).abs() < 1e-6);
    assert!((out[147] - 0.5).abs() < 1e-6);
    assert!((out[588] - 0.5).abs() < 1e-6);
    assert!(out[589].abs() < 1e-6);
    assert!(out[735].abs() < 1e-6);
  }

  #[test]
  fn partial_finalize_advances_base() {
    let mut s = VoiceStitcher::new();
    s.add_window(0, &ones_window());
    let part = s.take_finalized(100);
    assert_eq!(part.len(), 100);
    assert_eq!(s.base_frame(), 100);
    let rest = s.take_finalized(FRAMES_PER_WINDOW as u64);
    assert_eq!(rest.len(), FRAMES_PER_WINDOW - 100);
    assert_eq!(s.base_frame(), FRAMES_PER_WINDOW as u64);
  }

  #[test]
  fn tail_window_overlap_with_finalized_skipped() {
    // Drain [0, 100) first, then add a "tail" window starting at frame 50
    // (overlaps the finalized region).
    let mut s = VoiceStitcher::new();
    s.add_window(0, &ones_window());
    let _ = s.take_finalized(100);
    assert_eq!(s.base_frame(), 100);
    // Now add a window at start_frame=50 — frames 50..100 are already
    // finalized and should be dropped; frames 100..639 contribute.
    s.add_window(50, &zeros_window());
    // Drain everything available (window 0 covered [0, 589), so frames
    // 100..589 still in buffer with count=1 each from window 0; the tail
    // adds count for [100, 639), so frames [100, 589) have count=2 and
    // frames [589, 639) have count=1 from the tail only).
    let out = s.take_finalized(639);
    // Frame 100..589 average = (1.0 + 0.0) / 2 = 0.5
    assert!((out[0] - 0.5).abs() < 1e-6);
    assert!((out[488] - 0.5).abs() < 1e-6);
    // Frame 589..639 average = 0.0 / 1 = 0.0
    assert!(out[489].abs() < 1e-6);
  }

  #[test]
  fn clear_resets() {
    let mut s = VoiceStitcher::new();
    s.add_window(0, &ones_window());
    s.clear();
    assert_eq!(s.base_frame(), 0);
    assert!(s.take_finalized(100).is_empty());
  }
}
