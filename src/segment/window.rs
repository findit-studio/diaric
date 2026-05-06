//! Sliding-window scheduling.
//!
//! Windows step at `step_samples` intervals. If the regular grid does not
//! cover the entire stream, a final tail window is anchored to end-of-stream
//! so the last `WINDOW_SAMPLES` samples are always processed.

extern crate alloc;

use alloc::vec::Vec;

use crate::segment::options::WINDOW_SAMPLES;

/// Plan output: the start sample of each scheduled window. Each window
/// covers `[start, start + WINDOW_SAMPLES)`.
///
/// `total_samples` is the full stream length.
/// Returns at minimum one window (anchored at 0, possibly padded) when
/// `total_samples > 0`. Empty streams yield an empty plan.
pub(crate) fn plan_starts(total_samples: u64, step_samples: u32) -> Vec<u64> {
  if total_samples == 0 {
    return Vec::new();
  }
  let step = step_samples as u64;
  assert!(step > 0, "step_samples must be > 0");
  let win = WINDOW_SAMPLES as u64;

  let mut out = Vec::new();
  let mut s: u64 = 0;
  // Schedule regular windows that fully fit.
  while s + win <= total_samples {
    out.push(s);
    s += step;
  }
  // Tail anchor: ensure the final window ends at total_samples (or covers
  // [0, total_samples) if total < window).
  let tail_start = total_samples.saturating_sub(win);
  if out.last().copied() != Some(tail_start) {
    out.push(tail_start);
  }
  out
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn empty_stream_no_windows() {
    assert!(plan_starts(0, 40_000).is_empty());
  }

  #[test]
  fn shorter_than_one_window_yields_one_anchored_window() {
    let p = plan_starts(50_000, 40_000);
    assert_eq!(p, vec![0]); // tail_start = 0 (50_000 - 160_000 saturates).
  }

  #[test]
  fn exact_one_window_no_tail_duplicate() {
    let p = plan_starts(160_000, 40_000);
    // Regular schedule places a window at 0; tail_start is also 0.
    assert_eq!(p, vec![0]);
  }

  #[test]
  fn regular_grid_then_tail_anchor() {
    // 200_000 samples, step 40_000: regular fits at 0 and 40_000
    // (since 40_000 + 160_000 = 200_000 == total). Next would be 80_000
    // (80_000 + 160_000 = 240_000 > 200_000), so stop. tail_start = 40_000,
    // already last → no duplicate.
    let p = plan_starts(200_000, 40_000);
    assert_eq!(p, vec![0, 40_000]);
  }

  #[test]
  fn regular_grid_with_separate_tail() {
    // 230_000 samples, step 40_000: regular windows at 0, 40_000.
    // 80_000 + 160_000 = 240_000 > 230_000, stop. tail_start = 70_000,
    // distinct from 40_000 → push as tail.
    let p = plan_starts(230_000, 40_000);
    assert_eq!(p, vec![0, 40_000, 70_000]);
  }

  #[test]
  fn step_equal_to_window_no_overlap() {
    // step == window, total = 320_000 → windows at 0 and 160_000, tail same as last.
    let p = plan_starts(320_000, 160_000);
    assert_eq!(p, vec![0, 160_000]);
  }
}
