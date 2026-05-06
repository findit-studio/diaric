//! Two-threshold hysteresis state machine and run-length encoding.
//!
//! `binarize` walks a probability sequence with state. The state goes
//! inactive → active when `p >= onset`, and active → inactive when
//! `p < offset`. With `offset < onset` this gives stable boundaries.
//!
//! `runs_of_true` extracts half-open `[start, end)` index ranges where the
//! mask is true.

extern crate alloc;

use alloc::vec::Vec;

/// Stateful hysteresis cursor. Use [`Hysteresis::push`] for streaming use,
/// or [`binarize`] for whole-buffer use.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Hysteresis {
  onset: f32,
  offset: f32,
  active: bool,
}

impl Hysteresis {
  pub(crate) const fn new(onset: f32, offset: f32) -> Self {
    Self {
      onset,
      offset,
      active: false,
    }
  }
  /// Step one sample. Returns the new active state.
  pub(crate) fn push(&mut self, p: f32) -> bool {
    self.active = if self.active {
      p >= self.offset
    } else {
      p >= self.onset
    };
    self.active
  }
  pub(crate) fn is_active(&self) -> bool {
    self.active
  }
  pub(crate) fn reset(&mut self) {
    self.active = false;
  }
}

/// Apply hysteresis to a probability sequence (no carried state).
///
/// Bulk-mode helper used by tests; the segmenter uses [`Hysteresis::push`]
/// directly to maintain streaming state.
#[cfg(test)]
pub(crate) fn binarize(probs: &[f32], onset: f32, offset: f32) -> Vec<bool> {
  let mut h = Hysteresis::new(onset, offset);
  probs.iter().map(|&p| h.push(p)).collect()
}

/// RLE of a boolean mask into half-open `[start, end)` index ranges of true.
pub(crate) fn runs_of_true(mask: &[bool]) -> Vec<(usize, usize)> {
  let mut out = Vec::new();
  let mut start: Option<usize> = None;
  for (i, &b) in mask.iter().enumerate() {
    match (b, start) {
      (true, None) => start = Some(i),
      (false, Some(s)) => {
        out.push((s, i));
        start = None;
      }
      _ => {}
    }
  }
  if let Some(s) = start {
    out.push((s, mask.len()));
  }
  out
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn binarize_simple_step() {
    let probs = [0.0, 0.4, 0.6, 0.5, 0.4, 0.3, 0.0];
    // onset 0.5, offset 0.4. State: 0,0,1,1,1,0,0 (active until p<0.4 at index 5).
    let m = binarize(&probs, 0.5, 0.4);
    assert_eq!(m, [false, false, true, true, true, false, false]);
  }

  #[test]
  fn binarize_hysteresis_prevents_flicker() {
    // probabilities oscillate between 0.45 and 0.55 around the onset.
    // With onset 0.5, offset 0.4, once active we stay active because
    // p >= 0.4 throughout.
    let probs = [0.55, 0.45, 0.55, 0.45, 0.55];
    let m = binarize(&probs, 0.5, 0.4);
    assert_eq!(m, [true, true, true, true, true]);
  }

  #[test]
  fn binarize_empty() {
    let m = binarize(&[], 0.5, 0.4);
    assert!(m.is_empty());
  }

  #[test]
  fn binarize_all_below_onset_stays_inactive() {
    let probs = [0.0, 0.1, 0.2, 0.3, 0.49];
    let m = binarize(&probs, 0.5, 0.4);
    assert_eq!(m, [false, false, false, false, false]);
  }

  #[test]
  fn runs_basic() {
    let m = [false, true, true, false, true, false, true, true, true];
    assert_eq!(runs_of_true(&m), vec![(1, 3), (4, 5), (6, 9)]);
  }

  #[test]
  fn runs_all_false() {
    let m = [false; 5];
    assert!(runs_of_true(&m).is_empty());
  }

  #[test]
  fn runs_all_true() {
    let m = [true; 4];
    assert_eq!(runs_of_true(&m), vec![(0, 4)]);
  }

  #[test]
  fn runs_trailing_open_run_closes() {
    let m = [false, true, true];
    assert_eq!(runs_of_true(&m), vec![(1, 3)]);
  }

  #[test]
  fn runs_empty() {
    assert!(runs_of_true(&[]).is_empty());
  }

  #[test]
  fn streaming_hysteresis_matches_batch() {
    let probs = [0.0, 0.4, 0.6, 0.5, 0.4, 0.3, 0.0];
    let mut h = Hysteresis::new(0.5, 0.4);
    let online: Vec<bool> = probs.iter().map(|&p| h.push(p)).collect();
    assert_eq!(online, binarize(&probs, 0.5, 0.4));
  }
}
