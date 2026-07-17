//! Public types emitted by the segmentation state machine.

extern crate alloc;

use mediatime::{TimeRange, Timestamp};

/// Stable correlation handle for one inference round-trip.
///
/// Carries the window's sample range in `SAMPLE_RATE_TB` plus an opaque
/// generation token minted from a process-wide counter (see §11.9 of the
/// design spec). Two `WindowId`s compare equal iff both their range AND
/// generation match.
///
/// The generation counter eliminates two corruption scenarios:
///
/// 1. **Within one segmenter**, a stale `push_inference` from before a
///    `clear()` cannot match a new pending entry with the same range.
/// 2. **Across segmenters in the same process**, an `id` accidentally
///    fed to the wrong `Segmenter` cannot match because each
///    `Segmenter::new` consumes a fresh counter value.
///
/// The generation value is intentionally not exposed on the public API.
/// `Debug` shows it for diagnostics. `Ord`/`PartialOrd` order by
/// `(generation, start_pts)`; cross-generation ordering is deterministic
/// (suitable for `BTreeMap` lookup) but semantically meaningless — do not
/// use it for sample-position comparisons across cleared / different streams.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WindowId {
  range: TimeRange,
  generation: u64,
}

impl WindowId {
  pub(crate) const fn new(range: TimeRange, generation: u64) -> Self {
    Self { range, generation }
  }

  /// Sample-range covered by the window in `SAMPLE_RATE_TB`.
  pub const fn range(&self) -> TimeRange {
    self.range
  }

  /// Window start as a `Timestamp`.
  pub const fn start(&self) -> Timestamp {
    self.range.start()
  }

  /// Window end as a `Timestamp`.
  pub const fn end(&self) -> Timestamp {
    self.range.end()
  }

  /// Window duration (always 10 s for v0.1.0).
  pub const fn duration(&self) -> core::time::Duration {
    self.range.duration()
  }

  /// Internal accessor for the generation token. Crate-private and used
  /// only by tests; the public-facing diagnostic surface is `Debug`.
  /// Callers must not depend on this value being stable across releases.
  #[cfg(test)]
  pub(crate) const fn generation(&self) -> u64 {
    self.generation
  }
}

// Order by (generation, start_pts). End-PTS adds no information because
// `end == start + WINDOW_SAMPLES` for every window we produce. Within a
// single generation, ordering is "by sample position" and meaningful;
// across generations, ordering is deterministic (suitable for `BTreeMap`)
// but semantically meaningless.
impl Ord for WindowId {
  fn cmp(&self, other: &Self) -> core::cmp::Ordering {
    self
      .generation
      .cmp(&other.generation)
      .then_with(|| self.range.start_pts().cmp(&other.range.start_pts()))
  }
}

impl PartialOrd for WindowId {
  fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
    Some(self.cmp(other))
  }
}

/// One window-local speaker activity.
///
/// `speaker_slot` ∈ `0..=2` is local to the emitting window — slot identity
/// does NOT cross windows. Cross-window speaker identity is the job of a
/// future clustering layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SpeakerActivity {
  window_id: WindowId,
  speaker_slot: u8,
  range: TimeRange,
}

impl SpeakerActivity {
  pub(crate) const fn new(window_id: WindowId, speaker_slot: u8, range: TimeRange) -> Self {
    Self {
      window_id,
      speaker_slot,
      range,
    }
  }
  /// The window this activity was decoded from.
  pub const fn window_id(&self) -> WindowId {
    self.window_id
  }
  /// Window-local speaker slot (0, 1, or 2).
  pub const fn speaker_slot(&self) -> u8 {
    self.speaker_slot
  }
  /// Sample range of the activity within the stream, in `SAMPLE_RATE_TB`.
  pub const fn range(&self) -> TimeRange {
    self.range
  }
}

/// One output of the Layer-1 state machine.
///
/// Style note: enum-variant fields (`id`, `samples`) are public because they
/// participate in pattern matching, which is the standard Rust enum idiom.
/// Structs with invariants (`WindowId`, `SpeakerActivity`) use private
/// fields with accessors. The two conventions coexist deliberately.
///
/// **`#[non_exhaustive]`** (added in v0.X for the dia phase-2 release):
/// downstream `match` expressions must include `_ => ...` to remain
/// forward-compatible. New variants may be added in subsequent minor
/// versions without a breaking change.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum Action {
  /// The caller must run ONNX inference on `samples` and call
  /// [`Segmenter::push_inference`](crate::segment::Segmenter::push_inference)
  /// with the same `id`.
  NeedsInference {
    /// Correlation handle (the window's sample range plus generation).
    id: WindowId,
    /// Always `WINDOW_SAMPLES = 160_000` mono float32 samples at 16 kHz,
    /// zero-padded if the input stream is shorter.
    samples: alloc::boxed::Box<[f32]>,
  },
  /// A decoded window-local speaker activity.
  Activity(SpeakerActivity),
  /// A finalized speaker-agnostic voice region. Emit-only — never
  /// retracted once produced.
  VoiceSpan(TimeRange),
  /// Per-window per-speaker per-frame raw probabilities. Emitted from
  /// [`Segmenter::push_inference`](crate::segment::Segmenter::push_inference)
  /// **immediately before** the `Activity` events for the same `id`.
  ///
  /// Carries the powerset-decoded per-frame voice probabilities for
  /// each of the 3 speaker slots. Most callers can ignore this
  /// variant via the `_ => ...` arm of `match`.
  ///
  /// Layout: `raw_probs[slot][frame]`. `MAX_SPEAKER_SLOTS = 3`,
  /// `FRAMES_PER_WINDOW = 589`. ~7 KB allocation per emission;
  /// see spec §15 #53 for a v0.1.1 pooling optimization.
  SpeakerScores {
    /// Correlation handle of the window these scores belong to.
    id: WindowId,
    /// Window start in absolute samples (`id.range().start_pts()` in `SAMPLE_RATE_TB`).
    window_start: u64,
    /// Per-(slot, frame) raw probabilities.
    raw_probs: alloc::boxed::Box<
      [[f32; crate::segment::options::FRAMES_PER_WINDOW];
        crate::segment::options::MAX_SPEAKER_SLOTS as usize],
    >,
  },
}

/// Layer-2 emission events (Layer 2 hides `NeedsInference` from the caller).
#[derive(Debug, Clone)]
pub enum Event {
  /// A decoded window-local speaker activity.
  Activity(SpeakerActivity),
  /// A finalized speaker-agnostic voice region.
  VoiceSpan(TimeRange),
}

#[cfg(test)]
mod tests {
  use super::*;

  use crate::segment::options::SAMPLE_RATE_TB;

  fn tr(start: i64, end: i64) -> TimeRange {
    TimeRange::new(start, end, SAMPLE_RATE_TB)
  }

  fn id(start: i64, end: i64, generation: u64) -> WindowId {
    WindowId::new(tr(start, end), generation)
  }

  #[test]
  fn window_id_accessors() {
    let w = id(0, 160_000, 7);
    assert_eq!(w.range(), tr(0, 160_000));
    assert_eq!(w.start().pts(), 0);
    assert_eq!(w.end().pts(), 160_000);
    assert_eq!(w.duration(), core::time::Duration::from_secs(10));
    assert_eq!(w.generation(), 7);
  }

  #[test]
  fn window_id_eq_includes_generation() {
    assert_eq!(id(0, 160_000, 0), id(0, 160_000, 0));
    assert_ne!(id(0, 160_000, 0), id(0, 160_000, 1));
  }

  #[test]
  fn window_id_hash_includes_generation() {
    use std::collections::HashSet;
    let mut s = HashSet::new();
    s.insert(id(0, 160_000, 0));
    assert!(s.contains(&id(0, 160_000, 0)));
    assert!(!s.contains(&id(0, 160_000, 1)));
    assert!(!s.contains(&id(40_000, 200_000, 0)));
  }

  #[test]
  fn window_id_ord_by_generation_then_start() {
    use core::cmp::Ordering;
    // Same generation: ordered by start.
    assert_eq!(
      id(0, 160_000, 0).cmp(&id(40_000, 200_000, 0)),
      Ordering::Less
    );
    // Different generation: ordered by generation.
    assert_eq!(
      id(0, 160_000, 1).cmp(&id(40_000, 200_000, 0)),
      Ordering::Greater
    );
    assert_eq!(
      id(40_000, 200_000, 0).cmp(&id(0, 160_000, 1)),
      Ordering::Less
    );
  }

  #[test]
  fn speaker_activity_accessors() {
    let win = id(0, 160_000, 0);
    let act = SpeakerActivity::new(win, 1, tr(8_000, 24_000));
    assert_eq!(act.window_id(), win);
    assert_eq!(act.speaker_slot(), 1);
    assert_eq!(act.range(), tr(8_000, 24_000));
  }
}
