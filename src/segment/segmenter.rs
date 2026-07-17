//! Layer-1 Sans-I/O speaker segmentation state machine.

extern crate alloc;

use alloc::{
  boxed::Box,
  collections::{BTreeMap, VecDeque},
  vec,
  vec::Vec,
};

use core::sync::atomic::{AtomicU64, Ordering};

use mediatime::TimeRange;

use crate::segment::{
  error::Error,
  hysteresis::{Hysteresis, runs_of_true},
  options::{
    FRAMES_PER_WINDOW, MAX_SPEAKER_SLOTS, POWERSET_CLASSES, SAMPLE_RATE_HZ, SAMPLE_RATE_TB,
    SegmentOptions, WINDOW_SAMPLES,
  },
  powerset::{powerset_to_speakers, softmax_row, voice_prob},
  stitch::{VoiceStitcher, frame_index_of, frame_to_sample, frame_to_sample_u64},
  types::{Action, SpeakerActivity, WindowId},
  window::plan_starts,
};

/// Process-wide generation counter for `WindowId` minting. Bumped on every
/// [`Segmenter::new`] and every [`Segmenter::clear`]. Each `Segmenter`
/// captures its current value and stamps every `WindowId` it yields with
/// it.
///
/// `Relaxed` ordering is sufficient because the counter values are not
/// used to synchronize any other memory; their only purpose is to provide
/// a unique opaque token. Each `Segmenter` reads the value once at
/// construction or `clear`, stores it locally, and consults it from then
/// on under `&mut self`. There is no happens-before relationship across
/// `Segmenter` instances that needs to be established by the atomic.
///
/// Wraps at `2^64` (~600 years at 10⁹ clears/s); overflow is treated as
/// not-a-concern.
static GENERATION: AtomicU64 = AtomicU64::new(0);

/// Sans-I/O speaker segmentation state machine.
///
/// See the module docs and the design spec for the full data flow. Brief:
///
/// 1. Caller appends PCM via [`push_samples`](Self::push_samples).
/// 2. Caller drains [`Action`]s via [`poll`](Self::poll). When it sees
///    [`Action::NeedsInference`], it runs the model on the supplied
///    samples and calls [`push_inference`](Self::push_inference) with the
///    scores.
/// 3. After all PCM is delivered, caller calls [`finish`](Self::finish)
///    and drains remaining actions.
///
/// `Segmenter` auto-derives `Send + Sync`. State-machine calls all need
/// `&mut self`, so `Sync` is incidental — sharing one `Segmenter` between
/// threads buys nothing. Use one per concurrent stream.
pub struct Segmenter {
  pub(crate) opts: SegmentOptions,

  /// Generation token for every `WindowId` this segmenter mints.
  /// Initialized at construction; refreshed on `clear()`.
  pub(crate) generation: u64,

  /// Rolling sample buffer. Index 0 corresponds to absolute sample
  /// `consumed_samples`.
  pub(crate) input: VecDeque<f32>,
  pub(crate) consumed_samples: u64,

  /// Cumulative count of samples ever delivered via `push_samples`.
  /// Never decremented (window-driven trimming of `input` does not
  /// affect this). Resets only on `clear()`.
  pub(crate) total_samples_pushed: u64,

  /// Index of the next window to schedule (== how many windows have
  /// been emitted). Window k covers
  /// `[k * step_samples, k * step_samples + WINDOW_SAMPLES)` in absolute
  /// samples, except for the final tail-anchor window.
  pub(crate) next_window_idx: u32,

  /// Pending inference round-trips: id → window-start sample.
  pub(crate) pending: BTreeMap<WindowId, u64>,

  /// Output queue.
  pub(crate) pending_actions: VecDeque<Action>,

  /// Per-frame voice-probability accumulator.
  pub(crate) stitcher: VoiceStitcher,

  /// Streaming hysteresis cursor for the global voice timeline.
  pub(crate) voice_hyst: Hysteresis,
  /// Frame index where the currently-open voice run started (if any).
  pub(crate) voice_run_start: Option<u64>,

  /// Pending span buffered by the merge cursor — emitted when the next
  /// span is farther than `voice_merge_gap` away, or at end-of-stream.
  pub(crate) merge_pending: Option<(u64, u64)>,

  /// `finish()` has been called.
  pub(crate) finished: bool,
  /// Tail anchor window has been emitted.
  pub(crate) tail_emitted: bool,
  /// Total stream length latched at `finish()`.
  pub(crate) total_samples: u64,
  /// Stashed in-flight inference for the Layer-2 streaming API. Set by
  /// [`Segmenter::process_samples`] / [`Segmenter::finish_stream`]
  /// (under `feature = "ort"`) when [`SegmentModel::infer`] or
  /// [`Self::push_inference`] returns an error mid-drain. The next
  /// drain replays the stash before polling new actions; without it,
  /// `Action::NeedsInference` was popped + lost on every transient
  /// failure (the `WindowId` stayed in `pending`, but no caller-
  /// reachable handle remained to retry it).
  ///
  /// `cfg`-gated because it is only consumed by the `ort`-feature
  /// streaming helpers; the field stays always-present to keep
  /// `Segmenter` layout stable across feature builds.
  pub(crate) pending_inference: Option<(WindowId, alloc::boxed::Box<[f32]>)>,
}

impl Segmenter {
  /// Construct a new segmenter. Consumes one process-wide generation token.
  ///
  /// # Panics
  /// Panics if any [`SegmentOptions`] field violates its documented
  /// contract: `step_samples == 0` or `> WINDOW_SAMPLES`, or any
  /// hysteresis threshold outside `[0.0, 1.0]` (including NaN/±inf),
  /// or `offset_threshold > onset_threshold`. Defense-in-depth: the
  /// option setters already enforce these invariants on the builder
  /// path, so this trip only fires when a `SegmentOptions` value was
  /// constructed without them — most realistically, a serde-
  /// deserialized config (`#[serde(default)]` fields are never
  /// validated by the setters). Use [`Self::try_new`] to surface
  /// these preconditions as [`Error::InvalidOptions`] instead.
  pub fn new(opts: SegmentOptions) -> Self {
    Self::try_new(opts).expect("Segmenter::new: invalid options; use try_new to handle")
  }

  /// Fallible variant of [`Self::new`]. Returns
  /// [`Error::InvalidOptions`] for any of the contract violations
  /// described on [`Self::new`]; otherwise identical output.
  pub fn try_new(opts: SegmentOptions) -> Result<Self, Error> {
    use crate::segment::error::InvalidOptionsReason;
    if opts.step_samples() == 0 {
      return Err(InvalidOptionsReason::ZeroStepSamples.into());
    }
    if opts.step_samples() > WINDOW_SAMPLES {
      return Err(
        InvalidOptionsReason::StepSamplesExceedsWindow {
          step: opts.step_samples(),
          window: WINDOW_SAMPLES,
        }
        .into(),
      );
    }
    let onset = opts.onset_threshold();
    let offset = opts.offset_threshold();
    // `Hysteresis::new(NaN, _)` makes every `p >= threshold`
    // comparison false (sticky-silent state machine);
    // `Hysteresis::new(_, > 1.0)` makes the falling-edge unreachable
    // so a started voice run never closes. Mirror the predicate the
    // setters use so serde-bypassed configs cannot bypass it.
    if !is_finite_in_unit_interval(onset) {
      return Err(
        InvalidOptionsReason::HysteresisThresholdOutOfRange {
          which: "onset",
          value: onset,
        }
        .into(),
      );
    }
    if !is_finite_in_unit_interval(offset) {
      return Err(
        InvalidOptionsReason::HysteresisThresholdOutOfRange {
          which: "offset",
          value: offset,
        }
        .into(),
      );
    }
    // After NaN rejection above, both values are finite in [0,1] and
    // `offset > onset` is well-defined.
    if offset > onset {
      return Err(InvalidOptionsReason::OffsetAboveOnset { offset, onset }.into());
    }
    Ok(Self {
      opts,
      generation: GENERATION.fetch_add(1, Ordering::Relaxed),
      input: VecDeque::new(),
      consumed_samples: 0,
      total_samples_pushed: 0,
      next_window_idx: 0,
      pending: BTreeMap::new(),
      pending_actions: VecDeque::new(),
      stitcher: VoiceStitcher::new(),
      voice_hyst: Hysteresis::new(onset, offset),
      voice_run_start: None,
      merge_pending: None,
      finished: false,
      tail_emitted: false,
      total_samples: 0,
      pending_inference: None,
    })
  }

  /// Read-only access to the configured options.
  pub fn options(&self) -> &SegmentOptions {
    &self.opts
  }

  /// Append 16 kHz mono float32 PCM samples. Arbitrary chunk size.
  ///
  /// **Caller must enforce sample rate** — there is no runtime guard.
  ///
  /// `samples.len() == 0` is a no-op: the call is accepted but does NOT
  /// count toward the §11.7 tail-window threshold (a tail is scheduled
  /// only if at least one non-empty `push_samples` happened before
  /// `finish()`).
  ///
  /// Calling after [`finish`](Self::finish) is a programming bug; the
  /// call is silently ignored in release builds and panics in debug.
  pub fn push_samples(&mut self, samples: &[f32]) {
    debug_assert!(!self.finished, "push_samples after finish");
    if self.finished || samples.is_empty() {
      return;
    }
    self.input.extend(samples.iter().copied());
    self.total_samples_pushed += samples.len() as u64;
    self.schedule_ready_windows();
  }

  /// Schedule any regular windows fully covered by buffered audio. Tail
  /// scheduling happens in [`finish`](Self::finish).
  fn schedule_ready_windows(&mut self) {
    let step = self.opts.step_samples() as u64;
    let win = WINDOW_SAMPLES as u64;
    loop {
      let start = self.next_window_idx as u64 * step;
      let end = start + win;
      let buffered_end = self.consumed_samples + self.input.len() as u64;
      if buffered_end < end {
        return;
      }
      self.emit_window(start);
      self.next_window_idx += 1;
    }
  }

  /// Build a window starting at `start` (absolute samples), copy its
  /// samples (zero-padding when the input buffer is shorter than
  /// `WINDOW_SAMPLES`), enqueue `NeedsInference`, and trim the input buffer.
  pub(crate) fn emit_window(&mut self, start: u64) {
    let win = WINDOW_SAMPLES as u64;
    let buffered_end = self.consumed_samples + self.input.len() as u64;
    let mut samples: Vec<f32> = Vec::with_capacity(WINDOW_SAMPLES as usize);
    let avail_end = buffered_end.min(start + win);

    let copy_from = (start.saturating_sub(self.consumed_samples)) as usize;
    let copy_until = (avail_end.saturating_sub(self.consumed_samples)) as usize;
    for i in copy_from..copy_until {
      samples.push(self.input[i]);
    }
    while samples.len() < WINDOW_SAMPLES as usize {
      samples.push(0.0);
    }

    let id = WindowId::new(
      TimeRange::new(start as i64, (start + win) as i64, SAMPLE_RATE_TB),
      self.generation,
    );
    self.pending.insert(id, start);
    self.pending_actions.push_back(Action::NeedsInference {
      id,
      samples: Box::from(samples.as_slice()),
    });

    // Drop samples no future regular window OR finish() tail anchor will
    // need. The next regular window starts at (next_window_idx + 1) *
    // step_samples. The latest possible tail anchor (from plan_starts in
    // finish()) is at total_samples_pushed - WINDOW_SAMPLES. Keep at least
    // the rolling last-WINDOW_SAMPLES window so a later tail can replay
    // audio with correct absolute alignment.
    let next_regular_start = (self.next_window_idx + 1) as u64 * self.opts.step_samples() as u64;
    let tail_floor = self
      .total_samples_pushed
      .saturating_sub(WINDOW_SAMPLES as u64);
    let trim_to = next_regular_start.min(tail_floor);
    self.trim_input_to(trim_to);
  }

  fn trim_input_to(&mut self, abs_sample: u64) {
    let target = abs_sample.min(self.consumed_samples + self.input.len() as u64);
    let drop_n = (target.saturating_sub(self.consumed_samples)) as usize;
    for _ in 0..drop_n {
      self.input.pop_front();
    }
    self.consumed_samples += drop_n as u64;
  }

  /// Drain the next pending action.
  ///
  /// Returns `None` when nothing is currently ready. `None` does NOT
  /// imply end-of-stream — the caller signals that with
  /// [`finish`](Self::finish).
  pub fn poll(&mut self) -> Option<Action> {
    self.pending_actions.pop_front()
  }

  /// Provide ONNX inference results for a previously-yielded window.
  ///
  /// `scores.len()` must equal `FRAMES_PER_WINDOW * POWERSET_CLASSES = 4123`.
  ///
  /// Returns [`Error::UnknownWindow`] if `id` is not in the pending set.
  /// This covers four scenarios:
  ///
  /// 1. `id` was never yielded by [`poll`](Self::poll).
  /// 2. `id` was already consumed by an earlier `push_inference` call —
  ///    each pending entry is consumed exactly once.
  /// 3. `id` came from a previous stream that was reset by
  ///    [`clear`](Self::clear) (caught by the generation counter).
  /// 4. `id` was minted by a different `Segmenter` instance whose sample
  ///    range happens to match a current pending window's range
  ///    (different generation; rejected).
  ///
  /// Returns [`Error::InferenceShapeMismatch`] if `scores.len()` is wrong,
  /// or [`Error::NonFiniteScores`] if any score is NaN or infinite.
  ///
  /// On `NonFiniteScores`, the [`WindowId`] is left in the pending set so
  /// the caller can retry with valid logits (e.g. from a fallback model
  /// or a re-run of the same model). Without this validation, NaN
  /// propagates through `softmax_row` and downstream comparisons treat
  /// the entire window as silent — silently dropping the audio with no
  /// retry path.
  pub fn push_inference(&mut self, id: WindowId, scores: &[f32]) -> Result<(), Error> {
    let expected = FRAMES_PER_WINDOW * POWERSET_CLASSES;
    if scores.len() != expected {
      return Err(Error::InferenceShapeMismatch {
        expected,
        got: scores.len(),
      });
    }
    // Verify the window is pending BEFORE rejecting non-finite scores so
    // an unknown id keeps reporting `UnknownWindow` (a stable contract
    // for callers using stale ids after `clear()`).
    if !self.pending.contains_key(&id) {
      return Err(Error::UnknownWindow { id });
    }
    if !scores.iter().all(|x| x.is_finite()) {
      // Leave `id` in `pending` so the caller can retry with valid
      // logits. The window is not consumed.
      return Err(Error::NonFiniteScores { id });
    }
    let start = self.pending.remove(&id).expect("presence checked above");

    // Decode powerset row by row.
    let mut speaker_probs: [Vec<f32>; MAX_SPEAKER_SLOTS as usize] = [
      vec![0.0; FRAMES_PER_WINDOW],
      vec![0.0; FRAMES_PER_WINDOW],
      vec![0.0; FRAMES_PER_WINDOW],
    ];
    let mut voice_per_frame: Vec<f32> = Vec::with_capacity(FRAMES_PER_WINDOW);

    // The index drives slicing of `scores` AND parallel writes into the
    // three per-slot probability buffers; an iterator would not be cleaner.
    #[allow(clippy::needless_range_loop)]
    for f in 0..FRAMES_PER_WINDOW {
      let row_start = f * POWERSET_CLASSES;
      let mut row = [0f32; POWERSET_CLASSES];
      row.copy_from_slice(&scores[row_start..row_start + POWERSET_CLASSES]);
      let probs = softmax_row(&row);
      voice_per_frame.push(voice_prob(&probs));
      let s = powerset_to_speakers(&probs);
      speaker_probs[0][f] = s[0];
      speaker_probs[1][f] = s[1];
      speaker_probs[2][f] = s[2];
    }

    // Emit raw per-(slot, frame) probabilities BEFORE any activities for
    // the same window so a downstream consumer can buffer scores per
    // `WindowId` and then process the activities that follow.
    self.pending_actions.push_back(Action::SpeakerScores {
      id,
      window_start: start,
      raw_probs: Box::new(speaker_probs_to_array(&speaker_probs)),
    });

    // Emit per-window speaker activities.
    self.emit_speaker_activities(id, start, &speaker_probs);

    // Feed voice probabilities into the per-frame stitcher.
    let start_frame = frame_index_of(start);
    self.stitcher.add_window(start_frame, &voice_per_frame);

    self.process_voice_finalization();
    Ok(())
  }

  fn emit_speaker_activities(
    &mut self,
    id: WindowId,
    window_start: u64,
    speaker_probs: &[Vec<f32>; MAX_SPEAKER_SLOTS as usize],
  ) {
    let onset = self.opts.onset_threshold();
    let offset = self.opts.offset_threshold();
    let min_dur = self.opts.min_activity_duration();
    let min_samples = duration_to_samples(min_dur);

    // Tail windows (post-finish) may extend beyond actual audio; their
    // activities must be clamped to `total_samples`. Regular windows have
    // already been validated against buffered audio so no clamp is needed,
    // but applying it unconditionally when `finished` is harmless.
    let clamp_max = if self.finished {
      self.total_samples
    } else {
      u64::MAX
    };

    for slot in 0..MAX_SPEAKER_SLOTS {
      let probs = &speaker_probs[slot as usize];
      let mut h = Hysteresis::new(onset, offset);
      let mask: Vec<bool> = probs.iter().map(|&p| h.push(p)).collect();
      for (f0, f1) in runs_of_true(&mask) {
        let s0_raw = window_start + frame_to_sample(f0 as u32) as u64;
        let s1_raw = window_start + frame_to_sample(f1 as u32) as u64;
        let s0 = s0_raw.min(clamp_max);
        let s1 = s1_raw.min(clamp_max);
        if s1 <= s0 || s1 - s0 < min_samples {
          continue;
        }
        let range = TimeRange::new(s0 as i64, s1 as i64, SAMPLE_RATE_TB);
        self
          .pending_actions
          .push_back(Action::Activity(SpeakerActivity::new(id, slot, range)));
      }
    }
  }

  /// Drain finalizable frames from the stitcher, run streaming hysteresis,
  /// and emit voice spans.
  fn process_voice_finalization(&mut self) {
    let up_to = self.next_finalization_boundary();
    let probs = if up_to > self.stitcher.base_frame() {
      self.stitcher.take_finalized(up_to)
    } else {
      Vec::new()
    };
    let base_after = self.stitcher.base_frame();
    let base_before = base_after - probs.len() as u64;

    for (i, p) in probs.iter().enumerate() {
      let abs_frame = base_before + i as u64;
      let was_active = self.voice_hyst.is_active();
      let now_active = self.voice_hyst.push(*p);
      match (was_active, now_active) {
        (false, true) => self.voice_run_start = Some(abs_frame),
        (true, false) => {
          if let Some(start_frame) = self.voice_run_start.take() {
            self.feed_merge_cursor_frames(start_frame, abs_frame);
          }
        }
        _ => {}
      }
    }

    if self.finished && self.pending.is_empty() {
      // End-of-stream span closure (spec §5.6 step 3-5). Convert the
      // run's start frame to a sample index, but use `total_samples`
      // directly for the end (don't round-trip through `frame_to_sample`,
      // which can overshoot — e.g. for total_samples=250_000, total_frames
      // rounds to 921 and frame_to_sample(921) = 250_187).
      if let Some(start_frame) = self.voice_run_start.take() {
        // Absolute frame → sample: must be u64 end-to-end. The legacy
        // u32 cast wrapped after ~74 h at 16 kHz.
        let s0 = frame_to_sample_u64(start_frame).min(self.total_samples);
        self.feed_merge_cursor(s0, self.total_samples);
        self.voice_hyst.reset();
      }
      // Step 5: flush any pending merge buffer.
      self.flush_merge_pending();
    }
  }

  /// Smallest absolute frame index that no future or pending window can
  /// still contribute to.
  ///
  /// - **Pre-finish:** `min(next_window_start_frame, earliest_pending,
  ///   tail_safe_frame)`.
  ///   - Without `earliest_pending`, an out-of-order `push_inference`
  ///     (windows 0/1/2 pending; scores for 2 arrive first) would
  ///     advance the boundary past frames whose other contributing
  ///     windows haven't reported yet.
  ///   - Without `tail_safe_frame`, the not-yet-emitted tail-anchor
  ///     window (scheduled by [`Self::finish`] at
  ///     `max(0, total_samples_pushed - WINDOW_SAMPLES)`) could land
  ///     on frames that have already been finalized — its
  ///     contribution would be silently dropped.
  /// - **Post-finish + pending empty:** `total_frames` (entire stream
  ///   finalized).
  fn next_finalization_boundary(&self) -> u64 {
    if self.finished && self.pending.is_empty() {
      return total_frames_of(self.total_samples);
    }
    let step = self.opts.step_samples() as u64;
    let next_window_start = self.next_window_idx as u64 * step;
    let next_window_start_frame = frame_index_of(next_window_start);
    let earliest_pending_frame = self
      .pending
      .values()
      .copied()
      .map(frame_index_of)
      .min()
      .unwrap_or(u64::MAX);
    // Tail-safe cap: any tail anchor that finish() may schedule starts
    // no earlier than `total_samples_pushed - WINDOW_SAMPLES`. If we
    // were already finalized past that point, its frames would be
    // skipped by reconstruction (it has a defensive guard against
    // frames < base_frame). Pre-finish, we don't know whether finish
    // will be called soon, so we always include this term — it costs
    // at most one window of extra buffering.
    let tail_safe_frame = if self.finished {
      // After finish, the tail (if any) has already been scheduled
      // and is in `pending`; the earliest_pending term covers it.
      u64::MAX
    } else {
      let tail_safe_start = self
        .total_samples_pushed
        .saturating_sub(WINDOW_SAMPLES as u64);
      frame_index_of(tail_safe_start)
    };
    next_window_start_frame
      .min(earliest_pending_frame)
      .min(tail_safe_frame)
  }

  /// Receive one `[start_frame, end_frame)` span from the streaming
  /// hysteresis state machine, convert to samples, apply the merge-gap
  /// rule (§5.6.5).
  ///
  /// `start_frame` / `end_frame` are absolute stream-wide frame
  /// indices; we convert with the u64 helper so timestamps stay
  /// correct past ~74 h. The previous u32-clamp path silently wrapped
  /// `Action::VoiceSpan` ranges.
  fn feed_merge_cursor_frames(&mut self, start_frame: u64, end_frame: u64) {
    let s0 = frame_to_sample_u64(start_frame);
    let s1 = frame_to_sample_u64(end_frame);
    self.feed_merge_cursor(s0, s1);
  }

  fn feed_merge_cursor(&mut self, start_sample: u64, end_sample: u64) {
    let merge_gap = duration_to_samples(self.opts.voice_merge_gap());
    match self.merge_pending.take() {
      Some((p_start, p_end)) => {
        if start_sample.saturating_sub(p_end) <= merge_gap {
          // Merge: extend the pending span.
          self.merge_pending = Some((p_start, end_sample.max(p_end)));
        } else {
          // Gap too large: emit the pending span, buffer the new one.
          self.emit_voice_span(p_start, p_end);
          self.merge_pending = Some((start_sample, end_sample));
        }
      }
      None => {
        self.merge_pending = Some((start_sample, end_sample));
      }
    }
  }

  fn flush_merge_pending(&mut self) {
    if let Some((p_start, p_end)) = self.merge_pending.take() {
      self.emit_voice_span(p_start, p_end);
    }
  }

  fn emit_voice_span(&mut self, start_sample: u64, end_sample: u64) {
    let dur_samples = end_sample.saturating_sub(start_sample);
    let min = duration_to_samples(self.opts.min_voice_duration());
    if dur_samples < min || dur_samples == 0 {
      return;
    }
    let range = TimeRange::new(start_sample as i64, end_sample as i64, SAMPLE_RATE_TB);
    self.pending_actions.push_back(Action::VoiceSpan(range));
  }

  /// Signal end-of-stream. Schedules a tail-anchored window if needed and
  /// flushes any open voice span (the actual emission happens lazily as
  /// the tail's `push_inference` arrives, or immediately if no inference
  /// is pending).
  pub fn finish(&mut self) {
    if self.finished {
      return;
    }
    self.finished = true;
    self.total_samples = self.total_samples_pushed;

    if !self.tail_emitted && self.total_samples_pushed > 0 {
      let starts = plan_starts(self.total_samples_pushed, self.opts.step_samples());
      let regular_emitted = self.next_window_idx as usize;
      for &start in starts.iter().skip(regular_emitted) {
        self.emit_window(start);
      }
      self.tail_emitted = true;
    }

    // Flush voice finalization. If pending is empty, this drains everything
    // and closes the open span. If pending is non-empty (tail just
    // scheduled), the boundary stalls and we'll close on the tail's
    // push_inference.
    self.process_voice_finalization();
  }

  /// Reset to empty state for a new stream.
  ///
  /// - input buffer cleared,
  /// - pending inferences dropped,
  /// - voice/hysteresis state reset,
  /// - `finished`/`tail_emitted` flags cleared,
  /// - `total_samples_pushed` reset to 0,
  /// - **a fresh process-wide generation token consumed**, so any stale
  ///   `WindowId` from before the `clear()` will fail
  ///   [`push_inference`](Self::push_inference) with
  ///   [`Error::UnknownWindow`].
  ///
  /// Internal allocations are reused. Does NOT discard or warm down a
  /// paired `SegmentModel`.
  pub fn clear(&mut self) {
    self.generation = GENERATION.fetch_add(1, Ordering::Relaxed);
    self.input.clear();
    self.consumed_samples = 0;
    self.total_samples_pushed = 0;
    self.next_window_idx = 0;
    self.pending.clear();
    self.pending_actions.clear();
    self.stitcher.clear();
    self.voice_hyst.reset();
    self.voice_run_start = None;
    self.merge_pending = None;
    self.finished = false;
    self.tail_emitted = false;
    self.total_samples = 0;
    self.pending_inference = None;
  }

  /// Number of [`Action::NeedsInference`] yielded but not yet fulfilled
  /// via [`push_inference`](Self::push_inference). Stays at zero in steady
  /// state.
  pub fn pending_inferences(&self) -> usize {
    self.pending.len()
  }

  /// Number of input samples currently buffered (pushed via
  /// [`push_samples`](Self::push_samples) but not yet released because
  /// they're still part of some not-yet-scheduled or in-flight window).
  ///
  /// Useful for detecting pathological backpressure: a steady increase
  /// despite calls to [`poll`](Self::poll) means the caller's inference
  /// loop has fallen behind. Canonical pattern:
  ///
  /// ```ignore
  /// const MAX_PENDING: usize = 16;
  /// if seg.pending_inferences() > MAX_PENDING {
  ///     // throttle the audio source until inference catches up
  /// }
  /// ```
  pub fn buffered_samples(&self) -> usize {
    self.input.len()
  }

  /// Where the next regular sliding window will start, in absolute samples.
  ///
  /// After [`finish`](Self::finish) is called, returns `u64::MAX` (no
  /// future regular windows; any tail anchor is already scheduled).
  ///
  /// **Do not use this for finalization** in a downstream
  /// reconstruction pump — it ignores the not-yet-emitted tail anchor.
  /// Use [`Self::tail_safe_finalization_boundary_samples`] instead.
  ///
  #[cfg(test)]
  pub(crate) fn peek_next_window_start(&self) -> u64 {
    if self.finished {
      return u64::MAX;
    }
    self.next_window_idx as u64 * self.opts.step_samples() as u64
  }

  /// Smallest absolute SAMPLE position past which downstream
  /// reconstruction can safely finalize frames — i.e. no future or
  /// already-pending window can still contribute past this point.
  ///
  /// Pre-finish: `min(next regular window start, earliest pending
  /// window start, total_samples_pushed - WINDOW_SAMPLES)`. The third
  /// term is the load-bearing one fixed by:
  /// `finish()` schedules a tail anchor at `total_samples_pushed -
  /// WINDOW_SAMPLES` (clamped to 0), and frames before that are
  /// touched by the tail's contribution. Without it, a stream like
  /// `230_000` samples (regular grid covers 0..160k and 40k..200k →
  /// drain finalizes to 80_000; tail later anchored at 70_000) lost
  /// the tail's contribution silently because reconstruction had
  /// already advanced past frame_index_of(70_000).
  ///
  /// Post-finish + all pending consumed: `u64::MAX` (everything
  /// finalizes).
  #[cfg(test)]
  pub(crate) fn tail_safe_finalization_boundary_samples(&self) -> u64 {
    if self.finished && self.pending.is_empty() {
      return u64::MAX;
    }
    let step = self.opts.step_samples() as u64;
    let next_window_start = if self.finished {
      // No more regular windows after finish; tail (if any) is in pending.
      u64::MAX
    } else {
      self.next_window_idx as u64 * step
    };
    let earliest_pending_start = self.pending.values().copied().min().unwrap_or(u64::MAX);
    // Tail-safe cap: only relevant pre-finish (after finish, the tail
    // is in `pending` and `earliest_pending_start` covers it).
    let tail_safe_start = if self.finished {
      u64::MAX
    } else {
      self
        .total_samples_pushed
        .saturating_sub(WINDOW_SAMPLES as u64)
    };
    next_window_start
      .min(earliest_pending_start)
      .min(tail_safe_start)
  }
}

#[inline]
fn duration_to_samples(d: core::time::Duration) -> u64 {
  let nanos = d.as_nanos();
  (nanos * SAMPLE_RATE_HZ as u128 / 1_000_000_000u128) as u64
}

/// `v` is finite (not NaN/±inf) and within `[0.0, 1.0]`. Mirrors the
/// `check_hysteresis_threshold` predicate used by the option setters,
/// hand-coded with `v == v` (NaN check) and direct `>=`/`<=` so it
/// can be used in `Segmenter::try_new`'s runtime path. The setter
/// path is `const fn` (which constrains how `is_finite` is phrased);
/// this fn does not need to be `const`, but stays consistent with
/// the same idiom for clarity.
#[inline]
fn is_finite_in_unit_interval(v: f32) -> bool {
  #[allow(clippy::eq_op)] // intentional NaN check: NaN != NaN by IEEE 754.
  let not_nan = !(v != v);
  not_nan && (0.0..=1.0).contains(&v)
}

/// `total_frames = ceil(total_samples * FRAMES_PER_WINDOW / WINDOW_SAMPLES)`
/// — the smallest absolute frame index whose start sample is at or past
/// `total_samples`. See spec §5.4.1 terminal-case definition.
#[inline]
fn total_frames_of(total_samples: u64) -> u64 {
  (total_samples * FRAMES_PER_WINDOW as u64).div_ceil(WINDOW_SAMPLES as u64)
}

/// Copy a `[Vec<f32>; MAX_SPEAKER_SLOTS]` (each of length
/// `FRAMES_PER_WINDOW`) into a fixed-size array suitable for
/// [`Action::SpeakerScores::raw_probs`].
#[inline]
fn speaker_probs_to_array(
  probs: &[Vec<f32>; MAX_SPEAKER_SLOTS as usize],
) -> [[f32; FRAMES_PER_WINDOW]; MAX_SPEAKER_SLOTS as usize] {
  let mut out = [[0.0f32; FRAMES_PER_WINDOW]; MAX_SPEAKER_SLOTS as usize];
  for (s, slot_probs) in probs.iter().enumerate() {
    debug_assert_eq!(slot_probs.len(), FRAMES_PER_WINDOW);
    out[s].copy_from_slice(slot_probs);
  }
  out
}

#[cfg(test)]
mod tests {
  use super::*;
  use mediatime::TimeRange;

  fn opts() -> SegmentOptions {
    SegmentOptions::default()
  }

  /// Synthetic powerset logits: speaker A "dominant" (class 1) for frames
  /// in `active_frames`, otherwise silence (class 0).
  fn synth_logits(active_frames: core::ops::Range<usize>) -> Vec<f32> {
    let mut out = vec![0.0f32; FRAMES_PER_WINDOW * POWERSET_CLASSES];
    for f in 0..FRAMES_PER_WINDOW {
      let row_start = f * POWERSET_CLASSES;
      for c in 0..POWERSET_CLASSES {
        out[row_start + c] = -10.0;
      }
      let active = active_frames.contains(&f);
      let dominant = if active { 1 } else { 0 };
      out[row_start + dominant] = 10.0;
    }
    out
  }

  #[test]
  fn empty_no_actions() {
    let mut s = Segmenter::new(opts());
    assert!(s.poll().is_none());
    assert_eq!(s.pending_inferences(), 0);
    assert_eq!(s.buffered_samples(), 0);
  }

  #[test]
  fn first_window_emits_after_full_window_buffered() {
    let mut s = Segmenter::new(opts());
    s.push_samples(&vec![0.1f32; 80_000]);
    assert!(s.poll().is_none());
    assert_eq!(s.buffered_samples(), 80_000);
    s.push_samples(&vec![0.2f32; 80_000]);
    match s.poll() {
      Some(Action::NeedsInference { id, samples }) => {
        assert_eq!(samples.len(), WINDOW_SAMPLES as usize);
        assert_eq!(id.range(), TimeRange::new(0, 160_000, SAMPLE_RATE_TB));
        assert!((samples[0] - 0.1).abs() < 1e-6);
        assert!((samples[80_000] - 0.2).abs() < 1e-6);
      }
      other => panic!("expected NeedsInference, got {other:?}"),
    }
    assert_eq!(s.pending_inferences(), 1);
  }

  #[test]
  fn second_window_emits_after_one_step_more_audio() {
    let mut s = Segmenter::new(opts());
    s.push_samples(&vec![0.0f32; 160_000]);
    let _ = s.poll();
    s.push_samples(&vec![0.0f32; 40_000]);
    match s.poll() {
      Some(Action::NeedsInference { id, .. }) => {
        assert_eq!(id.range(), TimeRange::new(40_000, 200_000, SAMPLE_RATE_TB));
      }
      other => panic!("expected NeedsInference, got {other:?}"),
    }
  }

  #[test]
  fn push_inference_wrong_length_errors() {
    let mut s = Segmenter::new(opts());
    s.push_samples(&vec![0.0; 160_000]);
    let id = match s.poll().unwrap() {
      Action::NeedsInference { id, .. } => id,
      _ => unreachable!(),
    };
    let bogus = vec![0.0f32; 100];
    match s.push_inference(id, &bogus) {
      Err(Error::InferenceShapeMismatch { expected, got }) => {
        assert_eq!(expected, FRAMES_PER_WINDOW * POWERSET_CLASSES);
        assert_eq!(got, 100);
      }
      other => panic!("unexpected: {other:?}"),
    }
  }

  #[test]
  fn push_inference_unknown_window_errors() {
    let mut s = Segmenter::new(opts());
    let bogus_id = WindowId::new(TimeRange::new(0, 160_000, SAMPLE_RATE_TB), 999);
    let scores = vec![0.0f32; FRAMES_PER_WINDOW * POWERSET_CLASSES];
    match s.push_inference(bogus_id, &scores) {
      Err(Error::UnknownWindow { .. }) => {}
      other => panic!("unexpected: {other:?}"),
    }
  }

  /// Calling push_inference twice with the same id: first succeeds, second
  /// returns UnknownWindow because the entry was consumed.
  #[test]
  fn push_inference_twice_with_same_id() {
    let mut s = Segmenter::new(opts());
    s.push_samples(&vec![0.0; 160_000]);
    let id = match s.poll().unwrap() {
      Action::NeedsInference { id, .. } => id,
      _ => unreachable!(),
    };
    let scores = synth_logits(0..0);
    s.push_inference(id, &scores).expect("first call ok");
    match s.push_inference(id, &scores) {
      Err(Error::UnknownWindow { .. }) => {}
      other => panic!("expected UnknownWindow on second call, got {other:?}"),
    }
  }

  /// Non-finite logits (`NaN`, `+inf`, `-inf`) must be rejected BEFORE
  /// the pending entry is consumed so the caller can retry. Without
  /// this gate, `softmax_row` produces `NaN` probabilities, downstream
  /// comparisons treat the window as silent, and the audio is silently
  /// dropped.
  #[test]
  fn push_inference_rejects_non_finite_and_keeps_pending() {
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
      let mut s = Segmenter::new(opts());
      s.push_samples(&vec![0.0; 160_000]);
      let id = match s.poll().unwrap() {
        Action::NeedsInference { id, .. } => id,
        _ => unreachable!(),
      };
      assert_eq!(s.pending_inferences(), 1);

      // Inject `bad` somewhere in the middle of an otherwise valid slice.
      let mut scores = synth_logits(0..0);
      scores[100] = bad;
      match s.push_inference(id, &scores) {
        Err(Error::NonFiniteScores { id: ret_id }) => assert_eq!(ret_id, id),
        other => panic!("expected NonFiniteScores for {bad}, got {other:?}"),
      }
      // Crucial: pending entry must still be there so caller can retry.
      assert_eq!(s.pending_inferences(), 1);

      // Retry with valid scores succeeds.
      let good = synth_logits(0..0);
      s.push_inference(id, &good).expect("retry should succeed");
      assert_eq!(s.pending_inferences(), 0);
    }
  }

  /// All-non-finite row: every score is NaN. Same rejection path; the
  /// window stays pending.
  #[test]
  fn push_inference_rejects_all_nan_row() {
    let mut s = Segmenter::new(opts());
    s.push_samples(&vec![0.0; 160_000]);
    let id = match s.poll().unwrap() {
      Action::NeedsInference { id, .. } => id,
      _ => unreachable!(),
    };
    let scores = vec![f32::NAN; FRAMES_PER_WINDOW * POWERSET_CLASSES];
    match s.push_inference(id, &scores) {
      Err(Error::NonFiniteScores { .. }) => {}
      other => panic!("expected NonFiniteScores, got {other:?}"),
    }
    assert_eq!(s.pending_inferences(), 1);
  }

  /// Stale-id from before clear() is rejected (spec §11.9).
  #[test]
  fn push_inference_stale_after_clear() {
    let mut s = Segmenter::new(opts());
    s.push_samples(&vec![0.0; 160_000]);
    let stale_id = match s.poll().unwrap() {
      Action::NeedsInference { id, .. } => id,
      _ => unreachable!(),
    };
    s.clear();
    s.push_samples(&vec![0.0; 160_000]);
    let _ = s.poll(); // discard the new id
    let scores = vec![0.0f32; FRAMES_PER_WINDOW * POWERSET_CLASSES];
    match s.push_inference(stale_id, &scores) {
      Err(Error::UnknownWindow { .. }) => {}
      other => panic!("expected UnknownWindow on stale id, got {other:?}"),
    }
  }

  /// Cross-Segmenter id collision (spec §11.9 #2): two `Segmenter`s both
  /// yield ids with the same TimeRange but different generations. Using
  /// one's id with the other returns UnknownWindow.
  #[test]
  fn push_inference_cross_segmenter_collision() {
    let mut a = Segmenter::new(opts());
    let mut b = Segmenter::new(opts());
    a.push_samples(&vec![0.0; 160_000]);
    b.push_samples(&vec![0.0; 160_000]);
    let id_a = match a.poll().unwrap() {
      Action::NeedsInference { id, .. } => id,
      _ => unreachable!(),
    };
    let id_b = match b.poll().unwrap() {
      Action::NeedsInference { id, .. } => id,
      _ => unreachable!(),
    };
    // Both ids cover the same sample range (0..160_000) but their
    // generations differ.
    assert_eq!(id_a.range(), id_b.range());
    assert_ne!(id_a, id_b);
    let scores = vec![0.0f32; FRAMES_PER_WINDOW * POWERSET_CLASSES];
    match b.push_inference(id_a, &scores) {
      Err(Error::UnknownWindow { .. }) => {}
      other => panic!("expected UnknownWindow on cross-segmenter id, got {other:?}"),
    }
  }

  #[test]
  fn one_window_speaker_a_active_emits_activity() {
    let mut s = Segmenter::new(opts());
    s.push_samples(&vec![0.0; 160_000]);
    let id = match s.poll().unwrap() {
      Action::NeedsInference { id, .. } => id,
      _ => unreachable!(),
    };
    let scores = synth_logits(100..200);
    s.push_inference(id, &scores).unwrap();

    let mut saw_activity = false;
    while let Some(a) = s.poll() {
      if let Action::Activity(act) = a {
        assert_eq!(act.window_id(), id);
        assert_eq!(act.speaker_slot(), 0);
        assert_eq!(act.range().timebase(), SAMPLE_RATE_TB);
        saw_activity = true;
      }
    }
    assert!(saw_activity, "expected at least one Activity for slot 0");
  }

  #[test]
  fn finish_short_clip_schedules_tail_window() {
    let mut s = Segmenter::new(opts());
    s.push_samples(&vec![0.0; 50_000]);
    assert!(s.poll().is_none());
    s.finish();
    match s.poll() {
      Some(Action::NeedsInference { samples, .. }) => {
        assert_eq!(samples.len(), WINDOW_SAMPLES as usize);
        for i in 0..50_000 {
          assert_eq!(samples[i], 0.0);
        }
        for i in 50_000..160_000 {
          assert_eq!(samples[i], 0.0);
        }
      }
      other => panic!("unexpected: {other:?}"),
    }
  }

  /// Empty stream: finish() after no push_samples (or only empty pushes)
  /// produces zero actions. Spec §11.10.
  #[test]
  fn empty_stream_no_actions() {
    let mut s = Segmenter::new(opts());
    s.push_samples(&[]);
    s.finish();
    assert!(s.poll().is_none());
    assert_eq!(s.pending_inferences(), 0);
    assert_eq!(s.buffered_samples(), 0);
  }

  /// Tail-window activity range is clamped to total_samples (spec §5.5).
  #[test]
  fn tail_window_activity_clamped_to_total_samples() {
    let mut s = Segmenter::new(opts());
    s.push_samples(&vec![0.0; 50_000]);
    s.finish();
    let id = match s.poll().unwrap() {
      Action::NeedsInference { id, .. } => id,
      _ => unreachable!(),
    };
    // All frames "speaker A active" — without clamping, activity would
    // span [0, 160_000) sample-wise.
    let scores = synth_logits(0..FRAMES_PER_WINDOW);
    s.push_inference(id, &scores).unwrap();
    let mut saw_activity = false;
    while let Some(a) = s.poll() {
      if let Action::Activity(act) = a {
        let r = act.range();
        // Range must be clamped at total_samples = 50_000.
        assert!(
          r.end_pts() <= 50_000,
          "activity end {} exceeds total_samples 50000",
          r.end_pts()
        );
        saw_activity = true;
      }
    }
    assert!(saw_activity);
  }

  #[test]
  fn clear_resets_state() {
    let mut s = Segmenter::new(opts());
    s.push_samples(&vec![0.0; 160_000]);
    let _ = s.poll();
    s.clear();
    assert!(s.poll().is_none());
    assert_eq!(s.pending_inferences(), 0);
    assert_eq!(s.buffered_samples(), 0);
    s.push_samples(&vec![0.0; 160_000]);
    match s.poll().unwrap() {
      Action::NeedsInference { id, .. } => {
        assert_eq!(id.range().start_pts(), 0);
      }
      _ => unreachable!(),
    }
  }

  ///: `clear()` must drop any stashed Layer-2
  /// inference so a fresh session doesn't accidentally retry one from
  /// the previous session. We exercise the field directly here because
  /// the streaming helpers populating it require an ONNX runtime not
  /// available in unit tests.
  #[test]
  fn clear_drops_layer2_pending_inference() {
    let mut s = Segmenter::new(opts());
    assert!(s.pending_inference.is_none());
    // Inject a fake stash so we can verify `clear()` drops it. Real
    // population comes from the `ort`-feature helpers.
    let bogus_id = WindowId::new(TimeRange::new(0, 160_000, SAMPLE_RATE_TB), 0);
    s.pending_inference = Some((bogus_id, vec![0.0f32; 4].into_boxed_slice()));
    s.clear();
    assert!(
      s.pending_inference.is_none(),
      "clear() must drop pending_inference"
    );
  }

  #[test]
  fn end_of_stream_closes_open_voice_span() {
    let mut s = Segmenter::new(opts());
    s.push_samples(&vec![0.0; 160_000]);
    let id = match s.poll().unwrap() {
      Action::NeedsInference { id, .. } => id,
      _ => unreachable!(),
    };
    let scores = synth_logits(0..FRAMES_PER_WINDOW);
    s.push_inference(id, &scores).unwrap();
    s.finish();
    if let Some(Action::NeedsInference { id: tail_id, .. }) = s.poll() {
      s.push_inference(tail_id, &scores).unwrap();
    }
    let mut found_voice = false;
    while let Some(a) = s.poll() {
      if matches!(a, Action::VoiceSpan(_)) {
        found_voice = true;
      }
    }
    assert!(found_voice, "expected a closing voice span on finish");
  }

  /// Out-of-order push_inference must NOT advance boundary past frames
  /// whose earlier-pending windows haven't reported. Spec §5.4.1 / T1-A.
  #[test]
  fn out_of_order_push_inference_holds_boundary() {
    let mut s = Segmenter::new(opts());
    // Push enough audio for windows 0, 1, 2 to all schedule.
    s.push_samples(&vec![0.0; 240_000]); // 0..240_000 covers 0..3 windows
    let mut ids: Vec<WindowId> = Vec::new();
    while let Some(a) = s.poll() {
      if let Action::NeedsInference { id, .. } = a {
        ids.push(id);
      }
    }
    assert_eq!(ids.len(), 3, "expected 3 pending NeedsInference");

    let scores = synth_logits(0..FRAMES_PER_WINDOW);
    // Push window 2's inference first.
    s.push_inference(ids[2], &scores).unwrap();
    // Boundary should be clamped at window 0's start frame (= 0); no
    // VoiceSpan should be emitted yet.
    let mut spans_after_2 = 0;
    while let Some(a) = s.poll() {
      if matches!(a, Action::VoiceSpan(_)) {
        spans_after_2 += 1;
      }
    }
    assert_eq!(
      spans_after_2, 0,
      "voice span emitted prematurely before earlier windows reported"
    );

    // Now push window 0's and 1's inferences.
    s.push_inference(ids[0], &scores).unwrap();
    s.push_inference(ids[1], &scores).unwrap();
    // After all three, boundary should advance to next_window_idx * step.
    // We don't strictly assert a span here (depends on hysteresis crossing
    // a boundary); just confirm the pipeline ran without error.
  }

  #[test]
  fn peek_next_window_start_advances_on_window_emit() {
    let mut s = Segmenter::new(opts());
    let step = SegmentOptions::default().step_samples() as u64;
    assert_eq!(s.peek_next_window_start(), 0);

    s.push_samples(&vec![0.001f32; 160_000]);
    let id = match s.poll() {
      Some(Action::NeedsInference { id, .. }) => id,
      other => panic!("expected NeedsInference, got {other:?}"),
    };
    // After the first regular window has been scheduled (its
    // NeedsInference dequeued), the next regular window starts at `step`.
    assert_eq!(s.peek_next_window_start(), step);

    let scores = vec![1.0f32 / POWERSET_CLASSES as f32; FRAMES_PER_WINDOW * POWERSET_CLASSES];
    s.push_inference(id, &scores).unwrap();
    while s.poll().is_some() {}
    assert_eq!(s.peek_next_window_start(), step);
  }

  #[test]
  fn peek_next_window_start_max_after_finish() {
    let mut s = Segmenter::new(opts());
    s.push_samples(&[0.001; 16_000]);
    s.finish();
    assert_eq!(s.peek_next_window_start(), u64::MAX);
  }

  /// regression: with the default step of 40_000
  /// and WINDOW_SAMPLES=160_000, a 230_000-sample stream:
  ///   - schedules regular windows at 0 (covers 0..160k) and 40k
  ///     (covers 40k..200k); a window at 80k would need 240k samples
  ///     so it is NOT scheduled pre-finish.
  ///   - `peek_next_window_start` returns 80_000 (next regular grid
  ///     position).
  ///   - But `finish()` will schedule a tail anchor at 70_000
  ///     (= total - WINDOW_SAMPLES). Frames in 70k..80k can still be
  ///     touched by that tail, so finalization MUST stay below 70k.
  ///
  /// `tail_safe_finalization_boundary_samples` enforces the min over
  /// next-regular, earliest-pending, and `total - WINDOW_SAMPLES`.
  #[test]
  fn tail_safe_finalization_boundary_clamps_below_future_tail() {
    let mut s = Segmenter::new(opts());
    s.push_samples(&vec![0.001f32; 230_000]);
    // Drain both regular windows' NeedsInference + push valid scores.
    let scores = vec![1.0f32 / POWERSET_CLASSES as f32; FRAMES_PER_WINDOW * POWERSET_CLASSES];
    let ids: Vec<WindowId> = (0..2)
      .map(|_| match s.poll().unwrap() {
        Action::NeedsInference { id, .. } => id,
        other => panic!("expected NeedsInference, got {other:?}"),
      })
      .collect();
    for id in &ids {
      s.push_inference(*id, &scores).unwrap();
    }
    // Drain remaining actions.
    while s.poll().is_some() {}

    // peek_next_window_start says next regular start = 80_000 ...
    assert_eq!(s.peek_next_window_start(), 80_000);
    // ... but the tail-safe boundary clamps below 70_000 to leave room
    // for the future tail.
    let tail_safe = s.tail_safe_finalization_boundary_samples();
    assert!(
      tail_safe <= 70_000,
      "tail_safe_finalization_boundary_samples must be <= 70_000 \
       (= total - WINDOW_SAMPLES); got {tail_safe}"
    );
  }

  /// After finish + all pending consumed, the tail-safe boundary
  /// returns u64::MAX so downstream consumers can finalize everything.
  #[test]
  fn tail_safe_finalization_boundary_after_finish_and_drain() {
    let mut s = Segmenter::new(opts());
    s.push_samples(&vec![0.001f32; 160_000]);
    let id = match s.poll().unwrap() {
      Action::NeedsInference { id, .. } => id,
      _ => unreachable!(),
    };
    let scores = vec![1.0f32 / POWERSET_CLASSES as f32; FRAMES_PER_WINDOW * POWERSET_CLASSES];
    s.push_inference(id, &scores).unwrap();
    while s.poll().is_some() {}
    s.finish();
    // finish() schedules the tail; consume it.
    while let Some(action) = s.poll() {
      if let Action::NeedsInference { id, .. } = action {
        s.push_inference(id, &scores).unwrap();
      }
    }
    while s.poll().is_some() {}
    assert!(s.pending.is_empty(), "all pending should be consumed");
    assert_eq!(
      s.tail_safe_finalization_boundary_samples(),
      u64::MAX,
      "post-finish + pending empty must allow full finalization"
    );
  }

  #[test]
  fn tail_window_audio_aligned_with_claimed_start() {
    //-severity regression: with default step = 40_000
    // and WINDOW_SAMPLES = 160_000, push 230_000 samples in one shot.
    // Two regular windows fire (idx 0 → 0, idx 1 → 40_000). finish()
    // then schedules a tail window at 230_000 - 160_000 = 70_000.
    // Without the fix, consumed_samples advances to 80_000 after the
    // second regular emit, and the tail window's audio is shifted by
    // 10_000 samples while the WindowId still claims start = 70_000.
    let mut s = Segmenter::new(opts());

    // Build a sentinel signal: every sample equals its own absolute index
    // (cast to f32). Any misalignment shows up as a constant offset in
    // the emitted samples.
    let total: i32 = 230_000;
    let samples: Vec<f32> = (0..total).map(|i| i as f32).collect();
    s.push_samples(&samples);
    s.finish();

    // Drain all NeedsInference actions; record (claimed start, samples).
    let mut emitted: Vec<(u64, Box<[f32]>)> = Vec::new();
    let scores = vec![1.0f32 / POWERSET_CLASSES as f32; FRAMES_PER_WINDOW * POWERSET_CLASSES];
    while let Some(action) = s.poll() {
      if let Action::NeedsInference { id, samples } = action {
        emitted.push((id.range().start_pts() as u64, samples));
        s.push_inference(id, &scores).unwrap();
      }
    }

    // We expect exactly 3 windows: 0, 40_000, 70_000 (tail).
    assert!(
      emitted.iter().any(|(start, _)| *start == 0),
      "missing regular window at 0"
    );
    assert!(
      emitted.iter().any(|(start, _)| *start == 40_000),
      "missing regular window at 40_000"
    );
    let tail = emitted
      .iter()
      .find(|(start, _)| *start == 70_000)
      .expect("missing tail window at 70_000");

    // The tail window's samples must satisfy: samples[k] == 70_000 + k as f32
    // for every k in [0, total - 70_000) = [0, 160_000). Since the input
    // ended at sample 230_000 (== 70_000 + 160_000), the entire window
    // is covered with no zero padding. (If our fix is broken, samples[k]
    // for small k would be 80_000 + k or zero-padding instead.)
    for (k, &v) in tail.1.iter().enumerate() {
      let expected = 70_000.0 + k as f32;
      assert_eq!(
        v, expected,
        "tail window sample[{k}] = {v}, expected {expected} (audio misaligned)"
      );
    }
    assert_eq!(tail.1.len(), WINDOW_SAMPLES as usize);
  }

  #[test]
  fn push_inference_emits_speaker_scores_before_activities() {
    let mut s = Segmenter::new(opts());
    s.push_samples(&vec![0.001f32; 160_000]);
    let id = match s.poll() {
      Some(Action::NeedsInference { id, .. }) => id,
      other => panic!("expected NeedsInference, got {other:?}"),
    };
    let scores = vec![1.0f32 / POWERSET_CLASSES as f32; FRAMES_PER_WINDOW * POWERSET_CLASSES];
    s.push_inference(id, &scores).unwrap();

    let mut saw_scores = false;
    while let Some(action) = s.poll() {
      match action {
        Action::SpeakerScores {
          id: sid,
          window_start,
          raw_probs,
        } => {
          assert_eq!(sid, id);
          assert_eq!(window_start, 0);
          assert_eq!(raw_probs.len(), MAX_SPEAKER_SLOTS as usize);
          assert_eq!(raw_probs[0].len(), FRAMES_PER_WINDOW);
          saw_scores = true;
        }
        Action::Activity(_) => {
          assert!(saw_scores, "Activity emitted before SpeakerScores");
        }
        Action::VoiceSpan(_) => {}
        _ => {}
      }
    }
    assert!(saw_scores, "no SpeakerScores emitted");
  }

  // ── try_new option validation (serde-bypass guards) ──────────────
  //
  // SegmentOptions::with_*/set_* enforce the contract on the builder
  // path, but a #[serde] deserialize bypasses those entry points and
  // can construct a SegmentOptions with bad values directly. These
  // tests construct violating options manually and confirm that
  // try_new rejects them with a typed error.

  /// Build a SegmentOptions with a custom step bypassing the panic-
  /// validating setter. Round-trips defaults through the public
  /// API and then mutates the field via serde to avoid the assert.
  ///
  /// Without serde we cannot synthesize an out-of-range
  /// SegmentOptions in stable Rust (the field is private and the
  /// setters panic). The test gates on the `serde` feature.
  #[cfg(feature = "serde")]
  fn opts_from_json(json: &str) -> SegmentOptions {
    serde_json::from_str(json).expect("deserialize SegmentOptions")
  }

  /// Helper: assert `try_new(opts)` returned an `Err` matching
  /// `pat`. Uses `match` rather than `expect_err` because `Segmenter`
  /// is not `Debug` (it owns large internal state we deliberately
  /// don't expose for diagnostic dumps).
  #[cfg(feature = "serde")]
  fn assert_try_new_err<F>(opts: SegmentOptions, label: &str, check: F)
  where
    F: FnOnce(&Error) -> bool,
  {
    match Segmenter::try_new(opts) {
      Ok(_) => panic!("try_new must reject {label}, got Ok"),
      Err(e) => assert!(check(&e), "try_new returned wrong error for {label}: {e:?}"),
    }
  }

  #[cfg(feature = "serde")]
  #[test]
  fn try_new_rejects_step_above_window_via_serde() {
    use crate::segment::error::InvalidOptionsReason;
    let json = format!(r#"{{"step_samples": {}}}"#, WINDOW_SAMPLES + 1);
    assert_try_new_err(opts_from_json(&json), "step > WINDOW_SAMPLES", |e| {
      matches!(
        e,
        Error::InvalidOptions(InvalidOptionsReason::StepSamplesExceedsWindow { .. })
      )
    });
  }

  #[cfg(feature = "serde")]
  #[test]
  fn try_new_rejects_zero_step_via_serde() {
    use crate::segment::error::InvalidOptionsReason;
    assert_try_new_err(opts_from_json(r#"{"step_samples": 0}"#), "step == 0", |e| {
      matches!(
        e,
        Error::InvalidOptions(InvalidOptionsReason::ZeroStepSamples)
      )
    });
  }

  /// `is_finite_in_unit_interval` is the predicate `Segmenter::try_new`
  /// uses to gate hysteresis thresholds against NaN/±inf and out-of-
  /// `[0,1]` values. JSON cannot carry `NaN`, so we cannot exercise
  /// that path via serde — covering the predicate directly is the
  /// canonical alternative. Boundary cases (`< 0`, `> 1.0`) are
  /// covered by the `serde`-driven tests that follow.
  #[test]
  fn is_finite_in_unit_interval_predicate() {
    assert!(is_finite_in_unit_interval(0.0));
    assert!(is_finite_in_unit_interval(0.5));
    assert!(is_finite_in_unit_interval(1.0));
    assert!(!is_finite_in_unit_interval(-0.001));
    assert!(!is_finite_in_unit_interval(1.001));
    assert!(!is_finite_in_unit_interval(f32::NAN));
    assert!(!is_finite_in_unit_interval(f32::INFINITY));
    assert!(!is_finite_in_unit_interval(f32::NEG_INFINITY));
  }

  #[cfg(feature = "serde")]
  #[test]
  fn try_new_rejects_above_one_onset_via_serde() {
    use crate::segment::error::InvalidOptionsReason;
    assert_try_new_err(
      opts_from_json(r#"{"onset_threshold": 1.5}"#),
      "onset > 1.0",
      |e| {
        matches!(
          e,
          Error::InvalidOptions(InvalidOptionsReason::HysteresisThresholdOutOfRange {
            which: "onset",
            ..
          })
        )
      },
    );
  }

  #[cfg(feature = "serde")]
  #[test]
  fn try_new_rejects_negative_offset_via_serde() {
    use crate::segment::error::InvalidOptionsReason;
    assert_try_new_err(
      opts_from_json(r#"{"offset_threshold": -0.1}"#),
      "offset < 0.0",
      |e| {
        matches!(
          e,
          Error::InvalidOptions(InvalidOptionsReason::HysteresisThresholdOutOfRange {
            which: "offset",
            ..
          })
        )
      },
    );
  }

  /// `offset > onset` makes the falling-edge unreachable so a started
  /// voice run never closes. Bypass the setter check via serde and
  /// confirm try_new rejects the inverted ordering.
  #[cfg(feature = "serde")]
  #[test]
  fn try_new_rejects_offset_above_onset_via_serde() {
    use crate::segment::error::InvalidOptionsReason;
    assert_try_new_err(
      opts_from_json(r#"{"onset_threshold": 0.3, "offset_threshold": 0.6}"#),
      "offset > onset",
      |e| {
        matches!(
          e,
          Error::InvalidOptions(InvalidOptionsReason::OffsetAboveOnset { .. })
        )
      },
    );
  }

  /// At the boundary: step == WINDOW_SAMPLES is accepted.
  #[cfg(feature = "serde")]
  #[test]
  fn try_new_accepts_step_at_window_boundary() {
    let json = format!(r#"{{"step_samples": {WINDOW_SAMPLES}}}"#);
    let opts = opts_from_json(&json);
    let _ = Segmenter::try_new(opts).map_err(|e| {
      panic!("step == WINDOW_SAMPLES must be accepted, got {e:?}");
    });
  }
}
