//! Boundary tests for `diarization::offline::diarize_offline`.
//!
//! These tests exercise the Stage-0 boundary checks added to fail
//! fast on a malformed input *before* spill-backed
//! `embeddings`/`post_plda` allocation, PLDA projection, and the
//! `assign_embeddings` (AHC + VBx + Hungarian) chain. They use
//! synthetic inputs with the smallest valid dimensions so the only
//! failure surface is the targeted boundary check.

use crate::{
  embed::EMBEDDING_DIM,
  offline::{Error, OfflineInput, diarize_offline},
  plda::PldaTransform,
  reconstruct::{ShapeError as ReconstructShapeError, SlidingWindow},
  segment::options::MAX_SPEAKER_SLOTS,
};

/// Build a minimal valid `OfflineInput`-shaped data set: well-formed
/// raw_embeddings + segmentations matching `num_chunks * num_speakers
/// * num_frames_per_chunk`, default sliding windows, with the count
/// tensor controlled by the caller. The PLDA transform is bundled.
fn synthetic_inputs(
  num_chunks: usize,
  num_frames_per_chunk: usize,
) -> (
  Vec<f32>,
  Vec<f64>,
  PldaTransform,
  SlidingWindow,
  SlidingWindow,
) {
  let num_speakers = MAX_SPEAKER_SLOTS as usize;
  let raw = vec![0.5_f32; num_chunks * num_speakers * EMBEDDING_DIM];
  let seg = vec![0.5_f64; num_chunks * num_frames_per_chunk * num_speakers];
  let plda = PldaTransform::new().expect("PldaTransform::new");
  // Pyannote community-1 timing: 10 s chunk window, 1 s step,
  // 0.0167 s frame duration/step (16 ms ≈ 1/60 s).
  let chunks_sw = SlidingWindow::new(0.0, 10.0, 1.0);
  let frames_sw = SlidingWindow::new(0.0, 0.0167, 0.0167);
  (raw, seg, plda, chunks_sw, frames_sw)
}

/// `count.len() != num_output_frames` must surface
/// `Error::Reconstruct(Shape(CountLenMismatch))` *before* the offline
/// stage-1 filter pass and the spill-backed `embeddings` /
/// `post_plda` allocations. Using `num_output_frames = 64` and a
/// `count` of length 0 keeps every other field valid so the only
/// failure surface is this boundary check.
#[test]
fn rejects_count_length_mismatch_before_clustering() {
  let num_chunks = 1;
  let num_frames_per_chunk = 4;
  let (raw, seg, plda, chunks_sw, frames_sw) = synthetic_inputs(num_chunks, num_frames_per_chunk);
  let bad_count: Vec<u8> = Vec::new();
  let num_output_frames = 64;
  let input = OfflineInput::new(
    &raw,
    num_chunks,
    MAX_SPEAKER_SLOTS as usize,
    &seg,
    num_frames_per_chunk,
    &bad_count,
    num_output_frames,
    chunks_sw,
    frames_sw,
    &plda,
  );
  let r = diarize_offline(&input);
  assert!(
    matches!(
      r,
      Err(Error::Reconstruct(crate::reconstruct::Error::Shape(
        ReconstructShapeError::CountLenMismatch
      )))
    ),
    "expected Reconstruct(Shape(CountLenMismatch)), got {r:?}"
  );
}

/// `num_output_frames == 0` must fail at the offline
/// boundary with `ZeroNumOutputFrames`. The reconstruct module's own
/// check fires on the same predicate, but only after stage 1-4 burn
/// PLDA projection, AHC, VBx, and centroid work.
#[test]
fn rejects_zero_num_output_frames_before_clustering() {
  let num_chunks = 1;
  let num_frames_per_chunk = 4;
  let (raw, seg, plda, chunks_sw, frames_sw) = synthetic_inputs(num_chunks, num_frames_per_chunk);
  let bad_count: Vec<u8> = Vec::new();
  let num_output_frames = 0;
  let input = OfflineInput::new(
    &raw,
    num_chunks,
    MAX_SPEAKER_SLOTS as usize,
    &seg,
    num_frames_per_chunk,
    &bad_count,
    num_output_frames,
    chunks_sw,
    frames_sw,
    &plda,
  );
  let r = diarize_offline(&input);
  assert!(
    matches!(
      r,
      Err(Error::Reconstruct(crate::reconstruct::Error::Shape(
        ReconstructShapeError::ZeroNumOutputFrames
      )))
    ),
    "expected Reconstruct(Shape(ZeroNumOutputFrames)), got {r:?}"
  );
}

/// a single `count[t] > MAX_COUNT_PER_FRAME` must surface
/// `CountAboveMax`. `255` is the canonical `u8` sentinel-corruption
/// value that this gate is sized to catch (theoretical max for
/// community-1 is `~30`; the cap of `64` allows headroom while
/// rejecting upstream overflow).
#[test]
fn rejects_count_above_max_before_clustering() {
  let num_chunks = 1;
  let num_frames_per_chunk = 4;
  let num_output_frames = 64;
  let (raw, seg, plda, chunks_sw, frames_sw) = synthetic_inputs(num_chunks, num_frames_per_chunk);
  let mut bad_count: Vec<u8> = vec![1; num_output_frames];
  bad_count[5] = u8::MAX; // single poison cell, well above the cap of 64
  let input = OfflineInput::new(
    &raw,
    num_chunks,
    MAX_SPEAKER_SLOTS as usize,
    &seg,
    num_frames_per_chunk,
    &bad_count,
    num_output_frames,
    chunks_sw,
    frames_sw,
    &plda,
  );
  let r = diarize_offline(&input);
  assert!(
    matches!(
      r,
      Err(Error::Reconstruct(crate::reconstruct::Error::Shape(
        ReconstructShapeError::CountAboveMax
      )))
    ),
    "expected Reconstruct(Shape(CountAboveMax)), got {r:?}"
  );
}
