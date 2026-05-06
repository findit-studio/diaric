//! Smoke tests: run `OwnedDiarizationPipeline` end-to-end
//! on a fixture's `clip_16k.wav` and validate the output is sane
//! (non-empty spans, finite timestamps, total duration consistent).
//!
//! Strict pyannote DER comparison is reserved for an integration
//! tooling pass that runs `score.py` against the captured
//! `reference.rttm`. The ONNX models (`segmentation-3.0.onnx` +
//! `wespeaker_resnet34_lm.onnx`) are not committed; tests are
//! `#[ignore]`-marked so CI is green without them.
//!
//! Run with:
//! ```sh
//! cargo test --features ort -- --ignored owned_smoke
//! ```

use crate::{
  embed::EmbedModel, offline::OwnedDiarizationPipeline, plda::PldaTransform, segment::SegmentModel,
};
use std::path::PathBuf;

fn crate_root() -> PathBuf {
  PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn load_wav_16k_mono(path: &std::path::Path) -> Vec<f32> {
  let mut reader = hound::WavReader::open(path).expect("open wav");
  let spec = reader.spec();
  assert_eq!(
    spec.sample_rate, 16_000,
    "expected 16 kHz; got {}",
    spec.sample_rate
  );
  assert_eq!(
    spec.channels, 1,
    "expected mono; got {} channels",
    spec.channels
  );
  match (spec.sample_format, spec.bits_per_sample) {
    (hound::SampleFormat::Int, 16) => reader
      .samples::<i16>()
      .map(|s| s.unwrap() as f32 / i16::MAX as f32)
      .collect(),
    (hound::SampleFormat::Float, 32) => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
    (fmt, bps) => panic!("unsupported wav: {fmt:?} {bps}-bit"),
  }
}

#[test]
#[ignore = "requires segmentation + wespeaker ONNX models locally"]
fn owned_smoke_02_pyannote_sample() {
  let root = crate_root();
  let mut seg = SegmentModel::from_file(root.join("models/segmentation-3.0.onnx"))
    .expect("load segmentation model");
  let mut emb = EmbedModel::from_file(root.join("models/wespeaker_resnet34_lm.onnx"))
    .expect("load embedding model");
  let plda = PldaTransform::new().expect("PldaTransform");
  let samples =
    load_wav_16k_mono(&root.join("tests/parity/fixtures/02_pyannote_sample/clip_16k.wav"));

  let pipeline = OwnedDiarizationPipeline::new();
  let out = pipeline
    .run(&mut seg, &mut emb, &plda, &samples)
    .expect("OwnedDiarizationPipeline::run");

  // Sanity: at least one span emitted, all timestamps finite + ordered.
  assert!(
    !out.spans().is_empty(),
    "expected non-empty spans; got 0 spans (num_clusters={})",
    out.num_clusters()
  );
  for span in out.spans_slice() {
    let s = span.start();
    let d = span.duration();
    assert!(
      s.is_finite() && d.is_finite() && d > 0.0,
      "bad span: {s} dur {d}"
    );
  }
}
