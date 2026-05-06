//! Layer-1 throughput bench. Runs `Segmenter` with synthetic scores so we
//! measure state-machine cost only (no ort).

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use diarization::segment::{
  Action, FRAMES_PER_WINDOW, POWERSET_CLASSES, SegmentOptions, Segmenter,
};

fn synth_scores() -> Vec<f32> {
  let mut out = vec![-10.0f32; FRAMES_PER_WINDOW * POWERSET_CLASSES];
  for f in 0..FRAMES_PER_WINDOW {
    out[f * POWERSET_CLASSES + 1] = 10.0;
  }
  out
}

fn bench_one_minute(c: &mut Criterion) {
  let scores = synth_scores();
  let pcm = vec![0.0f32; 16_000 * 60]; // one minute at 16 kHz
  c.bench_function("segmenter_one_minute_layer1", |b| {
    b.iter_batched(
      || Segmenter::new(SegmentOptions::default()),
      |mut seg| {
        for chunk in pcm.chunks(1_600) {
          seg.push_samples(chunk);
          while let Some(a) = seg.poll() {
            match a {
              Action::NeedsInference { id, .. } => {
                seg.push_inference(id, &scores).unwrap();
              }
              Action::Activity(_) | Action::VoiceSpan(_) => {}
              _ => {}
            }
          }
        }
        seg.finish();
        while let Some(a) = seg.poll() {
          if let Action::NeedsInference { id, .. } = a {
            seg.push_inference(id, &scores).unwrap();
          }
        }
      },
      BatchSize::SmallInput,
    );
  });
}

criterion_group!(benches, bench_one_minute);
criterion_main!(benches);
