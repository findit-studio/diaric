//! AHC initialization throughput baseline.
//!
//! Times `diarization::cluster::ahc::ahc_init` (L2-normalize → centroid linkage
//! → fcluster + remap) on each captured fixture's training-embedding
//! subset.
//!
//! Per-fixture shape (N = num_train, D = 256 raw embed dim):
//!
//! - `01_dialogue`        — N=195
//! - `02_pyannote_sample` — N=37
//! - `03_dual_speaker`    — N=41
//! - `04_three_speaker`   — N=16
//! - `05_four_speaker`    — N=32
//!
//! `pdist_euclidean` cost ≈ N²·D/2 — `01_dialogue` is the dominant case
//! (~5M f64 ops). The other fixtures should run in microseconds.
//!
//! Run: `cargo bench --bench ahc --features _bench`.

use std::{fs::File, hint::black_box, io::BufReader, path::PathBuf};

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use diarization::{cluster::ahc::ahc_init, ops::spill::SpillOptions};
use npyz::npz::NpzArchive;

const FIXTURES: &[&str] = &[
  "01_dialogue",
  "02_pyannote_sample",
  "03_dual_speaker",
  "04_three_speaker",
  "05_four_speaker",
  "06_long_recording",
];

fn fixture(name: &str, file: &str) -> PathBuf {
  PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    .join("tests/parity/fixtures")
    .join(name)
    .join(file)
}

fn read_npz<T: npyz::Deserialize>(path: &PathBuf, key: &str) -> (Vec<T>, Vec<u64>) {
  let f = File::open(path).expect("open npz");
  let mut z = NpzArchive::new(BufReader::new(f)).expect("read npz");
  let npy = z
    .by_name(key)
    .expect("query archive")
    .unwrap_or_else(|| panic!("array `{key}` not in {}", path.display()));
  let shape = npy.shape().to_vec();
  let data = npy.into_vec().expect("decode array");
  (data, shape)
}

struct AhcInputs {
  train_embeddings: Vec<f64>,
  num_train: usize,
  embed_dim: usize,
  threshold: f64,
}

fn load(fixture_name: &str) -> AhcInputs {
  // Project raw embeddings to active subset via captured train_*idx.
  let raw_path = fixture(fixture_name, "raw_embeddings.npz");
  let plda_path = fixture(fixture_name, "plda_embeddings.npz");
  let ahc_path = fixture(fixture_name, "ahc_state.npz");
  let (raw_flat, raw_shape) = read_npz::<f32>(&raw_path, "embeddings");
  let num_speakers = raw_shape[1] as usize;
  let embed_dim = raw_shape[2] as usize;
  let (chunk_idx, _) = read_npz::<i64>(&plda_path, "train_chunk_idx");
  let (speaker_idx, _) = read_npz::<i64>(&plda_path, "train_speaker_idx");
  let num_train = chunk_idx.len();
  let mut train_embeddings = vec![0.0_f64; num_train * embed_dim];
  for i in 0..num_train {
    let c = chunk_idx[i] as usize;
    let s = speaker_idx[i] as usize;
    let base = (c * num_speakers + s) * embed_dim;
    for d in 0..embed_dim {
      train_embeddings[i * embed_dim + d] = raw_flat[base + d] as f64;
    }
  }
  let threshold = read_npz::<f64>(&ahc_path, "threshold").0[0];
  AhcInputs {
    train_embeddings,
    num_train,
    embed_dim,
    threshold,
  }
}

fn bench(c: &mut Criterion) {
  let mut group = c.benchmark_group("ahc_init");
  let spill_opts = SpillOptions::new();
  for &name in FIXTURES {
    let inputs = load(name);
    group.bench_with_input(BenchmarkId::from_parameter(name), &inputs, |b, inp| {
      b.iter(|| {
        let labels = ahc_init(
          black_box(&inp.train_embeddings),
          black_box(inp.num_train),
          black_box(inp.embed_dim),
          black_box(inp.threshold),
          black_box(&spill_opts),
        )
        .expect("ahc_init");
        black_box(labels);
      });
    });
  }
  group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
