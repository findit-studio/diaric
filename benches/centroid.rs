//! Weighted-centroid throughput baseline.
//!
//! Times `diarization::cluster::centroid::weighted_centroids` — the post-VBx
//! `W = q[:, sp > threshold]; centroids = W.T @ raw / W.sum(0).T`
//! AXPY accumulator. The dominant cost is the inner
//! `centroids[k, d] += w * embed[t, d]` loop, sized `K_alive · T · D`.
//!
//! Per-fixture shape (T = num_train, K_alive ≤ 2 in all fixtures, D = 256):
//!
//! - `01_dialogue`        — T=195, K=2 → ~100K f64 ops
//! - `02_pyannote_sample` — T=37,  K=2
//! - `03_dual_speaker`    — T=41,  K=1
//! - `04_three_speaker`   — T=16,  K=1
//! - `05_four_speaker`    — T=32,  K=1
//!
//! Run: `cargo bench --bench centroid --features _bench`.

use std::{fs::File, hint::black_box, io::BufReader, path::PathBuf};

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use diarization::cluster::centroid::{SP_ALIVE_THRESHOLD, weighted_centroids};
use nalgebra::{DMatrix, DVector};
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

struct CentroidInputs {
  q: DMatrix<f64>,
  sp: DVector<f64>,
  embeddings: Vec<f64>,
  num_train: usize,
  embed_dim: usize,
}

fn load(fixture_name: &str) -> CentroidInputs {
  let vbx_path = fixture(fixture_name, "vbx_state.npz");
  let raw_path = fixture(fixture_name, "raw_embeddings.npz");
  let plda_path = fixture(fixture_name, "plda_embeddings.npz");

  let (q_flat, q_shape) = read_npz::<f64>(&vbx_path, "q_final");
  let (sp_flat, _) = read_npz::<f64>(&vbx_path, "sp_final");
  let num_train = q_shape[0] as usize;
  let num_init = q_shape[1] as usize;
  let q = DMatrix::<f64>::from_row_slice(num_train, num_init, &q_flat);
  let sp = DVector::<f64>::from_vec(sp_flat);

  let (raw_flat, raw_shape) = read_npz::<f32>(&raw_path, "embeddings");
  let num_speakers = raw_shape[1] as usize;
  let embed_dim = raw_shape[2] as usize;
  let (chunk_idx, _) = read_npz::<i64>(&plda_path, "train_chunk_idx");
  let (speaker_idx, _) = read_npz::<i64>(&plda_path, "train_speaker_idx");
  let mut embeddings = vec![0.0_f64; num_train * embed_dim];
  for i in 0..num_train {
    let c = chunk_idx[i] as usize;
    let s = speaker_idx[i] as usize;
    let base = (c * num_speakers + s) * embed_dim;
    for d in 0..embed_dim {
      embeddings[i * embed_dim + d] = raw_flat[base + d] as f64;
    }
  }

  CentroidInputs {
    q,
    sp,
    embeddings,
    num_train,
    embed_dim,
  }
}

fn bench(c: &mut Criterion) {
  let mut group = c.benchmark_group("weighted_centroids");
  for &name in FIXTURES {
    let inputs = load(name);
    group.bench_with_input(BenchmarkId::from_parameter(name), &inputs, |b, inp| {
      b.iter(|| {
        let centroids = weighted_centroids(
          black_box(&inp.q),
          black_box(&inp.sp),
          black_box(&inp.embeddings),
          black_box(inp.num_train),
          black_box(inp.embed_dim),
          SP_ALIVE_THRESHOLD,
        )
        .expect("weighted_centroids");
        black_box(centroids);
      });
    });
  }
  group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
