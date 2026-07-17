//! VBx EM-iteration throughput baseline.
//!
//! Times `diarization::cluster::vbx::vbx_iterate` end-to-end on each captured
//! fixture, holding the inputs constant across iterations. The
//! per-iteration time covers all `max_iters = 20` EM rounds plus the
//! pre-loop matrix setup.
//!
//! Per-fixture shape (T = num_train, S = num_init_clusters, D = 128):
//!
//! - `01_dialogue`        — T=195, S=19
//! - `02_pyannote_sample` — T=37,  S=4
//! - `03_dual_speaker`    — T=41,  S=6
//! - `04_three_speaker`   — T=16,  S=4
//! - `05_four_speaker`    — T=32,  S=3
//!
//! Run: `cargo bench --bench vbx --features _bench`.

use std::{fs::File, hint::black_box, io::BufReader, path::PathBuf};

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use diarization::cluster::vbx::vbx_iterate;
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

fn read_npz<T: npyz::Deserialize>(path: &PathBuf, key: &str) -> Vec<T> {
  let f = File::open(path).expect("open npz");
  let mut z = NpzArchive::new(BufReader::new(f)).expect("read npz");
  let npy = z
    .by_name(key)
    .expect("query archive")
    .unwrap_or_else(|| panic!("array `{key}` not in {}", path.display()));
  npy.into_vec().expect("decode array")
}

struct VbxInputs {
  post_plda: DMatrix<f64>,
  phi: DVector<f64>,
  qinit: DMatrix<f64>,
  fa: f64,
  fb: f64,
  max_iters: usize,
}

fn load(fixture_name: &str) -> VbxInputs {
  let plda_path = fixture(fixture_name, "plda_embeddings.npz");
  let vbx_path = fixture(fixture_name, "vbx_state.npz");

  let post_plda_flat = read_npz::<f64>(&plda_path, "post_plda");
  let phi_flat = read_npz::<f64>(&plda_path, "phi");
  let qinit_flat = read_npz::<f64>(&vbx_path, "qinit");
  let fa = read_npz::<f64>(&vbx_path, "fa")[0];
  let fb = read_npz::<f64>(&vbx_path, "fb")[0];
  let max_iters = read_npz::<i64>(&vbx_path, "max_iters")[0] as usize;

  let plda_dim = phi_flat.len();
  let num_train = post_plda_flat.len() / plda_dim;
  let num_init = qinit_flat.len() / num_train;
  VbxInputs {
    post_plda: DMatrix::<f64>::from_row_slice(num_train, plda_dim, &post_plda_flat),
    phi: DVector::<f64>::from_vec(phi_flat),
    qinit: DMatrix::<f64>::from_row_slice(num_train, num_init, &qinit_flat),
    fa,
    fb,
    max_iters,
  }
}

fn bench(c: &mut Criterion) {
  let mut group = c.benchmark_group("vbx_iterate");
  for &name in FIXTURES {
    let inputs = load(name);
    group.bench_with_input(BenchmarkId::from_parameter(name), &inputs, |b, inp| {
      b.iter(|| {
        let out = vbx_iterate(
          black_box(inp.post_plda.as_view()),
          black_box(&inp.phi),
          black_box(&inp.qinit),
          black_box(inp.fa),
          black_box(inp.fb),
          black_box(inp.max_iters),
        )
        .expect("vbx_iterate");
        black_box(out);
      });
    });
  }
  group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
