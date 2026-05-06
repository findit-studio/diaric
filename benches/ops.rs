//! Per-primitive scalar-vs-SIMD A/B benchmark.
//!
//! Each `[crate::ops]` primitive is exercised at the production
//! dimensions used by the pipeline (D = 192 PLDA, D = 256 raw embed),
//! plus N values that bracket the per-fixture loads. `simd=true` /
//! `simd=false` are run as adjacent rows so criterion prints the
//! delta in one chart.
//!
//! Run: `cargo bench --bench ops --features _bench`.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use diarization::ops;
use rand::{SeedableRng, prelude::*};
use rand_chacha::ChaCha20Rng;

fn rand_vec(n: usize, seed: u64) -> Vec<f64> {
  let mut rng = ChaCha20Rng::seed_from_u64(seed);
  (0..n).map(|_| rng.random::<f64>() * 2.0 - 1.0).collect()
}

const DIMS: &[usize] = &[192, 256];
// `n_rows` for pdist — bracket the AHC fixture range. AHC is
// O(N²·D), so large N times exceed 1s; capped here for `--quick`.
const PDIST_N: &[usize] = &[64, 128, 200];

fn bench_dot(c: &mut Criterion) {
  let mut group = c.benchmark_group("dot");
  for &d in DIMS {
    let a = rand_vec(d, 0xa1);
    let b = rand_vec(d, 0xb2);
    group.bench_function(BenchmarkId::new(format!("d={d}"), "simd"), |bn| {
      bn.iter(|| {
        let r = ops::dot(black_box(&a), black_box(&b));
        black_box(r);
      });
    });
    group.bench_function(BenchmarkId::new(format!("d={d}"), "scalar"), |bn| {
      bn.iter(|| {
        let r = ops::scalar::dot(black_box(&a), black_box(&b));
        black_box(r);
      });
    });
  }
  group.finish();
}

fn bench_axpy(c: &mut Criterion) {
  let mut group = c.benchmark_group("axpy");
  for &d in DIMS {
    let x = rand_vec(d, 0xa1);
    let y_init = rand_vec(d, 0xb2);
    let alpha = 0.7_f64;
    group.bench_function(BenchmarkId::new(format!("d={d}"), "simd"), |bn| {
      bn.iter_batched(
        || y_init.clone(),
        |mut y| {
          ops::axpy(black_box(&mut y), black_box(alpha), black_box(&x));
          black_box(y);
        },
        criterion::BatchSize::SmallInput,
      );
    });
    group.bench_function(BenchmarkId::new(format!("d={d}"), "scalar"), |bn| {
      bn.iter_batched(
        || y_init.clone(),
        |mut y| {
          ops::scalar::axpy(black_box(&mut y), black_box(alpha), black_box(&x));
          black_box(y);
        },
        criterion::BatchSize::SmallInput,
      );
    });
  }
  group.finish();
}

fn bench_pdist(c: &mut Criterion) {
  let mut group = c.benchmark_group("pdist_euclidean");
  for &d in DIMS {
    for &n in PDIST_N {
      let rows = rand_vec(n * d, 0xc3 ^ d as u64 ^ ((n as u64) << 16));
      group.bench_function(BenchmarkId::new(format!("n={n},d={d}"), "simd"), |bn| {
        bn.iter(|| {
          let v = ops::pdist_euclidean(black_box(&rows), n, d);
          black_box(v);
        });
      });
      group.bench_function(BenchmarkId::new(format!("n={n},d={d}"), "scalar"), |bn| {
        bn.iter(|| {
          let v = ops::scalar::pdist_euclidean(black_box(&rows), n, d);
          black_box(v);
        });
      });
    }
  }
  group.finish();
}

criterion_group!(benches, bench_dot, bench_axpy, bench_pdist);
criterion_main!(benches);
