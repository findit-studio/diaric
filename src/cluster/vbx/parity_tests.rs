//! Parity tests for `diarization::cluster::vbx` against the captured artifacts.
//!
//! Loads `tests/parity/fixtures/01_dialogue/{plda_embeddings, vbx_state}.npz`
//! and asserts that `vbx_iterate(post_plda, phi, qinit, fa, fb, max_iters)`
//! reproduces pyannote's `q_final`, `sp_final`, and `elbo_trajectory`
//! within float-cast tolerance.
//!
//! **Hard-fails** when fixtures are absent (same convention as
//! `src/plda/parity_tests.rs`). The fixtures are committed to the
//! repo and ship via `cargo publish`; a missing one is a packaging
//! error, not an opt-out.

use std::{fs::File, io::BufReader, path::PathBuf};

use nalgebra::{DMatrix, DVector};
use npyz::npz::NpzArchive;

use crate::cluster::vbx::{StopReason, vbx_iterate};

fn repo_root() -> PathBuf {
  PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn fixture(rel: &str) -> PathBuf {
  repo_root().join(rel)
}

/// Hard-fail if the captured fixtures are absent. Mirrors
/// `src/plda/parity_tests.rs::require_fixtures`.
fn require_fixtures() {
  let required = [
    "tests/parity/fixtures/01_dialogue/plda_embeddings.npz",
    "tests/parity/fixtures/01_dialogue/vbx_state.npz",
  ];
  let missing: Vec<&str> = required
    .iter()
    .copied()
    .filter(|p| !repo_root().join(p).exists())
    .collect();
  assert!(
    missing.is_empty(),
    "VBx parity fixtures missing: {missing:?}. \
     These ship with the crate via `cargo publish`; a missing \
     fixture is a packaging error, not an opt-out. Re-run \
     `tests/parity/python/capture_intermediates.py` against the \
     reference clip to regenerate, or restore the files from a \
     full checkout."
  );
}

fn read_npz_array<T>(path: &PathBuf, key: &str) -> (Vec<T>, Vec<u64>)
where
  T: npyz::Deserialize,
{
  let f = File::open(path).expect("open npz");
  let mut z = NpzArchive::new(BufReader::new(f)).expect("read npz");
  let npy = z
    .by_name(key)
    .expect("query archive")
    .unwrap_or_else(|| panic!("array `{key}` not in {}", path.display()));
  let shape: Vec<u64> = npy.shape().to_vec();
  let data: Vec<T> = npy.into_vec().expect("decode array");
  (data, shape)
}

#[test]
#[ignore = "ad-hoc capture; localizes pyannote VBx parity on 10_mrbeast_clean_water"]
fn vbx_iterate_matches_pyannote_q_final_pi_elbo_10_mrbeast() {
  // Adapter: call run_vbx_parity on a different fixture. 01_dialogue
  // has T=195 (single chunk), 10_mrbeast_clean_water has T=611 — large
  // enough to expose VBx GEMM drift if it's the divergence source for
  // the testaudioset bench's segment-count differences.
  run_vbx_parity_for_fixture("10_mrbeast_clean_water");
}

fn run_vbx_parity_for_fixture(fixture_dir: &str) {
  let plda_path = fixture(&format!(
    "tests/parity/fixtures/{fixture_dir}/plda_embeddings.npz"
  ));
  let (post_plda_flat, post_plda_shape) = read_npz_array::<f64>(&plda_path, "post_plda");
  assert_eq!(post_plda_shape.len(), 2);
  let t = post_plda_shape[0] as usize;
  let d = post_plda_shape[1] as usize;
  assert_eq!(d, 128);
  let x = DMatrix::<f64>::from_row_slice(t, d, &post_plda_flat);

  let (phi_flat, _) = read_npz_array::<f64>(&plda_path, "phi");
  let phi = DVector::<f64>::from_vec(phi_flat);

  let vbx_path = fixture(&format!(
    "tests/parity/fixtures/{fixture_dir}/vbx_state.npz"
  ));
  let (qinit_flat, qinit_shape) = read_npz_array::<f64>(&vbx_path, "qinit");
  let s = qinit_shape[1] as usize;
  let qinit = DMatrix::<f64>::from_row_slice(t, s, &qinit_flat);

  let (fa_flat, _) = read_npz_array::<f64>(&vbx_path, "fa");
  let (fb_flat, _) = read_npz_array::<f64>(&vbx_path, "fb");
  let (max_iters_flat, _) = read_npz_array::<i64>(&vbx_path, "max_iters");
  let fa = fa_flat[0];
  let fb = fb_flat[0];
  let max_iters = max_iters_flat[0] as usize;

  let out = vbx_iterate(x.as_view(), &phi, &qinit, fa, fb, max_iters).expect("vbx_iterate");

  let (q_final_flat, _) = read_npz_array::<f64>(&vbx_path, "q_final");
  let q_final = DMatrix::<f64>::from_row_slice(t, s, &q_final_flat);
  let mut gamma_max_err = 0.0f64;
  for tt in 0..t {
    for sj in 0..s {
      let err = (out.gamma()[(tt, sj)] - q_final[(tt, sj)]).abs();
      if err > gamma_max_err {
        gamma_max_err = err;
      }
    }
  }
  let (sp_final_flat, _) = read_npz_array::<f64>(&vbx_path, "sp_final");
  let mut pi_max_err = 0.0f64;
  for (sj, want) in sp_final_flat.iter().enumerate() {
    let err = (out.pi()[sj] - want).abs();
    if err > pi_max_err {
      pi_max_err = err;
    }
  }
  let (elbo_flat, _) = read_npz_array::<f64>(&vbx_path, "elbo_trajectory");
  let elbo_max_err = out
    .elbo_trajectory()
    .iter()
    .zip(elbo_flat.iter())
    .map(|(g, w)| (g - w).abs())
    .fold(0.0_f64, f64::max);
  eprintln!(
    "[parity_vbx_{fixture_dir}] T={t} S={s} stop={:?} iters={} gamma_max_err={gamma_max_err:.3e} pi_max_err={pi_max_err:.3e} elbo_max_err={elbo_max_err:.3e}",
    out.stop_reason(),
    out.elbo_trajectory().len(),
  );
  // Use the same tolerances as the canonical parity test on 01_dialogue.
  assert!(gamma_max_err < 1.0e-12, "gamma_max_err={gamma_max_err}");
  assert!(pi_max_err < 1.0e-9, "pi_max_err={pi_max_err}");
  assert!(elbo_max_err < 1.0e-9, "elbo_max_err={elbo_max_err}");
}

#[test]
fn vbx_iterate_matches_pyannote_q_final_pi_elbo() {
  crate::parity_fixtures_or_skip!();
  require_fixtures();

  // ── Inputs (post_plda, phi from PLDA stage; qinit, fa, fb,
  //    max_iters from the captured VBx run) ────────────────────────
  let plda_path = fixture("tests/parity/fixtures/01_dialogue/plda_embeddings.npz");
  let (post_plda_flat, post_plda_shape) = read_npz_array::<f64>(&plda_path, "post_plda");
  assert_eq!(post_plda_shape.len(), 2);
  let t = post_plda_shape[0] as usize;
  let d = post_plda_shape[1] as usize;
  assert_eq!(d, 128);
  let x = DMatrix::<f64>::from_row_slice(t, d, &post_plda_flat);

  let (phi_flat, phi_shape) = read_npz_array::<f64>(&plda_path, "phi");
  assert_eq!(phi_shape, vec![128]);
  let phi = DVector::<f64>::from_vec(phi_flat);

  let vbx_path = fixture("tests/parity/fixtures/01_dialogue/vbx_state.npz");
  let (qinit_flat, qinit_shape) = read_npz_array::<f64>(&vbx_path, "qinit");
  assert_eq!(qinit_shape.len(), 2);
  assert_eq!(qinit_shape[0] as usize, t);
  let s = qinit_shape[1] as usize;
  let qinit = DMatrix::<f64>::from_row_slice(t, s, &qinit_flat);

  // Hyperparameters were captured alongside the VBx outputs (Task 0).
  // Reading from the fixture means a future model upgrade surfaces
  // as a parity failure rather than a silent drift.
  let (fa_flat, _) = read_npz_array::<f64>(&vbx_path, "fa");
  let (fb_flat, _) = read_npz_array::<f64>(&vbx_path, "fb");
  let (max_iters_flat, _) = read_npz_array::<i64>(&vbx_path, "max_iters");
  let fa = fa_flat[0];
  let fb = fb_flat[0];
  let max_iters = max_iters_flat[0] as usize;

  // ── Run ────────────────────────────────────────────────────────
  let out = vbx_iterate(x.as_view(), &phi, &qinit, fa, fb, max_iters).expect("vbx_iterate");

  // The captured run converged in 16 of 20 iterations — the
  // pyannote-equivalent should hit the convergence branch, not
  // exhaust max_iters.
  assert_eq!(
    out.stop_reason(),
    StopReason::Converged,
    "captured pyannote run converged within max_iters=20 in 16 iterations; \
     parity should also converge"
  );

  // ── Compare gamma (T x S) ──────────────────────────────────────
  let (q_final_flat, q_final_shape) = read_npz_array::<f64>(&vbx_path, "q_final");
  assert_eq!(q_final_shape, vec![t as u64, s as u64]);
  let q_final = DMatrix::<f64>::from_row_slice(t, s, &q_final_flat);
  let mut gamma_max_err = 0.0f64;
  let mut gamma_max_err_loc = (0usize, 0usize);
  let mut gamma_max_err_got = 0.0f64;
  let mut gamma_max_err_want = 0.0f64;
  for tt in 0..t {
    for sj in 0..s {
      let got = out.gamma()[(tt, sj)];
      let want = q_final[(tt, sj)];
      let err = (got - want).abs();
      if err > gamma_max_err {
        gamma_max_err = err;
        gamma_max_err_loc = (tt, sj);
        gamma_max_err_got = got;
        gamma_max_err_want = want;
      }
    }
  }
  eprintln!(
    "[parity_vbx] gamma max_abs_err = {gamma_max_err:.3e} at (t={}, s={}) got={:.6e} want={:.6e}",
    gamma_max_err_loc.0, gamma_max_err_loc.1, gamma_max_err_got, gamma_max_err_want,
  );
  assert!(
    gamma_max_err < 1.0e-12,
    "gamma parity failed: max_abs_err = {gamma_max_err:.3e} at (t={}, s={}) got={:.6e} want={:.6e}",
    gamma_max_err_loc.0,
    gamma_max_err_loc.1,
    gamma_max_err_got,
    gamma_max_err_want,
  );

  // ── Compare pi (S,) ────────────────────────────────────────────
  let (sp_final_flat, sp_final_shape) = read_npz_array::<f64>(&vbx_path, "sp_final");
  assert_eq!(sp_final_shape, vec![s as u64]);
  let mut pi_max_err = 0.0f64;
  let mut pi_max_err_loc = 0usize;
  let mut pi_max_err_got = 0.0f64;
  let mut pi_max_err_want = 0.0f64;
  for (sj, want) in sp_final_flat.iter().enumerate() {
    let got = out.pi()[sj];
    let err = (got - want).abs();
    if err > pi_max_err {
      pi_max_err = err;
      pi_max_err_loc = sj;
      pi_max_err_got = got;
      pi_max_err_want = *want;
    }
  }
  eprintln!(
    "[parity_vbx] pi max_abs_err = {pi_max_err:.3e} at s={pi_max_err_loc} got={pi_max_err_got:.6e} want={pi_max_err_want:.6e}",
  );
  assert!(
    pi_max_err < 1.0e-9,
    "pi parity failed: max_abs_err = {pi_max_err:.3e} at s={pi_max_err_loc} got={pi_max_err_got:.6e} want={pi_max_err_want:.6e}",
  );

  // ── Compare ELBO trajectory ────────────────────────────────────
  let (elbo_flat, elbo_shape) = read_npz_array::<f64>(&vbx_path, "elbo_trajectory");
  assert_eq!(elbo_shape.len(), 1);
  assert_eq!(
    out.elbo_trajectory().len(),
    elbo_flat.len(),
    "ELBO iteration count mismatch: rust={} pyannote={}",
    out.elbo_trajectory().len(),
    elbo_flat.len()
  );
  let mut elbo_max_err = 0.0f64;
  let mut elbo_max_err_iter = 0usize;
  let mut elbo_max_err_got = 0.0f64;
  let mut elbo_max_err_want = 0.0f64;
  for (ii, (got, want)) in out
    .elbo_trajectory()
    .iter()
    .zip(elbo_flat.iter())
    .enumerate()
  {
    let err = (got - want).abs();
    if err > elbo_max_err {
      elbo_max_err = err;
      elbo_max_err_iter = ii;
      elbo_max_err_got = *got;
      elbo_max_err_want = *want;
    }
  }
  eprintln!(
    "[parity_vbx] ELBO max_abs_err = {elbo_max_err:.3e} at iter {elbo_max_err_iter} got={elbo_max_err_got:.6e} want={elbo_max_err_want:.6e}",
  );
  assert!(
    elbo_max_err < 1.0e-9,
    "ELBO parity failed: max_abs_err = {elbo_max_err:.3e} at iter {elbo_max_err_iter} got={elbo_max_err_got:.6e} want={elbo_max_err_want:.6e}",
  );
}

/// CI guard for finding (MEDIUM): VBx reductions feed
/// the discrete `sp > SP_ALIVE_THRESHOLD` filter. AVX2/AVX-512
/// reductions diverge from scalar/NEON by O(1e-15) relative; if any
/// produced `pi[k]` lands inside that drift band of `SP_ALIVE_THRESHOLD
/// = 1e-7`, the alive-cluster set could differ across CPU families
/// → CPU-dependent speaker count → downstream Hungarian assignment
/// changes.
///
/// This test runs production `vbx_iterate` (SIMD via `ops::dot`) on
/// every captured fixture and asserts that for every produced `pi[k]`,
/// the value is at least `MIN_RATIO_TO_THRESHOLD`× larger or smaller
/// than `SP_ALIVE_THRESHOLD`. Empirically captured fixtures have alive
/// `pi` in O(0.1) and squashed `pi` in O(1e-14) — the closest value
/// to threshold is at least 1e6× away. With ulp drift bounded by
/// O(1e-15) relative (i.e. ~1e-22 absolute on the squashed values
/// and ~1e-16 absolute on alive), there is no realistic floating-point
/// path that flips the discrete decision. This test makes that
/// margin explicit and CI-checked: if a future model retraining or
/// algorithm change pushed any cluster's `pi` near the threshold,
/// the failure here would force us to re-evaluate whether SIMD is
/// safe for the VBx path.
#[test]
fn vbx_pi_has_safe_margin_from_sp_alive_threshold() {
  crate::parity_fixtures_or_skip!();
  use crate::cluster::centroid::SP_ALIVE_THRESHOLD;

  // pi must be at least this much away from the threshold (ratio).
  // 1e3 is generous: alive pi are in O(0.1), squashed in O(1e-14),
  // so realistic margins are O(1e6). 1e3 still catches any drift
  // worse than ~1e-10 absolute, which is far above any plausible
  // SIMD-induced ulp shift on these magnitudes.
  const MIN_RATIO_TO_THRESHOLD: f64 = 1.0e3;
  const ALIVE_FLOOR: f64 = SP_ALIVE_THRESHOLD * MIN_RATIO_TO_THRESHOLD; // 1e-4
  const SQUASHED_CEILING: f64 = SP_ALIVE_THRESHOLD / MIN_RATIO_TO_THRESHOLD; // 1e-10

  for fixture_dir in &[
    "01_dialogue",
    "02_pyannote_sample",
    "03_dual_speaker",
    "04_three_speaker",
    "05_four_speaker",
    "06_long_recording",
  ] {
    let plda_path = fixture(&format!(
      "tests/parity/fixtures/{fixture_dir}/plda_embeddings.npz"
    ));
    let vbx_path = fixture(&format!(
      "tests/parity/fixtures/{fixture_dir}/vbx_state.npz"
    ));
    if !plda_path.exists() || !vbx_path.exists() {
      panic!("fixture {fixture_dir} missing required npz files");
    }

    let (post_plda_flat, post_plda_shape) = read_npz_array::<f64>(&plda_path, "post_plda");
    let t = post_plda_shape[0] as usize;
    let d = post_plda_shape[1] as usize;
    let x = DMatrix::<f64>::from_row_slice(t, d, &post_plda_flat);
    let (phi_flat, _) = read_npz_array::<f64>(&plda_path, "phi");
    let phi = DVector::<f64>::from_vec(phi_flat);

    let (qinit_flat, qinit_shape) = read_npz_array::<f64>(&vbx_path, "qinit");
    let s = qinit_shape[1] as usize;
    let qinit = DMatrix::<f64>::from_row_slice(t, s, &qinit_flat);

    let (fa_flat, _) = read_npz_array::<f64>(&vbx_path, "fa");
    let (fb_flat, _) = read_npz_array::<f64>(&vbx_path, "fb");
    let (max_iters_flat, _) = read_npz_array::<i64>(&vbx_path, "max_iters");
    let fa = fa_flat[0];
    let fb = fb_flat[0];
    let max_iters = max_iters_flat[0] as usize;

    let out = vbx_iterate(x.as_view(), &phi, &qinit, fa, fb, max_iters).expect("vbx_iterate");

    for sj in 0..out.pi().len() {
      let p = out.pi()[sj];
      assert!(p.is_finite(), "{fixture_dir}: pi[{sj}] = {p} is non-finite");
      let alive = p > SP_ALIVE_THRESHOLD;
      if alive {
        assert!(
          p >= ALIVE_FLOOR,
          "{fixture_dir}: alive pi[{sj}] = {p:.3e} too close to SP_ALIVE_THRESHOLD ({SP_ALIVE_THRESHOLD:.0e}); \
           bound = {ALIVE_FLOOR:.0e}. SIMD vs scalar ulp drift could flip the alive decision."
        );
      } else {
        assert!(
          p <= SQUASHED_CEILING,
          "{fixture_dir}: squashed pi[{sj}] = {p:.3e} too close to SP_ALIVE_THRESHOLD ({SP_ALIVE_THRESHOLD:.0e}); \
           bound = {SQUASHED_CEILING:.0e}. SIMD vs scalar ulp drift could flip the squashed decision."
        );
      }
    }
    eprintln!(
      "[parity_vbx_margin] {fixture_dir}: {} pi values, alive ratio = {:.0e}× above threshold, squashed ratio = {:.0e}× below threshold",
      out.pi().len(),
      out
        .pi()
        .iter()
        .filter(|&&p| p > SP_ALIVE_THRESHOLD)
        .fold(f64::INFINITY, |a, &p| a.min(p))
        / SP_ALIVE_THRESHOLD,
      SP_ALIVE_THRESHOLD
        / out
          .pi()
          .iter()
          .filter(|&&p| p <= SP_ALIVE_THRESHOLD)
          .copied()
          .fold(f64::NEG_INFINITY, f64::max)
          .max(f64::MIN_POSITIVE),
    );
  }
}
