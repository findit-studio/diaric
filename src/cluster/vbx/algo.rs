//! VBx variational EM iterations.

use crate::cluster::vbx::error::Error;
use nalgebra::{DMatrix, DVector};

/// Hard upper bound on `max_iters`. Pyannote's community-1 default is
/// 20 and captured fixtures converge in 16-20 iterations; production
/// runs that hit even 50 would already indicate a misconfiguration.
/// `1_000` is ~50× the default — generous headroom for experimentation
/// while preventing a malformed config from turning one diarization
/// call into hours of unbounded matmul work.
pub const MAX_ITERS_CAP: usize = 1_000;

/// Why the EM loop stopped. Lets callers distinguish a converged
/// posterior from one that ran out of iterations — both have
/// `elbo_trajectory.len() == max_iters` when convergence happens
/// on the very last allowed iteration vs. when the cap was hit
/// without convergence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
  /// EM converged: an iteration's ELBO step delta landed within the
  /// scale-aware regression band and the loop exited early.
  Converged,
  /// The loop ran all `max_iters` iterations without ever firing
  /// the convergence check. The output is the best estimate seen,
  /// but downstream consumers should decide whether to accept it,
  /// retry with a higher cap, or reject.
  MaxIterationsReached,
}

/// Output of [`vbx_iterate`].
#[derive(Debug, Clone)]
pub struct VbxOutput {
  gamma: nalgebra::DMatrix<f64>,
  pi: nalgebra::DVector<f64>,
  elbo_trajectory: Vec<f64>,
  stop_reason: StopReason,
}

impl VbxOutput {
  /// Construct.
  pub fn new(
    gamma: nalgebra::DMatrix<f64>,
    pi: nalgebra::DVector<f64>,
    elbo_trajectory: Vec<f64>,
    stop_reason: StopReason,
  ) -> Self {
    Self {
      gamma,
      pi,
      elbo_trajectory,
      stop_reason,
    }
  }

  /// Final responsibilities, shape `(T, S)`.
  pub const fn gamma(&self) -> &nalgebra::DMatrix<f64> {
    &self.gamma
  }

  /// Final speaker priors, shape `(S,)`. Sums to 1.0.
  pub const fn pi(&self) -> &nalgebra::DVector<f64> {
    &self.pi
  }

  /// ELBO at each iteration (length ≤ `max_iters`).
  pub fn elbo_trajectory(&self) -> &[f64] {
    &self.elbo_trajectory
  }

  /// Why the loop stopped — converged vs. hit `max_iters`.
  pub const fn stop_reason(&self) -> StopReason {
    self.stop_reason
  }

  /// Decompose into the four owned fields.
  pub fn into_parts(
    self,
  ) -> (
    nalgebra::DMatrix<f64>,
    nalgebra::DVector<f64>,
    Vec<f64>,
    StopReason,
  ) {
    (self.gamma, self.pi, self.elbo_trajectory, self.stop_reason)
  }
}

/// Absolute floor for the ELBO regression tolerance. Caps the band
/// for tiny ELBOs where the relative term is negligible.
const ELBO_REGRESSION_ATOL: f64 = 1.0e-9;

/// Relative scaling for the ELBO regression tolerance. ELBO is an
/// accumulated sum over `T * S * D` matrix entries plus `T` per-frame
/// terms; float roundoff therefore scales with the working magnitude
/// of the ELBO itself. reproduced a final delta of
/// `~-2.47e-8` for finite community-Fa/Fb inputs at |ELBO| ≈ 2700,
/// well outside an absolute `1e-9` band but ~9× *inside* the
/// scale-aware band `1e-9 + 1e-9 * 2700 ≈ 2.7e-6`. The previous
/// fixture-only calibration would have rejected that as an algorithm
/// failure.
const ELBO_REGRESSION_RTOL: f64 = 1.0e-9;

/// Compute the regression tolerance for a given ELBO magnitude.
/// `band(prev, elbo) = atol + rtol * max(|prev|, |elbo|)`.
fn regression_tolerance(prev_elbo: f64, elbo: f64) -> f64 {
  ELBO_REGRESSION_ATOL + ELBO_REGRESSION_RTOL * prev_elbo.abs().max(elbo.abs())
}

/// Outcome of comparing one EM iteration's ELBO against the previous.
#[derive(Debug, PartialEq)]
pub(super) enum ElboStep {
  /// Improvement >= `epsilon` — keep iterating.
  Continue,
  /// Improvement < `epsilon` (including small negative deltas within
  /// the scale-aware regression-tolerance band) — converged, exit
  /// cleanly.
  Converged,
  /// Negative delta beyond the scale-aware regression-tolerance band
  /// — VB EM's monotonicity invariant is violated. Carries the
  /// offending delta.
  Regressed(f64),
}

/// Classify an ELBO step into the three convergence regimes.
///
/// The regression boundary is scale-aware: any delta within
/// `±(atol + rtol * max(|prev|, |elbo|))` is treated as float
/// roundoff and routed to `Converged`. Beyond that band on the
/// negative side: `Regressed`. This matters because ELBO accumulates
/// over `T * S * D` matrix entries plus `T` per-frame terms; float
/// roundoff therefore scales with magnitude, and an absolute
/// tolerance calibrated against a single fixture would error out on
/// numerically awkward but otherwise valid inputs.
///
/// Pyannote's `vbx.py:133-136` uses `if ELBO - prev < epsilon: break`
/// for both small-positive convergence AND any negative regression,
/// printing a warning for the regression case. The Rust port treats
/// a regression *beyond the float-roundoff band* as an error (no
/// print mechanism, and downstream clustering should not silently
/// consume a materially regressed posterior).
pub(super) fn classify_elbo_step(delta: f64, prev_elbo: f64, elbo: f64, epsilon: f64) -> ElboStep {
  let regression_tol = regression_tolerance(prev_elbo, elbo);
  if delta < -regression_tol {
    ElboStep::Regressed(delta)
  } else if delta < epsilon {
    ElboStep::Converged
  } else {
    ElboStep::Continue
  }
}

/// Row-wise `logsumexp` (numerically stable). For each row `r`:
///
/// ```text
/// out[r] = log(sum_j exp(m[r, j] - max_j m[r, j])) + max_j m[r, j]
/// ```
///
/// Matches `scipy.special.logsumexp(m, axis=-1)` modulo float roundoff
/// for finite or `-inf` rows. An all-NaN row returns `-inf` here vs
/// `NaN` in scipy — VBx callers reject NaN inputs upstream via
/// `Error::NonFinite`, so this divergence is unreachable in production.
/// An all-`-inf` row produces `-inf` (the shift trick is bypassed
/// because subtracting `-inf` from `-inf` yields `NaN`).
pub(super) fn logsumexp_rows(m: &DMatrix<f64>) -> DVector<f64> {
  let (rows, cols) = m.shape();
  let mut out = DVector::<f64>::zeros(rows);
  // Per-row stack buffer for the contiguous slice ops::logsumexp_row
  // expects. nalgebra is column-major so `m.row(r)` is strided; we
  // copy into `row_buf` once per row and dispatch.
  let mut row_buf: Vec<f64> = Vec::with_capacity(cols);
  for r in 0..rows {
    row_buf.clear();
    for c in 0..cols {
      row_buf.push(m[(r, c)]);
    }
    out[r] = crate::ops::logsumexp_row(&row_buf);
  }
  out
}

/// Variational Bayes HMM speaker clustering (the VBx EM core).
///
/// Mirrors `pyannote.audio.utils.vbx.VBx` (`utils/vbx.py:27-137` in
/// pyannote.audio 4.0.4). Inputs:
///
/// - `x`: `(T, D)` post-PLDA features (output of
///   `diarization::plda::PldaTransform::project()` stacked into a matrix).
/// - `phi`: `(D,)` eigenvalue diagonal (output of
///   `diarization::plda::PldaTransform::phi()`). Must be strictly positive.
/// - `qinit`: `(T, S)` initial responsibility matrix. Each row should
///   sum to 1 (the algorithm doesn't enforce this — pyannote's caller
///   pre-softmaxes a smoothed one-hot AHC initialization).
/// - `fa`: sufficient-statistics scale (community-1 uses 0.07).
/// - `fb`: speaker regularization (community-1 uses 0.8).
/// - `max_iters`: hard iteration cap. Inner convergence triggers early
///   exit when `ELBO_i - ELBO_{i-1} < 1e-4`.
///
/// Returns final `gamma`, `pi`, and the ELBO trajectory (one entry per
/// iteration actually run; length ≤ `max_iters`).
///
/// # Errors
///
/// - [`Error::Shape`] on mismatched dimensions, an `Fa`/`Fb` value
///   that's non-positive or non-finite, a `qinit` row that doesn't
///   sum to 1, a `qinit` entry that's negative, or `max_iters == 0`.
/// - [`Error::NonFinite`] if `x` or `qinit` contains a NaN/`±inf`
///   entry, or if a non-finite value appears in an algorithm
///   intermediate (the algorithm has no recovery; treat as a hard
///   failure).
/// - [`Error::NonPositivePhi`] if any `phi[d]` is not strictly
///   positive *and* finite (zero, negative, NaN, or `±inf`).
///
/// `qinit` row-sum tolerance is `1e-9` — pyannote's caller produces
/// a softmaxed initializer that is unit-normalized to within float
/// roundoff, and the captured rows are within `~1e-15` of 1.0.
pub fn vbx_iterate(
  x: nalgebra::DMatrixView<'_, f64>,
  phi: &DVector<f64>,
  qinit: &DMatrix<f64>,
  fa: f64,
  fb: f64,
  max_iters: usize,
) -> Result<VbxOutput, Error> {
  let (t, d) = x.shape();
  if d == 0 {
    // Zero feature columns silently runs the EM loop with no PLDA
    // evidence — gamma/pi end up driven only by `log_pi` priors and
    // the `(1 - fa/fb)` regularization term. The result is finite
    // and looks plausible, so a downstream caller treats it as a
    // valid clustering instead of a typed shape error. The pipeline
    // entrypoint has its own zero-PLDA-dim guard, but `vbx_iterate`
    // is public — direct callers must fail at this boundary too.
    //
    return Err(crate::cluster::vbx::error::ShapeError::ZeroXFeatureDim.into());
  }
  use crate::cluster::vbx::error::{NonFiniteField, ShapeError};
  if phi.len() != d {
    return Err(ShapeError::PhiXFeatureMismatch.into());
  }
  if qinit.nrows() != t {
    return Err(ShapeError::QinitXRowMismatch.into());
  }
  let s = qinit.ncols();
  if s == 0 {
    return Err(ShapeError::QinitNoClusters.into());
  }
  if !fa.is_finite() || fa <= 0.0 {
    return Err(ShapeError::InvalidFa.into());
  }
  if !fb.is_finite() || fb <= 0.0 {
    return Err(ShapeError::InvalidFb.into());
  }
  // Phi must be strictly positive AND finite. The previous check
  // accepted `+inf` because `inf > 0.0` is true and `inf.is_nan()`
  // is false; an infinite eigenvalue from a corrupted PLDA upstream
  // would have flowed into `sqrt(Phi)` and `1 + Fa/Fb * gamma_sum *
  // Phi`, producing NaN/Inf intermediates downstream.
  for (i, p) in phi.iter().enumerate() {
    if !p.is_finite() || *p <= 0.0 {
      return Err(Error::NonPositivePhi(*p, i));
    }
  }
  // X must be entirely finite. Without this, NaN/Inf in the
  // post-PLDA features would either:
  //   - silently return Ok at `max_iters = 0` with the unvalidated
  //     qinit as "gamma", or
  //   - poison G/rho in the pre-loop and surface as a generic
  //     `NonFinite("ELBO")` later instead of a clear input error.
  // The boundary contract is "non-finite intermediates are hard
  // failures"; admitting non-finite inputs violates that.
  if x.iter().any(|v| !v.is_finite()) {
    return Err(NonFiniteField::X.into());
  }
  // qinit value validation: each row must be a discrete probability
  // distribution over speakers (finite, nonnegative, row-sum ≈ 1).
  // Without this, a malformed initializer (negative entries, rows
  // not summing to 1, NaN) produces finite-looking posteriors after
  // the first update and biases the speaker model silently. Also
  // matters at `max_iters == 0`, which returns `qinit` directly as
  // the output `gamma`.
  const QINIT_ROW_SUM_TOLERANCE: f64 = 1.0e-9;
  for tt in 0..t {
    let mut row_sum = 0.0;
    for sj in 0..s {
      let v = qinit[(tt, sj)];
      if !v.is_finite() {
        return Err(NonFiniteField::Qinit.into());
      }
      if v < 0.0 {
        return Err(ShapeError::NegativeQinit.into());
      }
      row_sum += v;
    }
    if (row_sum - 1.0).abs() > QINIT_ROW_SUM_TOLERANCE {
      return Err(ShapeError::QinitRowSumMismatch.into());
    }
  }
  if max_iters == 0 {
    return Err(ShapeError::ZeroMaxIters.into());
  }
  if max_iters > MAX_ITERS_CAP {
    return Err(ShapeError::MaxItersAboveCap.into());
  }

  // Pre-compute G[t] = -0.5 * (sum(X[t]^2) + D * log(2*pi)) and rho via
  // a single row-major pack of X. nalgebra is column-major so `x.row(r)`
  // is strided — we copy into `x_row_major` once and reuse the slice
  // for the L2-norm-squared dot reduction.
  //
  // SIMD dot: scalar/NEON bit-identical contract (see
  // `ops::scalar::dot` module docs), so VBx EM trajectory, ELBO
  // convergence, and downstream `pi[s] > SP_ALIVE_THRESHOLD = 1e-7`
  // alive-cluster decisions are deterministic across backends.
  let log_2pi = (2.0_f64 * std::f64::consts::PI).ln();
  let mut x_row_major: Vec<f64> = Vec::with_capacity(t * d);
  for r in 0..t {
    for c in 0..d {
      x_row_major.push(x[(r, c)]);
    }
  }
  let mut g = DVector::<f64>::zeros(t);
  for r in 0..t {
    let row = &x_row_major[r * d..(r + 1) * d];
    let row_sq = crate::ops::dot(row, row);
    g[r] = -0.5 * (row_sq + d as f64 * log_2pi);
  }
  // V = sqrt(Phi); rho[t,d] = X[t,d] * V[d]. Column-major DMatrix
  // because the downstream `gamma.T @ rho` matmul (matrixmultiply
  // crate via nalgebra) exploits the column-major layout for its
  // cache-blocked GEMM. Hand-rolled dot-based and axpy-outer-product
  // matmul replacements in `ops::*` regressed the dominant
  // 01_dialogue fixture at the pipeline level: at our (T~200, S~10,
  // D=128) shape, matrixmultiply's blocked microkernel beats both
  // approaches. A proper hand-rolled cache-blocked GEMM is out of
  // scope here.
  let v_sqrt: DVector<f64> = phi.map(|p| p.sqrt());
  let mut rho = DMatrix::<f64>::zeros(t, d);
  for r in 0..t {
    for c in 0..d {
      rho[(r, c)] = x_row_major[r * d + c] * v_sqrt[c];
    }
  }

  let mut gamma = qinit.clone();
  // pi = ones(S) / S — matches pyannote's `VBx(..., pi=int(S), ...)`.
  let mut pi = DVector::<f64>::from_element(s, 1.0 / s as f64);

  let mut elbo_trajectory: Vec<f64> = Vec::new();
  let epsilon = 1e-4_f64;
  let eps_log = 1e-8_f64;
  let fa_over_fb = fa / fb;
  let mut converged = false;

  for ii in 0..max_iters {
    // ── E-step (speaker-model update) ────────────────────────────
    // gamma_sum, invL, alpha
    // gamma_sum[s] = column-sum of gamma over T rows (Eq. 17 input).
    let gamma_sum = DVector::<f64>::from_vec((0..s).map(|j| gamma.column(j).sum()).collect());

    // invL[s,d] = 1 / (1 + Fa/Fb * gamma_sum[s] * Phi[d])  (Eq. 17)
    let mut inv_l = DMatrix::<f64>::zeros(s, d);
    for sj in 0..s {
      for dk in 0..d {
        let denom = 1.0 + fa_over_fb * gamma_sum[sj] * phi[dk];
        inv_l[(sj, dk)] = 1.0 / denom;
      }
    }

    // alpha[s,d] = Fa/Fb * invL[s,d] * (gamma.T @ rho)[s,d]  (Eq. 16)
    let prod = gamma.transpose() * &rho; // (S, D)
    let mut alpha = DMatrix::<f64>::zeros(s, d);
    for sj in 0..s {
      for dk in 0..d {
        alpha[(sj, dk)] = fa_over_fb * inv_l[(sj, dk)] * prod[(sj, dk)];
      }
    }

    // ── log_p_ (per-(frame, speaker) log-likelihood, Eq. 23) ─────
    // log_p_[t,s] = Fa * (rho @ alpha.T - 0.5*(invL+alpha**2)@Phi + G) (Eq. 23)
    let rho_alpha_t = &rho * alpha.transpose(); // (T, S)
    // (invL + alpha**2) @ Phi : (S, D) · (D,) → (S,).
    //
    // Pack `(invL[s,:] + α[s,:]²)` into a contiguous scratch buffer
    // and reduce against `phi.as_slice()`. Buffer is reused across `s`
    // (one alloc per EM iter). SIMD dot — same scalar/NEON
    // bit-identical contract as the G norm-squared above.
    let mut sa_phi = DVector::<f64>::zeros(s);
    let mut sa_buf: Vec<f64> = vec![0.0; d];
    let phi_slice = phi.as_slice();
    for sj in 0..s {
      for dk in 0..d {
        let inv = inv_l[(sj, dk)];
        let a = alpha[(sj, dk)];
        sa_buf[dk] = inv + a * a;
      }
      sa_phi[sj] = crate::ops::dot(&sa_buf, phi_slice);
    }
    let mut log_p = DMatrix::<f64>::zeros(t, s);
    for tt in 0..t {
      for sj in 0..s {
        log_p[(tt, sj)] = fa * (rho_alpha_t[(tt, sj)] - 0.5 * sa_phi[sj] + g[tt]);
      }
    }

    // ── Responsibility update ────────────────────────────────────
    // log_pi, log_p_x via logsumexp, new gamma, new pi
    // log_pi[s] = log(pi[s] + eps_log)
    let log_pi: DVector<f64> = pi.map(|p| (p + eps_log).ln());
    // Fold log_pi into log_p in place — log_p is not referenced
    // outside this block, so we save the (T, S) clone.
    for tt in 0..t {
      for sj in 0..s {
        log_p[(tt, sj)] += log_pi[sj];
      }
    }
    // log_p_x[t] = logsumexp_t(log_p[t,:] + log_pi[:])
    let log_p_x = logsumexp_rows(&log_p);
    // gamma[t,s] = exp(log_p_[t,s] + log_pi[s] - log_p_x[t])
    //
    // The vectorized NEON exp polynomial in `crate::ops::arch::neon::exp`
    // is correct (parity tests pass at gamma 1e-12) but benchmarked
    // ~3–5% slower at the pipeline level on Apple Silicon: the extra
    // memory traffic of writing to and re-reading each column scratch
    // outweighs the polynomial's narrow SIMD gain over libm's
    // hand-tuned scalar `exp`. The primitive ships for future use on
    // x86_64 platforms (where AVX2/AVX-512 8-lane exp would have a
    // larger margin over scalar) and for architectures whose libm
    // exp is slower.
    let mut new_gamma = DMatrix::<f64>::zeros(t, s);
    for tt in 0..t {
      for sj in 0..s {
        // log_p now contains log_p + log_pi.
        new_gamma[(tt, sj)] = (log_p[(tt, sj)] - log_p_x[tt]).exp();
      }
    }
    gamma = new_gamma;
    // pi = gamma.sum(0); pi /= pi.sum()
    let mut new_pi = DVector::<f64>::zeros(s);
    for sj in 0..s {
      new_pi[sj] = gamma.column(sj).sum();
    }
    let pi_sum = new_pi.sum();
    if !pi_sum.is_finite() || pi_sum <= 0.0 {
      return Err(crate::cluster::vbx::error::NonFiniteField::PiSum.into());
    }
    pi = new_pi / pi_sum;

    // ── ELBO (Eq. 25) ────────────────────────────────────────────
    // ELBO = sum(log_p_x) + Fb * 0.5 * sum_{s,d}(log(invL) - invL - alpha**2 + 1)  (Eq. 25)
    let log_p_x_total: f64 = log_p_x.iter().sum();
    let mut bracket = 0.0;
    for sj in 0..s {
      for dk in 0..d {
        let inv = inv_l[(sj, dk)];
        let a2 = alpha[(sj, dk)] * alpha[(sj, dk)];
        bracket += inv.ln() - inv - a2 + 1.0;
      }
    }
    let elbo = log_p_x_total + fb * 0.5 * bracket;
    if !elbo.is_finite() {
      return Err(crate::cluster::vbx::error::NonFiniteField::Elbo.into());
    }
    elbo_trajectory.push(elbo);

    // ── Convergence check ────────────────────────────────────────
    if ii > 0 {
      let prev = elbo_trajectory[elbo_trajectory.len() - 2];
      let delta = elbo - prev;
      match classify_elbo_step(delta, prev, elbo, epsilon) {
        ElboStep::Continue => {}
        ElboStep::Converged => {
          converged = true;
          break;
        }
        ElboStep::Regressed(d) => {
          return Err(Error::ElboRegression { iter: ii, delta: d });
        }
      }
    }
  }
  let stop_reason = if converged {
    StopReason::Converged
  } else {
    StopReason::MaxIterationsReached
  };

  Ok(VbxOutput {
    gamma,
    pi,
    elbo_trajectory,
    stop_reason,
  })
}
