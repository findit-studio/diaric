//! Error variants for `diarization::cluster::vbx`.

use thiserror::Error;

/// Errors produced by `vbx_iterate`.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum Error {
  /// Input shapes do not satisfy the contract.
  #[error("shape mismatch: {0}")]
  Shape(#[from] ShapeError),

  /// A non-finite value (NaN / ±inf) appeared in an intermediate
  /// (rho, alpha, log_p_, ELBO, …). The algorithm has no recovery
  /// path; the caller should treat this as a hard failure.
  #[error("non-finite intermediate: {0}")]
  NonFinite(#[from] NonFiniteField),

  /// `Phi` (the eigenvalue diagonal from `PldaTransform::phi()`) had
  /// an entry that wasn't strictly positive *and* finite. The
  /// algorithm requires `0 < Phi[d] < ∞` for `sqrt(Phi)` and
  /// `1 + … * Phi` to be well-defined; `+inf` would poison
  /// downstream intermediates without surfacing a clear cause at
  /// the boundary.
  #[error("Phi must be strictly positive and finite; saw {0:.3e} at index {1}")]
  NonPositivePhi(f64, usize),

  /// The ELBO decreased by more than the float-roundoff tolerance
  /// between two consecutive iterations. VB EM's monotonicity is a
  /// fundamental invariant — a regression beyond float noise
  /// indicates a bug, numerical instability, or an out-of-distribution
  /// input that should not be silently accepted. The returned `gamma`
  /// and `pi` from the failing iteration are NOT propagated; if the
  /// caller wants the last-known-good state, re-invoke with
  /// `max_iters` set to `iter` (the regression-triggering iteration
  /// index). Pyannote prints a `WARNING:` to stdout and keeps the
  /// regressed state; this is a deliberate fail-fast divergence.
  #[error("ELBO regressed by {delta:.3e} at iteration {iter} (beyond float-roundoff tolerance)")]
  ElboRegression {
    /// Iteration index at which the regression was detected.
    iter: usize,
    /// `ELBO[iter] - ELBO[iter - 1]` (negative beyond the
    /// float-roundoff band).
    delta: f64,
  },
}

/// Specific shape-violation reasons for [`Error::Shape`].
#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum ShapeError {
  #[error("X must have at least one feature column")]
  ZeroXFeatureDim,
  #[error("Phi.len() must equal X.ncols()")]
  PhiXFeatureMismatch,
  #[error("qinit.nrows() must equal X.nrows()")]
  QinitXRowMismatch,
  #[error("qinit must have at least one cluster column")]
  QinitNoClusters,
  #[error("Fa must be a positive finite scalar")]
  InvalidFa,
  #[error("Fb must be a positive finite scalar")]
  InvalidFb,
  #[error("qinit entries must be nonnegative")]
  NegativeQinit,
  #[error("qinit rows must sum to 1")]
  QinitRowSumMismatch,
  #[error("max_iters must be at least 1")]
  ZeroMaxIters,
  #[error(
    "max_iters exceeds MAX_ITERS_CAP (1_000); pyannote's default is 20 \
     and realistic configurations converge well below the cap"
  )]
  MaxItersAboveCap,
}

/// Field that contained a non-finite value.
#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum NonFiniteField {
  #[error("x")]
  X,
  #[error("qinit")]
  Qinit,
  #[error("pi sum")]
  PiSum,
  #[error("ELBO")]
  Elbo,
}
