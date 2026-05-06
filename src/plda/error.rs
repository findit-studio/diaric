//! Error type for `diarization::plda`.

use thiserror::Error;

/// Errors produced by `PldaTransform` construction or transform calls.
///
/// PLDA weights are embedded into the dia binary at compile time via
/// `include_bytes!`, so I/O / file-not-found / shape-mismatch errors
/// are eliminated. The remaining failure modes are:
///
/// 1. A linear-algebra precondition fails at construction time —
///    [`Self::WNotPositiveDefinite`].
/// 2. The caller hands a degenerate / non-finite embedding to a
///    transform — [`Self::NonFiniteInput`] or [`Self::DegenerateInput`].
///
/// `xvec_transform`, `plda_transform`, and `project` return `Result`
/// so that a degraded upstream embedder (NaN/Inf from a misconfigured
/// ONNX runtime, near-zero output post-centering) surfaces as an
/// explicit error instead of silently producing NaN that propagates
/// into VBx / clustering.
#[derive(Debug, Error)]
pub enum Error {
  /// The within-class covariance matrix `W = inv(tr.T @ tr)` is not
  /// symmetric positive-definite. Either the embedded `tr.bin` is
  /// corrupted, or pyannote's PLDA weights have changed in a way
  /// that breaks the generalized-eigh preconditions.
  #[error("PLDA: W matrix not positive-definite (corrupted weights or upstream drift)")]
  WNotPositiveDefinite,

  /// Input embedding contained `NaN` or `±inf`, or an intermediate
  /// vector inside a transform stage produced a non-finite value
  /// (e.g. division-by-zero in L2 normalization fed by Inf input).
  /// Almost always indicates a degraded upstream embedder rather
  /// than an algorithmic bug in `diarization::plda`.
  #[error("PLDA: input or intermediate vector contains NaN or ±inf")]
  NonFiniteInput,

  /// Input vector is zero-norm or near-zero-norm (`< NORM_EPSILON`)
  /// after the centering step inside `xvec_transform`. The L2
  /// normalization that follows would divide by ~0 and amplify
  /// noise to dominate the signal. Real WeSpeaker outputs are never
  /// this close to the centering mean; if this fires the embedder
  /// is producing degenerate output.
  #[error("PLDA: centered input has near-zero norm; cannot L2-normalize")]
  DegenerateInput,

  /// Vector handed to the captured-fixture `PostXvecEmbedding`
  /// constructor (test-only) has a norm too far from
  /// `sqrt(PLDA_DIMENSION) ≈ 11.31` — i.e.
  /// it is not in the post-`xvec_tf` distribution that `plda_tf`
  /// requires.
  ///
  /// Common misuses this catches:
  /// - L2-normalized 128-d vector (norm = 1.0).
  /// - Stale or wrong-revision pyannote capture.
  /// - Random / hand-constructed input.
  ///
  /// Returning whitened features for any of these would silently
  /// drift VBx clustering off the captured pyannote distribution.
  ///
  #[error(
    "PLDA: post-xvec norm {actual:.6} too far from expected sqrt(D_out) {expected:.6} \
     (tolerance {tolerance:.0e}); not a post-xvec_tf vector"
  )]
  WrongPostXvecNorm {
    /// Actual L2 norm of the offending input.
    actual: f64,
    /// Expected L2 norm — `sqrt(PLDA_DIMENSION)`.
    expected: f64,
    /// Absolute tolerance applied for the check.
    tolerance: f64,
  },
}
