//! Scalar `logsumexp` over a single row.

/// Numerically-stable `ln(Σ exp(row[i]))`, computed via the standard
/// max-shift trick:
///
/// ```text
/// out = ln(Σ exp(row[i] - max)) + max
/// ```
///
/// Matches the `pyannote.audio.utils.vbx.logsumexp_axis(_, axis=-1)`
/// reduction used inside VBx's responsibility update.
///
/// All-`-inf` rows return `-inf` (the shift trick is bypassed because
/// subtracting `-inf` from `-inf` yields `NaN`). NaN rows propagate
/// to `-inf` here vs. `NaN` in scipy — VBx callers reject NaN
/// upstream via `Error::NonFinite`, so this divergence is unreachable
/// in production.
#[inline]
pub fn logsumexp_row(row: &[f64]) -> f64 {
  // Find max for stability shift.
  let mut max = f64::NEG_INFINITY;
  for &v in row {
    if v > max {
      max = v;
    }
  }
  if max == f64::NEG_INFINITY {
    return f64::NEG_INFINITY;
  }
  let mut sum_exp = 0.0;
  for &v in row {
    sum_exp += (v - max).exp();
  }
  sum_exp.ln() + max
}
