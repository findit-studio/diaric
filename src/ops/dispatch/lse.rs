//! `logsumexp_row` dispatcher.

use crate::ops::scalar;

/// `ln(Σ exp(row[i]))` via the max-shift trick. Scalar-only today.
#[inline]
pub fn logsumexp_row(row: &[f64]) -> f64 {
  scalar::logsumexp_row(row)
}
