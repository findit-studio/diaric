//! Compile-time-embedded PLDA weights.
//!
//! The six weight arrays (`mean1`, `mean2`, `lda`, `mu`, `tr`, `psi`)
//! ship as raw little-endian f64 binary blobs under `models/plda/`,
//! produced by `scripts/extract-plda-blobs.sh` from the upstream
//! `pyannote/speaker-diarization-community-1` `.npz` files. Embedding
//! them via `include_bytes!` means the dia binary is self-contained:
//! no runtime file I/O, no `npz` dependency, no
//! "did you put the weights in the right folder?" support burden.

use nalgebra::{DMatrix, DVector};

use crate::plda::{EMBEDDING_DIMENSION, PLDA_DIMENSION};

// ── Compile-time weight blobs ────────────────────────────────────────

const MEAN1_BYTES: &[u8] = include_bytes!("../../models/plda/mean1.bin");
const MEAN2_BYTES: &[u8] = include_bytes!("../../models/plda/mean2.bin");
const LDA_BYTES: &[u8] = include_bytes!("../../models/plda/lda.bin");
const MU_BYTES: &[u8] = include_bytes!("../../models/plda/mu.bin");
const TR_BYTES: &[u8] = include_bytes!("../../models/plda/tr.bin");
const PSI_BYTES: &[u8] = include_bytes!("../../models/plda/psi.bin");

/// PLDA eigenvectors_desc, derived offline via scipy's `eigh` and
/// shipped pre-computed. Sourced by
/// `scripts/extract-plda-eigenvectors.py`. We pin the eigenvectors
/// because LAPACK's eigenvector sign convention is implementation-
/// defined and varies across BLAS backends — nalgebra's
/// `SymmetricEigen` and scipy's `eigh` produced sign-flipped columns
/// on 67 of 128 dims for the community-1 weights, which propagated
/// through VBx as a 38% DER divergence on fixture 04 (heavy three-
/// speaker overlap). With pyannote's exact eigenvectors loaded here,
/// `post_plda` matches captured pyannote within ~1e-12 absolute,
/// across every (chunk, slot) row of every captured fixture.
const EIGENVECTORS_DESC_BYTES: &[u8] = include_bytes!("../../models/plda/eigenvectors_desc.bin");
const PHI_DESC_BYTES: &[u8] = include_bytes!("../../models/plda/phi_desc.bin");

// Compile-time size assertions. Catches blob/dimension drift the
// instant `cargo build` runs — far less surprising than a panic at
// `PldaTransform::new()` time.
const _: () = assert!(MEAN1_BYTES.len() == EMBEDDING_DIMENSION * 8);
const _: () = assert!(MEAN2_BYTES.len() == PLDA_DIMENSION * 8);
const _: () = assert!(LDA_BYTES.len() == EMBEDDING_DIMENSION * PLDA_DIMENSION * 8);
const _: () = assert!(MU_BYTES.len() == PLDA_DIMENSION * 8);
const _: () = assert!(TR_BYTES.len() == PLDA_DIMENSION * PLDA_DIMENSION * 8);
const _: () = assert!(PSI_BYTES.len() == PLDA_DIMENSION * 8);
const _: () = assert!(EIGENVECTORS_DESC_BYTES.len() == PLDA_DIMENSION * PLDA_DIMENSION * 8);
const _: () = assert!(PHI_DESC_BYTES.len() == PLDA_DIMENSION * 8);

// ── Public types ────────────────────────────────────────────────────

/// `xvec_tf`-stage weights extracted from `xvec_transform.npz`.
pub(super) struct XvecWeights {
  pub mean1: DVector<f64>, // (256,)
  pub mean2: DVector<f64>, // (128,)
  pub lda: DMatrix<f64>,   // (256, 128) row-major in the source numpy
}

/// `plda_tf`-stage weights consumed by `PldaTransform::new`.
///
/// The raw `tr` and `psi` source arrays from `plda.npz` are not stored
/// here — `PldaTransform` only needs the pre-computed
/// `eigenvectors_desc` / `phi_desc` derived from them, so the raw
/// matrices are loaded only by the loader's shape-validation tests.
pub(super) struct PldaWeights {
  pub mu: DVector<f64>, // (128,)
  /// Pre-computed eigenvectors of the generalized eigenvalue problem
  /// `B v = λ W v` (where `B = inv(tr.T / psi @ tr)` and `W = inv(tr.T
  /// @ tr)`), sorted descending by eigenvalue. Columns are unit-norm
  /// in `W`-metric. Captured offline from scipy's `eigh` to lock the
  /// eigenvector sign convention against pyannote's runtime stack.
  pub eigenvectors_desc: DMatrix<f64>, // (128, 128)
  /// Eigenvalues `λ_desc` matching `eigenvectors_desc`. Pyannote's
  /// `phi`. Pre-computed for parity (the eigenvalues themselves
  /// are sign-invariant, but we ship them anyway for byte-equal
  /// reproducibility against the captured fixture).
  pub phi_desc: DVector<f64>, // (128,)
}

// ── Loaders ─────────────────────────────────────────────────────────

pub(super) fn load_xvec() -> XvecWeights {
  XvecWeights {
    mean1: bytes_to_vector(MEAN1_BYTES, EMBEDDING_DIMENSION),
    mean2: bytes_to_vector(MEAN2_BYTES, PLDA_DIMENSION),
    lda: bytes_to_row_major_matrix(LDA_BYTES, EMBEDDING_DIMENSION, PLDA_DIMENSION),
  }
}

pub(super) fn load_plda() -> PldaWeights {
  PldaWeights {
    mu: bytes_to_vector(MU_BYTES, PLDA_DIMENSION),
    eigenvectors_desc: bytes_to_row_major_matrix(
      EIGENVECTORS_DESC_BYTES,
      PLDA_DIMENSION,
      PLDA_DIMENSION,
    ),
    phi_desc: bytes_to_vector(PHI_DESC_BYTES, PLDA_DIMENSION),
  }
}

// ── Byte-array → nalgebra helpers ───────────────────────────────────

/// Decode `len` little-endian f64 values from a byte slice into a
/// nalgebra `DVector`. Length is asserted at runtime; for the embedded
/// blobs the compile-time const-asserts above already guarantee the
/// right length, so this is defense-in-depth.
fn bytes_to_vector(bytes: &[u8], len: usize) -> DVector<f64> {
  debug_assert_eq!(bytes.len(), len * 8);
  let mut v = DVector::<f64>::zeros(len);
  for (i, chunk) in bytes.chunks_exact(8).enumerate() {
    v[i] = f64::from_le_bytes(chunk.try_into().expect("chunk_exact yields 8 bytes"));
  }
  v
}

/// Decode `rows × cols` little-endian f64 values from a row-major byte
/// slice (numpy C-order) into a nalgebra `DMatrix` with the same
/// element ordering. Note: nalgebra is column-major internally, but
/// `DMatrix::from_row_slice` does the transpose into the correct
/// element layout, so `m[(i, j)]` after the call returns the element
/// that was at offset `(i * cols + j) * 8` in `bytes`.
fn bytes_to_row_major_matrix(bytes: &[u8], rows: usize, cols: usize) -> DMatrix<f64> {
  debug_assert_eq!(bytes.len(), rows * cols * 8);
  let mut data = Vec::with_capacity(rows * cols);
  for chunk in bytes.chunks_exact(8) {
    data.push(f64::from_le_bytes(
      chunk.try_into().expect("chunk_exact yields 8 bytes"),
    ));
  }
  DMatrix::from_row_slice(rows, cols, &data)
}

#[cfg(test)]
mod loader_internal_tests {
  use super::*;

  /// Smoke-check the byte decoder against a known-shape vector.
  /// Catches endianness mistakes (numpy default is `<f8` so
  /// little-endian is correct on x86_64 / aarch64).
  #[test]
  fn bytes_to_vector_round_trip() {
    let v = vec![1.0_f64, 2.5, -3.25, 4.875];
    let mut bytes = Vec::with_capacity(v.len() * 8);
    for x in &v {
      bytes.extend_from_slice(&x.to_le_bytes());
    }
    let out = bytes_to_vector(&bytes, v.len());
    for (i, want) in v.iter().enumerate() {
      assert_eq!(out[i], *want);
    }
  }

  /// Smoke-check the matrix decoder. Numpy stores `lda[i, j]` at
  /// offset `(i * cols + j) * 8` in row-major (C-order); after
  /// `from_row_slice`, `m[(i, j)]` returns the same element.
  #[test]
  fn bytes_to_row_major_matrix_round_trip() {
    // 2x3 matrix:
    //   [[1.0, 2.0, 3.0],
    //    [4.0, 5.0, 6.0]]
    let v = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut bytes = Vec::with_capacity(v.len() * 8);
    for x in &v {
      bytes.extend_from_slice(&x.to_le_bytes());
    }
    let m = bytes_to_row_major_matrix(&bytes, 2, 3);
    assert_eq!(m[(0, 0)], 1.0);
    assert_eq!(m[(0, 1)], 2.0);
    assert_eq!(m[(0, 2)], 3.0);
    assert_eq!(m[(1, 0)], 4.0);
    assert_eq!(m[(1, 1)], 5.0);
    assert_eq!(m[(1, 2)], 6.0);
  }

  #[test]
  fn embedded_blobs_load_with_correct_shapes() {
    let xvec = load_xvec();
    assert_eq!(xvec.mean1.len(), EMBEDDING_DIMENSION);
    assert_eq!(xvec.mean2.len(), PLDA_DIMENSION);
    assert_eq!(xvec.lda.shape(), (EMBEDDING_DIMENSION, PLDA_DIMENSION));

    let plda = load_plda();
    assert_eq!(plda.mu.len(), PLDA_DIMENSION);
    assert_eq!(
      plda.eigenvectors_desc.shape(),
      (PLDA_DIMENSION, PLDA_DIMENSION)
    );
    assert_eq!(plda.phi_desc.len(), PLDA_DIMENSION);
  }

  /// Cross-check against Python-printed reference values from
  /// `pyannote/speaker-diarization-community-1` (snapshot
  /// `3533c8cf8e369892e6b79ff1bf80f7b0286a54ee`). Catches silent
  /// row/column transposition or endianness bugs in the byte
  /// decoder. Values come from
  /// `python3 -c "import numpy as np; x=np.load('models/plda/xvec_transform.npz'); print(x['mean1'][0], x['lda'][0,0], x['lda'][0,1], x['lda'][1,0])"`.
  #[test]
  fn embedded_xvec_blob_matches_python_reference_values() {
    let xvec = load_xvec();
    assert!((xvec.mean1[0] - (-0.1253200000_f64)).abs() < 1e-12);
    assert!((xvec.mean1[EMBEDDING_DIMENSION - 1] - (-0.0476597300_f64)).abs() < 1e-7);
    // The corner triplet (lda[0,0], lda[0,1], lda[1,0]) localizes
    // any silent transposition: if rows and columns swap, [0,1]
    // would land where [1,0] should.
    assert!((xvec.lda[(0, 0)] - (-1.1149908304_f64)).abs() < 1e-9);
    assert!((xvec.lda[(0, 1)] - 1.1986296177_f64).abs() < 1e-9);
    assert!((xvec.lda[(1, 0)] - 0.3553361595_f64).abs() < 1e-9);
    assert!((xvec.lda[(255, 127)] - (-6.7928419113_f64)).abs() < 1e-9);
  }
}
