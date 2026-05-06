//! Public output types for `diarization::embed`. All types are `Send + Sync`.

use core::time::Duration;

use crate::embed::options::{EMBEDDING_DIM, NORM_EPSILON};

/// A 256-d L2-normalized speaker embedding.
///
/// **Invariant:** `||embedding.as_array()||₂ > NORM_EPSILON`. The crate
/// guarantees this — the only public constructor (`normalize_from`)
/// returns `None` for degenerate inputs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Embedding(pub(crate) [f32; EMBEDDING_DIM]);

impl Embedding {
  /// Borrow the raw L2-normalized 256-d vector.
  pub const fn as_array(&self) -> &[f32; EMBEDDING_DIM] {
    &self.0
  }

  /// Borrow as a slice.
  pub fn as_slice(&self) -> &[f32] {
    &self.0
  }

  /// Cosine similarity. Both inputs are L2-normalized (per the
  /// `Embedding` invariant), so this reduces to a dot product.
  /// Returns a value in `[-1.0, 1.0]`.
  pub fn similarity(&self, other: &Embedding) -> f32 {
    self.0.iter().zip(other.0.iter()).map(|(a, b)| a * b).sum()
  }

  /// L2-normalize a raw 256-d inference output and wrap it.
  ///
  /// Returns `None` if the result would not satisfy the `Embedding`
  /// invariant `||embedding|| > NORM_EPSILON`. This covers two cases:
  /// - **Non-finite input**: any `raw[i]` that's NaN or infinity makes
  ///   the L2 norm non-finite, division would propagate the corruption,
  ///   and the returned `Embedding` would silently violate the invariant.
  /// - **Degenerate norm**: `||raw||_2 < NORM_EPSILON`, division would
  ///   amplify floating-point noise to no useful direction.
  ///
  /// Use after running raw `EmbedModel` inference plus your own
  /// aggregation. The higher-level `EmbedModel::embed*` methods
  /// surface `None` here as
  /// [`Error::DegenerateEmbedding`](crate::embed::Error::DegenerateEmbedding);
  /// callers who need to distinguish NaN/inf from zero-norm should
  /// validate `raw` is_finite themselves before calling.
  pub fn normalize_from(raw: [f32; EMBEDDING_DIM]) -> Option<Self> {
    // Compute ||raw||₂ in f64 for precision, then divide each
    // component in f32. Matches Python's typical behavior where
    // the L2 norm is computed in float32.
    let sq: f64 = raw.iter().map(|&x| (x as f64) * (x as f64)).sum();
    let n = sq.sqrt() as f32;
    // !n.is_finite() catches NaN/inf inputs — the squared sum + sqrt
    // chain propagates non-finite into n. The `n < NORM_EPSILON` clause
    // rejects degenerate (zero-or-near-zero norm) inputs.
    if !n.is_finite() || n < NORM_EPSILON {
      return None;
    }
    let mut out = [0.0f32; EMBEDDING_DIM];
    for (o, &r) in out.iter_mut().zip(raw.iter()) {
      *o = r / n;
    }
    Some(Self(out))
  }
}

/// Free-function form of [`Embedding::similarity`] for callers who
/// prefer it. Both styles are public; pick whichever reads more
/// naturally at the call site. **Bit-exactly equivalent** to the
/// method (same component-order dot product, no FMA rearrangement).
pub fn cosine_similarity(a: &Embedding, b: &Embedding) -> f32 {
  a.similarity(b)
}

/// Optional metadata that flows through `embed_with_meta` /
/// `embed_weighted_with_meta` / `embed_masked_with_meta` to
/// `EmbeddingResult`. Generic over the `audio_id` and `track_id`
/// types — callers use whatever string-like type fits their domain.
/// Defaults to `()` so the unit-typed metadata path allocates nothing.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct EmbeddingMeta<A = (), T = ()> {
  pub(crate) audio_id: A,
  pub(crate) track_id: T,
  pub(crate) correlation_id: Option<u64>,
}

impl<A, T> EmbeddingMeta<A, T> {
  /// Construct with `audio_id` and `track_id`.
  pub fn new(audio_id: A, track_id: T) -> Self {
    Self {
      audio_id,
      track_id,
      correlation_id: None,
    }
  }

  /// Attach a correlation id (e.g., a session-scoped sequence number)
  /// for downstream telemetry / log correlation.
  pub fn with_correlation_id(mut self, id: u64) -> Self {
    self.correlation_id = Some(id);
    self
  }

  /// Caller-supplied audio identifier propagated through the
  /// embedding pipeline.
  pub fn audio_id(&self) -> &A {
    &self.audio_id
  }

  /// Caller-supplied track identifier propagated through the
  /// embedding pipeline.
  pub fn track_id(&self) -> &T {
    &self.track_id
  }

  /// Optional correlation id (for telemetry / log correlation).
  pub fn correlation_id(&self) -> Option<u64> {
    self.correlation_id
  }
}

/// Result of one `EmbedModel::embed*` call.
///
/// Carries the embedding plus observability fields:
/// - `source_duration`: actual length of the source clip (NOT padded/cropped)
/// - `windows_used`: number of 2 s windows averaged (1 for clips ≤ 2 s)
/// - `total_weight`: sum of per-window weights
/// - `audio_id`/`track_id`/`correlation_id`: caller-supplied metadata
#[derive(Debug, Clone)]
pub struct EmbeddingResult<A = (), T = ()> {
  embedding: Embedding,
  source_duration: Duration,
  windows_used: u32,
  total_weight: f32,
  audio_id: A,
  track_id: T,
  correlation_id: Option<u64>,
}

impl<A, T> EmbeddingResult<A, T> {
  /// Construct (typically from inside `EmbedModel`).
  // The only caller (`crate::embed::embedder`) is gated behind feature
  // `ort`. Under `--no-default-features` the constructor is unused but
  // we keep it reachable so `cargo test --no-default-features` (used
  // by SDE / miri CI lanes) compiles under `-Dwarnings`.
  //
  #[allow(dead_code)]
  pub(crate) fn new(
    embedding: Embedding,
    source_duration: Duration,
    windows_used: u32,
    total_weight: f32,
    meta: EmbeddingMeta<A, T>,
  ) -> Self {
    let EmbeddingMeta {
      audio_id,
      track_id,
      correlation_id,
    } = meta;
    Self {
      embedding,
      source_duration,
      windows_used,
      total_weight,
      audio_id,
      track_id,
      correlation_id,
    }
  }

  /// L2-normalized 256-d speaker embedding.
  pub fn embedding(&self) -> &Embedding {
    &self.embedding
  }

  /// Duration of the source audio clip (pre-padding, pre-cropping).
  pub fn source_duration(&self) -> Duration {
    self.source_duration
  }

  /// Number of 2 s windows averaged into the embedding (1 for clips
  /// ≤ 2 s; sliding-window aggregation for longer clips).
  pub fn windows_used(&self) -> u32 {
    self.windows_used
  }

  /// Sum of per-window weights used during aggregation. Zero ⇒
  /// the result is degenerate; callers may want to inspect this for
  /// quality gating.
  pub fn total_weight(&self) -> f32 {
    self.total_weight
  }

  /// Caller-supplied audio identifier propagated from
  /// [`EmbeddingMeta::audio_id`].
  pub fn audio_id(&self) -> &A {
    &self.audio_id
  }

  /// Caller-supplied track identifier propagated from
  /// [`EmbeddingMeta::track_id`].
  pub fn track_id(&self) -> &T {
    &self.track_id
  }

  /// Optional correlation id propagated from
  /// [`EmbeddingMeta::correlation_id`].
  pub fn correlation_id(&self) -> Option<u64> {
    self.correlation_id
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn normalize_from_zero_returns_none() {
    assert!(Embedding::normalize_from([0.0; EMBEDDING_DIM]).is_none());
  }

  #[test]
  fn normalize_from_below_epsilon_returns_none() {
    let mut tiny = [0.0; EMBEDDING_DIM];
    tiny[0] = 1e-13; // < NORM_EPSILON
    assert!(Embedding::normalize_from(tiny).is_none());
  }

  #[test]
  fn normalize_from_nan_returns_none() {
    // regression: NaN raw input previously produced
    // Some(Embedding) containing NaNs because `n = NaN` and `NaN < eps`
    // is false. is_finite() check catches this.
    let mut v = [0.5f32; EMBEDDING_DIM];
    v[0] = f32::NAN;
    assert!(Embedding::normalize_from(v).is_none());
  }

  #[test]
  fn normalize_from_positive_infinity_returns_none() {
    let mut v = [0.5f32; EMBEDDING_DIM];
    v[0] = f32::INFINITY;
    assert!(Embedding::normalize_from(v).is_none());
  }

  #[test]
  fn normalize_from_negative_infinity_returns_none() {
    let mut v = [0.5f32; EMBEDDING_DIM];
    v[0] = f32::NEG_INFINITY;
    assert!(Embedding::normalize_from(v).is_none());
  }

  #[test]
  fn normalize_from_mixed_inf_returns_none() {
    // Mixed +inf and -inf produce NaN sum; should reject.
    let mut v = [0.0f32; EMBEDDING_DIM];
    v[0] = f32::INFINITY;
    v[1] = f32::NEG_INFINITY;
    assert!(Embedding::normalize_from(v).is_none());
  }

  #[test]
  fn normalize_from_unit_vector_round_trips() {
    let mut v = [0.0; EMBEDDING_DIM];
    v[0] = 1.0;
    let e = Embedding::normalize_from(v).unwrap();
    let n2: f32 = e.as_array().iter().map(|x| x * x).sum();
    assert!((n2 - 1.0).abs() < 1e-6, "||result|| ≈ 1, got n2 = {n2}");
    assert!((e.as_array()[0] - 1.0).abs() < 1e-6);
  }

  #[test]
  fn normalize_from_arbitrary_vector_norms_to_one() {
    let mut raw = [0.0; EMBEDDING_DIM];
    for (i, v) in raw.iter_mut().enumerate() {
      *v = (i as f32) * 0.01 + 0.1;
    }
    let e = Embedding::normalize_from(raw).unwrap();
    let n2: f32 = e.as_array().iter().map(|x| x * x).sum();
    assert!((n2 - 1.0).abs() < 1e-5, "n2 = {n2}");
  }

  #[test]
  fn similarity_self_is_one() {
    let mut v = [0.0; EMBEDDING_DIM];
    v[0] = 1.0;
    let e = Embedding::normalize_from(v).unwrap();
    assert!((e.similarity(&e) - 1.0).abs() < 1e-6);
  }

  #[test]
  fn similarity_orthogonal_is_zero() {
    let mut a = [0.0; EMBEDDING_DIM];
    a[0] = 1.0;
    let mut b = [0.0; EMBEDDING_DIM];
    b[1] = 1.0;
    let ea = Embedding::normalize_from(a).unwrap();
    let eb = Embedding::normalize_from(b).unwrap();
    assert!(ea.similarity(&eb).abs() < 1e-6);
  }

  #[test]
  fn similarity_antipodal_is_negative_one() {
    let mut a = [0.0; EMBEDDING_DIM];
    a[0] = 1.0;
    let mut b = [0.0; EMBEDDING_DIM];
    b[0] = -1.0;
    let ea = Embedding::normalize_from(a).unwrap();
    let eb = Embedding::normalize_from(b).unwrap();
    assert!((ea.similarity(&eb) + 1.0).abs() < 1e-6);
  }

  #[test]
  fn similarity_symmetric() {
    let mut a = [0.0; EMBEDDING_DIM];
    a[0] = 0.6;
    a[1] = 0.8;
    let mut b = [0.0; EMBEDDING_DIM];
    b[0] = 0.8;
    b[1] = 0.6;
    let ea = Embedding::normalize_from(a).unwrap();
    let eb = Embedding::normalize_from(b).unwrap();
    assert!((ea.similarity(&eb) - eb.similarity(&ea)).abs() < 1e-7);
  }

  #[test]
  fn cosine_similarity_matches_method() {
    let mut a = [0.0; EMBEDDING_DIM];
    let mut b = [0.0; EMBEDDING_DIM];
    for (i, (av, bv)) in a.iter_mut().zip(b.iter_mut()).enumerate() {
      *av = (i as f32 * 0.01).sin();
      *bv = (i as f32 * 0.013).cos();
    }
    let ea = Embedding::normalize_from(a).unwrap();
    let eb = Embedding::normalize_from(b).unwrap();
    // Free fn must equal method bit-exactly (same dot product,
    // same component order — no fma rearrangement).
    assert_eq!(cosine_similarity(&ea, &eb), ea.similarity(&eb));
  }

  #[test]
  fn embedding_meta_unit_default() {
    let m: EmbeddingMeta = EmbeddingMeta::default();
    assert_eq!(m.audio_id(), &());
    assert_eq!(m.track_id(), &());
    assert_eq!(m.correlation_id(), None);
  }

  #[test]
  fn embedding_meta_typed() {
    let m = EmbeddingMeta::new("audio_42".to_string(), 7u32);
    assert_eq!(m.audio_id(), "audio_42");
    assert_eq!(m.track_id(), &7u32);
    assert_eq!(m.correlation_id(), None);
  }

  #[test]
  fn embedding_meta_with_correlation_id() {
    let m = EmbeddingMeta::new((), ()).with_correlation_id(123);
    assert_eq!(m.correlation_id(), Some(123));
  }

  #[test]
  fn embedding_result_unit_meta_construction() {
    let mut v = [0.0; EMBEDDING_DIM];
    v[0] = 1.0;
    let e = Embedding::normalize_from(v).unwrap();
    let r: EmbeddingResult = EmbeddingResult::new(
      e,
      Duration::from_millis(1500),
      1,
      1.0,
      EmbeddingMeta::default(),
    );
    assert_eq!(r.embedding(), &e);
    assert_eq!(r.windows_used(), 1);
    assert!((r.total_weight() - 1.0).abs() < 1e-7);
  }

  #[test]
  fn embedding_result_typed_meta() {
    let mut v = [0.0; EMBEDDING_DIM];
    v[0] = 1.0;
    let e = Embedding::normalize_from(v).unwrap();
    let r = EmbeddingResult::new(
      e,
      Duration::from_millis(2000),
      2,
      1.5,
      EmbeddingMeta::new("clip_3".to_string(), 9u32).with_correlation_id(42),
    );
    assert_eq!(r.audio_id(), "clip_3");
    assert_eq!(r.track_id(), &9u32);
    assert_eq!(r.correlation_id(), Some(42));
  }
}
