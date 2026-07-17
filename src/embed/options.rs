//! Constants for `diarization::embed`. All values match spec §4.2 / §5.

/// 2 s @ 16 kHz; the WeSpeaker model's fixed input length.
///
/// Named with the `EMBED_` prefix to avoid collision with
/// `diarization::segment::WINDOW_SAMPLES` (160 000 = 10 s at the same rate).
pub const EMBED_WINDOW_SAMPLES: u32 = 32_000;

/// 1 s @ 16 kHz; sliding-window hop for the long-clip path (§5.1).
/// 50 % overlap with `EMBED_WINDOW_SAMPLES`.
pub const HOP_SAMPLES: u32 = 16_000;

/// ~25 ms @ 16 kHz; one kaldi window. Below this, `embed` returns
/// [`Error::InvalidClip`](crate::embed::Error::InvalidClip).
pub const MIN_CLIP_SAMPLES: u32 = 400;

/// Number of mel bins in the kaldi fbank features (spec §4.2).
pub const FBANK_NUM_MELS: usize = 80;

/// Number of fbank frames per `EMBED_WINDOW_SAMPLES` of audio
/// (25 ms frame length, 10 ms shift → 200 frames per 2 s).
pub const FBANK_FRAMES: usize = 200;

/// Output dimensionality of the WeSpeaker ResNet34 embedding.
pub const EMBEDDING_DIM: usize = 256;

/// Numerical floor used in L2-normalization to avoid divide-by-zero.
/// Matches `findit-speaker-embedding`'s `1e-12` (verified at
/// `embedder.py:85`); diverging would lose Python parity in edge cases.
pub const NORM_EPSILON: f32 = 1e-12;

/// 16 kHz mono — the WeSpeaker ResNet34 expected sample rate.
/// Matches [`diarization::segment::SAMPLE_RATE_HZ`](crate::segment::SAMPLE_RATE_HZ).
pub const SAMPLE_RATE_HZ: u32 = 16_000;

// ── EmbedModelOptions ─────────────────────────────────────────────────────

#[cfg(feature = "ort")]
use ort::ep::ExecutionProviderDispatch;
#[cfg(feature = "ort")]
use ort::session::builder::{GraphOptimizationLevel, SessionBuilder};

/// Builder for [`EmbedModel`](crate::embed::EmbedModel) runtime configuration.
///
/// Mirrors [`SegmentModelOptions`](crate::segment::SegmentModelOptions): the
/// same four ort knobs (graph optimization level, execution providers,
/// intra/inter-op thread counts), with both consuming `with_*` and
/// in-place `set_*` builders.
///
/// Default: ort defaults for optimization level and threading, no
/// execution providers configured beyond ort's default search.
#[cfg(feature = "ort")]
#[cfg_attr(docsrs, doc(cfg(feature = "ort")))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EmbedModelOptions {
  #[cfg_attr(
    feature = "serde",
    serde(
      default = "default_optimization_level",
      with = "crate::ort_serde::graph_optimization_level"
    )
  )]
  optimization_level: GraphOptimizationLevel,
  #[cfg_attr(feature = "serde", serde(skip, default))]
  providers: Vec<ExecutionProviderDispatch>,
  #[cfg_attr(feature = "serde", serde(default = "default_threads"))]
  intra_threads: usize,
  #[cfg_attr(feature = "serde", serde(default = "default_threads"))]
  inter_threads: usize,
}

#[cfg(feature = "ort")]
const fn default_optimization_level() -> GraphOptimizationLevel {
  GraphOptimizationLevel::Disable
}

#[cfg(feature = "ort")]
const fn default_threads() -> usize {
  1
}

#[cfg(feature = "ort")]
impl Default for EmbedModelOptions {
  fn default() -> Self {
    Self {
      optimization_level: default_optimization_level(),
      providers: Vec::new(),
      intra_threads: default_threads(),
      inter_threads: default_threads(),
    }
  }
}

#[cfg(feature = "ort")]
impl EmbedModelOptions {
  /// Construct with all-default options.
  pub fn new() -> Self {
    Self::default()
  }

  // ── Builder (consuming with_*) ───────────────────────────────────────

  /// Override the graph optimization level.
  pub fn with_optimization_level(mut self, level: GraphOptimizationLevel) -> Self {
    self.optimization_level = level;
    self
  }

  /// Configure execution providers in priority order. Default: ort's
  /// default execution-provider selection (typically CPU).
  ///
  /// **Caveat:** non-CPU providers may degrade WeSpeaker ResNet34 numerics
  /// and break the byte-determinism guarantees in spec §11.9. Do not enable
  /// without measuring against the pyannote parity harness (Task 46).
  pub fn with_providers(mut self, providers: Vec<ExecutionProviderDispatch>) -> Self {
    self.providers = providers;
    self
  }

  /// Override `intra_threads`. Default is `1` for bit-exact
  /// reproducibility across runs (parallel reductions are not
  /// deterministic).
  pub fn with_intra_threads(mut self, n: usize) -> Self {
    self.intra_threads = n;
    self
  }

  /// Override `inter_threads`. Default is `1`.
  pub fn with_inter_threads(mut self, n: usize) -> Self {
    self.inter_threads = n;
    self
  }

  // ── Mutators (in-place set_*) ────────────────────────────────────────

  /// Set the graph optimization level (in-place).
  pub fn set_optimization_level(&mut self, level: GraphOptimizationLevel) -> &mut Self {
    self.optimization_level = level;
    self
  }

  /// Set the execution providers (in-place).
  pub fn set_providers(&mut self, providers: Vec<ExecutionProviderDispatch>) -> &mut Self {
    self.providers = providers;
    self
  }

  /// Set `intra_threads` (in-place).
  pub fn set_intra_threads(&mut self, n: usize) -> &mut Self {
    self.intra_threads = n;
    self
  }

  /// Set `inter_threads` (in-place).
  pub fn set_inter_threads(&mut self, n: usize) -> &mut Self {
    self.inter_threads = n;
    self
  }

  // ── Internal apply ───────────────────────────────────────────────────

  /// Apply the option set to a `SessionBuilder`. Used internally by
  /// [`EmbedModel`](crate::embed::EmbedModel).
  pub(crate) fn apply(
    self,
    mut builder: SessionBuilder,
  ) -> Result<SessionBuilder, crate::embed::Error> {
    builder = builder
      .with_optimization_level(self.optimization_level)
      .map_err(ort::Error::from)?;
    builder = builder
      .with_intra_threads(self.intra_threads)
      .map_err(ort::Error::from)?;
    builder = builder
      .with_inter_threads(self.inter_threads)
      .map_err(ort::Error::from)?;
    if !self.providers.is_empty() {
      builder = builder
        .with_execution_providers(self.providers)
        .map_err(ort::Error::from)?;
    }
    Ok(builder)
  }
}
