//! Constants for `crate::embed`. All values match spec §4.2 / §5.

/// 2 s @ 16 kHz; the WeSpeaker model's fixed input length.
///
/// Named with the `EMBED_` prefix to avoid collision with
/// `crate::segment::WINDOW_SAMPLES` (160 000 = 10 s at the same rate).
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
/// Matches [`crate::segment::SAMPLE_RATE_HZ`](crate::segment::SAMPLE_RATE_HZ).
pub const SAMPLE_RATE_HZ: u32 = 16_000;
