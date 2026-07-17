//! Model-identity accessors so a downstream pipeline reads the loaded
//! model's family + version instead of hardcoding them, giving each
//! diarization result a reproducible record of which model artifacts
//! (segmentation, embedding, clustering/PLDA) produced it.

/// Model family of the segmentation stage. `diaric` bundles no segmentation
/// model — the ONNX weights live in the `diarization` crate (see
/// `models/SOURCE.md`); this only records the family for a result's identity.
pub const SEGMENTATION_FAMILY: &str = "pyannote-segmentation-3.0";
/// Model family of the WeSpeaker embedding model.
pub const WESPEAKER_FAMILY: &str = "wespeaker";
/// Model family of the clustering / PLDA stage (the diarization decision).
pub const DIARIZATION_FAMILY: &str = "pyannote-diarization-community-1";
/// Version tag of the `segmentation-3.0` model. The `.onnx` weights are
/// bundled by the `diarization` crate, not by `diaric`; this records the
/// version for a result's identity.
pub const BUNDLED_SEGMENTATION_VERSION: &str = "segmentation-3.0";
/// Version string for the embedded community-1 PLDA weights.
pub const DIARIZATION_PLDA_VERSION: &str = "community-1";

/// A loaded model's reproducible identity: a static `family` plus a
/// dynamic `version` (bundled artifact tag or loaded-file basename).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelIdentity {
  family: &'static str,
  version: String,
}

impl ModelIdentity {
  /// Construct from a static family and a dynamic version.
  pub fn new(family: &'static str, version: impl Into<String>) -> Self {
    Self {
      family,
      version: version.into(),
    }
  }
  /// The model family (e.g. `wespeaker`).
  pub fn family(&self) -> &str {
    self.family
  }
  /// The model version (bundled tag or loaded basename).
  pub fn version(&self) -> &str {
    &self.version
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn plda_identity_reports_community_one() {
    let plda = crate::plda::PldaTransform::new().expect("plda");
    let id = plda.identity();
    assert_eq!(id.family(), DIARIZATION_FAMILY);
    assert_eq!(id.version(), DIARIZATION_PLDA_VERSION);
  }

  #[test]
  fn model_identity_accessors() {
    let id = ModelIdentity::new(WESPEAKER_FAMILY, "wespeaker_resnet34_lm.onnx");
    assert_eq!(id.family(), "wespeaker");
    assert_eq!(id.version(), "wespeaker_resnet34_lm.onnx");
  }
}
