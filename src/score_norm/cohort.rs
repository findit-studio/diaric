//! The impostor [`Cohort`] and identity-keyed self-exclusion.

use crate::score_norm::{AsNormOptions, CohortStats, Error};

/// One cohort member: an item, plus the speaker identity it belongs to.
///
/// The key exists for one reason — so
/// [`Cohort::stats_excluding`] can keep a speaker out of its own cohort.
/// It is compared for equality and nothing else, so anything that names a
/// speaker works: a `u32` row id, a `String`, a cluster label. The
/// comparison is [`Eq`], not [`PartialEq`] — see
/// [`Cohort::stats_excluding`] for why the difference is load-bearing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CohortEntry<K, T> {
  speaker: K,
  item: T,
}

impl<K, T> CohortEntry<K, T> {
  /// Pair an item with the speaker it came from.
  pub const fn new(speaker: K, item: T) -> Self {
    Self { speaker, item }
  }

  /// The speaker identity this item belongs to.
  pub const fn speaker(&self) -> &K {
    &self.speaker
  }

  /// The item itself — whatever the scoring function consumes.
  pub const fn item(&self) -> &T {
    &self.item
  }
}

/// The impostor cohort each side of a trial is scored against.
///
/// Generic over the key type `K` and the item type `T`. `T` is whatever
/// the caller's scoring function takes — [`Embedding`](crate::embed::Embedding)
/// today, a [`PostXvecEmbedding`](crate::plda::PostXvecEmbedding) or a
/// future model's wider vector tomorrow. Nothing here constrains it, so
/// nothing here has to change when it changes.
///
/// # Composition
///
/// Matějka et al. 2017 §4.1 states the assumption the cohort is supposed
/// to satisfy: *"The cohort set has an assumption to contain only one
/// file per speaker"*, and recommends unsupervised clustering to enforce
/// it on unlabelled data. WeSpeaker implements exactly that by averaging
/// each speaker's utterances into a single embedding before using them as
/// the cohort. One entry per speaker is therefore the intended shape —
/// but [`stats_excluding`](Self::stats_excluding) does not depend on it,
/// and a cohort with several entries per speaker still excludes all of a
/// speaker's entries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cohort<K, T> {
  entries: Vec<CohortEntry<K, T>>,
}

impl<K, T> Default for Cohort<K, T> {
  /// An empty cohort. Hand-written rather than derived: the derive would
  /// demand `K: Default, T: Default`, which an empty `Vec` does not need.
  fn default() -> Self {
    Self {
      entries: Vec::new(),
    }
  }
}

impl<K, T> Cohort<K, T> {
  /// An empty cohort (see [`Default`]).
  pub fn new() -> Self {
    Self::default()
  }

  /// Wrap pre-built entries.
  pub const fn from_entries(entries: Vec<CohortEntry<K, T>>) -> Self {
    Self { entries }
  }

  /// Append one item under the speaker it belongs to.
  pub fn push(&mut self, speaker: K, item: T) {
    self.entries.push(CohortEntry::new(speaker, item));
  }

  /// The cohort's entries.
  pub fn entries(&self) -> &[CohortEntry<K, T>] {
    &self.entries
  }

  /// Number of cohort members.
  pub fn len(&self) -> usize {
    self.entries.len()
  }

  /// Whether the cohort holds no members.
  pub fn is_empty(&self) -> bool {
    self.entries.is_empty()
  }

  /// Statistics for `side` over **every** cohort member.
  ///
  /// This is the literature's own path, and its precondition is in the
  /// name. Matějka et al. 2017 §2.1 defines the cohort as one *"which we
  /// assume to be different from the speakers in utterances e and t"*,
  /// and every reference implementation satisfies that structurally — by
  /// drawing the cohort from a held-out corpus — rather than by filtering
  /// per trial. When your cohort is genuinely disjoint from the speakers
  /// you score, this is correct and marginally cheaper.
  ///
  /// When it is not, use [`Self::stats_excluding`]. Reaching for this
  /// method with a cohort sampled from the same library you are scoring
  /// is the self-contamination failure described there, and no type can
  /// catch it for you — which is why the promise is spelled out in the
  /// method name.
  ///
  /// # Errors
  ///
  /// See [`CohortStats::from_scores`].
  pub fn stats_assuming_disjoint<S, F>(
    &self,
    side: &S,
    mut score: F,
    options: &AsNormOptions,
  ) -> Result<CohortStats, Error>
  where
    F: FnMut(&S, &T) -> f64,
  {
    CohortStats::from_scores(self.entries.iter().map(|e| score(side, &e.item)), options)
  }
}

impl<K: Eq, T> Cohort<K, T> {
  /// Statistics for `side`, excluding every entry belonging to
  /// `speaker`.
  ///
  /// # Why this exists
  ///
  /// A cohort is supposed to be impostors. If the speaker being
  /// normalized is *in* its own cohort, `μ` is inflated and `σ` distorted
  /// in a way nothing downstream can detect — the side looks perfectly
  /// healthy and every score derived from it is wrong.
  ///
  /// Adaptive selection makes this worse than it would be for plain
  /// S-Norm, not better. Top-N takes the **largest** scores, and a
  /// speaker's own material scores higher against itself than any
  /// impostor can — for L2-normalized embeddings under cosine, a
  /// self-match is exactly `1.0`, the global maximum. A self-entry is
  /// therefore not merely *present* in the selected set, it is
  /// *guaranteed* to be selected and to sit at the top of it. The more
  /// material a library holds for a speaker, the more of that speaker's
  /// own scores crowd out the impostors, so the bias grows with exactly
  /// the speakers you know best.
  ///
  /// # Why an identity key, and not a score threshold
  ///
  /// Dropping cohort scores above some cutoff would also remove the
  /// self-matches — along with every genuinely-similar impostor, which
  /// are the informative part of the tail AS-Norm exists to measure.
  /// Identity is the only signal that separates "this is me" from "this
  /// impostor sounds like me", and only the caller has it.
  ///
  /// # Relation to the literature
  ///
  /// This is an extension, stated plainly: the primary sources assume a
  /// disjoint cohort (see [`Self::stats_assuming_disjoint`]) and no
  /// reference implementation performs a per-trial exclusion, because
  /// under that assumption none is needed. Sampling the cohort from the
  /// same library being scored — which is what a diarization or
  /// speaker-library deployment naturally does — falls outside what those
  /// sources cover. Excluding by identity is the conservative reading:
  /// it restores the disjointness the math was derived under.
  ///
  /// # Statistics stay per-side
  ///
  /// Only `speaker`'s **own** entries are removed — never the other side
  /// of the trial. That is what keeps a side a property of one speaker
  /// plus the cohort, and therefore reusable across every trial that
  /// speaker appears in. Excluding the partner too would make the
  /// statistics trial-dependent and give back the quadratic cost the
  /// precomputation exists to avoid.
  ///
  /// # Why `K: Eq` and not `K: PartialEq`
  ///
  /// Identity exclusion is `entry.speaker != *speaker`, and that is a
  /// correct exclusion test only if a key equals **itself**. [`Eq`] is
  /// precisely the marker for that reflexivity; [`PartialEq`] promises
  /// only symmetry and transitivity, and `f64` is the standard type that
  /// takes the licence: `f64::NAN != f64::NAN` is `true`, so a `NaN`
  /// speaker key does not match its own entry and the filter keeps it.
  ///
  /// That is not a near-miss. A retained self-entry scores `1.0` under
  /// cosine — the global maximum — so it is *guaranteed* to be selected
  /// into the top-N and to sit at the top of it. With cohort scores
  /// `[self 1.0, 0.8, 0.2]` and `top_n = 2` the selection becomes
  /// `[1.0, 0.8]` instead of `[0.8, 0.2]`, and every normalized score for
  /// that speaker is biased — silently, since the side still looks
  /// healthy.
  ///
  /// Requiring [`Eq`] makes that state unrepresentable rather than
  /// diagnosed: a non-reflexive key is rejected by the compiler, at the
  /// call site, with no runtime check to forget. Every sensible speaker
  /// identity — an integer id, a `String`, a `&str`, a cluster label, a
  /// UUID — is already [`Eq`]; only the floating-point types are not.
  ///
  /// ```compile_fail,E0277
  /// use diaric::score_norm::{AsNormOptions, Cohort};
  /// // `f64` is `PartialEq` but not `Eq`, so it cannot be a speaker key.
  /// let mut cohort: Cohort<f64, f64> = Cohort::new();
  /// cohort.push(f64::NAN, 0.5);
  /// let _ = cohort.stats_excluding(
  ///   &f64::NAN,
  ///   &(),
  ///   |_: &(), item: &f64| *item,
  ///   &AsNormOptions::new(),
  /// );
  /// ```
  ///
  /// The bound sits on this method's `impl` block alone. Building a
  /// cohort, reading it back and
  /// [`stats_assuming_disjoint`](Self::stats_assuming_disjoint) never
  /// compare keys, so they keep working for any `K` at all.
  ///
  /// # Errors
  ///
  /// See [`CohortStats::from_scores`]. In particular a cohort whose every
  /// entry belongs to `speaker` yields [`Error::EmptyCohort`].
  pub fn stats_excluding<S, F>(
    &self,
    speaker: &K,
    side: &S,
    mut score: F,
    options: &AsNormOptions,
  ) -> Result<CohortStats, Error>
  where
    F: FnMut(&S, &T) -> f64,
  {
    CohortStats::from_scores(
      self
        .entries
        .iter()
        .filter(|e| e.speaker != *speaker)
        .map(|e| score(side, &e.item)),
      options,
    )
  }
}
