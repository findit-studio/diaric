//! Adaptive score normalization (AS-Norm) over speaker-trial scores.
//!
//! Raw similarity scores are not comparable across speakers. One speaker
//! sits in a crowded region of the embedding space and scores highly
//! against everyone; another sits alone and scores lower against
//! everyone. A single global threshold therefore over-merges the first
//! and under-merges the second. AS-Norm rescales each trial score against
//! the *cohort distribution of the two speakers involved*, so one
//! threshold means the same thing everywhere.
//!
//! # The variant implemented here
//!
//! **AS-Norm1**, as defined by Matějka, Novotný, Plchot, Burget, Diez
//! Sánchez and Černocký, *"Analysis of Score Normalization in
//! Multilingual Speaker Recognition"*, Interspeech 2017, eq. (7). For a
//! trial between `e` and `t` scoring `s(e, t)`:
//!
//! ```text
//!                1   ⎛ s(e,t) − μ(Sₑ(Eₑᵗᵒᵖ))     s(e,t) − μ(Sₜ(Eₜᵗᵒᵖ)) ⎞
//! as-norm1  =   ─── ⎜ ───────────────────── + ───────────────────── ⎟
//!                2   ⎝      σ(Sₑ(Eₑᵗᵒᵖ))             σ(Sₜ(Eₜᵗᵒᵖ))       ⎠
//! ```
//!
//! Each side selects its own top-N of the shared cohort (`Eₑᵗᵒᵖ`,
//! `Eₜᵗᵒᵖ`) and takes the mean and standard deviation of exactly those
//! selected scores. That per-side independence is the whole reason this
//! variant is the practical one — see [Cost](#cost).
//!
//! The name AS-Norm is due to Cumani, Batzu, Colibro, Vair, Laface and
//! Vasilakakis, *"Comparison of Speaker Recognition Approaches for Real
//! Applications"*, Interspeech 2011, eq. (3), whose original formulation
//! is the **crossed** variant that Matějka later labels AS-Norm2 (eq. 8):
//! there each side's statistics are taken over the subset selected by the
//! *other* side. AS-Norm2 is not implemented here. Matějka's own
//! measurement is that it *"performs about the same as adaptive
//! S-norm1"*, and it cannot be precomputed per speaker — its statistics
//! depend on the pairing, so every trial would re-derive both sides.
//!
//! ## The `0.5` is load-bearing here
//!
//! Both primary sources carry the `1/2`, and so does WeSpeaker
//! (`wespeaker/bin/score_norm.py`). Because it is a global positive
//! constant it cannot change any *ranking*, so EER, AUC and minDCF are
//! identical with or without it. But a clustering stage compares the
//! normalized score against a **fixed absolute threshold**, and there a
//! factor of two is the difference between a tuned threshold and a
//! meaningless one. Keeping it is what makes a threshold derived from
//! published AS-Norm numbers directly usable.
//!
//! ## Population, not sample, standard deviation
//!
//! Divisor `N`, not `N − 1`. This matches the reference implementation,
//! which calls `np.std` without `ddof` — and NumPy's default is `ddof=0`.
//! With `top_n` in the recommended 200–500 band the two differ by well
//! under a percent, but matching the reference exactly is what makes a
//! published threshold transferable.
//!
//! # Nothing here knows what an embedding is
//!
//! AS-Norm is arithmetic on **scores**. Accordingly no type or function
//! in this module mentions an embedding, a vector, or a dimension:
//!
//! - [`CohortStats::from_scores`] takes an iterator of `f64` scores.
//! - [`Cohort<K, T>`] is generic over the item type `T`, and reaches it
//!   only through a caller-supplied `FnMut(&S, &T) -> f64`.
//!
//! So the scoring function is a parameter too — cosine
//! ([`cosine_similarity`](crate::embed::cosine_similarity)) today, PLDA
//! ([`PldaTransform`](crate::plda::PldaTransform)) later, both already in
//! this crate — and a future model emitting 192-d (ECAPA, CAM++) or 512-d
//! (XVEC) vectors instead of today's 256-d needs no change here at all.
//!
//! ```
//! use diaric::{
//!   embed::{Embedding, cosine_similarity},
//!   score_norm::{AsNormOptions, Cohort, as_norm},
//! };
//!
//! # // Stand-in for real inference output: spread-out unit vectors, so
//! # // the cohort scores actually vary. Real callers pass whatever their
//! # // embedder produced — the dimension is theirs, not this module's.
//! # fn speaker(seed: usize) -> Embedding {
//! #   let mut v = [0.0f32; diaric::embed::EMBEDDING_DIM];
//! #   for (i, x) in v.iter_mut().enumerate() {
//! #     *x = (((i * 37 + seed * 101) % 97) as f32) / 97.0 - 0.5;
//! #   }
//! #   Embedding::normalize_from(v).unwrap()
//! # }
//! let options = AsNormOptions::new();
//! let (alice, bob) = (speaker(200), speaker(201));
//!
//! // A cohort sampled from the speaker library itself — so it contains
//! // Alice and Bob, exactly the situation `stats_excluding` is for.
//! let mut cohort: Cohort<u32, Embedding> = Cohort::new();
//! for i in 0..64 {
//!   cohort.push(1000 + i as u32, speaker(i));
//! }
//! cohort.push(1, alice);
//! cohort.push(2, bob);
//!
//! // The dimension appears nowhere below: only the scoring closure,
//! // which is the single place the embedding type is named at all.
//! let score = |a: &Embedding, b: &Embedding| f64::from(cosine_similarity(a, b));
//!
//! // Computed once per speaker, then reused for every trial. Each side
//! // excludes its own entry: without that, Alice's self-match scores a
//! // perfect 1.0 and top-N selection is guaranteed to pick it up.
//! let alice_side = cohort.stats_excluding(&1, &alice, score, &options)?;
//! let bob_side = cohort.stats_excluding(&2, &bob, score, &options)?;
//! assert_eq!(alice_side.considered(), 65); // 66 members, less Alice's own
//!
//! let raw = score(&alice, &bob);
//! let normalized = as_norm(raw, &alice_side, &bob_side)?;
//! # assert!(normalized.is_finite());
//! # Ok::<(), diaric::score_norm::Error>(())
//! ```
//!
//! # Cohort self-contamination
//!
//! The failure this module works hardest to prevent. A cohort is meant to
//! be impostors; if the speaker being normalized is in it, top-N
//! selection is *guaranteed* to pick up its own material, because a
//! self-match is the highest score obtainable. Choose deliberately
//! between the two entrypoints:
//!
//! - [`Cohort::stats_excluding`] — drops every entry whose speaker key
//!   matches the one being normalized. Use this whenever the cohort is
//!   drawn from the same library being scored.
//! - [`Cohort::stats_assuming_disjoint`] — scores the whole cohort. Only
//!   correct for a held-out cohort, which is what the primary sources
//!   assume.
//!
//! # Cost
//!
//! A [`CohortStats`] depends only on its own side and the cohort, so it
//! is computed **once per speaker** and reused. Scoring `N` speakers
//! against each other with a cohort of `C`:
//!
//! | | cohort scores | trial scores |
//! |---|---|---|
//! | recomputing per trial | `N(N−1)·C` | `N(N−1)/2` |
//! | reusing [`CohortStats`] | `N·C` | `N(N−1)/2` |
//!
//! For `N = 1000`, `C = 300` that is 300 000 cohort scores instead of
//! 300 million. Selection itself is linear in the cohort, not
//! `O(C log C)`, since only the membership of the top-N matters.
//!
//! # Not implemented
//!
//! Matějka §4.1 also advises rejecting cohort scores outside ±4–5σ of the
//! cohort mean before taking statistics. No reference implementation does
//! this, and it is a second, independent policy question; it is left to
//! the caller, who can filter the score iterator before it reaches
//! [`CohortStats::from_scores`].

mod cohort;
mod error;
mod options;
mod stats;

#[cfg(test)]
mod tests;

pub use cohort::{Cohort, CohortEntry};
pub use error::{CohortTooSmall, DegenerateDeviation, Error};
pub use options::{AsNormOptions, DEFAULT_MIN_DEVIATION, DEFAULT_TOP_N, MIN_COHORT_SCORES};
pub use stats::{CohortStats, as_norm};

// Compile-time trait assertions. Catches a future field-type change that
// would silently regress Send/Sync auto-derive on the public types.
const _: fn() = || {
  fn assert_send_sync<T: Send + Sync>() {}
  assert_send_sync::<AsNormOptions>();
  assert_send_sync::<CohortStats>();
  assert_send_sync::<Error>();
  assert_send_sync::<CohortTooSmall>();
  assert_send_sync::<DegenerateDeviation>();
  assert_send_sync::<Cohort<u32, crate::embed::Embedding>>();
  assert_send_sync::<CohortEntry<u32, crate::embed::Embedding>>();
};
