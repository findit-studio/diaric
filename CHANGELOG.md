# UNRELEASED

# 0.2.0 (September 1st, 2026)

PUBLIC API

- **What this crate binds under semver.** `cluster` (with its `ahc`,
  `centroid`, `hungarian`, `online` and `vbx` submodules), `pipeline`,
  `reconstruct`, `plda`, `aggregate`, `offline`, `segment`, `embed`,
  `provenance`, `spill`, and — new in this release — `score_norm` are
  unconditionally `pub`. They are public surface, not incidental
  exposure: a breaking change to any of them moves the crate version.
  Every one but `score_norm` was already `pub` in the released 0.1.0, so
  this records the existing contract rather than extending it. `ops` is
  the sole exception and stays crate-private — it is `pub` only under
  the internal, `doc(hidden)` `_bench` feature, which promises nothing;
  the two re-exports out of it, `axpy_f32` at the crate root and the
  `spill` module's types, are public. Direct use of the `cluster`
  submodules remains uncommon — `pipeline` and `offline` are the
  supported entrypoints — but uncommon is not unstable. 0.1.0's feature
  table called the kernel modules internal and disclaimed the public
  API for them; that was wrong, and it and the manifest comment behind
  it are corrected here.

- **Every `nalgebra` major bump is a breaking change for this crate.** A
  known cost, not a defect. `pipeline::AssignEmbeddingsInput::phi`,
  `cluster::hungarian::constrained_argmax`,
  `cluster::centroid::weighted_centroids`, and `cluster::vbx`'s
  `vbx_iterate` and `VbxOutput` (`new`, `gamma`, `pi`, `into_parts`)
  hand `DMatrix<f64>`, `DVector<f64>` and `DMatrixView<'_, f64>` across
  the API boundary, so nalgebra is a public dependency and its versions
  are not interchangeable. The `0.34 → 0.35` bump below is exactly that,
  it recurs on every future nalgebra major, and a downstream pinning
  nalgebra must move in lockstep. Wrapping those accessors in owned
  types would sever the coupling; that is API design work and is not
  attempted here.

CHANGED

- **`mediatime` `0.1` → `0.4`.** mediatime is a public dependency —
  `segment::SAMPLE_RATE_TB` is a `Timebase`, and `WindowId`,
  `SpeakerActivity` and `SegmentEvent::VoiceSpan` hand back `Timestamp`
  and `TimeRange` — so its breakage is diaric's breakage, and the crate
  version goes to `0.2.0` for it.
  - **`Timebase` is signed.** `num: u32 → i32` and
    `den: NonZeroU32 → NonZeroI32`, matching ffmpeg's `AVRational`, so
    `Timebase::new`, `num`, `den` and the `with_*` / `set_*` setters all
    change signature. `SAMPLE_RATE_TB` moves its denominator literal to
    `NonZeroI32`; `SAMPLE_RATE_HZ` stays `u32`, because it counts samples
    rather than dividing them. `Timebase::new` also panics now on a
    negative numerator or denominator — unreachable here, the one
    construction site being the `1 / 16_000` constant.
  - **No value in this crate moves.** 0.3's other two breaking changes
    are the `checked_`/`saturating_` ladder replacing the bare-name
    arithmetic, and rounding corrected from truncation to
    nearest-with-ties-away-from-zero (`AV_ROUND_NEAR_INF`). diaric never
    rescales, so the first has no call site here at all. The second
    reaches exactly one conversion — `WindowId::range().duration()`,
    which turns a tick count into a `core::time::Duration` — and at
    `1 / 16_000` a tick is exactly 62 500 ns, so every count converts
    exactly under either rounding. Checked over the first 200 000 ticks:
    no value moves. Window and voice-span boundaries stay exact sample
    indices, and the segment suites are unchanged.
  - **0.4 adds a type and moves nothing.** Its only section is
    `### Added`: an unsigned `Duration` (`{ ticks: u64, timebase }`)
    beside `SignedDuration`, conversions both ways with
    `core::time::Duration` and with `SignedDuration`, `Display` /
    `FromStr`, and the serde / quickcheck / arbitrary treatments.
    Claiming the name `Duration` for it moved the standard type to an
    internal `StdDuration` alias crate-wide, which respells eleven
    existing signatures — `Timebase::checked_pts_to_duration`,
    `TimeRange::duration` and `Timestamp::duration_since` among them —
    without changing what any of them accepts or returns, `StdDuration`
    being `core::time::Duration` itself. Diffed signature-for-signature
    across the two versions, that respelling is the *whole* delta to the
    existing surface: nothing is removed, nothing changes type, and
    diaric compiles and passes its suites against 0.4 with no source
    change at all. Taken now because 0.2.0 has not shipped — cargo reads
    a 0.x minor as incompatible, so deferring it would spend a `0.3.0`
    on an upgrade that absorbs nothing.

- **`nalgebra` `0.34` → `0.35`.** nalgebra is a public dependency too,
  across three `cluster` submodules: `hungarian::constrained_argmax`
  takes `&[DMatrix<f64>]`; `centroid::weighted_centroids` takes
  `&DMatrix<f64>` and `&DVector<f64>` and returns `DMatrix<f64>`;
  `vbx::vbx_iterate` takes `DMatrixView<'_, f64>`, `&DVector<f64>` and
  `&DMatrix<f64>`, and `VbxOutput`'s `new`, `gamma`, `pi` and
  `into_parts` carry `DMatrix<f64>` and `DVector<f64>` in and out. So
  this bump is a breaking change for the same reason mediatime's is.
  - **No diaric signature moves — the break is type identity.** Not one
    line of this crate changed for it; the bump is a single line of
    `Cargo.toml`. What breaks is that `nalgebra 0.34`'s `DMatrix<f64>`
    and `nalgebra 0.35`'s are distinct types, so a caller still on 0.34
    can no longer hand a matrix to any of the functions above. Callers
    move in lockstep and change nothing but the version they name.
  - **Almost none of 0.35's own delta reaches here.** The additions
    (`Matrix::lblt`, `OneNorm` / `Matrix::one_norm`, `convert-glam033`)
    are unused by this crate. The `rand` `0.9 → 0.10` and `rand_distr`
    `0.5 → 0.6` updates sit behind nalgebra's optional `rand` feature,
    which diaric does not enable — `rand` enters the graph only through
    diaric's own direct dependency — so they unify nothing and move
    nothing. The `convert-glamXXX` removals drop features for `glam`
    below 0.30, none of them enabled. nalgebra's own MSRV goes
    `1.87 → 1.89`, still under this crate's declared `1.95`, which is
    unchanged and was built and checked at `1.95` for this release. The
    `SymmetricEigen` fix that the (private) spectral path would care
    about landed in `0.34.2`, so the previous `"0.34"` requirement
    already resolved to it.
  - **The one change that touches a type in this crate's public surface
    is a no-op here.** 0.35 corrects `ViewStorage`'s `unsafe impl Send`
    from `T: Send` to `T: Sync` — the sound bound for a `&T`-equivalent
    — and leaves `ViewStorageMut` at `T: Send`. That tightens the shared
    view rather than loosening it, which nalgebra's changelog line
    ("so view storages can be sent across threads when `T: Send`")
    describes only from the mutable side. `DMatrixView<'_, f64>` in
    `vbx_iterate` is `Send` under either bound, `f64` being both `Send`
    and `Sync`, so no caller can observe the difference.

FEATURES

- **`score_norm` — adaptive score normalization (AS-Norm1).** A new
  module implementing Matějka et al., Interspeech 2017, eq. (7):
  rescale each trial score against the cohort distribution of the two
  speakers involved, so one fixed absolute threshold means the same
  thing for a speaker in a crowded region of the embedding space and for
  one sitting alone. Additive — nothing else in the crate calls it, and
  it does not itself force the version.
  - **The API.** `Cohort<K, T>` / `CohortEntry<K, T>` hold the cohort;
    `CohortStats` is one side's precomputed `μ`, `σ`, `selected` and
    `considered`; `as_norm(raw, &enrollment, &test)` (or
    `CohortStats::normalize`) combines two sides into a normalized
    score. `AsNormOptions` carries `top_n` and `min_deviation` with
    `DEFAULT_TOP_N`, `DEFAULT_MIN_DEVIATION` and `MIN_COHORT_SCORES`
    published beside it. Failures are a typed `Error` — `EmptyCohort`,
    `CohortTooSmall`, `NonFiniteScore`, `DegenerateDeviation`,
    `NonFiniteResult`, `ZScoreCancellation`, `InvalidMinDeviation` —
    with the three structured variants carrying their own inspectable
    payload types.
  - **Generic over the score source; no embedding dimension anywhere in
    it.** AS-Norm is arithmetic on scores, and the module is written
    that way: `CohortStats::from_scores` takes an
    `IntoIterator<Item = f64>`, and `Cohort<K, T>` reaches its item type
    only through a caller-supplied `FnMut(&S, &T) -> f64`. The scoring
    function is therefore a parameter — cosine
    (`embed::cosine_similarity`) today, PLDA (`plda::PldaTransform`)
    later, both already in this crate — and a model emitting 192-d
    (ECAPA, CAM++) or 512-d (XVEC) vectors instead of today's 256-d
    needs no change here at all. No type or function in the module
    mentions an embedding, a vector, or a dimension.
  - **`stats_excluding` vs `stats_assuming_disjoint`.** Two entrypoints,
    because the literature's cohort assumption does not hold for the
    deployment this crate serves. Matějka et al. define the cohort as
    one *"which we assume to be different from the speakers in
    utterances e and t"*, and every reference implementation satisfies
    that structurally, by drawing the cohort from a held-out corpus —
    so none of them filters per trial, and none needs to.
    `stats_assuming_disjoint` is that path, with its precondition stated
    in its name. But a diarization or speaker-library deployment
    naturally samples the cohort from the very library it is scoring,
    and there a self-match scores the global maximum and is therefore
    *guaranteed* to be selected into the top-N and to sit at the top of
    it — biasing every score for that speaker while the side still looks
    healthy. `stats_excluding` drops every entry whose speaker key
    matches the side being normalized, restoring the disjointness the
    math was derived under. It removes only that side's own entries,
    never the partner's, which is what keeps a `CohortStats` a property
    of one speaker plus the cohort and so reusable across every trial
    that speaker appears in. Its `K: Eq` bound is load-bearing and sits
    on that method's `impl` block alone: `PartialEq` does not promise a
    key equals itself, `f64::NAN` takes exactly that licence, and a
    non-reflexive key would silently fail to match its own entry. `Eq`
    makes that unrepresentable at the call site instead of diagnosing it
    at runtime; building a cohort, reading it back and
    `stats_assuming_disjoint` never compare keys and stay open to any
    `K`.
  - **`MAX_NORMALIZED_ERROR` is a postcondition, not a hint.** A
    successful normalization asserts an accuracy and not merely a
    finiteness: the returned value lands within
    `MAX_NORMALIZED_ERROR * max(|returned|, 1)` of the exact average of
    the two z-scores the stored statistics define — `2^-20`, measured in
    the unit the result itself is in, one cohort standard deviation, so
    the tolerance is absolute up to one deviation and relative beyond.
    The absolute floor is the load-bearing half: a purely relative
    criterion would refuse every result that legitimately cancels to
    near zero, which is precisely where a match threshold lives. A trial
    that cannot meet it returns `Error::ZScoreCancellation` instead of
    an `Ok`, carrying the z-scores, the value and the bound that
    convicted it. The guard only decides whether a value is returned —
    it never alters one — and it decides in two tiers, an operand-side
    bound that clears the overwhelming majority of trials outright and,
    only for what that bound cannot clear, an exact residual identity
    recovered with `f64::mul_add` and a two-sum, so that an answer whose
    arithmetic was in fact exact is not thrown away by an upper bound
    that merely permitted an error nobody made.

# 0.1.0

Initial release: the backend-free diarization core extracted (history-preserving)
from the `diarization` crate — clustering (offline AHC→VBx, online), PLDA,
pipeline assembly, reconstruction/RTTM, kaldi-fbank DSP and embedding types, and
the SIMD/mmap numeric-ops layer. Carries no ONNX/Torch dependency; the model
runners remain in `diarization`.
