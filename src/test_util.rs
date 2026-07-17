//! Test-only shared helpers.
//!
//! In particular: the parity-fixture skip macro
//! [`parity_fixtures_or_skip!`] used by every `*_parity_tests.rs`
//! module under `src/`.
//!
//! ## Why skip instead of fail?
//!
//! `tests/parity/fixtures/` ships in the **git repo** (~5 MiB of
//! captured pyannote intermediates) but is **excluded** from the
//! published crate tarball via `[package] exclude = ["tests/parity/"]`
//! in `Cargo.toml` so we stay under the 10 MiB crates.io limit.
//! Crates.io users running `cargo test` against the published crate
//! therefore have the parity test source files (compiled into
//! `cargo test`) but no fixtures to feed them — without this skip
//! macro every parity test would `assert!` and panic on missing
//! files.
//!
//! Workspace developers (running `cargo test` from a checkout) have
//! the fixtures present and run the full parity suite. Crates.io
//! consumers see the parity tests skip cleanly with a one-line
//! stderr note.

#![cfg(test)]

use std::path::PathBuf;

/// Path to the dia crate root (the directory containing `Cargo.toml`).
pub fn repo_root() -> PathBuf {
  PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// `Some(...)` if `tests/parity/fixtures/` is present (workspace
/// build); `None` if the directory is absent (published crate
/// tarball — `[package] exclude` removes it).
pub fn parity_fixtures_root() -> Option<PathBuf> {
  let p = repo_root().join("tests/parity/fixtures");
  if p.is_dir() { Some(p) } else { None }
}

/// Skip the current test if `tests/parity/fixtures/` is not shipped
/// (e.g. the published crate tarball). Use at the top of every
/// `#[test]` (or its helper) that loads a parity fixture:
///
/// ```ignore
/// #[test]
/// fn my_parity_test() {
///   $crate::parity_fixtures_or_skip!();
///   // … rest of the test reads tests/parity/fixtures/…
/// }
/// ```
///
/// Expands to an early `return` from the calling fn when the
/// fixtures are absent. Prints a one-line skip note to stderr so
/// `cargo test --nocapture` makes the skip visible.
#[macro_export]
macro_rules! parity_fixtures_or_skip {
  () => {{
    if $crate::test_util::parity_fixtures_root().is_none() {
      ::std::eprintln!(
        "[parity-skip] tests/parity/fixtures/ not shipped in this build \
         (likely the published crate tarball — see `[package] exclude` \
         in Cargo.toml); skipping parity test."
      );
      return;
    }
  }};
}
