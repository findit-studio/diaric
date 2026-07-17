//! Regression fixture for `rand_chacha::ChaCha8Rng` keystream stability.
//!
//! `dia`'s public determinism contract (spec §11.9) commits us to bit-exact
//! cluster labels for a given `OfflineClusterOptions::seed`. That contract
//! depends on `ChaCha8Rng::seed_from_u64(seed).next_u64()` producing the
//! same byte sequence across versions of `rand_chacha`. This test pins
//! the first 8 `next_u64()` outputs for three seeds.
//!
//! If this test ever fails after a `cargo update`, the keystream changed
//! and we need to either (a) pin `rand_chacha` to the prior compatible
//! version, or (b) bump `dia` to a major version (per §11.9 policy).
//!
//! To regenerate FIXTURES intentionally (e.g., on a planned major-version bump),
//! run `cargo run --release --example chacha_fixture_gen` and paste the output
//! into the FIXTURES array, replacing each `(seed, [...])` block.
//!
//! Note: `rand_chacha`'s std feature affects `OsRng`, not `ChaCha8Rng`'s
//! keystream — the keystream is identical with and without `std`. So a
//! single test covers both feature configurations.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

const FIXTURES: &[(u64, [u64; 8])] = &[
  // (seed, [next_u64() × 8])
  // Generated with rand_chacha = "0.10" (default-features = false).
  // See spec §15 #52 for re-generation procedure if cipher is intentionally bumped.
  (
    0,
    [
      0xb585f767a79a3b6c,
      0x7746a55fbad8c037,
      0xb2fb0d3281e2a6e6,
      0x0f6760a48f9b887c,
      0xe10d666732024679,
      0x8cae14cb947eb0bd,
      0xd438539d6a2e923c,
      0xef781c7dd2d368ba,
    ],
  ),
  (
    42,
    [
      0xae90bfb5395d5ba1,
      0xf3453fc625799188,
      0x6d71b708c5b6538c,
      0xa09ab2f958166752,
      0x49e149d8bcb642b0,
      0x2663b45ba45d829e,
      0x4edbbf0150871314,
      0xcdca9b0d2a122884,
    ],
  ),
  (
    0xDEAD_BEEF,
    [
      0xff01307f43ec8df9,
      0x946b5cc52dc1b3db,
      0x017ff25ec6284944,
      0x408827c5ef521b39,
      0xad405c58500ab5ce,
      0x07dee5d6817b87ff,
      0xe3f4da5d913c5820,
      0x73e790c1503561d5,
    ],
  ),
];

#[test]
fn chacha8_keystream_byte_fixture() {
  for (seed, expected) in FIXTURES {
    let mut rng = ChaCha8Rng::seed_from_u64(*seed);
    let actual: [u64; 8] = std::array::from_fn(|_| rng.next_u64());
    assert_eq!(
      &actual, expected,
      "ChaCha8Rng keystream changed for seed {:#x}: actual={:?} expected={:?}\n\
             If intentional (rand_chacha cipher bump), regenerate FIXTURES and bump dia major version per §11.9.",
      seed, actual, expected
    );
  }
}
