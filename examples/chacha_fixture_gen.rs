//! One-shot generator to populate FIXTURES in tests/chacha_keystream_fixture.rs.
//!
//! Usage: `cargo run --example chacha_fixture_gen`
//! Output: paste into FIXTURES, replacing the PLACEHOLDER lines.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

fn main() {
  for seed in [0u64, 42, 0xDEAD_BEEF] {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let vals: Vec<String> = (0..8)
      .map(|_| format!("0x{:016x}", rng.next_u64()))
      .collect();
    println!("(0x{:x}, [", seed);
    for v in vals {
      println!("    {},", v);
    }
    println!("]),");
  }
}
