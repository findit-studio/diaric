//! Cross-component cluster tests for `cluster_offline` per spec §9.

use super::*;
use crate::cluster::test_util::perturbed_unit;

#[test]
fn agglomerative_average_matches_two_groups() {
  let mut e = Vec::new();
  for s in [0.0, 0.05, -0.05] {
    e.push(perturbed_unit(0, s));
  }
  for s in [0.0, 0.05, -0.05] {
    e.push(perturbed_unit(10, s));
  }
  let labels = cluster_offline(
    &e,
    &OfflineClusterOptions::default().with_method(OfflineMethod::Agglomerative {
      linkage: Linkage::Average,
    }),
  )
  .unwrap();
  // First three indices share a label, last three share another, and the
  // two groups have different labels.
  assert_eq!(labels[0], labels[1]);
  assert_eq!(labels[1], labels[2]);
  assert_eq!(labels[3], labels[4]);
  assert_eq!(labels[4], labels[5]);
  assert_ne!(
    labels[0], labels[3],
    "two well-separated groups must end up in different clusters"
  );
}
