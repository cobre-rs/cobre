//! Declaration-order permutation invariance of every committed deck's
//! stage-LP template facts: the permanent successor to the build-twice check,
//! subsuming it by running in-process over the full fact set
//! ([`template_fact_groups`]) instead of one seeded pair per golden case.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use cobre_sddp::test_support::template_fact_groups;

use common::decks::{SLOW_DECKS, committed_decks};
use common::fresh_setup_with;
use common::permute::permute_case;

const SEED: u64 = 0x5EED_C0BE_5EED_C0BE;

#[test]
fn every_deck_template_is_invariant_to_declaration_order() {
    let slow_tests_enabled = cfg!(feature = "slow-tests");
    let mut mismatches: Vec<(String, String)> = Vec::new();

    for deck in committed_decks() {
        if !slow_tests_enabled && SLOW_DECKS.contains(&deck.key.as_str()) {
            continue;
        }

        let base = template_fact_groups(&fresh_setup_with(&deck.dir, |_| {}));
        let permuted_dir = permute_case(&deck.dir, SEED);
        let permuted = template_fact_groups(&fresh_setup_with(permuted_dir.path(), |_| {}));

        assert!(
            base.keys().eq(permuted.keys()),
            "deck {}: permuted fact-group key set differs from base (base={:?}, permuted={:?})",
            deck.key,
            base.keys().collect::<Vec<_>>(),
            permuted.keys().collect::<Vec<_>>(),
        );

        for (group, base_bytes) in &base {
            if permuted.get(group) != Some(base_bytes) {
                mismatches.push((deck.key.clone(), (*group).to_string()));
            }
        }
    }

    assert!(
        mismatches.is_empty(),
        "template facts differ under declaration-order permutation for (deck, group) pairs:\n{}",
        mismatches
            .iter()
            .map(|(deck, group)| format!("  {deck}\t{group}"))
            .collect::<Vec<_>>()
            .join("\n")
    );
}
