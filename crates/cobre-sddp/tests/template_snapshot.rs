//! Template snapshot manifest over every committed deck (ADR-041): a compare
//! test that fails on any moved, added, or removed `(deck, group)` line, and
//! an ignored regeneration test that rewrites the manifest.
//!
//! Temporary safety net for the stage-LP builder consolidation: this manifest
//! and its two tests are deleted once the consolidation's last step lands.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::collections::BTreeMap;
use std::path::PathBuf;

use cobre_sddp::StudySetup;
use cobre_sddp::test_support::template_fact_groups;
use sha2::{Digest, Sha256};

use common::decks::{Deck, SLOW_DECKS, committed_decks};

fn manifest_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/template_snapshot_manifest.tsv")
}

/// Build `deck`'s [`StudySetup`], panicking with the deck's key on any
/// build failure — no deck is ever skipped silently.
fn build_deck_or_panic(deck: &Deck) -> StudySetup {
    let dir = deck.dir.clone();
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        common::fresh_setup_with(&dir, |_| {})
    }))
    .unwrap_or_else(|payload| {
        let msg = payload
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| payload.downcast_ref::<&str>().copied())
            .unwrap_or("<non-string panic payload>");
        panic!("deck {} failed to build: {msg}", deck.key);
    })
}

/// In-code fixtures, keyed by their manifest key: each isolates a stage-LP
/// builder axis no committed deck combines. Included in both the compare and
/// the regen tests, never slow-gated.
fn in_code_decks() -> Vec<(String, StudySetup)> {
    let (system, config) = common::in_code_studies::discounted_anticipated_study();
    vec![(
        "in-code/discounted-anticipated".to_string(),
        common::build_setup_in_code(system, &config),
    )]
}

/// One sorted line per `(key, group)`: `<key>\t<group>\t<sha256-hex>`.
fn fact_lines(key: &str, setup: &StudySetup) -> Vec<String> {
    template_fact_groups(setup)
        .into_iter()
        .map(|(group, bytes)| format!("{key}\t{group}\t{:x}", Sha256::digest(&bytes)))
        .collect()
}

fn manifest_lines(decks: &[Deck]) -> Vec<String> {
    let mut lines: Vec<String> = decks
        .iter()
        .flat_map(|deck| fact_lines(&deck.key, &build_deck_or_panic(deck)))
        .chain(
            in_code_decks()
                .into_iter()
                .flat_map(|(key, setup)| fact_lines(&key, &setup)),
        )
        .collect();
    lines.sort();
    lines
}

fn parse_lines(lines: &[String]) -> BTreeMap<(String, String), String> {
    lines
        .iter()
        .map(|line| {
            let mut fields = line.split('\t');
            let deck = fields.next().expect("deck field");
            let group = fields.next().expect("group field");
            let hash = fields.next().expect("hash field");
            assert!(
                fields.next().is_none(),
                "unexpected extra field in line: {line}"
            );
            ((deck.to_string(), group.to_string()), hash.to_string())
        })
        .collect()
}

fn active_decks() -> Vec<Deck> {
    let slow_tests_enabled = cfg!(feature = "slow-tests");
    committed_decks()
        .into_iter()
        .filter(|deck| slow_tests_enabled || !SLOW_DECKS.contains(&deck.key.as_str()))
        .collect()
}

fn format_pairs(pairs: &[(String, String)]) -> String {
    pairs
        .iter()
        .take(200)
        .map(|(deck, group)| format!("  {deck}\t{group}"))
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn template_snapshot_matches_manifest() {
    let slow_tests_enabled = cfg!(feature = "slow-tests");

    let manifest_text =
        std::fs::read_to_string(manifest_path()).expect("read template snapshot manifest");
    let committed_lines: Vec<String> = manifest_text.lines().map(str::to_string).collect();
    let committed: BTreeMap<(String, String), String> = parse_lines(&committed_lines)
        .into_iter()
        .filter(|((deck_key, _), _)| slow_tests_enabled || !SLOW_DECKS.contains(&deck_key.as_str()))
        .collect();

    let computed_lines = manifest_lines(&active_decks());
    let computed = parse_lines(&computed_lines);

    let mut moved = Vec::new();
    let mut added = Vec::new();
    for (key, computed_hash) in &computed {
        match committed.get(key) {
            Some(committed_hash) if committed_hash == computed_hash => {}
            Some(_) => moved.push(key.clone()),
            None => added.push(key.clone()),
        }
    }
    let mut removed: Vec<(String, String)> = committed
        .keys()
        .filter(|key| !computed.contains_key(*key))
        .cloned()
        .collect();

    if moved.is_empty() && added.is_empty() && removed.is_empty() {
        return;
    }

    moved.sort();
    added.sort();
    removed.sort();

    panic!(
        "template snapshot manifest mismatch (moved={}, added={}, removed={}):\n\
         moved:\n{}\nadded:\n{}\nremoved:\n{}",
        moved.len(),
        added.len(),
        removed.len(),
        format_pairs(&moved),
        format_pairs(&added),
        format_pairs(&removed),
    );
}

/// The discounted-anticipated fixture's stage-1 anticipated decision is
/// costed: `LeadStages(2)` on a 4-stage horizon decides stages 0 and 1, both
/// delivering after stage 0, and the decision column's objective coefficient
/// must be non-zero for the discount path to be exercised at all.
#[test]
fn discounted_anticipated_fixture_decides_after_stage_zero() {
    let (system, config) = common::in_code_studies::discounted_anticipated_study();
    let setup = common::build_setup_in_code(system, &config);

    let geometry = &setup.stage_data.stage_templates.geometry_per_stage[1];
    assert!(
        !geometry.anticipated_decision.is_empty(),
        "stage 1 must have an active anticipated-decision column"
    );
    let template = &setup.stage_data.stage_templates.templates[1];
    assert!(
        template.objective[geometry.anticipated_decision.start] > 0.0,
        "stage 1's anticipated decision must carry a nonzero costed objective coefficient"
    );
}

#[cfg(feature = "slow-tests")]
fn require_slow_tests() {}

#[cfg(not(feature = "slow-tests"))]
fn require_slow_tests() {
    panic!(
        "template_snapshot_regen must run with --features slow-tests so the \
         manifest always covers every committed deck, including SLOW_DECKS"
    );
}

#[test]
#[ignore = "rewrites the committed template snapshot manifest; run explicitly"]
fn template_snapshot_regen() {
    require_slow_tests();

    let lines = manifest_lines(&committed_decks());
    let mut content = lines.join("\n");
    content.push('\n');

    let path = manifest_path();
    let tmp_path = path.with_extension("tsv.tmp");
    std::fs::write(&tmp_path, content).expect("write temporary manifest");
    std::fs::rename(&tmp_path, &path).expect("rename temporary manifest into place");
}
