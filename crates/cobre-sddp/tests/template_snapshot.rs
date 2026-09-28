//! Template snapshot manifest over every committed deck: a compare
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
    let (evap_system, evap_config, evap_hydro_models) =
        common::in_code_studies::parallel_multiblock_evaporation_study();
    vec![
        (
            "in-code/discounted-anticipated".to_string(),
            common::build_setup_in_code(system, &config),
        ),
        (
            "in-code/parallel-multiblock-evaporation".to_string(),
            common::build_setup_in_code_with_models(evap_system, &evap_config, evap_hydro_models),
        ),
    ]
}

/// One sorted line per `(key, group)`: `<key>\t<group>\t<sha256-hex>`.
fn fact_lines(key: &str, setup: &StudySetup) -> Vec<String> {
    template_fact_groups(setup)
        .into_iter()
        .map(|(group, bytes)| format!("{key}\t{group}\t{:x}", Sha256::digest(&bytes)))
        .collect()
}

/// Per-field simulation digest for a fixed deck subset, trained under `HiGHS`
/// only. `lines()` skips training outside `highs`+`slow-tests` via a runtime
/// `cfg!` check, so the digest code below compiles and lints under a
/// CLP-only build too; only the `HighsSolver` import and call are gated on
/// `feature = "highs"`.
mod sim_view {
    use std::collections::BTreeMap;
    use std::path::Path;

    use cobre_sddp::{SimulationScenarioResult, StudySetup};
    #[cfg(feature = "highs")]
    use cobre_solver::highs::HighsSolver;
    use serde_json::{Map, Value};
    use sha2::{Digest, Sha256};

    const SIM_VIEW_DECKS: &[&str] = &[
        "crates/cobre-sddp/tests/fixtures/chronological_storage",
        "crates/cobre-sddp/tests/fixtures/parallel_storage",
        "examples/deterministic/d46-travel-time-chronological",
        "examples/deterministic/d49-travel-time-chronological-arrival",
        "examples/deterministic/d50-travel-time-plain-tributary-confluence",
    ];

    fn put_u64(buf: &mut Vec<u8>, value: u64) {
        buf.extend_from_slice(&value.to_le_bytes());
    }

    fn put_f64_bits(buf: &mut Vec<u8>, value: f64) {
        put_u64(buf, value.to_bits());
    }

    fn cut_digest_bytes(setup: &StudySetup) -> Vec<u8> {
        let mut buf = Vec::new();
        for pool in 0..setup.fcf.pools.len() {
            for (_slot, intercept, coefficients) in setup.fcf.active_cuts(pool) {
                put_u64(&mut buf, pool as u64);
                put_u64(&mut buf, coefficients.len() as u64);
                put_f64_bits(&mut buf, intercept);
                for &c in coefficients {
                    put_f64_bits(&mut buf, c);
                }
            }
        }
        buf
    }

    fn put_leaf(buf: &mut Vec<u8>, value: &Value) {
        match value {
            Value::Null => buf.push(0),
            Value::Bool(b) => {
                buf.push(1);
                buf.push(u8::from(*b));
            }
            Value::Number(n) => {
                if let Some(u) = n.as_u64() {
                    buf.push(2);
                    put_u64(buf, u);
                } else if let Some(i) = n.as_i64() {
                    buf.push(3);
                    buf.extend_from_slice(&i.to_le_bytes());
                } else {
                    buf.push(4);
                    put_f64_bits(
                        buf,
                        n.as_f64()
                            .expect("a JSON number that is neither u64 nor i64 must be an f64"),
                    );
                }
            }
            Value::String(s) => {
                buf.push(5);
                put_u64(buf, s.len() as u64);
                buf.extend_from_slice(s.as_bytes());
            }
            Value::Array(items) => {
                buf.push(6);
                put_u64(buf, items.len() as u64);
                for item in items {
                    put_leaf(buf, item);
                }
            }
            Value::Object(_) => {
                unreachable!("walk() routes every object through its own record branch")
            }
        }
    }

    /// A JSON object nested under a field becomes a record of that field's key;
    /// an array of objects becomes one record per element (`stage` for
    /// `stages`, else the field key); every other value is a leaf appended to
    /// `sim.<record_type>.<field>`. An empty array contributes nothing.
    fn walk(record_type: &str, obj: &Map<String, Value>, groups: &mut BTreeMap<String, Vec<u8>>) {
        for (key, value) in obj {
            match value {
                Value::Object(child) => walk(key, child, groups),
                Value::Array(items) if matches!(items.first(), Some(Value::Object(_))) => {
                    let child_type = if key == "stages" {
                        "stage"
                    } else {
                        key.as_str()
                    };
                    for item in items {
                        if let Value::Object(child) = item {
                            walk(child_type, child, groups);
                        }
                    }
                }
                Value::Array(items) if items.is_empty() => {}
                leaf => put_leaf(
                    groups
                        .entry(format!("sim.{record_type}.{key}"))
                        .or_default(),
                    leaf,
                ),
            }
        }
    }

    fn sim_digest_lines(
        deck_key: &str,
        setup: &StudySetup,
        mut results: Vec<SimulationScenarioResult>,
    ) -> Vec<String> {
        let mut groups: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        groups.insert("sim.cuts".to_string(), cut_digest_bytes(setup));

        results.sort_by_key(|r| r.scenario_id);
        for scenario in &mut results {
            scenario.stages.sort_by_key(|s| s.stage_id);
            let Value::Object(obj) =
                serde_json::to_value(&*scenario).expect("SimulationScenarioResult must serialize")
            else {
                panic!("SimulationScenarioResult must serialize to a JSON object");
            };
            walk("scenario", &obj, &mut groups);
        }

        groups
            .into_iter()
            .map(|(group, bytes)| format!("{deck_key}\t{group}\t{:x}", Sha256::digest(&bytes)))
            .collect()
    }

    #[cfg(feature = "highs")]
    fn train_and_simulate(dir: &Path) -> (StudySetup, Vec<SimulationScenarioResult>) {
        super::common::parity_hash::train_and_simulate_at_dir(dir, HighsSolver::new)
    }

    #[cfg(not(feature = "highs"))]
    fn train_and_simulate(_dir: &Path) -> (StudySetup, Vec<SimulationScenarioResult>) {
        unreachable!("lines() only calls train_and_simulate under feature = \"highs\"")
    }

    pub(super) fn lines() -> Vec<String> {
        if !cfg!(all(feature = "highs", feature = "slow-tests")) {
            return Vec::new();
        }
        super::common::decks::committed_decks()
            .into_iter()
            .filter(|deck| SIM_VIEW_DECKS.contains(&deck.key.as_str()))
            .flat_map(|deck| {
                let (setup, results) = train_and_simulate(&deck.dir);
                sim_digest_lines(&deck.key, &setup, results)
            })
            .collect()
    }
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
        .chain(sim_view::lines())
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
    let sim_view_enabled = cfg!(all(feature = "highs", feature = "slow-tests"));

    let manifest_text =
        std::fs::read_to_string(manifest_path()).expect("read template snapshot manifest");
    let committed_lines: Vec<String> = manifest_text.lines().map(str::to_string).collect();
    let committed: BTreeMap<(String, String), String> = parse_lines(&committed_lines)
        .into_iter()
        .filter(|((deck_key, group), _)| {
            (slow_tests_enabled || !SLOW_DECKS.contains(&deck_key.as_str()))
                && (sim_view_enabled || !group.starts_with("sim."))
        })
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

/// The parallel-multiblock-evaporation fixture's stage 0 is a 3-block
/// parallel stage with active evaporation.
#[test]
fn parallel_evaporation_fixture_evaporates_on_a_multiblock_parallel_stage() {
    let (system, config, hydro_models) =
        common::in_code_studies::parallel_multiblock_evaporation_study();
    let setup = common::build_setup_in_code_with_models(system, &config, hydro_models);

    let geometry = &setup.stage_data.stage_templates.geometry_per_stage[0];
    assert_eq!(geometry.block_mode, cobre_core::BlockMode::Parallel);
    assert_eq!(geometry.n_blks, 3);
    assert!(
        !geometry.evap_hydro_indices.is_empty(),
        "stage 0 must have an active evaporation slot"
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

#[cfg(feature = "highs")]
fn require_highs() {}

#[cfg(not(feature = "highs"))]
fn require_highs() {
    panic!(
        "template_snapshot_regen must run with --features highs so the \
         manifest's simulation-view lines are always regenerated"
    );
}

#[test]
#[ignore = "rewrites the committed template snapshot manifest; run explicitly"]
fn template_snapshot_regen() {
    require_slow_tests();
    require_highs();

    let lines = manifest_lines(&committed_decks());
    let mut content = lines.join("\n");
    content.push('\n');

    let path = manifest_path();
    let tmp_path = path.with_extension("tsv.tmp");
    std::fs::write(&tmp_path, content).expect("write temporary manifest");
    std::fs::rename(&tmp_path, &path).expect("rename temporary manifest into place");
}

#[test]
fn every_deck_workspace_pool_is_sized_from_its_owners() {
    use cobre_comm::LocalBackend;
    use cobre_solver::ActiveSolver;

    for deck in active_decks() {
        let setup = build_deck_or_panic(&deck);
        let stage_ctx = setup.stage_ctx();
        let training_ctx = setup.training_ctx();
        let state = training_ctx.state;
        let max_n_blks = stage_ctx
            .geometry_per_stage
            .iter()
            .map(|g| g.n_blks)
            .max()
            .unwrap_or(0);

        let comm = LocalBackend;
        let pool = setup
            .create_workspace_pool(&comm, 1, ActiveSolver::new)
            .unwrap_or_else(|e| panic!("deck {}: workspace pool: {e:?}", deck.key));

        for ws in &pool.workspaces {
            assert_eq!(
                ws.patch_buf.indices.len(),
                stage_ctx.load_bus_indices.len() * max_n_blks + state.hydro_count,
                "deck {}: patch_buf.indices length",
                deck.key
            );
            assert_eq!(
                ws.patch_buf.col_indices.len(),
                state.hydro_count * (1 + state.max_par_order)
                    + state.n_buckets
                    + state.n_anticipated * state.k_max,
                "deck {}: patch_buf.col_indices length",
                deck.key
            );
            assert!(
                ws.current_state.capacity() >= state.n_state,
                "deck {}: current_state capacity",
                deck.key
            );
        }
    }
}
