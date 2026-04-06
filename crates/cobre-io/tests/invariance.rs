//! Declaration-order invariance tests for `cobre_io::load_case`.
//!
//! Verifies that the full `load_case` pipeline produces bit-for-bit identical
//! [`System`] values regardless of the order in which entities are declared in
//! the input JSON files.  This is the core "declaration-order invariance"
//! guarantee required by the Cobre architecture: all entity collections are
//! stored in canonical ID-sorted order, so shuffling the input arrays must
//! produce an equal result.
//!
//! Each test creates two [`TempDir`] case directories -- one with canonical
//! ordering, one with reversed entity arrays -- loads both, and asserts
//! equality.
#![allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::too_many_lines,
    clippy::doc_markdown
)]

mod helpers;

use cobre_core::EntityId;
use cobre_io::load_case;
use tempfile::TempDir;

// ── make_shuffled_multi_entity_case ───────────────────────────────────────────

/// Populate `dir` with the same logical data as [`helpers::make_multi_entity_case`]
/// but with entity arrays in reversed ID order:
///
/// - `system/buses.json`: `[id=2, id=1]` instead of `[id=1, id=2]`
/// - `stages.json`: stages array `[id=1, id=0]` instead of `[id=0, id=1]`
///
/// Lines, hydros, and thermals each contain a single entity so reversing
/// them is a no-op -- only buses (2 entities) and stages (2 entities) are
/// meaningfully reordered.
fn make_shuffled_multi_entity_case(dir: &TempDir) {
    let root = dir.path();

    helpers::write_file(root, "config.json", helpers::VALID_CONFIG_JSON);
    helpers::write_file(root, "penalties.json", helpers::VALID_PENALTIES_JSON);

    // Two-stage finite horizon with stages in REVERSED order (id=1 before id=0).
    helpers::write_file(
        root,
        "stages.json",
        r#"{
    "policy_graph": {
        "type": "finite_horizon",
        "annual_discount_rate": 0.06,
        "transitions": [
            { "source_id": 0, "target_id": 1, "probability": 1.0 }
        ]
    },
    "scenario_source": { "seed": 42 },
    "stages": [
        {
            "id": 1,
            "start_date": "2024-02-01",
            "end_date": "2024-03-01",
            "blocks": [{ "id": 0, "name": "FLAT", "hours": 672.0 }],
            "num_scenarios": 10
        },
        {
            "id": 0,
            "start_date": "2024-01-01",
            "end_date": "2024-02-01",
            "blocks": [{ "id": 0, "name": "FLAT", "hours": 744.0 }],
            "num_scenarios": 10
        }
    ]
}"#,
    );

    helpers::write_file(
        root,
        "initial_conditions.json",
        helpers::VALID_INITIAL_CONDITIONS_JSON,
    );

    // Two buses in REVERSED order: id=2 first, id=1 second.
    helpers::write_file(
        root,
        "system/buses.json",
        r#"{
    "buses": [
        { "id": 2, "name": "BUS_S" },
        { "id": 1, "name": "BUS_SE" }
    ]
}"#,
    );

    // One transmission line (single entity -- order is irrelevant).
    helpers::write_file(
        root,
        "system/lines.json",
        r#"{
    "lines": [
        {
            "id": 1,
            "name": "SE-S",
            "source_bus_id": 1,
            "target_bus_id": 2,
            "capacity": { "direct_mw": 2000.0, "reverse_mw": 1500.0 }
        }
    ]
}"#,
    );

    // One hydro plant (single entity -- order is irrelevant).
    helpers::write_file(
        root,
        "system/hydros.json",
        r#"{
    "hydros": [
        {
            "id": 1,
            "name": "HYDRO_1",
            "bus_id": 1,
            "downstream_id": null,
            "reservoir": { "min_storage_hm3": 0.0, "max_storage_hm3": 1000.0 },
            "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": null },
            "generation": {
                "model": "constant_productivity",
                "productivity_mw_per_m3s": 1.0,
                "min_turbined_m3s": 0.0,
                "max_turbined_m3s": 200.0,
                "min_generation_mw": 0.0,
                "max_generation_mw": 200.0
            }
        }
    ]
}"#,
    );

    // One thermal plant (single entity -- order is irrelevant).
    helpers::write_file(
        root,
        "system/thermals.json",
        r#"{
    "thermals": [
        {
            "id": 1,
            "name": "THERMAL_1",
            "bus_id": 2,
            "cost_segments": [
                { "capacity_mw": 300.0, "cost_per_mwh": 80.0 }
            ],
            "generation": { "min_mw": 0.0, "max_mw": 300.0 }
        }
    ]
}"#,
    );
}

// ── test_bus_ordering_invariance ──────────────────────────────────────────────

/// Given a case with buses ordered `[id=1, id=2]` and a shuffled case with
/// buses ordered `[id=2, id=1]`, `load_case` must return equal Systems.
///
/// This verifies that the parser sorts buses into canonical ID order
/// regardless of their position in the JSON array.
#[test]
fn test_bus_ordering_invariance() {
    let canonical_dir = TempDir::new().unwrap();
    helpers::make_multi_entity_case(&canonical_dir);

    let shuffled_dir = TempDir::new().unwrap();
    make_shuffled_multi_entity_case(&shuffled_dir);

    let canonical = load_case(canonical_dir.path())
        .unwrap_or_else(|e| panic!("canonical case should load successfully, got: {e}"));

    let shuffled = load_case(shuffled_dir.path())
        .unwrap_or_else(|e| panic!("shuffled case should load successfully, got: {e}"));

    // Both Systems must have the same entity counts.
    assert_eq!(
        canonical.n_buses(),
        shuffled.n_buses(),
        "bus count must match between canonical and shuffled cases"
    );

    // O(1) lookup by ID must succeed in both cases and return the same name.
    let canonical_bus1 = canonical
        .bus(EntityId(1))
        .unwrap_or_else(|| panic!("bus id=1 must exist in canonical case"));
    let shuffled_bus1 = shuffled
        .bus(EntityId(1))
        .unwrap_or_else(|| panic!("bus id=1 must exist in shuffled case"));

    assert_eq!(
        canonical_bus1.name, shuffled_bus1.name,
        "bus id=1 name must be identical in canonical and shuffled cases"
    );

    let canonical_bus2 = canonical
        .bus(EntityId(2))
        .unwrap_or_else(|| panic!("bus id=2 must exist in canonical case"));
    let shuffled_bus2 = shuffled
        .bus(EntityId(2))
        .unwrap_or_else(|| panic!("bus id=2 must exist in shuffled case"));

    assert_eq!(
        canonical_bus2.name, shuffled_bus2.name,
        "bus id=2 name must be identical in canonical and shuffled cases"
    );

    // Full System equality via PartialEq (skips HashMap indices, compares sorted vecs).
    assert_eq!(
        canonical, shuffled,
        "Systems from canonical and shuffled bus orderings must be equal"
    );
}

// ── test_stage_ordering_invariance ────────────────────────────────────────────

/// Given a case with stages ordered `[id=0, id=1]` and a shuffled case with
/// stages ordered `[id=1, id=0]`, `load_case` must return Systems with stages
/// in the same canonical order `[id=0, id=1]` for both.
///
/// This verifies that the temporal parser sorts stages by ID so that
/// `system.stages()[0].id == 0` regardless of declaration order.
#[test]
fn test_stage_ordering_invariance() {
    let canonical_dir = TempDir::new().unwrap();
    helpers::make_multi_entity_case(&canonical_dir);

    let shuffled_dir = TempDir::new().unwrap();
    make_shuffled_multi_entity_case(&shuffled_dir);

    let canonical = load_case(canonical_dir.path())
        .unwrap_or_else(|e| panic!("canonical case should load successfully, got: {e}"));

    let shuffled = load_case(shuffled_dir.path())
        .unwrap_or_else(|e| panic!("shuffled case should load successfully, got: {e}"));

    assert_eq!(
        canonical.n_stages(),
        shuffled.n_stages(),
        "stage count must match between canonical and shuffled cases"
    );

    // Canonical order requires stages[0].id == 0 in both cases.
    let canonical_first = canonical
        .stages()
        .first()
        .unwrap_or_else(|| panic!("canonical case must have at least one stage"));
    let shuffled_first = shuffled
        .stages()
        .first()
        .unwrap_or_else(|| panic!("shuffled case must have at least one stage"));

    assert_eq!(
        canonical_first.id, 0,
        "canonical case: stages must be sorted so stages[0].id == 0"
    );
    assert_eq!(
        shuffled_first.id, 0,
        "shuffled case: stages must be sorted so stages[0].id == 0 even when declared in reversed order"
    );

    assert_eq!(
        canonical_first.id, shuffled_first.id,
        "first stage id must be the same in both canonical and shuffled cases"
    );
}

// ── test_full_case_ordering_invariance ────────────────────────────────────────

/// Given a full multi-entity case with all arrays reversed simultaneously
/// (buses and stages), `load_case` must produce Systems that are equal via `==`.
///
/// This is the comprehensive invariance test: it reverses ALL entity arrays
/// that have more than one element and verifies both the full structural
/// equality and selected entity-level spot checks.
#[test]
fn test_full_case_ordering_invariance() {
    let canonical_dir = TempDir::new().unwrap();
    helpers::make_multi_entity_case(&canonical_dir);

    let shuffled_dir = TempDir::new().unwrap();
    make_shuffled_multi_entity_case(&shuffled_dir);

    let canonical = load_case(canonical_dir.path())
        .unwrap_or_else(|e| panic!("canonical case should load successfully, got: {e}"));

    let shuffled = load_case(shuffled_dir.path())
        .unwrap_or_else(|e| panic!("shuffled case should load successfully, got: {e}"));

    // Full structural equality via PartialEq.
    assert_eq!(
        canonical, shuffled,
        "Systems built from canonical and fully-shuffled input must be structurally equal"
    );

    // Spot-check: bus id=1 name must be identical.
    let canonical_bus1 = canonical
        .bus(EntityId(1))
        .unwrap_or_else(|| panic!("bus id=1 must exist in canonical case"));
    let shuffled_bus1 = shuffled
        .bus(EntityId(1))
        .unwrap_or_else(|| panic!("bus id=1 must exist in shuffled case"));

    assert_eq!(
        canonical_bus1.name, shuffled_bus1.name,
        "bus(EntityId(1)).name must be identical regardless of declaration order"
    );

    // Spot-check: entity counts are preserved.
    assert_eq!(
        canonical.n_hydros(),
        shuffled.n_hydros(),
        "hydro count must match between canonical and shuffled cases"
    );
    assert_eq!(
        canonical.n_stages(),
        shuffled.n_stages(),
        "stage count must match between canonical and shuffled cases"
    );
}
