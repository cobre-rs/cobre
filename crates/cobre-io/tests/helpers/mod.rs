//! Test helper functions for integration tests.
//!
//! Provides fixture builders for creating case directories in a [`TempDir`],
//! following the same patterns as the unit test helpers in
//! `src/validation/schema.rs`.
//!
//! # Design
//!
//! All fixture helpers write JSON files programmatically — no binary fixtures
//! are committed to the repository. Parquet fixtures are constructed with
//! `ArrowWriter` when needed.
#![allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::too_many_lines,
    clippy::doc_markdown,
    dead_code
)]

use std::fs;
use std::path::Path;
use tempfile::TempDir;

// ── JSON content constants ────────────────────────────────────────────────────
//
// These replicate the VALID_*_JSON constants from `src/validation/schema.rs`
// tests.  They cannot be imported from that module (private test constants),
// so they are redefined here.

/// Minimal valid `config.json`.
pub const VALID_CONFIG_JSON: &str = r#"{
    "training": {
        "forward_passes": 10,
        "stopping_rules": [
            { "type": "iteration_limit", "limit": 100 }
        ]
    }
}"#;

/// Minimal valid `penalties.json` with all required top-level sections.
pub const VALID_PENALTIES_JSON: &str = r#"{
    "bus": {
        "deficit_segments": [
            { "depth_mw": 500.0, "cost": 1000.0 },
            { "depth_mw": null,  "cost": 5000.0 }
        ],
        "excess_cost": 100.0
    },
    "line": { "exchange_cost": 2.0 },
    "hydro": {
        "spillage_cost": 0.01,
        "fpha_turbined_cost": 0.05,
        "diversion_cost": 0.1,
        "storage_violation_below_cost": 10000.0,
        "filling_target_violation_cost": 50000.0,
        "turbined_violation_below_cost": 500.0,
        "outflow_violation_below_cost": 500.0,
        "outflow_violation_above_cost": 500.0,
        "generation_violation_below_cost": 1000.0,
        "evaporation_violation_cost": 5000.0,
        "water_withdrawal_violation_cost": 1000.0
    },
    "non_controllable_source": { "curtailment_cost": 0.005 }
}"#;

/// Single-stage finite-horizon `stages.json` with no transitions.
const VALID_STAGES_JSON: &str = r#"{
    "policy_graph": {
        "type": "finite_horizon",
        "annual_discount_rate": 0.06,
        "transitions": []
    },
    "scenario_source": { "sampling_scheme": "in_sample", "seed": 42 },
    "stages": [
        {
            "id": 0,
            "start_date": "2024-01-01",
            "end_date": "2024-02-01",
            "blocks": [{ "id": 0, "name": "FLAT", "hours": 744.0 }],
            "num_scenarios": 50
        }
    ]
}"#;

/// Minimal valid `initial_conditions.json`.
pub const VALID_INITIAL_CONDITIONS_JSON: &str = r#"{
    "storage": [],
    "filling_storage": []
}"#;

/// Single-bus `buses.json`.
const VALID_BUSES_JSON: &str = r#"{ "buses": [{ "id": 1, "name": "BUS_1" }] }"#;

/// Empty lines array.
const VALID_LINES_JSON: &str = r#"{ "lines": [] }"#;

/// Empty hydros array.
const VALID_HYDROS_JSON: &str = r#"{ "hydros": [] }"#;

/// Empty thermals array.
const VALID_THERMALS_JSON: &str = r#"{ "thermals": [] }"#;

// ── write_file ────────────────────────────────────────────────────────────────

/// Write `content` to `root.join(relative)`, creating all parent directories.
///
/// Mirrors the same helper in `src/validation/schema.rs` tests.
pub fn write_file(root: &Path, relative: &str, content: &str) {
    let full = root.join(relative);
    if let Some(parent) = full.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(&full, content).unwrap();
}

// ── make_minimal_case ─────────────────────────────────────────────────────────

/// Populate `dir` with the 8 required JSON files for a minimal valid case.
///
/// The case has:
/// - 1 bus (id=1, name="BUS\_1")
/// - 0 lines, 0 hydros, 0 thermals
/// - 1 study stage (id=0, 744 h flat block, 50 scenarios)
/// - A finite-horizon policy graph with no transitions
pub fn make_minimal_case(dir: &TempDir) {
    let root = dir.path();
    write_file(root, "config.json", VALID_CONFIG_JSON);
    write_file(root, "penalties.json", VALID_PENALTIES_JSON);
    write_file(root, "stages.json", VALID_STAGES_JSON);
    write_file(
        root,
        "initial_conditions.json",
        VALID_INITIAL_CONDITIONS_JSON,
    );
    write_file(root, "system/buses.json", VALID_BUSES_JSON);
    write_file(root, "system/lines.json", VALID_LINES_JSON);
    write_file(root, "system/hydros.json", VALID_HYDROS_JSON);
    write_file(root, "system/thermals.json", VALID_THERMALS_JSON);
}

// ── make_multi_entity_case ────────────────────────────────────────────────────

/// Populate `dir` with a richer 8-file case: 2 buses, 1 hydro, 1 thermal,
/// 1 line, 2 study stages with a transition.
///
/// Cross-references:
/// - Hydro id=1 -> bus_id=1
/// - Thermal id=1 -> bus_id=2
/// - Line id=1: source_bus_id=1, target_bus_id=2
///
/// All references are valid: both buses exist in the registry.
pub fn make_multi_entity_case(dir: &TempDir) {
    let root = dir.path();

    write_file(root, "config.json", VALID_CONFIG_JSON);
    write_file(root, "penalties.json", VALID_PENALTIES_JSON);

    // Two-stage finite horizon with a single deterministic transition.
    write_file(
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
    "scenario_source": { "sampling_scheme": "in_sample", "seed": 42 },
    "stages": [
        {
            "id": 0,
            "start_date": "2024-01-01",
            "end_date": "2024-02-01",
            "blocks": [{ "id": 0, "name": "FLAT", "hours": 744.0 }],
            "num_scenarios": 10
        },
        {
            "id": 1,
            "start_date": "2024-02-01",
            "end_date": "2024-03-01",
            "blocks": [{ "id": 0, "name": "FLAT", "hours": 672.0 }],
            "num_scenarios": 10
        }
    ]
}"#,
    );

    write_file(
        root,
        "initial_conditions.json",
        VALID_INITIAL_CONDITIONS_JSON,
    );

    // Two buses.
    write_file(
        root,
        "system/buses.json",
        r#"{
    "buses": [
        { "id": 1, "name": "BUS_SE" },
        { "id": 2, "name": "BUS_S" }
    ]
}"#,
    );

    // One transmission line connecting the two buses.
    write_file(
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

    // One hydro plant injecting into bus_id=1.
    write_file(
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

    // One thermal plant injecting into bus_id=2.
    write_file(
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

// ── make_referential_violation_case ───────────────────────────────────────────

/// Populate `dir` with a case that has a referential integrity violation:
/// a hydro plant whose `bus_id` (999) does not exist in the bus registry.
///
/// This is `make_multi_entity_case` with the hydro's `bus_id` overridden to 999.
pub fn make_referential_violation_case(dir: &TempDir) {
    // Start with the full multi-entity layout.
    make_multi_entity_case(dir);

    // Overwrite hydros.json with an invalid bus_id reference.
    write_file(
        dir.path(),
        "system/hydros.json",
        r#"{
    "hydros": [
        {
            "id": 1,
            "name": "HYDRO_1",
            "bus_id": 999,
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
}
