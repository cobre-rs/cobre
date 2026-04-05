//! End-to-end integration tests for `cobre_io::load_case`.
//!
//! Each test constructs a complete case directory in a [`TempDir`] using the
//! helpers in [`helpers`], calls [`load_case`], and verifies either the
//! returned [`System`] entity counts or the returned [`LoadError`] variant.
//!
//! These tests exercise the full five-layer validation pipeline and the
//! `SystemBuilder` assembly step in one shot.
#![allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::too_many_lines,
    clippy::doc_markdown
)]

mod helpers;

use cobre_core::EntityId;
use cobre_io::{deserialize_system, load_case, serialize_system};
use tempfile::TempDir;

// ── test_minimal_valid_case ────────────────────────────────────────────────────

/// Given the 8 required files for a minimal case, `load_case` must return an
/// `Ok(System)` with exactly 1 bus, 0 hydros, 0 thermals, 0 lines, and 1 stage.
#[test]
fn test_minimal_valid_case() {
    let dir = TempDir::new().unwrap();
    helpers::make_minimal_case(&dir);

    let result = load_case(dir.path());

    let system = match result {
        Ok(s) => s,
        Err(e) => panic!("expected Ok(System) for minimal case, got Err: {e}"),
    };

    assert_eq!(
        system.n_buses(),
        1,
        "minimal case should have exactly 1 bus"
    );
    assert_eq!(system.n_hydros(), 0, "minimal case should have 0 hydros");
    assert_eq!(
        system.n_thermals(),
        0,
        "minimal case should have 0 thermals"
    );
    assert_eq!(system.n_lines(), 0, "minimal case should have 0 lines");
    assert_eq!(
        system.n_stages(),
        1,
        "minimal case should have exactly 1 stage"
    );
    assert!(
        system.bus(EntityId(1)).is_some(),
        "bus with id=1 should be found by O(1) lookup"
    );
}

// ── test_multi_entity_case ────────────────────────────────────────────────────

/// Given a richer case with 2 buses, 1 hydro, 1 thermal, 1 line, and 2 stages,
/// `load_case` must return `Ok(System)` with matching entity counts.
#[test]
fn test_multi_entity_case() {
    let dir = TempDir::new().unwrap();
    helpers::make_multi_entity_case(&dir);

    let result = load_case(dir.path());

    let system = match result {
        Ok(s) => s,
        Err(e) => panic!("expected Ok(System) for multi-entity case, got Err: {e}"),
    };

    assert_eq!(
        system.n_buses(),
        2,
        "multi-entity case should have exactly 2 buses"
    );
    assert_eq!(
        system.n_hydros(),
        1,
        "multi-entity case should have exactly 1 hydro"
    );
    assert_eq!(
        system.n_thermals(),
        1,
        "multi-entity case should have exactly 1 thermal"
    );
    assert_eq!(
        system.n_lines(),
        1,
        "multi-entity case should have exactly 1 line"
    );
    assert_eq!(
        system.n_stages(),
        2,
        "multi-entity case should have exactly 2 stages"
    );
    assert!(
        system.bus(EntityId(1)).is_some(),
        "bus with id=1 should be accessible via O(1) lookup"
    );
    assert!(
        system.bus(EntityId(2)).is_some(),
        "bus with id=2 should be accessible via O(1) lookup"
    );
}

// ── test_missing_required_file ────────────────────────────────────────────────

/// Given a minimal case with `system/buses.json` removed, `load_case` must
/// return an `Err` whose display representation contains `"buses"`.
#[test]
fn test_missing_required_file() {
    let dir = TempDir::new().unwrap();
    helpers::make_minimal_case(&dir);

    // Remove the required buses file after populating the full case.
    std::fs::remove_file(dir.path().join("system/buses.json")).unwrap();

    let result = load_case(dir.path());

    match result {
        Err(err) => {
            let display = err.to_string();
            assert!(
                display.contains("buses"),
                "error display should mention 'buses', got: {display}"
            );
        }
        Ok(_) => panic!("expected Err when buses.json is missing, got Ok"),
    }
}

// ── test_malformed_json ───────────────────────────────────────────────────────

/// Given a minimal case with `system/hydros.json` containing invalid JSON,
/// `load_case` must return an `Err` (parse failure).
#[test]
fn test_malformed_json() {
    let dir = TempDir::new().unwrap();
    helpers::make_minimal_case(&dir);

    // Overwrite hydros.json with syntactically invalid content.
    helpers::write_file(
        dir.path(),
        "system/hydros.json",
        "{ this is not valid json }",
    );

    let result = load_case(dir.path());

    match result {
        Err(err) => {
            // The error must be a parse or constraint error — not an Ok.
            // We only assert it is an Err; the exact variant is implementation-
            // defined (ParseError or ConstraintError wrapping the parse failure).
            let display = err.to_string();
            assert!(
                !display.is_empty(),
                "error display should be non-empty for malformed JSON"
            );
        }
        Ok(_) => panic!("expected Err for malformed hydros.json, got Ok"),
    }
}

// ── test_referential_integrity_violation ─────────────────────────────────────

/// Given a case where a hydro references a non-existent bus (id=999),
/// `load_case` must return an `Err` whose display representation mentions
/// the invalid reference.
#[test]
fn test_referential_integrity_violation() {
    let dir = TempDir::new().unwrap();
    helpers::make_referential_violation_case(&dir);

    let result = load_case(dir.path());

    match result {
        Err(err) => {
            let display = err.to_string();
            // The error description must mention that a reference is missing —
            // the structural format from ValidationContext::into_result is:
            // "[InvalidReference] system/hydros.json (Hydro 1): ... 999 ..."
            assert!(
                display.contains("999") || display.contains("bus") || display.contains("Bus"),
                "error display should mention the invalid bus reference (999), got: {display}"
            );
        }
        Ok(_) => panic!("expected Err when hydro references non-existent bus_id=999, got Ok"),
    }
}

// ── test_inflow_history_wired_into_system ─────────────────────────────────────

/// Given a case directory with `scenarios/inflow_history.parquet` containing
/// 1 hydro x 10 years of monthly data (120 rows), `load_case` must return a
/// `System` whose `inflow_history()` slice has exactly 120 entries, all with
/// finite `value_m3s`.
///
/// The case also includes `inflow_seasonal_stats.parquet` (one row per stage for
/// the single hydro) so that the estimation path is bypassed and no
/// `season_definitions` are required in `stages.json`.
#[test]
fn test_inflow_history_wired_into_system() {
    use arrow::array::{Date32Array, Float64Array, Int32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use chrono::NaiveDate;
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;

    let dir = TempDir::new().unwrap();
    helpers::make_multi_entity_case(&dir);

    // Overwrite hydros.json with 3 hydros (all on bus_id=1).
    std::fs::write(
        dir.path().join("system/hydros.json"),
        r#"{ "hydros": [
            { "id": 1, "name": "H1", "bus_id": 1, "downstream_id": null,
              "reservoir": { "min_storage_hm3": 0.0, "max_storage_hm3": 1000.0 },
              "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": null },
              "generation": { "model": "constant_productivity", "productivity_mw_per_m3s": 1.0,
                "min_turbined_m3s": 0.0, "max_turbined_m3s": 200.0,
                "min_generation_mw": 0.0, "max_generation_mw": 200.0 } },
            { "id": 2, "name": "H2", "bus_id": 1, "downstream_id": null,
              "reservoir": { "min_storage_hm3": 0.0, "max_storage_hm3": 500.0 },
              "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": null },
              "generation": { "model": "constant_productivity", "productivity_mw_per_m3s": 1.0,
                "min_turbined_m3s": 0.0, "max_turbined_m3s": 100.0,
                "min_generation_mw": 0.0, "max_generation_mw": 100.0 } },
            { "id": 3, "name": "H3", "bus_id": 1, "downstream_id": null,
              "reservoir": { "min_storage_hm3": 0.0, "max_storage_hm3": 300.0 },
              "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": null },
              "generation": { "model": "constant_productivity", "productivity_mw_per_m3s": 1.0,
                "min_turbined_m3s": 0.0, "max_turbined_m3s": 80.0,
                "min_generation_mw": 0.0, "max_generation_mw": 80.0 } }
        ] }"#,
    )
    .unwrap();

    std::fs::create_dir_all(dir.path().join("scenarios")).unwrap();

    // ── Write inflow_seasonal_stats.parquet (3 hydros × 2 stages) ────────────
    {
        let stats_schema = Arc::new(Schema::new(vec![
            Field::new("hydro_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, false),
            Field::new("mean_m3s", DataType::Float64, false),
            Field::new("std_m3s", DataType::Float64, false),
        ]));
        let stats_batch = RecordBatch::try_new(
            Arc::clone(&stats_schema),
            vec![
                Arc::new(Int32Array::from(vec![1, 1, 2, 2, 3, 3])),
                Arc::new(Int32Array::from(vec![0, 1, 0, 1, 0, 1])),
                Arc::new(Float64Array::from(vec![
                    150.0, 120.0, 80.0, 70.0, 50.0, 45.0,
                ])),
                Arc::new(Float64Array::from(vec![20.0, 15.0, 10.0, 8.0, 6.0, 5.0])),
            ],
        )
        .unwrap();
        let file =
            std::fs::File::create(dir.path().join("scenarios/inflow_seasonal_stats.parquet"))
                .unwrap();
        let mut writer = ArrowWriter::try_new(file, stats_batch.schema(), None).unwrap();
        writer.write(&stats_batch).unwrap();
        writer.close().unwrap();
    }

    // ── Write inflow_ar_coefficients.parquet (3 hydros × 2 stages, lag=1) ────
    // Both stats AND coefficients must be present to skip estimation.
    {
        let ar_schema = Arc::new(Schema::new(vec![
            Field::new("hydro_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, false),
            Field::new("lag", DataType::Int32, false),
            Field::new("coefficient", DataType::Float64, false),
            Field::new("residual_std_ratio", DataType::Float64, false),
        ]));
        let ar_batch = RecordBatch::try_new(
            Arc::clone(&ar_schema),
            vec![
                Arc::new(Int32Array::from(vec![1, 1, 2, 2, 3, 3])),
                Arc::new(Int32Array::from(vec![0, 1, 0, 1, 0, 1])),
                Arc::new(Int32Array::from(vec![1, 1, 1, 1, 1, 1])),
                Arc::new(Float64Array::from(vec![0.3, 0.25, 0.4, 0.35, 0.2, 0.15])),
                Arc::new(Float64Array::from(vec![0.95, 0.92, 0.90, 0.88, 0.93, 0.91])),
            ],
        )
        .unwrap();
        let file =
            std::fs::File::create(dir.path().join("scenarios/inflow_ar_coefficients.parquet"))
                .unwrap();
        let mut writer = ArrowWriter::try_new(file, ar_batch.schema(), None).unwrap();
        writer.write(&ar_batch).unwrap();
        writer.close().unwrap();
    }

    // ── Write inflow_history.parquet (3 hydros × 10 years × 12 months = 360) ─
    let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
    let mut hydro_ids: Vec<i32> = Vec::with_capacity(360);
    let mut dates: Vec<i32> = Vec::with_capacity(360);
    let mut values: Vec<f64> = Vec::with_capacity(360);

    for hid in 1_i32..=3 {
        for year in 2000_i32..=2009 {
            for month in 1_u32..=12 {
                let date = NaiveDate::from_ymd_opt(year, month, 1).unwrap();
                let days = i32::try_from((date - epoch).num_days()).unwrap();
                hydro_ids.push(hid);
                dates.push(days);
                values.push(f64::from(hid) * 100.0 + f64::from(month));
            }
        }
    }

    let history_schema = Arc::new(Schema::new(vec![
        Field::new("hydro_id", DataType::Int32, false),
        Field::new("date", DataType::Date32, false),
        Field::new("value_m3s", DataType::Float64, false),
    ]));
    let history_batch = RecordBatch::try_new(
        Arc::clone(&history_schema),
        vec![
            Arc::new(Int32Array::from(hydro_ids)),
            Arc::new(Date32Array::from(dates)),
            Arc::new(Float64Array::from(values)),
        ],
    )
    .unwrap();
    let file = std::fs::File::create(dir.path().join("scenarios/inflow_history.parquet")).unwrap();
    let mut writer = ArrowWriter::try_new(file, history_batch.schema(), None).unwrap();
    writer.write(&history_batch).unwrap();
    writer.close().unwrap();

    let system = load_case(dir.path())
        .unwrap_or_else(|e| panic!("load_case failed for inflow_history case: {e}"));

    assert_eq!(
        system.inflow_history().len(),
        360,
        "system.inflow_history() must have 360 rows (3 hydros × 10 years × 12 months)"
    );
    for row in system.inflow_history() {
        assert!(
            row.value_m3s.is_finite(),
            "every inflow_history row must have a finite value_m3s"
        );
    }
}

// ── test_external_scenarios_wired_into_system ─────────────────────────────────

/// Given a case directory with `scenarios/external_inflow_scenarios.parquet` containing
/// 2 stages × 5 scenarios × 3 hydros (30 rows), `load_case` must return a
/// `System` whose `external_scenarios()` slice has exactly 30 entries.
#[test]
fn test_external_scenarios_wired_into_system() {
    use arrow::array::{Float64Array, Int32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;

    let dir = TempDir::new().unwrap();
    helpers::make_multi_entity_case(&dir);

    // Overwrite hydros.json with 3 hydros (all on bus_id=1).
    std::fs::write(
        dir.path().join("system/hydros.json"),
        r#"{ "hydros": [
            { "id": 1, "name": "H1", "bus_id": 1, "downstream_id": null,
              "reservoir": { "min_storage_hm3": 0.0, "max_storage_hm3": 1000.0 },
              "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": null },
              "generation": { "model": "constant_productivity", "productivity_mw_per_m3s": 1.0,
                "min_turbined_m3s": 0.0, "max_turbined_m3s": 200.0,
                "min_generation_mw": 0.0, "max_generation_mw": 200.0 } },
            { "id": 2, "name": "H2", "bus_id": 1, "downstream_id": null,
              "reservoir": { "min_storage_hm3": 0.0, "max_storage_hm3": 500.0 },
              "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": null },
              "generation": { "model": "constant_productivity", "productivity_mw_per_m3s": 1.0,
                "min_turbined_m3s": 0.0, "max_turbined_m3s": 100.0,
                "min_generation_mw": 0.0, "max_generation_mw": 100.0 } },
            { "id": 3, "name": "H3", "bus_id": 1, "downstream_id": null,
              "reservoir": { "min_storage_hm3": 0.0, "max_storage_hm3": 300.0 },
              "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": null },
              "generation": { "model": "constant_productivity", "productivity_mw_per_m3s": 1.0,
                "min_turbined_m3s": 0.0, "max_turbined_m3s": 80.0,
                "min_generation_mw": 0.0, "max_generation_mw": 80.0 } }
        ] }"#,
    )
    .unwrap();

    // Build 2 stages × 5 scenarios × 3 hydros = 30 rows.
    let mut stage_ids: Vec<i32> = Vec::with_capacity(30);
    let mut scenario_ids: Vec<i32> = Vec::with_capacity(30);
    let mut hydro_ids: Vec<i32> = Vec::with_capacity(30);
    let mut values: Vec<f64> = Vec::with_capacity(30);

    for stage_id in 0_i32..2 {
        for scenario_id in 0_i32..5 {
            for hid in 1_i32..=3 {
                stage_ids.push(stage_id);
                scenario_ids.push(scenario_id);
                hydro_ids.push(hid);
                values.push(
                    f64::from(stage_id) * 1000.0 + f64::from(scenario_id) * 10.0 + f64::from(hid),
                );
            }
        }
    }

    let schema = Arc::new(Schema::new(vec![
        Field::new("stage_id", DataType::Int32, false),
        Field::new("scenario_id", DataType::Int32, false),
        Field::new("hydro_id", DataType::Int32, false),
        Field::new("value_m3s", DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(stage_ids)),
            Arc::new(Int32Array::from(scenario_ids)),
            Arc::new(Int32Array::from(hydro_ids)),
            Arc::new(Float64Array::from(values)),
        ],
    )
    .unwrap();

    std::fs::create_dir_all(dir.path().join("scenarios")).unwrap();
    let file = std::fs::File::create(
        dir.path()
            .join("scenarios/external_inflow_scenarios.parquet"),
    )
    .unwrap();
    let mut writer = ArrowWriter::try_new(file, batch.schema(), None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();

    let system = load_case(dir.path())
        .unwrap_or_else(|e| panic!("load_case failed for external_inflow_scenarios case: {e}"));

    assert_eq!(
        system.external_scenarios().len(),
        30,
        "system.external_scenarios() must have 30 rows (2 stages × 5 scenarios × 3 hydros)"
    );
    for row in system.external_scenarios() {
        assert!(
            row.value_m3s.is_finite(),
            "every external_scenarios row must have a finite value_m3s"
        );
    }
}

// ── test_inflow_history_absent_returns_empty ──────────────────────────────────

/// Given a case directory without `scenarios/inflow_history.parquet`, `load_case`
/// must return a `System` whose `inflow_history()` is an empty slice.
#[test]
fn test_inflow_history_absent_returns_empty() {
    let dir = TempDir::new().unwrap();
    helpers::make_minimal_case(&dir);

    // No inflow_history.parquet is written — absence is the default.
    let system =
        load_case(dir.path()).unwrap_or_else(|e| panic!("load_case failed for minimal case: {e}"));

    assert!(
        system.inflow_history().is_empty(),
        "system.inflow_history() must be empty when inflow_history.parquet is absent"
    );
}

// ── test_external_scenarios_absent_returns_empty ──────────────────────────────

/// Given a case directory without `scenarios/external_inflow_scenarios.parquet`, `load_case`
/// must return a `System` whose `external_scenarios()` is an empty slice.
#[test]
fn test_external_scenarios_absent_returns_empty() {
    let dir = TempDir::new().unwrap();
    helpers::make_minimal_case(&dir);

    // No external_inflow_scenarios.parquet is written — absence is the default.
    let system =
        load_case(dir.path()).unwrap_or_else(|e| panic!("load_case failed for minimal case: {e}"));

    assert!(
        system.external_scenarios().is_empty(),
        "system.external_scenarios() must be empty when external_inflow_scenarios.parquet is absent"
    );
}

// ── test_postcard_round_trip ──────────────────────────────────────────────────

/// Given a System produced by `load_case`, serializing it with `serialize_system`
/// and deserializing with `deserialize_system` must produce a System with the
/// same entity counts and working O(1) lookups.
#[test]
fn test_postcard_round_trip() {
    let dir = TempDir::new().unwrap();
    helpers::make_minimal_case(&dir);

    let original = load_case(dir.path())
        .unwrap_or_else(|e| panic!("load_case should succeed for minimal case, got: {e}"));

    let bytes = serialize_system(&original)
        .unwrap_or_else(|e| panic!("serialize_system should succeed, got: {e}"));

    assert!(!bytes.is_empty(), "serialized bytes should be non-empty");

    let deserialized = deserialize_system(&bytes)
        .unwrap_or_else(|e| panic!("deserialize_system should succeed, got: {e}"));

    assert_eq!(
        deserialized.n_buses(),
        original.n_buses(),
        "bus count must match after postcard round-trip"
    );
    assert!(
        deserialized.bus(EntityId(1)).is_some(),
        "O(1) bus lookup must work after index rebuild on deserialized System"
    );

    // Verify no data was lost for other entity types.
    assert_eq!(
        deserialized.n_hydros(),
        original.n_hydros(),
        "hydro count must match after round-trip"
    );
    assert_eq!(
        deserialized.n_thermals(),
        original.n_thermals(),
        "thermal count must match after round-trip"
    );
    assert_eq!(
        deserialized.n_lines(),
        original.n_lines(),
        "line count must match after round-trip"
    );
    assert_eq!(
        deserialized.n_stages(),
        original.n_stages(),
        "stage count must match after round-trip"
    );
}
