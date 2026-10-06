//! Integration tests for the `cobre run` subcommand: fixtures are built
//! programmatically in temp dirs or point at committed example cases.
//! Shared harness helpers live in `common`.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp
)]

use std::fs;
use std::path::Path;

use arrow::array::{Array, BooleanArray, Float64Array, Int32Array, StringArray};
use assert_cmd::prelude::*;
use cobre_io::scenarios::parse_inflow_annual_component;
use cobre_io::{EntitySlot, StateFamily, deserialize_stage_cuts};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use predicates::prelude::*;
use tempfile::TempDir;

mod common;
use common::{
    case_dir, cobre, copy_dir_recursive, make_valid_case, restamp_policy_version, write_file,
    write_supplied_opening_tree_case,
};

#[test]
fn valid_case_exits_0() {
    let dir = TempDir::new().unwrap();
    make_valid_case(dir.path(), None, None, None, None);
    let out = TempDir::new().unwrap();

    cobre()
        .args([
            "run",
            dir.path().to_str().unwrap(),
            "--output",
            out.path().to_str().unwrap(),
            "--quiet",
        ])
        .assert()
        .success();
}

#[test]
fn valid_case_creates_training_metadata() {
    let dir = TempDir::new().unwrap();
    make_valid_case(dir.path(), None, None, None, None);
    let out = TempDir::new().unwrap();

    cobre()
        .args([
            "run",
            dir.path().to_str().unwrap(),
            "--output",
            out.path().to_str().unwrap(),
            "--quiet",
        ])
        .assert()
        .success();

    assert!(out.path().join("training/metadata.json").is_file());
}

#[test]
fn valid_case_creates_convergence_parquet() {
    let dir = TempDir::new().unwrap();
    make_valid_case(dir.path(), None, None, None, None);
    let out = TempDir::new().unwrap();

    cobre()
        .args([
            "run",
            dir.path().to_str().unwrap(),
            "--output",
            out.path().to_str().unwrap(),
            "--quiet",
        ])
        .assert()
        .success();

    assert!(out.path().join("training/convergence.parquet").is_file());
}

#[test]
fn disabled_simulation_does_not_produce_manifest() {
    let dir = TempDir::new().unwrap();
    make_valid_case(dir.path(), None, None, None, None);
    let out = TempDir::new().unwrap();

    cobre()
        .args([
            "run",
            dir.path().to_str().unwrap(),
            "--output",
            out.path().to_str().unwrap(),
            "--quiet",
        ])
        .assert()
        .success();

    assert!(!out.path().join("simulation/metadata.json").exists());
}

#[test]
fn custom_output_dir_receives_training_artifacts() {
    let dir = TempDir::new().unwrap();
    make_valid_case(dir.path(), None, None, None, None);
    let custom_out = TempDir::new().unwrap();
    assert_ne!(dir.path(), custom_out.path());

    cobre()
        .args([
            "run",
            dir.path().to_str().unwrap(),
            "--output",
            custom_out.path().to_str().unwrap(),
            "--quiet",
        ])
        .assert()
        .success();

    assert!(custom_out.path().join("training/metadata.json").is_file());
    assert!(!dir.path().join("output").exists());
}

#[test]
fn missing_required_file_exits_1() {
    let dir = TempDir::new().unwrap();
    make_valid_case(dir.path(), None, None, None, None);
    fs::remove_file(dir.path().join("system/buses.json")).unwrap();

    cobre()
        .args(["run", dir.path().to_str().unwrap(), "--quiet"])
        .assert()
        .failure()
        .code(1);
}

#[test]
fn missing_required_file_stderr_contains_validation_error() {
    let dir = TempDir::new().unwrap();
    make_valid_case(dir.path(), None, None, None, None);
    fs::remove_file(dir.path().join("system/buses.json")).unwrap();

    cobre()
        .args(["run", dir.path().to_str().unwrap()])
        .assert()
        .failure()
        .code(1)
        .stderr(predicate::str::contains("error"));
}

/// On the `run` path the "run `cobre validate`" hint is non-circular and
/// actionable, so it is kept on stderr — unlike the `validate` path, which
/// suppresses it. Removing it here breaks the run-path UX contract.
#[test]
fn missing_required_file_stderr_contains_report_and_hint() {
    let dir = TempDir::new().unwrap();
    make_valid_case(dir.path(), None, None, None, None);
    fs::remove_file(dir.path().join("system/buses.json")).unwrap();

    cobre()
        .args(["run", dir.path().to_str().unwrap()])
        .assert()
        .failure()
        .code(1)
        .stderr(predicate::str::contains("buses.json"))
        .stderr(predicate::str::contains(
            "run `cobre validate <CASE_DIR>` for a full diagnostic report",
        ));
}

#[test]
fn nonexistent_path_exits_2() {
    cobre()
        .args(["run", "/nonexistent/path/that/does/not/exist", "--quiet"])
        .assert()
        .failure()
        .code(2);
}

#[test]
fn nonexistent_path_stderr_contains_io_error() {
    cobre()
        .args(["run", "/nonexistent/path/that/does/not/exist"])
        .assert()
        .failure()
        .code(2)
        .stderr(predicate::str::contains("I/O error"));
}

#[test]
fn test_run_quiet_suppresses_banner_and_summary() {
    let dir = TempDir::new().unwrap();
    make_valid_case(dir.path(), None, None, None, None);
    let out = TempDir::new().unwrap();
    cobre()
        .args([
            "run",
            dir.path().to_str().unwrap(),
            "--output",
            out.path().to_str().unwrap(),
            "--quiet",
        ])
        .assert()
        .success()
        .stderr(predicate::str::contains("COBRE v").not())
        .stderr(predicate::str::contains("Training complete in").not());
}

const CONFIG_STOCHASTIC_PAR_A_JSON: &str = r#"{
    "training": {
        "selection": { "method": "sampled", "forward_passes": 1 },
        "stopping_rules": [
            { "type": "iteration_limit", "limit": 2 }
        ],
        "scenario_source": { "inflow": { "scheme": "in_sample" }, "seed": 42 }
    },
    "exports": { "stochastic": true },
    "estimation": { "order_selection": "pacf_annual" }
}"#;

/// On a case with no hydros the annual-component file is still written, with zero
/// data rows but a valid Arrow schema.
#[test]
fn cli_run_writes_inflow_annual_component_when_par_a_active() {
    let dir = TempDir::new().unwrap();
    make_valid_case(
        dir.path(),
        Some(CONFIG_STOCHASTIC_PAR_A_JSON),
        None,
        None,
        None,
    );

    let out = TempDir::new().unwrap();

    cobre()
        .args([
            "run",
            dir.path().to_str().unwrap(),
            "--output",
            out.path().to_str().unwrap(),
            "--quiet",
        ])
        .assert()
        .success();

    let parquet_path = out
        .path()
        .join("stochastic/inflow_annual_component.parquet");
    assert!(
        parquet_path.is_file(),
        "stochastic/inflow_annual_component.parquet must exist"
    );

    let rows = parse_inflow_annual_component(&parquet_path).unwrap();
    assert_eq!(
        rows.len(),
        0,
        "expected zero annual component rows for a case with no hydros"
    );
}

// ── Anticipated thermal columns (K=1) ─────────────────────────────────────────

const CONFIG_ANTICIPATED_JSON: &str = r#"{
    "training": {
        "selection": { "method": "sampled", "forward_passes": 1 },
        "stopping_rules": [
            { "type": "iteration_limit", "limit": 2 }
        ],
        "scenario_source": { "inflow": { "scheme": "in_sample" }, "seed": 42 }
    },
    "simulation": { "enabled": true, "selection": { "method": "sampled", "num_scenarios": 1 } }
}"#;

/// Anticipated thermal id=2 has `lead_stages=1`, so its commitment windows
/// must tile exactly one leading delivery stage — the prior commitment before
/// the study start, covering stage 0's `[2024-01-01, 2024-02-01)` span.
const INITIAL_CONDITIONS_ANTICIPATED_JSON: &str = r#"{
    "storage": [],
    "filling_storage": [],
    "past_anticipated_commitments": [
        { "thermal_id": 2, "start_date": "2024-01-01", "end_date": "2024-02-01", "value_mw": 0.0 }
    ]
}"#;

/// Two thermals: id=1 regular, id=2 anticipated with `lead_stages=1`.
/// IDs are ascending (declaration-order invariance rule: anticipated id > regular id).
const THERMALS_ANTICIPATED_JSON: &str = r#"{
    "thermals": [
        {
            "id": 1,
            "name": "REGULAR",
            "operational_start_date": "2024-01-01",
            "bus_id": 1,
            "cost_per_mwh": 30.0,
            "generation": { "min_mw": 0.0, "max_mw": 200.0 }
        },
        {
            "id": 2,
            "name": "ANTICIPATED",
            "operational_start_date": "2024-01-01",
            "bus_id": 1,
            "cost_per_mwh": 25.0,
            "generation": { "min_mw": 0.0, "max_mw": 100.0 },
            "anticipated_config": { "lead_stages": 1 }
        }
    ]
}"#;

struct ThermalRows {
    stage_ids: Vec<i32>,
    thermal_ids: Vec<i32>,
    is_anticipated: Vec<bool>,
    anticipated_committed_mw: Vec<Option<f64>>,
    anticipated_decision_mw: Vec<Option<f64>>,
}

fn read_thermals_parquet(path: &Path) -> ThermalRows {
    let file = fs::File::open(path)
        .unwrap_or_else(|e| panic!("failed to open thermals parquet at {}: {e}", path.display()));

    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .expect("failed to build ParquetRecordBatchReaderBuilder");
    let reader = builder.build().expect("failed to build reader");

    let mut stage_ids: Vec<i32> = Vec::new();
    let mut thermal_ids: Vec<i32> = Vec::new();
    let mut is_anticipated: Vec<bool> = Vec::new();
    let mut anticipated_committed_mw: Vec<Option<f64>> = Vec::new();
    let mut anticipated_decision_mw: Vec<Option<f64>> = Vec::new();

    for batch_result in reader {
        let batch = batch_result.expect("failed to read record batch");
        let schema = batch.schema();

        let stage_col_idx = schema
            .index_of("stage_id")
            .expect("thermals schema must have stage_id column");
        let thermal_col_idx = schema
            .index_of("thermal_id")
            .expect("thermals schema must have thermal_id column");
        let is_ant_col_idx = schema
            .index_of("is_anticipated")
            .expect("thermals schema must have is_anticipated column");
        let committed_col_idx = schema
            .index_of("anticipated_committed_mw")
            .expect("thermals schema must have anticipated_committed_mw column");
        let decision_col_idx = schema
            .index_of("anticipated_decision_mw")
            .expect("thermals schema must have anticipated_decision_mw column");

        let stage_col = batch
            .column(stage_col_idx)
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("stage_id must be Int32Array");
        let thermal_col = batch
            .column(thermal_col_idx)
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("thermal_id must be Int32Array");
        let is_ant_col = batch
            .column(is_ant_col_idx)
            .as_any()
            .downcast_ref::<BooleanArray>()
            .expect("is_anticipated must be BooleanArray");
        let committed_col = batch
            .column(committed_col_idx)
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("anticipated_committed_mw must be Float64Array");
        let decision_col = batch
            .column(decision_col_idx)
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("anticipated_decision_mw must be Float64Array");

        for i in 0..batch.num_rows() {
            stage_ids.push(stage_col.value(i));
            thermal_ids.push(thermal_col.value(i));
            is_anticipated.push(is_ant_col.value(i));
            anticipated_committed_mw.push(if committed_col.is_null(i) {
                None
            } else {
                Some(committed_col.value(i))
            });
            anticipated_decision_mw.push(if decision_col.is_null(i) {
                None
            } else {
                Some(decision_col.value(i))
            });
        }
    }

    ThermalRows {
        stage_ids,
        thermal_ids,
        is_anticipated,
        anticipated_committed_mw,
        anticipated_decision_mw,
    }
}

/// End-to-end run with anticipated thermals on a 2-stage fixture (one regular,
/// one anticipated thermal with `lead_stages=1`).
#[test]
#[allow(clippy::too_many_lines)]
fn cli_run_populates_anticipated_thermal_columns() {
    let regular_id: i32 = 1;
    let anticipated_id: i32 = 2;
    assert!(
        regular_id < anticipated_id,
        "regular_id ({regular_id}) must be less than anticipated_id ({anticipated_id})"
    );

    let tmp = TempDir::new().expect("create tempdir");
    let case = tmp.path().join("case");
    let output = tmp.path().join("output");
    fs::create_dir_all(&case).expect("create case dir");

    make_valid_case(
        &case,
        Some(CONFIG_ANTICIPATED_JSON),
        None,
        Some(INITIAL_CONDITIONS_ANTICIPATED_JSON),
        Some(THERMALS_ANTICIPATED_JSON),
    );

    cobre()
        .args([
            "run",
            case.to_str().expect("case path is valid UTF-8"),
            "--output",
            output.to_str().expect("output path is valid UTF-8"),
            "--threads",
            "1",
        ])
        .assert()
        .success();

    // One parquet per scenario; num_scenarios=1 gives the single scenario_id=0000 file.
    let parquet_path = output.join("simulation/thermals/scenario_id=0000/data.parquet");
    assert!(
        parquet_path.exists(),
        "simulation/thermals/scenario_id=0000/data.parquet must exist at {}",
        parquet_path.display()
    );

    let rows = read_thermals_parquet(&parquet_path);

    assert!(
        !rows.stage_ids.is_empty(),
        "thermals.parquet must contain at least one row"
    );

    for (row_idx, &tid) in rows.thermal_ids.iter().enumerate() {
        if tid != regular_id {
            continue;
        }
        let stage = rows.stage_ids[row_idx];
        assert!(
            !rows.is_anticipated[row_idx],
            "row {row_idx}: regular thermal (id={regular_id}, stage={stage}) must have \
             is_anticipated=false"
        );
        assert!(
            rows.anticipated_decision_mw[row_idx].is_none(),
            "row {row_idx}: regular thermal (id={regular_id}, stage={stage}) must have \
             anticipated_decision_mw=null"
        );
        assert!(
            rows.anticipated_committed_mw[row_idx].is_none(),
            "row {row_idx}: regular thermal (id={regular_id}, stage={stage}) must have \
             anticipated_committed_mw=null"
        );
    }

    let regular_row_count = rows
        .thermal_ids
        .iter()
        .filter(|&&id| id == regular_id)
        .count();
    assert!(
        regular_row_count > 0,
        "no rows found for regular thermal id={regular_id} in thermals.parquet"
    );

    for (row_idx, &tid) in rows.thermal_ids.iter().enumerate() {
        if tid != anticipated_id {
            continue;
        }
        let stage = rows.stage_ids[row_idx];
        assert!(
            rows.is_anticipated[row_idx],
            "row {row_idx}: anticipated thermal (id={anticipated_id}, stage={stage}) must have \
             is_anticipated=true"
        );
    }

    let anticipated_row_count = rows
        .thermal_ids
        .iter()
        .filter(|&&id| id == anticipated_id)
        .count();
    assert!(
        anticipated_row_count > 0,
        "no rows found for anticipated thermal id={anticipated_id} in thermals.parquet"
    );

    let stage_0_ant_rows: Vec<usize> = rows
        .thermal_ids
        .iter()
        .enumerate()
        .filter(|&(i, &tid)| tid == anticipated_id && rows.stage_ids[i] == 0)
        .map(|(i, _)| i)
        .collect();

    assert!(
        !stage_0_ant_rows.is_empty(),
        "no rows found for anticipated thermal id={anticipated_id} at stage 0"
    );

    for &row_idx in &stage_0_ant_rows {
        let decision = rows.anticipated_decision_mw[row_idx];
        assert!(
            decision.is_some(),
            "row {row_idx}: anticipated thermal at stage 0 must have non-null \
             anticipated_decision_mw"
        );
        let v = decision.unwrap();
        assert!(
            v >= 0.0 && v.is_finite(),
            "row {row_idx}: anticipated_decision_mw at stage 0 must be >= 0.0 and finite, \
             got {v}"
        );
        let committed = rows.anticipated_committed_mw[row_idx];
        assert!(
            committed.is_some(),
            "row {row_idx}: anticipated thermal at stage 0 under always-active fishing \
             must have non-null anticipated_committed_mw (reads slot 0)"
        );
        let c = committed.unwrap();
        assert!(
            c >= 0.0 && c.is_finite(),
            "row {row_idx}: anticipated_committed_mw at stage 0 must be >= 0.0 and finite, got {c}"
        );
    }

    let stage_1_ant_rows: Vec<usize> = rows
        .thermal_ids
        .iter()
        .enumerate()
        .filter(|&(i, &tid)| tid == anticipated_id && rows.stage_ids[i] == 1)
        .map(|(i, _)| i)
        .collect();

    assert!(
        !stage_1_ant_rows.is_empty(),
        "no rows found for anticipated thermal id={anticipated_id} at stage 1"
    );

    for &row_idx in &stage_1_ant_rows {
        let committed = rows.anticipated_committed_mw[row_idx];
        assert!(
            committed.is_some(),
            "row {row_idx}: anticipated thermal at stage 1 must have non-null \
             anticipated_committed_mw (matured delivery for K=1)"
        );
        let v = committed.unwrap();
        assert!(
            v.is_finite() && (0.0..=100.0).contains(&v),
            "row {row_idx}: anticipated_committed_mw at stage 1 must be finite and in \
             [0.0, 100.0] (the plant's generation envelope), got {v}"
        );
    }

    // Ring-buffer transport invariant: the stage-0 decision must equal the
    // stage-1 committed value (single plant/scenario/block — no aggregation).
    // Guards the simulation-shift contract (pipeline must shift the ring buffer).
    assert_eq!(
        stage_0_ant_rows.len(),
        stage_1_ant_rows.len(),
        "stage-0 anticipated rows must pair 1:1 with stage-1 rows in a single-scenario, \
         single-block fixture"
    );
    for (&stage_0_idx, &stage_1_idx) in stage_0_ant_rows.iter().zip(&stage_1_ant_rows) {
        let decision = rows.anticipated_decision_mw[stage_0_idx]
            .expect("stage-0 decision must be Some (asserted above)");
        let committed = rows.anticipated_committed_mw[stage_1_idx]
            .expect("stage-1 committed must be Some (asserted above)");
        // IEEE 754 +0.0/-0.0 are equivalent and either may surface from solver
        // vertex selection at an inactive slot; treat equal at zero, else require
        // bit equality.
        let zero_equiv =
            decision == 0.0 && committed == 0.0 && decision.is_finite() && committed.is_finite();
        assert!(
            zero_equiv || decision.to_bits() == committed.to_bits(),
            "ring-buffer transport: stage-0 decision {decision} must equal stage-1 \
             committed {committed} bit-for-bit (the simulation-shift contract)"
        );
    }
}

// ── Anticipated thermal columns (K=2) ─────────────────────────────────────────

/// 3-stage finite horizon. Anticipated thermal with `lead_stages=2` places
/// a decision at stage 0 that matures at stage 2; at stage 1 (horizon boundary:
/// t + `K_i` = 1 + 2 = 3 = `n_stages`) the decision is still active.
const STAGES_ANTICIPATED_K2_JSON: &str = r#"{
    "policy_graph": {
        "type": "finite_horizon",
        "annual_discount_rate": 0.06,
        "transitions": []
    },
    "stages": [
        {
            "id": 0,
            "start_date": "2024-01-01",
            "end_date": "2024-02-01",
            "blocks": [{ "id": 0, "name": "FLAT", "hours": 744.0 }],
            "num_openings": 2
        },
        {
            "id": 1,
            "start_date": "2024-02-01",
            "end_date": "2024-03-01",
            "blocks": [{ "id": 0, "name": "FLAT", "hours": 672.0 }],
            "num_openings": 2
        },
        {
            "id": 2,
            "start_date": "2024-03-01",
            "end_date": "2024-04-01",
            "blocks": [{ "id": 0, "name": "FLAT", "hours": 744.0 }],
            "num_openings": 2
        }
    ]
}"#;

/// Anticipated thermal id=2 has `lead_stages=2`, so its commitment windows
/// must tile exactly two leading delivery stages — the prior commitments
/// before the study start, covering stage 0's `[2024-01-01, 2024-02-01)` and
/// stage 1's `[2024-02-01, 2024-03-01)`. Coverage (`StageCalendar::coverage`)
/// is computed on each stage's own real calendar span, not its declared
/// block-hours, so the windows mirror `STAGES_ANTICIPATED_K2_JSON`'s
/// `start_date`/`end_date` exactly (stage 1's real span is 29 days despite its
/// 672 declared hours).
const INITIAL_CONDITIONS_ANTICIPATED_K2_JSON: &str = r#"{
    "storage": [],
    "filling_storage": [],
    "past_anticipated_commitments": [
        { "thermal_id": 2, "start_date": "2024-01-01", "end_date": "2024-02-01", "value_mw": 0.0 },
        { "thermal_id": 2, "start_date": "2024-02-01", "end_date": "2024-03-01", "value_mw": 0.0 }
    ]
}"#;

/// Two thermals: id=1 regular, id=2 anticipated with `lead_stages=2`.
/// IDs are ascending (declaration-order invariance rule: anticipated id > regular id).
const THERMALS_ANTICIPATED_K2_JSON: &str = r#"{
    "thermals": [
        {
            "id": 1,
            "name": "REGULAR",
            "operational_start_date": "2024-01-01",
            "bus_id": 1,
            "cost_per_mwh": 30.0,
            "generation": { "min_mw": 0.0, "max_mw": 200.0 }
        },
        {
            "id": 2,
            "name": "ANTICIPATED",
            "operational_start_date": "2024-01-01",
            "bus_id": 1,
            "cost_per_mwh": 25.0,
            "generation": { "min_mw": 0.0, "max_mw": 100.0 },
            "anticipated_config": { "lead_stages": 2 }
        }
    ]
}"#;

/// End-to-end run with K=2 anticipated thermals on a 3-stage fixture (one
/// regular, one anticipated thermal with `lead_stages=2`).
#[test]
#[allow(clippy::too_many_lines)]
fn cli_run_k2_populates_anticipated_columns_and_manifest() {
    let regular_id: i32 = 1;
    let anticipated_id: i32 = 2;
    assert!(regular_id < anticipated_id);

    let tmp = TempDir::new().expect("create tempdir");
    let case = tmp.path().join("case");
    let output = tmp.path().join("output");
    fs::create_dir_all(&case).expect("create case dir");

    make_valid_case(
        &case,
        Some(CONFIG_ANTICIPATED_JSON),
        Some(STAGES_ANTICIPATED_K2_JSON),
        Some(INITIAL_CONDITIONS_ANTICIPATED_K2_JSON),
        Some(THERMALS_ANTICIPATED_K2_JSON),
    );

    cobre()
        .args([
            "run",
            case.to_str().expect("case path is valid UTF-8"),
            "--output",
            output.to_str().expect("output path is valid UTF-8"),
            "--threads",
            "1",
        ])
        .assert()
        .success();

    // One parquet per scenario; num_scenarios=1 gives the single scenario_id=0000 file.
    let parquet_path = output.join("simulation/thermals/scenario_id=0000/data.parquet");
    assert!(
        parquet_path.exists(),
        "simulation/thermals/scenario_id=0000/data.parquet must exist at {}",
        parquet_path.display()
    );

    let rows = read_thermals_parquet(&parquet_path);

    assert!(
        !rows.stage_ids.is_empty(),
        "thermals.parquet must contain at least one row"
    );

    for (row_idx, &tid) in rows.thermal_ids.iter().enumerate() {
        if tid != regular_id {
            continue;
        }
        let stage = rows.stage_ids[row_idx];
        assert!(
            !rows.is_anticipated[row_idx],
            "row {row_idx}: regular thermal (id={regular_id}, stage={stage}) must have \
             is_anticipated=false"
        );
        assert!(
            rows.anticipated_decision_mw[row_idx].is_none(),
            "row {row_idx}: regular thermal (id={regular_id}, stage={stage}) must have \
             anticipated_decision_mw=null"
        );
        assert!(
            rows.anticipated_committed_mw[row_idx].is_none(),
            "row {row_idx}: regular thermal (id={regular_id}, stage={stage}) must have \
             anticipated_committed_mw=null"
        );
    }

    let regular_row_count = rows
        .thermal_ids
        .iter()
        .filter(|&&id| id == regular_id)
        .count();
    assert!(
        regular_row_count > 0,
        "no rows found for regular thermal id={regular_id} in thermals.parquet"
    );

    for (row_idx, &tid) in rows.thermal_ids.iter().enumerate() {
        if tid != anticipated_id {
            continue;
        }
        let stage = rows.stage_ids[row_idx];
        assert!(
            rows.is_anticipated[row_idx],
            "row {row_idx}: anticipated thermal (id={anticipated_id}, stage={stage}) must have \
             is_anticipated=true"
        );
    }

    let anticipated_row_count = rows
        .thermal_ids
        .iter()
        .filter(|&&id| id == anticipated_id)
        .count();
    assert!(
        anticipated_row_count > 0,
        "no rows found for anticipated thermal id={anticipated_id} in thermals.parquet"
    );

    let stage_0_ant_rows: Vec<usize> = rows
        .thermal_ids
        .iter()
        .enumerate()
        .filter(|&(i, &tid)| tid == anticipated_id && rows.stage_ids[i] == 0)
        .map(|(i, _)| i)
        .collect();

    assert!(
        !stage_0_ant_rows.is_empty(),
        "no rows found for anticipated thermal id={anticipated_id} at stage 0"
    );

    for &row_idx in &stage_0_ant_rows {
        let decision = rows.anticipated_decision_mw[row_idx];
        assert!(
            decision.is_some(),
            "row {row_idx}: anticipated thermal at stage 0 must have non-null \
             anticipated_decision_mw"
        );
        let v = decision.unwrap();
        assert!(
            v >= 0.0 && v.is_finite(),
            "row {row_idx}: anticipated_decision_mw at stage 0 must be >= 0.0 and finite, \
             got {v}"
        );
        let committed = rows.anticipated_committed_mw[row_idx];
        assert!(
            committed.is_some(),
            "row {row_idx}: anticipated thermal at stage 0 under always-active fishing \
             must have non-null anticipated_committed_mw (reads slot 0 of the ring buffer)"
        );
        let c = committed.unwrap();
        assert!(
            c >= 0.0 && c.is_finite(),
            "row {row_idx}: anticipated_committed_mw at stage 0 must be >= 0.0 and finite, got {c}"
        );
    }

    let stage_1_ant_rows: Vec<usize> = rows
        .thermal_ids
        .iter()
        .enumerate()
        .filter(|&(i, &tid)| tid == anticipated_id && rows.stage_ids[i] == 1)
        .map(|(i, _)| i)
        .collect();

    assert!(
        !stage_1_ant_rows.is_empty(),
        "no rows found for anticipated thermal id={anticipated_id} at stage 1"
    );

    for &row_idx in &stage_1_ant_rows {
        assert!(
            rows.anticipated_decision_mw[row_idx].is_none(),
            "row {row_idx}: anticipated thermal at stage 1 must have \
             anticipated_decision_mw=null (horizon-boundary inactive: \
             t + K_i = 3 >= n_stages = 3 under the strict predicate)"
        );
        let committed = rows.anticipated_committed_mw[row_idx];
        assert!(
            committed.is_some(),
            "row {row_idx}: anticipated thermal at stage 1 under always-active fishing \
             must have non-null anticipated_committed_mw (reads slot 0 of the ring buffer \
             regardless of K_i vs stage_index)"
        );
        let c = committed.unwrap();
        assert!(
            c >= 0.0 && c.is_finite(),
            "row {row_idx}: anticipated_committed_mw at stage 1 must be >= 0.0 and finite, got {c}"
        );
    }

    let stage_2_ant_rows: Vec<usize> = rows
        .thermal_ids
        .iter()
        .enumerate()
        .filter(|&(i, &tid)| tid == anticipated_id && rows.stage_ids[i] == 2)
        .map(|(i, _)| i)
        .collect();

    assert!(
        !stage_2_ant_rows.is_empty(),
        "no rows found for anticipated thermal id={anticipated_id} at stage 2"
    );

    for &row_idx in &stage_2_ant_rows {
        assert!(
            rows.anticipated_decision_mw[row_idx].is_none(),
            "row {row_idx}: anticipated thermal at stage 2 must have anticipated_decision_mw=null \
             (t + K_i = 4 > n_stages=3)"
        );
        let committed = rows.anticipated_committed_mw[row_idx];
        assert!(
            committed.is_some(),
            "row {row_idx}: anticipated thermal at stage 2 must have non-null \
             anticipated_committed_mw (delivery: K_i=2 <= stage_index=2)"
        );
        let v = committed.unwrap();
        assert!(
            v >= 0.0 && v.is_finite(),
            "row {row_idx}: anticipated_committed_mw at stage 2 must be >= 0.0 and finite, \
             got {v}"
        );
    }

    let cuts_path = output.join("policy/cuts/000.bin");
    assert!(
        cuts_path.exists(),
        "policy/cuts/000.bin must exist at {}",
        cuts_path.display()
    );

    let cuts_bytes = fs::read(&cuts_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", cuts_path.display()));
    let stage_cuts = deserialize_stage_cuts(&cuts_bytes)
        .unwrap_or_else(|e| panic!("failed to decode {}: {e:?}", cuts_path.display()));

    let anticipated_slots: Vec<&EntitySlot> = stage_cuts
        .entity_manifest
        .iter()
        .filter(|slot| slot.entity_type == StateFamily::AnticipatedThermalState.code())
        .collect();

    assert_eq!(
        anticipated_slots.len(),
        2,
        "stage 0 manifest must contain exactly 2 anticipated ring slots \
         (K_max=2, n_anticipated=1), found {}",
        anticipated_slots.len()
    );

    for slot in &anticipated_slots {
        assert_eq!(
            slot.entity_id, anticipated_id,
            "each anticipated slot must reference entity_id={anticipated_id}, got {}",
            slot.entity_id
        );
    }

    let ring_subindices: std::collections::HashSet<u32> =
        anticipated_slots.iter().map(|slot| slot.subindex).collect();
    assert_eq!(
        ring_subindices,
        [0u32, 1u32]
            .into_iter()
            .collect::<std::collections::HashSet<u32>>(),
        "anticipated slots must cover ring subindex values {{0, 1}}, found {ring_subindices:?}"
    );

    // Plant 2 has no commissioning window and K_i = K_max = 2, so both ring
    // slots are active at stage 0.
    for slot in &anticipated_slots {
        assert!(
            slot.was_active,
            "anticipated slot (subindex {}) must be active at stage 0",
            slot.subindex
        );
    }
}

// ── Evaporation model rows ─────────────────────────────────────────────────────

/// The CLI writes `hydro_models/evaporation_models.parquet` whose coefficient
/// rows equal the resolved evaporation models. Because the CLI and Python write
/// sites both serialize the identical `cobre_sddp::build_evaporation_model_rows`
/// output, asserting the written file matches that builder output establishes
/// CLI⇄Python file parity without launching a Python interpreter (the builder is
/// the single source both consume).
#[test]
fn cli_writes_evaporation_models_matching_resolver() {
    let case = case_dir("deterministic/d08-evaporation");
    assert!(
        case.join("config.json").is_file(),
        "D08 fixture must exist at {}",
        case.display()
    );

    // Temp output dir so the committed `output/` tree is untouched.
    let out = TempDir::new().expect("create temp output dir");

    cobre()
        .args([
            "run",
            case.to_str().expect("D08 path is valid UTF-8"),
            "--output",
            out.path().to_str().expect("temp path is valid UTF-8"),
            "--quiet",
        ])
        .assert()
        .success();

    let evaporation_path = out
        .path()
        .join("hydro_models")
        .join("evaporation_models.parquet");
    assert!(
        evaporation_path.is_file(),
        "evaporation_models.parquet must exist at {} (D08 models evaporation)",
        evaporation_path.display()
    );

    // The reader returns rows sorted by `(hydro_id, stage_id)`, matching the
    // builder's canonical order — the row-for-row zip below depends on it.
    let written =
        cobre_io::parse_evaporation_models(&evaporation_path).expect("parse evaporation_models");
    assert!(
        !written.is_empty(),
        "D08 must produce at least one evaporation row"
    );

    let system = cobre_io::load_case(&case).expect("load D08 system");
    let result = cobre_sddp::prepare_hydro_models(&system, &case, false)
        .expect("prepare hydro models for D08");
    let expected = cobre_sddp::build_evaporation_model_rows(&result, &system);

    assert_eq!(
        written.len(),
        expected.len(),
        "written row count must equal the resolver row count"
    );

    // One evaporation hydro × two study stages = two per-stage rows.
    assert_eq!(
        written.len(),
        2,
        "D08 (one evaporation hydro, two study stages) must yield two per-stage rows"
    );

    for (got, want) in written.iter().zip(&expected) {
        assert_eq!(got.hydro_id, want.hydro_id, "hydro_id must match");
        assert_eq!(got.stage_id, want.stage_id, "stage_id must match");
        assert_eq!(
            got.intercept_m3s.to_bits(),
            want.intercept_m3s.to_bits(),
            "intercept_m3s must be bit-identical to the resolver output"
        );
        assert_eq!(
            got.volume_slope_m3s_per_hm3.to_bits(),
            want.volume_slope_m3s_per_hm3.to_bits(),
            "volume_slope_m3s_per_hm3 must be bit-identical to the resolver output"
        );
        assert_eq!(
            got.reference_volume_hm3.to_bits(),
            want.reference_volume_hm3.to_bits(),
            "reference_volume_hm3 must be bit-identical to the resolver output"
        );
        assert_eq!(got.source, want.source, "source tag must match");
    }

    assert_eq!(written[0].stage_id, Some(0), "first row tags stage 0");
    assert_eq!(written[1].stage_id, Some(1), "second row tags stage 1");
}

// ── Generic constraint resolved-echo sidecar ──────────────────────────────────

fn run_case(case: &Path, out: &Path) {
    cobre()
        .args([
            "run",
            case.to_str().expect("case path is valid UTF-8"),
            "--output",
            out.to_str().expect("temp path is valid UTF-8"),
            "--quiet",
        ])
        .assert()
        .success();
}

/// `cobre run` emits `generic_constraints/resolved_echo.parquet` for a study
/// with generic constraints; the echo is a training-side sidecar carrying the
/// 13-column echo schema and the resolved interval a reader can compare
/// against the deck.
#[test]
fn cli_writes_generic_constraint_echo_for_d13() {
    let case = case_dir("deterministic/d13-generic-constraint");
    assert!(
        case.join("config.json").is_file(),
        "d13 fixture must exist at {}",
        case.display()
    );

    let out = TempDir::new().expect("create temp output dir");
    run_case(&case, out.path());

    let echo_path = out
        .path()
        .join("generic_constraints")
        .join("resolved_echo.parquet");
    assert!(
        echo_path.is_file(),
        "resolved_echo.parquet must exist at {} (d13 declares a generic constraint)",
        echo_path.display()
    );

    let file = fs::File::open(&echo_path).expect("open resolved_echo.parquet");
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).expect("build parquet reader");
    assert_eq!(
        builder.schema().fields().len(),
        13,
        "echo must carry the 13-column schema"
    );
    let reader = builder.build().expect("build reader");

    let mut saw_cap_at_ten = false;
    let mut total_rows = 0usize;
    for batch_result in reader {
        let batch = batch_result.expect("read record batch");
        total_rows += batch.num_rows();
        let schema = batch.schema();

        let shape_col = batch
            .column(
                schema
                    .index_of("derived_shape")
                    .expect("derived_shape column"),
            )
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("derived_shape must be Utf8");
        let upper_col = batch
            .column(schema.index_of("bound_upper").expect("bound_upper column"))
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("bound_upper must be Float64");

        for i in 0..batch.num_rows() {
            if shape_col.value(i) == "cap" && !upper_col.is_null(i) && upper_col.value(i) == 10.0 {
                saw_cap_at_ten = true;
            }
        }
    }

    assert!(total_rows > 0, "d13 echo must contain at least one row");
    assert!(
        saw_cap_at_ten,
        "d13's `thermal_generation(0) <= 10` must echo a `cap` row with bound_upper = 10.0"
    );
}

#[test]
fn cli_writes_no_echo_without_generic_constraints() {
    let case = case_dir("deterministic/d01-thermal-dispatch");
    assert!(
        case.join("config.json").is_file(),
        "d01 fixture must exist at {}",
        case.display()
    );

    let out = TempDir::new().expect("create temp output dir");
    run_case(&case, out.path());

    assert!(
        !out.path().join("generic_constraints").exists(),
        "a study with no generic constraints must write no generic_constraints/ directory"
    );
}

// ── Deterministic end-to-end golden (examples/1dtoy) ──────────────────────────

fn run_ok(args: &[&str]) -> (String, String) {
    let output = cobre().args(args).assert().success().get_output().clone();
    (
        String::from_utf8(output.stdout).unwrap(),
        String::from_utf8(output.stderr).unwrap(),
    )
}

fn assert_ordered(haystack: &str, needle_a: &str, needle_b: &str) {
    let pos_a = haystack
        .find(needle_a)
        .unwrap_or_else(|| panic!("expected to find {needle_a:?} in stderr"));
    let pos_b = haystack
        .find(needle_b)
        .unwrap_or_else(|| panic!("expected to find {needle_b:?} in stderr"));
    assert!(
        pos_a < pos_b,
        "expected {needle_a:?} (at {pos_a}) before {needle_b:?} (at {pos_b})"
    );
}

fn find_line<'a>(haystack: &'a str, label: &str) -> &'a str {
    haystack
        .lines()
        .find(|l| l.contains(label))
        .unwrap_or_else(|| panic!("expected a {label} Time-split line in run stderr"))
}

const EXPECTED_MEAN_COST: f64 = 9_679_385.922_404_844;

/// Golden LP counts and cost on input-committed `examples/1dtoy` (seed 42, all
/// `in_sample`). Goldens change only when the case, solver or algorithm changes.
#[test]
fn run_produces_deterministic_end_block_and_metadata() {
    let case = case_dir("1dtoy");
    assert!(
        case.is_dir(),
        "committed example case must exist at {}",
        case.display()
    );

    let out = TempDir::new().unwrap();
    let out_path = out.path();

    let (_, run_stderr) = run_ok(&[
        "run",
        case.to_str().unwrap(),
        "--output",
        out_path.to_str().unwrap(),
    ]);

    for rel in [
        "training/metadata.json",
        "simulation/metadata.json",
        "training/hydro_models.json",
        "training/model_provenance.json",
    ] {
        let path = out_path.join(rel);
        assert!(
            path.is_file(),
            "run must write {rel} (at {})",
            path.display()
        );
    }

    for needle in [
        "Lower bound:  1.55955e7",
        "Upper bound:  5.79592e5",
        "LP solves:    5632 (",
        "LP solves:    400 (",
        "Expected cost: 9.67939e6",
        "1 constant",
        "0 linearized, 1 without",
        "user_stats_white_noise",
    ] {
        assert!(
            run_stderr.contains(needle),
            "live run stderr must contain {needle:?}"
        );
    }

    let forward_line = find_line(&run_stderr, "Forward");
    let backward_line = find_line(&run_stderr, "Backward");
    let serial_line = find_line(&run_stderr, "Serial");

    for (label, line) in [
        ("Forward", forward_line),
        ("Backward", backward_line),
        ("Serial", serial_line),
    ] {
        assert!(
            line.contains('%'),
            "{label} Time-split line must carry a wall value with a percentage, got: {line:?}"
        );
    }
    assert!(
        forward_line.contains("solve") && forward_line.contains("wait"),
        "Forward line must show the solve/wait decomposition, got: {forward_line:?}"
    );
    assert!(
        backward_line.contains("solve") && backward_line.contains("wait"),
        "Backward line must show the solve/wait decomposition, got: {backward_line:?}"
    );
    assert_ordered(&run_stderr, "Forward", "Backward");
    assert_ordered(&run_stderr, "Backward", "Serial");

    let sim_metadata_content =
        std::fs::read_to_string(out_path.join("simulation/metadata.json")).unwrap();
    let sim_metadata: serde_json::Value = serde_json::from_str(&sim_metadata_content).unwrap();

    let mean_cost = sim_metadata["cost"]["mean_cost"]
        .as_f64()
        .expect(".cost.mean_cost must be present and non-null");
    let rel_err = (mean_cost - EXPECTED_MEAN_COST).abs() / EXPECTED_MEAN_COST;
    assert!(
        rel_err < 1e-3,
        "simulation metadata .cost.mean_cost = {mean_cost} is not within 1e-3 of \
         {EXPECTED_MEAN_COST} (relative error {rel_err})"
    );

    let training_metadata_content =
        std::fs::read_to_string(out_path.join("training/metadata.json")).unwrap();
    let training_metadata: serde_json::Value =
        serde_json::from_str(&training_metadata_content).unwrap();

    assert_eq!(
        training_metadata["solve_stats"]["total_lp_solves"].as_u64(),
        Some(5632),
        "training metadata .solve_stats.total_lp_solves must be 5632"
    );
    assert_eq!(
        sim_metadata["solve_stats"]["total_lp_solves"].as_u64(),
        Some(400),
        "simulation metadata .solve_stats.total_lp_solves must be 400"
    );
}

// ── Stored basis that does not fit, at simulation-only load ──────────────────

/// `examples/1dtoy`'s two thermals plus a third on the same bus: adds LP
/// columns but no state, so a policy trained on this variant passes every
/// `validate_policy_load` check against the original 1dtoy and reaches the
/// stored-basis fit rule.
const THERMALS_WITH_EXTRA_JSON: &str = r#"{
    "thermals": [
        {
            "id": 0, "name": "UTE1", "operational_start_date": "2020-01-01", "bus_id": 0,
            "generation": { "min_mw": 0.0, "max_mw": 15.0 }, "cost_per_mwh": 5.0
        },
        {
            "id": 1, "name": "UTE2", "operational_start_date": "2020-01-02", "bus_id": 0,
            "generation": { "min_mw": 0.0, "max_mw": 15.0 }, "cost_per_mwh": 10.0
        },
        {
            "id": 2, "name": "UTE3", "operational_start_date": "2020-01-01", "bus_id": 0,
            "generation": { "min_mw": 0.0, "max_mw": 15.0 }, "cost_per_mwh": 20.0
        }
    ]
}"#;

const CONFIG_VARIANT_TRAIN_JSON: &str = r#"{
    "training": {
        "selection": { "method": "sampled", "forward_passes": 1 },
        "stopping_rules": [ { "type": "iteration_limit", "limit": 2 } ],
        "scenario_source": {
            "seed": 42,
            "inflow": { "scheme": "in_sample" },
            "load": { "scheme": "in_sample" },
            "ncs": { "scheme": "in_sample" }
        }
    },
    "simulation": { "enabled": false },
    "modeling": { "inflow_non_negativity": { "method": "none" } }
}"#;

const CONFIG_SIMULATION_ONLY_JSON: &str = r#"{
    "training": {
        "enabled": false,
        "selection": { "method": "sampled", "forward_passes": 1 },
        "stopping_rules": [ { "type": "iteration_limit", "limit": 2 } ],
        "scenario_source": {
            "seed": 42,
            "inflow": { "scheme": "in_sample" },
            "load": { "scheme": "in_sample" },
            "ncs": { "scheme": "in_sample" }
        }
    },
    "simulation": { "enabled": true, "selection": { "method": "sampled", "num_scenarios": 1 } },
    "modeling": { "inflow_non_negativity": { "method": "none" } }
}"#;

/// A policy trained on a 1dtoy variant with one extra thermal (wider LP
/// columns, identical state) loads for simulation-only into the original 1dtoy:
/// each stored basis whose column count no longer matches its node's LP is left
/// out, one warning reports them, and the simulation runs. Ends with the
/// ordering assertion: the same policy, restamped to another cobre version, is
/// refused by the version check instead, before any basis is examined.
#[test]
fn simulation_only_loads_a_policy_with_a_wider_stored_basis_and_warns() {
    let variant_dir = TempDir::new().unwrap();
    copy_dir_recursive(&case_dir("1dtoy"), variant_dir.path());
    write_file(
        variant_dir.path(),
        "system/thermals.json",
        THERMALS_WITH_EXTRA_JSON,
    );
    write_file(variant_dir.path(), "config.json", CONFIG_VARIANT_TRAIN_JSON);

    let output = TempDir::new().unwrap();
    cobre()
        .args([
            "run",
            variant_dir.path().to_str().unwrap(),
            "--output",
            output.path().to_str().unwrap(),
            "--quiet",
        ])
        .assert()
        .success();

    let sim_only_dir = TempDir::new().unwrap();
    copy_dir_recursive(&case_dir("1dtoy"), sim_only_dir.path());
    write_file(
        sim_only_dir.path(),
        "config.json",
        CONFIG_SIMULATION_ONLY_JSON,
    );

    let run = cobre()
        .args([
            "run",
            sim_only_dir.path().to_str().unwrap(),
            "--output",
            output.path().to_str().unwrap(),
        ])
        .assert()
        .success();
    let stderr = String::from_utf8_lossy(&run.get_output().stderr).into_owned();
    assert_eq!(
        stderr.matches("stored bases not used: ").count(),
        1,
        "the misfit must be reported exactly once: {stderr}"
    );
    assert_empty_file(&output.path().join("simulation/_SUCCESS"));

    restamp_policy_version(&output.path().join("policy"), "0.0.1");

    cobre()
        .args([
            "run",
            sim_only_dir.path().to_str().unwrap(),
            "--output",
            output.path().to_str().unwrap(),
            "--quiet",
        ])
        .assert()
        .failure()
        .code(1)
        .stderr(predicate::str::contains("written by cobre 0.0.1"))
        .stderr(predicate::str::contains("stored bases not used").not());
}

/// The policy of the wider-basis variant, validated against the original
/// 1dtoy for simulation-only: stored bases that no longer fit are warnings, and
/// validate still exits 0, with the same count in the human line and in `--json`.
#[test]
fn simulation_only_validate_warns_about_unused_stored_bases_and_exits_0() {
    let variant_dir = TempDir::new().unwrap();
    copy_dir_recursive(&case_dir("1dtoy"), variant_dir.path());
    write_file(
        variant_dir.path(),
        "system/thermals.json",
        THERMALS_WITH_EXTRA_JSON,
    );
    write_file(variant_dir.path(), "config.json", CONFIG_VARIANT_TRAIN_JSON);

    let sim_only_dir = TempDir::new().unwrap();
    copy_dir_recursive(&case_dir("1dtoy"), sim_only_dir.path());
    write_file(
        sim_only_dir.path(),
        "config.json",
        CONFIG_SIMULATION_ONLY_JSON,
    );
    cobre()
        .args([
            "run",
            variant_dir.path().to_str().unwrap(),
            "--output",
            sim_only_dir.path().join("output").to_str().unwrap(),
            "--quiet",
        ])
        .assert()
        .success();

    let human = cobre()
        .args(["validate", sim_only_dir.path().to_str().unwrap()])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&human.get_output().stdout).into_owned();
    let line = stdout
        .lines()
        .find(|line| line.contains("warning:") && line.contains("stored bases not used: "))
        .unwrap_or_else(|| panic!("no unused-basis warning line: {stdout}"));
    let count: usize = line
        .split("stored bases not used: ")
        .nth(1)
        .and_then(|rest| rest.split(' ').next())
        .and_then(|n| n.parse().ok())
        .unwrap_or_else(|| panic!("no count in: {line}"));
    assert!(count > 0, "{line}");

    let json = cobre()
        .args(["validate", sim_only_dir.path().to_str().unwrap(), "--json"])
        .assert()
        .success();
    let value: serde_json::Value = serde_json::from_slice(&json.get_output().stdout).unwrap();
    assert_eq!(
        value["policy_load"],
        serde_json::json!({ "mode": "simulation_only", "unused_stored_bases": count })
    );
}

fn policy_mode_config(mode: &str) -> String {
    format!(
        r#"{{
    "training": {{
        "selection": {{ "method": "sampled", "forward_passes": 1 }},
        "stopping_rules": [ {{ "type": "iteration_limit", "limit": 2 }} ],
        "scenario_source": {{
            "seed": 42,
            "inflow": {{ "scheme": "in_sample" }},
            "load": {{ "scheme": "in_sample" }},
            "ncs": {{ "scheme": "in_sample" }}
        }}
    }},
    "simulation": {{ "enabled": false }},
    "modeling": {{ "inflow_non_negativity": {{ "method": "none" }} }},
    "policy": {{ "mode": "{mode}" }}
}}"#
    )
}

#[test]
fn missing_policy_directory_is_reported_for_each_load_kind() {
    let cases = [
        (
            policy_mode_config("warm_start"),
            "Cannot warm-start without a prior policy.",
        ),
        (
            policy_mode_config("resume"),
            "Cannot resume without a prior checkpoint.",
        ),
        (
            CONFIG_SIMULATION_ONLY_JSON.to_string(),
            "Cannot run simulation-only mode without a trained policy.",
        ),
    ];
    for (config, sentence) in cases {
        let case = TempDir::new().unwrap();
        copy_dir_recursive(&case_dir("1dtoy"), case.path());
        write_file(case.path(), "config.json", &config);
        let output = TempDir::new().unwrap();
        cobre()
            .args([
                "run",
                case.path().to_str().unwrap(),
                "--output",
                output.path().to_str().unwrap(),
                "--quiet",
            ])
            .assert()
            .failure()
            .code(1)
            .stderr(predicate::str::contains("Policy directory not found: "))
            .stderr(predicate::str::contains(sentence))
            .stderr(predicate::str::contains(
                "run `cobre validate <CASE_DIR>` for a full diagnostic report",
            ))
            .stderr(predicate::str::contains("report this at").not());
    }
}

#[test]
fn unreadable_policy_checkpoint_is_reported_as_a_read_failure() {
    let case = TempDir::new().unwrap();
    copy_dir_recursive(&case_dir("1dtoy"), case.path());
    write_file(case.path(), "config.json", &policy_mode_config("fresh"));
    let output = TempDir::new().unwrap();
    cobre()
        .args([
            "run",
            case.path().to_str().unwrap(),
            "--output",
            output.path().to_str().unwrap(),
            "--quiet",
        ])
        .assert()
        .success();

    fs::write(output.path().join("policy/manifest.bin"), b"garbage").unwrap();

    write_file(
        case.path(),
        "config.json",
        &policy_mode_config("warm_start"),
    );
    cobre()
        .args([
            "run",
            case.path().to_str().unwrap(),
            "--output",
            output.path().to_str().unwrap(),
            "--quiet",
        ])
        .assert()
        .failure()
        .code(1)
        .stderr(predicate::str::contains(
            "failed to read policy checkpoint: ",
        ))
        .stderr(predicate::str::contains("report this at").not());
}

fn warm_start_case_with_output_policy_dir() -> (TempDir, TempDir, std::path::PathBuf) {
    let case = TempDir::new().unwrap();
    copy_dir_recursive(&case_dir("1dtoy"), case.path());
    rewrite_json(&case.path().join("config.json"), |config| {
        config["policy"]["mode"] = serde_json::json!("warm_start");
    });
    let output = TempDir::new().unwrap();
    let policy = output.path().join("policy");
    fs::create_dir(&policy).unwrap();
    (case, output, policy)
}

#[test]
fn policy_directory_without_manifest_exits_1() {
    let (case, output, _policy) = warm_start_case_with_output_policy_dir();
    cobre()
        .args([
            "run",
            case.path().to_str().unwrap(),
            "--output",
            output.path().to_str().unwrap(),
            "--quiet",
        ])
        .assert()
        .failure()
        .code(1)
        .stderr(predicate::str::contains(
            "failed to read policy checkpoint: ",
        ))
        .stderr(predicate::str::contains("manifest.bin"))
        .stderr(predicate::str::contains("report this at").not());
}

#[cfg(unix)]
#[test]
fn policy_manifest_the_process_cannot_open_exits_2() {
    use std::os::unix::fs::PermissionsExt;

    let (case, output, policy) = warm_start_case_with_output_policy_dir();
    let manifest = policy.join("manifest.bin");
    fs::write(&manifest, b"garbage").unwrap();
    fs::set_permissions(&manifest, fs::Permissions::from_mode(0o000)).unwrap();
    if fs::read(&manifest).is_ok() {
        fs::set_permissions(&manifest, fs::Permissions::from_mode(0o644)).unwrap();
        println!("skipped: permission bits are not enforced for this process");
        return;
    }

    let assertion = cobre()
        .args([
            "run",
            case.path().to_str().unwrap(),
            "--output",
            output.path().to_str().unwrap(),
            "--quiet",
        ])
        .assert();
    fs::set_permissions(&manifest, fs::Permissions::from_mode(0o644)).unwrap();
    assertion
        .failure()
        .code(2)
        .stderr(predicate::str::contains("I/O error in"))
        .stderr(predicate::str::contains("manifest.bin"));
}

// ── Error classification of refusals and in-loop failures ────────────────────

fn rewrite_json(path: &Path, edit: impl FnOnce(&mut serde_json::Value)) {
    let mut value: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();
    edit(&mut value);
    fs::write(path, serde_json::to_string_pretty(&value).unwrap()).unwrap();
}

#[test]
fn stochastic_data_refusal_exits_1_without_a_bug_report_request() {
    let dir = TempDir::new().unwrap();
    copy_dir_recursive(&case_dir("1dtoy"), dir.path());
    rewrite_json(&dir.path().join("config.json"), |config| {
        config["training"]["scenario_source"]["inflow"] =
            serde_json::json!({ "scheme": "historical" });
    });

    cobre()
        .args(["run", dir.path().to_str().unwrap()])
        .assert()
        .failure()
        .code(1)
        .stderr(predicate::str::contains(
            "stochastic error: insufficient data: no valid historical windows found",
        ))
        .stderr(predicate::str::contains(
            "run `cobre validate <CASE_DIR>` for a full diagnostic report",
        ))
        .stderr(predicate::str::contains("report this at").not());
}

#[test]
fn infeasible_training_lp_exits_3_naming_stage_iteration_and_scenario() {
    let dir = TempDir::new().unwrap();
    copy_dir_recursive(
        &case_dir("deterministic/d13-generic-constraint"),
        dir.path(),
    );
    rewrite_json(
        &dir.path().join("constraints/generic_constraints.json"),
        |constraints| {
            constraints["constraints"][0]["slack"] = serde_json::json!({ "enabled": false });
        },
    );
    rewrite_json(&dir.path().join("system/thermals.json"), |thermals| {
        thermals["thermals"][0]["generation"]["min_mw"] = serde_json::json!(20.0);
    });

    cobre()
        .args(["run", dir.path().to_str().unwrap()])
        .assert()
        .failure()
        .code(3)
        .stderr(predicate::str::contains(
            "LP infeasible at stage 0, iteration 1, scenario 0",
        ))
        .stderr(predicate::str::contains(
            "Training failed after 0 iterations",
        ))
        .stderr(predicate::str::contains("report this at").not());
}

// ── Phase success markers ─────────────────────────────────────────────────────

fn assert_empty_file(path: &Path) {
    let metadata =
        fs::metadata(path).unwrap_or_else(|e| panic!("{} must exist: {e}", path.display()));
    assert!(metadata.is_file(), "{} must be a file", path.display());
    assert_eq!(metadata.len(), 0, "{} must be empty", path.display());
}

fn run_1dtoy_with_a_directory_at(blocked: &str) -> TempDir {
    let out = TempDir::new().unwrap();
    fs::create_dir_all(out.path().join(blocked)).unwrap();
    cobre()
        .args([
            "run",
            case_dir("1dtoy").to_str().unwrap(),
            "--output",
            out.path().to_str().unwrap(),
            "--quiet",
        ])
        .assert()
        .failure();
    out
}

#[test]
fn run_writes_a_success_marker_for_each_executed_phase() {
    let out = TempDir::new().unwrap();
    run_case(&case_dir("1dtoy"), out.path());

    assert_empty_file(&out.path().join("training/_SUCCESS"));
    assert_empty_file(&out.path().join("simulation/_SUCCESS"));
}

#[test]
fn training_only_run_writes_no_simulation_success_marker() {
    let dir = TempDir::new().unwrap();
    make_valid_case(dir.path(), None, None, None, None);
    let out = TempDir::new().unwrap();
    run_case(dir.path(), out.path());

    assert_empty_file(&out.path().join("training/_SUCCESS"));
    assert!(!out.path().join("simulation/_SUCCESS").exists());
}

#[test]
fn run_writes_no_training_marker_when_the_last_training_write_fails() {
    let out = run_1dtoy_with_a_directory_at("training/solver/retry_histogram.parquet.tmp");

    assert!(out.path().join("training/metadata.json").is_file());
    assert!(
        !out.path().join("training/_SUCCESS").exists(),
        "training/_SUCCESS must not exist when a training write failed"
    );
    assert!(!out.path().join("simulation/_SUCCESS").exists());
}

#[test]
fn run_writes_no_simulation_marker_when_the_last_simulation_write_fails() {
    let out = run_1dtoy_with_a_directory_at("simulation/scenario_summary.parquet.tmp");

    assert!(out.path().join("simulation/metadata.json").is_file());
    assert!(out.path().join("training/_SUCCESS").is_file());
    assert!(
        !out.path().join("simulation/_SUCCESS").exists(),
        "simulation/_SUCCESS must not exist when a simulation write failed"
    );
}

#[test]
fn run_clears_stale_markers_of_planned_phases_before_writing() {
    let out = TempDir::new().unwrap();
    write_file(out.path(), "training/_SUCCESS", "");
    write_file(out.path(), "simulation/_SUCCESS", "");
    fs::create_dir_all(
        out.path()
            .join("training/solver/retry_histogram.parquet.tmp"),
    )
    .unwrap();

    cobre()
        .args([
            "run",
            case_dir("1dtoy").to_str().unwrap(),
            "--output",
            out.path().to_str().unwrap(),
            "--quiet",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("training/solver"));

    assert!(out.path().join("training/metadata.json").is_file());
    assert!(
        !out.path().join("training/_SUCCESS").exists(),
        "the stale training/_SUCCESS must be removed before training writes"
    );
    assert!(
        !out.path().join("simulation/_SUCCESS").exists(),
        "the stale simulation/_SUCCESS must be removed before training when simulation is planned"
    );
}

#[test]
fn run_keeps_the_marker_of_a_phase_it_does_not_run() {
    let dir = TempDir::new().unwrap();
    make_valid_case(dir.path(), None, None, None, None);
    let out = TempDir::new().unwrap();
    write_file(out.path(), "simulation/_SUCCESS", "");
    write_file(out.path(), "simulation/paths.parquet", "");
    write_file(
        out.path(),
        "simulation/costs/scenario_id=0000/data.parquet",
        "",
    );

    run_case(dir.path(), out.path());

    assert_empty_file(&out.path().join("simulation/_SUCCESS"));
    assert_empty_file(&out.path().join("training/_SUCCESS"));
    assert!(out.path().join("simulation/paths.parquet").is_file());
    assert!(
        out.path()
            .join("simulation/costs/scenario_id=0000/data.parquet")
            .is_file()
    );
}

#[test]
fn simulation_only_run_keeps_the_training_marker() {
    let dir = TempDir::new().unwrap();
    copy_dir_recursive(&case_dir("1dtoy"), dir.path());
    let out = TempDir::new().unwrap();
    run_case(dir.path(), out.path());
    rewrite_json(&dir.path().join("config.json"), |config| {
        config["training"]["enabled"] = serde_json::json!(false);
    });
    write_file(
        out.path(),
        "training/cut_selection/iterations.parquet",
        "stale",
    );
    write_file(out.path(), "hydro_models/fpha_hyperplanes.parquet", "stale");

    run_case(dir.path(), out.path());

    assert_empty_file(&out.path().join("training/_SUCCESS"));
    assert_empty_file(&out.path().join("simulation/_SUCCESS"));
    for kept in [
        "training/cut_selection/iterations.parquet",
        "hydro_models/fpha_hyperplanes.parquet",
        "training/solver/iterations.parquet",
    ] {
        assert!(
            out.path().join(kept).is_file(),
            "{kept} must be kept by a run that does not train"
        );
    }
}

const STALE_SIMULATION_OUTPUTS: [(&str, &str); 6] = [
    ("simulation/costs/scenario_id=9999/data.parquet", ""),
    (
        "simulation/pumping_stations/scenario_id=0000/data.parquet",
        "",
    ),
    ("simulation/solver/iterations.parquet", "stale"),
    ("simulation/paths.parquet", ""),
    ("simulation/scenario_summary.parquet", ""),
    ("simulation/metadata.json", "{}"),
];

const FOREIGN_SIMULATION_FILES: [&str; 2] = ["simulation/solver/stale.txt", "simulation/notes.txt"];

fn seed_stale_simulation_outputs(out: &Path) {
    for (relative, content) in STALE_SIMULATION_OUTPUTS {
        write_file(out, relative, content);
    }
    for relative in FOREIGN_SIMULATION_FILES {
        write_file(out, relative, "");
    }
}

#[test]
fn run_clears_stale_simulation_outputs_before_training() {
    let out = TempDir::new().unwrap();
    seed_stale_simulation_outputs(out.path());
    write_file(out.path(), "training/solver", "");

    cobre()
        .args([
            "run",
            case_dir("1dtoy").to_str().unwrap(),
            "--output",
            out.path().to_str().unwrap(),
            "--quiet",
        ])
        .assert()
        .failure();

    for (relative, _) in STALE_SIMULATION_OUTPUTS {
        assert!(
            !out.path().join(relative).exists(),
            "the stale {relative} must be removed before training when simulation is planned"
        );
    }
    for relative in FOREIGN_SIMULATION_FILES {
        assert!(
            out.path().join(relative).is_file(),
            "{relative} is not a cobre output and must be kept"
        );
    }
}

#[test]
fn run_replaces_stale_simulation_outputs() {
    let out = TempDir::new().unwrap();
    seed_stale_simulation_outputs(out.path());

    run_case(&case_dir("1dtoy"), out.path());

    let sim = out.path().join("simulation");
    assert!(!sim.join("costs/scenario_id=9999").exists());
    assert!(!sim.join("pumping_stations").exists());
    for kept in [
        "costs/scenario_id=0000/data.parquet",
        "solver/iterations.parquet",
        "solver/stale.txt",
        "notes.txt",
        "_SUCCESS",
    ] {
        assert!(sim.join(kept).is_file(), "simulation/{kept} must exist");
    }
}

// ── Conditional training outputs ──────────────────────────────────────────────

/// The conditional training outputs 1dtoy never writes.
const STALE_CONDITIONAL_TRAINING_OUTPUTS: [&str; 6] = [
    "training/cut_selection/iterations.parquet",
    "hydro_models/fpha_hyperplanes.parquet",
    "hydro_models/evaporation_models.parquet",
    "hydro_models/fpha_deviation_points.parquet",
    "generic_constraints/resolved_echo.parquet",
    "anticipated/fixed_deliveries.parquet",
];

fn seed_stale_conditional_training_outputs(out: &Path) {
    for relative in STALE_CONDITIONAL_TRAINING_OUTPUTS {
        write_file(out, relative, "stale");
    }
    write_file(out, "hydro_models/notes.txt", "");
}

fn assert_stale_conditional_training_outputs_removed(out: &Path) {
    for relative in STALE_CONDITIONAL_TRAINING_OUTPUTS {
        assert!(
            !out.join(relative).exists(),
            "the stale {relative} must be removed before training"
        );
    }
}

fn copy_of_1dtoy_training_six_iterations() -> TempDir {
    let case = TempDir::new().unwrap();
    copy_dir_recursive(&case_dir("1dtoy"), case.path());
    rewrite_json(&case.path().join("config.json"), |config| {
        config["training"]["stopping_rules"] =
            serde_json::json!([{ "type": "iteration_limit", "limit": 6 }]);
        config["simulation"]["enabled"] = serde_json::json!(false);
    });
    case
}

#[test]
fn run_clears_stale_conditional_training_outputs() {
    let out = TempDir::new().unwrap();
    seed_stale_conditional_training_outputs(out.path());

    run_case(&case_dir("1dtoy"), out.path());

    assert_stale_conditional_training_outputs_removed(out.path());
    for emptied in [
        "training/cut_selection",
        "anticipated",
        "generic_constraints",
    ] {
        assert!(
            !out.path().join(emptied).exists(),
            "{emptied} must be removed once its stale output is gone"
        );
    }
    assert!(out.path().join("hydro_models/notes.txt").is_file());
    assert_empty_file(&out.path().join("training/_SUCCESS"));
}

#[test]
fn run_clears_stale_conditional_training_outputs_before_training() {
    let out = TempDir::new().unwrap();
    seed_stale_conditional_training_outputs(out.path());
    fs::create_dir_all(
        out.path()
            .join("training/solver/retry_histogram.parquet.tmp"),
    )
    .unwrap();

    cobre()
        .args([
            "run",
            case_dir("1dtoy").to_str().unwrap(),
            "--output",
            out.path().to_str().unwrap(),
            "--quiet",
        ])
        .assert()
        .failure();

    assert_stale_conditional_training_outputs_removed(out.path());
    assert!(out.path().join("training/metadata.json").is_file());
    assert!(
        !out.path().join("training/_SUCCESS").exists(),
        "training/_SUCCESS must not exist when a training write failed"
    );
}

#[test]
fn run_clears_cut_selection_output_after_cut_selection_is_disabled() {
    let case = copy_of_1dtoy_training_six_iterations();
    rewrite_json(&case.path().join("config.json"), |config| {
        config["training"]["cut_selection"] = serde_json::json!({
            "selection": { "method": "level1", "check_frequency": 2 }
        });
    });
    let out = TempDir::new().unwrap();
    run_case(case.path(), out.path());
    assert!(
        out.path()
            .join("training/cut_selection/iterations.parquet")
            .is_file()
    );

    rewrite_json(&case.path().join("config.json"), |config| {
        config["training"]
            .as_object_mut()
            .unwrap()
            .remove("cut_selection");
    });
    run_case(case.path(), out.path());

    assert_empty_file(&out.path().join("training/_SUCCESS"));
    assert!(!out.path().join("training/cut_selection").exists());
}

#[test]
fn run_clears_fpha_hyperplanes_after_switching_to_constant_productivity() {
    let case = TempDir::new().unwrap();
    copy_dir_recursive(&case_dir("deterministic/d07-fpha-computed"), case.path());
    let out = TempDir::new().unwrap();
    run_case(case.path(), out.path());
    assert!(
        out.path()
            .join("hydro_models/fpha_hyperplanes.parquet")
            .is_file()
    );

    rewrite_json(
        &case.path().join("system/hydro_production_models.json"),
        |models| {
            let range = &mut models["production_models"][0]["stage_ranges"][0];
            range["model"] = serde_json::json!("constant_productivity");
            range.as_object_mut().unwrap().remove("fpha_config");
        },
    );
    run_case(case.path(), out.path());

    assert_empty_file(&out.path().join("training/_SUCCESS"));
    assert!(!out.path().join("hydro_models").exists());
}

#[test]
fn warm_start_rerun_reads_the_policy_the_training_clear_keeps() {
    let case = copy_of_1dtoy_training_six_iterations();
    let out = TempDir::new().unwrap();
    run_case(case.path(), out.path());
    rewrite_json(&case.path().join("config.json"), |config| {
        config["policy"]["mode"] = serde_json::json!("warm_start");
    });

    run_case(case.path(), out.path());

    assert_empty_file(&out.path().join("training/_SUCCESS"));
}

/// A study that supplies its opening tree from a file runs although its
/// `historical_residuals` stages have no inflow history to build a library from.
#[test]
fn run_accepts_a_supplied_opening_tree_with_historical_residuals_stages() {
    let case = TempDir::new().unwrap();
    write_supplied_opening_tree_case(case.path());
    let out = TempDir::new().unwrap();

    cobre()
        .args(["run", case.path().to_str().unwrap()])
        .args(["--output", out.path().to_str().unwrap(), "--quiet"])
        .assert()
        .success()
        .stdout(predicate::str::contains("historical windows").not())
        .stdout(predicate::str::contains("V2.").not())
        .stderr(predicate::str::contains("historical windows").not())
        .stderr(predicate::str::contains("V2.").not());
}
