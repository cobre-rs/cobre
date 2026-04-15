//! Parsing for external scenario Parquet files — pre-computed scenario values.
//!
//! Three parsers are provided, one per entity class:
//!
//! - [`parse_external_inflow_scenarios`] — `scenarios/external_inflow_scenarios.parquet`
//! - [`parse_external_load_scenarios`] — `scenarios/external_load_scenarios.parquet`
//! - [`parse_external_ncs_scenarios`] — `scenarios/external_ncs_scenarios.parquet`
//!
//! ## Parquet schemas (spec E2)
//!
//! ### `external_inflow_scenarios.parquet`
//!
//! | Column        | Type   | Required | Description                      |
//! | ------------- | ------ | -------- | -------------------------------- |
//! | `stage_id`    | INT32  | Yes      | Stage ID                         |
//! | `scenario_id` | INT32  | Yes      | Scenario index (0-based)         |
//! | `hydro_id`    | INT32  | Yes      | Hydro plant ID                   |
//! | `value_m3s`   | DOUBLE | Yes      | Inflow value in m³/s             |
//!
//! ### `external_load_scenarios.parquet`
//!
//! | Column        | Type   | Required | Description                      |
//! | ------------- | ------ | -------- | -------------------------------- |
//! | `stage_id`    | INT32  | Yes      | Stage ID                         |
//! | `scenario_id` | INT32  | Yes      | Scenario index (0-based)         |
//! | `bus_id`      | INT32  | Yes      | Bus ID                           |
//! | `value_mw`    | DOUBLE | Yes      | Load value in MW                 |
//!
//! ### `external_ncs_scenarios.parquet`
//!
//! | Column        | Type   | Required | Description                      |
//! | ------------- | ------ | -------- | -------------------------------- |
//! | `stage_id`    | INT32  | Yes      | Stage ID                         |
//! | `scenario_id` | INT32  | Yes      | Scenario index (0-based)         |
//! | `ncs_id`      | INT32  | Yes      | NCS source ID                    |
//! | `value`       | DOUBLE | Yes      | Dimensionless availability factor|
//!
//! ## Output ordering
//!
//! All parsers sort by `(stage_id, scenario_id, entity_id)` ascending.
//!
//! ## Validation
//!
//! Per-row constraints enforced by all parsers:
//!
//! - All four columns must be present with the correct types.
//! - Value columns must be finite (NaN and infinity are rejected).
//! - `scenario_id` must be >= 0 (0-based indexing).
//!
//! Deferred validations (not performed here, Epic 06):
//!
//! - Entity ID existence in registries — Layer 3, Epic 06.
//! - `stage_id` existence in the stages registry — Layer 3, Epic 06.
//! - Scenario count matching `stage.num_scenarios` — Layer 3/5, Epic 06.

use cobre_core::EntityId;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::fs::File;
use std::path::Path;

use crate::parquet_helpers::{extract_required_float64, extract_required_int32};
use crate::LoadError;

pub use cobre_core::scenario::{ExternalLoadRow, ExternalNcsRow, ExternalScenarioRow};

/// Parse `scenarios/external_inflow_scenarios.parquet` and return a sorted row table.
///
/// Reads all record batches from the Parquet file at `path`, validates per-row
/// constraints, then returns all rows sorted by `(stage_id, scenario_id, hydro_id)`
/// ascending.
///
/// # Errors
///
/// | Condition                                            | Error variant              |
/// | ---------------------------------------------------- | -------------------------- |
/// | File not found or permission denied                  | [`LoadError::IoError`]     |
/// | Malformed Parquet (corrupt header, etc.)             | [`LoadError::ParseError`]  |
/// | Required column missing or wrong type                | [`LoadError::SchemaError`] |
/// | `value_m3s` not finite (NaN or infinity)             | [`LoadError::SchemaError`] |
/// | `scenario_id` < 0                                    | [`LoadError::SchemaError`] |
///
/// # Examples
///
/// ```no_run
/// use cobre_io::scenarios::parse_external_inflow_scenarios;
/// use std::path::Path;
///
/// let rows = parse_external_inflow_scenarios(
///     Path::new("scenarios/external_inflow_scenarios.parquet")
/// )
/// .expect("valid external inflow scenarios file");
/// println!("loaded {} external inflow scenario rows", rows.len());
/// ```
pub fn parse_external_inflow_scenarios(path: &Path) -> Result<Vec<ExternalScenarioRow>, LoadError> {
    let file = File::open(path).map_err(|e| LoadError::io(path, e))?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| LoadError::parse(path, e.to_string()))?;

    let reader = builder
        .build()
        .map_err(|e| LoadError::parse(path, e.to_string()))?;

    let mut rows: Vec<ExternalScenarioRow> = Vec::new();

    for batch_result in reader {
        let batch = batch_result.map_err(|e| LoadError::parse(path, e.to_string()))?;

        let stage_id_col = extract_required_int32(&batch, "stage_id", path)?;
        let scenario_id_col = extract_required_int32(&batch, "scenario_id", path)?;
        let hydro_id_col = extract_required_int32(&batch, "hydro_id", path)?;
        let value_m3s_col = extract_required_float64(&batch, "value_m3s", path)?;

        let n = batch.num_rows();
        let base_idx = rows.len();
        rows.reserve(n);

        for i in 0..n {
            let row_idx = base_idx + i;

            let stage_id = stage_id_col.value(i);
            let scenario_id = scenario_id_col.value(i);
            let hydro_id = EntityId::from(hydro_id_col.value(i));
            let value_m3s = value_m3s_col.value(i);

            // Validate stage_id: must be >= 0.
            if stage_id < 0 {
                return Err(LoadError::SchemaError {
                    path: path.to_path_buf(),
                    field: format!("external_inflow_scenarios[{row_idx}].stage_id"),
                    message: format!("stage_id must be >= 0, got {stage_id}"),
                });
            }

            // Validate scenario_id: must be >= 0 (0-based indexing).
            if scenario_id < 0 {
                return Err(LoadError::SchemaError {
                    path: path.to_path_buf(),
                    field: format!("external_inflow_scenarios[{row_idx}].scenario_id"),
                    message: format!("scenario_id must be >= 0 (0-based), got {scenario_id}"),
                });
            }

            // Validate value_m3s: must be finite (no NaN or infinity).
            if !value_m3s.is_finite() {
                return Err(LoadError::SchemaError {
                    path: path.to_path_buf(),
                    field: format!("external_inflow_scenarios[{row_idx}].value_m3s"),
                    message: format!("value_m3s must be finite, got {value_m3s}"),
                });
            }

            rows.push(ExternalScenarioRow {
                stage_id,
                scenario_id,
                hydro_id,
                value_m3s,
            });
        }
    }

    rows.sort_by(|a, b| {
        a.stage_id
            .cmp(&b.stage_id)
            .then_with(|| a.scenario_id.cmp(&b.scenario_id))
            .then_with(|| a.hydro_id.0.cmp(&b.hydro_id.0))
    });

    // Check for duplicate (stage_id, scenario_id, hydro_id) tuples.
    if let Some(window) = rows.windows(2).find(|w| {
        w[0].stage_id == w[1].stage_id
            && w[0].scenario_id == w[1].scenario_id
            && w[0].hydro_id == w[1].hydro_id
    }) {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: "external_inflow_scenarios".to_string(),
            message: format!(
                "duplicate row: stage_id={}, scenario_id={}, hydro_id={}",
                window[0].stage_id, window[0].scenario_id, window[0].hydro_id.0,
            ),
        });
    }

    Ok(rows)
}

/// Parse `scenarios/external_load_scenarios.parquet` and return a sorted row table.
///
/// Reads all record batches from the Parquet file at `path`, validates per-row
/// constraints, then returns all rows sorted by `(stage_id, scenario_id, bus_id)`
/// ascending.
///
/// # Errors
///
/// | Condition                                            | Error variant              |
/// | ---------------------------------------------------- | -------------------------- |
/// | File not found or permission denied                  | [`LoadError::IoError`]     |
/// | Malformed Parquet (corrupt header, etc.)             | [`LoadError::ParseError`]  |
/// | Required column missing or wrong type                | [`LoadError::SchemaError`] |
/// | `value_mw` not finite (NaN or infinity)              | [`LoadError::SchemaError`] |
/// | `scenario_id` < 0                                    | [`LoadError::SchemaError`] |
///
/// # Examples
///
/// ```no_run
/// use cobre_io::scenarios::parse_external_load_scenarios;
/// use std::path::Path;
///
/// let rows = parse_external_load_scenarios(
///     Path::new("scenarios/external_load_scenarios.parquet")
/// )
/// .expect("valid external load scenarios file");
/// println!("loaded {} external load scenario rows", rows.len());
/// ```
pub fn parse_external_load_scenarios(path: &Path) -> Result<Vec<ExternalLoadRow>, LoadError> {
    let file = File::open(path).map_err(|e| LoadError::io(path, e))?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| LoadError::parse(path, e.to_string()))?;

    let reader = builder
        .build()
        .map_err(|e| LoadError::parse(path, e.to_string()))?;

    let mut rows: Vec<ExternalLoadRow> = Vec::new();

    for batch_result in reader {
        let batch = batch_result.map_err(|e| LoadError::parse(path, e.to_string()))?;

        let stage_id_col = extract_required_int32(&batch, "stage_id", path)?;
        let scenario_id_col = extract_required_int32(&batch, "scenario_id", path)?;
        let bus_id_col = extract_required_int32(&batch, "bus_id", path)?;
        let value_mw_col = extract_required_float64(&batch, "value_mw", path)?;

        let n = batch.num_rows();
        let base_idx = rows.len();
        rows.reserve(n);

        for i in 0..n {
            let row_idx = base_idx + i;

            let stage_id = stage_id_col.value(i);
            let scenario_id = scenario_id_col.value(i);
            let bus_id = EntityId::from(bus_id_col.value(i));
            let value_mw = value_mw_col.value(i);

            // Validate stage_id: must be >= 0.
            if stage_id < 0 {
                return Err(LoadError::SchemaError {
                    path: path.to_path_buf(),
                    field: format!("external_load_scenarios[{row_idx}].stage_id"),
                    message: format!("stage_id must be >= 0, got {stage_id}"),
                });
            }

            // Validate scenario_id: must be >= 0 (0-based indexing).
            if scenario_id < 0 {
                return Err(LoadError::SchemaError {
                    path: path.to_path_buf(),
                    field: format!("external_load_scenarios[{row_idx}].scenario_id"),
                    message: format!("scenario_id must be >= 0 (0-based), got {scenario_id}"),
                });
            }

            // Validate value_mw: must be finite (no NaN or infinity).
            if !value_mw.is_finite() {
                return Err(LoadError::SchemaError {
                    path: path.to_path_buf(),
                    field: format!("external_load_scenarios[{row_idx}].value_mw"),
                    message: format!("value_mw must be finite, got {value_mw}"),
                });
            }

            rows.push(ExternalLoadRow {
                stage_id,
                scenario_id,
                bus_id,
                value_mw,
            });
        }
    }

    rows.sort_by(|a, b| {
        a.stage_id
            .cmp(&b.stage_id)
            .then_with(|| a.scenario_id.cmp(&b.scenario_id))
            .then_with(|| a.bus_id.0.cmp(&b.bus_id.0))
    });

    // Check for duplicate (stage_id, scenario_id, bus_id) tuples.
    if let Some(window) = rows.windows(2).find(|w| {
        w[0].stage_id == w[1].stage_id
            && w[0].scenario_id == w[1].scenario_id
            && w[0].bus_id == w[1].bus_id
    }) {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: "external_load_scenarios".to_string(),
            message: format!(
                "duplicate row: stage_id={}, scenario_id={}, bus_id={}",
                window[0].stage_id, window[0].scenario_id, window[0].bus_id.0,
            ),
        });
    }

    Ok(rows)
}

/// Parse `scenarios/external_ncs_scenarios.parquet` and return a sorted row table.
///
/// Reads all record batches from the Parquet file at `path`, validates per-row
/// constraints, then returns all rows sorted by `(stage_id, scenario_id, ncs_id)`
/// ascending.
///
/// # Errors
///
/// | Condition                                            | Error variant              |
/// | ---------------------------------------------------- | -------------------------- |
/// | File not found or permission denied                  | [`LoadError::IoError`]     |
/// | Malformed Parquet (corrupt header, etc.)             | [`LoadError::ParseError`]  |
/// | Required column missing or wrong type                | [`LoadError::SchemaError`] |
/// | `value` not finite (NaN or infinity)                 | [`LoadError::SchemaError`] |
/// | `scenario_id` < 0                                    | [`LoadError::SchemaError`] |
///
/// # Examples
///
/// ```no_run
/// use cobre_io::scenarios::parse_external_ncs_scenarios;
/// use std::path::Path;
///
/// let rows = parse_external_ncs_scenarios(
///     Path::new("scenarios/external_ncs_scenarios.parquet")
/// )
/// .expect("valid external NCS scenarios file");
/// println!("loaded {} external NCS scenario rows", rows.len());
/// ```
pub fn parse_external_ncs_scenarios(path: &Path) -> Result<Vec<ExternalNcsRow>, LoadError> {
    let file = File::open(path).map_err(|e| LoadError::io(path, e))?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| LoadError::parse(path, e.to_string()))?;

    let reader = builder
        .build()
        .map_err(|e| LoadError::parse(path, e.to_string()))?;

    let mut rows: Vec<ExternalNcsRow> = Vec::new();

    for batch_result in reader {
        let batch = batch_result.map_err(|e| LoadError::parse(path, e.to_string()))?;

        let stage_id_col = extract_required_int32(&batch, "stage_id", path)?;
        let scenario_id_col = extract_required_int32(&batch, "scenario_id", path)?;
        let ncs_id_col = extract_required_int32(&batch, "ncs_id", path)?;
        let value_col = extract_required_float64(&batch, "value", path)?;

        let n = batch.num_rows();
        let base_idx = rows.len();
        rows.reserve(n);

        for i in 0..n {
            let row_idx = base_idx + i;

            let stage_id = stage_id_col.value(i);
            let scenario_id = scenario_id_col.value(i);
            let ncs_id = EntityId::from(ncs_id_col.value(i));
            let value = value_col.value(i);

            // Validate stage_id: must be >= 0.
            if stage_id < 0 {
                return Err(LoadError::SchemaError {
                    path: path.to_path_buf(),
                    field: format!("external_ncs_scenarios[{row_idx}].stage_id"),
                    message: format!("stage_id must be >= 0, got {stage_id}"),
                });
            }

            // Validate scenario_id: must be >= 0 (0-based indexing).
            if scenario_id < 0 {
                return Err(LoadError::SchemaError {
                    path: path.to_path_buf(),
                    field: format!("external_ncs_scenarios[{row_idx}].scenario_id"),
                    message: format!("scenario_id must be >= 0 (0-based), got {scenario_id}"),
                });
            }

            // Validate value: must be finite (no NaN or infinity).
            if !value.is_finite() {
                return Err(LoadError::SchemaError {
                    path: path.to_path_buf(),
                    field: format!("external_ncs_scenarios[{row_idx}].value"),
                    message: format!("value must be finite, got {value}"),
                });
            }

            rows.push(ExternalNcsRow {
                stage_id,
                scenario_id,
                ncs_id,
                value,
            });
        }
    }

    rows.sort_by(|a, b| {
        a.stage_id
            .cmp(&b.stage_id)
            .then_with(|| a.scenario_id.cmp(&b.scenario_id))
            .then_with(|| a.ncs_id.0.cmp(&b.ncs_id.0))
    });

    // Check for duplicate (stage_id, scenario_id, ncs_id) tuples.
    if let Some(window) = rows.windows(2).find(|w| {
        w[0].stage_id == w[1].stage_id
            && w[0].scenario_id == w[1].scenario_id
            && w[0].ncs_id == w[1].ncs_id
    }) {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: "external_ncs_scenarios".to_string(),
            message: format!(
                "duplicate row: stage_id={}, scenario_id={}, ncs_id={}",
                window[0].stage_id, window[0].scenario_id, window[0].ncs_id.0,
            ),
        });
    }

    Ok(rows)
}

#[cfg(test)]
#[allow(
    clippy::doc_markdown,
    clippy::expect_used,
    clippy::panic,
    clippy::too_many_lines,
    clippy::unwrap_used
)]
mod tests {
    use super::*;
    use arrow::array::{Float64Array, Int32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;
    use tempfile::NamedTempFile;

    fn write_parquet(batch: &RecordBatch) -> NamedTempFile {
        let tmp = NamedTempFile::new().expect("tempfile");
        let mut writer = ArrowWriter::try_new(tmp.reopen().expect("reopen"), batch.schema(), None)
            .expect("ArrowWriter");
        writer.write(batch).expect("write batch");
        writer.close().expect("close writer");
        tmp
    }

    fn inflow_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("stage_id", DataType::Int32, false),
            Field::new("scenario_id", DataType::Int32, false),
            Field::new("hydro_id", DataType::Int32, false),
            Field::new("value_m3s", DataType::Float64, false),
        ]))
    }

    fn make_inflow_batch(
        stage_ids: &[i32],
        scenario_ids: &[i32],
        hydro_ids: &[i32],
        values: &[f64],
    ) -> RecordBatch {
        RecordBatch::try_new(
            inflow_schema(),
            vec![
                Arc::new(Int32Array::from(stage_ids.to_vec())),
                Arc::new(Int32Array::from(scenario_ids.to_vec())),
                Arc::new(Int32Array::from(hydro_ids.to_vec())),
                Arc::new(Float64Array::from(values.to_vec())),
            ],
        )
        .expect("valid batch")
    }

    fn load_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("stage_id", DataType::Int32, false),
            Field::new("scenario_id", DataType::Int32, false),
            Field::new("bus_id", DataType::Int32, false),
            Field::new("value_mw", DataType::Float64, false),
        ]))
    }

    fn make_load_batch(
        stage_ids: &[i32],
        scenario_ids: &[i32],
        bus_ids: &[i32],
        values: &[f64],
    ) -> RecordBatch {
        RecordBatch::try_new(
            load_schema(),
            vec![
                Arc::new(Int32Array::from(stage_ids.to_vec())),
                Arc::new(Int32Array::from(scenario_ids.to_vec())),
                Arc::new(Int32Array::from(bus_ids.to_vec())),
                Arc::new(Float64Array::from(values.to_vec())),
            ],
        )
        .expect("valid batch")
    }

    fn ncs_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("stage_id", DataType::Int32, false),
            Field::new("scenario_id", DataType::Int32, false),
            Field::new("ncs_id", DataType::Int32, false),
            Field::new("value", DataType::Float64, false),
        ]))
    }

    fn make_ncs_batch(
        stage_ids: &[i32],
        scenario_ids: &[i32],
        ncs_ids: &[i32],
        values: &[f64],
    ) -> RecordBatch {
        RecordBatch::try_new(
            ncs_schema(),
            vec![
                Arc::new(Int32Array::from(stage_ids.to_vec())),
                Arc::new(Int32Array::from(scenario_ids.to_vec())),
                Arc::new(Int32Array::from(ncs_ids.to_vec())),
                Arc::new(Float64Array::from(values.to_vec())),
            ],
        )
        .expect("valid batch")
    }

    #[test]
    fn test_parse_external_inflow_valid() {
        let batch = make_inflow_batch(
            &[1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0], // stage_id
            &[2, 0, 2, 0, 1, 1, 0, 0, 2, 2, 1, 1], // scenario_id
            &[2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1], // hydro_id
            &[
                120.0, 100.0, 110.0, 200.0, 105.0, 201.0, 205.0, 195.0, 115.0, 125.0, 210.0, 208.0,
            ],
        );
        let tmp = write_parquet(&batch);
        let rows = parse_external_inflow_scenarios(tmp.path()).unwrap();

        assert_eq!(rows.len(), 12);

        for w in rows.windows(2) {
            let a = &w[0];
            let b = &w[1];
            let cmp = a
                .stage_id
                .cmp(&b.stage_id)
                .then_with(|| a.scenario_id.cmp(&b.scenario_id))
                .then_with(|| a.hydro_id.0.cmp(&b.hydro_id.0));
            assert!(cmp != std::cmp::Ordering::Greater);
        }

        assert_eq!(rows[0].stage_id, 0);
        assert_eq!(rows[0].scenario_id, 0);
        assert_eq!(rows[0].hydro_id, EntityId::from(1));
        assert!((rows[0].value_m3s - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_parse_external_inflow_nan_rejected() {
        let batch = make_inflow_batch(&[0], &[0], &[1], &[f64::NAN]);
        let tmp = write_parquet(&batch);
        let err = parse_external_inflow_scenarios(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(field.contains("value_m3s"), "field: {field}");
                assert!(message.contains("finite"), "message: {message}");
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    #[test]
    fn test_parse_external_inflow_missing_hydro_id_column() {
        let schema_no_hydro = Arc::new(Schema::new(vec![
            Field::new("stage_id", DataType::Int32, false),
            Field::new("scenario_id", DataType::Int32, false),
            Field::new("value_m3s", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema_no_hydro,
            vec![
                Arc::new(Int32Array::from(vec![0_i32])),
                Arc::new(Int32Array::from(vec![0_i32])),
                Arc::new(Float64Array::from(vec![100.0_f64])),
            ],
        )
        .unwrap();
        let tmp = write_parquet(&batch);
        let err = parse_external_inflow_scenarios(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(field.contains("hydro_id"), "field: {field}");
                assert!(
                    message.contains("missing required column"),
                    "message: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    #[test]
    fn test_parse_external_inflow_negative_scenario_id_rejected() {
        let batch = make_inflow_batch(&[0], &[-1], &[1], &[100.0]);
        let tmp = write_parquet(&batch);
        let err = parse_external_inflow_scenarios(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(field.contains("scenario_id"), "field: {field}");
                assert!(message.contains('0'), "message: {message}");
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    #[test]
    fn test_parse_external_inflow_empty_parquet_returns_empty_vec() {
        let batch = make_inflow_batch(&[], &[], &[], &[]);
        let tmp = write_parquet(&batch);
        let rows = parse_external_inflow_scenarios(tmp.path()).unwrap();
        assert!(rows.is_empty());
    }

    #[test]
    fn test_load_external_inflow_scenarios_none_returns_empty() {
        let result = super::super::load_external_inflow_scenarios(None).unwrap();
        assert!(result.is_empty(), "expected empty vec for None path");
    }

    #[test]
    fn test_parse_external_load_valid() {
        let batch = make_load_batch(
            &[0, 1, 0, 1], // stage_id
            &[1, 0, 0, 1], // scenario_id
            &[3, 2, 1, 2], // bus_id
            &[500.0, 400.0, 300.0, 450.0],
        );
        let tmp = write_parquet(&batch);
        let rows = parse_external_load_scenarios(tmp.path()).unwrap();

        assert_eq!(rows.len(), 4);

        // Verify sorted by (stage_id, scenario_id, bus_id)
        for w in rows.windows(2) {
            let a = &w[0];
            let b = &w[1];
            let cmp = a
                .stage_id
                .cmp(&b.stage_id)
                .then_with(|| a.scenario_id.cmp(&b.scenario_id))
                .then_with(|| a.bus_id.0.cmp(&b.bus_id.0));
            assert!(cmp != std::cmp::Ordering::Greater, "rows not sorted");
        }

        // First row after sorting: stage=0, scenario=0, bus=1
        assert_eq!(rows[0].stage_id, 0);
        assert_eq!(rows[0].scenario_id, 0);
        assert_eq!(rows[0].bus_id, EntityId::from(1));
        assert!((rows[0].value_mw - 300.0).abs() < 1e-10);
    }

    #[test]
    fn test_parse_external_load_missing_bus_id_column() {
        let schema_no_bus = Arc::new(Schema::new(vec![
            Field::new("stage_id", DataType::Int32, false),
            Field::new("scenario_id", DataType::Int32, false),
            Field::new("value_mw", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema_no_bus,
            vec![
                Arc::new(Int32Array::from(vec![0_i32])),
                Arc::new(Int32Array::from(vec![0_i32])),
                Arc::new(Float64Array::from(vec![100.0_f64])),
            ],
        )
        .unwrap();
        let tmp = write_parquet(&batch);
        let err = parse_external_load_scenarios(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(field.contains("bus_id"), "field: {field}");
                assert!(
                    message.contains("missing required column"),
                    "message: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    #[test]
    fn test_parse_external_load_nonfinite_value_rejected() {
        let batch = make_load_batch(&[0], &[0], &[1], &[f64::NAN]);
        let tmp = write_parquet(&batch);
        let err = parse_external_load_scenarios(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(field.contains("value_mw"), "field: {field}");
                assert!(message.contains("finite"), "message: {message}");
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    #[test]
    fn test_load_external_load_scenarios_none_returns_empty() {
        let result = super::super::load_external_load_scenarios(None).unwrap();
        assert!(result.is_empty(), "expected empty vec for None path");
    }

    #[test]
    fn test_parse_external_ncs_valid() {
        let batch = make_ncs_batch(
            &[0, 1, 0, 1], // stage_id
            &[1, 0, 0, 1], // scenario_id
            &[2, 1, 3, 1], // ncs_id
            &[0.9, 0.8, 0.7, 0.85],
        );
        let tmp = write_parquet(&batch);
        let rows = parse_external_ncs_scenarios(tmp.path()).unwrap();

        assert_eq!(rows.len(), 4);

        // Verify sorted by (stage_id, scenario_id, ncs_id)
        for w in rows.windows(2) {
            let a = &w[0];
            let b = &w[1];
            let cmp = a
                .stage_id
                .cmp(&b.stage_id)
                .then_with(|| a.scenario_id.cmp(&b.scenario_id))
                .then_with(|| a.ncs_id.0.cmp(&b.ncs_id.0));
            assert!(cmp != std::cmp::Ordering::Greater, "rows not sorted");
        }

        // First row after sorting: stage=0, scenario=0, ncs=3
        assert_eq!(rows[0].stage_id, 0);
        assert_eq!(rows[0].scenario_id, 0);
        assert_eq!(rows[0].ncs_id, EntityId::from(3));
        assert!((rows[0].value - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_parse_external_ncs_negative_scenario_id_rejected() {
        let batch = make_ncs_batch(&[0], &[-1], &[1], &[0.5]);
        let tmp = write_parquet(&batch);
        let err = parse_external_ncs_scenarios(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(field.contains("scenario_id"), "field: {field}");
                assert!(message.contains('0'), "message: {message}");
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    #[test]
    fn test_parse_external_ncs_infinity_value_rejected() {
        let batch = make_ncs_batch(&[0], &[0], &[1], &[f64::INFINITY]);
        let tmp = write_parquet(&batch);
        let err = parse_external_ncs_scenarios(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(field.contains("value"), "field: {field}");
                assert!(message.contains("finite"), "message: {message}");
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    #[test]
    fn test_load_external_ncs_scenarios_none_returns_empty() {
        let result = super::super::load_external_ncs_scenarios(None).unwrap();
        assert!(result.is_empty(), "expected empty vec for None path");
    }
}
