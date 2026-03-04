//! Parsing for `scenarios/external_scenarios.parquet` — pre-computed scenario values.
//!
//! [`parse_external_scenarios`] reads `scenarios/external_scenarios.parquet` and returns
//! a sorted `Vec<ExternalScenarioRow>` with one row per (stage, scenario, hydro) triple.
//!
//! ## Parquet schema (spec SS2.5)
//!
//! | Column        | Type   | Required | Description                      |
//! | ------------- | ------ | -------- | -------------------------------- |
//! | `stage_id`    | INT32  | Yes      | Stage ID                         |
//! | `scenario_id` | INT32  | Yes      | Scenario index (0-based)         |
//! | `hydro_id`    | INT32  | Yes      | Hydro plant ID                   |
//! | `value_m3s`   | DOUBLE | Yes      | Inflow value in m³/s             |
//!
//! ## Output ordering
//!
//! Rows are sorted by `(stage_id, scenario_id, hydro_id)` ascending.
//!
//! ## Validation
//!
//! Per-row constraints enforced by this parser:
//!
//! - All four columns must be present with the correct types.
//! - `value_m3s` must be finite (NaN and infinity are rejected).
//! - `scenario_id` must be >= 0 (0-based indexing).
//!
//! Deferred validations (not performed here, Epic 06):
//!
//! - `hydro_id` existence in the hydro registry — Layer 3, Epic 06.
//! - `stage_id` existence in the stages registry — Layer 3, Epic 06.
//! - Scenario count matching `stage.num_scenarios` — Layer 3/5, Epic 06.

use cobre_core::EntityId;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::fs::File;
use std::path::Path;

use crate::parquet_helpers::{extract_required_float64, extract_required_int32};
use crate::LoadError;

/// A single row from `scenarios/external_scenarios.parquet`.
///
/// Each row defines the pre-computed inflow value for one (stage, scenario, hydro)
/// triple. Used when [`SamplingScheme::External`](cobre_core::scenario::SamplingScheme)
/// is active.
///
/// # Examples
///
/// ```
/// use cobre_io::scenarios::ExternalScenarioRow;
/// use cobre_core::EntityId;
///
/// let row = ExternalScenarioRow {
///     stage_id: 0,
///     scenario_id: 2,
///     hydro_id: EntityId::from(5),
///     value_m3s: 320.5,
/// };
/// assert_eq!(row.scenario_id, 2);
/// assert!((row.value_m3s - 320.5).abs() < 1e-10);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ExternalScenarioRow {
    /// Stage index (0-based within `System::stages`).
    pub stage_id: i32,

    /// Scenario index (0-based). Must be >= 0.
    pub scenario_id: i32,

    /// Hydro plant this inflow value belongs to.
    pub hydro_id: EntityId,

    /// Pre-computed inflow value in m³/s. Must be finite.
    pub value_m3s: f64,
}

/// Parse `scenarios/external_scenarios.parquet` and return a sorted row table.
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
/// use cobre_io::scenarios::parse_external_scenarios;
/// use std::path::Path;
///
/// let rows = parse_external_scenarios(Path::new("scenarios/external_scenarios.parquet"))
///     .expect("valid external scenarios file");
/// println!("loaded {} external scenario rows", rows.len());
/// ```
pub fn parse_external_scenarios(path: &Path) -> Result<Vec<ExternalScenarioRow>, LoadError> {
    let file = File::open(path).map_err(|e| LoadError::io(path, e))?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| LoadError::parse(path, e.to_string()))?;

    let reader = builder
        .build()
        .map_err(|e| LoadError::parse(path, e.to_string()))?;

    let mut rows: Vec<ExternalScenarioRow> = Vec::new();

    for batch_result in reader {
        let batch = batch_result.map_err(|e| LoadError::parse(path, e.to_string()))?;

        // ── Required columns ──────────────────────────────────────────────────
        let stage_id_col = extract_required_int32(&batch, "stage_id", path)?;
        let scenario_id_col = extract_required_int32(&batch, "scenario_id", path)?;
        let hydro_id_col = extract_required_int32(&batch, "hydro_id", path)?;
        let value_m3s_col = extract_required_float64(&batch, "value_m3s", path)?;

        // ── Build rows with per-row validation ────────────────────────────────
        let n = batch.num_rows();
        let base_idx = rows.len();
        rows.reserve(n);

        for i in 0..n {
            let row_idx = base_idx + i;

            let stage_id = stage_id_col.value(i);
            let scenario_id = scenario_id_col.value(i);
            let hydro_id = EntityId::from(hydro_id_col.value(i));
            let value_m3s = value_m3s_col.value(i);

            // Validate scenario_id: must be >= 0 (0-based indexing).
            if scenario_id < 0 {
                return Err(LoadError::SchemaError {
                    path: path.to_path_buf(),
                    field: format!("external_scenarios[{row_idx}].scenario_id"),
                    message: format!("scenario_id must be >= 0 (0-based), got {scenario_id}"),
                });
            }

            // Validate value_m3s: must be finite (no NaN or infinity).
            if !value_m3s.is_finite() {
                return Err(LoadError::SchemaError {
                    path: path.to_path_buf(),
                    field: format!("external_scenarios[{row_idx}].value_m3s"),
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

    // ── Sort by (stage_id, scenario_id, hydro_id) ascending ──────────────────
    rows.sort_by(|a, b| {
        a.stage_id
            .cmp(&b.stage_id)
            .then_with(|| a.scenario_id.cmp(&b.scenario_id))
            .then_with(|| a.hydro_id.0.cmp(&b.hydro_id.0))
    });

    Ok(rows)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

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

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("stage_id", DataType::Int32, false),
            Field::new("scenario_id", DataType::Int32, false),
            Field::new("hydro_id", DataType::Int32, false),
            Field::new("value_m3s", DataType::Float64, false),
        ]))
    }

    fn write_parquet(batch: &RecordBatch) -> NamedTempFile {
        let tmp = NamedTempFile::new().expect("tempfile");
        let mut writer = ArrowWriter::try_new(tmp.reopen().expect("reopen"), batch.schema(), None)
            .expect("ArrowWriter");
        writer.write(batch).expect("write batch");
        writer.close().expect("close writer");
        tmp
    }

    fn make_batch(
        stage_ids: &[i32],
        scenario_ids: &[i32],
        hydro_ids: &[i32],
        values: &[f64],
    ) -> RecordBatch {
        RecordBatch::try_new(
            schema(),
            vec![
                Arc::new(Int32Array::from(stage_ids.to_vec())),
                Arc::new(Int32Array::from(scenario_ids.to_vec())),
                Arc::new(Int32Array::from(hydro_ids.to_vec())),
                Arc::new(Float64Array::from(values.to_vec())),
            ],
        )
        .expect("valid batch")
    }

    // ── AC: 12 rows (2 stages x 3 scenarios x 2 hydros), verify sort order ───

    /// Valid file with 12 rows (2 stages x 3 scenarios x 2 hydros) in scrambled
    /// order. Result must be sorted by (stage_id, scenario_id, hydro_id).
    #[test]
    fn test_valid_12_rows_sorted_by_stage_scenario_hydro() {
        // Input: scrambled order. All stages, scenarios, and hydros mixed.
        // stage_ids, scenario_ids, hydro_ids, values — 12 rows.
        let batch = make_batch(
            &[1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0], // stage_id
            &[2, 0, 2, 0, 1, 1, 0, 0, 2, 2, 1, 1], // scenario_id
            &[2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1], // hydro_id
            &[
                120.0, 100.0, 110.0, 200.0, 105.0, 201.0, 205.0, 195.0, 115.0, 125.0, 210.0, 208.0,
            ],
        );
        let tmp = write_parquet(&batch);
        let rows = parse_external_scenarios(tmp.path()).unwrap();

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

    // ── AC: value_m3s NaN -> SchemaError ──────────────────────────────────────

    /// `value_m3s = NaN` -> SchemaError with field containing "value_m3s".
    #[test]
    fn test_nan_value_rejected() {
        let batch = make_batch(&[0], &[0], &[1], &[f64::NAN]);
        let tmp = write_parquet(&batch);
        let err = parse_external_scenarios(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("value_m3s"),
                    "field should contain 'value_m3s', got: {field}"
                );
                assert!(
                    message.contains("finite"),
                    "message should mention 'finite', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── AC: value_m3s infinity -> SchemaError ─────────────────────────────────

    /// `value_m3s = +inf` -> SchemaError with field containing "value_m3s".
    #[test]
    fn test_infinity_value_rejected() {
        let batch = make_batch(&[0], &[0], &[1], &[f64::INFINITY]);
        let tmp = write_parquet(&batch);
        let err = parse_external_scenarios(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("value_m3s"),
                    "field should contain 'value_m3s', got: {field}"
                );
                assert!(
                    message.contains("finite"),
                    "message should mention 'finite', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── AC: scenario_id negative -> SchemaError ───────────────────────────────

    /// `scenario_id = -1` -> SchemaError with field containing "scenario_id".
    #[test]
    fn test_negative_scenario_id_rejected() {
        let batch = make_batch(&[0], &[-1], &[1], &[100.0]);
        let tmp = write_parquet(&batch);
        let err = parse_external_scenarios(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("scenario_id"),
                    "field should contain 'scenario_id', got: {field}"
                );
                assert!(
                    message.contains('0'),
                    "message should mention >= 0, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── AC: missing hydro_id column -> SchemaError ────────────────────────────

    /// File missing `hydro_id` column -> SchemaError with field "hydro_id".
    #[test]
    fn test_missing_hydro_id_column() {
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
        let err = parse_external_scenarios(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("hydro_id"),
                    "field should contain 'hydro_id', got: {field}"
                );
                assert!(
                    message.contains("missing required column"),
                    "message should mention missing column, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── AC: empty file -> Ok(vec![]) ──────────────────────────────────────────

    /// Empty Parquet (0 rows) -> Ok(Vec::new()).
    #[test]
    fn test_empty_parquet_returns_empty_vec() {
        let batch = make_batch(&[], &[], &[], &[]);
        let tmp = write_parquet(&batch);
        let rows = parse_external_scenarios(tmp.path()).unwrap();
        assert!(rows.is_empty());
    }

    // ── AC: load_external_scenarios(None) -> Ok(vec![]) ──────────────────────

    /// `load_external_scenarios(None)` returns `Ok(Vec::new())` without I/O.
    #[test]
    fn test_load_external_scenarios_none_returns_empty() {
        let result = super::super::load_external_scenarios(None).unwrap();
        assert!(result.is_empty(), "expected empty vec for None path");
    }

    // ── AC: field values preserved ────────────────────────────────────────────

    /// Field values survive the Parquet round-trip.
    #[test]
    fn test_field_values_preserved() {
        let batch = make_batch(&[3], &[7], &[42], &[288.75]);
        let tmp = write_parquet(&batch);
        let rows = parse_external_scenarios(tmp.path()).unwrap();

        assert_eq!(rows.len(), 1);
        let row = &rows[0];
        assert_eq!(row.stage_id, 3);
        assert_eq!(row.scenario_id, 7);
        assert_eq!(row.hydro_id, EntityId::from(42));
        assert!((row.value_m3s - 288.75).abs() < 1e-10);
    }

    // ── AC: scenario_id = 0 is valid ──────────────────────────────────────────

    /// `scenario_id = 0` is the minimum valid value (0-based).
    #[test]
    fn test_scenario_id_zero_is_valid() {
        let batch = make_batch(&[0], &[0], &[1], &[150.0]);
        let tmp = write_parquet(&batch);
        let rows = parse_external_scenarios(tmp.path()).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].scenario_id, 0);
    }
}
