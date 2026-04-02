//! Parsing for `scenarios/inflow_history.parquet` — raw historical inflow
//! observations per (hydro, date).
//!
//! [`parse_inflow_history`] reads `scenarios/inflow_history.parquet` and
//! returns a sorted `Vec<InflowHistoryRow>`.
//!
//! ## Parquet schema (spec SS2.4)
//!
//! | Column       | Type   | Required | Description                      |
//! | ------------ | ------ | -------- | -------------------------------- |
//! | `hydro_id`   | INT32  | Yes      | Hydro plant ID                   |
//! | `date`       | DATE   | Yes      | Observation date (Date32)        |
//! | `value_m3s`  | DOUBLE | Yes      | Mean inflow (m³/s)               |
//!
//! ## Output ordering
//!
//! Rows are sorted by `(hydro_id, date)` ascending.
//!
//! ## Validation
//!
//! Per-row constraints enforced by this parser:
//!
//! - All three columns must be present with the correct types.
//! - `value_m3s` must be finite (NaN and ±inf are rejected).
//!
//! Deferred validations (not performed here):
//!
//! - `hydro_id` existence in the hydro registry — Layer 3, Epic 06.
//! - Date coverage checks — Layer 4/5, Epic 06.

use arrow::temporal_conversions::date32_to_datetime;
use cobre_core::EntityId;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::fs::File;
use std::path::Path;

use crate::parquet_helpers::{
    extract_required_date32, extract_required_float64, extract_required_int32,
};
use crate::LoadError;

/// A single row from `scenarios/inflow_history.parquet`.
///
/// Carries one historical inflow observation for a (hydro, date) pair.
/// These rows constitute the raw historical record used by PAR(p) fitting
/// routines in `cobre-stochastic`.
///
/// # Examples
///
/// ```
/// use cobre_io::scenarios::InflowHistoryRow;
/// use cobre_core::EntityId;
/// use chrono::NaiveDate;
///
/// let row = InflowHistoryRow {
///     hydro_id: EntityId::from(1),
///     date: NaiveDate::from_ymd_opt(2000, 1, 1).unwrap(),
///     value_m3s: 500.0,
/// };
/// assert_eq!(row.hydro_id, EntityId::from(1));
/// assert_eq!(row.value_m3s, 500.0);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct InflowHistoryRow {
    /// Hydro plant this observation belongs to.
    pub hydro_id: EntityId,
    /// Date of the observation (timezone-free calendar date).
    pub date: chrono::NaiveDate,
    /// Mean inflow for this observation period in m³/s. Must be finite.
    pub value_m3s: f64,
}

/// Parse `scenarios/inflow_history.parquet` and return a sorted row table.
///
/// Reads all record batches from the Parquet file at `path`, validates per-row
/// constraints, then returns all rows sorted by `(hydro_id, date)` ascending.
///
/// # Errors
///
/// | Condition                                     | Error variant              |
/// |---------------------------------------------- |--------------------------- |
/// | File not found or permission denied           | [`LoadError::IoError`]     |
/// | Malformed Parquet (corrupt header, etc.)      | [`LoadError::ParseError`]  |
/// | Required column missing or wrong type         | [`LoadError::SchemaError`] |
/// | `value_m3s` is NaN or infinite                | [`LoadError::SchemaError`] |
///
/// # Examples
///
/// ```no_run
/// use cobre_io::scenarios::parse_inflow_history;
/// use std::path::Path;
///
/// let rows = parse_inflow_history(Path::new("scenarios/inflow_history.parquet"))
///     .expect("valid inflow history file");
/// println!("loaded {} inflow history rows", rows.len());
/// ```
pub fn parse_inflow_history(path: &Path) -> Result<Vec<InflowHistoryRow>, LoadError> {
    let file = File::open(path).map_err(|e| LoadError::io(path, e))?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| LoadError::parse(path, e.to_string()))?;

    let reader = builder
        .build()
        .map_err(|e| LoadError::parse(path, e.to_string()))?;

    let mut rows: Vec<InflowHistoryRow> = Vec::new();

    for batch_result in reader {
        let batch = batch_result.map_err(|e| LoadError::parse(path, e.to_string()))?;

        // ── Required columns ──────────────────────────────────────────────────
        let hydro_id_col = extract_required_int32(&batch, "hydro_id", path)?;
        let date_col = extract_required_date32(&batch, "date", path)?;
        let value_col = extract_required_float64(&batch, "value_m3s", path)?;

        // ── Build rows with per-row validation ────────────────────────────────
        let n = batch.num_rows();
        let base_idx = rows.len();
        rows.reserve(n);

        for i in 0..n {
            let row_idx = base_idx + i;

            let hydro_id = EntityId::from(hydro_id_col.value(i));

            // Arrow Date32 stores days since Unix epoch (1970-01-01).
            // date32_to_datetime converts to NaiveDateTime; .date() gives NaiveDate.
            let date = date32_to_datetime(date_col.value(i))
                .ok_or_else(|| LoadError::SchemaError {
                    path: path.to_path_buf(),
                    field: format!("inflow_history[{row_idx}].date"),
                    message: format!(
                        "cannot convert date32 value {} to a valid calendar date",
                        date_col.value(i)
                    ),
                })?
                .date();

            let value_m3s = value_col.value(i);

            // Validate value_m3s: must be finite.
            if !value_m3s.is_finite() {
                return Err(LoadError::SchemaError {
                    path: path.to_path_buf(),
                    field: format!("inflow_history[{row_idx}].value_m3s"),
                    message: format!("value must be finite, got {value_m3s}"),
                });
            }

            rows.push(InflowHistoryRow {
                hydro_id,
                date,
                value_m3s,
            });
        }
    }

    // ── Sort by (hydro_id, date) ascending ────────────────────────────────────
    rows.sort_by(|a, b| {
        a.hydro_id
            .0
            .cmp(&b.hydro_id.0)
            .then_with(|| a.date.cmp(&b.date))
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
    use arrow::array::{Date32Array, Float64Array, Int32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use chrono::NaiveDate;
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;
    use tempfile::NamedTempFile;

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("hydro_id", DataType::Int32, false),
            Field::new("date", DataType::Date32, false),
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

    /// Convert a `NaiveDate` to a Date32 value (days since Unix epoch 1970-01-01).
    fn naive_date_to_date32(date: NaiveDate) -> i32 {
        let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
        i32::try_from((date - epoch).num_days()).expect("date out of Date32 range")
    }

    fn make_batch(hydro_ids: &[i32], dates: &[i32], values: &[f64]) -> RecordBatch {
        RecordBatch::try_new(
            schema(),
            vec![
                Arc::new(Int32Array::from(hydro_ids.to_vec())),
                Arc::new(Date32Array::from(dates.to_vec())),
                Arc::new(Float64Array::from(values.to_vec())),
            ],
        )
        .expect("valid batch")
    }

    // ── AC: valid 24 rows (2 hydros x 12 dates), verify sort ─────────────────

    /// Valid file with 2 hydros × 12 dates = 24 rows, scrambled input.
    /// Result: 24 rows sorted by (hydro_id, date).
    #[test]
    fn test_valid_24_rows_sorted_by_hydro_date() {
        let base_date = NaiveDate::from_ymd_opt(2000, 1, 1).unwrap();
        let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();

        // Build 12 monthly dates starting from 2000-01-01.
        let dates_hydro_1: Vec<i32> = (0..12)
            .map(|m| {
                let d = NaiveDate::from_ymd_opt(2000, 1 + m, 1).unwrap_or(base_date);
                i32::try_from((d - epoch).num_days()).expect("date out of Date32 range")
            })
            .collect();
        let dates_hydro_2 = dates_hydro_1.clone();

        // Interleave: hydro 2 dates then hydro 1 dates (scrambled order).
        let mut hydro_ids = vec![2_i32; 12];
        hydro_ids.extend(vec![1_i32; 12]);
        let mut date_vals: Vec<i32> = dates_hydro_2;
        date_vals.extend(dates_hydro_1.iter().copied());
        let values = vec![500.0_f64; 24];

        let batch = make_batch(&hydro_ids, &date_vals, &values);
        let tmp = write_parquet(&batch);
        let rows = parse_inflow_history(tmp.path()).unwrap();

        assert_eq!(rows.len(), 24, "expected 24 rows");
        rows.iter()
            .take(12)
            .for_each(|r| assert_eq!(r.hydro_id, EntityId::from(1)));
        rows.iter()
            .skip(12)
            .for_each(|r| assert_eq!(r.hydro_id, EntityId::from(2)));
        for w in rows[..12].windows(2) {
            assert!(w[0].date < w[1].date);
        }
    }

    // ── AC: value_m3s infinity -> SchemaError ─────────────────────────────────

    /// `value_m3s = +inf` -> SchemaError with field path containing "value_m3s".
    #[test]
    fn test_infinite_value_m3s() {
        let date = naive_date_to_date32(NaiveDate::from_ymd_opt(2000, 1, 1).unwrap());
        let batch = make_batch(&[1], &[date], &[f64::INFINITY]);
        let tmp = write_parquet(&batch);
        let err = parse_inflow_history(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, .. } => {
                assert!(
                    field.contains("value_m3s"),
                    "field should contain 'value_m3s', got: {field}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── AC: missing date column -> SchemaError ─────────────────────────────────

    /// File missing `date` column -> SchemaError with field "date".
    #[test]
    fn test_missing_date_column() {
        let schema_no_date = Arc::new(Schema::new(vec![
            Field::new("hydro_id", DataType::Int32, false),
            Field::new("value_m3s", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema_no_date,
            vec![
                Arc::new(Int32Array::from(vec![1_i32])),
                Arc::new(Float64Array::from(vec![500.0])),
            ],
        )
        .unwrap();
        let tmp = write_parquet(&batch);
        let err = parse_inflow_history(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("date"),
                    "field should contain 'date', got: {field}"
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
        let batch = make_batch(&[], &[], &[]);
        let tmp = write_parquet(&batch);
        let rows = parse_inflow_history(tmp.path()).unwrap();
        assert!(rows.is_empty());
    }

    // ── AC: date values round-tripped correctly ────────────────────────────────

    /// Date values survive the Parquet round-trip.
    #[test]
    fn test_date_values_preserved() {
        let expected_date = NaiveDate::from_ymd_opt(2024, 6, 15).unwrap();
        let date32_val = naive_date_to_date32(expected_date);
        let batch = make_batch(&[7], &[date32_val], &[250.5]);
        let tmp = write_parquet(&batch);
        let rows = parse_inflow_history(tmp.path()).unwrap();

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].hydro_id, EntityId::from(7));
        assert_eq!(rows[0].date, expected_date);
        assert!((rows[0].value_m3s - 250.5).abs() < 1e-10);
    }

    // ── AC: NaN value_m3s -> SchemaError ─────────────────────────────────────

    /// `value_m3s = NaN` -> SchemaError.
    #[test]
    fn test_nan_value_m3s() {
        let date = naive_date_to_date32(NaiveDate::from_ymd_opt(2000, 1, 1).unwrap());
        let batch = make_batch(&[1], &[date], &[f64::NAN]);
        let tmp = write_parquet(&batch);
        let err = parse_inflow_history(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, .. } => {
                assert!(
                    field.contains("value_m3s"),
                    "field should contain 'value_m3s', got: {field}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }
}
