//! Parsing for `scenarios/inflow_history.parquet` — historical inflow
//! observation windows per hydro.
//!
//! [`parse_inflow_history`] reads `scenarios/inflow_history.parquet` and
//! returns a sorted `Vec<InflowHistoryRow>`. These windows are the
//! default-seeding record [`cobre_stochastic::derive_inflow_seeds`] casts
//! from, layered under `recent_observations` conditioning.
//!
//! ## Parquet schema (spec SS2.4)
//!
//! | Column       | Type   | Required | Description                       |
//! | ------------ | ------ | -------- | ---------------------------------- |
//! | `hydro_id`   | INT32  | Yes      | Hydro plant ID                     |
//! | `start_date` | DATE   | Yes      | Window start, inclusive (Date32)   |
//! | `end_date`   | DATE   | Yes      | Window end, exclusive (Date32)     |
//! | `value_m3s`  | DOUBLE | Yes      | Mean inflow over the window (m³/s) |
//!
//! A file carrying the legacy `date` column, or missing `start_date`/
//! `end_date`, is rejected outright — there is no inference/conversion path
//! from the legacy point-dated layout.
//!
//! ## Output ordering
//!
//! Rows are sorted by `(hydro_id, start_date)` ascending.
//!
//! ## Validation
//!
//! Per-row and per-hydro constraints enforced by this parser:
//!
//! - All four columns must be present with the correct types.
//! - `end_date` must be strictly after `start_date`.
//! - `value_m3s` must be finite. A negative value is accepted — the quantity is
//!   incremental inflow, a difference between a plant's natural flow and its
//!   upstream plants' — and Layer 5a reports the count.
//! - For a given `hydro_id`, windows must not overlap; adjacent windows
//!   (`start_date == previous end_date`) are accepted.
//!
//! Deferred validations (not performed here):
//!
//! - `hydro_id` existence in the hydro registry — Layer 3.
//! - Season/coverage alignment checks — Layer 4/5.

use std::path::Path;

use arrow::temporal_conversions::date32_to_datetime;
use chrono::NaiveDate;
use cobre_core::EntityId;

use crate::LoadError;
use crate::parquet_helpers::{
    extract_required_date32, extract_required_float64, extract_required_int32,
    open_record_batch_reader,
};
use crate::windowed_history::{WindowedRecord, validate_windowed_records};

pub use cobre_core::scenario::InflowHistoryRow;

const LEGACY_LAYOUT_MESSAGE: &str = "scenarios/inflow_history.parquet uses the legacy \
    point-dated layout; re-emit windowed columns \"start_date\"/\"end_date\" in place of \"date\"";

/// Parse `scenarios/inflow_history.parquet` and return a sorted row table.
///
/// # Errors
///
/// | Condition                                          | Error variant              |
/// |-----------------------------------------------------|--------------------------- |
/// | File not found or permission denied                 | [`LoadError::IoError`]     |
/// | Malformed Parquet (corrupt header, etc.)            | [`LoadError::ParseError`]  |
/// | `start_date`/`end_date` missing, or `date` present   | [`LoadError::SchemaError`] |
/// | Required column missing or wrong type                | [`LoadError::SchemaError`] |
/// | `end_date` is not strictly after `start_date`         | [`LoadError::SchemaError`] |
/// | `value_m3s` is NaN or infinite                        | [`LoadError::SchemaError`] |
/// | Two windows for the same `hydro_id` overlap           | [`LoadError::SchemaError`] |
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
    let reader = open_record_batch_reader(path)?;

    let mut rows: Vec<InflowHistoryRow> = Vec::new();

    for batch_result in reader {
        let batch = batch_result.map_err(|e| LoadError::parse(path, e.to_string()))?;

        if batch.column_by_name("date").is_some() {
            return Err(legacy_layout_error(path, "date"));
        }
        if batch.column_by_name("start_date").is_none() {
            return Err(legacy_layout_error(path, "start_date"));
        }
        if batch.column_by_name("end_date").is_none() {
            return Err(legacy_layout_error(path, "end_date"));
        }

        let hydro_id_col = extract_required_int32(&batch, "hydro_id", path)?;
        let start_date_col = extract_required_date32(&batch, "start_date", path)?;
        let end_date_col = extract_required_date32(&batch, "end_date", path)?;
        let value_col = extract_required_float64(&batch, "value_m3s", path)?;

        let n = batch.num_rows();
        let base_idx = rows.len();
        rows.reserve(n);

        for i in 0..n {
            let row_idx = base_idx + i;

            let hydro_id = EntityId::from(hydro_id_col.value(i));
            let start_date = parse_date32(start_date_col.value(i), path, row_idx, "start_date")?;
            let end_date = parse_date32(end_date_col.value(i), path, row_idx, "end_date")?;
            let value_m3s = value_col.value(i);

            rows.push(InflowHistoryRow {
                hydro_id,
                start_date,
                end_date,
                value_m3s,
            });
        }
    }

    rows.sort_by(|a, b| {
        a.hydro_id
            .0
            .cmp(&b.hydro_id.0)
            .then_with(|| a.start_date.cmp(&b.start_date))
    });

    let windows: Vec<WindowedRecord> = rows
        .iter()
        .map(|r| WindowedRecord {
            entity_id: r.hydro_id.0,
            start_date: r.start_date,
            end_date: r.end_date,
            value: r.value_m3s,
        })
        .collect();
    validate_windowed_records(&windows, "inflow_history", "hydro_id", "value_m3s", path)?;

    Ok(rows)
}

fn legacy_layout_error(path: &Path, field: &str) -> LoadError {
    LoadError::SchemaError {
        path: path.to_path_buf(),
        field: field.to_string(),
        message: LEGACY_LAYOUT_MESSAGE.to_string(),
    }
}

fn parse_date32(
    raw: i32,
    path: &Path,
    row_idx: usize,
    field_name: &str,
) -> Result<NaiveDate, LoadError> {
    // Arrow Date32 stores days since the Unix epoch (1970-01-01).
    date32_to_datetime(raw)
        .map(|dt| dt.date())
        .ok_or_else(|| LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("inflow_history[{row_idx}].{field_name}"),
            message: format!("cannot convert date32 value {raw} to a valid calendar date"),
        })
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
    use crate::test_support::write_parquet;
    use arrow::array::{Date32Array, Float64Array, Int32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };
    use std::sync::Arc;

    fn schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("hydro_id", DataType::Int32, false),
            Field::new("start_date", DataType::Date32, false),
            Field::new("end_date", DataType::Date32, false),
            Field::new("value_m3s", DataType::Float64, false),
        ]))
    }

    fn legacy_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("hydro_id", DataType::Int32, false),
            Field::new("date", DataType::Date32, false),
            Field::new("value_m3s", DataType::Float64, false),
        ]))
    }

    fn naive_date_to_date32(date: NaiveDate) -> i32 {
        let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
        i32::try_from((date - epoch).num_days()).expect("date out of Date32 range")
    }

    fn make_batch(
        hydro_ids: &[i32],
        start_dates: &[i32],
        end_dates: &[i32],
        values: &[f64],
    ) -> RecordBatch {
        RecordBatch::try_new(
            schema(),
            vec![
                Arc::new(Int32Array::from(hydro_ids.to_vec())),
                Arc::new(Date32Array::from(start_dates.to_vec())),
                Arc::new(Date32Array::from(end_dates.to_vec())),
                Arc::new(Float64Array::from(values.to_vec())),
            ],
        )
        .expect("valid batch")
    }

    fn make_legacy_batch(hydro_ids: &[i32], dates: &[i32], values: &[f64]) -> RecordBatch {
        RecordBatch::try_new(
            legacy_schema(),
            vec![
                Arc::new(Int32Array::from(hydro_ids.to_vec())),
                Arc::new(Date32Array::from(dates.to_vec())),
                Arc::new(Float64Array::from(values.to_vec())),
            ],
        )
        .expect("valid legacy batch")
    }

    /// One calendar-month window `[start, start + 1 month)`, encoded as Date32.
    fn month_window(year: i32, month: u32) -> (i32, i32) {
        let start = NaiveDate::from_ymd_opt(year, month, 1).unwrap();
        let end = start.checked_add_months(chrono::Months::new(1)).unwrap();
        (naive_date_to_date32(start), naive_date_to_date32(end))
    }

    #[test]
    fn test_windowed_inflow_history_parses_and_sorts() {
        // 2 hydros x 12 monthly windows in 2000, deliberately scrambled
        // (hydro 2 before hydro 1) to confirm the (hydro_id, start_date) sort.
        let windows: Vec<(i32, i32)> = (1..=12).map(|m| month_window(2000, m)).collect();

        let mut hydro_ids = vec![2_i32; 12];
        hydro_ids.extend(vec![1_i32; 12]);
        let mut start_vals: Vec<i32> = windows.iter().map(|(s, _)| *s).collect();
        start_vals.extend(windows.iter().map(|(s, _)| *s));
        let mut end_vals: Vec<i32> = windows.iter().map(|(_, e)| *e).collect();
        end_vals.extend(windows.iter().map(|(_, e)| *e));
        let values = vec![500.0_f64; 24];

        let batch = make_batch(&hydro_ids, &start_vals, &end_vals, &values);
        let tmp = write_parquet(&batch);
        let rows = parse_inflow_history(tmp.path()).unwrap();

        assert_eq!(rows.len(), 24, "expected 24 rows");
        rows.iter()
            .take(12)
            .for_each(|r| assert_eq!(r.hydro_id, EntityId::from(1)));
        rows.iter()
            .skip(12)
            .for_each(|r| assert_eq!(r.hydro_id, EntityId::from(2)));
        for r in &rows {
            assert!(
                r.end_date > r.start_date,
                "end_date must be after start_date, got {r:?}"
            );
        }
        for w in rows[..12].windows(2) {
            assert!(w[0].start_date < w[1].start_date);
        }
    }

    #[test]
    fn test_legacy_date_column_rejected_with_message() {
        let date = naive_date_to_date32(NaiveDate::from_ymd_opt(2000, 1, 1).unwrap());
        let batch = make_legacy_batch(&[1], &[date], &[500.0]);
        let tmp = write_parquet(&batch);
        let err = parse_inflow_history(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { message, .. } => {
                assert!(
                    message.contains("legacy point-dated layout; re-emit windowed"),
                    "message should mention the legacy layout, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    #[test]
    fn test_missing_windowed_columns_rejected_as_legacy() {
        let schema_no_dates = Arc::new(Schema::new(vec![
            Field::new("hydro_id", DataType::Int32, false),
            Field::new("value_m3s", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema_no_dates,
            vec![
                Arc::new(Int32Array::from(vec![1_i32])),
                Arc::new(Float64Array::from(vec![500.0])),
            ],
        )
        .unwrap();
        let tmp = write_parquet(&batch);
        let err = parse_inflow_history(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { message, .. } => {
                assert!(
                    message.contains("legacy point-dated layout; re-emit windowed"),
                    "message should mention the legacy layout, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    #[test]
    fn test_windowed_overlap_rejected() {
        let (jan_start, jan_end) = month_window(2000, 1);
        // Second window starts inside January's window (Jan 15) — overlaps.
        let overlap_start = naive_date_to_date32(NaiveDate::from_ymd_opt(2000, 1, 15).unwrap());
        let (_, feb_end) = month_window(2000, 2);

        let batch = make_batch(
            &[1, 1],
            &[jan_start, overlap_start],
            &[jan_end, feb_end],
            &[500.0, 510.0],
        );
        let tmp = write_parquet(&batch);
        let err = parse_inflow_history(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { message, .. } => {
                assert!(
                    message.contains("overlapping windows") && message.contains("hydro_id 1"),
                    "message should name the overlapping hydro, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    #[test]
    fn test_windowed_adjacent_windows_accepted() {
        let (jan_start, jan_end) = month_window(2000, 1);
        let (feb_start, feb_end) = month_window(2000, 2);
        assert_eq!(jan_end, feb_start, "adjacent windows share a boundary");

        let batch = make_batch(
            &[1, 1],
            &[jan_start, feb_start],
            &[jan_end, feb_end],
            &[500.0, 510.0],
        );
        let tmp = write_parquet(&batch);
        let rows = parse_inflow_history(tmp.path()).unwrap();
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn test_inverted_window_rejected() {
        let start = naive_date_to_date32(NaiveDate::from_ymd_opt(2000, 1, 15).unwrap());
        let end = naive_date_to_date32(NaiveDate::from_ymd_opt(2000, 1, 1).unwrap());
        let batch = make_batch(&[1], &[start], &[end], &[500.0]);
        let tmp = write_parquet(&batch);
        let err = parse_inflow_history(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(field.contains("end_date"));
                assert!(
                    message.contains("end_date must be after start_date"),
                    "got: {message}"
                );
                assert!(
                    message.contains("hydro_id 1"),
                    "message should name the entity, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    #[test]
    fn test_infinite_value_m3s() {
        let (start, end) = month_window(2000, 1);
        let batch = make_batch(&[1], &[start], &[end], &[f64::INFINITY]);
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

    #[test]
    fn test_negative_value_m3s_accepted() {
        let (start, end) = month_window(2000, 1);
        let batch = make_batch(&[1], &[start], &[end], &[-1.0]);
        let tmp = write_parquet(&batch);
        let rows = parse_inflow_history(tmp.path()).unwrap();

        assert_eq!(rows.len(), 1);
        assert!((rows[0].value_m3s - (-1.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_empty_parquet_returns_empty_vec() {
        let batch = make_batch(&[], &[], &[], &[]);
        let tmp = write_parquet(&batch);
        let rows = parse_inflow_history(tmp.path()).unwrap();
        assert!(rows.is_empty());
    }

    #[test]
    fn test_window_dates_preserved() {
        let expected_start = NaiveDate::from_ymd_opt(2024, 6, 15).unwrap();
        let expected_end = NaiveDate::from_ymd_opt(2024, 7, 1).unwrap();
        let start_val = naive_date_to_date32(expected_start);
        let end_val = naive_date_to_date32(expected_end);
        let batch = make_batch(&[7], &[start_val], &[end_val], &[250.5]);
        let tmp = write_parquet(&batch);
        let rows = parse_inflow_history(tmp.path()).unwrap();

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].hydro_id, EntityId::from(7));
        assert_eq!(rows[0].start_date, expected_start);
        assert_eq!(rows[0].end_date, expected_end);
        assert!((rows[0].value_m3s - 250.5).abs() < 1e-10);
    }

    #[test]
    fn test_nan_value_m3s() {
        let (start, end) = month_window(2000, 1);
        let batch = make_batch(&[1], &[start], &[end], &[f64::NAN]);
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

    /// Build 12 monthly study stages with `season_ids` 0-11, covering `year`.
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    fn twelve_monthly_stages(year: i32) -> Vec<Stage> {
        (0_usize..12)
            .map(|i| {
                let month = (i as u32 % 12) + 1;
                let (end_year, end_month) = if month == 12 {
                    (year + 1, 1)
                } else {
                    (year, month + 1)
                };
                Stage {
                    index: i,
                    id: i as i32,
                    start_date: NaiveDate::from_ymd_opt(year, month, 1).unwrap(),
                    end_date: NaiveDate::from_ymd_opt(end_year, end_month, 1).unwrap(),
                    season_id: Some(i),
                    blocks: vec![Block {
                        index: 0,
                        name: "SINGLE".to_string(),
                        duration_hours: 720.0,
                    }],
                    block_mode: BlockMode::Parallel,
                    state_config: StageStateConfig {
                        storage: true,
                        inflow_lags: false,
                    },
                    risk_config: StageRiskConfig::Expectation,
                    scenario_config: ScenarioSourceConfig {
                        branching_factor: 1,
                        noise_method: NoiseMethod::Saa,
                    },
                }
            })
            .collect()
    }

    /// Full-coverage monthly windowed history for `hydro_id` spanning
    /// `[from_year, to_year]`: each row is exactly `[month start, next month
    /// start)`, matching a prior point-dated fixture one-for-one.
    fn full_coverage_monthly_history(
        hydro_id: EntityId,
        from_year: i32,
        to_year: i32,
    ) -> Vec<InflowHistoryRow> {
        (from_year..=to_year)
            .flat_map(|y| {
                (1u32..=12).map(move |m| {
                    let start_date = NaiveDate::from_ymd_opt(y, m, 1).unwrap();
                    let end_date = start_date
                        .checked_add_months(chrono::Months::new(1))
                        .unwrap();
                    InflowHistoryRow {
                        hydro_id,
                        start_date,
                        end_date,
                        value_m3s: 100.0,
                    }
                })
            })
            .collect()
    }

    /// A full-coverage windowed history — where every window is exactly one
    /// calendar month `[start, next month start)` with no gaps — must yield
    /// the identical discovered-window set a prior point-dated fixture
    /// produced, since `discover_historical_windows` keys occurrence
    /// resolution off each window's `start_date` alone.
    #[test]
    fn test_windowed_full_coverage_matches_pointdated_discovery() {
        let hydro1 = EntityId::from(1);
        let hydro2 = EntityId::from(2);
        let mut history = full_coverage_monthly_history(hydro1, 1990, 2010);
        history.extend(full_coverage_monthly_history(hydro2, 1990, 2010));

        let stages = twelve_monthly_stages(2024);
        let windows = cobre_stochastic::discover_historical_windows(
            &history,
            &[hydro1, hydro2],
            &stages,
            None,
            None,
            10,
        )
        .unwrap();

        let expected: Vec<i32> = (1990..=2010).collect();
        assert_eq!(
            windows, expected,
            "full-coverage windowed history must reproduce the prior point-dated \
             discovery result exactly"
        );
    }
}
