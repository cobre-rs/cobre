//! Parquet writer for the run-level fixed post-horizon commitment echo.
//!
//! [`write_fixed_delivery`] exports a slice of [`FixedDeliveryRow`] — one row
//! per anticipated plant × declared fixed window — to
//! `anticipated/fixed_deliveries.parquet`. The on-disk shape is owned by
//! `fixed_delivery_schema` in [`crate::output::schemas`].
//!
//! An empty slice writes nothing — no file, no `anticipated/` directory — so a
//! run with no fixed window declared keeps a byte-identical output tree. All
//! writes are atomic: data is first written to a `.tmp` suffix, then renamed.

use std::path::Path;
use std::sync::Arc;

use arrow::array::{Date32Builder, Float64Builder, Int32Builder, RecordBatch};
use chrono::NaiveDate;

use super::date32_days;

use crate::output::atomic::write_batch_atomic;
use crate::output::error::OutputError;
use crate::output::schemas::fixed_delivery_schema;

pub(crate) const FIXED_DELIVERIES_FILE: &str = "anticipated/fixed_deliveries.parquet";

/// One row of the run-level fixed post-horizon commitment echo.
///
/// Each row is one anticipated plant's declared fixed (class-4) commitment
/// window: its real delivery span and committed MW. Values are echoed from
/// resolved input, so they are scenario- and stage-independent.
#[derive(Debug, Clone)]
pub struct FixedDeliveryRow {
    /// Anticipated thermal plant id.
    pub thermal_id: i32,
    /// First delivery date of the fixed window.
    pub start_date: NaiveDate,
    /// Last delivery date of the fixed window.
    pub end_date: NaiveDate,
    /// Committed delivery, in MW.
    pub value_mw: f64,
}

/// Write a slice of [`FixedDeliveryRow`] to
/// `<output_dir>/anticipated/fixed_deliveries.parquet`, in the order given.
///
/// An empty slice writes nothing — no file and no `anticipated/` directory —
/// and returns `Ok(())`, so a run with no fixed window keeps a byte-identical
/// output tree. Otherwise the `anticipated/` parent is created if absent and the
/// write is atomic.
///
/// # Errors
///
/// - [`OutputError::IoError`] — directory creation, file open, or rename fails.
/// - [`OutputError::SerializationError`] — Arrow/Parquet construction fails.
///
/// # Examples
///
/// ```no_run
/// use cobre_io::{FixedDeliveryRow, write_fixed_delivery};
/// use chrono::NaiveDate;
/// use std::path::Path;
///
/// # fn main() -> Result<(), cobre_io::OutputError> {
/// let rows = vec![FixedDeliveryRow {
///     thermal_id: 3,
///     start_date: NaiveDate::from_ymd_opt(2030, 1, 1).expect("valid date"),
///     end_date: NaiveDate::from_ymd_opt(2030, 6, 30).expect("valid date"),
///     value_mw: 120.5,
/// }];
/// write_fixed_delivery(Path::new("/tmp/out"), &rows)?;
/// # Ok(())
/// # }
/// ```
pub fn write_fixed_delivery(
    output_dir: &Path,
    rows: &[FixedDeliveryRow],
) -> Result<(), OutputError> {
    if rows.is_empty() {
        return Ok(());
    }
    let path = output_dir.join(FIXED_DELIVERIES_FILE);
    let batch = build_fixed_delivery_batch(rows)?;
    write_batch_atomic(&path, &batch)
}

fn build_fixed_delivery_batch(rows: &[FixedDeliveryRow]) -> Result<RecordBatch, OutputError> {
    let n = rows.len();

    let mut thermal_id_col = Int32Builder::with_capacity(n);
    let mut start_date_col = Date32Builder::with_capacity(n);
    let mut end_date_col = Date32Builder::with_capacity(n);
    let mut value_mw_col = Float64Builder::with_capacity(n);

    for row in rows {
        thermal_id_col.append_value(row.thermal_id);
        start_date_col.append_value(date32_days(row.start_date));
        end_date_col.append_value(date32_days(row.end_date));
        value_mw_col.append_value(row.value_mw);
    }

    RecordBatch::try_new(
        Arc::new(fixed_delivery_schema()),
        vec![
            Arc::new(thermal_id_col.finish()),
            Arc::new(start_date_col.finish()),
            Arc::new(end_date_col.finish()),
            Arc::new(value_mw_col.finish()),
        ],
    )
    .map_err(|e| OutputError::serialization("fixed_delivery", e.to_string()))
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::float_cmp, clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::test_support::output::read_first_batch;
    use arrow::array::{Date32Array, Float64Array, Int32Array};
    use tempfile::tempdir;

    fn sample_rows() -> Vec<FixedDeliveryRow> {
        vec![
            FixedDeliveryRow {
                thermal_id: 3,
                start_date: NaiveDate::from_ymd_opt(2030, 1, 1).unwrap(),
                end_date: NaiveDate::from_ymd_opt(2030, 6, 30).unwrap(),
                value_mw: 120.5,
            },
            FixedDeliveryRow {
                thermal_id: 7,
                start_date: NaiveDate::from_ymd_opt(2031, 7, 1).unwrap(),
                end_date: NaiveDate::from_ymd_opt(2031, 12, 31).unwrap(),
                value_mw: 88.0,
            },
        ]
    }

    #[test]
    fn fixed_delivery_round_trip_preserves_rows_in_order() {
        let rows = sample_rows();
        let root = tempdir().expect("tempdir");
        write_fixed_delivery(root.path(), &rows).expect("write must succeed");

        let path = root
            .path()
            .join("anticipated")
            .join("fixed_deliveries.parquet");
        assert!(path.exists(), "file must exist after non-empty write");

        let batch = read_first_batch(&path);
        assert_eq!(batch.num_columns(), 4, "must have 4 columns");
        assert_eq!(batch.num_rows(), rows.len(), "one row per input record");

        let thermal_id = batch
            .column_by_name("thermal_id")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let start_date = batch
            .column_by_name("start_date")
            .unwrap()
            .as_any()
            .downcast_ref::<Date32Array>()
            .unwrap();
        let end_date = batch
            .column_by_name("end_date")
            .unwrap()
            .as_any()
            .downcast_ref::<Date32Array>()
            .unwrap();
        let value_mw = batch
            .column_by_name("value_mw")
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();

        for (i, row) in rows.iter().enumerate() {
            assert_eq!(thermal_id.value(i), row.thermal_id);
            assert_eq!(start_date.value(i), date32_days(row.start_date));
            assert_eq!(end_date.value(i), date32_days(row.end_date));
            assert!(value_mw.value(i) == row.value_mw);
        }
    }

    #[test]
    fn fixed_delivery_empty_slice_writes_no_file_and_no_directory() {
        let root = tempdir().expect("tempdir");
        write_fixed_delivery(root.path(), &[]).expect("empty write must succeed");

        let anticipated_dir = root.path().join("anticipated");
        assert!(
            !anticipated_dir.exists(),
            "empty slice must not create the anticipated/ directory"
        );
        assert!(
            !anticipated_dir.join("fixed_deliveries.parquet").exists(),
            "empty slice must not write the parquet file"
        );
    }
}
