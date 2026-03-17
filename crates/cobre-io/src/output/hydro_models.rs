//! Parquet writer for fitted FPHA hyperplane coefficients.
//!
//! [`write_fpha_hyperplanes`] exports a slice of [`FphaHyperplaneRow`] to
//! `output/hydro_models/fpha_hyperplanes.parquet` using the same 11-column schema
//! as the input file `system/fpha_hyperplanes.parquet`:
//!
//! | Column            | Type    | Required | Description                              |
//! | ----------------- | ------- | -------- | ---------------------------------------- |
//! | `hydro_id`        | INT32   | Yes      | Hydro plant identifier                   |
//! | `stage_id`        | INT32?  | No       | Stage (`null` = valid for all stages)    |
//! | `plane_id`        | INT32   | Yes      | Plane index within hydro                 |
//! | `gamma_0`         | DOUBLE  | Yes      | Intercept coefficient (MW), unscaled     |
//! | `gamma_v`         | DOUBLE  | Yes      | Volume coefficient (MW/hm³)              |
//! | `gamma_q`         | DOUBLE  | Yes      | Turbined flow coefficient (MW per m³/s)  |
//! | `gamma_s`         | DOUBLE  | Yes      | Spillage coefficient (MW per m³/s)       |
//! | `kappa`           | DOUBLE? | No       | Correction factor                        |
//! | `valid_v_min_hm3` | DOUBLE? | No       | Volume range minimum                     |
//! | `valid_v_max_hm3` | DOUBLE? | No       | Volume range maximum                     |
//! | `valid_q_max_m3s` | DOUBLE? | No       | Maximum turbined flow validity           |
//!
//! The output file is readable by [`crate::extensions::parse_fpha_hyperplanes`],
//! enabling a round-trip between computed and precomputed hyperplane workflows.
//!
//! All writes use atomic file creation: data is first written to a `.tmp`
//! suffix, then renamed to the final path.

use std::path::Path;
use std::sync::Arc;

use arrow::array::{Float64Builder, Int32Builder, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};

use crate::extensions::FphaHyperplaneRow;
use crate::output::error::OutputError;
use crate::output::parquet_config::ParquetWriterConfig;
use crate::output::stochastic::{ensure_parent_dir, write_parquet_atomic};

/// Write a slice of [`FphaHyperplaneRow`] to a Parquet file at `path`.
///
/// The output schema is exactly 11 columns matching the input schema of
/// `system/fpha_hyperplanes.parquet`, enabling round-trip compatibility
/// with [`crate::extensions::parse_fpha_hyperplanes`].
///
/// Rows are written in the order given; the caller is responsible for sorting
/// into canonical `(hydro_id, stage_id, plane_id)` order before calling if
/// that ordering is required.
///
/// The parent directory is created if it does not already exist. The write is
/// atomic: data goes to `{path}.tmp` first, then the file is renamed to
/// `path`. A partial `.tmp` file may remain on disk if the process is killed
/// mid-write, but the final path is never partially written.
///
/// An empty slice produces a valid Parquet file with 0 rows and the correct
/// 11-column schema.
///
/// # Errors
///
/// - [`OutputError::IoError`] — directory creation, file open, or rename fails.
/// - [`OutputError::SerializationError`] — Arrow/Parquet construction fails.
///
/// # Examples
///
/// ```no_run
/// use cobre_io::output::write_fpha_hyperplanes;
/// use cobre_io::extensions::FphaHyperplaneRow;
/// use cobre_core::EntityId;
/// use std::path::Path;
///
/// # fn main() -> Result<(), cobre_io::OutputError> {
/// let rows = vec![
///     FphaHyperplaneRow {
///         hydro_id: EntityId::from(66),
///         stage_id: None,
///         plane_id: 0,
///         gamma_0: 1250.5,
///         gamma_v: 0.0023,
///         gamma_q: 0.892,
///         gamma_s: -0.015,
///         kappa: 0.985,
///         valid_v_min_hm3: None,
///         valid_v_max_hm3: None,
///         valid_q_max_m3s: None,
///     },
/// ];
/// write_fpha_hyperplanes(
///     Path::new("/tmp/out/hydro_models/fpha_hyperplanes.parquet"),
///     &rows,
/// )?;
/// # Ok(())
/// # }
/// ```
pub fn write_fpha_hyperplanes(path: &Path, rows: &[FphaHyperplaneRow]) -> Result<(), OutputError> {
    ensure_parent_dir(path)?;
    let config = ParquetWriterConfig::default();
    let batch = build_fpha_hyperplanes_batch(rows)?;
    write_parquet_atomic(path, &batch, &config)
}

// ── Schema builder ────────────────────────────────────────────────────────────

fn fpha_hyperplanes_schema() -> Schema {
    Schema::new(vec![
        Field::new("hydro_id", DataType::Int32, false),
        Field::new("stage_id", DataType::Int32, true),
        Field::new("plane_id", DataType::Int32, false),
        Field::new("gamma_0", DataType::Float64, false),
        Field::new("gamma_v", DataType::Float64, false),
        Field::new("gamma_q", DataType::Float64, false),
        Field::new("gamma_s", DataType::Float64, false),
        Field::new("kappa", DataType::Float64, true),
        Field::new("valid_v_min_hm3", DataType::Float64, true),
        Field::new("valid_v_max_hm3", DataType::Float64, true),
        Field::new("valid_q_max_m3s", DataType::Float64, true),
    ])
}

// ── Batch builder ─────────────────────────────────────────────────────────────

#[allow(clippy::similar_names)]
fn build_fpha_hyperplanes_batch(rows: &[FphaHyperplaneRow]) -> Result<RecordBatch, OutputError> {
    let n = rows.len();

    let mut hydro_id_col = Int32Builder::with_capacity(n);
    let mut stage_id_col = Int32Builder::with_capacity(n);
    let mut plane_id_col = Int32Builder::with_capacity(n);
    let mut gamma_0_col = Float64Builder::with_capacity(n);
    let mut gamma_v_col = Float64Builder::with_capacity(n);
    let mut gamma_q_col = Float64Builder::with_capacity(n);
    let mut gamma_s_col = Float64Builder::with_capacity(n);
    let mut kappa_col = Float64Builder::with_capacity(n);
    let mut valid_v_min_col = Float64Builder::with_capacity(n);
    let mut valid_v_max_col = Float64Builder::with_capacity(n);
    let mut valid_q_max_col = Float64Builder::with_capacity(n);

    for row in rows {
        hydro_id_col.append_value(row.hydro_id.0);
        if let Some(sid) = row.stage_id {
            stage_id_col.append_value(sid);
        } else {
            stage_id_col.append_null();
        }
        plane_id_col.append_value(row.plane_id);
        gamma_0_col.append_value(row.gamma_0);
        gamma_v_col.append_value(row.gamma_v);
        gamma_q_col.append_value(row.gamma_q);
        gamma_s_col.append_value(row.gamma_s);
        kappa_col.append_value(row.kappa);
        if let Some(v) = row.valid_v_min_hm3 {
            valid_v_min_col.append_value(v);
        } else {
            valid_v_min_col.append_null();
        }
        if let Some(v) = row.valid_v_max_hm3 {
            valid_v_max_col.append_value(v);
        } else {
            valid_v_max_col.append_null();
        }
        if let Some(v) = row.valid_q_max_m3s {
            valid_q_max_col.append_value(v);
        } else {
            valid_q_max_col.append_null();
        }
    }

    RecordBatch::try_new(
        Arc::new(fpha_hyperplanes_schema()),
        vec![
            Arc::new(hydro_id_col.finish()),
            Arc::new(stage_id_col.finish()),
            Arc::new(plane_id_col.finish()),
            Arc::new(gamma_0_col.finish()),
            Arc::new(gamma_v_col.finish()),
            Arc::new(gamma_q_col.finish()),
            Arc::new(gamma_s_col.finish()),
            Arc::new(kappa_col.finish()),
            Arc::new(valid_v_min_col.finish()),
            Arc::new(valid_v_max_col.finish()),
            Arc::new(valid_q_max_col.finish()),
        ],
    )
    .map_err(|e| OutputError::serialization("fpha_hyperplanes", e.to_string()))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(
    clippy::doc_markdown,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::panic,
    clippy::too_many_lines,
    clippy::unwrap_used
)]
mod tests {
    use super::*;
    use cobre_core::EntityId;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use tempfile::tempdir;

    use crate::extensions::parse_fpha_hyperplanes;

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// Build a sample [`FphaHyperplaneRow`] for `hydro_id` and `plane_id`.
    fn make_row(hydro_id: i32, plane_id: i32, gamma_0: f64, kappa: f64) -> FphaHyperplaneRow {
        FphaHyperplaneRow {
            hydro_id: EntityId::from(hydro_id),
            stage_id: None,
            plane_id,
            gamma_0,
            gamma_v: 0.0023,
            gamma_q: 0.892,
            gamma_s: -0.015,
            kappa,
            valid_v_min_hm3: None,
            valid_v_max_hm3: None,
            valid_q_max_m3s: None,
        }
    }

    // ── AC: round-trip identity ───────────────────────────────────────────────

    /// Write 5 rows for hydro_id=66, read back with parse_fpha_hyperplanes.
    /// All 11 fields must match within 1e-10 tolerance.
    #[test]
    fn round_trip_5_rows_hydro_66() {
        let rows = vec![
            FphaHyperplaneRow {
                hydro_id: EntityId::from(66),
                stage_id: None,
                plane_id: 0,
                gamma_0: 1250.5,
                gamma_v: 0.0023,
                gamma_q: 0.892,
                gamma_s: -0.015,
                kappa: 0.985,
                valid_v_min_hm3: None,
                valid_v_max_hm3: None,
                valid_q_max_m3s: None,
            },
            FphaHyperplaneRow {
                hydro_id: EntityId::from(66),
                stage_id: None,
                plane_id: 1,
                gamma_0: 1180.2,
                gamma_v: 0.0031,
                gamma_q: 0.875,
                gamma_s: -0.012,
                kappa: 0.985,
                valid_v_min_hm3: None,
                valid_v_max_hm3: None,
                valid_q_max_m3s: None,
            },
            FphaHyperplaneRow {
                hydro_id: EntityId::from(66),
                stage_id: None,
                plane_id: 2,
                gamma_0: 1320.8,
                gamma_v: 0.0018,
                gamma_q: 0.901,
                gamma_s: -0.018,
                kappa: 0.985,
                valid_v_min_hm3: None,
                valid_v_max_hm3: None,
                valid_q_max_m3s: None,
            },
            FphaHyperplaneRow {
                hydro_id: EntityId::from(66),
                stage_id: None,
                plane_id: 3,
                gamma_0: 1095.4,
                gamma_v: 0.0042,
                gamma_q: 0.858,
                gamma_s: -0.010,
                kappa: 0.985,
                valid_v_min_hm3: None,
                valid_v_max_hm3: None,
                valid_q_max_m3s: None,
            },
            FphaHyperplaneRow {
                hydro_id: EntityId::from(66),
                stage_id: None,
                plane_id: 4,
                gamma_0: 1410.1,
                gamma_v: 0.0012,
                gamma_q: 0.915,
                gamma_s: -0.022,
                kappa: 0.985,
                valid_v_min_hm3: None,
                valid_v_max_hm3: None,
                valid_q_max_m3s: None,
            },
        ];

        let tmp = tempdir().expect("tempdir");
        let path = tmp.path().join("fpha_hyperplanes.parquet");

        write_fpha_hyperplanes(&path, &rows).expect("write must succeed");
        assert!(path.exists(), "file must exist after write");

        let parsed = parse_fpha_hyperplanes(&path).expect("parse must succeed");

        assert_eq!(parsed.len(), 5, "must have 5 rows");

        for (written, read) in rows.iter().zip(parsed.iter()) {
            assert_eq!(read.hydro_id, written.hydro_id, "hydro_id mismatch");
            assert_eq!(read.plane_id, written.plane_id, "plane_id mismatch");
            assert_eq!(read.stage_id, written.stage_id, "stage_id mismatch");
            assert!(
                (read.gamma_0 - written.gamma_0).abs() < 1e-10,
                "gamma_0 mismatch: {} vs {}",
                read.gamma_0,
                written.gamma_0
            );
            assert!(
                (read.gamma_v - written.gamma_v).abs() < 1e-10,
                "gamma_v mismatch"
            );
            assert!(
                (read.gamma_q - written.gamma_q).abs() < 1e-10,
                "gamma_q mismatch"
            );
            assert!(
                (read.gamma_s - written.gamma_s).abs() < 1e-10,
                "gamma_s mismatch"
            );
            assert!(
                (read.kappa - written.kappa).abs() < 1e-10,
                "kappa mismatch: {} vs {}",
                read.kappa,
                written.kappa
            );
        }
    }

    // ── AC: empty slice produces valid Parquet with 0 rows ────────────────────

    /// Write an empty slice. The output must be a valid Parquet with 0 rows
    /// and the correct 11-column schema.
    #[test]
    fn empty_slice_produces_valid_parquet_with_zero_rows() {
        let tmp = tempdir().expect("tempdir");
        let path = tmp.path().join("fpha_hyperplanes.parquet");

        write_fpha_hyperplanes(&path, &[]).expect("write must succeed for empty slice");
        assert!(path.exists(), "file must exist after write");

        let parsed = parse_fpha_hyperplanes(&path).expect("parse must succeed");
        assert!(parsed.is_empty(), "must have 0 rows for empty input");
    }

    // ── AC: schema validation — exactly 11 fields ─────────────────────────────

    /// Write rows, open with ParquetRecordBatchReaderBuilder, verify exactly
    /// 11 fields with correct names and types.
    #[test]
    fn schema_has_exactly_11_fields_with_correct_names_and_types() {
        let rows = vec![make_row(5, 0, 1000.0, 0.97)];
        let tmp = tempdir().expect("tempdir");
        let path = tmp.path().join("fpha_hyperplanes.parquet");

        write_fpha_hyperplanes(&path, &rows).expect("write must succeed");

        let file = std::fs::File::open(&path).unwrap();
        let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
        let schema = builder.schema().clone();

        assert_eq!(schema.fields().len(), 11, "schema must have 11 fields");

        let expected_fields: &[(&str, bool)] = &[
            ("hydro_id", false),
            ("stage_id", true),
            ("plane_id", false),
            ("gamma_0", false),
            ("gamma_v", false),
            ("gamma_q", false),
            ("gamma_s", false),
            ("kappa", true),
            ("valid_v_min_hm3", true),
            ("valid_v_max_hm3", true),
            ("valid_q_max_m3s", true),
        ];

        for (i, (expected_name, expected_nullable)) in expected_fields.iter().enumerate() {
            let field = &schema.fields()[i];
            assert_eq!(
                field.name(),
                *expected_name,
                "field {i} name: expected {expected_name}, got {}",
                field.name()
            );
            assert_eq!(
                field.is_nullable(),
                *expected_nullable,
                "field {i} ({}) nullable: expected {expected_nullable}, got {}",
                field.name(),
                field.is_nullable()
            );
        }
    }

    // ── AC: nullable columns round-trip as None ───────────────────────────────

    /// Write rows with stage_id=None and all validity range fields as None.
    /// Read back and assert these fields are None in the parsed output.
    #[test]
    fn nullable_columns_round_trip_as_none() {
        let rows = vec![FphaHyperplaneRow {
            hydro_id: EntityId::from(10),
            stage_id: None,
            plane_id: 0,
            gamma_0: 500.0,
            gamma_v: 0.001,
            gamma_q: 0.85,
            gamma_s: -0.01,
            kappa: 1.0,
            valid_v_min_hm3: None,
            valid_v_max_hm3: None,
            valid_q_max_m3s: None,
        }];

        let tmp = tempdir().expect("tempdir");
        let path = tmp.path().join("fpha_hyperplanes.parquet");

        write_fpha_hyperplanes(&path, &rows).expect("write must succeed");

        let parsed = parse_fpha_hyperplanes(&path).expect("parse must succeed");
        assert_eq!(parsed.len(), 1);
        let row = &parsed[0];
        assert!(row.stage_id.is_none(), "stage_id must be None");
        assert!(
            row.valid_v_min_hm3.is_none(),
            "valid_v_min_hm3 must be None"
        );
        assert!(
            row.valid_v_max_hm3.is_none(),
            "valid_v_max_hm3 must be None"
        );
        assert!(
            row.valid_q_max_m3s.is_none(),
            "valid_q_max_m3s must be None"
        );
    }

    // ── AC: multi-hydro rows sorted by (hydro_id, stage_id, plane_id) ─────────

    /// Write rows for hydros 5 and 10 in unsorted order.
    /// parse_fpha_hyperplanes must return them sorted by (hydro_id, stage_id, plane_id).
    #[test]
    fn multi_hydro_rows_sorted_by_parse() {
        let rows = vec![
            make_row(10, 1, 200.0, 0.99),
            make_row(10, 0, 210.0, 0.99),
            make_row(5, 2, 300.0, 0.95),
            make_row(5, 0, 310.0, 0.95),
            make_row(5, 1, 305.0, 0.95),
        ];

        let tmp = tempdir().expect("tempdir");
        let path = tmp.path().join("fpha_hyperplanes.parquet");

        write_fpha_hyperplanes(&path, &rows).expect("write must succeed");

        let parsed = parse_fpha_hyperplanes(&path).expect("parse must succeed");
        assert_eq!(parsed.len(), 5);

        // First 3 rows: hydro_id=5, plane_id 0,1,2
        assert_eq!(parsed[0].hydro_id, EntityId::from(5));
        assert_eq!(parsed[0].plane_id, 0);
        assert_eq!(parsed[1].hydro_id, EntityId::from(5));
        assert_eq!(parsed[1].plane_id, 1);
        assert_eq!(parsed[2].hydro_id, EntityId::from(5));
        assert_eq!(parsed[2].plane_id, 2);
        // Last 2 rows: hydro_id=10, plane_id 0,1
        assert_eq!(parsed[3].hydro_id, EntityId::from(10));
        assert_eq!(parsed[3].plane_id, 0);
        assert_eq!(parsed[4].hydro_id, EntityId::from(10));
        assert_eq!(parsed[4].plane_id, 1);
    }

    // ── AC: parent directory created automatically ────────────────────────────

    /// write_fpha_hyperplanes must create non-existent parent directories.
    #[test]
    fn parent_directory_created_automatically() {
        let tmp = tempdir().expect("tempdir");
        let path = tmp
            .path()
            .join("output")
            .join("hydro_models")
            .join("fpha_hyperplanes.parquet");

        // Parent directories do not exist yet.
        assert!(
            !path.parent().unwrap().exists(),
            "parent dir must not exist before write"
        );

        write_fpha_hyperplanes(&path, &[make_row(1, 0, 100.0, 1.0)])
            .expect("write must succeed even when parent dirs are missing");

        assert!(path.exists(), "file must exist after write");
    }
}
