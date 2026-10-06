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

use arrow::array::{Float64Builder, Int32Builder, RecordBatch, StringBuilder};
#[cfg(test)]
use arrow::datatypes::DataType;
use serde::Serialize;

use crate::extensions::{EvaporationModelRow, FphaDeviationPointRow, FphaHyperplaneRow};
use crate::output::atomic::{ensure_parent_dir, write_batch_atomic, write_json_atomic};
use crate::output::error::OutputError;
use crate::output::schemas::{
    evaporation_models_schema, fpha_deviation_points_schema, fpha_hyperplanes_schema,
};

/// The fitted FPHA hyperplanes file's path relative to the output directory.
pub const FPHA_HYPERPLANES_FILE: &str = "hydro_models/fpha_hyperplanes.parquet";
/// The evaporation models file's path relative to the output directory.
pub const EVAPORATION_MODELS_FILE: &str = "hydro_models/evaporation_models.parquet";
/// The FPHA deviation points file's path relative to the output directory.
pub const FPHA_DEVIATION_POINTS_FILE: &str = "hydro_models/fpha_deviation_points.parquet";

/// Write a slice of [`FphaHyperplaneRow`] to a Parquet file at `path`,
/// re-readable as `system/fpha_hyperplanes.parquet` by
/// [`crate::extensions::parse_fpha_hyperplanes`].
///
/// Rows are written in the order given; the caller sorts into canonical
/// `(hydro_id, stage_id, plane_id)` order. An empty slice produces a valid
/// 0-row file. The parent directory is created if absent; the write is atomic.
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
    let batch = build_fpha_hyperplanes_batch(rows)?;
    write_batch_atomic(path, &batch)
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
        stage_id_col.append_option(row.stage_id);
        plane_id_col.append_value(row.plane_id);
        gamma_0_col.append_value(row.gamma_0);
        gamma_v_col.append_value(row.gamma_v);
        gamma_q_col.append_value(row.gamma_q);
        gamma_s_col.append_value(row.gamma_s);
        kappa_col.append_value(row.kappa);
        valid_v_min_col.append_option(row.valid_v_min_hm3);
        valid_v_max_col.append_option(row.valid_v_max_hm3);
        valid_q_max_col.append_option(row.valid_q_max_m3s);
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

// ── Evaporation models ──────────────────────────────────────────────────────────

/// Write a slice of [`EvaporationModelRow`] to a Parquet file at `path`,
/// re-readable by [`crate::extensions::parse_evaporation_models`].
///
/// Rows are written in the order given; the caller sorts into canonical
/// `(hydro_id, stage_id)` order. An empty slice produces a valid 0-row file.
/// The parent directory is created if absent; the write is atomic.
///
/// # Errors
///
/// - [`OutputError::IoError`] — directory creation, file open, or rename fails.
/// - [`OutputError::SerializationError`] — Arrow/Parquet construction fails.
///
/// # Examples
///
/// ```no_run
/// use cobre_io::output::write_evaporation_models;
/// use cobre_io::extensions::EvaporationModelRow;
/// use cobre_core::EntityId;
/// use std::path::Path;
///
/// # fn main() -> Result<(), cobre_io::OutputError> {
/// let rows = vec![
///     EvaporationModelRow {
///         hydro_id: EntityId::from(66),
///         stage_id: None,
///         intercept_m3s: 12.5,
///         volume_slope_m3s_per_hm3: 0.0031,
///         reference_volume_hm3: 14_500.0,
///         source: "default_midpoint".to_string(),
///     },
/// ];
/// write_evaporation_models(
///     Path::new("/tmp/out/hydro_models/evaporation_models.parquet"),
///     &rows,
/// )?;
/// # Ok(())
/// # }
/// ```
pub fn write_evaporation_models(
    path: &Path,
    rows: &[EvaporationModelRow],
) -> Result<(), OutputError> {
    let batch = build_evaporation_models_batch(rows)?;
    write_batch_atomic(path, &batch)
}

fn build_evaporation_models_batch(
    rows: &[EvaporationModelRow],
) -> Result<RecordBatch, OutputError> {
    let n = rows.len();

    let mut hydro_id_col = Int32Builder::with_capacity(n);
    let mut stage_id_col = Int32Builder::with_capacity(n);
    let mut intercept_col = Float64Builder::with_capacity(n);
    let mut volume_slope_col = Float64Builder::with_capacity(n);
    let mut reference_volume_col = Float64Builder::with_capacity(n);
    let mut source_col = StringBuilder::with_capacity(n, n * 16);

    for row in rows {
        hydro_id_col.append_value(row.hydro_id.0);
        stage_id_col.append_option(row.stage_id);
        intercept_col.append_value(row.intercept_m3s);
        volume_slope_col.append_value(row.volume_slope_m3s_per_hm3);
        reference_volume_col.append_value(row.reference_volume_hm3);
        source_col.append_value(&row.source);
    }

    RecordBatch::try_new(
        Arc::new(evaporation_models_schema()),
        vec![
            Arc::new(hydro_id_col.finish()),
            Arc::new(stage_id_col.finish()),
            Arc::new(intercept_col.finish()),
            Arc::new(volume_slope_col.finish()),
            Arc::new(reference_volume_col.finish()),
            Arc::new(source_col.finish()),
        ],
    )
    .map_err(|e| OutputError::serialization("evaporation_models", e.to_string()))
}

// ── FPHA deviation points ─────────────────────────────────────────────────────

/// Write a slice of [`FphaDeviationPointRow`] to a Parquet file at `path`,
/// re-readable by [`crate::extensions::parse_fpha_deviation_points`].
///
/// Rows are written in the order given; the caller emits them in canonical
/// `(hydro_id, stage_id, grid)` order. An empty slice produces a valid 0-row
/// file. The parent directory is created if absent; the write is atomic.
///
/// # Errors
///
/// - [`OutputError::IoError`] — directory creation, file open, or rename fails.
/// - [`OutputError::SerializationError`] — Arrow/Parquet construction fails.
///
/// # Examples
///
/// ```no_run
/// use cobre_io::output::write_fpha_deviation_points;
/// use cobre_io::extensions::FphaDeviationPointRow;
/// use cobre_core::EntityId;
/// use std::path::Path;
///
/// # fn main() -> Result<(), cobre_io::OutputError> {
/// let rows = vec![
///     FphaDeviationPointRow {
///         hydro_id: EntityId::from(66),
///         stage_id: Some(3),
///         v: 14_500.0,
///         q: 1_200.0,
///         fph_exact: 980.5,
///         fpha_fitted: 985.0,
///         deviation: 4.5,
///         relative: 0.0046,
///     },
/// ];
/// write_fpha_deviation_points(
///     Path::new("/tmp/out/hydro_models/fpha_deviation_points.parquet"),
///     &rows,
/// )?;
/// # Ok(())
/// # }
/// ```
pub fn write_fpha_deviation_points(
    path: &Path,
    rows: &[FphaDeviationPointRow],
) -> Result<(), OutputError> {
    let batch = build_fpha_deviation_points_batch(rows)?;
    write_batch_atomic(path, &batch)
}

fn build_fpha_deviation_points_batch(
    rows: &[FphaDeviationPointRow],
) -> Result<RecordBatch, OutputError> {
    let n = rows.len();

    let mut hydro_id_col = Int32Builder::with_capacity(n);
    let mut stage_id_col = Int32Builder::with_capacity(n);
    let mut v_col = Float64Builder::with_capacity(n);
    let mut q_col = Float64Builder::with_capacity(n);
    let mut fph_exact_col = Float64Builder::with_capacity(n);
    let mut fpha_fitted_col = Float64Builder::with_capacity(n);
    let mut deviation_col = Float64Builder::with_capacity(n);
    let mut relative_col = Float64Builder::with_capacity(n);

    for row in rows {
        hydro_id_col.append_value(row.hydro_id.0);
        stage_id_col.append_option(row.stage_id);
        v_col.append_value(row.v);
        q_col.append_value(row.q);
        fph_exact_col.append_value(row.fph_exact);
        fpha_fitted_col.append_value(row.fpha_fitted);
        deviation_col.append_value(row.deviation);
        relative_col.append_value(row.relative);
    }

    RecordBatch::try_new(
        Arc::new(fpha_deviation_points_schema()),
        vec![
            Arc::new(hydro_id_col.finish()),
            Arc::new(stage_id_col.finish()),
            Arc::new(v_col.finish()),
            Arc::new(q_col.finish()),
            Arc::new(fph_exact_col.finish()),
            Arc::new(fpha_fitted_col.finish()),
            Arc::new(deviation_col.finish()),
            Arc::new(relative_col.finish()),
        ],
    )
    .map_err(|e| OutputError::serialization("fpha_deviation_points", e.to_string()))
}

// ── Structural hydro-model summary (generic JSON sidecar) ───────────────────────

/// Write a structural hydro-model summary as pretty-printed JSON.
///
/// Generic over `Serialize` so the summary struct can stay in the calling
/// algorithm crate, keeping this crate algorithm-agnostic.
///
/// # Errors
///
/// Returns [`OutputError::IoError`] on filesystem failures, or
/// [`OutputError::SerializationError`] if JSON serialization fails.
pub fn write_hydro_model_summary(path: &Path, summary: &impl Serialize) -> Result<(), OutputError> {
    ensure_parent_dir(path)?;
    write_json_atomic(path, summary, "hydro_models")
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

    use crate::extensions::{
        EvaporationModelRow, FphaDeviationPointRow, parse_evaporation_models,
        parse_fpha_deviation_points, parse_fpha_hyperplanes,
    };

    // ── Helpers ───────────────────────────────────────────────────────────────

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

        assert_eq!(parsed[0].hydro_id, EntityId::from(5));
        assert_eq!(parsed[0].plane_id, 0);
        assert_eq!(parsed[1].hydro_id, EntityId::from(5));
        assert_eq!(parsed[1].plane_id, 1);
        assert_eq!(parsed[2].hydro_id, EntityId::from(5));
        assert_eq!(parsed[2].plane_id, 2);
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

        assert!(
            !path.parent().unwrap().exists(),
            "parent dir must not exist before write"
        );

        write_fpha_hyperplanes(&path, &[make_row(1, 0, 100.0, 1.0)])
            .expect("write must succeed even when parent dirs are missing");

        assert!(path.exists(), "file must exist after write");
    }

    // ── Evaporation models ────────────────────────────────────────────────────

    /// Write two rows (one `stage_id: None`, one `stage_id: Some(3)`), read back
    /// with parse_evaporation_models, assert field-for-field equality.
    #[test]
    fn evaporation_models_round_trips() {
        let rows = vec![
            EvaporationModelRow {
                hydro_id: EntityId::from(66),
                stage_id: None,
                intercept_m3s: 12.5,
                volume_slope_m3s_per_hm3: 0.0031,
                reference_volume_hm3: 14_500.0,
                source: "default_midpoint".to_string(),
            },
            EvaporationModelRow {
                hydro_id: EntityId::from(66),
                stage_id: Some(3),
                intercept_m3s: 9.75,
                volume_slope_m3s_per_hm3: 0.0042,
                reference_volume_hm3: 12_000.0,
                source: "user_supplied".to_string(),
            },
        ];

        let tmp = tempdir().expect("tempdir");
        let path = tmp.path().join("evaporation_models.parquet");

        write_evaporation_models(&path, &rows).expect("write must succeed");
        assert!(path.exists(), "file must exist after write");

        let parsed = parse_evaporation_models(&path).expect("parse must succeed");
        assert_eq!(parsed, rows, "parsed rows must equal input field-for-field");
    }

    /// Open the evaporation parquet and verify exactly six columns with the
    /// expected names and nullability.
    #[test]
    fn evaporation_models_schema_has_expected_columns() {
        let rows = vec![EvaporationModelRow {
            hydro_id: EntityId::from(5),
            stage_id: None,
            intercept_m3s: 1.0,
            volume_slope_m3s_per_hm3: 0.001,
            reference_volume_hm3: 100.0,
            source: "default_midpoint".to_string(),
        }];
        let tmp = tempdir().expect("tempdir");
        let path = tmp.path().join("evaporation_models.parquet");

        write_evaporation_models(&path, &rows).expect("write must succeed");

        let file = std::fs::File::open(&path).unwrap();
        let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
        let schema = builder.schema().clone();

        assert_eq!(schema.fields().len(), 6, "schema must have 6 fields");

        let expected_fields: &[(&str, DataType, bool)] = &[
            ("hydro_id", DataType::Int32, false),
            ("stage_id", DataType::Int32, true),
            ("intercept_m3s", DataType::Float64, false),
            ("volume_slope_m3s_per_hm3", DataType::Float64, false),
            ("reference_volume_hm3", DataType::Float64, false),
            ("source", DataType::Utf8, false),
        ];

        for (i, (expected_name, expected_type, expected_nullable)) in
            expected_fields.iter().enumerate()
        {
            let field = &schema.fields()[i];
            assert_eq!(
                field.name(),
                *expected_name,
                "field {i} name: expected {expected_name}, got {}",
                field.name()
            );
            assert_eq!(
                field.data_type(),
                expected_type,
                "field {i} ({expected_name}) type: expected {expected_type:?}, got {:?}",
                field.data_type()
            );
            assert_eq!(
                field.is_nullable(),
                *expected_nullable,
                "field {i} ({expected_name}) nullable: expected {expected_nullable}, got {}",
                field.is_nullable()
            );
        }
    }

    /// An empty slice writes a valid parquet with zero rows.
    #[test]
    fn write_evaporation_models_empty_ok() {
        let tmp = tempdir().expect("tempdir");
        let path = tmp.path().join("evaporation_models.parquet");

        write_evaporation_models(&path, &[]).expect("write must succeed for empty slice");
        assert!(path.exists(), "file must exist after write");

        let parsed = parse_evaporation_models(&path).expect("parse must succeed");
        assert!(parsed.is_empty(), "must have 0 rows for empty input");
    }

    // ── FPHA deviation points ─────────────────────────────────────────────────

    /// Write two rows (one `stage_id: None`, one `stage_id: Some(3)`), read back
    /// with parse_fpha_deviation_points, assert field-for-field equality.
    #[test]
    fn fpha_deviation_points_round_trips() {
        let rows = vec![
            FphaDeviationPointRow {
                hydro_id: EntityId::from(66),
                stage_id: None,
                v: 0.0,
                q: 0.0,
                fph_exact: 0.0,
                fpha_fitted: 1.5,
                deviation: 1.5,
                relative: 0.0015,
            },
            FphaDeviationPointRow {
                hydro_id: EntityId::from(66),
                stage_id: Some(3),
                v: 14_500.0,
                q: 1_200.0,
                fph_exact: 980.5,
                fpha_fitted: 985.0,
                deviation: 4.5,
                relative: 0.0046,
            },
        ];

        let tmp = tempdir().expect("tempdir");
        let path = tmp.path().join("fpha_deviation_points.parquet");

        write_fpha_deviation_points(&path, &rows).expect("write must succeed");
        assert!(path.exists(), "file must exist after write");

        let parsed = parse_fpha_deviation_points(&path).expect("parse must succeed");
        assert_eq!(parsed, rows, "parsed rows must equal input field-for-field");
    }

    /// Open the deviation-points parquet and verify exactly eight columns with the
    /// expected names, types, and nullability in the declared order.
    #[test]
    fn fpha_deviation_points_schema_has_expected_columns() {
        let rows = vec![FphaDeviationPointRow {
            hydro_id: EntityId::from(5),
            stage_id: None,
            v: 100.0,
            q: 50.0,
            fph_exact: 40.0,
            fpha_fitted: 41.0,
            deviation: 1.0,
            relative: 0.025,
        }];
        let tmp = tempdir().expect("tempdir");
        let path = tmp.path().join("fpha_deviation_points.parquet");

        write_fpha_deviation_points(&path, &rows).expect("write must succeed");

        let file = std::fs::File::open(&path).unwrap();
        let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
        let schema = builder.schema().clone();

        assert_eq!(schema.fields().len(), 8, "schema must have 8 fields");

        let expected_fields: &[(&str, DataType, bool)] = &[
            ("hydro_id", DataType::Int32, false),
            ("stage_id", DataType::Int32, true),
            ("v", DataType::Float64, false),
            ("q", DataType::Float64, false),
            ("fph_exact", DataType::Float64, false),
            ("fpha_fitted", DataType::Float64, false),
            ("deviation", DataType::Float64, false),
            ("relative", DataType::Float64, false),
        ];

        for (i, (expected_name, expected_type, expected_nullable)) in
            expected_fields.iter().enumerate()
        {
            let field = &schema.fields()[i];
            assert_eq!(
                field.name(),
                *expected_name,
                "field {i} name: expected {expected_name}, got {}",
                field.name()
            );
            assert_eq!(
                field.data_type(),
                expected_type,
                "field {i} ({expected_name}) type: expected {expected_type:?}, got {:?}",
                field.data_type()
            );
            assert_eq!(
                field.is_nullable(),
                *expected_nullable,
                "field {i} ({expected_name}) nullable: expected {expected_nullable}, got {}",
                field.is_nullable()
            );
        }
    }

    /// An empty slice writes a valid parquet with zero rows.
    #[test]
    fn write_fpha_deviation_points_empty_ok() {
        let tmp = tempdir().expect("tempdir");
        let path = tmp.path().join("fpha_deviation_points.parquet");

        write_fpha_deviation_points(&path, &[]).expect("write must succeed for empty slice");
        assert!(path.exists(), "file must exist after write");

        let parsed = parse_fpha_deviation_points(&path).expect("parse must succeed");
        assert!(parsed.is_empty(), "must have 0 rows for empty input");
    }

    // ── Structural hydro-model summary (generic JSON sidecar) ─────────────────

    /// Local mock summary, defined here so the JSON sidecar tests never depend
    /// on an algorithm crate (genericity rule).
    #[derive(Debug, serde::Serialize, serde::Deserialize, PartialEq)]
    struct MockHydroModelSummary {
        n_constant: usize,
        n_fpha: usize,
        total_planes: usize,
    }

    #[test]
    fn write_and_read_hydro_model_summary_round_trips() {
        let tmp = tempdir().expect("tempdir");
        let path = tmp.path().join("training/hydro_models.json");

        let summary = MockHydroModelSummary {
            n_constant: 4,
            n_fpha: 7,
            total_planes: 35,
        };

        write_hydro_model_summary(&path, &summary).expect("write should succeed");

        let content = std::fs::read_to_string(&path).expect("read");
        let decoded: MockHydroModelSummary =
            serde_json::from_str(&content).expect("valid JSON after round-trip");
        assert_eq!(decoded, summary);
    }

    #[test]
    fn hydro_model_summary_write_is_atomic_no_tmp_remains() {
        let tmp = tempdir().expect("tempdir");
        let path = tmp.path().join("hydro_models.json");

        let summary = MockHydroModelSummary {
            n_constant: 1,
            n_fpha: 2,
            total_planes: 6,
        };

        write_hydro_model_summary(&path, &summary).expect("write should succeed");

        let tmp_path = path.with_extension("json.tmp");
        assert!(
            !tmp_path.exists(),
            "tmp file should be removed after rename"
        );
        assert!(path.exists(), "final file should exist");
    }
}
