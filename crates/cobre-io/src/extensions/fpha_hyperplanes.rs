//! Parsing for `system/fpha_hyperplanes.parquet` — pre-computed FPHA hyperplane coefficients.
//!
//! [`parse_fpha_hyperplanes`] reads `system/fpha_hyperplanes.parquet` and returns a sorted
//! `Vec<FphaHyperplaneRow>` containing the hyperplane coefficients for hydros configured with
//! `source: "precomputed"`.
//!
//! ## Parquet schema
//!
//! | Column            | Type         | Required | Description                              |
//! | ----------------- | ------------ | -------- | ---------------------------------------- |
//! | `hydro_id`        | INT32        | Yes      | Hydro plant identifier                   |
//! | `stage_id`        | INT32?       | No       | Stage (`null` = valid for all stages)    |
//! | `plane_id`        | INT32        | Yes      | Plane index within hydro                 |
//! | `gamma_0`         | DOUBLE       | Yes      | Intercept coefficient (MW)               |
//! | `gamma_v`         | DOUBLE       | Yes      | Volume coefficient (MW/hm³)              |
//! | `gamma_q`         | DOUBLE       | Yes      | Turbined flow coefficient (MW per m³/s)  |
//! | `gamma_s`         | DOUBLE       | Yes      | Spillage coefficient (MW per m³/s)       |
//! | `kappa`           | DOUBLE?      | No       | Correction factor (default: 1.0)         |
//! | `valid_v_min_hm3` | DOUBLE?      | No       | Volume range minimum                     |
//! | `valid_v_max_hm3` | DOUBLE?      | No       | Volume range maximum                     |
//! | `valid_q_max_m3s` | DOUBLE?      | No       | Maximum turbined flow validity           |
//!
//! ## Output ordering
//!
//! Rows are sorted by `(hydro_id, stage_id, plane_id)` ascending.
//! Null `stage_id` sorts before any non-null value.
//!
//! ## Validation
//!
//! Per-row constraints enforced by this parser:
//!
//! - Required columns (`hydro_id`, `plane_id`, `gamma_0`, `gamma_v`, `gamma_q`, `gamma_s`) must
//!   be present with the correct types.
//! - Optional columns that are present must have the correct types.
//! - `kappa` defaults to `1.0` when the column is entirely absent or the cell value is null.
//!
//! Deferred validations (not performed here):
//!
//! - Minimum planes per hydro-per-stage (at least 3) — Layer 5, Epic 06.
//! - `gamma_v` positive, `gamma_s` non-positive semantic checks — Layer 5, Epic 06.
//! - `hydro_id` existence in the hydro registry — Layer 3, Epic 06.

use arrow::array::Array;
use cobre_core::EntityId;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::fs::File;
use std::path::Path;

use crate::LoadError;
use crate::parquet_helpers::{
    extract_optional_float64, extract_optional_int32, extract_required_float64,
    extract_required_int32,
};

/// A single row from `system/fpha_hyperplanes.parquet`.
///
/// Each row defines one hyperplane of the piecewise-linear FPHA production
/// function approximation for the hydro identified by `hydro_id`.
///
/// # Examples
///
/// ```
/// use cobre_io::extensions::FphaHyperplaneRow;
/// use cobre_core::EntityId;
///
/// let row = FphaHyperplaneRow {
///     hydro_id: EntityId::from(66),
///     stage_id: None,
///     plane_id: 0,
///     gamma_0: 1250.5,
///     gamma_v: 0.0023,
///     gamma_q: 0.892,
///     gamma_s: -0.015,
///     kappa: 0.985,
///     valid_v_min_hm3: None,
///     valid_v_max_hm3: None,
///     valid_q_max_m3s: None,
/// };
/// assert_eq!(row.hydro_id, EntityId::from(66));
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct FphaHyperplaneRow {
    /// Hydro plant this hyperplane belongs to.
    pub hydro_id: EntityId,
    /// Stage this plane applies to. `None` means valid for all stages.
    pub stage_id: Option<i32>,
    /// Plane index within this hydro (and stage).
    pub plane_id: i32,
    /// Intercept coefficient (MW).
    pub gamma_0: f64,
    /// Volume coefficient (MW/hm³).
    pub gamma_v: f64,
    /// Turbined flow coefficient (MW per m³/s).
    pub gamma_q: f64,
    /// Spillage coefficient (MW per m³/s, typically ≤ 0).
    pub gamma_s: f64,
    /// Correction factor κ. Defaults to `1.0` when absent or null in the file.
    pub kappa: f64,
    /// Volume range minimum where this plane is valid (hm³). Optional.
    pub valid_v_min_hm3: Option<f64>,
    /// Volume range maximum where this plane is valid (hm³). Optional.
    pub valid_v_max_hm3: Option<f64>,
    /// Maximum turbined flow where this plane is valid (m³/s). Optional.
    pub valid_q_max_m3s: Option<f64>,
}

/// Parse `system/fpha_hyperplanes.parquet` and return a sorted FPHA coefficient table.
///
/// Reads all record batches from the Parquet file at `path`, validates per-row
/// constraints, then returns all rows sorted by `(hydro_id, stage_id, plane_id)` ascending.
/// Null `stage_id` values sort before any non-null stage.
///
/// # Errors
///
/// | Condition                                     | Error variant              |
/// |---------------------------------------------- |--------------------------- |
/// | File not found or permission denied           | [`LoadError::IoError`]     |
/// | Malformed Parquet (corrupt header, etc.)      | [`LoadError::ParseError`]  |
/// | Required column missing or wrong type         | [`LoadError::SchemaError`] |
/// | Optional column present with wrong type       | [`LoadError::SchemaError`] |
///
/// # Examples
///
/// ```no_run
/// use cobre_io::extensions::parse_fpha_hyperplanes;
/// use std::path::Path;
///
/// let rows = parse_fpha_hyperplanes(Path::new("system/fpha_hyperplanes.parquet"))
///     .expect("valid FPHA hyperplanes file");
/// println!("loaded {} hyperplane rows", rows.len());
/// ```
#[allow(clippy::similar_names)]
pub fn parse_fpha_hyperplanes(path: &Path) -> Result<Vec<FphaHyperplaneRow>, LoadError> {
    let file = File::open(path).map_err(|e| LoadError::io(path, e))?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| LoadError::parse(path, e.to_string()))?;

    let reader = builder
        .build()
        .map_err(|e| LoadError::parse(path, e.to_string()))?;

    let mut rows: Vec<FphaHyperplaneRow> = Vec::new();

    for batch_result in reader {
        let batch = batch_result.map_err(|e| LoadError::parse(path, e.to_string()))?;

        // ── Required columns ──────────────────────────────────────────────────
        let hydro_id_col = extract_required_int32(&batch, "hydro_id", path)?;
        let plane_id_col = extract_required_int32(&batch, "plane_id", path)?;
        let gamma_0_col = extract_required_float64(&batch, "gamma_0", path)?;
        let gamma_v_col = extract_required_float64(&batch, "gamma_v", path)?;
        let gamma_q_col = extract_required_float64(&batch, "gamma_q", path)?;
        let gamma_s_col = extract_required_float64(&batch, "gamma_s", path)?;

        // ── Optional columns — check existence first ──────────────────────────
        let stage_id_col = extract_optional_int32(&batch, "stage_id", path)?;
        let kappa_col = extract_optional_float64(&batch, "kappa", path)?;
        let valid_v_min_col = extract_optional_float64(&batch, "valid_v_min_hm3", path)?;
        let valid_v_max_col = extract_optional_float64(&batch, "valid_v_max_hm3", path)?;
        let valid_q_max_col = extract_optional_float64(&batch, "valid_q_max_m3s", path)?;

        // ── Build rows ────────────────────────────────────────────────────────
        let n = batch.num_rows();
        rows.reserve(n);

        for i in 0..n {
            let hydro_id = EntityId::from(hydro_id_col.value(i));
            let plane_id = plane_id_col.value(i);
            let gamma_0 = gamma_0_col.value(i);
            let gamma_v = gamma_v_col.value(i);
            let gamma_q = gamma_q_col.value(i);
            let gamma_s = gamma_s_col.value(i);

            // stage_id: None if column is absent or null at this row.
            let stage_id = stage_id_col
                .filter(|col| !col.is_null(i))
                .map(|col| col.value(i));

            // kappa defaults to 1.0 when column is absent or null.
            let kappa = kappa_col
                .filter(|col| !col.is_null(i))
                .map_or(1.0, |col| col.value(i));

            // Optional float columns: None when absent or null.
            let valid_v_min_hm3 = valid_v_min_col
                .filter(|col| !col.is_null(i))
                .map(|col| col.value(i));
            let valid_v_max_hm3 = valid_v_max_col
                .filter(|col| !col.is_null(i))
                .map(|col| col.value(i));
            let valid_q_max_m3s = valid_q_max_col
                .filter(|col| !col.is_null(i))
                .map(|col| col.value(i));

            rows.push(FphaHyperplaneRow {
                hydro_id,
                stage_id,
                plane_id,
                gamma_0,
                gamma_v,
                gamma_q,
                gamma_s,
                kappa,
                valid_v_min_hm3,
                valid_v_max_hm3,
                valid_q_max_m3s,
            });
        }
    }

    // ── Sort by (hydro_id, stage_id, plane_id) ascending ─────────────────────
    // Null stage_id sorts before any non-null value (None < Some(_)).
    rows.sort_by(|a, b| {
        a.hydro_id
            .0
            .cmp(&b.hydro_id.0)
            .then_with(|| a.stage_id.cmp(&b.stage_id))
            .then_with(|| a.plane_id.cmp(&b.plane_id))
    });

    Ok(rows)
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
    use arrow::array::{Float64Array, Int32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;
    use tempfile::NamedTempFile;

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// Minimum required-column schema (no optional columns).
    fn required_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("hydro_id", DataType::Int32, false),
            Field::new("plane_id", DataType::Int32, false),
            Field::new("gamma_0", DataType::Float64, false),
            Field::new("gamma_v", DataType::Float64, false),
            Field::new("gamma_q", DataType::Float64, false),
            Field::new("gamma_s", DataType::Float64, false),
        ]))
    }

    /// Full schema including all optional columns.
    fn full_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
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
        ]))
    }

    /// Write a single [`RecordBatch`] to a temporary Parquet file.
    fn write_parquet(batch: &RecordBatch) -> NamedTempFile {
        let tmp = NamedTempFile::new().expect("tempfile");
        let mut writer = ArrowWriter::try_new(tmp.reopen().expect("reopen"), batch.schema(), None)
            .expect("ArrowWriter");
        writer.write(batch).expect("write batch");
        writer.close().expect("close writer");
        tmp
    }

    /// Build a minimal required-column batch with the given data.
    fn make_required_batch(
        hydro_ids: &[i32],
        plane_ids: &[i32],
        g0: &[f64],
        gv: &[f64],
        gq: &[f64],
        gs: &[f64],
    ) -> RecordBatch {
        RecordBatch::try_new(
            required_schema(),
            vec![
                Arc::new(Int32Array::from(hydro_ids.to_vec())),
                Arc::new(Int32Array::from(plane_ids.to_vec())),
                Arc::new(Float64Array::from(g0.to_vec())),
                Arc::new(Float64Array::from(gv.to_vec())),
                Arc::new(Float64Array::from(gq.to_vec())),
                Arc::new(Float64Array::from(gs.to_vec())),
            ],
        )
        .expect("valid required batch")
    }

    // ── AC: valid file with all columns present (Itaipu example) ─────────────

    /// 5 planes for hydro 66, all with kappa = 0.985 (Itaipu from spec example).
    /// Result: Ok with 5 rows, all hydro_id = EntityId(66), sorted by plane_id.
    #[test]
    fn test_valid_itaipu_5_planes_all_columns() {
        let schema = full_schema();
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![66, 66, 66, 66, 66])),
                // stage_id: all null
                Arc::new(Int32Array::from(vec![None::<i32>; 5])),
                Arc::new(Int32Array::from(vec![0, 1, 2, 3, 4])),
                Arc::new(Float64Array::from(vec![
                    1250.5, 1180.2, 1320.8, 1095.4, 1410.1,
                ])),
                Arc::new(Float64Array::from(vec![
                    0.0023, 0.0031, 0.0018, 0.0042, 0.0012,
                ])),
                Arc::new(Float64Array::from(vec![0.892, 0.875, 0.901, 0.858, 0.915])),
                Arc::new(Float64Array::from(vec![
                    -0.015, -0.012, -0.018, -0.010, -0.022,
                ])),
                Arc::new(Float64Array::from(vec![0.985, 0.985, 0.985, 0.985, 0.985])),
                Arc::new(Float64Array::from(vec![None::<f64>; 5])),
                Arc::new(Float64Array::from(vec![None::<f64>; 5])),
                Arc::new(Float64Array::from(vec![None::<f64>; 5])),
            ],
        )
        .unwrap();
        let tmp = write_parquet(&batch);
        let rows = parse_fpha_hyperplanes(tmp.path()).unwrap();

        assert_eq!(rows.len(), 5, "expected 5 rows");
        for row in &rows {
            assert_eq!(row.hydro_id, EntityId::from(66));
            assert!(row.stage_id.is_none());
            assert!(
                (row.kappa - 0.985).abs() < 1e-10,
                "kappa should be 0.985, got {}",
                row.kappa
            );
        }
        // Verify sort by plane_id.
        let plane_ids: Vec<i32> = rows.iter().map(|r| r.plane_id).collect();
        assert_eq!(plane_ids, vec![0, 1, 2, 3, 4]);

        // Spot-check specific values from the spec example.
        assert!((rows[0].gamma_0 - 1250.5).abs() < 1e-10);
        assert!((rows[0].gamma_v - 0.0023).abs() < 1e-12);
        assert!((rows[0].gamma_q - 0.892).abs() < 1e-10);
        assert!((rows[0].gamma_s - (-0.015)).abs() < 1e-12);
    }

    // ── AC: optional columns absent — kappa defaults to 1.0 ──────────────────

    /// File with only required columns — kappa defaults to 1.0 per row.
    #[test]
    fn test_optional_columns_absent_kappa_defaults_to_1() {
        let batch = make_required_batch(
            &[10, 10, 10],
            &[0, 1, 2],
            &[500.0, 480.0, 520.0],
            &[0.001, 0.002, 0.0015],
            &[0.85, 0.83, 0.87],
            &[-0.01, -0.01, -0.01],
        );
        let tmp = write_parquet(&batch);
        let rows = parse_fpha_hyperplanes(tmp.path()).unwrap();

        assert_eq!(rows.len(), 3);
        for row in &rows {
            assert!(
                (row.kappa - 1.0).abs() < f64::EPSILON,
                "kappa should default to 1.0, got {}",
                row.kappa
            );
            assert!(row.stage_id.is_none());
            assert!(row.valid_v_min_hm3.is_none());
            assert!(row.valid_v_max_hm3.is_none());
            assert!(row.valid_q_max_m3s.is_none());
        }
    }

    // ── AC: kappa null in column still defaults to 1.0 ───────────────────────

    /// File has a kappa column but all values are null — defaults to 1.0.
    #[test]
    fn test_kappa_column_present_but_null_defaults_to_1() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("hydro_id", DataType::Int32, false),
            Field::new("plane_id", DataType::Int32, false),
            Field::new("gamma_0", DataType::Float64, false),
            Field::new("gamma_v", DataType::Float64, false),
            Field::new("gamma_q", DataType::Float64, false),
            Field::new("gamma_s", DataType::Float64, false),
            Field::new("kappa", DataType::Float64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![5])),
                Arc::new(Int32Array::from(vec![0])),
                Arc::new(Float64Array::from(vec![100.0])),
                Arc::new(Float64Array::from(vec![0.001])),
                Arc::new(Float64Array::from(vec![0.9])),
                Arc::new(Float64Array::from(vec![-0.005])),
                Arc::new(Float64Array::from(vec![None::<f64>])),
            ],
        )
        .unwrap();
        let tmp = write_parquet(&batch);
        let rows = parse_fpha_hyperplanes(tmp.path()).unwrap();

        assert_eq!(rows.len(), 1);
        assert!(
            (rows[0].kappa - 1.0).abs() < f64::EPSILON,
            "null kappa should default to 1.0, got {}",
            rows[0].kappa
        );
    }

    // ── AC: missing required column -> SchemaError ────────────────────────────

    /// File missing `gamma_0` column -> SchemaError with field "gamma_0".
    #[test]
    fn test_missing_gamma_0_column() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("hydro_id", DataType::Int32, false),
            Field::new("plane_id", DataType::Int32, false),
            Field::new("gamma_v", DataType::Float64, false),
            Field::new("gamma_q", DataType::Float64, false),
            Field::new("gamma_s", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![1])),
                Arc::new(Int32Array::from(vec![0])),
                Arc::new(Float64Array::from(vec![0.001])),
                Arc::new(Float64Array::from(vec![0.9])),
                Arc::new(Float64Array::from(vec![-0.005])),
            ],
        )
        .unwrap();
        let tmp = write_parquet(&batch);
        let err = parse_fpha_hyperplanes(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert_eq!(field, "gamma_0", "field should be 'gamma_0', got: {field}");
                assert!(
                    message.contains("missing required column"),
                    "message should mention missing column, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// File missing `hydro_id` column -> SchemaError with field "hydro_id".
    #[test]
    fn test_missing_hydro_id_column() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("plane_id", DataType::Int32, false),
            Field::new("gamma_0", DataType::Float64, false),
            Field::new("gamma_v", DataType::Float64, false),
            Field::new("gamma_q", DataType::Float64, false),
            Field::new("gamma_s", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![0])),
                Arc::new(Float64Array::from(vec![100.0])),
                Arc::new(Float64Array::from(vec![0.001])),
                Arc::new(Float64Array::from(vec![0.9])),
                Arc::new(Float64Array::from(vec![-0.005])),
            ],
        )
        .unwrap();
        let tmp = write_parquet(&batch);
        let err = parse_fpha_hyperplanes(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, .. } => {
                assert_eq!(field, "hydro_id");
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── AC: wrong column type -> SchemaError ──────────────────────────────────

    /// `gamma_0` provided as Int32 instead of Float64 -> SchemaError.
    #[test]
    fn test_wrong_type_gamma_0_as_int32() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("hydro_id", DataType::Int32, false),
            Field::new("plane_id", DataType::Int32, false),
            Field::new("gamma_0", DataType::Int32, false), // wrong type
            Field::new("gamma_v", DataType::Float64, false),
            Field::new("gamma_q", DataType::Float64, false),
            Field::new("gamma_s", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![66])),
                Arc::new(Int32Array::from(vec![0])),
                Arc::new(Int32Array::from(vec![1250])), // wrong type
                Arc::new(Float64Array::from(vec![0.0023])),
                Arc::new(Float64Array::from(vec![0.892])),
                Arc::new(Float64Array::from(vec![-0.015])),
            ],
        )
        .unwrap();
        let tmp = write_parquet(&batch);
        let err = parse_fpha_hyperplanes(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, .. } => {
                assert_eq!(field, "gamma_0");
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── AC: sorted output (hydro_id, stage_id, plane_id) ─────────────────────

    /// Rows for two hydros in reverse order -> sorted by (hydro_id, stage_id, plane_id).
    #[test]
    fn test_sorted_output() {
        let batch = make_required_batch(
            &[20, 20, 5, 5, 5],
            &[1, 0, 2, 0, 1],
            &[200.0, 210.0, 300.0, 310.0, 305.0],
            &[0.002, 0.002, 0.003, 0.003, 0.003],
            &[0.8, 0.8, 0.7, 0.7, 0.7],
            &[-0.01, -0.01, -0.02, -0.02, -0.02],
        );
        let tmp = write_parquet(&batch);
        let rows = parse_fpha_hyperplanes(tmp.path()).unwrap();

        assert_eq!(rows.len(), 5);
        // First 3: hydro_id=5, sorted by plane_id
        assert_eq!(rows[0].hydro_id, EntityId::from(5));
        assert_eq!(rows[0].plane_id, 0);
        assert_eq!(rows[1].hydro_id, EntityId::from(5));
        assert_eq!(rows[1].plane_id, 1);
        assert_eq!(rows[2].hydro_id, EntityId::from(5));
        assert_eq!(rows[2].plane_id, 2);
        // Last 2: hydro_id=20, sorted by plane_id
        assert_eq!(rows[3].hydro_id, EntityId::from(20));
        assert_eq!(rows[3].plane_id, 0);
        assert_eq!(rows[4].hydro_id, EntityId::from(20));
        assert_eq!(rows[4].plane_id, 1);
    }

    // ── AC: null stage_id sorts before non-null ───────────────────────────────

    /// Rows for same hydro with null and non-null stage_id — null sorts first.
    #[test]
    fn test_null_stage_id_sorts_before_non_null() {
        let schema = full_schema();
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![10, 10, 10])),
                // stage_id: null, 1, 0
                Arc::new(Int32Array::from(vec![None, Some(1), Some(0)])),
                Arc::new(Int32Array::from(vec![0, 0, 0])),
                Arc::new(Float64Array::from(vec![100.0, 200.0, 300.0])),
                Arc::new(Float64Array::from(vec![0.001, 0.001, 0.001])),
                Arc::new(Float64Array::from(vec![0.9, 0.9, 0.9])),
                Arc::new(Float64Array::from(vec![-0.01, -0.01, -0.01])),
                Arc::new(Float64Array::from(vec![1.0, 1.0, 1.0])),
                Arc::new(Float64Array::from(vec![None::<f64>; 3])),
                Arc::new(Float64Array::from(vec![None::<f64>; 3])),
                Arc::new(Float64Array::from(vec![None::<f64>; 3])),
            ],
        )
        .unwrap();
        let tmp = write_parquet(&batch);
        let rows = parse_fpha_hyperplanes(tmp.path()).unwrap();

        assert_eq!(rows.len(), 3);
        // null stage_id should sort first
        assert!(
            rows[0].stage_id.is_none(),
            "null stage_id should sort first"
        );
        assert_eq!(rows[1].stage_id, Some(0));
        assert_eq!(rows[2].stage_id, Some(1));
    }

    // ── AC: file not found -> IoError ─────────────────────────────────────────

    /// Non-existent path -> IoError with the matching path.
    #[test]
    fn test_file_not_found() {
        let path = Path::new("/nonexistent/path/fpha_hyperplanes.parquet");
        let err = parse_fpha_hyperplanes(path).unwrap_err();

        match &err {
            LoadError::IoError { path: p, .. } => {
                assert_eq!(p, path);
            }
            other => panic!("expected IoError, got: {other:?}"),
        }
    }

    // ── AC: empty file -> Ok(Vec::new()) ──────────────────────────────────────

    /// Empty Parquet (zero rows) -> Ok(Vec::new()).
    #[test]
    fn test_empty_parquet_returns_empty_vec() {
        let batch = make_required_batch(&[], &[], &[], &[], &[], &[]);
        let tmp = write_parquet(&batch);
        let rows = parse_fpha_hyperplanes(tmp.path()).unwrap();
        assert!(rows.is_empty(), "expected empty vec for empty Parquet");
    }

    // ── AC: optional validity range columns preserved ─────────────────────────

    /// Rows with valid_v_min_hm3, valid_v_max_hm3, valid_q_max_m3s — values preserved.
    #[test]
    fn test_optional_validity_ranges_preserved() {
        let schema = full_schema();
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![7])),
                Arc::new(Int32Array::from(vec![None::<i32>])),
                Arc::new(Int32Array::from(vec![0])),
                Arc::new(Float64Array::from(vec![1000.0])),
                Arc::new(Float64Array::from(vec![0.002])),
                Arc::new(Float64Array::from(vec![0.88])),
                Arc::new(Float64Array::from(vec![-0.01])),
                Arc::new(Float64Array::from(vec![0.97])),
                Arc::new(Float64Array::from(vec![5000.0])), // valid_v_min_hm3
                Arc::new(Float64Array::from(vec![25000.0])), // valid_v_max_hm3
                Arc::new(Float64Array::from(vec![1500.0])), // valid_q_max_m3s
            ],
        )
        .unwrap();
        let tmp = write_parquet(&batch);
        let rows = parse_fpha_hyperplanes(tmp.path()).unwrap();

        assert_eq!(rows.len(), 1);
        let row = &rows[0];
        assert_eq!(row.valid_v_min_hm3, Some(5000.0));
        assert_eq!(row.valid_v_max_hm3, Some(25000.0));
        assert_eq!(row.valid_q_max_m3s, Some(1500.0));
        assert!((row.kappa - 0.97).abs() < 1e-10);
    }

    // ── AC: declaration-order invariance ─────────────────────────────────────

    /// Reordering the Parquet rows does not change the output ordering.
    #[test]
    fn test_declaration_order_invariance() {
        let batch_asc = make_required_batch(
            &[1, 1, 5, 5],
            &[0, 1, 0, 1],
            &[100.0, 110.0, 200.0, 210.0],
            &[0.001, 0.001, 0.002, 0.002],
            &[0.9, 0.9, 0.8, 0.8],
            &[-0.01, -0.01, -0.01, -0.01],
        );
        let batch_desc = make_required_batch(
            &[5, 5, 1, 1],
            &[1, 0, 1, 0],
            &[210.0, 200.0, 110.0, 100.0],
            &[0.002, 0.002, 0.001, 0.001],
            &[0.8, 0.8, 0.9, 0.9],
            &[-0.01, -0.01, -0.01, -0.01],
        );

        let tmp_asc = write_parquet(&batch_asc);
        let tmp_desc = write_parquet(&batch_desc);
        let rows_asc = parse_fpha_hyperplanes(tmp_asc.path()).unwrap();
        let rows_desc = parse_fpha_hyperplanes(tmp_desc.path()).unwrap();

        let keys_asc: Vec<(i32, Option<i32>, i32)> = rows_asc
            .iter()
            .map(|r| (r.hydro_id.0, r.stage_id, r.plane_id))
            .collect();
        let keys_desc: Vec<(i32, Option<i32>, i32)> = rows_desc
            .iter()
            .map(|r| (r.hydro_id.0, r.stage_id, r.plane_id))
            .collect();

        assert_eq!(
            keys_asc, keys_desc,
            "output order must be (hydro_id, stage_id, plane_id)-sorted regardless of input"
        );
    }

    // ── AC: field values round-tripped correctly ──────────────────────────────

    /// All required field values are correctly preserved through the Parquet read path.
    #[test]
    fn test_field_values_preserved() {
        let batch = make_required_batch(&[42], &[3], &[987.654], &[0.00321], &[0.777], &[-0.00123]);
        let tmp = write_parquet(&batch);
        let rows = parse_fpha_hyperplanes(tmp.path()).unwrap();

        assert_eq!(rows.len(), 1);
        let row = &rows[0];
        assert_eq!(row.hydro_id, EntityId::from(42));
        assert_eq!(row.plane_id, 3);
        assert!((row.gamma_0 - 987.654).abs() < 1e-10);
        assert!((row.gamma_v - 0.00321).abs() < 1e-15);
        assert!((row.gamma_q - 0.777).abs() < 1e-10);
        assert!((row.gamma_s - (-0.00123)).abs() < 1e-15);
        assert!((row.kappa - 1.0).abs() < f64::EPSILON);
    }
}
