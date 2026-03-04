//! Shared Parquet column extraction helpers.
//!
//! These helpers centralise the typed-downcast logic used by every Parquet parser
//! in `cobre-io`. They are `pub(crate)` — not part of the public API.

use arrow::array::{Array, Date32Array, Float64Array, Int32Array};
use std::path::Path;

use crate::LoadError;

/// Extract a required column as [`Int32Array`] by name.
///
/// Returns `SchemaError` if the column is absent or has the wrong Arrow type.
pub(crate) fn extract_required_int32<'a>(
    batch: &'a arrow::record_batch::RecordBatch,
    name: &str,
    path: &Path,
) -> Result<&'a Int32Array, LoadError> {
    let col = batch
        .column_by_name(name)
        .ok_or_else(|| LoadError::SchemaError {
            path: path.to_path_buf(),
            field: name.to_string(),
            message: format!("missing required column \"{name}\""),
        })?;
    col.as_any()
        .downcast_ref::<Int32Array>()
        .ok_or_else(|| LoadError::SchemaError {
            path: path.to_path_buf(),
            field: name.to_string(),
            message: format!(
                "column \"{name}\" has type {} but Int32 is required",
                col.data_type()
            ),
        })
}

/// Extract a required column as [`Float64Array`] by name.
///
/// Returns `SchemaError` if the column is absent or has the wrong Arrow type.
pub(crate) fn extract_required_float64<'a>(
    batch: &'a arrow::record_batch::RecordBatch,
    name: &str,
    path: &Path,
) -> Result<&'a Float64Array, LoadError> {
    let col = batch
        .column_by_name(name)
        .ok_or_else(|| LoadError::SchemaError {
            path: path.to_path_buf(),
            field: name.to_string(),
            message: format!("missing required column \"{name}\""),
        })?;
    col.as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| LoadError::SchemaError {
            path: path.to_path_buf(),
            field: name.to_string(),
            message: format!(
                "column \"{name}\" has type {} but Float64 is required",
                col.data_type()
            ),
        })
}

/// Extract an optional column as [`Int32Array`] by name, returning `None` if absent.
///
/// Returns `SchemaError` if the column exists but has the wrong Arrow type.
pub(crate) fn extract_optional_int32<'a>(
    batch: &'a arrow::record_batch::RecordBatch,
    name: &str,
    path: &Path,
) -> Result<Option<&'a Int32Array>, LoadError> {
    let Some(col) = batch.column_by_name(name) else {
        return Ok(None);
    };
    let arr = col
        .as_any()
        .downcast_ref::<Int32Array>()
        .ok_or_else(|| LoadError::SchemaError {
            path: path.to_path_buf(),
            field: name.to_string(),
            message: format!(
                "column \"{name}\" has type {} but Int32 is required",
                col.data_type()
            ),
        })?;
    Ok(Some(arr))
}

/// Extract an optional column as [`Float64Array`] by name, returning `None` if absent.
///
/// Returns `SchemaError` if the column exists but has the wrong Arrow type.
pub(crate) fn extract_optional_float64<'a>(
    batch: &'a arrow::record_batch::RecordBatch,
    name: &str,
    path: &Path,
) -> Result<Option<&'a Float64Array>, LoadError> {
    let Some(col) = batch.column_by_name(name) else {
        return Ok(None);
    };
    let arr =
        col.as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| LoadError::SchemaError {
                path: path.to_path_buf(),
                field: name.to_string(),
                message: format!(
                    "column \"{name}\" has type {} but Float64 is required",
                    col.data_type()
                ),
            })?;
    Ok(Some(arr))
}

/// Extract a required column as [`Date32Array`] by name.
///
/// Returns `SchemaError` if the column is absent or has the wrong Arrow type.
pub(crate) fn extract_required_date32<'a>(
    batch: &'a arrow::record_batch::RecordBatch,
    name: &str,
    path: &Path,
) -> Result<&'a Date32Array, LoadError> {
    let col = batch
        .column_by_name(name)
        .ok_or_else(|| LoadError::SchemaError {
            path: path.to_path_buf(),
            field: name.to_string(),
            message: format!("missing required column \"{name}\""),
        })?;
    col.as_any()
        .downcast_ref::<Date32Array>()
        .ok_or_else(|| LoadError::SchemaError {
            path: path.to_path_buf(),
            field: name.to_string(),
            message: format!(
                "column \"{name}\" has type {} but Date32 is required",
                col.data_type()
            ),
        })
}
