//! Result loading functions exposed as `cobre.results`.
//!
//! Provides lightweight inspection of output artifacts written by
//! `cobre.run.run()`. JSON manifest and metadata files are read in Rust
//! and returned as Python dicts. Parquet file paths are returned as strings
//! so that callers can load them with `polars` or `pandas`.
//!
//! ## Design
//!
//! - [`load_results`] reads the JSON manifest/metadata files and returns a
//!   nested dict with training and simulation sections.
//! - [`load_convergence`] reads `training/convergence.parquet` using the
//!   `parquet` + `arrow` crates and returns a list of dicts (one per row).
//! - [`load_simulation`] reads Hive-partitioned Parquet files under
//!   `simulation/{entity_type}/scenario_id=NNNN/data.parquet` with dynamic
//!   schema discovery and returns rows as Python dicts.
//! - [`load_policy`] reads a `FlatBuffers` policy checkpoint from
//!   `training/policy/` via `cobre_io::read_policy_checkpoint` and returns
//!   a nested Python dict.

use std::fs;
use std::path::Path;
use std::path::PathBuf;

use arrow::array::{Array, BooleanArray, Float64Array, Int32Array, Int64Array, Int8Array};
use arrow::compute::concat_batches;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::ipc::writer::StreamWriter;
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use pyo3::exceptions::{PyFileNotFoundError, PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyList, PyString};
use pyo3::BoundObject;

/// Canonicalize a path and return an appropriate Python error on failure.
fn canonicalize_dir(path: &Path) -> PyResult<PathBuf> {
    path.canonicalize().map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            PyFileNotFoundError::new_err(format!("directory not found: {}", path.display()))
        } else {
            PyOSError::new_err(format!("failed to access {}: {e}", path.display()))
        }
    })
}

/// Convert a `serde_json::Value` to a Python object recursively.
fn json_value_to_py(py: Python<'_>, val: &serde_json::Value) -> PyResult<Py<PyAny>> {
    match val {
        serde_json::Value::Null => Ok(py.None()),

        serde_json::Value::Bool(b) => Ok(PyBool::new(py, *b).to_owned().unbind().into()),

        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.into_pyobject(py)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?
                    .unbind()
                    .into())
            } else if let Some(u) = n.as_u64() {
                Ok(u.into_pyobject(py)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?
                    .unbind()
                    .into())
            } else {
                let f = n.as_f64().ok_or_else(|| {
                    PyValueError::new_err("JSON number is not representable as f64")
                })?;
                Ok(f.into_pyobject(py)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?
                    .unbind()
                    .into())
            }
        }

        serde_json::Value::String(s) => Ok(PyString::new(py, s).unbind().into()),

        serde_json::Value::Array(arr) => {
            let list = PyList::empty(py);
            for item in arr {
                list.append(json_value_to_py(py, item)?)?;
            }
            Ok(list.unbind().into())
        }

        serde_json::Value::Object(map) => {
            let dict = PyDict::new(py);
            for (k, v) in map {
                dict.set_item(k, json_value_to_py(py, v)?)?;
            }
            Ok(dict.unbind().into())
        }
    }
}

/// Read a JSON file and return its contents as a `serde_json::Value`.
fn read_json_file(path: &std::path::Path) -> PyResult<serde_json::Value> {
    let content = fs::read_to_string(path).map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            PyFileNotFoundError::new_err(format!("file not found: {}", path.display()))
        } else {
            PyOSError::new_err(format!("failed to read {}: {e}", path.display()))
        }
    })?;

    serde_json::from_str(&content)
        .map_err(|e| PyValueError::new_err(format!("malformed JSON in {}: {e}", path.display())))
}

/// Load and inspect the output artifacts produced by a completed solver run.
///
/// Returns a nested dict with the following structure:
///
/// ```python
/// {
///     "training": {
///         "manifest": { ... },           # contents of training/metadata.json
///         "metadata": { ... },           # contents of training/metadata.json
///         "convergence_path": "/abs/...", # absolute path to convergence.parquet
///         "timing_path": "/abs/...",      # absolute path to timing/iterations.parquet
///         "complete": True,               # whether training/_SUCCESS exists
///     },
///     "simulation": {
///         "manifest": { ... } | None,    # contents of simulation/metadata.json, or None
///         "complete": False,             # whether simulation/_SUCCESS exists
///     },
/// }
/// ```
///
/// # Errors
///
/// - `FileNotFoundError` if `output_dir` does not exist or `training/_SUCCESS`
///   is missing (indicating that the training run did not complete).
/// - `ValueError` if JSON files are malformed.
/// - `OSError` for other I/O errors.
///
/// # Examples (Python)
///
/// ```python
/// import cobre.results
///
/// result = cobre.results.load_results("output/")
/// print(result["training"]["manifest"]["status"])
/// df = polars.read_parquet(result["training"]["convergence_path"])
/// ```
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub fn load_results(py: Python<'_>, output_dir: PathBuf) -> PyResult<Py<PyAny>> {
    let output_dir = canonicalize_dir(&output_dir)?;

    let training_dir = output_dir.join("training");
    let training_success = training_dir.join("_SUCCESS");

    if !training_success.exists() {
        return Err(PyFileNotFoundError::new_err(format!(
            "training run did not complete — no _SUCCESS marker at {}",
            training_success.display()
        )));
    }

    let manifest_val = read_json_file(&training_dir.join("metadata.json"))?;
    let metadata_val = read_json_file(&training_dir.join("metadata.json"))?;

    let convergence_path = training_dir
        .join("convergence.parquet")
        .to_string_lossy()
        .into_owned();
    let timing_path = training_dir
        .join("timing")
        .join("iterations.parquet")
        .to_string_lossy()
        .into_owned();

    let simulation_dir = output_dir.join("simulation");
    let sim_manifest_path = simulation_dir.join("metadata.json");
    let sim_manifest = if sim_manifest_path.exists() {
        json_value_to_py(py, &read_json_file(&sim_manifest_path)?)?
    } else {
        py.None()
    };
    let sim_complete = simulation_dir.join("_SUCCESS").exists();

    let result = PyDict::new(py);

    let training_dict = PyDict::new(py);
    training_dict.set_item("manifest", json_value_to_py(py, &manifest_val)?)?;
    training_dict.set_item("metadata", json_value_to_py(py, &metadata_val)?)?;
    training_dict.set_item("convergence_path", &convergence_path)?;
    training_dict.set_item("timing_path", &timing_path)?;
    training_dict.set_item("complete", true)?;
    result.set_item("training", training_dict)?;

    let simulation_dict = PyDict::new(py);
    simulation_dict.set_item("manifest", sim_manifest)?;
    simulation_dict.set_item("complete", sim_complete)?;
    result.set_item("simulation", simulation_dict)?;

    Ok(result.unbind().into())
}

/// Read `training/convergence.parquet` and return its rows as a list of dicts.
///
/// Each dict in the returned list corresponds to one training iteration and
/// contains the following keys (matching the `training/convergence.parquet`
/// schema):
///
/// | Key                | Type            | Description                                         |
/// |--------------------|-----------------|-----------------------------------------------------|
/// | `iteration`        | `int`           | Iteration number (1-based).                         |
/// | `lower_bound`      | `float`         | Lower bound on the optimal value.                   |
/// | `upper_bound_mean` | `float`         | Mean upper bound across forward-pass scenarios.     |
/// | `upper_bound_std`  | `float`         | Std-dev of the upper bound.                         |
/// | `gap_percent`      | `float \| None` | Relative gap as a percentage (None if ill-defined). |
/// | `cuts_added`       | `int`           | Cuts added to the pool this iteration.              |
/// | `cuts_removed`     | `int`           | Cuts removed from the pool this iteration.          |
/// | `cuts_active`      | `int`           | Active cuts after this iteration.                   |
/// | `time_forward_ms`  | `int`           | Forward-pass wall time (ms).                        |
/// | `time_backward_ms` | `int`           | Backward-pass wall time (ms).                       |
/// | `time_total_ms`    | `int`           | Total iteration wall time (ms).                     |
/// | `forward_passes`   | `int`           | Number of forward-pass scenarios.                   |
/// | `lp_solves`        | `int`           | Total LP solves in this iteration.                  |
///
/// Returns an empty list if `training/convergence.parquet` has zero rows.
///
/// # Errors
///
/// - `FileNotFoundError` if `output_dir` or `training/convergence.parquet`
///   does not exist.
/// - `OSError` for other I/O errors or Parquet decoding failures.
///
/// # Examples (Python)
///
/// ```python
/// import cobre.results
///
/// rows = cobre.results.load_convergence("output/")
/// for row in rows:
///     print(row["iteration"], row["lower_bound"], row["upper_bound_mean"])
/// ```
#[pyfunction]
#[allow(clippy::too_many_lines, clippy::needless_pass_by_value)]
pub fn load_convergence(py: Python<'_>, output_dir: PathBuf) -> PyResult<Py<PyAny>> {
    let output_dir = canonicalize_dir(&output_dir)?;

    let parquet_path = output_dir.join("training").join("convergence.parquet");

    let file = fs::File::open(&parquet_path).map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            PyFileNotFoundError::new_err(format!(
                "parquet file not found: {}",
                parquet_path.display()
            ))
        } else {
            PyOSError::new_err(format!("failed to open {}: {e}", parquet_path.display()))
        }
    })?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| PyOSError::new_err(format!("failed to open Parquet file: {e}")))?;

    let reader = builder
        .build()
        .map_err(|e| PyOSError::new_err(format!("failed to build Parquet reader: {e}")))?;

    let result_list = PyList::empty(py);

    for batch_result in reader {
        let batch = batch_result
            .map_err(|e| PyOSError::new_err(format!("error reading Parquet batch: {e}")))?;

        let n_rows = batch.num_rows();

        let col_iteration = get_column_by_name::<Int32Array>(&batch, "iteration")?;
        let col_lower_bound = get_column_by_name::<Float64Array>(&batch, "lower_bound")?;
        let col_upper_bound_mean = get_column_by_name::<Float64Array>(&batch, "upper_bound_mean")?;
        let col_upper_bound_std = get_column_by_name::<Float64Array>(&batch, "upper_bound_std")?;
        let col_gap_percent = get_column_by_name::<Float64Array>(&batch, "gap_percent")?;
        let col_cuts_added = get_column_by_name::<Int32Array>(&batch, "cuts_added")?;
        let col_cuts_removed = get_column_by_name::<Int32Array>(&batch, "cuts_removed")?;
        let col_cuts_active = get_column_by_name::<Int64Array>(&batch, "cuts_active")?;
        let col_time_forward_ms = get_column_by_name::<Int64Array>(&batch, "time_forward_ms")?;
        let col_time_backward_ms = get_column_by_name::<Int64Array>(&batch, "time_backward_ms")?;
        let col_time_total_ms = get_column_by_name::<Int64Array>(&batch, "time_total_ms")?;
        let col_forward_passes = get_column_by_name::<Int32Array>(&batch, "forward_passes")?;
        let col_lp_solves = get_column_by_name::<Int64Array>(&batch, "lp_solves")?;

        for i in 0..n_rows {
            let row = PyDict::new(py);

            row.set_item("iteration", col_iteration.value(i))?;
            row.set_item("lower_bound", col_lower_bound.value(i))?;
            row.set_item("upper_bound_mean", col_upper_bound_mean.value(i))?;
            row.set_item("upper_bound_std", col_upper_bound_std.value(i))?;

            if col_gap_percent.is_null(i) {
                row.set_item("gap_percent", py.None())?;
            } else {
                row.set_item("gap_percent", col_gap_percent.value(i))?;
            }

            row.set_item("cuts_added", col_cuts_added.value(i))?;
            row.set_item("cuts_removed", col_cuts_removed.value(i))?;
            row.set_item("cuts_active", col_cuts_active.value(i))?;
            row.set_item("time_forward_ms", col_time_forward_ms.value(i))?;
            row.set_item("time_backward_ms", col_time_backward_ms.value(i))?;
            row.set_item("time_total_ms", col_time_total_ms.value(i))?;
            row.set_item("forward_passes", col_forward_passes.value(i))?;
            row.set_item("lp_solves", col_lp_solves.value(i))?;

            result_list.append(row)?;
        }
    }

    Ok(result_list.unbind().into())
}

/// Read `training/convergence.parquet` and return its contents as a `pyarrow.Table`.
///
/// Reads the same file as [`load_convergence`] but returns the data in Arrow
/// IPC format, which `pyarrow` can deserialise without copying individual values.
/// The data is serialised to an in-memory Arrow IPC stream in Rust and handed
/// to `pyarrow.ipc.open_stream` on the Python side, which reconstructs a
/// `pyarrow.Table` from the stream. For convergence data (typically < 10 KB)
/// the single IPC buffer copy is negligible compared to the Python-object
/// allocation avoided by [`load_convergence`].
///
/// The returned `pyarrow.Table` can be consumed directly by `polars.from_arrow()`
/// or any library that supports the Arrow `PyCapsule` / interchange protocol.
///
/// # Requirements
///
/// `pyarrow` must be installed in the Python environment. If it is not present
/// the function raises `ImportError`.
///
/// # Schema
///
/// Matches the convergence Parquet schema written by the solver:
///
/// | Column              | Arrow type   | Nullable |
/// |---------------------|-------------|----------|
/// | `iteration`         | Int32        | No       |
/// | `lower_bound`       | Float64      | No       |
/// | `upper_bound_mean`  | Float64      | No       |
/// | `upper_bound_std`   | Float64      | No       |
/// | `gap_percent`       | Float64      | Yes      |
/// | `cuts_added`        | Int32        | No       |
/// | `cuts_removed`      | Int32        | No       |
/// | `cuts_active`       | Int64        | No       |
/// | `time_forward_ms`   | Int64        | No       |
/// | `time_backward_ms`  | Int64        | No       |
/// | `time_total_ms`     | Int64        | No       |
/// | `forward_passes`    | Int32        | No       |
/// | `lp_solves`         | Int64        | No       |
///
/// # Errors
///
/// - `FileNotFoundError` if `output_dir` or `training/convergence.parquet`
///   does not exist.
/// - `OSError` for Parquet decoding failures or IPC serialisation errors.
/// - `ImportError` if `pyarrow` is not installed.
///
/// # Examples (Python)
///
/// ```python
/// import cobre.results
/// import polars as pl
///
/// table = cobre.results.load_convergence_arrow("output/")
/// df = pl.from_arrow(table)
/// print(df.head())
/// ```
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub fn load_convergence_arrow(py: Python<'_>, output_dir: PathBuf) -> PyResult<Py<PyAny>> {
    let output_dir = canonicalize_dir(&output_dir)?;

    let parquet_path = output_dir.join("training").join("convergence.parquet");

    // Read all RecordBatches from the Parquet file with the GIL released.
    let ipc_bytes = py.detach(|| -> PyResult<Vec<u8>> {
        let file = fs::File::open(&parquet_path).map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                PyFileNotFoundError::new_err(format!(
                    "parquet file not found: {}",
                    parquet_path.display()
                ))
            } else {
                PyOSError::new_err(format!("failed to open {}: {e}", parquet_path.display()))
            }
        })?;

        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .map_err(|e| PyOSError::new_err(format!("failed to open Parquet file: {e}")))?;

        let schema = builder.schema().clone();

        let reader = builder
            .build()
            .map_err(|e| PyOSError::new_err(format!("failed to build Parquet reader: {e}")))?;

        // Serialise all batches into a single Arrow IPC stream buffer.
        let mut buf = Vec::new();
        let mut writer = StreamWriter::try_new(&mut buf, &schema)
            .map_err(|e| PyOSError::new_err(format!("failed to create IPC writer: {e}")))?;

        for batch_result in reader {
            let batch = batch_result
                .map_err(|e| PyOSError::new_err(format!("error reading Parquet batch: {e}")))?;
            writer
                .write(&batch)
                .map_err(|e| PyOSError::new_err(format!("failed to write IPC batch: {e}")))?;
        }

        writer
            .finish()
            .map_err(|e| PyOSError::new_err(format!("failed to finish IPC stream: {e}")))?;

        Ok(buf)
    })?;

    // Import pyarrow.ipc and reconstruct the Table from the IPC bytes.
    // ImportError propagates naturally if pyarrow is not installed.
    let pa_ipc = py.import("pyarrow.ipc")?;
    let py_bytes = pyo3::types::PyBytes::new(py, &ipc_bytes);
    let reader = pa_ipc.call_method1("open_stream", (py_bytes,))?;
    let table = reader.call_method0("read_all")?;

    Ok(table.unbind())
}

/// Extract a column from a batch and downcast to its expected type, or return an error.
fn get_column_by_name<'a, T: Array + 'static>(
    batch: &'a RecordBatch,
    name: &str,
) -> PyResult<&'a T> {
    batch
        .column_by_name(name)
        .ok_or_else(|| PyOSError::new_err(format!("convergence.parquet missing '{name}' column")))?
        .as_any()
        .downcast_ref::<T>()
        .ok_or_else(|| PyOSError::new_err(format!("'{name}' column has unexpected type")))
}

/// Convert an Arrow column value at row `i` to a Python object based on the array's data type.
///
/// Handles the Arrow types present in simulation output schemas:
/// `Float64`, `Int32`, `Int64`, `Int8`, and `Boolean`. Nullable columns
/// return `None` when the cell is null. Unsupported types fall back to a
/// string representation via `format!("{:?}", data_type)`.
fn arrow_value_to_py(py: Python<'_>, col: &dyn Array, i: usize) -> PyResult<Py<PyAny>> {
    if col.is_null(i) {
        return Ok(py.None());
    }

    match col.data_type() {
        DataType::Float64 => {
            let arr = col
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| PyOSError::new_err("Float64 column downcast failed"))?;
            Ok(arr
                .value(i)
                .into_pyobject(py)
                .map_err(|e| PyValueError::new_err(e.to_string()))?
                .unbind()
                .into())
        }
        DataType::Int32 => {
            let arr = col
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| PyOSError::new_err("Int32 column downcast failed"))?;
            Ok(arr
                .value(i)
                .into_pyobject(py)
                .map_err(|e| PyValueError::new_err(e.to_string()))?
                .unbind()
                .into())
        }
        DataType::Int64 => {
            let arr = col
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| PyOSError::new_err("Int64 column downcast failed"))?;
            Ok(arr
                .value(i)
                .into_pyobject(py)
                .map_err(|e| PyValueError::new_err(e.to_string()))?
                .unbind()
                .into())
        }
        DataType::Int8 => {
            let arr = col
                .as_any()
                .downcast_ref::<Int8Array>()
                .ok_or_else(|| PyOSError::new_err("Int8 column downcast failed"))?;
            Ok(i32::from(arr.value(i))
                .into_pyobject(py)
                .map_err(|e| PyValueError::new_err(e.to_string()))?
                .unbind()
                .into())
        }
        DataType::Boolean => {
            let arr = col
                .as_any()
                .downcast_ref::<BooleanArray>()
                .ok_or_else(|| PyOSError::new_err("Boolean column downcast failed"))?;
            Ok(PyBool::new(py, arr.value(i)).to_owned().unbind().into())
        }
        other => Ok(PyString::new(py, &format!("<unsupported type: {other}>"))
            .unbind()
            .into()),
    }
}

/// Convert a value to a Python object, mapping errors appropriately.
fn into_py<'py, T>(py: Python<'py>, val: T) -> PyResult<Py<PyAny>>
where
    T: pyo3::IntoPyObject<'py>,
    <T as pyo3::IntoPyObject<'py>>::Error: std::fmt::Display,
{
    val.into_pyobject(py)
        .map_err(|e| PyValueError::new_err(e.to_string()))
        .map(|b| b.into_any().unbind())
}

/// Convert a [`cobre_io::OutputError`] to an appropriate Python exception.
///
/// - [`cobre_io::OutputError::IoError`] with `NotFound` kind → `FileNotFoundError`
/// - All other variants → `OSError`
fn output_error_to_py(err: &cobre_io::OutputError) -> PyErr {
    match err {
        cobre_io::OutputError::IoError { source, .. }
            if source.kind() == std::io::ErrorKind::NotFound =>
        {
            PyFileNotFoundError::new_err(err.to_string())
        }
        _ => PyOSError::new_err(err.to_string()),
    }
}

/// Read one `scenario_id=NNNN/data.parquet` partition and append rows to `result_list`.
///
/// Each row is a Python dict of column values. The `scenario_id` integer is injected
/// into every row from `scenario_id_val`.
fn read_parquet_partition_into(
    py: Python<'_>,
    parquet_path: &std::path::Path,
    scenario_id_val: i64,
    result_list: &Bound<'_, PyList>,
) -> PyResult<()> {
    let file = fs::File::open(parquet_path).map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            PyFileNotFoundError::new_err(format!(
                "simulation Parquet file not found: {}",
                parquet_path.display()
            ))
        } else {
            PyOSError::new_err(format!("failed to open {}: {e}", parquet_path.display()))
        }
    })?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(|e| {
        PyOSError::new_err(format!(
            "failed to open Parquet file {}: {e}",
            parquet_path.display()
        ))
    })?;

    let reader = builder.build().map_err(|e| {
        PyOSError::new_err(format!(
            "failed to build Parquet reader for {}: {e}",
            parquet_path.display()
        ))
    })?;

    for batch_result in reader {
        let batch = batch_result.map_err(|e| {
            PyOSError::new_err(format!(
                "error reading batch from {}: {e}",
                parquet_path.display()
            ))
        })?;

        let schema = batch.schema();
        let n_rows = batch.num_rows();

        for i in 0..n_rows {
            let row = PyDict::new(py);
            row.set_item("scenario_id", scenario_id_val)?;
            for (col_idx, field) in schema.fields().iter().enumerate() {
                let col = batch.column(col_idx);
                let val = arrow_value_to_py(py, col.as_ref(), i)?;
                row.set_item(field.name(), val)?;
            }
            result_list.append(row)?;
        }
    }

    Ok(())
}

/// Load simulation output rows for one entity type directory.
///
/// Enumerates `scenario_id=NNNN` subdirectories under `entity_dir`,
/// parses the `scenario_id` integer from the directory name, reads each
/// `data.parquet` file, and appends all rows (with the `scenario_id` field
/// injected) to a newly-allocated `PyList`.
///
/// Returns an empty list if the directory exists but contains no scenario
/// subdirectories.
fn load_entity_type(py: Python<'_>, entity_dir: &std::path::Path) -> PyResult<Py<PyList>> {
    let result_list = PyList::empty(py);

    let read_dir = fs::read_dir(entity_dir).map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            PyFileNotFoundError::new_err(format!(
                "simulation entity directory not found: {}",
                entity_dir.display()
            ))
        } else {
            PyOSError::new_err(format!(
                "failed to read directory {}: {e}",
                entity_dir.display()
            ))
        }
    })?;

    // Collect entries so we can sort them by scenario_id for deterministic order.
    let mut entries: Vec<(i64, std::path::PathBuf)> = Vec::new();

    for dir_entry in read_dir {
        let dir_entry = dir_entry.map_err(|e| {
            PyOSError::new_err(format!("failed to enumerate {}: {e}", entity_dir.display()))
        })?;

        let file_name = dir_entry.file_name();
        let name = file_name.to_string_lossy();

        // Only process directories matching `scenario_id=NNNN`.
        if !name.starts_with("scenario_id=") {
            continue;
        }

        let scenario_str = &name["scenario_id=".len()..];
        let scenario_id: i64 = scenario_str.parse().map_err(|_| {
            PyOSError::new_err(format!(
                "malformed scenario directory name '{name}': expected scenario_id=<integer>"
            ))
        })?;

        let parquet_path = dir_entry.path().join("data.parquet");
        entries.push((scenario_id, parquet_path));
    }

    entries.sort_by_key(|(id, _)| *id);

    for (scenario_id, parquet_path) in &entries {
        read_parquet_partition_into(py, parquet_path, *scenario_id, &result_list)?;
    }

    Ok(result_list.unbind())
}

/// Entity types supported by the simulation output.
const ENTITY_TYPES: &[&str] = &[
    "costs",
    "buses",
    "hydros",
    "thermals",
    "exchanges",
    "pumping_stations",
    "contracts",
    "non_controllables",
    "inflow_lags",
    "violations/generic",
];

/// Load simulation results from Hive-partitioned Parquet files.
///
/// Reads `simulation/{entity_type}/scenario_id=NNNN/data.parquet` files
/// and returns the rows with a `scenario_id` integer column added from the
/// partition path. Column schemas vary by entity type and are discovered
/// dynamically from the Parquet file metadata.
///
/// ## Parameters
///
/// - `output_dir` — root output directory (same as passed to `cobre.run.run()`).
/// - `entity_type` — optional entity type name (`"costs"`, `"buses"`, `"hydros"`,
///   `"thermals"`, or `"inflow_lags"`). When provided, only that entity type is
///   loaded and a flat list of dicts is returned. When `None`, all available
///   entity types are loaded and a dict of lists is returned.
///
/// ## Returns
///
/// - When `entity_type` is specified: a `list[dict]` — one dict per row.
/// - When `entity_type` is `None`: a `dict[str, list[dict]]` keyed by entity type.
///
/// ## Errors
///
/// - `FileNotFoundError` if `output_dir` does not exist.
/// - `FileNotFoundError` if a specific `entity_type` directory is absent.
/// - `OSError` for corrupt Parquet files or other I/O failures.
///
/// ## Examples (Python)
///
/// ```python
/// import cobre.results
///
/// # Load one entity type as a list of dicts
/// rows = cobre.results.load_simulation("output/", entity_type="costs")
/// for row in rows:
///     print(row["scenario_id"], row["stage_id"], row["total_cost"])
///
/// # Load all entity types as a dict of lists
/// data = cobre.results.load_simulation("output/")
/// hydro_rows = data["hydros"]
/// ```
#[pyfunction]
#[pyo3(signature = (output_dir, entity_type=None))]
#[allow(clippy::needless_pass_by_value)]
pub fn load_simulation(
    py: Python<'_>,
    output_dir: PathBuf,
    entity_type: Option<String>,
) -> PyResult<Py<PyAny>> {
    let output_dir = canonicalize_dir(&output_dir)?;

    let simulation_dir = output_dir.join("simulation");

    if !simulation_dir.exists() {
        return Err(PyFileNotFoundError::new_err(format!(
            "simulation directory not found: {}",
            simulation_dir.display()
        )));
    }

    if let Some(ref et) = entity_type {
        let entity_dir = simulation_dir.join(et);
        load_entity_type(py, &entity_dir).map(Py::from)
    } else {
        let result = PyDict::new(py);
        for et in ENTITY_TYPES {
            let entity_dir = simulation_dir.join(et);
            if entity_dir.exists() {
                let rows = load_entity_type(py, &entity_dir)?;
                result.set_item(et, rows)?;
            }
        }
        Ok(result.unbind().into())
    }
}

/// Read one `scenario_id=NNNN/data.parquet` partition into `RecordBatch`es with a
/// prepended `scenario_id` column.
///
/// Each batch from the file is extended with a leading `scenario_id` (Int64) column
/// whose value is `scenario_id_val` for every row. The extended batches are appended
/// to `out_batches`. The extended schema is written to `out_schema` on the first call
/// (i.e. when `out_schema` is `None`); subsequent calls assert the schema is
/// consistent within the entity type.
fn read_parquet_partition_as_batches(
    parquet_path: &std::path::Path,
    scenario_id_val: i64,
    out_batches: &mut Vec<RecordBatch>,
    out_schema: &mut Option<std::sync::Arc<Schema>>,
) -> PyResult<()> {
    let file = fs::File::open(parquet_path).map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            PyFileNotFoundError::new_err(format!(
                "simulation Parquet file not found: {}",
                parquet_path.display()
            ))
        } else {
            PyOSError::new_err(format!("failed to open {}: {e}", parquet_path.display()))
        }
    })?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(|e| {
        PyOSError::new_err(format!(
            "failed to open Parquet file {}: {e}",
            parquet_path.display()
        ))
    })?;

    let reader = builder.build().map_err(|e| {
        PyOSError::new_err(format!(
            "failed to build Parquet reader for {}: {e}",
            parquet_path.display()
        ))
    })?;

    for batch_result in reader {
        let batch = batch_result.map_err(|e| {
            PyOSError::new_err(format!(
                "error reading batch from {}: {e}",
                parquet_path.display()
            ))
        })?;

        let n_rows = batch.num_rows();
        let scenario_id_array = std::sync::Arc::new(Int64Array::from(vec![scenario_id_val; n_rows]))
            as std::sync::Arc<dyn arrow::array::Array>;

        let extended_schema = if let Some(schema) = out_schema.as_ref() {
            schema.clone()
        } else {
            let orig_schema = batch.schema();
            let mut fields: Vec<Field> = vec![Field::new("scenario_id", DataType::Int64, false)];
            fields.extend(orig_schema.fields().iter().map(|f| f.as_ref().clone()));
            let schema = std::sync::Arc::new(Schema::new(fields));
            *out_schema = Some(schema.clone());
            schema
        };

        let mut columns: Vec<std::sync::Arc<dyn arrow::array::Array>> =
            Vec::with_capacity(batch.num_columns() + 1);
        columns.push(scenario_id_array);
        columns.extend(batch.columns().iter().cloned());

        let extended_batch = RecordBatch::try_new(extended_schema, columns).map_err(|e| {
            PyOSError::new_err(format!("failed to construct extended RecordBatch: {e}"))
        })?;

        out_batches.push(extended_batch);
    }

    Ok(())
}

/// Load simulation output for one entity type as a concatenated `RecordBatch`.
///
/// Enumerates `scenario_id=NNNN` subdirectories under `entity_dir` sorted by
/// `scenario_id` ascending, reads each `data.parquet`, prepends a `scenario_id`
/// column, and concatenates all batches into a single `RecordBatch`.
///
/// Returns `None` when the directory exists but contains no scenario subdirectories.
fn load_entity_type_as_batch(
    entity_dir: &std::path::Path,
) -> PyResult<Option<(RecordBatch, std::sync::Arc<Schema>)>> {
    let read_dir = fs::read_dir(entity_dir).map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            PyFileNotFoundError::new_err(format!(
                "simulation entity directory not found: {}",
                entity_dir.display()
            ))
        } else {
            PyOSError::new_err(format!(
                "failed to read directory {}: {e}",
                entity_dir.display()
            ))
        }
    })?;

    let mut entries: Vec<(i64, std::path::PathBuf)> = Vec::new();

    for dir_entry in read_dir {
        let dir_entry = dir_entry.map_err(|e| {
            PyOSError::new_err(format!("failed to enumerate {}: {e}", entity_dir.display()))
        })?;

        let file_name = dir_entry.file_name();
        let name = file_name.to_string_lossy();

        if !name.starts_with("scenario_id=") {
            continue;
        }

        let scenario_str = &name["scenario_id=".len()..];
        let scenario_id: i64 = scenario_str.parse().map_err(|_| {
            PyOSError::new_err(format!(
                "malformed scenario directory name '{name}': expected scenario_id=<integer>"
            ))
        })?;

        let parquet_path = dir_entry.path().join("data.parquet");
        entries.push((scenario_id, parquet_path));
    }

    if entries.is_empty() {
        return Ok(None);
    }

    entries.sort_by_key(|(id, _)| *id);

    let mut all_batches: Vec<RecordBatch> = Vec::new();
    let mut schema: Option<std::sync::Arc<Schema>> = None;

    for (scenario_id, parquet_path) in &entries {
        read_parquet_partition_as_batches(
            parquet_path,
            *scenario_id,
            &mut all_batches,
            &mut schema,
        )?;
    }

    let Some(schema) = schema else {
        return Ok(None);
    };

    let concatenated = concat_batches(&schema, &all_batches)
        .map_err(|e| PyOSError::new_err(format!("failed to concatenate RecordBatches: {e}")))?;

    Ok(Some((concatenated, schema)))
}

/// Serialize a `RecordBatch` to an Arrow IPC stream buffer.
fn batch_to_ipc_bytes(batch: &RecordBatch, schema: &Schema) -> PyResult<Vec<u8>> {
    let mut buf = Vec::new();
    let mut writer = StreamWriter::try_new(&mut buf, schema)
        .map_err(|e| PyOSError::new_err(format!("failed to create IPC writer: {e}")))?;
    writer
        .write(batch)
        .map_err(|e| PyOSError::new_err(format!("failed to write IPC batch: {e}")))?;
    writer
        .finish()
        .map_err(|e| PyOSError::new_err(format!("failed to finish IPC stream: {e}")))?;
    Ok(buf)
}

/// Reconstruct a `pyarrow.Table` from a raw Arrow IPC stream buffer.
fn ipc_bytes_to_py_table<'py>(
    py: Python<'py>,
    ipc_bytes: &[u8],
) -> PyResult<Bound<'py, pyo3::PyAny>> {
    let pa_ipc = py.import("pyarrow.ipc")?;
    let py_bytes = pyo3::types::PyBytes::new(py, ipc_bytes);
    let reader = pa_ipc.call_method1("open_stream", (py_bytes,))?;
    reader.call_method0("read_all")
}

/// Load simulation results from Hive-partitioned Parquet files as a `pyarrow.Table`.
///
/// Reads `simulation/{entity_type}/scenario_id=NNNN/data.parquet` files and
/// returns the data as an Arrow table with a prepended `scenario_id` (Int64) column.
/// Data is serialised to an in-memory Arrow IPC stream in Rust and handed to
/// `pyarrow.ipc.open_stream` on the Python side, which reconstructs a
/// `pyarrow.Table` without per-row Python object construction.
///
/// ## Parameters
///
/// - `output_dir` — root output directory (same as passed to `cobre.run.run()`).
/// - `entity_type` — optional entity type name (`"costs"`, `"buses"`, `"hydros"`,
///   `"thermals"`, or `"inflow_lags"`). When provided, only that entity type is
///   loaded and a single `pyarrow.Table` is returned. When `None`, all available
///   entity types are loaded and a `dict[str, pyarrow.Table]` is returned.
///
/// ## Returns
///
/// - When `entity_type` is specified: a `pyarrow.Table` — all scenarios concatenated.
/// - When `entity_type` is `None`: a `dict[str, pyarrow.Table]` keyed by entity type.
///
/// The `scenario_id` column is always the first column in the returned table(s).
/// Scenarios are ordered by `scenario_id` ascending.
///
/// ## Requirements
///
/// `pyarrow` must be installed in the Python environment. If it is not present
/// the function raises `ImportError`.
///
/// ## Errors
///
/// - `FileNotFoundError` if `output_dir` or the simulation directory does not exist.
/// - `FileNotFoundError` if a specific `entity_type` directory is absent.
/// - `OSError` for corrupt Parquet files or IPC serialisation errors.
/// - `ImportError` if `pyarrow` is not installed.
///
/// ## Examples (Python)
///
/// ```python
/// import cobre.results
/// import polars as pl
///
/// # Load one entity type as a pyarrow.Table
/// table = cobre.results.load_simulation_arrow("output/", entity_type="costs")
/// df = pl.from_arrow(table)
/// print(df.head())
///
/// # Load all entity types as a dict of pyarrow.Tables
/// tables = cobre.results.load_simulation_arrow("output/")
/// hydro_df = pl.from_arrow(tables["hydros"])
/// ```
#[pyfunction]
#[pyo3(signature = (output_dir, entity_type=None))]
#[allow(clippy::needless_pass_by_value)]
pub fn load_simulation_arrow(
    py: Python<'_>,
    output_dir: PathBuf,
    entity_type: Option<String>,
) -> PyResult<Py<PyAny>> {
    let output_dir = canonicalize_dir(&output_dir)?;

    let simulation_dir = output_dir.join("simulation");

    if !simulation_dir.exists() {
        return Err(PyFileNotFoundError::new_err(format!(
            "simulation directory not found: {}",
            simulation_dir.display()
        )));
    }

    if let Some(ref et) = entity_type {
        let entity_dir = simulation_dir.join(et);

        // Read all partitions with the GIL released.
        let ipc_bytes = py.detach(|| -> PyResult<Vec<u8>> {
            if let Some((batch, schema)) = load_entity_type_as_batch(&entity_dir)? {
                batch_to_ipc_bytes(&batch, &schema)
            } else {
                // Empty entity directory: return an empty IPC stream.
                // Build an empty schema with only the scenario_id column since
                // there are no Parquet files to discover the entity schema from.
                let empty_schema =
                    Schema::new(vec![Field::new("scenario_id", DataType::Int64, false)]);
                let empty_batch = RecordBatch::new_empty(std::sync::Arc::new(empty_schema.clone()));
                batch_to_ipc_bytes(&empty_batch, &empty_schema)
            }
        })?;

        let table = ipc_bytes_to_py_table(py, &ipc_bytes)?;
        Ok(table.unbind())
    } else {
        // Build a dict mapping each available entity type to its pyarrow.Table.
        let result = PyDict::new(py);

        for et in ENTITY_TYPES {
            let entity_dir = simulation_dir.join(et);
            if !entity_dir.exists() {
                continue;
            }

            let ipc_bytes = py.detach(|| -> PyResult<Vec<u8>> {
                if let Some((batch, schema)) = load_entity_type_as_batch(&entity_dir)? {
                    batch_to_ipc_bytes(&batch, &schema)
                } else {
                    let empty_schema =
                        Schema::new(vec![Field::new("scenario_id", DataType::Int64, false)]);
                    let empty_batch =
                        RecordBatch::new_empty(std::sync::Arc::new(empty_schema.clone()));
                    batch_to_ipc_bytes(&empty_batch, &empty_schema)
                }
            })?;

            let table = ipc_bytes_to_py_table(py, &ipc_bytes)?;
            result.set_item(et, table)?;
        }

        Ok(result.unbind().into())
    }
}

/// Load a `FlatBuffers` policy checkpoint from `training/policy/`.
///
/// Reads the policy metadata, per-stage cut pools, and per-stage solver bases
/// written by `cobre-io`'s policy checkpoint writer and returns them as a
/// nested Python dict.
///
/// ## Returns
///
/// ```python
/// {
///     "metadata": {
///         "version": "1.0.0",
///         "completed_iterations": 128,
///         "state_dimension": 4,
///         ...
///     },
///     "stage_cuts": [
///         {
///             "stage_id": 0,
///             "state_dimension": 4,
///             "capacity": 100,
///             "warm_start_count": 0,
///             "populated_count": 50,
///             "cuts": [
///                 {
///                     "cut_id": 0,
///                     "slot_index": 0,
///                     "iteration": 1,
///                     "forward_pass_index": 0,
///                     "intercept": 42.0,
///                     "coefficients": [1.0, 2.0, ...],
///                     "is_active": True,
///                     "domination_count": 0,
///                 },
///                 ...
///             ]
///         },
///         ...
///     ],
///     "stage_bases": [
///         {
///             "stage_id": 0,
///             "iteration": 1,
///             "column_status": [0, 1, ...],
///             "row_status": [1, 0, ...],
///             "num_cut_rows": 50,
///         },
///         ...
///     ]
/// }
/// ```
///
/// ## Errors
///
/// - `FileNotFoundError` if `output_dir` or `training/policy/` does not exist.
/// - `OSError` for corrupt `FlatBuffers` files or other I/O failures.
///
/// ## Examples (Python)
///
/// ```python
/// import cobre.results
///
/// policy = cobre.results.load_policy("output/")
/// print(policy["metadata"]["completed_iterations"])
/// first_stage_cuts = policy["stage_cuts"][0]["cuts"]
/// ```
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub fn load_policy(py: Python<'_>, output_dir: PathBuf) -> PyResult<Py<PyAny>> {
    let output_dir = canonicalize_dir(&output_dir)?;

    let policy_dir = output_dir.join("training").join("policy");

    if !policy_dir.exists() {
        return Err(PyFileNotFoundError::new_err(format!(
            "policy directory not found: {}",
            policy_dir.display()
        )));
    }

    let checkpoint =
        cobre_io::read_policy_checkpoint(&policy_dir).map_err(|e| output_error_to_py(&e))?;

    // Convert metadata via serde_json (it derives Serialize).
    let metadata_json = serde_json::to_value(&checkpoint.metadata)
        .map_err(|e| PyValueError::new_err(format!("failed to serialize policy metadata: {e}")))?;
    let metadata_py = json_value_to_py(py, &metadata_json)?;

    let stage_cuts_list = PyList::empty(py);
    for sc in &checkpoint.stage_cuts {
        let sc_dict = PyDict::new(py);
        sc_dict.set_item("stage_id", into_py(py, sc.stage_id)?)?;
        sc_dict.set_item("state_dimension", into_py(py, sc.state_dimension)?)?;
        sc_dict.set_item("capacity", into_py(py, sc.capacity)?)?;
        sc_dict.set_item("warm_start_count", into_py(py, sc.warm_start_count)?)?;
        sc_dict.set_item("populated_count", into_py(py, sc.populated_count)?)?;

        let cuts_list = PyList::empty(py);
        for cut in &sc.cuts {
            let cut_dict = PyDict::new(py);
            cut_dict.set_item("cut_id", into_py(py, cut.cut_id)?)?;
            cut_dict.set_item("slot_index", into_py(py, cut.slot_index)?)?;
            cut_dict.set_item("iteration", into_py(py, cut.iteration)?)?;
            cut_dict.set_item("forward_pass_index", into_py(py, cut.forward_pass_index)?)?;
            cut_dict.set_item("intercept", into_py(py, cut.intercept)?)?;

            let coeffs_list = PyList::empty(py);
            for &c in &cut.coefficients {
                coeffs_list.append(into_py(py, c)?)?;
            }
            cut_dict.set_item("coefficients", coeffs_list)?;

            cut_dict.set_item("is_active", PyBool::new(py, cut.is_active).to_owned())?;
            cut_dict.set_item("domination_count", into_py(py, cut.domination_count)?)?;

            cuts_list.append(cut_dict)?;
        }

        sc_dict.set_item("cuts", cuts_list)?;
        stage_cuts_list.append(sc_dict)?;
    }

    let stage_bases_list = PyList::empty(py);
    for basis in &checkpoint.stage_bases {
        let basis_dict = PyDict::new(py);
        basis_dict.set_item("stage_id", into_py(py, basis.stage_id)?)?;
        basis_dict.set_item("iteration", into_py(py, basis.iteration)?)?;

        let col_status_list = PyList::empty(py);
        for &b in &basis.column_status {
            col_status_list.append(into_py(py, i32::from(b))?)?;
        }
        basis_dict.set_item("column_status", col_status_list)?;

        let row_status_list = PyList::empty(py);
        for &b in &basis.row_status {
            row_status_list.append(into_py(py, i32::from(b))?)?;
        }
        basis_dict.set_item("row_status", row_status_list)?;

        basis_dict.set_item("num_cut_rows", into_py(py, basis.num_cut_rows)?)?;

        stage_bases_list.append(basis_dict)?;
    }

    let result = PyDict::new(py);
    result.set_item("metadata", metadata_py)?;
    result.set_item("stage_cuts", stage_cuts_list)?;
    result.set_item("stage_bases", stage_bases_list)?;

    Ok(result.unbind().into())
}
