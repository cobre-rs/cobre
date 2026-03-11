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

use std::fs;
use std::path::PathBuf;

use arrow::array::{Array, Float64Array, Int32Array, Int64Array};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use pyo3::exceptions::{PyFileNotFoundError, PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyList, PyString};

/// Convert a `serde_json::Value` to a Python object recursively.
fn json_value_to_py(py: Python<'_>, val: &serde_json::Value) -> PyResult<PyObject> {
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
///         "manifest": { ... },           # contents of training/_manifest.json
///         "metadata": { ... },           # contents of training/metadata.json
///         "convergence_path": "/abs/...", # absolute path to convergence.parquet
///         "timing_path": "/abs/...",      # absolute path to timing/iterations.parquet
///         "complete": True,               # whether training/_SUCCESS exists
///     },
///     "simulation": {
///         "manifest": { ... } | None,    # contents of simulation/_manifest.json, or None
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
pub fn load_results(py: Python<'_>, output_dir: PathBuf) -> PyResult<PyObject> {
    // Canonicalize to get absolute paths even if a relative path is given.
    let output_dir = output_dir.canonicalize().map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            PyFileNotFoundError::new_err(format!(
                "output directory not found: {}",
                output_dir.display()
            ))
        } else {
            PyOSError::new_err(format!(
                "failed to access output directory {}: {e}",
                output_dir.display()
            ))
        }
    })?;

    let training_dir = output_dir.join("training");
    let training_success = training_dir.join("_SUCCESS");

    if !training_success.exists() {
        return Err(PyFileNotFoundError::new_err(format!(
            "training run did not complete — no _SUCCESS marker at {}",
            training_success.display()
        )));
    }

    // Read training JSON files.
    let manifest_val = read_json_file(&training_dir.join("_manifest.json"))?;
    let metadata_val = read_json_file(&training_dir.join("metadata.json"))?;

    // Build absolute path strings for Parquet files.
    let convergence_path = training_dir
        .join("convergence.parquet")
        .to_string_lossy()
        .into_owned();
    let timing_path = training_dir
        .join("timing")
        .join("iterations.parquet")
        .to_string_lossy()
        .into_owned();

    // Build simulation section.
    let simulation_dir = output_dir.join("simulation");
    let sim_manifest_path = simulation_dir.join("_manifest.json");
    let sim_success = simulation_dir.join("_SUCCESS");

    let sim_manifest_py: PyObject = if sim_manifest_path.exists() {
        let sim_manifest_val = read_json_file(&sim_manifest_path)?;
        json_value_to_py(py, &sim_manifest_val)?
    } else {
        py.None()
    };
    let sim_complete = sim_success.exists();

    // Assemble result dict.
    let result = PyDict::new(py);

    let training_dict = PyDict::new(py);
    training_dict.set_item("manifest", json_value_to_py(py, &manifest_val)?)?;
    training_dict.set_item("metadata", json_value_to_py(py, &metadata_val)?)?;
    training_dict.set_item("convergence_path", &convergence_path)?;
    training_dict.set_item("timing_path", &timing_path)?;
    training_dict.set_item("complete", true)?;
    result.set_item("training", training_dict)?;

    let simulation_dict = PyDict::new(py);
    simulation_dict.set_item("manifest", sim_manifest_py)?;
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
pub fn load_convergence(py: Python<'_>, output_dir: PathBuf) -> PyResult<PyObject> {
    let output_dir = output_dir.canonicalize().map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            PyFileNotFoundError::new_err(format!(
                "output directory not found: {}",
                output_dir.display()
            ))
        } else {
            PyOSError::new_err(format!(
                "failed to access output directory {}: {e}",
                output_dir.display()
            ))
        }
    })?;

    let parquet_path = output_dir.join("training").join("convergence.parquet");

    let file = fs::File::open(&parquet_path).map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            PyFileNotFoundError::new_err(format!(
                "convergence.parquet not found: {}",
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

        // Downcast all columns; fail fast with descriptive errors.
        let col_iteration = batch
            .column_by_name("iteration")
            .ok_or_else(|| PyOSError::new_err("convergence.parquet missing 'iteration' column"))?
            .as_any()
            .downcast_ref::<Int32Array>()
            .ok_or_else(|| PyOSError::new_err("'iteration' column is not Int32"))?;

        let col_lower_bound = batch
            .column_by_name("lower_bound")
            .ok_or_else(|| PyOSError::new_err("convergence.parquet missing 'lower_bound' column"))?
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| PyOSError::new_err("'lower_bound' column is not Float64"))?;

        let col_upper_bound_mean = batch
            .column_by_name("upper_bound_mean")
            .ok_or_else(|| {
                PyOSError::new_err("convergence.parquet missing 'upper_bound_mean' column")
            })?
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| PyOSError::new_err("'upper_bound_mean' column is not Float64"))?;

        let col_upper_bound_std = batch
            .column_by_name("upper_bound_std")
            .ok_or_else(|| {
                PyOSError::new_err("convergence.parquet missing 'upper_bound_std' column")
            })?
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| PyOSError::new_err("'upper_bound_std' column is not Float64"))?;

        let col_gap_percent = batch
            .column_by_name("gap_percent")
            .ok_or_else(|| PyOSError::new_err("convergence.parquet missing 'gap_percent' column"))?
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| PyOSError::new_err("'gap_percent' column is not Float64"))?;

        let col_cuts_added = batch
            .column_by_name("cuts_added")
            .ok_or_else(|| PyOSError::new_err("convergence.parquet missing 'cuts_added' column"))?
            .as_any()
            .downcast_ref::<Int32Array>()
            .ok_or_else(|| PyOSError::new_err("'cuts_added' column is not Int32"))?;

        let col_cuts_removed = batch
            .column_by_name("cuts_removed")
            .ok_or_else(|| PyOSError::new_err("convergence.parquet missing 'cuts_removed' column"))?
            .as_any()
            .downcast_ref::<Int32Array>()
            .ok_or_else(|| PyOSError::new_err("'cuts_removed' column is not Int32"))?;

        let col_cuts_active = batch
            .column_by_name("cuts_active")
            .ok_or_else(|| PyOSError::new_err("convergence.parquet missing 'cuts_active' column"))?
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| PyOSError::new_err("'cuts_active' column is not Int64"))?;

        let col_time_forward_ms = batch
            .column_by_name("time_forward_ms")
            .ok_or_else(|| {
                PyOSError::new_err("convergence.parquet missing 'time_forward_ms' column")
            })?
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| PyOSError::new_err("'time_forward_ms' column is not Int64"))?;

        let col_time_backward_ms = batch
            .column_by_name("time_backward_ms")
            .ok_or_else(|| {
                PyOSError::new_err("convergence.parquet missing 'time_backward_ms' column")
            })?
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| PyOSError::new_err("'time_backward_ms' column is not Int64"))?;

        let col_time_total_ms = batch
            .column_by_name("time_total_ms")
            .ok_or_else(|| {
                PyOSError::new_err("convergence.parquet missing 'time_total_ms' column")
            })?
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| PyOSError::new_err("'time_total_ms' column is not Int64"))?;

        let col_forward_passes = batch
            .column_by_name("forward_passes")
            .ok_or_else(|| {
                PyOSError::new_err("convergence.parquet missing 'forward_passes' column")
            })?
            .as_any()
            .downcast_ref::<Int32Array>()
            .ok_or_else(|| PyOSError::new_err("'forward_passes' column is not Int32"))?;

        let col_lp_solves = batch
            .column_by_name("lp_solves")
            .ok_or_else(|| PyOSError::new_err("convergence.parquet missing 'lp_solves' column"))?
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| PyOSError::new_err("'lp_solves' column is not Int64"))?;

        for i in 0..n_rows {
            let row = PyDict::new(py);

            row.set_item("iteration", col_iteration.value(i))?;
            row.set_item("lower_bound", col_lower_bound.value(i))?;
            row.set_item("upper_bound_mean", col_upper_bound_mean.value(i))?;
            row.set_item("upper_bound_std", col_upper_bound_std.value(i))?;

            // gap_percent is nullable — return None when the cell is null.
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
