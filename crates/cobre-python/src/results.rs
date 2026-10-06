//! Result loading functions exposed as `cobre.results`.
//!
//! Provides lightweight inspection of output artifacts written by
//! `cobre.run.run()`. JSON manifest and metadata files are read in Rust
//! and returned as Python dicts. Parquet file paths are returned as strings
//! so that callers can load them with `polars` or `pandas`.

use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

use arrow::array::{
    Array, BooleanArray, Float64Array, Int8Array, Int32Array, Int64Array, StringArray, UInt32Array,
};
use arrow::compute::concat_batches;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::ipc::writer::StreamWriter;
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use pyo3::BoundObject;
use pyo3::exceptions::{PyFileNotFoundError, PyIndexError, PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyList, PyString};

use crate::errors::{ErrorSource, convert_error};

fn canonicalize_dir(path: &Path) -> PyResult<PathBuf> {
    path.canonicalize().map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            PyFileNotFoundError::new_err(format!("directory not found: {}", path.display()))
        } else {
            PyOSError::new_err(format!("failed to access {}: {e}", path.display()))
        }
    })
}

fn json_value_to_py(py: Python<'_>, val: &serde_json::Value) -> PyResult<Py<PyAny>> {
    match val {
        serde_json::Value::Null => Ok(py.None()),

        serde_json::Value::Bool(b) => Ok(PyBool::new(py, *b).to_owned().unbind().into()),

        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                into_py(py, i)
            } else if let Some(u) = n.as_u64() {
                into_py(py, u)
            } else {
                let f = n.as_f64().ok_or_else(|| {
                    PyValueError::new_err("JSON number is not representable as f64")
                })?;
                into_py(py, f)
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

fn read_json_file(path: &Path) -> PyResult<serde_json::Value> {
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

fn open_parquet_file(path: &Path, missing_label: &str) -> PyResult<fs::File> {
    fs::File::open(path).map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            PyFileNotFoundError::new_err(format!("{missing_label} not found: {}", path.display()))
        } else {
            PyOSError::new_err(format!("failed to open {}: {e}", path.display()))
        }
    })
}

/// Load and inspect the output artifacts produced by a completed solver run.
///
/// Returns a nested dict with the following structure:
///
/// ```python
/// {
///     "training": {
///         "manifest": { ... },           # alias for metadata (same contents)
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
    training_dict.set_item("manifest", json_value_to_py(py, &metadata_val)?)?;
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
/// | `upper_bound`      | `float`         | Upper bound estimate (mean if sampled, exact if enumerated). |
/// | `upper_bound_std`  | `float \| None` | Std-dev of the upper bound (None under an exact bound). |
/// | `upper_bound_kind` | `str`           | Bound regime: `"statistical"` or `"exact"`.         |
/// | `gap_percent`      | `float \| None` | Relative gap as a percentage (None if ill-defined). |
/// | `cuts_added`       | `int`           | Cuts added to the pool this iteration.              |
/// | `cuts_removed`     | `int`           | Cuts removed from the pool this iteration.          |
/// | `cuts_active`      | `int`           | Active cuts after this iteration.                   |
/// | `time_forward_ms`  | `int`           | Forward-pass wall time (ms).                        |
/// | `time_backward_ms` | `int`           | Backward-pass wall time (ms).                       |
/// | `time_total_ms`    | `int`           | Total iteration wall time (ms).                     |
/// | `forward_passes`   | `int`           | Number of forward-pass scenarios.                   |
/// | `lp_solves`        | `int`           | Total LP solves in this iteration.                  |
/// | `mean_rows_in_lp`  | `float`         | Mean resident rows per lazy-selection LP solve (0 if none ran). |
///
/// The reader iterates the file's own schema, so the keys are exactly the
/// columns the convergence Parquet schema declares; the rows above name the
/// stable ones a caller indexes by hand.
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
///     print(row["iteration"], row["lower_bound"], row["upper_bound"])
/// ```
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub fn load_convergence(py: Python<'_>, output_dir: PathBuf) -> PyResult<Py<PyAny>> {
    let output_dir = canonicalize_dir(&output_dir)?;

    let parquet_path = output_dir.join("training").join("convergence.parquet");

    let file = open_parquet_file(&parquet_path, "parquet file")?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| PyOSError::new_err(format!("failed to open Parquet file: {e}")))?;

    let reader = builder
        .build()
        .map_err(|e| PyOSError::new_err(format!("failed to build Parquet reader: {e}")))?;

    let result_list = PyList::empty(py);

    for batch_result in reader {
        let batch = batch_result
            .map_err(|e| PyOSError::new_err(format!("error reading Parquet batch: {e}")))?;

        let schema = batch.schema();
        let n_rows = batch.num_rows();

        for i in 0..n_rows {
            let row = PyDict::new(py);
            for (col_idx, field) in schema.fields().iter().enumerate() {
                let col = batch.column(col_idx);
                let val: Py<PyAny> = if !col.is_null(i) && matches!(col.data_type(), DataType::Utf8)
                {
                    let arr = col
                        .as_any()
                        .downcast_ref::<StringArray>()
                        .ok_or_else(|| PyOSError::new_err("Utf8 column downcast failed"))?;
                    PyString::new(py, arr.value(i)).unbind().into()
                } else {
                    arrow_value_to_py(py, col.as_ref(), i)?
                };
                row.set_item(field.name(), val)?;
            }
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
/// The returned table carries the convergence Parquet schema written by the
/// solver (`convergence_schema` in `cobre-io`): its columns, types, and
/// nullability are exactly the fields that schema declares, so a schema change
/// flows through without editing this doc.
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

    let ipc_bytes = py.detach(|| -> PyResult<Vec<u8>> {
        let file = open_parquet_file(&parquet_path, "parquet file")?;

        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .map_err(|e| PyOSError::new_err(format!("failed to open Parquet file: {e}")))?;

        let schema = builder.schema().clone();

        let reader = builder
            .build()
            .map_err(|e| PyOSError::new_err(format!("failed to build Parquet reader: {e}")))?;

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

    let table = ipc_bytes_to_py_table(py, &ipc_bytes)?;

    Ok(table.unbind())
}

// ── Stochastic-model introspection (`cobre.results.load_stochastic`) ───────────

/// Number of columns in the flat `par_coefficients()` table:
/// `[hydro_id, stage_id, lag, coefficient]`.
const PAR_COEFFICIENT_COLUMNS: usize = 4;

/// Parsed rows of `stochastic/inflow_ar_coefficients.parquet`.
///
/// Stored as four parallel owned vectors (one per column), in the file's
/// on-disk row order (`(hydro_id, stage_id, lag)` ascending). All vectors share
/// the same length (`n_rows`).
struct ParRows {
    hydro_id: Vec<i32>,
    stage_id: Vec<i32>,
    lag: Vec<i32>,
    coefficient: Vec<f64>,
}

/// Parsed rows of `stochastic/noise_openings.parquet`.
///
/// Stored as four parallel owned vectors (one per column), in the file's
/// on-disk row order (`(stage_id, opening_index, entity_index)` ascending). All
/// vectors share the same length (`n_rows`).
struct OpeningRows {
    stage_id: Vec<i32>,
    opening_index: Vec<u32>,
    entity_index: Vec<u32>,
    value: Vec<f64>,
}

/// Open a Parquet file, mapping a missing file to a `FileNotFoundError` that
/// names the path and notes the `exports.stochastic` requirement.
///
/// Other open failures (permissions, etc.) map to `OSError`, mirroring
/// [`load_convergence_arrow`].
fn open_stochastic_parquet(path: &Path) -> PyResult<fs::File> {
    fs::File::open(path).map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            PyFileNotFoundError::new_err(format!(
                "{} not found — stochastic artifacts are written only when \
                 exports.stochastic is enabled in config.json",
                path.display()
            ))
        } else {
            PyOSError::new_err(format!("failed to open {}: {e}", path.display()))
        }
    })
}

fn stochastic_column<'a, T: Array + 'static>(
    batch: &'a RecordBatch,
    file: &str,
    name: &str,
) -> PyResult<&'a T> {
    batch
        .column_by_name(name)
        .ok_or_else(|| PyOSError::new_err(format!("{file} missing '{name}' column")))?
        .as_any()
        .downcast_ref::<T>()
        .ok_or_else(|| PyOSError::new_err(format!("{file}: '{name}' column has unexpected type")))
}

/// Read every record batch of a stochastic Parquet file into a flat owned
/// representation, with the GIL already released by the caller.
///
/// `extract` pulls the per-row values out of one decoded [`RecordBatch`] and
/// appends them to the accumulator. A missing file raises `FileNotFoundError`;
/// any decode failure raises `OSError`.
fn read_stochastic_parquet<A>(
    path: &Path,
    mut acc: A,
    mut extract: impl FnMut(&RecordBatch, &mut A) -> PyResult<()>,
) -> PyResult<A> {
    let file = open_stochastic_parquet(path)?;
    let display = path.display().to_string();

    let reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| PyOSError::new_err(format!("failed to open {display}: {e}")))?
        .build()
        .map_err(|e| PyOSError::new_err(format!("failed to build reader for {display}: {e}")))?;

    for batch_result in reader {
        let batch = batch_result
            .map_err(|e| PyOSError::new_err(format!("error reading {display}: {e}")))?;
        extract(&batch, &mut acc)?;
    }

    Ok(acc)
}

/// Read `stochastic/inflow_ar_coefficients.parquet` into [`ParRows`].
///
/// GIL-free: takes a `&Path` and returns owned vectors, raising `PyErr` only on
/// the not-found / decode paths. Columns are read by name so the read is robust
/// to schema column reordering.
fn read_par_rows(path: &Path) -> PyResult<ParRows> {
    let file = "inflow_ar_coefficients.parquet";
    read_stochastic_parquet(
        path,
        ParRows {
            hydro_id: Vec::new(),
            stage_id: Vec::new(),
            lag: Vec::new(),
            coefficient: Vec::new(),
        },
        |batch, rows| {
            let hydro_id = stochastic_column::<Int32Array>(batch, file, "hydro_id")?;
            let stage_id = stochastic_column::<Int32Array>(batch, file, "stage_id")?;
            let lag = stochastic_column::<Int32Array>(batch, file, "lag")?;
            let coefficient = stochastic_column::<Float64Array>(batch, file, "coefficient")?;

            rows.hydro_id.extend(hydro_id.values().iter().copied());
            rows.stage_id.extend(stage_id.values().iter().copied());
            rows.lag.extend(lag.values().iter().copied());
            rows.coefficient
                .extend(coefficient.values().iter().copied());
            Ok(())
        },
    )
}

/// Read `stochastic/noise_openings.parquet` into [`OpeningRows`].
///
/// GIL-free: takes a `&Path` and returns owned vectors, raising `PyErr` only on
/// the not-found / decode paths. Enforces the `(stage_id, opening_index,
/// entity_index)` sort order that `opening_tree`'s reshape relies on.
fn read_opening_rows(path: &Path) -> PyResult<OpeningRows> {
    let file = "noise_openings.parquet";
    let rows = read_stochastic_parquet(
        path,
        OpeningRows {
            stage_id: Vec::new(),
            opening_index: Vec::new(),
            entity_index: Vec::new(),
            value: Vec::new(),
        },
        |batch, rows| {
            let stage_id = stochastic_column::<Int32Array>(batch, file, "stage_id")?;
            let opening_index = stochastic_column::<UInt32Array>(batch, file, "opening_index")?;
            let entity_index = stochastic_column::<UInt32Array>(batch, file, "entity_index")?;
            let value = stochastic_column::<Float64Array>(batch, file, "value")?;

            rows.stage_id.extend(stage_id.values().iter().copied());
            rows.opening_index
                .extend(opening_index.values().iter().copied());
            rows.entity_index
                .extend(entity_index.values().iter().copied());
            rows.value.extend(value.values().iter().copied());
            Ok(())
        },
    )?;

    // `opening_tree`'s reshape is only correct on sorted rows; enforce it here so
    // a corrupted or third-party parquet fails loudly rather than returning a
    // scrambled array in a release build (a debug_assert would not fire there).
    if !is_opening_order_sorted(&rows) {
        return Err(PyOSError::new_err(
            "noise_openings.parquet rows are not sorted by \
             (stage_id, opening_index, entity_index); cannot reshape the opening tree",
        ));
    }

    Ok(rows)
}

/// True when `rows` are sorted ascending by `(stage_id, opening_index, entity_index)`,
/// the order `opening_tree` relies on for its reshape.
fn is_opening_order_sorted(rows: &OpeningRows) -> bool {
    rows.stage_id
        .iter()
        .zip(&rows.opening_index)
        .zip(&rows.entity_index)
        .map(|((s, o), e)| (*s, *o, *e))
        .is_sorted()
}

/// Read-only view of a run's fitted stochastic model, projected from the
/// on-disk artifacts under `{output_dir}/stochastic/`.
///
/// Construct with [`load_stochastic`]; the two accessor methods lazily import
/// `numpy` and return `float64` arrays. Constructing the handle does **not**
/// require `numpy`.
#[pyclass(name = "Stochastic", frozen, module = "cobre.results")]
pub struct Stochastic {
    par_rows: ParRows,
    opening_rows: OpeningRows,
}

#[pymethods]
impl Stochastic {
    /// Return the fitted PAR(p) coefficients as a `(n_rows, 4)` `float64` array.
    ///
    /// Columns, in fixed order, are
    /// `[hydro_id, stage_id, lag, coefficient]` — a lossless projection of
    /// `stochastic/inflow_ar_coefficients.parquet` in its on-disk row order
    /// (`(hydro_id, stage_id, lag)` ascending). The three integer columns are
    /// cast to `float64`. `lag` is 1-based (ψ₁ = lag 1).
    ///
    /// Lazily imports `numpy`; an `ImportError` propagates if it is absent.
    fn par_coefficients(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let n_rows = self.par_rows.hydro_id.len();
        let mut flat = Vec::with_capacity(n_rows * PAR_COEFFICIENT_COLUMNS);
        for i in 0..n_rows {
            flat.push(f64::from(self.par_rows.hydro_id[i]));
            flat.push(f64::from(self.par_rows.stage_id[i]));
            flat.push(f64::from(self.par_rows.lag[i]));
            flat.push(self.par_rows.coefficient[i]);
        }

        reshape_f64(py, flat, (n_rows, PAR_COEFFICIENT_COLUMNS))
    }

    /// Return the opening tree at 0-based `stage` as a `(n_openings, dim)`
    /// `float64` array.
    ///
    /// Row `k` is the noise vector of opening `k` at `stage`, with `dim` the
    /// number of distinct `entity_index` values within the stage. Values are
    /// taken from `stochastic/noise_openings.parquet`, which is stored in
    /// `(opening_index, entity_index)` order within each stage, so the flat
    /// slice reshapes directly.
    ///
    /// Raises `IndexError` if `stage` is not present in the file, and lazily
    /// imports `numpy` (`ImportError` propagates if absent).
    fn opening_tree(&self, py: Python<'_>, stage: usize) -> PyResult<Py<PyAny>> {
        let stage_i32 = i32::try_from(stage).map_err(|_| {
            PyIndexError::new_err(format!(
                "stage {stage} is out of range for the opening tree"
            ))
        })?;

        let rows = &self.opening_rows;
        let start = rows.stage_id.partition_point(|&s| s < stage_i32);
        let end = rows.stage_id.partition_point(|&s| s <= stage_i32);

        if start == end {
            let valid = stage_range_message(rows);
            return Err(PyIndexError::new_err(format!(
                "stage {stage} not present in the opening tree ({valid})"
            )));
        }

        // Rows are sorted, so the matching rows form a contiguous block; dim and
        // n_openings derive from the maxima within it.
        let opening_slice = &rows.opening_index[start..end];
        let entity_slice = &rows.entity_index[start..end];
        let value_slice = &rows.value[start..end];

        let n_openings = opening_slice
            .iter()
            .copied()
            .max()
            .map_or(0usize, |m| m as usize + 1);
        let dim = entity_slice
            .iter()
            .copied()
            .max()
            .map_or(0usize, |m| m as usize + 1);

        let expected = n_openings.checked_mul(dim).ok_or_else(|| {
            PyOSError::new_err("noise_openings.parquet: opening-tree dimensions overflow usize")
        })?;
        if value_slice.len() != expected {
            return Err(PyOSError::new_err(format!(
                "noise_openings.parquet: stage {stage} has {} values, expected {expected} \
                 (n_openings={n_openings} × dim={dim}) — the opening tree is ragged",
                value_slice.len()
            )));
        }

        reshape_f64(py, value_slice.to_vec(), (n_openings, dim))
    }
}

/// Lazily import `numpy` and return `np.asarray(flat).reshape(shape)`.
///
/// `flat` must already be in row-major order for `shape`. An `ImportError`
/// propagates verbatim if `numpy` is absent.
fn reshape_f64(py: Python<'_>, flat: Vec<f64>, shape: (usize, usize)) -> PyResult<Py<PyAny>> {
    let numpy = py.import("numpy")?;
    let array = numpy.call_method1("asarray", (flat,))?;
    let reshaped = array.call_method1("reshape", (shape,))?;
    Ok(reshaped.unbind())
}

/// Human-readable description of the stage values present in `rows`, for the
/// `IndexError` raised by [`Stochastic::opening_tree`] on an absent stage.
fn stage_range_message(rows: &OpeningRows) -> String {
    match (rows.stage_id.iter().min(), rows.stage_id.iter().max()) {
        (Some(&lo), Some(&hi)) => format!("valid stages are {lo}..={hi}"),
        _ => "the opening tree is empty".to_string(),
    }
}

/// Load a run's fitted stochastic model for read-only introspection.
///
/// Reads `{output_dir}/stochastic/inflow_ar_coefficients.parquet` and
/// `{output_dir}/stochastic/noise_openings.parquet` (written only when
/// `exports.stochastic` is enabled) and returns a [`Stochastic`] handle. The
/// parquet reads run with the GIL released; constructing the handle requires no
/// `numpy`.
///
/// # Errors
///
/// - `FileNotFoundError` — `output_dir` does not exist, or either required
///   parquet (or the `stochastic/` directory) is missing. The message names the
///   missing path and notes it requires `exports.stochastic`.
/// - `OSError` — a parquet file fails to decode.
///
/// # Examples (Python)
///
/// ```python
/// import cobre.results
///
/// stoch = cobre.results.load_stochastic("output/")
/// par = stoch.par_coefficients()        # (n_rows, 4) float64
/// tree = stoch.opening_tree(0)          # (n_openings, dim) float64 at stage 0
/// ```
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub fn load_stochastic(py: Python<'_>, output_dir: PathBuf) -> PyResult<Stochastic> {
    let output_dir = canonicalize_dir(&output_dir)?;
    let stochastic_dir = output_dir.join("stochastic");
    let par_path = stochastic_dir.join("inflow_ar_coefficients.parquet");
    let openings_path = stochastic_dir.join("noise_openings.parquet");

    let (par_rows, opening_rows) = py.detach(|| -> PyResult<(ParRows, OpeningRows)> {
        let par_rows = read_par_rows(&par_path)?;
        let opening_rows = read_opening_rows(&openings_path)?;
        Ok((par_rows, opening_rows))
    })?;

    Ok(Stochastic {
        par_rows,
        opening_rows,
    })
}

/// Convert an Arrow column value at row `i` to a Python object based on the array's data type.
///
/// Handles the Arrow types present in simulation output schemas (`Float64`,
/// `Int32`, `Int64`, `Int8`, `Boolean`); a null cell returns `None` and an
/// unsupported type falls back to a string placeholder.
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
            into_py(py, arr.value(i))
        }
        DataType::Int32 => {
            let arr = col
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| PyOSError::new_err("Int32 column downcast failed"))?;
            into_py(py, arr.value(i))
        }
        DataType::Int64 => {
            let arr = col
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| PyOSError::new_err("Int64 column downcast failed"))?;
            into_py(py, arr.value(i))
        }
        DataType::Int8 => {
            let arr = col
                .as_any()
                .downcast_ref::<Int8Array>()
                .ok_or_else(|| PyOSError::new_err("Int8 column downcast failed"))?;
            into_py(py, i32::from(arr.value(i)))
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

fn into_py<'py, T>(py: Python<'py>, val: T) -> PyResult<Py<PyAny>>
where
    T: pyo3::IntoPyObject<'py>,
    <T as pyo3::IntoPyObject<'py>>::Error: std::fmt::Display,
{
    val.into_pyobject(py)
        .map_err(|e| PyValueError::new_err(e.to_string()))
        .map(|b| b.into_any().unbind())
}

/// Build the `metadata` dict field-by-field from a [`cobre_io::CheckpointManifest`],
/// mirroring the `stage_cuts` surface in [`load_policy`] so the emitted dict shape
/// does not depend on a whole-struct serde path.
fn metadata_to_py<'py>(
    py: Python<'py>,
    metadata: &cobre_io::CheckpointManifest,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("format_version", into_py(py, metadata.format_version)?)?;
    dict.set_item("software", into_py(py, metadata.software.as_deref())?)?;
    dict.set_item(
        "software_version",
        into_py(py, metadata.software_version.as_str())?,
    )?;
    dict.set_item("created_at", into_py(py, metadata.created_at.as_str())?)?;
    dict.set_item("num_stages", into_py(py, metadata.num_stages)?)?;

    let graph = &metadata.graph_manifest;
    let graph_dict = PyDict::new(py);
    graph_dict.set_item("n_pools", into_py(py, graph.n_pools)?)?;
    let nodes = PyList::empty(py);
    for node in &graph.nodes {
        let node_dict = PyDict::new(py);
        node_dict.set_item("id", into_py(py, node.id)?)?;
        node_dict.set_item("stage_id", into_py(py, node.stage_id)?)?;
        node_dict.set_item("pool_id", into_py(py, node.pool_id)?)?;
        nodes.append(node_dict)?;
    }
    graph_dict.set_item("nodes", nodes)?;
    let edges = PyList::empty(py);
    for edge in &graph.edges {
        let edge_dict = PyDict::new(py);
        edge_dict.set_item("source_id", into_py(py, edge.source_id)?)?;
        edge_dict.set_item("target_id", into_py(py, edge.target_id)?)?;
        edge_dict.set_item("probability", into_py(py, edge.probability)?)?;
        edges.append(edge_dict)?;
    }
    graph_dict.set_item("edges", edges)?;
    dict.set_item("graph_manifest", graph_dict)?;

    let season = &metadata.season_manifest;
    let season_dict = PyDict::new(py);
    season_dict.set_item("cycle_code", into_py(py, season.cycle_code)?)?;
    season_dict.set_item("n_seasons", into_py(py, season.n_seasons)?)?;
    let hydro_orders = PyList::empty(py);
    for h in &season.hydro_orders {
        let h_dict = PyDict::new(py);
        h_dict.set_item("hydro_id", into_py(py, h.hydro_id)?)?;
        let orders = PyList::empty(py);
        for &o in &h.orders {
            orders.append(into_py(py, o)?)?;
        }
        h_dict.set_item("orders", orders)?;
        hydro_orders.append(h_dict)?;
    }
    season_dict.set_item("hydro_orders", hydro_orders)?;
    dict.set_item("season_manifest", season_dict)?;

    let producer = &metadata.producer;
    let producer_dict = PyDict::new(py);
    producer_dict.set_item(
        "completed_iterations",
        into_py(py, producer.completed_iterations)?,
    )?;
    producer_dict.set_item(
        "final_lower_bound",
        into_py(py, producer.final_lower_bound)?,
    )?;
    producer_dict.set_item("best_upper_bound", into_py(py, producer.best_upper_bound)?)?;
    producer_dict.set_item("max_iterations", into_py(py, producer.max_iterations)?)?;
    producer_dict.set_item("forward_passes", into_py(py, producer.forward_passes)?)?;
    producer_dict.set_item("warm_start_cuts", into_py(py, producer.warm_start_cuts)?)?;
    let warm_start_counts = PyList::empty(py);
    for &count in &producer.warm_start_counts {
        warm_start_counts.append(into_py(py, count)?)?;
    }
    producer_dict.set_item("warm_start_counts", warm_start_counts)?;
    producer_dict.set_item("rng_seed", into_py(py, producer.rng_seed)?)?;
    producer_dict.set_item(
        "total_visited_states",
        into_py(py, producer.total_visited_states)?,
    )?;
    producer_dict.set_item(
        "training_block_mode",
        into_py(py, producer.training_block_mode.as_str())?,
    )?;
    let per_stage = PyList::empty(py);
    for mode in &producer.training_block_mode_per_stage {
        per_stage.append(into_py(py, mode.as_str())?)?;
    }
    producer_dict.set_item("training_block_mode_per_stage", per_stage)?;
    producer_dict.set_item(
        "cost_scale_factor",
        into_py(py, producer.cost_scale_factor)?,
    )?;
    let lower_bound_history = PyList::empty(py);
    for &lb in &producer.lower_bound_history {
        lower_bound_history.append(into_py(py, lb)?)?;
    }
    producer_dict.set_item("lower_bound_history", lower_bound_history)?;
    dict.set_item("producer", producer_dict)?;

    Ok(dict)
}

/// Read one `scenario_id=NNNN/data.parquet` partition and append rows to `result_list`.
///
/// Each row is a Python dict of column values. The `scenario_id` integer is injected
/// into every row from `scenario_id_val`.
fn read_parquet_partition_into(
    py: Python<'_>,
    parquet_path: &Path,
    scenario_id_val: i64,
    result_list: &Bound<'_, PyList>,
) -> PyResult<()> {
    let file = open_parquet_file(parquet_path, "simulation Parquet file")?;

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

/// Collect the `scenario_id=NNNN` subdirectories of `entity_dir`, pairing each
/// parsed id with its `data.parquet` path, sorted ascending by id for
/// deterministic output order.
fn collect_sorted_scenario_entries(entity_dir: &Path) -> PyResult<Vec<(i64, PathBuf)>> {
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

    let mut entries: Vec<(i64, PathBuf)> = Vec::new();

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

    entries.sort_by_key(|(id, _)| *id);
    Ok(entries)
}

/// Load simulation output rows for one entity type directory into a `PyList` of
/// dicts (with the `scenario_id` field injected), ordered by `scenario_id`.
///
/// Returns an empty list if the directory exists but contains no scenario
/// subdirectories.
fn load_entity_type(py: Python<'_>, entity_dir: &Path) -> PyResult<Py<PyList>> {
    let result_list = PyList::empty(py);

    let entries = collect_sorted_scenario_entries(entity_dir)?;

    for (scenario_id, parquet_path) in &entries {
        read_parquet_partition_into(py, parquet_path, *scenario_id, &result_list)?;
    }

    Ok(result_list.unbind())
}

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
///   `"thermals"`, `"exchanges"`, `"pumping_stations"`, `"contracts"`,
///   `"non_controllables"`, `"inflow_lags"`, or `"violations/generic"`). When
///   provided, only that entity type is loaded and a flat list of dicts is
///   returned. When `None`, all available entity types are loaded and a dict of
///   lists is returned.
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
        for et in cobre_io::simulation_family_subpaths() {
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
/// leading `scenario_id` (Int64) column, appending them to `out_batches`.
///
/// `out_schema` is the extended schema, written on the first call (when `None`)
/// and reused by later calls so all batches share one schema for concatenation.
fn read_parquet_partition_as_batches(
    parquet_path: &Path,
    scenario_id_val: i64,
    out_batches: &mut Vec<RecordBatch>,
    out_schema: &mut Option<Arc<Schema>>,
) -> PyResult<()> {
    let file = open_parquet_file(parquet_path, "simulation Parquet file")?;

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

        // The file may already carry a `scenario_id` column (written as a real
        // column alongside the Hive partition). When it does, it is authoritative
        // — surface it as-is rather than prepending a second, duplicate column.
        if batch
            .schema()
            .fields()
            .iter()
            .any(|f| f.name() == "scenario_id")
        {
            if out_schema.is_none() {
                *out_schema = Some(batch.schema());
            }
            out_batches.push(batch);
            continue;
        }

        let n_rows = batch.num_rows();
        let scenario_id_array =
            Arc::new(Int64Array::from(vec![scenario_id_val; n_rows])) as Arc<dyn Array>;

        let extended_schema = if let Some(schema) = out_schema.as_ref() {
            schema.clone()
        } else {
            let orig_schema = batch.schema();
            let mut fields: Vec<Field> = vec![Field::new("scenario_id", DataType::Int64, false)];
            fields.extend(orig_schema.fields().iter().map(|f| f.as_ref().clone()));
            let schema = Arc::new(Schema::new(fields));
            *out_schema = Some(schema.clone());
            schema
        };

        let mut columns: Vec<Arc<dyn Array>> = Vec::with_capacity(batch.num_columns() + 1);
        columns.push(scenario_id_array);
        columns.extend(batch.columns().iter().cloned());

        let extended_batch = RecordBatch::try_new(extended_schema, columns).map_err(|e| {
            PyOSError::new_err(format!("failed to construct extended RecordBatch: {e}"))
        })?;

        out_batches.push(extended_batch);
    }

    Ok(())
}

/// Load simulation output for one entity type as a single `RecordBatch`
/// concatenated across scenarios (ordered by `scenario_id` ascending).
///
/// Returns `None` when the directory exists but contains no scenario subdirectories.
fn load_entity_type_as_batch(entity_dir: &Path) -> PyResult<Option<(RecordBatch, Arc<Schema>)>> {
    let entries = collect_sorted_scenario_entries(entity_dir)?;

    if entries.is_empty() {
        return Ok(None);
    }

    let mut all_batches: Vec<RecordBatch> = Vec::new();
    let mut schema: Option<Arc<Schema>> = None;

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

/// Serialize one entity type's concatenated batch to Arrow IPC bytes, falling
/// back to a `scenario_id`-only empty schema when the directory has no scenarios
/// (no Parquet file exists to discover the entity schema from).
fn entity_type_ipc_bytes(entity_dir: &Path) -> PyResult<Vec<u8>> {
    if let Some((batch, schema)) = load_entity_type_as_batch(entity_dir)? {
        batch_to_ipc_bytes(&batch, &schema)
    } else {
        let empty_schema = Schema::new(vec![Field::new("scenario_id", DataType::Int64, false)]);
        let empty_batch = RecordBatch::new_empty(Arc::new(empty_schema.clone()));
        batch_to_ipc_bytes(&empty_batch, &empty_schema)
    }
}

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
///   `"thermals"`, `"exchanges"`, `"pumping_stations"`, `"contracts"`,
///   `"non_controllables"`, `"inflow_lags"`, or `"violations/generic"`). When
///   provided, only that entity type is loaded and a single `pyarrow.Table` is
///   returned. When `None`, all available entity types are loaded and a
///   `dict[str, pyarrow.Table]` is returned.
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

        let ipc_bytes = py.detach(|| entity_type_ipc_bytes(&entity_dir))?;

        let table = ipc_bytes_to_py_table(py, &ipc_bytes)?;
        Ok(table.unbind())
    } else {
        let result = PyDict::new(py);

        for et in cobre_io::simulation_family_subpaths() {
            let entity_dir = simulation_dir.join(et);
            if !entity_dir.exists() {
                continue;
            }

            let ipc_bytes = py.detach(|| entity_type_ipc_bytes(&entity_dir))?;

            let table = ipc_bytes_to_py_table(py, &ipc_bytes)?;
            result.set_item(et, table)?;
        }

        Ok(result.unbind().into())
    }
}

/// Load a `FlatBuffers` policy checkpoint from `<output_dir>/<policy_subdir>`.
///
/// Reads the policy metadata, per-stage cut pools, and per-stage solver bases
/// written by `cobre-io`'s policy checkpoint writer and returns them as a
/// nested Python dict.
///
/// `policy_subdir` selects the checkpoint sub-directory under `output_dir` and
/// defaults to `"policy"` — the location the standard solve lifecycle writes
/// (matching the default `policy_path` of `"./policy"`). A study configured
/// with a non-default `policy_path` passes that sub-directory explicitly.
///
/// ## Returns
///
/// ```python
/// {
///     "metadata": {
///         "format_version": 3,
///         "software": "cobre",
///         "software_version": "1.0.0",
///         "created_at": "2026-01-15T12:00:00Z",
///         "num_stages": 60,
///         "graph_manifest": { "n_pools": 60, "nodes": [ ... ], "edges": [ ... ] },
///         "producer": {
///             "completed_iterations": 128,
///             ...
///         },
///     },
///     "stage_cuts": [
///         {
///             "stage_id": 0,
///             "state_dimension": 4,
///             "capacity": 100,
///             "warm_start_count": 0,
///             "populated_count": 50,
///             "cost_scale_factor": 2500000.0,
///             "node_id": 0,
///             "graph_stage_id": 0,
///             "priced_state_date": -2147483648,
///             "entity_manifest": [
///                 {
///                     "entity_type": 0,
///                     "entity_id": 0,
///                     "subindex": 0,
///                     "was_active": True,
///                     "reference_date": -2147483648,
///                     "interval_start": -2147483648,
///                     "interval_end": -2147483648,
///                 },
///                 ...
///             ],
///             "cuts": [
///                 {
///                     "cut_id": 0,
///                     "slot_index": 0,
///                     "iteration": 1,
///                     "forward_pass_index": 0,
///                     "intercept": 42.0,
///                     "coefficients": [1.0, 2.0, ...],
///                     "is_active": True,
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
/// - `FileNotFoundError` if `output_dir` does not exist, or if
///   `<output_dir>/<policy_subdir>` (for a symbolic link, its target) does not
///   exist and neither `.staging` nor `.previous` beside it holds a
///   `manifest.bin`.
/// - `OSError` for corrupt `FlatBuffers` files or other I/O failures.
///
/// ## Examples (Python)
///
/// ```python
/// import cobre.results
///
/// policy = cobre.results.load_policy("output/")
/// print(policy["metadata"]["producer"]["completed_iterations"])
/// first_stage_cuts = policy["stage_cuts"][0]["cuts"]
///
/// # Non-default policy_path: pass the sub-directory explicitly.
/// policy = cobre.results.load_policy("output/", policy_subdir="my_policy")
/// ```
#[pyfunction]
#[pyo3(signature = (output_dir, policy_subdir = "policy"))]
#[allow(clippy::needless_pass_by_value)]
pub fn load_policy(
    py: Python<'_>,
    output_dir: PathBuf,
    policy_subdir: &str,
) -> PyResult<Py<PyAny>> {
    let output_dir = canonicalize_dir(&output_dir)?;

    let policy_dir = output_dir.join(policy_subdir);

    let resolved = cobre_io::resolve_policy_checkpoint(&policy_dir)
        .map_err(|e| convert_error(ErrorSource::Output(&e)))?;
    if resolved == cobre_io::ResolvedCheckpoint::NoDirectory {
        return Err(PyFileNotFoundError::new_err(format!(
            "policy directory not found: {}",
            policy_dir.display()
        )));
    }

    let checkpoint = cobre_io::read_policy_checkpoint(&policy_dir)
        .map_err(|e| convert_error(ErrorSource::Output(&e)))?;

    let metadata_py = metadata_to_py(py, &checkpoint.metadata)?;

    let stage_cuts_list = PyList::empty(py);
    for sc in &checkpoint.stage_cuts {
        let sc_dict = PyDict::new(py);
        sc_dict.set_item("stage_id", into_py(py, sc.stage_id)?)?;
        sc_dict.set_item("state_dimension", into_py(py, sc.state_dimension)?)?;
        sc_dict.set_item("capacity", into_py(py, sc.capacity)?)?;
        sc_dict.set_item("warm_start_count", into_py(py, sc.warm_start_count)?)?;
        sc_dict.set_item("populated_count", into_py(py, sc.populated_count)?)?;

        // Surface the self-describing stage facts so a rewrite preserves them —
        // without node_id/graph_stage_id a genuine single-node pool collapses to
        // the missing-key sentinel and fails boundary-cut load.
        sc_dict.set_item("cost_scale_factor", into_py(py, sc.cost_scale_factor)?)?;
        sc_dict.set_item("node_id", into_py(py, sc.node_id)?)?;
        sc_dict.set_item("graph_stage_id", into_py(py, sc.graph_stage_id)?)?;
        sc_dict.set_item("priced_state_date", into_py(py, sc.priced_state_date)?)?;

        // Emit the per-slot entity manifest so a loaded checkpoint round-trips
        // through `write_policy_checkpoint` (whose binding already accepts this
        // exact shape). Without it the manifest — present on disk and required by
        // external boundary-cut authoring — is silently dropped on the read side.
        let manifest_list = PyList::empty(py);
        for slot in &sc.entity_manifest {
            let slot_dict = PyDict::new(py);
            slot_dict.set_item("entity_type", into_py(py, slot.entity_type)?)?;
            slot_dict.set_item("entity_id", into_py(py, slot.entity_id)?)?;
            slot_dict.set_item("subindex", into_py(py, slot.subindex)?)?;
            slot_dict.set_item("was_active", PyBool::new(py, slot.was_active).to_owned())?;
            slot_dict.set_item("reference_date", into_py(py, slot.reference_date)?)?;
            slot_dict.set_item("interval_start", into_py(py, slot.interval_start)?)?;
            slot_dict.set_item("interval_end", into_py(py, slot.interval_end)?)?;
            manifest_list.append(slot_dict)?;
        }
        sc_dict.set_item("entity_manifest", manifest_list)?;

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
