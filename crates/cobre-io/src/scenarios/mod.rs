//! Parsers for scenario data files in the `scenarios/` subdirectory.
//!
//! Scenario files carry the stochastic inputs used by the SDDP training loop:
//! inflow statistics, AR coefficients, historical observations, load statistics,
//! load scaling factors, spatial correlation profiles, and pre-computed external
//! scenarios. All files are optional; when absent the loader returns an empty
//! `Vec` (or default type) via the `load_*` wrapper functions.
//!
//! ## Parsing convention
//!
//! Parquet parsers follow the canonical pattern:
//!
//! 1. Open the file with `std::fs::File::open`.
//! 2. Build a `ParquetRecordBatchReaderBuilder` and consume all record batches.
//! 3. Extract typed columns by name; return `SchemaError` for missing or wrong-type columns.
//! 4. Validate per-row constraints; return `SchemaError` on violation.
//! 5. Sort the output by the documented sort key and return.
//!
//! JSON parsers follow the four-step pipeline:
//!
//! 1. Read the file with `std::fs::read_to_string`.
//! 2. Deserialize with `serde_json::from_str` through intermediate raw types.
//! 3. Validate with a `validate_raw` function.
//! 4. Convert raw types to public types and sort the output.
//!
//! Cross-reference validation (checking that referenced entity IDs exist in
//! their registries) is deferred to Layer 3 (Epic 06). Dimensional/semantic
//! constraints (lag contiguity, AR coefficient count matching, block count
//! matching) are deferred to Layer 3/5 (Epic 06).

pub mod ar_coefficients;
pub mod assembly;
pub mod correlation;
pub mod external;
pub mod inflow_history;
pub mod inflow_stats;
pub mod load_factors;
pub mod load_stats;

pub use ar_coefficients::{parse_inflow_ar_coefficients, InflowArCoefficientRow};
pub use assembly::{assemble_inflow_models, assemble_load_models};
pub use correlation::parse_correlation;
pub use external::{parse_external_scenarios, ExternalScenarioRow};
pub use inflow_history::{parse_inflow_history, InflowHistoryRow};
pub use inflow_stats::{parse_inflow_seasonal_stats, InflowSeasonalStatsRow};
pub use load_factors::{parse_load_factors, BlockFactor, LoadFactorEntry};
pub use load_stats::{parse_load_seasonal_stats, LoadSeasonalStatsRow};

use cobre_core::scenario::{CorrelationModel, InflowModel, LoadModel};

use crate::validation::structural::FileManifest;
use crate::LoadError;
use std::path::Path;

/// Load `scenarios/inflow_seasonal_stats.parquet` when the path is known, or
/// return an empty `Vec` when the file is absent (optional file).
///
/// When `path` is `None` (the structural validation step found no file at the
/// expected location), returns `Ok(Vec::new())` without touching the filesystem.
///
/// # Errors
///
/// Propagates [`LoadError`] from [`parse_inflow_seasonal_stats`] when `path` is `Some`.
///
/// # Examples
///
/// ```
/// use cobre_io::scenarios::load_inflow_seasonal_stats;
///
/// // No file present — returns empty vec.
/// let rows = load_inflow_seasonal_stats(None).expect("no file is fine");
/// assert!(rows.is_empty());
/// ```
pub fn load_inflow_seasonal_stats(
    path: Option<&Path>,
) -> Result<Vec<InflowSeasonalStatsRow>, LoadError> {
    match path {
        None => Ok(Vec::new()),
        Some(p) => parse_inflow_seasonal_stats(p),
    }
}

/// Load `scenarios/inflow_ar_coefficients.parquet` when the path is known, or
/// return an empty `Vec` when the file is absent (optional file).
///
/// When `path` is `None`, returns `Ok(Vec::new())` without touching the filesystem.
///
/// # Errors
///
/// Propagates [`LoadError`] from [`parse_inflow_ar_coefficients`] when `path` is `Some`.
///
/// # Examples
///
/// ```
/// use cobre_io::scenarios::load_inflow_ar_coefficients;
///
/// // No file present — returns empty vec.
/// let rows = load_inflow_ar_coefficients(None).expect("no file is fine");
/// assert!(rows.is_empty());
/// ```
pub fn load_inflow_ar_coefficients(
    path: Option<&Path>,
) -> Result<Vec<InflowArCoefficientRow>, LoadError> {
    match path {
        None => Ok(Vec::new()),
        Some(p) => parse_inflow_ar_coefficients(p),
    }
}

/// Load `scenarios/inflow_history.parquet` when the path is known, or return
/// an empty `Vec` when the file is absent (optional file).
///
/// When `path` is `None`, returns `Ok(Vec::new())` without touching the filesystem.
///
/// # Errors
///
/// Propagates [`LoadError`] from [`parse_inflow_history`] when `path` is `Some`.
///
/// # Examples
///
/// ```
/// use cobre_io::scenarios::load_inflow_history;
///
/// // No file present — returns empty vec.
/// let rows = load_inflow_history(None).expect("no file is fine");
/// assert!(rows.is_empty());
/// ```
pub fn load_inflow_history(path: Option<&Path>) -> Result<Vec<InflowHistoryRow>, LoadError> {
    match path {
        None => Ok(Vec::new()),
        Some(p) => parse_inflow_history(p),
    }
}

/// Load `scenarios/load_seasonal_stats.parquet` when the path is known, or
/// return an empty `Vec` when the file is absent (optional file).
///
/// When `path` is `None` (the structural validation step found no file at the
/// expected location), returns `Ok(Vec::new())` without touching the filesystem.
///
/// # Errors
///
/// Propagates [`LoadError`] from [`parse_load_seasonal_stats`] when `path` is `Some`.
///
/// # Examples
///
/// ```
/// use cobre_io::scenarios::load_load_seasonal_stats;
///
/// // No file present — returns empty vec.
/// let rows = load_load_seasonal_stats(None).expect("no file is fine");
/// assert!(rows.is_empty());
/// ```
pub fn load_load_seasonal_stats(
    path: Option<&Path>,
) -> Result<Vec<LoadSeasonalStatsRow>, LoadError> {
    match path {
        None => Ok(Vec::new()),
        Some(p) => parse_load_seasonal_stats(p),
    }
}

/// Load `scenarios/load_factors.json` when the path is known, or return an
/// empty `Vec` when the file is absent (optional file).
///
/// When `path` is `None`, returns `Ok(Vec::new())` without touching the filesystem.
///
/// # Errors
///
/// Propagates [`LoadError`] from [`parse_load_factors`] when `path` is `Some`.
///
/// # Examples
///
/// ```
/// use cobre_io::scenarios::load_load_factors;
///
/// // No file present — returns empty vec.
/// let entries = load_load_factors(None).expect("no file is fine");
/// assert!(entries.is_empty());
/// ```
pub fn load_load_factors(path: Option<&Path>) -> Result<Vec<LoadFactorEntry>, LoadError> {
    match path {
        None => Ok(Vec::new()),
        Some(p) => parse_load_factors(p),
    }
}

/// Load `scenarios/correlation.json` when the path is known, or return a
/// default [`CorrelationModel`] when the file is absent (optional file).
///
/// Unlike the Parquet parsers, `load_correlation` returns
/// `Ok(CorrelationModel::default())` for `None` (not an empty `Vec`), because
/// the target type is a structured model rather than a collection.
///
/// When `path` is `None`, returns `Ok(CorrelationModel::default())` without
/// touching the filesystem. The default model has an empty profiles map and
/// an empty schedule, meaning no inter-entity correlation is applied.
///
/// # Errors
///
/// Propagates [`LoadError`] from [`parse_correlation`] when `path` is `Some`.
///
/// # Examples
///
/// ```
/// use cobre_io::scenarios::load_correlation;
///
/// // No file present — returns default (empty) correlation model.
/// let model = load_correlation(None).expect("no file is fine");
/// assert!(model.profiles.is_empty());
/// assert!(model.schedule.is_empty());
/// ```
pub fn load_correlation(path: Option<&Path>) -> Result<CorrelationModel, LoadError> {
    match path {
        None => Ok(CorrelationModel::default()),
        Some(p) => parse_correlation(p),
    }
}

/// Load `scenarios/external_scenarios.parquet` when the path is known, or
/// return an empty `Vec` when the file is absent (optional file).
///
/// When `path` is `None`, returns `Ok(Vec::new())` without touching the filesystem.
///
/// # Errors
///
/// Propagates [`LoadError`] from [`parse_external_scenarios`] when `path` is `Some`.
///
/// # Examples
///
/// ```
/// use cobre_io::scenarios::load_external_scenarios;
///
/// // No file present — returns empty vec.
/// let rows = load_external_scenarios(None).expect("no file is fine");
/// assert!(rows.is_empty());
/// ```
pub fn load_external_scenarios(path: Option<&Path>) -> Result<Vec<ExternalScenarioRow>, LoadError> {
    match path {
        None => Ok(Vec::new()),
        Some(p) => parse_external_scenarios(p),
    }
}

// ── ScenarioData ──────────────────────────────────────────────────────────────

/// All assembled scenario pipeline data for one case directory.
///
/// Produced by [`load_scenarios`] after loading and assembling all seven
/// scenario files. Each field maps directly to the corresponding field in
/// [`cobre_core::System`].
///
/// Optional collections are empty `Vec`s when the corresponding file is absent
/// from the manifest. [`correlation`] is [`CorrelationModel::default()`] when
/// `scenarios/correlation.json` is absent.
///
/// # Examples
///
/// ```
/// use cobre_io::ScenarioData;
/// use cobre_core::scenario::CorrelationModel;
///
/// // Default-like state produced when all manifest flags are false.
/// let data = ScenarioData {
///     inflow_models: vec![],
///     load_models: vec![],
///     correlation: CorrelationModel::default(),
///     inflow_history: vec![],
///     external_scenarios: vec![],
///     load_factors: vec![],
/// };
/// assert!(data.inflow_models.is_empty());
/// assert!(data.correlation.profiles.is_empty());
/// ```
#[derive(Debug, Clone)]
pub struct ScenarioData {
    /// PAR(p) inflow models, sorted by `(hydro_id, stage_id)`.
    pub inflow_models: Vec<InflowModel>,
    /// Load seasonal statistics, sorted by `(bus_id, stage_id)`.
    pub load_models: Vec<LoadModel>,
    /// Correlation model (profiles + schedule).
    pub correlation: CorrelationModel,
    /// Inflow history rows, sorted by `(hydro_id, date)`. Empty when absent.
    pub inflow_history: Vec<InflowHistoryRow>,
    /// External scenario rows, sorted by `(stage_id, scenario_id, hydro_id)`. Empty when absent.
    pub external_scenarios: Vec<ExternalScenarioRow>,
    /// Load factor entries, sorted by `(bus_id, stage_id)`. Empty when absent.
    pub load_factors: Vec<LoadFactorEntry>,
}

// ── load_scenarios ────────────────────────────────────────────────────────────

/// Orchestrate loading of all seven scenario files and assemble the results.
///
/// Reads files indicated as present in `manifest`, calls the corresponding
/// `load_*` wrappers (passing `None` for absent files), and then assembles the
/// flat row types into the structured [`ScenarioData`] output.
///
/// `case_root` is the root case directory. File paths are constructed as
/// `case_root.join("scenarios/<filename>")`.
///
/// # Errors
///
/// | Condition                                              | Error variant              |
/// |--------------------------------------------------------|----------------------------|
/// | Any file read or parse failure                         | Propagated from parser     |
/// | AR coefficient rows without matching stats row         | [`LoadError::SchemaError`] |
/// | AR coefficient count does not match `ar_order`         | [`LoadError::SchemaError`] |
///
/// # Examples
///
/// ```no_run
/// use std::path::Path;
/// use cobre_io::validation::structural::FileManifest;
/// use cobre_io::scenarios::load_scenarios;
///
/// let manifest = FileManifest::default(); // all flags false
/// let data = load_scenarios(Path::new("/case"), &manifest)
///     .expect("empty manifest always succeeds");
/// assert!(data.inflow_models.is_empty());
/// assert!(data.correlation.profiles.is_empty());
/// ```
pub fn load_scenarios(
    case_root: &Path,
    manifest: &FileManifest,
) -> Result<ScenarioData, LoadError> {
    // Construct optional paths from manifest flags.
    let stats_path = manifest
        .scenarios_inflow_seasonal_stats_parquet
        .then(|| case_root.join("scenarios/inflow_seasonal_stats.parquet"));
    let coeff_path = manifest
        .scenarios_inflow_ar_coefficients_parquet
        .then(|| case_root.join("scenarios/inflow_ar_coefficients.parquet"));
    let history_path = manifest
        .scenarios_inflow_history_parquet
        .then(|| case_root.join("scenarios/inflow_history.parquet"));
    let load_stats_path = manifest
        .scenarios_load_seasonal_stats_parquet
        .then(|| case_root.join("scenarios/load_seasonal_stats.parquet"));
    let load_factors_path = manifest
        .scenarios_load_factors_json
        .then(|| case_root.join("scenarios/load_factors.json"));
    let correlation_path = manifest
        .scenarios_correlation_json
        .then(|| case_root.join("scenarios/correlation.json"));
    let external_path = manifest
        .scenarios_external_scenarios_parquet
        .then(|| case_root.join("scenarios/external_scenarios.parquet"));

    // Load each file (None → empty / default).
    let raw_stats = load_inflow_seasonal_stats(stats_path.as_deref())?;
    let raw_coefficients = load_inflow_ar_coefficients(coeff_path.as_deref())?;
    let inflow_history = load_inflow_history(history_path.as_deref())?;
    let raw_load_stats = load_load_seasonal_stats(load_stats_path.as_deref())?;
    let load_factors = load_load_factors(load_factors_path.as_deref())?;
    let correlation = load_correlation(correlation_path.as_deref())?;
    let external_scenarios = load_external_scenarios(external_path.as_deref())?;

    // Assemble inflow models (join stats + coefficients). Call even when both
    // are empty so orphaned coefficients are detected when stats is empty but
    // coefficients is not.
    let inflow_models = assemble_inflow_models(raw_stats, raw_coefficients)?;

    // Assemble load models (1:1 map).
    let load_models = assemble_load_models(raw_load_stats);

    Ok(ScenarioData {
        inflow_models,
        load_models,
        correlation,
        inflow_history,
        external_scenarios,
        load_factors,
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(
    clippy::doc_markdown,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::too_many_lines
)]
mod tests {
    use super::*;
    use crate::validation::structural::FileManifest;
    use tempfile::TempDir;

    /// `load_inflow_seasonal_stats(None)` returns `Ok(Vec::new())` without I/O.
    #[test]
    fn test_load_inflow_seasonal_stats_none_returns_empty() {
        let result = load_inflow_seasonal_stats(None).unwrap();
        assert!(result.is_empty());
    }

    /// `load_inflow_ar_coefficients(None)` returns `Ok(Vec::new())` without I/O.
    #[test]
    fn test_load_inflow_ar_coefficients_none_returns_empty() {
        let result = load_inflow_ar_coefficients(None).unwrap();
        assert!(result.is_empty());
    }

    /// `load_inflow_history(None)` returns `Ok(Vec::new())` without I/O.
    #[test]
    fn test_load_inflow_history_none_returns_empty() {
        let result = load_inflow_history(None).unwrap();
        assert!(result.is_empty());
    }

    /// `load_load_seasonal_stats(None)` returns `Ok(Vec::new())` without I/O.
    #[test]
    fn test_load_load_seasonal_stats_none_returns_empty() {
        let result = load_load_seasonal_stats(None).unwrap();
        assert!(result.is_empty());
    }

    /// `load_load_factors(None)` returns `Ok(Vec::new())` without I/O.
    #[test]
    fn test_load_load_factors_none_returns_empty() {
        let result = load_load_factors(None).unwrap();
        assert!(result.is_empty());
    }

    /// `load_correlation(None)` returns `Ok(CorrelationModel::default())` without I/O.
    #[test]
    fn test_load_correlation_none_returns_default() {
        let result = load_correlation(None).unwrap();
        assert!(result.profiles.is_empty());
        assert!(result.schedule.is_empty());
    }

    /// `load_external_scenarios(None)` returns `Ok(Vec::new())` without I/O.
    #[test]
    fn test_load_external_scenarios_none_returns_empty() {
        let result = load_external_scenarios(None).unwrap();
        assert!(result.is_empty());
    }

    /// `load_scenarios` with all manifest flags `false` returns empty collections
    /// and a default `CorrelationModel` without any filesystem access.
    #[test]
    fn test_load_scenarios_all_flags_false_returns_empty() {
        let dir = TempDir::new().unwrap();
        let manifest = FileManifest::default(); // all flags false

        let data =
            load_scenarios(dir.path(), &manifest).expect("empty manifest should always succeed");

        assert!(
            data.inflow_models.is_empty(),
            "inflow_models should be empty when no scenario files present"
        );
        assert!(
            data.load_models.is_empty(),
            "load_models should be empty when no scenario files present"
        );
        assert!(
            data.correlation.profiles.is_empty(),
            "correlation.profiles should be empty (default) when correlation.json absent"
        );
        assert!(
            data.correlation.schedule.is_empty(),
            "correlation.schedule should be empty (default) when correlation.json absent"
        );
        assert!(
            data.inflow_history.is_empty(),
            "inflow_history should be empty when file absent"
        );
        assert!(
            data.external_scenarios.is_empty(),
            "external_scenarios should be empty when file absent"
        );
        assert!(
            data.load_factors.is_empty(),
            "load_factors should be empty when file absent"
        );
    }
}
