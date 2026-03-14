//! Parsers for scenario data files in the `scenarios/` subdirectory.
//!
//! Scenario files carry the stochastic inputs used by multi-stage optimization solvers:
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
pub mod noise_openings;

pub use ar_coefficients::{parse_inflow_ar_coefficients, InflowArCoefficientRow};
pub use assembly::{assemble_inflow_models, assemble_load_models};
pub use correlation::parse_correlation;
pub use external::{parse_external_scenarios, ExternalScenarioRow};
pub use inflow_history::{parse_inflow_history, InflowHistoryRow};
pub use inflow_stats::{parse_inflow_seasonal_stats, InflowSeasonalStatsRow};
pub use load_factors::{parse_load_factors, BlockFactor, LoadFactorEntry};
pub use load_stats::{parse_load_seasonal_stats, LoadSeasonalStatsRow};
pub use noise_openings::{
    assemble_opening_tree, parse_noise_openings, validate_noise_openings, NoiseOpeningRow,
};

use cobre_core::scenario::{CorrelationModel, InflowModel, LoadModel};

use crate::validation::structural::FileManifest;
use crate::LoadError;
use std::path::Path;

/// Load `scenarios/inflow_seasonal_stats.parquet`, returning an empty `Vec` when absent.
///
/// # Errors
///
/// Propagates [`LoadError`] from the parser when `path` is `Some`.
pub fn load_inflow_seasonal_stats(
    path: Option<&Path>,
) -> Result<Vec<InflowSeasonalStatsRow>, LoadError> {
    match path {
        None => Ok(Vec::new()),
        Some(p) => parse_inflow_seasonal_stats(p),
    }
}

/// Load `scenarios/inflow_ar_coefficients.parquet`, returning an empty `Vec` when absent.
///
/// # Errors
///
/// Propagates [`LoadError`] from the parser when `path` is `Some`.
pub fn load_inflow_ar_coefficients(
    path: Option<&Path>,
) -> Result<Vec<InflowArCoefficientRow>, LoadError> {
    match path {
        None => Ok(Vec::new()),
        Some(p) => parse_inflow_ar_coefficients(p),
    }
}

/// Load `scenarios/inflow_history.parquet`, returning an empty `Vec` when absent.
///
/// # Errors
///
/// Propagates [`LoadError`] from the parser when `path` is `Some`.
pub fn load_inflow_history(path: Option<&Path>) -> Result<Vec<InflowHistoryRow>, LoadError> {
    match path {
        None => Ok(Vec::new()),
        Some(p) => parse_inflow_history(p),
    }
}

/// Load `scenarios/load_seasonal_stats.parquet`, returning an empty `Vec` when absent.
///
/// # Errors
///
/// Propagates [`LoadError`] from the parser when `path` is `Some`.
pub fn load_load_seasonal_stats(
    path: Option<&Path>,
) -> Result<Vec<LoadSeasonalStatsRow>, LoadError> {
    match path {
        None => Ok(Vec::new()),
        Some(p) => parse_load_seasonal_stats(p),
    }
}

/// Load `scenarios/load_factors.json`, returning an empty `Vec` when absent.
///
/// # Errors
///
/// Propagates [`LoadError`] from the parser when `path` is `Some`.
pub fn load_load_factors(path: Option<&Path>) -> Result<Vec<LoadFactorEntry>, LoadError> {
    match path {
        None => Ok(Vec::new()),
        Some(p) => parse_load_factors(p),
    }
}

/// Load `scenarios/correlation.json`, returning a default model when absent.
///
/// Unlike the Parquet loaders, this returns `Ok(CorrelationModel::default())` for
/// `None` rather than an empty collection, since the target type is a structured model.
///
/// # Errors
///
/// Propagates [`LoadError`] from the parser when `path` is `Some`.
pub fn load_correlation(path: Option<&Path>) -> Result<CorrelationModel, LoadError> {
    match path {
        None => Ok(CorrelationModel::default()),
        Some(p) => parse_correlation(p),
    }
}

/// Load `scenarios/external_scenarios.parquet`, returning an empty `Vec` when absent.
///
/// # Errors
///
/// Propagates [`LoadError`] from the parser when `path` is `Some`.
pub fn load_external_scenarios(path: Option<&Path>) -> Result<Vec<ExternalScenarioRow>, LoadError> {
    match path {
        None => Ok(Vec::new()),
        Some(p) => parse_external_scenarios(p),
    }
}

/// Load `scenarios/noise_openings.parquet`, returning an empty `Vec` when absent.
///
/// # Errors
///
/// Propagates [`LoadError`] from the parser when `path` is `Some`.
pub fn load_noise_openings(path: Option<&Path>) -> Result<Vec<NoiseOpeningRow>, LoadError> {
    match path {
        None => Ok(Vec::new()),
        Some(p) => parse_noise_openings(p),
    }
}

// ── ScenarioData ──────────────────────────────────────────────────────────────

/// All assembled scenario pipeline data for one case directory.
///
/// Produced by [`load_scenarios`] after loading and assembling all eight
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
///     noise_openings: vec![],
/// };
/// assert!(data.inflow_models.is_empty());
/// assert!(data.correlation.profiles.is_empty());
/// assert!(data.noise_openings.is_empty());
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
    /// Noise opening rows, sorted by `(stage_id, opening_index, entity_index)`. Empty when absent.
    pub noise_openings: Vec<NoiseOpeningRow>,
}

// ── load_scenarios ────────────────────────────────────────────────────────────

/// Orchestrate loading of all eight scenario files and assemble the results.
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
/// | AR coefficient rows exist for unknown (hydro, stage)   | [`LoadError::SchemaError`] |
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
    let scenarios_dir = case_root.join("scenarios");

    let raw_stats = load_inflow_seasonal_stats(
        manifest
            .scenarios_inflow_seasonal_stats_parquet
            .then(|| scenarios_dir.join("inflow_seasonal_stats.parquet"))
            .as_deref(),
    )?;
    let raw_coefficients = load_inflow_ar_coefficients(
        manifest
            .scenarios_inflow_ar_coefficients_parquet
            .then(|| scenarios_dir.join("inflow_ar_coefficients.parquet"))
            .as_deref(),
    )?;
    let inflow_history = load_inflow_history(
        manifest
            .scenarios_inflow_history_parquet
            .then(|| scenarios_dir.join("inflow_history.parquet"))
            .as_deref(),
    )?;
    let raw_load_stats = load_load_seasonal_stats(
        manifest
            .scenarios_load_seasonal_stats_parquet
            .then(|| scenarios_dir.join("load_seasonal_stats.parquet"))
            .as_deref(),
    )?;
    let load_factors = load_load_factors(
        manifest
            .scenarios_load_factors_json
            .then(|| scenarios_dir.join("load_factors.json"))
            .as_deref(),
    )?;
    let correlation = load_correlation(
        manifest
            .scenarios_correlation_json
            .then(|| scenarios_dir.join("correlation.json"))
            .as_deref(),
    )?;
    let external_scenarios = load_external_scenarios(
        manifest
            .scenarios_external_scenarios_parquet
            .then(|| scenarios_dir.join("external_scenarios.parquet"))
            .as_deref(),
    )?;
    let noise_openings = load_noise_openings(
        manifest
            .scenarios_noise_openings_parquet
            .then(|| scenarios_dir.join("noise_openings.parquet"))
            .as_deref(),
    )?;

    let inflow_models = assemble_inflow_models(raw_stats, raw_coefficients)?;
    let load_models = assemble_load_models(raw_load_stats);

    Ok(ScenarioData {
        inflow_models,
        load_models,
        correlation,
        inflow_history,
        external_scenarios,
        load_factors,
        noise_openings,
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

    /// `load_noise_openings(None)` returns `Ok(Vec::new())` without I/O.
    #[test]
    fn test_load_noise_openings_none_returns_empty() {
        let result = load_noise_openings(None).unwrap();
        assert!(result.is_empty());
    }

    /// `load_scenarios` with `scenarios_noise_openings_parquet` flag `false` produces
    /// an empty `noise_openings` collection in the resulting [`ScenarioData`].
    #[test]
    fn test_load_scenarios_noise_openings_absent() {
        let dir = TempDir::new().unwrap();
        let manifest = FileManifest::default(); // all flags false, including noise_openings

        let data =
            load_scenarios(dir.path(), &manifest).expect("empty manifest should always succeed");

        assert!(
            data.noise_openings.is_empty(),
            "noise_openings should be empty when scenarios_noise_openings_parquet flag is false"
        );
    }
}
