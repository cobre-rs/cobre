//! Configuration types for `config.json`.
//!
//! [`Config`] is the top-level deserialized representation of `config.json`.
//! Use [`parse_config`] to load and validate the file.
//!
//! All optional sections use `#[serde(default)]` so that a minimal `config.json`
//! containing only the mandatory `training` fields deserializes cleanly.
//!
//! # Mandatory fields
//!
//! The following fields have no defaults and must be present in `config.json`:
//!
//! - `training.selection` — the scenario-selection method (carries the
//!   forward-pass count in its `sampled` arm)
//! - `training.stopping_rules` — at least one rule entry (must include `iteration_limit`)
//!
//! # Examples
//!
//! ```no_run
//! use cobre_io::config::parse_config;
//! use std::path::Path;
//!
//! let cfg = parse_config(Path::new("case/config.json")).unwrap();
//! println!("forward passes = {:?}", cfg.resolve_forward_passes());
//! ```

use serde_json::{Map, Value};
pub mod estimation;
pub mod exports;
pub mod modeling;
pub mod policy;
pub mod scenario_source;
pub mod simulation;
pub mod training;

pub use estimation::{EstimationConfig, OrderSelectionMethod};
pub use exports::ExportsConfig;
pub use modeling::{InflowNonNegativityConfig, InflowNonNegativityMethod, ModelingConfig};
pub use policy::{BoundaryPolicy, CheckpointingConfig, PolicyConfig, PolicyMode};
pub use scenario_source::{
    HistoricalYearRange, Openings, RawClassConfigEntry, RawHistoricalYearsConfig,
    RawSamplingScheme, RawScenarioSourceConfig,
};
pub use simulation::{NumScenariosResolution, SimulationConfig, SimulationSelection};
pub use training::{
    BackwardScheduler, DualEdgeWeight, ForwardPassesResolution, LipschitzConfig, ParallelismConfig,
    PhaseSolverProfileConfig, PresolveMode, PriceStrategy, RowSelectionConfig, ScaleStrategy,
    SelectionMethod, StoppingMode, StoppingRuleConfig, TrainingConfig, TrainingSelection,
    TrainingSolverConfig, UpperBoundEvaluationConfig,
};

use simulation::DEFAULT_NUM_SCENARIOS;

use cobre_core::scenario::{HistoricalYears, SamplingScheme, ScenarioSource};

use crate::LoadError;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Top-level deserialized representation of `config.json`.
///
/// All sections except `training` are optional; their defaults are applied by
/// serde when the section is absent from the JSON.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct Config {
    /// JSON schema URI — informational, not validated.
    #[serde(rename = "$schema")]
    pub schema: Option<String>,

    /// Modeling options (inflow non-negativity treatment).
    #[serde(default)]
    pub modeling: ModelingConfig,

    /// Training parameters — contains mandatory fields.
    pub training: TrainingConfig,

    /// Upper-bound evaluation via inner approximation.
    #[serde(default)]
    pub upper_bound_evaluation: UpperBoundEvaluationConfig,

    /// Policy directory settings (warm-start / resume).
    #[serde(default)]
    pub policy: PolicyConfig,

    /// Post-training simulation settings.
    #[serde(default)]
    pub simulation: SimulationConfig,

    /// Export flags controlling which outputs are written to disk.
    #[serde(default)]
    pub exports: ExportsConfig,

    /// Time series estimation settings for automatic model parameter fitting.
    #[serde(default)]
    pub estimation: EstimationConfig,
}

/// Load and validate `config.json` from `path`.
///
/// Reads the JSON file, deserializes it into a [`Config`] struct (applying
/// `#[serde(default)]` for optional sections), then performs post-deserialization
/// validation of mandatory fields.
///
/// # Errors
///
/// | Condition                         | Error variant                 |
/// | --------------------------------- | ----------------------------- |
/// | File not found / read failure     | [`LoadError::IoError`]        |
/// | Invalid JSON syntax               | [`LoadError::ParseError`]     |
/// | `training.selection` missing      | [`LoadError::SchemaError`]    |
/// | `training.stopping_rules` missing | [`LoadError::SchemaError`]    |
/// | Unknown stopping rule `"type"`    | [`LoadError::SchemaError`]    |
///
/// # Examples
///
/// ```no_run
/// use cobre_io::config::parse_config;
/// use std::path::Path;
///
/// let cfg = parse_config(Path::new("case/config.json")).unwrap();
/// assert!(cfg.resolve_forward_passes().is_some());
/// ```
pub fn parse_config(path: &Path) -> Result<Config, LoadError> {
    let raw = std::fs::read_to_string(path).map_err(|e| LoadError::io(path, e))?;

    let config: Config = serde_json::from_str(&raw).map_err(|e| {
        let msg = e.to_string();
        if msg.contains("unknown variant") || msg.contains("missing field") {
            LoadError::SchemaError {
                path: path.to_path_buf(),
                field: extract_field_from_serde_msg(&msg),
                message: msg,
            }
        } else {
            LoadError::parse(path, msg)
        }
    })?;

    validate_config(&config, path)?;

    Ok(config)
}

/// Extract a field-name hint (the first backtick-quoted identifier) from a
/// `serde_json` error message, or `"<unknown>"` when none is present.
fn extract_field_from_serde_msg(msg: &str) -> String {
    if let Some(start) = msg.find('`')
        && let Some(end) = msg[start + 1..].find('`')
    {
        return msg[start + 1..start + 1 + end].to_string();
    }
    "<unknown>".to_string()
}

/// Post-deserialization validation that the mandatory `forward_passes` and
/// `stopping_rules` fields are present.
pub(crate) fn validate_config(config: &Config, path: &Path) -> Result<(), LoadError> {
    if config.resolve_forward_passes().is_none() {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: "training.selection".to_string(),
            message: "a forward-pass count is required via training.selection".to_string(),
        });
    }

    if config.training.stopping_rules.is_none() {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: "training.stopping_rules".to_string(),
            message: "required field is missing".to_string(),
        });
    }

    config.training_scenario_source(path)?;
    config.simulation_scenario_source(path)?;

    Ok(())
}

// ── ScenarioSource helpers ───────────────────────────────────────────────────

/// Map a per-class config entry to its [`SamplingScheme`], defaulting to
/// `InSample` when the entry is absent. Infallible: an unknown scheme is
/// rejected at parse by [`RawSamplingScheme`]'s `Deserialize`.
fn convert_class_scheme_cfg(class: Option<&RawClassConfigEntry>) -> SamplingScheme {
    match class.map(|c| c.scheme) {
        None | Some(RawSamplingScheme::InSample) => SamplingScheme::InSample,
        Some(RawSamplingScheme::OutOfSample) => SamplingScheme::OutOfSample,
        Some(RawSamplingScheme::External) => SamplingScheme::External,
        Some(RawSamplingScheme::Historical) => SamplingScheme::Historical,
    }
}

/// Convert `Option<RawScenarioSourceConfig>` into a [`ScenarioSource`].
///
/// `section` is either `"training"` or `"simulation"`, used to build field
/// paths in error messages that reference `config.json`.
///
/// Returns `ScenarioSource::default()` (all `InSample`, no seed, no years)
/// when `raw` is `None`.
fn convert_scenario_source_config(
    raw: Option<&RawScenarioSourceConfig>,
    section: &str,
    path: &Path,
) -> Result<ScenarioSource, LoadError> {
    let Some(r) = raw else {
        return Ok(ScenarioSource::default());
    };

    let inflow_scheme = convert_class_scheme_cfg(r.inflow.as_ref());
    let load_scheme = convert_class_scheme_cfg(r.load.as_ref());
    let ncs_scheme = convert_class_scheme_cfg(r.ncs.as_ref());

    let source = ScenarioSource {
        inflow_scheme,
        load_scheme,
        ncs_scheme,
        seed: r.seed,
        historical_years: r.historical_years.as_ref().map(|hy| match hy {
            RawHistoricalYearsConfig::List(years) => HistoricalYears::List(years.clone()),
            RawHistoricalYearsConfig::Range(range) => HistoricalYears::Range {
                from: range.from,
                to: range.to,
            },
        }),
    };

    validate_scenario_source_cfg(&source, section, path)?;
    validate_openings_cfg(r.openings.as_ref(), section, path)?;

    Ok(source)
}

/// Validate a declared `openings` source from `config.json`: `generated` and
/// `file` are both admitted under `training`; any declaration outside the
/// `training` section is rejected.
fn validate_openings_cfg(
    openings: Option<&Openings>,
    section: &str,
    path: &Path,
) -> Result<(), LoadError> {
    if openings.is_none() {
        return Ok(());
    }

    if section != "training" {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("{section}.scenario_source.openings"),
            message: format!(
                "openings is only valid under training.scenario_source, not \
                 {section}.scenario_source"
            ),
        });
    }

    Ok(())
}

/// Tier-1 structural validation of a parsed [`ScenarioSource`] from `config.json`.
fn validate_scenario_source_cfg(
    source: &ScenarioSource,
    section: &str,
    path: &Path,
) -> Result<(), LoadError> {
    let uses_historical = source.inflow_scheme == SamplingScheme::Historical
        || source.load_scheme == SamplingScheme::Historical
        || source.ncs_scheme == SamplingScheme::Historical;

    if source.historical_years.is_some() && !uses_historical {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("{section}.scenario_source.historical_years"),
            message: "historical_years is specified but no class uses the 'historical' scheme"
                .to_string(),
        });
    }

    if source.load_scheme == SamplingScheme::Historical {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("{section}.scenario_source.load.scheme"),
            message: "historical scheme is only valid for the inflow class".to_string(),
        });
    }

    if source.ncs_scheme == SamplingScheme::Historical {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("{section}.scenario_source.ncs.scheme"),
            message: "historical scheme is only valid for the inflow class".to_string(),
        });
    }

    let all_in_sample = source.inflow_scheme == SamplingScheme::InSample
        && source.load_scheme == SamplingScheme::InSample
        && source.ncs_scheme == SamplingScheme::InSample;
    if !all_in_sample && source.seed.is_none() {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("{section}.scenario_source.seed"),
            message:
                "seed is required when any class uses out_of_sample, historical, or external scheme"
                    .to_string(),
        });
    }

    if let Some(HistoricalYears::Range { from, to }) = source.historical_years
        && from > to
    {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("{section}.scenario_source.historical_years"),
            message: format!("range 'from' ({from}) must be <= 'to' ({to})"),
        });
    }

    Ok(())
}

impl Config {
    /// Resolve the training-phase [`ScenarioSource`].
    ///
    /// When `training.scenario_source` is absent, returns `ScenarioSource::default()`
    /// (all classes `InSample`, no seed, no historical years).
    ///
    /// # Errors
    ///
    /// Returns `LoadError::SchemaError` if the raw config contains an invalid
    /// scheme string, Historical on a non-inflow class, or seed/year validation
    /// failures.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cobre_io::config::parse_config;
    /// use std::path::Path;
    ///
    /// let cfg = parse_config(Path::new("case/config.json")).unwrap();
    /// let source = cfg.training_scenario_source(Path::new("case/config.json")).unwrap();
    /// ```
    pub fn training_scenario_source(&self, path: &Path) -> Result<ScenarioSource, LoadError> {
        convert_scenario_source_config(self.training.scenario_source.as_ref(), "training", path)
    }

    /// Resolve the simulation-phase [`ScenarioSource`].
    ///
    /// Falls back to `training_scenario_source()` when
    /// `simulation.scenario_source` is absent.
    ///
    /// # Errors
    ///
    /// Returns `LoadError::SchemaError` on validation failures in either the
    /// simulation or training scenario source.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cobre_io::config::parse_config;
    /// use std::path::Path;
    ///
    /// let cfg = parse_config(Path::new("case/config.json")).unwrap();
    /// let source = cfg.simulation_scenario_source(Path::new("case/config.json")).unwrap();
    /// ```
    pub fn simulation_scenario_source(&self, path: &Path) -> Result<ScenarioSource, LoadError> {
        if self.simulation.scenario_source.is_some() {
            convert_scenario_source_config(
                self.simulation.scenario_source.as_ref(),
                "simulation",
                path,
            )
        } else {
            self.training_scenario_source(path)
        }
    }

    /// The training-phase `openings` source declaration, or `None` when absent
    /// (equivalent to `generated`). Single owner of the training `openings`
    /// lookup that downstream opening-tree selection reads.
    #[must_use]
    pub fn training_openings(&self) -> Option<&Openings> {
        self.training
            .scenario_source
            .as_ref()
            .and_then(|s| s.openings.as_ref())
    }

    /// Resolve the effective training forward-pass method from
    /// `training.selection`.
    ///
    /// Returns `None` when `selection` is absent, leaving the mandatory-count
    /// rejection to [`validate_config`]. An `enumerated` selection resolves to
    /// [`ForwardPassesResolution::Enumerated`] — config load holds no policy
    /// graph, so the count itself is derived downstream.
    #[must_use]
    pub fn resolve_forward_passes(&self) -> Option<ForwardPassesResolution> {
        match &self.training.selection {
            Some(TrainingSelection::Enumerated {}) => Some(ForwardPassesResolution::Enumerated),
            Some(TrainingSelection::Sampled { forward_passes }) => {
                Some(ForwardPassesResolution::Sampled(*forward_passes))
            }
            None => None,
        }
    }

    /// Resolve the effective simulation scenario-count method from
    /// `simulation.selection`, defaulting to [`DEFAULT_NUM_SCENARIOS`] when
    /// absent.
    ///
    /// An `enumerated` selection resolves to
    /// [`NumScenariosResolution::Enumerated`] — config load holds no policy
    /// graph, so the derived count happens downstream.
    #[must_use]
    pub fn resolve_num_scenarios(&self) -> NumScenariosResolution {
        match &self.simulation.selection {
            Some(SimulationSelection::Enumerated {}) => NumScenariosResolution::Enumerated,
            Some(SimulationSelection::Sampled { num_scenarios }) => {
                NumScenariosResolution::Sampled(*num_scenarios)
            }
            None => NumScenariosResolution::Sampled(DEFAULT_NUM_SCENARIOS),
        }
    }

    /// Deep-merge a flat map of dotted-key overrides into `base` and re-deserialize
    /// the result into a validated [`Config`].
    ///
    /// `base` is the parsed-but-not-typed `config.json` (a [`serde_json::Value::Object`]).
    /// `overrides` is a flat map whose keys are dotted paths into the config schema
    /// (e.g. `"training.tree_seed"`, `"policy.checkpointing.compress"`). Intermediate
    /// objects are reused rather than replaced, so setting `policy.checkpointing.compress`
    /// does not clobber sibling keys under `policy` or `policy.checkpointing`.
    ///
    /// After merging, the value is re-deserialized into [`Config`]. Because `Config`
    /// is `#[serde(deny_unknown_fields)]`, an override key that does not exist in the
    /// schema (a typo such as `trainning.tree_seed`) fails loudly. The same
    /// post-deserialization checks as [`parse_config`] then run via `validate_config`.
    ///
    /// All errors carry the synthetic path `"<config_overrides>"` so callers can
    /// recognize override-originated failures.
    ///
    /// # Errors
    ///
    /// - [`LoadError::SchemaError`] if `base` is not a JSON object.
    /// - [`LoadError::SchemaError`] if any override key contains an empty path segment
    ///   (e.g. `"training..seed"` or a leading/trailing dot).
    /// - [`LoadError::SchemaError`] if the merged value fails to deserialize into
    ///   [`Config`] (e.g. an unknown field) or fails `validate_config`.
    pub fn with_overrides(
        base: &Value,
        overrides: &Map<String, Value>,
    ) -> Result<Config, LoadError> {
        if !base.is_object() {
            return Err(LoadError::SchemaError {
                path: PathBuf::from("<config_overrides>"),
                field: "<root>".to_string(),
                message: "base config must be a JSON object".to_string(),
            });
        }

        let mut merged = base.clone();
        for (dotted_key, value) in overrides {
            Self::set_dotted(&mut merged, dotted_key, value.clone())?;
        }

        let config: Config = serde_json::from_value(merged).map_err(|e| {
            let msg = e.to_string();
            LoadError::SchemaError {
                path: PathBuf::from("<config_overrides>"),
                field: extract_field_from_serde_msg(&msg),
                message: msg,
            }
        })?;

        validate_config(&config, Path::new("<config_overrides>")).map(|()| config)
    }

    /// Deep-merge `value` into `target` at the dotted path `dotted_key`, reusing
    /// existing intermediate objects so sibling keys survive.
    ///
    /// # Errors
    ///
    /// Returns [`LoadError::SchemaError`] (with `field` set to the offending
    /// `dotted_key`) when any path segment is empty — i.e. an empty key, a leading or
    /// trailing dot, or a doubled dot such as `"training..seed"`.
    fn set_dotted(target: &mut Value, dotted_key: &str, value: Value) -> Result<(), LoadError> {
        let segments: Vec<&str> = dotted_key.split('.').collect();
        if segments.iter().any(|s| s.is_empty()) {
            return Err(LoadError::SchemaError {
                path: PathBuf::from("<config_overrides>"),
                field: dotted_key.to_string(),
                message: format!("override key has an empty path segment: `{dotted_key}`"),
            });
        }

        let mut current = target;
        for segment in &segments[..segments.len() - 1] {
            // Reuse the existing intermediate object rather than replacing it; a
            // replace clobbers sibling keys, breaking the deep-merge contract.
            if !current.is_object() {
                *current = serde_json::Value::Object(serde_json::Map::new());
            }
            let serde_json::Value::Object(map) = current else {
                unreachable!("current was just coerced to an object")
            };
            current = map
                .entry((*segment).to_string())
                .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()));
        }

        // The empty-segment guard above rejects the only `dotted_key` that could
        // make `segments` empty, so this last index never panics.
        let last = segments[segments.len() - 1];
        if !current.is_object() {
            *current = serde_json::Value::Object(serde_json::Map::new());
        }
        let serde_json::Value::Object(map) = current else {
            unreachable!("current was just coerced to an object")
        };
        map.insert(last.to_string(), value);

        Ok(())
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::too_many_lines,
    clippy::doc_markdown
)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn write_config(content: &str) -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f
    }

    /// Minimal config returns Ok with correct forward_passes and all
    /// optional sections at their default values.
    #[test]
    fn test_parse_minimal_config() {
        let f = write_config(
            r#"{"training": {"tree_seed": 42, "selection": {"method": "sampled", "forward_passes": 192}, "stopping_rules": [{"type": "iteration_limit", "limit": 50}]}}"#,
        );
        let cfg = parse_config(f.path()).unwrap();

        assert_eq!(
            cfg.resolve_forward_passes(),
            Some(ForwardPassesResolution::Sampled(192))
        );
        assert_eq!(cfg.training.tree_seed, Some(42));
        assert_eq!(cfg.training.stopping_mode, StoppingMode::Any);
        assert!(cfg.training.enabled);
        assert_eq!(
            cfg.modeling.inflow_non_negativity.method,
            InflowNonNegativityMethod::Penalty
        );
        assert!(!cfg.simulation.enabled);
        assert_eq!(
            cfg.resolve_num_scenarios(),
            NumScenariosResolution::Sampled(2000),
            "absent simulation selection resolves to the default sampled count"
        );
        assert_eq!(cfg.policy.mode, PolicyMode::Fresh);
        assert_eq!(cfg.policy.path, "./policy");
    }

    /// The retired `state_space` section (inflow-lag depth is now always inferred
    /// from the boundary policy) is a `deny_unknown_fields` reject, not a silent
    /// ignore — a stale `config.json` fails loudly with the field name.
    #[test]
    fn test_retired_state_space_section_is_rejected() {
        let f = write_config(
            r#"{"training": {"selection": {"method": "sampled", "forward_passes": 1}, "stopping_rules": [{"type": "iteration_limit", "limit": 10}]}, "state_space": {"inflow_lag_depth": 12}}"#,
        );
        let err = parse_config(f.path()).unwrap_err();
        assert!(
            err.to_string().contains("state_space"),
            "expected an unknown-field error naming state_space, got: {err}"
        );
    }

    /// Missing `training.selection` (no forward-pass count) → SchemaError
    /// with field name.
    #[test]
    fn test_missing_forward_passes() {
        let f = write_config(
            r#"{"training": {"tree_seed": 1, "stopping_rules": [{"type": "iteration_limit", "limit": 10}]}}"#,
        );
        let err = parse_config(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { field, .. } => {
                assert!(
                    field.contains("selection"),
                    "field should name training.selection, got: {field}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// Missing `training.stopping_rules` → SchemaError.
    #[test]
    fn test_missing_stopping_rules() {
        let f = write_config(
            r#"{"training": {"tree_seed": 1, "selection": {"method": "sampled", "forward_passes": 100}}}"#,
        );
        let err = parse_config(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { field, .. } => {
                assert!(
                    field.contains("stopping_rules"),
                    "field should contain 'stopping_rules', got: {field}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// Nonexistent file → IoError with matching path.
    #[test]
    fn test_nonexistent_file() {
        let path = std::path::Path::new("/nonexistent/path/config.json");
        let err = parse_config(path).unwrap_err();
        match &err {
            LoadError::IoError { path: p, .. } => {
                assert_eq!(p, path);
            }
            other => panic!("expected IoError, got: {other:?}"),
        }
    }

    /// Full config with all sections → Ok with non-default values.
    #[test]
    fn test_parse_full_config() {
        let json = r#"{
          "$schema": "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main/schemas/config.schema.json",
          "modeling": {
            "inflow_non_negativity": {
              "method": "penalty"
            }
          },
          "training": {
            "tree_seed": 42,
            "selection": {"method": "sampled", "forward_passes": 192},
            "stopping_rules": [
              {"type": "iteration_limit", "limit": 50},
              {"type": "bound_stalling", "iterations": 10, "tolerance": 0.0001}
            ],
            "stopping_mode": "any",
            "cut_selection": {
              "selection": {
                "method": "domination",
                "domination_tolerance": 1e-6
              }
            }
          },
          "upper_bound_evaluation": {
            "enabled": true,
            "initial_iteration": 10,
            "interval_iterations": 5
          },
          "policy": {
            "path": "./policy",
            "mode": "fresh",
            "checkpointing": {
              "enabled": true,
              "initial_iteration": 10,
              "interval_iterations": 10,
              "store_basis": true,
              "compress": true
            }
          },
          "simulation": {
            "enabled": true,
            "selection": {"method": "sampled", "num_scenarios": 2000}
          },
          "exports": {
            "states": true,
            "stochastic": true
          }
        }"#;

        let f = write_config(json);
        let cfg = parse_config(f.path()).unwrap();

        assert_eq!(
            cfg.modeling.inflow_non_negativity.method,
            InflowNonNegativityMethod::Penalty
        );

        assert_eq!(
            cfg.resolve_forward_passes(),
            Some(ForwardPassesResolution::Sampled(192))
        );
        assert_eq!(cfg.training.stopping_mode, StoppingMode::Any);
        let rules = cfg.training.stopping_rules.as_ref().unwrap();
        assert_eq!(rules.len(), 2);
        let cut_sel = &cfg.training.cut_selection;
        match cut_sel.selection.as_ref().expect("selection present") {
            SelectionMethod::Domination {
                domination_tolerance,
                check_frequency,
            } => {
                assert!((domination_tolerance - 1e-6).abs() < f64::EPSILON);
                assert_eq!(*check_frequency, 5);
            }
            other => panic!("expected Domination, got {other:?}"),
        }

        assert_eq!(cfg.upper_bound_evaluation.enabled, Some(true));
        assert_eq!(cfg.upper_bound_evaluation.initial_iteration, Some(10));

        assert_eq!(cfg.policy.mode, PolicyMode::Fresh);
        assert_eq!(cfg.policy.checkpointing.enabled, Some(true));

        assert!(cfg.simulation.enabled);
        assert_eq!(
            cfg.resolve_num_scenarios(),
            NumScenariosResolution::Sampled(2000)
        );

        assert!(cfg.exports.states);
        assert!(cfg.exports.stochastic);
    }

    /// Invalid JSON syntax → ParseError.
    #[test]
    fn test_invalid_json_syntax() {
        let f = write_config(r#"{"training": {not valid json}}"#);
        let err = parse_config(f.path()).unwrap_err();
        assert!(
            matches!(err, LoadError::ParseError { .. }),
            "expected ParseError, got: {err:?}"
        );
    }

    /// All 4 JSON-configurable stopping rule variants deserialize correctly.
    ///
    /// The `GracefulShutdown` variant is runtime-only and has no JSON representation
    /// per the stopping-rule-trait spec (SS4.1).
    #[test]
    fn test_stopping_rule_variants() {
        let json = r#"{
          "training": {
            "selection": {"method": "sampled", "forward_passes": 10},
            "stopping_rules": [
              {"type": "iteration_limit", "limit": 100},
              {"type": "time_limit", "seconds": 3600.0},
              {"type": "bound_stalling", "iterations": 10, "tolerance": 0.0001},
              {
                "type": "gap",
                "tolerance": 1000.0
              }
            ]
          }
        }"#;

        let f = write_config(json);
        let cfg = parse_config(f.path()).unwrap();
        let rules = cfg.training.stopping_rules.unwrap();
        assert_eq!(rules.len(), 4);

        assert!(matches!(
            rules[0],
            StoppingRuleConfig::IterationLimit { limit: 100 }
        ));
        assert!(
            matches!(rules[1], StoppingRuleConfig::TimeLimit { seconds } if (seconds - 3600.0).abs() < f64::EPSILON)
        );
        assert!(matches!(
            rules[2],
            StoppingRuleConfig::BoundStalling { iterations: 10, .. }
        ));
        assert!(matches!(
            rules[3],
            StoppingRuleConfig::Gap {
                tolerance: Some(t),
                ..
            } if (t - 1000.0).abs() < f64::EPSILON
        ));
    }

    /// Unknown stopping rule type → SchemaError (not a panic or ParseError).
    #[test]
    fn test_unknown_stopping_rule_type() {
        let f = write_config(
            r#"{"training": {"selection": {"method": "sampled", "forward_passes": 10}, "stopping_rules": [{"type": "nonexistent_rule"}]}}"#,
        );
        let err = parse_config(f.path()).unwrap_err();
        assert!(
            matches!(err, LoadError::SchemaError { .. }),
            "expected SchemaError for unknown rule type, got: {err:?}"
        );
    }

    /// The retired `simulation` stopping-rule type (replaced by `gap`) is
    /// rejected as an unknown variant, not a recognized-but-invalid one.
    #[test]
    fn old_simulation_stopping_rule_type_is_unknown_variant() {
        let f = write_config(
            r#"{"training": {"selection": {"method": "sampled", "forward_passes": 10}, "stopping_rules": [
                {"type": "simulation", "replications": 100, "period": 20,
                 "bound_window": 5, "distance_tol": 0.01, "bound_tol": 0.0001}
            ]}}"#,
        );
        let err = parse_config(f.path()).unwrap_err();
        assert!(
            matches!(err, LoadError::SchemaError { .. }),
            "expected SchemaError for the retired simulation rule type, got: {err:?}"
        );
    }

    /// `Config` has no `version` field — the struct does not
    /// expose `.version` and the field is not present after deserialization.
    #[test]
    fn test_config_has_no_version_field() {
        let f = write_config(
            r#"{"training": {"selection": {"method": "sampled", "forward_passes": 1}, "stopping_rules": [{"type": "iteration_limit", "limit": 10}]}}"#,
        );
        let cfg = parse_config(f.path()).unwrap();
        assert!(cfg.schema.is_none(), "schema should be None when absent");
    }

    /// JSON with `"$schema"` property is accepted and the field
    /// value is stored correctly.
    #[test]
    fn test_schema_field_accepted() {
        let f = write_config(
            r#"{
            "$schema": "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main/schemas/config.schema.json",
            "training": {
                "selection": {"method": "sampled", "forward_passes": 1},
                "stopping_rules": [{"type": "iteration_limit", "limit": 10}]
            }
        }"#,
        );
        let cfg = parse_config(f.path()).unwrap();
        assert_eq!(
            cfg.schema.as_deref(),
            Some(
                "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main/schemas/config.schema.json"
            ),
            "schema field should be stored when present in JSON"
        );
    }

    /// Invalid `policy.mode` values are rejected at parse time.
    #[test]
    fn test_invalid_policy_mode_rejected() {
        let f = write_config(
            r#"{"training": {"selection": {"method": "sampled", "forward_passes": 1}, "stopping_rules": [{"type": "iteration_limit", "limit": 10}]}, "policy": {"mode": "warmstart"}}"#,
        );
        let err = parse_config(f.path()).unwrap_err();
        assert!(
            matches!(err, LoadError::SchemaError { .. }),
            "expected SchemaError for invalid policy.mode, got: {err:?}"
        );
    }

    /// JSON that contains the dead `"version"` property must now be rejected
    /// because `Config` uses `deny_unknown_fields`. Old case dirs that still
    /// contain this key will fail to parse — which is the desired behaviour.
    #[test]
    fn test_legacy_version_field_rejected() {
        let f = write_config(
            r#"{
            "version": "1.0.0",
            "training": {
                "selection": {"method": "sampled", "forward_passes": 1},
                "stopping_rules": [{"type": "iteration_limit", "limit": 10}]
            }
        }"#,
        );
        let err = parse_config(f.path()).unwrap_err();
        assert!(
            matches!(
                err,
                LoadError::ParseError { .. } | LoadError::SchemaError { .. }
            ),
            "expected parse/schema error for unknown 'version' field, got: {err:?}"
        );
    }

    /// A config that still sets the removed `policy.validate_compatibility` flag
    /// is rejected: `PolicyConfig` uses `deny_unknown_fields`, so the stale key
    /// fails deserialization with an error naming the field.
    #[test]
    fn test_stale_validate_compatibility_field_rejected() {
        let f = write_config(
            r#"{
            "training": {
                "selection": {"method": "sampled", "forward_passes": 1},
                "stopping_rules": [{"type": "iteration_limit", "limit": 10}]
            },
            "policy": {
                "validate_compatibility": false
            }
        }"#,
        );
        let err = parse_config(f.path()).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("validate_compatibility"),
            "expected the stale field name in the rejection error, got: {msg}"
        );
    }

    /// `"truncation"` is accepted as a method value and round-trips correctly
    /// through `parse_config`.
    #[test]
    fn test_truncation_method_accepted() {
        let f = write_config(
            r#"{
            "modeling": {
                "inflow_non_negativity": {
                    "method": "truncation"
                }
            },
            "training": {
                "selection": {"method": "sampled", "forward_passes": 10},
                "stopping_rules": [{"type": "iteration_limit", "limit": 5}]
            }
        }"#,
        );
        let cfg = parse_config(f.path()).unwrap();
        assert_eq!(
            cfg.modeling.inflow_non_negativity.method,
            InflowNonNegativityMethod::Truncation,
            "method field should round-trip as Truncation"
        );
    }

    /// An unknown inflow non-negativity method string is rejected at parse time.
    #[test]
    fn test_unknown_inflow_method_rejected() {
        let f = write_config(
            r#"{
            "modeling": {
                "inflow_non_negativity": {
                    "method": "bogus_method"
                }
            },
            "training": {
                "selection": {"method": "sampled", "forward_passes": 10},
                "stopping_rules": [{"type": "iteration_limit", "limit": 5}]
            }
        }"#,
        );
        let err = parse_config(f.path()).unwrap_err();
        assert!(
            matches!(
                err,
                LoadError::SchemaError { .. } | LoadError::ParseError { .. }
            ),
            "expected parse/schema error for unknown method, got: {err:?}"
        );
    }

    /// `config.json` without `"estimation"` section → all three defaults applied.
    #[test]
    fn test_estimation_config_defaults() {
        let f = write_config(
            r#"{"training": {"selection": {"method": "sampled", "forward_passes": 10}, "stopping_rules": [{"type": "iteration_limit", "limit": 5}]}}"#,
        );
        let cfg = parse_config(f.path()).unwrap();
        assert_eq!(cfg.estimation.max_order, 6);
        assert!(
            matches!(cfg.estimation.order_selection, OrderSelectionMethod::Pacf),
            "default order_selection should be Pacf"
        );
        assert_eq!(cfg.estimation.min_observations_per_season, 30);
    }

    /// `"order_selection": "fixed"` is now a hard parse error.
    #[test]
    fn test_estimation_config_order_selection_fixed_rejected() {
        let f = write_config(
            r#"{
            "training": {"selection": {"method": "sampled", "forward_passes": 10}, "stopping_rules": [{"type": "iteration_limit", "limit": 5}]},
            "estimation": {"max_order": 3, "order_selection": "fixed", "min_observations_per_season": 20}
        }"#,
        );
        let result = parse_config(f.path());
        assert!(
            result.is_err(),
            "\"fixed\" order_selection must now be a parse error"
        );
    }

    /// `"order_selection": "pacf"` deserializes to `Pacf` with no warning.
    #[test]
    fn test_estimation_config_order_selection_pacf() {
        let f = write_config(
            r#"{
            "training": {"selection": {"method": "sampled", "forward_passes": 10}, "stopping_rules": [{"type": "iteration_limit", "limit": 5}]},
            "estimation": {"max_order": 4, "order_selection": "pacf", "min_observations_per_season": 15}
        }"#,
        );
        let cfg = parse_config(f.path()).unwrap();
        assert_eq!(cfg.estimation.max_order, 4);
        assert!(
            matches!(cfg.estimation.order_selection, OrderSelectionMethod::Pacf),
            "explicit 'pacf' must deserialize to Pacf"
        );
        assert_eq!(cfg.estimation.min_observations_per_season, 15);
    }

    /// Unknown `order_selection` value → `LoadError::SchemaError` with
    /// message containing `"unknown variant"`.
    #[test]
    fn test_estimation_config_unknown_order_selection() {
        let f = write_config(
            r#"{
            "training": {"selection": {"method": "sampled", "forward_passes": 10}, "stopping_rules": [{"type": "iteration_limit", "limit": 5}]},
            "estimation": {"order_selection": "bogus"}
        }"#,
        );
        let err = parse_config(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { message, .. } => {
                assert!(
                    message.contains("unknown variant"),
                    "message should contain 'unknown variant', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// `exports.stochastic: true` deserializes correctly.
    #[test]
    fn test_exports_stochastic_explicit_true() {
        let f = write_config(
            r#"{
            "training": {"selection": {"method": "sampled", "forward_passes": 10}, "stopping_rules": [{"type": "iteration_limit", "limit": 5}]},
            "exports": {"stochastic": true}
        }"#,
        );
        let cfg = parse_config(f.path()).unwrap();
        assert!(
            cfg.exports.stochastic,
            "exports.stochastic should be true when set in config"
        );
    }

    /// `exports.stochastic` defaults to `false` when the field is absent.
    #[test]
    fn test_exports_stochastic_defaults_to_false() {
        let f = write_config(
            r#"{
            "training": {"selection": {"method": "sampled", "forward_passes": 10}, "stopping_rules": [{"type": "iteration_limit", "limit": 5}]}
        }"#,
        );
        let cfg = parse_config(f.path()).unwrap();
        assert!(
            !cfg.exports.stochastic,
            "exports.stochastic should default to false when absent"
        );
    }

    /// `exports.fpha_deviation_points: true` deserializes correctly.
    #[test]
    fn test_exports_fpha_deviation_points_explicit_true() {
        let f = write_config(
            r#"{
            "training": {"selection": {"method": "sampled", "forward_passes": 10}, "stopping_rules": [{"type": "iteration_limit", "limit": 5}]},
            "exports": {"fpha_deviation_points": true}
        }"#,
        );
        let cfg = parse_config(f.path()).unwrap();
        assert!(
            cfg.exports.fpha_deviation_points,
            "exports.fpha_deviation_points should be true when set in config"
        );
    }

    /// `exports.fpha_deviation_points` defaults to `false` when absent, so the
    /// flag-off run emits no deviation-points file and is byte-identical.
    #[test]
    fn test_exports_fpha_deviation_points_defaults_to_false() {
        let f = write_config(
            r#"{
            "training": {"selection": {"method": "sampled", "forward_passes": 10}, "stopping_rules": [{"type": "iteration_limit", "limit": 5}]}
        }"#,
        );
        let cfg = parse_config(f.path()).unwrap();
        assert!(
            !cfg.exports.fpha_deviation_points,
            "exports.fpha_deviation_points should default to false when absent"
        );
    }

    // ── ScenarioSource parsing tests ──────────────────────────────────────────

    const MINIMAL_TRAINING: &str = r#"{"selection": {"method": "sampled", "forward_passes": 10}, "stopping_rules": [{"type": "iteration_limit", "limit": 5}]}"#;

    fn write_with_training_scenario_source(scenario_source_json: &str) -> NamedTempFile {
        write_config(&format!(
            r#"{{"training": {{"selection": {{"method": "sampled", "forward_passes": 10}}, "stopping_rules": [{{"type": "iteration_limit", "limit": 5}}], "scenario_source": {scenario_source_json}}}}}"#
        ))
    }

    fn write_with_both_scenario_sources(
        training_json: &str,
        simulation_json: &str,
    ) -> NamedTempFile {
        write_config(&format!(
            r#"{{"training": {{"selection": {{"method": "sampled", "forward_passes": 10}}, "stopping_rules": [{{"type": "iteration_limit", "limit": 5}}], "scenario_source": {training_json}}}, "simulation": {{"scenario_source": {simulation_json}}}}}"#
        ))
    }

    /// Absent `training.scenario_source` → all InSample, no seed, no historical_years.
    #[test]
    fn test_training_scenario_source_default() {
        let f = write_config(&format!(r#"{{"training": {MINIMAL_TRAINING}}}"#));
        let cfg = parse_config(f.path()).unwrap();
        let source = cfg.training_scenario_source(f.path()).unwrap();
        assert_eq!(source, ScenarioSource::default());
        assert_eq!(source.inflow_scheme, SamplingScheme::InSample);
        assert_eq!(source.load_scheme, SamplingScheme::InSample);
        assert_eq!(source.ncs_scheme, SamplingScheme::InSample);
        assert_eq!(source.seed, None);
        assert_eq!(source.historical_years, None);
    }

    /// Explicit per-class schemes are parsed correctly.
    #[test]
    fn test_training_scenario_source_explicit() {
        let f = write_with_training_scenario_source(
            r#"{"seed": 42, "inflow": {"scheme": "historical"}, "historical_years": [1940, 1953]}"#,
        );
        let cfg = parse_config(f.path()).unwrap();
        let source = cfg.training_scenario_source(f.path()).unwrap();
        assert_eq!(source.inflow_scheme, SamplingScheme::Historical);
        assert_eq!(source.load_scheme, SamplingScheme::InSample);
        assert_eq!(source.ncs_scheme, SamplingScheme::InSample);
        assert_eq!(source.seed, Some(42));
        assert_eq!(
            source.historical_years,
            Some(HistoricalYears::List(vec![1940, 1953]))
        );
    }

    /// Absent `simulation.scenario_source` falls back to `training_scenario_source()`.
    #[test]
    fn test_simulation_scenario_source_fallback() {
        let f = write_with_training_scenario_source(
            r#"{"seed": 7, "inflow": {"scheme": "out_of_sample"}}"#,
        );
        let cfg = parse_config(f.path()).unwrap();
        let training = cfg.training_scenario_source(f.path()).unwrap();
        let simulation = cfg.simulation_scenario_source(f.path()).unwrap();
        assert_eq!(training, simulation);
        assert_eq!(simulation.inflow_scheme, SamplingScheme::OutOfSample);
        assert_eq!(simulation.seed, Some(7));
    }

    /// Both sections present with different schemes → different `ScenarioSource` values returned.
    #[test]
    fn test_simulation_scenario_source_independent() {
        let f = write_with_both_scenario_sources(
            r#"{"seed": 1, "inflow": {"scheme": "out_of_sample"}}"#,
            r#"{"seed": 2, "load": {"scheme": "out_of_sample"}}"#,
        );
        let cfg = parse_config(f.path()).unwrap();
        let training = cfg.training_scenario_source(f.path()).unwrap();
        let simulation = cfg.simulation_scenario_source(f.path()).unwrap();
        assert_ne!(training, simulation);
        assert_eq!(training.inflow_scheme, SamplingScheme::OutOfSample);
        assert_eq!(training.load_scheme, SamplingScheme::InSample);
        assert_eq!(simulation.inflow_scheme, SamplingScheme::InSample);
        assert_eq!(simulation.load_scheme, SamplingScheme::OutOfSample);
    }

    /// Historical scheme on inflow class is accepted.
    #[test]
    fn test_scenario_source_historical_inflow_valid() {
        let f = write_with_training_scenario_source(
            r#"{"seed": 99, "inflow": {"scheme": "historical"}}"#,
        );
        let cfg = parse_config(f.path()).unwrap();
        let source = cfg.training_scenario_source(f.path()).unwrap();
        assert_eq!(source.inflow_scheme, SamplingScheme::Historical);
    }

    /// Historical on load class → SchemaError.
    #[test]
    fn test_scenario_source_historical_load_rejected() {
        let f = write_config(&format!(
            r#"{{"training": {MINIMAL_TRAINING}, "simulation": {{"scenario_source": {{"seed": 1, "load": {{"scheme": "historical"}}}}}}}}"#
        ));
        let err = parse_config(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { message, field, .. } => {
                assert!(
                    message.contains("historical scheme is only valid for the inflow class"),
                    "unexpected message: {message}"
                );
                assert!(field.contains("load.scheme"), "unexpected field: {field}");
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// Historical on ncs class → SchemaError.
    #[test]
    fn test_scenario_source_historical_ncs_rejected() {
        let f =
            write_with_training_scenario_source(r#"{"seed": 1, "ncs": {"scheme": "historical"}}"#);
        let err = parse_config(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { message, field, .. } => {
                assert!(
                    message.contains("historical scheme is only valid for the inflow class"),
                    "unexpected message: {message}"
                );
                assert!(field.contains("ncs.scheme"), "unexpected field: {field}");
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// An unknown per-class `scheme` is rejected during parse, and the error
    /// names the accepted set.
    #[test]
    fn unknown_scheme_is_rejected_naming_accepted_set() {
        let f = write_with_training_scenario_source(r#"{"inflow": {"scheme": "antithetic"}}"#);
        let err = parse_config(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { message, .. } => {
                assert!(
                    message.contains("in_sample")
                        && message.contains("out_of_sample")
                        && message.contains("external")
                        && message.contains("historical"),
                    "message should name the accepted set, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// `stopping_mode = "any"` / `"all"` both parse into the enum — the accepted
    /// set is unchanged by the representation promotion.
    #[test]
    fn stopping_mode_any_and_all_parse() {
        for (value, expected) in [("any", StoppingMode::Any), ("all", StoppingMode::All)] {
            let f = write_config(&format!(
                r#"{{"training": {{"selection": {{"method": "sampled", "forward_passes": 10}}, "stopping_rules": [{{"type": "iteration_limit", "limit": 5}}], "stopping_mode": "{value}"}}}}"#
            ));
            let cfg = parse_config(f.path()).unwrap();
            assert_eq!(cfg.training.stopping_mode, expected);
        }
    }

    /// An unknown `stopping_mode` is rejected during parse (the tightening: no
    /// silent fallback to `any`), and the error names the accepted set.
    #[test]
    fn unknown_stopping_mode_is_rejected_naming_accepted_set() {
        let f = write_config(
            r#"{"training": {"selection": {"method": "sampled", "forward_passes": 10}, "stopping_rules": [{"type": "iteration_limit", "limit": 5}], "stopping_mode": "either"}}"#,
        );
        let err = parse_config(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { message, .. } => {
                assert!(
                    message.contains("any") && message.contains("all"),
                    "message should name the accepted set, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// OutOfSample without seed → SchemaError.
    #[test]
    fn test_scenario_source_seed_required_for_oos() {
        let f = write_with_training_scenario_source(r#"{"inflow": {"scheme": "out_of_sample"}}"#);
        let err = parse_config(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { message, field, .. } => {
                assert!(
                    message.contains("seed is required"),
                    "unexpected message: {message}"
                );
                assert!(field.contains("seed"), "unexpected field: {field}");
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// Range form of `historical_years` parses correctly.
    #[test]
    fn test_scenario_source_historical_years_range() {
        let f = write_with_training_scenario_source(
            r#"{"seed": 5, "inflow": {"scheme": "historical"}, "historical_years": {"from": 1940, "to": 2010}}"#,
        );
        let cfg = parse_config(f.path()).unwrap();
        let source = cfg.training_scenario_source(f.path()).unwrap();
        assert_eq!(
            source.historical_years,
            Some(HistoricalYears::Range {
                from: 1940,
                to: 2010
            })
        );
    }

    /// `historical_years` specified without any Historical scheme → SchemaError.
    #[test]
    fn test_scenario_source_historical_years_without_historical_scheme() {
        let f = write_with_training_scenario_source(
            r#"{"seed": 1, "inflow": {"scheme": "out_of_sample"}, "historical_years": [1990, 2000]}"#,
        );
        let err = parse_config(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { message, .. } => {
                assert!(
                    message.contains(
                        "historical_years is specified but no class uses the 'historical' scheme"
                    ),
                    "unexpected message: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// `historical_years` range with `from > to` → SchemaError.
    #[test]
    fn parse_config_rejects_historical_years_inverted_range() {
        let f = write_with_training_scenario_source(
            r#"{"seed": 1, "inflow": {"scheme": "historical"}, "historical_years": {"from": 2010, "to": 1990}}"#,
        );
        let err = parse_config(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { message, field, .. } => {
                assert!(
                    message.contains("must be <= 'to'"),
                    "unexpected message: {message}"
                );
                assert!(
                    field.contains("scenario_source.historical_years"),
                    "unexpected field: {field}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── openings source tests ─────────────────────────────────────────────────

    /// A declared `generated` openings source under training is admitted, and
    /// the resolved `ScenarioSource` is unchanged (openings lives on the raw
    /// config, not `ScenarioSource`).
    #[test]
    fn openings_generated_accepted_under_training() {
        let f = write_with_training_scenario_source(r#"{"openings": {"source": "generated"}}"#);
        let cfg = parse_config(f.path()).unwrap();
        let source = cfg.training_scenario_source(f.path()).unwrap();
        assert_eq!(source, ScenarioSource::default());
    }

    /// The dropped `external` openings source is now an unknown-variant parse
    /// error — the arm no longer exists (`generated` and `file` remain).
    #[test]
    fn openings_external_is_unknown_variant_parse_error() {
        let f = write_with_training_scenario_source(r#"{"openings": {"source": "external"}}"#);
        let err = parse_config(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { message, .. } => {
                assert!(
                    message.contains("unknown variant"),
                    "unexpected message: {message}"
                );
            }
            other => panic!("expected SchemaError (unknown variant), got: {other:?}"),
        }
    }

    /// A `file` openings source under training is admitted, and
    /// `training_openings()` surfaces the declared `File` arm.
    #[test]
    fn openings_file_accepted_under_training() {
        let f = write_with_training_scenario_source(r#"{"openings": {"source": "file"}}"#);
        let cfg = parse_config(f.path()).unwrap();
        cfg.training_scenario_source(f.path())
            .expect("file openings source must load under training");
        assert_eq!(cfg.training_openings(), Some(&Openings::File {}));
    }

    /// The dropped `path` field on the `file` openings source is now an
    /// unknown-field parse error — the arm is convention-located, no user path.
    #[test]
    fn openings_file_path_field_rejected() {
        let f = write_with_training_scenario_source(
            r#"{"openings": {"source": "file", "path": "scenarios/openings.parquet"}}"#,
        );
        let err = parse_config(f.path()).unwrap_err();
        assert!(
            matches!(
                err,
                LoadError::SchemaError { .. } | LoadError::ParseError { .. }
            ),
            "a path field on the file arm must be rejected, got: {err:?}"
        );
    }

    /// `openings` under `simulation.scenario_source` is rejected naming the
    /// field — a declared openings source is valid only for training.
    #[test]
    fn openings_under_simulation_rejected() {
        let f = write_with_both_scenario_sources(
            r#"{"inflow": {"scheme": "in_sample"}}"#,
            r#"{"openings": {"source": "generated"}}"#,
        );
        let err = parse_config(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert_eq!(field, "simulation.scenario_source.openings");
                assert!(
                    message.contains("only valid under training.scenario_source"),
                    "unexpected message: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// A `file` openings source round-trips through serde into the `File`
    /// variant (no path field).
    #[test]
    fn openings_file_variant_round_trips() {
        let parsed: Openings = serde_json::from_str(r#"{"source": "file"}"#).unwrap();
        assert_eq!(parsed, Openings::File {});
    }

    /// `simulation.sampling_scheme` (dead field) is now rejected because
    /// `SimulationConfig` uses `deny_unknown_fields`. Old case dirs must remove
    /// this key before loading.
    #[test]
    fn test_dead_sampling_scheme_field_rejected() {
        let f = write_config(
            r#"{
            "training": {"selection": {"method": "sampled", "forward_passes": 10}, "stopping_rules": [{"type": "iteration_limit", "limit": 5}]},
            "simulation": {"enabled": true, "sampling_scheme": {"type": "in_sample"}}
        }"#,
        );
        let err = parse_config(f.path()).unwrap_err();
        assert!(
            matches!(
                err,
                LoadError::ParseError { .. } | LoadError::SchemaError { .. }
            ),
            "expected parse/schema error for unknown 'sampling_scheme' field, got: {err:?}"
        );
    }

    /// `RowSelectionConfig` round-trips through JSON: the parent always-on knobs
    /// and a tagged `selection` block survive serialize → deserialize.
    #[test]
    fn row_selection_config_serde_roundtrip() {
        let original = RowSelectionConfig {
            row_activity_tolerance: Some(1e-6),
            max_active_per_stage: Some(100),
            selection: Some(SelectionMethod::Level1 {
                tie_tolerance: 1e-9,
                check_frequency: 7,
            }),
        };
        let json = serde_json::to_string(&original).unwrap();
        let roundtripped: RowSelectionConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(roundtripped.max_active_per_stage, Some(100));
        assert_eq!(roundtripped.row_activity_tolerance, Some(1e-6));
        match roundtripped.selection.expect("selection present") {
            SelectionMethod::Level1 {
                tie_tolerance,
                check_frequency,
            } => {
                assert!((tie_tolerance - 1e-9).abs() < f64::EPSILON);
                assert_eq!(check_frequency, 7);
            }
            other => panic!("expected Level1, got {other:?}"),
        }
    }

    /// max_active_per_stage absent from JSON deserializes to None.
    #[test]
    fn max_active_per_stage_absent_defaults_none() {
        let f = write_config(
            r#"{
            "training": {
                "selection": {"method": "sampled", "forward_passes": 10},
                "stopping_rules": [{"type": "iteration_limit", "limit": 5}],
                "cut_selection": {"selection": {"method": "level1"}}
            }
        }"#,
        );
        let cfg = parse_config(f.path()).unwrap();
        assert!(
            cfg.training.cut_selection.max_active_per_stage.is_none(),
            "max_active_per_stage must be None when absent from config.json"
        );
    }

    /// `policy.boundary` with `path` and `source_stage` deserializes
    /// to `Some(BoundaryPolicy { .. })` with the correct field values.
    #[test]
    fn test_boundary_policy_present() {
        let f = write_config(
            r#"{
            "training": {
                "selection": {"method": "sampled", "forward_passes": 10},
                "stopping_rules": [{"type": "iteration_limit", "limit": 5}]
            },
            "policy": {
                "mode": "fresh",
                "boundary": {
                    "path": "../monthly/policy",
                    "source_stage": 2
                }
            }
        }"#,
        );
        let cfg = parse_config(f.path()).unwrap();
        let boundary = cfg.policy.boundary.unwrap();
        assert_eq!(boundary.path, "../monthly/policy");
        assert_eq!(boundary.source_stage, Some(2));
    }

    /// `policy.boundary` with `path` but no `source_stage` deserializes to
    /// `Some(BoundaryPolicy { source_stage: None, .. })`; an unknown key
    /// under `boundary` is still rejected by `deny_unknown_fields`.
    #[test]
    fn test_boundary_policy_source_stage_absent_is_none() {
        let f = write_config(
            r#"{
            "training": {
                "selection": {"method": "sampled", "forward_passes": 10},
                "stopping_rules": [{"type": "iteration_limit", "limit": 5}]
            },
            "policy": {
                "mode": "fresh",
                "boundary": {
                    "path": "../monthly/policy"
                }
            }
        }"#,
        );
        let cfg = parse_config(f.path()).unwrap();
        let boundary = cfg.policy.boundary.unwrap();
        assert_eq!(boundary.path, "../monthly/policy");
        assert_eq!(boundary.source_stage, None);

        let unknown_key_json = r#"{
            "training": {
                "selection": {"method": "sampled", "forward_passes": 10},
                "stopping_rules": [{"type": "iteration_limit", "limit": 5}]
            },
            "policy": {
                "mode": "fresh",
                "boundary": { "path": "../monthly/policy", "unexpected": true }
            }
        }"#;
        assert!(
            serde_json::from_str::<Config>(unknown_key_json).is_err(),
            "an unknown key under policy.boundary must still be rejected"
        );
    }

    /// `policy` without a `boundary` key deserializes to `None`.
    #[test]
    fn test_boundary_policy_absent() {
        let f = write_config(
            r#"{
            "training": {
                "selection": {"method": "sampled", "forward_passes": 10},
                "stopping_rules": [{"type": "iteration_limit", "limit": 5}]
            },
            "policy": {}
        }"#,
        );
        let cfg = parse_config(f.path()).unwrap();
        assert!(
            cfg.policy.boundary.is_none(),
            "boundary must be None when the key is absent"
        );
    }

    /// `"boundary": null` deserializes to `None`.
    #[test]
    fn test_boundary_policy_explicit_null() {
        let f = write_config(
            r#"{
            "training": {
                "selection": {"method": "sampled", "forward_passes": 10},
                "stopping_rules": [{"type": "iteration_limit", "limit": 5}]
            },
            "policy": { "boundary": null }
        }"#,
        );
        let cfg = parse_config(f.path()).unwrap();
        assert!(
            cfg.policy.boundary.is_none(),
            "boundary must be None when explicitly null"
        );
    }

    /// `PolicyConfig::default()` has `boundary` set to `None`.
    #[test]
    fn test_policy_config_default_boundary_is_none() {
        assert!(
            PolicyConfig::default().boundary.is_none(),
            "default PolicyConfig must have boundary = None"
        );
    }

    /// Round-trip: serialize `PolicyConfig` with `Some(BoundaryPolicy)`
    /// to JSON and deserialize back; values are preserved.
    #[test]
    fn test_boundary_policy_round_trip() {
        let original = PolicyConfig {
            path: "./policy".to_string(),
            mode: PolicyMode::Fresh,
            checkpointing: CheckpointingConfig::default(),
            boundary: Some(BoundaryPolicy {
                path: "../monthly/policy".to_string(),
                source_stage: Some(5),
            }),
        };
        let json = serde_json::to_string(&original).unwrap();
        let restored: PolicyConfig = serde_json::from_str(&json).unwrap();
        let boundary = restored.boundary.unwrap();
        assert_eq!(boundary.path, "../monthly/policy");
        assert_eq!(boundary.source_stage, Some(5));
    }

    /// Stale `exports` keys (`training`, `cuts`, `vertices`, `simulation`,
    /// `forward_detail`, `backward_detail`, `compression`) are now rejected
    /// because `ExportsConfig` uses `deny_unknown_fields`. Old case dirs that
    /// still contain these keys must remove them before loading.
    #[test]
    fn parse_config_rejects_removed_exports_fields() {
        let json = r#"{
            "training": { "selection": {"method": "sampled", "forward_passes": 4}, "stopping_rules": [] },
            "exports": {
                "training": true,
                "cuts": false,
                "vertices": true,
                "simulation": true,
                "forward_detail": true,
                "backward_detail": true,
                "compression": "zstd"
            }
        }"#;
        let result = serde_json::from_str::<Config>(json);
        assert!(
            result.is_err(),
            "expected parse error for stale exports fields, got Ok"
        );
    }

    // ── OrderSelectionMethod::PacfAnnual tests ────────────────────────────────

    /// `"pacf_annual"` round-trips through serde_json.
    ///
    /// Deserialization must produce `PacfAnnual`; serialization must produce
    /// the `"pacf_annual"` string.
    #[test]
    fn order_selection_pacf_annual_round_trip() {
        let parsed: OrderSelectionMethod = serde_json::from_str("\"pacf_annual\"").unwrap();
        assert!(
            matches!(parsed, OrderSelectionMethod::PacfAnnual),
            "\"pacf_annual\" must deserialize to PacfAnnual, got: {parsed:?}"
        );
        let serialized = serde_json::to_string(&OrderSelectionMethod::PacfAnnual).unwrap();
        assert_eq!(
            serialized, "\"pacf_annual\"",
            "PacfAnnual must serialize to \"pacf_annual\", got: {serialized}"
        );
    }

    /// An unknown variant error must mention `"pacf_annual"` as an expected
    /// variant so users know the option exists.
    #[test]
    fn order_selection_unknown_variant_lists_pacf_annual() {
        let err = serde_json::from_str::<OrderSelectionMethod>("\"pacf_seasonal\"").unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("pacf_annual"),
            "error message must contain \"pacf_annual\", got: {msg}"
        );
    }

    /// The default variant must remain `Pacf`; `PacfAnnual` is opt-in.
    #[test]
    fn order_selection_default_is_pacf() {
        assert!(
            matches!(OrderSelectionMethod::default(), OrderSelectionMethod::Pacf),
            "default must be Pacf, not PacfAnnual"
        );
    }

    /// `"fixed"` is no longer a valid value and must hard-error on parse.
    #[test]
    fn order_selection_fixed_rejected() {
        let result: Result<OrderSelectionMethod, _> = serde_json::from_str("\"fixed\"");
        assert!(
            result.is_err(),
            "\"fixed\" must be rejected; expected an error"
        );
    }

    // ── with_overrides ────────────────────────────────────────────────────────

    /// Minimal valid config used as the `base` Value in override tests.
    const OVERRIDE_BASE_CONFIG: &str = r#"{
      "training": {
        "tree_seed": 42,
        "selection": {"method": "sampled", "forward_passes": 192},
        "stopping_rules": [{"type": "iteration_limit", "limit": 50}],
        "stopping_mode": "any"
      },
      "policy": {
        "checkpointing": {"enabled": true}
      }
    }"#;

    fn base_value(json: &str) -> serde_json::Value {
        serde_json::from_str(json).unwrap()
    }

    fn override_map(
        pairs: &[(&str, serde_json::Value)],
    ) -> serde_json::Map<String, serde_json::Value> {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), v.clone()))
            .collect()
    }

    /// A scalar override sets the value and leaves sibling `training` fields intact.
    #[test]
    fn with_overrides_sets_scalar_and_preserves_siblings() {
        let base = base_value(OVERRIDE_BASE_CONFIG);
        let overrides = override_map(&[("training.tree_seed", serde_json::json!(7))]);

        let cfg = Config::with_overrides(&base, &overrides).unwrap();

        assert_eq!(cfg.training.tree_seed, Some(7));
        // Siblings unchanged from base.
        assert_eq!(
            cfg.resolve_forward_passes(),
            Some(ForwardPassesResolution::Sampled(192))
        );
        assert_eq!(cfg.training.stopping_mode, StoppingMode::Any);
        let rules = cfg.training.stopping_rules.as_deref().unwrap();
        assert!(matches!(
            rules,
            [StoppingRuleConfig::IterationLimit { limit: 50 }]
        ));
    }

    /// An array override deserializes into the expected typed vector.
    #[test]
    fn with_overrides_accepts_array_value() {
        let base = base_value(OVERRIDE_BASE_CONFIG);
        let overrides = override_map(&[(
            "training.stopping_rules",
            serde_json::json!([{"type": "iteration_limit", "limit": 50}]),
        )]);

        let cfg = Config::with_overrides(&base, &overrides).unwrap();

        let rules = cfg.training.stopping_rules.as_deref().unwrap();
        assert!(matches!(
            rules,
            [StoppingRuleConfig::IterationLimit { limit: 50 }]
        ));
    }

    /// A typo key produces SchemaError whose message contains "unknown field".
    #[test]
    fn with_overrides_typo_key_is_schema_error() {
        let base = base_value(OVERRIDE_BASE_CONFIG);
        let overrides = override_map(&[("trainning.tree_seed", serde_json::json!(7))]);

        let err = Config::with_overrides(&base, &overrides).unwrap_err();
        match &err {
            LoadError::SchemaError { message, path, .. } => {
                assert!(
                    message.contains("unknown field"),
                    "message should contain 'unknown field', got: {message}"
                );
                assert_eq!(path, std::path::Path::new("<config_overrides>"));
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// Deep-merge into a nested object does not clobber sibling keys.
    #[test]
    fn with_overrides_deep_merge_preserves_nested_sibling() {
        let base = base_value(OVERRIDE_BASE_CONFIG);
        let overrides = override_map(&[("policy.checkpointing.compress", serde_json::json!(true))]);

        let cfg = Config::with_overrides(&base, &overrides).unwrap();

        assert_eq!(cfg.policy.checkpointing.compress, Some(true));
        // Sibling `enabled` (true in base) must survive the merge.
        assert_eq!(cfg.policy.checkpointing.enabled, Some(true));
    }

    /// An override that clears a required field fails post-merge validation —
    /// the override pipeline runs `validate_config`, not just type-checking.
    #[test]
    fn with_overrides_invalid_value_fails_validation() {
        let base = base_value(OVERRIDE_BASE_CONFIG);
        let overrides = override_map(&[("training.selection", serde_json::Value::Null)]);

        let err = Config::with_overrides(&base, &overrides).unwrap_err();
        match &err {
            LoadError::SchemaError { field, .. } => {
                assert!(
                    field.contains("training.selection"),
                    "field should name training.selection, got: {field}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// An override that makes `simulation.scenario_source` violate an admission
    /// rule fails post-merge validation, carrying the synthetic override path.
    #[test]
    fn with_overrides_rejects_invalid_simulation_scenario_source() {
        let base = base_value(OVERRIDE_BASE_CONFIG);
        let overrides = override_map(&[(
            "simulation.scenario_source",
            serde_json::json!({"seed": 1, "load": {"scheme": "historical"}}),
        )]);

        let err = Config::with_overrides(&base, &overrides).unwrap_err();
        match &err {
            LoadError::SchemaError {
                field,
                message,
                path,
            } => {
                assert!(
                    field.contains("simulation.scenario_source.load.scheme"),
                    "unexpected field: {field}"
                );
                assert!(
                    message.contains("historical scheme is only valid for the inflow class"),
                    "unexpected message: {message}"
                );
                assert_eq!(path, std::path::Path::new("<config_overrides>"));
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// Empty override map yields a Config equal to `from_value(base)`.
    #[test]
    fn with_overrides_empty_map_equals_direct_deserialize() {
        let base = base_value(OVERRIDE_BASE_CONFIG);
        let overrides = serde_json::Map::new();

        let cfg = Config::with_overrides(&base, &overrides).unwrap();
        let direct: Config = serde_json::from_value(base.clone()).unwrap();

        // `Config` has no `PartialEq`; compare via canonical JSON round-trip instead.
        assert_eq!(
            serde_json::to_value(&cfg).unwrap(),
            serde_json::to_value(&direct).unwrap()
        );
    }

    /// An empty path segment (`"training..seed"`) is a SchemaError naming the key.
    #[test]
    fn with_overrides_empty_segment_is_schema_error() {
        let base = base_value(OVERRIDE_BASE_CONFIG);
        let overrides = override_map(&[("training..seed", serde_json::json!(7))]);

        let err = Config::with_overrides(&base, &overrides).unwrap_err();
        match &err {
            LoadError::SchemaError { field, .. } => {
                assert_eq!(field, "training..seed");
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// A non-object `base` is rejected with a SchemaError naming `<root>`.
    #[test]
    fn with_overrides_non_object_base_is_schema_error() {
        let base = serde_json::json!([1, 2, 3]);
        let overrides = serde_json::Map::new();

        let err = Config::with_overrides(&base, &overrides).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert_eq!(field, "<root>");
                assert!(message.contains("must be a JSON object"));
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// A stray key in the `historical_years` range form is a deserialize
    /// error, never a silently dropped key (the untagged enum's `Range`
    /// variant routes through `HistoricalYearRange`'s `deny_unknown_fields`).
    #[test]
    fn historical_years_range_stray_key_is_deserialize_error() {
        let json = r#"{
            "training": {
                "selection": {"method": "sampled", "forward_passes": 4},
                "stopping_rules": [{ "type": "iteration_limit", "limit": 100 }],
                "scenario_source": {
                    "seed": 7,
                    "inflow": { "scheme": "historical" },
                    "historical_years": { "from": 1940, "to": 2010, "step": 2 }
                }
            }
        }"#;
        let result = serde_json::from_str::<Config>(json);
        assert!(
            result.is_err(),
            "a stray key in the historical_years range form must be rejected"
        );
    }

    // ------------------------------------------------------------------
    // Unknown-key injection sweep
    // ------------------------------------------------------------------

    /// Maximal valid configs for the injection sweep: together they exercise
    /// every config section and each internally-tagged / untagged variant
    /// family (both scheduler methods, a `level1` and a `dynamic` selection,
    /// all four stopping-rule types, both `historical_years` forms).
    fn injection_sweep_base_configs() -> Vec<serde_json::Value> {
        let by_scenario_flavored = serde_json::json!({
            "modeling": {
                "inflow_non_negativity": { "method": "penalty" },
                "cost_scale_factor": 1_000_000.0
            },
            "training": {
                "enabled": true,
                "tree_seed": 42,
                "selection": {"method": "sampled", "forward_passes": 4},
                "stopping_rules": [
                    { "type": "iteration_limit", "limit": 10 },
                    { "type": "time_limit", "seconds": 60.0 },
                    { "type": "bound_stalling", "iterations": 5, "tolerance": 0.001 },
                    { "type": "gap", "tolerance": 1000.0, "relative_tolerance": 0.01 }
                ],
                "stopping_mode": "any",
                "cut_selection": {
                    "row_activity_tolerance": 1e-6,
                    "max_active_per_stage": 1000,
                    "selection": {
                        "method": "level1", "tie_tolerance": 1e-10, "check_frequency": 5
                    }
                },
                "solver": {
                    "retry_max_attempts": 3,
                    "retry_time_budget_seconds": 10.0,
                    "backward": {
                        "dual_edge_weight": "devex",
                        "scale": "off",
                        "price": "row",
                        "primal_feasibility_tolerance": 1e-9,
                        "dual_feasibility_tolerance": 1e-9,
                        "presolve": "on",
                        "simplex_update_limit": 5000,
                        "cost_perturbation": 0.0,
                        "refactor_error_tolerance": 1e-6,
                        "factor_pivot_threshold": 0.1,
                        "use_warm_start": true,
                        "steepest_edge_devex_fallback_threshold": 10.0
                    },
                    "forward": { "price": "row_hyper_sparse" }
                },
                "parallelism": {
                    "backward_scheduler": { "method": "by_scenario" }
                },
                "scenario_source": {
                    "seed": 7,
                    "historical_years": { "from": 1940, "to": 2010 },
                    "inflow": { "scheme": "historical" },
                    "load": { "scheme": "in_sample" },
                    "ncs": { "scheme": "in_sample" }
                }
            },
            "upper_bound_evaluation": {
                "enabled": true,
                "initial_iteration": 5,
                "interval_iterations": 10,
                "lipschitz": { "mode": "auto", "fallback_value": 1.0, "scale_factor": 1.1 }
            },
            "policy": {
                "path": "./policy",
                "mode": "fresh",
                "checkpointing": {
                    "enabled": true, "initial_iteration": 1, "interval_iterations": 5,
                    "store_basis": true, "compress": false
                },
                "boundary": { "path": "./boundary", "source_stage": 3 }
            },
            "simulation": {
                "enabled": true,
                "selection": {"method": "sampled", "num_scenarios": 100},
                "io_channel_capacity": 64,
                "scenario_source": {
                    "seed": 9,
                    "historical_years": [1940, 1953],
                    "inflow": { "scheme": "historical" }
                },
                "solver": { "price": "row" },
                "selection": { "method": "sampled", "num_scenarios": 100 }
            },
            "exports": { "states": true, "stochastic": true, "fpha_deviation_points": true },
            "estimation": {
                "max_order": 6,
                "order_selection": "pacf",
                "min_observations_per_season": 30,
                "max_coefficient_magnitude": 2.0
            }
        });
        let by_node_flavored = serde_json::json!({
            "training": {
                "selection": {"method": "sampled", "forward_passes": 4},
                "stopping_rules": [{ "type": "iteration_limit", "limit": 10 }],
                "cut_selection": {
                    "selection": {
                        "method": "dynamic",
                        "start_iteration": 2,
                        "seed_window": 5,
                        "candidate_recency": 20,
                        "max_added_per_round": 10,
                        "violation_tolerance": 1e-10
                    }
                },
                "parallelism": {
                    "backward_scheduler": { "method": "by_node", "block_size": 4 }
                },
                "selection": { "method": "sampled", "forward_passes": 4 }
            }
        });
        vec![by_scenario_flavored, by_node_flavored]
    }

    /// Collect the JSON Pointer of every object node in `value`.
    fn collect_object_pointers(value: &serde_json::Value, pointer: &str, out: &mut Vec<String>) {
        match value {
            serde_json::Value::Object(map) => {
                out.push(pointer.to_string());
                for (key, child) in map {
                    let escaped = key.replace('~', "~0").replace('/', "~1");
                    collect_object_pointers(child, &format!("{pointer}/{escaped}"), out);
                }
            }
            serde_json::Value::Array(items) => {
                for (idx, child) in items.iter().enumerate() {
                    collect_object_pointers(child, &format!("{pointer}/{idx}"), out);
                }
            }
            _ => {}
        }
    }

    /// Every JSON object node in a maximal valid config rejects an injected
    /// unknown key. This is the mechanical closure over serde's per-attribute
    /// enforcement gaps — an internally-tagged unit variant, an untagged
    /// inline struct variant, or a plain missing `deny_unknown_fields` each
    /// silently ignore unknown keys, and a per-type reject test only covers
    /// the types someone remembered to test.
    #[test]
    fn unknown_key_injection_is_rejected_at_every_object_path() {
        for (i, base) in injection_sweep_base_configs().into_iter().enumerate() {
            serde_json::from_value::<Config>(base.clone())
                .unwrap_or_else(|e| panic!("sweep base config {i} must be valid: {e}"));

            let mut pointers = Vec::new();
            collect_object_pointers(&base, "", &mut pointers);
            assert!(
                pointers.len() > 1,
                "sweep base config {i} must contain nested objects"
            );

            for pointer in &pointers {
                let mut mutated = base.clone();
                mutated
                    .pointer_mut(pointer)
                    .and_then(serde_json::Value::as_object_mut)
                    .unwrap_or_else(|| panic!("pointer {pointer:?} must resolve to an object"))
                    .insert("__unknown_key__".to_string(), serde_json::json!(1));
                let result = serde_json::from_value::<Config>(mutated);
                assert!(
                    result.is_err(),
                    "config {i}: an unknown key injected at {pointer:?} must be rejected, \
                     but the config loaded successfully"
                );
            }
        }
    }

    // ── Scenario-selection count resolution ───────────────────────────────────

    /// A `sampled` training selection resolves the forward-pass count.
    #[test]
    fn training_sampled_selection_resolves_count() {
        let via_selection = write_config(
            r#"{"training": {"selection": {"method": "sampled", "forward_passes": 8}, "stopping_rules": [{"type": "iteration_limit", "limit": 5}]}}"#,
        );
        let cfg_sel = parse_config(via_selection.path()).unwrap();
        assert_eq!(
            cfg_sel.resolve_forward_passes(),
            Some(ForwardPassesResolution::Sampled(8))
        );
    }

    /// A config setting the removed root `training.forward_passes` alias fails
    /// to load under `deny_unknown_fields`, the error naming the unknown field;
    /// a flat `simulation.num_scenarios` alias likewise. The count lives solely
    /// in the `selection.sampled` arm.
    #[test]
    fn removed_selection_aliases_fail_to_load() {
        let root_fp = write_config(
            r#"{"training": {"forward_passes": 8, "stopping_rules": [{"type": "iteration_limit", "limit": 5}]}}"#,
        );
        let err = parse_config(root_fp.path()).unwrap_err();
        assert!(
            err.to_string().contains("forward_passes"),
            "root training.forward_passes must be an unknown-field load error naming it, got: {err}"
        );

        let flat_ns = write_config(
            r#"{"training": {"selection": {"method": "sampled", "forward_passes": 4}, "stopping_rules": [{"type": "iteration_limit", "limit": 5}]}, "simulation": {"enabled": true, "num_scenarios": 500}}"#,
        );
        let err = parse_config(flat_ns.path()).unwrap_err();
        assert!(
            err.to_string().contains("num_scenarios"),
            "flat simulation.num_scenarios must be an unknown-field load error naming it, got: {err}"
        );
    }

    /// A `training.selection` of `enumerated` loads and resolves to
    /// [`ForwardPassesResolution::Enumerated`] — the count itself is a
    /// setup-layer concern (the graph-derived count), not config load's.
    #[test]
    fn training_enumerated_selection_resolves() {
        let f = write_config(
            r#"{"training": {"selection": {"method": "enumerated"}, "stopping_rules": [{"type": "iteration_limit", "limit": 5}]}}"#,
        );
        let cfg = parse_config(f.path()).unwrap();
        assert_eq!(
            cfg.resolve_forward_passes(),
            Some(ForwardPassesResolution::Enumerated)
        );
    }

    /// The simulation mirror of `enumerated` loads and resolves to
    /// [`NumScenariosResolution::Enumerated`]; the census count is graph-derived
    /// downstream, carried with no config-side count.
    #[test]
    fn simulation_enumerated_selection_resolves() {
        let f = write_config(
            r#"{"training": {"selection": {"method": "sampled", "forward_passes": 4}, "stopping_rules": [{"type": "iteration_limit", "limit": 5}]}, "simulation": {"enabled": true, "selection": {"method": "enumerated"}}}"#,
        );
        let cfg = parse_config(f.path()).unwrap();
        assert_eq!(
            cfg.resolve_num_scenarios(),
            NumScenariosResolution::Enumerated
        );
    }

    /// A `sampled` simulation selection resolves its scenario count; an absent
    /// selection resolves to the default sampled count.
    #[test]
    fn simulation_sampled_and_default_num_scenarios_resolve() {
        let via_selection = write_config(
            r#"{"training": {"selection": {"method": "sampled", "forward_passes": 4}, "stopping_rules": [{"type": "iteration_limit", "limit": 5}]}, "simulation": {"enabled": true, "selection": {"method": "sampled", "num_scenarios": 500}}}"#,
        );
        let cfg_sel = parse_config(via_selection.path()).unwrap();
        assert_eq!(
            cfg_sel.resolve_num_scenarios(),
            NumScenariosResolution::Sampled(500)
        );

        let via_default = write_config(
            r#"{"training": {"selection": {"method": "sampled", "forward_passes": 4}, "stopping_rules": [{"type": "iteration_limit", "limit": 5}]}, "simulation": {"enabled": true}}"#,
        );
        let cfg_default = parse_config(via_default.path()).unwrap();
        assert_eq!(
            cfg_default.resolve_num_scenarios(),
            NumScenariosResolution::Sampled(2000)
        );
    }
}
