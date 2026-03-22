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
//! - `training.forward_passes` — number of scenario trajectories per iteration
//! - `training.stopping_rules` — at least one rule entry (must include `iteration_limit`)
//!
//! # Examples
//!
//! ```no_run
//! use cobre_io::config::parse_config;
//! use std::path::Path;
//!
//! let cfg = parse_config(Path::new("case/config.json")).unwrap();
//! println!("forward_passes = {:?}", cfg.training.forward_passes);
//! ```

use crate::LoadError;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Top-level deserialized representation of `config.json`.
///
/// All sections except `training` are optional; their defaults are applied by
/// serde when the section is absent from the JSON.
#[derive(Debug, Clone, Deserialize, Serialize)]
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

/// Modeling options (`config.json → modeling`).
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ModelingConfig {
    /// Strategy for handling non-negative inflow constraints.
    #[serde(default)]
    pub inflow_non_negativity: InflowNonNegativityConfig,
}

/// Inflow non-negativity treatment settings.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct InflowNonNegativityConfig {
    /// Method: `"none"`, `"penalty"`, or `"truncation"`.
    pub method: String,

    /// Penalty coefficient $c^{inf}$ applied when `method` is `"penalty"`.
    pub penalty_cost: f64,
}

impl Default for InflowNonNegativityConfig {
    fn default() -> Self {
        Self {
            method: "penalty".to_string(),
            penalty_cost: 1000.0,
        }
    }
}

/// Training parameters (`config.json → training`).
///
/// `forward_passes` and `stopping_rules` are mandatory — the loader returns
/// [`LoadError::SchemaError`] if either is absent.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct TrainingConfig {
    /// Enable the training phase. When `false`, skip directly to simulation.
    #[serde(default = "TrainingConfig::default_enabled")]
    pub enabled: bool,

    /// Random seed for reproducible scenario generation.
    #[serde(default)]
    pub seed: Option<i64>,

    /// Number of forward-pass scenario trajectories $M$ per iteration.
    ///
    /// **Mandatory** — no default. The loader rejects any config that omits this field.
    pub forward_passes: Option<u32>,

    /// List of stopping rule configurations.
    ///
    /// **Mandatory** — no default. Must contain at least one `iteration_limit` rule.
    pub stopping_rules: Option<Vec<StoppingRuleConfig>>,

    /// How multiple stopping rules combine: `"any"` (OR) or `"all"` (AND).
    #[serde(default = "TrainingConfig::default_stopping_mode")]
    pub stopping_mode: String,

    /// Cut formulation: `"single"` or `"multi"`.
    #[serde(default)]
    pub cut_formulation: Option<String>,

    /// Forward pass configuration.
    #[serde(default)]
    pub forward_pass: Option<ForwardPassConfig>,

    /// Cut selection settings.
    #[serde(default)]
    pub cut_selection: CutSelectionConfig,

    /// LP solver retry settings.
    #[serde(default)]
    pub solver: TrainingSolverConfig,
}

impl TrainingConfig {
    fn default_enabled() -> bool {
        true
    }

    fn default_stopping_mode() -> String {
        "any".to_string()
    }

    // Note: Default impl is not provided for TrainingConfig because forward_passes
    // and stopping_rules are mandatory and have no sensible defaults.
}

/// Forward pass mode configuration.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ForwardPassConfig {
    /// Forward pass type: `"default"` or other variants.
    #[serde(rename = "type")]
    pub pass_type: String,
}

/// Cut selection settings (`config.json → training.cut_selection`).
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct CutSelectionConfig {
    /// Enable cut pruning.
    #[serde(default)]
    pub enabled: Option<bool>,

    /// Method: `"level1"`, `"lml1"`, or `"domination"`.
    #[serde(default)]
    pub method: Option<String>,

    /// Minimum iterations before first pruning pass.
    #[serde(default)]
    pub threshold: Option<u32>,

    /// Iterations between pruning checks.
    #[serde(default)]
    pub check_frequency: Option<u32>,

    /// Minimum dual multiplier for a cut to count as binding.
    #[serde(default)]
    pub cut_activity_tolerance: Option<f64>,
}

/// LP solver retry settings (`config.json → training.solver`).
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct TrainingSolverConfig {
    /// Maximum solver retry attempts before propagating a hard error.
    pub retry_max_attempts: u32,

    /// Total time budget in seconds across all retry attempts for one solve.
    pub retry_time_budget_seconds: f64,
}

impl Default for TrainingSolverConfig {
    fn default() -> Self {
        Self {
            retry_max_attempts: 5,
            retry_time_budget_seconds: 30.0,
        }
    }
}

/// Deserialized configuration for one entry in `training.stopping_rules[]`.
///
/// Uses a `"type"` discriminator field (internally tagged) with `snake_case`
/// variant names matching the JSON schema.
///
/// The `GracefulShutdown` rule has no JSON representation — it is injected at
/// runtime by `StoppingRuleSet` construction and is never deserialized.
///
/// # Examples
///
/// ```
/// use cobre_io::config::StoppingRuleConfig;
///
/// let json = r#"{"type": "iteration_limit", "limit": 100}"#;
/// let rule: StoppingRuleConfig = serde_json::from_str(json).unwrap();
/// assert!(matches!(rule, StoppingRuleConfig::IterationLimit { limit: 100 }));
/// ```
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub enum StoppingRuleConfig {
    /// Stop after a fixed number of iterations. **Mandatory** — every rule set must
    /// contain at least one `iteration_limit` rule.
    IterationLimit {
        /// Maximum iteration count $k_{max}$.
        limit: u32,
    },
    /// Stop after a wall-clock time limit.
    TimeLimit {
        /// Time limit in seconds.
        seconds: f64,
    },
    /// Stop when the lower bound stalls (relative improvement falls below tolerance).
    BoundStalling {
        /// Window size $\tau$ (number of past iterations to compare).
        iterations: u32,
        /// Relative improvement threshold.
        tolerance: f64,
    },
    /// Stop when both the bound and simulated policy costs have stabilized.
    Simulation {
        /// Number of Monte Carlo forward simulations per check.
        replications: u32,
        /// Iterations between checks.
        period: u32,
        /// Number of past iterations for bound stability check.
        bound_window: u32,
        /// Normalized distance threshold between consecutive simulation results.
        distance_tol: f64,
        /// Relative tolerance for bound stability.
        bound_tol: f64,
    },
}

/// Upper-bound evaluation settings (`config.json → upper_bound_evaluation`).
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct UpperBoundEvaluationConfig {
    /// Enable vertex-based inner approximation for upper bound computation.
    #[serde(default)]
    pub enabled: Option<bool>,

    /// First iteration to compute the upper bound.
    #[serde(default)]
    pub initial_iteration: Option<u32>,

    /// Iterations between upper-bound evaluations.
    #[serde(default)]
    pub interval_iterations: Option<u32>,

    /// Lipschitz constant settings.
    #[serde(default)]
    pub lipschitz: LipschitzConfig,
}

/// Lipschitz constant settings for inner approximation.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct LipschitzConfig {
    /// Computation mode: `"auto"`.
    #[serde(default)]
    pub mode: Option<String>,

    /// Fallback value when automatic computation fails.
    #[serde(default)]
    pub fallback_value: Option<f64>,

    /// Multiplicative safety margin applied to computed Lipschitz constants.
    #[serde(default)]
    pub scale_factor: Option<f64>,
}

/// Policy directory settings (`config.json → policy`).
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct PolicyConfig {
    /// Directory for policy data (cuts, states, vertices, basis).
    pub path: String,

    /// Initialization mode: `"fresh"`, `"warm_start"`, or `"resume"`.
    pub mode: String,

    /// Verify state dimension and entity compatibility when loading.
    pub validate_compatibility: bool,

    /// Checkpoint settings.
    pub checkpointing: CheckpointingConfig,
}

impl Default for PolicyConfig {
    fn default() -> Self {
        Self {
            path: "./policy".to_string(),
            mode: "fresh".to_string(),
            validate_compatibility: true,
            checkpointing: CheckpointingConfig::default(),
        }
    }
}

/// Checkpoint settings (`config.json → policy.checkpointing`).
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct CheckpointingConfig {
    /// Enable periodic checkpointing.
    #[serde(default)]
    pub enabled: Option<bool>,

    /// First iteration to write a checkpoint.
    #[serde(default)]
    pub initial_iteration: Option<u32>,

    /// Iterations between checkpoints.
    #[serde(default)]
    pub interval_iterations: Option<u32>,

    /// Include LP basis in checkpoints for warm-start.
    #[serde(default)]
    pub store_basis: Option<bool>,

    /// Compress checkpoint files.
    #[serde(default)]
    pub compress: Option<bool>,
}

/// Post-training simulation settings (`config.json → simulation`).
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct SimulationConfig {
    /// Enable post-training simulation.
    pub enabled: bool,

    /// Number of simulation scenarios.
    pub num_scenarios: u32,

    /// Policy representation: `"outer"` (cuts) or `"inner"` (vertices).
    pub policy_type: String,

    /// Directory for simulation output files.
    pub output_path: Option<String>,

    /// Output mode: `"streaming"` or `"batched"`.
    pub output_mode: Option<String>,

    /// Bounded channel capacity between simulation threads and the I/O writer thread.
    pub io_channel_capacity: u32,

    /// Sampling scheme for simulation scenarios.
    pub sampling_scheme: SimulationSamplingConfig,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            num_scenarios: 2000,
            policy_type: "outer".to_string(),
            output_path: None,
            output_mode: None,
            io_channel_capacity: 64,
            sampling_scheme: SimulationSamplingConfig::default(),
        }
    }
}

/// Sampling scheme for the post-training simulation.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct SimulationSamplingConfig {
    /// Scheme type: `"in_sample"`, `"out_of_sample"`, or `"external"`.
    #[serde(rename = "type")]
    pub scheme_type: String,
}

impl Default for SimulationSamplingConfig {
    fn default() -> Self {
        Self {
            scheme_type: "in_sample".to_string(),
        }
    }
}

/// Order selection criterion for autoregressive model fitting.
///
/// Controls how the lag order is chosen when fitting a time series model.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub enum OrderSelectionMethod {
    /// Use a fixed maximum lag order specified by `max_order`.
    Fixed,
    /// Select the lag order minimising the Akaike Information Criterion.
    #[default]
    Aic,
    /// Select the lag order using partial autocorrelation significance testing.
    ///
    /// Tests each lag against a 95% confidence interval (`1.96 / sqrt(N)`)
    /// and selects the maximum lag with a statistically significant partial
    /// autocorrelation. Generally more conservative than AIC.
    Pacf,
}

/// Time series estimation settings (`config.json → estimation`).
///
/// Controls automatic parameter estimation when historical inflow data is
/// provided without explicit model statistics or coefficients.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct EstimationConfig {
    /// Maximum lag order considered during autoregressive model fitting.
    pub max_order: u32,

    /// Order selection criterion: fixed maximum or information-criterion-based.
    pub order_selection: OrderSelectionMethod,

    /// Minimum number of observations required per (entity, season) group
    /// to proceed with estimation. Groups below this threshold are skipped.
    pub min_observations_per_season: u32,
}

impl Default for EstimationConfig {
    fn default() -> Self {
        Self {
            max_order: 6,
            order_selection: OrderSelectionMethod::Aic,
            min_observations_per_season: 30,
        }
    }
}

/// Export flags controlling which outputs are written to disk
/// (`config.json → exports`).
///
/// The struct uses multiple `bool` fields because each flag maps directly to a
/// JSON field name in the `exports` section of `config.json`. A state machine
/// would not improve clarity for a flat set of independent output toggles.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ExportsConfig {
    /// Export training summary metrics.
    pub training: bool,

    /// Export cut pool (outer approximation).
    pub cuts: bool,

    /// Export visited states.
    pub states: bool,

    /// Export inner approximation vertices.
    pub vertices: bool,

    /// Export simulation results.
    pub simulation: bool,

    /// Export per-scenario forward-pass detail.
    pub forward_detail: bool,

    /// Export per-scenario backward-pass detail.
    pub backward_detail: bool,

    /// Export stochastic preprocessing artifacts to `output/stochastic/`.
    pub stochastic: bool,

    /// Compression algorithm for output files: `"zstd"`, `"lz4"`, or `"none"`.
    pub compression: Option<String>,
}

impl Default for ExportsConfig {
    fn default() -> Self {
        Self {
            training: true,
            cuts: true,
            states: true,
            vertices: true,
            simulation: true,
            forward_detail: false,
            backward_detail: false,
            stochastic: false,
            compression: None,
        }
    }
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
/// | `training.forward_passes` missing | [`LoadError::SchemaError`]    |
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
/// assert!(cfg.training.forward_passes.unwrap_or(0) > 0);
/// ```
pub fn parse_config(path: &Path) -> Result<Config, LoadError> {
    let raw = std::fs::read_to_string(path).map_err(|e| LoadError::io(path, e))?;

    let config: Config = serde_json::from_str(&raw).map_err(|e| {
        // serde_json errors carry a message that describes the field or syntax problem.
        // Unknown enum variants in a tagged enum produce a deserialization error whose
        // message contains the unknown variant name — surfaced to the caller as
        // SchemaError when the field is identifiable, otherwise as ParseError.
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

/// Extract a field name hint from a `serde_json` error message.
///
/// Extracts the identifier between backticks, returning a best-effort field name
/// or `"<unknown>"` when no match is found.
fn extract_field_from_serde_msg(msg: &str) -> String {
    if let Some(start) = msg.find('`') {
        if let Some(end) = msg[start + 1..].find('`') {
            return msg[start + 1..start + 1 + end].to_string();
        }
    }
    "<unknown>".to_string()
}

/// Post-deserialization validation for mandatory fields.
///
/// Checks that `forward_passes` and `stopping_rules` are present in the config.
fn validate_config(config: &Config, path: &Path) -> Result<(), LoadError> {
    if config.training.forward_passes.is_none() {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: "training.forward_passes".to_string(),
            message: "required field is missing".to_string(),
        });
    }

    if config.training.stopping_rules.is_none() {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: "training.stopping_rules".to_string(),
            message: "required field is missing".to_string(),
        });
    }

    Ok(())
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
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

    /// AC-1: minimal config returns Ok with correct forward_passes and all
    /// optional sections at their default values.
    #[test]
    fn test_parse_minimal_config() {
        let f = write_config(
            r#"{"training": {"seed": 42, "forward_passes": 192, "stopping_rules": [{"type": "iteration_limit", "limit": 50}]}}"#,
        );
        let cfg = parse_config(f.path()).unwrap();

        // Mandatory field present and correct
        assert_eq!(cfg.training.forward_passes, Some(192));

        // Seed is optional
        assert_eq!(cfg.training.seed, Some(42));

        // Defaults applied to optional sections
        assert_eq!(cfg.training.stopping_mode, "any");
        assert!(cfg.training.enabled);
        assert_eq!(
            cfg.modeling.inflow_non_negativity.method,
            "penalty".to_string()
        );
        assert!((cfg.modeling.inflow_non_negativity.penalty_cost - 1000.0).abs() < f64::EPSILON);
        assert!(!cfg.simulation.enabled);
        assert_eq!(cfg.simulation.num_scenarios, 2000);
        assert_eq!(cfg.policy.mode, "fresh");
        assert_eq!(cfg.policy.path, "./policy");
        assert!(cfg.policy.validate_compatibility);
        assert!(cfg.exports.training);
        assert!(cfg.exports.cuts);
    }

    /// AC-2: missing `training.forward_passes` → SchemaError with field name.
    #[test]
    fn test_missing_forward_passes() {
        let f = write_config(
            r#"{"training": {"seed": 1, "stopping_rules": [{"type": "iteration_limit", "limit": 10}]}}"#,
        );
        let err = parse_config(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { field, .. } => {
                assert!(
                    field.contains("forward_passes"),
                    "field should contain 'forward_passes', got: {field}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// AC-2 variant: missing `training.stopping_rules` → SchemaError.
    #[test]
    fn test_missing_stopping_rules() {
        let f = write_config(r#"{"training": {"seed": 1, "forward_passes": 100}}"#);
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

    /// AC-3: nonexistent file → IoError with matching path.
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

    /// AC-4: full config with all sections → Ok with non-default values.
    #[test]
    fn test_parse_full_config() {
        let json = r#"{
          "$schema": "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main/book/src/schemas/config.schema.json",
          "modeling": {
            "inflow_non_negativity": {
              "method": "penalty",
              "penalty_cost": 500.0
            }
          },
          "training": {
            "seed": 42,
            "forward_passes": 192,
            "stopping_rules": [
              {"type": "iteration_limit", "limit": 50},
              {"type": "bound_stalling", "iterations": 10, "tolerance": 0.0001}
            ],
            "stopping_mode": "any",
            "cut_formulation": "single",
            "forward_pass": {"type": "default"},
            "cut_selection": {
              "enabled": true,
              "method": "domination",
              "threshold": 0
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
            },
            "validate_compatibility": true
          },
          "simulation": {
            "enabled": true,
            "num_scenarios": 2000,
            "policy_type": "outer",
            "output_path": "./simulation",
            "output_mode": "streaming",
            "sampling_scheme": {"type": "in_sample"}
          },
          "exports": {
            "training": true,
            "cuts": true,
            "states": true,
            "vertices": true,
            "simulation": true,
            "forward_detail": false,
            "backward_detail": false,
            "compression": "zstd"
          }
        }"#;

        let f = write_config(json);
        let cfg = parse_config(f.path()).unwrap();

        // Modeling
        assert_eq!(cfg.modeling.inflow_non_negativity.method, "penalty");
        assert!((cfg.modeling.inflow_non_negativity.penalty_cost - 500.0).abs() < f64::EPSILON);

        // Training
        assert_eq!(cfg.training.forward_passes, Some(192));
        assert_eq!(cfg.training.stopping_mode, "any");
        let rules = cfg.training.stopping_rules.as_ref().unwrap();
        assert_eq!(rules.len(), 2);
        assert_eq!(cfg.training.cut_formulation.as_deref(), Some("single"));
        let cut_sel = &cfg.training.cut_selection;
        assert_eq!(cut_sel.enabled, Some(true));
        assert_eq!(cut_sel.method.as_deref(), Some("domination"));

        // Upper bound
        assert_eq!(cfg.upper_bound_evaluation.enabled, Some(true));
        assert_eq!(cfg.upper_bound_evaluation.initial_iteration, Some(10));

        // Policy
        assert_eq!(cfg.policy.mode, "fresh");
        assert!(cfg.policy.validate_compatibility);
        assert_eq!(cfg.policy.checkpointing.enabled, Some(true));

        // Simulation
        assert!(cfg.simulation.enabled);
        assert_eq!(cfg.simulation.num_scenarios, 2000);
        assert_eq!(cfg.simulation.policy_type, "outer");

        // Exports
        assert!(cfg.exports.training);
        assert_eq!(cfg.exports.compression.as_deref(), Some("zstd"));
        assert!(!cfg.exports.forward_detail);
    }

    /// AC-5: invalid JSON syntax → ParseError.
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
            "forward_passes": 10,
            "stopping_rules": [
              {"type": "iteration_limit", "limit": 100},
              {"type": "time_limit", "seconds": 3600.0},
              {"type": "bound_stalling", "iterations": 10, "tolerance": 0.0001},
              {
                "type": "simulation",
                "replications": 100,
                "period": 20,
                "bound_window": 5,
                "distance_tol": 0.01,
                "bound_tol": 0.0001
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
            StoppingRuleConfig::Simulation {
                replications: 100,
                period: 20,
                ..
            }
        ));
    }

    /// Unknown stopping rule type → SchemaError (not a panic or ParseError).
    #[test]
    fn test_unknown_stopping_rule_type() {
        let f = write_config(
            r#"{"training": {"forward_passes": 10, "stopping_rules": [{"type": "nonexistent_rule"}]}}"#,
        );
        let err = parse_config(f.path()).unwrap_err();
        assert!(
            matches!(err, LoadError::SchemaError { .. }),
            "expected SchemaError for unknown rule type, got: {err:?}"
        );
    }

    /// AC (ticket-007b): `Config` has no `version` field — the struct does not
    /// expose `.version` and the field is not present after deserialization.
    #[test]
    fn test_config_has_no_version_field() {
        let f = write_config(
            r#"{"training": {"forward_passes": 1, "stopping_rules": [{"type": "iteration_limit", "limit": 10}]}}"#,
        );
        let cfg = parse_config(f.path()).unwrap();
        // The struct must not have a `version` field — verified by compilation.
        // We also check that the $schema field is None when absent from JSON.
        assert!(cfg.schema.is_none(), "schema should be None when absent");
    }

    /// AC (ticket-007b): JSON with `"$schema"` property is accepted and the field
    /// value is stored correctly.
    #[test]
    fn test_schema_field_accepted() {
        let f = write_config(
            r#"{
            "$schema": "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main/book/src/schemas/config.schema.json",
            "training": {
                "forward_passes": 1,
                "stopping_rules": [{"type": "iteration_limit", "limit": 10}]
            }
        }"#,
        );
        let cfg = parse_config(f.path()).unwrap();
        assert_eq!(
            cfg.schema.as_deref(),
            Some(
                "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main/book/src/schemas/config.schema.json"
            ),
            "schema field should be stored when present in JSON"
        );
    }

    /// AC (ticket-007b): JSON that still contains a `"version"` property is
    /// silently accepted because `Config` has no `deny_unknown_fields` and the
    /// removed field is treated as an unknown key that serde ignores.
    #[test]
    fn test_legacy_version_field_silently_ignored() {
        let f = write_config(
            r#"{
            "version": "1.0.0",
            "training": {
                "forward_passes": 1,
                "stopping_rules": [{"type": "iteration_limit", "limit": 10}]
            }
        }"#,
        );
        // Must parse successfully — backward compatibility for existing case dirs.
        let cfg = parse_config(f.path()).unwrap();
        assert_eq!(cfg.training.forward_passes, Some(1));
    }

    /// AC (ticket-007): `"truncation"` is accepted as a method string and
    /// round-trips correctly through `parse_config`. The `penalty_cost` field
    /// falls back to its default (1000.0) when absent from the JSON.
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
                "forward_passes": 10,
                "stopping_rules": [{"type": "iteration_limit", "limit": 5}]
            }
        }"#,
        );
        let cfg = parse_config(f.path()).unwrap();
        assert_eq!(
            cfg.modeling.inflow_non_negativity.method, "truncation",
            "method field should round-trip as 'truncation'"
        );
        assert!(
            (cfg.modeling.inflow_non_negativity.penalty_cost - 1000.0).abs() < f64::EPSILON,
            "penalty_cost should be the default 1000.0 when absent from JSON"
        );
    }

    /// AC-035-1: `config.json` without `"estimation"` section → all three defaults applied.
    #[test]
    fn test_estimation_config_defaults() {
        let f = write_config(
            r#"{"training": {"forward_passes": 10, "stopping_rules": [{"type": "iteration_limit", "limit": 5}]}}"#,
        );
        let cfg = parse_config(f.path()).unwrap();
        assert_eq!(cfg.estimation.max_order, 6);
        assert!(
            matches!(cfg.estimation.order_selection, OrderSelectionMethod::Aic),
            "default order_selection should be Aic"
        );
        assert_eq!(cfg.estimation.min_observations_per_season, 30);
    }

    /// AC-035-2: explicit estimation section round-trips all three fields.
    #[test]
    fn test_estimation_config_explicit() {
        let f = write_config(
            r#"{
            "training": {"forward_passes": 10, "stopping_rules": [{"type": "iteration_limit", "limit": 5}]},
            "estimation": {"max_order": 3, "order_selection": "fixed", "min_observations_per_season": 20}
        }"#,
        );
        let cfg = parse_config(f.path()).unwrap();
        assert_eq!(cfg.estimation.max_order, 3);
        assert!(
            matches!(cfg.estimation.order_selection, OrderSelectionMethod::Fixed),
            "order_selection should be Fixed"
        );
        assert_eq!(cfg.estimation.min_observations_per_season, 20);
    }

    /// AC-035-3: unknown `order_selection` value → `LoadError::SchemaError` with
    /// message containing `"unknown variant"`.
    #[test]
    fn test_estimation_config_unknown_order_selection() {
        let f = write_config(
            r#"{
            "training": {"forward_passes": 10, "stopping_rules": [{"type": "iteration_limit", "limit": 5}]},
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
    ///
    /// Verifies that a `config.json` with `"exports": {"stochastic": true}` round-trips
    /// the field as `true` in `ExportsConfig`.
    #[test]
    fn test_exports_stochastic_explicit_true() {
        let f = write_config(
            r#"{
            "training": {"forward_passes": 10, "stopping_rules": [{"type": "iteration_limit", "limit": 5}]},
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
    ///
    /// Verifies that a `config.json` without the `stochastic` field in the
    /// `exports` section resolves to `false` via `#[serde(default)]`.
    #[test]
    fn test_exports_stochastic_defaults_to_false() {
        let f = write_config(
            r#"{
            "training": {"forward_passes": 10, "stopping_rules": [{"type": "iteration_limit", "limit": 5}]}
        }"#,
        );
        let cfg = parse_config(f.path()).unwrap();
        assert!(
            !cfg.exports.stochastic,
            "exports.stochastic should default to false when absent"
        );
    }
}
