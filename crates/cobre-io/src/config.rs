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
use serde::Deserialize;
use std::path::Path;

// ── Top-level Config ─────────────────────────────────────────────────────────

/// Top-level deserialized representation of `config.json`.
///
/// All sections except `training` are optional; their defaults are applied by
/// serde when the section is absent from the JSON.
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    /// JSON schema URI — informational, not validated.
    #[serde(rename = "$schema")]
    pub schema: Option<String>,

    /// Config format version (e.g. `"2.0.0"`).
    pub version: Option<String>,

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
}

// ── Modeling Configuration ───────────────────────────────────────────────────

/// Modeling options (`config.json → modeling`).
#[derive(Debug, Clone, Deserialize, Default)]
pub struct ModelingConfig {
    /// Strategy for handling non-negative inflow constraints.
    #[serde(default)]
    pub inflow_non_negativity: InflowNonNegativityConfig,
}

/// Inflow non-negativity treatment settings.
#[derive(Debug, Clone, Deserialize)]
pub struct InflowNonNegativityConfig {
    /// Method: `"none"`, `"penalty"`, `"truncation"`, or `"truncation_with_penalty"`.
    #[serde(default = "InflowNonNegativityConfig::default_method")]
    pub method: String,

    /// Penalty coefficient $c^{inf}$ applied when `method` is `"penalty"` or
    /// `"truncation_with_penalty"`.
    #[serde(default = "InflowNonNegativityConfig::default_penalty_cost")]
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

impl InflowNonNegativityConfig {
    fn default_method() -> String {
        "penalty".to_string()
    }

    fn default_penalty_cost() -> f64 {
        1000.0
    }
}

// ── Training Configuration ───────────────────────────────────────────────────

/// Training parameters (`config.json → training`).
///
/// `forward_passes` and `stopping_rules` are mandatory — the loader returns
/// [`LoadError::SchemaError`] if either is absent.
#[derive(Debug, Clone, Deserialize)]
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
#[derive(Debug, Clone, Deserialize)]
pub struct ForwardPassConfig {
    /// Forward pass type: `"default"` or other variants.
    #[serde(rename = "type")]
    pub pass_type: String,
}

/// Cut selection settings (`config.json → training.cut_selection`).
#[derive(Debug, Clone, Deserialize, Default)]
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
#[derive(Debug, Clone, Deserialize)]
pub struct TrainingSolverConfig {
    /// Maximum solver retry attempts before propagating a hard error.
    #[serde(default = "TrainingSolverConfig::default_retry_max_attempts")]
    pub retry_max_attempts: u32,

    /// Total time budget in seconds across all retry attempts for one solve.
    #[serde(default = "TrainingSolverConfig::default_retry_time_budget_seconds")]
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

impl TrainingSolverConfig {
    fn default_retry_max_attempts() -> u32 {
        5
    }

    fn default_retry_time_budget_seconds() -> f64 {
        30.0
    }
}

// ── Stopping Rules ───────────────────────────────────────────────────────────

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
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
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

// ── Upper-Bound Evaluation ───────────────────────────────────────────────────

/// Upper-bound evaluation settings (`config.json → upper_bound_evaluation`).
#[derive(Debug, Clone, Deserialize, Default)]
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
#[derive(Debug, Clone, Deserialize, Default)]
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

// ── Policy Configuration ─────────────────────────────────────────────────────

/// Policy directory settings (`config.json → policy`).
#[derive(Debug, Clone, Deserialize)]
pub struct PolicyConfig {
    /// Directory for policy data (cuts, states, vertices, basis).
    #[serde(default = "PolicyConfig::default_path")]
    pub path: String,

    /// Initialization mode: `"fresh"`, `"warm_start"`, or `"resume"`.
    #[serde(default = "PolicyConfig::default_mode")]
    pub mode: String,

    /// Verify state dimension and entity compatibility when loading.
    #[serde(default = "PolicyConfig::default_validate_compatibility")]
    pub validate_compatibility: bool,

    /// Checkpoint settings.
    #[serde(default)]
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

impl PolicyConfig {
    fn default_path() -> String {
        "./policy".to_string()
    }

    fn default_mode() -> String {
        "fresh".to_string()
    }

    fn default_validate_compatibility() -> bool {
        true
    }
}

/// Checkpoint settings (`config.json → policy.checkpointing`).
#[derive(Debug, Clone, Deserialize, Default)]
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

// ── Simulation Configuration ─────────────────────────────────────────────────

/// Post-training simulation settings (`config.json → simulation`).
#[derive(Debug, Clone, Deserialize)]
pub struct SimulationConfig {
    /// Enable post-training simulation.
    #[serde(default = "SimulationConfig::default_enabled")]
    pub enabled: bool,

    /// Number of simulation scenarios.
    #[serde(default = "SimulationConfig::default_num_scenarios")]
    pub num_scenarios: u32,

    /// Policy representation: `"outer"` (cuts) or `"inner"` (vertices).
    #[serde(default = "SimulationConfig::default_policy_type")]
    pub policy_type: String,

    /// Directory for simulation output files.
    #[serde(default)]
    pub output_path: Option<String>,

    /// Output mode: `"streaming"` or `"batched"`.
    #[serde(default)]
    pub output_mode: Option<String>,

    /// Bounded channel capacity between simulation threads and the I/O writer thread.
    #[serde(default = "SimulationConfig::default_io_channel_capacity")]
    pub io_channel_capacity: u32,

    /// Sampling scheme for simulation scenarios.
    #[serde(default)]
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

impl SimulationConfig {
    fn default_enabled() -> bool {
        false
    }

    fn default_num_scenarios() -> u32 {
        2000
    }

    fn default_policy_type() -> String {
        "outer".to_string()
    }

    fn default_io_channel_capacity() -> u32 {
        64
    }
}

/// Sampling scheme for the post-training simulation.
#[derive(Debug, Clone, Deserialize)]
pub struct SimulationSamplingConfig {
    /// Scheme type: `"in_sample"`, `"out_of_sample"`, or `"external"`.
    #[serde(rename = "type", default = "SimulationSamplingConfig::default_type")]
    pub scheme_type: String,
}

impl Default for SimulationSamplingConfig {
    fn default() -> Self {
        Self {
            scheme_type: "in_sample".to_string(),
        }
    }
}

impl SimulationSamplingConfig {
    fn default_type() -> String {
        "in_sample".to_string()
    }
}

// ── Exports Configuration ────────────────────────────────────────────────────

/// Export flags controlling which outputs are written to disk
/// (`config.json → exports`).
///
/// The struct uses multiple `bool` fields because each flag maps directly to a
/// JSON field name in the `exports` section of `config.json`. A state machine
/// would not improve clarity for a flat set of independent output toggles.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, Deserialize)]
pub struct ExportsConfig {
    /// Export training summary metrics.
    #[serde(default = "ExportsConfig::default_training")]
    pub training: bool,

    /// Export cut pool (outer approximation).
    #[serde(default = "ExportsConfig::default_cuts")]
    pub cuts: bool,

    /// Export visited states.
    #[serde(default = "ExportsConfig::default_states")]
    pub states: bool,

    /// Export inner approximation vertices.
    #[serde(default = "ExportsConfig::default_vertices")]
    pub vertices: bool,

    /// Export simulation results.
    #[serde(default = "ExportsConfig::default_simulation")]
    pub simulation: bool,

    /// Export per-scenario forward-pass detail.
    #[serde(default)]
    pub forward_detail: bool,

    /// Export per-scenario backward-pass detail.
    #[serde(default)]
    pub backward_detail: bool,

    /// Compression algorithm for output files: `"zstd"`, `"lz4"`, or `"none"`.
    #[serde(default)]
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
            compression: None,
        }
    }
}

impl ExportsConfig {
    fn default_training() -> bool {
        true
    }

    fn default_cuts() -> bool {
        true
    }

    fn default_states() -> bool {
        true
    }

    fn default_vertices() -> bool {
        true
    }

    fn default_simulation() -> bool {
        true
    }
}

// ── parse_config ─────────────────────────────────────────────────────────────

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
/// `serde_json` error messages follow patterns such as:
/// - `"missing field 'forward_passes' at line 1 column 2"`
/// - `"unknown variant 'foo', expected one of …"`
///
/// This helper extracts the identifier between backticks, returning a best-effort
/// field name or `"<unknown>"` when no match is found.
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
/// serde can only detect missing mandatory fields when they have no `Option`
/// wrapper and no `#[serde(default)]`. We model mandatory fields as
/// `Option<T>` so that the rest of the struct still deserializes on partial
/// input, then validate presence here for better error messages.
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

    // ── helpers ──

    fn write_config(content: &str) -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f
    }

    // ── unit tests ──

    /// AC-1: minimal config returns Ok with correct forward_passes and all
    /// optional sections at their default values.
    #[test]
    fn test_parse_minimal_config() {
        let f = write_config(
            r#"{"version": "2.0.0", "training": {"seed": 42, "forward_passes": 192, "stopping_rules": [{"type": "iteration_limit", "limit": 50}]}}"#,
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
          "$schema": "https://cobre.dev/schemas/v2/config.schema.json",
          "version": "2.0.0",
          "modeling": {
            "inflow_non_negativity": {
              "method": "truncation",
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
        assert_eq!(cfg.modeling.inflow_non_negativity.method, "truncation");
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
}
