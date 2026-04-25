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

use cobre_core::scenario::{HistoricalYears, SamplingScheme, ScenarioSource};

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
    ///
    /// **Deprecated:** Use `penalties.json` -> `hydro.inflow_nonnegativity_cost`
    /// instead. When both are specified, the penalty cascade takes precedence.
    /// This field is retained for backward compatibility with existing cases
    /// that do not yet have `inflow_nonnegativity_cost` in their `penalties.json`.
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

    /// Random seed for the opening scenario tree (reproducible training).
    #[serde(default)]
    pub tree_seed: Option<i64>,

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

    /// Row formulation: `"single"` or `"multi"`.
    #[serde(default)]
    pub cut_formulation: Option<String>,

    /// Forward pass configuration.
    #[serde(default)]
    pub forward_pass: Option<ForwardPassConfig>,

    /// Row-selection settings.
    #[serde(default)]
    pub cut_selection: RowSelectionConfig,

    /// LP solver retry settings.
    #[serde(default)]
    pub solver: TrainingSolverConfig,

    /// Scenario source configuration for the training forward pass.
    /// When absent, all classes default to `in_sample`.
    #[serde(default)]
    pub scenario_source: Option<RawScenarioSourceConfig>,
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

/// Row-selection settings (`config.json → training.cut_selection`).
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct RowSelectionConfig {
    /// Enable row pruning.
    #[serde(default)]
    pub enabled: Option<bool>,

    /// Method: `"level1"`, `"lml1"`, or `"domination"`.
    #[serde(default)]
    pub method: Option<String>,

    /// Generic threshold (deprecated — prefer method-specific fields below).
    ///
    /// Interpretation depends on the method:
    /// - `"level1"`: minimum iterations before first pruning pass
    /// - `"lml1"`: memory window size (iterations)
    /// - `"domination"`: epsilon for domination test (integer-limited)
    ///
    /// Use `memory_window` for lml1 and `domination_epsilon` for domination
    /// to avoid the integer limitation. This field is retained for backwards
    /// compatibility.
    #[serde(default, deserialize_with = "deserialize_deprecated_threshold")]
    pub threshold: Option<u32>,

    /// Memory window size for the `"lml1"` method (iterations).
    ///
    /// Overrides `threshold` when the method is `"lml1"`. Ignored for other methods.
    #[serde(default)]
    pub memory_window: Option<u32>,

    /// Epsilon for the `"domination"` method.
    ///
    /// Overrides `threshold` when the method is `"domination"`. Accepts
    /// fractional values (e.g., `1e-6`) unlike the integer-limited `threshold`.
    #[serde(default)]
    pub domination_epsilon: Option<f64>,

    /// Iterations between pruning checks.
    #[serde(default)]
    pub check_frequency: Option<u32>,

    /// Minimum dual multiplier for a row to count as binding.
    #[serde(default)]
    pub cut_activity_tolerance: Option<f64>,

    /// Activity-window size for the basis-reconstruction classifier and
    /// Scheme 1 sort popcount. Bit `i` of `activity_window` counts toward the
    /// classifier and popcount mask when `i < basis_activity_window`.
    ///
    /// Validated range: 1..=31. Default when absent: 5.
    #[serde(default)]
    pub basis_activity_window: Option<u32>,

    /// Maximum number of active rows per stage (stage 2 of the row-selection
    /// pipeline — hard cap on LP size).
    ///
    /// When `Some(n)`, the training loop enforces a hard cap of `n` active rows
    /// per stage after strategy selection has completed. Rows are evicted in
    /// order of staleness (`last_active_iter` ascending), tie-broken by usage
    /// frequency (`active_count` ascending). Rows generated in the current
    /// iteration are never evicted.
    ///
    /// When `None` (the default), no hard cap is enforced.
    #[serde(default)]
    pub max_active_per_stage: Option<u32>,
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

/// Intermediate serde type for per-class scenario source configuration in `config.json`.
///
/// Scoped to `config.json` fields (`training.scenario_source` /
/// `simulation.scenario_source`).
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct RawScenarioSourceConfig {
    /// Optional random seed for reproducible scenario generation.
    #[serde(default)]
    pub seed: Option<i64>,

    /// Historical year pool. Absent means `None` (auto-discover at validation time).
    #[serde(default)]
    pub historical_years: Option<RawHistoricalYearsConfig>,

    /// Inflow class scenario config. Absent defaults to `in_sample`.
    #[serde(default)]
    pub inflow: Option<RawClassConfigEntry>,

    /// Load class scenario config. Absent defaults to `in_sample`.
    #[serde(default)]
    pub load: Option<RawClassConfigEntry>,

    /// NCS class scenario config. Absent defaults to `in_sample`.
    #[serde(default)]
    pub ncs: Option<RawClassConfigEntry>,
}

/// Intermediate serde type for a single per-class scenario scheme in `config.json`.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct RawClassConfigEntry {
    /// Scheme string: `"in_sample"`, `"out_of_sample"`, `"external"`, or `"historical"`.
    pub scheme: String,
}

/// Intermediate serde type for `historical_years` in `config.json`.
///
/// Handles two JSON representations via `#[serde(untagged)]`:
/// - Array: `[1940, 1953, 1971]` → [`RawHistoricalYearsConfig::List`]
/// - Object: `{"from": 1940, "to": 2010}` → [`RawHistoricalYearsConfig::Range`]
///
/// The `List` variant must be declared first so serde tries it before `Range`
/// (an integer array is tried before an object).
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub enum RawHistoricalYearsConfig {
    /// Explicit list of year integers.
    List(Vec<i32>),
    /// Inclusive range shorthand.
    Range {
        /// First year (inclusive).
        from: i32,
        /// Last year (inclusive).
        to: i32,
    },
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

/// Policy initialization mode (`config.json → policy.mode`).
///
/// Controls whether the training phase starts from scratch, warm-starts from
/// a prior policy's rows, or resumes a checkpointed training run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub enum PolicyMode {
    /// Start training from an empty future-cost function.
    Fresh,
    /// Load rows from a prior policy checkpoint and continue training.
    WarmStart,
    /// Resume a previously interrupted training run from its checkpoint.
    Resume,
}

impl std::fmt::Display for PolicyMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PolicyMode::Fresh => f.write_str("fresh"),
            PolicyMode::WarmStart => f.write_str("warm_start"),
            PolicyMode::Resume => f.write_str("resume"),
        }
    }
}

/// Boundary-row configuration for terminal-stage FCF coupling.
///
/// When present, the solver loads rows from a source Cobre policy
/// checkpoint and injects them as fixed boundary conditions at the
/// terminal stage of the current study.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct BoundaryPolicy {
    /// Path to the source policy checkpoint directory.
    pub path: String,
    /// 0-based stage index in the source checkpoint to load rows from.
    pub source_stage: u32,
}

/// Policy directory settings (`config.json → policy`).
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct PolicyConfig {
    /// Directory for policy data (rows, states, vertices, basis).
    pub path: String,

    /// Initialization mode: `"fresh"`, `"warm_start"`, or `"resume"`.
    pub mode: PolicyMode,

    /// Verify state dimension and entity compatibility when loading.
    pub validate_compatibility: bool,

    /// Checkpoint settings.
    pub checkpointing: CheckpointingConfig,

    /// Optional boundary-row policy for terminal-stage coupling.
    #[serde(default)]
    pub boundary: Option<BoundaryPolicy>,
}

impl Default for PolicyConfig {
    fn default() -> Self {
        Self {
            path: "./policy".to_string(),
            mode: PolicyMode::Fresh,
            validate_compatibility: true,
            checkpointing: CheckpointingConfig::default(),
            boundary: None,
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

    /// Policy representation: `"outer"` (envelope rows) or `"inner"` (vertices).
    pub policy_type: String,

    /// Directory for simulation output files.
    pub output_path: Option<String>,

    /// Output mode: `"streaming"` or `"batched"`.
    pub output_mode: Option<String>,

    /// Bounded channel capacity between simulation threads and the I/O writer thread.
    pub io_channel_capacity: u32,

    /// Scenario source configuration for the post-training simulation forward pass.
    /// When absent, falls back to the training scenario source.
    #[serde(default)]
    pub scenario_source: Option<RawScenarioSourceConfig>,
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
            scenario_source: None,
        }
    }
}

/// Order selection criterion for autoregressive model fitting.
///
/// Controls how the lag order is chosen when fitting a time series model.
///
/// The `"fixed"` JSON value is deprecated; it is accepted for backwards
/// compatibility but mapped to `Pacf` at parse time with a warning.
#[derive(Debug, Clone, Serialize, Default)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub enum OrderSelectionMethod {
    /// Periodic Yule-Walker partial autocorrelation method (PACF).
    #[default]
    Pacf,
}

impl<'de> serde::Deserialize<'de> for OrderSelectionMethod {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        match s.as_str() {
            "pacf" => Ok(Self::Pacf),
            "fixed" => {
                tracing::warn!(
                    "OrderSelectionMethod::Fixed is deprecated and will be removed \
                     in a future release. The PACF method is now used for all order \
                     selection. Please update your config.json to use \"pacf\"."
                );
                Ok(Self::Pacf)
            }
            other => Err(serde::de::Error::unknown_variant(other, &["pacf", "fixed"])),
        }
    }
}

/// Deserialize `RowSelectionConfig::threshold`, emitting a deprecation warning
/// when a non-`None` value is present.
///
/// Used as the target of `#[serde(deserialize_with = ...)]` on the `threshold`
/// field. The warning fires once per parse — i.e. once per config load — and
/// mirrors the phrasing used for the deprecated `"fixed"` value of
/// [`OrderSelectionMethod`].
fn deserialize_deprecated_threshold<'de, D>(deserializer: D) -> Result<Option<u32>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value: Option<u32> = Option::deserialize(deserializer)?;
    if value.is_some() {
        tracing::warn!(
            "RowSelectionConfig::threshold is deprecated and will be removed in a \
             future release. Use `memory_window` for the \"lml1\" method and \
             `domination_epsilon` for the \"domination\" method. Please update \
             your config.json."
        );
    }
    Ok(value)
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

    /// Maximum allowed absolute magnitude for any AR coefficient.
    ///
    /// When set, any (entity, season) pair with `|coefficient| > threshold`
    /// is immediately reduced to order 0 before the contribution analysis
    /// runs. This acts as a fast-path safety net for the most extreme
    /// explosive models. Defaults to `None` (disabled; contribution analysis
    /// is the primary guard).
    #[serde(default)]
    pub max_coefficient_magnitude: Option<f64>,
}

impl Default for EstimationConfig {
    fn default() -> Self {
        Self {
            max_order: 6,
            order_selection: OrderSelectionMethod::Pacf,
            min_observations_per_season: 30,
            max_coefficient_magnitude: None,
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

    /// Export row pool (piecewise-linear envelope).
    pub cuts: bool,

    /// Export visited forward-pass trial points to the policy checkpoint.
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
            states: false,
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

// ── ScenarioSource helpers ───────────────────────────────────────────────────

/// Convert a `scheme` string from `config.json` to [`SamplingScheme`].
///
/// `field` is the dot-separated JSON path to the scheme key (e.g.
/// `"training.scenario_source.inflow.scheme"`), used verbatim in the error
/// message so the caller can identify which field has the invalid value.
fn convert_sampling_scheme_cfg(
    s: &str,
    field: &str,
    path: &Path,
) -> Result<SamplingScheme, LoadError> {
    match s {
        "in_sample" => Ok(SamplingScheme::InSample),
        "out_of_sample" => Ok(SamplingScheme::OutOfSample),
        "external" => Ok(SamplingScheme::External),
        "historical" => Ok(SamplingScheme::Historical),
        other => Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: field.to_string(),
            message: format!(
                "unknown scheme '{other}', expected one of: in_sample, out_of_sample, external, historical"
            ),
        }),
    }
}

/// Convert a per-class config entry to its [`SamplingScheme`], defaulting to
/// `in_sample` when the entry is absent.
fn convert_class_scheme_cfg(
    class: Option<&RawClassConfigEntry>,
    section: &str,
    class_name: &str,
    path: &Path,
) -> Result<SamplingScheme, LoadError> {
    convert_sampling_scheme_cfg(
        class.map_or("in_sample", |c| c.scheme.as_str()),
        &format!("{section}.scenario_source.{class_name}.scheme"),
        path,
    )
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

    let inflow_scheme = convert_class_scheme_cfg(r.inflow.as_ref(), section, "inflow", path)?;
    let load_scheme = convert_class_scheme_cfg(r.load.as_ref(), section, "load", path)?;
    let ncs_scheme = convert_class_scheme_cfg(r.ncs.as_ref(), section, "ncs", path)?;

    let source = ScenarioSource {
        inflow_scheme,
        load_scheme,
        ncs_scheme,
        seed: r.seed,
        historical_years: r.historical_years.as_ref().map(|hy| match hy {
            RawHistoricalYearsConfig::List(years) => HistoricalYears::List(years.clone()),
            RawHistoricalYearsConfig::Range { from, to } => HistoricalYears::Range {
                from: *from,
                to: *to,
            },
        }),
    };

    validate_scenario_source_cfg(&source, section, path)?;

    Ok(source)
}

/// Tier-1 structural validation of a parsed [`ScenarioSource`] from `config.json`.
///
/// ## Checks performed
///
/// - `historical_years` must not be specified if no class uses `Historical`.
/// - `seed` is required when any class uses `OutOfSample`, `Historical`, or `External`.
/// - `Historical` scheme is only valid for the `inflow` class.
/// - If `historical_years` is a `Range`, `from` must be `<= to`.
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

    // Historical scheme is only valid for inflow class
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

    // Seed is required unless all classes are InSample
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

    if let Some(HistoricalYears::Range { from, to }) = source.historical_years {
        if from > to {
            return Err(LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!("{section}.scenario_source.historical_years"),
                message: format!("range 'from' ({from}) must be <= 'to' ({to})"),
            });
        }
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
            r#"{"training": {"tree_seed": 42, "forward_passes": 192, "stopping_rules": [{"type": "iteration_limit", "limit": 50}]}}"#,
        );
        let cfg = parse_config(f.path()).unwrap();

        // Mandatory field present and correct
        assert_eq!(cfg.training.forward_passes, Some(192));

        // tree_seed is optional
        assert_eq!(cfg.training.tree_seed, Some(42));

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
        assert_eq!(cfg.policy.mode, PolicyMode::Fresh);
        assert_eq!(cfg.policy.path, "./policy");
        assert!(cfg.policy.validate_compatibility);
        assert!(cfg.exports.training);
        assert!(cfg.exports.cuts);
    }

    /// AC-2: missing `training.forward_passes` → SchemaError with field name.
    #[test]
    fn test_missing_forward_passes() {
        let f = write_config(
            r#"{"training": {"tree_seed": 1, "stopping_rules": [{"type": "iteration_limit", "limit": 10}]}}"#,
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
        let f = write_config(r#"{"training": {"tree_seed": 1, "forward_passes": 100}}"#);
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
            "tree_seed": 42,
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
            "output_mode": "streaming"
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
        assert_eq!(cfg.policy.mode, PolicyMode::Fresh);
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

    /// `Config` has no `version` field — the struct does not
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

    /// JSON with `"$schema"` property is accepted and the field
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

    /// Invalid `policy.mode` values are rejected at parse time.
    #[test]
    fn test_invalid_policy_mode_rejected() {
        let f = write_config(
            r#"{"training": {"forward_passes": 1, "stopping_rules": [{"type": "iteration_limit", "limit": 10}]}, "policy": {"mode": "warmstart"}}"#,
        );
        let err = parse_config(f.path()).unwrap_err();
        assert!(
            matches!(err, LoadError::SchemaError { .. }),
            "expected SchemaError for invalid policy.mode, got: {err:?}"
        );
    }

    /// JSON that still contains a `"version"` property is
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

    /// `"truncation"` is accepted as a method string and
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
            matches!(cfg.estimation.order_selection, OrderSelectionMethod::Pacf),
            "default order_selection should be Pacf"
        );
        assert_eq!(cfg.estimation.min_observations_per_season, 30);
    }

    /// AC-035-2: `"order_selection": "fixed"` deserializes to `Pacf` (deprecated alias).
    #[test]
    fn test_estimation_config_order_selection_fixed_deprecated() {
        let f = write_config(
            r#"{
            "training": {"forward_passes": 10, "stopping_rules": [{"type": "iteration_limit", "limit": 5}]},
            "estimation": {"max_order": 3, "order_selection": "fixed", "min_observations_per_season": 20}
        }"#,
        );
        let cfg = parse_config(f.path()).unwrap();
        assert_eq!(cfg.estimation.max_order, 3);
        assert!(
            matches!(cfg.estimation.order_selection, OrderSelectionMethod::Pacf),
            "deprecated 'fixed' must deserialize to Pacf"
        );
        assert_eq!(cfg.estimation.min_observations_per_season, 20);
    }

    /// AC-035-2b: `"order_selection": "pacf"` deserializes to `Pacf` with no warning.
    #[test]
    fn test_estimation_config_order_selection_pacf() {
        let f = write_config(
            r#"{
            "training": {"forward_passes": 10, "stopping_rules": [{"type": "iteration_limit", "limit": 5}]},
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

    // ── ScenarioSource parsing tests ──────────────────────────────────────────

    const MINIMAL_TRAINING: &str =
        r#"{"forward_passes": 10, "stopping_rules": [{"type": "iteration_limit", "limit": 5}]}"#;

    fn write_with_training_scenario_source(scenario_source_json: &str) -> NamedTempFile {
        write_config(&format!(
            r#"{{"training": {{"forward_passes": 10, "stopping_rules": [{{"type": "iteration_limit", "limit": 5}}], "scenario_source": {scenario_source_json}}}}}"#
        ))
    }

    fn write_with_both_scenario_sources(
        training_json: &str,
        simulation_json: &str,
    ) -> NamedTempFile {
        write_config(&format!(
            r#"{{"training": {{"forward_passes": 10, "stopping_rules": [{{"type": "iteration_limit", "limit": 5}}], "scenario_source": {training_json}}}, "simulation": {{"scenario_source": {simulation_json}}}}}"#
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
        let cfg = parse_config(f.path()).unwrap();
        let err = cfg.simulation_scenario_source(f.path()).unwrap_err();
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
        let cfg = parse_config(f.path()).unwrap();
        let err = cfg.training_scenario_source(f.path()).unwrap_err();
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

    /// OutOfSample without seed → SchemaError.
    #[test]
    fn test_scenario_source_seed_required_for_oos() {
        let f = write_with_training_scenario_source(r#"{"inflow": {"scheme": "out_of_sample"}}"#);
        let cfg = parse_config(f.path()).unwrap();
        let err = cfg.training_scenario_source(f.path()).unwrap_err();
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
        let cfg = parse_config(f.path()).unwrap();
        let err = cfg.training_scenario_source(f.path()).unwrap_err();
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

    /// `simulation.sampling_scheme` (old dead field) is silently ignored since
    /// `SimulationConfig` does not use `deny_unknown_fields`.
    #[test]
    fn test_dead_sampling_scheme_field_removed() {
        // The old `sampling_scheme` key is an unknown field — serde ignores it.
        let f = write_config(
            r#"{
            "training": {"forward_passes": 10, "stopping_rules": [{"type": "iteration_limit", "limit": 5}]},
            "simulation": {"enabled": true, "sampling_scheme": {"type": "in_sample"}}
        }"#,
        );
        let cfg = parse_config(f.path()).unwrap();
        // Parsed successfully — old field silently ignored.
        assert!(cfg.simulation.enabled);
        // The new scenario_source field is absent.
        assert!(cfg.simulation.scenario_source.is_none());
    }

    /// max_active_per_stage serde roundtrip: Some(100) serializes and deserializes correctly.
    #[test]
    fn max_active_per_stage_serde_roundtrip() {
        let original = RowSelectionConfig {
            enabled: Some(true),
            method: Some("level1".to_string()),
            threshold: None,
            memory_window: None,
            domination_epsilon: None,
            check_frequency: None,
            cut_activity_tolerance: None,
            max_active_per_stage: Some(100),
            basis_activity_window: Some(7),
        };
        let json = serde_json::to_string(&original).unwrap();
        let roundtripped: RowSelectionConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(roundtripped.max_active_per_stage, Some(100));
        assert_eq!(roundtripped.enabled, Some(true));
        assert_eq!(roundtripped.method.as_deref(), Some("level1"));
        assert_eq!(roundtripped.basis_activity_window, Some(7));
    }

    /// max_active_per_stage absent from JSON deserializes to None.
    #[test]
    fn max_active_per_stage_absent_defaults_none() {
        let f = write_config(
            r#"{
            "training": {
                "forward_passes": 10,
                "stopping_rules": [{"type": "iteration_limit", "limit": 5}],
                "cut_selection": {"enabled": true, "method": "level1"}
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
                "forward_passes": 10,
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
        assert_eq!(boundary.source_stage, 2);
    }

    /// `policy` without a `boundary` key deserializes to `None`.
    #[test]
    fn test_boundary_policy_absent() {
        let f = write_config(
            r#"{
            "training": {
                "forward_passes": 10,
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
                "forward_passes": 10,
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
            validate_compatibility: true,
            checkpointing: CheckpointingConfig::default(),
            boundary: Some(BoundaryPolicy {
                path: "../monthly/policy".to_string(),
                source_stage: 5,
            }),
        };
        let json = serde_json::to_string(&original).unwrap();
        let restored: PolicyConfig = serde_json::from_str(&json).unwrap();
        let boundary = restored.boundary.unwrap();
        assert_eq!(boundary.path, "../monthly/policy");
        assert_eq!(boundary.source_stage, 5);
    }

    // ── RowSelectionConfig::threshold deprecation warning tests ──────────────

    /// Minimal tracing subscriber that records WARN-level event messages for
    /// use in unit tests. Thread-safe via `Arc<Mutex<Vec<String>>>`.
    mod test_subscriber {
        use std::sync::{Arc, Mutex};
        use tracing::{
            Event, Level, Metadata, Subscriber,
            span::{Attributes, Id, Record},
        };

        pub(super) struct WarnRecorder {
            pub(super) messages: Arc<Mutex<Vec<String>>>,
        }

        impl WarnRecorder {
            pub(super) fn new() -> (Self, Arc<Mutex<Vec<String>>>) {
                let messages = Arc::new(Mutex::new(Vec::new()));
                (
                    Self {
                        messages: Arc::clone(&messages),
                    },
                    messages,
                )
            }
        }

        impl Subscriber for WarnRecorder {
            fn enabled(&self, metadata: &Metadata<'_>) -> bool {
                *metadata.level() <= Level::WARN
            }

            fn new_span(&self, _attrs: &Attributes<'_>) -> Id {
                Id::from_u64(1)
            }

            fn record(&self, _span: &Id, _values: &Record<'_>) {}

            fn record_follows_from(&self, _span: &Id, _follows: &Id) {}

            fn event(&self, event: &Event<'_>) {
                if *event.metadata().level() == Level::WARN {
                    struct MessageVisitor(String);
                    impl tracing::field::Visit for MessageVisitor {
                        fn record_debug(
                            &mut self,
                            field: &tracing::field::Field,
                            value: &dyn std::fmt::Debug,
                        ) {
                            if field.name() == "message" {
                                self.0 = format!("{value:?}");
                            }
                        }
                        fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
                            if field.name() == "message" {
                                self.0 = value.to_string();
                            }
                        }
                    }
                    let mut visitor = MessageVisitor(String::new());
                    event.record(&mut visitor);
                    self.messages.lock().unwrap().push(visitor.0);
                }
            }

            fn enter(&self, _span: &Id) {}

            fn exit(&self, _span: &Id) {}
        }
    }

    /// AC: parsing `RowSelectionConfig` with `threshold: Some(5)` emits exactly
    /// one WARN event whose message contains "threshold" and "deprecated".
    #[test]
    fn test_row_selection_threshold_deprecated_warning() {
        let (subscriber, messages) = test_subscriber::WarnRecorder::new();
        tracing::subscriber::with_default(subscriber, || {
            let json = r#"{"threshold": 5}"#;
            let cfg: RowSelectionConfig = serde_json::from_str(json).unwrap();
            assert_eq!(cfg.threshold, Some(5), "threshold must be stored");
        });
        let recorded = messages.lock().unwrap();
        let warn_events: Vec<&str> = recorded
            .iter()
            .map(std::string::String::as_str)
            .filter(|msg| msg.contains("threshold") && msg.contains("deprecated"))
            .collect();
        assert!(
            !warn_events.is_empty(),
            "expected at least one WARN event containing 'threshold' and 'deprecated', got: {recorded:?}"
        );
    }

    /// AC: parsing `RowSelectionConfig` without `threshold` emits no WARN event
    /// from this code path.
    #[test]
    fn test_row_selection_threshold_absent_no_warning() {
        let (subscriber, messages) = test_subscriber::WarnRecorder::new();
        tracing::subscriber::with_default(subscriber, || {
            let json = r#"{"enabled": true, "method": "lml1", "memory_window": 10}"#;
            let cfg: RowSelectionConfig = serde_json::from_str(json).unwrap();
            assert!(
                cfg.threshold.is_none(),
                "threshold must be None when absent"
            );
        });
        let recorded = messages.lock().unwrap();
        let threshold_warns: Vec<&str> = recorded
            .iter()
            .map(std::string::String::as_str)
            .filter(|msg| msg.contains("threshold") && msg.contains("deprecated"))
            .collect();
        assert!(
            threshold_warns.is_empty(),
            "expected no WARN events about threshold deprecation when field is absent, got: {threshold_warns:?}"
        );
    }
}
