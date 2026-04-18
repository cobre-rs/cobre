//! `StudyParams`, `ConstructionConfig`, and associated constants extracted from `setup/mod.rs`.

use cobre_io::config::StoppingRuleConfig;

use crate::{
    CanonicalStateStrategy, InflowNonNegativityMethod, SddpError,
    cut_selection::{CutSelectionStrategy, parse_cut_selection_config},
    stopping_rule::{StoppingMode, StoppingRule, StoppingRuleSet},
};

/// Default number of forward-pass trajectories when not specified in config.
pub const DEFAULT_FORWARD_PASSES: u32 = 1;

/// Default maximum iterations when no stopping rule specifies an iteration limit.
pub const DEFAULT_MAX_ITERATIONS: u64 = 100;

/// Default random seed for stochastic scenario generation.
pub const DEFAULT_SEED: u64 = 42;

// ---------------------------------------------------------------------------
// StudyParams
// ---------------------------------------------------------------------------

/// Scalar parameters extracted from a [`cobre_io::Config`].
///
/// Centralises config-to-domain conversion for both [`StudySetup::new`](super::StudySetup::new)
/// and `BroadcastConfig::from_config`. The struct owns all
/// values so it can be passed by value without lifetime dependencies.
#[derive(Debug, Clone)]
pub struct StudyParams {
    /// Random seed for noise generation.
    pub seed: u64,
    /// Number of forward-pass trajectories per training iteration.
    pub forward_passes: u32,
    /// Stopping rule set (rules + mode) governing when training halts.
    pub stopping_rule_set: StoppingRuleSet,
    /// Number of simulation scenarios (0 if simulation is disabled).
    pub n_scenarios: u32,
    /// Buffer capacity for the simulation output channel.
    pub io_channel_capacity: usize,
    /// Policy directory path string.
    pub policy_path: String,
    /// Inflow non-negativity enforcement method.
    pub inflow_method: InflowNonNegativityMethod,
    /// Optional cut selection strategy (None means cut selection is disabled).
    pub cut_selection: Option<CutSelectionStrategy>,
    /// Minimum dual multiplier for a cut to count as binding (`0.0` if unset).
    pub cut_activity_tolerance: f64,
    /// Maximum number of active cuts per stage (hard cap on LP size).
    ///
    /// `None` means no cap is enforced. Derived from
    /// `config.training.cut_selection.max_active_per_stage`.
    pub budget: Option<u32>,
    /// Canonical-state strategy for the backward pass.
    ///
    /// Parsed from `config.training.solver.canonical_state`. Defaults to
    /// `Disabled` when the key is absent from the config file.
    pub canonical_state_strategy: CanonicalStateStrategy,
}

impl StudyParams {
    /// Extract study parameters from a validated [`cobre_io::Config`].
    ///
    /// # Errors
    ///
    /// - [`SddpError::Validation`] if cut selection config is invalid.
    pub fn from_config(config: &cobre_io::Config) -> Result<Self, SddpError> {
        let seed = config
            .training
            .tree_seed
            .map_or(DEFAULT_SEED, i64::unsigned_abs);

        let forward_passes = config
            .training
            .forward_passes
            .unwrap_or(DEFAULT_FORWARD_PASSES);

        let rule_configs = match &config.training.stopping_rules {
            Some(rules) if !rules.is_empty() => rules.clone(),
            _ => vec![StoppingRuleConfig::IterationLimit {
                limit: u32::try_from(DEFAULT_MAX_ITERATIONS).unwrap_or(u32::MAX),
            }],
        };

        let stopping_rules: Vec<StoppingRule> = rule_configs
            .into_iter()
            .map(|c| match c {
                StoppingRuleConfig::IterationLimit { limit } => StoppingRule::IterationLimit {
                    limit: u64::from(limit),
                },
                StoppingRuleConfig::TimeLimit { seconds } => StoppingRule::TimeLimit { seconds },
                StoppingRuleConfig::BoundStalling {
                    iterations,
                    tolerance,
                } => StoppingRule::BoundStalling {
                    iterations: u64::from(iterations),
                    tolerance,
                },
                StoppingRuleConfig::Simulation { .. } => {
                    // Not implemented in the minimal viable solver; fold into
                    // an iteration limit so the stopping rule set is valid.
                    StoppingRule::IterationLimit {
                        limit: DEFAULT_MAX_ITERATIONS,
                    }
                }
            })
            .collect();

        let stopping_mode = if config.training.stopping_mode.eq_ignore_ascii_case("all") {
            StoppingMode::All
        } else {
            StoppingMode::Any
        };

        let stopping_rule_set = StoppingRuleSet {
            rules: stopping_rules,
            mode: stopping_mode,
        };

        let n_scenarios = if config.simulation.enabled {
            config.simulation.num_scenarios
        } else {
            0
        };

        let io_channel_capacity =
            usize::try_from(config.simulation.io_channel_capacity).unwrap_or(64);

        let policy_path = config.policy.path.clone();

        let inflow_method = InflowNonNegativityMethod::from(&config.modeling.inflow_non_negativity);

        let cut_selection = parse_cut_selection_config(&config.training.cut_selection)
            .map_err(|msg| SddpError::Validation(format!("cut_selection config error: {msg}")))?;

        let cut_activity_tolerance = config
            .training
            .cut_selection
            .cut_activity_tolerance
            .unwrap_or(0.0);

        let budget = config.training.cut_selection.max_active_per_stage;

        // Warn when the budget is so tight that every iteration will immediately
        // evict all cuts older than the current one.  This is not an error —
        // the solver remains correct — but it usually indicates a misconfiguration.
        if let Some(b) = budget {
            // world_size is not available here; use 1 as a conservative estimate.
            // The CLI/Python layer may emit a more precise warning with the real
            // world_size after broadcast.
            if u64::from(b) < u64::from(forward_passes) {
                eprintln!(
                    "warning: max_active_per_stage ({b}) is less than forward_passes \
                     ({forward_passes}); budget enforcement will evict all \
                     non-current-iteration cuts every iteration"
                );
            }
        }

        let canonical_state_strategy = match config.training.solver.canonical_state.as_str() {
            "disabled" => CanonicalStateStrategy::Disabled,
            "clear_solver" => CanonicalStateStrategy::ClearSolver,
            other => {
                return Err(SddpError::Validation(format!(
                    "invalid canonical_state: '{other}'. Expected 'disabled' or 'clear_solver'."
                )));
            }
        };

        Ok(Self {
            seed,
            forward_passes,
            stopping_rule_set,
            n_scenarios,
            io_channel_capacity,
            policy_path,
            inflow_method,
            cut_selection,
            cut_activity_tolerance,
            budget,
            canonical_state_strategy,
        })
    }

    /// Convert into a [`ConstructionConfig`] for [`StudySetup::from_broadcast_params`](super::StudySetup::from_broadcast_params).
    ///
    /// Sets `export_states = false`; callers should use
    /// [`StudySetup::set_export_states`](super::StudySetup::set_export_states) to enable state export after construction.
    #[must_use]
    pub fn into_construction_config(self) -> ConstructionConfig {
        ConstructionConfig {
            seed: self.seed,
            forward_passes: self.forward_passes,
            stopping_rule_set: self.stopping_rule_set,
            n_scenarios: self.n_scenarios,
            io_channel_capacity: self.io_channel_capacity,
            policy_path: self.policy_path,
            inflow_method: self.inflow_method,
            cut_selection: self.cut_selection,
            cut_activity_tolerance: self.cut_activity_tolerance,
            budget: self.budget,
            canonical_state_strategy: self.canonical_state_strategy,
            export_states: false,
        }
    }
}

// ---------------------------------------------------------------------------
// ConstructionConfig
// ---------------------------------------------------------------------------

/// Scalar and config parameters bundled for [`StudySetup::from_broadcast_params`](super::StudySetup::from_broadcast_params).
///
/// Groups parameters to reduce argument count. Construct via
/// [`StudyParams::into_construction_config`] from a [`cobre_io::Config`],
/// or populate fields directly from a broadcast config.
#[derive(Debug, Clone)]
pub struct ConstructionConfig {
    /// Random seed for noise generation.
    pub seed: u64,
    /// Number of forward-pass trajectories per training iteration.
    pub forward_passes: u32,
    /// Stopping rule set (rules + mode) governing when training halts.
    pub stopping_rule_set: StoppingRuleSet,
    /// Number of simulation scenarios (0 if simulation is disabled).
    pub n_scenarios: u32,
    /// Buffer capacity for the simulation output channel.
    pub io_channel_capacity: usize,
    /// Policy directory path string.
    pub policy_path: String,
    /// Inflow non-negativity enforcement method.
    pub inflow_method: InflowNonNegativityMethod,
    /// Optional cut selection strategy (`None` means cut selection is disabled).
    pub cut_selection: Option<CutSelectionStrategy>,
    /// Minimum dual multiplier for a cut to count as binding (`0.0` if unset).
    pub cut_activity_tolerance: f64,
    /// Maximum number of active cuts per stage (hard cap on LP size).
    ///
    /// `None` means no cap is enforced. Derived from
    /// `config.training.cut_selection.max_active_per_stage`.
    pub budget: Option<u32>,
    /// Canonical-state strategy for the backward pass.
    ///
    /// Propagated from [`StudyParams::canonical_state_strategy`]. Defaults to
    /// `Disabled` when the config key is absent.
    pub canonical_state_strategy: CanonicalStateStrategy,
    /// Whether the caller wants the visited-states archive for export.
    ///
    /// When `true`, the archive is allocated during training regardless of the
    /// cut selection strategy. Defaults to `false`; set based on
    /// `exports.states`.
    pub export_states: bool,
}

#[cfg(test)]
mod tests {
    use cobre_io::config::{
        Config, CutSelectionConfig, EstimationConfig, ExportsConfig, ModelingConfig, PolicyConfig,
        SimulationConfig, StoppingRuleConfig, TrainingConfig, TrainingSolverConfig,
        UpperBoundEvaluationConfig,
    };

    use super::StudyParams;
    use crate::CanonicalStateStrategy;

    fn minimal_config(canonical_state: &str) -> Config {
        Config {
            schema: None,
            modeling: ModelingConfig::default(),
            training: TrainingConfig {
                enabled: true,
                tree_seed: None,
                forward_passes: Some(1),
                stopping_rules: Some(vec![StoppingRuleConfig::IterationLimit { limit: 1 }]),
                stopping_mode: "any".to_string(),
                cut_formulation: None,
                forward_pass: None,
                cut_selection: CutSelectionConfig::default(),
                solver: TrainingSolverConfig {
                    canonical_state: canonical_state.to_string(),
                    ..TrainingSolverConfig::default()
                },
                scenario_source: None,
            },
            upper_bound_evaluation: UpperBoundEvaluationConfig::default(),
            policy: PolicyConfig::default(),
            simulation: SimulationConfig {
                enabled: false,
                ..SimulationConfig::default()
            },
            exports: ExportsConfig::default(),
            estimation: EstimationConfig::default(),
        }
    }

    #[test]
    fn canonical_state_disabled_parses_to_disabled() {
        let config = minimal_config("disabled");
        let params = StudyParams::from_config(&config).unwrap();
        assert_eq!(
            params.canonical_state_strategy,
            CanonicalStateStrategy::Disabled
        );
    }

    #[test]
    fn canonical_state_clear_solver_parses_to_clear_solver() {
        let config = minimal_config("clear_solver");
        let params = StudyParams::from_config(&config).unwrap();
        assert_eq!(
            params.canonical_state_strategy,
            CanonicalStateStrategy::ClearSolver
        );
    }

    #[test]
    fn canonical_state_missing_key_defaults_to_disabled() {
        // TrainingSolverConfig::default() sets canonical_state = "disabled",
        // which mirrors the serde #[serde(default)] behaviour for absent JSON keys.
        let config = minimal_config("disabled");
        let params = StudyParams::from_config(&config).unwrap();
        assert_eq!(
            params.canonical_state_strategy,
            CanonicalStateStrategy::Disabled
        );
    }

    #[test]
    fn canonical_state_invalid_value_returns_validation_error() {
        let config = minimal_config("foo");
        let err = StudyParams::from_config(&config).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("canonical_state"),
            "error message must mention 'canonical_state': {msg}"
        );
        assert!(
            msg.contains("disabled") || msg.contains("clear_solver"),
            "error message must list valid alternatives: {msg}"
        );
    }
}
