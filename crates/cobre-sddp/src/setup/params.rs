//! `StudyParams`, `ConstructionConfig`, and associated constants extracted from `setup/mod.rs`.

use cobre_io::config::StoppingRuleConfig;

use crate::{
    InflowNonNegativityMethod, SddpError,
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
    /// Activity-window size for the basis-reconstruction classifier.
    /// Validated range 1..=31. Default: [`crate::basis_reconstruct::DEFAULT_BASIS_ACTIVITY_WINDOW`].
    pub basis_activity_window: u32,
    /// Maximum number of active cuts per stage (hard cap on LP size).
    ///
    /// `None` means no cap is enforced. Derived from
    /// `config.training.cut_selection.max_active_per_stage`.
    pub budget: Option<u32>,
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

        let basis_activity_window = config
            .training
            .cut_selection
            .basis_activity_window
            .unwrap_or(crate::basis_reconstruct::DEFAULT_BASIS_ACTIVITY_WINDOW);
        if !(1..=31).contains(&basis_activity_window) {
            return Err(SddpError::Validation(format!(
                "basis_activity_window must be in 1..=31, got {basis_activity_window}"
            )));
        }

        let budget = config.training.cut_selection.max_active_per_stage;

        // Warn when the budget is so tight that every iteration will immediately
        // evict all cuts older than the current one.  This is not an error —
        // the solver remains correct — but it usually indicates a misconfiguration.
        if let Some(b) = budget {
            // world_size is not available here; use 1 as a conservative estimate.
            // The CLI/Python layer may emit a more precise warning with the real
            // world_size after broadcast.
            if u64::from(b) < u64::from(forward_passes) {
                tracing::warn!(
                    "max_active_per_stage ({b}) is less than forward_passes \
                     ({forward_passes}); budget enforcement will evict all \
                     non-current-iteration cuts every iteration"
                );
            }
        }

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
            basis_activity_window,
            budget,
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
            basis_activity_window: self.basis_activity_window,
            budget: self.budget,
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
    /// Activity-window size for the basis-reconstruction classifier.
    ///
    /// Validated range 1..=31. Default:
    /// [`crate::basis_reconstruct::DEFAULT_BASIS_ACTIVITY_WINDOW`].
    pub basis_activity_window: u32,
    /// Maximum number of active cuts per stage (hard cap on LP size).
    ///
    /// `None` means no cap is enforced. Derived from
    /// `config.training.cut_selection.max_active_per_stage`.
    pub budget: Option<u32>,
    /// Whether the caller wants the visited-states archive for export.
    ///
    /// When `true`, the archive is allocated during training regardless of the
    /// cut selection strategy. Defaults to `false`; set based on
    /// `exports.states`.
    pub export_states: bool,
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::sync::{Arc, Mutex};

    use cobre_io::config::{
        Config, EstimationConfig, ExportsConfig, InflowNonNegativityConfig, ModelingConfig,
        PolicyConfig, RowSelectionConfig, SimulationConfig as IoSimulationConfig,
        StoppingRuleConfig, TrainingConfig, TrainingSolverConfig, UpperBoundEvaluationConfig,
    };
    use tracing::{Event, Level, Metadata, Subscriber, span};

    use super::StudyParams;

    // ---------------------------------------------------------------------------
    // Minimal WARN-capturing subscriber for use in tests.
    // ---------------------------------------------------------------------------

    /// Records all WARN-level event messages into a shared `Vec<String>`.
    struct WarnRecorder {
        messages: Arc<Mutex<Vec<String>>>,
    }

    impl WarnRecorder {
        fn new() -> (Self, Arc<Mutex<Vec<String>>>) {
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

        fn new_span(&self, _attrs: &span::Attributes<'_>) -> span::Id {
            span::Id::from_u64(1)
        }

        fn record(&self, _span: &span::Id, _values: &span::Record<'_>) {}

        fn record_follows_from(&self, _span: &span::Id, _follows: &span::Id) {}

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

        fn enter(&self, _span: &span::Id) {}

        fn exit(&self, _span: &span::Id) {}
    }

    /// Build a minimal `cobre_io::Config` with the given
    /// `basis_activity_window` value in `training.cut_selection`.
    fn config_with_window(window: Option<u32>) -> Config {
        Config {
            schema: None,
            modeling: ModelingConfig {
                inflow_non_negativity: InflowNonNegativityConfig {
                    method: "penalty".to_string(),
                    penalty_cost: 1000.0,
                },
            },
            training: TrainingConfig {
                enabled: true,
                tree_seed: Some(42),
                forward_passes: Some(1),
                stopping_rules: Some(vec![StoppingRuleConfig::IterationLimit { limit: 1 }]),
                stopping_mode: "any".to_string(),
                cut_formulation: None,
                forward_pass: None,
                cut_selection: RowSelectionConfig {
                    basis_activity_window: window,
                    ..RowSelectionConfig::default()
                },
                solver: TrainingSolverConfig::default(),
                scenario_source: None,
            },
            upper_bound_evaluation: UpperBoundEvaluationConfig::default(),
            policy: PolicyConfig::default(),
            simulation: IoSimulationConfig::default(),
            exports: ExportsConfig::default(),
            estimation: EstimationConfig::default(),
        }
    }

    #[test]
    fn study_params_rejects_basis_activity_window_out_of_range() {
        // Value 0 must be rejected.
        let err = StudyParams::from_config(&config_with_window(Some(0)))
            .expect_err("window=0 must be rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("basis_activity_window"),
            "error must mention field name; got: {msg}"
        );

        // Value 32 must be rejected.
        let err = StudyParams::from_config(&config_with_window(Some(32)))
            .expect_err("window=32 must be rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("basis_activity_window"),
            "error must mention field name; got: {msg}"
        );

        // None defaults to 5.
        let params =
            StudyParams::from_config(&config_with_window(None)).expect("None must succeed");
        assert_eq!(
            params.basis_activity_window,
            crate::basis_reconstruct::DEFAULT_BASIS_ACTIVITY_WINDOW,
            "None must default to DEFAULT_BASIS_ACTIVITY_WINDOW"
        );

        // Explicit 5 must pass.
        let params =
            StudyParams::from_config(&config_with_window(Some(5))).expect("window=5 must succeed");
        assert_eq!(params.basis_activity_window, 5);

        // Boundary values 1 and 31 must pass.
        let params =
            StudyParams::from_config(&config_with_window(Some(1))).expect("window=1 must succeed");
        assert_eq!(params.basis_activity_window, 1);

        let params = StudyParams::from_config(&config_with_window(Some(31)))
            .expect("window=31 must succeed");
        assert_eq!(params.basis_activity_window, 31);
    }

    /// Build a minimal `cobre_io::Config` with `max_active_per_stage` and
    /// `forward_passes` set so that the budget-below-forward-passes warning fires.
    fn config_with_budget_below_forward_passes() -> Config {
        Config {
            schema: None,
            modeling: ModelingConfig {
                inflow_non_negativity: InflowNonNegativityConfig {
                    method: "penalty".to_string(),
                    penalty_cost: 1000.0,
                },
            },
            training: TrainingConfig {
                enabled: true,
                tree_seed: Some(42),
                forward_passes: Some(2),
                stopping_rules: Some(vec![StoppingRuleConfig::IterationLimit { limit: 1 }]),
                stopping_mode: "any".to_string(),
                cut_formulation: None,
                forward_pass: None,
                cut_selection: RowSelectionConfig {
                    max_active_per_stage: Some(1),
                    ..RowSelectionConfig::default()
                },
                solver: TrainingSolverConfig::default(),
                scenario_source: None,
            },
            upper_bound_evaluation: UpperBoundEvaluationConfig::default(),
            policy: PolicyConfig::default(),
            simulation: IoSimulationConfig::default(),
            exports: ExportsConfig::default(),
            estimation: EstimationConfig::default(),
        }
    }

    /// AC: when `max_active_per_stage` is less than `forward_passes`, `StudyParams::from_config`
    /// emits a WARN-level tracing event whose message contains `max_active_per_stage`.
    #[test]
    fn study_params_warns_when_budget_below_forward_passes() {
        let (subscriber, messages) = WarnRecorder::new();
        tracing::subscriber::with_default(subscriber, || {
            let _params = StudyParams::from_config(&config_with_budget_below_forward_passes())
                .expect("config is valid; warning must not prevent construction");
        });
        let recorded = messages.lock().unwrap();
        let relevant: Vec<&str> = recorded
            .iter()
            .map(std::string::String::as_str)
            .filter(|msg| msg.contains("max_active_per_stage"))
            .collect();
        assert!(
            !relevant.is_empty(),
            "expected at least one WARN event containing 'max_active_per_stage', got: {recorded:?}"
        );
    }
}
