//! `StudyParams` and associated constants.

use std::path::Path;

use cobre_core::ScalarParameter;
use cobre_io::Config;
use cobre_io::config::{
    BackwardScheduler, CheckpointSchedule, ForwardPassesResolution, NumScenariosResolution,
    PhaseSolverProfileConfig, StoppingRuleConfig,
};
use serde::{Deserialize, Serialize};

use crate::{
    InflowNonNegativityMethod, SddpError,
    cut_selection::{CutSelectionStrategy, parse_cut_selection_config},
    stopping_rule::{StoppingMode, StoppingRule, StoppingRuleSet},
};

/// Simulation's `enumerated`-selection declaration, carried on
/// [`StudyParams`] until the node graph resolves
/// [`StudyParams::n_scenarios`] (config load holds no graph to derive the
/// count from). A plain externally-tagged enum carrying no
/// `#[serde(tag = ...)]`, so it round-trips over the MPI broadcast wire
/// without a postcard mirror type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SimulationEnumeratedRequest {
    /// `sampled` selection (or the default); [`StudyParams::n_scenarios`] is
    /// already final.
    Sampled,
    /// `enumerated` selection declared; the count is graph-derived downstream.
    Enumerated,
}

/// Default number of forward-pass trajectories when not specified in config.
pub const DEFAULT_FORWARD_PASSES: u32 = 1;

/// Default random seed for stochastic scenario generation.
pub const DEFAULT_SEED: u64 = 42;

/// Default `modeling.cost_scale_factor` when absent from config — the value
/// every golden parity baseline is pinned at.
pub const DEFAULT_COST_SCALE_FACTOR: f64 = 1_000_000.0;

/// Advisory range for `modeling.cost_scale_factor`; values outside this range
/// are accepted but logged via `tracing::warn!`.
const COST_SCALE_FACTOR_ADVISORY_MIN: f64 = 1.0;
const COST_SCALE_FACTOR_ADVISORY_MAX: f64 = 1e12;

// ---------------------------------------------------------------------------
// BoundaryStateRequirements
// ---------------------------------------------------------------------------

/// The boundary-derived setup facts a study needs to size its state space,
/// resolved once from `config.policy.boundary` (and the source checkpoint) by
/// [`resolve_boundary_state_requirements`](crate::resolve_boundary_state_requirements)
/// and threaded onto the config-projection carriers as one value. Both facts are
/// derived together by that resolver — the single owner — so [`Self::is_present`]
/// and [`Self::inflow_lag_depth`] cannot disagree. A new externally-authored
/// boundary state family adds a field and accessor here, never a new scalar
/// threaded through the carriers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundaryStateRequirements {
    present: bool,
    inflow_lag_depth: Option<u32>,
}

impl BoundaryStateRequirements {
    /// No boundary policy: the study reserves nothing on its behalf.
    #[must_use]
    pub fn none() -> Self {
        Self {
            present: false,
            inflow_lag_depth: None,
        }
    }

    /// A boundary policy is present, requiring `required_lag_depth` inflow-lag
    /// slots; `0` reserves none (an all-storage source policy).
    #[must_use]
    pub fn present(required_lag_depth: u32) -> Self {
        Self {
            present: true,
            inflow_lag_depth: (required_lag_depth > 0).then_some(required_lag_depth),
        }
    }

    /// Whether the study declares a terminal boundary FCF
    /// (`config.policy.boundary.is_some()`).
    #[must_use]
    pub fn is_present(&self) -> bool {
        self.present
    }

    /// Effective inflow-lag state depth the loaded boundary requires (`None` when
    /// no boundary is loaded or it carries no inflow-lag slot). Widens `L_state`
    /// in `resolve_state_layout` via `widen_lag_state_depth`.
    #[must_use]
    pub fn inflow_lag_depth(&self) -> Option<u32> {
        self.inflow_lag_depth
    }
}

// ---------------------------------------------------------------------------
// StudyParams
// ---------------------------------------------------------------------------

/// Scalar parameters extracted from a [`Config`].
///
/// Centralises config-to-domain conversion for both [`StudySetup::new`](super::StudySetup::new)
/// and `BroadcastConfig::from_config`. The struct owns all
/// values so it can be passed by value without lifetime dependencies.
#[derive(Debug, Clone)]
pub struct StudyParams {
    /// Random seed for noise generation.
    pub seed: u64,
    /// Number of forward-pass trajectories per training iteration. A
    /// placeholder ([`DEFAULT_FORWARD_PASSES`]) when [`Self::training_enumerated`]
    /// is `true`, until the node graph resolves the derived count.
    pub forward_passes: u32,
    /// `true` when `training.selection = enumerated` is declared — the setup
    /// layer re-resolves [`Self::forward_passes`] from the node graph once it
    /// exists (config load holds no graph to derive the count from).
    pub training_enumerated: bool,
    /// Stopping rule set (rules + mode) governing when training halts.
    pub stopping_rule_set: StoppingRuleSet,
    /// Number of simulation scenarios (0 if simulation is disabled, or a
    /// placeholder while [`Self::simulation_enumerated`] is
    /// [`SimulationEnumeratedRequest::Enumerated`], until the node graph
    /// resolves the derived count).
    pub n_scenarios: u32,
    /// `simulation.selection`'s resolution — the setup layer re-resolves
    /// [`Self::n_scenarios`] from the node graph once it exists when this is
    /// [`SimulationEnumeratedRequest::Enumerated`].
    pub simulation_enumerated: SimulationEnumeratedRequest,
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
    /// Backward-pass solver profile override (`training.solver.backward`).
    pub training_solver_backward: Option<PhaseSolverProfileConfig>,
    /// Forward-pass solver profile override (`training.solver.forward`).
    pub training_solver_forward: Option<PhaseSolverProfileConfig>,
    /// Simulation solver profile override (`simulation.solver`).
    pub simulation_solver: Option<PhaseSolverProfileConfig>,
    /// Backward-pass scheduler (`training.parallelism.backward_scheduler`),
    /// carrying the opening-block size when the `by_node` method is
    /// selected.
    pub backward_scheduler: BackwardScheduler,
    /// Resolved objective cost-scale factor (`modeling.cost_scale_factor`,
    /// default [`DEFAULT_COST_SCALE_FACTOR`]). Baked into the template at build
    /// time — one value per study.
    pub cost_scale_factor: f64,
    /// Boundary-derived state-space requirements. `none()` out of
    /// [`Self::from_config`]; the resolved value is patched by
    /// [`StudySetup::new_with_boundary_requirements`](super::StudySetup::new_with_boundary_requirements)
    /// (local) or the broadcast carrier (MPI), before any consumer reads it.
    pub boundary: BoundaryStateRequirements,
    /// Whether the visited-states archive is allocated for export
    /// (`config.exports.states`). Overridable post-construction via
    /// [`StudySetup::set_export_states`](super::StudySetup::set_export_states).
    pub export_states: bool,
    /// `policy.checkpointing`'s resolved schedule; `None` when periodic
    /// checkpointing is off.
    pub checkpoint_schedule: Option<CheckpointSchedule>,
    /// Caller-supplied scalar parameters from `constraints/generic_parameters.json`
    /// to ensure boundary-configured studies cannot build against an empty table by omission.
    pub scalar_parameters: Vec<ScalarParameter>,
}

impl StudyParams {
    /// Extract study parameters from a validated [`Config`] plus the
    /// caller-loaded scalar-parameter table.
    ///
    /// # Errors
    ///
    /// - [`SddpError::Validation`] if cut selection config is invalid.
    /// - [`SddpError::Io`] if `policy.checkpointing` is enabled without an
    ///   interval of at least 1.
    pub fn from_config(
        config: &Config,
        scalar_parameters: Vec<ScalarParameter>,
    ) -> Result<Self, SddpError> {
        let seed = config
            .training
            .tree_seed
            .map_or(DEFAULT_SEED, i64::unsigned_abs);

        let (forward_passes, training_enumerated) = match config.resolve_forward_passes() {
            Some(ForwardPassesResolution::Sampled(n)) => (n, false),
            None => (DEFAULT_FORWARD_PASSES, false),
            // The node graph does not exist yet; DEFAULT_FORWARD_PASSES is a
            // placeholder `from_broadcast_params` overwrites once it does.
            Some(ForwardPassesResolution::Enumerated) => (DEFAULT_FORWARD_PASSES, true),
        };

        let rule_configs = config.training.stopping_rules.clone().unwrap_or_default();

        let stopping_rules: Vec<StoppingRule> = rule_configs
            .into_iter()
            .map(|c| match c {
                StoppingRuleConfig::IterationLimit { limit } => Ok(StoppingRule::IterationLimit {
                    limit: u64::from(limit),
                }),
                StoppingRuleConfig::TimeLimit { seconds } => {
                    Ok(StoppingRule::TimeLimit { seconds })
                }
                StoppingRuleConfig::BoundStalling {
                    iterations,
                    tolerance,
                } => Ok(StoppingRule::BoundStalling {
                    iterations: u64::from(iterations),
                    tolerance,
                }),
                StoppingRuleConfig::Gap {
                    tolerance,
                    relative_tolerance,
                } => {
                    if tolerance.is_none() && relative_tolerance.is_none() {
                        return Err(SddpError::Validation(
                            "gap stopping rule requires at least one of tolerance / \
                             relative_tolerance to be present"
                                .to_string(),
                        ));
                    }
                    Ok(StoppingRule::Gap {
                        tolerance,
                        relative_tolerance,
                    })
                }
            })
            .collect::<Result<Vec<_>, _>>()?;

        let stopping_mode = match config.training.stopping_mode {
            cobre_io::config::StoppingMode::Any => StoppingMode::Any,
            cobre_io::config::StoppingMode::All => StoppingMode::All,
        };

        let stopping_rule_set = StoppingRuleSet {
            rules: stopping_rules,
            mode: stopping_mode,
        };

        let (n_scenarios, simulation_enumerated) = if config.simulation.enabled {
            match config.resolve_num_scenarios() {
                NumScenariosResolution::Sampled(n) => (n, SimulationEnumeratedRequest::Sampled),
                // The node graph does not exist yet; 0 is a placeholder
                // `from_broadcast_params` overwrites once it does.
                NumScenariosResolution::Enumerated => (0, SimulationEnumeratedRequest::Enumerated),
            }
        } else {
            (0, SimulationEnumeratedRequest::Sampled)
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
            .row_activity_tolerance
            .unwrap_or(0.0);

        let budget = config.training.cut_selection.max_active_per_stage;

        let training_solver_backward = config.training.solver.backward.clone();
        let training_solver_forward = config.training.solver.forward.clone();
        let simulation_solver = config.simulation.solver.clone();
        let backward_scheduler = config.training.parallelism.backward_scheduler;

        if let Some(b) = budget
            && u64::from(b) < u64::from(forward_passes)
        {
            tracing::warn!(
                "max_active_per_stage ({b}) is less than forward_passes \
                 ({forward_passes}); budget enforcement will evict all \
                 non-current-iteration cuts every iteration"
            );
        }

        let cost_scale_factor = config
            .modeling
            .cost_scale_factor
            .unwrap_or(DEFAULT_COST_SCALE_FACTOR);
        if !cost_scale_factor.is_finite() || cost_scale_factor <= 0.0 {
            return Err(SddpError::Validation(format!(
                "modeling.cost_scale_factor ({cost_scale_factor}) must be finite and > 0"
            )));
        }
        if !(COST_SCALE_FACTOR_ADVISORY_MIN..=COST_SCALE_FACTOR_ADVISORY_MAX)
            .contains(&cost_scale_factor)
        {
            tracing::warn!(
                "modeling.cost_scale_factor ({cost_scale_factor}) is outside the advisory \
                 range [{COST_SCALE_FACTOR_ADVISORY_MIN}, {COST_SCALE_FACTOR_ADVISORY_MAX}]; \
                 this only affects LP conditioning, not results in exact arithmetic, but an \
                 extreme value may change how many economically-meaningless pivots run \
                 before optimality"
            );
        }

        Ok(Self {
            seed,
            forward_passes,
            training_enumerated,
            stopping_rule_set,
            n_scenarios,
            simulation_enumerated,
            io_channel_capacity,
            policy_path,
            inflow_method,
            cut_selection,
            cut_activity_tolerance,
            budget,
            training_solver_backward,
            training_solver_forward,
            simulation_solver,
            backward_scheduler,
            cost_scale_factor,
            // resolve_boundary_state_requirements owns both facts; no config knob
            // feeds them, so from_config leaves the placeholder for the caller to patch.
            boundary: BoundaryStateRequirements::none(),
            export_states: config.exports.states,
            checkpoint_schedule: config.checkpoint_schedule(Path::new("config.json"))?,
            scalar_parameters,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use cobre_core::{EntityId, ParameterKind, ScalarParameter};
    use cobre_io::config::{
        Config, EstimationConfig, ExportsConfig, InflowNonNegativityConfig,
        InflowNonNegativityMethod as CfgInflowMethod, ModelingConfig, ParallelismConfig,
        PolicyConfig, RowSelectionConfig, SimulationConfig as IoSimulationConfig,
        SimulationSelection, StoppingMode, StoppingRuleConfig, TrainingConfig, TrainingSelection,
        TrainingSolverConfig, UpperBoundEvaluationConfig,
    };
    use tracing::{Event, Level, Metadata, Subscriber, span};

    use super::{BoundaryStateRequirements, StudyParams};

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

    /// Minimal valid `Config` the `config_with_*` fixtures below mutate: one
    /// forward pass, an iteration-limit stopping rule, every other section
    /// default.
    fn base_test_config() -> Config {
        Config {
            schema: None,
            modeling: ModelingConfig {
                inflow_non_negativity: InflowNonNegativityConfig {
                    method: CfgInflowMethod::Penalty,
                },
                cost_scale_factor: None,
            },
            training: TrainingConfig {
                enabled: true,
                tree_seed: Some(42),
                stopping_rules: Some(vec![StoppingRuleConfig::IterationLimit { limit: 1 }]),
                stopping_mode: StoppingMode::Any,
                cut_selection: RowSelectionConfig::default(),
                solver: TrainingSolverConfig::default(),
                parallelism: ParallelismConfig::default(),
                scenario_source: None,
                selection: Some(TrainingSelection::Sampled { forward_passes: 1 }),
            },
            upper_bound_evaluation: UpperBoundEvaluationConfig::default(),
            policy: PolicyConfig::default(),
            simulation: IoSimulationConfig::default(),
            exports: ExportsConfig::default(),
            estimation: EstimationConfig::default(),
        }
    }

    /// `max_active_per_stage` below `forward_passes`, so the
    /// budget-below-forward-passes warning fires.
    fn config_with_budget_below_forward_passes() -> Config {
        let mut config = base_test_config();
        config.training.selection = Some(TrainingSelection::Sampled { forward_passes: 2 });
        config.training.cut_selection = RowSelectionConfig {
            max_active_per_stage: Some(1),
            ..RowSelectionConfig::default()
        };
        config
    }

    /// Stopping rules containing a `Gap` entry with neither `tolerance` nor
    /// `relative_tolerance` set.
    fn config_with_gap_stopping_rule_neither_field() -> Config {
        let mut config = base_test_config();
        config.training.stopping_rules = Some(vec![
            StoppingRuleConfig::IterationLimit { limit: 1 },
            StoppingRuleConfig::Gap {
                tolerance: None,
                relative_tolerance: None,
            },
        ]);
        config
    }

    /// Stopping rules containing a well-formed absolute-only `Gap` entry.
    fn config_with_gap_stopping_rule() -> Config {
        let mut config = base_test_config();
        config.training.stopping_rules = Some(vec![
            StoppingRuleConfig::IterationLimit { limit: 1 },
            StoppingRuleConfig::Gap {
                tolerance: Some(1000.0),
                relative_tolerance: None,
            },
        ]);
        config
    }

    /// A relative-only `Gap` entry with no user `BoundStalling`.
    fn config_with_gap_relative_only() -> Config {
        let mut config = base_test_config();
        config.training.stopping_rules = Some(vec![
            StoppingRuleConfig::IterationLimit { limit: 1 },
            StoppingRuleConfig::Gap {
                tolerance: None,
                relative_tolerance: Some(0.01),
            },
        ]);
        config
    }

    /// A relative-only `Gap` entry alongside a user-declared `BoundStalling` —
    /// both rules pass through unchanged.
    fn config_with_gap_relative_and_user_bound_stalling() -> Config {
        let mut config = base_test_config();
        config.training.stopping_rules = Some(vec![
            StoppingRuleConfig::IterationLimit { limit: 1 },
            StoppingRuleConfig::Gap {
                tolerance: None,
                relative_tolerance: Some(0.01),
            },
            StoppingRuleConfig::BoundStalling {
                iterations: 7,
                tolerance: 0.5,
            },
        ]);
        config
    }

    /// `modeling.cost_scale_factor` set to `value` (`None` reproduces the
    /// byte-neutral default-absent shape).
    fn config_with_cost_scale_factor(value: Option<f64>) -> Config {
        let mut config = base_test_config();
        config.modeling.cost_scale_factor = value;
        config
    }

    #[test]
    fn from_config_adds_no_rule_to_an_absent_or_empty_list() {
        for stopping_rules in [None, Some(Vec::new())] {
            let mut config = base_test_config();
            config.training.stopping_rules = stopping_rules;
            let params = StudyParams::from_config(&config, Vec::new())
                .expect("an absent or empty rule list maps without error");
            assert!(
                params.stopping_rule_set.rules.is_empty(),
                "no rule may be added: {:?}",
                params.stopping_rule_set.rules
            );
        }
    }

    /// An absent `modeling.cost_scale_factor` resolves to
    /// [`DEFAULT_COST_SCALE_FACTOR`] — the byte-neutral-at-default contract.
    #[test]
    fn cost_scale_factor_absent_resolves_to_default() {
        let params = StudyParams::from_config(&config_with_cost_scale_factor(None), Vec::new())
            .expect("absent cost_scale_factor is valid");
        assert_eq!(params.cost_scale_factor, super::DEFAULT_COST_SCALE_FACTOR);
        assert_eq!(params.cost_scale_factor, 1_000_000.0);
    }

    /// A valid custom `modeling.cost_scale_factor` resolves verbatim.
    #[test]
    fn cost_scale_factor_custom_value_resolves_verbatim() {
        let params =
            StudyParams::from_config(&config_with_cost_scale_factor(Some(500.0)), Vec::new())
                .expect("500.0 is within [1.0, 1e12] and > 0");
        assert_eq!(params.cost_scale_factor, 500.0);
    }

    /// `from_config` rejects a non-finite `cost_scale_factor` (NaN, +inf,
    /// -inf) with `SddpError::Validation` naming the field.
    #[test]
    fn cost_scale_factor_rejects_non_finite() {
        use crate::SddpError;

        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let err =
                StudyParams::from_config(&config_with_cost_scale_factor(Some(bad)), Vec::new())
                    .expect_err("non-finite cost_scale_factor must be rejected");
            assert!(
                matches!(err, SddpError::Validation(_)),
                "expected SddpError::Validation for {bad}, got: {err:?}"
            );
            assert!(
                err.to_string().contains("cost_scale_factor"),
                "error message must mention 'cost_scale_factor'; got: {err}"
            );
        }
    }

    /// `from_config` rejects a non-positive `cost_scale_factor` (zero and
    /// negative) with `SddpError::Validation` naming the field.
    #[test]
    fn cost_scale_factor_rejects_non_positive() {
        use crate::SddpError;

        for bad in [0.0, -1.0, -1e6] {
            let err =
                StudyParams::from_config(&config_with_cost_scale_factor(Some(bad)), Vec::new())
                    .expect_err("non-positive cost_scale_factor must be rejected");
            assert!(
                matches!(err, SddpError::Validation(_)),
                "expected SddpError::Validation for {bad}, got: {err:?}"
            );
            assert!(
                err.to_string().contains("cost_scale_factor"),
                "error message must mention 'cost_scale_factor'; got: {err}"
            );
        }
    }

    /// A `cost_scale_factor` outside the advisory range `[1.0, 1e12]` is
    /// ACCEPTED (construction succeeds) but emits a WARN-level tracing event
    /// naming the field — never a hard rejection.
    #[test]
    fn cost_scale_factor_outside_advisory_range_warns_but_succeeds() {
        for outside in [0.5, 1e13] {
            let (subscriber, messages) = WarnRecorder::new();
            let params = tracing::subscriber::with_default(subscriber, || {
                StudyParams::from_config(&config_with_cost_scale_factor(Some(outside)), Vec::new())
                    .expect("outside-advisory-range value must still construct successfully")
            });
            assert_eq!(params.cost_scale_factor, outside);
            let recorded = messages.lock().unwrap();
            let relevant: Vec<&str> = recorded
                .iter()
                .map(std::string::String::as_str)
                .filter(|msg| msg.contains("cost_scale_factor"))
                .collect();
            assert!(
                !relevant.is_empty(),
                "expected a WARN event containing 'cost_scale_factor' for {outside}, got: {recorded:?}"
            );
        }
    }

    /// The advisory-range boundaries `1.0` and `1e12` are inclusive — no
    /// warning fires exactly at either edge.
    #[test]
    fn cost_scale_factor_advisory_range_boundaries_are_inclusive() {
        for boundary in [1.0, 1e12] {
            let (subscriber, messages) = WarnRecorder::new();
            let _params = tracing::subscriber::with_default(subscriber, || {
                StudyParams::from_config(&config_with_cost_scale_factor(Some(boundary)), Vec::new())
                    .expect("boundary value must construct successfully")
            });
            let recorded = messages.lock().unwrap();
            let relevant: Vec<&str> = recorded
                .iter()
                .map(std::string::String::as_str)
                .filter(|msg| msg.contains("cost_scale_factor"))
                .collect();
            assert!(
                relevant.is_empty(),
                "boundary value {boundary} must not warn, got: {recorded:?}"
            );
        }
    }

    /// `from_config` leaves `boundary` at the `none()` placeholder (the
    /// resolver, `resolve_boundary_state_requirements`, owns both boundary
    /// facts and the caller patches the resolved value in) and echoes the
    /// caller-supplied `scalar_parameters` table verbatim.
    #[test]
    fn from_config_leaves_boundary_requirements_none() {
        let table = vec![ScalarParameter {
            id: EntityId(1),
            name: "p".to_string(),
            kind: ParameterKind::Constant { value: 1.0 },
        }];
        let params = StudyParams::from_config(&base_test_config(), table.clone())
            .expect("base config is valid");
        assert!(!params.boundary.is_present());
        assert_eq!(params.boundary.inflow_lag_depth(), None);
        assert_eq!(params.scalar_parameters, table);
    }

    /// `BoundaryStateRequirements` derives both facts together: `present(depth)`
    /// is present with the depth `> 0` folded to `Some`, `none()` is absent with
    /// no depth — the two can never disagree.
    #[test]
    fn boundary_state_requirements_derives_both_facts_together() {
        let absent = BoundaryStateRequirements::none();
        assert!(!absent.is_present());
        assert_eq!(absent.inflow_lag_depth(), None);

        let present_no_lag = BoundaryStateRequirements::present(0);
        assert!(present_no_lag.is_present());
        assert_eq!(present_no_lag.inflow_lag_depth(), None);

        let present_with_lag = BoundaryStateRequirements::present(3);
        assert!(present_with_lag.is_present());
        assert_eq!(present_with_lag.inflow_lag_depth(), Some(3));
    }

    /// `from_config` rejects a `Gap` rule with neither `tolerance` nor
    /// `relative_tolerance` set, naming both fields.
    #[test]
    fn from_config_rejects_gap_stopping_rule_with_neither_field() {
        use crate::SddpError;

        let err =
            StudyParams::from_config(&config_with_gap_stopping_rule_neither_field(), Vec::new())
                .expect_err("Gap stopping rule with neither field must be rejected");
        assert!(
            matches!(err, SddpError::Validation(_)),
            "expected SddpError::Validation, got: {err:?}"
        );
        let msg = err.to_string();
        assert!(
            msg.contains("tolerance"),
            "error message must mention 'tolerance'; got: {msg}"
        );
        assert!(
            msg.contains("relative_tolerance"),
            "error message must mention 'relative_tolerance'; got: {msg}"
        );
    }

    /// `from_config` maps a well-formed absolute-only `Gap` rule to the runtime
    /// rule and injects no `BoundStalling` — no rule is ever auto-added.
    #[test]
    fn from_config_maps_absolute_gap_rule() {
        use crate::stopping_rule::StoppingRule;

        let params = StudyParams::from_config(&config_with_gap_stopping_rule(), Vec::new())
            .expect("a well-formed Gap rule maps successfully");
        let rules = &params.stopping_rule_set.rules;
        assert!(
            rules.iter().any(|r| matches!(
                r,
                StoppingRule::Gap {
                    tolerance: Some(_),
                    relative_tolerance: None
                }
            )),
            "the Gap rule must be present as a runtime rule: {rules:?}"
        );
        assert!(
            !rules
                .iter()
                .any(|r| matches!(r, StoppingRule::BoundStalling { .. })),
            "no bound_stalling rule may be auto-added: {rules:?}"
        );
    }

    /// A relative-tolerance `Gap` with no user `BoundStalling` maps to the `Gap`
    /// rule alone — no `BoundStalling` companion is auto-injected, and nothing is
    /// logged about one.
    #[test]
    fn from_config_relative_gap_does_not_inject_bound_stalling_companion() {
        use crate::stopping_rule::StoppingRule;

        let (subscriber, messages) = WarnRecorder::new();
        let params = tracing::subscriber::with_default(subscriber, || {
            StudyParams::from_config(&config_with_gap_relative_only(), Vec::new())
                .expect("a relative-only Gap rule maps successfully")
        });
        let rules = &params.stopping_rule_set.rules;
        assert!(
            !rules
                .iter()
                .any(|r| matches!(r, StoppingRule::BoundStalling { .. })),
            "a relative-only Gap must not auto-inject a bound_stalling companion: {rules:?}"
        );
        let recorded = messages.lock().unwrap();
        assert!(
            !recorded.iter().any(|m| m.contains("companion")),
            "no companion-injection advisory may be logged: {recorded:?}"
        );
    }

    /// A shutdown request alone is reported as `graceful_shutdown`; one that
    /// coincides with the iteration limit is reported as `iteration_limit`.
    #[test]
    fn production_rule_set_reports_graceful_shutdown_on_a_shutdown_request() {
        use crate::stopping_rule::MonitorState;

        let mut config = base_test_config();
        config.training.stopping_rules =
            Some(vec![StoppingRuleConfig::IterationLimit { limit: 10 }]);
        let params = StudyParams::from_config(&config, Vec::new())
            .expect("the base config maps successfully");
        let rule_set = &params.stopping_rule_set;

        let shutdown_at = |iteration| MonitorState {
            iteration,
            wall_time_seconds: 0.0,
            lower_bound: 0.0,
            upper_bound: 0.0,
            lower_bound_history: Vec::new(),
            shutdown_requested: true,
        };
        let first = rule_set.evaluate(&shutdown_at(1));
        assert_eq!(first.termination_reason(), Some("graceful_shutdown"));
        assert!(first.ended_by_shutdown());

        let at_limit = rule_set.evaluate(&shutdown_at(10));
        assert_eq!(at_limit.termination_reason(), Some("iteration_limit"));
        assert!(!at_limit.ended_by_shutdown());
    }

    /// A user-declared `BoundStalling` alongside a `Gap` rule passes through
    /// unchanged and is never doubled.
    #[test]
    fn from_config_gap_with_user_bound_stalling_passes_through() {
        use crate::stopping_rule::StoppingRule;

        let params = StudyParams::from_config(
            &config_with_gap_relative_and_user_bound_stalling(),
            Vec::new(),
        )
        .expect("Gap + explicit BoundStalling maps successfully");
        let rules = &params.stopping_rule_set.rules;
        let bound_stallings: Vec<&StoppingRule> = rules
            .iter()
            .filter(|r| matches!(r, StoppingRule::BoundStalling { .. }))
            .collect();
        assert_eq!(
            bound_stallings.len(),
            1,
            "the user's BoundStalling must be the only one present: {rules:?}"
        );
        assert!(
            matches!(
                bound_stallings[0],
                StoppingRule::BoundStalling {
                    tolerance,
                    iterations
                } if *tolerance == 0.5 && *iterations == 7
            ),
            "the user's BoundStalling values must survive: {rules:?}"
        );
    }

    /// `from_config` resolves the forward-pass count from a `sampled` selection.
    #[test]
    fn from_config_resolves_forward_passes_from_selection() {
        let mut via_selection = base_test_config();
        via_selection.training.selection = Some(TrainingSelection::Sampled { forward_passes: 8 });
        let params_selection = StudyParams::from_config(&via_selection, Vec::new())
            .expect("sampled selection is a valid forward-pass source");
        assert_eq!(params_selection.forward_passes, 8);
    }

    /// `from_config` resolves the simulation scenario count from a `sampled`
    /// selection when simulation is enabled.
    #[test]
    fn from_config_resolves_num_scenarios_from_selection() {
        let mut config = base_test_config();
        config.simulation.enabled = true;
        config.simulation.selection = Some(SimulationSelection::Sampled { num_scenarios: 500 });
        let params = StudyParams::from_config(&config, Vec::new())
            .expect("sampled simulation selection is valid");
        assert_eq!(params.n_scenarios, 500);
        assert_eq!(
            params.simulation_enumerated,
            super::SimulationEnumeratedRequest::Sampled
        );
    }

    /// `from_config` accepts a training `enumerated` selection (the load-time
    /// rejection is lifted): `forward_passes` carries the
    /// [`super::DEFAULT_FORWARD_PASSES`] placeholder and `training_enumerated`
    /// signals the setup layer to re-resolve it from the node graph.
    #[test]
    fn from_config_accepts_training_enumerated_selection() {
        let mut config = base_test_config();
        config.training.selection = Some(TrainingSelection::Enumerated {});
        let params = StudyParams::from_config(&config, Vec::new())
            .expect("enumerated training selection is valid");
        assert_eq!(params.forward_passes, super::DEFAULT_FORWARD_PASSES);
        assert!(params.training_enumerated);
    }

    /// `from_config` accepts a simulation `enumerated` selection: `n_scenarios`
    /// carries the `0` placeholder and the request is
    /// [`super::SimulationEnumeratedRequest::Enumerated`], with the census count
    /// resolved from the node graph downstream.
    #[test]
    fn from_config_accepts_simulation_enumerated_selection() {
        let mut config = base_test_config();
        config.simulation.enabled = true;
        config.simulation.selection = Some(SimulationSelection::Enumerated {});
        let params = StudyParams::from_config(&config, Vec::new())
            .expect("enumerated simulation selection is valid");
        assert_eq!(params.n_scenarios, 0);
        assert_eq!(
            params.simulation_enumerated,
            super::SimulationEnumeratedRequest::Enumerated
        );
    }

    /// When `max_active_per_stage` is less than `forward_passes`, `StudyParams::from_config`
    /// emits a WARN-level tracing event whose message contains `max_active_per_stage`.
    #[test]
    fn study_params_warns_when_budget_below_forward_passes() {
        let (subscriber, messages) = WarnRecorder::new();
        tracing::subscriber::with_default(subscriber, || {
            let _params =
                StudyParams::from_config(&config_with_budget_below_forward_passes(), Vec::new())
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
