//! Study setup struct that owns all precomputed state for a solve run.
//!
//! [`StudySetup`] centralises orchestration from CLI/Python entry points, built
//! from a validated [`System`] and [`cobre_io::Config`].
//!
//! **Ownership**: `StudySetup` owns all data; callers borrow for `TrainingContext`
//! and `StageContext` construction. The [`StochasticContext`] lifetime matches setup.
//!
//! **Not included**: MPI communication (in CLI/Python), solver instances (caller-created),
//! progress bars, event channels (caller-managed).
//!
//! ## Example
//!
//! ```rust,no_run
//! use cobre_sddp::setup::StudySetup;
//! use cobre_sddp::hydro_models::PrepareHydroModelsResult;
//! use cobre_stochastic::{ClassSchemes, OpeningTreeInputs, build_stochastic_context};
//!
//! # fn example(system: &cobre_core::System, config: &cobre_io::Config)
//! #     -> Result<(), cobre_sddp::SddpError> {
//! let stochastic = build_stochastic_context(system, 42, None, &[], &[], OpeningTreeInputs::default(), ClassSchemes { inflow: None, load: None, ncs: None })?;
//! let hydro_models = PrepareHydroModelsResult::default_from_system(system);
//! let setup = StudySetup::new(system, config, stochastic, hydro_models, Vec::new())?;
//! assert!(!setup.inputs.stage_data.stage_templates.templates.is_empty());
//! # Ok(())
//! # }
//! ```

#![deny(clippy::allow_attributes, clippy::allow_attributes_without_reason)]

use chrono::NaiveDate;
use cobre_core::ContractType;
use cobre_core::temporal::SeasonCycleType::Monthly;
use cobre_core::temporal::SeasonMap;
use cobre_core::temporal::StageLagTransition;
use cobre_core::temporal::StageStateConfig;
use cobre_io::Config;
use cobre_io::config::BackwardScheduler;
use cobre_solver::ActiveProfile;
use cobre_stochastic::DerivedInflowSeeds;
use cobre_stochastic::DerivedSeed;
use cobre_stochastic::derive_inflow_seeds;
use cobre_stochastic::noise_entity_order;
use cobre_stochastic::par::lag_transition::derive_downstream_par_order;
use cobre_stochastic::par::lag_transition::precompute_noise_groups;
use cobre_stochastic::par::lag_transition::precompute_stage_lag_transitions;
use cobre_stochastic::season_cast::{DatedWindow, StageCalendar};

use crate::StageTemplates;
use crate::bucket_topology;
use crate::config::LoopParams;
use crate::resolved_parameters::{ResolvedParameters, build_resolved_parameters};
use crate::scaling_report::ScalingReport;
use crate::simulation::SimulationConfig;
use crate::solve::solver_phase::{Phase, validate_phase_solver_config};
use crate::stochastic::noise_key::build_noise_key_table;
mod accessors;
pub(crate) mod lp_build_inputs;
pub mod node_graph;
mod orchestration;
pub mod params;
pub(crate) mod scenario_libraries;
pub mod scenario_library_set;
mod solve_inputs;
pub mod stage_data;
pub mod stochastic_pipeline;
pub(crate) mod template_postprocess;

pub(crate) use lp_build_inputs::resolve_lp_build_inputs;
pub use node_graph::{
    EnumeratedPlan, NodeGraph, NodeId, NodeOpenings, NodePos, NodeRuntime, NodeSuccessor,
    OpeningSource, StageIdx, Traversal, TypedVec,
};
pub use params::{
    BoundaryStateRequirements, DEFAULT_COST_SCALE_FACTOR, DEFAULT_FORWARD_PASSES, DEFAULT_SEED,
    SimulationEnumeratedRequest, StudyParams,
};
pub use scenario_library_set::{PhaseLibraries, ScenarioLibraries};
pub use solve_inputs::SolveInputs;
pub use stage_data::StageData;
pub use stochastic_pipeline::{
    PrepareStochasticResult, build_ncs_factor_entries, build_stochastic_context_for_study,
    load_load_factors_for_stochastic, prepare_stochastic, study_stage_noise_group_ids,
};

use std::collections::HashMap;
use std::path::Path;

use cobre_core::{
    AffineBound, AnticipatedConfig, CoefficientRef, EntityId, GenericConstraint, Hydro,
    HydroPastDefluence, ScalarParameter, Stage, StageId, System, Thermal,
    scenario::{SamplingScheme, ScenarioSource},
};
use cobre_io::StageIdResolver;
use cobre_io::build_hydro_reference_volumes_resolved;
use cobre_stochastic::par::precompute::PrecomputedPar;
use cobre_stochastic::{
    ClassSchemes, ExternalScenarioLibrary, HistoricalScenarioLibrary, StochasticContext,
};

use crate::{
    InflowNonNegativityMethod,
    block_clock::M3S_TO_HM3,
    config::{CutManagementConfig, EventParams, ShutdownSource},
    cut::FutureCostFunction,
    cut_selection::CutSelectionStrategy,
    energy_conversion::{EnergyConversionSet, build_energy_conversion_set},
    error::SddpError,
    horizon_mode::HorizonMode,
    hydro_models::PrepareHydroModelsResult,
    lead_time::{AnticipatedResolution, DeliveryAxis, LeadTime, PointResolution},
    lp::builder::{
        LpBuildInputs, StageGeometry, StateBox, build_stage_templates, contract_family_slot,
    },
    lp::indexer::{
        AnticipatedLocal, AnticipatedPlants, CutStateProjection, HydroCellIndex, HydroSys,
        StateSpace, StudyDimensions, ThermalSys,
    },
    policy::orchestration::PeriodicCheckpoint,
    risk_measure::{RiskMeasure, uniform_effective_measure},
    simulation::EntityCounts,
    simulation::extraction::TransitSeedArc,
    stopping_rule::{StopDecision, StopMask, StoppingRule, StoppingRuleSet},
    time_value::{DeliveryCalendar, TimeValue},
    workspace::CapturedBasis,
};

// ---------------------------------------------------------------------------
// StudySetup
// ---------------------------------------------------------------------------

/// All precomputed study state built once before training and simulation.
///
/// Constructed by [`StudySetup::new`] from a validated [`System`] and
/// [`cobre_io::Config`]. Owns all data so it can be held across async
/// boundaries (e.g., Python GIL release) without lifetime issues.
///
/// Callers build `TrainingContext` and `StageContext` by borrowing
/// from `StudySetup`.
///
/// Commissioning windows (NCS, anticipated) are carried as per-slot
/// `(entry, exit)` pairs rather than per-stage activity masks, so the per-stage
/// patch sites compute dormancy inline and activity stays out of per-stage
/// storage.
#[derive(Debug)]
pub struct StudySetup {
    /// The resolved study inputs shared by the stage, training, and
    /// simulation contexts, disjoint from [`Self::fcf`] so [`Self::train`]
    /// can build a context while holding `&mut fcf`.
    pub inputs: SolveInputs,

    /// Future cost function (cut pool) updated by the backward pass during training.
    pub fcf: FutureCostFunction,

    /// Pre-computed hydro production models (FPHA, turbine curves, etc.).
    pub hydro_models: PrepareHydroModelsResult,

    /// Extended delivery-stage anchors ([`build_extended_delivery_anchors`]):
    /// the `YYYYMM01` anchor of each delivery target stage, indexed by delivery
    /// target `m` (study stages then the synthetic post-study continuation).
    /// Threaded into the simulation
    /// [`SimulationOutputSpec`](crate::simulation::SimulationOutputSpec)'s
    /// `extended_delivery_anchors` so the `anticipated_lanes` extractor
    /// dates a post-study-targeted decision without re-deriving the calendar
    /// walk. Study-only when the study declares no post-study stage.
    pub(crate) extended_delivery_anchors: Vec<i32>,

    /// Declared travel-time arcs (upstream hydro id + travel time), projected
    /// from the bucket topology's arc list ([`build_transit_seed_arcs`]).
    /// Threaded into
    /// [`SimulationOutputSpec`](crate::simulation::SimulationOutputSpec) so
    /// the rolling-seed emitter never re-derives it from `System`. Empty when
    /// the study declares no travel-time arc.
    pub(crate) transit_seed_arcs: Vec<TransitSeedArc>,

    /// This run's own `system.initial_conditions().past_defluences`, retained
    /// for the rolling-seed emitter's pre-study input-tail stitch (nonempty
    /// only when a declared arc's travel time exceeds the study horizon).
    pub(crate) past_defluences: Vec<HydroPastDefluence>,

    /// `study_stage_dates[t] = (stage.start_date, stage.end_date)` per study
    /// stage index, parallel to `inputs.study_stage_ids`. Threaded into
    /// [`SimulationOutputSpec`](crate::simulation::SimulationOutputSpec) so the
    /// rolling-seed emitter's per-stage windows never re-derive the calendar.
    pub(crate) study_stage_dates: Vec<(NaiveDate, NaiveDate)>,

    /// Resolved `(parameter_id, stage)` coefficients; consumed by the LP builder
    /// and the generic-constraint echo.
    pub(crate) resolved_parameters: ResolvedParameters,

    /// Iteration-loop parameters projected from [`crate::config::LoopConfig`].
    ///
    /// `n_fwd_threads` is excluded (derived at runtime) and supplied as a per-call
    /// argument to [`StudySetup::train`].
    pub loop_params: LoopParams,

    /// Simulation pipeline parameters, stored directly as [`crate::simulation::SimulationConfig`].
    pub simulation_config: SimulationConfig,

    /// Whether simulation's scenario source is a declared census
    /// (`simulation.selection = enumerated`) or Monte Carlo sampling —
    /// resolved once the node graph exists, mirroring
    /// [`Self::simulation_config`]'s `n_scenarios`. The caller reads this to
    /// select [`crate::simulation::SimulationWeighting::Census`] vs
    /// [`crate::simulation::SimulationWeighting::Uniform`] for
    /// `aggregate_simulation`.
    pub simulation_enumerated: SimulationEnumeratedRequest,

    /// Relative path to the policy output directory (e.g. `"training/policy"`).
    pub policy_path: String,

    /// Pure-data event flags (output-side).
    ///
    /// Runtime handles (`event_sender`, `shutdown_flag`) are excluded and
    /// supplied per-call in [`StudySetup::train`].
    pub(crate) events: EventParams,

    /// Set by [`StudySetup::enable_periodic_checkpoints`]; every
    /// [`StudySetup::train`] call writes it on its schedule.
    pub(crate) periodic_checkpoint: Option<PeriodicCheckpoint>,

    /// Resolved backward-pass solver profile (`training.solver.backward`, layered
    /// over the current per-phase constant — see
    /// [`crate::solve::solver_phase::Phase::resolve_profile`]). Threaded into
    /// [`StudySetup::train`].
    pub(crate) backward_profile: ActiveProfile,

    /// Resolved forward-pass solver profile (`training.solver.forward`).
    pub(crate) forward_profile: ActiveProfile,

    /// Backward-pass scheduler (`training.parallelism.backward_scheduler`,
    /// carrying the opening-block size), threaded into [`StudySetup::train`]
    /// alongside [`Self::backward_profile`].
    pub(crate) backward_scheduler: BackwardScheduler,

    /// Opening-block-scheduler claim-order override, threaded into
    /// [`StudySetup::train`] alongside [`Self::backward_scheduler`]. No
    /// `training.*` config field resolves this yet — a reserved test-support
    /// seam; production always resolves `true` (see
    /// [`crate::solve::solver_phase::SolverProfiles::hardest_first_claim_order`]).
    pub(crate) hardest_first_claim_order: bool,

    /// Energy-conversion scalars (`ρ_eq`, `V_ref`, `Q_ref`, `ρ_acum`) per
    /// `(hydro, stage)`, consumed by the energy-balance LP constraints and
    /// inflow-/stored-energy extraction.
    pub(crate) energy_conversion: EnergyConversionSet,

    /// `V_min` (`min_storage_hm3`) per hydro, in declaration order; threaded into
    /// the simulation pipeline for stored-energy calculations.
    pub(crate) hydro_min_storage_hm3: Vec<f64>,

    /// Per-stage warm-start basis cache for warm-start / resume training.
    ///
    /// Populated by the CLI / Python paths via
    /// [`StudySetup::set_warm_start_basis_cache`] from the checkpoint's stored
    /// solver bases; [`StudySetup::train`] seeds it into the session's
    /// [`BasisStore`](crate::workspace::BasisStore) so iteration 1's LPs warm-start.
    /// `None` for a fresh start, leaving fresh-mode behavior untouched.
    pub(crate) warm_start_basis_cache: Option<Vec<Option<CapturedBasis>>>,

    /// Boundary-derived state requirements this study was built against, resolved
    /// once and carried identically on every rank. The boundary-cut load path
    /// reads its inflow-lag depth here instead of re-reading the source checkpoint.
    pub(crate) boundary_requirements: BoundaryStateRequirements,
}

impl StudySetup {
    /// Build all precomputed study state from a validated system and config.
    ///
    /// # Errors
    ///
    /// - [`SddpError::Validation`] — if the template list is empty ("system
    ///   has no study stages").
    /// - [`SddpError::Validation`] — if `parse_cut_selection_config` returns
    ///   an invalid config string.
    /// - [`SddpError::Validation`] — if `stochastic`'s precomputed inflow
    ///   model shape does not match `system` (see `validate_par_shape`).
    pub fn new(
        system: &System,
        config: &Config,
        stochastic: StochasticContext,
        hydro_models: PrepareHydroModelsResult,
        scalar_parameters: Vec<ScalarParameter>,
    ) -> Result<Self, SddpError> {
        // No case dir to read the source checkpoint, so the depth is unresolved
        // (None); presence still follows the config, matching the boundary-mask
        // gate an entry point resolves via `resolve_boundary_state_requirements`.
        let boundary = if config.policy.boundary.is_some() {
            BoundaryStateRequirements::present(0)
        } else {
            BoundaryStateRequirements::none()
        };
        Self::new_with_boundary_requirements(
            system,
            config,
            stochastic,
            hydro_models,
            boundary,
            scalar_parameters,
        )
    }

    /// [`Self::new`] with the boundary-derived state requirements supplied
    /// explicitly.
    ///
    /// The requirements are resolved from a loaded boundary policy
    /// (`resolve_boundary_state_requirements`, an I/O read the caller performs),
    /// not from any config knob; [`BoundaryStateRequirements::none`] sizes the lag
    /// block from the PAR model alone. The MPI broadcast path carries the same
    /// value on `StudyParams::boundary` for non-root ranks.
    ///
    /// # Errors
    ///
    /// Same as [`Self::new`].
    pub fn new_with_boundary_requirements(
        system: &System,
        config: &Config,
        stochastic: StochasticContext,
        hydro_models: PrepareHydroModelsResult,
        boundary: BoundaryStateRequirements,
        scalar_parameters: Vec<ScalarParameter>,
    ) -> Result<Self, SddpError> {
        let mut params = StudyParams::from_config(config, scalar_parameters)?;
        params.boundary = boundary;
        // Sentinel: the scenario-source resolvers use the path only for error
        // messages and the historical-years look-up, neither exercised here with a
        // validated Config.
        let sentinel_path = Path::new("config.json");
        let training_source = config
            .training_scenario_source(sentinel_path)
            .map_err(|e| SddpError::Validation(e.to_string()))?;
        let simulation_source = config
            .simulation_scenario_source(sentinel_path)
            .map_err(|e| SddpError::Validation(e.to_string()))?;
        Self::from_broadcast_params(
            system,
            stochastic,
            params,
            hydro_models,
            &training_source,
            &simulation_source,
        )
    }

    /// Build all precomputed study state from pre-resolved broadcast parameters.
    ///
    /// This constructor accepts the scalar fields already extracted from either a
    /// [`cobre_io::Config`] (on rank 0) or a broadcast config struct (on non-root
    /// ranks), performing the expensive computation steps that cannot be serialised.
    ///
    /// # Errors
    ///
    /// - [`SddpError::Validation`] — a per-phase solver profile config sets a
    ///   field the compiled backend does not support (see
    ///   `validate_phase_solver_config`).
    /// - [`SddpError::Validation`] — if the template list is empty ("system
    ///   has no study stages").
    /// - [`SddpError::Validation`] — if `stochastic`'s precomputed inflow
    ///   model shape does not match `system` (see `validate_par_shape`).
    pub fn from_broadcast_params(
        system: &System,
        mut stochastic: StochasticContext,
        config: StudyParams,
        hydro_models: PrepareHydroModelsResult,
        training_source: &ScenarioSource,
        simulation_source: &ScenarioSource,
    ) -> Result<Self, SddpError> {
        validate_par_shape(system, stochastic.par())?;

        let (backward_profile, forward_profile, simulation_profile) =
            resolve_solver_profiles(&config)?;

        // Keys are a pure function of the synced tree + fixed σ, so every rank
        // computes the identical permutation and cuts stay bit-identical across
        // thread/rank counts (canonical-ω aggregation is order-independent).
        let solve_order_keys = build_noise_key_table(system, &stochastic)?;
        stochastic
            .set_solve_order(&solve_order_keys)
            .map_err(|e| SddpError::Validation(e.to_string()))?;

        let (stage_data, initial, energy_conversion, resolved_parameters, transit_seed_arcs) =
            resolve_stage_data(system, &config, &stochastic, &hydro_models)?;
        let n_stages = stage_data.stage_templates.templates.len();

        let study_stage_ids: Vec<i32> = stage_data.stages.iter().map(|s| s.id).collect();
        let study_stage_dates: Vec<(NaiveDate, NaiveDate)> = stage_data
            .stages
            .iter()
            .map(|s| (s.start_date, s.end_date))
            .collect();

        let scenario_libraries = build_scenario_libraries(
            system,
            &stochastic,
            &stage_data,
            &initial,
            config.forward_passes,
            training_source,
            simulation_source,
        )?;

        let node_graph = build_checked_node_graph(
            system,
            &stochastic,
            &study_stage_ids,
            n_stages,
            config.training_enumerated,
        )?;

        let (loop_params, simulation_config) = resolve_phase_configs(
            &node_graph,
            &config,
            simulation_profile,
            stochastic_pipeline::forward_seed_for(simulation_source),
        )?;

        let (fcf, cut_state_layouts) =
            build_future_cost_function(system, &stage_data.state, &node_graph, &loop_params);

        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        // Rejects a degenerate single-stage problem (`num_stages < 2`, no
        // predecessor to generate cuts for); reachable since the empty case rejected
        // above still leaves `n_stages == 1` possible.
        horizon.validate()?;

        let ncs = build_ncs_entity_data(system, &stage_data, &stochastic)?;

        let risk_measures = build_risk_measures(system);
        admission_gate(
            &risk_measures,
            &config.stopping_rule_set,
            config.training_enumerated,
            config.cut_selection.as_ref(),
        )?;

        let extended_delivery_anchors =
            build_extended_delivery_anchors(system, stage_data.time_value.calendar());

        Ok(Self {
            inputs: SolveInputs {
                stage_data,
                stochastic,
                scenario_libraries,
                node_graph,
                initial,
                ncs,
                study_stage_ids,
                horizon,
                cut_management: CutManagementConfig {
                    cut_selection: config.cut_selection,
                    budget: config.budget,
                    cut_activity_tolerance: config.cut_activity_tolerance,
                    risk_measures,
                },
                cut_state_layouts,
            },
            fcf,
            hydro_models,
            extended_delivery_anchors,
            transit_seed_arcs,
            past_defluences: system.initial_conditions().past_defluences.clone(),
            study_stage_dates,
            resolved_parameters,
            loop_params,
            simulation_config,
            simulation_enumerated: config.simulation_enumerated,
            policy_path: config.policy_path,
            events: EventParams {
                export_states: config.export_states,
                checkpoint_schedule: config.checkpoint_schedule,
            },
            periodic_checkpoint: None,
            backward_profile,
            forward_profile,
            backward_scheduler: config.backward_scheduler,
            hardest_first_claim_order: true,
            energy_conversion,
            hydro_min_storage_hm3: system.hydros().iter().map(|h| h.min_storage_hm3).collect(),
            warm_start_basis_cache: None,
            boundary_requirements: config.boundary,
        })
    }
}

// ---------------------------------------------------------------------------
// RunPhasePlan
// ---------------------------------------------------------------------------

/// A run's top-level shape: whether training runs, and whether simulation
/// runs from a stored policy instead. The single owner both L4 entry points
/// (CLI, Python) match on instead of each re-deriving the same predicate
/// pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunPhasePlan {
    /// Training runs; [`PostTrainingSimulation::resolve`] decides whether
    /// simulation follows.
    TrainedThenSimulated,
    /// Training is disabled; simulation runs from a stored policy.
    SimulateFromPolicy,
    /// Training is disabled and no simulation was requested.
    Nothing,
}

impl RunPhasePlan {
    /// Resolves the plan from `training_enabled` and whether simulation was
    /// requested (`simulation_config.n_scenarios > 0`, already normalized).
    #[must_use]
    pub fn resolve(training_enabled: bool, simulate_requested: bool) -> Self {
        match (training_enabled, simulate_requested) {
            (true, _) => Self::TrainedThenSimulated,
            (false, true) => Self::SimulateFromPolicy,
            (false, false) => Self::Nothing,
        }
    }
}

#[cfg(test)]
mod run_phase_plan_tests {
    use super::RunPhasePlan;

    #[test]
    fn resolve_truth_table() {
        assert_eq!(
            RunPhasePlan::resolve(true, true),
            RunPhasePlan::TrainedThenSimulated
        );
        assert_eq!(
            RunPhasePlan::resolve(true, false),
            RunPhasePlan::TrainedThenSimulated
        );
        assert_eq!(
            RunPhasePlan::resolve(false, true),
            RunPhasePlan::SimulateFromPolicy
        );
        assert_eq!(RunPhasePlan::resolve(false, false), RunPhasePlan::Nothing);
    }
}

/// What a trained run does about its configured simulation, matched by both
/// L4 entry points after the training outputs are written.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PostTrainingSimulation {
    /// The configured simulation runs.
    Run,
    /// A signal stopped the run; the simulation is recorded as partial with no scenario run.
    SkipAfterSignalStop,
    /// No simulation was configured.
    NotRequested,
}

impl PostTrainingSimulation {
    /// Resolves the decision from whether simulation was requested, training's
    /// final stop `decision`, and `post_write_level`, the shutdown level the
    /// entry point sampled once after writing the training outputs (`0` is no
    /// request, otherwise a [`ShutdownSource::level`]).
    ///
    /// A requested simulation is skipped when [`signal_stop_requested`] holds,
    /// whatever rule, budget or cooperative request ended training; a
    /// cooperative request alone keeps the simulation.
    #[must_use]
    pub fn resolve(
        simulate_requested: bool,
        decision: &StopDecision,
        post_write_level: usize,
    ) -> Self {
        if !simulate_requested {
            return Self::NotRequested;
        }
        if signal_stop_requested(decision, post_write_level) {
            Self::SkipAfterSignalStop
        } else {
            Self::Run
        }
    }
}

/// Whether a signal stopped the run, through the stop decision or the post-write level.
#[must_use]
pub fn signal_stop_requested(decision: &StopDecision, post_write_level: usize) -> bool {
    decision.mask().contains(StopMask::SIGNAL)
        || ShutdownSource::from_level(post_write_level) == Some(ShutdownSource::Signal)
}

#[cfg(test)]
mod post_training_simulation_tests {
    use super::PostTrainingSimulation::{self, NotRequested, Run, SkipAfterSignalStop};
    use super::signal_stop_requested;
    use crate::config::ShutdownSource;
    use crate::{
        ConvergenceMonitor, StopDecision, StopMask, StoppingMode, StoppingRule, StoppingRuleSet,
        SyncResult,
    };

    fn first_iteration_decision(
        iteration_limit: u64,
        iteration_budget: u64,
        shutdown: Option<ShutdownSource>,
    ) -> StopDecision {
        let rules = StoppingRuleSet {
            rules: vec![StoppingRule::IterationLimit {
                limit: iteration_limit,
            }],
            mode: StoppingMode::Any,
        };
        let mut monitor = ConvergenceMonitor::with_iteration_budget(rules, iteration_budget);
        if let Some(source) = shutdown {
            monitor.set_shutdown(source);
        }
        monitor.update(
            100.0,
            &SyncResult {
                global_ub_mean: 110.0,
                global_ub_std: 1.0,
                ci_95_half_width: 0.5,
                sync_time_ms: 0,
            },
            0.0,
        )
    }

    #[test]
    fn post_training_simulation_resolve_truth_table() {
        let no_stop = first_iteration_decision(100, 100, None);
        let configured_stop = first_iteration_decision(1, 100, None);
        let budget_stop = first_iteration_decision(100, 1, None);
        let cooperative_stop =
            first_iteration_decision(100, 100, Some(ShutdownSource::Cooperative));
        let signal_stop = first_iteration_decision(100, 100, Some(ShutdownSource::Signal));
        let coincident_signal_stop = first_iteration_decision(1, 100, Some(ShutdownSource::Signal));

        assert!(!no_stop.should_stop());
        assert!(configured_stop.configured_stop());
        assert!(budget_stop.mask().contains(StopMask::BUDGET_EXHAUSTED));
        assert!(!budget_stop.configured_stop());
        assert!(cooperative_stop.ended_by_shutdown());
        assert!(signal_stop.ended_by_shutdown());
        assert!(coincident_signal_stop.configured_stop());
        assert!(coincident_signal_stop.mask().contains(StopMask::SIGNAL));

        let levels = [
            0,
            ShutdownSource::Cooperative.level(),
            ShutdownSource::Signal.level(),
        ];
        let cases = [
            ("no stop", no_stop, [Run, Run, SkipAfterSignalStop]),
            (
                "configured stop",
                configured_stop,
                [Run, Run, SkipAfterSignalStop],
            ),
            ("budget stop", budget_stop, [Run, Run, SkipAfterSignalStop]),
            (
                "cooperative shutdown",
                cooperative_stop,
                [Run, Run, SkipAfterSignalStop],
            ),
            (
                "signal shutdown",
                signal_stop,
                [
                    SkipAfterSignalStop,
                    SkipAfterSignalStop,
                    SkipAfterSignalStop,
                ],
            ),
            (
                "signal at a configured stop",
                coincident_signal_stop,
                [
                    SkipAfterSignalStop,
                    SkipAfterSignalStop,
                    SkipAfterSignalStop,
                ],
            ),
        ];
        for (name, decision, expected_when_requested) in cases {
            for (level, expected) in levels.into_iter().zip(expected_when_requested) {
                assert_eq!(
                    PostTrainingSimulation::resolve(false, &decision, level),
                    NotRequested,
                    "{name}, post-write level {level}, not requested"
                );
                assert_eq!(
                    PostTrainingSimulation::resolve(true, &decision, level),
                    expected,
                    "{name}, post-write level {level}, requested"
                );
            }
        }
    }

    #[test]
    fn signal_stop_requested_truth_table() {
        let levels = [
            0,
            ShutdownSource::Cooperative.level(),
            ShutdownSource::Signal.level(),
        ];
        let cases = [
            (
                "no stop",
                first_iteration_decision(100, 100, None),
                [false, false, true],
            ),
            (
                "configured stop",
                first_iteration_decision(1, 100, None),
                [false, false, true],
            ),
            (
                "budget stop",
                first_iteration_decision(100, 1, None),
                [false, false, true],
            ),
            (
                "cooperative shutdown",
                first_iteration_decision(100, 100, Some(ShutdownSource::Cooperative)),
                [false, false, true],
            ),
            (
                "signal shutdown",
                first_iteration_decision(100, 100, Some(ShutdownSource::Signal)),
                [true, true, true],
            ),
            (
                "signal at a configured stop",
                first_iteration_decision(1, 100, Some(ShutdownSource::Signal)),
                [true, true, true],
            ),
        ];
        for (name, decision, expected_per_level) in cases {
            for (level, expected) in levels.into_iter().zip(expected_per_level) {
                assert_eq!(
                    signal_stop_requested(&decision, level),
                    expected,
                    "{name}, post-write level {level}"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// from_broadcast_params sub-phase helpers
// ---------------------------------------------------------------------------

/// Grouped per-slot NCS entity data, held on [`SolveInputs::ncs`].
#[derive(Debug)]
pub(crate) struct NcsEntityData {
    /// Stage-invariant stochastic-slot → dense NCS column index map (slot in
    /// `StochasticContext::ncs_entity_ids` id-sorted order).
    ///
    /// The NCS bound patch sites stride the per-opening cap through
    /// [`StageGeometry::ncs_generation_col`](crate::lp::builder::StageGeometry::ncs_generation_col)
    /// at `stochastic_dense_col[slot]`. Length equals `n_stochastic_ncs`;
    /// empty when the study has no stochastic NCS.
    pub(crate) stochastic_dense_col: Vec<usize>,
    /// Stage-invariant `(entry_stage_id, exit_stage_id)` per stochastic NCS slot
    /// (id-sorted to match `stochastic_dense_col` and the `transform_ncs_noise`
    /// buffer order).
    ///
    /// The dormant-slot `[0, 0]` cap MUST stay identical across the forward,
    /// backward, and lower-bound patch sites — the `evaluate_lower_bound`
    /// "patch NCS per opening" contract; a divergence understates the bound (D15).
    /// Length equals `n_stochastic_ncs`; empty when no stochastic NCS.
    pub(crate) stochastic_windows: Vec<(Option<i32>, Option<i32>)>,
    /// Max generation \[MW\] per stochastic NCS entity, sorted by entity ID.
    pub(crate) max_gen: Vec<f64>,
    /// Whether each stochastic NCS entity may be curtailed, aligned 1:1 with
    /// [`Self::max_gen`]. `false` = must-run: the patch sites pin
    /// `col_lower = col_upper` (not `[0, cap]`), and non-simulated must-run
    /// generation is pre-netted from load.
    pub(crate) allow_curtailment: Vec<bool>,
}

/// Build the per-slot NCS entity data from the system.
///
/// `stochastic_dense_col`, `stochastic_windows`, `max_gen`, and
/// `allow_curtailment` are aligned 1:1 in stochastic NCS-entity (slot) order;
/// see [`NcsEntityData::stochastic_dense_col`] and
/// [`NcsEntityData::stochastic_windows`] for what each carries.
///
/// # Errors
///
/// Returns [`SddpError::Validation`] when a stochastic NCS entity has no match
/// in the system's `non_controllable_sources`.
fn build_ncs_entity_data(
    system: &System,
    stage_data: &StageData,
    stochastic: &StochasticContext,
) -> Result<NcsEntityData, SddpError> {
    let stoch_ncs_ids = stochastic.ncs_entity_ids();

    // Bridge each slot to its dense column via entity id (not a direct index) so the
    // map stays correct when only a subset of NCS are stochastic or the orders
    // diverge. Keyed on the id-sorted slot order, not entity declaration order.
    let mut stochastic_dense_col: Vec<usize> = Vec::with_capacity(stoch_ncs_ids.len());
    let mut stochastic_windows: Vec<(Option<i32>, Option<i32>)> =
        Vec::with_capacity(stoch_ncs_ids.len());
    let mut max_gen: Vec<f64> = Vec::with_capacity(stoch_ncs_ids.len());
    let mut allow_curtailment: Vec<bool> = Vec::with_capacity(stoch_ncs_ids.len());
    for slot_id in stoch_ncs_ids {
        let not_found = || {
            SddpError::Validation(format!(
                "stochastic NCS entity {slot_id:?} not found in system non_controllable_sources"
            ))
        };
        let dense_col = stage_data
            .entity_counts
            .non_controllable_ids
            .iter()
            .position(|&id| id == slot_id.0)
            .ok_or_else(not_found)?;
        let ncs = system
            .non_controllable_sources()
            .iter()
            .find(|n| n.id == *slot_id)
            .ok_or_else(not_found)?;
        stochastic_dense_col.push(dense_col);
        stochastic_windows.push((ncs.entry_stage_id, ncs.exit_stage_id));
        max_gen.push(ncs.max_generation_mw);
        allow_curtailment.push(ncs.allow_curtailment);
    }

    Ok(NcsEntityData {
        stochastic_dense_col,
        stochastic_windows,
        max_gen,
        allow_curtailment,
    })
}

/// Build the stage LP templates and post-process them (scaling, state boxes).
///
/// # Errors
///
/// [`SddpError::Validation`] when the post-processed template list is empty.
fn build_postprocessed_templates(
    system: &System,
    stochastic: &StochasticContext,
    hydro_models: &PrepareHydroModelsResult,
    state: &StateSpace,
    topology: &bucket_topology::TransitBucketTopology,
    inputs: LpBuildInputs<'_>,
) -> Result<(StageTemplates, ScalingReport), SddpError> {
    let resolved_parameters = inputs.resolved_parameters;
    let study_dims = inputs.study_dims;
    let time_value = inputs.time_value;

    let mut stage_templates = build_stage_templates(
        system,
        stochastic.par(),
        &hydro_models.production,
        &hydro_models.evaporation,
        state,
        topology,
        inputs,
    );

    let scaling_report = template_postprocess::postprocess_templates(
        &mut stage_templates,
        system,
        state,
        &study_dims.anticipated_plants,
        resolved_parameters.cost_scale_factor,
        time_value,
    );

    if stage_templates.templates.is_empty() {
        return Err(SddpError::Validation(
            "system has no study stages".to_string(),
        ));
    }

    Ok((stage_templates, scaling_report))
}

/// Build the energy-conversion set and the resolved-parameter table, then fail
/// loud on a generic constraint that references an unresolved scalar-parameter
/// id — the shared prefix of [`resolve_stage_data`] and the
/// validate-time [`validate_generic_constraint_parameters`].
///
/// # Errors
///
/// [`SddpError::Validation`] on energy-conversion or resolved-parameter
/// construction failure, or a generic constraint referencing an id the resolved
/// table never held (via [`check_scalar_parameters_present`]).
fn build_energy_conversion_and_resolved_parameters(
    system: &System,
    hydro_models: &PrepareHydroModelsResult,
    scalar_parameters: &[ScalarParameter],
    cost_scale_factor: f64,
) -> Result<(EnergyConversionSet, ResolvedParameters), SddpError> {
    let study_stage_ids: Vec<StageId> = system
        .stages()
        .iter()
        .filter(|s| s.id >= 0)
        .map(|s| StageId(s.id))
        .collect();
    let stage_to_season: Vec<i32> = system
        .stages()
        .iter()
        .filter(|s| s.id >= 0)
        .map(|s| i32::try_from(s.season_id.unwrap_or(0)).unwrap_or(0))
        .collect();
    // Single source of truth for `reference_volume_hm3`, identical to the source the
    // FPHA backwater path uses, so the productivity reference and the backwater level
    // never drift.
    let reference_volume_fractions =
        build_hydro_reference_volumes_resolved(&hydro_models.reference_volumes_hm3, 0.0);
    let energy_conversion = build_energy_conversion_set(
        system.hydros(),
        &study_stage_ids,
        system.cascade(),
        &reference_volume_fractions,
        // Feeds the FPHA ρ_eq derivation only for plants with no parquet override
        // (the override still wins when present). Per-rank, never broadcast, so every
        // rank sees the same map.
        &hydro_models.vha_geometry_by_hydro,
        Some(&hydro_models.productivity_override),
        Some(&hydro_models.production),
    )
    .map_err(|e| SddpError::Validation(e.to_string()))?;
    let stage_block_counts: Vec<usize> = system
        .stages()
        .iter()
        .filter(|s| s.id >= 0)
        .map(|s| s.blocks.len())
        .collect();
    let resolved_parameters = build_resolved_parameters(
        scalar_parameters,
        &energy_conversion,
        &hydro_models.productivity_override,
        system.hydros(),
        &stage_to_season,
        &study_stage_ids,
        &stage_block_counts,
        cost_scale_factor,
    )
    .map_err(|e| SddpError::Validation(e.to_string()))?;

    check_scalar_parameters_present(system.generic_constraints(), &resolved_parameters)?;

    Ok((energy_conversion, resolved_parameters))
}

/// Run study construction's scalar-parameter presence guard for `system`
/// without building stage templates or a full [`StudySetup`], so a deck with no
/// boundary policy still rejects an unresolved generic-constraint parameter at
/// validate time (a boundary deck runs the same guard inside [`StudySetup::new`]).
///
/// # Errors
///
/// [`SddpError::Validation`] on a scalar parameter that fails to resolve, or a
/// generic constraint referencing an id the resolved table never held.
pub fn validate_generic_constraint_parameters(
    system: &System,
    hydro_models: &PrepareHydroModelsResult,
    scalar_parameters: &[ScalarParameter],
    cost_scale_factor: f64,
) -> Result<(), SddpError> {
    build_energy_conversion_and_resolved_parameters(
        system,
        hydro_models,
        scalar_parameters,
        cost_scale_factor,
    )?;
    Ok(())
}

/// Fails loud when a generic constraint references a scalar-parameter id
/// `resolved_parameters` never resolved, instead of letting
/// [`ResolvedParameters::get`] fall through to its `0.0` sentinel once the LP
/// build starts reading rows.
///
/// # Errors
///
/// [`SddpError::Validation`] naming the constraint and the missing id.
fn check_scalar_parameters_present(
    generic_constraints: &[GenericConstraint],
    resolved_parameters: &ResolvedParameters,
) -> Result<(), SddpError> {
    let is_resolved = |id: EntityId| {
        resolved_parameters
            .id_to_slot
            .binary_search_by_key(&id.0, |(k, _)| *k)
            .is_ok()
    };
    for constraint in generic_constraints {
        let expression_ids =
            constraint
                .expression
                .terms
                .iter()
                .filter_map(|term| match term.coefficient {
                    CoefficientRef::Parameter(id) => Some(id),
                    CoefficientRef::Literal(_) => None,
                });
        let bound_ids = [
            &constraint.bound_lower_affine,
            &constraint.bound_upper_affine,
        ]
        .into_iter()
        .flatten()
        .flat_map(AffineBound::params);
        for id in expression_ids.chain(bound_ids) {
            if !is_resolved(id) {
                return Err(SddpError::Validation(format!(
                    "generic constraint '{}' references scalar parameter id={} not \
                     present in the resolved parameter table",
                    constraint.name, id.0
                )));
            }
        }
    }
    Ok(())
}

/// Validate that `par`'s shape matches `system`, once, at setup — every
/// other reader trusts a validated [`PrecomputedPar`] and checks presence
/// only (`n_stages() > 0`).
///
/// # Errors
///
/// Returns [`SddpError::Validation`] when `par.n_stages() > 0` and either its
/// stage or hydro count differs from `system`'s.
pub(crate) fn validate_par_shape(system: &System, par: &PrecomputedPar) -> Result<(), SddpError> {
    let par_stages = par.n_stages();
    if par_stages == 0 {
        return Ok(());
    }
    let study_stages = system.stages().iter().filter(|s| s.id >= 0).count();
    let hydros = system.hydros().len();
    let par_hydros = par.n_hydros();
    if par_stages == study_stages && par_hydros == hydros {
        return Ok(());
    }
    Err(SddpError::Validation(format!(
        "precomputed inflow model shape mismatch: the model covers {par_stages} stages x \
         {par_hydros} hydros, the study has {study_stages} stages x {hydros} hydros"
    )))
}

/// Validate every per-phase solver-config override, then resolve the three
/// active profiles.
///
/// Validation covers all three phases before any profile resolves, so a
/// backend-unsupported field is rejected before any template exists — on
/// every rank identically, since `from_broadcast_params` is the shared setup
/// path.
///
/// # Errors
///
/// [`SddpError::Validation`] when a phase's solver-config override sets a
/// field the compiled backend does not support.
fn resolve_solver_profiles(
    config: &StudyParams,
) -> Result<(ActiveProfile, ActiveProfile, ActiveProfile), SddpError> {
    validate_phase_solver_config(config.training_solver_backward.as_ref(), Phase::Backward)?;
    validate_phase_solver_config(config.training_solver_forward.as_ref(), Phase::Forward)?;
    validate_phase_solver_config(config.simulation_solver.as_ref(), Phase::Simulation)?;

    // `resolve_profile` is a pure function of the (identically broadcast)
    // config, so every rank resolving independently is sufficient — the
    // resolved `ActiveProfile` itself never needs to go on the wire.
    let backward_profile =
        Phase::Backward.resolve_profile(config.training_solver_backward.as_ref());
    let forward_profile = Phase::Forward.resolve_profile(config.training_solver_forward.as_ref());
    let simulation_profile = Phase::Simulation.resolve_profile(config.simulation_solver.as_ref());

    Ok((backward_profile, forward_profile, simulation_profile))
}

/// `L_state = max(computed_order, boundary_depth)` — the single widening the
/// lag-state depth: `resolve_state_layout`'s dense stride and per-hydro
/// activeness mask, and the seed depth (`resolve_inflow_seeds`). It never
/// widens a historical library, whose width and coverage are the applied
/// PAR's own order (`scenario_libraries::build_historical_inflow_library`).
/// `None` (no boundary) leaves `computed_order` unchanged.
#[must_use]
pub fn widen_lag_state_depth(computed_order: usize, boundary_depth: Option<u32>) -> usize {
    boundary_depth.map_or(computed_order, |d| computed_order.max(d as usize))
}

/// Grouped output of [`resolve_state_layout`].
pub(crate) struct ResolvedStateLayout {
    pub(crate) state: StateSpace,
    pub(crate) anticipated_plants: AnticipatedPlants,
}

/// Resolve every anticipated thermal's delivery-anchored commitment and
/// construct the single role-(a) [`StateSpace`] — before stage templates
/// exist, since none of the state dimensions depend on the built LP.
///
/// The returned `anticipated_plants` is the exact value the layout was built
/// from; [`build_study_dimensions`] takes it as a parameter instead of
/// re-deriving it from the built templates.
///
/// # Errors
///
/// - [`SddpError::Validation`] — a `LeadTime` anticipated plant's resolution
///   fans out (`AnticipatedResolution::max_fanout() > 1`); per-delivery-stage
///   fan-out simulation output is not yet supported.
pub(crate) fn resolve_state_layout(
    system: &System,
    calendar: &DeliveryCalendar,
    par_lp: &PrecomputedPar,
    transit_bucket_topology: &bucket_topology::TransitBucketTopology,
    inflow_lag_depth: Option<u32>,
) -> Result<ResolvedStateLayout, SddpError> {
    let anticipated_plants = AnticipatedPlants::build(system.thermals());

    // Single resolve_point consumer: map each anticipated plant's config to a
    // delivery-anchored PointResolution and derive the constant-lead K_i the
    // still-live ring machinery reads (the resolve_point decider contract). A
    // second resolve_point call site is forbidden — this resolution threads onto
    // the state layout instead.
    let (anticipated_resolution, anticipated_lead_stages) =
        resolve_anticipated_commitments(system, calendar, &anticipated_plants);

    // TODO(anticipated-fanout-output): the coupled output extractor is
    // compute_anticipated_decision_mw
    if anticipated_resolution.max_fanout() > 1 {
        let plant_id = first_fanned_plant_id(system, &anticipated_plants, &anticipated_resolution);
        debug_assert!(
            plant_id.is_some(),
            "max_fanout > 1 must locate the fanning plant"
        );
        return Err(SddpError::Validation(format!(
            "anticipated thermal {}: LeadTime fan-out (a coarse decision stage anchoring \
             several delivery stages) — per-delivery-stage fan-out simulation output is \
             not yet supported",
            plant_id.unwrap_or(EntityId(-1))
        )));
    }

    let hydro_count = system.hydros().len();
    let max_par_order = widen_lag_state_depth(
        system
            .inflow_models()
            .iter()
            .filter(|m| m.stage_id >= 0)
            .map(|m| m.ar_coefficients.len())
            .max()
            .unwrap_or(0)
            .max(par_lp.max_order()),
        inflow_lag_depth,
    );

    // Per-hydro lag-state-slot count for the cut sparse mask: `max_par_order` (the
    // widened psi stride) when PAR(p)-A annual is active, else the classical AR
    // order, each further raised to `inflow_lag_depth` via `widen_lag_state_depth`
    // — the same `L_state = max(AR order, declared depth)` formula `max_par_order`
    // above applies, so a declared depth widens every hydro's activeness mask in
    // lockstep with the dense stride. `par.order(h)` here would silently truncate
    // the cut row's coefficients on the annual-`ψ̂/12` lag slots and produce
    // over-estimating cuts. Falls back to the dense (already-widened) `max_par_order`
    // stride for a hydro `par_lp` omits (`h >= par_lp.n_hydros()`) — production's
    // `par_lp` always covers every system hydro, so the fallback is inert there; a
    // hydro-free `PrecomputedPar` test fixture paired with a hydro-bearing system
    // relies on it to satisfy the `StateSpace::build` length contract.
    let effective_lag_counts: Vec<usize> = if max_par_order > 0 {
        (0..hydro_count)
            .map(|h| {
                if h < par_lp.n_hydros() {
                    widen_lag_state_depth(par_lp.effective_lag_count(h), inflow_lag_depth)
                } else {
                    max_par_order
                }
            })
            .collect()
    } else {
        vec![0; hydro_count]
    };

    // `StateSpace` is the sole role-(a) owner; its constructor finalizes the
    // nonzero mask unconditionally, so every study (storage-only or pure-thermal)
    // has a finalized mask for the single-path mask-driven cut-row loop. There is
    // no separate post-horizon commitment-hold block: a post-study-targeted
    // delivery is carried by the in-study ring slot its modular residue
    // resolves to.
    let state = StateSpace::build(
        system.hydros(),
        max_par_order,
        &effective_lag_counts,
        transit_bucket_topology,
        anticipated_lead_stages,
        anticipated_resolution,
    );

    debug_assert_eq!(
        state.n_anticipated,
        anticipated_plants.len(),
        "state and the anticipated-plant set must agree on n_anticipated"
    );
    Ok(ResolvedStateLayout {
        state,
        anticipated_plants,
    })
}

/// [`bucket_topology::build_transit_bucket_topology`] then [`resolve_state_layout`]
/// — the shared prefix [`resolve_stage_data`] and
/// [`lp_build_inputs::build_stage_templates_resolving_layout`] both need before
/// diverging.
///
/// # Errors
///
/// Propagates [`resolve_state_layout`]'s `LeadTime` fan-out rejection.
pub(crate) fn resolve_state_and_topology(
    system: &System,
    calendar: &DeliveryCalendar,
    par_lp: &PrecomputedPar,
    inflow_lag_depth: Option<u32>,
    boundary_present: bool,
) -> Result<(bucket_topology::TransitBucketTopology, ResolvedStateLayout), SddpError> {
    let topology =
        bucket_topology::build_transit_bucket_topology(system, calendar, boundary_present);
    let layout = resolve_state_layout(system, calendar, par_lp, &topology, inflow_lag_depth)?;
    Ok((topology, layout))
}

/// Canonical absolute delivery/arrival calendar date of a stage `start_date`,
/// encoded `year * 10000 + month * 100 + day` (`YYYYMMDD`). The day is pinned to
/// `01` so the anchor stays month-granular — the same calendar month maps to the
/// same date whether resolved from a weekly or a monthly stage. This is the
/// `anticipated_lanes` output column's month key, not a reconciliation input.
/// Pinned by `year_month_day_anchor_same_month_dates_are_equal` and
/// `year_month_day_anchor_always_normalizes_to_day_01`.
pub(crate) fn year_month_day_anchor(date: NaiveDate) -> i32 {
    use chrono::Datelike;
    // `month()` is 1..=12, so the conversion never fails.
    date.year() * 10_000 + i32::try_from(date.month()).unwrap_or(1) * 100 + 1
}

/// The study's boundary date: the last study stage's (`id >= 0`, highest
/// `id`) exclusive `end_date` — the instant a terminal boundary policy must
/// price. The sole owner of the last-non-negative-stage lookup; every site
/// deriving this date calls it rather than repeating the walk. `None` only
/// when the system declares no study stages.
#[must_use]
pub fn study_horizon_end(system: &System) -> Option<NaiveDate> {
    system
        .stages()
        .iter()
        .rfind(|s| s.id >= 0)
        .map(|s| s.end_date)
}

/// The extended dating calendar: the `study_stages` view chained with the
/// borrowed `post_study_calendar`, so a slot maturing past the horizon dates
/// onto its real post-study stage (study-only when the calendar is empty).
/// Shared by [`build_extended_delivery_anchors`] and the policy manifest
/// builders, so all derive one calendar.
pub(crate) fn extended_delivery_stages<'a>(
    study_stages: &[&'a Stage],
    post_study_calendar: &'a [Stage],
) -> Vec<&'a Stage> {
    study_stages
        .iter()
        .copied()
        .chain(post_study_calendar)
        .collect()
}

/// Extended delivery-stage anchors: the `YYYYMM01` anchor of each delivery
/// target stage — the study stages (`id >= 0`) followed by the synthetic
/// post-study continuation ([`DeliveryCalendar::post_study_stages`]) — indexed
/// by delivery target `m`. The dating input the `anticipated_lanes` output
/// extractor reads for a post-study-targeted decision (`delivery_dates[m]`),
/// matching the policy manifest's `delivery_anchor_at` walk over the same
/// extended calendar. Study-only, byte-identical to a study-stages walk, when
/// no post-study stage is declared.
fn build_extended_delivery_anchors(system: &System, calendar: &DeliveryCalendar) -> Vec<i32> {
    let study_stages: Vec<&Stage> = system.stages().iter().filter(|s| s.id >= 0).collect();
    let anchors: Vec<i32> = extended_delivery_stages(&study_stages, calendar.post_study_stages())
        .iter()
        .map(|s| year_month_day_anchor(s.start_date))
        .collect();
    debug_assert!(
        anchors.len() >= calendar.n_delivery(),
        "extended delivery anchors ({}) must cover the delivery axis ({})",
        anchors.len(),
        calendar.n_delivery(),
    );
    anchors
}

/// Declared travel-time arcs (upstream hydro id + travel time), projected
/// from [`bucket_topology::TransitBucketTopology::arcs`], in that list's
/// order.
fn build_transit_seed_arcs(
    system: &System,
    topology: &bucket_topology::TransitBucketTopology,
) -> Vec<TransitSeedArc> {
    let hydros = system.hydros();
    topology
        .arcs()
        .iter()
        .map(|arc| TransitSeedArc {
            upstream_hydro_id: hydros[arc.upstream.get()].id.0,
            travel_time_hours: arc.travel_time_hours,
        })
        .collect()
}

/// Build the study-invariant, non-state [`StudyDimensions`] from the system
/// alone, before the stage templates exist.
///
/// `anticipated_plants` is threaded from [`resolve_state_layout`] — the same
/// value its [`StateSpace`] was built from.
pub(crate) fn build_study_dimensions(
    system: &System,
    inflow_method: InflowNonNegativityMethod,
    anticipated_plants: AnticipatedPlants,
    downstream_par_order: usize,
) -> StudyDimensions {
    let max_deficit_segments = system
        .buses()
        .iter()
        .map(|b| b.deficit_segments.len())
        .max()
        .unwrap_or(0);

    // Single owner of the study-invariant, non-state LP shape. `n_blks` is
    // deliberately absent — it is per-stage, owned by the per-stage geometry, never
    // study-global.
    StudyDimensions {
        max_deficit_segments,
        inflow_method,
        anticipated_plants,
        downstream_par_order,
    }
}

/// The first (canonical-order) anticipated plant whose `LeadTime` resolution
/// fans out — `|genuine C(t)| > 1` at some decision stage `t` — or `None` if
/// none does. Shares the exact per-plant/per-stage predicate
/// [`AnticipatedResolution::max_fanout`] maxes over, so `Some(_)` iff
/// `resolution.max_fanout() > 1`; `anticipated_plants` and
/// `resolution.per_plant` are both in canonical (anticipated-local) order, so
/// the first match is declaration-order-invariant.
fn first_fanned_plant_id(
    system: &System,
    anticipated_plants: &AnticipatedPlants,
    resolution: &AnticipatedResolution,
) -> Option<EntityId> {
    resolution
        .per_plant
        .iter()
        .enumerate()
        .find_map(|(local_idx, point)| {
            let fans_out =
                (0..point.decision_sets.len()).any(|t| point.genuine_decisions_at(t).count() > 1);
            fans_out.then(|| {
                let thermal_idx = anticipated_plants
                    .thermal_of(AnticipatedLocal::new(local_idx))
                    .get();
                system.thermals()[thermal_idx].id
            })
        })
}

/// Resolve every anticipated thermal's delivery-anchored point commitment and
/// derive the constant-lead per-plant `K_i` the still-live ring machinery reads.
///
/// The sole `resolve_point` consumer (via [`AnticipatedResolution::resolve`]).
/// Warn-free: [`resolve_anticipated_commitments`] wraps this with the setup-time
/// `K = 0` advisory. Returns the
/// per-plant resolution and the anticipated-local constant leads: a
/// `LeadStages(ℓ)` plant keeps `ℓ` byte-for-byte; a `LeadTime` plant takes its
/// per-plant ring depth ([`PointResolution::ring_depth`]) —
/// [`crate::lp::indexer::for_each_live_commitment_slot`] owns which slots that
/// ring depth reaches, for both the LP fill and the policy manifest read.
///
/// The delivery axis is EXTENDED: `n_delivery = n_stages + n_post` while
/// `n_decision` stays `n_stages` (decisions are only ever made in-study), so a
/// `LeadTime` plant's resolution can target a post-study delivery.
pub(crate) fn resolve_anticipated_commitments_core(
    system: &System,
    calendar: &DeliveryCalendar,
    anticipated_plants: &AnticipatedPlants,
) -> (AnticipatedResolution, Vec<usize>) {
    let anticipated_thermals: Vec<&Thermal> = anticipated_plants
        .thermals()
        .map(|t| &system.thermals()[t.get()])
        .collect();
    let leads: Vec<LeadTime> = anticipated_thermals
        .iter()
        .filter_map(|t| t.anticipated_config.as_ref())
        .map(|cfg| match cfg {
            AnticipatedConfig::LeadStages(l) => LeadTime::Stages(*l),
            AnticipatedConfig::LeadTime(h) => LeadTime::Time(*h),
        })
        .collect();
    if leads.is_empty() {
        return (AnticipatedResolution::default(), Vec::new());
    }

    let n_stages = calendar.n_study();
    let resolution = AnticipatedResolution::resolve(
        &leads,
        DeliveryAxis {
            study_stage_hours: calendar.study_total_hours(),
            post_study_stage_hours: calendar.post_study_total_hours(),
        },
    );

    let lead_stages: Vec<usize> = leads
        .iter()
        .zip(&resolution.per_plant)
        .map(|(lead, point)| match lead {
            LeadTime::Stages(l) => {
                let l = usize::try_from(*l).unwrap_or(usize::MAX);
                // LeadStages byte-identity anchor: c(m)=m−ℓ ⇒ depth ≤ ℓ and each
                // in-horizon C(t) is the singleton {t+ℓ}.
                debug_assert!(
                    point.depth.iter().all(|&d| d <= l),
                    "LeadStages depth must be bounded by ℓ"
                );
                debug_assert!(
                    leadstages_decision_sets_are_singletons(point, l, n_stages),
                    "LeadStages c(m)=m−ℓ ⇒ each in-horizon C(t)={{t+ℓ}}"
                );
                debug_assert!(
                    point.ring_depth() <= l,
                    "LeadStages ring_depth must stay bounded by ℓ: returning ℓ verbatim under-sizes if a resolver change breaks this"
                );
                l
            }
            LeadTime::Time(_) => point.ring_depth(),
        })
        .collect();

    (resolution, lead_stages)
}

/// [`resolve_anticipated_commitments_core`] plus the setup-time `K = 0`
/// advisory ([`warn_on_sub_stage_lead`]) — the single owner of that advisory, so
/// it is emitted once per setup.
pub(crate) fn resolve_anticipated_commitments(
    system: &System,
    calendar: &DeliveryCalendar,
    anticipated_plants: &AnticipatedPlants,
) -> (AnticipatedResolution, Vec<usize>) {
    let (resolution, lead_stages) =
        resolve_anticipated_commitments_core(system, calendar, anticipated_plants);
    let anticipated_thermals: Vec<&Thermal> = anticipated_plants
        .thermals()
        .map(|t| &system.thermals()[t.get()])
        .collect();
    warn_on_sub_stage_lead(&anticipated_thermals, &resolution);
    (resolution, lead_stages)
}

/// Emit a per-stage setup-time advisory (exclude-with-advisory, never a
/// hard error) for every `K = 0` sub-stage-lead delivery a `LeadTime` plant's
/// calendar resolves to (`PointResolution::self_delivered_stages`): names the
/// plant, the stage, and the effective `lead_stages == 0` alternative.
/// `LeadStages` plants never trigger it (a positive stage-count lead never
/// resolves `c(m) = m`). Called once from [`resolve_anticipated_commitments`]
/// at setup/load time — the established `tracing::warn!` advisory channel
/// (mirrors `StudyParams::from_config`'s budget-below-forward-passes warning);
/// never from a per-scenario/per-trajectory function (log-spam rule).
fn warn_on_sub_stage_lead(thermals: &[&Thermal], resolution: &AnticipatedResolution) {
    for (thermal, point) in thermals.iter().zip(&resolution.per_plant) {
        for stage in point.self_delivered_stages() {
            tracing::warn!(
                "anticipated thermal {} ({}): stage {stage} resolves to a K=0 sub-stage \
                 lead (lead_stages == 0 at this stage); no anticipation binds and this \
                 plant's generation dispatches as ordinary, unconstrained thermal output",
                thermal.id,
                thermal.name,
            );
        }
    }
}

/// Emit a single setup-time advisory when the resolved anticipated axis
/// carries at least one post-study-targeted delivery — a plant's decider names
/// an in-study decision stage for some delivery target `m >= n_stages`
/// (class-3), or a plant declares at least one non-zero fixed post-horizon
/// (class-4) window in `past_anticipated_commitments` — but the study
/// declares no `config.policy.boundary`: both price at zero terminal value
/// until a boundary is loaded. Never a reject: a `min_mw == max_mw` replay
/// deck is a legitimate use of a fixed post-horizon profile with no boundary;
/// silence would instead hide a modelling error where the user expected the
/// commitment valued against a real future. An all-zero window (including the
/// horizon-end 0 MW stub) never qualifies — a zero value is provably inert.
/// Mirrors [`warn_on_sub_stage_lead`]'s channel and once-at-setup shape (a
/// distinct condition from it and from the class-3 arm, sharing only the
/// event), naming every affected plant in the one emitted event.
fn warn_on_boundary_absent_post_study_delivery(
    system: &System,
    calendar: &DeliveryCalendar,
    anticipated_plants: &AnticipatedPlants,
    resolution: &AnticipatedResolution,
    boundary_present: bool,
) {
    if boundary_present {
        return;
    }
    let n_stages = calendar.n_study();
    let thermals = system.thermals();
    let horizon_end = study_horizon_end(system);
    let past = &system.initial_conditions().past_anticipated_commitments;
    let has_nonzero_fixed = |thermal_id: i32| -> bool {
        horizon_end.is_some_and(|end| {
            past.iter()
                .any(|w| w.thermal_id.0 == thermal_id && w.start_date >= end && w.value_mw != 0.0)
        })
    };
    let affected: Vec<String> = anticipated_plants
        .thermals()
        .zip(&resolution.per_plant)
        .filter(|&(t, point)| {
            let class3 = point.decider.get(n_stages..).is_some_and(|post_study| {
                post_study.iter().any(|c| c.is_some_and(|t| t < n_stages))
            });
            class3 || has_nonzero_fixed(thermals[t.get()].id.0)
        })
        .map(|(t, _)| format!("{} ({})", thermals[t.get()].id, thermals[t.get()].name))
        .collect();
    if affected.is_empty() {
        return;
    }
    tracing::warn!(
        "{} anticipated thermal(s) resolve a post-study-targeted delivery with no \
         config.policy.boundary declared: {} — the delivery prices at zero terminal \
         value until a boundary policy is loaded",
        affected.len(),
        affected.join(", "),
    );
}

/// Whether every in-horizon delivery stage's decision set is the singleton
/// `{t+ℓ}` — the `LeadStages` byte-identity anchor. Edge stages (`t+ℓ ≥
/// n_stages`) carry empty sets and are skipped.
fn leadstages_decision_sets_are_singletons(
    point: &PointResolution,
    lead: usize,
    n_stages: usize,
) -> bool {
    point.decision_sets.iter().enumerate().all(|(t, set)| {
        t.checked_add(lead)
            .filter(|&m| m < n_stages)
            .is_none_or(|m| set.as_slice() == [m])
    })
}

/// Build the per-pool [`CutStateProjection`], one per pool id, projecting the
/// global [`StateSpace`] onto the cut-state dimensions each pool carries.
///
/// Pool `p`, owned by a non-leaf node `n` (`n.pool_id == p`), is sized by its
/// successor's `state_config` — the cost-to-go node `n`'s successor generates
/// for it (pool `p` is populated by the backward pass when it solves the
/// successor's LP and reads the successor's incoming-state reduced costs).
/// Every edge in the node graph goes `t -> t+1` (asserted in
/// `node_graph::build_declared_node_graph`), so all of `n`'s successors sit at
/// one stage and agree on that stage's `state_config` — the dimension is
/// well-defined by construction, no heterogeneity rule needed. Sizing pool `p`
/// from node `n`'s OWN stage's config instead of its successor's is the
/// off-by-one that compiles but stores cuts at the wrong dimension.
///
/// A leaf node has no successor, so the `successor.state_config` rule does not
/// apply to its pool (the trailing shared leaf pool on a declared graph; the
/// terminal pool `n_stages - 1` on a chain): it is sized by the **full global
/// `n_state`**. With `config.policy.boundary` set, the injected boundary cuts
/// come from the external study and are validated and rebuilt against
/// `fcf.state_dimension` (the global `n_state`) by `load_boundary_cuts` /
/// `inject_boundary_cuts`, so the global dimension is exactly the size
/// injection requires — never a DECOMP stage's reduced config. (Per-slot
/// identity reconciliation between a differently-scoped boundary manifest and
/// the local layout is out of scope here.)
///
/// On the chain degeneracy (`nodes[]` absent), `node_graph.n_pools ==
/// n_stages` and `node.pool_id == t`, so this reduces byte-for-byte to the
/// pre-node-native per-stage projection.
fn build_cut_state_layouts(
    system: &System,
    state_layout: &StateSpace,
    node_graph: &NodeGraph,
) -> Vec<CutStateProjection> {
    let study_stages: Vec<&Stage> = system.stages().iter().filter(|s| s.id >= 0).collect();
    // Every pool defaults to the full-dimension projection — the correct value
    // for a leaf-owned pool (no successor) — then non-leaf nodes overwrite
    // their own (disjoint) pool id with the successor-sized projection below.
    let mut layouts =
        vec![CutStateProjection::new(state_layout, FULL_STATE_CONFIG); node_graph.n_pools];
    for (pos, node) in node_graph.nodes.iter_indexed() {
        let Some(succ) = node_graph.successors[pos].first() else {
            continue;
        };
        let config = study_stages[node_graph.nodes[succ.child].stage.0].state_config;
        layouts[node.pool_id] = CutStateProjection::new(state_layout, config);
    }
    layouts
}

/// The all-dimensions cut-state config, sizing a pool to the full global
/// `n_state`. Used for a leaf-owned pool (no successor to govern it) — the
/// terminal pool on a chain.
const FULL_STATE_CONFIG: StageStateConfig = StageStateConfig {
    storage: true,
    inflow_lags: true,
};

/// Build the runtime node graph and run its admissibility rejects.
///
/// # Errors
///
/// Propagates [`node_graph::build_node_graph`]'s construction error,
/// [`reject_scenario_id_under_sampled_selection`], and
/// [`reject_insample_class_under_external_nodes`].
fn build_checked_node_graph(
    system: &System,
    stochastic: &StochasticContext,
    study_stage_ids: &[i32],
    n_stages: usize,
    training_enumerated: bool,
) -> Result<NodeGraph, SddpError> {
    // Binds after `build_scenario_libraries`: an `External`-bound node's Ω
    // addresses the standardized library's raw scenario axis, so binding
    // earlier would race the library's own standardization.
    let stage_id_resolver = StageIdResolver::from_study_stage_ids(study_stage_ids);
    let node_graph = node_graph::build_node_graph(
        system.policy_graph(),
        n_stages,
        &stage_id_resolver,
        stochastic,
    )?;

    reject_scenario_id_under_sampled_selection(&node_graph, training_enumerated)?;
    let prov = stochastic.provenance();
    reject_insample_class_under_external_nodes(
        &node_graph,
        (prov.inflow_scheme, stochastic.n_hydros()),
        (prov.load_scheme, stochastic.n_load_buses()),
        (prov.ncs_scheme, stochastic.n_stochastic_ncs()),
    )?;

    Ok(node_graph)
}

/// Resolve the training and simulation phase configs: re-resolve any
/// `enumerated`-declared forward-pass/scenario count against the now-built
/// node graph, then assemble [`LoopParams`] and [`SimulationConfig`].
///
/// # Errors
///
/// Propagates [`resolve_enumerated_training_count`]'s and
/// [`resolve_enumerated_simulation_count`]'s admissibility and overflow errors,
/// and [`max_iterations_from_rules`]'s error for a rule set with no
/// `IterationLimit` rule.
fn resolve_phase_configs(
    node_graph: &NodeGraph,
    config: &StudyParams,
    simulation_profile: ActiveProfile,
    simulation_forward_seed: Option<u64>,
) -> Result<(LoopParams, SimulationConfig), SddpError> {
    // Resolves any `enumerated`-declared phase's actual count now that the
    // graph exists — config load could only signal the request, never the
    // count. `forward_passes`/`n_scenarios` carry a `sampled`-shaped
    // placeholder until this point when enumerated was requested.
    warn_on_enumeration_asymmetry(
        config.training_enumerated,
        matches!(
            config.simulation_enumerated,
            SimulationEnumeratedRequest::Enumerated
        ),
    );
    let forward_passes = if config.training_enumerated {
        resolve_enumerated_training_count(node_graph)?
    } else {
        config.forward_passes
    };
    let n_scenarios = match config.simulation_enumerated {
        SimulationEnumeratedRequest::Enumerated => resolve_enumerated_simulation_count(node_graph)?,
        SimulationEnumeratedRequest::Sampled => config.n_scenarios,
    };
    let max_iterations = max_iterations_from_rules(&config.stopping_rule_set)?;

    Ok((
        LoopParams {
            seed: config.seed,
            forward_passes,
            training_enumerated: config.training_enumerated,
            max_iterations,
            start_iteration: 0,
            resume_lower_bound_history: Vec::new(),
            stopping_rules: config.stopping_rule_set.clone(),
        },
        SimulationConfig {
            n_scenarios,
            io_channel_capacity: config.io_channel_capacity,
            profile: simulation_profile,
            forward_seed: simulation_forward_seed,
        },
    ))
}

/// Build the per-pool [`FutureCostFunction`] and its [`CutStateProjection`]s
/// from the resolved node graph and phase parameters.
fn build_future_cost_function(
    system: &System,
    state: &StateSpace,
    node_graph: &NodeGraph,
    loop_params: &LoopParams,
) -> (FutureCostFunction, Vec<CutStateProjection>) {
    // Cannot fail: `loop_params` comes from `resolve_phase_configs`, which already
    // ran the enumerated admissibility guards for a `true` `training_enumerated`.
    let traversal = node_graph::Traversal::resolve(
        node_graph,
        loop_params.training_enumerated,
        loop_params.forward_passes,
    );

    let cut_state_layouts = build_cut_state_layouts(system, state, node_graph);
    let pool_state_dimensions: Vec<usize> = cut_state_layouts
        .iter()
        .map(CutStateProjection::n_slots)
        .collect();
    // Cut-RECEIPT stride selected through the resolved traversal. The
    // `Sampled` arm keeps `pool_cut_stride` — the mean+σ statistical margin
    // capped at `forward_passes`, one candidate cut per TRIAL POINT — and
    // NEVER `forward_solve_counts`, the enumerated engine's node-deduplicated
    // per-pool FORWARD-SOLVE count, which under-reserves a branched pool's
    // slots (the backward still produces one cut per trial point, so the next
    // trial collides with a still-active slot — `CutPool::add_cut`'s
    // double-insert panic). The `Enumerated` arm sizes at the node-native cut
    // count, `enumerated_pool_cut_stride`: exactly 1 per non-leaf node
    // (in-degree 1, one distinct incoming state, one cut per iteration) and 0
    // for the shared leaf pool — NOT the sampled bound, which would keep the
    // per-pool capacity/basis/broadcast/checkpoint reservation the node-native
    // backward never fills.
    let visit_bounds = match &traversal {
        node_graph::Traversal::Sampled { forward_passes } => {
            node_graph.pool_cut_stride(*forward_passes)
        }
        node_graph::Traversal::Enumerated(_) => node_graph::enumerated_pool_cut_stride(node_graph),
    };
    let fcf = FutureCostFunction::new_per_pool(
        &pool_state_dimensions,
        state.n_state,
        loop_params.forward_passes,
        loop_params.max_iterations.saturating_add(1),
        &vec![0; node_graph.n_pools],
        &visit_bounds,
    );

    (fcf, cut_state_layouts)
}

/// No-op fallback `SeasonMap`, shared by [`resolve_stage_lag_transitions`] and
/// [`resolve_inflow_seeds`].
static NOOP_SEASON_MAP: SeasonMap = SeasonMap {
    cycle_type: Monthly,
    seasons: Vec::new(),
};

/// Downstream PAR order and per-stage lag transitions over one `SeasonMap`.
/// The sole owner of both derivations — `resolve_stage_data` and
/// `scenario_libraries::build_historical_inflow_library` each call it once,
/// over their own PAR model.
pub(crate) fn resolve_stage_lag_transitions(
    stages: &[Stage],
    par: &PrecomputedPar,
    season_map: Option<&SeasonMap>,
) -> (usize, Vec<StageLagTransition>) {
    let downstream_par_order = derive_downstream_par_order(stages, par, season_map);
    let effective_season_map = season_map.unwrap_or(&NOOP_SEASON_MAP);
    let stage_lag_transitions =
        precompute_stage_lag_transitions(stages, effective_season_map, downstream_par_order);
    (downstream_par_order, stage_lag_transitions)
}

/// Derived per-hydro PAR lag-slot and accumulator seeds from the system's
/// first study stage. The sole owner — `resolve_initial_conditions` and the
/// opening tree each call it once, at the same depth.
fn resolve_inflow_seeds(system: &System, max_par_order: usize) -> DerivedInflowSeeds {
    let season_map = system
        .policy_graph()
        .season_map
        .as_ref()
        .unwrap_or(&NOOP_SEASON_MAP);
    match system.stages().iter().find(|s| s.id >= 0) {
        None => DerivedInflowSeeds::zero(system.hydros().len(), max_par_order),
        Some(first_stage) => derive_inflow_seeds(
            system.inflow_history(),
            &system.initial_conditions().recent_observations,
            system.hydros(),
            first_stage,
            season_map,
            max_par_order,
        ),
    }
}

/// Initial state vector and derived per-hydro PAR lag-slot/accumulator seeds,
/// held on [`SolveInputs::initial`].
#[derive(Debug)]
pub(crate) struct InitialConditions {
    pub(crate) state: Vec<f64>,
    /// Applied to the stage-0 lag block and to every trajectory start in the
    /// forward pass and simulation pipeline instead of zero-filling. All-zero
    /// when the derivation has no resolvable data.
    pub(crate) inflow_seeds: DerivedInflowSeeds,
}

/// Build the initial state vector and its inflow-lag seeds together —
/// [`build_initial_state`]'s lag block reads [`resolve_inflow_seeds`]'s output.
fn resolve_initial_conditions(
    system: &System,
    state: &StateSpace,
    study_dims: &StudyDimensions,
    topology: &bucket_topology::TransitBucketTopology,
    stage0_box: Option<&StateBox>,
) -> InitialConditions {
    let inflow_seeds = resolve_inflow_seeds(system, state.max_par_order);
    let mut initial_state =
        build_initial_state(system, study_dims, state, &inflow_seeds.lag_values);
    splice_transit_bucket_seed(&mut initial_state, state, system, topology);
    if let Some(stage0_box) = stage0_box {
        canonicalize_initial_state(&mut initial_state, state, stage0_box);
    }
    InitialConditions {
        state: initial_state,
        inflow_seeds,
    }
}

/// Resolve the stage phase: the state layout, LP templates, per-stage data,
/// and initial conditions built once before scenario libraries and the node
/// graph.
///
/// `energy_conversion`, `resolved_parameters`, and `transit_seed_arcs` return
/// alongside [`StageData`] rather than fold into it — all three are
/// `StudySetup`'s own fields, not part of the stage/training/simulation
/// contexts' shared inputs.
///
/// # Errors
///
/// Propagates [`resolve_state_layout`]'s, [`build_energy_conversion_and_resolved_parameters`]'s
/// and [`build_postprocessed_templates`]'s errors.
fn resolve_stage_data(
    system: &System,
    config: &StudyParams,
    stochastic: &StochasticContext,
    hydro_models: &PrepareHydroModelsResult,
) -> Result<
    (
        StageData,
        InitialConditions,
        EnergyConversionSet,
        ResolvedParameters,
        Vec<TransitSeedArc>,
    ),
    SddpError,
> {
    let calendar = DeliveryCalendar::from_system(system);
    let (transit_bucket_topology, layout) = resolve_state_and_topology(
        system,
        &calendar,
        stochastic.par(),
        config.boundary.inflow_lag_depth(),
        config.boundary.is_present(),
    )?;
    let transit_seed_arcs = build_transit_seed_arcs(system, &transit_bucket_topology);
    warn_on_boundary_absent_post_study_delivery(
        system,
        &calendar,
        &layout.anticipated_plants,
        &layout.state.anticipated_resolution,
        config.boundary.is_present(),
    );

    let (energy_conversion, resolved_parameters) = build_energy_conversion_and_resolved_parameters(
        system,
        hydro_models,
        &config.scalar_parameters,
        config.cost_scale_factor,
    )?;

    let stages: Vec<Stage> = system
        .stages()
        .iter()
        .filter(|s| s.id >= 0)
        .cloned()
        .collect();
    let (downstream_par_order, stage_lag_transitions) = resolve_stage_lag_transitions(
        &stages,
        stochastic.par(),
        system.policy_graph().season_map.as_ref(),
    );
    let study_dims = build_study_dimensions(
        system,
        config.inflow_method,
        layout.anticipated_plants,
        downstream_par_order,
    );

    let time_value = TimeValue::from_system(system, &study_dims.anticipated_plants, calendar);
    let hydro_cell_index = HydroCellIndex::build(system.hydros());
    let load_bus_ids = &stochastic.entity_order()[stochastic.class_dimensions().load_bus_range()];
    let inputs = resolve_lp_build_inputs(
        system,
        load_bus_ids,
        &hydro_models.production,
        &study_dims,
        &time_value,
        &hydro_cell_index,
        &resolved_parameters,
    );

    let (stage_templates, scaling_report) = build_postprocessed_templates(
        system,
        stochastic,
        hydro_models,
        &layout.state,
        &transit_bucket_topology,
        inputs,
    )?;

    let noise_group_ids = precompute_noise_groups(&stages);

    let initial = resolve_initial_conditions(
        system,
        &layout.state,
        &study_dims,
        &transit_bucket_topology,
        stage_templates.state_boxes().first(),
    );

    let stage_data = stage_data::StageData {
        entity_counts: build_entity_counts(system),
        pumping_consumption_mw_per_m3s: build_pumping_consumption(system),
        contract_prices_per_stage: build_contract_prices_per_stage(
            system,
            &stage_templates.geometry_per_stage,
        ),
        contract_slots: build_contract_slots(system),
        stage_templates,
        time_value,
        state: layout.state,
        study_dims,
        hydro_cell_index,
        stages,
        stage_lag_transitions,
        noise_group_ids,
        scaling_report,
    };

    Ok((
        stage_data,
        initial,
        energy_conversion,
        resolved_parameters,
        transit_seed_arcs,
    ))
}

/// Build one phase's per-class [`PhaseLibraries`].
///
/// A class is built when `source`'s scheme for it matches the class's target
/// scheme (`Historical` or `External`) and `training` is `None` (this call
/// builds the training phase itself) or names a different scheme for that
/// class than `source`'s own — the dedupe [`ScenarioLibraries`] documents.
///
/// # Errors
///
/// Propagates [`SddpError`] from the individual library builders on validation
/// or padding failure.
fn build_phase_libraries(
    system: &System,
    stochastic: &StochasticContext,
    stage_data: &StageData,
    seed: DerivedSeed<'_>,
    forward_passes: u32,
    source: &ScenarioSource,
    training: Option<&ScenarioSource>,
) -> Result<PhaseLibraries, SddpError> {
    let inflow_scheme = source.inflow_scheme;
    let load_scheme = source.load_scheme;
    let ncs_scheme = source.ncs_scheme;
    let inflow_differs = training.is_none_or(|t| t.inflow_scheme != inflow_scheme);
    let load_differs = training.is_none_or(|t| t.load_scheme != load_scheme);
    let ncs_differs = training.is_none_or(|t| t.ncs_scheme != ncs_scheme);

    let historical: Option<HistoricalScenarioLibrary> =
        if inflow_scheme == SamplingScheme::Historical && inflow_differs {
            Some(scenario_libraries::build_historical_inflow_library(
                system,
                stochastic.par(),
                seed,
                source.historical_years.as_ref(),
                forward_passes,
            )?)
        } else {
            None
        };

    let external_inflow: Option<ExternalScenarioLibrary> =
        if inflow_scheme == SamplingScheme::External && inflow_differs {
            Some(scenario_libraries::build_external_inflow_library(
                system,
                stochastic.par(),
                seed,
                &stage_data.stage_lag_transitions,
                forward_passes,
                stage_data.study_dims.downstream_par_order,
            )?)
        } else {
            None
        };

    // Shared by training and a simulation phase whose own scheme diverges —
    // see `build_external_load_library`'s doc for why.
    let normal_load_bus_ids =
        system.load_noise_member_bus_ids(training.unwrap_or(source).load_scheme);

    let external_load: Option<ExternalScenarioLibrary> =
        if load_scheme == SamplingScheme::External && load_differs {
            Some(scenario_libraries::build_external_load_library(
                system,
                load_scheme,
                forward_passes,
                stochastic.normal(),
                &normal_load_bus_ids,
            )?)
        } else {
            None
        };

    let external_ncs: Option<ExternalScenarioLibrary> =
        if ncs_scheme == SamplingScheme::External && ncs_differs {
            Some(scenario_libraries::build_external_ncs_library(
                system,
                forward_passes,
                stochastic.ncs_normal(),
                stochastic.ncs_entity_ids(),
            )?)
        } else {
            None
        };

    Ok(PhaseLibraries {
        inflow_scheme,
        load_scheme,
        ncs_scheme,
        historical,
        external_inflow,
        external_load,
        external_ncs,
    })
}

/// Build the training and simulation [`ScenarioLibraries`].
///
/// # Errors
///
/// Propagates [`SddpError`] from [`build_phase_libraries`] or from
/// [`assert_external_library_widths`]'s width check.
fn build_scenario_libraries(
    system: &System,
    stochastic: &StochasticContext,
    stage_data: &StageData,
    initial: &InitialConditions,
    forward_passes: u32,
    training_source: &ScenarioSource,
    simulation_source: &ScenarioSource,
) -> Result<ScenarioLibraries, SddpError> {
    let seed = initial.inflow_seeds.as_seed(stage_data.state.max_par_order);
    let training = build_phase_libraries(
        system,
        stochastic,
        stage_data,
        seed,
        forward_passes,
        training_source,
        None,
    )?;
    let simulation = build_phase_libraries(
        system,
        stochastic,
        stage_data,
        seed,
        forward_passes,
        simulation_source,
        Some(training_source),
    )?;
    let libraries = ScenarioLibraries {
        training,
        simulation,
    };
    assert_external_library_widths(system, &libraries, training_source)?;
    Ok(libraries)
}

/// G2 (rule 49): every standardized external library's `n_entities()` matches its
/// `noise_entity_order` block width. Reuses [`noise_entity_order`] — the single
/// owner of the three-block entity order — rather than re-deriving a class's
/// entity count a third time; a mismatch is a hard [`SddpError::Validation`]
/// naming the class and both widths. Runs at setup because the standardized
/// libraries exist only after [`build_scenario_libraries`]. `training_source`
/// resolves the same [`ClassSchemes`] every `noise_entity_order` caller in the
/// setup path passes, so training and simulation phases agree on membership.
fn assert_external_library_widths(
    system: &System,
    libraries: &ScenarioLibraries,
    training_source: &ScenarioSource,
) -> Result<(), SddpError> {
    let schemes = ClassSchemes {
        inflow: Some(training_source.inflow_scheme),
        load: Some(training_source.load_scheme),
        ncs: Some(training_source.ncs_scheme),
    };
    let order = noise_entity_order(system, &schemes);
    let check = |library: Option<&ExternalScenarioLibrary>, block_width: usize| {
        library.map_or(Ok(()), |lib| {
            if lib.n_entities() == block_width {
                Ok(())
            } else {
                Err(SddpError::Validation(format!(
                    "external {} library width mismatch: n_entities() = {} but the \
                     noise_entity_order block width is {block_width}",
                    lib.entity_class(),
                    lib.n_entities(),
                )))
            }
        })
    };
    for phase in [&libraries.training, &libraries.simulation] {
        check(phase.external_inflow.as_ref(), order.hydro_ids.len())?;
        check(phase.external_load.as_ref(), order.load_bus_ids.len())?;
        check(phase.external_ncs.as_ref(), order.ncs_entity_ids.len())?;
    }
    Ok(())
}

/// The run's iteration budget: the largest `IterationLimit` limit. Used for FCF pre-sizing.
///
/// # Errors
///
/// Returns [`SddpError::Validation`] when the rule set has no `IterationLimit` rule.
fn max_iterations_from_rules(rules: &StoppingRuleSet) -> Result<u64, SddpError> {
    rules
        .rules
        .iter()
        .filter_map(|r| {
            if let StoppingRule::IterationLimit { limit } = r {
                Some(*limit)
            } else {
                None
            }
        })
        .max()
        .ok_or_else(|| {
            SddpError::Validation(
                "the stopping rule set has no iteration_limit rule; every run needs one for its iteration budget"
                    .to_string(),
            )
        })
}

/// Build the per-study-stage risk measures from the system's stage risk configs.
///
/// One entry per study stage (`id >= 0`), in stage-index order, matching the
/// `stage_templates.geometry_per_stage` / template ordering the cut-management
/// pipeline indexes by stage.
fn build_risk_measures(system: &System) -> Vec<RiskMeasure> {
    system
        .stages()
        .iter()
        .filter(|s| s.id >= 0)
        .map(|s| RiskMeasure::from(s.risk_config))
        .collect()
}

// ---------------------------------------------------------------------------
// Admission gate
// ---------------------------------------------------------------------------

/// The setup-time admission gate: the permanent arms that survive the
/// node-native collapse, evaluated once from
/// [`StudySetup::from_broadcast_params`]. Absent the gated features (no `gap`
/// stopping rule, an expectation measure at every stage, and no dynamic cut
/// selection under enumerated forwards) it returns `Ok(())` unconditionally, so
/// a default study is byte-neutral.
///
/// # Errors
///
/// Returns [`SddpError::Validation`] when a `gap` stopping rule is present under
/// any stage's effective non-expectation risk measure, under sampled forward
/// selection, or when dynamic cut selection is paired with enumerated forward
/// traversal.
fn admission_gate(
    risk_measures: &[RiskMeasure],
    stopping_rules: &StoppingRuleSet,
    training_enumerated: bool,
    cut_selection: Option<&CutSelectionStrategy>,
) -> Result<(), SddpError> {
    reject_gap_under_effective_risk_aversion(risk_measures, stopping_rules, training_enumerated)?;
    reject_gap_under_sampled_selection(stopping_rules, training_enumerated)?;
    reject_dynamic_cut_selection_under_enumerated(cut_selection, training_enumerated)
}

/// Reject a `gap` stopping rule that has no exact bound to compare against.
///
/// Under **sampled** forwards the upper bound is a statistical estimate under any
/// measure — the [`reject_gap_under_sampled_selection`] companion rejects that
/// separately; this function additionally names the offending risk measure so a
/// risk-averse sampled study gets the more specific message. Under **enumerated**
/// forwards every path is visited, so the exact risk-adjusted upper bound is
/// computable and a `gap` rule IS admissible under `CVaR` — **provided the
/// measure is uniform across stages** (see [`reject_gap_under_nonuniform_risk`]):
/// the bound applies one static risk measure to the enumerated path costs, which
/// is undefined when stages differ. No `gap` rule present ⇒ `Ok(())`.
///
/// # Errors
///
/// Returns [`SddpError::Validation`] naming the rule, the offending stage's
/// measure, and the admitting condition.
fn reject_gap_under_effective_risk_aversion(
    risk_measures: &[RiskMeasure],
    stopping_rules: &StoppingRuleSet,
    training_enumerated: bool,
) -> Result<(), SddpError> {
    if !stopping_rules.rules.iter().any(rule_is_gap) {
        return Ok(());
    }
    if training_enumerated {
        return reject_gap_under_nonuniform_risk(risk_measures);
    }
    for (stage, measure) in risk_measures.iter().enumerate() {
        if is_effective_non_expectation(measure) {
            return Err(SddpError::Validation(format!(
                "gap stopping rule is inadmissible under the effective non-expectation \
                 risk measure at stage {stage} ({measure:?}) with sampled forward selection; \
                 enumerated forwards admit a gap rule under a uniform risk measure"
            )));
        }
    }
    Ok(())
}

/// Reject a `gap` stopping rule under enumerated forwards whose per-stage risk
/// measures are not uniform. The enumerated risk-adjusted upper bound applies one
/// static risk measure to the whole-path costs, so a measure that varies stage to
/// stage has no single bound to gap against. Uniformity is checked on the
/// [`effective`](RiskMeasure::effective) form, so a mix of `Expectation` and
/// `CVaR { lambda: 0 }` is uniform. Uniform (or empty) ⇒ `Ok(())`.
///
/// # Errors
///
/// Returns [`SddpError::Validation`] naming the first stage whose measure differs
/// and the admitting condition (a uniform measure).
fn reject_gap_under_nonuniform_risk(risk_measures: &[RiskMeasure]) -> Result<(), SddpError> {
    // `uniform_effective_measure` is the single owner of the uniformity predicate,
    // so this admission gate cannot drift from the bound the session applies. The
    // loop below runs only to name the offending stage for the diagnostic.
    if risk_measures.is_empty() || uniform_effective_measure(risk_measures).is_some() {
        return Ok(());
    }
    let first = risk_measures[0].effective();
    let Some(stage) = risk_measures.iter().position(|m| m.effective() != first) else {
        return Ok(());
    };
    Err(SddpError::Validation(format!(
        "gap stopping rule under enumerated forwards requires a uniform risk measure \
         across all stages; stage {stage} ({:?}) differs from stage 0 ({:?}). The \
         risk-adjusted upper bound applies one static CVaR measure to the enumerated \
         path costs, undefined when stages differ",
        risk_measures[stage], risk_measures[0]
    )))
}

/// Reject a `gap` stopping rule under sampled forward selection: the exact upper
/// bound a `gap` rule compares the lower bound against is produced only by the
/// enumerated engine; under sampled forwards the upper bound is a noisy
/// statistical estimate, so their difference is not a valid gap. No `gap` rule
/// present, or enumerated forwards ⇒ `Ok(())`.
///
/// # Errors
///
/// Returns [`SddpError::Validation`] naming the rule, the offending selection
/// (sampled), and the admitting condition (enumerated forwards).
fn reject_gap_under_sampled_selection(
    stopping_rules: &StoppingRuleSet,
    training_enumerated: bool,
) -> Result<(), SddpError> {
    if training_enumerated {
        return Ok(());
    }
    if stopping_rules.rules.iter().any(rule_is_gap) {
        return Err(SddpError::Validation(
            "gap stopping rule is inadmissible under sampled forward selection; the upper \
             bound is then a statistical estimate, not the exact bound a gap rule requires — \
             a gap rule admits only enumerated forward selection"
                .to_string(),
        ));
    }
    Ok(())
}

/// Reject dynamic cut selection paired with enumerated forward traversal. The
/// enumerated engine seeds each pool at its node-native cut stride, while
/// dynamic cut selection assumes the sampled-selection eviction-key discipline
/// its downstream budget-eviction reader depends on; the pairing would drive
/// that reader down an untested eviction path. Any non-[`Dynamic`] strategy (or
/// none) under enumerated forwards, and [`Dynamic`] under sampled forwards, are
/// admitted.
///
/// [`Dynamic`]: CutSelectionStrategy::Dynamic
///
/// # Errors
///
/// Returns [`SddpError::Validation`] naming the pairing when `training_enumerated`
/// and `cut_selection` is [`CutSelectionStrategy::Dynamic`].
fn reject_dynamic_cut_selection_under_enumerated(
    cut_selection: Option<&CutSelectionStrategy>,
    training_enumerated: bool,
) -> Result<(), SddpError> {
    if training_enumerated && matches!(cut_selection, Some(CutSelectionStrategy::Dynamic { .. })) {
        return Err(SddpError::Validation(
            "dynamic cut selection is inadmissible under enumerated forward traversal; the \
             enumerated engine seeds each cut pool at its node-native stride, whereas dynamic \
             cut selection assumes the sampled-selection eviction-key discipline — pair \
             enumerated forwards with a value-based cut selection strategy, or none"
                .to_string(),
        ));
    }
    Ok(())
}

/// Whether `rule` is the `gap` stopping-rule variant. Total match (every variant
/// named, `Gap` destructured with no `..`) so a new field on
/// [`StoppingRule::Gap`] or a new [`StoppingRule`] variant must be dispositioned
/// here rather than silently falling through.
fn rule_is_gap(rule: &StoppingRule) -> bool {
    match rule {
        StoppingRule::Gap {
            tolerance: _,
            relative_tolerance: _,
        } => true,
        StoppingRule::IterationLimit { .. }
        | StoppingRule::TimeLimit { .. }
        | StoppingRule::BoundStalling { .. } => false,
    }
}

/// Whether `measure` is *effectively* non-expectation (risk-averse) — i.e. its
/// [`effective`](RiskMeasure::effective) form is not `Expectation`.
/// `CVaR { lambda: 0 }` is documented-equivalent to `Expectation`, so only a
/// positive risk-aversion weight counts; the variant disposition lives on
/// `RiskMeasure::effective`, the single owner of the `lambda > 0` predicate.
fn is_effective_non_expectation(measure: &RiskMeasure) -> bool {
    measure.effective() != RiskMeasure::Expectation
}

/// Advisory (never a reject) for an asymmetric enumeration declaration: when
/// exactly one phase declares `enumerated` scenario selection, one census-only
/// capability is unavailable. Names both phases and the specific missing
/// capability — the exact lower bound (needs enumerated training) or the
/// weighted census simulation statistics (needs enumerated simulation) — never
/// a generic "census required". Symmetric declarations warn nothing.
fn warn_on_enumeration_asymmetry(training_enumerated: bool, simulation_enumerated: bool) {
    match (training_enumerated, simulation_enumerated) {
        (true, false) => tracing::warn!(
            "training declares enumerated scenario selection but simulation declares \
             sampled: the exact lower bound from exhaustive training enumeration is \
             available, but the weighted census simulation statistics are not, since \
             simulation samples its scenarios"
        ),
        (false, true) => tracing::warn!(
            "simulation declares enumerated scenario selection but training declares \
             sampled: the weighted census simulation statistics are available, but the \
             exact lower bound is not, since training samples its scenarios"
        ),
        (true, true) | (false, false) => {}
    }
}

/// Shared enumerated admissibility guard, called by both
/// [`resolve_enumerated_training_count`] and
/// [`resolve_enumerated_simulation_count`] so the two enumerated axes cannot
/// admit different graph shapes: derives the graph's path count via
/// [`node_graph::enumerated_scenario_count`] (propagating its `K^T` u64
/// overflow guard unchanged), rejects a non-singleton within-node opening set
/// via [`reject_within_node_opening_enumeration`], rejects a recombination
/// join via [`reject_recombining_node_enumeration`] — the two preconditions
/// exact node-dedup traversal needs, not merely a fence — then narrows the
/// result to `u32`. `axis` and `count_noun` phrase only the caller's own
/// overflow message (e.g. `("training", "forward-pass")`,
/// `("simulation", "scenario")`).
///
/// # Errors
///
/// Propagates [`node_graph::enumerated_scenario_count`]'s overflow
/// [`SddpError::Validation`]; returns [`SddpError::Validation`] when a node
/// carries more than one opening, when a node has two or more predecessors (a
/// recombination join), or when the derived count exceeds `u32`.
fn enumerated_admissible_count(
    node_graph: &NodeGraph,
    axis: &str,
    count_noun: &str,
) -> Result<u32, SddpError> {
    let derived = node_graph::enumerated_scenario_count(node_graph)?;
    reject_within_node_opening_enumeration(node_graph)?;
    reject_recombining_node_enumeration(node_graph)?;
    u32::try_from(derived).map_err(|_| {
        SddpError::Validation(format!(
            "{axis} enumerated scenario selection derived {derived} paths from the policy \
             graph, exceeding the u32 {count_noun} count the engine addresses"
        ))
    })
}

/// Resolve the `enumerated`-declared TRAINING forward-pass count once the node
/// graph exists, via the shared guard [`enumerated_admissible_count`]: any
/// derived count `>= 1` executes — the enumerated all-paths forward engine is
/// the consumer.
///
/// # Errors
///
/// See [`enumerated_admissible_count`].
fn resolve_enumerated_training_count(node_graph: &NodeGraph) -> Result<u32, SddpError> {
    enumerated_admissible_count(node_graph, "training", "forward-pass")
}

/// Resolve the `enumerated`-declared SIMULATION scenario count once the node
/// graph exists, via the shared guard [`enumerated_admissible_count`]: any
/// derived count `>= 1` executes — the node-native census simulation engine is
/// the consumer, weighting each resolved leaf path through
/// [`node_graph::Traversal::simulation_weighting`].
///
/// # Errors
///
/// See [`enumerated_admissible_count`].
fn resolve_enumerated_simulation_count(node_graph: &NodeGraph) -> Result<u32, SddpError> {
    enumerated_admissible_count(node_graph, "simulation", "scenario")
}

/// The first node pinning an [`OpeningSource::External`] scenario column, in
/// canonical position order — the shared trigger condition
/// [`reject_scenario_id_under_sampled_selection`] and
/// [`reject_insample_class_under_external_nodes`] both gate on.
fn find_external_bound_node(node_graph: &NodeGraph) -> Option<(NodePos, &NodeRuntime)> {
    node_graph
        .nodes
        .iter_indexed()
        .find(|(_, n)| n.openings.source == OpeningSource::External)
}

/// Reject a node carrying a scenario pointer under sampled forward selection: a
/// node's `scenario_id` (surfaced as an `External` opening) selects a
/// deterministic external-library column, which only the enumerated forward
/// engine consumes. Under sampled forwards every node draws its openings by hash,
/// so a declared pointer would be validated at load and then silently ignored;
/// an explicit rejection closes that footgun. Enumerated selection, or a graph
/// carrying no external-bound node, ⇒ `Ok(())`.
///
/// # Errors
///
/// Returns [`SddpError::Validation`] naming the first offending node id, its
/// stage, and the admitting condition (enumerated forward selection).
fn reject_scenario_id_under_sampled_selection(
    node_graph: &NodeGraph,
    training_enumerated: bool,
) -> Result<(), SddpError> {
    if training_enumerated {
        return Ok(());
    }
    if let Some((pos, node)) = find_external_bound_node(node_graph) {
        return Err(SddpError::Validation(format!(
            "node {} (stage {}) declares a scenario_id but training uses sampled forward \
             selection; scenario_id requires enumerated selection",
            node_graph.node_ids[pos], node.stage
        )));
    }
    Ok(())
}

/// Reject a non-empty in-sample class alongside an external-column node graph. An
/// [`OpeningSource::External`] node pins a scenario column that only the external
/// libraries carry; a class with real entities drawing under
/// [`SamplingScheme::InSample`] instead reads the generated opening tree at that
/// column offset, silently sampling a wrong opening (or, where the tree lacks that
/// column, tripping the sampler's opening-range assert). The mixed config is
/// unsupported: for an external-column graph every non-empty class must draw
/// external. A zero-entity class draws nothing and is exempt (the degenerate
/// no-entity class an all-external study still carries).
///
/// Takes each class's `(scheme, entity_count)` directly so it is unit-testable
/// without a [`StochasticContext`].
///
/// # Errors
///
/// Returns [`SddpError::Validation`] naming the first offending class and the
/// admitting condition (all non-empty classes external).
fn reject_insample_class_under_external_nodes(
    node_graph: &NodeGraph,
    inflow: (Option<SamplingScheme>, usize),
    load: (Option<SamplingScheme>, usize),
    ncs: (Option<SamplingScheme>, usize),
) -> Result<(), SddpError> {
    let Some((pos, node)) = find_external_bound_node(node_graph) else {
        return Ok(());
    };
    for (class, (scheme, count)) in [("inflow", inflow), ("load", load), ("ncs", ncs)] {
        if count > 0 && scheme == Some(SamplingScheme::InSample) {
            return Err(SddpError::Validation(format!(
                "node {} (stage {}) pins an external scenario column, but the {class} class draws \
                 {count} entities under in-sample selection; an external-column node graph admits \
                 only all-external non-empty classes (a zero-entity class is exempt) — set the \
                 {class} class to external selection",
                node_graph.node_ids[pos], node.stage
            )));
        }
    }
    Ok(())
}

/// Reject an `enumerated` graph whose branching is expressed as within-node
/// openings rather than structurally as distinct nodes: every enumerated axis
/// (training's forward engine, the census simulation driver) solves each node
/// once per distinct incoming state and does not enumerate a node's own
/// opening set, so a `|Ω_n| > 1` node would be sampled at a single realization
/// while the exact bound weights it as if fully enumerated. Declare the
/// branching structurally (one realization per node) or use sampled
/// selection.
///
/// # Errors
///
/// Returns [`SddpError::Validation`] naming the first offending node id, its
/// stage, and its opening count.
fn reject_within_node_opening_enumeration(node_graph: &NodeGraph) -> Result<(), SddpError> {
    if let Some((pos, node)) = node_graph
        .nodes
        .iter_indexed()
        .find(|(_, n)| n.openings.len > 1)
    {
        return Err(SddpError::Validation(format!(
            "enumerated scenario selection requires a singleton within-node opening set at \
             every node, but node id {} (stage {}) carries {} openings; within-node weighted \
             opening enumeration is not yet wired — declare the branching structurally (one \
             realization per node) or use sampled selection",
            node_graph.node_ids[pos], node.stage, node.openings.len
        )));
    }
    Ok(())
}

/// Reject an `enumerated` graph carrying a recombination join — a node reached
/// from two or more predecessor nodes (in-degree ≥ 2, counting how many
/// successor edges name it as a child). Every enumerated axis reconstructs
/// each visited node's incoming state through the single-predecessor
/// [`node_graph::NodeGraph::build_parent_map`] (via [`EnumeratedPlan`]); a multi-parent
/// node would, in a release build, be solved once under one arbitrarily
/// chosen parent's outgoing state while paths arriving through its other
/// parent silently read that wrong state — an invalid exact bound, not a
/// compile error. This setup-time guard precedes and makes release-active
/// `build_parent_map`'s single-predecessor `debug_assert`. Sampled selection is
/// unaffected: it carries each trajectory's own incoming state and resolves
/// recombination natively.
///
/// # Errors
///
/// Returns [`SddpError::Validation`] naming the first offending node id and its
/// stage (sibling to the within-node-opening rejection above).
fn reject_recombining_node_enumeration(node_graph: &NodeGraph) -> Result<(), SddpError> {
    let mut in_degree: TypedVec<NodePos, usize> = vec![0usize; node_graph.nodes.len()].into();
    for succ in node_graph.successors.iter().flatten() {
        in_degree[succ.child] += 1;
    }
    if let Some(pos) = in_degree.iter().position(|&d| d >= 2).map(NodePos) {
        return Err(SddpError::Validation(format!(
            "enumerated scenario selection requires a single-predecessor (tree) policy graph, \
             but node id {} (stage {}) is reached from {} predecessor nodes (a recombination \
             join); per-prefix state reconstruction for a multi-parent node is not yet wired — \
             use sampled selection, which handles recombination, or declare a non-recombining \
             graph (sibling requirement: a singleton within-node opening set at every node)",
            node_graph.node_ids[pos], node_graph.nodes[pos].stage, in_degree[pos]
        )));
    }
    Ok(())
}

fn build_entity_counts(system: &System) -> EntityCounts {
    EntityCounts {
        hydro_ids: system.hydros().iter().map(|h| h.id.0).collect(),
        hydro_productivities: vec![0.0; system.hydros().len()],
        thermal_ids: system.thermals().iter().map(|t| t.id.0).collect(),
        line_ids: system.lines().iter().map(|l| l.id.0).collect(),
        bus_ids: system.buses().iter().map(|b| b.id.0).collect(),
        pumping_station_ids: system.pumping_stations().iter().map(|p| p.id.0).collect(),
        contract_ids: system.contracts().iter().map(|c| c.id.0).collect(),
        non_controllable_ids: system
            .non_controllable_sources()
            .iter()
            .map(|n| n.id.0)
            .collect(),
    }
}

/// Build the per-station pumping power-consumption rates \[MW/(m³/s)\].
///
/// ID-sorted parallel to `EntityCounts::pumping_station_ids` (both derive from the
/// canonical ID-ordered `system.pumping_stations()` slice), so a row's position
/// matches its station ID's position in `pumping_station_ids`.
fn build_pumping_consumption(system: &System) -> Vec<f64> {
    system
        .pumping_stations()
        .iter()
        .map(|p| p.consumption_mw_per_m3s)
        .collect()
}

/// Build the per-stage RESOLVED contract prices \[$/`MWh`\], per block.
///
/// Outer index is the study-stage index `t` (0-based, matching
/// [`ResolvedBounds`](cobre_core::ResolvedBounds)'s contract stage axis); each
/// inner slice is flat with the per-stage stride `geometry_per_stage[t].n_blks`
/// — index `c * n_blks + blk`, `c` ID-sorted parallel to `system.contracts()`
/// (the same order `EntityCounts::contract_ids` is built in) — carrying
/// `contract_bounds_at_block(c, t, blk).price_per_mwh`. Empty inner slices for
/// a contract-free system or a zero-block stage.
fn build_contract_prices_per_stage(
    system: &System,
    geometry_per_stage: &[StageGeometry],
) -> Vec<Vec<f64>> {
    let bounds = system.bounds();
    let n_contracts = system.contracts().len();
    (0..geometry_per_stage.len())
        .map(|t| {
            let n_blks = geometry_per_stage[t].n_blks;
            (0..n_contracts)
                .flat_map(|c| {
                    (0..n_blks)
                        .map(move |blk| bounds.contract_bounds_at_block(c, t, blk).price_per_mwh)
                })
                .collect()
        })
        .collect()
}

/// Build the per-contract `(ContractType, per-family slot)`, ID-sorted
/// parallel to `system.contracts()` — the same order `EntityCounts::contract_ids`
/// is built in — from [`contract_family_slot`], the LP builder's own slot
/// derivation.
fn build_contract_slots(system: &System) -> Vec<(ContractType, usize)> {
    let contracts = system.contracts();
    (0..contracts.len())
        .map(|c| contract_family_slot(contracts, c))
        .collect()
}

/// Map each entity's declared numeric ID to its position in a canonically
/// ordered slice (`System::hydros()` / `System::thermals()`).
///
/// Canonical order sorts by `(operational_start_date, id)`
/// (`cobre_core::system::builder::sort_canonical`), which is id-ascending only
/// when every entity shares one operational start date. A staggered-
/// commissioning system (filling reservoirs, future-entry plants) breaks that
/// coincidence, so any id-keyed initial-condition lookup MUST resolve through
/// this map — `binary_search_by_key` over the canonical slice itself silently
/// returns `Err` (or the wrong index) for an out-of-id-order entry, dropping
/// its seed to the default `0.0`.
fn id_to_position<T>(entities: &[T], id_of: impl Fn(&T) -> i32) -> HashMap<i32, usize> {
    entities
        .iter()
        .enumerate()
        .map(|(idx, e)| (id_of(e), idx))
        .collect()
}

/// The contiguous study-stage slice (`Stage::id >= 0`), found by position since
/// study stages are a contiguous suffix of `System::stages()`. Empty when the
/// system declares no study stages.
fn study_stages_slice(system: &System) -> &[Stage] {
    match system.stages().iter().position(|s| s.id >= 0) {
        Some(idx) => &system.stages()[idx..],
        None => &[],
    }
}

/// Project the stage-0 initial (incoming) state onto the stage-0 admissible box
/// for the box-stable families — storage (and its `PreFilling` seed) and
/// travel-time buckets — the setup-time analog of the read-back seam's clamp on
/// the OUTGOING state. Inflow lags are unbounded, so the box leaves them
/// untouched. The commitment-hold ring is deliberately NOT clamped here: the
/// stage-0 box is anchored on the ring's OUTGOING delivery window, so its
/// residue-0 slot bounds a later delivery than the incoming seed carried there —
/// that family is projected onto its own delivery-stage bound at seed time in
/// [`build_initial_state`] instead.
fn canonicalize_initial_state(state: &mut [f64], layout: &StateSpace, stage0_box: &StateBox) {
    for j in layout
        .storage
        .clone()
        .chain(layout.transit_buckets_out.clone())
    {
        state[j] = state[j].clamp(stage0_box.lower[j], stage0_box.upper[j]);
    }
}

/// Build the initial state vector from the system's initial conditions.
///
/// Layout `[storage(0..N), lags(N..N*(1+L))]` (N hydros, L = max PAR order),
/// storage indexed by each hydro's position in `system.hydros()`'s canonical
/// order. Lag slots come from `derived_lag_values` (entity-major,
/// `derived_lag_values[pos * L + lag]`, lag 0 = most recent) — already
/// pre-ordered by canonical hydro position at its single derivation site
/// ([`derive_inflow_seeds`]), so `pos` here needs no id lookup. Storage-only
/// when `max_par_order == 0`.
fn build_initial_state(
    system: &System,
    study_dims: &StudyDimensions,
    layout: &StateSpace,
    derived_lag_values: &[f64],
) -> Vec<f64> {
    let mut state = vec![0.0_f64; layout.n_state];
    let hydros = system.hydros();
    let hydro_positions = id_to_position(hydros, |h: &Hydro| h.id.0);
    let ic = system.initial_conditions();

    for hs in &ic.storage {
        if let Some(&idx) = hydro_positions.get(&hs.hydro_id.0) {
            state[layout.storage_state_dim(HydroSys::new(idx)).get()] = hs.value_hm3;
        }
    }

    for hs in &ic.filling_storage {
        // The seed writes the same coordinate the PreFilling pin
        // (`fill_prefilling_shortcircuit`) freezes to `[seed, seed]`; do not merge
        // the two collections or re-index the column — a separate index would
        // silently desync from that pin.
        if let Some(&idx) = hydro_positions.get(&hs.hydro_id.0) {
            state[layout.storage_state_dim(HydroSys::new(idx)).get()] = hs.value_hm3;
        }
    }

    if layout.max_par_order > 0 {
        let n_h = layout.hydro_count;
        let l = layout.max_par_order;
        for idx in 0..n_h {
            for lag in 0..l {
                state[layout.lag_state_dim(lag, HydroSys::new(idx)).get()] =
                    derived_lag_values[idx * l + lag];
            }
        }
    }

    if layout.n_anticipated > 0 && layout.k_max > 0 {
        let thermals = system.thermals();
        let thermal_positions = id_to_position(thermals, |t: &Thermal| t.id.0);
        let calendar = StageCalendar::new(study_stages_slice(system));
        for history in &ic.past_anticipated_commitments {
            let Some(&global_idx) = thermal_positions.get(&history.thermal_id.0) else {
                // Defense-in-depth — the cobre-io validator rejects an unknown ID in
                // production.
                continue;
            };
            let Some(local_idx) = study_dims
                .anticipated_plants
                .local_of(ThermalSys::new(global_idx))
                .map(AnticipatedLocal::get)
            else {
                // Not one of AnticipatedPlants::build's plants — skip.
                continue;
            };
            // A covered stage at or beyond K_i is a resolver/validator desync,
            // unreachable through valid input — cobre-io's coverage rule rejects
            // it before setup runs.
            let k_i = layout.anticipated_lead_stages[local_idx];
            let window = DatedWindow {
                start_date: history.start_date,
                end_date: history.end_date,
            };
            #[expect(
                clippy::float_cmp,
                reason = "whole-day-hours coverage makes a full-coverage ratio exactly 1.0"
            )]
            for (slot, fraction) in calendar.coverage(&window).into_iter().enumerate() {
                if fraction == 1.0 {
                    if slot < k_i {
                        let off = layout.commit_out.start
                            + layout.commitment_hold_in_study_offset(local_idx, slot);
                        // Project the seed onto its delivery stage's generation
                        // bound — the setup-time analog of the read-back seam's
                        // clamp — so a sub-tolerance input overshoot cannot drive
                        // the no-slack fishing equality at stage `slot` infeasible.
                        // The delivery-stage bound, NOT the stage-0 state box the
                        // storage/bucket families clamp against: that box is
                        // anchored on the ring's OUTGOING delivery window, so its
                        // residue-0 slot bounds a later delivery than the seed held
                        // here.
                        let cap = system.bounds().thermal_block_base(global_idx, slot);
                        state[off] = history
                            .value_mw
                            .clamp(cap.min_generation_mw, cap.max_generation_mw);
                    } else {
                        debug_assert!(
                            false,
                            "covered stage beyond plant's own lead: plant local_idx={local_idx}, slot={slot}, K_i={k_i}, k_max={}",
                            layout.k_max
                        );
                    }
                }
            }
            #[expect(
                clippy::float_cmp,
                reason = "padding slots must be exactly 0.0, because a non-zero value corrupts the ring buffer and makes the LP infeasible"
            )]
            for slot in k_i..layout.k_max {
                let off = layout.commit_out.start
                    + layout.commitment_hold_in_study_offset(local_idx, slot);
                debug_assert_eq!(
                    state[off], 0.0,
                    "padding slot must be zero: plant local_idx={local_idx}, slot={slot}, K_i={k_i}, k_max={}",
                    layout.k_max
                );
            }
        }
    }

    state
}

/// Unroll every declared arc's `past_defluences` windows into the stage-0
/// incoming bucket seed, in [`bucket_topology::TransitBucketTopology::column_order`]
/// order. Runs single-threaded in that canonical order — never a
/// rank-count-dependent parallel reduction.
///
/// Each window `[start_date, end_date)` for upstream hydro `i` contributes
/// `k_d · D_i` (`D_i` the width-scaled volume, `k_d` from
/// [`StageCalendar::hour_window_shares`] anchored at
/// `e_off = start_0 − end_date`, width `end_date − start_date`) into every
/// bucket it reaches. A hydro may carry multiple, non-contiguous windows; each
/// is `filter`ed and deposited independently — never `find`, which would
/// silently keep only the first window and drop the rest, understating the
/// seed with no error.
///
/// `cobre-io`'s `validate_travel_time` coverage gate guarantees every declared
/// arc's windows cover `[start_0 − t_v, start_0)` before this runs; there is no
/// fallback for incomplete coverage.
fn build_initial_transit_bucket_state(
    system: &System,
    topology: &bucket_topology::TransitBucketTopology,
    state: &StateSpace,
) -> Vec<f64> {
    let mut seed = vec![0.0_f64; state.n_buckets];
    if state.n_buckets == 0 {
        return seed;
    }

    let Some(start_0) = study_start_date(system) else {
        debug_assert!(
            false,
            "n_buckets > 0 implies build_transit_bucket_topology sized a depth from a non-empty \
             study calendar, so at least one study stage must exist here"
        );
        return seed;
    };
    let calendar = StageCalendar::new(study_stages_slice(system));
    let ic = system.initial_conditions();
    let hydros = system.hydros();

    for (plant, local) in state.transit_bucket_plants() {
        let depth = local.len();

        for arc in topology.arcs().iter().filter(|arc| arc.downstream == plant) {
            let upstream = &hydros[arc.upstream.get()];
            let t_v = arc.travel_time_hours;

            for window in ic
                .past_defluences
                .iter()
                .filter(|w| w.hydro_id == upstream.id)
            {
                debug_assert!(
                    window.end_date <= start_0,
                    "past_defluences window must end at or before start_0 ({start_0}); \
                     cobre-io's validate_travel_time row-5b gate guarantees this"
                );
                let e_off = hours_between(start_0, window.end_date);
                let width = hours_between(window.end_date, window.start_date);
                let volume = width * M3S_TO_HM3 * window.value_m3s;

                let k = calendar.hour_window_shares(t_v, e_off, width);
                for (transit_bucket_offset, &k_val) in k.iter().enumerate().take(depth) {
                    if k_val != 0.0 {
                        seed[local.start + transit_bucket_offset] += k_val * volume;
                    }
                }
            }
        }
    }

    debug_assert_eq!(seed.len(), topology.n_buckets());
    seed
}

/// The first study stage's (`id >= 0`, lowest `id`) start date — `start_0`, the
/// anchor every `past_defluences` window's `(e_off, width)` measures against.
/// `None` only when the system declares no study stages.
fn study_start_date(system: &System) -> Option<NaiveDate> {
    system
        .stages()
        .iter()
        .filter(|s| s.id >= 0)
        .min_by_key(|s| s.id)
        .map(|s| s.start_date)
}

/// Hours of wall clock between `earlier` and `later` (`later − earlier`),
/// positive when `earlier` precedes `later`.
#[expect(
    clippy::cast_precision_loss,
    reason = "pre-study spans are years long, far inside f64's exact-integer range"
)]
fn hours_between(later: NaiveDate, earlier: NaiveDate) -> f64 {
    (later - earlier).num_hours() as f64
}

/// Write the travel-time bucket seed into `state`'s declared `transit_buckets_out`
/// slots — the same index space [`StateSpace::state_to_lp_incoming_column`]
/// remaps to the pinned `transit_buckets_in` LP column, so no separate pin wiring is
/// needed beyond this splice.
fn splice_transit_bucket_seed(
    state: &mut [f64],
    layout: &StateSpace,
    system: &System,
    topology: &bucket_topology::TransitBucketTopology,
) {
    let seed = build_initial_transit_bucket_state(system, topology, layout);
    state[layout.transit_buckets_out.clone()].copy_from_slice(&seed);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests;

/// Round-trip fidelity: the rolling-seed emitter's output, re-anchored at the
/// next run's `start_0`, must reproduce the identical
/// [`build_initial_transit_bucket_state`] seed a direct re-anchoring of the
/// SAME underlying history (pre-study `past_defluences` plus the elapsed
/// in-study releases) would produce — the property that lets a rolling run
/// hand off water state across runs with no separate input path.
#[cfg(test)]
mod transit_seed_round_trip_tests {
    use chrono::{Duration, NaiveDate};
    use cobre_core::entities::bus::{Bus, DeficitSegment};
    use cobre_core::entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties};
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };
    use cobre_core::{EntityId, HydroPastDefluence, InitialConditions, System, SystemBuilder};

    use super::{TransitSeedArc, build_initial_transit_bucket_state};
    use crate::bucket_topology;
    use crate::simulation::extraction::build_transit_seed;
    use crate::simulation::types::{SimulationHydroResult, SimulationStageResult};
    use crate::time_value::DeliveryCalendar;

    fn date(y: i32, m: u32, d: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, d).unwrap_or_else(|| unreachable!("hardcoded date is valid"))
    }

    fn zero_penalties() -> HydroPenalties {
        HydroPenalties {
            spillage_cost: 0.0,
            diversion_cost: 0.0,
            turbined_cost: 0.0,
            storage_violation_below_cost: 0.0,
            filling_target_violation_cost: 0.0,
            turbined_violation_below_cost: 0.0,
            outflow_violation_below_cost: 0.0,
            outflow_violation_above_cost: 0.0,
            generation_violation_below_cost: 0.0,
            evaporation_violation_cost: 0.0,
            water_withdrawal_violation_cost: 0.0,
            water_withdrawal_violation_pos_cost: 0.0,
            water_withdrawal_violation_neg_cost: 0.0,
            evaporation_violation_pos_cost: 0.0,
            evaporation_violation_neg_cost: 0.0,
            inflow_nonnegativity_cost: 0.0,
        }
    }

    fn hydro(id: i32, downstream_id: Option<i32>, travel_time_hours: Option<f64>) -> Hydro {
        let mut h = Hydro {
            unit_groups: Vec::new(),
            id: EntityId(id),
            name: format!("H{id}"),
            operational_start_date: date(2024, 1, 1),
            downstream_id: downstream_id.map(EntityId),
            travel_time_hours,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 100.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 100.0,
            specific_productivity_mw_per_m3s_per_m: None,
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: zero_penalties(),
        };
        h.declare_mirror_unit_group(EntityId(1));
        h
    }

    /// One study stage, `id`-indexed from 0, a single `hours`-long block, each
    /// anchored one real calendar day apart (`NaiveDate` has no sub-day
    /// resolution; `StageCalendar::hour_window_shares` reads only
    /// `duration_hours`, never the calendar span).
    fn stages_from(start: NaiveDate, n: i32, hours: f64) -> Vec<Stage> {
        (0..n)
            .map(|id| {
                let start_date = start + Duration::days(i64::from(id));
                Stage {
                    index: usize::try_from(id).unwrap_or(0),
                    id,
                    start_date,
                    end_date: start_date + Duration::days(1),
                    season_id: None,
                    blocks: vec![Block {
                        index: 0,
                        name: "FLAT".to_string(),
                        duration_hours: hours,
                    }],
                    block_mode: BlockMode::Parallel,
                    state_config: StageStateConfig {
                        storage: true,
                        inflow_lags: false,
                    },
                    risk_config: StageRiskConfig::Expectation,
                    scenario_config: ScenarioSourceConfig {
                        branching_factor: 1,
                        noise_method: NoiseMethod::Saa,
                    },
                }
            })
            .collect()
    }

    fn build_system(
        hydros: Vec<Hydro>,
        stages: Vec<Stage>,
        past: Vec<HydroPastDefluence>,
    ) -> System {
        let bus = Bus {
            id: EntityId(1),
            name: "B1".to_string(),
            operational_start_date: date(2024, 1, 1),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
        };
        SystemBuilder::new()
            .buses(vec![bus])
            .hydros(hydros)
            .stages(stages)
            .initial_conditions(InitialConditions {
                past_defluences: past,
                ..InitialConditions::default()
            })
            .build()
            .expect("valid system")
    }

    const UPSTREAM_ID: i32 = 2;
    const DOWNSTREAM_ID: i32 = 1;

    fn hydros() -> Vec<Hydro> {
        vec![
            hydro(DOWNSTREAM_ID, None, None),
            hydro(UPSTREAM_ID, Some(DOWNSTREAM_ID), Some(100.0)),
        ]
    }

    fn stage_release(stage_id: u32, hydro_id: i32, rate_m3s: f64) -> SimulationStageResult {
        SimulationStageResult {
            stage_id,
            node_id: crate::setup::NodeId(i32::try_from(stage_id).unwrap_or(0)),
            costs: vec![],
            hydros: vec![SimulationHydroResult {
                stage_id,
                block_id: Some(0),
                hydro_id,
                turbined_m3s: rate_m3s,
                spillage_m3s: 0.0,
                evaporation_m3s: None,
                diverted_inflow_m3s: None,
                diverted_outflow_m3s: None,
                incremental_inflow_m3s: 0.0,
                inflow_m3s: 0.0,
                storage_initial_hm3: 0.0,
                storage_final_hm3: 0.0,
                generation_mw: 0.0,
                equivalent_productivity_mw_per_m3s: 0.0,
                accumulated_productivity_mw_per_m3s: 0.0,
                incremental_inflow_energy_mw: 0.0,
                stored_energy_initial_mwh: 0.0,
                stored_energy_final_mwh: 0.0,
                spillage_cost: 0.0,
                water_value_per_hm3: 0.0,
                storage_binding_code: 0,
                operative_state_code: 0,
                turbined_slack_m3s: 0.0,
                outflow_slack_below_m3s: 0.0,
                outflow_slack_above_m3s: 0.0,
                generation_slack_mw: 0.0,
                storage_violation_below_hm3: 0.0,
                filling_target_violation_hm3: 0.0,
                evaporation_violation_pos_m3s: 0.0,
                evaporation_violation_neg_m3s: 0.0,
                inflow_nonnegativity_slack_m3s: 0.0,
                water_withdrawal_violation_pos_m3s: 0.0,
                water_withdrawal_violation_neg_m3s: 0.0,
                integrated_equivalent_productivity_mw_per_m3s: 0.0,
                integrated_accumulated_productivity_mw_per_m3s: 0.0,
                stored_energy_initial_mw: 0.0,
                stored_energy_final_mw: 0.0,
            }],
            hydro_bus_generation: vec![],
            thermals: vec![],
            exchanges: vec![],
            buses: vec![],
            pumping_stations: vec![],
            contracts: vec![],
            non_controllables: vec![],
            inflow_lags: vec![],
            transit_buckets: vec![],
            generic_violations: vec![],
            anticipated_lanes: vec![],
        }
    }

    /// `t_v = 100h` exceeds the 48h in-study horizon, exercising the stitch:
    /// the emitted windows must cover both the elapsed in-study releases and
    /// the run's own pre-study `past_defluences` tail. Re-anchoring the SAME
    /// underlying history (the pre-study window plus the two in-study
    /// releases) at `study_end` directly must give the identical seed the
    /// emitted windows reproduce when fed to a continuing run starting there.
    #[test]
    fn emitted_windows_reproduce_the_directly_reanchored_seed() {
        let study_start_a = date(2024, 1, 1);
        let study_end_a = study_start_a + Duration::days(2); // 2 stages, 24h each
        let pre_study_window = HydroPastDefluence {
            hydro_id: EntityId(UPSTREAM_ID),
            start_date: study_start_a - Duration::days(2),
            end_date: study_start_a,
            value_m3s: 50.0,
        };

        let stages_a = stages_from(study_start_a, 2, 24.0);
        let study_stage_dates: Vec<(NaiveDate, NaiveDate)> = stages_a
            .iter()
            .map(|s| (s.start_date, s.end_date))
            .collect();
        let stage_results = vec![
            stage_release(0, UPSTREAM_ID, 100.0),
            stage_release(1, UPSTREAM_ID, 200.0),
        ];
        let arcs = [TransitSeedArc {
            upstream_hydro_id: UPSTREAM_ID,
            travel_time_hours: 100.0,
        }];
        let block_hours = vec![vec![24.0]; 2];

        let emitted = build_transit_seed(
            &stage_results,
            &study_stage_dates,
            &arcs,
            std::slice::from_ref(&pre_study_window),
            &block_hours,
        );
        assert_eq!(
            emitted.len(),
            3,
            "t_v=100h must pull in both in-study stages and the pre-study tail"
        );

        let system_b = build_system(
            hydros(),
            stages_from(study_end_a, 1, 24.0),
            emitted
                .into_iter()
                .map(|w| HydroPastDefluence {
                    hydro_id: EntityId(w.hydro_id),
                    start_date: w.start_date,
                    end_date: w.end_date,
                    value_m3s: w.value_m3s,
                })
                .collect(),
        );
        let calendar_b = DeliveryCalendar::from_system(&system_b);
        let topology_b =
            bucket_topology::build_transit_bucket_topology(&system_b, &calendar_b, false);
        let state_b = crate::test_support::bucket_seed_state(&system_b, &topology_b);
        let seed_from_emission =
            build_initial_transit_bucket_state(&system_b, &topology_b, &state_b);

        let system_reference = build_system(
            hydros(),
            stages_from(study_end_a, 1, 24.0),
            vec![
                pre_study_window,
                HydroPastDefluence {
                    hydro_id: EntityId(UPSTREAM_ID),
                    start_date: stages_a[0].start_date,
                    end_date: stages_a[0].end_date,
                    value_m3s: 100.0,
                },
                HydroPastDefluence {
                    hydro_id: EntityId(UPSTREAM_ID),
                    start_date: stages_a[1].start_date,
                    end_date: stages_a[1].end_date,
                    value_m3s: 200.0,
                },
            ],
        );
        let calendar_reference = DeliveryCalendar::from_system(&system_reference);
        let topology_reference = bucket_topology::build_transit_bucket_topology(
            &system_reference,
            &calendar_reference,
            false,
        );
        let state_reference =
            crate::test_support::bucket_seed_state(&system_reference, &topology_reference);
        let seed_reference = build_initial_transit_bucket_state(
            &system_reference,
            &topology_reference,
            &state_reference,
        );

        assert_eq!(seed_from_emission.len(), seed_reference.len());
        for (a, b) in seed_from_emission.iter().zip(&seed_reference) {
            assert!(
                (a - b).abs() < 1e-9,
                "round-trip seed must match the directly re-anchored reference to 1e-9: \
                 {seed_from_emission:?} vs {seed_reference:?}"
            );
        }
        assert!(
            seed_reference.iter().any(|&v| v.abs() > f64::EPSILON),
            "the reference seed must be non-degenerate (not all-zero) for this to be a \
             meaningful fidelity check"
        );
    }
}

/// The scalar-parameter table is now a `StudySetup` constructor input (never an
/// empty placeholder — see [`StudyParams::from_config`]): each gap class
/// `build_resolved_parameters` can raise surfaces through
/// [`StudySetup::new_with_boundary_requirements`], and a generic constraint
/// referencing an id the table never resolved fails loud via
/// `check_scalar_parameters_present` instead of reaching
/// [`ResolvedParameters::get`]'s `0.0` sentinel.
#[cfg(test)]
mod scalar_parameter_construction_tests {
    use cobre_core::scenario::SamplingScheme;
    use cobre_core::{
        AffineBound, ComputedParameter, ConstraintExpression, EntityId, GenericConstraint,
        ParameterKind, ScalarParameter, SlackConfig, SystemBuilder,
    };
    use cobre_io::Config;
    use cobre_stochastic::{ClassSchemes, OpeningTreeInputs, build_stochastic_context};

    use super::{BoundaryStateRequirements, StudySetup};
    use crate::SddpError;
    use crate::hydro_models::PrepareHydroModelsResult;
    use crate::test_support::{k_fan_config, k_fan_system};

    fn build(
        system: &cobre_core::System,
        config: &Config,
        scalar_parameters: Vec<ScalarParameter>,
    ) -> Result<StudySetup, SddpError> {
        let stochastic = build_stochastic_context(
            system,
            42,
            None,
            &[],
            &[],
            OpeningTreeInputs::default(),
            ClassSchemes {
                inflow: Some(SamplingScheme::InSample),
                load: Some(SamplingScheme::InSample),
                ncs: Some(SamplingScheme::InSample),
            },
        )
        .expect("build_stochastic_context must succeed for a valid fixture system");
        let hydro_models = PrepareHydroModelsResult::default_from_system(system);
        StudySetup::new_with_boundary_requirements(
            system,
            config,
            stochastic,
            hydro_models,
            BoundaryStateRequirements::present(0),
            scalar_parameters,
        )
    }

    fn scalar_param(kind: ParameterKind) -> ScalarParameter {
        ScalarParameter {
            id: EntityId(1),
            name: "probe".to_string(),
            kind,
        }
    }

    #[test]
    fn missing_season_rejects_at_construction() {
        let system = k_fan_system(3, false);
        let config = k_fan_config(1, 1);
        // Every k_fan_system stage has `season_id: None`, resolving to season 0
        // (`unwrap_or(0)`); a Seasonal table with no season-0 entry misses.
        let table = vec![scalar_param(ParameterKind::Seasonal {
            values: vec![(1, 100.0)],
        })];
        let err = build(&system, &config, table).expect_err("must reject the season gap");
        assert!(
            matches!(err, SddpError::Validation(ref msg) if msg.contains("season")),
            "expected a MissingSeason message, got: {err:?}"
        );
    }

    #[test]
    fn per_stage_block_coverage_gap_rejects_at_construction() {
        let system = k_fan_system(3, false);
        let config = k_fan_config(1, 1);
        // Every k_fan_system stage has exactly one block; an empty PerStageBlock
        // table covers no (stage, block) cell.
        let table = vec![scalar_param(ParameterKind::PerStageBlock {
            values: vec![],
        })];
        let err = build(&system, &config, table).expect_err("must reject the coverage gap");
        assert!(
            matches!(err, SddpError::Validation(ref msg) if msg.contains("not covered")),
            "expected a PerStageBlockCoverage message, got: {err:?}"
        );
    }

    #[test]
    fn missing_specific_productivity_rejects_at_construction() {
        let base = k_fan_system(3, false);
        let mut hydros = base.hydros().to_vec();
        hydros[0].specific_productivity_mw_per_m3s_per_m = None;
        let hydro_id = hydros[0].id;
        let system = SystemBuilder::new()
            .buses(base.buses().to_vec())
            .hydros(hydros)
            .stages(base.stages().to_vec())
            .inflow_models(base.inflow_models().to_vec())
            .load_models(base.load_models().to_vec())
            .bounds(base.bounds().clone())
            .penalties(base.penalties().clone())
            .initial_conditions(base.initial_conditions().clone())
            .policy_graph(base.policy_graph().clone())
            .build()
            .expect("clearing specific_productivity keeps the fixture valid");
        let config = k_fan_config(1, 1);
        let table = vec![scalar_param(ParameterKind::Computed {
            computed_spec: ComputedParameter::SpecificProductivity { hydro_id },
        })];
        let err = build(&system, &config, table).expect_err("must reject the missing rho_esp");
        assert!(
            matches!(err, SddpError::Validation(ref msg) if msg.contains("specific productivity")),
            "expected a MissingSpecificProductivity message, got: {err:?}"
        );
    }

    #[test]
    fn generic_constraint_unresolved_parameter_fails_loud_at_construction() {
        let base = k_fan_system(3, false);
        let constraint = GenericConstraint {
            id: EntityId(500),
            name: "probe_constraint".to_string(),
            description: None,
            expression: ConstraintExpression { terms: vec![] },
            slack: SlackConfig {
                enabled: false,
                penalty: None,
            },
            bound_lower_affine: Some(AffineBound::single(EntityId(999))),
            bound_upper_affine: None,
        };
        let system = SystemBuilder::new()
            .buses(base.buses().to_vec())
            .hydros(base.hydros().to_vec())
            .stages(base.stages().to_vec())
            .inflow_models(base.inflow_models().to_vec())
            .load_models(base.load_models().to_vec())
            .bounds(base.bounds().clone())
            .penalties(base.penalties().clone())
            .initial_conditions(base.initial_conditions().clone())
            .policy_graph(base.policy_graph().clone())
            .generic_constraints(vec![constraint])
            .build()
            .expect("adding a generic constraint keeps the fixture valid");
        let config = k_fan_config(1, 1);
        let err = build(&system, &config, Vec::new())
            .expect_err("must reject the unresolved parameter reference before the LP builds");
        assert!(
            matches!(err, SddpError::Validation(ref msg) if msg.contains("probe_constraint") && msg.contains("999")),
            "expected the fail-loud check naming the constraint and id=999, got: {err:?}"
        );
    }

    #[test]
    fn validate_generic_constraint_parameters_rejects_unresolved_reference_without_a_setup() {
        let base = k_fan_system(3, false);
        let constraint = GenericConstraint {
            id: EntityId(500),
            name: "probe_constraint".to_string(),
            description: None,
            expression: ConstraintExpression { terms: vec![] },
            slack: SlackConfig {
                enabled: false,
                penalty: None,
            },
            bound_lower_affine: Some(AffineBound::single(EntityId(999))),
            bound_upper_affine: None,
        };
        let system = SystemBuilder::new()
            .buses(base.buses().to_vec())
            .hydros(base.hydros().to_vec())
            .stages(base.stages().to_vec())
            .inflow_models(base.inflow_models().to_vec())
            .load_models(base.load_models().to_vec())
            .bounds(base.bounds().clone())
            .penalties(base.penalties().clone())
            .initial_conditions(base.initial_conditions().clone())
            .policy_graph(base.policy_graph().clone())
            .generic_constraints(vec![constraint])
            .build()
            .expect("adding a generic constraint keeps the fixture valid");
        let hydro_models = PrepareHydroModelsResult::default_from_system(&system);
        let err = super::validate_generic_constraint_parameters(
            &system,
            &hydro_models,
            &[],
            crate::DEFAULT_COST_SCALE_FACTOR,
        )
        .expect_err("the validate-time guard must reject the unresolved reference");
        assert!(
            matches!(err, SddpError::Validation(ref msg) if msg.contains("probe_constraint") && msg.contains("999")),
            "expected the same fail-loud message the construction path emits, got: {err:?}"
        );
    }

    #[test]
    fn validate_generic_constraint_parameters_accepts_a_gap_free_deck() {
        let system = k_fan_system(3, false);
        let hydro_models = PrepareHydroModelsResult::default_from_system(&system);
        super::validate_generic_constraint_parameters(
            &system,
            &hydro_models,
            &[],
            crate::DEFAULT_COST_SCALE_FACTOR,
        )
        .expect("a deck with no generic-constraint parameter gap must pass the guard");
    }
}

#[cfg(test)]
mod admission_gate_dcs_tests {
    use super::{CutSelectionStrategy, SddpError, admission_gate};
    use crate::risk_measure::RiskMeasure;
    use crate::stopping_rule::{StoppingMode, StoppingRule, StoppingRuleSet};

    fn no_gap_rules() -> StoppingRuleSet {
        StoppingRuleSet {
            rules: vec![StoppingRule::IterationLimit { limit: 100 }],
            mode: StoppingMode::Any,
        }
    }

    fn dynamic() -> CutSelectionStrategy {
        CutSelectionStrategy::Dynamic {
            k1: None,
            k2: 1,
            nadic: 1,
            epsilon_viol: 1e-6,
            start_iteration: 1,
        }
    }

    fn level1() -> CutSelectionStrategy {
        CutSelectionStrategy::Level1 {
            check_frequency: 5,
            tie_tolerance: 1e-10,
        }
    }

    /// Enumerated forward traversal paired with dynamic cut selection is rejected
    /// at the real admission gate, the message naming the pairing; either
    /// configuration alone is admitted, and a value-based strategy under
    /// enumerated forwards is admitted — so the rejection discriminates on the
    /// `Dynamic` variant, not on any strategy being present. Every other arm of
    /// the gate is neutralised here (expectation measures, no `gap` rule).
    #[test]
    fn admission_gate_rejects_dynamic_cut_selection_under_enumerated() {
        let measures = vec![RiskMeasure::Expectation, RiskMeasure::Expectation];
        let rules = no_gap_rules();
        let dcs = dynamic();
        let l1 = level1();

        match admission_gate(&measures, &rules, true, Some(&dcs)) {
            Err(SddpError::Validation(msg)) => {
                assert!(
                    msg.contains("dynamic cut selection"),
                    "names dynamic cut selection: {msg}"
                );
                assert!(
                    msg.contains("enumerated"),
                    "names enumerated traversal: {msg}"
                );
            }
            other => panic!("expected a Validation reject for enumerated + Dynamic, got {other:?}"),
        }

        assert!(
            admission_gate(&measures, &rules, true, None).is_ok(),
            "enumerated forwards without dynamic cut selection must be admitted"
        );
        assert!(
            admission_gate(&measures, &rules, true, Some(&l1)).is_ok(),
            "a value-based cut selection strategy under enumerated forwards must be admitted"
        );
        assert!(
            admission_gate(&measures, &rules, false, Some(&dcs)).is_ok(),
            "dynamic cut selection under sampled forwards must be admitted"
        );
    }
}
