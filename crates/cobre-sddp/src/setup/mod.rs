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
//! assert!(!setup.stage_data.stage_templates.templates.is_empty());
//! # Ok(())
//! # }
//! ```

use chrono::NaiveDate;
use cobre_core::ContractType::Import;
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
use cobre_stochastic::season_cast::{DatedWindow, StageCalendar, post_study_calendar_stages};

use crate::StageTemplates;
use crate::config::LoopParams;
use crate::resolved_parameters::{ResolvedParameters, build_resolved_parameters};
use crate::scaling_report::ScalingReport;
use crate::simulation::SimulationConfig;
use crate::solve::solver_phase::{Phase, validate_phase_solver_config};
use crate::stochastic::noise_key::build_noise_key_table;
mod accessors;
pub(crate) mod bucket_topology;
pub mod node_graph;
mod orchestration;
pub mod params;
pub(crate) mod scenario_libraries;
pub mod scenario_library_set;
pub mod stage_data;
pub mod stochastic_pipeline;
pub(crate) mod template_postprocess;

pub use node_graph::{
    EnumeratedPlan, NodeGraph, NodeId, NodeOpenings, NodePos, NodeRuntime, NodeSuccessor,
    OpeningSource, StageIdx, Traversal, TypedVec,
};
pub use params::{
    BoundaryStateRequirements, DEFAULT_COST_SCALE_FACTOR, DEFAULT_FORWARD_PASSES,
    DEFAULT_MAX_ITERATIONS, DEFAULT_SEED, SimulationEnumeratedRequest, StudyParams,
};
pub use scenario_library_set::{PhaseLibraries, ScenarioLibraries};
pub use stage_data::StageData;
pub use stochastic_pipeline::{
    PrepareStochasticResult, build_ncs_factor_entries, build_stochastic_context_for_study,
    load_load_factors_for_stochastic, prepare_stochastic, study_stage_noise_group_ids,
};

use std::collections::HashMap;
use std::path::Path;

use cobre_core::{
    AffineBound, AnticipatedConfig, CoefficientRef, EntityId, GenericConstraint, HorizonGraph,
    Hydro, HydroPastDefluence, PostStudyStages, PostStudyThermalBound, ScalarParameter, Stage,
    StageId, System, Thermal,
    scenario::{SamplingScheme, ScenarioSource},
};
use cobre_io::StageIdResolver;
use cobre_io::build_hydro_reference_volumes_resolved;
use cobre_stochastic::par::precompute::PrecomputedPar;
use cobre_stochastic::{
    ClassSchemes, ExternalScenarioLibrary, HistoricalScenarioLibrary, StochasticContext,
};

use crate::{
    config::{CutManagementConfig, EventParams},
    cut::FutureCostFunction,
    cut_selection::CutSelectionStrategy,
    energy_conversion::{EnergyConversionSet, build_energy_conversion_set},
    error::SddpError,
    horizon_mode::HorizonMode,
    hydro_models::PrepareHydroModelsResult,
    inflow_method::InflowNonNegativityMethod,
    lead_time::{AnticipatedResolution, DeliveryAxis, LeadTime, PointResolution, SpreadResolution},
    lp::builder::{M3S_TO_HM3, StateBox, build_stage_templates},
    lp::indexer::{
        AnticipatedLocal, CutStateProjection, HydroCellIndex, StateSpace, StudyDimensions,
    },
    risk_measure::{RiskMeasure, uniform_effective_measure},
    simulation::EntityCounts,
    simulation::extraction::TransitSeedArc,
    stopping_rule::{StoppingRule, StoppingRuleSet},
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
    /// Stage-indexed data: LP templates, indexer, stages, entity counts, blocks,
    /// lag transitions, noise groups, and scaling report.
    pub stage_data: stage_data::StageData,

    /// Stochastic context holding sampling distributions, libraries, and provenance.
    pub stochastic: StochasticContext,
    /// Future cost function (cut pool) updated by the backward pass during training.
    pub fcf: FutureCostFunction,
    pub(crate) initial_state: Vec<f64>,

    /// Pre-computed hydro production models (FPHA, turbine curves, etc.).
    pub hydro_models: PrepareHydroModelsResult,
    pub(crate) ncs_entity_ids_per_stage: Vec<Vec<i32>>,
    /// Stage-invariant stochastic-slot → dense NCS column index map (slot in
    /// `StochasticContext::ncs_entity_ids` id-sorted order).
    ///
    /// The NCS bound patch sites stride the per-opening cap onto
    /// `ncs_col_starts[s] + ncs_stochastic_dense_col[slot] * n_blks_s + blk`.
    /// Length equals `n_stochastic_ncs`; empty when the study has no stochastic NCS.
    pub(crate) ncs_stochastic_dense_col: Vec<usize>,
    /// Stage-invariant `(entry_stage_id, exit_stage_id)` per stochastic NCS slot
    /// (id-sorted to match `ncs_stochastic_dense_col` and the `transform_ncs_noise`
    /// buffer order).
    ///
    /// The dormant-slot `[0, 0]` cap MUST stay identical across the forward,
    /// backward, and lower-bound patch sites — the `evaluate_lower_bound`
    /// "patch NCS per opening" contract; a divergence understates the bound (D15).
    /// Length equals `n_stochastic_ncs`; empty when no stochastic NCS.
    pub(crate) ncs_stochastic_windows: Vec<(Option<i32>, Option<i32>)>,
    /// Max generation \[MW\] per stochastic NCS entity, sorted by entity ID.
    pub(crate) ncs_max_gen: Vec<f64>,
    /// Whether each stochastic NCS entity may be curtailed, aligned 1:1 with
    /// [`Self::ncs_max_gen`]. `false` = must-run: the patch sites pin
    /// `col_lower = col_upper` (not `[0, cap]`), and non-simulated must-run
    /// generation is pre-netted from load.
    pub(crate) ncs_allow_curtailment: Vec<bool>,

    /// Stage-invariant `(entry_stage_id, exit_stage_id)` per anticipated thermal,
    /// in anticipated-local order matching
    /// `stage_data.study_dims.anticipated_thermal_indices`.
    ///
    /// Threaded into the simulation
    /// [`StageExtractionSpec`](crate::simulation::extraction::StageExtractionSpec)
    /// so the anticipated-decision read gates on the same
    /// `is_anticipated_decision_active` predicate the LP builder used,
    /// keying its operation-window clause on the DELIVERY stage's `stage.id`. Empty
    /// when there are no anticipated thermals.
    pub(crate) anticipated_windows: Vec<(Option<i32>, Option<i32>)>,

    /// `study_stage_ids[t] = stage.id` per study stage index; the simulation
    /// context borrows it to map a delivery stage index to its commissioning id
    /// for the `anticipated_windows` gate.
    pub(crate) study_stage_ids: Vec<i32>,

    /// Extended delivery-stage anchors ([`build_extended_delivery_anchors`]):
    /// the `YYYYMM01` anchor of each delivery target stage, indexed by delivery
    /// target `m` (study stages then the synthetic post-study continuation).
    /// Threaded into the simulation
    /// [`SimulationOutputSpec`](crate::simulation::SimulationOutputSpec)'s
    /// `extended_delivery_anchors` so the `anticipated_lanes` extractor
    /// dates a post-study-targeted decision without re-deriving the calendar
    /// walk. Study-only when the study declares no post-study stage.
    pub(crate) extended_delivery_anchors: Vec<i32>,

    /// Declared travel-time arcs (upstream hydro id + travel time), resolved
    /// once from [`System::hydros`] ([`build_transit_seed_arcs`]). Threaded
    /// into [`SimulationOutputSpec`](crate::simulation::SimulationOutputSpec)
    /// so the rolling-seed emitter never re-derives it from `System`. Empty
    /// when the study declares no travel-time arc.
    pub(crate) transit_seed_arcs: Vec<TransitSeedArc>,

    /// This run's own `system.initial_conditions().past_defluences`, retained
    /// for the rolling-seed emitter's pre-study input-tail stitch (nonempty
    /// only when a declared arc's travel time exceeds the study horizon).
    pub(crate) past_defluences: Vec<HydroPastDefluence>,

    /// `study_stage_dates[t] = (stage.start_date, stage.end_date)` per study
    /// stage index, parallel to [`Self::study_stage_ids`]. Threaded into
    /// [`SimulationOutputSpec`](crate::simulation::SimulationOutputSpec) so the
    /// rolling-seed emitter's per-stage windows never re-derive the calendar.
    pub(crate) study_stage_dates: Vec<(NaiveDate, NaiveDate)>,

    /// Resolved `(parameter_id, stage)` coefficients; consumed by the LP builder
    /// and the generic-constraint echo.
    pub(crate) resolved_parameters: ResolvedParameters,

    /// Sampling schemes and pre-built libraries for training and simulation phases.
    pub scenario_libraries: ScenarioLibraries,

    /// The runtime node graph: node identity/order, the `node → pool`
    /// map, and per-node Ω views/out-edges. Absent `nodes[]` this is the
    /// byte-exact chain degeneracy. Reached through
    /// [`crate::context::TrainingContext::node_graph`] on the hot path.
    pub node_graph: node_graph::NodeGraph,
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

    /// Two-stage cut management pipeline configuration.
    pub(crate) cut_management: CutManagementConfig,

    /// Pure-data event flags (output-side).
    ///
    /// Runtime handles (`event_sender`, `shutdown_flag`) and deferred fields
    /// (`checkpoint_interval`) are excluded and supplied per-call in
    /// [`StudySetup::train`].
    pub(crate) events: EventParams,

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

    /// Study horizon mode (finite vs. infinite-horizon approximation).
    pub(crate) horizon: HorizonMode,
    /// Inflow non-negativity enforcement method.
    pub(crate) inflow_method: InflowNonNegativityMethod,

    /// Derived per-hydro PAR lag-slot and accumulator seeds ([`derive_inflow_seeds`]),
    /// applied to the stage-0 lag block and to every trajectory start in the
    /// forward pass and simulation pipeline instead of zero-filling. All-zero
    /// when the derivation has no resolvable data.
    pub(crate) derived_inflow_seeds: DerivedInflowSeeds,

    /// PAR order of the downstream (coarser) resolution model. Non-zero only when
    /// the study includes stages with `season_id >= 12` (a monthly-to-quarterly
    /// transition); zero for uniform-resolution studies. Sizes the downstream
    /// scratch buffers via `WorkspaceSizing`.
    pub(crate) downstream_par_order: usize,

    /// Energy-conversion scalars (`ρ_eq`, `V_ref`, `Q_ref`, `ρ_acum`) per
    /// `(hydro, stage)`, consumed by the energy-balance LP constraints and
    /// inflow-/stored-energy extraction.
    pub(crate) energy_conversion: EnergyConversionSet,

    /// `V_min` (`min_storage_hm3`) per hydro, in declaration order; threaded into
    /// the simulation pipeline for stored-energy calculations.
    pub(crate) hydro_min_storage_hm3: Vec<f64>,

    /// Water travel-time in-transit bucket topology: canonical column order,
    /// global bucket count, per-stage reachability mask, and the three
    /// resolved arc tables (stage-clock weights, chronological spread,
    /// arrival density) — the single derivation site for all of them. Empty
    /// (`n_buckets == 0`) when the system declares no travel-time arc.
    // Every field is consumed via the constructor's threaded LOCAL
    // (state-layout sizing, the LP builder's arc-table threading, the bucket
    // IC seed) before this STORED field is set below; no post-construction
    // reader exists yet. `#[allow(dead_code)]` refires once one lands.
    #[allow(dead_code)]
    pub(crate) transit_bucket_topology: bucket_topology::TransitBucketTopology,

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
    /// - [`SddpError::Validation`] — if `build_stage_templates` succeeds but
    ///   the template list is empty ("system has no study stages").
    /// - [`SddpError::Solver`] — propagated from `build_stage_templates`
    ///   on LP construction failure.
    /// - [`SddpError::Validation`] — if `parse_cut_selection_config` returns
    ///   an invalid config string.
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
    /// - [`SddpError::Validation`] — if `build_stage_templates` succeeds but
    ///   the template list is empty ("system has no study stages").
    /// - [`SddpError::Solver`] — propagated from `build_stage_templates` on LP
    ///   construction failure.
    // Rationale (too_many_lines): a single linear pass building the `StudySetup`
    // literal from per-entity prep blocks; splitting it would scatter the
    // construction the literal reads.
    #[allow(clippy::too_many_lines)]
    pub fn from_broadcast_params(
        system: &System,
        mut stochastic: StochasticContext,
        config: StudyParams,
        hydro_models: PrepareHydroModelsResult,
        training_source: &ScenarioSource,
        simulation_source: &ScenarioSource,
    ) -> Result<Self, SddpError> {
        let StudyParams {
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
            export_states,
            scalar_parameters,
            training_solver_backward,
            training_solver_forward,
            simulation_solver,
            backward_scheduler,
            cost_scale_factor,
            boundary,
        } = config;

        // Fail fast on a backend-unsupported field before any template exists;
        // validation runs on every rank (`from_broadcast_params` is the shared
        // setup path), so it is deterministic across the run.
        validate_phase_solver_config(training_solver_backward.as_ref(), Phase::Backward)?;
        validate_phase_solver_config(training_solver_forward.as_ref(), Phase::Forward)?;
        validate_phase_solver_config(simulation_solver.as_ref(), Phase::Simulation)?;

        // `resolve_profile` is a pure function of the (identically broadcast)
        // config, so every rank resolving independently is sufficient — the
        // resolved `ActiveProfile` itself never needs to go on the wire.
        let backward_profile = Phase::Backward.resolve_profile(training_solver_backward.as_ref());
        let forward_profile = Phase::Forward.resolve_profile(training_solver_forward.as_ref());
        let simulation_profile = Phase::Simulation.resolve_profile(simulation_solver.as_ref());

        // Keys are a pure function of the synced tree + fixed σ, so every rank
        // computes the identical permutation and cuts stay bit-identical across
        // thread/rank counts (canonical-ω aggregation is order-independent).
        let solve_order_keys = build_noise_key_table(system, &stochastic)?;
        stochastic
            .set_solve_order(&solve_order_keys)
            .map_err(|e| SddpError::Validation(e.to_string()))?;

        // Computed here (not inside `build_energy_and_templates`) so the one
        // `TransitBucketTopology` this constructor derives from `system` also seeds the
        // `StudySetup.transit_bucket_topology` field below, with no second call.
        // `boundary.is_present()` gates the terminal deep-lag mask (the
        // Delivery-family right-boundary pricing contract) — every rank resolves it
        // identically from the broadcast config, before `inject_boundary_cuts` runs.
        let transit_bucket_topology =
            bucket_topology::build_transit_bucket_topology(system, boundary.is_present());

        // Resolved before the LP templates: none of the state dimensions depend on
        // the built LP, and `build_stage_templates` needs the finished `StateSpace`
        // threaded in as a parameter (the single role-(a) owner — see
        // `resolve_state_layout`).
        let (state_layout, hydro_count, anticipated_thermal_indices) = resolve_state_layout(
            system,
            stochastic.par(),
            &transit_bucket_topology,
            boundary.inflow_lag_depth(),
        )?;
        warn_on_boundary_absent_post_study_delivery(
            system,
            &anticipated_thermal_indices,
            &state_layout.anticipated_resolution,
            boundary.is_present(),
        );

        // The sole `derive_inflow_seeds` call site: every consumer (the lag block
        // below, `StudySetup::derived_inflow_seeds`) reads this one value — do not
        // add a second call. Computed locally on every rank from the already-
        // broadcast `system` rather than carried over the wire: the derivation is
        // a pure function of `system`, so every rank derives a bit-identical seed
        // with no extra broadcast.
        let noop_season_map = SeasonMap {
            cycle_type: Monthly,
            seasons: Vec::new(),
        };
        let season_map_ref = system
            .policy_graph()
            .season_map
            .as_ref()
            .unwrap_or(&noop_season_map);
        let derived_inflow_seeds = match system.stages().iter().find(|s| s.id >= 0) {
            None => DerivedInflowSeeds::zero(system.hydros().len(), state_layout.max_par_order),
            Some(first_stage) => derive_inflow_seeds(
                system.inflow_history(),
                &system.initial_conditions().recent_observations,
                system.hydros(),
                first_stage,
                season_map_ref,
                state_layout.max_par_order,
            ),
        };

        // Built here, before the stage templates: `TemplateBuildCtx` reads it during
        // `StageLayout::new`, and the SAME value (never rebuilt or cloned) is stored
        // on `StageData` below.
        let hydro_cell_index = HydroCellIndex::build(system.hydros());

        let EnergyAndTemplates {
            energy_conversion,
            stage_templates,
            scaling_report,
            resolved_parameters,
        } = build_energy_and_templates(
            system,
            inflow_method,
            &stochastic,
            &hydro_models,
            &scalar_parameters,
            &state_layout,
            cost_scale_factor,
            &transit_bucket_topology.per_stage_mask,
            &transit_bucket_topology.arc_stage_weights,
            &transit_bucket_topology.arc_spread_chrono,
            &transit_bucket_topology.arc_arrival_density,
            &hydro_cell_index,
        )?;

        let study_dims = build_study_dimensions(
            system,
            &stage_templates,
            inflow_method,
            hydro_count,
            anticipated_thermal_indices,
        );

        let mut initial_state = build_initial_state(
            system,
            &study_dims,
            &state_layout,
            &derived_inflow_seeds.lag_values,
        );
        splice_transit_bucket_seed(
            &mut initial_state,
            &state_layout,
            system,
            &transit_bucket_topology,
        );
        if let Some(stage0_box) = stage_templates.state_boxes.first() {
            canonicalize_initial_state(&mut initial_state, &state_layout, stage0_box);
        }

        let n_stages = stage_templates.templates.len();
        let max_iterations = max_iterations_from_rules(&stopping_rule_set);
        let fcf_capacity_iterations = max_iterations.saturating_add(1);

        let stages: Vec<Stage> = system
            .stages()
            .iter()
            .filter(|s| s.id >= 0)
            .cloned()
            .collect();
        let study_stage_ids: Vec<i32> = stages.iter().map(|s| s.id).collect();
        let study_stage_dates: Vec<(NaiveDate, NaiveDate)> =
            stages.iter().map(|s| (s.start_date, s.end_date)).collect();

        let LagData {
            stage_lag_transitions,
            noise_group_ids,
            downstream_par_order,
        } = precompute_lag_data(system, &stages, &stochastic, season_map_ref);

        let hydro_ids: Vec<EntityId> = system.hydros().iter().map(|h| h.id).collect();

        let scenario_libraries = build_scenario_libraries(
            system,
            &stages,
            &hydro_ids,
            &stochastic,
            &stage_lag_transitions,
            training_source,
            simulation_source,
            forward_passes,
            downstream_par_order,
            derived_inflow_seeds.as_seed(state_layout.max_par_order),
        )?;

        // G1: binds after `build_scenario_libraries` — an `External`-bound
        // node's Ω addresses the standardized library's raw scenario axis,
        // so binding earlier would race the library's own standardization.
        // Also binds BEFORE the FCF / cut_state_layouts construction below: the
        // pool axis they use is resolved through this graph's `node → pool` map.
        let stage_id_resolver = StageIdResolver::from_study_stage_ids(&study_stage_ids);
        let node_graph = node_graph::build_node_graph(
            system.policy_graph(),
            n_stages,
            &stage_id_resolver,
            &stochastic,
        )?;

        reject_scenario_id_under_sampled_selection(&node_graph, training_enumerated)?;
        let prov = stochastic.provenance();
        reject_insample_class_under_external_nodes(
            &node_graph,
            (prov.inflow_scheme, stochastic.n_hydros()),
            (prov.load_scheme, stochastic.n_load_buses()),
            (prov.ncs_scheme, stochastic.n_stochastic_ncs()),
        )?;

        // Resolves any `enumerated`-declared phase's actual count now that the
        // graph exists — config load could only signal the request, never the
        // count. `forward_passes`/`n_scenarios` carry a `sampled`-shaped
        // placeholder until this point when enumerated was requested.
        warn_on_enumeration_asymmetry(
            training_enumerated,
            matches!(
                simulation_enumerated,
                SimulationEnumeratedRequest::Enumerated
            ),
        );
        let forward_passes = if training_enumerated {
            resolve_enumerated_training_count(&node_graph)?
        } else {
            forward_passes
        };
        let n_scenarios = match simulation_enumerated {
            SimulationEnumeratedRequest::Enumerated => {
                resolve_enumerated_simulation_count(&node_graph)?
            }
            SimulationEnumeratedRequest::Sampled => n_scenarios,
        };

        // Resolved AFTER the guard-checked counts above (`resolve_enumerated_training_count`
        // has already run the enumerated admissibility guards for a `true`
        // `training_enumerated`), so this resolution cannot fail — it is the
        // typed reification of what the two calls above already validated.
        let traversal =
            node_graph::Traversal::resolve(&node_graph, training_enumerated, forward_passes);

        let cut_state_layouts = build_cut_state_layouts(system, &state_layout, &node_graph);
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
            node_graph::Traversal::Enumerated(_) => {
                node_graph::enumerated_pool_cut_stride(&node_graph)
            }
        };
        let fcf = FutureCostFunction::new_per_pool(
            &pool_state_dimensions,
            state_layout.n_state,
            forward_passes,
            fcf_capacity_iterations,
            &vec![0; node_graph.n_pools],
            &visit_bounds,
        );

        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        // Rejects a degenerate single-stage problem (`num_stages < 2`, no
        // predecessor to generate cuts for); reachable since the empty case rejected
        // above still leaves `n_stages == 1` possible.
        horizon.validate()?;

        let risk_measures = build_risk_measures(system);

        let NcsEntityData {
            entity_counts,
            ncs_entity_ids_per_stage,
            ncs_stochastic_dense_col,
            ncs_stochastic_windows,
            ncs_max_gen,
            ncs_allow_curtailment,
        } = build_ncs_entity_data(system, &stage_templates, &stochastic)?;
        let block_counts_per_stage: Vec<usize> = stage_templates
            .block_hours_per_stage
            .iter()
            .map(Vec::len)
            .collect();
        let max_blocks = block_counts_per_stage.iter().copied().max().unwrap_or(0);

        let pumping_consumption_mw_per_m3s = build_pumping_consumption(system);
        let contract_prices_per_stage =
            build_contract_prices_per_stage(system, n_stages, &block_counts_per_stage);
        let contract_is_import = build_contract_is_import(system);

        let anticipated_windows = build_anticipated_windows(system);
        let extended_delivery_anchors =
            build_extended_delivery_anchors(system, &state_layout, n_stages);
        let transit_seed_arcs = build_transit_seed_arcs(system);
        let past_defluences = system.initial_conditions().past_defluences.clone();

        admission_gate(
            &risk_measures,
            &stopping_rule_set,
            training_enumerated,
            cut_selection.as_ref(),
        )?;

        let hydro_min_storage_hm3: Vec<f64> =
            system.hydros().iter().map(|h| h.min_storage_hm3).collect();

        Ok(Self {
            stage_data: stage_data::StageData {
                stage_templates,
                state: state_layout,
                study_dims,
                hydro_cell_index,
                cut_state_layouts,
                stages,
                entity_counts,
                pumping_consumption_mw_per_m3s,
                contract_prices_per_stage,
                contract_is_import,
                block_counts_per_stage,
                stage_lag_transitions,
                noise_group_ids,
                scaling_report,
            },
            stochastic,
            fcf,
            initial_state,
            hydro_models,
            ncs_entity_ids_per_stage,
            ncs_stochastic_dense_col,
            ncs_stochastic_windows,
            ncs_max_gen,
            ncs_allow_curtailment,
            anticipated_windows,
            study_stage_ids,
            extended_delivery_anchors,
            transit_seed_arcs,
            past_defluences,
            study_stage_dates,
            resolved_parameters,
            scenario_libraries,
            node_graph,
            loop_params: LoopParams {
                seed,
                forward_passes,
                training_enumerated,
                max_iterations,
                start_iteration: 0,
                max_blocks,
                stopping_rules: stopping_rule_set,
            },
            simulation_config: SimulationConfig {
                n_scenarios,
                io_channel_capacity,
                profile: simulation_profile,
            },
            simulation_enumerated,
            policy_path,
            cut_management: CutManagementConfig {
                cut_selection,
                budget,
                cut_activity_tolerance,
                risk_measures,
            },
            events: EventParams { export_states },
            backward_profile,
            forward_profile,
            backward_scheduler,
            hardest_first_claim_order: true,
            horizon,
            inflow_method,
            derived_inflow_seeds,
            downstream_par_order,
            energy_conversion,
            hydro_min_storage_hm3,
            transit_bucket_topology,
            warm_start_basis_cache: None,
            boundary_requirements: boundary,
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
    /// Training runs; whether simulation follows is a separate post-training
    /// check the caller still makes.
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

// ---------------------------------------------------------------------------
// from_broadcast_params sub-phase helpers
// ---------------------------------------------------------------------------

/// Grouped output of [`build_ncs_entity_data`].
struct NcsEntityData {
    entity_counts: EntityCounts,
    ncs_entity_ids_per_stage: Vec<Vec<i32>>,
    ncs_stochastic_dense_col: Vec<usize>,
    ncs_stochastic_windows: Vec<(Option<i32>, Option<i32>)>,
    ncs_max_gen: Vec<f64>,
    ncs_allow_curtailment: Vec<bool>,
}

/// Build entity counts and the dense NCS column/window maps from the system.
///
/// `ncs_stochastic_dense_col`, `ncs_stochastic_windows`, `ncs_max_gen`, and
/// `ncs_allow_curtailment` are aligned 1:1 in stochastic NCS-entity (slot) order;
/// see [`StudySetup::ncs_stochastic_dense_col`] and
/// [`StudySetup::ncs_stochastic_windows`] for what each carries.
///
/// # Errors
///
/// Returns [`SddpError::Validation`] when a stochastic NCS entity has no match
/// in the system's `non_controllable_sources`.
fn build_ncs_entity_data(
    system: &System,
    stage_templates: &StageTemplates,
    stochastic: &StochasticContext,
) -> Result<NcsEntityData, SddpError> {
    let entity_counts = build_entity_counts(system);

    let n_study = stage_templates.templates.len();

    // Every stage repeats the full id-sorted NCS list, so a dormant NCS still
    // occupies its slot and reports a zero row rather than being absent.
    let ncs_entity_ids_per_stage: Vec<Vec<i32>> =
        vec![entity_counts.non_controllable_ids.clone(); n_study];

    let stoch_ncs_ids = stochastic.ncs_entity_ids();

    // Bridge each slot to its dense column via entity id (not a direct index) so the
    // map stays correct when only a subset of NCS are stochastic or the orders
    // diverge. Keyed on the id-sorted slot order, not entity declaration order.
    let mut ncs_stochastic_dense_col: Vec<usize> = Vec::with_capacity(stoch_ncs_ids.len());
    let mut ncs_stochastic_windows: Vec<(Option<i32>, Option<i32>)> =
        Vec::with_capacity(stoch_ncs_ids.len());
    let mut ncs_max_gen: Vec<f64> = Vec::with_capacity(stoch_ncs_ids.len());
    let mut ncs_allow_curtailment: Vec<bool> = Vec::with_capacity(stoch_ncs_ids.len());
    for slot_id in stoch_ncs_ids {
        let not_found = || {
            SddpError::Validation(format!(
                "stochastic NCS entity {slot_id:?} not found in system non_controllable_sources"
            ))
        };
        let dense_col = entity_counts
            .non_controllable_ids
            .iter()
            .position(|&id| id == slot_id.0)
            .ok_or_else(not_found)?;
        let ncs = system
            .non_controllable_sources()
            .iter()
            .find(|n| n.id == *slot_id)
            .ok_or_else(not_found)?;
        ncs_stochastic_dense_col.push(dense_col);
        ncs_stochastic_windows.push((ncs.entry_stage_id, ncs.exit_stage_id));
        ncs_max_gen.push(ncs.max_generation_mw);
        ncs_allow_curtailment.push(ncs.allow_curtailment);
    }

    Ok(NcsEntityData {
        entity_counts,
        ncs_entity_ids_per_stage,
        ncs_stochastic_dense_col,
        ncs_stochastic_windows,
        ncs_max_gen,
        ncs_allow_curtailment,
    })
}

/// Grouped output of [`build_energy_and_templates`].
struct EnergyAndTemplates {
    energy_conversion: EnergyConversionSet,
    stage_templates: StageTemplates,
    scaling_report: ScalingReport,
    resolved_parameters: ResolvedParameters,
}

/// Build the energy-conversion set, the resolved parameter table, and the
/// post-processed stage LP templates.
///
/// The energy-conversion set and resolved parameter table are built before the
/// LP templates so the builder can resolve `CoefficientRef::Parameter` values.
/// The resolved parameter table feeds `build_stage_templates` and is returned
/// for the generic-constraint echo. Seasonless stages collapse to season 0,
/// consistent with every other season-indexed lookup.
///
/// # Errors
///
/// - [`SddpError::Validation`] — on energy-conversion / resolved-parameter
///   construction failure, or when the post-processed template list is empty.
/// - [`SddpError::Solver`] — propagated from `build_stage_templates`.
// Rationale (too_many_arguments): each of the three arc-table parameters threads
// the single setup-owned derivation (`build_transit_bucket_topology`) into
// `build_stage_templates`, mirroring the existing `per_stage_mask` thread; a
// wrapper struct used at this one call site would rename the coupling, not
// remove it.
#[allow(clippy::too_many_arguments)]
fn build_energy_and_templates(
    system: &System,
    inflow_method: crate::InflowNonNegativityMethod,
    stochastic: &StochasticContext,
    hydro_models: &PrepareHydroModelsResult,
    scalar_parameters: &[cobre_core::ScalarParameter],
    state_layout: &StateSpace,
    cost_scale_factor: f64,
    per_stage_mask: &[Vec<usize>],
    arc_stage_weights: &HashMap<usize, Vec<Vec<f64>>>,
    arc_spread_chrono: &HashMap<usize, Vec<Option<SpreadResolution>>>,
    arc_arrival_density: &HashMap<usize, Vec<Option<Vec<f64>>>>,
    hydro_cell_index: &HydroCellIndex,
) -> Result<EnergyAndTemplates, SddpError> {
    let (energy_conversion, resolved_parameters) = build_energy_conversion_and_resolved_parameters(
        system,
        hydro_models,
        scalar_parameters,
        cost_scale_factor,
    )?;

    let mut stage_templates = build_stage_templates(
        system,
        inflow_method,
        stochastic.par(),
        stochastic.normal(),
        &hydro_models.production,
        &hydro_models.evaporation,
        &resolved_parameters,
        state_layout,
        per_stage_mask,
        arc_stage_weights,
        arc_spread_chrono,
        arc_arrival_density,
        hydro_cell_index,
        stochastic
            .provenance()
            .load_scheme
            .unwrap_or(SamplingScheme::InSample),
    )?;

    let scaling_report = template_postprocess::postprocess_templates(
        &mut stage_templates,
        system,
        state_layout,
        cost_scale_factor,
    );

    if stage_templates.templates.is_empty() {
        return Err(SddpError::Validation(
            "system has no study stages".to_string(),
        ));
    }

    Ok(EnergyAndTemplates {
        energy_conversion,
        stage_templates,
        scaling_report,
        resolved_parameters,
    })
}

/// Build the energy-conversion set and the resolved-parameter table, then fail
/// loud on a generic constraint that references an unresolved scalar-parameter
/// id — the shared prefix of [`build_energy_and_templates`] and the
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

/// `L_state = max(computed_order, boundary_depth)` — the single widening
/// every lag-state-slot source (`resolve_state_layout`'s dense stride and
/// per-hydro activeness mask, `build_opening_tree_library`,
/// `rebuild_historical_library_non_root`) applies in lockstep so a
/// boundary-inferred depth never truncates on one source while widening another.
/// `None` (no boundary) leaves `computed_order` unchanged.
#[must_use]
pub fn widen_lag_state_depth(computed_order: usize, boundary_depth: Option<u32>) -> usize {
    boundary_depth.map_or(computed_order, |d| computed_order.max(d as usize))
}

/// Resolve every anticipated thermal's delivery-anchored commitment and
/// construct the single role-(a) [`StateSpace`] — before stage templates
/// exist, since none of the state dimensions depend on the built LP.
///
/// The returned `hydro_count` and `anticipated_thermal_indices` are the exact
/// values the layout was built from; [`build_study_dimensions`] takes them as
/// parameters instead of re-deriving them from the built templates.
///
/// # Errors
///
/// - [`SddpError::Validation`] — a `LeadTime` anticipated plant's resolution
///   fans out (`AnticipatedResolution::max_fanout > 1`); per-delivery-stage
///   fan-out simulation output is not yet supported.
pub(crate) fn resolve_state_layout(
    system: &System,
    par_lp: &PrecomputedPar,
    transit_bucket_topology: &bucket_topology::TransitBucketTopology,
    inflow_lag_depth: Option<u32>,
) -> Result<(StateSpace, usize, Vec<usize>), SddpError> {
    let anticipated_thermal_indices: Vec<usize> = system
        .thermals()
        .iter()
        .enumerate()
        .filter_map(|(t_idx, thermal)| thermal.anticipated_config.is_some().then_some(t_idx))
        .collect();
    let n_anticipated = anticipated_thermal_indices.len();

    // Single resolve_point consumer: map each anticipated plant's config to a
    // delivery-anchored PointResolution and derive the constant-lead K_i the
    // still-live ring machinery reads (the resolve_point decider contract). A
    // second resolve_point call site is forbidden — this resolution threads onto
    // the state layout instead.
    let (anticipated_resolution, anticipated_lead_stages) = resolve_anticipated_commitments(system);
    debug_assert_eq!(anticipated_lead_stages.len(), n_anticipated);

    // TODO(anticipated-fanout-output): the coupled output extractor is
    // compute_anticipated_decision_mw
    if anticipated_resolution.max_fanout > 1 {
        let plant_id = first_fanned_plant_id(
            system,
            &anticipated_thermal_indices,
            &anticipated_resolution,
        );
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

    // Ring depth: the delivery-anchored max_t K_i(t), clamped up to the
    // constant-lead machinery's per-plant K_i so its slot indexing stays in range.
    // A LeadStages plant's depth is bounded by ℓ, so this equals the pre-anchor
    // max(lead_stages) and the ring sizing is byte-for-byte unchanged.
    let k_max: usize = anticipated_resolution
        .k_max
        .max(anticipated_lead_stages.iter().copied().max().unwrap_or(0));

    let hydro_count = system.hydros().len();
    let max_par_order: usize = widen_lag_state_depth(
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
    // relies on it to satisfy the `StateSpace::new` length contract.
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
    let mut state = StateSpace::new(
        hydro_count,
        max_par_order,
        transit_bucket_topology.n_buckets,
        transit_bucket_topology.column_order.clone(),
        n_anticipated,
        k_max,
        anticipated_lead_stages,
        &effective_lag_counts,
    );
    state.set_anticipated_resolution(anticipated_resolution);

    Ok((state, hydro_count, anticipated_thermal_indices))
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

/// The study's post-study delivery calendar ([`post_study_calendar_stages`]),
/// empty when none is declared. Appended after the study stages to extend the
/// ring's dating calendar so a slot maturing past the horizon dates onto its
/// real post-study stage. Shared by [`build_extended_delivery_anchors`] and
/// the policy manifest builder `build_stage_entity_manifest`, so every
/// delivery-dating site derives one calendar. The synthetic `Stage::id`
/// restarts at `0` and collides with study ids — only
/// `start_date`/`end_date` are ever read.
pub(crate) fn post_study_delivery_calendar(system: &System) -> Vec<Stage> {
    system
        .post_study_stages()
        .map(|post_study| post_study_calendar_stages(&post_study.stages))
        .unwrap_or_default()
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
/// post-study continuation ([`post_study_delivery_calendar`]) — indexed by
/// delivery target `m`. The dating input the `anticipated_lanes` output
/// extractor reads for a post-study-targeted decision (`delivery_dates[m]`),
/// matching the policy manifest's `delivery_anchor_at` walk over the same
/// extended calendar. Study-only, byte-identical to a study-stages walk, when
/// no post-study stage is declared.
fn build_extended_delivery_anchors(
    system: &System,
    state: &StateSpace,
    n_stages: usize,
) -> Vec<i32> {
    let study_stages: Vec<&Stage> = system.stages().iter().filter(|s| s.id >= 0).collect();
    let post_study_calendar = post_study_delivery_calendar(system);
    let anchors: Vec<i32> = extended_delivery_stages(&study_stages, &post_study_calendar)
        .iter()
        .map(|s| year_month_day_anchor(s.start_date))
        .collect();
    debug_assert!(
        anchors.len() >= state.delivery_stage_count(n_stages),
        "extended delivery anchors ({}) must cover the delivery axis ({})",
        anchors.len(),
        state.delivery_stage_count(n_stages),
    );
    anchors
}

/// Declared travel-time arcs (upstream hydro id + travel time), one entry per
/// hydro declaring `travel_time_hours > 0.0` and a `downstream_id` — the same
/// predicate [`bucket_topology::declared_arcs`] uses, applied per upstream
/// hydro rather than grouped by downstream plant.
fn build_transit_seed_arcs(system: &System) -> Vec<TransitSeedArc> {
    system
        .hydros()
        .iter()
        .filter_map(|h| {
            let t_v = h.travel_time_hours.filter(|&t| t > 0.0)?;
            h.downstream_id?;
            Some(TransitSeedArc {
                upstream_hydro_id: h.id.0,
                travel_time_hours: t_v,
            })
        })
        .collect()
}

/// Per-`(thermal, post-study stage)` cost/bounds lookup — [`PostStudyStages::
/// thermal_bounds`] verbatim, never rebuilt into a nondeterministic-
/// iteration-order map.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct PostStudyThermalLookup {
    bounds: Vec<PostStudyThermalBound>,
}

impl PostStudyThermalLookup {
    fn new(bounds: Vec<PostStudyThermalBound>) -> Self {
        debug_assert!(
            bounds.is_sorted_by_key(|b| (b.thermal_id, b.post_study_stage_index)),
            "PostStudyStages::thermal_bounds must already be canonically sorted by \
             (thermal_id, post_study_stage_index) — the cobre-io parser's own invariant"
        );
        Self { bounds }
    }

    /// `(cost_per_mwh, min_mw, max_mw)` declared for `(thermal_id,
    /// post_study_stage_index)`; `None` when undeclared.
    #[must_use]
    pub(crate) fn lookup(
        &self,
        thermal_id: EntityId,
        post_study_stage_index: usize,
    ) -> Option<(f64, f64, f64)> {
        self.bounds
            .binary_search_by_key(&(thermal_id, post_study_stage_index), |b| {
                (b.thermal_id, b.post_study_stage_index)
            })
            .ok()
            .map(|i| {
                let b = &self.bounds[i];
                (b.cost_per_mwh, b.min_mw, b.max_mw)
            })
    }
}

/// Setup-side resolved post-study boundary artifacts
/// ([`System::post_study_stages`]), built once so the LP builder
/// (`TemplateBuildCtx`/`StageLayout`) and `policy_export` read them without
/// re-deriving the calendar walk, the discount continuation, or the
/// per-thermal lookup. Every field is empty without `post_study_stages` —
/// inert: a study with no post-horizon commitment leaves the rest of setup
/// unchanged.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct PostStudyResolved {
    /// Post-study stage `j`'s own duration in hours
    /// (`PostStudyStage::duration_hours` verbatim).
    pub(crate) total_hours: Vec<f64>,
    /// Cumulative discount factor continued past the study horizon — the exact
    /// values [`template_postprocess::compute_cumulative_discount_factors`]
    /// would hold for these stages had the horizon been extended to cover them
    /// (the study's last cumulative factor bridged by the last study stage's own
    /// one-step factor, then multiplied stage-by-stage).
    pub(crate) cumulative_discount_factors: Vec<f64>,
    /// Per-`(thermal, post-study stage)` cost/bounds lookup.
    pub(crate) thermal_bounds: PostStudyThermalLookup,
    /// Dense row-major `[anticipated_local][post_study_stage]` projection of
    /// [`PostStudyThermalLookup::lookup`] — one cell per anticipated plant times
    /// post-study stage, `None` where the deck declares none. `anticipated_local`
    /// MUST be [`resolve_state_layout`]'s own `anticipated_thermal_indices`
    /// canonical order (`system.thermals()` filtered on
    /// `anticipated_config.is_some()`); a mismatched order silently prices one
    /// plant's post-study commitment with another's fuel cost. Never index this
    /// directly — read it only through [`Self::anticipated_bound`], and never
    /// rebuild it into a `Vec<Vec<_>>` (a per-plant allocation) or an
    /// `EntityId`-keyed map (a nondeterministic-iteration-order read).
    anticipated_bounds: Vec<Option<(f64, f64, f64)>>,
    /// Row stride of [`Self::anticipated_bounds`] — the post-study stage count,
    /// `total_hours.len()`.
    anticipated_bounds_stride: usize,
}

impl PostStudyResolved {
    /// `(cost_per_mwh, min_mw, max_mw)` declared for anticipated-local plant
    /// `local_idx` at post-study stage `post_study_stage`, or `None` when the
    /// deck declares no cell there — never a panic, including on an empty table
    /// (`PostStudyResolved::default()`) or an out-of-range `local_idx`/
    /// `post_study_stage`.
    #[must_use]
    pub(crate) fn anticipated_bound(
        &self,
        local_idx: AnticipatedLocal,
        post_study_stage: usize,
    ) -> Option<(f64, f64, f64)> {
        if post_study_stage >= self.anticipated_bounds_stride {
            return None;
        }
        self.anticipated_bounds
            .get(local_idx.get() * self.anticipated_bounds_stride + post_study_stage)
            .copied()
            .flatten()
    }
}

/// Resolve [`System::post_study_stages`] into the setup-side artifacts:
/// post-study `total_hours`, the discount continuation, the per-thermal
/// cost/bounds lookup, and its dense anticipated-local projection. `None`/empty
/// `post_study` returns [`PostStudyResolved::default`] — inert.
///
/// `anticipated_thermal_ids` is the anticipated plants' `EntityId`s in
/// anticipated-local order — [`resolve_state_layout`]'s own
/// `anticipated_thermal_indices` order, handed in rather than re-derived here,
/// since a second derivation could silently diverge from it.
///
/// `last_real_cumulative` and `last_real_per_stage` are the study's own last
/// cumulative and per-stage discount factors — [`crate::StageTemplates::
/// cumulative_discount_factors`]/[`crate::StageTemplates::discount_factors`]'s
/// last entries, or (`crate::lp::builder::build_stage_templates`'s own
/// `TemplateBuildCtx` build) the identical values computed from the same
/// `compute_per_stage_discount_factors`/`compute_cumulative_discount_factors`
/// pair before those output slices exist. The first post-study cumulative
/// factor bridges the horizon by the LAST STUDY stage's own one-step factor
/// (`last_real_cumulative * last_real_per_stage`), NEVER the first post-study
/// stage's (`* per_stage_post[0]`): the continuation must equal what
/// `cumulative_discount_factors` would hold had the horizon been extended to
/// cover the post-study stages.
pub(crate) fn resolve_post_study_artifacts(
    post_study: Option<&PostStudyStages>,
    anticipated_thermal_ids: &[EntityId],
    pg: &HorizonGraph,
    last_real_cumulative: f64,
    last_real_per_stage: f64,
) -> PostStudyResolved {
    let Some(post_study) = post_study else {
        return PostStudyResolved::default();
    };
    if post_study.stages.is_empty() {
        return PostStudyResolved::default();
    }

    let total_hours: Vec<f64> = post_study.stages.iter().map(|s| s.duration_hours).collect();

    let calendar_stages = post_study_calendar_stages(&post_study.stages);
    // `PostStudyStage` declares no rate-override field (unlike a dispatched
    // `Stage`); a `HorizonGraph` carrying only `annual_discount_rate` keeps this
    // call from resolving a synthetic post-study stage id against a REAL study
    // stage's override in `pg.stage_discount_rate_overrides`.
    let rate_graph = HorizonGraph {
        annual_discount_rate: pg.annual_discount_rate,
        ..HorizonGraph::default()
    };
    let calendar_stage_refs: Vec<&Stage> = calendar_stages.iter().collect();
    let per_stage_post =
        template_postprocess::compute_per_stage_discount_factors(&calendar_stage_refs, &rate_graph);

    let mut cumulative_discount_factors = Vec::with_capacity(per_stage_post.len());
    let mut cumulative = last_real_cumulative * last_real_per_stage;
    for &factor in &per_stage_post {
        cumulative_discount_factors.push(cumulative);
        cumulative *= factor;
    }

    let thermal_bounds = PostStudyThermalLookup::new(post_study.thermal_bounds.clone());

    // Dense [anticipated_local][post_study_stage] projection of `thermal_bounds`,
    // built once here so the ring fill (`fill_anticipated_columns`) never
    // reconstructs an `EntityId` from an anticipated-local index or re-searches
    // `thermal_bounds` per fill call.
    let anticipated_bounds_stride = total_hours.len();
    let mut anticipated_bounds =
        Vec::with_capacity(anticipated_thermal_ids.len() * anticipated_bounds_stride);
    for &thermal_id in anticipated_thermal_ids {
        for post_study_stage in 0..anticipated_bounds_stride {
            anticipated_bounds.push(thermal_bounds.lookup(thermal_id, post_study_stage));
        }
    }
    debug_assert_eq!(
        anticipated_bounds.len(),
        anticipated_thermal_ids.len() * anticipated_bounds_stride,
        "PostStudyResolved.anticipated_bounds row count must equal \
         anticipated_thermal_ids.len()"
    );

    PostStudyResolved {
        total_hours,
        cumulative_discount_factors,
        thermal_bounds,
        anticipated_bounds,
        anticipated_bounds_stride,
    }
}

/// Build the study-invariant, non-state [`StudyDimensions`] from the system
/// and the post-processed stage templates.
///
/// `hydro_count` and `anticipated_thermal_indices` are threaded from
/// [`resolve_state_layout`] — the same values its [`StateSpace`] was built
/// from — so the only per-stage template field this reads is
/// `ncs_col_starts`, the one dimension genuinely derived from the built LP.
fn build_study_dimensions(
    system: &System,
    stage_templates: &StageTemplates,
    inflow_method: crate::InflowNonNegativityMethod,
    hydro_count: usize,
    anticipated_thermal_indices: Vec<usize>,
) -> StudyDimensions {
    let has_inflow_penalty = inflow_method.has_slack_columns() && hydro_count > 0;

    let max_deficit_segments = system
        .buses()
        .iter()
        .map(|b| b.deficit_segments.len())
        .max()
        .unwrap_or(0);

    // Single owner of the study-invariant, non-state LP shape. `has_ncs` only flags
    // presence; the per-(ncs, block) column base is read per stage from
    // `StageContext::ncs_col_starts`, never a global handle. `n_blks` is deliberately
    // absent — it is per-stage, owned by the per-stage geometry, never study-global.
    StudyDimensions {
        n_thermals: system.thermals().len(),
        n_lines: system.lines().len(),
        n_buses: system.buses().len(),
        max_deficit_segments,
        has_ncs: !stage_templates.ncs_col_starts.is_empty(),
        has_inflow_penalty,
        has_withdrawal: hydro_count > 0,
        has_operational_violations: hydro_count != 0,
        anticipated_thermal_indices,
        n_pumping: system.n_pumping_stations(),
    }
}

/// The first (canonical-order) anticipated plant whose `LeadTime` resolution
/// fans out — `|genuine C(t)| > 1` at some decision stage `t` — or `None` if
/// none does. Shares the exact per-plant/per-stage predicate
/// [`AnticipatedResolution::max_fanout`] maxes over, so `Some(_)` iff
/// `resolution.max_fanout > 1`; `anticipated_thermal_indices` and
/// `resolution.per_plant` are both in canonical (anticipated-local) order, so
/// the first match is declaration-order-invariant.
fn first_fanned_plant_id(
    system: &System,
    anticipated_thermal_indices: &[usize],
    resolution: &AnticipatedResolution,
) -> Option<EntityId> {
    resolution
        .per_plant
        .iter()
        .enumerate()
        .find_map(|(local_idx, point)| {
            let fans_out =
                (0..point.decision_sets.len()).any(|t| point.genuine_decisions_at(t).count() > 1);
            fans_out.then(|| system.thermals()[anticipated_thermal_indices[local_idx]].id)
        })
}

/// The study calendar followed by every declared post-study stage duration;
/// byte-identical to the study-only vector when none is declared. Two
/// consumers derive their own extended calendar from this one vector: the
/// anticipated delivery axis ([`DeliveryAxis::stage_lengths_hours`], where it
/// lets `n_delivery` span `n_stages + n_post`) and the water ring's arrival
/// resolution ([`bucket_topology::extend_for_resolution`]'s base calendar).
fn delivery_stage_durations(mut study_durations: Vec<f64>, system: &System) -> Vec<f64> {
    if let Some(post_study) = system.post_study_stages() {
        study_durations.extend(post_study.stages.iter().map(|s| s.duration_hours));
    }
    study_durations
}

/// Resolve every anticipated thermal's delivery-anchored point commitment and
/// derive the constant-lead per-plant `K_i` the still-live ring machinery reads.
///
/// The sole `resolve_point` consumer (via [`AnticipatedResolution::resolve`]).
/// Warn-free: [`resolve_anticipated_commitments`] wraps this with the setup-time
/// `K = 0` advisory; [`crate::lp::builder::build_stage_templates`] calls this core
/// directly to attach an identical resolution onto its own `StateSpace` — the
/// same accepted redundant-but-deterministic recompute this crate already
/// applies to the bucket topology, not a second advisory emission. Returns the
/// per-plant resolution and the anticipated-local constant leads: a
/// `LeadStages(ℓ)` plant keeps `ℓ` byte-for-byte; a `LeadTime` plant takes its
/// per-plant ring depth ([`PointResolution::ring_depth`]) — the plant's own
/// `k_i` reachability/padding bound the slot masking and policy manifest read.
///
/// The delivery axis is EXTENDED: `n_delivery = n_stages + n_post` while
/// `n_decision` stays `n_stages` (decisions are only ever made in-study), so a
/// `LeadTime` plant's resolution can target a post-study delivery. Widening
/// this site alone is a half-switch — [`crate::lp::indexer::anticipated_gate::anticipated_resolution_for`]'s
/// fixture fallback must widen in lockstep or the two resolution paths desync
/// the moment a study declares `post_study_stages`.
pub(crate) fn resolve_anticipated_commitments_core(
    system: &System,
) -> (AnticipatedResolution, Vec<usize>) {
    let anticipated_thermals: Vec<&Thermal> = system
        .thermals()
        .iter()
        .filter(|t| t.anticipated_config.is_some())
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

    let study_durations = bucket_topology::study_stage_durations(system);
    let n_stages = study_durations.len();
    let durations = delivery_stage_durations(study_durations, system);
    let n_delivery = durations.len();
    let resolution = AnticipatedResolution::resolve(
        &leads,
        DeliveryAxis {
            stage_lengths_hours: &durations,
            n_decision: n_stages,
            n_delivery,
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
/// advisory ([`warn_on_sub_stage_lead`]) — the single owner of that advisory.
/// Every other caller (e.g. [`crate::lp::builder::build_stage_templates`]) uses
/// the core directly so the advisory never double-emits.
pub(crate) fn resolve_anticipated_commitments(
    system: &System,
) -> (AnticipatedResolution, Vec<usize>) {
    let (resolution, lead_stages) = resolve_anticipated_commitments_core(system);
    let anticipated_thermals: Vec<&Thermal> = system
        .thermals()
        .iter()
        .filter(|t| t.anticipated_config.is_some())
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
    anticipated_thermal_indices: &[usize],
    resolution: &AnticipatedResolution,
    boundary_present: bool,
) {
    if boundary_present {
        return;
    }
    let n_stages = bucket_topology::study_stage_durations(system).len();
    let thermals = system.thermals();
    let horizon_end = study_horizon_end(system);
    let past = &system.initial_conditions().past_anticipated_commitments;
    let has_nonzero_fixed = |thermal_id: i32| -> bool {
        horizon_end.is_some_and(|end| {
            past.iter()
                .any(|w| w.thermal_id.0 == thermal_id && w.start_date >= end && w.value_mw != 0.0)
        })
    };
    let affected: Vec<String> = anticipated_thermal_indices
        .iter()
        .zip(&resolution.per_plant)
        .filter(|&(&t_idx, point)| {
            let class3 = point.decider.get(n_stages..).is_some_and(|post_study| {
                post_study.iter().any(|c| c.is_some_and(|t| t < n_stages))
            });
            class3 || has_nonzero_fixed(thermals[t_idx].id.0)
        })
        .map(|(&t_idx, _)| format!("{} ({})", thermals[t_idx].id, thermals[t_idx].name))
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

/// Grouped output of [`precompute_lag_data`].
struct LagData {
    stage_lag_transitions: Vec<StageLagTransition>,
    noise_group_ids: Vec<u32>,
    downstream_par_order: usize,
}

/// Precompute per-stage lag accumulation weights, noise-group ids, and the
/// downstream PAR order. `season_map_ref` is the caller's already-resolved
/// no-op-fallback season map (see the `from_broadcast_params` hoist).
fn precompute_lag_data(
    system: &System,
    stages: &[Stage],
    stochastic: &StochasticContext,
    season_map_ref: &SeasonMap,
) -> LagData {
    // Proxy: the global `max_par_order` stands in for the quarterly PAR order until a
    // separate quarterly stochastic context exists.
    let downstream_par_order = derive_downstream_par_order(
        stages,
        stochastic.par().max_order(),
        system.policy_graph().season_map.as_ref(),
    );
    let stage_lag_transitions =
        precompute_stage_lag_transitions(stages, season_map_ref, downstream_par_order);
    // Both outputs derive from `stages`, so they cannot disagree about which
    // stages are in scope; `study_stage_noise_group_ids` re-derives that scope
    // from `System` and is for callers that have no filtered slice.
    let noise_group_ids = precompute_noise_groups(stages);

    LagData {
        stage_lag_transitions,
        noise_group_ids,
        downstream_par_order,
    }
}

/// Build the training and simulation [`ScenarioLibraries`].
///
/// Each phase's per-class library (`historical`, `external_inflow`,
/// `external_load`, `external_ncs`) is constructed only when that class uses
/// the matching sampling scheme. Simulation-specific libraries are built only
/// when the simulation scheme differs from the training scheme; when identical,
/// the simulation phase stores `None` and `simulation_ctx()` falls back to the
/// training library references.
///
/// # Errors
///
/// Propagates [`SddpError`] from the individual library builders on validation
/// or padding failure.
// Rationale: mirrors build_historical_inflow_library/build_external_inflow_library's
// own arity; a context struct would just relocate the arity, not reduce it.
#[allow(clippy::too_many_arguments)]
// Rationale: a flat training/simulation x 4-class enumeration; splitting it
// would relocate the enumeration into a same-shaped helper, not shrink it.
#[allow(clippy::too_many_lines)]
fn build_scenario_libraries(
    system: &System,
    stages: &[Stage],
    hydro_ids: &[EntityId],
    stochastic: &StochasticContext,
    stage_lag_transitions: &[StageLagTransition],
    training_source: &ScenarioSource,
    simulation_source: &ScenarioSource,
    forward_passes: u32,
    downstream_par_order: usize,
    seed: DerivedSeed<'_>,
) -> Result<ScenarioLibraries, SddpError> {
    let inflow_scheme = training_source.inflow_scheme;
    let load_scheme = training_source.load_scheme;
    let ncs_scheme = training_source.ncs_scheme;
    let sim_inflow_scheme = simulation_source.inflow_scheme;
    let sim_load_scheme = simulation_source.load_scheme;
    let sim_ncs_scheme = simulation_source.ncs_scheme;
    // Shared by every external LOAD call below, training and simulation alike
    // — see `build_external_load_library`'s doc for why.
    let normal_load_bus_ids = system.load_noise_member_bus_ids(load_scheme);

    let training_historical: Option<HistoricalScenarioLibrary> =
        if inflow_scheme == SamplingScheme::Historical {
            Some(scenario_libraries::build_historical_inflow_library(
                system.inflow_history(),
                hydro_ids,
                stages,
                stochastic.par(),
                system.policy_graph().season_map.as_ref(),
                seed,
                stage_lag_transitions,
                training_source.historical_years.as_ref(),
                forward_passes,
                downstream_par_order,
            )?)
        } else {
            None
        };

    let training_external_inflow: Option<ExternalScenarioLibrary> =
        if inflow_scheme == SamplingScheme::External {
            Some(scenario_libraries::build_external_inflow_library(
                system.external_scenarios(),
                hydro_ids,
                stages,
                stochastic.par(),
                seed,
                stage_lag_transitions,
                forward_passes,
                downstream_par_order,
            )?)
        } else {
            None
        };

    let training_external_load: Option<ExternalScenarioLibrary> =
        if load_scheme == SamplingScheme::External {
            Some(scenario_libraries::build_external_load_library(
                system,
                load_scheme,
                stages,
                forward_passes,
                stochastic.normal(),
                &normal_load_bus_ids,
            )?)
        } else {
            None
        };

    let training_external_ncs: Option<ExternalScenarioLibrary> =
        if ncs_scheme == SamplingScheme::External {
            Some(scenario_libraries::build_external_ncs_library(
                system,
                stages,
                forward_passes,
                stochastic.ncs_normal(),
                stochastic.ncs_entity_ids(),
            )?)
        } else {
            None
        };

    let simulation_historical: Option<HistoricalScenarioLibrary> =
        if sim_inflow_scheme == SamplingScheme::Historical && sim_inflow_scheme != inflow_scheme {
            Some(scenario_libraries::build_historical_inflow_library(
                system.inflow_history(),
                hydro_ids,
                stages,
                stochastic.par(),
                system.policy_graph().season_map.as_ref(),
                seed,
                stage_lag_transitions,
                simulation_source.historical_years.as_ref(),
                forward_passes,
                downstream_par_order,
            )?)
        } else {
            None
        };

    let simulation_external_inflow: Option<ExternalScenarioLibrary> =
        if sim_inflow_scheme == SamplingScheme::External && sim_inflow_scheme != inflow_scheme {
            Some(scenario_libraries::build_external_inflow_library(
                system.external_scenarios(),
                hydro_ids,
                stages,
                stochastic.par(),
                seed,
                stage_lag_transitions,
                forward_passes,
                downstream_par_order,
            )?)
        } else {
            None
        };

    let simulation_external_load: Option<ExternalScenarioLibrary> =
        if sim_load_scheme == SamplingScheme::External && sim_load_scheme != load_scheme {
            Some(scenario_libraries::build_external_load_library(
                system,
                sim_load_scheme,
                stages,
                forward_passes,
                stochastic.normal(),
                &normal_load_bus_ids,
            )?)
        } else {
            None
        };

    let simulation_external_ncs: Option<ExternalScenarioLibrary> =
        if sim_ncs_scheme == SamplingScheme::External && sim_ncs_scheme != ncs_scheme {
            Some(scenario_libraries::build_external_ncs_library(
                system,
                stages,
                forward_passes,
                stochastic.ncs_normal(),
                stochastic.ncs_entity_ids(),
            )?)
        } else {
            None
        };

    let libraries = ScenarioLibraries {
        training: PhaseLibraries {
            inflow_scheme,
            load_scheme,
            ncs_scheme,
            historical: training_historical,
            external_inflow: training_external_inflow,
            external_load: training_external_load,
            external_ncs: training_external_ncs,
        },
        simulation: PhaseLibraries {
            inflow_scheme: sim_inflow_scheme,
            load_scheme: sim_load_scheme,
            ncs_scheme: sim_ncs_scheme,
            historical: simulation_historical,
            external_inflow: simulation_external_inflow,
            external_load: simulation_external_load,
            external_ncs: simulation_external_ncs,
        },
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

/// Return the maximum iteration budget from the stopping rule set.
///
/// Used for FCF pre-sizing. If no iteration limit is present, returns
/// [`DEFAULT_MAX_ITERATIONS`].
fn max_iterations_from_rules(rules: &StoppingRuleSet) -> u64 {
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
        .unwrap_or(DEFAULT_MAX_ITERATIONS)
}

/// Build the per-study-stage risk measures from the system's stage risk configs.
///
/// One entry per study stage (`id >= 0`), in stage-index order, matching the
/// `block_counts_per_stage` / template ordering the cut-management pipeline
/// indexes by stage.
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
        | StoppingRule::BoundStalling { .. }
        | StoppingRule::GracefulShutdown => false,
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
/// inner slice is flat with the per-stage stride `block_counts_per_stage[t]` —
/// index `c * n_blks + blk`, `c` ID-sorted parallel to `system.contracts()`
/// (the same order `EntityCounts::contract_ids` is built in) — carrying
/// `contract_bounds_at_block(c, t, blk).price_per_mwh`. Empty inner slices for
/// a contract-free system or a zero-block stage.
fn build_contract_prices_per_stage(
    system: &System,
    n_stages: usize,
    block_counts_per_stage: &[usize],
) -> Vec<Vec<f64>> {
    let bounds = system.bounds();
    let n_contracts = system.contracts().len();
    (0..n_stages)
        .map(|t| {
            let n_blks = block_counts_per_stage[t];
            (0..n_contracts)
                .flat_map(|c| {
                    (0..n_blks)
                        .map(move |blk| bounds.contract_bounds_at_block(c, t, blk).price_per_mwh)
                })
                .collect()
        })
        .collect()
}

/// Build the per-contract direction flags (`true` = import).
///
/// ID-sorted parallel to `system.contracts()` — the same order
/// `EntityCounts::contract_ids` is built in — so extraction's running per-direction
/// slot count reproduces the LP builder's `fill_contract_columns` slot assignment.
fn build_contract_is_import(system: &System) -> Vec<bool> {
    system
        .contracts()
        .iter()
        .map(|c| c.contract_type == Import)
        .collect()
}

/// Build the per-plant commissioning windows for the anticipated thermals.
///
/// In anticipated-local declaration order — the same order
/// `anticipated_thermal_indices` and the LP-builder `anticipated_windows` use, so
/// the simulation decision gate reads the matching window per index. Empty when
/// there are no anticipated thermals.
fn build_anticipated_windows(system: &System) -> Vec<(Option<i32>, Option<i32>)> {
    system
        .thermals()
        .iter()
        .filter(|t| t.anticipated_config.is_some())
        .map(|t| (t.entry_stage_id, t.exit_stage_id))
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
            state[idx] = hs.value_hm3;
        }
    }

    for hs in &ic.filling_storage {
        // The seed writes the same coordinate the PreFilling pin
        // (`fill_prefilling_shortcircuit`) freezes to `[seed, seed]`; do not merge
        // the two collections or re-index the column — a separate index would
        // silently desync from that pin.
        if let Some(&idx) = hydro_positions.get(&hs.hydro_id.0) {
            state[idx] = hs.value_hm3;
        }
    }

    if layout.max_par_order > 0 {
        let n_h = layout.hydro_count;
        let l = layout.max_par_order;
        for idx in 0..n_h {
            for lag in 0..l {
                let slot = layout.inflow_lags.start + lag * n_h + idx;
                state[slot] = derived_lag_values[idx * l + lag];
            }
        }
    }

    if layout.n_anticipated > 0 && layout.k_max > 0 {
        debug_assert_eq!(
            study_dims.anticipated_thermal_indices.len(),
            layout.n_anticipated,
            "anticipated_thermal_indices length must equal n_anticipated",
        );
        let thermals = system.thermals();
        let thermal_positions = id_to_position(thermals, |t: &Thermal| t.id.0);
        let calendar = StageCalendar::new(study_stages_slice(system));
        for history in &ic.past_anticipated_commitments {
            let Some(&global_idx) = thermal_positions.get(&history.thermal_id.0) else {
                // Defense-in-depth — the cobre-io validator rejects an unknown ID in
                // production.
                continue;
            };
            // O(n) over the small `n_anticipated` list, not a map.
            let Some(local_idx) = study_dims
                .anticipated_thermal_indices
                .iter()
                .position(|&g| g == global_idx)
            else {
                // Not an anticipated plant (`anticipated_config: None`) — skip.
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
            // coverage's whole-day-hours arithmetic keeps a full-coverage ratio
            // bit-exact (mirrors StageCalendar::covers_exactly).
            #[allow(clippy::float_cmp)]
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
            // Padding slots `[K_i, k_max)` must stay 0.0 — a non-zero value corrupts
            // the ring buffer and causes LP infeasibility.
            #[allow(clippy::float_cmp)]
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
) -> Vec<f64> {
    let mut seed = vec![0.0_f64; topology.n_buckets];
    if topology.n_buckets == 0 {
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

    let mut start = 0_usize;
    for &depth in &topology.per_plant_depth {
        let plant_id = hydros[topology.column_order[start].0].id;

        for upstream in hydros {
            let Some(t_v) = upstream.travel_time_hours.filter(|&t| t > 0.0) else {
                continue;
            };
            if upstream.downstream_id != Some(plant_id) {
                continue;
            }

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
                        seed[start + transit_bucket_offset] += k_val * volume;
                    }
                }
            }
        }

        start += depth;
    }

    debug_assert_eq!(seed.len(), topology.n_buckets);
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
// Rationale: pre-study spans are on the order of years, far under f64's
// exact-integer range; a checked conversion buys nothing.
#[allow(clippy::cast_precision_loss)]
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
    let seed = build_initial_transit_bucket_state(system, topology);
    debug_assert_eq!(seed.len(), layout.n_buckets);
    for (b, &value) in seed.iter().enumerate() {
        state[layout.transit_buckets_out.start + b] = value;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests;

#[cfg(test)]
mod post_study_resolution_tests {
    use super::{PostStudyResolved, resolve_post_study_artifacts, template_postprocess};
    use chrono::NaiveDate;
    use cobre_core::{
        EntityId, HorizonGraph, PostStudyStage, PostStudyStages, PostStudyThermalBound,
    };
    use cobre_stochastic::season_cast::post_study_calendar_stages;

    fn two_stage_post_study() -> PostStudyStages {
        PostStudyStages {
            stages: vec![
                PostStudyStage {
                    start_date: NaiveDate::from_ymd_opt(2026, 11, 1)
                        .unwrap_or_else(|| unreachable!("hardcoded date is valid")),
                    duration_hours: 720.0,
                },
                PostStudyStage {
                    start_date: NaiveDate::from_ymd_opt(2026, 12, 1)
                        .unwrap_or_else(|| unreachable!("hardcoded date is valid")),
                    duration_hours: 744.0,
                },
            ],
            thermal_bounds: vec![
                PostStudyThermalBound {
                    thermal_id: EntityId(1),
                    post_study_stage_index: 0,
                    cost_per_mwh: 210.0,
                    min_mw: 0.0,
                    max_mw: 350.0,
                },
                PostStudyThermalBound {
                    thermal_id: EntityId(1),
                    post_study_stage_index: 1,
                    cost_per_mwh: 220.0,
                    min_mw: 0.0,
                    max_mw: 300.0,
                },
            ],
        }
    }

    #[test]
    fn post_study_absent_returns_default() {
        let resolved = resolve_post_study_artifacts(None, &[], &HorizonGraph::default(), 1.0, 1.0);
        assert_eq!(resolved, PostStudyResolved::default());
    }

    #[test]
    fn post_study_with_no_stages_returns_default() {
        let empty = PostStudyStages {
            stages: Vec::new(),
            thermal_bounds: Vec::new(),
        };
        let resolved =
            resolve_post_study_artifacts(Some(&empty), &[], &HorizonGraph::default(), 1.0, 1.0);
        assert_eq!(resolved, PostStudyResolved::default());
    }

    #[test]
    fn total_hours_matches_declared_duration() {
        let post_study = two_stage_post_study();
        let resolved = resolve_post_study_artifacts(
            Some(&post_study),
            &[],
            &HorizonGraph::default(),
            1.0,
            1.0,
        );
        assert_eq!(resolved.total_hours, vec![720.0, 744.0]);
    }

    #[test]
    fn continued_cumulative_discount_is_seed_at_zero_rate() {
        let post_study = two_stage_post_study();
        let resolved = resolve_post_study_artifacts(
            Some(&post_study),
            &[],
            &HorizonGraph::default(),
            0.9,
            1.0,
        );
        assert_eq!(resolved.cumulative_discount_factors, vec![0.9, 0.9]);
    }

    #[test]
    fn continued_cumulative_discount_matches_extended_horizon() {
        let post_study = two_stage_post_study();
        let pg = HorizonGraph {
            annual_discount_rate: 0.08,
            ..HorizonGraph::default()
        };
        // A synthetic two-stage study whose per-stage one-step factors are
        // `[0.95, 0.93]`: its last cumulative factor is `0.95` (the product of
        // the stages strictly before the last) and its last per-stage factor is
        // `0.93`.
        let study_per_stage = [0.95_f64, 0.93_f64];
        let last_real_cumulative = study_per_stage[0];
        let last_real_per_stage = study_per_stage[1];

        let resolved = resolve_post_study_artifacts(
            Some(&post_study),
            &[],
            &pg,
            last_real_cumulative,
            last_real_per_stage,
        );

        // Ground truth: extend the horizon with the post-study stages, take the
        // cumulative product over the whole thing, and read off the post-study
        // tail. A resolver that bridged by `per_stage_post[0]` instead of the
        // last study factor would diverge here.
        let calendar_stages = post_study_calendar_stages(&post_study.stages);
        let calendar_stage_refs: Vec<_> = calendar_stages.iter().collect();
        let per_stage_post =
            template_postprocess::compute_per_stage_discount_factors(&calendar_stage_refs, &pg);
        let mut extended_per_stage = study_per_stage.to_vec();
        extended_per_stage.extend_from_slice(&per_stage_post);
        let extended_cumulative =
            template_postprocess::compute_cumulative_discount_factors(&extended_per_stage);

        assert_eq!(
            resolved.cumulative_discount_factors,
            extended_cumulative[study_per_stage.len()..].to_vec()
        );
    }

    #[test]
    fn thermal_bound_lookup_returns_declared_triple() {
        let post_study = two_stage_post_study();
        let resolved = resolve_post_study_artifacts(
            Some(&post_study),
            &[],
            &HorizonGraph::default(),
            1.0,
            1.0,
        );

        assert_eq!(
            resolved.thermal_bounds.lookup(EntityId(1), 0),
            Some((210.0, 0.0, 350.0))
        );
        assert_eq!(
            resolved.thermal_bounds.lookup(EntityId(1), 1),
            Some((220.0, 0.0, 300.0))
        );
        assert_eq!(resolved.thermal_bounds.lookup(EntityId(2), 0), None);
    }
}

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

    use super::{TransitSeedArc, bucket_topology, build_initial_transit_bucket_state};
    use crate::simulation::extraction::build_transit_seed;
    use crate::simulation::types::{SimulationHydroResult, SimulationStageResult};

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
        let topology_b = bucket_topology::build_transit_bucket_topology(&system_b, false);
        let seed_from_emission = build_initial_transit_bucket_state(&system_b, &topology_b);

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
        let topology_reference =
            bucket_topology::build_transit_bucket_topology(&system_reference, false);
        let seed_reference =
            build_initial_transit_bucket_state(&system_reference, &topology_reference);

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
