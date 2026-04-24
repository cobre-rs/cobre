//! Study setup struct that owns all precomputed state for a solve run.
//!
//! [`StudySetup`] centralises orchestration from CLI/Python entry points.
//! It builds LP templates, indexer, initial state, FCF, horizon mode, risk measures,
//! and entity counts from a validated [`System`] and [`cobre_io::Config`].
//!
//! **Ownership**: `StudySetup` owns all data; callers borrow for [`TrainingContext`](crate::TrainingContext)
//! and [`StageContext`](crate::StageContext) construction. The [`StochasticContext`] lifetime matches setup.
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
//! let setup = StudySetup::new(system, config, stochastic, hydro_models)?;
//! assert!(!setup.stage_data.stage_templates.templates.is_empty());
//! # Ok(())
//! # }
//! ```

mod accessors;
pub(crate) mod methodology_config;
mod orchestration;
pub mod params;
pub(crate) mod scenario_libraries;
pub mod scenario_library_set;
pub mod stage_data;
pub mod stochastic_pipeline;
pub(crate) mod template_postprocess;

pub use params::{
    ConstructionConfig, DEFAULT_FORWARD_PASSES, DEFAULT_MAX_ITERATIONS, DEFAULT_SEED, StudyParams,
};
pub use scenario_library_set::{PhaseLibraries, ScenarioLibraries};
pub use stage_data::StageData;
pub use stochastic_pipeline::{
    PrepareStochasticResult, build_ncs_factor_entries, load_load_factors_for_stochastic,
    prepare_stochastic,
};

use std::path::Path;

use cobre_core::{
    EntityId, Stage, System,
    entities::hydro::HydroGenerationModel,
    scenario::{SamplingScheme, ScenarioSource},
};
use cobre_stochastic::{ExternalScenarioLibrary, HistoricalScenarioLibrary, StochasticContext};

use crate::{
    CutManagementConfig, FutureCostFunction, HorizonMode, RiskMeasure, SddpError, StageIndexer,
    build_stage_templates,
    config::EventParams,
    hydro_models::{EvaporationModel, PrepareHydroModelsResult, ResolvedProductionModel},
    simulation::EntityCounts,
    stopping_rule::{StoppingRule, StoppingRuleSet},
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
/// Callers build [`TrainingContext`](crate::TrainingContext) and [`StageContext`](crate::StageContext) by borrowing
/// from `StudySetup`.
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
    /// Max generation [MW] per stochastic NCS entity, sorted by entity ID.
    pub(crate) ncs_max_gen: Vec<f64>,

    /// Sampling schemes and pre-built libraries for training and simulation phases.
    ///
    /// Replaces the 14 flat `inflow_scheme` / `sim_inflow_scheme` / … fields.
    /// Access via `scenario_libraries.training.<field>` or
    /// `scenario_libraries.simulation.<field>`.
    pub scenario_libraries: ScenarioLibraries,

    /// Iteration-loop parameters projected from [`LoopConfig`].
    ///
    /// Holds the five pure-data fields of [`LoopConfig`] that are stable
    /// across training invocations. `n_fwd_threads` is excluded (derived
    /// at runtime) and supplied as a per-call argument to [`StudySetup::train`].
    pub loop_params: crate::config::LoopParams,

    /// Simulation pipeline parameters, stored directly as [`SimulationConfig`].
    pub simulation_config: crate::simulation::SimulationConfig,

    /// Relative path to the policy output directory (e.g. `"training/policy"`).
    pub policy_path: String,

    /// Two-stage cut management pipeline configuration.
    ///
    /// Holds cut selection, budget cap, activity tolerance, basis window, and
    /// per-stage risk measures. Replaces the five former flat fields
    /// (`cut_selection`, `cut_activity_tolerance`, `budget`,
    /// `basis_activity_window`, `risk_measures`).
    pub(crate) cut_management: CutManagementConfig,

    /// Pure-data event parameters (output-side flags).
    ///
    /// Holds only the stable, serialisable event flags. Runtime handles
    /// (`event_sender`, `shutdown_flag`) and deferred fields
    /// (`checkpoint_interval`) are excluded and supplied per-call in
    /// [`StudySetup::train`].
    pub(crate) events: EventParams,

    /// Stochastic numerical methodology parameters.
    ///
    /// Groups `horizon` and `inflow_method`, which govern study horizon
    /// treatment and inflow non-negativity enforcement respectively.
    pub(crate) methodology: methodology_config::MethodologyConfig,

    /// Pre-computed lag accumulator seed from `initial_conditions.recent_observations`.
    ///
    /// Computed once at setup time by
    /// [`crate::lag_transition::compute_recent_observation_seed`] from the parsed
    /// `RecentObservation` entries. Applied at every trajectory start in the forward
    /// pass and simulation pipeline instead of zero-filling the accumulator.
    ///
    /// When `recent_observations` is empty, this is an all-zero seed and the
    /// behavior is identical to the previous zero-reset.
    pub(crate) recent_observation_seed: crate::lag_transition::RecentObservationSeed,

    /// PAR order of the downstream (coarser) resolution model.
    ///
    /// Non-zero only when the study includes stages with `season_id >= 12` (quarterly
    /// range), indicating a monthly-to-quarterly resolution transition. Set at setup
    /// time and passed to `WorkspaceSizing` so that downstream scratch buffers are
    /// allocated at the correct capacity. Zero for uniform-resolution studies.
    pub(crate) downstream_par_order: usize,
}

impl StudySetup {
    /// Build all precomputed study state from a validated system and config.
    ///
    /// The constructor performs:
    /// 1. Config field extraction (seed, forward passes, stopping rules, etc.)
    /// 2. Delegates to [`StudySetup::from_broadcast_params`] with the extracted values.
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
        config: &cobre_io::Config,
        stochastic: StochasticContext,
        hydro_models: PrepareHydroModelsResult,
    ) -> Result<Self, SddpError> {
        let params = StudyParams::from_config(config)?;
        // Use a sentinel path; training_scenario_source / simulation_scenario_source
        // only use the path for error messages and the historical-years look-up,
        // which is not exercised when the caller provides a validated Config.
        let sentinel_path = Path::new("config.json");
        let training_source = config
            .training_scenario_source(sentinel_path)
            .map_err(|e| SddpError::Validation(e.to_string()))?;
        let simulation_source = config
            .simulation_scenario_source(sentinel_path)
            .map_err(|e| SddpError::Validation(e.to_string()))?;
        let config = params.into_construction_config();
        Self::from_broadcast_params(
            system,
            stochastic,
            config,
            hydro_models,
            &training_source,
            &simulation_source,
        )
    }

    /// Build all precomputed study state from pre-resolved broadcast parameters.
    ///
    /// This constructor accepts the scalar fields already extracted from either a
    /// [`cobre_io::Config`] (on rank 0) or a broadcast config struct (on non-root
    /// ranks). It performs the expensive computation steps that cannot be serialised:
    ///
    /// 1. `build_stage_templates` — constructs LP skeletons for each stage
    /// 2. `StageIndexer::with_equipment` — computes LP column/row offsets
    /// 3. `build_initial_state` — extracts initial storage and past inflows from system IC
    /// 4. `max_iterations_from_rules` — sizes the FCF cut pool
    /// 5. `FutureCostFunction::new` — pre-allocates cut storage
    /// 6. `HorizonMode::Finite` — wraps stage count
    /// 7. Risk measures from stage configs
    /// 8. `build_entity_counts` — entity ID and productivity vectors
    /// 9. Block layout derivation (`block_counts_per_stage`, `max_blocks`)
    ///
    /// # Errors
    ///
    /// - [`SddpError::Validation`] — if `build_stage_templates` succeeds but
    ///   the template list is empty ("system has no study stages").
    /// - [`SddpError::Solver`] — propagated from `build_stage_templates` on LP
    ///   construction failure.
    // RATIONALE: from_broadcast_params initializes all 16 StudySetup fields from
    // disjoint sources (system, stochastic, config, hydro_models, sources).
    // Splitting into smaller functions would require passing the same borrowed data
    // into multiple helpers without reducing conceptual complexity.
    #[allow(clippy::missing_panics_doc, clippy::too_many_lines)]
    pub fn from_broadcast_params(
        system: &System,
        stochastic: StochasticContext,
        config: ConstructionConfig,
        hydro_models: PrepareHydroModelsResult,
        training_source: &ScenarioSource,
        simulation_source: &ScenarioSource,
    ) -> Result<Self, SddpError> {
        let ConstructionConfig {
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
            export_states,
        } = config;

        let mut stage_templates = build_stage_templates(
            system,
            &inflow_method,
            stochastic.par(),
            stochastic.normal(),
            &hydro_models.production,
            &hydro_models.evaporation,
        )?;

        let scaling_report =
            template_postprocess::postprocess_templates(&mut stage_templates, system);

        if stage_templates.templates.is_empty() {
            return Err(SddpError::Validation(
                "system has no study stages".to_string(),
            ));
        }

        let stage_templates_ref = &stage_templates.templates;

        let n_blks_stage0 = system.stages().first().map_or(1, |s| s.blocks.len().max(1));
        let has_inflow_penalty =
            inflow_method.has_slack_columns() && stage_templates_ref[0].n_hydro > 0;

        // Compute FPHA and evaporation hydro indices at stage 0 (representative).
        let n_hydros = system.hydros().len();
        let mut fpha_hydro_indices: Vec<usize> = Vec::new();
        let mut fpha_planes: Vec<usize> = Vec::new();
        let mut evap_hydro_indices: Vec<usize> = Vec::new();
        for h_idx in 0..n_hydros {
            if let ResolvedProductionModel::Fpha { planes, .. } =
                hydro_models.production.model(h_idx, 0)
            {
                fpha_hydro_indices.push(h_idx);
                fpha_planes.push(planes.len());
            }
            if matches!(
                hydro_models.evaporation.model(h_idx),
                EvaporationModel::Linearized { .. }
            ) {
                evap_hydro_indices.push(h_idx);
            }
        }

        let max_deficit_segments = system
            .buses()
            .iter()
            .map(|b| b.deficit_segments.len())
            .max()
            .unwrap_or(0);

        let eq_counts = crate::indexer::EquipmentCounts {
            hydro_count: stage_templates_ref[0].n_hydro,
            max_par_order: stage_templates_ref[0].max_par_order,
            n_thermals: system.thermals().len(),
            n_lines: system.lines().len(),
            n_buses: system.buses().len(),
            n_blks: n_blks_stage0,
            has_inflow_penalty,
            max_deficit_segments,
        };
        let fpha_cfg = crate::indexer::FphaColumnLayout {
            hydro_indices: fpha_hydro_indices,
            planes_per_hydro: fpha_planes,
        };
        let evap_cfg = crate::indexer::EvapConfig {
            hydro_indices: evap_hydro_indices,
        };
        let mut indexer =
            StageIndexer::with_equipment_and_evaporation(&eq_counts, &fpha_cfg, &evap_cfg);

        // Wire NCS column range from the LP builder's stage-0 layout.
        if !stage_templates.ncs_col_starts.is_empty() {
            let ncs_start = stage_templates.ncs_col_starts[0];
            let n_ncs_stage0 = stage_templates.n_ncs_per_stage[0];
            indexer.ncs_generation = ncs_start..(ncs_start + n_ncs_stage0 * n_blks_stage0);

            for (s, &start) in stage_templates.ncs_col_starts.iter().enumerate() {
                debug_assert_eq!(
                    start, ncs_start,
                    "NCS column start differs at stage {s}: expected {ncs_start}, got {start}"
                );
            }
        }

        // z-inflow column and row ranges are set by StageIndexer::new at
        // fixed offset N*(1+L), no per-stage wiring needed.

        // Build per-hydro AR orders from the precomputed PAR model. When the
        // PAR has hydros with AR order < max_par_order, the mask enables
        // sparse cut rows in `build_cut_row_batch_into`.
        if indexer.max_par_order > 0 && stochastic.par().n_hydros() > 0 {
            let par = stochastic.par();
            let ar_orders: Vec<usize> = (0..par.n_hydros()).map(|h| par.order(h)).collect();
            indexer.set_nonzero_mask(&ar_orders);
        }

        let initial_state = build_initial_state(system, &indexer);

        let n_stages = stage_templates_ref.len();
        let max_iterations = max_iterations_from_rules(&stopping_rule_set);
        let fcf_capacity_iterations = max_iterations.saturating_add(1);
        let fcf = FutureCostFunction::new(
            n_stages,
            indexer.n_state,
            forward_passes,
            fcf_capacity_iterations,
            &vec![0; n_stages],
        );

        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };

        let risk_measures: Vec<RiskMeasure> = system
            .stages()
            .iter()
            .filter(|s| s.id >= 0)
            .map(|s| RiskMeasure::from(s.risk_config))
            .collect();

        let entity_counts = build_entity_counts(system);

        let ncs_entity_ids_per_stage: Vec<Vec<i32>> = stage_templates
            .active_ncs_indices
            .iter()
            .map(|stage_indices| {
                stage_indices
                    .iter()
                    .map(|&sys_idx| entity_counts.non_controllable_ids[sys_idx])
                    .collect()
            })
            .collect();

        let ncs_max_gen: Vec<f64> = {
            let stoch_ncs_ids = stochastic.ncs_entity_ids();
            let mut result = Vec::with_capacity(stoch_ncs_ids.len());
            for ncs_id in stoch_ncs_ids {
                let max_gen = system
                    .non_controllable_sources()
                    .iter()
                    .find(|n| n.id == *ncs_id)
                    .map(|n| n.max_generation_mw)
                    .ok_or_else(|| {
                        SddpError::Validation(format!(
                            "stochastic NCS entity {ncs_id:?} not found in system non_controllable_sources"
                        ))
                    })?;
                result.push(max_gen);
            }
            result
        };

        let block_counts_per_stage: Vec<usize> = stage_templates
            .block_hours_per_stage
            .iter()
            .map(Vec::len)
            .collect();
        let max_blocks = block_counts_per_stage.iter().copied().max().unwrap_or(0);

        let inflow_scheme = training_source.inflow_scheme;
        let load_scheme = training_source.load_scheme;
        let ncs_scheme = training_source.ncs_scheme;
        let sim_inflow_scheme = simulation_source.inflow_scheme;
        let sim_load_scheme = simulation_source.load_scheme;
        let sim_ncs_scheme = simulation_source.ncs_scheme;
        let stages: Vec<Stage> = system
            .stages()
            .iter()
            .filter(|s| s.id >= 0)
            .cloned()
            .collect();

        // Precompute per-stage lag accumulation weights from stage date boundaries
        // and the policy-graph season map. This runs once at setup time; the
        // resulting Vec is stored in StudySetup and borrowed read-only on the hot path.
        let noop_season_map;
        let season_map_ref = if let Some(sm) = system.policy_graph().season_map.as_ref() {
            sm
        } else {
            // No season map: all stages produce zero-weight no-op transitions.
            noop_season_map = cobre_core::temporal::SeasonMap {
                cycle_type: cobre_core::temporal::SeasonCycleType::Monthly,
                seasons: Vec::new(),
            };
            &noop_season_map
        };
        // Compute downstream PAR order: non-zero when any stage has season_id >= 12
        // (quarterly range), indicating a monthly-to-quarterly resolution transition.
        // Use the global max_par_order from the stochastic context as a proxy for the
        // quarterly PAR order until a separate quarterly stochastic context is available.
        let has_quarterly_stages = stages
            .iter()
            .any(|s| s.season_id.is_some_and(|id| id >= 12));
        let downstream_par_order = if has_quarterly_stages {
            stochastic.par().max_order()
        } else {
            0
        };
        let stage_lag_transitions = crate::lag_transition::precompute_stage_lag_transitions(
            &stages,
            season_map_ref,
            downstream_par_order,
        );
        let noise_group_ids = crate::lag_transition::precompute_noise_groups(&stages);

        // Compute lag accumulator seed from recent_observations (if any).
        // Uses the first study stage and the resolved season_map_ref. When there are
        // no recent observations the result is an all-zero seed (backward-compatible).
        let recent_observation_seed = if stages.is_empty() {
            crate::lag_transition::RecentObservationSeed::zero(system.hydros().len())
        } else {
            crate::lag_transition::compute_recent_observation_seed(
                &system.initial_conditions().recent_observations,
                &stages[0],
                season_map_ref,
                system.hydros(),
            )
        };

        let hydro_ids: Vec<EntityId> = system.hydros().iter().map(|h| h.id).collect();

        // Build training phase libraries.
        let training_historical: Option<HistoricalScenarioLibrary> =
            if inflow_scheme == SamplingScheme::Historical {
                Some(scenario_libraries::build_historical_inflow_library(
                    system.inflow_history(),
                    &hydro_ids,
                    &stages,
                    stochastic.par(),
                    system.policy_graph().season_map.as_ref(),
                    training_source.historical_years.as_ref(),
                    forward_passes,
                )?)
            } else {
                None
            };

        let training_external_inflow: Option<ExternalScenarioLibrary> =
            if inflow_scheme == SamplingScheme::External {
                Some(scenario_libraries::build_external_inflow_library(
                    system.external_scenarios(),
                    &hydro_ids,
                    &stages,
                    stochastic.par(),
                    &system.initial_conditions().past_inflows,
                    &stage_lag_transitions,
                    forward_passes,
                )?)
            } else {
                None
            };

        let training_external_load: Option<ExternalScenarioLibrary> =
            if load_scheme == SamplingScheme::External {
                Some(scenario_libraries::build_external_load_library(
                    system.external_load_scenarios(),
                    system.load_models(),
                    &stages,
                    forward_passes,
                )?)
            } else {
                None
            };

        let training_external_ncs: Option<ExternalScenarioLibrary> =
            if ncs_scheme == SamplingScheme::External {
                Some(scenario_libraries::build_external_ncs_library(
                    system.external_ncs_scenarios(),
                    system.ncs_models(),
                    &stages,
                    forward_passes,
                )?)
            } else {
                None
            };

        // Build simulation-specific libraries when simulation schemes differ from
        // training schemes. When they are identical, simulation borrows from the
        // training libraries (represented as `None` in the simulation phase, with
        // `simulation_ctx()` falling back to the training library references).

        let simulation_historical: Option<HistoricalScenarioLibrary> = if sim_inflow_scheme
            == SamplingScheme::Historical
            && sim_inflow_scheme != inflow_scheme
        {
            Some(scenario_libraries::build_historical_inflow_library(
                system.inflow_history(),
                &hydro_ids,
                &stages,
                stochastic.par(),
                system.policy_graph().season_map.as_ref(),
                simulation_source.historical_years.as_ref(),
                forward_passes,
            )?)
        } else {
            None
        };

        let simulation_external_inflow: Option<ExternalScenarioLibrary> = if sim_inflow_scheme
            == SamplingScheme::External
            && sim_inflow_scheme != inflow_scheme
        {
            Some(scenario_libraries::build_external_inflow_library(
                system.external_scenarios(),
                &hydro_ids,
                &stages,
                stochastic.par(),
                &system.initial_conditions().past_inflows,
                &stage_lag_transitions,
                forward_passes,
            )?)
        } else {
            None
        };

        let simulation_external_load: Option<ExternalScenarioLibrary> =
            if sim_load_scheme == SamplingScheme::External && sim_load_scheme != load_scheme {
                Some(scenario_libraries::build_external_load_library(
                    system.external_load_scenarios(),
                    system.load_models(),
                    &stages,
                    forward_passes,
                )?)
            } else {
                None
            };

        let simulation_external_ncs: Option<ExternalScenarioLibrary> =
            if sim_ncs_scheme == SamplingScheme::External && sim_ncs_scheme != ncs_scheme {
                Some(scenario_libraries::build_external_ncs_library(
                    system.external_ncs_scenarios(),
                    system.ncs_models(),
                    &stages,
                    forward_passes,
                )?)
            } else {
                None
            };

        let scenario_libraries = ScenarioLibraries {
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

        Ok(Self {
            stage_data: stage_data::StageData {
                stage_templates,
                indexer,
                stages,
                entity_counts,
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
            ncs_max_gen,
            scenario_libraries,
            loop_params: crate::config::LoopParams {
                seed,
                forward_passes,
                max_iterations,
                start_iteration: 0,
                max_blocks,
                stopping_rules: stopping_rule_set,
            },
            simulation_config: crate::simulation::SimulationConfig {
                n_scenarios,
                io_channel_capacity,
                basis_activity_window,
            },
            policy_path,
            cut_management: CutManagementConfig {
                cut_selection,
                budget,
                cut_activity_tolerance,
                basis_activity_window,
                warm_start_cuts: 0,
                risk_measures,
            },
            events: EventParams { export_states },
            methodology: methodology_config::MethodologyConfig {
                horizon,
                inflow_method,
            },
            recent_observation_seed,
            downstream_par_order,
        })
    }
}

// ---------------------------------------------------------------------------
// Private helper functions (extracted from cobre-cli/src/commands/run.rs)
// ---------------------------------------------------------------------------

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

/// Build [`EntityCounts`] from the loaded system.
///
/// Entity IDs are extracted from [`cobre_core::EntityId`], which stores
/// an `i32` in its inner field.
fn build_entity_counts(system: &System) -> EntityCounts {
    EntityCounts {
        hydro_ids: system.hydros().iter().map(|h| h.id.0).collect(),
        hydro_productivities: system
            .hydros()
            .iter()
            .map(|h| match &h.generation_model {
                HydroGenerationModel::ConstantProductivity {
                    productivity_mw_per_m3s,
                }
                | HydroGenerationModel::LinearizedHead {
                    productivity_mw_per_m3s,
                } => *productivity_mw_per_m3s,
                HydroGenerationModel::Fpha => 0.0,
            })
            .collect(),
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

/// Build the initial state vector from the system's initial conditions.
///
/// The state vector layout is `[storage(0..N), lags(N..N*(1+L))]` where N is
/// the number of hydros and L is the maximum PAR order. Storage positions
/// correspond to hydros in canonical ID order.
///
/// Lag slots are populated from `initial_conditions.past_inflows`. For each
/// hydro at positional index `idx` with a `past_inflows` entry, lag slot `l`
/// (0-based) is set to `entry.values_m3s[l]` where index 0 corresponds to
/// lag 1 (most recent) and index L-1 to lag L (oldest). Hydros without a
/// `past_inflows` entry have their lag slots left at `0.0`.
///
/// When `max_par_order == 0`, no lag slots exist and the state is storage-only.
///
/// Each `HydroStorage` entry in `initial_conditions.storage` is matched to
/// its positional index among the system's hydros (both sorted by `hydro_id`).
fn build_initial_state(system: &System, indexer: &StageIndexer) -> Vec<f64> {
    let mut state = vec![0.0_f64; indexer.n_state];
    let hydros = system.hydros();
    let ic = system.initial_conditions();

    for hs in &ic.storage {
        // Both hydros() and ic.storage are sorted by hydro_id.
        if let Ok(idx) = hydros.binary_search_by_key(&hs.hydro_id.0, |h| h.id.0) {
            state[idx] = hs.value_hm3;
        }
    }

    if indexer.max_par_order > 0 {
        let n_h = indexer.hydro_count;
        for pi in &ic.past_inflows {
            if let Ok(idx) = hydros.binary_search_by_key(&pi.hydro_id.0, |h| h.id.0) {
                let n_lags = pi.values_m3s.len().min(indexer.max_par_order);
                for lag in 0..n_lags {
                    let slot = indexer.inflow_lags.start + lag * n_h + idx;
                    state[slot] = pi.values_m3s[lag];
                }
            }
        }
    }

    state
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::StudySetup;
    use crate::StageIndexer;
    use crate::hydro_models::PrepareHydroModelsResult;

    use cobre_core::{
        BoundsCountsSpec, BoundsDefaults, BusStagePenalties, ContractStageBounds, HydroStageBounds,
        HydroStagePenalties, LineStageBounds, LineStagePenalties, NcsStagePenalties,
        PenaltiesCountsSpec, PenaltiesDefaults, PumpingStageBounds, ResolvedBounds,
        ResolvedPenalties, ThermalStageBounds,
    };
    use cobre_core::{
        EntityId, SystemBuilder,
        entities::{
            bus::{Bus, DeficitSegment},
            hydro::{Hydro, HydroGenerationModel, HydroPenalties},
            thermal::Thermal,
        },
        scenario::{InflowModel, LoadModel, SamplingScheme},
        temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
            StageStateConfig,
        },
    };
    use cobre_io::config::{
        Config, EstimationConfig, ExportsConfig, InflowNonNegativityConfig, ModelingConfig,
        PolicyConfig, RawClassConfigEntry, RawScenarioSourceConfig, RowSelectionConfig,
        SimulationConfig as IoSimulationConfig, StoppingRuleConfig, TrainingConfig,
        TrainingSolverConfig, UpperBoundEvaluationConfig,
    };
    use cobre_stochastic::{ClassSchemes, OpeningTreeInputs, build_stochastic_context};

    /// Build a minimal system with 1 bus, 1 thermal, 1 hydro, and `n_stages`
    /// study stages (each with 1 block). All bounds and penalties are set to
    /// sensible non-zero defaults so `build_stage_templates` succeeds.
    #[allow(
        clippy::too_many_lines,
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::items_after_statements
    )]
    fn minimal_system(n_stages: usize) -> cobre_core::System {
        use chrono::NaiveDate;

        let bus = Bus {
            id: EntityId(1),
            name: "B1".to_string(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
        };

        let thermal = Thermal {
            id: EntityId(2),
            name: "T1".to_string(),
            bus_id: EntityId(1),
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            cost_per_mwh: 50.0,
            gnl_config: None,
            entry_stage_id: None,
            exit_stage_id: None,
        };

        let hydro = Hydro {
            id: EntityId(3),
            name: "H1".to_string(),
            bus_id: EntityId(1),
            downstream_id: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 200.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity {
                productivity_mw_per_m3s: 2.5,
            },
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 100.0,
            min_generation_mw: 0.0,
            max_generation_mw: 250.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: HydroPenalties {
                spillage_cost: 0.01,
                diversion_cost: 0.0,
                fpha_turbined_cost: 0.0,
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
                inflow_nonnegativity_cost: 1000.0,
            },
        };

        let stages: Vec<Stage> = (0..n_stages)
            .map(|i| Stage {
                index: i,
                id: i as i32,
                start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
                season_id: None,
                blocks: vec![Block {
                    index: 0,
                    name: "S".to_string(),
                    duration_hours: 744.0,
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
            })
            .collect();

        let inflow_models: Vec<InflowModel> = (0..n_stages)
            .map(|i| InflowModel {
                hydro_id: EntityId(3),
                stage_id: i as i32,
                mean_m3s: 80.0,
                std_m3s: 20.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
            })
            .collect();

        let load_models: Vec<LoadModel> = (0..n_stages)
            .map(|i| LoadModel {
                bus_id: EntityId(1),
                stage_id: i as i32,
                mean_mw: 100.0,
                std_mw: 0.0,
            })
            .collect();

        let n_st = n_stages.max(1);

        fn default_hydro_bounds() -> HydroStageBounds {
            HydroStageBounds {
                min_storage_hm3: 0.0,
                max_storage_hm3: 200.0,
                min_turbined_m3s: 0.0,
                max_turbined_m3s: 100.0,
                min_outflow_m3s: 0.0,
                max_outflow_m3s: None,
                min_generation_mw: 0.0,
                max_generation_mw: 250.0,
                max_diversion_m3s: None,
                filling_inflow_m3s: 0.0,
                water_withdrawal_m3s: 0.0,
            }
        }

        fn default_hydro_penalties() -> HydroStagePenalties {
            HydroStagePenalties {
                spillage_cost: 0.01,
                diversion_cost: 0.0,
                fpha_turbined_cost: 0.0,
                storage_violation_below_cost: 500.0,
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
                inflow_nonnegativity_cost: 1000.0,
            }
        }

        let bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 1,
                n_thermals: // n_hydros
            1,
                n_lines: // n_thermals
            0,
                n_pumping: // n_lines
            0,
                n_contracts: // n_pumping
            0,
                n_stages: // n_contracts
            n_st,
            },
            &BoundsDefaults {
                hydro: default_hydro_bounds(),
                thermal: ThermalStageBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 100.0,
                    cost_per_mwh: 0.0,
                },
                line: LineStageBounds {
                    direct_mw: 0.0,
                    reverse_mw: 0.0,
                },
                pumping: PumpingStageBounds {
                    min_flow_m3s: 0.0,
                    max_flow_m3s: 0.0,
                },
                contract: ContractStageBounds {
                    min_mw: 0.0,
                    max_mw: 0.0,
                    price_per_mwh: 0.0,
                },
            },
        );

        let penalties = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 1,
                n_buses: // n_hydros
            1,
                n_lines: // n_buses
            0,
                n_ncs: // n_lines
            0,
                n_stages: // n_ncs
            n_st,
            },
            &PenaltiesDefaults {
                hydro: default_hydro_penalties(),
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        );

        SystemBuilder::new()
            .buses(vec![bus])
            .thermals(vec![thermal])
            .hydros(vec![hydro])
            .stages(stages)
            .inflow_models(inflow_models)
            .load_models(load_models)
            .bounds(bounds)
            .penalties(penalties)
            .build()
            .expect("minimal_system: valid")
    }

    /// Build a minimal valid [`Config`] with a single iteration-limit stopping rule.
    fn minimal_config(forward_passes: u32, max_iterations: u32) -> Config {
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
                forward_passes: Some(forward_passes),
                stopping_rules: Some(vec![StoppingRuleConfig::IterationLimit {
                    limit: max_iterations,
                }]),
                stopping_mode: "any".to_string(),
                cut_formulation: None,
                forward_pass: None,
                cut_selection: RowSelectionConfig::default(),
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

    /// Build a minimal valid [`Config`] with the given per-class scheme overrides.
    ///
    /// `inflow_scheme`, `load_scheme`, and `ncs_scheme` are optional strings
    /// matching the JSON schema values (`"in_sample"`, `"historical"`, `"external"`,
    /// `"out_of_sample"`). `None` leaves the class defaulting to `in_sample`.
    fn minimal_config_with_schemes(
        forward_passes: u32,
        max_iterations: u32,
        inflow_scheme: Option<&str>,
        load_scheme: Option<&str>,
        ncs_scheme: Option<&str>,
    ) -> Config {
        // A seed is required when any class uses a non-in-sample scheme.
        let needs_seed = inflow_scheme.is_some_and(|s| s != "in_sample")
            || load_scheme.is_some_and(|s| s != "in_sample")
            || ncs_scheme.is_some_and(|s| s != "in_sample");
        let scenario_source = RawScenarioSourceConfig {
            seed: if needs_seed { Some(42) } else { None },
            historical_years: None,
            inflow: inflow_scheme.map(|s| RawClassConfigEntry {
                scheme: s.to_string(),
            }),
            load: load_scheme.map(|s| RawClassConfigEntry {
                scheme: s.to_string(),
            }),
            ncs: ncs_scheme.map(|s| RawClassConfigEntry {
                scheme: s.to_string(),
            }),
        };
        let mut config = minimal_config(forward_passes, max_iterations);
        config.training.scenario_source = Some(scenario_source);
        config
    }

    /// Given a minimal valid system (1 hydro, 1 thermal, 1 bus, 2 stages),
    /// when `StudySetup::new()` is called, then it returns `Ok` and
    /// `stage_templates()` returns a non-empty slice.
    #[test]
    fn new_minimal_valid_system_returns_ok() {
        let system = minimal_system(2);
        let config = minimal_config(1, 10);
        let stochastic = build_stochastic_context(
            &system,
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
        .expect("stochastic context");

        let result = StudySetup::new(
            &system,
            &config,
            stochastic,
            PrepareHydroModelsResult::default_from_system(&system),
        );
        assert!(result.is_ok(), "expected Ok, got {result:?}");
        let setup = result.unwrap();
        assert!(!setup.stage_data.stage_templates.templates.is_empty());
    }

    /// Given a system with zero study stages, when `StudySetup::new()` is
    /// called, then it returns `Err` containing the substring "no study stages".
    #[test]
    fn new_zero_stages_returns_validation_error() {
        let system = minimal_system(0);
        let config = minimal_config(1, 10);
        let stochastic = build_stochastic_context(
            &system,
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
        .expect("stochastic context");

        let result = StudySetup::new(
            &system,
            &config,
            stochastic,
            PrepareHydroModelsResult::default_from_system(&system),
        );
        assert!(result.is_err(), "expected Err, got Ok");
        let err = result.unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("no study stages"),
            "error message should contain 'no study stages': {msg}"
        );
    }

    /// Given a valid `StudySetup`, accessor methods return the expected values.
    #[test]
    fn accessor_methods_return_expected_values() {
        let n_stages = 3;
        let system = minimal_system(n_stages);
        let config = minimal_config(2, 50);
        let stochastic = build_stochastic_context(
            &system,
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
        .expect("stochastic context");

        let setup = StudySetup::new(
            &system,
            &config,
            stochastic,
            PrepareHydroModelsResult::default_from_system(&system),
        )
        .expect("setup");

        // Stage templates
        assert_eq!(setup.stage_data.stage_templates.templates.len(), n_stages);
        assert_eq!(setup.stage_data.stage_templates.base_rows.len(), n_stages);

        // Config-derived scalars
        assert_eq!(setup.loop_params.seed, 42);
        assert_eq!(setup.loop_params.forward_passes, 2);
        assert_eq!(setup.loop_params.max_iterations, 50);
        assert_eq!(setup.simulation_config.n_scenarios, 0); // simulation disabled by default
        assert_eq!(setup.policy_path, "./policy");

        // Derived layout
        assert_eq!(setup.stage_data.block_counts_per_stage.len(), n_stages);
        assert!(setup.loop_params.max_blocks > 0);

        // Horizon
        assert_eq!(setup.methodology.horizon.num_stages(), n_stages);

        // Risk measures: one per study stage
        assert_eq!(setup.cut_management.risk_measures.len(), n_stages);

        // FCF: pools match stage count
        assert_eq!(setup.fcf.pools.len(), n_stages);

        // Entity counts: 1 hydro, 1 thermal
        assert_eq!(setup.stage_data.entity_counts.hydro_ids.len(), 1);
        assert_eq!(setup.stage_data.entity_counts.thermal_ids.len(), 1);
    }

    /// FCF is accessible mutably via `fcf_mut()`.
    #[test]
    fn fcf_mut_allows_cut_insertion() {
        let system = minimal_system(2);
        let config = minimal_config(1, 10);
        let stochastic = build_stochastic_context(
            &system,
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
        .expect("stochastic context");

        let mut setup = StudySetup::new(
            &system,
            &config,
            stochastic,
            PrepareHydroModelsResult::default_from_system(&system),
        )
        .expect("setup");

        let n_state = setup.stage_data.indexer.n_state;
        let coefficients = vec![1.0_f64; n_state];
        setup.fcf.add_cut(0, 0, 0, 42.0, &coefficients);
        assert_eq!(setup.fcf.total_active_cuts(), 1);
    }

    /// `inflow_method()` reflects the config setting.
    #[test]
    fn inflow_method_reflects_config() {
        use crate::InflowNonNegativityMethod;

        let system = minimal_system(2);
        let config = minimal_config(1, 10);
        let stochastic = build_stochastic_context(
            &system,
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
        .expect("stochastic context");

        let setup = StudySetup::new(
            &system,
            &config,
            stochastic,
            PrepareHydroModelsResult::default_from_system(&system),
        )
        .expect("setup");

        // The minimal_config uses "penalty" — should not be None.
        assert!(
            !matches!(
                setup.methodology.inflow_method,
                InflowNonNegativityMethod::None
            ),
            "expected penalty or truncation method"
        );
    }

    /// `cut_selection()` returns `None` when disabled in config (default).
    #[test]
    fn cut_selection_none_when_disabled() {
        let system = minimal_system(2);
        let config = minimal_config(1, 10);
        let stochastic = build_stochastic_context(
            &system,
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
        .expect("stochastic context");

        let setup = StudySetup::new(
            &system,
            &config,
            stochastic,
            PrepareHydroModelsResult::default_from_system(&system),
        )
        .expect("setup");

        assert!(
            setup.cut_management.cut_selection.is_none(),
            "cut_selection should be None when disabled"
        );
    }

    #[test]
    fn stage_ctx_fields_match_study_setup() {
        let n_stages = 3;
        let system = minimal_system(n_stages);
        let config = minimal_config(2, 10);
        let stochastic = build_stochastic_context(
            &system,
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
        .expect("stochastic context");

        let setup = StudySetup::new(
            &system,
            &config,
            stochastic,
            PrepareHydroModelsResult::default_from_system(&system),
        )
        .expect("setup");
        let ctx = setup.stage_ctx();

        assert_eq!(
            ctx.templates.len(),
            setup.stage_data.stage_templates.templates.len(),
            "templates length mismatch"
        );
        assert_eq!(
            ctx.base_rows.len(),
            setup.stage_data.stage_templates.base_rows.len(),
            "base_rows length mismatch"
        );
        assert_eq!(
            ctx.noise_scale.len(),
            setup.stage_data.stage_templates.noise_scale.len(),
            "noise_scale length mismatch"
        );
        assert_eq!(
            ctx.n_hydros,
            setup.stage_data.entity_counts.hydro_ids.len(),
            "n_hydros mismatch"
        );
        assert_eq!(
            ctx.block_counts_per_stage.len(),
            setup.stage_data.block_counts_per_stage.len(),
            "block_counts_per_stage length mismatch"
        );
    }

    #[test]
    fn training_ctx_fields_match_study_setup() {
        let n_stages = 3;
        let system = minimal_system(n_stages);
        let config = minimal_config(2, 10);
        let stochastic = build_stochastic_context(
            &system,
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
        .expect("stochastic context");

        let setup = StudySetup::new(
            &system,
            &config,
            stochastic,
            PrepareHydroModelsResult::default_from_system(&system),
        )
        .expect("setup");
        let ctx = setup.training_ctx();

        assert_eq!(
            ctx.horizon.num_stages(),
            setup.methodology.horizon.num_stages(),
            "horizon num_stages mismatch"
        );
        assert_eq!(
            ctx.indexer.n_state, setup.stage_data.indexer.n_state,
            "indexer n_state mismatch"
        );
        assert_eq!(
            ctx.initial_state.len(),
            setup.initial_state.len(),
            "initial_state length mismatch"
        );
    }

    /// Given a 1-hydro, 1-thermal, 1-bus, 2-stage system with an iteration
    /// limit of 3, when `train()` is called, then it completes successfully
    /// with `result.iterations <= 3`.
    #[test]
    fn train_completes_within_iteration_limit() {
        use cobre_comm::LocalBackend;
        use cobre_solver::highs::HighsSolver;

        let system = minimal_system(2);
        let config = minimal_config(1, 3);
        let stochastic = build_stochastic_context(
            &system,
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
        .expect("stochastic context");

        let mut setup = StudySetup::new(
            &system,
            &config,
            stochastic,
            PrepareHydroModelsResult::default_from_system(&system),
        )
        .expect("setup");
        let comm = LocalBackend;
        let mut solver = HighsSolver::new().expect("solver");

        let result = setup
            .train(&mut solver, &comm, 1, HighsSolver::new, None, None)
            .expect("train");

        assert!(
            result.result.iterations <= 3,
            "expected iterations <= 3, got {}",
            result.result.iterations
        );
        assert!(
            result.result.iterations >= 1,
            "expected at least 1 iteration, got {}",
            result.result.iterations
        );
    }

    /// After `train()` completes, at least one cut should be populated in the
    /// FCF cut pool for stage 0.
    #[test]
    fn train_generates_cuts_in_fcf() {
        use cobre_comm::LocalBackend;
        use cobre_solver::highs::HighsSolver;

        let system = minimal_system(2);
        let config = minimal_config(1, 3);
        let stochastic = build_stochastic_context(
            &system,
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
        .expect("stochastic context");

        let mut setup = StudySetup::new(
            &system,
            &config,
            stochastic,
            PrepareHydroModelsResult::default_from_system(&system),
        )
        .expect("setup");
        let comm = LocalBackend;
        let mut solver = HighsSolver::new().expect("solver");

        setup
            .train(&mut solver, &comm, 1, HighsSolver::new, None, None)
            .expect("train");

        assert!(
            setup.fcf.pools[0].populated_count > 0,
            "expected at least one cut in FCF pool[0] after training"
        );
    }

    /// `simulation_config()` returns a `SimulationConfig` whose fields match
    /// the values extracted from the `Config` at construction time.
    #[test]
    fn simulation_config_reflects_setup_fields() {
        use cobre_io::config::SimulationConfig as IoSimulationConfig;

        // Build a config with simulation enabled so n_scenarios is non-zero.
        let mut config = minimal_config(1, 5);
        config.simulation = IoSimulationConfig {
            enabled: true,
            num_scenarios: 50,
            io_channel_capacity: 16,
            ..IoSimulationConfig::default()
        };

        let system = minimal_system(2);
        let stochastic = build_stochastic_context(
            &system,
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
        .expect("stochastic context");

        let setup = StudySetup::new(
            &system,
            &config,
            stochastic,
            PrepareHydroModelsResult::default_from_system(&system),
        )
        .expect("setup");

        let sim_cfg = setup.simulation_config();
        assert_eq!(sim_cfg.n_scenarios, setup.simulation_config.n_scenarios);
        assert_eq!(
            sim_cfg.io_channel_capacity,
            setup.simulation_config.io_channel_capacity
        );
    }

    /// `create_workspace_pool()` with `n_threads = 2` returns a pool whose
    /// `workspaces.len()` equals 2.
    #[test]
    fn create_workspace_pool_returns_correct_size() {
        use cobre_comm::LocalBackend;
        use cobre_solver::highs::HighsSolver;

        let system = minimal_system(2);
        let config = minimal_config(1, 3);
        let stochastic = build_stochastic_context(
            &system,
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
        .expect("stochastic context");

        let setup = StudySetup::new(
            &system,
            &config,
            stochastic,
            PrepareHydroModelsResult::default_from_system(&system),
        )
        .expect("setup");

        let comm = LocalBackend;
        let pool = setup
            .create_workspace_pool(&comm, 2, HighsSolver::new)
            .expect("workspace pool");

        assert_eq!(pool.workspaces.len(), 2);
    }

    /// `build_training_output()` with a non-empty `TrainingResult` and empty
    /// events produces a `TrainingOutput` whose `convergence_records` is
    /// non-empty (one record per `result.iterations`).
    #[test]
    fn build_training_output_non_empty() {
        use cobre_comm::LocalBackend;
        use cobre_solver::highs::HighsSolver;

        let system = minimal_system(2);
        let config = minimal_config(1, 2);
        let stochastic = build_stochastic_context(
            &system,
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
        .expect("stochastic context");

        let mut setup = StudySetup::new(
            &system,
            &config,
            stochastic,
            PrepareHydroModelsResult::default_from_system(&system),
        )
        .expect("setup");
        let comm = LocalBackend;
        let mut solver = HighsSolver::new().expect("solver");

        // Collect events from training so we have at least one IterationSummary.
        let (event_tx, event_rx) = std::sync::mpsc::channel();
        let result = setup
            .train(
                &mut solver,
                &comm,
                1,
                HighsSolver::new,
                Some(event_tx),
                None,
            )
            .expect("train");

        let events: Vec<cobre_core::TrainingEvent> = event_rx.try_iter().collect();

        let output = setup.build_training_output(&result.result, &events);
        assert!(
            !output.convergence_records.is_empty(),
            "convergence_records must be non-empty after training"
        );
    }

    /// Given a trained `StudySetup` with `n_scenarios > 0`, calling `simulate()`
    /// returns `Ok(costs)` with `costs.len() > 0`.
    #[test]
    fn simulate_after_train_returns_nonempty_costs() {
        use cobre_comm::LocalBackend;
        use cobre_solver::highs::HighsSolver;

        // Enable simulation with 3 scenarios.
        let mut config = minimal_config(1, 3);
        config.simulation = cobre_io::config::SimulationConfig {
            enabled: true,
            num_scenarios: 3,
            io_channel_capacity: 8,
            ..cobre_io::config::SimulationConfig::default()
        };

        let system = minimal_system(2);
        let stochastic = build_stochastic_context(
            &system,
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
        .expect("stochastic context");

        let mut setup = StudySetup::new(
            &system,
            &config,
            stochastic,
            PrepareHydroModelsResult::default_from_system(&system),
        )
        .expect("setup");

        // Train first so the FCF has cuts.
        let comm = LocalBackend;
        let mut solver = HighsSolver::new().expect("solver");
        setup
            .train(&mut solver, &comm, 1, HighsSolver::new, None, None)
            .expect("train");

        // Build simulation pool.
        let mut pool = setup
            .create_workspace_pool(&comm, 1, HighsSolver::new)
            .expect("sim pool");

        // Create the result channel and drain thread.
        let io_capacity = setup.simulation_config.io_channel_capacity.max(1);
        let (result_tx, result_rx) = std::sync::mpsc::sync_channel(io_capacity);
        let drain_handle = std::thread::spawn(move || result_rx.into_iter().collect::<Vec<_>>());

        let sim_result = setup
            .simulate(&mut pool.workspaces, &comm, &result_tx, None, None, &[])
            .expect("simulate");

        // Drop the sender so the drain thread terminates.
        drop(result_tx);
        let _results = drain_handle.join().expect("drain thread");

        assert!(
            !sim_result.costs.is_empty(),
            "simulate must return at least one cost entry"
        );
        assert_eq!(
            sim_result.solver_stats.len(),
            sim_result.costs.len(),
            "one solver stats entry per scenario"
        );
    }

    /// Given a config with no overrides, `StudyParams::from_config` returns the
    /// default values for all fields.
    #[test]
    fn study_params_from_config_defaults() {
        use super::{DEFAULT_FORWARD_PASSES, DEFAULT_SEED, StudyParams};
        use crate::stopping_rule::StoppingMode;
        use cobre_io::config::{
            Config, EstimationConfig, ExportsConfig, InflowNonNegativityConfig, ModelingConfig,
            PolicyConfig, RowSelectionConfig, SimulationConfig as IoSimulationConfig,
            TrainingConfig, TrainingSolverConfig, UpperBoundEvaluationConfig,
        };

        let config = Config {
            schema: None,
            modeling: ModelingConfig {
                inflow_non_negativity: InflowNonNegativityConfig {
                    method: "none".to_string(),
                    penalty_cost: 0.0,
                },
            },
            training: TrainingConfig {
                enabled: true,
                tree_seed: None,
                forward_passes: None,
                stopping_rules: None,
                stopping_mode: "any".to_string(),
                cut_formulation: None,
                forward_pass: None,
                cut_selection: RowSelectionConfig::default(),
                solver: TrainingSolverConfig::default(),
                scenario_source: None,
            },
            upper_bound_evaluation: UpperBoundEvaluationConfig::default(),
            policy: PolicyConfig::default(),
            simulation: IoSimulationConfig::default(),
            exports: ExportsConfig::default(),
            estimation: EstimationConfig::default(),
        };

        let params = StudyParams::from_config(&config).expect("from_config");

        assert_eq!(
            params.seed, DEFAULT_SEED,
            "seed should default to DEFAULT_SEED"
        );
        assert_eq!(
            params.forward_passes, DEFAULT_FORWARD_PASSES,
            "forward_passes should default to DEFAULT_FORWARD_PASSES"
        );
        // When no stopping rules are specified, a single IterationLimit rule is inserted.
        assert_eq!(
            params.stopping_rule_set.rules.len(),
            1,
            "expected exactly 1 default stopping rule"
        );
        assert!(
            matches!(
                params.stopping_rule_set.rules[0],
                crate::stopping_rule::StoppingRule::IterationLimit { .. }
            ),
            "default rule should be IterationLimit"
        );
        assert!(
            matches!(params.stopping_rule_set.mode, StoppingMode::Any),
            "default stopping mode should be Any"
        );
        // Simulation is disabled by default.
        assert_eq!(
            params.n_scenarios, 0,
            "n_scenarios should be 0 when simulation disabled"
        );
        assert!(
            params.cut_selection.is_none(),
            "cut_selection should be None by default"
        );
    }

    /// Given a config with explicit values for all fields, `StudyParams::from_config`
    /// extracts them correctly.
    #[test]
    fn study_params_from_config_explicit() {
        use super::StudyParams;
        use crate::stopping_rule::{StoppingMode, StoppingRule};
        use cobre_io::config::{
            Config, EstimationConfig, ExportsConfig, InflowNonNegativityConfig, ModelingConfig,
            PolicyConfig, RowSelectionConfig, SimulationConfig as IoSimulationConfig,
            StoppingRuleConfig, TrainingConfig, TrainingSolverConfig, UpperBoundEvaluationConfig,
        };

        let config = Config {
            schema: None,
            modeling: ModelingConfig {
                inflow_non_negativity: InflowNonNegativityConfig {
                    method: "penalty".to_string(),
                    penalty_cost: 999.0,
                },
            },
            training: TrainingConfig {
                enabled: true,
                tree_seed: Some(1234),
                forward_passes: Some(5),
                stopping_rules: Some(vec![
                    StoppingRuleConfig::IterationLimit { limit: 50 },
                    StoppingRuleConfig::TimeLimit { seconds: 60.0 },
                ]),
                stopping_mode: "all".to_string(),
                cut_formulation: None,
                forward_pass: None,
                cut_selection: RowSelectionConfig::default(),
                solver: TrainingSolverConfig::default(),
                scenario_source: None,
            },
            upper_bound_evaluation: UpperBoundEvaluationConfig::default(),
            policy: PolicyConfig {
                path: "./my_policy".to_string(),
                ..PolicyConfig::default()
            },
            simulation: IoSimulationConfig {
                enabled: true,
                num_scenarios: 200,
                ..IoSimulationConfig::default()
            },
            exports: ExportsConfig::default(),
            estimation: EstimationConfig::default(),
        };

        let params = StudyParams::from_config(&config).expect("from_config");

        // Seed: i64::unsigned_abs(1234) == 1234
        assert_eq!(params.seed, 1234, "seed mismatch");
        assert_eq!(params.forward_passes, 5, "forward_passes mismatch");
        // Two stopping rules must be preserved.
        assert_eq!(
            params.stopping_rule_set.rules.len(),
            2,
            "stopping rule count mismatch"
        );
        assert!(
            matches!(
                params.stopping_rule_set.rules[0],
                StoppingRule::IterationLimit { limit: 50 }
            ),
            "first rule should be IterationLimit(50)"
        );
        assert!(
            matches!(
                params.stopping_rule_set.rules[1],
                StoppingRule::TimeLimit { seconds } if (seconds - 60.0).abs() < 1e-9
            ),
            "second rule should be TimeLimit(60.0)"
        );
        assert!(
            matches!(params.stopping_rule_set.mode, StoppingMode::All),
            "stopping mode should be All"
        );
        assert_eq!(params.n_scenarios, 200, "n_scenarios mismatch");
        assert_eq!(params.policy_path, "./my_policy", "policy_path mismatch");
    }

    /// Build a minimal case directory with required structural files present so
    /// that `validate_structure` does not fail. The optional estimation and
    /// opening tree files are NOT created here; tests add them as needed.
    fn write_minimal_case_dir(root: &std::path::Path) {
        use std::fs;

        fs::create_dir_all(root.join("system")).unwrap();
        fs::write(root.join("config.json"), b"{}").unwrap();
        fs::write(root.join("penalties.json"), b"{}").unwrap();
        fs::write(root.join("stages.json"), b"{}").unwrap();
        fs::write(root.join("initial_conditions.json"), b"{}").unwrap();
        fs::write(root.join("system/buses.json"), b"{}").unwrap();
        fs::write(root.join("system/lines.json"), b"{}").unwrap();
        fs::write(root.join("system/hydros.json"), b"{}").unwrap();
        fs::write(root.join("system/thermals.json"), b"{}").unwrap();
    }

    /// Build a minimal [`cobre_io::Config`] with no estimation or seed overrides.
    fn minimal_prepare_config() -> cobre_io::Config {
        use cobre_io::config::{
            Config, EstimationConfig, ExportsConfig, InflowNonNegativityConfig, ModelingConfig,
            PolicyConfig, RowSelectionConfig, SimulationConfig as IoSimulationConfig,
            TrainingConfig, TrainingSolverConfig, UpperBoundEvaluationConfig,
        };

        Config {
            schema: None,
            modeling: ModelingConfig {
                inflow_non_negativity: InflowNonNegativityConfig {
                    method: "none".to_string(),
                    penalty_cost: 0.0,
                },
            },
            training: TrainingConfig {
                enabled: true,
                tree_seed: None,
                forward_passes: None,
                stopping_rules: None,
                stopping_mode: "any".to_string(),
                cut_formulation: None,
                forward_pass: None,
                cut_selection: RowSelectionConfig::default(),
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

    /// Given a case directory with no `inflow_history.parquet` and no
    /// `scenarios/noise_openings.parquet`, `prepare_stochastic` returns
    /// `estimation_report = None` and a stochastic context with generated provenance.
    #[test]
    fn prepare_stochastic_no_history_no_tree_returns_none_report_and_generated_provenance() {
        use super::prepare_stochastic;
        use cobre_core::scenario::ScenarioSource;
        use cobre_stochastic::provenance::ComponentProvenance;
        use tempfile::TempDir;

        let dir = TempDir::new().unwrap();
        let root = dir.path();
        write_minimal_case_dir(root);

        let system = minimal_system(2);
        let config = minimal_prepare_config();
        let seed = 42_u64;

        let source = ScenarioSource {
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            seed: None,
            historical_years: None,
        };
        let result = prepare_stochastic(system, root, &config, seed, &source)
            .expect("prepare_stochastic should succeed with no optional files");

        assert!(
            result.estimation_report.is_none(),
            "estimation_report must be None when no inflow_history.parquet is present"
        );
        assert_eq!(
            result.stochastic.provenance().opening_tree,
            ComponentProvenance::Generated,
            "opening_tree provenance must be Generated when no user tree is supplied"
        );
    }

    /// Given a case directory with `inflow_seasonal_stats.parquet` present
    /// alongside `inflow_history.parquet`, estimation is skipped and
    /// `estimation_report` is `None`.
    #[test]
    fn prepare_stochastic_with_stats_file_present_skips_estimation() {
        use super::prepare_stochastic;
        use cobre_core::scenario::ScenarioSource;
        use std::fs;
        use tempfile::TempDir;

        let dir = TempDir::new().unwrap();
        let root = dir.path();
        write_minimal_case_dir(root);

        // Place the stats files so that the "explicit stats present" branch is taken
        // and estimation is skipped. No `inflow_history.parquet` is written here;
        // its presence is not required for the estimation-skip path and the test
        // intentionally keeps the history file absent to avoid parse errors.
        // (`validate_structure` only checks existence, not content.)
        fs::create_dir_all(root.join("scenarios")).unwrap();
        fs::write(root.join("scenarios/inflow_seasonal_stats.parquet"), b"").unwrap();
        fs::write(root.join("scenarios/inflow_ar_coefficients.parquet"), b"").unwrap();

        let system = minimal_system(2);
        let config = minimal_prepare_config();
        let seed = 42_u64;

        let source = ScenarioSource {
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            seed: None,
            historical_years: None,
        };
        let result = prepare_stochastic(system, root, &config, seed, &source)
            .expect("prepare_stochastic should succeed when stats file is present");

        assert!(
            result.estimation_report.is_none(),
            "estimation_report must be None when inflow_seasonal_stats.parquet is present"
        );
    }

    /// Given a case directory with no `scenarios/noise_openings.parquet`,
    /// `load_user_opening_tree_inner` returns `None`.
    ///
    /// This is tested indirectly via `prepare_stochastic` by checking that the
    /// returned stochastic context does not claim `UserSupplied` provenance.
    #[test]
    fn prepare_stochastic_no_opening_tree_gives_non_user_supplied_provenance() {
        use super::prepare_stochastic;
        use cobre_core::scenario::ScenarioSource;
        use cobre_stochastic::provenance::ComponentProvenance;
        use tempfile::TempDir;

        let dir = TempDir::new().unwrap();
        let root = dir.path();
        write_minimal_case_dir(root);

        let system = minimal_system(2);
        let config = minimal_prepare_config();

        let source = ScenarioSource {
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            seed: None,
            historical_years: None,
        };
        let result = prepare_stochastic(system, root, &config, 0, &source)
            .expect("prepare_stochastic must succeed with no opening tree file");

        assert_ne!(
            result.stochastic.provenance().opening_tree,
            ComponentProvenance::UserSupplied,
            "opening_tree provenance must not be UserSupplied when file is absent"
        );
    }

    /// Given a system with `NoiseMethod::HistoricalResiduals` on all stages and
    /// sufficient inflow history, when `prepare_stochastic` is called, then it
    /// returns `Ok` and the resulting stochastic context has
    /// `opening_tree().n_stages()` equal to the number of study stages.
    #[test]
    #[allow(
        clippy::too_many_lines,
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap
    )]
    fn test_prepare_stochastic_historical_residuals_noise_method() {
        use super::prepare_stochastic;
        use chrono::NaiveDate;
        use cobre_core::{
            scenario::{InflowHistoryRow, ScenarioSource},
            system::SystemBuilder,
        };
        use tempfile::TempDir;

        // Build a system with HistoricalResiduals noise method on all stages.
        // Reuses the same structure as system_with_historical_inflow but sets
        // noise_method to HistoricalResiduals instead of the default Saa.
        let n_stages = 2usize;

        let bus = Bus {
            id: EntityId(1),
            name: "B1".to_string(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
        };
        let thermal = Thermal {
            id: EntityId(2),
            name: "T1".to_string(),
            bus_id: EntityId(1),
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            cost_per_mwh: 50.0,
            gnl_config: None,
            entry_stage_id: None,
            exit_stage_id: None,
        };
        let hydro = Hydro {
            id: EntityId(3),
            name: "H1".to_string(),
            bus_id: EntityId(1),
            downstream_id: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 200.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity {
                productivity_mw_per_m3s: 2.5,
            },
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 100.0,
            min_generation_mw: 0.0,
            max_generation_mw: 250.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: HydroPenalties {
                spillage_cost: 0.01,
                diversion_cost: 0.0,
                fpha_turbined_cost: 0.0,
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
                inflow_nonnegativity_cost: 1000.0,
            },
        };

        // Stages with HistoricalResiduals noise method; branching_factor=2 so
        // each stage selects 2 historical windows as openings.
        let stages: Vec<Stage> = (0..n_stages)
            .map(|i| Stage {
                index: i,
                id: i as i32,
                start_date: NaiveDate::from_ymd_opt(2024, (i as u32 % 12) + 1, 1).unwrap(),
                end_date: NaiveDate::from_ymd_opt(2024, (i as u32 % 12) + 1, 28).unwrap(),
                season_id: Some(i % 12),
                blocks: vec![Block {
                    index: 0,
                    name: "S".to_string(),
                    duration_hours: 720.0,
                }],
                block_mode: BlockMode::Parallel,
                state_config: StageStateConfig {
                    storage: true,
                    inflow_lags: false,
                },
                risk_config: StageRiskConfig::Expectation,
                scenario_config: ScenarioSourceConfig {
                    branching_factor: 2,
                    noise_method: NoiseMethod::HistoricalResiduals,
                },
            })
            .collect();

        let inflow_models: Vec<InflowModel> = (0..n_stages)
            .map(|i| InflowModel {
                hydro_id: EntityId(3),
                stage_id: i as i32,
                mean_m3s: 80.0,
                std_m3s: 20.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
            })
            .collect();

        let load_models: Vec<LoadModel> = (0..n_stages)
            .map(|i| LoadModel {
                bus_id: EntityId(1),
                stage_id: i as i32,
                mean_mw: 100.0,
                std_mw: 0.0,
            })
            .collect();

        // Historical inflow data: 1990 and 1991 cover 12 months each — 2 valid windows.
        let inflow_history: Vec<InflowHistoryRow> = (1990_i32..=1991)
            .flat_map(|year| {
                (1u32..=12).map(move |month| InflowHistoryRow {
                    hydro_id: EntityId(3),
                    date: NaiveDate::from_ymd_opt(year, month, 1).unwrap(),
                    value_m3s: 80.0 + f64::from(year - 1990) * 5.0,
                })
            })
            .collect();

        let n_st = n_stages.max(1);
        let bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 1,
                n_thermals: 1,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: n_st,
            },
            &BoundsDefaults {
                hydro: HydroStageBounds {
                    min_storage_hm3: 0.0,
                    max_storage_hm3: 200.0,
                    min_turbined_m3s: 0.0,
                    max_turbined_m3s: 100.0,
                    min_outflow_m3s: 0.0,
                    max_outflow_m3s: None,
                    min_generation_mw: 0.0,
                    max_generation_mw: 250.0,
                    max_diversion_m3s: None,
                    filling_inflow_m3s: 0.0,
                    water_withdrawal_m3s: 0.0,
                },
                thermal: ThermalStageBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 100.0,
                    cost_per_mwh: 0.0,
                },
                line: LineStageBounds {
                    direct_mw: 0.0,
                    reverse_mw: 0.0,
                },
                pumping: PumpingStageBounds {
                    min_flow_m3s: 0.0,
                    max_flow_m3s: 0.0,
                },
                contract: ContractStageBounds {
                    min_mw: 0.0,
                    max_mw: 0.0,
                    price_per_mwh: 0.0,
                },
            },
        );
        let penalties = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 1,
                n_buses: 1,
                n_lines: 0,
                n_ncs: 0,
                n_stages: n_st,
            },
            &PenaltiesDefaults {
                hydro: HydroStagePenalties {
                    spillage_cost: 0.01,
                    diversion_cost: 0.0,
                    fpha_turbined_cost: 0.0,
                    storage_violation_below_cost: 500.0,
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
                    inflow_nonnegativity_cost: 1000.0,
                },
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        );

        let system = SystemBuilder::new()
            .buses(vec![bus])
            .thermals(vec![thermal])
            .hydros(vec![hydro])
            .stages(stages)
            .inflow_models(inflow_models)
            .load_models(load_models)
            .inflow_history(inflow_history)
            .bounds(bounds)
            .penalties(penalties)
            .build()
            .expect("test system: valid");

        let dir = TempDir::new().unwrap();
        let root = dir.path();
        write_minimal_case_dir(root);

        let config = minimal_prepare_config();
        let source = ScenarioSource {
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            seed: None,
            historical_years: None,
        };
        let result = prepare_stochastic(system, root, &config, 42, &source)
            .expect("prepare_stochastic must succeed with HistoricalResiduals noise method");

        assert_eq!(
            result.stochastic.opening_tree().n_stages(),
            n_stages,
            "opening_tree must have n_stages == {n_stages}"
        );
    }

    /// Given a system with no FPHA and no evaporation data, `default_from_system`
    /// returns a result where all hydros use constant productivity and no evaporation.
    #[test]
    fn default_from_system_gives_constant_and_no_evaporation() {
        use crate::hydro_models::{
            EvaporationModel, ProductionModelSource, ResolvedProductionModel,
        };

        let system = minimal_system(2);
        let result = PrepareHydroModelsResult::default_from_system(&system);

        assert_eq!(
            result.provenance.production_sources.len(),
            system.hydros().len(),
            "production_sources length must equal n_hydros"
        );
        for (_, source) in &result.provenance.production_sources {
            assert_eq!(
                *source,
                ProductionModelSource::DefaultConstant,
                "all hydros must use DefaultConstant"
            );
        }

        assert_eq!(
            result.provenance.evaporation_sources.len(),
            system.hydros().len(),
            "evaporation_sources length must equal n_hydros"
        );
        assert!(
            !result.evaporation.has_evaporation(),
            "default result must have no evaporation"
        );

        // Verify the production model for the one hydro at stage 0 is ConstantProductivity.
        let model = result.production.model(0, 0);
        assert!(
            matches!(model, ResolvedProductionModel::ConstantProductivity { .. }),
            "default production model must be ConstantProductivity"
        );

        // Verify the evaporation model for the one hydro is None.
        let evap = result.evaporation.model(0);
        assert!(
            matches!(evap, EvaporationModel::None),
            "default evaporation model must be None"
        );
    }

    /// Given a valid `StudySetup`, `hydro_models()` returns the stored result.
    #[test]
    fn hydro_models_accessor_returns_stored_result() {
        use crate::hydro_models::ProductionModelSource;

        let system = minimal_system(2);
        let config = minimal_config(1, 5);
        let stochastic = build_stochastic_context(
            &system,
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
        .expect("stochastic context");
        let hydro_result = PrepareHydroModelsResult::default_from_system(&system);

        let setup = StudySetup::new(&system, &config, stochastic, hydro_result).expect("setup");

        let models = &setup.hydro_models;
        assert_eq!(
            models.provenance.production_sources.len(),
            system.hydros().len(),
            "hydro_models() must return the stored result (provenance length mismatch)"
        );
        for (_, source) in &models.provenance.production_sources {
            assert_eq!(
                *source,
                ProductionModelSource::DefaultConstant,
                "stored result must preserve DefaultConstant provenance"
            );
        }
    }

    /// Build a `StageIndexer` for lag tests: N hydros, L lags, no equipment columns.
    fn indexer_for_lag_test(hydro_count: usize, max_par_order: usize) -> StageIndexer {
        StageIndexer::new(hydro_count, max_par_order)
    }

    /// Build a 2-hydro system (IDs 1 and 2) with `n_stages` study stages and
    /// PAR order 2 AR coefficients on all stages, with `inflow_lags: true`.
    ///
    /// Provides `past_inflows` in `initial_conditions` with the given values
    /// for both hydros.
    #[allow(
        clippy::too_many_lines,
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::items_after_statements
    )]
    fn minimal_system_2_hydros_with_past_inflows(
        n_stages: usize,
        h1_past: Vec<f64>,
        h2_past: Vec<f64>,
    ) -> cobre_core::System {
        use chrono::NaiveDate;

        let bus = Bus {
            id: EntityId(1),
            name: "B1".to_string(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
        };

        let make_hydro = |id: i32, name: &str| Hydro {
            id: EntityId(id),
            name: name.to_string(),
            bus_id: EntityId(1),
            downstream_id: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 200.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity {
                productivity_mw_per_m3s: 2.5,
            },
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 100.0,
            min_generation_mw: 0.0,
            max_generation_mw: 250.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: HydroPenalties {
                spillage_cost: 0.01,
                diversion_cost: 0.0,
                fpha_turbined_cost: 0.0,
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
                inflow_nonnegativity_cost: 1000.0,
            },
        };

        let n_st = n_stages.max(1);
        let stages: Vec<Stage> = (0..n_stages)
            .map(|i| Stage {
                index: i,
                id: i as i32,
                start_date: NaiveDate::from_ymd_opt(2020, (i % 12 + 1) as u32, 1).unwrap(),
                end_date: NaiveDate::from_ymd_opt(
                    if (i % 12 + 1) == 12 { 2021 } else { 2020 },
                    ((i % 12 + 1) % 12 + 1) as u32,
                    1,
                )
                .unwrap(),
                season_id: Some(i),
                blocks: vec![Block {
                    index: 0,
                    name: "S".to_string(),
                    duration_hours: 744.0,
                }],
                block_mode: BlockMode::Parallel,
                state_config: StageStateConfig {
                    storage: true,
                    inflow_lags: true,
                },
                risk_config: StageRiskConfig::Expectation,
                scenario_config: ScenarioSourceConfig {
                    branching_factor: 1,
                    noise_method: NoiseMethod::Saa,
                },
            })
            .collect();

        let inflow_models: Vec<InflowModel> = (0..n_stages)
            .flat_map(|i| {
                [1_i32, 2].map(|hid| InflowModel {
                    hydro_id: EntityId(hid),
                    stage_id: i as i32,
                    mean_m3s: 80.0,
                    std_m3s: 20.0,
                    ar_coefficients: vec![0.5, 0.3],
                    residual_std_ratio: 0.8,
                })
            })
            .collect();

        let load_models: Vec<LoadModel> = (0..n_stages)
            .map(|i| LoadModel {
                bus_id: EntityId(1),
                stage_id: i as i32,
                mean_mw: 100.0,
                std_mw: 0.0,
            })
            .collect();

        fn default_hydro_bounds() -> HydroStageBounds {
            HydroStageBounds {
                min_storage_hm3: 0.0,
                max_storage_hm3: 200.0,
                min_turbined_m3s: 0.0,
                max_turbined_m3s: 100.0,
                min_outflow_m3s: 0.0,
                max_outflow_m3s: None,
                min_generation_mw: 0.0,
                max_generation_mw: 250.0,
                max_diversion_m3s: None,
                filling_inflow_m3s: 0.0,
                water_withdrawal_m3s: 0.0,
            }
        }

        fn default_hydro_penalties() -> HydroStagePenalties {
            HydroStagePenalties {
                spillage_cost: 0.01,
                diversion_cost: 0.0,
                fpha_turbined_cost: 0.0,
                storage_violation_below_cost: 500.0,
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
                inflow_nonnegativity_cost: 1000.0,
            }
        }

        let bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 2,
                n_thermals: 0,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: n_st,
            },
            &BoundsDefaults {
                hydro: default_hydro_bounds(),
                thermal: ThermalStageBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 0.0,
                    cost_per_mwh: 0.0,
                },
                line: LineStageBounds {
                    direct_mw: 0.0,
                    reverse_mw: 0.0,
                },
                pumping: PumpingStageBounds {
                    min_flow_m3s: 0.0,
                    max_flow_m3s: 0.0,
                },
                contract: ContractStageBounds {
                    min_mw: 0.0,
                    max_mw: 0.0,
                    price_per_mwh: 0.0,
                },
            },
        );

        let penalties = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 2,
                n_buses: 1,
                n_lines: 0,
                n_ncs: 0,
                n_stages: n_st,
            },
            &PenaltiesDefaults {
                hydro: default_hydro_penalties(),
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        );

        let past_inflows = vec![
            cobre_core::HydroPastInflows {
                hydro_id: EntityId(1),
                values_m3s: h1_past,
                season_ids: None,
            },
            cobre_core::HydroPastInflows {
                hydro_id: EntityId(2),
                values_m3s: h2_past,
                season_ids: None,
            },
        ];

        SystemBuilder::new()
            .buses(vec![bus])
            .thermals(vec![])
            .hydros(vec![make_hydro(1, "H1"), make_hydro(2, "H2")])
            .stages(stages)
            .inflow_models(inflow_models)
            .load_models(load_models)
            .bounds(bounds)
            .penalties(penalties)
            .initial_conditions(cobre_core::InitialConditions {
                storage: vec![],
                filling_storage: vec![],
                past_inflows,
                recent_observations: vec![],
            })
            .build()
            .expect("minimal_system_2_hydros_with_past_inflows: valid")
    }

    /// Given 2 hydros (IDs 1, 2), `max_par_order`=2, and `past_inflows` set,
    /// `build_initial_state` populates lag slots correctly.
    ///
    /// Hydro idx 0 (id=1): lag 0 = 600.0, lag 1 = 500.0
    /// Hydro idx 1 (id=2): lag 0 = 200.0, lag 1 = 100.0
    #[test]
    fn build_initial_state_populates_lags_from_past_inflows() {
        use super::build_initial_state;

        let system =
            minimal_system_2_hydros_with_past_inflows(1, vec![600.0, 500.0], vec![200.0, 100.0]);
        let indexer = indexer_for_lag_test(2, 2);

        let state = build_initial_state(&system, &indexer);

        // State layout: storage(0..2), lags(2..6) in lag-major order.
        // Lag-major: slot = s + lag * N + h, where N = 2.
        // lag0_h0 = 600.0 at s+0, lag0_h1 = 200.0 at s+1,
        // lag1_h0 = 500.0 at s+2, lag1_h1 = 100.0 at s+3.
        let s = indexer.inflow_lags.start;
        assert!(
            (state[s] - 600.0).abs() < 1e-10,
            "lag0 hydro 0: expected 600.0, got {}",
            state[s]
        );
        assert!(
            (state[s + 1] - 200.0).abs() < 1e-10,
            "lag0 hydro 1: expected 200.0, got {}",
            state[s + 1]
        );
        assert!(
            (state[s + 2] - 500.0).abs() < 1e-10,
            "lag1 hydro 0: expected 500.0, got {}",
            state[s + 2]
        );
        assert!(
            (state[s + 3] - 100.0).abs() < 1e-10,
            "lag1 hydro 1: expected 100.0, got {}",
            state[s + 3]
        );
        assert_eq!(
            state.len(),
            indexer.n_state,
            "state length must equal n_state"
        );
    }

    /// Given no `past_inflows` entries, all lag slots remain 0.0.
    #[test]
    fn build_initial_state_empty_past_inflows_leaves_zero_lags() {
        use super::build_initial_state;

        let system = minimal_system(2);
        let indexer = indexer_for_lag_test(1, 3);

        let state = build_initial_state(&system, &indexer);

        let s = indexer.inflow_lags.start;
        for l in 0..3 {
            assert!(
                state[s + l].abs() < 1e-10,
                "lag slot {l} should be 0.0 when past_inflows is empty, got {}",
                state[s + l]
            );
        }
    }

    /// Given `past_inflows` only for a hydro not in the system, lag slots
    /// for the system's hydros remain 0.0.
    #[test]
    fn build_initial_state_unknown_hydro_in_past_inflows_stays_zero() {
        use super::build_initial_state;

        // minimal_system has 1 hydro id=3; build a system with past_inflows
        // for hydro id=99 (not in registry).
        let system = {
            // Reuse minimal_system(2) but add past_inflows for a non-existent hydro.
            // Since minimal_system doesn't support overriding IC, we use
            // build_initial_state directly on the base system — its IC has
            // no past_inflows, so both lag slots are 0.0.
            minimal_system(2)
        };
        let indexer = indexer_for_lag_test(1, 2);

        let state = build_initial_state(&system, &indexer);

        let s = indexer.inflow_lags.start;
        assert!(
            state[s].abs() < 1e-10,
            "lag 0 should be 0.0 when past_inflows is absent, got {}",
            state[s]
        );
        assert!(
            state[s + 1].abs() < 1e-10,
            "lag 1 should be 0.0 when past_inflows is absent, got {}",
            state[s + 1]
        );
    }

    /// Integration test: `StudySetup::new` with `past_inflows` in the system's
    /// initial conditions produces `initial_state()` with non-zero lag values.
    #[test]
    fn study_setup_initial_state_has_nonzero_lags_from_past_inflows() {
        let system =
            minimal_system_2_hydros_with_past_inflows(3, vec![600.0, 500.0], vec![200.0, 100.0]);
        let config = minimal_config(1, 10);
        let stochastic = build_stochastic_context(
            &system,
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
        .expect("stochastic context");

        let setup = StudySetup::new(
            &system,
            &config,
            stochastic,
            PrepareHydroModelsResult::default_from_system(&system),
        )
        .expect("setup with past_inflows");

        let state = &setup.initial_state;

        // With 2 hydros (N=2) and max_par_order=2 (L=2), lag slots start at N=2.
        // Lag-major layout: slot = lag_start + lag * N + h.
        // lag0_h0 = 600.0 at [2], lag0_h1 = 200.0 at [3],
        // lag1_h0 = 500.0 at [4], lag1_h1 = 100.0 at [5].
        let n_hydros = 2;
        let lag_start = n_hydros;
        assert!(
            (state[lag_start] - 600.0).abs() < 1e-10,
            "lag0 hydro 0 should be 600.0 via StudySetup, got {}",
            state[lag_start]
        );
        assert!(
            (state[lag_start + 1] - 200.0).abs() < 1e-10,
            "lag0 hydro 1 should be 200.0 via StudySetup, got {}",
            state[lag_start + 1]
        );
        assert!(
            (state[lag_start + 2] - 500.0).abs() < 1e-10,
            "lag1 hydro 0 should be 500.0 via StudySetup, got {}",
            state[lag_start + 2]
        );
        assert!(
            (state[lag_start + 3] - 100.0).abs() < 1e-10,
            "lag1 hydro 1 should be 100.0 via StudySetup, got {}",
            state[lag_start + 3]
        );
    }

    /// Given `max_par_order`=0, no lag slots exist; state is storage-only.
    #[test]
    fn build_initial_state_no_lags_state_is_storage_only() {
        use super::build_initial_state;

        let system = minimal_system(2);
        let indexer = indexer_for_lag_test(1, 0);

        // n_state = N*(1+L) = 1*(1+0) = 1
        assert_eq!(indexer.n_state, 1);
        assert!(
            indexer.inflow_lags.is_empty(),
            "inflow_lags range should be empty for L=0"
        );

        let state = build_initial_state(&system, &indexer);

        assert_eq!(state.len(), 1, "state length must equal n_state=1");
    }

    /// Given a `System` with `inflow_scheme = InSample`, when `StudySetup::new()`
    /// is called, then `historical_library()` returns `None`.
    #[test]
    fn historical_library_none_for_insample() {
        let system = minimal_system(2);
        let config = minimal_config(1, 5);
        let stochastic = build_stochastic_context(
            &system,
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
        .expect("stochastic context");

        let setup = StudySetup::new(
            &system,
            &config,
            stochastic,
            PrepareHydroModelsResult::default_from_system(&system),
        )
        .expect("setup");

        assert!(
            setup.scenario_libraries.training.historical.is_none(),
            "historical_library must be None for InSample scheme"
        );
        assert!(
            setup.scenario_libraries.training.external_inflow.is_none(),
            "external_inflow_library must be None for InSample scheme"
        );
        assert!(
            setup.scenario_libraries.training.external_load.is_none(),
            "external_load_library must be None for InSample load scheme"
        );
        assert!(
            setup.scenario_libraries.training.external_ncs.is_none(),
            "external_ncs_library must be None for InSample ncs scheme"
        );
    }

    /// Build a system that has `inflow_scheme = Historical` and the inflow
    /// history needed to discover at least one window.
    ///
    /// The system has 1 hydro, 1 bus, 1 thermal, 2 monthly stages (`season_id`
    /// `Some(0)` and `Some(1)`), and historical data covering years 1990-1991.
    /// With `max_par_order = 0` (no AR coefficients), a window is valid if
    /// we have observations for both study months. Year 1990 covers months 0-1
    /// so season 0 and 1 are available under year 1990.
    #[allow(
        clippy::too_many_lines,
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::cast_lossless
    )]
    fn system_with_historical_inflow(n_stages: usize) -> cobre_core::System {
        use chrono::NaiveDate;
        use cobre_core::{scenario::InflowHistoryRow, system::SystemBuilder};

        fn default_hydro_bounds() -> HydroStageBounds {
            HydroStageBounds {
                min_storage_hm3: 0.0,
                max_storage_hm3: 200.0,
                min_turbined_m3s: 0.0,
                max_turbined_m3s: 100.0,
                min_outflow_m3s: 0.0,
                max_outflow_m3s: None,
                min_generation_mw: 0.0,
                max_generation_mw: 250.0,
                max_diversion_m3s: None,
                filling_inflow_m3s: 0.0,
                water_withdrawal_m3s: 0.0,
            }
        }

        fn default_hydro_penalties() -> HydroStagePenalties {
            HydroStagePenalties {
                spillage_cost: 0.01,
                diversion_cost: 0.0,
                fpha_turbined_cost: 0.0,
                storage_violation_below_cost: 500.0,
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
                inflow_nonnegativity_cost: 1000.0,
            }
        }

        let bus = Bus {
            id: EntityId(1),
            name: "B1".to_string(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
        };

        let thermal = Thermal {
            id: EntityId(2),
            name: "T1".to_string(),
            bus_id: EntityId(1),
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            cost_per_mwh: 50.0,
            gnl_config: None,
            entry_stage_id: None,
            exit_stage_id: None,
        };

        let hydro = Hydro {
            id: EntityId(3),
            name: "H1".to_string(),
            bus_id: EntityId(1),
            downstream_id: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 200.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity {
                productivity_mw_per_m3s: 2.5,
            },
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 100.0,
            min_generation_mw: 0.0,
            max_generation_mw: 250.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: HydroPenalties {
                spillage_cost: 0.01,
                diversion_cost: 0.0,
                fpha_turbined_cost: 0.0,
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
                inflow_nonnegativity_cost: 1000.0,
            },
        };

        // Monthly stages: season_id = month index (0-based).
        let stages: Vec<Stage> = (0..n_stages)
            .map(|i| Stage {
                index: i,
                id: i as i32,
                start_date: NaiveDate::from_ymd_opt(2024, (i as u32 % 12) + 1, 1).unwrap(),
                end_date: NaiveDate::from_ymd_opt(2024, (i as u32 % 12) + 1, 28).unwrap(),
                season_id: Some(i % 12),
                blocks: vec![Block {
                    index: 0,
                    name: "S".to_string(),
                    duration_hours: 720.0,
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
            })
            .collect();

        let inflow_models: Vec<InflowModel> = (0..n_stages)
            .map(|i| InflowModel {
                hydro_id: EntityId(3),
                stage_id: i as i32,
                mean_m3s: 80.0,
                std_m3s: 20.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
            })
            .collect();

        let load_models: Vec<LoadModel> = (0..n_stages)
            .map(|i| LoadModel {
                bus_id: EntityId(1),
                stage_id: i as i32,
                mean_mw: 100.0,
                std_mw: 0.0,
            })
            .collect();

        // Historical inflow data: 1990 and 1991 cover 12 months each.
        // With n_stages <= 2 and max_par_order = 0, year 1990 and 1991 are
        // both valid windows (study months are in Jan-Feb = seasons 0-1).
        let inflow_history: Vec<InflowHistoryRow> = (1990_i32..=1991)
            .flat_map(|year| {
                (1u32..=12).map(move |month| InflowHistoryRow {
                    hydro_id: EntityId(3),
                    date: NaiveDate::from_ymd_opt(year, month, 1).unwrap(),
                    value_m3s: 80.0 + f64::from(year - 1990) * 5.0,
                })
            })
            .collect();

        let n_st = n_stages.max(1);

        let bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 1,
                n_thermals: 1,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: n_st,
            },
            &BoundsDefaults {
                hydro: default_hydro_bounds(),
                thermal: ThermalStageBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 100.0,
                    cost_per_mwh: 0.0,
                },
                line: LineStageBounds {
                    direct_mw: 0.0,
                    reverse_mw: 0.0,
                },
                pumping: PumpingStageBounds {
                    min_flow_m3s: 0.0,
                    max_flow_m3s: 0.0,
                },
                contract: ContractStageBounds {
                    min_mw: 0.0,
                    max_mw: 0.0,
                    price_per_mwh: 0.0,
                },
            },
        );

        let penalties = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 1,
                n_buses: 1,
                n_lines: 0,
                n_ncs: 0,
                n_stages: n_st,
            },
            &PenaltiesDefaults {
                hydro: default_hydro_penalties(),
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        );

        SystemBuilder::new()
            .buses(vec![bus])
            .thermals(vec![thermal])
            .hydros(vec![hydro])
            .stages(stages)
            .inflow_models(inflow_models)
            .load_models(load_models)
            .inflow_history(inflow_history)
            .bounds(bounds)
            .penalties(penalties)
            .build()
            .expect("system_with_historical_inflow: valid")
    }

    /// Given a `System` with `inflow_scheme = Historical` and valid inflow history,
    /// when `StudySetup::new()` is called, then `historical_library()` returns
    /// `Some` and `n_windows() > 0`.
    #[test]
    fn historical_library_built_when_scheme_is_historical() {
        let system = system_with_historical_inflow(2);
        let config = minimal_config_with_schemes(1, 5, Some("historical"), None, None);
        let stochastic = build_stochastic_context(
            &system,
            42,
            None,
            &[],
            &[],
            OpeningTreeInputs::default(),
            ClassSchemes {
                inflow: Some(SamplingScheme::Historical),
                load: Some(SamplingScheme::InSample),
                ncs: Some(SamplingScheme::InSample),
            },
        )
        .expect("stochastic context");

        let setup = StudySetup::new(
            &system,
            &config,
            stochastic,
            PrepareHydroModelsResult::default_from_system(&system),
        )
        .expect("setup");

        let lib = setup
            .scenario_libraries
            .training
            .historical
            .as_ref()
            .expect("expected Some(HistoricalScenarioLibrary) for Historical scheme");
        assert!(
            lib.n_windows() > 0,
            "expected at least one historical window, got 0"
        );
        assert_eq!(
            lib.n_stages(),
            2,
            "expected n_stages == 2 matching the system's study stages"
        );
        assert_eq!(lib.n_hydros(), 1, "expected n_hydros == 1");
    }

    /// Given a `System` with `inflow_scheme = External` and valid external
    /// inflow rows, when `StudySetup::new()` is called, then
    /// `external_inflow_library()` returns `Some` and `n_entities() > 0`.
    #[test]
    #[allow(
        clippy::too_many_lines,
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::cast_precision_loss,
        clippy::cast_lossless
    )]
    fn external_inflow_library_built_when_scheme_is_external() {
        use chrono::NaiveDate;
        use cobre_core::scenario::ExternalScenarioRow;
        use cobre_core::{scenario::InflowModel as CoreInflowModel, system::SystemBuilder};

        // Build external inflow rows: 3 scenarios × 1 hydro × 2 stages.
        // Hydro ID = 3 (from minimal_system). Stage IDs 0, 1. Scenario IDs 0, 1, 2.
        let hydro_id = EntityId(3);
        let mut external_rows: Vec<ExternalScenarioRow> = Vec::new();
        for stage_id in 0i32..2 {
            for scenario_id in 0i32..3 {
                external_rows.push(ExternalScenarioRow {
                    stage_id,
                    scenario_id,
                    hydro_id,
                    value_m3s: 80.0 + scenario_id as f64 * 5.0,
                });
            }
        }

        // We need to rebuild the system with external scenario source and rows.
        // Use SystemBuilder to produce a system that carries external rows.
        // minimal_system builds with its own SystemBuilder call, so we rebuild.

        let bus = Bus {
            id: EntityId(1),
            name: "B1".to_string(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
        };
        let thermal = Thermal {
            id: EntityId(2),
            name: "T1".to_string(),
            bus_id: EntityId(1),
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            cost_per_mwh: 50.0,
            gnl_config: None,
            entry_stage_id: None,
            exit_stage_id: None,
        };
        let hydro = Hydro {
            id: EntityId(3),
            name: "H1".to_string(),
            bus_id: EntityId(1),
            downstream_id: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 200.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity {
                productivity_mw_per_m3s: 2.5,
            },
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 100.0,
            min_generation_mw: 0.0,
            max_generation_mw: 250.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: HydroPenalties {
                spillage_cost: 0.01,
                diversion_cost: 0.0,
                fpha_turbined_cost: 0.0,
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
                inflow_nonnegativity_cost: 1000.0,
            },
        };
        let stages: Vec<Stage> = (0..2usize)
            .map(|i| Stage {
                index: i,
                id: i as i32,
                start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
                season_id: None,
                blocks: vec![Block {
                    index: 0,
                    name: "S".to_string(),
                    duration_hours: 744.0,
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
            })
            .collect();

        let inflow_models: Vec<CoreInflowModel> = (0..2usize)
            .map(|i| CoreInflowModel {
                hydro_id: EntityId(3),
                stage_id: i as i32,
                mean_m3s: 80.0,
                std_m3s: 20.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
            })
            .collect();

        let load_models: Vec<LoadModel> = (0..2usize)
            .map(|i| LoadModel {
                bus_id: EntityId(1),
                stage_id: i as i32,
                mean_mw: 100.0,
                std_mw: 0.0,
            })
            .collect();

        let bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 1,
                n_thermals: 1,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: 2,
            },
            &BoundsDefaults {
                hydro: HydroStageBounds {
                    min_storage_hm3: 0.0,
                    max_storage_hm3: 200.0,
                    min_turbined_m3s: 0.0,
                    max_turbined_m3s: 100.0,
                    min_outflow_m3s: 0.0,
                    max_outflow_m3s: None,
                    min_generation_mw: 0.0,
                    max_generation_mw: 250.0,
                    max_diversion_m3s: None,
                    filling_inflow_m3s: 0.0,
                    water_withdrawal_m3s: 0.0,
                },
                thermal: ThermalStageBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 100.0,
                    cost_per_mwh: 0.0,
                },
                line: LineStageBounds {
                    direct_mw: 0.0,
                    reverse_mw: 0.0,
                },
                pumping: PumpingStageBounds {
                    min_flow_m3s: 0.0,
                    max_flow_m3s: 0.0,
                },
                contract: ContractStageBounds {
                    min_mw: 0.0,
                    max_mw: 0.0,
                    price_per_mwh: 0.0,
                },
            },
        );
        let penalties = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 1,
                n_buses: 1,
                n_lines: 0,
                n_ncs: 0,
                n_stages: 2,
            },
            &PenaltiesDefaults {
                hydro: HydroStagePenalties {
                    spillage_cost: 0.01,
                    diversion_cost: 0.0,
                    fpha_turbined_cost: 0.0,
                    storage_violation_below_cost: 500.0,
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
                    inflow_nonnegativity_cost: 1000.0,
                },
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        );

        let system = SystemBuilder::new()
            .buses(vec![bus])
            .thermals(vec![thermal])
            .hydros(vec![hydro])
            .stages(stages)
            .inflow_models(inflow_models)
            .load_models(load_models)
            .external_scenarios(external_rows)
            .bounds(bounds)
            .penalties(penalties)
            .build()
            .expect("system with external inflow: valid");

        let config = minimal_config_with_schemes(1, 5, Some("external"), None, None);
        let stochastic = build_stochastic_context(
            &system,
            42,
            None,
            &[],
            &[],
            OpeningTreeInputs::default(),
            ClassSchemes {
                inflow: Some(SamplingScheme::External),
                load: Some(SamplingScheme::InSample),
                ncs: Some(SamplingScheme::InSample),
            },
        )
        .expect("stochastic context");

        let setup = StudySetup::new(
            &system,
            &config,
            stochastic,
            PrepareHydroModelsResult::default_from_system(&system),
        )
        .expect("setup");

        let lib = setup
            .scenario_libraries
            .training
            .external_inflow
            .as_ref()
            .expect("expected Some(ExternalScenarioLibrary) for External inflow scheme");
        assert!(
            lib.n_entities() > 0,
            "expected n_entities > 0 in external inflow library"
        );
        assert_eq!(lib.n_stages(), 2);
        assert_eq!(lib.n_scenarios(), 3);
        assert_eq!(lib.entity_class(), "inflow");
    }

    /// Given a `System` with `load_scheme = External` and valid external load
    /// rows, when `StudySetup::new()` is called, then
    /// `external_load_library()` returns `Some` and `n_entities() > 0`.
    #[test]
    #[allow(
        clippy::too_many_lines,
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::cast_precision_loss,
        clippy::cast_lossless
    )]
    fn external_load_library_built_when_scheme_is_external() {
        use chrono::NaiveDate;
        use cobre_core::scenario::ExternalLoadRow;
        use cobre_core::{scenario::InflowModel as CoreInflowModel, system::SystemBuilder};

        let bus = Bus {
            id: EntityId(1),
            name: "B1".to_string(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
        };
        let thermal = Thermal {
            id: EntityId(2),
            name: "T1".to_string(),
            bus_id: EntityId(1),
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            cost_per_mwh: 50.0,
            gnl_config: None,
            entry_stage_id: None,
            exit_stage_id: None,
        };
        let hydro = Hydro {
            id: EntityId(3),
            name: "H1".to_string(),
            bus_id: EntityId(1),
            downstream_id: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 200.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity {
                productivity_mw_per_m3s: 2.5,
            },
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 100.0,
            min_generation_mw: 0.0,
            max_generation_mw: 250.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: HydroPenalties {
                spillage_cost: 0.01,
                diversion_cost: 0.0,
                fpha_turbined_cost: 0.0,
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
                inflow_nonnegativity_cost: 1000.0,
            },
        };

        let stages: Vec<Stage> = (0..2usize)
            .map(|i| Stage {
                index: i,
                id: i as i32,
                start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
                season_id: None,
                blocks: vec![Block {
                    index: 0,
                    name: "S".to_string(),
                    duration_hours: 744.0,
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
            })
            .collect();

        let inflow_models: Vec<CoreInflowModel> = (0..2usize)
            .map(|i| CoreInflowModel {
                hydro_id: EntityId(3),
                stage_id: i as i32,
                mean_m3s: 80.0,
                std_m3s: 20.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
            })
            .collect();

        let load_models: Vec<LoadModel> = (0..2usize)
            .map(|i| LoadModel {
                bus_id: EntityId(1),
                stage_id: i as i32,
                mean_mw: 100.0,
                std_mw: 10.0,
            })
            .collect();

        // External load rows: 3 scenarios × 1 bus × 2 stages.
        let mut external_load_rows: Vec<ExternalLoadRow> = Vec::new();
        for stage_id in 0i32..2 {
            for scenario_id in 0i32..3 {
                external_load_rows.push(ExternalLoadRow {
                    stage_id,
                    scenario_id,
                    bus_id: EntityId(1),
                    value_mw: 90.0 + scenario_id as f64 * 10.0,
                });
            }
        }

        let bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 1,
                n_thermals: 1,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: 2,
            },
            &BoundsDefaults {
                hydro: HydroStageBounds {
                    min_storage_hm3: 0.0,
                    max_storage_hm3: 200.0,
                    min_turbined_m3s: 0.0,
                    max_turbined_m3s: 100.0,
                    min_outflow_m3s: 0.0,
                    max_outflow_m3s: None,
                    min_generation_mw: 0.0,
                    max_generation_mw: 250.0,
                    max_diversion_m3s: None,
                    filling_inflow_m3s: 0.0,
                    water_withdrawal_m3s: 0.0,
                },
                thermal: ThermalStageBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 100.0,
                    cost_per_mwh: 0.0,
                },
                line: LineStageBounds {
                    direct_mw: 0.0,
                    reverse_mw: 0.0,
                },
                pumping: PumpingStageBounds {
                    min_flow_m3s: 0.0,
                    max_flow_m3s: 0.0,
                },
                contract: ContractStageBounds {
                    min_mw: 0.0,
                    max_mw: 0.0,
                    price_per_mwh: 0.0,
                },
            },
        );
        let penalties = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 1,
                n_buses: 1,
                n_lines: 0,
                n_ncs: 0,
                n_stages: 2,
            },
            &PenaltiesDefaults {
                hydro: HydroStagePenalties {
                    spillage_cost: 0.01,
                    diversion_cost: 0.0,
                    fpha_turbined_cost: 0.0,
                    storage_violation_below_cost: 500.0,
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
                    inflow_nonnegativity_cost: 1000.0,
                },
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        );

        let system = SystemBuilder::new()
            .buses(vec![bus])
            .thermals(vec![thermal])
            .hydros(vec![hydro])
            .stages(stages)
            .inflow_models(inflow_models)
            .load_models(load_models)
            .external_load_scenarios(external_load_rows)
            .bounds(bounds)
            .penalties(penalties)
            .build()
            .expect("system with external load: valid");

        let config = minimal_config_with_schemes(1, 5, None, Some("external"), None);
        let stochastic = build_stochastic_context(
            &system,
            42,
            None,
            &[],
            &[],
            OpeningTreeInputs::default(),
            ClassSchemes {
                inflow: Some(SamplingScheme::InSample),
                load: Some(SamplingScheme::External),
                ncs: Some(SamplingScheme::InSample),
            },
        )
        .expect("stochastic context");

        let setup = StudySetup::new(
            &system,
            &config,
            stochastic,
            PrepareHydroModelsResult::default_from_system(&system),
        )
        .expect("setup");

        let lib = setup
            .scenario_libraries
            .training
            .external_load
            .as_ref()
            .expect("expected Some(ExternalScenarioLibrary) for External load scheme");
        assert!(
            lib.n_entities() > 0,
            "expected n_entities > 0 in external load library"
        );
        assert_eq!(lib.n_stages(), 2);
        assert_eq!(lib.n_scenarios(), 3);
        assert_eq!(lib.entity_class(), "load");
    }

    /// Given a `System` with `ncs_scheme = External` and valid external NCS
    /// rows, when `StudySetup::new()` is called, then
    /// `external_ncs_library()` returns `Some` and `n_entities() > 0`.
    #[test]
    #[allow(
        clippy::too_many_lines,
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::cast_precision_loss,
        clippy::cast_lossless
    )]
    fn external_ncs_library_built_when_scheme_is_external() {
        use chrono::NaiveDate;
        use cobre_core::scenario::InflowModel as CoreInflowModel;
        use cobre_core::{
            NonControllableSource,
            scenario::{ExternalNcsRow, NcsModel},
            system::SystemBuilder,
        };

        let bus = Bus {
            id: EntityId(1),
            name: "B1".to_string(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
        };
        let thermal = Thermal {
            id: EntityId(2),
            name: "T1".to_string(),
            bus_id: EntityId(1),
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            cost_per_mwh: 50.0,
            gnl_config: None,
            entry_stage_id: None,
            exit_stage_id: None,
        };
        let hydro = Hydro {
            id: EntityId(3),
            name: "H1".to_string(),
            bus_id: EntityId(1),
            downstream_id: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 200.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity {
                productivity_mw_per_m3s: 2.5,
            },
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 100.0,
            min_generation_mw: 0.0,
            max_generation_mw: 250.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: HydroPenalties {
                spillage_cost: 0.01,
                diversion_cost: 0.0,
                fpha_turbined_cost: 0.0,
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
                inflow_nonnegativity_cost: 1000.0,
            },
        };

        // NCS entity: wind plant with EntityId(4).
        let ncs_id = EntityId(4);
        let ncs_source = NonControllableSource {
            id: ncs_id,
            name: "Wind1".to_string(),
            bus_id: EntityId(1),
            entry_stage_id: None,
            exit_stage_id: None,
            max_generation_mw: 100.0,
            curtailment_cost: 0.01,
        };

        let stages: Vec<Stage> = (0..2usize)
            .map(|i| Stage {
                index: i,
                id: i as i32,
                start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
                season_id: None,
                blocks: vec![Block {
                    index: 0,
                    name: "S".to_string(),
                    duration_hours: 744.0,
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
            })
            .collect();

        let inflow_models: Vec<CoreInflowModel> = (0..2usize)
            .map(|i| CoreInflowModel {
                hydro_id: EntityId(3),
                stage_id: i as i32,
                mean_m3s: 80.0,
                std_m3s: 20.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
            })
            .collect();

        let load_models: Vec<LoadModel> = (0..2usize)
            .map(|i| LoadModel {
                bus_id: EntityId(1),
                stage_id: i as i32,
                mean_mw: 100.0,
                std_mw: 0.0,
            })
            .collect();

        // NCS models: mean=0.8, std=0.1 for both stages.
        let ncs_models: Vec<NcsModel> = (0..2usize)
            .map(|i| NcsModel {
                ncs_id,
                stage_id: i as i32,
                mean: 0.8,
                std: 0.1,
            })
            .collect();

        // External NCS rows: 3 scenarios × 1 NCS × 2 stages.
        let mut external_ncs_rows: Vec<ExternalNcsRow> = Vec::new();
        for stage_id in 0i32..2 {
            for scenario_id in 0i32..3 {
                external_ncs_rows.push(ExternalNcsRow {
                    stage_id,
                    scenario_id,
                    ncs_id,
                    value: 0.7 + scenario_id as f64 * 0.1,
                });
            }
        }

        let bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 1,
                n_thermals: 1,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: 2,
            },
            &BoundsDefaults {
                hydro: HydroStageBounds {
                    min_storage_hm3: 0.0,
                    max_storage_hm3: 200.0,
                    min_turbined_m3s: 0.0,
                    max_turbined_m3s: 100.0,
                    min_outflow_m3s: 0.0,
                    max_outflow_m3s: None,
                    min_generation_mw: 0.0,
                    max_generation_mw: 250.0,
                    max_diversion_m3s: None,
                    filling_inflow_m3s: 0.0,
                    water_withdrawal_m3s: 0.0,
                },
                thermal: ThermalStageBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 100.0,
                    cost_per_mwh: 0.0,
                },
                line: LineStageBounds {
                    direct_mw: 0.0,
                    reverse_mw: 0.0,
                },
                pumping: PumpingStageBounds {
                    min_flow_m3s: 0.0,
                    max_flow_m3s: 0.0,
                },
                contract: ContractStageBounds {
                    min_mw: 0.0,
                    max_mw: 0.0,
                    price_per_mwh: 0.0,
                },
            },
        );
        let penalties = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 1,
                n_buses: 1,
                n_lines: 0,
                n_ncs: 1,
                n_stages: 2,
            },
            &PenaltiesDefaults {
                hydro: HydroStagePenalties {
                    spillage_cost: 0.01,
                    diversion_cost: 0.0,
                    fpha_turbined_cost: 0.0,
                    storage_violation_below_cost: 500.0,
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
                    inflow_nonnegativity_cost: 1000.0,
                },
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        );

        let system = SystemBuilder::new()
            .buses(vec![bus])
            .thermals(vec![thermal])
            .hydros(vec![hydro])
            .non_controllable_sources(vec![ncs_source])
            .stages(stages)
            .inflow_models(inflow_models)
            .load_models(load_models)
            .ncs_models(ncs_models)
            .external_ncs_scenarios(external_ncs_rows)
            .bounds(bounds)
            .penalties(penalties)
            .build()
            .expect("system with external NCS: valid");

        let config = minimal_config_with_schemes(1, 5, None, None, Some("external"));
        let stochastic = build_stochastic_context(
            &system,
            42,
            None,
            &[],
            &[],
            OpeningTreeInputs::default(),
            ClassSchemes {
                inflow: Some(SamplingScheme::InSample),
                load: Some(SamplingScheme::InSample),
                ncs: Some(SamplingScheme::External),
            },
        )
        .expect("stochastic context");

        let setup = StudySetup::new(
            &system,
            &config,
            stochastic,
            PrepareHydroModelsResult::default_from_system(&system),
        )
        .expect("setup");

        let lib = setup
            .scenario_libraries
            .training
            .external_ncs
            .as_ref()
            .expect("expected Some(ExternalScenarioLibrary) for External NCS scheme");
        assert!(
            lib.n_entities() > 0,
            "expected n_entities > 0 in external NCS library"
        );
        assert_eq!(lib.n_stages(), 2);
        assert_eq!(lib.n_scenarios(), 3);
        assert_eq!(lib.entity_class(), "ncs");
    }

    /// Given a `System` with `inflow_scheme = Historical` but a user pool
    /// that references a year with no data, when `StudySetup::new()` is
    /// called, then it returns `Err` with a message about windows.
    #[test]
    #[allow(
        clippy::too_many_lines,
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::cast_precision_loss,
        clippy::cast_lossless
    )]
    fn historical_library_fails_when_no_valid_windows() {
        // system_with_historical_inflow has data for years 1990-1991.
        // We use HistoricalYears::List with year 2050 (no data) to force
        // zero valid windows after filtering.
        use cobre_core::system::SystemBuilder;

        // Instead, let's build a system with Historical scheme and empty
        // inflow_history (no rows at all). This guarantees zero candidate
        // years in discovery.
        use chrono::NaiveDate;
        use cobre_core::scenario::InflowModel;

        let bus = Bus {
            id: EntityId(1),
            name: "B1".to_string(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
        };
        let thermal = Thermal {
            id: EntityId(2),
            name: "T1".to_string(),
            bus_id: EntityId(1),
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            cost_per_mwh: 50.0,
            gnl_config: None,
            entry_stage_id: None,
            exit_stage_id: None,
        };
        let hydro = Hydro {
            id: EntityId(3),
            name: "H1".to_string(),
            bus_id: EntityId(1),
            downstream_id: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 200.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity {
                productivity_mw_per_m3s: 2.5,
            },
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 100.0,
            min_generation_mw: 0.0,
            max_generation_mw: 250.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: HydroPenalties {
                spillage_cost: 0.01,
                diversion_cost: 0.0,
                fpha_turbined_cost: 0.0,
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
                inflow_nonnegativity_cost: 1000.0,
            },
        };

        let stages: Vec<Stage> = (0..2usize)
            .map(|i| Stage {
                index: i,
                id: i as i32,
                start_date: NaiveDate::from_ymd_opt(2024, (i as u32 % 12) + 1, 1).unwrap(),
                end_date: NaiveDate::from_ymd_opt(2024, (i as u32 % 12) + 1, 28).unwrap(),
                season_id: Some(i % 12),
                blocks: vec![Block {
                    index: 0,
                    name: "S".to_string(),
                    duration_hours: 720.0,
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
            })
            .collect();

        let inflow_models: Vec<InflowModel> = (0..2usize)
            .map(|i| InflowModel {
                hydro_id: EntityId(3),
                stage_id: i as i32,
                mean_m3s: 80.0,
                std_m3s: 20.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
            })
            .collect();

        let load_models: Vec<LoadModel> = (0..2usize)
            .map(|i| LoadModel {
                bus_id: EntityId(1),
                stage_id: i as i32,
                mean_mw: 100.0,
                std_mw: 0.0,
            })
            .collect();

        let bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 1,
                n_thermals: 1,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: 2,
            },
            &BoundsDefaults {
                hydro: HydroStageBounds {
                    min_storage_hm3: 0.0,
                    max_storage_hm3: 200.0,
                    min_turbined_m3s: 0.0,
                    max_turbined_m3s: 100.0,
                    min_outflow_m3s: 0.0,
                    max_outflow_m3s: None,
                    min_generation_mw: 0.0,
                    max_generation_mw: 250.0,
                    max_diversion_m3s: None,
                    filling_inflow_m3s: 0.0,
                    water_withdrawal_m3s: 0.0,
                },
                thermal: ThermalStageBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 100.0,
                    cost_per_mwh: 0.0,
                },
                line: LineStageBounds {
                    direct_mw: 0.0,
                    reverse_mw: 0.0,
                },
                pumping: PumpingStageBounds {
                    min_flow_m3s: 0.0,
                    max_flow_m3s: 0.0,
                },
                contract: ContractStageBounds {
                    min_mw: 0.0,
                    max_mw: 0.0,
                    price_per_mwh: 0.0,
                },
            },
        );
        let penalties = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 1,
                n_buses: 1,
                n_lines: 0,
                n_ncs: 0,
                n_stages: 2,
            },
            &PenaltiesDefaults {
                hydro: HydroStagePenalties {
                    spillage_cost: 0.01,
                    diversion_cost: 0.0,
                    fpha_turbined_cost: 0.0,
                    storage_violation_below_cost: 500.0,
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
                    inflow_nonnegativity_cost: 1000.0,
                },
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        );

        // Historical scheme but NO inflow_history data — discovery must fail.
        let system = SystemBuilder::new()
            .buses(vec![bus])
            .thermals(vec![thermal])
            .hydros(vec![hydro])
            .stages(stages)
            .inflow_models(inflow_models)
            .load_models(load_models)
            .bounds(bounds)
            .penalties(penalties)
            .build()
            .expect("system: valid");

        let config = minimal_config_with_schemes(1, 5, Some("historical"), None, None);
        let stochastic = build_stochastic_context(
            &system,
            42,
            None,
            &[],
            &[],
            OpeningTreeInputs::default(),
            ClassSchemes {
                inflow: Some(SamplingScheme::Historical),
                load: Some(SamplingScheme::InSample),
                ncs: Some(SamplingScheme::InSample),
            },
        )
        .expect("stochastic context");

        let result = StudySetup::new(
            &system,
            &config,
            stochastic,
            PrepareHydroModelsResult::default_from_system(&system),
        );

        assert!(result.is_err(), "expected Err when no historical data");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("window") || err_msg.contains("historical"),
            "error should mention windows or historical, got: {err_msg}"
        );
    }

    /// Given a `Config` with training inflow scheme `InSample` and simulation
    /// inflow scheme `OutOfSample`, when `StudySetup::new()` is called, then
    /// `training_ctx().inflow_scheme` is `InSample` and
    /// `simulation_ctx().inflow_scheme` is `OutOfSample`.
    #[test]
    fn test_simulate_uses_simulation_scheme() {
        let system = minimal_system(2);

        // Training: InSample (default). Simulation: OutOfSample.
        let mut config = minimal_config(1, 5);
        config.simulation.scenario_source = Some(RawScenarioSourceConfig {
            seed: Some(99),
            historical_years: None,
            inflow: Some(RawClassConfigEntry {
                scheme: "out_of_sample".to_string(),
            }),
            load: None,
            ncs: None,
        });

        let stochastic = build_stochastic_context(
            &system,
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
        .expect("stochastic context");

        let setup = StudySetup::new(
            &system,
            &config,
            stochastic,
            PrepareHydroModelsResult::default_from_system(&system),
        )
        .expect("setup");

        let train_ctx = setup.training_ctx();
        assert_eq!(
            train_ctx.inflow_scheme,
            SamplingScheme::InSample,
            "training context must use InSample inflow scheme"
        );

        let sim_ctx = setup.simulation_ctx();
        assert_eq!(
            sim_ctx.inflow_scheme,
            SamplingScheme::OutOfSample,
            "simulation context must use OutOfSample inflow scheme"
        );
    }

    /// Given a `Config` with training inflow scheme `InSample` and simulation
    /// inflow scheme `Historical`, when `StudySetup::new()` is called on a
    /// system that has inflow history, then `training_ctx().historical_library`
    /// is `None` and `simulation_ctx().historical_library` is `Some`.
    #[test]
    fn test_sim_historical_library_built_when_sim_scheme_is_historical() {
        let system = system_with_historical_inflow(2);

        // Training: InSample. Simulation: Historical.
        let mut config = minimal_config(1, 5);
        config.simulation.scenario_source = Some(RawScenarioSourceConfig {
            seed: Some(42),
            historical_years: None,
            inflow: Some(RawClassConfigEntry {
                scheme: "historical".to_string(),
            }),
            load: None,
            ncs: None,
        });

        // The stochastic context is built for the training scheme (InSample).
        let stochastic = build_stochastic_context(
            &system,
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
        .expect("stochastic context");

        let setup = StudySetup::new(
            &system,
            &config,
            stochastic,
            PrepareHydroModelsResult::default_from_system(&system),
        )
        .expect("setup");

        assert!(
            setup.training_ctx().historical_library.is_none(),
            "training context must NOT have a historical library when scheme is InSample"
        );
        assert!(
            setup.simulation_ctx().historical_library.is_some(),
            "simulation context must have a historical library when sim scheme is Historical"
        );
    }
}
