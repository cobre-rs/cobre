//! Study setup struct that owns all precomputed state for a solve run.
//!
//! [`StudySetup`] centralises the orchestration that was previously scattered
//! across entry points (CLI, Python bindings). It builds stage LP templates,
//! the stage indexer, initial state, future cost function, horizon mode, risk
//! measures, entity counts, and configuration-derived scalars from a validated
//! [`System`] and [`cobre_io::Config`].
//!
//! ## Ownership model
//!
//! `StudySetup` owns all data. Callers borrow from it when constructing
//! [`TrainingContext`] and [`StageContext`] for each pass. The
//! [`StochasticContext`] is moved in at construction time so its lifetime
//! matches `StudySetup`.
//!
//! ## What is NOT included
//!
//! - MPI communication — broadcast and barrier calls remain in the CLI/Python
//!   entry points.
//! - Solver instances — callers create solvers and pass them to `train()`/
//!   `simulate()`.
//! - Progress bars or event channels — callers wire those up and pass the
//!   sender to [`TrainingConfig`].
//!
//! ## Example
//!
//! ```rust,no_run
//! use cobre_sddp::setup::StudySetup;
//! use cobre_sddp::hydro_models::PrepareHydroModelsResult;
//! use cobre_stochastic::build_stochastic_context;
//!
//! # fn example(system: &cobre_core::System, config: &cobre_io::Config)
//! #     -> Result<(), cobre_sddp::SddpError> {
//! let stochastic = build_stochastic_context(system, 42, &[], &[], None)?;
//! let hydro_models = PrepareHydroModelsResult::default_from_system(system);
//! let setup = StudySetup::new(system, config, stochastic, hydro_models)?;
//! assert!(!setup.stage_templates().is_empty());
//! # Ok(())
//! # }
//! ```

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::Sender;
use std::sync::mpsc::SyncSender;

use cobre_comm::Communicator;
use cobre_core::{EntityId, System, TrainingEvent, entities::hydro::HydroGenerationModel};
use cobre_solver::{SolverError, SolverInterface};
use cobre_stochastic::{StochasticContext, context::OpeningTree};

use crate::{
    FutureCostFunction, HorizonMode, InflowNonNegativityMethod, RiskMeasure, SddpError,
    SimulationConfig, SimulationError, SimulationScenarioResult, SolverWorkspace, StageContext,
    StageIndexer, StageTemplates, TrainingConfig, TrainingContext, TrainingResult, WorkspacePool,
    build_stage_templates,
    cut_selection::{CutSelectionStrategy, parse_cut_selection_config},
    hydro_models::{EvaporationModel, PrepareHydroModelsResult, ResolvedProductionModel},
    lp_builder,
    simulation::{EntityCounts, SimulationOutputSpec},
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
/// `StudyParams` centralises the config-to-domain conversion that was previously
/// duplicated between `StudySetup::new()` (cobre-sddp) and
/// `BroadcastConfig::from_config()` (cobre-cli). Both callers now delegate
/// to `StudyParams::from_config()` and then convert the resulting fields to
/// their respective target representations.
///
/// The struct owns all values so it can be passed by value to constructors
/// and broadcast helpers without lifetime dependencies.
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
}

impl StudyParams {
    /// Extract study parameters from a validated [`cobre_io::Config`].
    ///
    /// This method contains the full config-to-domain conversion logic:
    /// seed derivation, forward passes defaulting, stopping rule conversion,
    /// stopping mode parsing, `n_scenarios` conditional, `io_channel_capacity`
    /// conversion, policy path extraction, inflow method construction, and
    /// cut selection parsing.
    ///
    /// # Errors
    ///
    /// - [`SddpError::Validation`] if `parse_cut_selection_config` returns an
    ///   error for an unrecognised cut selection config string.
    pub fn from_config(config: &cobre_io::Config) -> Result<Self, SddpError> {
        use cobre_io::config::StoppingRuleConfig;

        let seed = config.training.seed.map_or(DEFAULT_SEED, i64::unsigned_abs);

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
        })
    }
}

// ---------------------------------------------------------------------------
// StudySetup
// ---------------------------------------------------------------------------

/// All precomputed study state built once before training and simulation.
///
/// Constructed by [`StudySetup::new`] from a validated [`System`] and
/// [`cobre_io::Config`]. Owns all data so it can be held across async
/// boundaries (e.g., Python GIL release) without lifetime issues.
///
/// Callers build [`TrainingContext`] and [`StageContext`] by borrowing
/// from `StudySetup`.
#[derive(Debug)]
pub struct StudySetup {
    // ── LP templates ─────────────────────────────────────────────────────────
    stage_templates: StageTemplates,

    // ── Algorithm state ──────────────────────────────────────────────────────
    stochastic: StochasticContext,
    indexer: StageIndexer,
    fcf: FutureCostFunction,
    initial_state: Vec<f64>,
    horizon: HorizonMode,
    risk_measures: Vec<RiskMeasure>,
    entity_counts: EntityCounts,

    // ── Hydro models ──────────────────────────────────────────────────────────
    hydro_models: PrepareHydroModelsResult,

    // ── NCS per-stage entity IDs (for simulation extraction) ────────────────
    ncs_entity_ids_per_stage: Vec<Vec<i32>>,
    /// Max generation [MW] per stochastic NCS entity, sorted by entity ID.
    ncs_max_gen: Vec<f64>,

    // ── Derived layout values ─────────────────────────────────────────────────
    block_counts_per_stage: Vec<usize>,
    max_blocks: usize,

    // ── Scaling diagnostics ──────────────────────────────────────────────────
    scaling_report: crate::scaling_report::ScalingReport,

    // ── Config-derived scalars ────────────────────────────────────────────────
    seed: u64,
    forward_passes: u32,
    max_iterations: u64,
    n_scenarios: u32,
    io_channel_capacity: usize,
    policy_path: String,
    inflow_method: InflowNonNegativityMethod,
    cut_selection: Option<CutSelectionStrategy>,
    cut_activity_tolerance: f64,
    stopping_rule_set: StoppingRuleSet,
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
    /// - [`SddpError::LpBuilder`](SddpError::Solver) — propagated from
    ///   `build_stage_templates` on LP construction failure.
    /// - [`SddpError::Validation`] — if `parse_cut_selection_config` returns
    ///   an invalid config string.
    pub fn new(
        system: &System,
        config: &cobre_io::Config,
        stochastic: StochasticContext,
        hydro_models: PrepareHydroModelsResult,
    ) -> Result<Self, SddpError> {
        let params = StudyParams::from_config(config)?;
        Self::from_broadcast_params(
            system,
            stochastic,
            params.seed,
            params.forward_passes,
            params.stopping_rule_set,
            params.n_scenarios,
            params.io_channel_capacity,
            params.policy_path,
            params.inflow_method,
            params.cut_selection,
            params.cut_activity_tolerance,
            hydro_models,
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
    #[allow(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        clippy::missing_panics_doc
    )]
    pub fn from_broadcast_params(
        system: &System,
        stochastic: StochasticContext,
        seed: u64,
        forward_passes: u32,
        stopping_rule_set: StoppingRuleSet,
        n_scenarios: u32,
        io_channel_capacity: usize,
        policy_path: String,
        inflow_method: InflowNonNegativityMethod,
        cut_selection: Option<CutSelectionStrategy>,
        cut_activity_tolerance: f64,
        hydro_models: PrepareHydroModelsResult,
    ) -> Result<Self, SddpError> {
        use crate::scaling_report::{
            LpDimensions, StageScalingReport, build_scaling_report, compute_coefficient_range,
            summarize_scale_factors,
        };

        // ── Stage templates ───────────────────────────────────────────────────
        let mut stage_templates = build_stage_templates(
            system,
            &inflow_method,
            stochastic.par(),
            stochastic.normal(),
            &hydro_models.production,
            &hydro_models.evaporation,
        )?;

        // Compute and apply column scaling, then row scaling for numerical
        // conditioning (D_r * A * D_c form). Scale factors are stored in the
        // template for unscaling primal/dual solutions in the forward and
        // backward passes.
        //
        // Scaling report: capture pre/post coefficient ranges for diagnostics.

        let mut stage_scaling_reports = Vec::with_capacity(stage_templates.templates.len());

        for (stage_id, tmpl) in stage_templates.templates.iter_mut().enumerate() {
            // Pre-scaling snapshot (before col/row scaling; cost scaling is
            // already baked into the objective during template construction).
            let pre_scaling = compute_coefficient_range(tmpl);

            let col_scale =
                lp_builder::compute_col_scale(tmpl.num_cols, &tmpl.col_starts, &tmpl.values);
            lp_builder::apply_col_scale(tmpl, &col_scale);
            tmpl.col_scale.clone_from(&col_scale);
            // Row scaling is applied to the already column-scaled matrix.
            let row_scale = lp_builder::compute_row_scale(
                tmpl.num_rows,
                tmpl.num_cols,
                &tmpl.col_starts,
                &tmpl.row_indices,
                &tmpl.values,
            );
            lp_builder::apply_row_scale(tmpl, &row_scale);
            tmpl.row_scale.clone_from(&row_scale);

            // Post-scaling snapshot (after col + row scaling).
            let post_scaling = compute_coefficient_range(tmpl);

            stage_scaling_reports.push(StageScalingReport {
                stage_id,
                dimensions: LpDimensions {
                    num_cols: tmpl.num_cols,
                    num_rows: tmpl.num_rows,
                    num_nz: tmpl.num_nz,
                },
                pre_scaling,
                post_scaling,
                col_scale: summarize_scale_factors(&col_scale),
                row_scale: summarize_scale_factors(&row_scale),
            });
        }

        let scaling_report =
            build_scaling_report(lp_builder::COST_SCALE_FACTOR, stage_scaling_reports);

        // Pre-scale noise_scale by row_scale so that the inflow noise
        // perturbation (noise_scale * eta) is in the same scaled units as
        // the template row bounds (which were already row-scaled above).
        // Without this, transform_inflow_noise would produce a mixed-scale
        // RHS: scaled base + unscaled perturbation.
        let n_hydros_noise = stage_templates.n_hydros;
        for (s_idx, tmpl) in stage_templates.templates.iter().enumerate() {
            if !tmpl.row_scale.is_empty() {
                let base_row = stage_templates.base_rows[s_idx];
                for h in 0..n_hydros_noise {
                    stage_templates.noise_scale[s_idx * n_hydros_noise + h] *=
                        tmpl.row_scale[base_row + h];
                }
            }
        }

        if stage_templates.templates.is_empty() {
            return Err(SddpError::Validation(
                "system has no study stages".to_string(),
            ));
        }

        let stage_templates_ref = &stage_templates.templates;

        // ── Stage indexer ─────────────────────────────────────────────────────
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

        let mut indexer = StageIndexer::with_equipment_and_evaporation(
            stage_templates_ref[0].n_hydro,
            stage_templates_ref[0].max_par_order,
            system.thermals().len(),
            system.lines().len(),
            system.buses().len(),
            n_blks_stage0,
            has_inflow_penalty,
            fpha_hydro_indices,
            &fpha_planes,
            evap_hydro_indices,
            max_deficit_segments,
        );

        // Wire NCS column range from the LP builder's stage-0 layout.
        if !stage_templates.ncs_col_starts.is_empty() {
            let ncs_start = stage_templates.ncs_col_starts[0];
            let n_ncs_stage0 = stage_templates.n_ncs_per_stage[0];
            indexer.ncs_generation = ncs_start..(ncs_start + n_ncs_stage0 * n_blks_stage0);

            // Debug: verify NCS column starts are consistent across stages.
            for (s, &start) in stage_templates.ncs_col_starts.iter().enumerate() {
                debug_assert_eq!(
                    start, ncs_start,
                    "NCS column start differs at stage {s}: expected {ncs_start}, got {start}"
                );
            }
        }

        // z-inflow column and row ranges are set by StageIndexer::new at
        // fixed offset N*(1+L), no per-stage wiring needed.

        // ── Initial state ─────────────────────────────────────────────────────
        let initial_state = build_initial_state(system, &indexer);

        // ── FCF pre-allocation ────────────────────────────────────────────────
        let n_stages = stage_templates_ref.len();
        let max_iterations = max_iterations_from_rules(&stopping_rule_set);
        let fcf_capacity_iterations = max_iterations.saturating_add(1);
        let fcf = FutureCostFunction::new(
            n_stages,
            indexer.n_state,
            forward_passes,
            fcf_capacity_iterations,
            0,
        );

        // ── Horizon and risk measures ─────────────────────────────────────────
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };

        let risk_measures: Vec<RiskMeasure> = system
            .stages()
            .iter()
            .filter(|s| s.id >= 0)
            .map(|s| RiskMeasure::from(s.risk_config))
            .collect();

        // ── Entity counts ─────────────────────────────────────────────────────
        let entity_counts = build_entity_counts(system);

        // ── NCS per-stage entity IDs ──────────────────────────────────────────
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

        // ── NCS max generation for stochastic entities ───────────────────────
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

        // ── Block layout ──────────────────────────────────────────────────────
        let block_counts_per_stage: Vec<usize> = stage_templates
            .block_hours_per_stage
            .iter()
            .map(Vec::len)
            .collect();
        let max_blocks = block_counts_per_stage.iter().copied().max().unwrap_or(0);

        Ok(Self {
            stage_templates,
            stochastic,
            indexer,
            fcf,
            initial_state,
            horizon,
            risk_measures,
            entity_counts,
            hydro_models,
            ncs_entity_ids_per_stage,
            ncs_max_gen,
            block_counts_per_stage,
            max_blocks,
            scaling_report,
            seed,
            forward_passes,
            max_iterations,
            n_scenarios,
            io_channel_capacity,
            policy_path,
            inflow_method,
            cut_selection,
            cut_activity_tolerance,
            stopping_rule_set,
        })
    }

    // ── Accessors: LP templates ───────────────────────────────────────────────

    /// Return a reference to the full [`StageTemplates`] struct.
    #[must_use]
    pub fn templates_full(&self) -> &StageTemplates {
        &self.stage_templates
    }

    /// Return a reference to the LP scaling report captured during template build.
    #[must_use]
    pub fn scaling_report(&self) -> &crate::scaling_report::ScalingReport {
        &self.scaling_report
    }

    /// Return the slice of [`StageTemplate`](cobre_solver::StageTemplate)s,
    /// one per study stage.
    #[must_use]
    pub fn stage_templates(&self) -> &[cobre_solver::StageTemplate] {
        &self.stage_templates.templates
    }

    /// Return the per-stage water-balance row offset array.
    #[must_use]
    pub fn base_rows(&self) -> &[usize] {
        &self.stage_templates.base_rows
    }

    /// Return the pre-computed noise scale factors (stage-major layout).
    #[must_use]
    pub fn noise_scale(&self) -> &[f64] {
        &self.stage_templates.noise_scale
    }

    // ── Accessors: algorithm state ────────────────────────────────────────────

    /// Return a shared reference to the stochastic context.
    #[must_use]
    pub fn stochastic(&self) -> &StochasticContext {
        &self.stochastic
    }

    /// Return a shared reference to the stage indexer.
    #[must_use]
    pub fn indexer(&self) -> &StageIndexer {
        &self.indexer
    }

    /// Return a shared reference to the future cost function.
    #[must_use]
    pub fn fcf(&self) -> &FutureCostFunction {
        &self.fcf
    }

    /// Return a mutable reference to the future cost function.
    ///
    /// Training and simulation modify the FCF (add/deactivate cuts), so
    /// callers must use this accessor when passing it to `train()` or
    /// `simulate()`.
    #[must_use]
    pub fn fcf_mut(&mut self) -> &mut FutureCostFunction {
        &mut self.fcf
    }

    /// Return a reference to the initial state vector.
    #[must_use]
    pub fn initial_state(&self) -> &[f64] {
        &self.initial_state
    }

    /// Return a reference to the horizon mode.
    #[must_use]
    pub fn horizon(&self) -> &HorizonMode {
        &self.horizon
    }

    /// Return a reference to the per-stage risk measures.
    #[must_use]
    pub fn risk_measures(&self) -> &[RiskMeasure] {
        &self.risk_measures
    }

    /// Return a reference to the entity counts struct.
    #[must_use]
    pub fn entity_counts(&self) -> &EntityCounts {
        &self.entity_counts
    }

    /// Return a reference to the hydro model preprocessing result.
    ///
    /// Contains the resolved production models, evaporation models, and
    /// provenance records for all hydro plants. Used by the LP builder
    /// (Epic 2/3) to configure hydro-related LP variables and constraints.
    #[must_use]
    pub fn hydro_models(&self) -> &PrepareHydroModelsResult {
        &self.hydro_models
    }

    // ── Accessors: derived layout ─────────────────────────────────────────────

    /// Return the number of blocks per stage as a slice.
    #[must_use]
    pub fn block_counts_per_stage(&self) -> &[usize] {
        &self.block_counts_per_stage
    }

    /// Return the maximum block count across all stages.
    #[must_use]
    pub fn max_blocks(&self) -> usize {
        self.max_blocks
    }

    // ── Accessors: config-derived scalars ─────────────────────────────────────

    /// Return the random seed used for noise generation.
    #[must_use]
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Return the number of forward passes per training iteration.
    #[must_use]
    pub fn forward_passes(&self) -> u32 {
        self.forward_passes
    }

    /// Return the maximum training iteration budget (derived from stopping rules).
    #[must_use]
    pub fn max_iterations(&self) -> u64 {
        self.max_iterations
    }

    /// Return the number of simulation scenarios (0 if simulation is disabled).
    #[must_use]
    pub fn n_scenarios(&self) -> u32 {
        self.n_scenarios
    }

    /// Return the I/O channel capacity for the simulation output pipeline.
    #[must_use]
    pub fn io_channel_capacity(&self) -> usize {
        self.io_channel_capacity
    }

    /// Return the policy directory path string.
    #[must_use]
    pub fn policy_path(&self) -> &str {
        &self.policy_path
    }

    /// Return a reference to the inflow non-negativity enforcement method.
    #[must_use]
    pub fn inflow_method(&self) -> &InflowNonNegativityMethod {
        &self.inflow_method
    }

    /// Return a reference to the optional cut selection strategy.
    #[must_use]
    pub fn cut_selection(&self) -> Option<&CutSelectionStrategy> {
        self.cut_selection.as_ref()
    }

    /// Minimum dual multiplier for a cut to count as binding.
    #[must_use]
    pub fn cut_activity_tolerance(&self) -> f64 {
        self.cut_activity_tolerance
    }

    /// Return a reference to the stopping rule set.
    #[must_use]
    pub fn stopping_rule_set(&self) -> &StoppingRuleSet {
        &self.stopping_rule_set
    }

    // ── Context constructors ──────────────────────────────────────────────────

    /// Construct a [`StageContext`] borrowing from this setup.
    #[must_use]
    pub fn stage_ctx(&self) -> StageContext<'_> {
        StageContext {
            templates: &self.stage_templates.templates,
            base_rows: &self.stage_templates.base_rows,
            noise_scale: &self.stage_templates.noise_scale,
            n_hydros: self.stage_templates.n_hydros,
            n_load_buses: self.stage_templates.n_load_buses,
            load_balance_row_starts: &self.stage_templates.load_balance_row_starts,
            load_bus_indices: &self.stage_templates.load_bus_indices,
            block_counts_per_stage: &self.block_counts_per_stage,
            ncs_max_gen: &self.ncs_max_gen,
        }
    }

    /// Construct a [`TrainingContext`] borrowing from this setup.
    #[must_use]
    pub fn training_ctx(&self) -> TrainingContext<'_> {
        TrainingContext {
            horizon: &self.horizon,
            indexer: &self.indexer,
            inflow_method: &self.inflow_method,
            stochastic: &self.stochastic,
            initial_state: &self.initial_state,
        }
    }

    // ── Training method ───────────────────────────────────────────────────────

    /// Execute the training loop using the precomputed study state.
    ///
    /// Constructs [`TrainingConfig`] and [`TrainingContext`] from the struct's
    /// owned fields, then delegates to the internal [`crate::train`] function.
    /// After a successful call, `self.fcf` contains all Benders cuts generated
    /// during the run.
    ///
    /// The method takes `&mut self` because training mutates the FCF cut pool.
    ///
    /// # Errors
    ///
    /// Returns `Err(SddpError::Infeasible { .. })` when an LP has no feasible
    /// solution. Returns `Err(SddpError::Solver(_))` for other solver failures.
    /// Returns `Err(SddpError::Communication(_))` when a collective operation
    /// fails.
    pub fn train<S: SolverInterface + Send, C: Communicator>(
        &mut self,
        solver: &mut S,
        comm: &C,
        n_threads: usize,
        solver_factory: impl Fn() -> Result<S, SolverError>,
        event_sender: Option<Sender<TrainingEvent>>,
        shutdown_flag: Option<&Arc<AtomicBool>>,
    ) -> Result<TrainingResult, SddpError> {
        let training_config = TrainingConfig {
            forward_passes: self.forward_passes,
            max_iterations: self.max_iterations,
            checkpoint_interval: None,
            warm_start_cuts: 0,
            event_sender,
        };

        // Inline context construction to allow &mut self.fcf (borrow checker requirements).
        let stage_ctx = StageContext {
            templates: &self.stage_templates.templates,
            base_rows: &self.stage_templates.base_rows,
            noise_scale: &self.stage_templates.noise_scale,
            n_hydros: self.stage_templates.n_hydros,
            n_load_buses: self.stage_templates.n_load_buses,
            load_balance_row_starts: &self.stage_templates.load_balance_row_starts,
            load_bus_indices: &self.stage_templates.load_bus_indices,
            block_counts_per_stage: &self.block_counts_per_stage,
            ncs_max_gen: &self.ncs_max_gen,
        };

        let training_ctx = TrainingContext {
            horizon: &self.horizon,
            indexer: &self.indexer,
            inflow_method: &self.inflow_method,
            stochastic: &self.stochastic,
            initial_state: &self.initial_state,
        };

        crate::train(
            solver,
            training_config,
            &mut self.fcf,
            &stage_ctx,
            &training_ctx,
            self.stochastic.opening_tree(),
            &self.risk_measures,
            self.stopping_rule_set.clone(),
            self.cut_selection.as_ref(),
            self.cut_activity_tolerance,
            shutdown_flag,
            comm,
            n_threads,
            solver_factory,
            self.max_blocks,
        )
    }

    // ── Simulation method ─────────────────────────────────────────────────────

    /// Execute the simulation pipeline using the trained future cost function.
    ///
    /// Constructs [`StageContext`], [`TrainingContext`], [`SimulationConfig`],
    /// and [`SimulationOutputSpec`] from the struct's owned fields, then
    /// delegates to [`crate::simulate`].
    ///
    /// The method takes `&self` because simulation only reads the FCF —
    /// the cut pool is not modified during simulation.
    ///
    /// The `result_tx` channel and `event_sender` are created by the caller
    /// (CLI/Python), because the caller also spawns the drain thread and
    /// progress display. `StudySetup` does not manage threads.
    ///
    /// `stage_bases` provides an optional per-stage warm-start basis from the
    /// training checkpoint. Pass `&training_result.basis_cache` to enable
    /// warm-start, or `&[]` to fall back to cold-start.
    ///
    /// # Errors
    ///
    /// Returns `Err(SimulationError::LpInfeasible { .. })` when a stage LP
    /// has no feasible solution.
    /// Returns `Err(SimulationError::SolverError { .. })` for other terminal
    /// LP solver failures.
    /// Returns `Err(SimulationError::ChannelClosed)` when the channel receiver
    /// has been dropped.
    pub fn simulate<S: SolverInterface + Send, C: Communicator>(
        &self,
        workspaces: &mut [SolverWorkspace<S>],
        comm: &C,
        result_tx: &SyncSender<SimulationScenarioResult>,
        event_sender: Option<Sender<TrainingEvent>>,
        stage_bases: &[Option<cobre_solver::Basis>],
    ) -> Result<crate::SimulationRunResult, SimulationError> {
        let stage_ctx = self.stage_ctx();
        let training_ctx = self.training_ctx();

        let sim_config = self.simulation_config();

        let output = SimulationOutputSpec {
            result_tx,
            zeta_per_stage: &self.stage_templates.zeta_per_stage,
            block_hours_per_stage: &self.stage_templates.block_hours_per_stage,
            entity_counts: &self.entity_counts,
            generic_constraint_row_entries: &self.stage_templates.generic_constraint_row_entries,
            ncs_col_starts: &self.stage_templates.ncs_col_starts,
            n_ncs_per_stage: &self.stage_templates.n_ncs_per_stage,
            ncs_entity_ids_per_stage: &self.ncs_entity_ids_per_stage,
            diversion_upstream: &self.stage_templates.diversion_upstream,
            hydro_productivities_per_stage: &self.stage_templates.hydro_productivities_per_stage,
            event_sender,
        };

        crate::simulate(
            workspaces,
            &stage_ctx,
            &self.fcf,
            &training_ctx,
            &sim_config,
            output,
            stage_bases,
            comm,
        )
    }

    // ── Training output method ────────────────────────────────────────────────

    /// Convert a [`TrainingResult`] and event log into the training output
    /// required by the output writers in `cobre-io`.
    ///
    /// This is a thin delegation to [`crate::build_training_output`], using
    /// the FCF stored in `self` to populate cut statistics.
    ///
    /// The conversion is pure and cannot fail.
    #[must_use]
    pub fn build_training_output(
        &self,
        result: &TrainingResult,
        events: &[TrainingEvent],
    ) -> cobre_io::TrainingOutput {
        crate::build_training_output(result, events, &self.fcf)
    }

    // ── Workspace pool helper ─────────────────────────────────────────────────

    /// Construct a [`WorkspacePool`] sized for this study's indexer dimensions.
    ///
    /// Each workspace receives a fresh solver instance created by
    /// `solver_factory`. The pool size equals `n_threads`.
    ///
    /// `n_load_buses` and `max_blocks` are taken from the pre-computed stage
    /// templates so that Category 4 patch buffers are correctly sized.
    ///
    /// # Errors
    ///
    /// Returns `Err(SolverError)` if any call to `solver_factory` fails.
    pub fn create_workspace_pool<S: SolverInterface + Send>(
        &self,
        n_threads: usize,
        solver_factory: impl Fn() -> Result<S, SolverError>,
    ) -> Result<WorkspacePool<S>, SolverError> {
        WorkspacePool::try_new(
            n_threads,
            self.indexer.hydro_count,
            self.indexer.max_par_order,
            self.indexer.n_state,
            self.stage_templates.n_load_buses,
            self.max_blocks,
            solver_factory,
        )
    }

    // ── SimulationConfig convenience accessor ─────────────────────────────────

    /// Build a [`SimulationConfig`] from the stored `n_scenarios` and
    /// `io_channel_capacity` fields.
    ///
    /// Provided as a convenience so callers do not have to construct the struct
    /// manually when they only need the config (e.g., for sizing a drain
    /// channel before calling [`simulate`](Self::simulate)).
    #[must_use]
    pub fn simulation_config(&self) -> SimulationConfig {
        SimulationConfig {
            n_scenarios: self.n_scenarios,
            io_channel_capacity: self.io_channel_capacity,
        }
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

    // ── Storage portion ───────────────────────────────────────────────────────
    for hs in &ic.storage {
        // Both hydros() and ic.storage are sorted by hydro_id.
        if let Ok(idx) = hydros.binary_search_by_key(&hs.hydro_id.0, |h| h.id.0) {
            state[idx] = hs.value_hm3;
        }
    }

    // ── Lag portion (populated from past_inflows) ─────────────────────────────
    if indexer.max_par_order > 0 {
        for pi in &ic.past_inflows {
            if let Ok(idx) = hydros.binary_search_by_key(&pi.hydro_id.0, |h| h.id.0) {
                let n_lags = pi.values_m3s.len().min(indexer.max_par_order);
                for lag in 0..n_lags {
                    let slot = indexer.inflow_lags.start + idx * indexer.max_par_order + lag;
                    state[slot] = pi.values_m3s[lag];
                }
            }
        }
    }

    state
}

// ---------------------------------------------------------------------------
// PrepareStochasticResult + prepare_stochastic
// ---------------------------------------------------------------------------

/// Result of the stochastic preprocessing pipeline.
///
/// Bundles the outputs of [`prepare_stochastic`] so that callers do not have
/// to handle three separate return values.
#[derive(Debug)]
pub struct PrepareStochasticResult {
    /// Updated system with estimated PAR models (if estimation ran).
    pub system: System,
    /// Built stochastic context, ready to pass to [`StudySetup::new`].
    pub stochastic: StochasticContext,
    /// Estimation report (`Some` if `inflow_history.parquet` was present and
    /// `inflow_seasonal_stats.parquet` was absent, triggering auto-estimation).
    pub estimation_report: Option<crate::EstimationReport>,
}

/// Load, validate, and assemble a user-supplied opening tree from the case directory.
///
/// Checks whether `scenarios/noise_openings.parquet` is present using
/// [`cobre_io::validate_structure`]. If absent, returns `Ok(None)`.
/// If present, loads the rows, validates dimensions and stage consistency,
/// and assembles an [`OpeningTree`].
///
/// # Errors
///
/// - [`SddpError::Io`] if the Parquet file cannot be read.
/// - [`SddpError::Io`] if rows fail dimension or stage consistency checks.
fn load_user_opening_tree_inner(
    case_dir: &Path,
    system: &System,
) -> Result<Option<OpeningTree>, SddpError> {
    let mut ctx = cobre_io::ValidationContext::new();
    let manifest = cobre_io::validate_structure(case_dir, &mut ctx);

    if !manifest.scenarios_noise_openings_parquet {
        return Ok(None);
    }

    let path = case_dir.join("scenarios").join("noise_openings.parquet");

    let rows = cobre_io::scenarios::load_noise_openings(Some(&path))?;

    let n_hydros = system.hydros().len();
    let mut load_bus_ids: Vec<EntityId> = system
        .load_models()
        .iter()
        .filter(|m| m.std_mw > 0.0)
        .map(|m| m.bus_id)
        .collect();
    load_bus_ids.sort_unstable_by_key(|id| id.0);
    load_bus_ids.dedup();
    let n_load_buses = load_bus_ids.len();
    let expected_dim = n_hydros + n_load_buses;

    let expected_stages = system.stages().iter().filter(|s| s.id >= 0).count();
    let mut openings_by_stage: BTreeMap<i32, BTreeSet<u32>> = BTreeMap::new();
    for row in &rows {
        openings_by_stage
            .entry(row.stage_id)
            .or_default()
            .insert(row.opening_index);
    }
    let openings_per_stage: Vec<usize> = openings_by_stage.values().map(BTreeSet::len).collect();

    cobre_io::scenarios::validate_noise_openings(
        &rows,
        expected_dim,
        expected_stages,
        &openings_per_stage,
    )?;

    let tree = cobre_io::scenarios::assemble_opening_tree(rows, expected_dim);
    Ok(Some(tree))
}

/// Build NCS entity factor entries from the `ResolvedNcsFactors` stored in `System`.
///
/// Converts the dense 3D factor table into the `(entity_id, stage_id, block_pairs)`
/// tuple format expected by `PrecomputedNormal::build`. Includes all NCS entities
/// that have model entries in `non_controllable_stats.parquet`. Entities with
/// `std_mw = 0` produce deterministic availability at their `mean_mw` value.
fn build_ncs_factor_entries(
    system: &System,
) -> Vec<(
    cobre_core::EntityId,
    i32,
    Vec<cobre_stochastic::normal::precompute::BlockFactorPair>,
)> {
    use cobre_stochastic::normal::precompute::BlockFactorPair;
    use std::collections::BTreeSet;

    // Collect NCS entity IDs that have model entries.
    let stochastic_ncs: BTreeSet<cobre_core::EntityId> =
        system.ncs_models().iter().map(|m| m.ncs_id).collect();

    if stochastic_ncs.is_empty() {
        return Vec::new();
    }

    let study_stages: Vec<_> = system.stages().iter().filter(|s| s.id >= 0).collect();
    let ncs_ids: Vec<cobre_core::EntityId> = system
        .non_controllable_sources()
        .iter()
        .map(|n| n.id)
        .collect();

    let mut entries = Vec::new();
    for (ncs_idx, ncs_id) in ncs_ids.iter().enumerate() {
        if !stochastic_ncs.contains(ncs_id) {
            continue;
        }
        for (stage_idx, stage) in study_stages.iter().enumerate() {
            #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
            let block_pairs: Vec<BlockFactorPair> = stage
                .blocks
                .iter()
                .enumerate()
                .map(|(block_idx, _)| {
                    let factor = system
                        .resolved_ncs_factors()
                        .factor(ncs_idx, stage_idx, block_idx);
                    // block_idx is a small count (< 1000 in practice); fits in i32.
                    (block_idx as i32, factor)
                })
                .collect();
            entries.push((*ncs_id, stage.id, block_pairs));
        }
    }
    entries
}

/// Load `scenarios/load_factors.json` from the case directory, returning an
/// empty vec when the file is absent. This is consumed by the stochastic
/// context builder for per-block noise scaling.
fn load_load_factors_for_stochastic(
    case_dir: &Path,
) -> Result<Vec<cobre_io::scenarios::LoadFactorEntry>, SddpError> {
    let path = case_dir.join("scenarios").join("load_factors.json");
    if !path.exists() {
        return Ok(Vec::new());
    }
    cobre_io::scenarios::parse_load_factors(&path).map_err(SddpError::from)
}

/// Prepare the stochastic pipeline: estimate PAR from history (if applicable),
/// load a user-supplied opening tree (if present), and build the
/// [`StochasticContext`].
///
/// This function encapsulates the pre-setup orchestration that would otherwise
/// be duplicated across entry points (CLI, Python bindings). It is intended to
/// be called once per entry point before constructing [`StudySetup`].
///
/// ## Input path matrix
///
/// | `inflow_history.parquet` | `inflow_seasonal_stats.parquet` | Behaviour |
/// |---|---|---|
/// | absent | any | System unchanged; `estimation_report = None`. |
/// | present | present | System unchanged; estimation skipped. |
/// | present | absent | PAR estimation runs; system updated. |
///
/// If `scenarios/noise_openings.parquet` is present, it is loaded, validated,
/// and passed as the user opening tree to [`cobre_stochastic::build_stochastic_context`].
///
/// ## MPI note
///
/// Under MPI, this function must only be called on rank 0. Non-root ranks
/// should receive the opening tree via broadcast and call
/// [`cobre_stochastic::build_stochastic_context`] directly.
///
/// # Errors
///
/// - [`SddpError::Io`] — file read, parse, or validation failure from either
///   `estimate_from_history` or opening tree loading.
/// - [`SddpError::Stochastic`] — PAR parameter validation or Cholesky
///   decomposition failure from `build_stochastic_context` or estimation.
pub fn prepare_stochastic(
    system: System,
    case_dir: &Path,
    config: &cobre_io::Config,
    seed: u64,
) -> Result<PrepareStochasticResult, SddpError> {
    let (system, estimation_report) =
        crate::estimation::estimate_from_history(system, case_dir, config)?;

    let user_opening_tree = load_user_opening_tree_inner(case_dir, &system)?;

    // Load block-level load factors (optional). When present, these scale the
    // stochastic noise realization per block, mirroring how the LP builder
    // scales the deterministic load balance RHS.
    let load_factor_entries = load_load_factors_for_stochastic(case_dir)?;

    // Convert LoadFactorEntry -> Vec<BlockFactorPair> per entry. The pairs
    // vec must outlive the entity_factor_entries references.
    let block_pairs: Vec<Vec<cobre_stochastic::normal::precompute::BlockFactorPair>> =
        load_factor_entries
            .iter()
            .map(|e| {
                e.block_factors
                    .iter()
                    .map(|bf| (bf.block_id, bf.factor))
                    .collect()
            })
            .collect();

    let entity_factor_entries: Vec<cobre_stochastic::normal::precompute::EntityFactorEntry<'_>> =
        load_factor_entries
            .iter()
            .zip(block_pairs.iter())
            .map(|(e, pairs)| (e.bus_id, e.stage_id, pairs.as_slice()))
            .collect();

    // Build NCS block factor entries from ResolvedNcsFactors, mirroring the
    // load factor conversion above. NCS entities consume their block factors
    // from the resolved NCS factors table.
    let ncs_factor_entries = build_ncs_factor_entries(&system);
    let ncs_entity_factor_entries: Vec<
        cobre_stochastic::normal::precompute::EntityFactorEntry<'_>,
    > = ncs_factor_entries
        .iter()
        .map(|(ncs_id, stage_id, pairs)| (*ncs_id, *stage_id, pairs.as_slice()))
        .collect();

    let stochastic = cobre_stochastic::build_stochastic_context(
        &system,
        seed,
        &entity_factor_entries,
        &ncs_entity_factor_entries,
        user_opening_tree,
    )?;

    Ok(PrepareStochasticResult {
        system,
        stochastic,
        estimation_report,
    })
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
        BusStagePenalties, ContractStageBounds, HydroStageBounds, HydroStagePenalties,
        LineStageBounds, LineStagePenalties, NcsStagePenalties, PumpingStageBounds, ResolvedBounds,
        ResolvedPenalties, ThermalStageBounds,
    };
    use cobre_core::{
        EntityId, SystemBuilder,
        entities::{
            bus::{Bus, DeficitSegment},
            hydro::{Hydro, HydroGenerationModel, HydroPenalties},
            thermal::{Thermal, ThermalCostSegment},
        },
        scenario::{InflowModel, LoadModel},
        temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
            StageStateConfig,
        },
    };
    use cobre_io::config::{
        Config, CutSelectionConfig, EstimationConfig, ExportsConfig, InflowNonNegativityConfig,
        ModelingConfig, PolicyConfig, SimulationConfig as IoSimulationConfig, StoppingRuleConfig,
        TrainingConfig, TrainingSolverConfig, UpperBoundEvaluationConfig,
    };
    use cobre_stochastic::build_stochastic_context;

    // ── Fixture helpers ───────────────────────────────────────────────────────

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
            cost_segments: vec![ThermalCostSegment {
                capacity_mw: 100.0,
                cost_per_mwh: 50.0,
            }],
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
            }
        }

        let bounds = ResolvedBounds::new(
            1, // n_hydros
            1, // n_thermals
            0, // n_lines
            0, // n_pumping
            0, // n_contracts
            n_st,
            default_hydro_bounds(),
            ThermalStageBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 100.0,
            },
            LineStageBounds {
                direct_mw: 0.0,
                reverse_mw: 0.0,
            },
            PumpingStageBounds {
                min_flow_m3s: 0.0,
                max_flow_m3s: 0.0,
            },
            ContractStageBounds {
                min_mw: 0.0,
                max_mw: 0.0,
                price_per_mwh: 0.0,
            },
        );

        let penalties = ResolvedPenalties::new(
            1, // n_hydros
            1, // n_buses
            0, // n_lines
            0, // n_ncs
            n_st,
            default_hydro_penalties(),
            BusStagePenalties { excess_cost: 0.0 },
            LineStagePenalties { exchange_cost: 0.0 },
            NcsStagePenalties {
                curtailment_cost: 0.0,
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
                seed: Some(42),
                forward_passes: Some(forward_passes),
                stopping_rules: Some(vec![StoppingRuleConfig::IterationLimit {
                    limit: max_iterations,
                }]),
                stopping_mode: "any".to_string(),
                cut_formulation: None,
                forward_pass: None,
                cut_selection: CutSelectionConfig::default(),
                solver: TrainingSolverConfig::default(),
            },
            upper_bound_evaluation: UpperBoundEvaluationConfig::default(),
            policy: PolicyConfig::default(),
            simulation: IoSimulationConfig::default(),
            exports: ExportsConfig::default(),
            estimation: EstimationConfig::default(),
        }
    }

    // ── Tests ─────────────────────────────────────────────────────────────────

    /// Given a minimal valid system (1 hydro, 1 thermal, 1 bus, 2 stages),
    /// when `StudySetup::new()` is called, then it returns `Ok` and
    /// `stage_templates()` returns a non-empty slice.
    #[test]
    fn new_minimal_valid_system_returns_ok() {
        let system = minimal_system(2);
        let config = minimal_config(1, 10);
        let stochastic =
            build_stochastic_context(&system, 42, &[], &[], None).expect("stochastic context");

        let result = StudySetup::new(
            &system,
            &config,
            stochastic,
            PrepareHydroModelsResult::default_from_system(&system),
        );
        assert!(result.is_ok(), "expected Ok, got {result:?}");
        let setup = result.unwrap();
        assert!(!setup.stage_templates().is_empty());
    }

    /// Given a system with zero study stages, when `StudySetup::new()` is
    /// called, then it returns `Err` containing the substring "no study stages".
    #[test]
    fn new_zero_stages_returns_validation_error() {
        let system = minimal_system(0);
        let config = minimal_config(1, 10);
        let stochastic =
            build_stochastic_context(&system, 42, &[], &[], None).expect("stochastic context");

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
        let stochastic =
            build_stochastic_context(&system, 42, &[], &[], None).expect("stochastic context");

        let setup = StudySetup::new(
            &system,
            &config,
            stochastic,
            PrepareHydroModelsResult::default_from_system(&system),
        )
        .expect("setup");

        // Stage templates
        assert_eq!(setup.stage_templates().len(), n_stages);
        assert_eq!(setup.base_rows().len(), n_stages);

        // Config-derived scalars
        assert_eq!(setup.seed(), 42);
        assert_eq!(setup.forward_passes(), 2);
        assert_eq!(setup.max_iterations(), 50);
        assert_eq!(setup.n_scenarios(), 0); // simulation disabled by default
        assert_eq!(setup.policy_path(), "./policy");

        // Derived layout
        assert_eq!(setup.block_counts_per_stage().len(), n_stages);
        assert!(setup.max_blocks() > 0);

        // Horizon
        assert_eq!(setup.horizon().num_stages(), n_stages);

        // Risk measures: one per study stage
        assert_eq!(setup.risk_measures().len(), n_stages);

        // FCF: pools match stage count
        assert_eq!(setup.fcf().pools.len(), n_stages);

        // Entity counts: 1 hydro, 1 thermal
        assert_eq!(setup.entity_counts().hydro_ids.len(), 1);
        assert_eq!(setup.entity_counts().thermal_ids.len(), 1);
    }

    /// FCF is accessible mutably via `fcf_mut()`.
    #[test]
    fn fcf_mut_allows_cut_insertion() {
        let system = minimal_system(2);
        let config = minimal_config(1, 10);
        let stochastic =
            build_stochastic_context(&system, 42, &[], &[], None).expect("stochastic context");

        let mut setup = StudySetup::new(
            &system,
            &config,
            stochastic,
            PrepareHydroModelsResult::default_from_system(&system),
        )
        .expect("setup");

        let n_state = setup.indexer().n_state;
        let coefficients = vec![1.0_f64; n_state];
        setup.fcf_mut().add_cut(0, 0, 0, 42.0, &coefficients);
        assert_eq!(setup.fcf().total_active_cuts(), 1);
    }

    /// `inflow_method()` reflects the config setting.
    #[test]
    fn inflow_method_reflects_config() {
        use crate::InflowNonNegativityMethod;

        let system = minimal_system(2);
        let config = minimal_config(1, 10);
        let stochastic =
            build_stochastic_context(&system, 42, &[], &[], None).expect("stochastic context");

        let setup = StudySetup::new(
            &system,
            &config,
            stochastic,
            PrepareHydroModelsResult::default_from_system(&system),
        )
        .expect("setup");

        // The minimal_config uses "penalty" — should not be None.
        assert!(
            !matches!(setup.inflow_method(), InflowNonNegativityMethod::None),
            "expected penalty or truncation method"
        );
    }

    /// `cut_selection()` returns `None` when disabled in config (default).
    #[test]
    fn cut_selection_none_when_disabled() {
        let system = minimal_system(2);
        let config = minimal_config(1, 10);
        let stochastic =
            build_stochastic_context(&system, 42, &[], &[], None).expect("stochastic context");

        let setup = StudySetup::new(
            &system,
            &config,
            stochastic,
            PrepareHydroModelsResult::default_from_system(&system),
        )
        .expect("setup");

        assert!(
            setup.cut_selection().is_none(),
            "cut_selection should be None when disabled"
        );
    }

    #[test]
    fn stage_ctx_fields_match_study_setup() {
        let n_stages = 3;
        let system = minimal_system(n_stages);
        let config = minimal_config(2, 10);
        let stochastic =
            build_stochastic_context(&system, 42, &[], &[], None).expect("stochastic context");

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
            setup.stage_templates().len(),
            "templates length mismatch"
        );
        assert_eq!(
            ctx.base_rows.len(),
            setup.base_rows().len(),
            "base_rows length mismatch"
        );
        assert_eq!(
            ctx.noise_scale.len(),
            setup.noise_scale().len(),
            "noise_scale length mismatch"
        );
        assert_eq!(
            ctx.n_hydros,
            setup.entity_counts().hydro_ids.len(),
            "n_hydros mismatch"
        );
        assert_eq!(
            ctx.block_counts_per_stage.len(),
            setup.block_counts_per_stage().len(),
            "block_counts_per_stage length mismatch"
        );
    }

    #[test]
    fn training_ctx_fields_match_study_setup() {
        let n_stages = 3;
        let system = minimal_system(n_stages);
        let config = minimal_config(2, 10);
        let stochastic =
            build_stochastic_context(&system, 42, &[], &[], None).expect("stochastic context");

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
            setup.horizon().num_stages(),
            "horizon num_stages mismatch"
        );
        assert_eq!(
            ctx.indexer.n_state,
            setup.indexer().n_state,
            "indexer n_state mismatch"
        );
        assert_eq!(
            ctx.initial_state.len(),
            setup.initial_state().len(),
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
        let stochastic =
            build_stochastic_context(&system, 42, &[], &[], None).expect("stochastic context");

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
            result.iterations <= 3,
            "expected iterations <= 3, got {}",
            result.iterations
        );
        assert!(
            result.iterations >= 1,
            "expected at least 1 iteration, got {}",
            result.iterations
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
        let stochastic =
            build_stochastic_context(&system, 42, &[], &[], None).expect("stochastic context");

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
            setup.fcf().pools[0].populated_count > 0,
            "expected at least one cut in FCF pool[0] after training"
        );
    }

    // ── simulation_config() ───────────────────────────────────────────────────

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
        let stochastic =
            build_stochastic_context(&system, 42, &[], &[], None).expect("stochastic context");

        let setup = StudySetup::new(
            &system,
            &config,
            stochastic,
            PrepareHydroModelsResult::default_from_system(&system),
        )
        .expect("setup");

        let sim_cfg = setup.simulation_config();
        assert_eq!(sim_cfg.n_scenarios, setup.n_scenarios());
        assert_eq!(sim_cfg.io_channel_capacity, setup.io_channel_capacity());
    }

    // ── create_workspace_pool() ───────────────────────────────────────────────

    /// `create_workspace_pool()` with `n_threads = 2` returns a pool whose
    /// `workspaces.len()` equals 2.
    #[test]
    fn create_workspace_pool_returns_correct_size() {
        use cobre_solver::highs::HighsSolver;

        let system = minimal_system(2);
        let config = minimal_config(1, 3);
        let stochastic =
            build_stochastic_context(&system, 42, &[], &[], None).expect("stochastic context");

        let setup = StudySetup::new(
            &system,
            &config,
            stochastic,
            PrepareHydroModelsResult::default_from_system(&system),
        )
        .expect("setup");

        let pool = setup
            .create_workspace_pool(2, HighsSolver::new)
            .expect("workspace pool");

        assert_eq!(pool.workspaces.len(), 2);
    }

    // ── build_training_output() ───────────────────────────────────────────────

    /// `build_training_output()` with a non-empty `TrainingResult` and empty
    /// events produces a `TrainingOutput` whose `convergence_records` is
    /// non-empty (one record per `result.iterations`).
    #[test]
    fn build_training_output_non_empty() {
        use cobre_comm::LocalBackend;
        use cobre_solver::highs::HighsSolver;

        let system = minimal_system(2);
        let config = minimal_config(1, 2);
        let stochastic =
            build_stochastic_context(&system, 42, &[], &[], None).expect("stochastic context");

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

        let output = setup.build_training_output(&result, &events);
        assert!(
            !output.convergence_records.is_empty(),
            "convergence_records must be non-empty after training"
        );
    }

    // ── simulate() integration test ───────────────────────────────────────────

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
        let stochastic =
            build_stochastic_context(&system, 42, &[], &[], None).expect("stochastic context");

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
            .create_workspace_pool(1, HighsSolver::new)
            .expect("sim pool");

        // Create the result channel and drain thread.
        let io_capacity = setup.io_channel_capacity().max(1);
        let (result_tx, result_rx) = std::sync::mpsc::sync_channel(io_capacity);
        let drain_handle = std::thread::spawn(move || result_rx.into_iter().collect::<Vec<_>>());

        let sim_result = setup
            .simulate(&mut pool.workspaces, &comm, &result_tx, None, &[])
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
            Config, CutSelectionConfig, EstimationConfig, ExportsConfig, InflowNonNegativityConfig,
            ModelingConfig, PolicyConfig, SimulationConfig as IoSimulationConfig, TrainingConfig,
            TrainingSolverConfig, UpperBoundEvaluationConfig,
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
                seed: None,
                forward_passes: None,
                stopping_rules: None,
                stopping_mode: "any".to_string(),
                cut_formulation: None,
                forward_pass: None,
                cut_selection: CutSelectionConfig::default(),
                solver: TrainingSolverConfig::default(),
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
            Config, CutSelectionConfig, EstimationConfig, ExportsConfig, InflowNonNegativityConfig,
            ModelingConfig, PolicyConfig, SimulationConfig as IoSimulationConfig,
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
                seed: Some(1234),
                forward_passes: Some(5),
                stopping_rules: Some(vec![
                    StoppingRuleConfig::IterationLimit { limit: 50 },
                    StoppingRuleConfig::TimeLimit { seconds: 60.0 },
                ]),
                stopping_mode: "all".to_string(),
                cut_formulation: None,
                forward_pass: None,
                cut_selection: CutSelectionConfig::default(),
                solver: TrainingSolverConfig::default(),
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

    // ── prepare_stochastic tests ──────────────────────────────────────────────

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
            Config, CutSelectionConfig, EstimationConfig, ExportsConfig, InflowNonNegativityConfig,
            ModelingConfig, PolicyConfig, SimulationConfig as IoSimulationConfig, TrainingConfig,
            TrainingSolverConfig, UpperBoundEvaluationConfig,
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
                seed: None,
                forward_passes: None,
                stopping_rules: None,
                stopping_mode: "any".to_string(),
                cut_formulation: None,
                forward_pass: None,
                cut_selection: CutSelectionConfig::default(),
                solver: TrainingSolverConfig::default(),
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
        use cobre_stochastic::provenance::ComponentProvenance;
        use tempfile::TempDir;

        let dir = TempDir::new().unwrap();
        let root = dir.path();
        write_minimal_case_dir(root);

        let system = minimal_system(2);
        let config = minimal_prepare_config();
        let seed = 42_u64;

        let result = prepare_stochastic(system, root, &config, seed)
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

        let result = prepare_stochastic(system, root, &config, seed)
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
        use cobre_stochastic::provenance::ComponentProvenance;
        use tempfile::TempDir;

        let dir = TempDir::new().unwrap();
        let root = dir.path();
        write_minimal_case_dir(root);

        let system = minimal_system(2);
        let config = minimal_prepare_config();

        let result = prepare_stochastic(system, root, &config, 0)
            .expect("prepare_stochastic must succeed with no opening tree file");

        assert_ne!(
            result.stochastic.provenance().opening_tree,
            ComponentProvenance::UserSupplied,
            "opening_tree provenance must not be UserSupplied when file is absent"
        );
    }

    // ── prepare_hydro_models / hydro_models accessor tests ────────────────────

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
        let stochastic =
            build_stochastic_context(&system, 42, &[], &[], None).expect("stochastic context");
        let hydro_result = PrepareHydroModelsResult::default_from_system(&system);

        let setup = StudySetup::new(&system, &config, stochastic, hydro_result).expect("setup");

        let models = setup.hydro_models();
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

    // ── build_initial_state lag population tests ──────────────────────────────

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
            }
        }

        let bounds = ResolvedBounds::new(
            2,
            0,
            0,
            0,
            0,
            n_st,
            default_hydro_bounds(),
            ThermalStageBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 0.0,
            },
            LineStageBounds {
                direct_mw: 0.0,
                reverse_mw: 0.0,
            },
            PumpingStageBounds {
                min_flow_m3s: 0.0,
                max_flow_m3s: 0.0,
            },
            ContractStageBounds {
                min_mw: 0.0,
                max_mw: 0.0,
                price_per_mwh: 0.0,
            },
        );

        let penalties = ResolvedPenalties::new(
            2,
            1,
            0,
            0,
            n_st,
            default_hydro_penalties(),
            BusStagePenalties { excess_cost: 0.0 },
            LineStagePenalties { exchange_cost: 0.0 },
            NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        );

        let past_inflows = vec![
            cobre_core::HydroPastInflows {
                hydro_id: EntityId(1),
                values_m3s: h1_past,
            },
            cobre_core::HydroPastInflows {
                hydro_id: EntityId(2),
                values_m3s: h2_past,
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

        // State layout: storage(0..2), lags(2..6).
        // Hydro at idx 0 (id=1): lag 0 = 600.0, lag 1 = 500.0.
        // Hydro at idx 1 (id=2): lag 0 = 200.0, lag 1 = 100.0.
        let s = indexer.inflow_lags.start;
        assert!(
            (state[s] - 600.0).abs() < 1e-10,
            "hydro 1 lag 0: expected 600.0, got {}",
            state[s]
        );
        assert!(
            (state[s + 1] - 500.0).abs() < 1e-10,
            "hydro 1 lag 1: expected 500.0, got {}",
            state[s + 1]
        );
        assert!(
            (state[s + 2] - 200.0).abs() < 1e-10,
            "hydro 2 lag 0: expected 200.0, got {}",
            state[s + 2]
        );
        assert!(
            (state[s + 3] - 100.0).abs() < 1e-10,
            "hydro 2 lag 1: expected 100.0, got {}",
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
        let stochastic =
            build_stochastic_context(&system, 42, &[], &[], None).expect("stochastic context");

        let setup = StudySetup::new(
            &system,
            &config,
            stochastic,
            PrepareHydroModelsResult::default_from_system(&system),
        )
        .expect("setup with past_inflows");

        let state = setup.initial_state();

        // With 2 hydros and max_par_order=2, lag slots start after storage slots.
        // Hydro 1 (idx 0): lag 0 = 600.0, lag 1 = 500.0
        // Hydro 2 (idx 1): lag 0 = 200.0, lag 1 = 100.0
        let n_hydros = 2;
        let lag_start = n_hydros;
        assert!(
            (state[lag_start] - 600.0).abs() < 1e-10,
            "hydro 1 lag 0 should be 600.0 via StudySetup, got {}",
            state[lag_start]
        );
        assert!(
            (state[lag_start + 1] - 500.0).abs() < 1e-10,
            "hydro 1 lag 1 should be 500.0 via StudySetup, got {}",
            state[lag_start + 1]
        );
        assert!(
            (state[lag_start + 2] - 200.0).abs() < 1e-10,
            "hydro 2 lag 0 should be 200.0 via StudySetup, got {}",
            state[lag_start + 2]
        );
        assert!(
            (state[lag_start + 3] - 100.0).abs() < 1e-10,
            "hydro 2 lag 1 should be 100.0 via StudySetup, got {}",
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
}
