//! Orchestration methods: train, simulate, and workspace pool construction.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::{Sender, SyncSender};

use cobre_comm::Communicator;
use cobre_core::TrainingEvent;
use cobre_solver::{SolverError, SolverInterface};

use crate::{
    CapturedBasis, CutManagementConfig, EventConfig, LoopConfig, SddpError, SimulationError,
    SimulationRunResult, SimulationScenarioResult, SolverWorkspace, StageContext, TrainingConfig,
    TrainingContext, TrainingOutcome, TrainingResult, WorkspacePool, WorkspaceSizing,
    simulation::SimulationOutputSpec,
};

use super::StudySetup;

impl StudySetup {
    /// Execute the training loop.
    ///
    /// Constructs [`TrainingConfig`] and [`TrainingContext`], then delegates to
    /// [`crate::train`]. Mutates `self.fcf` to store generated cuts.
    ///
    /// # Errors
    ///
    /// Returns `SddpError::Infeasible`, `SddpError::Solver`, or
    /// `SddpError::Communication` on LP, solver, or MPI failure.
    pub fn train<S: SolverInterface + Send, C: Communicator>(
        &mut self,
        solver: &mut S,
        comm: &C,
        n_threads: usize,
        solver_factory: impl Fn() -> Result<S, SolverError>,
        event_sender: Option<Sender<TrainingEvent>>,
        shutdown_flag: Option<&Arc<AtomicBool>>,
    ) -> Result<TrainingOutcome, SddpError> {
        let training_config = TrainingConfig {
            loop_config: LoopConfig {
                forward_passes: self.forward_passes,
                max_iterations: self.max_iterations,
                start_iteration: self.start_iteration,
                n_fwd_threads: n_threads,
                max_blocks: self.max_blocks,
                stopping_rules: self.stopping_rule_set.clone(),
            },
            cut_management: CutManagementConfig {
                cut_selection: self.cut_selection.clone(),
                budget: self.budget,
                cut_activity_tolerance: self.cut_activity_tolerance,
                warm_start_cuts: 0,
                risk_measures: self.risk_measures.clone(),
            },
            events: EventConfig {
                event_sender,
                checkpoint_interval: None,
                shutdown_flag: shutdown_flag.map(Arc::clone),
                export_states: self.export_states,
            },
        };

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
            discount_factors: &self.stage_templates.discount_factors,
            cumulative_discount_factors: &self.stage_templates.cumulative_discount_factors,
            stage_lag_transitions: &self.stage_lag_transitions,
            noise_group_ids: &self.noise_group_ids,
            downstream_par_order: self.downstream_par_order,
        };

        let training_ctx = TrainingContext {
            horizon: &self.horizon,
            indexer: &self.indexer,
            inflow_method: &self.inflow_method,
            stochastic: &self.stochastic,
            initial_state: &self.initial_state,
            inflow_scheme: self.inflow_scheme,
            load_scheme: self.load_scheme,
            ncs_scheme: self.ncs_scheme,
            stages: &self.stages,
            historical_library: self.historical_library.as_ref(),
            external_inflow_library: self.external_inflow_library.as_ref(),
            external_load_library: self.external_load_library.as_ref(),
            external_ncs_library: self.external_ncs_library.as_ref(),
            recent_accum_seed: &self.recent_observation_seed.accum_seed,
            recent_weight_seed: self.recent_observation_seed.weight_seed,
        };

        crate::train(
            solver,
            training_config,
            &mut self.fcf,
            &stage_ctx,
            &training_ctx,
            comm,
            solver_factory,
        )
    }

    /// Run simulation using the trained future cost function.
    ///
    /// The caller provides channels, event sender, and thread management.
    /// `baked_templates` enables the baked-template LP load path (no `add_rows`
    /// per stage); pass `None` for the legacy `load_model + add_rows` fallback.
    /// `stage_bases` enables warm-start; pass `&[]` for cold-start.
    ///
    /// # Errors
    ///
    /// Returns `SimulationError` on LP infeasibility, solver failure, channel closure,
    /// or if `baked_templates.len() != num_stages`.
    pub fn simulate<S: SolverInterface + Send, C: Communicator>(
        &self,
        workspaces: &mut [SolverWorkspace<S>],
        comm: &C,
        result_tx: &SyncSender<SimulationScenarioResult>,
        event_sender: Option<Sender<TrainingEvent>>,
        baked_templates: Option<&[cobre_solver::StageTemplate]>,
        stage_bases: &[Option<CapturedBasis>],
    ) -> Result<SimulationRunResult, SimulationError> {
        let stage_ctx = self.stage_ctx();
        let training_ctx = self.simulation_ctx();

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
            baked_templates,
            stage_bases,
            comm,
        )
    }

    /// Convert [`TrainingResult`] and events into training output.
    ///
    /// Delegates to [`crate::build_training_output`] with cut statistics from `self.fcf`.
    #[must_use]
    pub fn build_training_output(
        &self,
        result: &TrainingResult,
        events: &[TrainingEvent],
    ) -> cobre_io::TrainingOutput {
        crate::build_training_output(result, events, &self.fcf)
    }

    /// Create a [`WorkspacePool`] sized for this study.
    ///
    /// Pool size equals `n_threads`. Each workspace gets a fresh solver instance.
    /// `comm` is used to read the MPI rank that is stamped into each workspace's
    /// `rank` field for downstream per-worker observability (epic-04b).
    ///
    /// # Errors
    ///
    /// Returns `SolverError` if solver creation fails.
    ///
    /// # Panics
    ///
    /// Panics if `comm.rank() > i32::MAX`. MPI world sizes are bounded well
    /// below this on all real systems.
    #[allow(clippy::expect_used)]
    pub fn create_workspace_pool<S: SolverInterface + Send, C: Communicator>(
        &self,
        comm: &C,
        n_threads: usize,
        solver_factory: impl Fn() -> Result<S, SolverError>,
    ) -> Result<WorkspacePool<S>, SolverError> {
        let rank = i32::try_from(comm.rank()).expect("MPI rank fits in i32");
        let mut pool = WorkspacePool::try_new(
            rank,
            n_threads,
            self.indexer.n_state,
            WorkspaceSizing {
                hydro_count: self.indexer.hydro_count,
                max_par_order: self.indexer.max_par_order,
                n_load_buses: self.stage_templates.n_load_buses,
                max_blocks: self.max_blocks,
                downstream_par_order: self.downstream_par_order,
                max_openings: (0..self.num_stages())
                    .map(|t| self.stochastic.opening_tree().n_openings(t))
                    .max()
                    .unwrap_or(0),
                initial_pool_capacity: 0,
                n_state: self.indexer.n_state,
            },
            solver_factory,
        )?;
        // Always pre-size scratch bases — basis reconstruction runs
        // unconditionally on every forward/backward apply with a stored basis.
        let max_cols = self
            .stage_templates
            .templates
            .iter()
            .map(|t| t.num_cols)
            .max()
            .unwrap_or(0);
        let max_rows = self
            .stage_templates
            .templates
            .iter()
            .map(|t| t.num_rows)
            .max()
            .unwrap_or(0);
        pool.resize_scratch_bases(max_cols, max_rows);
        Ok(pool)
    }
}
