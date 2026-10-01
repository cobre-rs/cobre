//! Orchestration methods: train, simulate, and workspace pool construction.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::{Sender, SyncSender};

use cobre_comm::Communicator;
use cobre_core::TrainingEvent;
use cobre_io::TrainingOutput;
use cobre_solver::ActiveProfile;
use cobre_solver::StageTemplate;
use cobre_solver::{SolverError, SolverInterface};

use crate::{
    config::{CutManagementConfig, EventConfig, LoopConfig, TrainingConfig},
    error::SddpError,
    simulation::{
        SimulationOutputSpec, error::SimulationError, pipeline::SimulationRunResult,
        types::SimulationScenarioResult,
    },
    solve::solver_phase::SolverProfiles,
    training::{TrainingOutcome, TrainingResult},
    workspace::{
        CapturedBasis, NoisePreallocation, SolverWorkspace, WorkspacePool, WorkspaceSizing,
    },
};

use super::node_graph::pool_fill_basis_cache;
use super::{SimulationEnumeratedRequest, StudySetup, Traversal};
use crate::build_training_output;
use crate::simulate;
use crate::train;

impl StudySetup {
    /// Execute the training loop. Mutates `self.fcf` to store generated cuts.
    ///
    /// # Errors
    ///
    /// Returns `SddpError::Infeasible`, `SddpError::Solver`, or
    /// `SddpError::Communication` on LP, solver, or MPI failure.
    pub fn train<S, C: Communicator>(
        &mut self,
        solver: &mut S,
        comm: &C,
        n_threads: usize,
        solver_factory: impl Fn() -> Result<S, SolverError>,
        event_sender: Option<Sender<TrainingEvent>>,
        shutdown_flag: Option<&Arc<AtomicBool>>,
    ) -> Result<TrainingOutcome, SddpError>
    where
        S: SolverInterface<Profile = ActiveProfile> + Send,
    {
        let solver_profiles = SolverProfiles {
            forward: self.forward_profile,
            backward: self.backward_profile,
            backward_scheduler: self.backward_scheduler,
            hardest_first_claim_order: self.hardest_first_claim_order,
        };
        self.train_inner(
            solver,
            comm,
            n_threads,
            solver_factory,
            event_sender,
            shutdown_flag,
            solver_profiles,
        )
    }

    /// Test-support hook: [`Self::train`] with an explicit [`SolverProfiles`]
    /// override, bypassing the config-resolved `self.forward_profile`/
    /// `self.backward_profile`. Exists to force a low `simplex_iteration_limit`
    /// for the retry-armed determinism gate — a value the config surface
    /// deliberately does not expose (see `PhaseSolverProfileConfig`).
    ///
    /// # Errors
    ///
    /// Returns `SddpError::Infeasible`, `SddpError::Solver`, or
    /// `SddpError::Communication` on LP, solver, or MPI failure.
    #[cfg(any(test, feature = "test-support"))]
    pub fn train_with_solver_profiles<S, C: Communicator>(
        &mut self,
        solver: &mut S,
        comm: &C,
        n_threads: usize,
        solver_factory: impl Fn() -> Result<S, SolverError>,
        solver_profiles: SolverProfiles,
    ) -> Result<TrainingOutcome, SddpError>
    where
        S: SolverInterface<Profile = ActiveProfile> + Send,
    {
        self.train_inner(
            solver,
            comm,
            n_threads,
            solver_factory,
            None,
            None,
            solver_profiles,
        )
    }

    fn train_inner<S, C: Communicator>(
        &mut self,
        solver: &mut S,
        comm: &C,
        n_threads: usize,
        solver_factory: impl Fn() -> Result<S, SolverError>,
        event_sender: Option<Sender<TrainingEvent>>,
        shutdown_flag: Option<&Arc<AtomicBool>>,
        solver_profiles: SolverProfiles,
    ) -> Result<TrainingOutcome, SddpError>
    where
        S: SolverInterface<Profile = ActiveProfile> + Send,
    {
        let training_config = TrainingConfig {
            loop_config: LoopConfig {
                forward_passes: self.loop_params.forward_passes,
                training_enumerated: self.loop_params.training_enumerated,
                max_iterations: self.loop_params.max_iterations,
                start_iteration: self.loop_params.start_iteration,
                n_fwd_threads: n_threads,
                stopping_rules: self.loop_params.stopping_rules.clone(),
            },
            cut_management: CutManagementConfig {
                cut_selection: self.inputs.cut_management.cut_selection.clone(),
                budget: self.inputs.cut_management.budget,
                cut_activity_tolerance: self.inputs.cut_management.cut_activity_tolerance,
                risk_measures: self.inputs.cut_management.risk_measures.clone(),
            },
            events: EventConfig {
                event_sender,
                checkpoint_interval: None,
                shutdown_flag: shutdown_flag.map(Arc::clone),
                export_states: self.events.export_states,
            },
        };

        let stage_ctx = self.inputs.stage_ctx();
        let training_ctx = self.inputs.training_ctx();

        let warm_start_basis_cache = self.warm_start_basis_cache.take();

        train(
            solver,
            training_config,
            &mut self.fcf,
            &stage_ctx,
            &training_ctx,
            comm,
            solver_factory,
            warm_start_basis_cache,
            solver_profiles,
        )
    }

    /// Run simulation using the trained future cost function.
    ///
    /// The caller provides channels, event sender, and thread management.
    /// `frozen_templates` enables the frozen-template LP load path (no `add_rows`
    /// per stage); pass `None` for the legacy `load_model + add_rows` fallback.
    /// `stage_bases` enables warm-start; pass `&[]` for cold-start.
    ///
    /// # Errors
    ///
    /// Returns `SimulationError` on LP infeasibility, solver failure, channel closure,
    /// or if `frozen_templates.len() != n_pools`.
    pub fn simulate<S, C: Communicator>(
        &self,
        workspaces: &mut [SolverWorkspace<S>],
        comm: &C,
        result_tx: &SyncSender<SimulationScenarioResult>,
        event_sender: Option<Sender<TrainingEvent>>,
        frozen_templates: Option<&[StageTemplate]>,
        stage_bases: &[Option<CapturedBasis>],
    ) -> Result<SimulationRunResult, SimulationError>
    where
        S: SolverInterface<Profile = ActiveProfile> + Send,
    {
        let stage_ctx = self.stage_ctx();
        let training_ctx = self.simulation_ctx();
        let is_enumerated = matches!(
            self.simulation_enumerated,
            SimulationEnumeratedRequest::Enumerated
        );
        let traversal = Traversal::resolve(
            &self.inputs.node_graph,
            is_enumerated,
            self.simulation_config().n_scenarios,
        );

        // Pool-fill confined to this simulate-local buffer: the source cache
        // (and any checkpoint exported from it) stays sparse, so a warm-start
        // resume-training reseeds only genuine captures — never a filled leaf.
        let filled_bases: Option<Vec<Option<CapturedBasis>>> = is_enumerated.then(|| {
            let mut owned = stage_bases.to_vec();
            pool_fill_basis_cache(
                &mut owned,
                &self.inputs.node_graph.node_pool_ids(),
                &self.inputs.node_graph.node_ids,
            );
            owned
        });
        let stage_bases = filled_bases.as_deref().unwrap_or(stage_bases);

        let output = SimulationOutputSpec {
            result_tx,
            block_hours_per_stage: &self.inputs.stage_data.stage_templates.block_hours_per_stage,
            entity_counts: &self.inputs.stage_data.entity_counts,
            generic_constraint_row_entries: &self
                .inputs
                .stage_data
                .stage_templates
                .generic_constraint_row_entries,
            hydro_cell_index: &self.inputs.stage_data.hydro_cell_index,
            pumping_consumption_mw_per_m3s: &self.inputs.stage_data.pumping_consumption_mw_per_m3s,
            contract_prices_per_stage: &self.inputs.stage_data.contract_prices_per_stage,
            contract_slots: &self.inputs.stage_data.contract_slots,
            diversion_upstream: &self.inputs.stage_data.stage_templates.diversion_upstream,
            hydro_productivities_per_stage: &self
                .inputs
                .stage_data
                .stage_templates
                .hydro_productivities_per_stage,
            energy_conversion: &self.energy_conversion,
            hydro_min_storage_hm3: &self.hydro_min_storage_hm3,
            event_sender,
            extended_delivery_anchors: &self.extended_delivery_anchors,
            transit_seed_arcs: &self.transit_seed_arcs,
            past_defluences: &self.past_defluences,
            study_stage_dates: &self.study_stage_dates,
        };

        simulate(
            workspaces,
            &stage_ctx,
            &self.fcf,
            &training_ctx,
            self.simulation_config(),
            output,
            frozen_templates,
            stage_bases,
            comm,
            &traversal,
        )
    }

    /// Convert [`TrainingResult`] and events into training output.
    #[must_use]
    pub fn build_training_output(
        &self,
        result: &TrainingResult,
        events: &[TrainingEvent],
    ) -> TrainingOutput {
        build_training_output(
            result,
            events,
            &self.fcf,
            self.loop_params.training_enumerated,
        )
    }

    /// Create a [`WorkspacePool`] of `n_threads` workspaces sized for this study.
    ///
    /// # Errors
    ///
    /// Returns `SolverError` if solver creation fails.
    ///
    /// # Panics
    ///
    /// Panics if `comm.rank() > i32::MAX`. MPI world sizes are bounded well
    /// below this on all real systems.
    #[expect(
        clippy::expect_used,
        reason = "an MPI rank is a C int, so its i32 conversion cannot fail"
    )]
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
            &self.training_ctx(),
            &self.stage_ctx(),
            WorkspaceSizing {
                max_openings: (0..self.inputs.stage_data.stage_templates.templates.len())
                    .map(|t| self.inputs.stochastic.opening_tree().n_openings(t))
                    .max()
                    .unwrap_or(0),
                initial_pool_capacity: 0,
                // Simulation-only pool: forward-worker scratch fields unused.
                max_local_fwd: 0,
                noise: NoisePreallocation::OnDemand,
            },
            solver_factory,
        )?;
        // Always pre-size scratch bases — basis reconstruction runs
        // unconditionally on every forward/backward apply with a stored basis.
        let templates = &self.inputs.stage_data.stage_templates.templates;
        let max_cols = templates.iter().map(|t| t.num_cols).max().unwrap_or(0);
        let max_rows = templates.iter().map(|t| t.num_rows).max().unwrap_or(0);
        pool.resize_scratch_bases(max_cols, max_rows);
        Ok(pool)
    }
}
