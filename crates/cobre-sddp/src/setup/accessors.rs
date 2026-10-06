//! Accessor methods and context builders for [`StudySetup`].

use std::path::Path;

use cobre_core::AnticipatedCommitmentHistory;
use cobre_core::System;
#[cfg(any(test, feature = "test-support"))]
use cobre_core::commissioning::commissioning_active;
use cobre_io::EntitySlot;

#[cfg(any(test, feature = "test-support"))]
use cobre_io::config::BackwardScheduler;

#[cfg(any(test, feature = "test-support"))]
use crate::convergence::risk_measure::RiskMeasure;
use crate::{
    context::{StageContext, TrainingContext},
    cut::FutureCostFunction,
    energy_conversion::EnergyConversionSet,
    indexer::StateSpace,
    policy::orchestration::{CheckpointLayout, CheckpointParams, PeriodicCheckpoint},
    simulation::SimulationConfig,
    workspace::CapturedBasis,
};

use super::StudySetup;
use super::study_horizon_end;
use crate::policy_export::{build_graph_manifest, build_stage_entity_manifest};

impl StudySetup {
    /// Replace the FCF with a pre-loaded policy.
    pub fn replace_fcf(&mut self, fcf: FutureCostFunction) {
        self.fcf = fcf;
    }

    /// The boundary-derived state requirements this study was built against — the
    /// boundary-cut load path reads its inflow-lag depth here rather than
    /// re-resolving from the source checkpoint.
    #[must_use]
    pub fn boundary_requirements(&self) -> &super::BoundaryStateRequirements {
        &self.boundary_requirements
    }

    /// Set the resume point: the iterations the earlier run completed and the
    /// lower bound it recorded for each of them.
    pub fn set_resume_point(&mut self, completed_iterations: u64, lower_bound_history: Vec<f64>) {
        self.loop_params.start_iteration = completed_iterations;
        self.loop_params.resume_lower_bound_history = lower_bound_history;
    }

    /// Seed the per-stage warm-start basis cache for warm-start / resume
    /// training.
    ///
    /// `cache` carries one entry per stage (as built by
    /// [`build_basis_cache_from_checkpoint`](crate::build_basis_cache_from_checkpoint)).
    /// Leave unset (the default `None`) for a fresh start.
    pub fn set_warm_start_basis_cache(&mut self, cache: Vec<Option<CapturedBasis>>) {
        self.warm_start_basis_cache = Some(cache);
    }

    /// Enable state archiving for export.
    pub fn set_export_states(&mut self, export: bool) {
        self.events.export_states = export;
    }

    /// Have every later [`Self::train`] write a checkpoint to
    /// `output_dir.join(&self.policy_path)` on the iterations the
    /// `policy.checkpointing` schedule fires; does nothing when it is off.
    ///
    /// Call it on every rank: each rank evaluates the schedule and joins the
    /// write's error agreement, and rank 0 writes. `system` is passed explicitly
    /// because [`StudySetup`] does not own it.
    pub fn enable_periodic_checkpoints(&mut self, system: &System, output_dir: &Path) {
        let Some(schedule) = self.events.checkpoint_schedule else {
            return;
        };
        let layout = CheckpointLayout::new(
            self,
            system,
            CheckpointParams {
                max_iterations: self.loop_params.max_iterations,
                forward_passes: self.loop_params.forward_passes,
                seed: self.loop_params.seed,
                export_states: self.events.export_states,
            },
        );
        self.periodic_checkpoint = Some(PeriodicCheckpoint::new(
            schedule,
            output_dir.join(&self.policy_path),
            layout,
        ));
    }

    /// Test-support hook: override the per-stage backward-pass risk measures
    /// (`length` must equal `num_stages`), e.g. to swap `Expectation` for
    /// `CVaR { alpha, lambda }` without a config file exposing it per case.
    #[cfg(any(test, feature = "test-support"))]
    pub fn set_risk_measures(&mut self, risk_measures: Vec<RiskMeasure>) {
        self.inputs.cut_management.risk_measures = risk_measures;
    }

    /// Test-support hook: override the backward-pass scheduler
    /// (`training.parallelism.backward_scheduler`) to force `by_node`
    /// without a config file edit.
    #[cfg(any(test, feature = "test-support"))]
    pub fn set_scheduler(&mut self, scheduler: BackwardScheduler) {
        self.backward_scheduler = scheduler;
    }

    /// Test-support hook: override the opening-block scheduler's claim order
    /// (`BackwardPassState::set_hardest_first_claim_order`) — `false` forces
    /// the canonical ascending block order for the byte-neutrality gate.
    /// Production always resolves `true`; no config field surfaces this.
    #[cfg(any(test, feature = "test-support"))]
    pub fn set_hardest_first_claim_order(&mut self, enabled: bool) {
        self.hardest_first_claim_order = enabled;
    }

    /// Return the pre-computed [`EnergyConversionSet`] for this study.
    #[must_use]
    pub fn energy_conversion(&self) -> &EnergyConversionSet {
        &self.energy_conversion
    }

    /// Return a reference to the simulation configuration.
    #[must_use]
    pub fn simulation_config(&self) -> &SimulationConfig {
        &self.simulation_config
    }

    /// Return a reference to the stage-invariant role-(a) state-vector layout.
    ///
    /// State-region offsets are pure functions of `(N, L, A, k_max)`, so the
    /// single layout resolves onto the correct column at every stage regardless
    /// of per-stage block counts.
    #[must_use]
    pub fn stage_state(&self) -> &StateSpace {
        &self.inputs.stage_data.state
    }

    /// Resolve the terminal cut pool's ordinal and its owning study stage id.
    ///
    /// `terminal_idx` is a pool ordinal (`== n_pools - 1`); its owning stage
    /// resolves through `node_graph.pool_stage`, never `study_stage_ids[terminal_idx]`,
    /// which is OOB once `n_pools > n_stages` on a branching graph. Sole owner of
    /// this resolution so [`Self::build_terminal_entity_manifest`] resolves the
    /// pool's stage once, consistently.
    fn terminal_pool_stage_id(&self) -> (usize, i32) {
        let terminal_idx = self.inputs.cut_state_layouts.len() - 1;
        let stage_id =
            self.inputs.study_stage_ids[self.inputs.node_graph.pool_stage[terminal_idx].0];
        (terminal_idx, stage_id)
    }

    /// Build the per-slot entity-identity manifest for the terminal cut pool —
    /// the pool a boundary policy injects into.
    ///
    /// Delegates to
    /// [`build_stage_entity_manifest`],
    /// the single owner of identity resolution shared with the checkpoint writer,
    /// against the terminal pool's projection (the last entry of
    /// `cut_state_layouts`, pool-id-indexed; on the chain degeneracy this is the
    /// terminal stage). The caller passes the result to
    /// [`load_boundary_cuts`](crate::load_boundary_cuts) so a boundary cut whose
    /// slot identity diverges from the current study is rejected rather than
    /// silently mis-loaded.
    ///
    /// `system` is passed explicitly because [`StudySetup`] does not own it.
    #[must_use]
    pub fn build_terminal_entity_manifest(&self, system: &System) -> Vec<EntitySlot> {
        let (terminal_idx, stage_id) = self.terminal_pool_stage_id();
        build_stage_entity_manifest(
            system,
            &self.inputs.stage_data.state,
            &self.inputs.stage_data.study_dims.anticipated_plants,
            &self.inputs.cut_state_layouts[terminal_idx],
            stage_id,
        )
    }

    /// Build the study's fixed post-horizon (class-4) anticipated commitment
    /// windows — declared post-study deliveries the terminal boundary FCF
    /// prices by folding into each boundary cut's intercept — in canonical
    /// anticipated-plant order, then per-plant ascending `start_date`.
    ///
    /// A window is class-4 iff its `start_date` is at or after the study
    /// horizon end (the last study stage's `end_date`).
    /// `past_anticipated_commitments` carries only pre-study-decided class-2
    /// (in-study) and class-4 windows, and the calendar validation rejects any
    /// window straddling into an in-study-decided/never-priced stage, so the
    /// date threshold classifies unambiguously; the in-study (class-2) windows
    /// are the ones the terminal boundary does not price and this accessor
    /// omits.
    ///
    /// `system` is passed explicitly because [`StudySetup`] does not own it.
    #[must_use]
    pub fn build_terminal_fixed_post_horizon_windows(
        &self,
        system: &System,
    ) -> Vec<AnticipatedCommitmentHistory> {
        let Some(horizon_end) = study_horizon_end(system) else {
            return Vec::new();
        };
        let ic = system.initial_conditions();
        let thermals = system.thermals();
        let mut windows = Vec::new();
        for t in self
            .inputs
            .stage_data
            .study_dims
            .anticipated_plants
            .thermals()
        {
            let thermal = &thermals[t.get()];
            let mut plant_windows: Vec<AnticipatedCommitmentHistory> = ic
                .past_anticipated_commitments
                .iter()
                .filter(|w| w.thermal_id == thermal.id && w.start_date >= horizon_end)
                .cloned()
                .collect();
            plant_windows.sort_by_key(|w| w.start_date);
            windows.extend(plant_windows);
        }
        windows
    }

    /// Number of stages in the planning horizon.
    #[must_use]
    pub fn num_stages(&self) -> usize {
        self.inputs.horizon.num_stages()
    }

    /// Build the value-function artifact's graph manifest for the current study
    /// — node list, edges, and node → pool map — from the runtime node graph.
    ///
    /// Delegates to [`build_graph_manifest`], the single owner shared with the
    /// checkpoint writer, so the manifest written into an artifact and the one
    /// the full-FCF load path validates against can never diverge.
    #[must_use]
    pub fn build_graph_manifest(&self) -> cobre_io::GraphManifest {
        build_graph_manifest(&self.inputs.node_graph, &self.inputs.study_stage_ids)
    }

    /// Per-stage stochastic-NCS dormancy mask, reconstructed for the out-of-crate
    /// `tests/` harness (the dormancy mask is not stored — the patch path applies
    /// the commissioning predicate inline to `ncs_stochastic_windows`).
    ///
    /// Outer index is the study stage; inner is the stochastic slot (id-sorted
    /// `StochasticContext::ncs_entity_ids` order). `true` marks a
    /// commissioning-dormant slot whose dense NCS column is zeroed at that stage.
    #[cfg(any(test, feature = "test-support"))]
    #[must_use]
    pub fn ncs_stochastic_dormant_for_test(&self) -> Vec<Vec<bool>> {
        self.inputs
            .stage_data
            .stages
            .iter()
            .map(|stage| {
                self.inputs
                    .ncs
                    .stochastic_windows
                    .iter()
                    .map(|&(entry, exit)| !commissioning_active(entry, exit, stage.id))
                    .collect()
            })
            .collect()
    }

    /// Construct a [`StageContext`] borrowing from this setup.
    #[must_use]
    pub fn stage_ctx(&self) -> StageContext<'_> {
        self.inputs.stage_ctx()
    }

    /// Construct a [`TrainingContext`] borrowing from this setup.
    ///
    /// `create_workspace_pool`'s own owner-based sizing reads it; it is also
    /// reachable from downstream integration tests so a probe can drive
    /// production entry points (e.g. `forward::run_forward_pass`,
    /// `solve::stage_solve::run_stage_solve`) that take a `&TrainingContext`
    /// without duplicating this crate's private field layout.
    #[must_use]
    pub fn training_ctx(&self) -> TrainingContext<'_> {
        self.inputs.training_ctx()
    }

    /// Build simulation [`TrainingContext`] with simulation-specific schemes and libraries.
    #[must_use]
    pub(crate) fn simulation_ctx(&self) -> TrainingContext<'_> {
        self.inputs.simulation_ctx()
    }
}
