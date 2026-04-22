//! Accessor methods and context builders for [`StudySetup`].

use cobre_core::scenario::SamplingScheme;
use cobre_stochastic::{ExternalScenarioLibrary, HistoricalScenarioLibrary, StochasticContext};

use crate::{
    FutureCostFunction, HorizonMode, InflowNonNegativityMethod, RiskMeasure, SimulationConfig,
    StageContext, StageIndexer, StageTemplates, TrainingContext,
    cut_selection::CutSelectionStrategy, hydro_models::PrepareHydroModelsResult,
    simulation::EntityCounts, stopping_rule::StoppingRuleSet,
};

use super::StudySetup;

impl StudySetup {
    // -------------------------------------------------------------------------
    // Getters — remain `pub` (called from cobre-cli or cobre-python)
    // -------------------------------------------------------------------------

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

    /// Replace the FCF with a pre-loaded one (for simulation-only mode).
    ///
    /// This swaps the internal `FutureCostFunction` with the provided one,
    /// enabling simulation against a policy loaded from disk without training.
    pub fn replace_fcf(&mut self, fcf: FutureCostFunction) {
        self.fcf = fcf;
    }

    /// Return the number of study stages.
    #[must_use]
    pub fn num_stages(&self) -> usize {
        self.stage_templates.templates.len()
    }

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

    /// Set the starting iteration for resumed training.
    ///
    /// When resuming from a checkpoint, call this with the checkpoint's
    /// `completed_iterations` so the training loop starts at `start_iteration + 1`.
    pub fn set_start_iteration(&mut self, iteration: u64) {
        self.start_iteration = iteration;
    }

    /// Enable the visited-states archive for state export.
    ///
    /// When `true`, the archive is allocated during training regardless of
    /// whether cut selection requires it. Set this based on the
    /// `exports.states` configuration flag.
    pub fn set_export_states(&mut self, export: bool) {
        self.export_states = export;
    }

    /// Set the active-cut budget cap for training.
    ///
    /// When `Some(n)`, the training loop enforces a hard cap of `n` active cuts
    /// per stage after each backward pass. When `None`, no cap is enforced.
    pub fn set_budget(&mut self, budget: Option<u32>) {
        self.budget = budget;
    }

    /// Return the number of simulation scenarios (0 if simulation is disabled).
    #[must_use]
    pub fn n_scenarios(&self) -> u32 {
        self.n_scenarios
    }

    /// Return the policy directory path string.
    #[must_use]
    pub fn policy_path(&self) -> &str {
        &self.policy_path
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

    /// Return the historical inflow scenario library, if built.
    ///
    /// `Some` when `inflow_scheme == SamplingScheme::Historical`, else `None`.
    #[must_use]
    pub fn historical_library(&self) -> Option<&HistoricalScenarioLibrary> {
        self.historical_library.as_ref()
    }

    /// Return the external inflow scenario library, if built.
    ///
    /// `Some` when `inflow_scheme == SamplingScheme::External`, else `None`.
    #[must_use]
    pub fn external_inflow_library(&self) -> Option<&ExternalScenarioLibrary> {
        self.external_inflow_library.as_ref()
    }

    /// Return the external load scenario library, if built.
    ///
    /// `Some` when `load_scheme == SamplingScheme::External`, else `None`.
    #[must_use]
    pub fn external_load_library(&self) -> Option<&ExternalScenarioLibrary> {
        self.external_load_library.as_ref()
    }

    /// Return the external NCS scenario library, if built.
    ///
    /// `Some` when `ncs_scheme == SamplingScheme::External`, else `None`.
    #[must_use]
    pub fn external_ncs_library(&self) -> Option<&ExternalScenarioLibrary> {
        self.external_ncs_library.as_ref()
    }

    /// Build a [`SimulationConfig`] from stored fields.
    ///
    /// Convenience helper for callers who only need the config.
    #[must_use]
    pub fn simulation_config(&self) -> SimulationConfig {
        SimulationConfig {
            n_scenarios: self.n_scenarios,
            io_channel_capacity: self.io_channel_capacity,
            basis_activity_window: self.basis_activity_window,
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // Getters — `pub(crate)` only (used within cobre-sddp and in tests)
    // ─────────────────────────────────────────────────────────────────────

    /// Return the per-stage water-balance row offset array.
    #[must_use]
    #[allow(dead_code)]
    pub(crate) fn base_rows(&self) -> &[usize] {
        &self.stage_templates.base_rows
    }

    /// Return the pre-computed noise scale factors (stage-major layout).
    #[must_use]
    #[allow(dead_code)]
    pub(crate) fn noise_scale(&self) -> &[f64] {
        &self.stage_templates.noise_scale
    }

    /// Return a reference to the initial state vector.
    #[must_use]
    #[allow(dead_code)]
    pub(crate) fn initial_state(&self) -> &[f64] {
        &self.initial_state
    }

    /// Return a reference to the horizon mode.
    #[must_use]
    #[allow(dead_code)]
    pub(crate) fn horizon(&self) -> &HorizonMode {
        &self.horizon
    }

    /// Return a reference to the per-stage risk measures.
    #[must_use]
    #[allow(dead_code)]
    pub(crate) fn risk_measures(&self) -> &[RiskMeasure] {
        &self.risk_measures
    }

    /// Return a reference to the entity counts struct.
    #[must_use]
    #[allow(dead_code)]
    pub(crate) fn entity_counts(&self) -> &EntityCounts {
        &self.entity_counts
    }

    /// Return the number of blocks per stage as a slice.
    #[must_use]
    #[allow(dead_code)]
    pub(crate) fn block_counts_per_stage(&self) -> &[usize] {
        &self.block_counts_per_stage
    }

    /// Return the maximum block count across all stages.
    #[must_use]
    #[allow(dead_code)]
    pub(crate) fn max_blocks(&self) -> usize {
        self.max_blocks
    }

    /// Return the I/O channel capacity for the simulation output pipeline.
    #[must_use]
    pub fn io_channel_capacity(&self) -> usize {
        self.io_channel_capacity
    }

    /// Return a reference to the inflow non-negativity enforcement method.
    #[must_use]
    #[allow(dead_code)]
    pub(crate) fn inflow_method(&self) -> &InflowNonNegativityMethod {
        &self.inflow_method
    }

    /// Return a reference to the optional cut selection strategy.
    #[must_use]
    #[allow(dead_code)]
    pub(crate) fn cut_selection(&self) -> Option<&CutSelectionStrategy> {
        self.cut_selection.as_ref()
    }

    /// Minimum dual multiplier for a cut to count as binding.
    #[must_use]
    #[allow(dead_code)]
    pub(crate) fn cut_activity_tolerance(&self) -> f64 {
        self.cut_activity_tolerance
    }

    /// Activity-window size for the basis-reconstruction classifier (1..=31).
    #[must_use]
    #[allow(dead_code)]
    pub(crate) fn basis_activity_window(&self) -> u32 {
        self.basis_activity_window
    }

    /// Return a reference to the stopping rule set.
    #[must_use]
    #[allow(dead_code)]
    pub(crate) fn stopping_rule_set(&self) -> &StoppingRuleSet {
        &self.stopping_rule_set
    }

    /// Return the precomputed noise group IDs, indexed by stage.
    ///
    /// Stages sharing the same `(season_id, year)` have the same group ID.
    /// Consumed by `ForwardSampler` and `generate_opening_tree` in Epic 2
    /// to share noise draws across weekly stages within the same monthly bucket.
    #[must_use]
    pub fn noise_group_ids(&self) -> &[u32] {
        &self.noise_group_ids
    }

    // ─────────────────────────────────────────────────────────────────────
    // Context builders — `pub(crate)` only
    // ─────────────────────────────────────────────────────────────────────

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
            discount_factors: &self.stage_templates.discount_factors,
            cumulative_discount_factors: &self.stage_templates.cumulative_discount_factors,
            stage_lag_transitions: &self.stage_lag_transitions,
            noise_group_ids: &self.noise_group_ids,
            downstream_par_order: self.downstream_par_order,
        }
    }

    /// Construct a [`TrainingContext`] borrowing from this setup.
    #[must_use]
    #[allow(dead_code)]
    pub(crate) fn training_ctx(&self) -> TrainingContext<'_> {
        TrainingContext {
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
        }
    }

    /// Build simulation [`TrainingContext`] with simulation-specific schemes and libraries.
    ///
    /// Reuses training libraries when simulation schemes match. Selects per-class
    /// libraries in this order: simulation-specific, then training (shared).
    #[must_use]
    pub(crate) fn simulation_ctx(&self) -> TrainingContext<'_> {
        // For each class, prefer the simulation-specific library when present;
        // fall back to the training library when schemes are identical.
        let historical_library = self.sim_historical_library.as_ref().or(
            if self.sim_inflow_scheme == SamplingScheme::Historical {
                self.historical_library.as_ref()
            } else {
                None
            },
        );
        let external_inflow_library = self.sim_external_inflow_library.as_ref().or(
            if self.sim_inflow_scheme == SamplingScheme::External {
                self.external_inflow_library.as_ref()
            } else {
                None
            },
        );
        let external_load_library = self.sim_external_load_library.as_ref().or(
            if self.sim_load_scheme == SamplingScheme::External {
                self.external_load_library.as_ref()
            } else {
                None
            },
        );
        let external_ncs_library = self.sim_external_ncs_library.as_ref().or(
            if self.sim_ncs_scheme == SamplingScheme::External {
                self.external_ncs_library.as_ref()
            } else {
                None
            },
        );

        TrainingContext {
            horizon: &self.horizon,
            indexer: &self.indexer,
            inflow_method: &self.inflow_method,
            stochastic: &self.stochastic,
            initial_state: &self.initial_state,
            inflow_scheme: self.sim_inflow_scheme,
            load_scheme: self.sim_load_scheme,
            ncs_scheme: self.sim_ncs_scheme,
            stages: &self.stages,
            historical_library,
            external_inflow_library,
            external_load_library,
            external_ncs_library,
            recent_accum_seed: &self.recent_observation_seed.accum_seed,
            recent_weight_seed: self.recent_observation_seed.weight_seed,
        }
    }
}
