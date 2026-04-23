//! Accessor methods and context builders for [`StudySetup`].

use cobre_core::scenario::SamplingScheme;

use crate::{FutureCostFunction, StageContext, TrainingContext, simulation::SimulationConfig};

use super::StudySetup;

impl StudySetup {
    // -------------------------------------------------------------------------
    // Mutation setters — remain `pub` (called from cobre-cli or cobre-python)
    // -------------------------------------------------------------------------

    /// Replace the FCF with a pre-loaded policy.
    pub fn replace_fcf(&mut self, fcf: FutureCostFunction) {
        self.fcf = fcf;
    }

    /// Set the starting iteration for resumed training.
    pub fn set_start_iteration(&mut self, iteration: u64) {
        self.loop_params.start_iteration = iteration;
    }

    /// Enable state archiving for export.
    pub fn set_export_states(&mut self, export: bool) {
        self.events.export_states = export;
    }

    /// Set the active-cut budget cap per stage.
    pub fn set_budget(&mut self, budget: Option<u32>) {
        self.cut_management.budget = budget;
    }

    // ─────────────────────────────────────────────────────────────────────
    // Context builders — span multiple sub-structs
    // ─────────────────────────────────────────────────────────────────────

    /// Return a reference to the simulation configuration.
    #[must_use]
    pub fn simulation_config(&self) -> &SimulationConfig {
        &self.simulation_config
    }

    /// Construct a [`StageContext`] borrowing from this setup.
    #[must_use]
    pub fn stage_ctx(&self) -> StageContext<'_> {
        StageContext {
            templates: &self.stage_data.stage_templates.templates,
            base_rows: &self.stage_data.stage_templates.base_rows,
            noise_scale: &self.stage_data.stage_templates.noise_scale,
            n_hydros: self.stage_data.stage_templates.n_hydros,
            n_load_buses: self.stage_data.stage_templates.n_load_buses,
            load_balance_row_starts: &self.stage_data.stage_templates.load_balance_row_starts,
            load_bus_indices: &self.stage_data.stage_templates.load_bus_indices,
            block_counts_per_stage: &self.stage_data.block_counts_per_stage,
            ncs_max_gen: &self.ncs_max_gen,
            discount_factors: &self.stage_data.stage_templates.discount_factors,
            cumulative_discount_factors: &self
                .stage_data
                .stage_templates
                .cumulative_discount_factors,
            stage_lag_transitions: &self.stage_data.stage_lag_transitions,
            noise_group_ids: &self.stage_data.noise_group_ids,
            downstream_par_order: self.downstream_par_order,
        }
    }

    /// Construct a [`TrainingContext`] borrowing from this setup. Test-only.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn training_ctx(&self) -> TrainingContext<'_> {
        let tr = &self.scenario_libraries.training;
        TrainingContext {
            horizon: &self.methodology.horizon,
            indexer: &self.stage_data.indexer,
            inflow_method: &self.methodology.inflow_method,
            stochastic: &self.stochastic,
            initial_state: &self.initial_state,
            inflow_scheme: tr.inflow_scheme,
            load_scheme: tr.load_scheme,
            ncs_scheme: tr.ncs_scheme,
            stages: &self.stage_data.stages,
            historical_library: tr.historical.as_ref(),
            external_inflow_library: tr.external_inflow.as_ref(),
            external_load_library: tr.external_load.as_ref(),
            external_ncs_library: tr.external_ncs.as_ref(),
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
        let tr = &self.scenario_libraries.training;
        let sim = &self.scenario_libraries.simulation;

        // For each class, prefer the simulation-specific library when present;
        // fall back to the training library when schemes are identical.
        let historical_library =
            sim.historical
                .as_ref()
                .or(if sim.inflow_scheme == SamplingScheme::Historical {
                    tr.historical.as_ref()
                } else {
                    None
                });
        let external_inflow_library =
            sim.external_inflow
                .as_ref()
                .or(if sim.inflow_scheme == SamplingScheme::External {
                    tr.external_inflow.as_ref()
                } else {
                    None
                });
        let external_load_library =
            sim.external_load
                .as_ref()
                .or(if sim.load_scheme == SamplingScheme::External {
                    tr.external_load.as_ref()
                } else {
                    None
                });
        let external_ncs_library =
            sim.external_ncs
                .as_ref()
                .or(if sim.ncs_scheme == SamplingScheme::External {
                    tr.external_ncs.as_ref()
                } else {
                    None
                });

        TrainingContext {
            horizon: &self.methodology.horizon,
            indexer: &self.stage_data.indexer,
            inflow_method: &self.methodology.inflow_method,
            stochastic: &self.stochastic,
            initial_state: &self.initial_state,
            inflow_scheme: sim.inflow_scheme,
            load_scheme: sim.load_scheme,
            ncs_scheme: sim.ncs_scheme,
            stages: &self.stage_data.stages,
            historical_library,
            external_inflow_library,
            external_load_library,
            external_ncs_library,
            recent_accum_seed: &self.recent_observation_seed.accum_seed,
            recent_weight_seed: self.recent_observation_seed.weight_seed,
        }
    }
}
