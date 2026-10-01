//! [`SolveInputs`]: the resolved study inputs the stage, training, and
//! simulation contexts all borrow from.

use cobre_core::scenario::SamplingScheme;
use cobre_stochastic::StochasticContext;

use crate::{
    config::CutManagementConfig,
    context::{StageContext, TrainingContext},
    dcs::DcsParams,
    horizon_mode::HorizonMode,
    lp::indexer::CutStateProjection,
};

use super::{InitialConditions, NcsEntityData, NodeGraph, ScenarioLibraries, StageData};

/// The resolved study inputs shared by [`StageContext`] and [`TrainingContext`]
/// (training and simulation phases), disjoint from
/// [`StudySetup::fcf`](super::StudySetup::fcf) so `train_inner` can build a
/// context while holding `&mut fcf`.
#[derive(Debug)]
pub struct SolveInputs {
    /// Stage-indexed data: LP templates, indexer, stages, entity counts, blocks,
    /// lag transitions, noise groups, and scaling report.
    pub stage_data: StageData,
    /// Stochastic context holding sampling distributions, libraries, and provenance.
    pub stochastic: StochasticContext,
    /// Sampling schemes and pre-built libraries for training and simulation phases.
    pub scenario_libraries: ScenarioLibraries,
    /// The runtime node graph: node identity/order, the `node → pool`
    /// map, and per-node Ω views/out-edges. Absent `nodes[]` this is the
    /// byte-exact chain degeneracy.
    pub node_graph: NodeGraph,
    /// Initial state vector and derived inflow-lag seeds.
    pub(crate) initial: InitialConditions,
    /// Per-stage and per-slot NCS entity data.
    pub(crate) ncs: NcsEntityData,
    /// `study_stage_ids[t] = stage.id` per study stage index.
    pub(crate) study_stage_ids: Vec<i32>,
    /// Study horizon mode (finite vs. infinite-horizon approximation).
    pub(crate) horizon: HorizonMode,
    /// Two-stage cut management pipeline configuration.
    pub(crate) cut_management: CutManagementConfig,
    /// Per-pool cut-state projection, indexed by pool id, paired 1:1 with
    /// [`crate::FutureCostFunction::pools`] — the single owner of each pool's
    /// cut-state dimension.
    pub(crate) cut_state_layouts: Vec<CutStateProjection>,
}

impl SolveInputs {
    /// Construct a [`StageContext`] borrowing from these inputs.
    pub(crate) fn stage_ctx(&self) -> StageContext<'_> {
        StageContext {
            templates: &self.stage_data.stage_templates.templates,
            state_boxes: self.stage_data.stage_templates.state_boxes(),
            geometry_per_stage: &self.stage_data.stage_templates.geometry_per_stage,
            cost_scale_factor: self.stage_data.stage_templates.cost_scale_factor,
            load_bus_indices: &self.stage_data.stage_templates.load_bus_indices,
            ncs_stochastic_dense_col: &self.ncs.stochastic_dense_col,
            ncs_stochastic_windows: &self.ncs.stochastic_windows,
            anticipated_windows: self.stage_data.study_dims.anticipated_plants.windows(),
            study_stage_ids: &self.study_stage_ids,
            ncs_max_gen: &self.ncs.max_gen,
            ncs_allow_curtailment: &self.ncs.allow_curtailment,
            discount_factors: self.stage_data.time_value.discount_factors(),
            cumulative_discount_factors: self.stage_data.time_value.cumulative_discount_factors(),
            stage_lag_transitions: &self.stage_data.stage_lag_transitions,
            noise_group_ids: &self.stage_data.noise_group_ids,
        }
    }

    /// Construct the training-phase [`TrainingContext`] borrowing from these inputs.
    pub(crate) fn training_ctx(&self) -> TrainingContext<'_> {
        let tr = &self.scenario_libraries.training;
        TrainingContext {
            horizon: &self.horizon,
            state: &self.stage_data.state,
            cut_state_layouts: &self.cut_state_layouts,
            study_dims: &self.stage_data.study_dims,
            inflow_method: &self.stage_data.study_dims.inflow_method,
            stochastic: &self.stochastic,
            initial_state: &self.initial.state,
            inflow_scheme: tr.inflow_scheme,
            load_scheme: tr.load_scheme,
            ncs_scheme: tr.ncs_scheme,
            stages: &self.stage_data.stages,
            historical_library: tr.historical.as_ref(),
            external_inflow_library: tr.external_inflow.as_ref(),
            external_load_library: tr.external_load.as_ref(),
            external_ncs_library: tr.external_ncs.as_ref(),
            lag_accum_seed: &self.initial.inflow_seeds.accum,
            lag_weight_seed: &self.initial.inflow_seeds.weight,
            dcs: self
                .cut_management
                .cut_selection
                .as_ref()
                .and_then(DcsParams::from_strategy),
            node_graph: &self.node_graph,
        }
    }

    /// Construct the simulation-phase [`TrainingContext`], with
    /// simulation-specific schemes and libraries falling back to the training
    /// phase's own when the simulation scheme reuses it.
    pub(crate) fn simulation_ctx(&self) -> TrainingContext<'_> {
        fn sim_or_training<'a, T>(
            sim_value: Option<&'a T>,
            scheme: SamplingScheme,
            matches_scheme: SamplingScheme,
            training_value: Option<&'a T>,
        ) -> Option<&'a T> {
            sim_value.or(if scheme == matches_scheme {
                training_value
            } else {
                None
            })
        }

        let tr = &self.scenario_libraries.training;
        let sim = &self.scenario_libraries.simulation;

        let historical_library = sim_or_training(
            sim.historical.as_ref(),
            sim.inflow_scheme,
            SamplingScheme::Historical,
            tr.historical.as_ref(),
        );
        let external_inflow_library = sim_or_training(
            sim.external_inflow.as_ref(),
            sim.inflow_scheme,
            SamplingScheme::External,
            tr.external_inflow.as_ref(),
        );
        let external_load_library = sim_or_training(
            sim.external_load.as_ref(),
            sim.load_scheme,
            SamplingScheme::External,
            tr.external_load.as_ref(),
        );
        let external_ncs_library = sim_or_training(
            sim.external_ncs.as_ref(),
            sim.ncs_scheme,
            SamplingScheme::External,
            tr.external_ncs.as_ref(),
        );

        TrainingContext {
            inflow_scheme: sim.inflow_scheme,
            load_scheme: sim.load_scheme,
            ncs_scheme: sim.ncs_scheme,
            historical_library,
            external_inflow_library,
            external_load_library,
            external_ncs_library,
            ..self.training_ctx()
        }
    }
}
