//! Context structs for reducing parameter count in hot-path functions.

use cobre_core::{Stage, scenario::SamplingScheme, temporal::StageLagTransition};
use cobre_solver::StageTemplate;
use cobre_stochastic::par::resolve_stage_lag_transition;
use cobre_stochastic::{ExternalScenarioLibrary, HistoricalScenarioLibrary, StochasticContext};

use crate::{
    dcs::DcsParams,
    horizon_mode::HorizonMode,
    inflow_method::InflowNonNegativityMethod,
    lp::builder::{StageGeometry, StateBox},
    lp::indexer::{CutStateProjection, StateSpace, StudyDimensions},
    setup::node_graph::{NodeGraph, StageIdx},
};

/// Immutable per-stage LP layout parameters.
///
/// Read-only parameters shared by the forward pass, backward pass, and
/// simulation pipeline. Slice fields are indexed by study stage `t` unless
/// noted; per-stage NCS/anticipated slices are empty when the study lacks that
/// entity class.
pub struct StageContext<'a> {
    /// Stage LP templates.
    pub templates: &'a [StageTemplate],
    /// Per-stage admissible box for the outgoing state vector.
    pub state_boxes: &'a [StateBox],
    /// Per-stage equipment geometry: `geometry_per_stage[t]` holds stage `t`'s
    /// column and row ranges; a single global stage-0 geometry would carry
    /// `n_blks`-striped bases that misread any stage with a differing block count.
    pub geometry_per_stage: &'a [StageGeometry],
    /// Resolved objective cost-scale factor (`modeling.cost_scale_factor`),
    /// mirroring [`StageTemplates::cost_scale_factor`](crate::lp::builder::StageTemplates::cost_scale_factor).
    /// Multiplies a scaled-objective quantity back to currency units at the
    /// stage-cost / immediate-cost reporting boundary.
    pub cost_scale_factor: f64,
    /// Bus indices for stochastic load mapping.
    pub load_bus_indices: &'a [usize],
    /// Stage-invariant stochastic-slot → dense NCS column index map, id-sorted in
    /// `StochasticContext::ncs_entity_ids` order — the order `transform_ncs_noise`
    /// emits its bound buffers. Length equals `n_stochastic_ncs`.
    pub ncs_stochastic_dense_col: &'a [usize],
    /// Stage-invariant commissioning window `(entry, exit)` per stochastic NCS
    /// slot, id-sorted to match [`Self::ncs_stochastic_dense_col`]; length equals
    /// `n_stochastic_ncs`. The forward, backward, and lower-bound patch sites force
    /// a dormant slot's cap to `[0, 0]` identically — the "patch NCS identically"
    /// contract; a divergence understates the bound (D15).
    pub ncs_stochastic_windows: &'a [(Option<i32>, Option<i32>)],
    /// Stage-invariant commissioning window `(entry, exit)` per anticipated
    /// thermal, in anticipated-local order (matching
    /// `StudyDimensions::anticipated_plants`). The simulation
    /// anticipated-decision read gates on the DELIVERY stage's `stage.id`, the same
    /// predicate the LP builder uses — never the decision stage.
    pub anticipated_windows: &'a [(Option<i32>, Option<i32>)],
    /// `study_stage_ids[t] = stage.id`. The anticipated decision gate maps the
    /// delivery index `t + K_i` to the DELIVERY stage's `stage.id` (not the index)
    /// through this slice.
    pub study_stage_ids: &'a [i32],
    /// Maximum generation (MW) per stochastic NCS entity, sorted by entity ID.
    pub ncs_max_gen: &'a [f64],
    /// Per-stochastic-NCS curtailment policy, aligned 1:1 with
    /// [`Self::ncs_max_gen`]. `true` = dispatchable in `[0, cap]`; `false` =
    /// must-run, pinned `col_lower = col_upper = cap` for every scenario.
    pub ncs_allow_curtailment: &'a [bool],
    /// One-step discount factor for the transition departing each stage:
    /// `1 / (1 + r)^(Dt / 365.25)` for annual rate `r` and stage duration `Dt`
    /// in days. All `1.0` when no discount.
    pub discount_factors: &'a [f64],
    /// Cumulative discount factor for present-value costing:
    /// `cumulative_discount_factors[t]` is the product of all one-step factors
    /// for transitions preceding stage `t`. `[0] == 1.0` always.
    pub cumulative_discount_factors: &'a [f64],
    /// Precomputed per-stage lag accumulation weights and period-finalization
    /// flags.
    pub stage_lag_transitions: &'a [StageLagTransition],
    /// Noise group IDs for noise-group sharing, indexed by stage array index.
    /// Stages sharing a group ID share a noise draw in the opening tree and
    /// forward pass; uniform monthly studies give each stage a unique ID (no
    /// sharing).
    pub noise_group_ids: &'a [u32],
}

impl StageContext<'_> {
    /// Returns the noise group ID for stage index `t`.
    #[inline]
    #[must_use]
    pub fn noise_group_id_at(&self, t: StageIdx) -> u32 {
        if self.noise_group_ids.is_empty() {
            #[allow(clippy::cast_possible_truncation)]
            return t.0 as u32;
        }
        debug_assert!(
            t.0 < self.noise_group_ids.len(),
            "stage index {t} out of bounds for noise_group_ids (len={})",
            self.noise_group_ids.len()
        );
        self.noise_group_ids[t.0]
    }

    /// Stage `t`'s LP template.
    #[inline]
    #[must_use]
    pub fn template(&self, t: StageIdx) -> &StageTemplate {
        &self.templates[t.0]
    }

    /// Stage `t`'s admissible box for the outgoing state vector.
    #[inline]
    #[must_use]
    pub fn state_box(&self, t: StageIdx) -> &StateBox {
        &self.state_boxes[t.0]
    }

    /// Blocks at stage `t`.
    #[inline]
    #[must_use]
    pub fn block_count(&self, t: StageIdx) -> usize {
        self.geometry_per_stage[t.0].n_blks
    }

    /// One-step discount factor for the transition departing stage `t`, or
    /// `1.0` when `discount_factors` is unpopulated.
    #[inline]
    #[must_use]
    pub fn discount_factor(&self, t: StageIdx) -> f64 {
        self.discount_factors.get(t.0).copied().unwrap_or(1.0)
    }

    /// Cumulative discount factor for present-value costing at stage `t`, or
    /// `1.0` when `cumulative_discount_factors` is unpopulated.
    #[inline]
    #[must_use]
    pub fn cumulative_discount_factor(&self, t: StageIdx) -> f64 {
        self.cumulative_discount_factors
            .get(t.0)
            .copied()
            .unwrap_or(1.0)
    }

    /// Declared study `stage_id` at position `t`, or `None` if `study_stage_ids`
    /// is shorter than the study (never in production).
    #[inline]
    #[must_use]
    pub fn study_stage_id(&self, t: StageIdx) -> Option<i32> {
        self.study_stage_ids.get(t.0).copied()
    }

    /// Precomputed lag accumulation weights and period-finalization flags for
    /// stage `t`.
    #[inline]
    #[must_use]
    pub fn stage_lag(&self, t: StageIdx) -> StageLagTransition {
        resolve_stage_lag_transition(self.stage_lag_transitions, t.0)
    }
}

/// Immutable algorithm-level configuration for the training loop.
///
/// Read-only parameters shared by the training loop, forward pass, backward
/// pass, and simulation pipeline. The external-scenario `Option` libraries are
/// `Some` exactly when their entity class's `SamplingScheme` selects them.
pub struct TrainingContext<'a> {
    /// Horizon mode determining stage count.
    pub horizon: &'a HorizonMode,
    /// Single owner of the state-vector layout: the state column ranges,
    /// `n_state`, the resolvers, and the mask. Every hot-path state-column read
    /// resolves through this handle.
    pub state: &'a StateSpace,
    /// Per-pool cut-state projection, indexed by pool id (paired 1:1 with
    /// `FutureCostFunction::pools`), resolved from the node graph's
    /// `node → pool` map. The backward pass reads `cut_state_layouts[pool_id]`
    /// when solving a node's successor to size that node's own pool's
    /// extracted subgradient and every per-stage backward buffer. On the
    /// chain degeneracy `pool_id == stage`. Empty on the non-training paths
    /// (simulation, lower-bound eval), which never extract cuts.
    pub cut_state_layouts: &'a [CutStateProjection],
    /// Single owner of the study-invariant, non-state LP shape.
    pub study_dims: &'a StudyDimensions,
    /// Inflow non-negativity enforcement strategy.
    pub inflow_method: &'a InflowNonNegativityMethod,
    /// Stochastic context providing noise generation and PAR model.
    pub stochastic: &'a StochasticContext,
    /// Initial state vector for stage 0.
    pub initial_state: &'a [f64],
    /// Forward-pass noise source scheme for the inflow entity class.
    pub inflow_scheme: SamplingScheme,
    /// Forward-pass noise source scheme for the load entity class.
    pub load_scheme: SamplingScheme,
    /// Forward-pass noise source scheme for the NCS entity class.
    pub ncs_scheme: SamplingScheme,
    /// Study stages (id >= 0) in index order; required by [`cobre_stochastic::build_forward_sampler`].
    pub stages: &'a [Stage],
    /// Pre-standardized historical inflow windows library.
    pub historical_library: Option<&'a HistoricalScenarioLibrary>,
    /// Pre-standardized external inflow scenario library.
    pub external_inflow_library: Option<&'a ExternalScenarioLibrary>,
    /// Pre-standardized external load scenario library.
    pub external_load_library: Option<&'a ExternalScenarioLibrary>,
    /// Pre-standardized external NCS scenario library.
    pub external_ncs_library: Option<&'a ExternalScenarioLibrary>,
    /// Per-hydro derived in-progress accumulator seed
    /// (`DerivedInflowSeeds::accum`), copied into `ws.scratch.lag_accumulator`
    /// at every trajectory start. Empty slice when the derivation has no data
    /// (the zero-fill path is taken instead).
    pub lag_accum_seed: &'a [f64],
    /// Per-hydro coverage-fraction seed (`DerivedInflowSeeds::weight`),
    /// copied into `ws.scratch.lag_weight_accum` at every trajectory start;
    /// length matches [`Self::lag_accum_seed`].
    pub lag_weight_seed: &'a [f64],
    /// Dynamic Cut Selection hyperparameters, `Some` only when the dynamic
    /// cut-selection method is configured. When `Some` and the iteration is at or
    /// past `start_iteration`, the backward pass solves each stage LP lazily;
    /// otherwise the frozen all-cuts path is used.
    pub dcs: Option<DcsParams>,
    /// The runtime node graph: node identity/order, the `node → pool`
    /// map, and per-node Ω views/out-edges. Absent `nodes[]` this is the
    /// byte-exact chain degeneracy (one node per stage). Discount is NOT
    /// carried here — it stays per-stage on
    /// [`StageContext::cumulative_discount_factors`], reached through a
    /// node's own `stage` field.
    pub node_graph: &'a NodeGraph,
}
