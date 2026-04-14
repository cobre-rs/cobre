//! Context structs for reducing parameter count in hot-path functions.

use cobre_core::{Stage, scenario::SamplingScheme, temporal::StageLagTransition};
use cobre_solver::StageTemplate;
use cobre_stochastic::{ExternalScenarioLibrary, HistoricalScenarioLibrary, StochasticContext};

use crate::{HorizonMode, InflowNonNegativityMethod, StageIndexer};

/// Immutable per-stage LP layout and noise scaling parameters.
///
/// Groups the read-only parameters shared by the forward pass, backward pass,
/// and simulation pipeline. Constructed once at algorithm startup and passed
/// by reference to all hot-path functions.
pub struct StageContext<'a> {
    /// Stage LP templates (one per study stage).
    pub templates: &'a [StageTemplate],
    /// Row index of the first water-balance row in each stage template.
    pub base_rows: &'a [usize],
    /// Noise scaling factors, layout: `[stage * n_hydros + hydro]`.
    pub noise_scale: &'a [f64],
    /// Number of hydro plants with LP variables.
    pub n_hydros: usize,
    /// Number of buses with stochastic load noise.
    pub n_load_buses: usize,
    /// Row index of the first load-balance row in each stage template.
    pub load_balance_row_starts: &'a [usize],
    /// Bus indices for stochastic load mapping.
    pub load_bus_indices: &'a [usize],
    /// Number of blocks per stage.
    pub block_counts_per_stage: &'a [usize],
    /// Maximum generation (MW) per stochastic NCS entity, sorted by entity ID.
    /// Length equals the number of stochastic NCS entities. Empty when none exist.
    pub ncs_max_gen: &'a [f64],
    /// One-step discount factor for the transition departing each stage.
    ///
    /// `discount_factors[t] = 1 / (1 + r)^(Dt / 365.25)` where `r` is the
    /// annual discount rate and `Dt` is the stage duration in days.
    /// Length equals the number of study stages. All `1.0` when no discount.
    pub discount_factors: &'a [f64],
    /// Cumulative discount factor at each stage for present-value costing.
    ///
    /// `cumulative_discount_factors[t]` is the product of all one-step discount
    /// factors for transitions preceding stage `t`. `[0] == 1.0` always.
    /// Length equals the number of study stages.
    pub cumulative_discount_factors: &'a [f64],
    /// Precomputed lag accumulation weights and period-finalization flags, one
    /// entry per study stage. Indexed by stage: `stage_lag_transitions[t]`.
    ///
    /// Populated by [`crate::lag_transition::precompute_stage_lag_transitions`]
    /// at setup time. Used by the forward pass and simulation pipeline starting
    /// in Epic 2 ticket-006.
    #[allow(dead_code)]
    pub stage_lag_transitions: &'a [StageLagTransition],
}

/// Immutable algorithm-level configuration for the training loop.
///
/// Groups the read-only parameters shared by the training loop, forward pass,
/// backward pass, and simulation pipeline. Constructed once by the caller of
/// [`train`](crate::train) and passed by reference to all pass functions.
pub struct TrainingContext<'a> {
    /// Horizon mode (finite/infinite) determining stage count.
    pub horizon: &'a HorizonMode,
    /// Stage indexer providing state layout and entity counts.
    pub indexer: &'a StageIndexer,
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
    ///
    /// `Some` when `inflow_scheme == SamplingScheme::Historical`, `None` otherwise.
    pub historical_library: Option<&'a HistoricalScenarioLibrary>,
    /// Pre-standardized external inflow scenario library.
    ///
    /// `Some` when `inflow_scheme == SamplingScheme::External`, `None` otherwise.
    pub external_inflow_library: Option<&'a ExternalScenarioLibrary>,
    /// Pre-standardized external load scenario library.
    ///
    /// `Some` when `load_scheme == SamplingScheme::External`, `None` otherwise.
    pub external_load_library: Option<&'a ExternalScenarioLibrary>,
    /// Pre-standardized external NCS scenario library.
    ///
    /// `Some` when `ncs_scheme == SamplingScheme::External`, `None` otherwise.
    pub external_ncs_library: Option<&'a ExternalScenarioLibrary>,
    /// Whether basis-aware warm-start padding is enabled (config-gated).
    pub basis_padding_enabled: bool,
}
