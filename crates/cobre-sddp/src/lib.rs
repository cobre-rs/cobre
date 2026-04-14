//! # cobre-sddp
//!
//! Stochastic Dual Dynamic Programming for the [Cobre](https://github.com/cobre-rs/cobre) power systems ecosystem.
//!
//! This crate implements the SDDP algorithm (Pereira & Pinto, 1991) for
//! long-term hydrothermal dispatch and energy planning:
//!
//! - **Forward pass**: scenario-based simulation with policy evaluation.
//! - **Backward pass**: Benders cut generation for the cost-to-go function.
//! - **Cut management**: single-cut and multi-cut strategies, cut selection,
//!   and dominance pruning.
//! - **Risk measures**: expected value, `CVaR`, and convex combinations.
//! - **Convergence**: statistical stopping criteria and bound gap monitoring.
//! - **Parallelism**: designed for hybrid MPI + thread-level parallelism
//!   via [ferrompi](https://github.com/cobre-rs/ferrompi).
//!
//! Built on `cobre-core` for system data, `cobre-stochastic` for inflow
//! modeling, and `cobre-solver` for LP subproblems.
//!
//! ## Status
//!
//! This crate is in early development. The API **will** change.
//!
//! See the [repository](https://github.com/cobre-rs/cobre) for the current status.

// Relax strict production lints for test builds (normal in test contexts).
#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::float_cmp,
    )
)]

pub mod angular_pruning;
pub mod backward;
pub mod basis_padding;
pub mod config;
pub mod context;
pub mod convergence;
pub mod conversion;
pub mod cut;
pub mod cut_selection;
pub mod cut_sync;
pub mod error;
pub mod estimation;
pub mod forward;
pub(crate) mod fpha_fitting;
pub mod generic_constraints;
pub mod horizon_mode;
pub mod hydro_models;
pub mod indexer;
pub mod inflow_method;
pub(crate) mod lag_transition;
pub mod lower_bound;
pub mod lp_builder;
pub(crate) mod noise;
pub mod policy_export;
pub mod policy_load;
pub mod provenance;
pub mod risk_measure;
pub mod scaling_report;
pub mod setup;
pub mod simulation;
pub mod solver_stats;
pub mod state_exchange;
pub mod stochastic_summary;
pub mod stopping_rule;
pub mod training;
pub mod training_output;
pub mod trajectory;
pub mod visited_states;
pub mod workspace;

pub use angular_pruning::{
    AngularPruningParams, AngularPruningResult, parse_angular_pruning_config,
    select_angular_dominated,
};
pub use backward::{BackwardPassSpec, BackwardResult, run_backward_pass};
pub use config::TrainingConfig;
pub use context::{StageContext, TrainingContext};
pub use convergence::ConvergenceMonitor;
pub use cut::{CutPool, FutureCostFunction, WARM_START_ITERATION};
pub use cut_selection::{
    CutMetadata, CutSelectionStrategy, DeactivationSet, parse_cut_selection_config,
};
pub use cut_sync::CutSyncBuffers;
pub use error::SddpError;
pub use estimation::{
    EstimationError, EstimationPath, EstimationReport, LagScaleWarning, StdRatioDivergence,
};
pub use forward::{ForwardResult, SyncResult, run_forward_pass, sync_forward};
pub use horizon_mode::HorizonMode;
pub use hydro_models::{
    EvaporationModel, EvaporationModelSet, EvaporationReferenceSource, EvaporationSource,
    FphaHydroDetail, FphaPlane, HydroModelProvenance, HydroModelSummary, LinearizedEvaporation,
    PrepareHydroModelsResult, ProductionModelSet, ProductionModelSource, ResolvedProductionModel,
    build_hydro_model_summary, prepare_hydro_models, resolve_evaporation_models,
    resolve_production_models,
};
pub use indexer::{
    EquipmentCounts, EvapConfig, EvaporationIndices, FphaColumnLayout, FphaRowRange, StageIndexer,
};
pub use inflow_method::InflowNonNegativityMethod;
pub use lower_bound::{LbEvalSpec, evaluate_lower_bound};
pub use lp_builder::{
    GenericConstraintRowEntry, PatchBuffer, StageTemplates, ar_dynamics_row_offset,
    build_stage_templates,
};
pub use policy_load::{
    build_basis_cache_from_checkpoint, resolve_warm_start_counts, validate_policy_compatibility,
};
pub use provenance::{ModelProvenanceReport, ProvenanceSource, build_provenance_report};
pub use risk_measure::{BackwardOutcome, RiskMeasure};
pub use scaling_report::ScalingReport;
pub use setup::{
    DEFAULT_FORWARD_PASSES, DEFAULT_MAX_ITERATIONS, DEFAULT_SEED, PrepareStochasticResult,
    StudyParams, StudySetup, prepare_stochastic,
};
pub use simulation::{
    CategoryCostStats, EntityCounts, ScenarioCategoryCosts, SimulationBusResult, SimulationConfig,
    SimulationContractResult, SimulationCostResult, SimulationError, SimulationExchangeResult,
    SimulationGenericViolationResult, SimulationHydroResult, SimulationInflowLagResult,
    SimulationNonControllableResult, SimulationOutputSpec, SimulationPumpingResult,
    SimulationRunResult, SimulationScenarioResult, SimulationStageResult, SimulationSummary,
    SimulationThermalResult, StageSummaryStats, accumulate_category_costs, aggregate_simulation,
    assign_scenarios, extract_stage_result, simulate,
};
pub use solver_stats::{
    SCENARIO_STATS_STRIDE, SOLVER_STATS_DELTA_SCALAR_FIELDS, SolverStatsDelta, SolverStatsEntry,
    aggregate_solver_statistics, pack_delta_scalars, pack_scenario_stats, unpack_delta_scalars,
    unpack_scenario_stats,
};
pub use state_exchange::ExchangeBuffers;
pub use stochastic_summary::{
    ArOrderSummary, StochasticSource, StochasticSummary, build_stochastic_summary,
    estimation_report_to_fitting_report, inflow_models_to_ar_rows, inflow_models_to_stats_rows,
};
pub use stopping_rule::{MonitorState, StoppingMode, StoppingRule, StoppingRuleSet};
pub use training::{TrainingOutcome, TrainingResult, train};
pub use training_output::build_training_output;
pub use trajectory::TrajectoryRecord;
pub use visited_states::VisitedStatesArchive;
pub use workspace::{BasisStore, BasisStoreSliceMut, SolverWorkspace, WorkspacePool};
