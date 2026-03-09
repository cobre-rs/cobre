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
//! See the [repository](https://github.com/cobre-rs/cobre) for the full roadmap.

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

pub mod backward;
pub mod config;
pub mod convergence;
pub mod cut;
pub mod cut_selection;
pub mod cut_sync;
pub mod error;
pub mod forward;
pub mod horizon_mode;
pub mod indexer;
pub mod lower_bound;
pub mod lp_builder;
pub mod risk_measure;
pub mod simulation;
pub mod state_exchange;
pub mod stopping_rule;
pub mod training;
pub mod training_output;
pub mod trajectory;
pub mod workspace;

pub use backward::{run_backward_pass, BackwardResult};
pub use config::TrainingConfig;
pub use convergence::ConvergenceMonitor;
pub use cut::{CutPool, FutureCostFunction};
pub use cut_selection::{CutMetadata, CutSelectionStrategy, DeactivationSet};
pub use cut_sync::CutSyncBuffers;
pub use error::SddpError;
pub use forward::{run_forward_pass, sync_forward, ForwardResult, SyncResult};
pub use horizon_mode::HorizonMode;
pub use indexer::StageIndexer;
pub use lower_bound::evaluate_lower_bound;
pub use lp_builder::{ar_dynamics_row_offset, build_stage_templates, PatchBuffer, StageTemplates};
pub use risk_measure::{BackwardOutcome, RiskMeasure};
pub use simulation::{
    accumulate_category_costs, aggregate_simulation, assign_scenarios, extract_stage_result,
    simulate, CategoryCostStats, EntityCounts, ScenarioCategoryCosts, SimulationBusResult,
    SimulationConfig, SimulationContractResult, SimulationCostResult, SimulationError,
    SimulationExchangeResult, SimulationGenericViolationResult, SimulationHydroResult,
    SimulationInflowLagResult, SimulationNonControllableResult, SimulationPumpingResult,
    SimulationScenarioResult, SimulationStageResult, SimulationSummary, SimulationThermalResult,
    StageSummaryStats,
};
pub use state_exchange::ExchangeBuffers;
pub use stopping_rule::{MonitorState, StoppingMode, StoppingRule, StoppingRuleSet};
pub use training::{train, TrainingResult};
pub use training_output::build_training_output;
pub use trajectory::TrajectoryRecord;
pub use workspace::{BasisStore, BasisStoreSliceMut, SolverWorkspace, WorkspacePool};
