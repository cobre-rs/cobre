//! Simulation types for the SDDP policy evaluation phase.
//!
//! The simulation phase evaluates the trained SDDP policy on a large number
//! of scenarios and streams per-scenario results through a bounded channel to
//! a background I/O thread for Parquet output writing.

pub mod aggregation;
pub mod config;
pub mod error;
pub mod extraction;
pub mod pipeline;
pub mod types;

pub use aggregation::aggregate_simulation;
pub use config::SimulationConfig;
pub use error::SimulationError;
pub use extraction::{
    EntityCounts, SolutionView, StageExtractionSpec, accumulate_category_costs, assign_scenarios,
    extract_stage_result,
};
pub use pipeline::{SimulationOutputSpec, simulate};
pub use types::{
    CategoryCostStats, ScenarioCategoryCosts, SimulationBusResult, SimulationContractResult,
    SimulationCostResult, SimulationExchangeResult, SimulationGenericViolationResult,
    SimulationHydroResult, SimulationInflowLagResult, SimulationNonControllableResult,
    SimulationPumpingResult, SimulationScenarioResult, SimulationStageResult, SimulationSummary,
    SimulationThermalResult, StageSummaryStats,
};
