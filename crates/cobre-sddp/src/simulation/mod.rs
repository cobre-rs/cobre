//! Simulation types for the SDDP policy evaluation phase.
//!
//! The simulation phase evaluates the trained SDDP policy on a large number
//! of scenarios and streams per-scenario results through a bounded channel to
//! a background I/O thread for Parquet output writing.

pub mod config;
pub mod error;
pub mod types;

pub use config::SimulationConfig;
pub use error::SimulationError;
pub use types::{
    CategoryCostStats, ScenarioCategoryCosts, SimulationBusResult, SimulationContractResult,
    SimulationCostResult, SimulationExchangeResult, SimulationGenericViolationResult,
    SimulationHydroResult, SimulationInflowLagResult, SimulationNonControllableResult,
    SimulationPumpingResult, SimulationScenarioResult, SimulationStageResult, SimulationSummary,
    SimulationThermalResult, StageSummaryStats,
};
