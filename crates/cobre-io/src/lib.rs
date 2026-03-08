//! # cobre-io
//!
//! Case directory loading, validation, and result writing for the
//! [Cobre](https://github.com/cobre-rs/cobre) power systems ecosystem.
//!
//! This crate provides two top-level entry points:
//!
//! - [`load_case`] — reads a case directory and produces a fully-validated
//!   [`cobre_core::System`] ready for use by the solver.
//! - [`write_results`] — accepts aggregate result types and writes all output
//!   artifacts to a specified root directory.
//!
//! ## Loading pipeline
//!
//! [`load_case`] executes a five-layer validation pipeline:
//!
//! 1. **Structural validation** — checks that required files exist on disk and records
//!    which optional files are present ([`validation::structural`]).
//! 2. **Schema validation** — verifies required fields, types, and value ranges.
//! 3. **Referential integrity** — checks all entity ID cross-references are resolvable.
//! 4. **Dimensional consistency** — cross-file coverage checks (e.g., inflow params
//!    cover all hydros).
//! 5. **Semantic validation** — domain business rules (acyclic cascade, penalty ordering,
//!    PAR stationarity, etc.).
//!
//! All validation diagnostics are collected by [`validation::ValidationContext`] before
//! failing, so users see every problem in a single report.  Final errors are reported
//! via [`LoadError`], which carries enough context for diagnostic messages without
//! re-reading input files.
//!
//! ## Status
//!
//! This crate is in early development. The API **will** change.
//!
//! See the [repository](https://github.com/cobre-rs/cobre) for the full roadmap.

pub mod broadcast;
pub mod config;
pub mod constraints;
pub mod error;
pub mod extensions;
pub mod initial_conditions;
pub mod output;
pub(crate) mod parquet_helpers;
pub mod penalties;
pub(crate) mod pipeline;
pub mod report;
pub mod resolution;
pub mod scenarios;
pub mod stages;
pub mod system;
pub mod validation;

pub use broadcast::{deserialize_system, serialize_system};
pub use config::{parse_config, Config};
pub use constraints::{
    load_contract_bounds, load_exchange_factors, load_generic_constraint_bounds,
    load_generic_constraints, load_hydro_bounds, load_line_bounds, load_penalty_overrides_bus,
    load_penalty_overrides_hydro, load_penalty_overrides_line, load_penalty_overrides_ncs,
    load_pumping_bounds, load_thermal_bounds, parse_contract_bounds, parse_exchange_factors,
    parse_generic_constraint_bounds, parse_generic_constraints, parse_hydro_bounds,
    parse_line_bounds, parse_penalty_overrides_bus, parse_penalty_overrides_hydro,
    parse_penalty_overrides_line, parse_penalty_overrides_ncs, parse_pumping_bounds,
    parse_thermal_bounds, BlockExchangeFactor, BusPenaltyOverrideRow, ContractBoundsRow,
    ExchangeFactorEntry, GenericConstraintBoundsRow, HydroBoundsRow, HydroPenaltyOverrideRow,
    LineBoundsRow, LinePenaltyOverrideRow, NcsPenaltyOverrideRow, PumpingBoundsRow,
    ThermalBoundsRow,
};
pub use error::LoadError;
pub use extensions::{
    load_fpha_hyperplanes, load_production_models, parse_fpha_hyperplanes, parse_hydro_geometry,
    parse_production_models, FittingWindow, FphaConfig, FphaHyperplaneRow, HydroGeometryRow,
    ProductionModelConfig, SeasonConfig, SelectionMode, StageRange,
};
pub use initial_conditions::parse_initial_conditions;
pub use output::{
    write_results, CutStatistics, IterationRecord, OutputError, ParquetWriterConfig,
    SimulationOutput, TrainingOutput,
};
pub use penalties::parse_penalties;
pub use report::{generate_report, ReportEntry, ValidationReport};
pub use resolution::{resolve_bounds, resolve_penalties};
pub use scenarios::{
    assemble_inflow_models, assemble_load_models, load_correlation, load_external_scenarios,
    load_inflow_ar_coefficients, load_inflow_history, load_inflow_seasonal_stats,
    load_load_factors, load_load_seasonal_stats, load_scenarios, parse_correlation,
    parse_external_scenarios, parse_inflow_ar_coefficients, parse_inflow_history,
    parse_inflow_seasonal_stats, parse_load_factors, parse_load_seasonal_stats, BlockFactor,
    ExternalScenarioRow, InflowArCoefficientRow, InflowHistoryRow, InflowSeasonalStatsRow,
    LoadFactorEntry, LoadSeasonalStatsRow, ScenarioData,
};
pub use stages::{parse_stages, StagesData};
pub use system::{
    load_energy_contracts, load_non_controllable_sources, load_pumping_stations, parse_buses,
    parse_energy_contracts, parse_hydros, parse_lines, parse_non_controllable_sources,
    parse_pumping_stations, parse_thermals,
};
pub use validation::structural::{validate_structure, FileManifest};
pub use validation::{ErrorKind, Severity, ValidationContext, ValidationEntry};

use cobre_core::System;
use std::path::Path;

/// Load a case directory and return a fully-validated [`System`].
///
/// `path` must point to the root case directory containing `config.json` and the
/// standard subdirectories (`system/`, `scenarios/`, `constraints/`, `policy/`).
///
/// The function executes a five-layer validation pipeline:
///
/// 1. **Structural** — all required files exist on disk.
/// 2. **Schema** — required fields, types, and value ranges are valid.
/// 3. **Referential integrity** — all cross-entity ID references resolve.
/// 4. **Dimensional consistency** — entity coverage across optional files.
/// 5. **Semantic** — domain business rules (acyclic cascade, penalty ordering, etc.).
///
/// After all layers pass, three-tier penalty/bound resolution and scenario assembly
/// are performed before constructing the [`System`].
///
/// # Errors
///
/// - [`LoadError::IoError`] — a required file is missing or cannot be read.
/// - [`LoadError::ParseError`] — a file contains malformed JSON or invalid Parquet.
/// - [`LoadError::SchemaError`] — a domain constraint violation detected
///   post-deserialization (e.g., AR coefficient count mismatch).
/// - [`LoadError::ConstraintError`] — one or more validation errors collected
///   across Layers 1-5, or `SystemBuilder` rejected the assembled data.
pub fn load_case(path: &Path) -> Result<System, LoadError> {
    pipeline::run_pipeline(path)
}
