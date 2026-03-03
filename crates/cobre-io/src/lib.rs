//! # cobre-io
//!
//! Case directory loading and validation for the [Cobre](https://github.com/cobre-rs/cobre)
//! power systems ecosystem.
//!
//! This crate provides the [`load_case`] function, which reads a case directory and
//! produces a fully-validated [`cobre_core::System`] ready for use by the solver.
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

pub mod config;
pub mod error;
pub mod extensions;
pub mod initial_conditions;
pub mod penalties;
pub mod system;
pub mod validation;

pub use config::{Config, parse_config};
pub use error::LoadError;
pub use extensions::{
    FittingWindow, FphaConfig, FphaHyperplaneRow, HydroGeometryRow, ProductionModelConfig,
    SeasonConfig, SelectionMode, StageRange, load_fpha_hyperplanes, load_production_models,
    parse_fpha_hyperplanes, parse_hydro_geometry, parse_production_models,
};
pub use initial_conditions::parse_initial_conditions;
pub use penalties::parse_penalties;
pub use system::{
    load_energy_contracts, load_non_controllable_sources, load_pumping_stations, parse_buses,
    parse_energy_contracts, parse_hydros, parse_lines, parse_non_controllable_sources,
    parse_pumping_stations, parse_thermals,
};
pub use validation::structural::{FileManifest, validate_structure};
pub use validation::{ErrorKind, Severity, ValidationContext, ValidationEntry};

use cobre_core::System;
use std::path::Path;

/// Load a case directory and return a fully-validated [`System`].
///
/// `path` must point to the root case directory containing `config.json` and the
/// standard subdirectories (`system/`, `scenarios/`, `constraints/`, `policy/`).
///
/// # Errors
///
/// Returns [`LoadError`] if any file cannot be read, parsed, or validated.
///
/// # Panics
///
/// This function is not yet implemented.
pub fn load_case(path: &Path) -> Result<System, LoadError> {
    let _ = path;
    todo!("load_case is not yet implemented")
}
