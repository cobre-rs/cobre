//! Entity registry parsers for the `system/` subdirectory.
//!
//! Each sub-module implements a `parse_*` function that reads one entity registry
//! file from the case `system/` directory, validates it, and returns a sorted
//! `Vec` of core entity types.
//!
//! ## Parsing convention
//!
//! All parsers follow the canonical four-step pipeline:
//!
//! 1. `fs::read_to_string` — read the file into a string.
//! 2. `serde_json::from_str` — deserialize into a `Raw*` intermediate type.
//! 3. `validate_raw(&raw, path)?` — post-deserialization invariant checks.
//! 4. `Ok(convert(raw))` — convert to core types and sort by ID.
//!
//! Cross-reference validation (e.g., checking that `bus_id` exists in the bus
//! registry) is deferred to Layer 3 (Epic 06). Only schema-level invariants
//! are checked here.
//!
//! ## Optional files
//!
//! Three entity registries are optional: `non_controllable_sources.json`,
//! `pumping_stations.json`, and `energy_contracts.json`. When the file is absent
//! from the case directory the corresponding `load_*` wrapper (which accepts
//! `Option<&Path>`) returns `Ok(Vec::new())` without error.

pub mod buses;
pub mod energy_contracts;
pub mod hydros;
pub mod lines;
pub mod non_controllable;
pub mod pumping_stations;
pub mod thermals;

pub use buses::parse_buses;
pub use energy_contracts::parse_energy_contracts;
pub use hydros::parse_hydros;
pub use lines::parse_lines;
pub use non_controllable::parse_non_controllable_sources;
pub use pumping_stations::parse_pumping_stations;
pub use thermals::parse_thermals;

use cobre_core::{
    entities::{EnergyContract, NonControllableSource, PumpingStation},
    penalty::GlobalPenaltyDefaults,
};
use std::path::Path;

use crate::LoadError;

/// Load `system/non_controllable_sources.json`, or return an empty vec when absent.
///
/// When `path` is `None` (the file is not present in the case directory), returns
/// `Ok(Vec::new())` without error. When `path` is `Some`, delegates to
/// [`parse_non_controllable_sources`].
///
/// # Errors
///
/// Propagates errors from [`parse_non_controllable_sources`] when `path` is `Some`.
pub fn load_non_controllable_sources(
    path: Option<&Path>,
    global_penalties: &GlobalPenaltyDefaults,
) -> Result<Vec<NonControllableSource>, LoadError> {
    match path {
        Some(p) => non_controllable::parse_non_controllable_sources(p, global_penalties),
        None => Ok(Vec::new()),
    }
}

/// Load `system/pumping_stations.json`, or return an empty vec when absent.
///
/// When `path` is `None` (the file is not present in the case directory), returns
/// `Ok(Vec::new())` without error. When `path` is `Some`, delegates to
/// [`parse_pumping_stations`].
///
/// # Errors
///
/// Propagates errors from [`parse_pumping_stations`] when `path` is `Some`.
pub fn load_pumping_stations(path: Option<&Path>) -> Result<Vec<PumpingStation>, LoadError> {
    match path {
        Some(p) => pumping_stations::parse_pumping_stations(p),
        None => Ok(Vec::new()),
    }
}

/// Load `system/energy_contracts.json`, or return an empty vec when absent.
///
/// When `path` is `None` (the file is not present in the case directory), returns
/// `Ok(Vec::new())` without error. When `path` is `Some`, delegates to
/// [`parse_energy_contracts`].
///
/// # Errors
///
/// Propagates errors from [`parse_energy_contracts`] when `path` is `Some`.
pub fn load_energy_contracts(path: Option<&Path>) -> Result<Vec<EnergyContract>, LoadError> {
    match path {
        Some(p) => energy_contracts::parse_energy_contracts(p),
        None => Ok(Vec::new()),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use cobre_core::entities::{DeficitSegment, HydroPenalties};

    /// Build a canonical `GlobalPenaltyDefaults` for test use.
    fn make_global() -> GlobalPenaltyDefaults {
        GlobalPenaltyDefaults {
            bus_deficit_segments: vec![
                DeficitSegment {
                    depth_mw: Some(500.0),
                    cost_per_mwh: 1000.0,
                },
                DeficitSegment {
                    depth_mw: None,
                    cost_per_mwh: 5000.0,
                },
            ],
            bus_excess_cost: 100.0,
            line_exchange_cost: 2.0,
            hydro: HydroPenalties {
                spillage_cost: 0.01,
                fpha_turbined_cost: 0.05,
                diversion_cost: 0.1,
                storage_violation_below_cost: 10_000.0,
                filling_target_violation_cost: 50_000.0,
                turbined_violation_below_cost: 500.0,
                outflow_violation_below_cost: 500.0,
                outflow_violation_above_cost: 500.0,
                generation_violation_below_cost: 1_000.0,
                evaporation_violation_cost: 5_000.0,
                water_withdrawal_violation_cost: 1_000.0,
                water_withdrawal_violation_pos_cost: 1_000.0,
                water_withdrawal_violation_neg_cost: 1_000.0,
                evaporation_violation_pos_cost: 5_000.0,
                evaporation_violation_neg_cost: 5_000.0,
                inflow_nonnegativity_cost: 1000.0,
            },
            ncs_curtailment_cost: 0.005,
        }
    }

    // ── AC: optional file wrapper returns empty vec for None ──────────────────

    /// `load_non_controllable_sources(None, ...)` returns `Ok(Vec::new())`.
    #[test]
    fn test_load_ncs_none_returns_empty() {
        let global = make_global();
        let result = load_non_controllable_sources(None, &global).unwrap();
        assert!(
            result.is_empty(),
            "expected empty vec when path is None, got {result:?}"
        );
    }

    /// `load_pumping_stations(None)` returns `Ok(Vec::new())`.
    #[test]
    fn test_load_pumping_stations_none_returns_empty() {
        let result = load_pumping_stations(None).unwrap();
        assert!(
            result.is_empty(),
            "expected empty vec when path is None, got {result:?}"
        );
    }

    /// `load_energy_contracts(None)` returns `Ok(Vec::new())`.
    #[test]
    fn test_load_energy_contracts_none_returns_empty() {
        let result = load_energy_contracts(None).unwrap();
        assert!(
            result.is_empty(),
            "expected empty vec when path is None, got {result:?}"
        );
    }
}
