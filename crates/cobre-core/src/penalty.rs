//! Penalty resolution and pre-resolved penalty structures.
//!
//! The penalty system uses a three-tier resolution cascade: global defaults,
//! entity-level overrides, and stage-level overrides. After resolution,
//! penalties are stored as pre-computed per-(entity, stage) values so solvers
//! do not need to re-evaluate the cascade during execution.
//!
//! This module implements the first two tiers (global → entity). Stage-varying
//! overrides are handled by `cobre-io` and are not implemented here.

use crate::entities::{DeficitSegment, HydroPenalties};

/// Global default penalty values for all entity types.
///
/// Mirrors the structure of `penalties.json`. These values are used as
/// fallbacks when entity-level overrides are not specified.
/// See Penalty System spec section 3.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GlobalPenaltyDefaults {
    // Bus defaults
    /// Default piecewise-linear deficit cost segments for buses.
    pub bus_deficit_segments: Vec<DeficitSegment>,
    /// Default excess cost for buses \[$/`MWh`\].
    pub bus_excess_cost: f64,

    // Line defaults
    /// Default exchange cost for lines \[$/`MWh`\].
    pub line_exchange_cost: f64,

    // Hydro defaults
    /// Default hydro penalty values. Applied to any hydro field not
    /// overridden at the entity level.
    pub hydro: HydroPenalties,

    // Non-controllable source defaults
    /// Default curtailment cost for non-controllable sources \[$/`MWh`\].
    pub ncs_curtailment_cost: f64,
}

/// Optional entity-level hydro penalty overrides.
///
/// Each field corresponds to a field in [`HydroPenalties`]. A value of `None`
/// means "use global default". A value of `Some(x)` means "override with x".
///
/// This is an intermediate type used during System construction; the resolved
/// [`HydroPenalties`] (with no `Option`s) is stored on the `Hydro` entity.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct HydroPenaltyOverrides {
    /// Override for spillage cost [$/m³/s]. `None` = use global default.
    pub spillage_cost: Option<f64>,
    /// Override for diversion cost [$/m³/s]. `None` = use global default.
    pub diversion_cost: Option<f64>,
    /// Override for FPHA turbined cost \[$/`MWh`\]. `None` = use global default.
    pub fpha_turbined_cost: Option<f64>,
    /// Override for storage violation below cost [$/hm³]. `None` = use global default.
    pub storage_violation_below_cost: Option<f64>,
    /// Override for filling target violation cost [$/hm³]. `None` = use global default.
    pub filling_target_violation_cost: Option<f64>,
    /// Override for turbined violation below cost [$/m³/s]. `None` = use global default.
    pub turbined_violation_below_cost: Option<f64>,
    /// Override for outflow violation below cost [$/m³/s]. `None` = use global default.
    pub outflow_violation_below_cost: Option<f64>,
    /// Override for outflow violation above cost [$/m³/s]. `None` = use global default.
    pub outflow_violation_above_cost: Option<f64>,
    /// Override for generation violation below cost [$/MW]. `None` = use global default.
    pub generation_violation_below_cost: Option<f64>,
    /// Override for evaporation violation cost [$/mm]. `None` = use global default.
    pub evaporation_violation_cost: Option<f64>,
    /// Override for water withdrawal violation cost [$/m³/s]. `None` = use global default.
    pub water_withdrawal_violation_cost: Option<f64>,
}

/// Resolve a bus's deficit segments: use entity override if present, else global default.
///
/// Returns an owned `Vec<DeficitSegment>` cloned from whichever source wins the cascade.
///
/// # Examples
///
/// ```
/// use cobre_core::penalty::{GlobalPenaltyDefaults, resolve_bus_deficit_segments};
/// use cobre_core::entities::{DeficitSegment, HydroPenalties};
///
/// let global = GlobalPenaltyDefaults {
///     bus_deficit_segments: vec![DeficitSegment { depth_mw: None, cost_per_mwh: 500.0 }],
///     bus_excess_cost: 10.0,
///     line_exchange_cost: 5.0,
///     hydro: HydroPenalties {
///         spillage_cost: 0.01, diversion_cost: 0.02, fpha_turbined_cost: 0.03,
///         storage_violation_below_cost: 1.0, filling_target_violation_cost: 2.0,
///         turbined_violation_below_cost: 3.0, outflow_violation_below_cost: 4.0,
///         outflow_violation_above_cost: 5.0, generation_violation_below_cost: 6.0,
///         evaporation_violation_cost: 7.0, water_withdrawal_violation_cost: 8.0,
///     },
///     ncs_curtailment_cost: 50.0,
/// };
///
/// // No entity override -> returns global default segments.
/// let resolved = resolve_bus_deficit_segments(&None, &global);
/// assert_eq!(resolved.len(), 1);
/// assert_eq!(resolved[0].cost_per_mwh, 500.0);
/// ```
#[must_use]
pub fn resolve_bus_deficit_segments(
    entity_deficit_segments: &Option<Vec<DeficitSegment>>,
    global: &GlobalPenaltyDefaults,
) -> Vec<DeficitSegment> {
    entity_deficit_segments
        .clone()
        .unwrap_or_else(|| global.bus_deficit_segments.clone())
}

/// Resolve a bus's excess cost: use entity override if present, else global default.
///
/// # Examples
///
/// ```
/// use cobre_core::penalty::{GlobalPenaltyDefaults, resolve_bus_excess_cost};
/// use cobre_core::entities::{DeficitSegment, HydroPenalties};
///
/// let global = GlobalPenaltyDefaults {
///     bus_deficit_segments: vec![DeficitSegment { depth_mw: None, cost_per_mwh: 500.0 }],
///     bus_excess_cost: 100.0,
///     line_exchange_cost: 5.0,
///     hydro: HydroPenalties {
///         spillage_cost: 0.01, diversion_cost: 0.02, fpha_turbined_cost: 0.03,
///         storage_violation_below_cost: 1.0, filling_target_violation_cost: 2.0,
///         turbined_violation_below_cost: 3.0, outflow_violation_below_cost: 4.0,
///         outflow_violation_above_cost: 5.0, generation_violation_below_cost: 6.0,
///         evaporation_violation_cost: 7.0, water_withdrawal_violation_cost: 8.0,
///     },
///     ncs_curtailment_cost: 50.0,
/// };
///
/// assert_eq!(resolve_bus_excess_cost(None, &global), 100.0);
/// assert_eq!(resolve_bus_excess_cost(Some(250.0), &global), 250.0);
/// ```
#[must_use]
pub fn resolve_bus_excess_cost(
    entity_excess_cost: Option<f64>,
    global: &GlobalPenaltyDefaults,
) -> f64 {
    entity_excess_cost.unwrap_or(global.bus_excess_cost)
}

/// Resolve a line's exchange cost: use entity override if present, else global default.
///
/// # Examples
///
/// ```
/// use cobre_core::penalty::{GlobalPenaltyDefaults, resolve_line_exchange_cost};
/// use cobre_core::entities::{DeficitSegment, HydroPenalties};
///
/// let global = GlobalPenaltyDefaults {
///     bus_deficit_segments: vec![DeficitSegment { depth_mw: None, cost_per_mwh: 500.0 }],
///     bus_excess_cost: 100.0,
///     line_exchange_cost: 5.0,
///     hydro: HydroPenalties {
///         spillage_cost: 0.01, diversion_cost: 0.02, fpha_turbined_cost: 0.03,
///         storage_violation_below_cost: 1.0, filling_target_violation_cost: 2.0,
///         turbined_violation_below_cost: 3.0, outflow_violation_below_cost: 4.0,
///         outflow_violation_above_cost: 5.0, generation_violation_below_cost: 6.0,
///         evaporation_violation_cost: 7.0, water_withdrawal_violation_cost: 8.0,
///     },
///     ncs_curtailment_cost: 50.0,
/// };
///
/// assert_eq!(resolve_line_exchange_cost(None, &global), 5.0);
/// assert_eq!(resolve_line_exchange_cost(Some(12.0), &global), 12.0);
/// ```
#[must_use]
pub fn resolve_line_exchange_cost(
    entity_exchange_cost: Option<f64>,
    global: &GlobalPenaltyDefaults,
) -> f64 {
    entity_exchange_cost.unwrap_or(global.line_exchange_cost)
}

/// Resolve a hydro plant's penalty values.
///
/// For each of the 11 penalty fields, uses the entity override if `Some`, or the
/// global default from `global.hydro` if `None`. When `entity_overrides` is `None`
/// (no entity-level penalty block was specified at all), all 11 fields fall back to
/// the global defaults, and the result equals `global.hydro` exactly.
///
/// # Examples
///
/// ```
/// use cobre_core::penalty::{
///     GlobalPenaltyDefaults, HydroPenaltyOverrides, resolve_hydro_penalties,
/// };
/// use cobre_core::entities::{DeficitSegment, HydroPenalties};
///
/// let global_hydro = HydroPenalties {
///     spillage_cost: 0.01, diversion_cost: 0.02, fpha_turbined_cost: 0.03,
///     storage_violation_below_cost: 1.0, filling_target_violation_cost: 2.0,
///     turbined_violation_below_cost: 3.0, outflow_violation_below_cost: 4.0,
///     outflow_violation_above_cost: 5.0, generation_violation_below_cost: 6.0,
///     evaporation_violation_cost: 7.0, water_withdrawal_violation_cost: 8.0,
/// };
/// let global = GlobalPenaltyDefaults {
///     bus_deficit_segments: vec![],
///     bus_excess_cost: 100.0,
///     line_exchange_cost: 5.0,
///     hydro: global_hydro,
///     ncs_curtailment_cost: 50.0,
/// };
///
/// // All-None overrides -> result equals global hydro exactly.
/// let resolved = resolve_hydro_penalties(&None, &global);
/// assert_eq!(resolved, global.hydro);
///
/// // Partial override: only spillage_cost overridden.
/// let overrides = HydroPenaltyOverrides {
///     spillage_cost: Some(0.05),
///     ..Default::default()
/// };
/// let resolved = resolve_hydro_penalties(&Some(overrides), &global);
/// assert!((resolved.spillage_cost - 0.05).abs() < f64::EPSILON);
/// assert!((resolved.diversion_cost - 0.02).abs() < f64::EPSILON);
/// ```
#[must_use]
pub fn resolve_hydro_penalties(
    entity_overrides: &Option<HydroPenaltyOverrides>,
    global: &GlobalPenaltyDefaults,
) -> HydroPenalties {
    let g = &global.hydro;
    match entity_overrides {
        None => *g,
        Some(ov) => HydroPenalties {
            spillage_cost: ov.spillage_cost.unwrap_or(g.spillage_cost),
            diversion_cost: ov.diversion_cost.unwrap_or(g.diversion_cost),
            fpha_turbined_cost: ov.fpha_turbined_cost.unwrap_or(g.fpha_turbined_cost),
            storage_violation_below_cost: ov
                .storage_violation_below_cost
                .unwrap_or(g.storage_violation_below_cost),
            filling_target_violation_cost: ov
                .filling_target_violation_cost
                .unwrap_or(g.filling_target_violation_cost),
            turbined_violation_below_cost: ov
                .turbined_violation_below_cost
                .unwrap_or(g.turbined_violation_below_cost),
            outflow_violation_below_cost: ov
                .outflow_violation_below_cost
                .unwrap_or(g.outflow_violation_below_cost),
            outflow_violation_above_cost: ov
                .outflow_violation_above_cost
                .unwrap_or(g.outflow_violation_above_cost),
            generation_violation_below_cost: ov
                .generation_violation_below_cost
                .unwrap_or(g.generation_violation_below_cost),
            evaporation_violation_cost: ov
                .evaporation_violation_cost
                .unwrap_or(g.evaporation_violation_cost),
            water_withdrawal_violation_cost: ov
                .water_withdrawal_violation_cost
                .unwrap_or(g.water_withdrawal_violation_cost),
        },
    }
}

/// Resolve a non-controllable source's curtailment cost: use entity override if present,
/// else global default.
///
/// # Examples
///
/// ```
/// use cobre_core::penalty::{GlobalPenaltyDefaults, resolve_ncs_curtailment_cost};
/// use cobre_core::entities::{DeficitSegment, HydroPenalties};
///
/// let global = GlobalPenaltyDefaults {
///     bus_deficit_segments: vec![],
///     bus_excess_cost: 100.0,
///     line_exchange_cost: 5.0,
///     hydro: HydroPenalties {
///         spillage_cost: 0.01, diversion_cost: 0.02, fpha_turbined_cost: 0.03,
///         storage_violation_below_cost: 1.0, filling_target_violation_cost: 2.0,
///         turbined_violation_below_cost: 3.0, outflow_violation_below_cost: 4.0,
///         outflow_violation_above_cost: 5.0, generation_violation_below_cost: 6.0,
///         evaporation_violation_cost: 7.0, water_withdrawal_violation_cost: 8.0,
///     },
///     ncs_curtailment_cost: 50.0,
/// };
///
/// assert_eq!(resolve_ncs_curtailment_cost(None, &global), 50.0);
/// assert_eq!(resolve_ncs_curtailment_cost(Some(75.0), &global), 75.0);
/// ```
#[must_use]
pub fn resolve_ncs_curtailment_cost(
    entity_curtailment_cost: Option<f64>,
    global: &GlobalPenaltyDefaults,
) -> f64 {
    entity_curtailment_cost.unwrap_or(global.ncs_curtailment_cost)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_global() -> GlobalPenaltyDefaults {
        GlobalPenaltyDefaults {
            bus_deficit_segments: vec![
                DeficitSegment {
                    depth_mw: Some(100.0),
                    cost_per_mwh: 500.0,
                },
                DeficitSegment {
                    depth_mw: None,
                    cost_per_mwh: 5000.0,
                },
            ],
            bus_excess_cost: 100.0,
            line_exchange_cost: 5.0,
            hydro: HydroPenalties {
                spillage_cost: 0.01,
                diversion_cost: 0.02,
                fpha_turbined_cost: 0.03,
                storage_violation_below_cost: 1.0,
                filling_target_violation_cost: 2.0,
                turbined_violation_below_cost: 3.0,
                outflow_violation_below_cost: 4.0,
                outflow_violation_above_cost: 5.0,
                generation_violation_below_cost: 6.0,
                evaporation_violation_cost: 7.0,
                water_withdrawal_violation_cost: 8.0,
            },
            ncs_curtailment_cost: 50.0,
        }
    }

    #[test]
    fn test_resolve_bus_excess_cost_global() {
        let global = make_global();
        let result = resolve_bus_excess_cost(None, &global);
        assert!((result - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_resolve_bus_excess_cost_override() {
        let global = make_global();
        let result = resolve_bus_excess_cost(Some(250.0), &global);
        assert!((result - 250.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_resolve_bus_deficit_segments_global() {
        let global = make_global();
        let result = resolve_bus_deficit_segments(&None, &global);
        assert_eq!(result, global.bus_deficit_segments);
        assert_eq!(result.len(), 2);
        assert!((result[0].cost_per_mwh - 500.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_resolve_bus_deficit_segments_override() {
        let global = make_global();
        let override_segments = vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 9999.0,
        }];
        let result = resolve_bus_deficit_segments(&Some(override_segments.clone()), &global);
        assert_eq!(result, override_segments);
        assert_eq!(result.len(), 1);
        assert!((result[0].cost_per_mwh - 9999.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_resolve_line_exchange_cost_global() {
        let global = make_global();
        let result = resolve_line_exchange_cost(None, &global);
        assert!((result - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_resolve_line_exchange_cost_override() {
        let global = make_global();
        let result = resolve_line_exchange_cost(Some(12.0), &global);
        assert!((result - 12.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_resolve_hydro_penalties_all_global() {
        let global = make_global();
        let result = resolve_hydro_penalties(&None, &global);
        assert_eq!(result, global.hydro);
    }

    #[test]
    fn test_resolve_hydro_penalties_partial_override() {
        let global = make_global();
        let overrides = HydroPenaltyOverrides {
            spillage_cost: Some(0.05),
            ..Default::default()
        };
        let result = resolve_hydro_penalties(&Some(overrides), &global);

        assert!((result.spillage_cost - 0.05).abs() < f64::EPSILON);
        assert!((result.diversion_cost - 0.02).abs() < f64::EPSILON);
        assert!((result.fpha_turbined_cost - 0.03).abs() < f64::EPSILON);
        assert!((result.storage_violation_below_cost - 1.0).abs() < f64::EPSILON);
        assert!((result.filling_target_violation_cost - 2.0).abs() < f64::EPSILON);
        assert!((result.turbined_violation_below_cost - 3.0).abs() < f64::EPSILON);
        assert!((result.outflow_violation_below_cost - 4.0).abs() < f64::EPSILON);
        assert!((result.outflow_violation_above_cost - 5.0).abs() < f64::EPSILON);
        assert!((result.generation_violation_below_cost - 6.0).abs() < f64::EPSILON);
        assert!((result.evaporation_violation_cost - 7.0).abs() < f64::EPSILON);
        assert!((result.water_withdrawal_violation_cost - 8.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_resolve_hydro_penalties_all_override() {
        let global = make_global();
        let overrides = HydroPenaltyOverrides {
            spillage_cost: Some(10.0),
            diversion_cost: Some(20.0),
            fpha_turbined_cost: Some(30.0),
            storage_violation_below_cost: Some(40.0),
            filling_target_violation_cost: Some(50.0),
            turbined_violation_below_cost: Some(60.0),
            outflow_violation_below_cost: Some(70.0),
            outflow_violation_above_cost: Some(80.0),
            generation_violation_below_cost: Some(90.0),
            evaporation_violation_cost: Some(100.0),
            water_withdrawal_violation_cost: Some(110.0),
        };
        let result = resolve_hydro_penalties(&Some(overrides), &global);

        assert!((result.spillage_cost - 10.0).abs() < f64::EPSILON);
        assert!((result.diversion_cost - 20.0).abs() < f64::EPSILON);
        assert!((result.fpha_turbined_cost - 30.0).abs() < f64::EPSILON);
        assert!((result.storage_violation_below_cost - 40.0).abs() < f64::EPSILON);
        assert!((result.filling_target_violation_cost - 50.0).abs() < f64::EPSILON);
        assert!((result.turbined_violation_below_cost - 60.0).abs() < f64::EPSILON);
        assert!((result.outflow_violation_below_cost - 70.0).abs() < f64::EPSILON);
        assert!((result.outflow_violation_above_cost - 80.0).abs() < f64::EPSILON);
        assert!((result.generation_violation_below_cost - 90.0).abs() < f64::EPSILON);
        assert!((result.evaporation_violation_cost - 100.0).abs() < f64::EPSILON);
        assert!((result.water_withdrawal_violation_cost - 110.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_resolve_hydro_penalties_default_overrides_equals_global() {
        let global = make_global();
        let result = resolve_hydro_penalties(&Some(HydroPenaltyOverrides::default()), &global);
        assert_eq!(result, global.hydro);
    }

    #[test]
    fn test_resolve_ncs_curtailment_cost_global() {
        let global = make_global();
        let result = resolve_ncs_curtailment_cost(None, &global);
        assert!((result - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_resolve_ncs_curtailment_cost_override() {
        let global = make_global();
        let result = resolve_ncs_curtailment_cost(Some(75.0), &global);
        assert!((result - 75.0).abs() < f64::EPSILON);
    }
}
