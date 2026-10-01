//! User-defined generic linear constraints.
//!
//! This module defines the in-memory representation of generic constraints
//! that users can specify to add custom linear relationships between LP
//! variables. The expression parser (string → [`ConstraintExpression`])
//! lives in `cobre-io`, not here. This module contains only the output types.
//!
//! See `internal-structures.md §15` and `input-constraints.md §3` for the
//! full specification, grammar, and validation rules.
//!
//! # Variable Reference Catalog
//!
//! [`VariableRef`] covers every LP variable type defined in the spec (§15).
//! Each variant carries the entity ID and, for block-capable variables, an
//! optional block ID. `Some(i)` references block `i`. `None` is not
//! block-specific and resolves by the variable's nature: per-block flows (e.g.
//! [`VariableRef::HydroInflow`]) follow the materialized row's block — a single
//! collapsed stage-level row for a block-independent expression, one row per
//! block otherwise; stage-level stocks ([`VariableRef::HydroStorage`] and the
//! storage-boundary variants at their stage endpoints S⁰/Sᴷ) resolve to a single
//! fixed column; [`VariableRef::HydroEvaporation`] resolves `None` to block 0, and
//! on a parallel stage every named block resolves to that same single
//! stage-level slot. No variant sums over blocks.
//!
//! [`VariableRef::HydroTurbined`] and [`VariableRef::HydroGeneration`] additionally
//! carry `bus_id: Option<EntityId>`, selecting one cell of a plant split across
//! several buses: the LP holds one turbine/generation column per `(hydro, bus)`
//! cell, so the addressable unit is a bus, not a unit group — two same-bus groups
//! share one column, and a group-level selector would resolve ambiguously to it.
//! `None` sums over the plant's cells, preserving the plant-level meaning every
//! other variant carries. No other variant accepts a bus selector.
//!
//! # Examples
//!
//! ```
//! use cobre_core::{
//!     EntityId, GenericConstraint, ConstraintExpression,
//!     LinearTerm, SlackConfig, VariableRef,
//! };
//!
//! let expr = ConstraintExpression {
//!     terms: vec![
//!         LinearTerm::literal(1.0, VariableRef::HydroGeneration {
//!             hydro_id: EntityId(10),
//!             block_id: None,
//!             bus_id: None,
//!         }),
//!         LinearTerm::literal(1.0, VariableRef::HydroGeneration {
//!             hydro_id: EntityId(11),
//!             block_id: None,
//!             bus_id: None,
//!         }),
//!     ],
//! };
//!
//! assert_eq!(expr.terms.len(), 2);
//!
//! let gc = GenericConstraint {
//!     id: EntityId(0),
//!     name: "min_southeast_hydro".to_string(),
//!     description: Some("Minimum hydro generation in Southeast region".to_string()),
//!     expression: expr,
//!     slack: SlackConfig { enabled: true, penalty: Some(5_000.0) },
//!     bound_lower_affine: None,
//!     bound_upper_affine: None,
//! };
//!
//! assert_eq!(gc.expression.terms.len(), 2);
//! ```

use crate::CoefficientRef;
use crate::EntityId;

/// Reference to a single LP variable in a generic constraint expression.
///
/// See the module header for the `block_id = None` block-independent vs
/// block-dependent row-materialization contract, and for why
/// [`VariableRef::HydroTurbined`]/[`VariableRef::HydroGeneration`]'s `bus_id`
/// selector names a bus rather than a unit group.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum VariableRef {
    /// Reservoir storage level for a hydro plant (stage-level, not block-specific).
    HydroStorage {
        /// Hydro plant identifier.
        hydro_id: EntityId,
    },
    /// Turbined water flow for a hydro plant (m³/s).
    HydroTurbined {
        /// Hydro plant identifier.
        hydro_id: EntityId,
        /// Block selector.
        block_id: Option<usize>,
        /// Bus selector: `None` sums over the plant's cells; `Some(b)` selects
        /// the single cell whose unit groups sit on bus `b`.
        bus_id: Option<EntityId>,
    },
    /// Spillage flow for a hydro plant (m³/s).
    HydroSpillage {
        /// Hydro plant identifier.
        hydro_id: EntityId,
        /// Block selector.
        block_id: Option<usize>,
    },
    /// Diversion flow for a hydro plant (m³/s). Only valid for hydros with diversion.
    HydroDiversion {
        /// Hydro plant identifier.
        hydro_id: EntityId,
        /// Block selector.
        block_id: Option<usize>,
    },
    /// Total outflow for a hydro plant (m³/s): a derived alias for
    /// turbined + spillage, not an independent LP column.
    HydroOutflow {
        /// Hydro plant identifier.
        hydro_id: EntityId,
        /// Block selector.
        block_id: Option<usize>,
    },
    /// Electrical generation from a hydro plant (MW).
    HydroGeneration {
        /// Hydro plant identifier.
        hydro_id: EntityId,
        /// Block selector.
        block_id: Option<usize>,
        /// Bus selector: `None` sums over the plant's cells; `Some(b)` selects
        /// the single cell whose unit groups sit on bus `b`.
        bus_id: Option<EntityId>,
    },
    /// Signed evaporation flow from a hydro reservoir (m³/s). Positive values
    /// represent net evaporative outflow; negative values represent net rainfall
    /// input absorbed by the reservoir. `Some(k)` selects block `k`; `None`
    /// selects block 0. On a parallel stage every block shares one stage-level
    /// evaporation slot on the stage endpoints, so `None`/`Some(0)` resolve to
    /// that slot; generic-constraint validation rejects `Some(k)` for `k >= 1`
    /// there. In chronological mode with `K > 1` each block has its own slot, so
    /// a `None` reference is rejected by generic-constraint validation — a block
    /// must be named.
    HydroEvaporation {
        /// Hydro plant identifier.
        hydro_id: EntityId,
        /// Block selector; `None` = block 0 (the stage-level slot on a parallel
        /// stage).
        block_id: Option<usize>,
    },
    /// Water withdrawal from a hydro reservoir (m³/s). Stage-level, not block-specific.
    HydroWithdrawal {
        /// Hydro plant identifier.
        hydro_id: EntityId,
    },
    /// Electrical generation from a thermal unit (MW).
    ThermalGeneration {
        /// Thermal unit identifier.
        thermal_id: EntityId,
        /// Block selector.
        block_id: Option<usize>,
    },
    /// Direct (forward) power flow on a transmission line (MW).
    LineDirect {
        /// Transmission line identifier.
        line_id: EntityId,
        /// Block selector.
        block_id: Option<usize>,
    },
    /// Reverse power flow on a transmission line (MW).
    LineReverse {
        /// Transmission line identifier.
        line_id: EntityId,
        /// Block selector.
        block_id: Option<usize>,
    },
    /// Net exchange flow on a transmission line (direct - reverse) (MW).
    ///
    /// Derived: maps to two LP columns (forward `+1.0`, reverse `-1.0`),
    /// net flow source-to-target.
    LineExchange {
        /// Transmission line identifier.
        line_id: EntityId,
        /// Block selector.
        block_id: Option<usize>,
    },
    /// Load deficit (unserved energy) at a bus (MW).
    BusDeficit {
        /// Bus identifier.
        bus_id: EntityId,
        /// Block selector.
        block_id: Option<usize>,
    },
    /// Load excess (over-generation) at a bus (MW).
    BusExcess {
        /// Bus identifier.
        bus_id: EntityId,
        /// Block selector.
        block_id: Option<usize>,
    },
    /// Pumped water flow at a pumping station (m³/s).
    PumpingFlow {
        /// Pumping station identifier.
        station_id: EntityId,
        /// Block selector.
        block_id: Option<usize>,
    },
    /// Electrical power consumed by a pumping station (MW).
    PumpingPower {
        /// Pumping station identifier.
        station_id: EntityId,
        /// Block selector.
        block_id: Option<usize>,
    },
    /// Energy imported via a contract (MW).
    ContractImport {
        /// Energy contract identifier.
        contract_id: EntityId,
        /// Block selector.
        block_id: Option<usize>,
    },
    /// Energy exported via a contract (MW).
    ContractExport {
        /// Energy contract identifier.
        contract_id: EntityId,
        /// Block selector.
        block_id: Option<usize>,
    },
    /// Generation from a non-controllable source (wind, solar, etc.) (MW).
    NonControllableGeneration {
        /// Non-controllable source identifier.
        source_id: EntityId,
        /// Block selector.
        block_id: Option<usize>,
    },
    /// Curtailment of a non-controllable source (MW).
    NonControllableCurtailment {
        /// Non-controllable source identifier.
        source_id: EntityId,
        /// Block selector.
        block_id: Option<usize>,
    },
    /// Forward-commitment decision for an anticipated thermal unit (MW).
    ///
    /// The commitment placed at stage `t` for delivery at `t + lead_stages`. A
    /// per-plant per-stage scalar — **no `block_id`**, the commitment is uniform
    /// across blocks. The column exists only for plants with
    /// `anticipated_config: Some(_)`; referencing a non-anticipated thermal is a
    /// referential-validation error. At boundary stages (`t + K_i >= n_stages`)
    /// the column has `[0.0, 0.0]` bounds, so a constraint there is a no-op.
    ///
    /// Appended at the END of the enum to preserve every existing variant's
    /// postcard discriminant (postcard encodes variants by declaration order; a
    /// mid-enum insertion breaks serialized policies — pinned by
    /// `test_variable_ref_postcard_discriminant_pin`).
    AnticipatedDecision {
        /// Thermal unit identifier. Must satisfy `anticipated_config: Some(_)`.
        thermal_id: EntityId,
    },
    /// Total realized inflow into a hydro reservoir (m³/s). Block-capable.
    ///
    /// Local natural inflow PLUS the immediately-upstream cascade releases
    /// (turbined + spilled + diverted) routed down — the inflow side of the water
    /// balance, not the `z_inflow` column alone. For an operating reservoir, an
    /// upstream release enters with the share of it that the water balance credits
    /// to this block. That is the whole release when the arc has no water travel
    /// time. The upstream transit water that matures into this block also enters,
    /// as a rate. While an upstream plant is not yet built or is retired, its
    /// local inflow, the releases of the plants above it and the flows diverted
    /// into it pass straight through to the first operating plant below, and
    /// enter that plant's term whole, in the block they occur. Pumping into the
    /// plant is not part of the term. This is a rate
    /// identity in m³/s, NOT the `−τ` volume weighting of the storage-balance row.
    /// Block-dependent (upstream releases are per-block LP columns), so `None`
    /// always expands to one row per block, never a collapsed stage-level row.
    ///
    /// Appended at the END of the enum to preserve every existing variant's
    /// postcard discriminant.
    HydroInflow {
        /// Hydro plant identifier.
        hydro_id: EntityId,
        /// Block selector; `None` expands to one row per block.
        block_id: Option<usize>,
    },
    /// Start-of-block storage boundary for a hydro reservoir (hm³).
    ///
    /// `Some(k)` references the incoming storage column of block `k` (boundary
    /// `Sᵏ`); `None` references the stage-initial anchor `S⁰`. A stage-level stock:
    /// resolves to a single fixed column, never a per-block expansion.
    ///
    /// Appended at the END of the enum to keep its postcard discriminant stable.
    HydroStorageInitial {
        /// Hydro plant identifier.
        hydro_id: EntityId,
        /// Block selector; `None` = stage-initial `S⁰`.
        block_id: Option<usize>,
    },
    /// End-of-block storage boundary for a hydro reservoir (hm³).
    ///
    /// `Some(k)` references the outgoing storage column of block `k` (boundary
    /// `Sᵏ⁺¹`); `None` references the stage-final `Sᴷ`, equal to `HydroStorage`.
    /// A stage-level stock: resolves to a single fixed column, never a per-block
    /// expansion.
    ///
    /// Appended at the END of the enum to keep its postcard discriminant stable.
    HydroStorageFinal {
        /// Hydro plant identifier.
        hydro_id: EntityId,
        /// Block selector; `None` = stage-final `Sᴷ`.
        block_id: Option<usize>,
    },
    /// Start-of-block useful-volume boundary for a hydro reservoir (hm³):
    /// `storage_initial - V_lo`, the `-V_lo` shift applied at bound-fold time.
    ///
    /// Appended at the END of the enum to keep its postcard discriminant stable.
    HydroUsefulVolumeInitial {
        /// Hydro plant identifier.
        hydro_id: EntityId,
        /// Block selector; `None` = stage-initial `S⁰`.
        block_id: Option<usize>,
    },
    /// End-of-block useful-volume boundary for a hydro reservoir (hm³):
    /// `storage_final - V_lo`, the `-V_lo` shift applied at bound-fold time.
    ///
    /// Appended at the END of the enum to keep its postcard discriminant stable.
    HydroUsefulVolumeFinal {
        /// Hydro plant identifier.
        hydro_id: EntityId,
        /// Block selector; `None` = stage-final `Sᴷ`.
        block_id: Option<usize>,
    },
}

/// One term in a linear constraint expression: `coefficient * scale * variable`.
///
/// The LP coefficient is `resolve(coefficient, stage) * scale`, the
/// stage-resolved scalar for a `Parameter` or the literal value for a `Literal`.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LinearTerm {
    /// Coefficient reference, resolved to a scalar per stage at LP build time.
    pub coefficient: CoefficientRef,
    /// Multiplicative scale applied after the coefficient is resolved.
    pub scale: f64,
    /// The LP variable being referenced.
    pub variable: VariableRef,
}

impl LinearTerm {
    /// Construct a literal-coefficient `LinearTerm` with `scale = 1.0`.
    #[must_use]
    pub fn literal(coef: f64, variable: VariableRef) -> Self {
        Self {
            coefficient: CoefficientRef::Literal(coef),
            scale: 1.0,
            variable,
        }
    }

    /// Construct a named-parameter-coefficient `LinearTerm`.
    ///
    /// `scale` carries the literal multiplier from the expression (e.g. `2.5`
    /// for `"2.5 * @rho_eq * x"`, or `sign` for `"@rho_eq * x"`).
    #[must_use]
    pub fn parameter(id: EntityId, scale: f64, variable: VariableRef) -> Self {
        Self {
            coefficient: CoefficientRef::Parameter(id),
            scale,
            variable,
        }
    }
}

/// Parsed left-hand side of a generic constraint as weighted variable
/// references. An empty `terms` vector is valid (constant-only expression, not
/// rejected at this layer).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ConstraintExpression {
    /// Ordered list of linear terms that form the left-hand side of the constraint.
    pub terms: Vec<LinearTerm>,
}

impl ConstraintExpression {
    /// Sort `terms` into the content-derived canonical order ([`canonical_term_key`]),
    /// so the flattened order is a pure function of the terms — independent of how the
    /// expression was authored or the order references were substituted in. Upholds the
    /// declaration-order-invariance contract: two expressions with the same terms in
    /// different orders converge to the same term sequence.
    pub fn canonicalize(&mut self) {
        self.terms.sort_by_key(canonical_term_key);
    }
}

/// Sort key over a [`VariableRef`], as `(variant_tag, entity_id, block_sentinel,
/// bus_sentinel)`. The tag numbers variants in declaration order; keep it an explicit
/// match, never a derived `Ord` — the tail variants are appended out of catalog order
/// to pin their wire discriminants, and a derive would couple this order to that
/// convention.
fn canonical_variable_key(v: &VariableRef) -> (u8, i32, i64, i64) {
    match *v {
        VariableRef::HydroStorage { hydro_id } => (0, hydro_id.0, -1, -1),
        VariableRef::HydroTurbined {
            hydro_id,
            block_id,
            bus_id,
        } => (
            1,
            hydro_id.0,
            block_sentinel(block_id),
            bus_sentinel(bus_id),
        ),
        VariableRef::HydroSpillage { hydro_id, block_id } => block_key(2, hydro_id, block_id),
        VariableRef::HydroDiversion { hydro_id, block_id } => block_key(3, hydro_id, block_id),
        VariableRef::HydroOutflow { hydro_id, block_id } => block_key(4, hydro_id, block_id),
        VariableRef::HydroGeneration {
            hydro_id,
            block_id,
            bus_id,
        } => (
            5,
            hydro_id.0,
            block_sentinel(block_id),
            bus_sentinel(bus_id),
        ),
        VariableRef::HydroEvaporation { hydro_id, block_id } => block_key(6, hydro_id, block_id),
        VariableRef::HydroWithdrawal { hydro_id } => (7, hydro_id.0, -1, -1),
        VariableRef::ThermalGeneration {
            thermal_id,
            block_id,
        } => block_key(8, thermal_id, block_id),
        VariableRef::LineDirect { line_id, block_id } => block_key(9, line_id, block_id),
        VariableRef::LineReverse { line_id, block_id } => block_key(10, line_id, block_id),
        VariableRef::LineExchange { line_id, block_id } => block_key(11, line_id, block_id),
        VariableRef::BusDeficit { bus_id, block_id } => block_key(12, bus_id, block_id),
        VariableRef::BusExcess { bus_id, block_id } => block_key(13, bus_id, block_id),
        VariableRef::PumpingFlow {
            station_id,
            block_id,
        } => block_key(14, station_id, block_id),
        VariableRef::PumpingPower {
            station_id,
            block_id,
        } => block_key(15, station_id, block_id),
        VariableRef::ContractImport {
            contract_id,
            block_id,
        } => block_key(16, contract_id, block_id),
        VariableRef::ContractExport {
            contract_id,
            block_id,
        } => block_key(17, contract_id, block_id),
        VariableRef::NonControllableGeneration {
            source_id,
            block_id,
        } => block_key(18, source_id, block_id),
        VariableRef::NonControllableCurtailment {
            source_id,
            block_id,
        } => block_key(19, source_id, block_id),
        VariableRef::AnticipatedDecision { thermal_id } => (20, thermal_id.0, -1, -1),
        VariableRef::HydroInflow { hydro_id, block_id } => block_key(21, hydro_id, block_id),
        VariableRef::HydroStorageInitial { hydro_id, block_id } => {
            block_key(22, hydro_id, block_id)
        }
        VariableRef::HydroStorageFinal { hydro_id, block_id } => block_key(23, hydro_id, block_id),
        VariableRef::HydroUsefulVolumeInitial { hydro_id, block_id } => {
            block_key(24, hydro_id, block_id)
        }
        VariableRef::HydroUsefulVolumeFinal { hydro_id, block_id } => {
            block_key(25, hydro_id, block_id)
        }
    }
}

fn block_key(tag: u8, id: EntityId, block_id: Option<usize>) -> (u8, i32, i64, i64) {
    (tag, id.0, block_sentinel(block_id), -1)
}

/// `None` maps to `-1` (a block-independent reference orders before block 0).
// Rationale (cast_possible_wrap): block indices are small stage-local counts far
// below i64::MAX; the widening signed cast never wraps.
#[allow(clippy::cast_possible_wrap)]
fn block_sentinel(block_id: Option<usize>) -> i64 {
    block_id.map_or(-1, |b| b as i64)
}

/// `None` maps to `-1` (a bus-independent reference orders before bus 0).
fn bus_sentinel(bus_id: Option<EntityId>) -> i64 {
    bus_id.map_or(-1, |b| i64::from(b.0))
}

/// Total-order sort key over a [`LinearTerm`], a pure function of content: variable
/// key, then coefficient kind (`Literal` = 0 before `Parameter` = 1) and its payload
/// (`to_bits` / entity id), then `scale.to_bits`. `to_bits` gives exact float
/// ordering; ties occur only between fully-identical terms, where the order is
/// immaterial to the emitted bytes.
fn canonical_term_key(term: &LinearTerm) -> ((u8, i32, i64, i64), u8, u64, u64) {
    let (coef_kind, coef_payload) = match term.coefficient {
        CoefficientRef::Literal(v) => (0, v.to_bits()),
        CoefficientRef::Parameter(id) => (1, u64::from(id.0.cast_unsigned())),
    };
    (
        canonical_variable_key(&term.variable),
        coef_kind,
        coef_payload,
        term.scale.to_bits(),
    )
}

/// A generic constraint's symbolic bound endpoint: `constant + Σ coefᵢ·@paramᵢ`.
///
/// Each `(coef, param_id)` term resolves through `param_id`'s `(stage, block)`
/// axis and is summed with `constant`. [`AffineBound::single`] is the
/// single-`@name`-ref special case: `constant = 0.0` and one term at
/// coefficient `1.0`, which resolves to exactly the named parameter's value
/// (`0.0 + 1.0 * x == x` in `f64`).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AffineBound {
    /// The constant term of the affine remainder.
    pub constant: f64,
    /// Ordered `(coefficient, parameter_id)` terms summed onto `constant`.
    pub terms: Vec<(f64, EntityId)>,
}

impl AffineBound {
    /// A single named-parameter reference: `constant = 0.0`, one term at coefficient `1.0`.
    #[must_use]
    pub fn single(id: EntityId) -> Self {
        Self {
            constant: 0.0,
            terms: vec![(1.0, id)],
        }
    }

    /// Iterate over the parameter ids referenced by this bound's terms.
    pub fn params(&self) -> impl Iterator<Item = EntityId> + '_ {
        self.terms.iter().map(|&(_, id)| id)
    }
}

/// Slack variable configuration for a generic constraint.
///
/// An enabled slack lets the constraint be violated at a cost (entering the LP
/// objective), preventing infeasibility under tight or conflicting bounds.
/// `penalty` must be `Some(positive)` when `enabled`, `None` otherwise.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SlackConfig {
    /// Whether a slack variable is added to allow soft violation of the constraint.
    pub enabled: bool,
    /// Penalty cost per unit of constraint violation.
    pub penalty: Option<f64>,
}

/// A user-defined generic linear constraint.
///
/// Sorted by `id` after loading to satisfy declaration-order invariance.
/// Parsing, referential validation, and bounds loading happen in `cobre-io`.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GenericConstraint {
    /// Unique constraint identifier.
    pub id: EntityId,
    /// Short name used in reports and log output.
    pub name: String,
    /// Optional human-readable description.
    pub description: Option<String>,
    /// Parsed left-hand-side expression of the constraint.
    pub expression: ConstraintExpression,
    /// Slack variable configuration.
    pub slack: SlackConfig,
    /// Lower RHS bound as an affine remainder, composed with any numeric
    /// parquet base on the same endpoint: the LP builder's `fold_endpoint`
    /// resolves the endpoint to `base + bound` when both are present, or
    /// either alone when only one is.
    pub bound_lower_affine: Option<AffineBound>,
    /// Upper RHS bound as an affine remainder; upper counterpart of
    /// `bound_lower_affine`.
    pub bound_upper_affine: Option<AffineBound>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_variable_ref_variants() {
        let variants: &[(&str, VariableRef)] = &[
            (
                "HydroStorage",
                VariableRef::HydroStorage {
                    hydro_id: EntityId(0),
                },
            ),
            (
                "HydroTurbined",
                VariableRef::HydroTurbined {
                    hydro_id: EntityId(0),
                    block_id: None,
                    bus_id: None,
                },
            ),
            (
                "HydroSpillage",
                VariableRef::HydroSpillage {
                    hydro_id: EntityId(0),
                    block_id: Some(1),
                },
            ),
            (
                "HydroDiversion",
                VariableRef::HydroDiversion {
                    hydro_id: EntityId(0),
                    block_id: None,
                },
            ),
            (
                "HydroOutflow",
                VariableRef::HydroOutflow {
                    hydro_id: EntityId(0),
                    block_id: None,
                },
            ),
            (
                "HydroGeneration",
                VariableRef::HydroGeneration {
                    hydro_id: EntityId(0),
                    block_id: Some(0),
                    bus_id: None,
                },
            ),
            (
                "HydroEvaporation",
                VariableRef::HydroEvaporation {
                    hydro_id: EntityId(0),
                    block_id: None,
                },
            ),
            (
                "HydroWithdrawal",
                VariableRef::HydroWithdrawal {
                    hydro_id: EntityId(0),
                },
            ),
            (
                "ThermalGeneration",
                VariableRef::ThermalGeneration {
                    thermal_id: EntityId(0),
                    block_id: None,
                },
            ),
            (
                "LineDirect",
                VariableRef::LineDirect {
                    line_id: EntityId(0),
                    block_id: None,
                },
            ),
            (
                "LineReverse",
                VariableRef::LineReverse {
                    line_id: EntityId(0),
                    block_id: None,
                },
            ),
            (
                "LineExchange",
                VariableRef::LineExchange {
                    line_id: EntityId(0),
                    block_id: None,
                },
            ),
            (
                "BusDeficit",
                VariableRef::BusDeficit {
                    bus_id: EntityId(0),
                    block_id: None,
                },
            ),
            (
                "BusExcess",
                VariableRef::BusExcess {
                    bus_id: EntityId(0),
                    block_id: None,
                },
            ),
            (
                "PumpingFlow",
                VariableRef::PumpingFlow {
                    station_id: EntityId(0),
                    block_id: None,
                },
            ),
            (
                "PumpingPower",
                VariableRef::PumpingPower {
                    station_id: EntityId(0),
                    block_id: None,
                },
            ),
            (
                "ContractImport",
                VariableRef::ContractImport {
                    contract_id: EntityId(0),
                    block_id: None,
                },
            ),
            (
                "ContractExport",
                VariableRef::ContractExport {
                    contract_id: EntityId(0),
                    block_id: None,
                },
            ),
            (
                "NonControllableGeneration",
                VariableRef::NonControllableGeneration {
                    source_id: EntityId(0),
                    block_id: None,
                },
            ),
            (
                "NonControllableCurtailment",
                VariableRef::NonControllableCurtailment {
                    source_id: EntityId(0),
                    block_id: None,
                },
            ),
            (
                "AnticipatedDecision",
                VariableRef::AnticipatedDecision {
                    thermal_id: EntityId(0),
                },
            ),
            (
                "HydroInflow",
                VariableRef::HydroInflow {
                    hydro_id: EntityId(0),
                    block_id: None,
                },
            ),
            (
                "HydroStorageInitial",
                VariableRef::HydroStorageInitial {
                    hydro_id: EntityId(0),
                    block_id: Some(0),
                },
            ),
            (
                "HydroStorageFinal",
                VariableRef::HydroStorageFinal {
                    hydro_id: EntityId(0),
                    block_id: None,
                },
            ),
            (
                "HydroUsefulVolumeInitial",
                VariableRef::HydroUsefulVolumeInitial {
                    hydro_id: EntityId(0),
                    block_id: None,
                },
            ),
            (
                "HydroUsefulVolumeFinal",
                VariableRef::HydroUsefulVolumeFinal {
                    hydro_id: EntityId(0),
                    block_id: None,
                },
            ),
        ];

        assert_eq!(
            variants.len(),
            26,
            "VariableRef must have exactly 26 variants"
        );

        for (name, variant) in variants {
            let debug_str = format!("{variant:?}");
            assert!(
                debug_str.contains(name),
                "Debug output for {name} does not contain the variant name: {debug_str}"
            );
        }
    }

    /// Pins the postcard discriminants of the tail variants. Postcard encodes
    /// the variant index as a varint; for discriminants `< 0x80` the first byte
    /// equals the discriminant. `AnticipatedDecision` (index 20 = `0x14`),
    /// `HydroInflow` (index 21 = `0x15`), `HydroStorageInitial` (index 22 =
    /// `0x16`), `HydroStorageFinal` (index 23 = `0x17`), `HydroUsefulVolumeInitial`
    /// (index 24 = `0x18`), and `HydroUsefulVolumeFinal` (index 25 = `0x19`) must
    /// keep their indices — a mid-enum insertion would shift them and silently
    /// break previously serialized policies.
    #[cfg(feature = "serde")]
    #[test]
    fn test_variable_ref_postcard_discriminant_pin() {
        let hydro_inflow = postcard::to_allocvec(&VariableRef::HydroInflow {
            hydro_id: EntityId(0),
            block_id: None,
        })
        .expect("HydroInflow serializes");
        assert_eq!(
            hydro_inflow[0], 0x15,
            "HydroInflow must serialize to postcard discriminant 0x15"
        );

        let anticipated = postcard::to_allocvec(&VariableRef::AnticipatedDecision {
            thermal_id: EntityId(0),
        })
        .expect("AnticipatedDecision serializes");
        assert_eq!(
            anticipated[0], 0x14,
            "AnticipatedDecision must keep postcard discriminant 0x14"
        );

        let storage_initial = postcard::to_allocvec(&VariableRef::HydroStorageInitial {
            hydro_id: EntityId(0),
            block_id: None,
        })
        .expect("HydroStorageInitial serializes");
        assert_eq!(
            storage_initial[0], 0x16,
            "HydroStorageInitial must serialize to postcard discriminant 0x16"
        );

        let storage_final = postcard::to_allocvec(&VariableRef::HydroStorageFinal {
            hydro_id: EntityId(0),
            block_id: None,
        })
        .expect("HydroStorageFinal serializes");
        assert_eq!(
            storage_final[0], 0x17,
            "HydroStorageFinal must serialize to postcard discriminant 0x17"
        );

        let useful_volume_initial = postcard::to_allocvec(&VariableRef::HydroUsefulVolumeInitial {
            hydro_id: EntityId(0),
            block_id: None,
        })
        .expect("HydroUsefulVolumeInitial serializes");
        assert_eq!(
            useful_volume_initial[0], 0x18,
            "HydroUsefulVolumeInitial must serialize to postcard discriminant 0x18"
        );

        let useful_volume_final = postcard::to_allocvec(&VariableRef::HydroUsefulVolumeFinal {
            hydro_id: EntityId(0),
            block_id: None,
        })
        .expect("HydroUsefulVolumeFinal serializes");
        assert_eq!(
            useful_volume_final[0], 0x19,
            "HydroUsefulVolumeFinal must serialize to postcard discriminant 0x19"
        );
    }

    /// Full postcard round-trip for `HydroUsefulVolumeInitial` across both
    /// block selectors; `test_variable_ref_postcard_discriminant_pin` owns the
    /// discriminant byte.
    #[cfg(feature = "serde")]
    #[test]
    fn hydro_useful_volume_initial_postcard_roundtrip() {
        let stage_initial = VariableRef::HydroUsefulVolumeInitial {
            hydro_id: EntityId(7),
            block_id: None,
        };
        let bytes = postcard::to_allocvec(&stage_initial).expect("serialize");
        let recovered: VariableRef = postcard::from_bytes(&bytes).expect("deserialize");
        assert_eq!(stage_initial, recovered);

        let block = VariableRef::HydroUsefulVolumeInitial {
            hydro_id: EntityId(7),
            block_id: Some(2),
        };
        let bytes = postcard::to_allocvec(&block).expect("serialize");
        let recovered: VariableRef = postcard::from_bytes(&bytes).expect("deserialize");
        assert_eq!(block, recovered);
    }

    /// Full postcard round-trip for `HydroUsefulVolumeFinal` across both
    /// block selectors; `test_variable_ref_postcard_discriminant_pin` owns the
    /// discriminant byte.
    #[cfg(feature = "serde")]
    #[test]
    fn hydro_useful_volume_final_postcard_roundtrip() {
        let stage_final = VariableRef::HydroUsefulVolumeFinal {
            hydro_id: EntityId(7),
            block_id: None,
        };
        let bytes = postcard::to_allocvec(&stage_final).expect("serialize");
        let recovered: VariableRef = postcard::from_bytes(&bytes).expect("deserialize");
        assert_eq!(stage_final, recovered);

        let block = VariableRef::HydroUsefulVolumeFinal {
            hydro_id: EntityId(7),
            block_id: Some(2),
        };
        let bytes = postcard::to_allocvec(&block).expect("serialize");
        let recovered: VariableRef = postcard::from_bytes(&bytes).expect("deserialize");
        assert_eq!(block, recovered);
    }

    #[test]
    fn test_generic_constraint_construction() {
        let expr = ConstraintExpression {
            terms: vec![
                LinearTerm::literal(
                    1.0,
                    VariableRef::HydroGeneration {
                        hydro_id: EntityId(10),
                        block_id: None,
                        bus_id: None,
                    },
                ),
                LinearTerm::literal(
                    1.0,
                    VariableRef::HydroGeneration {
                        hydro_id: EntityId(11),
                        block_id: None,
                        bus_id: None,
                    },
                ),
            ],
        };

        let gc = GenericConstraint {
            id: EntityId(0),
            name: "min_southeast_hydro".to_string(),
            description: Some("Minimum hydro generation in Southeast region".to_string()),
            expression: expr,
            slack: SlackConfig {
                enabled: true,
                penalty: Some(5_000.0),
            },
            bound_lower_affine: None,
            bound_upper_affine: None,
        };

        assert_eq!(gc.expression.terms.len(), 2);
        assert_eq!(gc.id, EntityId(0));
        assert_eq!(gc.name, "min_southeast_hydro");
        assert!(gc.description.is_some());
        assert!(gc.slack.enabled);
        assert_eq!(gc.slack.penalty, Some(5_000.0));
    }

    #[test]
    fn canonicalize_orders_by_content_not_authoring_order() {
        let a = LinearTerm::literal(
            1.0,
            VariableRef::HydroGeneration {
                hydro_id: EntityId(1),
                block_id: None,
                bus_id: None,
            },
        );
        let b = LinearTerm::literal(
            1.0,
            VariableRef::HydroGeneration {
                hydro_id: EntityId(0),
                block_id: None,
                bus_id: None,
            },
        );
        let mut forward = ConstraintExpression {
            terms: vec![a.clone(), b.clone()],
        };
        let mut reversed = ConstraintExpression { terms: vec![b, a] };
        forward.canonicalize();
        reversed.canonicalize();
        assert_eq!(forward.terms, reversed.terms);
        assert_eq!(
            forward.terms[0].variable,
            VariableRef::HydroGeneration {
                hydro_id: EntityId(0),
                block_id: None,
                bus_id: None,
            }
        );
        assert_eq!(
            forward.terms[1].variable,
            VariableRef::HydroGeneration {
                hydro_id: EntityId(1),
                block_id: None,
                bus_id: None,
            }
        );
    }

    #[test]
    fn canonicalize_is_idempotent() {
        let mut expr = ConstraintExpression {
            terms: vec![
                LinearTerm::literal(
                    2.0,
                    VariableRef::ThermalGeneration {
                        thermal_id: EntityId(5),
                        block_id: None,
                    },
                ),
                LinearTerm::literal(
                    1.0,
                    VariableRef::HydroGeneration {
                        hydro_id: EntityId(3),
                        block_id: None,
                        bus_id: None,
                    },
                ),
                LinearTerm::parameter(
                    EntityId(7),
                    -1.0,
                    VariableRef::HydroStorage {
                        hydro_id: EntityId(2),
                    },
                ),
            ],
        };
        expr.canonicalize();
        let once = expr.terms.clone();
        expr.canonicalize();
        assert_eq!(expr.terms, once);
    }

    #[test]
    fn canonical_key_orders_literal_before_parameter() {
        let var = VariableRef::HydroGeneration {
            hydro_id: EntityId(4),
            block_id: None,
            bus_id: None,
        };
        let mut expr = ConstraintExpression {
            terms: vec![
                LinearTerm::parameter(EntityId(7), 1.0, var),
                LinearTerm::literal(1.0, var),
            ],
        };
        expr.canonicalize();
        assert_eq!(expr.terms[0].coefficient, CoefficientRef::Literal(1.0));
        assert!(matches!(
            expr.terms[1].coefficient,
            CoefficientRef::Parameter(_)
        ));
    }

    #[test]
    fn canonical_variable_key_is_injective_over_distinct_variants() {
        let vars = [
            VariableRef::HydroStorage {
                hydro_id: EntityId(0),
            },
            VariableRef::HydroGeneration {
                hydro_id: EntityId(0),
                block_id: None,
                bus_id: None,
            },
            VariableRef::HydroGeneration {
                hydro_id: EntityId(0),
                block_id: Some(0),
                bus_id: None,
            },
            VariableRef::HydroGeneration {
                hydro_id: EntityId(0),
                block_id: None,
                bus_id: Some(EntityId(0)),
            },
            VariableRef::ThermalGeneration {
                thermal_id: EntityId(0),
                block_id: None,
            },
            VariableRef::AnticipatedDecision {
                thermal_id: EntityId(0),
            },
            VariableRef::HydroStorageFinal {
                hydro_id: EntityId(0),
                block_id: None,
            },
            VariableRef::HydroStorageInitial {
                hydro_id: EntityId(0),
                block_id: None,
            },
            VariableRef::HydroUsefulVolumeInitial {
                hydro_id: EntityId(0),
                block_id: None,
            },
            VariableRef::HydroUsefulVolumeFinal {
                hydro_id: EntityId(0),
                block_id: None,
            },
        ];
        let mut keys: Vec<_> = vars.iter().map(canonical_variable_key).collect();
        let n = keys.len();
        keys.sort_unstable();
        keys.dedup();
        assert_eq!(
            keys.len(),
            n,
            "distinct variables must map to distinct canonical keys"
        );
    }

    #[test]
    fn test_slack_config_disabled_has_no_penalty() {
        let slack = SlackConfig {
            enabled: false,
            penalty: None,
        };
        assert!(!slack.enabled);
        assert!(slack.penalty.is_none());
    }

    fn lit(term: &LinearTerm) -> f64 {
        match term.coefficient {
            CoefficientRef::Literal(v) => v,
            CoefficientRef::Parameter(_) => panic!("expected literal"),
        }
    }

    #[test]
    fn test_linear_term_with_coefficient() {
        let term = LinearTerm::literal(
            2.5,
            VariableRef::ThermalGeneration {
                thermal_id: EntityId(5),
                block_id: None,
            },
        );
        assert!((lit(&term) - 2.5).abs() < f64::EPSILON);
        let debug = format!("{:?}", term.variable);
        assert!(debug.contains("ThermalGeneration"));
    }

    #[test]
    fn linear_term_literal_constructor() {
        let term = LinearTerm::literal(
            3.0,
            VariableRef::ThermalGeneration {
                thermal_id: EntityId(1),
                block_id: None,
            },
        );
        assert_eq!(term.coefficient, CoefficientRef::Literal(3.0));
        assert!((term.scale - 1.0).abs() < f64::EPSILON);
        assert_eq!(
            term.variable,
            VariableRef::ThermalGeneration {
                thermal_id: EntityId(1),
                block_id: None,
            }
        );
    }

    #[test]
    fn linear_term_explicit_scale() {
        let term = LinearTerm {
            coefficient: CoefficientRef::Literal(2.0),
            scale: 0.5,
            variable: VariableRef::HydroStorage {
                hydro_id: EntityId(1),
            },
        };
        assert!((term.scale - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_variable_ref_block_none_vs_some() {
        let all_blocks = VariableRef::HydroTurbined {
            hydro_id: EntityId(3),
            block_id: None,
            bus_id: None,
        };
        let specific_block = VariableRef::HydroTurbined {
            hydro_id: EntityId(3),
            block_id: Some(0),
            bus_id: None,
        };
        assert_ne!(all_blocks, specific_block);
    }

    #[test]
    fn anticipated_decision_constructs() {
        let v = VariableRef::AnticipatedDecision {
            thermal_id: EntityId(5),
        };
        let debug_str = format!("{v:?}");
        assert!(
            debug_str.contains("AnticipatedDecision"),
            "Debug output should contain variant name: {debug_str}"
        );
        assert!(
            debug_str.contains("thermal_id"),
            "Debug output should contain field name: {debug_str}"
        );
    }

    #[test]
    fn anticipated_decision_copy_and_eq() {
        let v1 = VariableRef::AnticipatedDecision {
            thermal_id: EntityId(5),
        };
        let v2 = v1;
        assert_eq!(v1, v2);

        let v3 = VariableRef::AnticipatedDecision {
            thermal_id: EntityId(9),
        };
        assert_ne!(v1, v3);
    }

    /// Postcard round-trip preserves `AnticipatedDecision`, and pins its
    /// discriminant byte so no variant can be inserted before it without
    /// breaking this assertion (postcard encodes variants by declaration order).
    #[cfg(feature = "serde")]
    #[test]
    fn anticipated_decision_postcard_roundtrip() {
        let original = VariableRef::AnticipatedDecision {
            thermal_id: EntityId(5),
        };
        let bytes = postcard::to_allocvec(&original).expect("serialize");
        let recovered: VariableRef = postcard::from_bytes(&bytes).expect("deserialize");
        assert_eq!(
            original, recovered,
            "postcard round-trip must preserve the variant"
        );

        assert_eq!(
            bytes[0], 20,
            "AnticipatedDecision must be discriminant 20 (end-of-enum); \
             got {}. Did you insert a variant before it?",
            bytes[0]
        );
    }

    /// Regression: no insertion before `NonControllableCurtailment` shifted its
    /// discriminant (index 19).
    #[cfg(feature = "serde")]
    #[test]
    fn non_controllable_curtailment_discriminant_is_19() {
        let v = VariableRef::NonControllableCurtailment {
            source_id: EntityId(0),
            block_id: None,
        };
        let bytes = postcard::to_allocvec(&v).expect("serialize");
        assert_eq!(
            bytes[0], 19,
            "NonControllableCurtailment must remain discriminant 19; \
             got {}. A variant was inserted before it.",
            bytes[0]
        );
    }

    /// A `Some(b)` bus selector round-trips through postcard on both
    /// bus-selectable variants — a field addition to an existing variant, not a
    /// new variant, so it carries no discriminant of its own to pin.
    #[cfg(feature = "serde")]
    #[test]
    fn variable_ref_bus_selector_postcard_roundtrip() {
        let turbined = VariableRef::HydroTurbined {
            hydro_id: EntityId(7),
            block_id: Some(1),
            bus_id: Some(EntityId(3)),
        };
        let bytes = postcard::to_allocvec(&turbined).expect("serialize");
        let recovered: VariableRef = postcard::from_bytes(&bytes).expect("deserialize");
        assert_eq!(turbined, recovered);

        let generation = VariableRef::HydroGeneration {
            hydro_id: EntityId(7),
            block_id: None,
            bus_id: Some(EntityId(3)),
        };
        let bytes = postcard::to_allocvec(&generation).expect("serialize");
        let recovered: VariableRef = postcard::from_bytes(&bytes).expect("deserialize");
        assert_eq!(generation, recovered);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_generic_constraint_serde_roundtrip() {
        let gc = GenericConstraint {
            id: EntityId(0),
            name: "test".to_string(),
            description: None,
            expression: ConstraintExpression {
                terms: vec![
                    LinearTerm::literal(
                        1.0,
                        VariableRef::HydroGeneration {
                            hydro_id: EntityId(10),
                            block_id: None,
                            bus_id: None,
                        },
                    ),
                    LinearTerm::literal(
                        1.0,
                        VariableRef::HydroGeneration {
                            hydro_id: EntityId(11),
                            block_id: None,
                            bus_id: None,
                        },
                    ),
                ],
            },
            slack: SlackConfig {
                enabled: true,
                penalty: Some(5_000.0),
            },
            bound_lower_affine: None,
            bound_upper_affine: Some(AffineBound::single(EntityId(42))),
        };

        let json = serde_json::to_string(&gc).unwrap();
        let deserialized: GenericConstraint = serde_json::from_str(&json).unwrap();
        assert_eq!(gc, deserialized);
        assert_eq!(deserialized.expression.terms.len(), 2);
        assert_eq!(
            deserialized.bound_upper_affine,
            Some(AffineBound::single(EntityId(42)))
        );
    }

    #[test]
    fn affine_bound_single_is_zero_constant_plus_one_term() {
        let bound = AffineBound::single(EntityId(7));
        assert_eq!(bound.constant, 0.0);
        assert_eq!(bound.terms, vec![(1.0, EntityId(7))]);
    }

    #[test]
    fn affine_bound_params_iterates_term_ids() {
        let bound = AffineBound {
            constant: 12.0,
            terms: vec![(0.5, EntityId(7)), (-2.0, EntityId(9))],
        };
        assert_eq!(
            bound.params().collect::<Vec<_>>(),
            vec![EntityId(7), EntityId(9)]
        );
    }

    /// The byte-neutral load-bearing identity for `AffineBound::single`: any
    /// resolved parameter value `x` passed through `single`'s
    /// `constant + coef * x` shape returns `x` unchanged in `f64` — the
    /// arithmetic fact that lets a single-ref remainder stand in for the
    /// bare parameter reference it replaces.
    #[test]
    fn affine_bound_single_resolves_to_the_bare_value() {
        for x in [0.0_f64, 1.0, -3.5, 42.75, f64::MIN_POSITIVE, 1e300] {
            let bound = AffineBound::single(EntityId(1));
            let (coef, _) = bound.terms[0];
            assert_eq!(bound.constant + coef * x, x);
        }
    }
}
