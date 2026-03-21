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
//! [`VariableRef`] covers all 20 LP variable types defined in the spec (SS15).
//! Each variant carries the entity ID and, for block-specific variables, an
//! optional block ID (`None` = sum over all blocks, `Some(i)` = block `i`).
//!
//! # Examples
//!
//! ```
//! use cobre_core::{
//!     EntityId, GenericConstraint, ConstraintExpression, ConstraintSense,
//!     LinearTerm, SlackConfig, VariableRef,
//! };
//!
//! // Represents: hydro_generation(10) + hydro_generation(11)
//! let expr = ConstraintExpression {
//!     terms: vec![
//!         LinearTerm {
//!             coefficient: 1.0,
//!             variable: VariableRef::HydroGeneration {
//!                 hydro_id: EntityId(10),
//!                 block_id: None,
//!             },
//!         },
//!         LinearTerm {
//!             coefficient: 1.0,
//!             variable: VariableRef::HydroGeneration {
//!                 hydro_id: EntityId(11),
//!                 block_id: None,
//!             },
//!         },
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
//!     sense: ConstraintSense::GreaterEqual,
//!     slack: SlackConfig { enabled: true, penalty: Some(5_000.0) },
//! };
//!
//! assert_eq!(gc.expression.terms.len(), 2);
//! ```

use crate::EntityId;

/// Reference to a single LP variable in a generic constraint expression.
///
/// Each variant names the variable type and carries the entity ID. For
/// block-specific variables, `block_id` is `None` to sum over all blocks or
/// `Some(i)` to reference block `i` specifically.
///
/// The 20 variants cover the full variable catalog defined in
/// `internal-structures.md §15` (table in the "Variable References" section).
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
        /// Block index. `None` = sum over all blocks; `Some(i)` = block `i`.
        block_id: Option<usize>,
    },
    /// Spillage flow for a hydro plant (m³/s).
    HydroSpillage {
        /// Hydro plant identifier.
        hydro_id: EntityId,
        /// Block index. `None` = sum over all blocks; `Some(i)` = block `i`.
        block_id: Option<usize>,
    },
    /// Diversion flow for a hydro plant (m³/s). Only valid for hydros with diversion.
    HydroDiversion {
        /// Hydro plant identifier.
        hydro_id: EntityId,
        /// Block index. `None` = sum over all blocks; `Some(i)` = block `i`.
        block_id: Option<usize>,
    },
    /// Total outflow (turbined + spillage) for a hydro plant (m³/s).
    ///
    /// Currently an alias for turbined + spillage. Future CEPEL formulations
    /// may turn this into an independent variable.
    HydroOutflow {
        /// Hydro plant identifier.
        hydro_id: EntityId,
        /// Block index. `None` = sum over all blocks; `Some(i)` = block `i`.
        block_id: Option<usize>,
    },
    /// Electrical generation from a hydro plant (MW).
    HydroGeneration {
        /// Hydro plant identifier.
        hydro_id: EntityId,
        /// Block index. `None` = sum over all blocks; `Some(i)` = block `i`.
        block_id: Option<usize>,
    },
    /// Evaporation flow from a hydro reservoir (m³/s). Stage-level, not block-specific.
    HydroEvaporation {
        /// Hydro plant identifier.
        hydro_id: EntityId,
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
        /// Block index. `None` = sum over all blocks; `Some(i)` = block `i`.
        block_id: Option<usize>,
    },
    /// Direct (forward) power flow on a transmission line (MW).
    LineDirect {
        /// Transmission line identifier.
        line_id: EntityId,
        /// Block index. `None` = sum over all blocks; `Some(i)` = block `i`.
        block_id: Option<usize>,
    },
    /// Reverse power flow on a transmission line (MW).
    LineReverse {
        /// Transmission line identifier.
        line_id: EntityId,
        /// Block index. `None` = sum over all blocks; `Some(i)` = block `i`.
        block_id: Option<usize>,
    },
    /// Net exchange flow on a transmission line (direct - reverse) (MW).
    ///
    /// This is a derived variable: the resolver maps it to two LP columns
    /// (forward flow with +1.0 and reverse flow with -1.0), representing
    /// net flow in the source-to-target direction.
    LineExchange {
        /// Transmission line identifier.
        line_id: EntityId,
        /// Block index. `None` = sum over all blocks; `Some(i)` = block `i`.
        block_id: Option<usize>,
    },
    /// Load deficit (unserved energy) at a bus (MW).
    BusDeficit {
        /// Bus identifier.
        bus_id: EntityId,
        /// Block index. `None` = sum over all blocks; `Some(i)` = block `i`.
        block_id: Option<usize>,
    },
    /// Load excess (over-generation) at a bus (MW).
    BusExcess {
        /// Bus identifier.
        bus_id: EntityId,
        /// Block index. `None` = sum over all blocks; `Some(i)` = block `i`.
        block_id: Option<usize>,
    },
    /// Pumped water flow at a pumping station (m³/s).
    PumpingFlow {
        /// Pumping station identifier.
        station_id: EntityId,
        /// Block index. `None` = sum over all blocks; `Some(i)` = block `i`.
        block_id: Option<usize>,
    },
    /// Electrical power consumed by a pumping station (MW).
    PumpingPower {
        /// Pumping station identifier.
        station_id: EntityId,
        /// Block index. `None` = sum over all blocks; `Some(i)` = block `i`.
        block_id: Option<usize>,
    },
    /// Energy imported via a contract (MW).
    ContractImport {
        /// Energy contract identifier.
        contract_id: EntityId,
        /// Block index. `None` = sum over all blocks; `Some(i)` = block `i`.
        block_id: Option<usize>,
    },
    /// Energy exported via a contract (MW).
    ContractExport {
        /// Energy contract identifier.
        contract_id: EntityId,
        /// Block index. `None` = sum over all blocks; `Some(i)` = block `i`.
        block_id: Option<usize>,
    },
    /// Generation from a non-controllable source (wind, solar, etc.) (MW).
    NonControllableGeneration {
        /// Non-controllable source identifier.
        source_id: EntityId,
        /// Block index. `None` = sum over all blocks; `Some(i)` = block `i`.
        block_id: Option<usize>,
    },
    /// Curtailment of a non-controllable source (MW).
    NonControllableCurtailment {
        /// Non-controllable source identifier.
        source_id: EntityId,
        /// Block index. `None` = sum over all blocks; `Some(i)` = block `i`.
        block_id: Option<usize>,
    },
}

/// One term in a linear constraint expression: `coefficient * variable`.
///
/// The expression is `coefficient × variable_ref`. A coefficient of `1.0`
/// represents an unweighted variable reference.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LinearTerm {
    /// Scalar multiplier for the variable reference.
    pub coefficient: f64,
    /// The LP variable being referenced.
    pub variable: VariableRef,
}

/// Parsed linear constraint expression.
///
/// Represents the left-hand side of a generic constraint as a list of weighted
/// variable references. An empty `terms` vector is valid (constant-only
/// expression, unusual but not rejected at this layer).
///
/// The expression parser (string → `ConstraintExpression`) lives in `cobre-io`.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ConstraintExpression {
    /// Ordered list of linear terms that form the left-hand side of the constraint.
    pub terms: Vec<LinearTerm>,
}

/// Comparison sense for a generic constraint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ConstraintSense {
    /// The expression must be greater than or equal to the bound (`>=`).
    GreaterEqual,
    /// The expression must be less than or equal to the bound (`<=`).
    LessEqual,
    /// The expression must be exactly equal to the bound (`==`).
    Equal,
}

/// Slack variable configuration for a generic constraint.
///
/// When `enabled` is `true`, a slack variable is added to the LP so that the
/// constraint can be violated at a cost. This prevents infeasibility when
/// bounds are tight or conflicting. The penalty cost enters the LP objective
/// function.
///
/// `penalty` must be `Some(value)` with a positive value when `enabled` is
/// `true`, and `None` when `enabled` is `false`.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SlackConfig {
    /// Whether a slack variable is added to allow soft violation of the constraint.
    pub enabled: bool,
    /// Penalty cost per unit of constraint violation. `None` when `enabled` is `false`.
    pub penalty: Option<f64>,
}

/// A user-defined generic linear constraint.
///
/// Stored in [`crate::System::generic_constraints`] after loading and
/// validation. Constraints are sorted by `id` after loading to satisfy the
/// declaration-order invariance requirement.
///
/// The expression parser, referential validation (entity IDs exist), and
/// bounds loading (from `generic_constraint_bounds.parquet`) are all
/// performed by `cobre-io`, not here.
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
    /// Comparison sense (`>=`, `<=`, or `==`).
    pub sense: ConstraintSense,
    /// Slack variable configuration.
    pub slack: SlackConfig,
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
                },
            ),
            (
                "HydroEvaporation",
                VariableRef::HydroEvaporation {
                    hydro_id: EntityId(0),
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
        ];

        assert_eq!(
            variants.len(),
            20,
            "VariableRef must have exactly 20 variants"
        );

        for (name, variant) in variants {
            let debug_str = format!("{variant:?}");
            assert!(
                debug_str.contains(name),
                "Debug output for {name} does not contain the variant name: {debug_str}"
            );
        }
    }

    #[test]
    fn test_generic_constraint_construction() {
        let expr = ConstraintExpression {
            terms: vec![
                LinearTerm {
                    coefficient: 1.0,
                    variable: VariableRef::HydroGeneration {
                        hydro_id: EntityId(10),
                        block_id: None,
                    },
                },
                LinearTerm {
                    coefficient: 1.0,
                    variable: VariableRef::HydroGeneration {
                        hydro_id: EntityId(11),
                        block_id: None,
                    },
                },
            ],
        };

        let gc = GenericConstraint {
            id: EntityId(0),
            name: "min_southeast_hydro".to_string(),
            description: Some("Minimum hydro generation in Southeast region".to_string()),
            expression: expr,
            sense: ConstraintSense::GreaterEqual,
            slack: SlackConfig {
                enabled: true,
                penalty: Some(5_000.0),
            },
        };

        assert_eq!(gc.expression.terms.len(), 2);
        assert_eq!(gc.id, EntityId(0));
        assert_eq!(gc.name, "min_southeast_hydro");
        assert!(gc.description.is_some());
        assert_eq!(gc.sense, ConstraintSense::GreaterEqual);
        assert!(gc.slack.enabled);
        assert_eq!(gc.slack.penalty, Some(5_000.0));
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

    #[test]
    fn test_constraint_sense_variants() {
        assert_ne!(ConstraintSense::GreaterEqual, ConstraintSense::LessEqual);
        assert_ne!(ConstraintSense::GreaterEqual, ConstraintSense::Equal);
        assert_ne!(ConstraintSense::LessEqual, ConstraintSense::Equal);
    }

    #[test]
    fn test_linear_term_with_coefficient() {
        let term = LinearTerm {
            coefficient: 2.5,
            variable: VariableRef::ThermalGeneration {
                thermal_id: EntityId(5),
                block_id: None,
            },
        };
        assert_eq!(term.coefficient, 2.5);
        let debug = format!("{:?}", term.variable);
        assert!(debug.contains("ThermalGeneration"));
    }

    #[test]
    fn test_variable_ref_block_none_vs_some() {
        let all_blocks = VariableRef::HydroTurbined {
            hydro_id: EntityId(3),
            block_id: None,
        };
        let specific_block = VariableRef::HydroTurbined {
            hydro_id: EntityId(3),
            block_id: Some(0),
        };
        assert_ne!(all_blocks, specific_block);
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
                    LinearTerm {
                        coefficient: 1.0,
                        variable: VariableRef::HydroGeneration {
                            hydro_id: EntityId(10),
                            block_id: None,
                        },
                    },
                    LinearTerm {
                        coefficient: 1.0,
                        variable: VariableRef::HydroGeneration {
                            hydro_id: EntityId(11),
                            block_id: None,
                        },
                    },
                ],
            },
            sense: ConstraintSense::GreaterEqual,
            slack: SlackConfig {
                enabled: true,
                penalty: Some(5_000.0),
            },
        };

        let json = serde_json::to_string(&gc).unwrap();
        let deserialized: GenericConstraint = serde_json::from_str(&json).unwrap();
        assert_eq!(gc, deserialized);
        assert_eq!(deserialized.expression.terms.len(), 2);
    }
}
