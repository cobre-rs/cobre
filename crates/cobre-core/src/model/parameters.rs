//! Scalar parameter types for user-defined and computed coefficients.
//!
//! This module defines the in-memory representation of scalar parameters that
//! users or the system can attach to entities and reference from constraint
//! expressions. A [`ScalarParameter`] bundles a unique identifier, a human-readable
//! name, and a [`ParameterKind`] that describes how the numeric value is
//! determined at solve time.
//!
//! The parameter system is deliberately stratified:
//!
//! - **Resolution** (mapping kinds to concrete `f64` values) belongs to a
//!   dedicated resolver layer in the solver crate.
//! - **Loading** (reading the JSON file, validating IDs and lengths) belongs to
//!   the I/O layer.
//! - **Consumption** (substituting resolved values into LP rows) belongs to the
//!   LP builder in the solver crate.
//!
//! This module defines only the structural types — it contains no I/O, no
//! resolution logic, and no LP wiring.

use crate::EntityId;

#[cfg(feature = "serde")]
use serde::Deserializer;

/// Reference to a scalar coefficient in a linear term.
///
/// Allows constraint coefficients to be either a literal `f64` value known at
/// input time, or a named parameter whose value is resolved later. The `Parameter`
/// variant carries the [`EntityId`] of a [`ScalarParameter`] stored in the
/// study's parameter collection.
///
/// # Examples
///
/// ```
/// use cobre_core::{CoefficientRef, EntityId};
///
/// // A literal coefficient of 3.6:
/// let literal = CoefficientRef::Literal(3.6);
///
/// // A parameter-backed coefficient referencing parameter ID 42:
/// let param = CoefficientRef::Parameter(EntityId(42));
///
/// // CoefficientRef is Copy, so it can be used after moving:
/// let a = CoefficientRef::Literal(1.0);
/// let b = a;
/// assert_eq!(a, b);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum CoefficientRef {
    /// A constant scalar value known at input parse time.
    Literal(f64),
    /// An indirect reference to a [`ScalarParameter`] by its entity ID.
    Parameter(EntityId),
}

/// A Cobre-computed quantity indexed by hydro plant.
///
/// Each variant names a scalar quantity that the resolver derives from hydro
/// geometry and operational data. All variants carry a single `hydro_id` field
/// identifying the hydro plant.
///
/// # Examples
///
/// ```
/// use cobre_core::{ComputedParameter, EntityId};
///
/// let param = ComputedParameter::EquivalentProductivity {
///     hydro_id: EntityId(1),
/// };
///
/// // ComputedParameter is Copy:
/// let copy = param;
/// assert_eq!(param, copy);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(tag = "tag", rename_all = "snake_case"))]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "schema",
    schemars(description = "A quantity computed from hydro data for the plant `hydro_id`.")
)]
pub enum ComputedParameter {
    /// Equivalent productivity coefficient (`ρ_eq`).
    EquivalentProductivity {
        /// Hydro plant identifier.
        hydro_id: EntityId,
    },
    /// Accumulated productivity coefficient (`ρ_acum`).
    AccumulatedProductivity {
        /// Hydro plant identifier.
        hydro_id: EntityId,
    },
    /// Reference reservoir volume (`V_ref`).
    ReferenceVolume {
        /// Hydro plant identifier.
        hydro_id: EntityId,
    },
    /// Reference turbine flow (`q_ref`).
    ReferenceTurbine {
        /// Hydro plant identifier.
        hydro_id: EntityId,
    },
    /// Minimum operational storage (`V_min`).
    MinStorage {
        /// Hydro plant identifier.
        hydro_id: EntityId,
    },
    /// Maximum operational storage (`V_max`).
    MaxStorage {
        /// Hydro plant identifier.
        hydro_id: EntityId,
    },
    /// Specific productivity (`ρ_esp`).
    SpecificProductivity {
        /// Hydro plant identifier.
        hydro_id: EntityId,
    },
    /// Equivalent productivity evaluated as the mean over the plant's storage
    /// range rather than at the reference volume (own scope).
    IntegratedEquivalentProductivity {
        /// Hydro plant identifier.
        hydro_id: EntityId,
    },
    /// Accumulated productivity evaluated as the mean over the plant's storage
    /// range rather than at the reference volume (cascade scope).
    IntegratedAccumulatedProductivity {
        /// Hydro plant identifier.
        hydro_id: EntityId,
    },
    /// Maximum stored energy: `integrated_accumulated_productivity * (V_hi -
    /// V_lo)` over the hydro plant's physical storage range (stage-invariant).
    /// Unit is productivity `[MW/(m3/s)]` times volume `[hm3]` — the same unit
    /// as a productivity-times-storage-volume constraint term, not `MWh`.
    MaxStoredEnergy {
        /// Hydro plant identifier.
        hydro_id: EntityId,
    },
}

/// How the numeric value of a [`ScalarParameter`] is determined at solve time.
///
/// The variants cover the full range from compile-time constants to values
/// computed from physical plant data, plus a `(stage, block)`-indexed table:
///
/// - [`Constant`](ParameterKind::Constant) — one value for all stages.
/// - [`PerStage`](ParameterKind::PerStage) — one value per study stage;
///   the resolver validates `len() == n_stages`.
/// - [`Seasonal`](ParameterKind::Seasonal) — one value per season, keyed by
///   `season_id` (`i32`). Use [`ParameterKind::new_seasonal`] to construct
///   this variant with the sort-and-dedup invariant enforced.
/// - [`Computed`](ParameterKind::Computed) — derived by the resolver from
///   hydro geometry data; no explicit user value is required.
/// - [`PerStageBlock`](ParameterKind::PerStageBlock) — one value per
///   `(stage_id, block_id)` pair, keys unique and stored sorted.
///
/// # JSON schema
///
/// Serialization uses an internally-tagged JSON form. Each variant produces a
/// `"kind"` discriminant field alongside its payload fields:
///
/// ```
/// # #[cfg(feature = "serde")] {
/// use cobre_core::{ComputedParameter, EntityId, ParameterKind};
///
/// // {"kind":"constant","value":3.6}
/// let c = ParameterKind::Constant { value: 3.6 };
/// assert_eq!(
///     serde_json::to_string(&c).unwrap(),
///     r#"{"kind":"constant","value":3.6}"#
/// );
///
/// // {"kind":"per_stage","values":[[0,1.0],[1,1.1],[2,0.9]]}
/// let ps = ParameterKind::PerStage { values: vec![1.0, 1.1, 0.9] };
/// assert_eq!(
///     serde_json::to_string(&ps).unwrap(),
///     r#"{"kind":"per_stage","values":[[0,1.0],[1,1.1],[2,0.9]]}"#
/// );
///
/// // {"kind":"seasonal","values":[[1,0.5],[2,1.5]]}
/// let s = ParameterKind::Seasonal { values: vec![(1, 0.5), (2, 1.5)] };
/// assert_eq!(
///     serde_json::to_string(&s).unwrap(),
///     r#"{"kind":"seasonal","values":[[1,0.5],[2,1.5]]}"#
/// );
///
/// // {"kind":"computed","computed_spec":{"tag":"equivalent_productivity","hydro_id":7}}
/// let comp = ParameterKind::Computed {
///     computed_spec: ComputedParameter::EquivalentProductivity { hydro_id: EntityId(7) },
/// };
/// assert_eq!(
///     serde_json::to_string(&comp).unwrap(),
///     r#"{"kind":"computed","computed_spec":{"tag":"equivalent_productivity","hydro_id":7}}"#
/// );
///
/// // {"kind":"per_stage_block","values":[[0,0,1.0],[0,1,2.0]]}
/// let psb = ParameterKind::PerStageBlock { values: vec![(0, 0, 1.0), (0, 1, 2.0)] };
/// assert_eq!(
///     serde_json::to_string(&psb).unwrap(),
///     r#"{"kind":"per_stage_block","values":[[0,0,1.0],[0,1,2.0]]}"#
/// );
/// # }
/// ```
///
/// Every variant round-trips through `serde_json::from_str` back to the same
/// in-memory value.
///
/// # Examples
///
/// ```
/// use cobre_core::{ComputedParameter, EntityId, ParameterKind};
///
/// let constant = ParameterKind::Constant { value: 3.6 };
/// let per_stage = ParameterKind::PerStage { values: vec![1.0, 2.0, 3.0] };
/// let seasonal = ParameterKind::new_seasonal(vec![(2, 1.5), (1, 0.5)]);
/// let computed = ParameterKind::Computed {
///     computed_spec: ComputedParameter::EquivalentProductivity {
///         hydro_id: EntityId(1),
///     },
/// };
///
/// // Seasonal entries are sorted ascending by season_id:
/// assert_eq!(
///     seasonal,
///     ParameterKind::Seasonal { values: vec![(1, 0.5), (2, 1.5)] }
/// );
/// ```
// Deserialize is manual, not derived — see the impl below for why.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[cfg_attr(feature = "serde", serde(into = "ParameterKindJson"))]
pub enum ParameterKind {
    /// A single scalar value applied to every stage.
    Constant {
        /// The scalar value for all stages.
        value: f64,
    },
    /// One scalar value per study stage; length must equal `n_stages`.
    PerStage {
        /// Dense array of values indexed by stage (0-based).
        values: Vec<f64>,
    },
    /// One scalar value per season, keyed by `season_id` (`i32`).
    ///
    /// Entries are stored sorted ascending by `season_id` with unique keys.
    /// Construct via [`ParameterKind::new_seasonal`] to enforce this invariant,
    /// or supply a pre-sorted, deduplicated vector directly.
    Seasonal {
        /// Sorted, deduplicated `(season_id, value)` pairs.
        values: Vec<(i32, f64)>,
    },
    /// A value derived from physical plant data by the resolver.
    Computed {
        /// The computed quantity specification.
        computed_spec: ComputedParameter,
    },
    /// One scalar value per `(stage_id, block_id)` pair.
    ///
    /// Entries are stored sorted ascending by `(stage_id, block_id)` with unique
    /// keys. A flat triple list is used rather than a nested `[stage][block]`
    /// array because block counts vary per stage, so no rectangular array
    /// expresses every study.
    PerStageBlock {
        /// Sorted, unique-keyed `(stage_id, block_id, value)` triples.
        values: Vec<(i32, i32, f64)>,
    },
}

impl ParameterKind {
    /// Construct a [`ParameterKind::Seasonal`] from an unsorted, possibly
    /// duplicate list of `(season_id, value)` pairs.
    ///
    /// The constructor sorts the pairs ascending by `season_id` and removes
    /// duplicate keys, keeping the **first** occurrence of each key. Downstream
    /// loaders are responsible for reporting duplicate-key errors to the user;
    /// this constructor only guarantees the invariant that the internal
    /// `Vec<(i32, f64)>` is sorted and contains unique keys.
    ///
    /// # Examples
    ///
    /// ```
    /// use cobre_core::ParameterKind;
    ///
    /// // Duplicates: key 1 appears twice; first occurrence (0.5) is kept.
    /// let seasonal = ParameterKind::new_seasonal(vec![(3, 1.5), (1, 0.5), (1, 0.9), (2, 1.0)]);
    /// assert_eq!(
    ///     seasonal,
    ///     ParameterKind::Seasonal { values: vec![(1, 0.5), (2, 1.0), (3, 1.5)] }
    /// );
    /// ```
    #[must_use]
    pub fn new_seasonal(mut pairs: Vec<(i32, f64)>) -> Self {
        pairs.sort_by_key(|(k, _)| *k);
        pairs.dedup_by_key(|(k, _)| *k);
        Self::Seasonal { values: pairs }
    }
}

/// A named scalar parameter whose value is resolved before LP construction.
///
/// Parameters are identified by a unique [`EntityId`] and a human-readable
/// `name`. The `kind` field describes how the numeric value is determined at
/// solve time (see [`ParameterKind`]).
///
/// Parameters are loaded from the JSON case file by the I/O layer and resolved
/// to concrete `f64` values by the resolver before the LP builder consumes them.
///
/// # Examples
///
/// ```
/// use cobre_core::{EntityId, ParameterKind, ScalarParameter};
///
/// let param = ScalarParameter {
///     id: EntityId(1),
///     name: "rho_eq_h1".to_string(),
///     kind: ParameterKind::Constant { value: 3.6 },
/// };
///
/// assert_eq!(param.id, EntityId(1));
/// assert_eq!(param.name, "rho_eq_h1");
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ScalarParameter {
    /// Unique parameter identifier.
    pub id: EntityId,
    /// Short name used in reports and log output.
    pub name: String,
    /// How the numeric value of this parameter is determined at solve time.
    pub kind: ParameterKind,
}

// Serde intermediate: `PerStage` is a dense `Vec<f64>` in memory but
// `[[stage_id, value], ...]` on the wire; the other variants are shape-identical.

#[cfg(feature = "serde")]
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum ParameterKindJson {
    Constant { value: f64 },
    PerStage { values: Vec<(i32, f64)> },
    Seasonal { values: Vec<(i32, f64)> },
    Computed { computed_spec: ComputedParameter },
    PerStageBlock { values: Vec<(i32, i32, f64)> },
}

#[cfg(feature = "serde")]
impl From<ParameterKind> for ParameterKindJson {
    fn from(kind: ParameterKind) -> Self {
        match kind {
            ParameterKind::Constant { value } => ParameterKindJson::Constant { value },
            ParameterKind::PerStage { values } => ParameterKindJson::PerStage {
                values: values
                    .into_iter()
                    .enumerate()
                    .map(|(i, v)| {
                        // saturate rather than panic on the impossible >i32::MAX stage count
                        (i32::try_from(i).unwrap_or(i32::MAX), v)
                    })
                    .collect(),
            },
            ParameterKind::Seasonal { values } => ParameterKindJson::Seasonal { values },
            ParameterKind::Computed { computed_spec } => {
                ParameterKindJson::Computed { computed_spec }
            }
            ParameterKind::PerStageBlock { values } => ParameterKindJson::PerStageBlock { values },
        }
    }
}

// Manual (not `#[serde(from = "ParameterKindJson")]`): infallible `From` cannot
// surface the PerStage contiguity/duplicate-key error this validation requires.

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for ParameterKind {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let json = ParameterKindJson::deserialize(deserializer)?;
        match json {
            ParameterKindJson::Constant { value } => Ok(ParameterKind::Constant { value }),
            ParameterKindJson::PerStage { mut values } => {
                values.sort_by_key(|(k, _)| *k);

                for window in values.windows(2) {
                    if window[0].0 == window[1].0 {
                        return Err(serde::de::Error::custom(format!(
                            "duplicate stage_id {} in per_stage values",
                            window[0].0
                        )));
                    }
                }

                for (expected, (actual, _)) in values.iter().enumerate() {
                    let expected_i32 = i32::try_from(expected).unwrap_or(i32::MAX);
                    if *actual != expected_i32 {
                        return Err(serde::de::Error::custom(format!(
                            "per_stage values must have contiguous stage_ids starting at 0; \
                             expected stage_id {expected_i32} but got {actual}"
                        )));
                    }
                }

                let dense: Vec<f64> = values.into_iter().map(|(_, v)| v).collect();
                Ok(ParameterKind::PerStage { values: dense })
            }
            ParameterKindJson::Seasonal { values } => {
                // new_seasonal silently dedups; reject duplicate season_ids first.
                let mut seen: std::collections::HashSet<i32> = std::collections::HashSet::new();
                for &(season_id, _) in &values {
                    if !seen.insert(season_id) {
                        return Err(serde::de::Error::custom(format!(
                            "duplicate season_id {season_id} in seasonal values"
                        )));
                    }
                }
                Ok(ParameterKind::new_seasonal(values))
            }
            ParameterKindJson::Computed { computed_spec } => {
                Ok(ParameterKind::Computed { computed_spec })
            }
            ParameterKindJson::PerStageBlock { mut values } => {
                values.sort_by_key(|&(stage_id, block_id, _)| (stage_id, block_id));

                for window in values.windows(2) {
                    if window[0].0 == window[1].0 && window[0].1 == window[1].1 {
                        return Err(serde::de::Error::custom(format!(
                            "duplicate (stage_id, block_id) pair ({}, {}) in per_stage_block values",
                            window[0].0, window[0].1
                        )));
                    }
                }

                Ok(ParameterKind::PerStageBlock { values })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "serde")]
    use super::ScalarParameter;
    use super::{CoefficientRef, ComputedParameter, EntityId, ParameterKind};

    #[test]
    fn computed_parameter_variant_coverage_is_exhaustive() {
        let variants = [
            ComputedParameter::EquivalentProductivity {
                hydro_id: EntityId(1),
            },
            ComputedParameter::AccumulatedProductivity {
                hydro_id: EntityId(2),
            },
            ComputedParameter::ReferenceVolume {
                hydro_id: EntityId(3),
            },
            ComputedParameter::ReferenceTurbine {
                hydro_id: EntityId(4),
            },
            ComputedParameter::MinStorage {
                hydro_id: EntityId(5),
            },
            ComputedParameter::MaxStorage {
                hydro_id: EntityId(6),
            },
            ComputedParameter::SpecificProductivity {
                hydro_id: EntityId(7),
            },
            ComputedParameter::IntegratedEquivalentProductivity {
                hydro_id: EntityId(8),
            },
            ComputedParameter::IntegratedAccumulatedProductivity {
                hydro_id: EntityId(9),
            },
            ComputedParameter::MaxStoredEnergy {
                hydro_id: EntityId(10),
            },
        ];

        assert_eq!(
            variants.len(),
            10,
            "ComputedParameter must have exactly 10 variants"
        );

        // No `_` arm: adding a variant without updating here is a compile error.
        for variant in &variants {
            let _name = match variant {
                ComputedParameter::EquivalentProductivity { .. } => "EquivalentProductivity",
                ComputedParameter::AccumulatedProductivity { .. } => "AccumulatedProductivity",
                ComputedParameter::ReferenceVolume { .. } => "ReferenceVolume",
                ComputedParameter::ReferenceTurbine { .. } => "ReferenceTurbine",
                ComputedParameter::MinStorage { .. } => "MinStorage",
                ComputedParameter::MaxStorage { .. } => "MaxStorage",
                ComputedParameter::SpecificProductivity { .. } => "SpecificProductivity",
                ComputedParameter::IntegratedEquivalentProductivity { .. } => {
                    "IntegratedEquivalentProductivity"
                }
                ComputedParameter::IntegratedAccumulatedProductivity { .. } => {
                    "IntegratedAccumulatedProductivity"
                }
                ComputedParameter::MaxStoredEnergy { .. } => "MaxStoredEnergy",
            };
        }
    }

    #[test]
    fn parameter_kind_five_variants() {
        let variants = [
            ParameterKind::Constant { value: 1.0 },
            ParameterKind::PerStage {
                values: vec![1.0, 2.0],
            },
            ParameterKind::Seasonal {
                values: vec![(1, 0.5)],
            },
            ParameterKind::Computed {
                computed_spec: ComputedParameter::EquivalentProductivity {
                    hydro_id: EntityId(1),
                },
            },
            ParameterKind::PerStageBlock {
                values: vec![(0, 0, 1.0)],
            },
        ];

        // No `_` arm: adding a variant without updating here is a compile error.
        for variant in &variants {
            let _name = match variant {
                ParameterKind::Constant { .. } => "Constant",
                ParameterKind::PerStage { .. } => "PerStage",
                ParameterKind::Seasonal { .. } => "Seasonal",
                ParameterKind::Computed { .. } => "Computed",
                ParameterKind::PerStageBlock { .. } => "PerStageBlock",
            };
        }
    }

    #[test]
    fn seasonal_constructor_sorts_and_dedups() {
        let input = vec![(3, 1.5), (1, 0.5), (1, 0.9), (2, 1.0)];
        let result = ParameterKind::new_seasonal(input);
        assert_eq!(
            result,
            ParameterKind::Seasonal {
                values: vec![(1, 0.5), (2, 1.0), (3, 1.5)]
            }
        );
    }

    #[test]
    fn coefficient_ref_copy_semantics() {
        let a = CoefficientRef::Literal(1.0);
        let b = a;
        assert_eq!(a, b);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn scalar_parameter_seasonal_serde_roundtrip() {
        let param = ScalarParameter {
            id: EntityId(7),
            name: "rho_acum_h1".to_string(),
            kind: ParameterKind::Seasonal {
                values: vec![(1, 0.5), (2, 1.0)],
            },
        };

        let json = serde_json::to_string(&param).unwrap();
        let deserialized: ScalarParameter = serde_json::from_str(&json).unwrap();
        assert_eq!(param, deserialized);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn parameter_kind_serde_constant_form() {
        let kind = ParameterKind::Constant { value: 3.6 };
        let json = serde_json::to_string(&kind).unwrap();
        assert_eq!(json, r#"{"kind":"constant","value":3.6}"#);
        let roundtrip: ParameterKind = serde_json::from_str(&json).unwrap();
        assert_eq!(roundtrip, kind);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn parameter_kind_serde_per_stage_form() {
        let kind = ParameterKind::PerStage {
            values: vec![1.0, 1.1, 0.9],
        };
        let json = serde_json::to_string(&kind).unwrap();
        assert_eq!(
            json,
            r#"{"kind":"per_stage","values":[[0,1.0],[1,1.1],[2,0.9]]}"#
        );
        let roundtrip: ParameterKind = serde_json::from_str(&json).unwrap();
        assert_eq!(roundtrip, kind);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn parameter_kind_serde_seasonal_form() {
        let kind = ParameterKind::Seasonal {
            values: vec![(1, 0.5), (2, 1.5)],
        };
        let json = serde_json::to_string(&kind).unwrap();
        assert_eq!(json, r#"{"kind":"seasonal","values":[[1,0.5],[2,1.5]]}"#);
        let roundtrip: ParameterKind = serde_json::from_str(&json).unwrap();
        assert_eq!(roundtrip, kind);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn parameter_kind_serde_computed_form() {
        let kind = ParameterKind::Computed {
            computed_spec: ComputedParameter::EquivalentProductivity {
                hydro_id: EntityId(7),
            },
        };
        let json = serde_json::to_string(&kind).unwrap();
        assert_eq!(
            json,
            r#"{"kind":"computed","computed_spec":{"tag":"equivalent_productivity","hydro_id":7}}"#
        );
        let roundtrip: ParameterKind = serde_json::from_str(&json).unwrap();
        assert_eq!(roundtrip, kind);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn computed_parameter_integrated_tags_serialize_to_snake_case() {
        let eq = ComputedParameter::IntegratedEquivalentProductivity {
            hydro_id: EntityId(3),
        };
        let json = serde_json::to_string(&eq).unwrap();
        assert_eq!(
            json,
            r#"{"tag":"integrated_equivalent_productivity","hydro_id":3}"#
        );
        assert_eq!(
            serde_json::from_str::<ComputedParameter>(&json).unwrap(),
            eq
        );

        let acum = ComputedParameter::IntegratedAccumulatedProductivity {
            hydro_id: EntityId(5),
        };
        let json = serde_json::to_string(&acum).unwrap();
        assert_eq!(
            json,
            r#"{"tag":"integrated_accumulated_productivity","hydro_id":5}"#
        );
        assert_eq!(
            serde_json::from_str::<ComputedParameter>(&json).unwrap(),
            acum
        );
    }

    #[cfg(feature = "serde")]
    #[test]
    fn computed_parameter_max_stored_energy_serializes_to_snake_case() {
        let p = ComputedParameter::MaxStoredEnergy {
            hydro_id: EntityId(4),
        };
        let json = serde_json::to_string(&p).unwrap();
        assert_eq!(json, r#"{"tag":"max_stored_energy","hydro_id":4}"#);
        assert_eq!(serde_json::from_str::<ComputedParameter>(&json).unwrap(), p);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn computed_parameter_existing_tags_are_byte_identical() {
        let eq = ComputedParameter::EquivalentProductivity {
            hydro_id: EntityId(7),
        };
        assert_eq!(
            serde_json::to_string(&eq).unwrap(),
            r#"{"tag":"equivalent_productivity","hydro_id":7}"#
        );
        let acum = ComputedParameter::AccumulatedProductivity {
            hydro_id: EntityId(7),
        };
        assert_eq!(
            serde_json::to_string(&acum).unwrap(),
            r#"{"tag":"accumulated_productivity","hydro_id":7}"#
        );
    }

    #[cfg(feature = "serde")]
    #[test]
    fn parameter_kind_serde_per_stage_block_form() {
        let kind = ParameterKind::PerStageBlock {
            values: vec![(0, 0, 1.0), (1, 0, 2.0)],
        };
        let json = serde_json::to_string(&kind).unwrap();
        assert_eq!(
            json,
            r#"{"kind":"per_stage_block","values":[[0,0,1.0],[1,0,2.0]]}"#
        );
        let roundtrip: ParameterKind = serde_json::from_str(&json).unwrap();
        assert_eq!(roundtrip, kind);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn parameter_kind_per_stage_block_sorts_on_deserialize() {
        let json = r#"{"kind":"per_stage_block","values":[[1,0,2.0],[0,1,9.0],[0,0,1.0]]}"#;
        let parsed: ParameterKind = serde_json::from_str(json).unwrap();
        assert_eq!(
            parsed,
            ParameterKind::PerStageBlock {
                values: vec![(0, 0, 1.0), (0, 1, 9.0), (1, 0, 2.0)]
            }
        );
    }

    #[cfg(feature = "serde")]
    #[test]
    fn parameter_kind_per_stage_block_rejects_duplicate_pair_via_serde() {
        let json = r#"{"kind":"per_stage_block","values":[[0,0,1.0],[0,0,2.0]]}"#;
        let result: Result<ParameterKind, _> = serde_json::from_str(json);
        let err = result.expect_err("expected an error for duplicate (stage_id, block_id) pair");
        let msg = err.to_string();
        assert!(
            msg.contains("duplicate") && msg.contains("(0, 0)"),
            "error message must mention the duplicate pair; got: {err}"
        );
    }

    #[cfg(feature = "serde")]
    #[test]
    fn parameter_kind_per_stage_rejects_non_contiguous_stage_ids() {
        let json = r#"{"kind":"per_stage","values":[[0,1.0],[2,0.9]]}"#;
        let result: Result<ParameterKind, _> = serde_json::from_str(json);
        assert!(
            result.is_err(),
            "expected an error for non-contiguous stage_ids, got: {result:?}"
        );
    }

    #[cfg(feature = "serde")]
    #[test]
    fn parameter_kind_seasonal_rejects_duplicate_season_id_via_serde() {
        let json = r#"{"kind":"seasonal","values":[[1,0.5],[1,0.9],[2,1.0]]}"#;
        let result: Result<ParameterKind, _> = serde_json::from_str(json);
        let err = result.expect_err("expected an error for duplicate season_id");
        assert!(
            err.to_string().contains("duplicate season_id 1"),
            "error message must mention the duplicate id; got: {err}"
        );
    }
}
