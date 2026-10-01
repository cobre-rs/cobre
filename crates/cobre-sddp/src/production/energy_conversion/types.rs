//! Energy-conversion data layout: the per-`(hydro, stage)` scalar cell
//! [`EnergyConversion`], the indexed grid [`EnergyConversionSet`], and the
//! [`EnergyConversionError`] enum raised by the builder.

use cobre_core::{EntityId, Hydro};
use thiserror::Error;

/// Per-`(hydro, stage)` scalars used for inflow-energy / stored-energy accounting.
///
/// All three values are scalar reductions of the underlying production model at a
/// representative operating point (`V_ref`, `Q_ref`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnergyConversion {
    /// Equivalent productivity `ρ_eq` \[MW / (m³/s)\] at the reference point.
    pub equivalent_productivity_mw_per_m3s: f64,
    /// Reference reservoir storage `V_ref` \[hm³\] used to evaluate `h_eq`.
    pub reference_volume_hm3: f64,
    /// Reference turbined flow `Q_ref` \[m³/s\] used to evaluate `h_eq`.
    pub reference_outflow_m3s: f64,
}

/// Indexed grid of `(EnergyConversion, ρ_acum)` per `(hydro, stage)`.
///
/// Indexing convention mirrors [`ProductionModelSet`](crate::hydro_models::ProductionModelSet):
/// outer dimension is the hydro plant, inner dimension is the stage.
#[derive(Debug, Clone)]
pub struct EnergyConversionSet {
    per_hydro_stage: Vec<Vec<EnergyConversion>>,
    accumulated: Vec<Vec<f64>>,
    integrated_equivalent: Vec<Vec<f64>>,
    integrated_accumulated: Vec<Vec<f64>>,
    n_stages: usize,
}

impl EnergyConversionSet {
    /// Build an [`EnergyConversionSet`] from raw grids.
    ///
    /// # Panics
    ///
    /// Panics if the outer dimensions do not match `hydros.len()` or any
    /// inner row has length `!= n_stages`.
    #[must_use]
    pub fn new(
        per_hydro_stage: Vec<Vec<EnergyConversion>>,
        accumulated: Vec<Vec<f64>>,
        hydros: &[Hydro],
        n_stages: usize,
    ) -> Self {
        assert_eq!(
            per_hydro_stage.len(),
            hydros.len(),
            "per_hydro_stage outer length must equal hydros.len()"
        );
        assert_eq!(
            accumulated.len(),
            hydros.len(),
            "accumulated outer length must equal hydros.len()"
        );
        assert!(
            per_hydro_stage.iter().all(|row| row.len() == n_stages),
            "each per_hydro_stage row must have length n_stages"
        );
        assert!(
            accumulated.iter().all(|row| row.len() == n_stages),
            "each accumulated row must have length n_stages"
        );
        let integrated_equivalent = per_hydro_stage
            .iter()
            .map(|row| {
                row.iter()
                    .map(|c| c.equivalent_productivity_mw_per_m3s)
                    .collect()
            })
            .collect();
        let integrated_accumulated = accumulated.clone();
        Self {
            per_hydro_stage,
            accumulated,
            integrated_equivalent,
            integrated_accumulated,
            n_stages,
        }
    }

    /// Install the two integrated-productivity grids, overriding the
    /// `new(...)` reference-point defaults.
    ///
    /// # Panics
    ///
    /// In debug builds, panics if either grid's outer length does not equal
    /// `n_hydros` or any inner row's length does not equal `n_stages`.
    #[must_use]
    pub fn with_integrated(
        mut self,
        integrated_equivalent: Vec<Vec<f64>>,
        integrated_accumulated: Vec<Vec<f64>>,
    ) -> Self {
        debug_assert_eq!(
            integrated_equivalent.len(),
            self.per_hydro_stage.len(),
            "integrated_equivalent outer length must equal n_hydros"
        );
        debug_assert!(
            integrated_equivalent
                .iter()
                .all(|row| row.len() == self.n_stages),
            "each integrated_equivalent row must have length n_stages"
        );
        debug_assert_eq!(
            integrated_accumulated.len(),
            self.per_hydro_stage.len(),
            "integrated_accumulated outer length must equal n_hydros"
        );
        debug_assert!(
            integrated_accumulated
                .iter()
                .all(|row| row.len() == self.n_stages),
            "each integrated_accumulated row must have length n_stages"
        );
        self.integrated_equivalent = integrated_equivalent;
        self.integrated_accumulated = integrated_accumulated;
        self
    }

    fn check_indices(&self, hydro: usize, stage: usize) {
        debug_assert!(
            hydro < self.per_hydro_stage.len(),
            "hydro index {hydro} out of bounds (n_hydros = {})",
            self.per_hydro_stage.len()
        );
        debug_assert!(
            stage < self.n_stages,
            "stage index {stage} out of bounds (n_stages = {})",
            self.n_stages
        );
    }

    /// Return the [`EnergyConversion`] for `(hydro, stage)`.
    ///
    /// # Panics
    ///
    /// In debug builds, panics on out-of-range indices.
    #[must_use]
    pub fn conversion(&self, hydro: usize, stage: usize) -> &EnergyConversion {
        self.check_indices(hydro, stage);
        &self.per_hydro_stage[hydro][stage]
    }

    /// Return the accumulated productivity `ρ_acum` for `(hydro, stage)`.
    ///
    /// # Panics
    ///
    /// In debug builds, panics on out-of-range indices.
    #[must_use]
    pub fn accumulated_productivity(&self, hydro: usize, stage: usize) -> f64 {
        self.check_indices(hydro, stage);
        self.accumulated[hydro][stage]
    }

    /// Return the own-scope, mean-evaluator integrated productivity for `(hydro, stage)`.
    ///
    /// # Panics
    ///
    /// In debug builds, panics on out-of-range indices.
    #[must_use]
    pub fn integrated_equivalent_productivity(&self, hydro: usize, stage: usize) -> f64 {
        self.check_indices(hydro, stage);
        self.integrated_equivalent[hydro][stage]
    }

    /// Return the cascade-scope, mean-evaluator integrated productivity for `(hydro, stage)`.
    ///
    /// # Panics
    ///
    /// In debug builds, panics on out-of-range indices.
    #[must_use]
    pub fn integrated_accumulated_productivity(&self, hydro: usize, stage: usize) -> f64 {
        self.check_indices(hydro, stage);
        self.integrated_accumulated[hydro][stage]
    }

    /// Number of hydro plants (outer grid dimension).
    #[must_use]
    pub fn n_hydros(&self) -> usize {
        self.per_hydro_stage.len()
    }

    /// Number of stages (inner grid dimension).
    #[must_use]
    pub fn n_stages(&self) -> usize {
        self.n_stages
    }
}

/// Errors raised by [`build_energy_conversion_set`](super::build_energy_conversion_set) and successor derivations.
#[derive(Debug, Error)]
pub enum EnergyConversionError {
    /// Dimensions of input data did not match each other.
    #[error("energy-conversion dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch {
        /// Description of the expected dimension.
        expected: String,
        /// Description of the actual dimension.
        got: String,
    },
    /// `max_storage_hm3 < min_storage_hm3` for a hydro plant.
    #[error(
        "hydro {hydro_id:?} has invalid storage range: max_storage_hm3={v_max} < min_storage_hm3={v_min}"
    )]
    InvalidStorageRange {
        /// Identifier of the offending hydro.
        hydro_id: EntityId,
        /// Stored minimum reservoir volume \[hm³\].
        v_min: f64,
        /// Stored maximum reservoir volume \[hm³\].
        v_max: f64,
    },
    /// `max_turbined_m3s` is negative.
    #[error("hydro {hydro_id:?} has negative max_turbined_m3s={q_max}")]
    NegativeMaxTurbined {
        /// Identifier of the offending hydro.
        hydro_id: EntityId,
        /// Stored maximum turbined flow \[m³/s\].
        q_max: f64,
    },
    /// FPHA equivalent head `h_eq = h_fore − h_tail − h_loss` is non-positive,
    /// which would yield a non-physical `ρ_eq`.
    #[error("hydro {hydro_id:?} has non-positive equivalent head h_eq={h_eq}")]
    NonPositiveEquivalentHead {
        /// Identifier of the offending hydro.
        hydro_id: EntityId,
        /// Computed equivalent head \[m\].
        h_eq: f64,
    },
    /// Building a `ForebayTable` from the VHA rows failed validation.
    #[error("hydro {hydro_id:?} forebay table construction failed: {message}")]
    ForebayTableInvalid {
        /// Identifier of the offending hydro.
        hydro_id: EntityId,
        /// Human-readable description of the underlying FPHA-fitting error.
        /// The concrete error type is intentionally kept crate-private.
        message: String,
    },
    /// The cascade topology contains a different number of entries than the
    /// `hydros` slice, indicating that the topology was built from a different
    /// set of hydros.
    #[error("cascade topological order length {got} does not match hydros slice length {expected}")]
    CascadeIndexMismatch {
        /// Number of hydros in the `hydros` slice (expected).
        expected: usize,
        /// Number of entries in `cascade.topological_order()` (actual).
        got: usize,
    },
    /// A downstream reference in the cascade points to an `EntityId` that is
    /// not present in the `hydros` slice.
    #[error(
        "cascade has dangling downstream reference: hydro {hydro_id:?} points to {downstream_id:?} which is not in the hydros slice"
    )]
    DanglingDownstream {
        /// The hydro whose `downstream_id` is dangling.
        hydro_id: EntityId,
        /// The downstream `EntityId` that has no matching entry.
        downstream_id: EntityId,
    },
    /// An FPHA hydro has no way to derive `ρ_eq`: no VHA geometry, no
    /// `ρ_esp`, and no entry in the override table.
    #[error(
        "FPHA hydro '{hydro_name}' ({hydro_id:?}) cannot derive ρ_eq for stage {stage}: \
        no VHA geometry + ρ_esp pair is present and no override entry exists. \
        Remediation: (1) supply VHA geometry rows and specific_productivity (ρ_esp) for this hydro, \
        (2) add an entry in system/hydro_energy_productivity.parquet, \
        or (3) change the hydro's generation_model away from FPHA."
    )]
    FphaMissingEquivalentProductivity {
        /// Identifier of the offending FPHA hydro.
        hydro_id: EntityId,
        /// Name of the offending FPHA hydro, for user-facing diagnostics.
        hydro_name: String,
        /// First stage index for which `ρ_eq` could not be derived.
        stage: usize,
    },
}

#[cfg(test)]
#[allow(
    clippy::doc_markdown,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::panic,
    clippy::unwrap_used
)]
mod tests {
    use super::*;
    use crate::test_support;

    #[test]
    fn new_round_trips_grid_dimensions() {
        let grid = vec![
            vec![
                EnergyConversion {
                    equivalent_productivity_mw_per_m3s: 0.5,
                    reference_volume_hm3: 100.0,
                    reference_outflow_m3s: 50.0,
                },
                EnergyConversion {
                    equivalent_productivity_mw_per_m3s: 0.6,
                    reference_volume_hm3: 110.0,
                    reference_outflow_m3s: 55.0,
                },
                EnergyConversion {
                    equivalent_productivity_mw_per_m3s: 0.7,
                    reference_volume_hm3: 120.0,
                    reference_outflow_m3s: 60.0,
                },
            ],
            vec![
                EnergyConversion {
                    equivalent_productivity_mw_per_m3s: 1.0,
                    reference_volume_hm3: 200.0,
                    reference_outflow_m3s: 80.0,
                },
                EnergyConversion {
                    equivalent_productivity_mw_per_m3s: 1.1,
                    reference_volume_hm3: 210.0,
                    reference_outflow_m3s: 85.0,
                },
                EnergyConversion {
                    equivalent_productivity_mw_per_m3s: 1.2,
                    reference_volume_hm3: 220.0,
                    reference_outflow_m3s: 90.0,
                },
            ],
        ];
        let acc = vec![vec![10.0, 11.0, 12.0], vec![20.0, 21.0, 22.0]];
        let set = EnergyConversionSet::new(
            grid.clone(),
            acc.clone(),
            &test_support::minimal_hydros(2),
            3,
        );

        assert_eq!(set.n_hydros(), 2);
        assert_eq!(set.n_stages(), 3);
        for (h, row) in grid.iter().enumerate() {
            for (s, expected) in row.iter().enumerate() {
                assert_eq!(set.conversion(h, s), expected);
            }
        }
    }

    #[test]
    fn accessors_return_correct_cell() {
        let grid = vec![
            vec![EnergyConversion {
                equivalent_productivity_mw_per_m3s: 0.5,
                reference_volume_hm3: 100.0,
                reference_outflow_m3s: 50.0,
            }],
            vec![EnergyConversion {
                equivalent_productivity_mw_per_m3s: 0.9,
                reference_volume_hm3: 180.0,
                reference_outflow_m3s: 70.0,
            }],
        ];
        let acc = vec![vec![3.5_f64], vec![2.5_f64]];
        let set = EnergyConversionSet::new(grid, acc, &test_support::minimal_hydros(2), 1);

        assert_eq!(set.conversion(0, 0).equivalent_productivity_mw_per_m3s, 0.5);
        assert_eq!(set.conversion(1, 0).reference_outflow_m3s, 70.0);
        assert_eq!(set.accumulated_productivity(0, 0), 3.5);
        assert_eq!(set.accumulated_productivity(1, 0), 2.5);
    }

    #[test]
    #[should_panic(expected = "per_hydro_stage outer length must equal hydros.len()")]
    fn new_panics_on_hydro_count_mismatch() {
        let grid = vec![vec![EnergyConversion {
            equivalent_productivity_mw_per_m3s: 0.5,
            reference_volume_hm3: 100.0,
            reference_outflow_m3s: 50.0,
        }]];
        let acc = vec![vec![3.5_f64]];
        let _ = EnergyConversionSet::new(grid, acc, &test_support::minimal_hydros(2), 1);
    }

    #[test]
    fn new_defaults_integrated_grids_to_reference_point_grids() {
        let grid = vec![
            vec![EnergyConversion {
                equivalent_productivity_mw_per_m3s: 0.5,
                reference_volume_hm3: 100.0,
                reference_outflow_m3s: 50.0,
            }],
            vec![EnergyConversion {
                equivalent_productivity_mw_per_m3s: 0.9,
                reference_volume_hm3: 180.0,
                reference_outflow_m3s: 70.0,
            }],
        ];
        let acc = vec![vec![3.5_f64], vec![2.5_f64]];
        let set = EnergyConversionSet::new(grid, acc, &test_support::minimal_hydros(2), 1);

        for h in 0..2 {
            assert_eq!(
                set.integrated_equivalent_productivity(h, 0).to_bits(),
                set.conversion(h, 0)
                    .equivalent_productivity_mw_per_m3s
                    .to_bits()
            );
            assert_eq!(
                set.integrated_accumulated_productivity(h, 0).to_bits(),
                set.accumulated_productivity(h, 0).to_bits()
            );
        }
    }
}
