//! Assembly logic for scenario pipeline data.
//!
//! This module joins the flat row types produced by individual parsers into the
//! assembled types expected by [`cobre_core::System`]:
//!
//! - [`assemble_inflow_models`] — joins [`InflowSeasonalStatsRow`] with
//!   [`InflowArCoefficientRow`] by `(hydro_id, stage_id)` to produce
//!   `Vec<InflowModel>`.
//! - [`assemble_load_models`] — maps [`LoadSeasonalStatsRow`] 1:1 to
//!   `Vec<LoadModel>`.
//!
//! Both inputs are assumed to be pre-sorted by the individual parsers (tickets 020
//! and 021). The assembly step preserves that sort order in its output.

use std::collections::HashMap;
use std::path::Path;

use cobre_core::{
    EntityId,
    scenario::{InflowModel, LoadModel},
};

use crate::LoadError;
use crate::scenarios::{InflowArCoefficientRow, InflowSeasonalStatsRow, LoadSeasonalStatsRow};

/// Assemble `Vec<InflowModel>` by joining seasonal stats with AR coefficients.
///
/// Both inputs must be pre-sorted by their respective parsers:
/// - `stats` sorted by `(hydro_id, stage_id)` ascending.
/// - `coefficients` sorted by `(hydro_id, stage_id, lag)` ascending.
///
/// For each [`InflowSeasonalStatsRow`], all [`InflowArCoefficientRow`] entries
/// with a matching `(hydro_id, stage_id)` are collected into `ar_coefficients`,
/// preserving lag order. The coefficient count determines the AR order; there is
/// no cross-check against a separate `ar_order` field.
///
/// When no coefficient rows exist for a (hydro, stage) pair, the resulting
/// [`InflowModel`] has an empty `ar_coefficients` vec and `residual_std_ratio = 1.0`
/// (white-noise identity).
///
/// When `stats` is empty (regardless of whether `coefficients` is non-empty),
/// the function returns an empty `Vec` without error. The P7 estimation path
/// (`UserArHistoryStats`) loads AR coefficients independently via
/// `parse_inflow_ar_coefficients` and does not route them through this function.
///
/// # Errors
///
/// | Condition                                              | Error variant              |
/// |--------------------------------------------------------|----------------------------|
/// | Coefficient rows exist for a pair not in `stats`      | [`LoadError::SchemaError`] |
///
/// # Examples
///
/// ```
/// use cobre_core::EntityId;
/// use cobre_io::scenarios::{InflowSeasonalStatsRow, InflowArCoefficientRow};
/// use cobre_io::scenarios::assembly::assemble_inflow_models;
///
/// let stats = vec![
///     InflowSeasonalStatsRow { hydro_id: EntityId(1), stage_id: 0, mean_m3s: 100.0, std_m3s: 10.0 },
///     InflowSeasonalStatsRow { hydro_id: EntityId(1), stage_id: 1, mean_m3s: 80.0, std_m3s: 8.0 },
/// ];
/// let coefficients = vec![
///     InflowArCoefficientRow { hydro_id: EntityId(1), stage_id: 0, lag: 1, coefficient: 0.5, residual_std_ratio: 0.85 },
///     InflowArCoefficientRow { hydro_id: EntityId(1), stage_id: 0, lag: 2, coefficient: 0.2, residual_std_ratio: 0.85 },
/// ];
/// let models = assemble_inflow_models(stats, coefficients).expect("valid join");
/// assert_eq!(models.len(), 2);
/// assert_eq!(models[0].ar_order(), 2);
/// assert!((models[0].residual_std_ratio - 0.85).abs() < f64::EPSILON);
/// assert!(models[1].ar_coefficients.is_empty());
/// assert!((models[1].residual_std_ratio - 1.0).abs() < f64::EPSILON);
/// ```
pub fn assemble_inflow_models(
    stats: Vec<InflowSeasonalStatsRow>,
    coefficients: Vec<InflowArCoefficientRow>,
) -> Result<Vec<InflowModel>, LoadError> {
    // Short-circuit: no stats means there is no base to join AR coefficients onto.
    // When stats is empty the result is always an empty vec regardless of whether
    // coefficients are present. The P7 estimation path (UserArHistoryStats) loads
    // AR independently and does not depend on this join to produce inflow models.
    if stats.is_empty() {
        return Ok(Vec::new());
    }

    // Group coefficients by (hydro_id, stage_id) preserving lag order (pre-sorted by parser).
    // The tuple carries (coefficient_vec, residual_std_ratio) where the ratio is taken
    // from the first row in the group (consistency across lags is validated in Layer 5).
    let mut coeff_map: HashMap<(EntityId, i32), (Vec<f64>, f64)> =
        HashMap::with_capacity(coefficients.len());
    for row in coefficients {
        let entry = coeff_map
            .entry((row.hydro_id, row.stage_id))
            .or_insert_with(|| (Vec::new(), row.residual_std_ratio));
        entry.0.push(row.coefficient);
    }

    let total_coeff_keys = coeff_map.len();
    let mut consumed_keys: usize = 0;

    let mut models = Vec::with_capacity(stats.len());

    for row in stats {
        let key = (row.hydro_id, row.stage_id);

        let (ar_coefficients, residual_std_ratio) =
            if let Some((coeffs, ratio)) = coeff_map.remove(&key) {
                consumed_keys += 1;
                (coeffs, ratio)
            } else {
                // No coefficients for this (hydro, stage): white-noise model.
                (Vec::new(), 1.0_f64)
            };

        models.push(InflowModel {
            hydro_id: row.hydro_id,
            stage_id: row.stage_id,
            mean_m3s: row.mean_m3s,
            std_m3s: row.std_m3s,
            ar_coefficients,
            residual_std_ratio,
        });
    }

    // Any remaining keys in coeff_map are orphaned (no matching stats row).
    if consumed_keys < total_coeff_keys {
        let mut orphan_keys: Vec<_> = coeff_map.keys().collect();
        orphan_keys.sort_by_key(|(id, stage)| (id.0, *stage));
        let orphan_descriptions: Vec<String> = orphan_keys
            .iter()
            .map(|(id, stage)| format!("(hydro_id={}, stage_id={})", id.0, stage))
            .collect();
        return Err(LoadError::SchemaError {
            path: Path::new("scenarios/inflow_ar_coefficients.parquet").to_path_buf(),
            field: "inflow_ar_coefficients".to_string(),
            message: format!(
                "orphaned AR coefficients for {} have no matching inflow_seasonal_stats row",
                orphan_descriptions.join(", "),
            ),
        });
    }

    Ok(models)
}

/// Assemble `Vec<LoadModel>` by mapping [`LoadSeasonalStatsRow`] 1:1 to [`LoadModel`].
///
/// This is a direct field mapping with no join logic. The input is expected to be
/// pre-sorted by `(bus_id, stage_id)` ascending (as produced by the parser).
/// That order is preserved in the output.
///
/// # Examples
///
/// ```
/// use cobre_core::EntityId;
/// use cobre_io::scenarios::LoadSeasonalStatsRow;
/// use cobre_io::scenarios::assembly::assemble_load_models;
///
/// let stats = vec![
///     LoadSeasonalStatsRow { bus_id: EntityId(1), stage_id: 0, mean_mw: 300.0, std_mw: 30.0 },
///     LoadSeasonalStatsRow { bus_id: EntityId(1), stage_id: 1, mean_mw: 280.0, std_mw: 28.0 },
///     LoadSeasonalStatsRow { bus_id: EntityId(2), stage_id: 0, mean_mw: 500.0, std_mw: 50.0 },
/// ];
/// let models = assemble_load_models(stats);
/// assert_eq!(models.len(), 3);
/// assert_eq!(models[0].bus_id, EntityId(1));
/// assert_eq!(models[0].mean_mw, 300.0);
/// ```
#[must_use]
pub fn assemble_load_models(stats: Vec<LoadSeasonalStatsRow>) -> Vec<LoadModel> {
    stats
        .into_iter()
        .map(|row| LoadModel {
            bus_id: row.bus_id,
            stage_id: row.stage_id,
            mean_mw: row.mean_mw,
            std_mw: row.std_mw,
        })
        .collect()
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::too_many_lines,
    clippy::doc_markdown
)]
mod tests {
    use super::*;
    use cobre_core::EntityId;

    #[test]
    fn test_assemble_inflow_models_matching_join() {
        let stats = vec![
            InflowSeasonalStatsRow {
                hydro_id: EntityId(1),
                stage_id: 0,
                mean_m3s: 100.0,
                std_m3s: 10.0,
            },
            InflowSeasonalStatsRow {
                hydro_id: EntityId(1),
                stage_id: 1,
                mean_m3s: 80.0,
                std_m3s: 8.0,
            },
            InflowSeasonalStatsRow {
                hydro_id: EntityId(2),
                stage_id: 0,
                mean_m3s: 200.0,
                std_m3s: 20.0,
            },
        ];
        let coefficients = vec![
            InflowArCoefficientRow {
                hydro_id: EntityId(1),
                stage_id: 0,
                lag: 1,
                coefficient: 0.45,
                residual_std_ratio: 0.85,
            },
            InflowArCoefficientRow {
                hydro_id: EntityId(1),
                stage_id: 0,
                lag: 2,
                coefficient: 0.22,
                residual_std_ratio: 0.85,
            },
            InflowArCoefficientRow {
                hydro_id: EntityId(2),
                stage_id: 0,
                lag: 1,
                coefficient: 0.60,
                residual_std_ratio: 0.72,
            },
        ];

        let models = assemble_inflow_models(stats, coefficients).unwrap();
        assert_eq!(models.len(), 3);

        let m0 = &models[0];
        assert_eq!(m0.hydro_id, EntityId(1));
        assert_eq!(m0.stage_id, 0);
        assert_eq!(m0.ar_order(), 2);
        assert_eq!(m0.ar_coefficients.len(), 2);
        assert!((m0.ar_coefficients[0] - 0.45).abs() < f64::EPSILON);
        assert!((m0.ar_coefficients[1] - 0.22).abs() < f64::EPSILON);
        assert!((m0.residual_std_ratio - 0.85).abs() < f64::EPSILON);

        let m1 = &models[1];
        assert_eq!(m1.hydro_id, EntityId(1));
        assert_eq!(m1.stage_id, 1);
        assert_eq!(m1.ar_order(), 0);
        assert!(m1.ar_coefficients.is_empty());
        assert!((m1.residual_std_ratio - 1.0).abs() < f64::EPSILON);

        let m2 = &models[2];
        assert_eq!(m2.hydro_id, EntityId(2));
        assert_eq!(m2.stage_id, 0);
        assert_eq!(m2.ar_order(), 1);
        assert_eq!(m2.ar_coefficients.len(), 1);
        assert!((m2.ar_coefficients[0] - 0.60).abs() < f64::EPSILON);
        assert!((m2.residual_std_ratio - 0.72).abs() < f64::EPSILON);
    }

    #[test]
    fn test_assemble_inflow_models_no_coefficients() {
        let stats = vec![InflowSeasonalStatsRow {
            hydro_id: EntityId(3),
            stage_id: 5,
            mean_m3s: 50.0,
            std_m3s: 5.0,
        }];
        let models = assemble_inflow_models(stats, vec![]).unwrap();
        assert_eq!(models.len(), 1);
        assert!(models[0].ar_coefficients.is_empty());
        assert_eq!(models[0].ar_order(), 0);
        assert!((models[0].residual_std_ratio - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_assemble_inflow_models_orphaned_coefficients() {
        let stats = vec![InflowSeasonalStatsRow {
            hydro_id: EntityId(1),
            stage_id: 0,
            mean_m3s: 100.0,
            std_m3s: 10.0,
        }];
        // Coefficients for hydro 5 stage 0 — no matching stats row.
        let coefficients = vec![InflowArCoefficientRow {
            hydro_id: EntityId(5),
            stage_id: 0,
            lag: 1,
            coefficient: 0.3,
            residual_std_ratio: 0.85,
        }];

        let err = assemble_inflow_models(stats, coefficients).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("inflow_ar_coefficients"),
                    "field should mention inflow_ar_coefficients, got: {field}"
                );
                assert!(
                    message.contains("orphaned"),
                    "message should contain 'orphaned', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    #[test]
    fn test_assemble_inflow_models_both_empty() {
        let models = assemble_inflow_models(vec![], vec![]).unwrap();
        assert!(models.is_empty());
    }

    /// AC-009-1: `assemble_inflow_models(vec![], non_empty_coefficients)` must
    /// return `Ok(Vec::new())` without error.
    ///
    /// This is the P7 (UserArHistoryStats) case: `inflow_seasonal_stats.parquet`
    /// is absent so `stats` is empty, but `inflow_ar_coefficients.parquet` is
    /// present. The function must not raise a SchemaError for "orphaned" AR
    /// entries — instead it returns an empty vec because there is no base to
    /// join onto. The estimation path loads AR independently.
    #[test]
    fn test_assemble_inflow_models_empty_stats_non_empty_ar_returns_empty() {
        let coefficients = vec![InflowArCoefficientRow {
            hydro_id: EntityId(1),
            stage_id: 0,
            lag: 1,
            coefficient: 0.5,
            residual_std_ratio: 0.85,
        }];

        let result = assemble_inflow_models(vec![], coefficients);
        assert!(
            result.is_ok(),
            "empty stats + non-empty AR must return Ok, got: {:?}",
            result
        );
        let models = result.unwrap();
        assert!(
            models.is_empty(),
            "empty stats must produce empty InflowModel vec, got {} models",
            models.len()
        );
    }

    #[test]
    fn test_assemble_load_models_four_rows() {
        let stats = vec![
            LoadSeasonalStatsRow {
                bus_id: EntityId(1),
                stage_id: 0,
                mean_mw: 300.0,
                std_mw: 30.0,
            },
            LoadSeasonalStatsRow {
                bus_id: EntityId(1),
                stage_id: 1,
                mean_mw: 280.0,
                std_mw: 28.0,
            },
            LoadSeasonalStatsRow {
                bus_id: EntityId(2),
                stage_id: 0,
                mean_mw: 500.0,
                std_mw: 50.0,
            },
            LoadSeasonalStatsRow {
                bus_id: EntityId(2),
                stage_id: 1,
                mean_mw: 450.0,
                std_mw: 45.0,
            },
        ];

        let models = assemble_load_models(stats);
        assert_eq!(models.len(), 4);

        assert_eq!(models[0].bus_id, EntityId(1));
        assert_eq!(models[0].stage_id, 0);
        assert!((models[0].mean_mw - 300.0).abs() < f64::EPSILON);
        assert!((models[0].std_mw - 30.0).abs() < f64::EPSILON);

        assert_eq!(models[1].bus_id, EntityId(1));
        assert_eq!(models[1].stage_id, 1);
        assert!((models[1].mean_mw - 280.0).abs() < f64::EPSILON);
        assert!((models[1].std_mw - 28.0).abs() < f64::EPSILON);

        assert_eq!(models[2].bus_id, EntityId(2));
        assert_eq!(models[2].stage_id, 0);
        assert!((models[2].mean_mw - 500.0).abs() < f64::EPSILON);
        assert!((models[2].std_mw - 50.0).abs() < f64::EPSILON);

        assert_eq!(models[3].bus_id, EntityId(2));
        assert_eq!(models[3].stage_id, 1);
        assert!((models[3].mean_mw - 450.0).abs() < f64::EPSILON);
        assert!((models[3].std_mw - 45.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_assemble_load_models_empty() {
        let models = assemble_load_models(vec![]);
        assert!(models.is_empty());
    }
}
