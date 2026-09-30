//! Energy-conversion builder: derives the [`EnergyConversionSet`] for the case.

use std::collections::HashMap;
use std::hash::BuildHasher;

use cobre_core::{CascadeTopology, EntityId, Hydro, HydroGenerationModel, StageId, StudyPos};
use cobre_io::{HydroGeometryRow, HydroReferenceVolumeFractions};

use super::productivity_override::HydroEnergyProductivityOverride;
use super::types::{EnergyConversion, EnergyConversionError, EnergyConversionSet};
use crate::fpha_fitting::{ForebayTable, evaluate_losses, evaluate_tailrace};
use crate::hydro_models::ProductionModelSet;
use crate::hydro_models::ResolvedProductionModel::ConstantProductivity;
use crate::hydro_models::ResolvedProductionModel::Fpha;

/// Build the [`EnergyConversionSet`] for the case.
///
/// Fills two per-`(hydro, stage)` own-term evaluators:
///
/// - **Reference-point** (`equivalent_productivity_mw_per_m3s`, gated by the
///   generation model): for FPHA hydros `ρ_eq` resolves in priority order —
///   (1) an `override_table` `ρ_eq` value, (2) the resolved `ρ_esp` (override →
///   entity) times `h_eq(V_ref, Q_ref)` from VHA geometry, else (3)
///   [`EnergyConversionError::FphaMissingEquivalentProductivity`]; for non-FPHA
///   hydros `ρ_eq` comes from `production_models` (passing `None` yields `0.0`,
///   only appropriate in tests that do not require accurate `ρ_eq`).
/// - **Mean** (`integrated_equivalent_productivity`, gated by geometry, not the
///   generation model): for ANY hydro with VHA geometry and a resolved `ρ_esp`
///   the own term is (1) an `override_table` `ρ_eq` value, else (2) the
///   reference-point value when the entity physical range is collapsed or the
///   hydro has no geometry, else (3) the resolved `ρ_esp` times
///   `(mean_height(V_lo, V_hi) − cf − losses)` over the entity physical range
///   `hydro.{min,max}_storage_hm3`, reusing `Q_ref` and the reference-point
///   `cf`/`losses` evaluation.
///
/// # Errors
///
/// - [`EnergyConversionError::InvalidStorageRange`] — `max_storage_hm3 < min_storage_hm3`.
/// - [`EnergyConversionError::NegativeMaxTurbined`] — `max_turbined_m3s < 0`.
/// - [`EnergyConversionError::ForebayTableInvalid`] — VHA rows fail forebay-table
///   validation, for any hydro carrying VHA geometry and `ρ_esp`.
/// - [`EnergyConversionError::NonPositiveEquivalentHead`] — a reference-point equivalent
///   head `≤ 0` on an FPHA hydro, or a mean equivalent head `≤ 0` on any hydro whose
///   per-stage range is genuine (geometry, `ρ_esp`, no override).
/// - [`EnergyConversionError::FphaMissingEquivalentProductivity`] — FPHA hydro has no
///   usable `ρ_eq` source for a given stage and no override entry.
/// - [`EnergyConversionError::CascadeIndexMismatch`] — cascade built from different hydro set.
/// - [`EnergyConversionError::DanglingDownstream`] — dangling downstream reference.
#[allow(clippy::missing_errors_doc)]
pub fn build_energy_conversion_set<S: BuildHasher>(
    hydros: &[Hydro],
    stage_ids: &[StageId],
    cascade: &CascadeTopology,
    reference_volume_fractions: &HydroReferenceVolumeFractions,
    vha_rows_by_hydro: &HashMap<EntityId, Vec<HydroGeometryRow>, S>,
    override_table: Option<&HydroEnergyProductivityOverride>,
    production_models: Option<&ProductionModelSet>,
) -> Result<EnergyConversionSet, EnergyConversionError> {
    let n_hydros = hydros.len();
    let n_stages = stage_ids.len();

    let mut per_hydro_stage: Vec<Vec<EnergyConversion>> = Vec::with_capacity(n_hydros);
    let mut mean_own_grid: Vec<Vec<f64>> = Vec::with_capacity(n_hydros);

    for (h_idx, hydro) in hydros.iter().enumerate() {
        let v_min = hydro.min_storage_hm3;
        let v_max = hydro.max_storage_hm3;
        let q_max = hydro.max_turbined_m3s;
        let is_fpha = matches!(hydro.generation_model, HydroGenerationModel::Fpha);

        if v_max < v_min {
            return Err(EnergyConversionError::InvalidStorageRange {
                hydro_id: hydro.id,
                v_min,
                v_max,
            });
        }
        if q_max < 0.0 {
            return Err(EnergyConversionError::NegativeMaxTurbined {
                hydro_id: hydro.id,
                q_max,
            });
        }

        // Resolved once per stage (override → entity) via the shared resolver,
        // so the builder and the SpecificProductivity tag can never drift.
        let entity_esp = hydro.specific_productivity_mw_per_m3s_per_m;
        let resolved_rho_esp: Vec<Option<f64>> = stage_ids
            .iter()
            .map(|&stage_id| {
                override_table.map_or(entity_esp, |o| {
                    o.resolve_specific_productivity(hydro.id, stage_id, entity_esp)
                })
            })
            .collect();

        // Geometry, not the generation model, gates the mean-evaluator: build the
        // forebay table for ANY hydro with VHA rows and a resolved ρ_esp at any
        // stage, so a construction failure surfaces as ForebayTableInvalid
        // uniformly (never gated on FPHA).
        let forebay_table = match vha_rows_by_hydro.get(&hydro.id) {
            Some(rows) if resolved_rho_esp.iter().any(Option::is_some) => {
                Some(ForebayTable::new(rows, &hydro.name).map_err(|e| {
                    EnergyConversionError::ForebayTableInvalid {
                        hydro_id: hydro.id,
                        message: e.to_string(),
                    }
                })?)
            }
            _ => None,
        };

        let mut row: Vec<EnergyConversion> = Vec::with_capacity(n_stages);
        let mut mean_own_row: Vec<f64> = Vec::with_capacity(n_stages);
        for (stage_pos, &stage_id) in stage_ids.iter().enumerate() {
            let reference_volume_hm3 =
                reference_volume_fractions.get(hydro.id, StudyPos(stage_pos));

            let productivity = if is_fpha {
                0.0
            } else {
                production_models.map_or(0.0, |pm| match pm.model(h_idx, stage_pos) {
                    ConstantProductivity { productivity } => *productivity,
                    Fpha { .. } => 0.0,
                })
            };
            let mut conversion =
                derive_conversion_for_hydro(hydro, reference_volume_hm3, productivity);

            // Keyed by the domain StageId (matches how the table is built and how
            // cobre_io's validator keys it) — never the study position.
            let parquet_rho_eq =
                override_table.and_then(|o| o.equivalent_productivity(hydro.id, stage_id));
            let rho_esp_at_stage = resolved_rho_esp[stage_pos];

            if is_fpha {
                let rho_eq = if let Some(value) = parquet_rho_eq {
                    value
                } else if let (Some(table), Some(rho_esp)) =
                    (forebay_table.as_ref(), rho_esp_at_stage)
                {
                    let h_eq = fpha_equivalent_head(
                        hydro,
                        conversion.reference_volume_hm3,
                        conversion.reference_outflow_m3s,
                        table,
                    )?;
                    rho_esp * h_eq
                } else {
                    return Err(EnergyConversionError::FphaMissingEquivalentProductivity {
                        hydro_id: hydro.id,
                        hydro_name: hydro.name.clone(),
                        stage: stage_pos,
                    });
                };
                conversion.equivalent_productivity_mw_per_m3s = rho_eq;
            }

            // Mean-evaluator own term (geometry gates it, not the generation model).
            // An override or a collapsed/absent range copies the reference-point value
            // bit-for-bit; only a genuine range integrates the forebay height.
            let mean_own = if parquet_rho_eq.is_some() {
                conversion.equivalent_productivity_mw_per_m3s
            } else if let (Some(table), Some(rho_esp)) = (forebay_table.as_ref(), rho_esp_at_stage)
            {
                let (v_lo, v_hi) = (v_min, v_max);
                if v_hi <= v_lo {
                    conversion.equivalent_productivity_mw_per_m3s
                } else {
                    let mean_head = mean_equivalent_head(
                        hydro,
                        table,
                        v_lo,
                        v_hi,
                        conversion.reference_outflow_m3s,
                    );
                    if mean_head <= 0.0 {
                        return Err(EnergyConversionError::NonPositiveEquivalentHead {
                            hydro_id: hydro.id,
                            h_eq: mean_head,
                        });
                    }
                    rho_esp * mean_head
                }
            } else {
                conversion.equivalent_productivity_mw_per_m3s
            };

            row.push(conversion);
            mean_own_row.push(mean_own);
        }
        per_hydro_stage.push(row);
        mean_own_grid.push(mean_own_row);
    }

    let grids =
        accumulate_cascade_grids(cascade, hydros, &per_hydro_stage, &mean_own_grid, n_stages)?;

    Ok(
        EnergyConversionSet::new(per_hydro_stage, grids.accumulated, hydros, n_stages)
            .with_integrated(grids.integrated_equivalent, grids.integrated_accumulated),
    )
}

/// Reference-point (`ρ_acum`) and mean-evaluator cascade grids.
struct CascadeGrids {
    accumulated: Vec<Vec<f64>>,
    integrated_equivalent: Vec<Vec<f64>>,
    integrated_accumulated: Vec<Vec<f64>>,
}

/// Sum each plant's own term down the reverse-topological cascade, for the
/// reference-point (`ρ_acum`) and mean evaluators together.
///
/// # Errors
///
/// - [`EnergyConversionError::CascadeIndexMismatch`] — `cascade` length ≠ `hydros` length.
/// - [`EnergyConversionError::DanglingDownstream`] — a downstream id absent from `hydros`.
fn accumulate_cascade_grids(
    cascade: &CascadeTopology,
    hydros: &[Hydro],
    per_hydro_stage: &[Vec<EnergyConversion>],
    mean_own_grid: &[Vec<f64>],
    n_stages: usize,
) -> Result<CascadeGrids, EnergyConversionError> {
    let n_hydros = hydros.len();
    let topo_len = cascade.topological_order().len();
    if topo_len != n_hydros {
        return Err(EnergyConversionError::CascadeIndexMismatch {
            expected: n_hydros,
            got: topo_len,
        });
    }

    let mut id_to_index: HashMap<EntityId, usize> = HashMap::with_capacity(n_hydros);
    for (idx, h) in hydros.iter().enumerate() {
        id_to_index.insert(h.id, idx);
    }

    // Reverse topological order (downstream before upstream): each plant's
    // downstream ρ_acum is fully computed before it is summed in.
    let mut accumulated = vec![vec![0.0_f64; n_stages]; n_hydros];
    let mut integrated_equivalent = vec![vec![0.0_f64; n_stages]; n_hydros];
    let mut integrated_accumulated = vec![vec![0.0_f64; n_stages]; n_hydros];
    for t in 0..n_stages {
        for id in cascade.topological_order().iter().rev() {
            let h_idx =
                *id_to_index
                    .get(id)
                    .ok_or(EnergyConversionError::CascadeIndexMismatch {
                        expected: n_hydros,
                        got: topo_len,
                    })?;
            let rho_eq = per_hydro_stage[h_idx][t].equivalent_productivity_mw_per_m3s;
            let mean_own = mean_own_grid[h_idx][t];
            let downstream_contrib = if let Some(ds_id) = cascade.downstream(*id) {
                let ds_idx =
                    *id_to_index
                        .get(&ds_id)
                        .ok_or(EnergyConversionError::DanglingDownstream {
                            hydro_id: *id,
                            downstream_id: ds_id,
                        })?;
                accumulated[ds_idx][t]
            } else {
                0.0
            };
            accumulated[h_idx][t] = rho_eq + downstream_contrib;

            integrated_equivalent[h_idx][t] = mean_own;
            let integrated_downstream_contrib = if let Some(ds_id) = cascade.downstream(*id) {
                let ds_idx =
                    *id_to_index
                        .get(&ds_id)
                        .ok_or(EnergyConversionError::DanglingDownstream {
                            hydro_id: *id,
                            downstream_id: ds_id,
                        })?;
                integrated_accumulated[ds_idx][t]
            } else {
                0.0
            };
            integrated_accumulated[h_idx][t] = mean_own + integrated_downstream_contrib;
        }
    }

    Ok(CascadeGrids {
        accumulated,
        integrated_equivalent,
        integrated_accumulated,
    })
}

/// Derive the per-`(hydro, stage)` [`EnergyConversion`] cell.
///
/// `reference_volume_hm3` (absolute hm³) is stored verbatim — the
/// `v_min + fraction·(..)` span is applied once at resolver construction, never
/// re-applied here. For FPHA hydros pass `productivity = 0.0`;
/// `build_energy_conversion_set` overwrites it via the FPHA derivation.
fn derive_conversion_for_hydro(
    hydro: &Hydro,
    reference_volume_hm3: f64,
    productivity: f64,
) -> EnergyConversion {
    let reference_outflow_m3s = hydro.max_turbined_m3s;
    EnergyConversion {
        equivalent_productivity_mw_per_m3s: productivity,
        reference_volume_hm3,
        reference_outflow_m3s,
    }
}

/// Equivalent head `h_fore − h_tail(Q_ref) − h_loss` for a given forebay elevation.
///
/// The tailrace and hydraulic-loss terms are all the reference-point and mean
/// evaluators share; only `h_fore` differs between them.
fn equivalent_head_from_forebay(hydro: &Hydro, h_fore: f64, q_ref: f64) -> f64 {
    let h_tail = hydro
        .tailrace
        .as_ref()
        .map_or(0.0, |t| evaluate_tailrace(t, q_ref));
    let h_loss = hydro
        .hydraulic_losses
        .as_ref()
        .map_or(0.0, |m| evaluate_losses(m, h_fore - h_tail, q_ref));
    h_fore - h_tail - h_loss
}

/// Reference-point equivalent head at `h_fore = height(V_ref)`.
///
/// May be non-positive; [`fpha_equivalent_head`] surfaces that case as an error.
fn equivalent_head(hydro: &Hydro, table: &ForebayTable, v_ref: f64, q_ref: f64) -> f64 {
    equivalent_head_from_forebay(hydro, table.height(v_ref), q_ref)
}

/// Mean equivalent head at `h_fore = mean_height(V_lo, V_hi)` — the reference-point
/// head evaluated over the physical range instead of at a single point.
fn mean_equivalent_head(
    hydro: &Hydro,
    table: &ForebayTable,
    v_lo: f64,
    v_hi: f64,
    q_ref: f64,
) -> f64 {
    equivalent_head_from_forebay(hydro, table.mean_height(v_lo, v_hi), q_ref)
}

/// FPHA equivalent head ([`equivalent_head`]), erroring on non-positive results.
///
/// Returns [`EnergyConversionError::NonPositiveEquivalentHead`] when `h_eq <= 0.0`
/// (which would yield a non-physical `ρ_eq`).
fn fpha_equivalent_head(
    hydro: &Hydro,
    v_ref: f64,
    q_ref: f64,
    table: &ForebayTable,
) -> Result<f64, EnergyConversionError> {
    let h_eq = equivalent_head(hydro, table, v_ref, q_ref);
    if h_eq > 0.0 {
        Ok(h_eq)
    } else {
        Err(EnergyConversionError::NonPositiveEquivalentHead {
            hydro_id: hydro.id,
            h_eq,
        })
    }
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
    use cobre_core::{
        CascadeTopology, EntityId, HydraulicLossesModel, Hydro, HydroGenerationModel,
        HydroPenalties, TailraceModel,
    };
    use cobre_io::{
        HydroEnergyProductivityRow, HydroGeometryRow, HydroReferenceVolumeFractions,
        build_hydro_reference_volumes_resolved,
    };

    use super::super::productivity_override::build_hydro_energy_productivity_override;
    use super::super::types::EnergyConversionError;
    use super::*;
    use crate::hydro_models::{ProductionModelSet, ResolvedProductionModel};
    use crate::test_support;

    fn penalties_zero() -> HydroPenalties {
        HydroPenalties {
            spillage_cost: 0.0,
            diversion_cost: 0.0,
            turbined_cost: 0.0,
            storage_violation_below_cost: 0.0,
            filling_target_violation_cost: 0.0,
            turbined_violation_below_cost: 0.0,
            outflow_violation_below_cost: 0.0,
            outflow_violation_above_cost: 0.0,
            generation_violation_below_cost: 0.0,
            evaporation_violation_cost: 0.0,
            water_withdrawal_violation_cost: 0.0,
            water_withdrawal_violation_pos_cost: 0.0,
            water_withdrawal_violation_neg_cost: 0.0,
            evaporation_violation_pos_cost: 0.0,
            evaporation_violation_neg_cost: 0.0,
            inflow_nonnegativity_cost: 1000.0,
        }
    }

    fn make_hydro(id: i32, downstream: Option<i32>) -> Hydro {
        let mut hydro = Hydro {
            unit_groups: Vec::new(),
            id: EntityId::from(id),
            name: format!("Hydro {id}"),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            downstream_id: downstream.map(EntityId::from),
            travel_time_hours: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 100.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 50.0,
            specific_productivity_mw_per_m3s_per_m: None,
            min_generation_mw: 0.0,
            max_generation_mw: 45.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: penalties_zero(),
        };
        hydro.declare_mirror_unit_group(EntityId::from(1));
        hydro
    }

    /// Resolver returning, for every `(hydro, stage)`, the absolute hm³ for the
    /// 0.65 default fraction resolved against each plant's band — the value the
    /// production pipeline feeds after wiring the JSON-sourced reference volume.
    fn make_resolver(hydros: &[Hydro]) -> HydroReferenceVolumeFractions {
        resolved_resolver(hydros, 0.65, 2)
    }

    /// Build a resolver whose `get(hydro, stage)` returns the absolute hm³ for
    /// `fraction` resolved against each plant's `[v_min, v_max]` band — the
    /// resolved-hm³ semantics the consumers read after this wiring.
    fn resolved_resolver(
        hydros: &[Hydro],
        fraction: f64,
        n_stages: usize,
    ) -> HydroReferenceVolumeFractions {
        let resolved: Vec<(EntityId, StudyPos, f64)> = hydros
            .iter()
            .flat_map(|h| {
                let v = h.min_storage_hm3 + fraction * (h.max_storage_hm3 - h.min_storage_hm3);
                (0..n_stages).map(move |s| (h.id, StudyPos(s), v))
            })
            .collect();
        build_hydro_reference_volumes_resolved(&resolved, 0.0)
    }

    /// `StageId(0)..StageId(n_stages - 1)`: the 0-based domain ids every test
    /// fixture in this module uses (no pre-study-stage offset), so study
    /// position and domain id coincide.
    fn stage_ids_0_based(n_stages: usize) -> Vec<StageId> {
        (0..n_stages)
            .map(|s| StageId(i32::try_from(s).expect("test stage count fits in i32")))
            .collect()
    }

    #[test]
    fn builder_returns_grid_with_expected_dimensions() {
        // hydro id=1 (downstream=2) and hydro id=2 (terminal), both ρ_eq=1.0.
        // After the cascade walk:
        //   ρ_acum(id=2) = 1.0            (terminal, no downstream contrib)
        //   ρ_acum(id=1) = 1.0 + 1.0 = 2.0
        let n_stages = 2;
        let hydros = vec![make_hydro(1, Some(2)), make_hydro(2, None)];
        let cascade = CascadeTopology::build(&hydros);
        let resolver = make_resolver(&hydros);
        let pm = production_set(&[1.0, 1.0], n_stages);

        let set = build_energy_conversion_set(
            &hydros,
            &stage_ids_0_based(n_stages),
            &cascade,
            &resolver,
            &HashMap::new(),
            None,
            Some(&pm),
        )
        .expect("builder succeeds");

        assert_eq!(set.n_hydros(), 2);
        assert_eq!(set.n_stages(), n_stages);
        // hydro index 0 = id=1 (upstream), index 1 = id=2 (terminal).
        for s in 0..n_stages {
            assert_eq!(set.accumulated_productivity(1, s), 1.0); // id=2, terminal
            assert_eq!(set.accumulated_productivity(0, s), 2.0); // id=1, upstream of id=2
        }
    }

    fn make_hydro_with(
        id: i32,
        model: HydroGenerationModel,
        v_min: f64,
        v_max: f64,
        q_max: f64,
        specific: Option<f64>,
    ) -> Hydro {
        let mut h = make_hydro(id, None);
        h.generation_model = model;
        h.min_storage_hm3 = v_min;
        h.max_storage_hm3 = v_max;
        h.max_turbined_m3s = q_max;
        h.specific_productivity_mw_per_m3s_per_m = specific;
        // Re-declare against the post-mutation q_max — make_hydro()'s group already
        // mirrors the pre-mutation 50.0 and would otherwise go stale.
        h.unit_groups.clear();
        h.declare_mirror_unit_group(EntityId::from(1));
        h
    }

    fn constant_resolver(
        hydros: &[Hydro],
        fraction: f64,
        n_stages: usize,
    ) -> HydroReferenceVolumeFractions {
        resolved_resolver(hydros, fraction, n_stages)
    }

    /// Build a `ProductionModelSet` where every (hydro, stage) cell uses
    /// `ConstantProductivity` with the given per-hydro productivity values.
    ///
    /// `productivities[h]` is the productivity for hydro at declaration index `h`.
    fn production_set(productivities: &[f64], n_stages: usize) -> ProductionModelSet {
        let n_hydros = productivities.len();
        let models = productivities
            .iter()
            .map(|&p| {
                vec![ResolvedProductionModel::ConstantProductivity { productivity: p }; n_stages]
            })
            .collect();
        ProductionModelSet::new(models, &test_support::minimal_hydros(n_hydros), n_stages)
    }

    #[test]
    fn constant_productivity_yields_input_scalar() {
        let n_stages = 2;
        let hydros = vec![make_hydro_with(
            1,
            HydroGenerationModel::ConstantProductivity,
            100.0,
            200.0,
            50.0,
            None,
        )];
        let cascade = CascadeTopology::build(&hydros);
        let resolver = constant_resolver(&hydros, 0.65, n_stages);
        let pm = production_set(&[0.9], n_stages);

        let set = build_energy_conversion_set(
            &hydros,
            &stage_ids_0_based(n_stages),
            &cascade,
            &resolver,
            &HashMap::new(),
            None,
            Some(&pm),
        )
        .expect("builder succeeds");

        let c = set.conversion(0, 0);
        assert_eq!(c.equivalent_productivity_mw_per_m3s, 0.9);
        assert_eq!(c.reference_volume_hm3, 100.0 + 0.65 * (200.0 - 100.0));
        assert_eq!(c.reference_outflow_m3s, 50.0);
    }

    #[test]
    fn linearized_head_yields_input_scalar() {
        let n_stages = 1;
        let hydros = vec![make_hydro_with(
            1,
            HydroGenerationModel::LinearizedHead,
            100.0,
            200.0,
            40.0,
            None,
        )];
        let cascade = CascadeTopology::build(&hydros);
        let resolver = constant_resolver(&hydros, 0.5, n_stages);
        let pm = production_set(&[1.2], n_stages);

        let set = build_energy_conversion_set(
            &hydros,
            &stage_ids_0_based(n_stages),
            &cascade,
            &resolver,
            &HashMap::new(),
            None,
            Some(&pm),
        )
        .expect("builder succeeds");

        assert_eq!(set.conversion(0, 0).equivalent_productivity_mw_per_m3s, 1.2);
    }

    #[test]
    fn reference_volume_uses_fraction() {
        let hydros = vec![make_hydro_with(
            1,
            HydroGenerationModel::ConstantProductivity,
            100.0,
            200.0,
            50.0,
            None,
        )];
        let cascade = CascadeTopology::build(&hydros);

        for f in [0.1_f64, 0.5, 1.0] {
            let resolver = constant_resolver(&hydros, f, 1);
            let set = build_energy_conversion_set(
                &hydros,
                &stage_ids_0_based(1),
                &cascade,
                &resolver,
                &HashMap::new(),
                None,
                None,
            )
            .expect("builder succeeds");
            let expected = 100.0 + f * (200.0 - 100.0);
            assert!(
                (set.conversion(0, 0).reference_volume_hm3 - expected).abs() < f64::EPSILON,
                "fraction {f}: V_ref expected {expected}, got {}",
                set.conversion(0, 0).reference_volume_hm3
            );
        }
    }

    #[test]
    fn per_season_override_produces_different_v_ref_per_stage() {
        let n_stages = 4;
        let hydros = vec![make_hydro_with(
            1,
            HydroGenerationModel::ConstantProductivity,
            100.0,
            200.0,
            50.0,
            None,
        )];
        let cascade = CascadeTopology::build(&hydros);
        // 4 stages alternating season 0 (fraction 0.50 → 150 hm³) and season 1
        // (fraction 0.70 → 170 hm³) on the [100, 200] band, resolved per stage.
        let id = hydros[0].id;
        let per_stage_hm3 = vec![
            (id, StudyPos(0), 150.0),
            (id, StudyPos(1), 170.0),
            (id, StudyPos(2), 150.0),
            (id, StudyPos(3), 170.0),
        ];
        let resolver = build_hydro_reference_volumes_resolved(&per_stage_hm3, 0.0);
        let pm = production_set(&[0.9], n_stages);
        let set = build_energy_conversion_set(
            &hydros,
            &stage_ids_0_based(n_stages),
            &cascade,
            &resolver,
            &HashMap::new(),
            None,
            Some(&pm),
        )
        .expect("builder succeeds");

        let expected = [150.0_f64, 170.0, 150.0, 170.0];
        for (s, want) in expected.iter().enumerate() {
            assert!(
                (set.conversion(0, s).reference_volume_hm3 - want).abs() < f64::EPSILON,
                "stage {s}: expected V_ref {want}, got {}",
                set.conversion(0, s).reference_volume_hm3
            );
            assert_eq!(set.conversion(0, s).equivalent_productivity_mw_per_m3s, 0.9);
        }
    }

    #[test]
    fn invalid_storage_range_is_rejected() {
        let hydros = vec![make_hydro_with(
            42,
            HydroGenerationModel::ConstantProductivity,
            200.0,
            100.0,
            50.0,
            None,
        )];
        let cascade = CascadeTopology::build(&hydros);
        let resolver = constant_resolver(&hydros, 0.65, 1);

        let err = build_energy_conversion_set(
            &hydros,
            &stage_ids_0_based(1),
            &cascade,
            &resolver,
            &HashMap::new(),
            None,
            None,
        )
        .unwrap_err();
        match err {
            EnergyConversionError::InvalidStorageRange { hydro_id, .. } => {
                assert_eq!(hydro_id, hydros[0].id);
            }
            other => panic!("expected InvalidStorageRange, got: {other:?}"),
        }
    }

    #[test]
    fn negative_max_turbined_is_rejected() {
        let hydros = vec![make_hydro_with(
            42,
            HydroGenerationModel::ConstantProductivity,
            100.0,
            200.0,
            -1.0,
            None,
        )];
        let cascade = CascadeTopology::build(&hydros);
        let resolver = constant_resolver(&hydros, 0.65, 1);

        let err = build_energy_conversion_set(
            &hydros,
            &stage_ids_0_based(1),
            &cascade,
            &resolver,
            &HashMap::new(),
            None,
            None,
        )
        .unwrap_err();
        match err {
            EnergyConversionError::NegativeMaxTurbined { hydro_id, q_max } => {
                assert_eq!(hydro_id, hydros[0].id);
                assert_eq!(q_max, -1.0);
            }
            other => panic!("expected NegativeMaxTurbined, got: {other:?}"),
        }
    }

    /// An FPHA hydro that supplies ρ_esp but no VHA geometry is rejected with
    /// `FphaMissingEquivalentProductivity` (a permissive fallback to ρ_eq=0.0
    /// would silently mis-model the plant).
    #[test]
    fn fpha_hydro_missing_vha_is_rejected_with_actionable_error() {
        let hydros = vec![make_hydro_with(
            5,
            HydroGenerationModel::Fpha,
            100.0,
            200.0,
            50.0,
            Some(0.01),
        )];
        let cascade = CascadeTopology::build(&hydros);
        let resolver = constant_resolver(&hydros, 0.5, 1);
        let err = build_energy_conversion_set(
            &hydros,
            &stage_ids_0_based(1),
            &cascade,
            &resolver,
            &HashMap::new(),
            None,
            None,
        )
        .unwrap_err();
        match err {
            EnergyConversionError::FphaMissingEquivalentProductivity {
                hydro_id,
                ref hydro_name,
                stage,
            } => {
                assert_eq!(hydro_id, hydros[0].id);
                assert!(
                    hydro_name.contains("Hydro 5"),
                    "error should mention the hydro name, got: {hydro_name}"
                );
                assert_eq!(stage, 0);
            }
            other => panic!("expected FphaMissingEquivalentProductivity, got: {other:?}"),
        }
    }

    // ── FPHA derivation tests ──────────────────────────────────────────────

    /// Create a flat VHA table (constant height) for testing.
    fn vha_constant_height(hydro_id: EntityId, height: f64) -> (EntityId, Vec<HydroGeometryRow>) {
        (
            hydro_id,
            vec![
                HydroGeometryRow {
                    hydro_id,
                    volume_hm3: 0.0,
                    height_m: height,
                    area_km2: 1.0,
                },
                HydroGeometryRow {
                    hydro_id,
                    volume_hm3: 1000.0,
                    height_m: height,
                    area_km2: 1.0,
                },
            ],
        )
    }

    fn fpha_hydro_for_tests(id: i32) -> Hydro {
        make_hydro_with(
            id,
            HydroGenerationModel::Fpha,
            100.0,
            200.0,
            50.0,
            Some(0.0090),
        )
    }

    #[test]
    fn fpha_rho_eq_from_vha_no_tailrace_no_losses() {
        let mut hydro = fpha_hydro_for_tests(7);
        hydro.tailrace = None;
        hydro.hydraulic_losses = None;
        let hydros = vec![hydro];
        let cascade = CascadeTopology::build(&hydros);
        let resolver = constant_resolver(&hydros, 0.65, 1);
        let (id, rows) = vha_constant_height(hydros[0].id, 400.0);
        let mut map = HashMap::new();
        map.insert(id, rows);

        let set = build_energy_conversion_set(
            &hydros,
            &stage_ids_0_based(1),
            &cascade,
            &resolver,
            &map,
            None,
            None,
        )
        .expect("builder succeeds");

        let got = set.conversion(0, 0).equivalent_productivity_mw_per_m3s;
        let expected = 0.0090 * 400.0;
        assert!(
            (got - expected).abs() < 1e-12,
            "got {got}, expected {expected}"
        );
    }

    #[test]
    fn fpha_rho_eq_with_factor_losses() {
        let mut hydro = fpha_hydro_for_tests(7);
        hydro.tailrace = None;
        hydro.hydraulic_losses = Some(HydraulicLossesModel::Factor { value: 0.05 });
        let hydros = vec![hydro];
        let cascade = CascadeTopology::build(&hydros);
        let resolver = constant_resolver(&hydros, 0.65, 1);
        let (id, rows) = vha_constant_height(hydros[0].id, 400.0);
        let mut map = HashMap::new();
        map.insert(id, rows);

        let set = build_energy_conversion_set(
            &hydros,
            &stage_ids_0_based(1),
            &cascade,
            &resolver,
            &map,
            None,
            None,
        )
        .expect("builder succeeds");

        // h_eq = 400 - 0 - 0.05 * 400 = 380; rho_eq = 0.009 * 380 = 3.42
        let got = set.conversion(0, 0).equivalent_productivity_mw_per_m3s;
        let expected = 0.0090 * 380.0;
        assert!(
            (got - expected).abs() < 1e-12,
            "got {got}, expected {expected}"
        );
    }

    #[test]
    fn fpha_rho_eq_with_constant_losses() {
        let mut hydro = fpha_hydro_for_tests(7);
        hydro.tailrace = None;
        hydro.hydraulic_losses = Some(HydraulicLossesModel::Constant { value_m: 5.0 });
        let hydros = vec![hydro];
        let cascade = CascadeTopology::build(&hydros);
        let resolver = constant_resolver(&hydros, 0.65, 1);
        let (id, rows) = vha_constant_height(hydros[0].id, 400.0);
        let mut map = HashMap::new();
        map.insert(id, rows);

        let set = build_energy_conversion_set(
            &hydros,
            &stage_ids_0_based(1),
            &cascade,
            &resolver,
            &map,
            None,
            None,
        )
        .expect("builder succeeds");

        // h_eq = 400 - 0 - 5 = 395; rho_eq = 0.009 * 395 = 3.555
        let got = set.conversion(0, 0).equivalent_productivity_mw_per_m3s;
        let expected = 0.0090 * 395.0;
        assert!(
            (got - expected).abs() < 1e-12,
            "got {got}, expected {expected}"
        );
    }

    /// An FPHA hydro that supplies VHA geometry but no ρ_esp is rejected (a
    /// permissive fallback to ρ_eq=0.0 would silently mis-model the plant).
    #[test]
    fn fpha_missing_rho_esp_is_rejected() {
        let mut hydro = fpha_hydro_for_tests(7);
        hydro.specific_productivity_mw_per_m3s_per_m = None;
        hydro.tailrace = None;
        hydro.hydraulic_losses = None;
        let hydros = vec![hydro];
        let cascade = CascadeTopology::build(&hydros);
        let resolver = constant_resolver(&hydros, 0.65, 1);
        let (id, rows) = vha_constant_height(hydros[0].id, 400.0);
        let mut map = HashMap::new();
        map.insert(id, rows);

        let err = build_energy_conversion_set(
            &hydros,
            &stage_ids_0_based(1),
            &cascade,
            &resolver,
            &map,
            None,
            None,
        )
        .unwrap_err();
        match err {
            EnergyConversionError::FphaMissingEquivalentProductivity {
                hydro_id,
                ref hydro_name,
                stage: _,
            } => {
                assert_eq!(hydro_id, hydros[0].id);
                assert!(
                    hydro_name.contains("Hydro 7"),
                    "error should mention hydro name, got: {hydro_name}"
                );
            }
            other => panic!("expected FphaMissingEquivalentProductivity, got: {other:?}"),
        }
    }

    /// An FPHA hydro that supplies ρ_esp but no VHA geometry is rejected (a
    /// permissive fallback to ρ_eq=0.0 would silently mis-model the plant).
    #[test]
    fn fpha_missing_vha_is_rejected() {
        let hydro = fpha_hydro_for_tests(7); // has rho_esp = Some(0.009)
        let hydros = vec![hydro];
        let cascade = CascadeTopology::build(&hydros);
        let resolver = constant_resolver(&hydros, 0.65, 1);

        let err = build_energy_conversion_set(
            &hydros,
            &stage_ids_0_based(1),
            &cascade,
            &resolver,
            &HashMap::new(),
            None,
            None,
        )
        .unwrap_err();
        match err {
            EnergyConversionError::FphaMissingEquivalentProductivity {
                hydro_id,
                ref hydro_name,
                stage: _,
            } => {
                assert_eq!(hydro_id, hydros[0].id);
                assert!(
                    hydro_name.contains("Hydro 7"),
                    "error should mention hydro name, got: {hydro_name}"
                );
            }
            other => panic!("expected FphaMissingEquivalentProductivity, got: {other:?}"),
        }
    }

    #[test]
    fn fpha_rejects_non_positive_h_eq() {
        let mut hydro = fpha_hydro_for_tests(7);
        hydro.tailrace = None;
        // Constant loss equal to h_fore -> h_eq = 0 -> non-positive -> Err.
        hydro.hydraulic_losses = Some(HydraulicLossesModel::Constant { value_m: 400.0 });
        let hydros = vec![hydro];
        let cascade = CascadeTopology::build(&hydros);
        let resolver = constant_resolver(&hydros, 0.65, 1);
        let (id, rows) = vha_constant_height(hydros[0].id, 400.0);
        let mut map = HashMap::new();
        map.insert(id, rows);

        let err = build_energy_conversion_set(
            &hydros,
            &stage_ids_0_based(1),
            &cascade,
            &resolver,
            &map,
            None,
            None,
        )
        .unwrap_err();
        match err {
            EnergyConversionError::NonPositiveEquivalentHead { hydro_id, h_eq } => {
                assert_eq!(hydro_id, hydros[0].id);
                assert!(h_eq <= 0.0);
            }
            other => panic!("expected NonPositiveEquivalentHead, got: {other:?}"),
        }
    }

    #[test]
    fn fpha_propagates_forebay_table_error() {
        // An EMPTY VHA fails ForebayTable::new (InsufficientPoints). A single row
        // is now valid (constant run-of-river forebay), so only the zero-row case
        // still propagates as ForebayTableInvalid.
        let hydro = fpha_hydro_for_tests(7);
        let hydros = vec![hydro];
        let cascade = CascadeTopology::build(&hydros);
        let resolver = constant_resolver(&hydros, 0.65, 1);
        let mut map = HashMap::new();
        map.insert(hydros[0].id, Vec::<HydroGeometryRow>::new());

        let err = build_energy_conversion_set(
            &hydros,
            &stage_ids_0_based(1),
            &cascade,
            &resolver,
            &map,
            None,
            None,
        )
        .unwrap_err();
        match err {
            EnergyConversionError::ForebayTableInvalid { hydro_id, .. } => {
                assert_eq!(hydro_id, hydros[0].id);
            }
            other => panic!("expected ForebayTableInvalid, got: {other:?}"),
        }
    }

    // ── cascade accumulator tests ─────────────────────────────────────────────

    /// A->B->C linear cascade. ρ_eq values: A=2.0, B=3.0, C=5.0 at stage 0.
    /// Expected ρ_acum: C=5.0, B=3+5=8.0, A=2+8=10.0.
    #[test]
    fn linear_cascade_accumulates_three_levels() {
        let mut a = make_hydro(0, Some(1));
        a.generation_model = HydroGenerationModel::ConstantProductivity;
        let mut b = make_hydro(1, Some(2));
        b.generation_model = HydroGenerationModel::ConstantProductivity;
        let mut c = make_hydro(2, None);
        c.generation_model = HydroGenerationModel::ConstantProductivity;
        let hydros = vec![a, b, c];
        let cascade = CascadeTopology::build(&hydros);
        let resolver = constant_resolver(&hydros, 0.65, 1);
        // Declaration order: A(idx=0)=2.0, B(idx=1)=3.0, C(idx=2)=5.0.
        let pm = production_set(&[2.0, 3.0, 5.0], 1);

        let set = build_energy_conversion_set(
            &hydros,
            &stage_ids_0_based(1),
            &cascade,
            &resolver,
            &HashMap::new(),
            None,
            Some(&pm),
        )
        .expect("builder succeeds");

        // Index by declaration order: A=0, B=1, C=2.
        assert_eq!(set.accumulated_productivity(0, 0), 10.0); // A
        assert_eq!(set.accumulated_productivity(1, 0), 8.0); // B
        assert_eq!(set.accumulated_productivity(2, 0), 5.0); // C
    }

    /// A->C and B->C branching cascade. ρ_eq: A=1.0, B=2.0, C=4.0.
    /// ρ_acum: C=4.0, A=1+4=5.0, B=2+4=6.0.
    #[test]
    fn branching_cascade_accumulates_correctly() {
        let mut a = make_hydro(0, Some(2));
        a.generation_model = HydroGenerationModel::ConstantProductivity;
        let mut b = make_hydro(1, Some(2));
        b.generation_model = HydroGenerationModel::ConstantProductivity;
        let mut c = make_hydro(2, None);
        c.generation_model = HydroGenerationModel::ConstantProductivity;
        let hydros = vec![a, b, c];
        let cascade = CascadeTopology::build(&hydros);
        let resolver = constant_resolver(&hydros, 0.65, 1);
        // Declaration order: A(idx=0)=1.0, B(idx=1)=2.0, C(idx=2)=4.0.
        let pm = production_set(&[1.0, 2.0, 4.0], 1);

        let set = build_energy_conversion_set(
            &hydros,
            &stage_ids_0_based(1),
            &cascade,
            &resolver,
            &HashMap::new(),
            None,
            Some(&pm),
        )
        .expect("builder succeeds");

        // A(idx=0), B(idx=1), C(idx=2).
        assert_eq!(set.accumulated_productivity(2, 0), 4.0); // C terminal
        assert_eq!(set.accumulated_productivity(0, 0), 5.0); // A = 1 + 4
        assert_eq!(set.accumulated_productivity(1, 0), 6.0); // B = 2 + 4
    }

    /// Assert `integrated_accumulated_productivity(h,t) ==
    /// integrated_equivalent_productivity(h,t) + Σ
    /// integrated_accumulated_productivity(direct_downstream, t)` for every hydro
    /// and stage, walking the cascade's own downstream links.
    fn assert_integrated_cascade_recurrence_holds(
        hydros: &[Hydro],
        cascade: &CascadeTopology,
        set: &EnergyConversionSet,
        n_stages: usize,
    ) {
        for t in 0..n_stages {
            for (h_idx, hydro) in hydros.iter().enumerate() {
                assert_ne!(
                    set.integrated_equivalent_productivity(h_idx, t).to_bits(),
                    set.conversion(h_idx, t)
                        .equivalent_productivity_mw_per_m3s
                        .to_bits(),
                    "hydro {h_idx}, stage {t}: fixture must make the two evaluators differ"
                );
                let downstream_contrib = cascade.downstream(hydro.id).map_or(0.0, |ds_id| {
                    let ds_idx = hydros
                        .iter()
                        .position(|h| h.id == ds_id)
                        .expect("downstream id present in hydros");
                    set.integrated_accumulated_productivity(ds_idx, t)
                });
                let expected =
                    set.integrated_equivalent_productivity(h_idx, t) + downstream_contrib;
                assert_eq!(
                    set.integrated_accumulated_productivity(h_idx, t).to_bits(),
                    expected.to_bits(),
                    "hydro {h_idx}, stage {t}: integrated cascade recurrence"
                );
            }
        }
    }

    /// A->B->C linear cascade where every plant has real VHA geometry and a
    /// resolved ρ_esp, so the mean own term genuinely diverges from the
    /// reference-point own term; the integrated cascade grid must still satisfy
    /// the downstream-sum recurrence.
    #[test]
    fn integrated_cascade_recurrence_holds_on_a_linear_topology() {
        let n_stages = 1;
        let mut a = make_hydro(0, Some(1));
        a.generation_model = HydroGenerationModel::ConstantProductivity;
        a.min_storage_hm3 = 100.0;
        a.max_storage_hm3 = 300.0;
        a.specific_productivity_mw_per_m3s_per_m = Some(0.01);
        let mut b = make_hydro(1, Some(2));
        b.generation_model = HydroGenerationModel::ConstantProductivity;
        b.min_storage_hm3 = 200.0;
        b.max_storage_hm3 = 600.0;
        b.specific_productivity_mw_per_m3s_per_m = Some(0.02);
        let mut c = make_hydro(2, None);
        c.generation_model = HydroGenerationModel::ConstantProductivity;
        c.min_storage_hm3 = 50.0;
        c.max_storage_hm3 = 150.0;
        c.specific_productivity_mw_per_m3s_per_m = Some(0.03);
        let hydros = vec![a, b, c];
        let cascade = CascadeTopology::build(&hydros);
        let resolver = constant_resolver(&hydros, 0.65, n_stages);
        let pm = production_set(&[2.0, 3.0, 5.0], n_stages);

        let mut vha_map = HashMap::new();
        let (id_a, rows_a) = vha_rows(hydros[0].id, &[(0.0, 500.0), (1000.0, 700.0)]);
        vha_map.insert(id_a, rows_a);
        let (id_b, rows_b) = vha_rows(hydros[1].id, &[(0.0, 300.0), (1000.0, 500.0)]);
        vha_map.insert(id_b, rows_b);
        let (id_c, rows_c) = vha_rows(hydros[2].id, &[(0.0, 200.0), (1000.0, 260.0)]);
        vha_map.insert(id_c, rows_c);

        let set = build_energy_conversion_set(
            &hydros,
            &stage_ids_0_based(n_stages),
            &cascade,
            &resolver,
            &vha_map,
            None,
            Some(&pm),
        )
        .expect("builder succeeds");

        assert_integrated_cascade_recurrence_holds(&hydros, &cascade, &set, n_stages);
    }

    /// A->C, B->C branching cascade with the same real-range/VHA setup as the
    /// linear case; the integrated cascade grid must satisfy the downstream-sum
    /// recurrence at the branch point too.
    #[test]
    fn integrated_cascade_recurrence_holds_on_a_branching_topology() {
        let n_stages = 1;
        let mut a = make_hydro(0, Some(2));
        a.generation_model = HydroGenerationModel::ConstantProductivity;
        a.min_storage_hm3 = 100.0;
        a.max_storage_hm3 = 300.0;
        a.specific_productivity_mw_per_m3s_per_m = Some(0.01);
        let mut b = make_hydro(1, Some(2));
        b.generation_model = HydroGenerationModel::ConstantProductivity;
        b.min_storage_hm3 = 150.0;
        b.max_storage_hm3 = 450.0;
        b.specific_productivity_mw_per_m3s_per_m = Some(0.02);
        let mut c = make_hydro(2, None);
        c.generation_model = HydroGenerationModel::ConstantProductivity;
        c.min_storage_hm3 = 50.0;
        c.max_storage_hm3 = 150.0;
        c.specific_productivity_mw_per_m3s_per_m = Some(0.03);
        let hydros = vec![a, b, c];
        let cascade = CascadeTopology::build(&hydros);
        let resolver = constant_resolver(&hydros, 0.65, n_stages);
        let pm = production_set(&[1.0, 2.0, 4.0], n_stages);

        let mut vha_map = HashMap::new();
        let (id_a, rows_a) = vha_rows(hydros[0].id, &[(0.0, 500.0), (1000.0, 700.0)]);
        vha_map.insert(id_a, rows_a);
        let (id_b, rows_b) = vha_rows(hydros[1].id, &[(0.0, 400.0), (1000.0, 600.0)]);
        vha_map.insert(id_b, rows_b);
        let (id_c, rows_c) = vha_rows(hydros[2].id, &[(0.0, 200.0), (1000.0, 260.0)]);
        vha_map.insert(id_c, rows_c);

        let set = build_energy_conversion_set(
            &hydros,
            &stage_ids_0_based(n_stages),
            &cascade,
            &resolver,
            &vha_map,
            None,
            Some(&pm),
        )
        .expect("builder succeeds");

        assert_integrated_cascade_recurrence_holds(&hydros, &cascade, &set, n_stages);
    }

    /// Build the same A->C, B->C branching cascade with two different
    /// declaration orders, non-zero mutually distinct productivities, and
    /// confirm all four scope x evaluator accessors are bit-for-bit identical
    /// when indexed by EntityId.
    #[test]
    fn declaration_order_invariance() {
        let downstream = |id: i32| if id == 2 { None } else { Some(2) };
        let productivity_for = |id: i32| match id {
            0 => 1.0,
            1 => 2.0,
            2 => 4.0,
            _ => unreachable!(),
        };
        let make_branching = |order: &[i32]| -> Vec<Hydro> {
            order
                .iter()
                .map(|&id| {
                    let mut h = make_hydro(id, downstream(id));
                    h.generation_model = HydroGenerationModel::ConstantProductivity;
                    h
                })
                .collect()
        };

        let order_abc = [0_i32, 1, 2];
        let order_cab = [2_i32, 0, 1];

        let hydros_abc = make_branching(&order_abc);
        let hydros_cab = make_branching(&order_cab);

        let cascade_abc = CascadeTopology::build(&hydros_abc);
        let cascade_cab = CascadeTopology::build(&hydros_cab);

        let resolver_abc = constant_resolver(&hydros_abc, 0.65, 1);
        let resolver_cab = constant_resolver(&hydros_cab, 0.65, 1);

        let pm_abc = production_set(
            &order_abc
                .iter()
                .map(|&id| productivity_for(id))
                .collect::<Vec<_>>(),
            1,
        );
        let pm_cab = production_set(
            &order_cab
                .iter()
                .map(|&id| productivity_for(id))
                .collect::<Vec<_>>(),
            1,
        );

        let set_abc = build_energy_conversion_set(
            &hydros_abc,
            &stage_ids_0_based(1),
            &cascade_abc,
            &resolver_abc,
            &HashMap::new(),
            None,
            Some(&pm_abc),
        )
        .expect("abc order");
        let set_cab = build_energy_conversion_set(
            &hydros_cab,
            &stage_ids_0_based(1),
            &cascade_cab,
            &resolver_cab,
            &HashMap::new(),
            None,
            Some(&pm_cab),
        )
        .expect("cab order");

        let idx_abc: HashMap<i32, usize> = hydros_abc
            .iter()
            .enumerate()
            .map(|(i, h)| (h.id.0, i))
            .collect();
        let idx_cab: HashMap<i32, usize> = hydros_cab
            .iter()
            .enumerate()
            .map(|(i, h)| (h.id.0, i))
            .collect();

        for entity_id in [0_i32, 1, 2] {
            let i_abc = idx_abc[&entity_id];
            let i_cab = idx_cab[&entity_id];

            let own_abc = set_abc
                .conversion(i_abc, 0)
                .equivalent_productivity_mw_per_m3s;
            let own_cab = set_cab
                .conversion(i_cab, 0)
                .equivalent_productivity_mw_per_m3s;
            assert_eq!(
                own_abc.to_bits(),
                own_cab.to_bits(),
                "entity {entity_id}: own reference-point"
            );

            let cascade_abc_val = set_abc.accumulated_productivity(i_abc, 0);
            let cascade_cab_val = set_cab.accumulated_productivity(i_cab, 0);
            assert_eq!(
                cascade_abc_val.to_bits(),
                cascade_cab_val.to_bits(),
                "entity {entity_id}: cascade reference-point"
            );

            let integrated_own_abc = set_abc.integrated_equivalent_productivity(i_abc, 0);
            let integrated_own_cab = set_cab.integrated_equivalent_productivity(i_cab, 0);
            assert_eq!(
                integrated_own_abc.to_bits(),
                integrated_own_cab.to_bits(),
                "entity {entity_id}: integrated own"
            );

            let integrated_cascade_abc = set_abc.integrated_accumulated_productivity(i_abc, 0);
            let integrated_cascade_cab = set_cab.integrated_accumulated_productivity(i_cab, 0);
            assert_eq!(
                integrated_cascade_abc.to_bits(),
                integrated_cascade_cab.to_bits(),
                "entity {entity_id}: integrated cascade"
            );
        }
    }

    #[test]
    fn integrated_grids_match_reference_point_grids() {
        let n_stages = 2;
        let mut a = make_hydro(0, Some(1));
        a.generation_model = HydroGenerationModel::ConstantProductivity;
        let mut b = make_hydro(1, Some(2));
        b.generation_model = HydroGenerationModel::ConstantProductivity;
        let mut c = make_hydro(2, None);
        c.generation_model = HydroGenerationModel::ConstantProductivity;
        let hydros = vec![a, b, c];
        let cascade = CascadeTopology::build(&hydros);
        let resolver = constant_resolver(&hydros, 0.65, n_stages);
        let pm = production_set(&[2.0, 3.0, 5.0], n_stages);

        let set = build_energy_conversion_set(
            &hydros,
            &stage_ids_0_based(n_stages),
            &cascade,
            &resolver,
            &HashMap::new(),
            None,
            Some(&pm),
        )
        .expect("builder succeeds");

        for h in 0..hydros.len() {
            for t in 0..n_stages {
                let own = set.conversion(h, t).equivalent_productivity_mw_per_m3s;
                assert_eq!(
                    set.integrated_equivalent_productivity(h, t).to_bits(),
                    own.to_bits(),
                    "hydro {h}, stage {t}: integrated own vs reference-point"
                );
                assert_eq!(
                    set.integrated_accumulated_productivity(h, t).to_bits(),
                    set.accumulated_productivity(h, t).to_bits(),
                    "hydro {h}, stage {t}: integrated cascade vs reference-point"
                );
            }
        }
    }

    /// A hydro whose downstream_id points to an EntityId not in the hydros slice
    /// must return DanglingDownstream.
    #[test]
    fn dangling_downstream_is_rejected() {
        // Two hydros (id=0 downstream=99, id=1 terminal) so topo length stays 2
        // == hydros.len(); otherwise the shorter topo would fire
        // CascadeIndexMismatch before DanglingDownstream can.
        let mut h0 = make_hydro(0, Some(99));
        h0.generation_model = HydroGenerationModel::ConstantProductivity;
        let mut h1 = make_hydro(1, None);
        h1.generation_model = HydroGenerationModel::ConstantProductivity;
        let hydros = vec![h0, h1];
        let cascade = CascadeTopology::build(&hydros);
        let resolver = constant_resolver(&hydros, 0.65, 1);

        let err = build_energy_conversion_set(
            &hydros,
            &stage_ids_0_based(1),
            &cascade,
            &resolver,
            &HashMap::new(),
            None,
            None,
        )
        .unwrap_err();
        match err {
            EnergyConversionError::DanglingDownstream {
                hydro_id,
                downstream_id,
            } => {
                assert_eq!(hydro_id, EntityId::from(0));
                assert_eq!(downstream_id, EntityId::from(99));
            }
            other => panic!("expected DanglingDownstream, got: {other:?}"),
        }
    }

    /// When cascade.topological_order().len() != hydros.len(), return
    /// CascadeIndexMismatch.
    #[test]
    fn cascade_index_mismatch_is_rejected() {
        let h0 = make_hydro(0, Some(1));
        let h1 = make_hydro(1, None);
        let h2 = make_hydro(2, None);

        let short_cascade = CascadeTopology::build(&[h0.clone(), h1.clone()]);

        let hydros_three = vec![h0, h1, h2];
        let resolver = constant_resolver(&hydros_three, 0.65, 1);

        let err = build_energy_conversion_set(
            &hydros_three,
            &stage_ids_0_based(1),
            &short_cascade,
            &resolver,
            &HashMap::new(),
            None,
            None,
        )
        .unwrap_err();
        match err {
            EnergyConversionError::CascadeIndexMismatch { expected, got } => {
                assert_eq!(expected, 3);
                assert_eq!(got, 2);
            }
            other => panic!("expected CascadeIndexMismatch, got: {other:?}"),
        }
    }

    /// FPHA hydro with no VHA and no rho_esp but an override that returns a
    /// constant value for every stage succeeds, and the resolved
    /// equivalent_productivity equals the override value.
    #[test]
    fn fpha_with_override_only_succeeds() {
        let mut hydro = fpha_hydro_for_tests(7);
        hydro.specific_productivity_mw_per_m3s_per_m = None;
        hydro.tailrace = None;
        hydro.hydraulic_losses = None;
        let hydros = vec![hydro];
        let cascade = CascadeTopology::build(&hydros);
        let resolver = constant_resolver(&hydros, 0.65, 3);
        let override_table =
            build_hydro_energy_productivity_override(&[HydroEnergyProductivityRow {
                hydro_id: hydros[0].id,
                stage_id: None,
                equivalent_productivity_mw_per_m3s: Some(2.5),
                reference_outflow_m3s: None,
                specific_productivity_mw_per_m3s_per_m: None,
            }])
            .expect("override builds");

        let set = build_energy_conversion_set(
            &hydros,
            &stage_ids_0_based(3),
            &cascade,
            &resolver,
            &HashMap::new(),
            Some(&override_table),
            None,
        )
        .expect("builder succeeds with override only");

        for s in 0..3 {
            let cell = set.conversion(0, s);
            assert!(
                (cell.equivalent_productivity_mw_per_m3s - 2.5).abs() < 1e-12,
                "stage {s}: expected 2.5, got {}",
                cell.equivalent_productivity_mw_per_m3s
            );
        }
    }

    /// A stage-specific override keyed at domain `stage_id = 60` resolves at
    /// the study's ONLY stage, whose position is 0 but whose domain id is 60
    /// (a non-0-based study). Keying by position instead of `StageId` would
    /// find nothing at this key and fall through to `FphaMissingEquivalentProductivity`.
    #[test]
    fn fpha_override_resolves_by_non_zero_based_domain_stage_id() {
        let mut hydro = fpha_hydro_for_tests(7);
        hydro.specific_productivity_mw_per_m3s_per_m = None;
        hydro.tailrace = None;
        hydro.hydraulic_losses = None;
        let hydros = vec![hydro];
        let cascade = CascadeTopology::build(&hydros);
        let resolver = constant_resolver(&hydros, 0.65, 1);
        let override_table =
            build_hydro_energy_productivity_override(&[HydroEnergyProductivityRow {
                hydro_id: hydros[0].id,
                stage_id: Some(60),
                equivalent_productivity_mw_per_m3s: Some(9.1),
                reference_outflow_m3s: None,
                specific_productivity_mw_per_m3s_per_m: None,
            }])
            .expect("override builds");

        let set = build_energy_conversion_set(
            &hydros,
            &[StageId(60)],
            &cascade,
            &resolver,
            &HashMap::new(),
            Some(&override_table),
            None,
        )
        .expect("builder resolves the override at the stage's domain id");

        let cell = set.conversion(0, 0);
        assert!(
            (cell.equivalent_productivity_mw_per_m3s - 9.1).abs() < 1e-12,
            "position 0 (domain id 60) must read the StageId(60) override, got {}",
            cell.equivalent_productivity_mw_per_m3s
        );
    }

    /// A ConstantProductivity hydro with no VHA, no rho_esp, and no override
    /// must succeed — the FPHA correctness gate must not apply to non-FPHA
    /// generation models. The productivity is supplied via `ProductionModelSet`;
    /// the test verifies the value is passed through to the conversion cell.
    #[test]
    fn constant_productivity_bypasses_gate() {
        let mut hydro = make_hydro(0, None);
        hydro.generation_model = HydroGenerationModel::ConstantProductivity;
        hydro.specific_productivity_mw_per_m3s_per_m = None;
        hydro.tailrace = None;
        hydro.hydraulic_losses = None;
        let hydros = vec![hydro];
        let cascade = CascadeTopology::build(&hydros);
        let resolver = constant_resolver(&hydros, 0.65, 1);
        let pm = production_set(&[1.5], 1);

        let set = build_energy_conversion_set(
            &hydros,
            &stage_ids_0_based(1),
            &cascade,
            &resolver,
            &HashMap::new(),
            None,
            Some(&pm),
        )
        .expect("non-FPHA hydro succeeds despite missing FPHA inputs");

        let cell = set.conversion(0, 0);
        assert!((cell.equivalent_productivity_mw_per_m3s - 1.5).abs() < 1e-12);
    }

    /// For a non-FPHA hydro at stage 0 with both a JSON-resolved productivity
    /// (0.9) and a parquet override row supplying 0.85, the override wins.
    ///
    /// For a non-FPHA hydro, `build_energy_conversion_set` reads `ρ_eq` from
    /// `ProductionModelSet` as the single source of truth. Whether the value
    /// originated from JSON or from the parquet override is resolved upstream
    /// in `prepare_hydro_models`. This test confirms the value-flow contract:
    /// what `pm` says is what the conversion stores.
    #[test]
    fn test_non_fpha_reads_productivity_from_production_model_set() {
        let n_stages = 3;
        let hydros = vec![make_hydro_with(
            1,
            HydroGenerationModel::ConstantProductivity,
            100.0,
            200.0,
            50.0,
            None,
        )];
        let cascade = CascadeTopology::build(&hydros);
        let resolver = constant_resolver(&hydros, 0.65, n_stages);
        let pm = production_set(&[0.85], n_stages);

        let set = build_energy_conversion_set(
            &hydros,
            &stage_ids_0_based(n_stages),
            &cascade,
            &resolver,
            &HashMap::new(),
            None,
            Some(&pm),
        )
        .expect("builder succeeds");

        for s in 0..n_stages {
            let cell = set.conversion(0, s);
            assert!(
                (cell.equivalent_productivity_mw_per_m3s - 0.85).abs() < 1e-12,
                "stage {s}: expected 0.85 from pm, got {}",
                cell.equivalent_productivity_mw_per_m3s
            );
        }
    }

    /// The override table is consulted ONLY for FPHA hydros (where it
    /// replaces the VHA + `ρ_esp` derivation). For non-FPHA hydros the
    /// override has no effect at this layer because `ProductionModelSet`
    /// is already authoritative; the `prepare_hydro_models` pipeline is
    /// responsible for folding the override into pm.
    #[test]
    fn test_non_fpha_override_table_not_consulted_at_build_site() {
        let n_stages = 1;
        let hydros = vec![make_hydro_with(
            1,
            HydroGenerationModel::ConstantProductivity,
            100.0,
            200.0,
            50.0,
            None,
        )];
        let cascade = CascadeTopology::build(&hydros);
        let resolver = constant_resolver(&hydros, 0.65, n_stages);
        // pm supplies 0.9 — that's the resolved value the LP and conversion
        // both see.
        let pm = production_set(&[0.9], n_stages);
        // Override table contains an inconsistent value that must NOT win at
        // this layer (such an inconsistency would be caught upstream by
        // `cobre_io::validation::productivity_resolution`).
        let override_table =
            build_hydro_energy_productivity_override(&[HydroEnergyProductivityRow {
                hydro_id: hydros[0].id,
                stage_id: Some(0),
                equivalent_productivity_mw_per_m3s: Some(0.42),
                reference_outflow_m3s: None,
                specific_productivity_mw_per_m3s_per_m: None,
            }])
            .expect("override builds");

        let set = build_energy_conversion_set(
            &hydros,
            &stage_ids_0_based(n_stages),
            &cascade,
            &resolver,
            &HashMap::new(),
            Some(&override_table),
            Some(&pm),
        )
        .expect("builder succeeds");

        let cell = set.conversion(0, 0);
        assert!(
            (cell.equivalent_productivity_mw_per_m3s - 0.9).abs() < 1e-12,
            "non-FPHA path must read from pm, not from override; got {}",
            cell.equivalent_productivity_mw_per_m3s
        );
    }

    /// For a non-FPHA hydro with no parquet override and a JSON-resolved
    /// productivity of 0.9, every stage receives 0.9. Regression for the
    /// JSON-only path.
    #[test]
    fn test_non_fpha_json_only_path_unchanged() {
        let n_stages = 2;
        let hydros = vec![make_hydro_with(
            1,
            HydroGenerationModel::ConstantProductivity,
            100.0,
            200.0,
            50.0,
            None,
        )];
        let cascade = CascadeTopology::build(&hydros);
        let resolver = constant_resolver(&hydros, 0.65, n_stages);
        let pm = production_set(&[0.9], n_stages);

        let set = build_energy_conversion_set(
            &hydros,
            &stage_ids_0_based(n_stages),
            &cascade,
            &resolver,
            &HashMap::new(),
            None,
            Some(&pm),
        )
        .expect("builder succeeds");

        for s in 0..n_stages {
            let cell = set.conversion(0, s);
            assert!(
                (cell.equivalent_productivity_mw_per_m3s - 0.9).abs() < 1e-12,
                "stage {s}: expected 0.9 from JSON-only path, got {}",
                cell.equivalent_productivity_mw_per_m3s
            );
        }
    }

    // ── mean-evaluator own-term tests ─────────────────────────────────────────

    /// Build a VHA geometry map entry from `(volume_hm3, height_m)` breakpoints.
    fn vha_rows(hydro_id: EntityId, points: &[(f64, f64)]) -> (EntityId, Vec<HydroGeometryRow>) {
        (
            hydro_id,
            points
                .iter()
                .map(|&(volume_hm3, height_m)| HydroGeometryRow {
                    hydro_id,
                    volume_hm3,
                    height_m,
                    area_km2: 1.0,
                })
                .collect(),
        )
    }

    /// A multi-breakpoint FPHA plant with Factor losses: the mean-evaluator own term
    /// integrates the forebay height over the physical range and equals a hand oracle,
    /// while the reference-point value (at V_ref) is unchanged and distinct.
    #[test]
    fn mean_evaluator_own_term_matches_hand_oracle() {
        let mut hydro = make_hydro_with(
            1,
            HydroGenerationModel::Fpha,
            100.0,
            700.0,
            40.0,
            Some(0.02),
        );
        hydro.hydraulic_losses = Some(HydraulicLossesModel::Factor { value: 0.1 });
        let hydros = vec![hydro];
        let cascade = CascadeTopology::build(&hydros);
        // V_ref=650 decoupled from the physical range [100, 700].
        let resolver =
            build_hydro_reference_volumes_resolved(&[(hydros[0].id, StudyPos(0), 650.0)], 0.0);
        let (id, rows) = vha_rows(
            hydros[0].id,
            &[(0.0, 500.0), (300.0, 560.0), (1000.0, 700.0)],
        );
        let mut map = HashMap::new();
        map.insert(id, rows);

        let set = build_energy_conversion_set(
            &hydros,
            &stage_ids_0_based(1),
            &cascade,
            &resolver,
            &map,
            None,
            None,
        )
        .expect("builder succeeds");

        // mean_height([100,700]) = 580; h_loss = 0.1·580 = 58; ρ_esp·(580−58) = 0.02·522.
        let integrated = set.integrated_equivalent_productivity(0, 0);
        let expected_mean = 0.02 * 522.0;
        assert!(
            (integrated - expected_mean).abs() <= 1e-9 * expected_mean.abs().max(1.0),
            "integrated own {integrated}, expected {expected_mean}"
        );
        // height(650)=630; h_eq=630−63=567; reference-point ρ_eq = 0.02·567.
        let reference = set.conversion(0, 0).equivalent_productivity_mw_per_m3s;
        let expected_ref = 0.02 * 567.0;
        assert!(
            (reference - expected_ref).abs() <= 1e-9 * expected_ref.abs().max(1.0),
            "reference-point {reference}, expected {expected_ref}"
        );
        assert!(
            (reference - integrated).abs() > 1e-6,
            "the mean term must differ from the reference-point value here"
        );
    }

    #[test]
    fn mean_evaluator_tracks_physical_range_not_v_ref() {
        let hydro = make_hydro_with(
            1,
            HydroGenerationModel::Fpha,
            100.0,
            300.0,
            50.0,
            Some(0.01),
        );
        let hydros = vec![hydro];
        let cascade = CascadeTopology::build(&hydros);
        // V_ref sits at 900, far above the physical range [100, 300].
        let resolver =
            build_hydro_reference_volumes_resolved(&[(hydros[0].id, StudyPos(0), 900.0)], 0.0);
        let (id, rows) = vha_rows(hydros[0].id, &[(0.0, 100.0), (1000.0, 200.0)]);
        let mut map = HashMap::new();
        map.insert(id, rows);

        let set = build_energy_conversion_set(
            &hydros,
            &stage_ids_0_based(1),
            &cascade,
            &resolver,
            &map,
            None,
            None,
        )
        .expect("builder succeeds");

        // Linear table: mean over [100,300] is height(200) = 120.
        let integrated = set.integrated_equivalent_productivity(0, 0);
        assert!(
            (integrated - 0.01 * 120.0).abs() <= 1e-9,
            "integrated own tracks [100,300] → 0.01·120, got {integrated}"
        );
        // Reference-point anchors on V_ref=900 → height(900) = 190.
        let reference = set.conversion(0, 0).equivalent_productivity_mw_per_m3s;
        assert!(
            (reference - 0.01 * 190.0).abs() <= 1e-9,
            "reference-point tracks V_ref=900 → 0.01·190, got {reference}"
        );
    }

    /// A collapsed physical range (`V_lo == V_hi`) copies the reference-point own value
    /// bit-for-bit rather than re-deriving it.
    #[test]
    fn collapsed_range_copies_reference_point_bit_for_bit() {
        let v_ref = 650.0;
        let hydro = make_hydro_with(
            1,
            HydroGenerationModel::Fpha,
            v_ref,
            v_ref,
            50.0,
            Some(0.01),
        );
        let hydros = vec![hydro];
        let cascade = CascadeTopology::build(&hydros);
        let resolver =
            build_hydro_reference_volumes_resolved(&[(hydros[0].id, StudyPos(0), v_ref)], 0.0);
        let (id, rows) = vha_rows(hydros[0].id, &[(0.0, 100.0), (1000.0, 200.0)]);
        let mut map = HashMap::new();
        map.insert(id, rows);

        let set = build_energy_conversion_set(
            &hydros,
            &stage_ids_0_based(1),
            &cascade,
            &resolver,
            &map,
            None,
            None,
        )
        .expect("builder succeeds");

        let reference = set.conversion(0, 0).equivalent_productivity_mw_per_m3s;
        assert_eq!(
            set.integrated_equivalent_productivity(0, 0).to_bits(),
            reference.to_bits(),
            "collapsed range must copy the reference-point own value bit-for-bit"
        );
    }

    /// A single-row VHA table is degenerate in a DIFFERENT way than a collapsed
    /// physical range: the range is genuine (`V_lo != V_hi`), but the table
    /// returns one constant height for any query, so the integrated own term
    /// still copies the reference-point own term bit-for-bit.
    #[test]
    fn single_row_vha_table_copies_reference_point_own_term_bit_for_bit() {
        let hydro = make_hydro_with(
            1,
            HydroGenerationModel::Fpha,
            100.0,
            700.0,
            50.0,
            Some(0.01),
        );
        let hydros = vec![hydro];
        let cascade = CascadeTopology::build(&hydros);
        let resolver =
            build_hydro_reference_volumes_resolved(&[(hydros[0].id, StudyPos(0), 650.0)], 0.0);
        let (id, rows) = vha_rows(hydros[0].id, &[(400.0, 386.5)]);
        let mut map = HashMap::new();
        map.insert(id, rows);

        let set = build_energy_conversion_set(
            &hydros,
            &stage_ids_0_based(1),
            &cascade,
            &resolver,
            &map,
            None,
            None,
        )
        .expect("builder succeeds");

        let reference = set.conversion(0, 0).equivalent_productivity_mw_per_m3s;
        assert_eq!(
            set.integrated_equivalent_productivity(0, 0).to_bits(),
            reference.to_bits(),
            "a single-row VHA table's constant height must copy the reference-point own term"
        );
    }

    /// A parquet `ρ_eq` override supplies the own term for BOTH evaluators and
    /// suppresses the mean computation, even on a genuine (non-collapsed) range.
    #[test]
    fn override_wins_both_evaluators() {
        // A genuine range whose mean term (0.01·height(250)=1.25) must be suppressed.
        let hydro = make_hydro_with(
            1,
            HydroGenerationModel::Fpha,
            200.0,
            300.0,
            50.0,
            Some(0.01),
        );
        let hydros = vec![hydro];
        let cascade = CascadeTopology::build(&hydros);
        let resolver = constant_resolver(&hydros, 0.65, 1);
        let (id, rows) = vha_rows(hydros[0].id, &[(0.0, 100.0), (1000.0, 200.0)]);
        let mut map = HashMap::new();
        map.insert(id, rows);
        let override_table =
            build_hydro_energy_productivity_override(&[HydroEnergyProductivityRow {
                hydro_id: hydros[0].id,
                stage_id: None,
                equivalent_productivity_mw_per_m3s: Some(2.5),
                reference_outflow_m3s: None,
                specific_productivity_mw_per_m3s_per_m: None,
            }])
            .expect("override builds");

        let set = build_energy_conversion_set(
            &hydros,
            &stage_ids_0_based(1),
            &cascade,
            &resolver,
            &map,
            Some(&override_table),
            None,
        )
        .expect("builder succeeds");

        let reference = set.conversion(0, 0).equivalent_productivity_mw_per_m3s;
        assert_eq!(
            reference.to_bits(),
            2.5_f64.to_bits(),
            "override wins the reference-point evaluator"
        );
        assert_eq!(
            set.integrated_equivalent_productivity(0, 0).to_bits(),
            2.5_f64.to_bits(),
            "override wins the mean evaluator (its computation is suppressed)"
        );
    }

    /// A non-positive mean equivalent head is rejected for any generation model — here
    /// a constant-productivity plant with geometry, a genuine range, and a tailrace
    /// above the mean forebay height — naming the offending hydro.
    #[test]
    fn non_positive_mean_head_is_rejected_for_any_generation_model() {
        let mut hydro = make_hydro_with(
            3,
            HydroGenerationModel::ConstantProductivity,
            100.0,
            300.0,
            50.0,
            Some(0.01),
        );
        // Constant tailrace at 200 m, above the mean forebay height (120 m over [100,300]).
        hydro.tailrace = Some(TailraceModel::Polynomial {
            coefficients: vec![200.0],
        });
        let hydros = vec![hydro];
        let cascade = CascadeTopology::build(&hydros);
        let resolver = constant_resolver(&hydros, 0.65, 1);
        let (id, rows) = vha_rows(hydros[0].id, &[(0.0, 100.0), (1000.0, 200.0)]);
        let mut map = HashMap::new();
        map.insert(id, rows);

        let err = build_energy_conversion_set(
            &hydros,
            &stage_ids_0_based(1),
            &cascade,
            &resolver,
            &map,
            None,
            None,
        )
        .unwrap_err();
        match err {
            EnergyConversionError::NonPositiveEquivalentHead { hydro_id, h_eq } => {
                assert_eq!(hydro_id, hydros[0].id);
                assert!(h_eq <= 0.0, "expected non-positive mean head, got {h_eq}");
            }
            other => panic!("expected NonPositiveEquivalentHead, got: {other:?}"),
        }
    }

    // ── ρ_esp parquet override resolution tests ────────────────────────────

    /// A single-row override setting only `specific_productivity_mw_per_m3s_per_m`
    /// for `hydro_id` at `stage_id` (`None` = per-hydro default); every other
    /// column stays `None`.
    fn rho_esp_override_row(
        hydro_id: EntityId,
        stage_id: Option<i32>,
        rho_esp: Option<f64>,
    ) -> HydroEnergyProductivityRow {
        HydroEnergyProductivityRow {
            hydro_id,
            stage_id,
            equivalent_productivity_mw_per_m3s: None,
            reference_outflow_m3s: None,
            specific_productivity_mw_per_m3s_per_m: rho_esp,
        }
    }

    /// A per-stage ρ_esp override shifts BOTH the FPHA reference-point ρ_eq and
    /// the mean own term at that stage only; the other stage, with no override,
    /// keeps the entity ρ_esp. Every operand is exactly representable in binary
    /// so the comparison is `to_bits()`, not a tolerance.
    #[test]
    fn per_stage_rho_esp_override_shifts_mean_own_and_point_fpha_at_that_stage_only() {
        let mut hydro = make_hydro_with(
            1,
            HydroGenerationModel::Fpha,
            256.0,
            768.0,
            50.0,
            Some(0.125),
        );
        hydro.tailrace = None;
        hydro.hydraulic_losses = None;
        let hydros = vec![hydro];
        let cascade = CascadeTopology::build(&hydros);
        // V_ref = 896 at every stage, decoupled from the physical range [256, 768].
        let resolver = build_hydro_reference_volumes_resolved(
            &[
                (hydros[0].id, StudyPos(0), 896.0),
                (hydros[0].id, StudyPos(1), 896.0),
            ],
            0.0,
        );
        let (id, rows) = vha_rows(hydros[0].id, &[(0.0, 100.0), (1024.0, 356.0)]);
        let mut map = HashMap::new();
        map.insert(id, rows);
        let override_esp = 0.375;
        let override_table = build_hydro_energy_productivity_override(&[rho_esp_override_row(
            hydros[0].id,
            Some(1),
            Some(override_esp),
        )])
        .expect("override builds");

        let set = build_energy_conversion_set(
            &hydros,
            &stage_ids_0_based(2),
            &cascade,
            &resolver,
            &map,
            Some(&override_table),
            None,
        )
        .expect("builder succeeds");

        // height(896) = 324, mean_height(256,768) = 228 on this table.
        // Stage 0: no override -> resolved ρ_esp = entity 0.125.
        assert_eq!(
            set.conversion(0, 0)
                .equivalent_productivity_mw_per_m3s
                .to_bits(),
            40.5_f64.to_bits(),
            "stage 0 point ρ_eq must use the entity ρ_esp (0.125 * 324 = 40.5)"
        );
        assert_eq!(
            set.integrated_equivalent_productivity(0, 0).to_bits(),
            28.5_f64.to_bits(),
            "stage 0 mean own must use the entity ρ_esp (0.125 * 228 = 28.5)"
        );

        // Stage 1: override -> resolved ρ_esp = 0.375.
        assert_eq!(
            set.conversion(0, 1)
                .equivalent_productivity_mw_per_m3s
                .to_bits(),
            121.5_f64.to_bits(),
            "stage 1 point ρ_eq must use the override ρ_esp (0.375 * 324 = 121.5)"
        );
        assert_eq!(
            set.integrated_equivalent_productivity(0, 1).to_bits(),
            85.5_f64.to_bits(),
            "stage 1 mean own must use the override ρ_esp (0.375 * 228 = 85.5)"
        );
    }

    /// A per-hydro default override (`stage_id = NULL`) applies at every stage.
    #[test]
    fn per_hydro_default_rho_esp_override_applies_at_every_stage() {
        let hydro = make_hydro_with(
            1,
            HydroGenerationModel::ConstantProductivity,
            256.0,
            768.0,
            50.0,
            None,
        );
        let hydros = vec![hydro];
        let cascade = CascadeTopology::build(&hydros);
        let resolver = constant_resolver(&hydros, 0.5, 3);
        let (id, rows) = vha_rows(hydros[0].id, &[(0.0, 100.0), (1024.0, 356.0)]);
        let mut map = HashMap::new();
        map.insert(id, rows);
        let override_esp = 0.375;
        let override_table = build_hydro_energy_productivity_override(&[rho_esp_override_row(
            hydros[0].id,
            None,
            Some(override_esp),
        )])
        .expect("override builds");

        let set = build_energy_conversion_set(
            &hydros,
            &stage_ids_0_based(3),
            &cascade,
            &resolver,
            &map,
            Some(&override_table),
            None,
        )
        .expect("builder succeeds");

        // mean_height(256, 768) = 228 on this table.
        let expected = override_esp * 228.0;
        for s in 0..3 {
            assert_eq!(
                set.integrated_equivalent_productivity(0, s).to_bits(),
                expected.to_bits(),
                "stage {s}: per-hydro default override must apply, got {}",
                set.integrated_equivalent_productivity(0, s)
            );
        }
    }

    /// An FPHA hydro with entity ρ_esp `None` and an override ρ_esp `Some`
    /// derives ρ_eq via the resolved override instead of failing
    /// `FphaMissingEquivalentProductivity`.
    #[test]
    fn fpha_with_override_only_rho_esp_derives_rho_eq() {
        let mut hydro = make_hydro_with(1, HydroGenerationModel::Fpha, 100.0, 200.0, 50.0, None);
        hydro.tailrace = None;
        hydro.hydraulic_losses = None;
        let hydros = vec![hydro];
        let cascade = CascadeTopology::build(&hydros);
        let resolver = constant_resolver(&hydros, 0.5, 1);
        let (id, rows) = vha_constant_height(hydros[0].id, 400.0);
        let mut map = HashMap::new();
        map.insert(id, rows);
        let override_table = build_hydro_energy_productivity_override(&[rho_esp_override_row(
            hydros[0].id,
            None,
            Some(0.02),
        )])
        .expect("override builds");

        let set = build_energy_conversion_set(
            &hydros,
            &stage_ids_0_based(1),
            &cascade,
            &resolver,
            &map,
            Some(&override_table),
            None,
        )
        .expect("override-only ρ_esp must derive ρ_eq, not FphaMissingEquivalentProductivity");

        assert_eq!(
            set.conversion(0, 0)
                .equivalent_productivity_mw_per_m3s
                .to_bits(),
            (0.02_f64 * 400.0).to_bits(),
            "override ρ_esp * flat height(400) must derive ρ_eq"
        );
    }

    /// A ρ_eq override still wins outright over a CO-PRESENT ρ_esp override —
    /// not merely over the entity ρ_esp (already covered by
    /// `override_wins_both_evaluators`).
    #[test]
    fn rho_eq_override_wins_over_rho_esp_override() {
        let hydro = make_hydro_with(1, HydroGenerationModel::Fpha, 200.0, 300.0, 50.0, None);
        let hydros = vec![hydro];
        let cascade = CascadeTopology::build(&hydros);
        let resolver = constant_resolver(&hydros, 0.65, 1);
        let (id, rows) = vha_rows(hydros[0].id, &[(0.0, 100.0), (1000.0, 200.0)]);
        let mut map = HashMap::new();
        map.insert(id, rows);
        let override_table =
            build_hydro_energy_productivity_override(&[HydroEnergyProductivityRow {
                hydro_id: hydros[0].id,
                stage_id: None,
                equivalent_productivity_mw_per_m3s: Some(2.5),
                reference_outflow_m3s: None,
                specific_productivity_mw_per_m3s_per_m: Some(0.01),
            }])
            .expect("override builds");

        let set = build_energy_conversion_set(
            &hydros,
            &stage_ids_0_based(1),
            &cascade,
            &resolver,
            &map,
            Some(&override_table),
            None,
        )
        .expect("builder succeeds");

        assert_eq!(
            set.conversion(0, 0)
                .equivalent_productivity_mw_per_m3s
                .to_bits(),
            2.5_f64.to_bits(),
            "ρ_eq override wins the reference-point evaluator over a co-present ρ_esp override"
        );
        assert_eq!(
            set.integrated_equivalent_productivity(0, 0).to_bits(),
            2.5_f64.to_bits(),
            "ρ_eq override wins the mean evaluator over a co-present ρ_esp override"
        );
    }

    /// Permuting the hydro declaration order leaves every ρ_esp-override-derived
    /// grid cell bit-identical when re-keyed by hydro id — the resolver and the
    /// geometry gate key by `hydro.id`, never by declaration position.
    #[test]
    fn rho_esp_override_grids_are_declaration_order_invariant() {
        let mut fpha_plant =
            make_hydro_with(1, HydroGenerationModel::Fpha, 100.0, 300.0, 50.0, None);
        fpha_plant.tailrace = None;
        fpha_plant.hydraulic_losses = None;
        let constant_plant = make_hydro_with(
            2,
            HydroGenerationModel::ConstantProductivity,
            50.0,
            150.0,
            30.0,
            None,
        );

        let (id_a, rows_a) = vha_constant_height(fpha_plant.id, 400.0);
        let (id_b, rows_b) = vha_constant_height(constant_plant.id, 200.0);
        let mut map = HashMap::new();
        map.insert(id_a, rows_a);
        map.insert(id_b, rows_b);

        let override_table = build_hydro_energy_productivity_override(&[
            rho_esp_override_row(fpha_plant.id, None, Some(0.02)),
            rho_esp_override_row(constant_plant.id, None, Some(0.05)),
        ])
        .expect("override builds");

        for hydros in [
            vec![fpha_plant.clone(), constant_plant.clone()],
            vec![constant_plant.clone(), fpha_plant.clone()],
        ] {
            let cascade = CascadeTopology::build(&hydros);
            let resolver = constant_resolver(&hydros, 0.5, 1);
            let set = build_energy_conversion_set(
                &hydros,
                &stage_ids_0_based(1),
                &cascade,
                &resolver,
                &map,
                Some(&override_table),
                None,
            )
            .expect("builder succeeds");

            let fpha_idx = hydros.iter().position(|h| h.id == fpha_plant.id).unwrap();
            let constant_idx = hydros
                .iter()
                .position(|h| h.id == constant_plant.id)
                .unwrap();

            assert_eq!(
                set.conversion(fpha_idx, 0)
                    .equivalent_productivity_mw_per_m3s
                    .to_bits(),
                (0.02_f64 * 400.0).to_bits(),
                "hydro A's overridden ρ_eq must be order-invariant"
            );
            assert_eq!(
                set.integrated_equivalent_productivity(fpha_idx, 0)
                    .to_bits(),
                (0.02_f64 * 400.0).to_bits(),
                "hydro A's overridden mean own term must be order-invariant"
            );
            assert_eq!(
                set.integrated_equivalent_productivity(constant_idx, 0)
                    .to_bits(),
                (0.05_f64 * 200.0).to_bits(),
                "hydro B's overridden mean own term must be order-invariant"
            );
        }
    }

    /// An override table with every column NULL for every hydro yields grids
    /// bit-identical to passing `None` — an all-absent override degenerates to
    /// the entity-only path exactly.
    #[test]
    fn all_null_rho_esp_override_is_bit_identical_to_no_override() {
        let mut hydro = make_hydro_with(
            1,
            HydroGenerationModel::Fpha,
            100.0,
            300.0,
            50.0,
            Some(0.02),
        );
        hydro.tailrace = None;
        hydro.hydraulic_losses = None;
        let hydros = vec![hydro];
        let cascade = CascadeTopology::build(&hydros);
        let resolver = constant_resolver(&hydros, 0.5, 2);
        let (id, rows) = vha_rows(hydros[0].id, &[(0.0, 100.0), (1000.0, 200.0)]);
        let mut map = HashMap::new();
        map.insert(id, rows);

        let all_null_override = build_hydro_energy_productivity_override(&[rho_esp_override_row(
            hydros[0].id,
            None,
            None,
        )])
        .expect("override builds");

        let without_table = build_energy_conversion_set(
            &hydros,
            &stage_ids_0_based(2),
            &cascade,
            &resolver,
            &map,
            None,
            None,
        )
        .expect("builder succeeds without a table");
        let with_all_null_table = build_energy_conversion_set(
            &hydros,
            &stage_ids_0_based(2),
            &cascade,
            &resolver,
            &map,
            Some(&all_null_override),
            None,
        )
        .expect("builder succeeds with an all-NULL table");

        for s in 0..2 {
            assert_eq!(
                without_table
                    .conversion(0, s)
                    .equivalent_productivity_mw_per_m3s
                    .to_bits(),
                with_all_null_table
                    .conversion(0, s)
                    .equivalent_productivity_mw_per_m3s
                    .to_bits(),
                "stage {s}: point ρ_eq must be bit-identical"
            );
            assert_eq!(
                without_table
                    .integrated_equivalent_productivity(0, s)
                    .to_bits(),
                with_all_null_table
                    .integrated_equivalent_productivity(0, s)
                    .to_bits(),
                "stage {s}: mean own term must be bit-identical"
            );
        }
    }
}
