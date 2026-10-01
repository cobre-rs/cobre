//! Evaporation model resolution: per-hydro linearized evaporation from geometry.
//!
//! Resolves the per-`(hydro, stage)` linearized evaporation coefficients from
//! reservoir Volume-Height-Area geometry, plus the supporting area
//! interpolation and finite-difference derivative helpers. The resolved
//! coefficients feed the water-balance row in the LP builder.

use std::collections::HashMap;

use chrono::{Datelike, NaiveDate};
use cobre_core::temporal::Stage;
use cobre_core::{EntityId, Hydro, System, month_of};
use cobre_io::CaseArtifacts;
use cobre_io::extensions::HydroGeometryRow;

use super::types::{
    EvaporationModel, EvaporationModelSet, EvaporationReferenceSource, EvaporationSource,
    LinearizedEvaporation,
};
use crate::SddpError;
use crate::block_clock::BlockClock;
// ── Evaporation model resolution ──────────────────────────────────────────────

/// Resolve per-hydro linearized evaporation models from a pre-parsed
/// [`cobre_io::CaseArtifacts`] bundle.
///
/// Plants without `evaporation_coefficients_mm` get `EvaporationModel::None`; if
/// no plant has them, no geometry is consulted. Otherwise the model is a
/// first-order Taylor linearization around the reference volume:
///
/// ```text
/// evaporation_outflow = intercept_m3s + volume_slope_m3s_per_hm3 * v
/// ```
///
/// where:
///
/// ```text
/// mm_km2_to_m3s            = 1.0 / (3.6 * month_hours)   -- mm·km² → m³/s
/// volume_slope_m3s_per_hm3 = mm_km2_to_m3s * monthly_evaporation_mm[month] * dA/dv|_{reference_volume}
/// intercept_m3s            = mm_km2_to_m3s * monthly_evaporation_mm[month] * A(reference_volume)
///                            - volume_slope_m3s_per_hm3 * reference_volume
/// ```
///
/// `reference_volume = (v_min + v_max) / 2` is the linearization reference volume.
/// `month_hours` is the CALENDAR month's hours (leap-aware). It is the divisor —
/// not the stage's own hours — because the water-balance coupling multiplies this
/// flow by the stage-duration factor `zeta` (∝ `stage_seconds`); dividing by the
/// stage would cancel and deposit a whole month of evaporation on any stage,
/// whereas dividing by the month makes it a monthly-average rate, so a stage
/// deposits only its `stage_hours / month_hours` share.
/// `month` is the 0-based calendar month [`month_of`]
/// derives from `stage.start_date` — not `stage.season_id`, whose meaning is
/// cycle-dependent (`Monthly`, `Weekly`, `Custom`) and only equals the calendar
/// month under the `Monthly` convention.
///
/// # Errors
///
/// | Condition                                                        | Error variant             |
/// | ---------------------------------------------------------------- | ------------------------- |
/// | Computed slope or intercept is NaN or infinite                   | [`SddpError::Validation`] |
///
/// A hydro with evaporation coefficients but no usable surface-area data — no
/// geometry rows, or every `area_km2` zero — does NOT error: evaporation is
/// disabled for that hydro ([`EvaporationSource::DisabledNoArea`]) with a
/// `tracing::warn!`, because zero surface area yields zero evaporation.
// Rationale: a type alias for this three-output tuple would hide the concrete
// types callers destructure at every call site.
#[allow(clippy::type_complexity)]
pub fn resolve_evaporation_models_from_artifacts(
    system: &System,
    artifacts: &CaseArtifacts,
) -> Result<
    (
        EvaporationModelSet,
        Vec<(EntityId, EvaporationSource)>,
        Vec<(EntityId, EvaporationReferenceSource)>,
    ),
    SddpError,
> {
    let any_evaporation = system
        .hydros()
        .iter()
        .any(|h| h.evaporation_coefficients_mm.is_some());

    if !any_evaporation {
        let models = system
            .hydros()
            .iter()
            .map(|_| EvaporationModel::None)
            .collect();
        let provenance = system
            .hydros()
            .iter()
            .map(|h| (h.id, EvaporationSource::NotModeled))
            .collect();
        let reference_sources = system
            .hydros()
            .iter()
            .map(|h| (h.id, EvaporationReferenceSource::DefaultMidpoint))
            .collect();
        return Ok((
            EvaporationModelSet::new(models),
            provenance,
            reference_sources,
        ));
    }

    let mut geometry_map: HashMap<EntityId, Vec<&HydroGeometryRow>> = HashMap::new();
    for row in &artifacts.hydro_geometry {
        geometry_map.entry(row.hydro_id).or_default().push(row);
    }
    // Interpolation below assumes ascending volume order.
    for rows in geometry_map.values_mut() {
        rows.sort_by(|a, b| a.volume_hm3.total_cmp(&b.volume_hm3));
    }

    let study_stages: Vec<&Stage> = system.stages().iter().filter(|s| s.id >= 0).collect();

    resolve_evaporation_core(system.hydros(), &geometry_map, &study_stages)
}

/// Hours in `date`'s calendar month, leap-aware. The evaporation-rate divisor
/// (see [`resolve_evaporation_models_from_artifacts`]): a stage deposits its
/// `stage_hours / month_hours` share of the month's evaporation.
fn hours_in_calendar_month(date: NaiveDate) -> f64 {
    let days = match date.month() {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31.0,
        2 if is_leap_year(date.year()) => 29.0,
        2 => 28.0,
        // April, June, September, November have 30 days; the wildcard also
        // absorbs the unreachable out-of-range month (`month()` is 1..=12).
        _ => 30.0,
    };
    days * 24.0
}

fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

/// Core evaporation linearization over pre-loaded data, split from
/// [`resolve_evaporation_models_from_artifacts`] so unit tests can run without disk I/O.
///
/// # Errors
///
/// Same error conditions as [`resolve_evaporation_models_from_artifacts`].
// Rationale: a type alias would hide the three concrete output types; splitting
// the per-stage loop would thread several computed intermediates across helper
// boundaries.
#[allow(clippy::type_complexity, clippy::too_many_lines)]
fn resolve_evaporation_core(
    hydros: &[Hydro],
    geometry_map: &HashMap<EntityId, Vec<&HydroGeometryRow>>,
    study_stages: &[&Stage],
) -> Result<
    (
        EvaporationModelSet,
        Vec<(EntityId, EvaporationSource)>,
        Vec<(EntityId, EvaporationReferenceSource)>,
    ),
    SddpError,
> {
    let n_stages = study_stages.len();
    let mut all_models: Vec<EvaporationModel> = Vec::with_capacity(hydros.len());
    let mut provenance: Vec<(EntityId, EvaporationSource)> = Vec::with_capacity(hydros.len());
    let mut reference_provenance: Vec<(EntityId, EvaporationReferenceSource)> =
        Vec::with_capacity(hydros.len());

    for hydro in hydros {
        let Some(coefficients_mm) = hydro.evaporation_coefficients_mm else {
            all_models.push(EvaporationModel::None);
            provenance.push((hydro.id, EvaporationSource::NotModeled));
            reference_provenance.push((hydro.id, EvaporationReferenceSource::DefaultMidpoint));
            continue;
        };

        let geo_rows: &[&HydroGeometryRow] = geometry_map.get(&hydro.id).map_or(&[], Vec::as_slice);

        // Evaporation needs a usable area-volume curve. A new or being-filled
        // reservoir legitimately may have none (no geometry rows, or a single
        // dead-volume point with zero area). Zero surface area yields zero
        // evaporation, so disable it for this hydro and warn rather than failing
        // the whole run — DisabledNoArea is the provenance the summary surfaces.
        if geo_rows.is_empty() {
            tracing::warn!(
                "hydro {} (id={}) has evaporation_coefficients_mm but no geometry data \
                 in hydro_geometry.parquet; disabling evaporation for this hydro",
                hydro.name,
                hydro.id.0
            );
            all_models.push(EvaporationModel::None);
            provenance.push((hydro.id, EvaporationSource::DisabledNoArea));
            reference_provenance.push((hydro.id, EvaporationReferenceSource::DefaultMidpoint));
            continue;
        }

        if geo_rows.iter().all(|r| r.area_km2 == 0.0) {
            tracing::warn!(
                "hydro {} (id={}) has evaporation_coefficients_mm but every area_km2 in \
                 hydro_geometry.parquet is zero; disabling evaporation for this hydro \
                 (zero surface area produces zero evaporation)",
                hydro.name,
                hydro.id.0
            );
            all_models.push(EvaporationModel::None);
            provenance.push((hydro.id, EvaporationSource::DisabledNoArea));
            reference_provenance.push((hydro.id, EvaporationReferenceSource::DefaultMidpoint));
            continue;
        }

        let ref_source = if hydro.evaporation_reference_volumes_hm3.is_some() {
            EvaporationReferenceSource::UserSupplied
        } else {
            EvaporationReferenceSource::DefaultMidpoint
        };

        // Midpoint-path values, read only when there are no per-season volumes.
        let midpoint_v = f64::midpoint(hydro.min_storage_hm3, hydro.max_storage_hm3);
        let (midpoint_area, midpoint_slope) = if hydro.evaporation_reference_volumes_hm3.is_none() {
            (
                interpolate_area(geo_rows, midpoint_v),
                area_derivative(geo_rows, midpoint_v),
            )
        } else {
            (0.0, 0.0)
        };

        let mut stage_coefficients: Vec<LinearizedEvaporation> = Vec::with_capacity(n_stages);
        let mut stage_ref_volumes: Vec<f64> = Vec::with_capacity(n_stages);

        for stage in study_stages {
            let month_index = month_of(stage).index();

            let monthly_evaporation_mm = coefficients_mm[month_index];

            let (reference_volume, a_ref, da_dv) =
                if let Some(ref_vols) = hydro.evaporation_reference_volumes_hm3 {
                    let v = ref_vols[month_index];
                    (
                        v,
                        interpolate_area(geo_rows, v),
                        area_derivative(geo_rows, v),
                    )
                } else {
                    (midpoint_v, midpoint_area, midpoint_slope)
                };

            let stage_hours = BlockClock::new(stage).total_hours();

            // A zero-duration stage no longer surfaces as a non-finite coefficient
            // below (the divisor is now the calendar month, never zero), so reject
            // it explicitly here.
            if stage_hours <= 0.0 {
                return Err(SddpError::Validation(format!(
                    "hydro {} (id={}) stage {}: total block duration is {stage_hours} h; \
                     evaporation needs a positive stage duration.",
                    hydro.name, hydro.id.0, stage.id
                )));
            }

            // Divide by the CALENDAR month's hours, not the stage's: the
            // water-balance `zeta` (∝ stage_seconds) cancels a stage-hours divisor
            // and would deposit a whole month of evaporation per stage.
            let month_hours = hours_in_calendar_month(stage.start_date);
            let mm_km2_to_m3s = 1.0 / (3.6 * month_hours);

            let volume_slope_m3s_per_hm3 = mm_km2_to_m3s * monthly_evaporation_mm * da_dv;
            let intercept_m3s = mm_km2_to_m3s * monthly_evaporation_mm * a_ref
                - volume_slope_m3s_per_hm3 * reference_volume;

            if !volume_slope_m3s_per_hm3.is_finite() {
                return Err(SddpError::Validation(format!(
                    "hydro {} (id={}) stage {}: computed volume_slope_m3s_per_hm3 = \
                     {volume_slope_m3s_per_hm3} is not finite. Check geometry data for \
                     degenerate area-volume curve points.",
                    hydro.name, hydro.id.0, stage.id
                )));
            }
            if !intercept_m3s.is_finite() {
                return Err(SddpError::Validation(format!(
                    "hydro {} (id={}) stage {}: computed intercept_m3s = {intercept_m3s} is not \
                     finite. Check geometry data for degenerate area-volume curve points.",
                    hydro.name, hydro.id.0, stage.id
                )));
            }

            stage_coefficients.push(LinearizedEvaporation {
                intercept_m3s,
                volume_slope_m3s_per_hm3,
            });
            stage_ref_volumes.push(reference_volume);
        }

        all_models.push(EvaporationModel::Linearized {
            coefficients: stage_coefficients,
            reference_volumes_hm3: stage_ref_volumes,
        });
        provenance.push((hydro.id, EvaporationSource::LinearizedFromGeometry));
        reference_provenance.push((hydro.id, ref_source));
    }

    Ok((
        EvaporationModelSet::new(all_models),
        provenance,
        reference_provenance,
    ))
}

// ── Evaporation geometry helpers ──────────────────────────────────────────────

/// Linearly interpolate reservoir surface area at volume `v` from the sorted
/// geometry table. Out-of-range `v` clamps to the first/last point's area (no
/// extrapolation). Assumes `geometry` is ascending by `volume_hm3`; returns `0.0`
/// for an empty slice.
fn interpolate_area(geometry: &[&HydroGeometryRow], v: f64) -> f64 {
    if geometry.is_empty() {
        return 0.0;
    }

    let n = geometry.len();

    if v <= geometry[0].volume_hm3 {
        return geometry[0].area_km2;
    }

    if v >= geometry[n - 1].volume_hm3 {
        return geometry[n - 1].area_km2;
    }

    let mut lo = 0usize;
    let mut hi = n - 1;
    while hi - lo > 1 {
        let mid = lo.midpoint(hi);
        if geometry[mid].volume_hm3 <= v {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    let v0 = geometry[lo].volume_hm3;
    let v1 = geometry[hi].volume_hm3;
    let a0 = geometry[lo].area_km2;
    let a1 = geometry[hi].area_km2;

    // Guard against degenerate (identical-volume) points.
    let dv = v1 - v0;
    if dv == 0.0 {
        return a0;
    }

    a0 + (a1 - a0) * (v - v0) / dv
}

/// Finite-difference derivative `dA/dv` at volume `v` from the sorted geometry
/// table, using the enclosing interval's slope (the edge interval when `v` is
/// out of range). Returns `0.0` for a single-point geometry. Assumes `geometry`
/// is ascending by `volume_hm3`.
fn area_derivative(geometry: &[&HydroGeometryRow], v: f64) -> f64 {
    let n = geometry.len();

    if n < 2 {
        return 0.0;
    }

    let (lo, hi) = if v <= geometry[0].volume_hm3 {
        (0, 1)
    } else if v >= geometry[n - 1].volume_hm3 {
        (n - 2, n - 1)
    } else {
        let mut l = 0usize;
        let mut r = n - 1;
        while r - l > 1 {
            let mid = l.midpoint(r);
            if geometry[mid].volume_hm3 <= v {
                l = mid;
            } else {
                r = mid;
            }
        }
        (l, r)
    };

    let dv = geometry[hi].volume_hm3 - geometry[lo].volume_hm3;
    let da = geometry[hi].area_km2 - geometry[lo].area_km2;

    // Guard against degenerate (identical-volume) points.
    if dv == 0.0 {
        return 0.0;
    }

    da / dv
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(
    clippy::doc_markdown,
    clippy::match_wildcard_for_single_variants,
    clippy::cast_precision_loss,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic
)]
mod tests {
    use std::collections::HashMap;

    use chrono::NaiveDate;
    use cobre_core::{
        EntityId,
        entities::hydro::{HydroGenerationModel, HydroPenalties},
        temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
            StageStateConfig,
        },
    };

    use crate::SddpError;

    use super::*;

    // ── Test helpers ──────────────────────────────────────────────────────────

    fn zero_penalties() -> HydroPenalties {
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

    /// Helper: build a slice of HydroGeometryRow references for interpolation tests.
    fn make_geo_rows(volume_area: &[(f64, f64)]) -> Vec<HydroGeometryRow> {
        volume_area
            .iter()
            .map(|&(v, a)| HydroGeometryRow {
                hydro_id: EntityId::from(1),
                volume_hm3: v,
                height_m: 0.0,
                area_km2: a,
            })
            .collect()
    }

    /// Helper: build a Monthly-cycle Stage whose `start_date` falls in the
    /// given 0-based calendar month, with `season_id` set equal to it (the
    /// `Monthly` convention `season_id == month0(start_date)`).
    fn make_stage_with_month(id: i32, month: usize) -> Stage {
        make_stage_with_date_and_season(
            id,
            NaiveDate::from_ymd_opt(2024, u32::try_from(month).unwrap_or(0) + 1, 1)
                .unwrap_or_default(),
            Some(month),
        )
    }

    /// Helper: build a Stage anchored at an explicit `start_date` and
    /// `season_id`, for fixtures where the two diverge (`Custom`, `Weekly`).
    fn make_stage_with_date_and_season(
        id: i32,
        start_date: NaiveDate,
        season_id: Option<usize>,
    ) -> Stage {
        Stage {
            index: usize::try_from(id.max(0)).unwrap_or(0),
            id,
            start_date,
            end_date: start_date
                .checked_add_months(chrono::Months::new(1))
                .unwrap_or(start_date),
            season_id,
            blocks: vec![Block {
                index: 0,
                name: "SINGLE".to_string(),
                duration_hours: 744.0,
            }],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: true,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 50,
                noise_method: NoiseMethod::Saa,
            },
        }
    }

    /// Helper: build a Hydro with the given id and evaporation coefficients.
    fn make_hydro_with_evaporation(
        id: i32,
        min_storage: f64,
        max_storage: f64,
        evap_mm: Option<[f64; 12]>,
    ) -> Hydro {
        let mut hydro = Hydro {
            unit_groups: Vec::new(),
            id: EntityId::from(id),
            name: format!("Hydro{id}"),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            downstream_id: None,
            travel_time_hours: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: min_storage,
            max_storage_hm3: max_storage,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 500.0,
            specific_productivity_mw_per_m3s_per_m: None,
            min_generation_mw: 0.0,
            max_generation_mw: 1000.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: evap_mm,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: zero_penalties(),
        };
        hydro.declare_mirror_unit_group(EntityId::from(10));
        hydro
    }

    #[test]
    fn interpolate_area_exact_first_point() {
        let rows = make_geo_rows(&[(100.0, 1.0), (200.0, 1.5), (300.0, 2.0)]);
        let refs: Vec<_> = rows.iter().collect();
        let result = super::interpolate_area(&refs, 100.0);
        assert!(
            (result - 1.0).abs() < 1e-10,
            "exact first point: expected 1.0, got {result}"
        );
    }

    #[test]
    fn interpolate_area_exact_last_point() {
        let rows = make_geo_rows(&[(100.0, 1.0), (200.0, 1.5), (300.0, 2.0)]);
        let refs: Vec<_> = rows.iter().collect();
        let result = super::interpolate_area(&refs, 300.0);
        assert!(
            (result - 2.0).abs() < 1e-10,
            "exact last point: expected 2.0, got {result}"
        );
    }

    #[test]
    fn interpolate_area_exact_middle_point() {
        let rows = make_geo_rows(&[(100.0, 1.0), (200.0, 1.5), (300.0, 2.0)]);
        let refs: Vec<_> = rows.iter().collect();
        let result = super::interpolate_area(&refs, 200.0);
        assert!(
            (result - 1.5).abs() < 1e-10,
            "exact middle point: expected 1.5, got {result}"
        );
    }

    /// interpolate_area: midpoint between two geometry points is linearly interpolated.
    ///
    /// Geometry: volumes [100, 200, 300, 400, 500], areas [1.0, 1.5, 2.0, 2.5, 3.0].
    /// At v=300, A(300) = 2.0 (exact match). At v=250, A(250) = 1.75 (midpoint of [1.5, 2.0]).
    #[test]
    fn interpolate_area_midpoint_between_two_points() {
        let rows = make_geo_rows(&[
            (100.0, 1.0),
            (200.0, 1.5),
            (300.0, 2.0),
            (400.0, 2.5),
            (500.0, 3.0),
        ]);
        let refs: Vec<_> = rows.iter().collect();
        let result = super::interpolate_area(&refs, 250.0);
        // Midpoint between (200, 1.5) and (300, 2.0): 1.5 + 0.5 * (2.0 - 1.5) = 1.75
        assert!(
            (result - 1.75).abs() < 1e-10,
            "midpoint: expected 1.75, got {result}"
        );
    }

    #[test]
    fn interpolate_area_clamps_below_first_point() {
        let rows = make_geo_rows(&[(100.0, 1.0), (200.0, 1.5), (300.0, 2.0)]);
        let refs: Vec<_> = rows.iter().collect();
        let result = super::interpolate_area(&refs, 50.0);
        assert!(
            (result - 1.0).abs() < 1e-10,
            "below first point: expected clamped area 1.0, got {result}"
        );
    }

    #[test]
    fn interpolate_area_clamps_above_last_point() {
        let rows = make_geo_rows(&[(100.0, 1.0), (200.0, 1.5), (300.0, 2.0)]);
        let refs: Vec<_> = rows.iter().collect();
        let result = super::interpolate_area(&refs, 400.0);
        assert!(
            (result - 2.0).abs() < 1e-10,
            "above last point: expected clamped area 2.0, got {result}"
        );
    }

    // ── area_derivative unit tests ────────────────────────────────────────────

    /// area_derivative: correct finite difference between two points spanning v.
    ///
    /// Geometry: volumes [100, 200, 300, 400, 500], areas [1.0, 1.5, 2.0, 2.5, 3.0].
    /// dA/dv at v=300 uses the interval [200, 300]: (2.0 - 1.5) / (300 - 200) = 0.005.
    #[test]
    fn area_derivative_correct_finite_difference() {
        let rows = make_geo_rows(&[
            (100.0, 1.0),
            (200.0, 1.5),
            (300.0, 2.0),
            (400.0, 2.5),
            (500.0, 3.0),
        ]);
        let refs: Vec<_> = rows.iter().collect();
        let result = super::area_derivative(&refs, 300.0);
        // Interval [200, 300]: (2.0 - 1.5) / (300 - 200) = 0.005
        assert!(
            (result - 0.005).abs() < 1e-10,
            "dA/dv at 300: expected 0.005, got {result}"
        );
    }

    #[test]
    fn area_derivative_single_point_returns_zero() {
        let rows = make_geo_rows(&[(200.0, 1.5)]);
        let refs: Vec<_> = rows.iter().collect();
        let result = super::area_derivative(&refs, 200.0);
        assert!(
            result.abs() < 1e-10,
            "single-point geometry: expected dA/dv = 0.0, got {result}"
        );
    }

    #[test]
    fn area_derivative_at_or_below_first_point_uses_first_interval() {
        let rows = make_geo_rows(&[(100.0, 1.0), (200.0, 1.5), (300.0, 2.0)]);
        let refs: Vec<_> = rows.iter().collect();
        // First interval: (1.5 - 1.0) / (200 - 100) = 0.005
        let result = super::area_derivative(&refs, 50.0);
        assert!(
            (result - 0.005).abs() < 1e-10,
            "below first point: expected first-interval slope 0.005, got {result}"
        );
    }

    #[test]
    fn area_derivative_at_or_above_last_point_uses_last_interval() {
        let rows = make_geo_rows(&[(100.0, 1.0), (200.0, 1.5), (300.0, 2.0)]);
        let refs: Vec<_> = rows.iter().collect();
        // Last interval: (2.0 - 1.5) / (300 - 200) = 0.005
        let result = super::area_derivative(&refs, 400.0);
        assert!(
            (result - 0.005).abs() < 1e-10,
            "above last point: expected last-interval slope 0.005, got {result}"
        );
    }

    /// resolve_evaporation_models core logic: all-no-evaporation system returns all None
    /// without geometry lookup.
    ///
    /// This test calls the internal core logic directly without loading from disk by
    /// using an empty geometry map.
    #[test]
    fn resolve_evaporation_all_none_when_no_hydro_has_coefficients() {
        let hydros = vec![
            make_hydro_with_evaporation(0, 100.0, 500.0, None),
            make_hydro_with_evaporation(1, 200.0, 1000.0, None),
        ];

        // Build the geometry map (empty, since no hydro needs it).
        let geometry_map: HashMap<EntityId, Vec<&HydroGeometryRow>> = HashMap::new();
        let study_stages = [make_stage_with_month(0, 0)];
        let stage_refs: Vec<_> = study_stages.iter().collect();

        let (models, provenance, _ref_provenance) =
            super::resolve_evaporation_core(&hydros, &geometry_map, &stage_refs)
                .expect("should succeed for all-no-evaporation");

        assert_eq!(models.n_hydros(), 2);
        assert!(
            matches!(models.model(0), EvaporationModel::None),
            "hydro 0 must be None"
        );
        assert!(
            matches!(models.model(1), EvaporationModel::None),
            "hydro 1 must be None"
        );
        assert!(!models.has_evaporation(), "has_evaporation() must be false");
        assert_eq!(provenance.len(), 2);
        assert!(
            provenance
                .iter()
                .all(|(_, src)| *src == EvaporationSource::NotModeled)
        );
    }

    /// resolve_evaporation_models core logic: known geometry + coefficient gives correct intercept and slope.
    ///
    /// Spec (acceptance criterion 2):
    ///   hydro: v_min=100, v_max=500, evaporation_coefficients_mm=[5.0; 12]
    ///   geometry: volumes [100, 200, 300, 400, 500], areas [1.0, 1.5, 2.0, 2.5, 3.0]
    ///   reference_volume = (100 + 500) / 2 = 300
    ///   A(300) = 2.0
    ///   dA/dv|_300 = (2.0 - 1.5) / (300 - 200) = 0.005
    ///   January (season_id=0): calendar month = 31·24 = 744 h
    ///   mm_km2_to_m3s = 1 / (3.6 * 744) = 1 / 2678.4
    ///   monthly_evaporation_mm = 5.0
    ///   volume_slope_m3s_per_hm3 = mm_km2_to_m3s * 5.0 * 0.005
    ///   intercept_m3s            = mm_km2_to_m3s * 5.0 * 2.0 - volume_slope_m3s_per_hm3 * 300
    #[test]
    fn resolve_evaporation_known_geometry_produces_correct_coefficients() {
        let evap_mm = [5.0f64; 12];
        let hydro = make_hydro_with_evaporation(0, 100.0, 500.0, Some(evap_mm));

        let geo_rows = make_geo_rows(&[
            (100.0, 1.0),
            (200.0, 1.5),
            (300.0, 2.0),
            (400.0, 2.5),
            (500.0, 3.0),
        ]);
        let geo_refs: Vec<_> = geo_rows.iter().collect();
        let mut geometry_map: HashMap<EntityId, Vec<&HydroGeometryRow>> = HashMap::new();
        geometry_map.insert(EntityId::from(0), geo_refs);

        let study_stages = [make_stage_with_month(0, 0)]; // January
        let stage_refs: Vec<_> = study_stages.iter().collect();

        let (models, provenance, _ref_provenance) =
            super::resolve_evaporation_core(&[hydro], &geometry_map, &stage_refs)
                .expect("should succeed");

        assert_eq!(models.n_hydros(), 1);
        assert_eq!(provenance.len(), 1);
        assert_eq!(provenance[0].1, EvaporationSource::LinearizedFromGeometry);

        match models.model(0) {
            EvaporationModel::Linearized {
                coefficients,
                reference_volumes_hm3,
            } => {
                assert_eq!(
                    reference_volumes_hm3.len(),
                    1,
                    "must have one ref volume per stage"
                );
                assert!(
                    (reference_volumes_hm3[0] - 300.0).abs() < 1e-10,
                    "reference_volume must be (100+500)/2 = 300, got {}",
                    reference_volumes_hm3[0]
                );
                assert_eq!(coefficients.len(), 1);

                let reference_volume = 300.0_f64;
                let a_ref = 2.0_f64;
                let da_dv = 0.005_f64;
                let monthly_evaporation_mm = 5.0_f64;
                let month_hours = 744.0_f64;
                let mm_km2_to_m3s = 1.0 / (3.6 * month_hours);

                let expected_slope = mm_km2_to_m3s * monthly_evaporation_mm * da_dv;
                let expected_intercept = mm_km2_to_m3s * monthly_evaporation_mm * a_ref
                    - expected_slope * reference_volume;

                let coeff = &coefficients[0];
                assert!(
                    (coeff.volume_slope_m3s_per_hm3 - expected_slope).abs() < 1e-10,
                    "volume_slope_m3s_per_hm3: expected {expected_slope}, got {}",
                    coeff.volume_slope_m3s_per_hm3
                );
                assert!(
                    (coeff.intercept_m3s - expected_intercept).abs() < 1e-10,
                    "intercept_m3s: expected {expected_intercept}, got {}",
                    coeff.intercept_m3s
                );
            }
            other => panic!("expected Linearized, got {other:?}"),
        }
    }

    /// resolve_evaporation_models core logic: negative evaporation coefficients produce valid results.
    ///
    /// Net precipitation (negative monthly evaporation) is physically valid; the
    /// volume slope can be negative.
    #[test]
    fn resolve_evaporation_negative_coefficient_produces_valid_results() {
        let mut evap_mm = [0.0f64; 12];
        evap_mm[0] = -3.0; // net precipitation in January
        let hydro = make_hydro_with_evaporation(0, 100.0, 500.0, Some(evap_mm));

        let geo_rows = make_geo_rows(&[(100.0, 1.0), (200.0, 1.5), (300.0, 2.0)]);
        let geo_refs: Vec<_> = geo_rows.iter().collect();
        let mut geometry_map: HashMap<EntityId, Vec<&HydroGeometryRow>> = HashMap::new();
        geometry_map.insert(EntityId::from(0), geo_refs);

        let study_stages = [make_stage_with_month(0, 0)]; // January
        let stage_refs: Vec<_> = study_stages.iter().collect();

        let (models, provenance, _ref_provenance) =
            super::resolve_evaporation_core(&[hydro], &geometry_map, &stage_refs)
                .expect("negative evaporation must succeed");

        assert_eq!(provenance[0].1, EvaporationSource::LinearizedFromGeometry);

        match models.model(0) {
            EvaporationModel::Linearized { coefficients, .. } => {
                let coeff = &coefficients[0];
                assert!(
                    coeff.volume_slope_m3s_per_hm3.is_finite(),
                    "volume_slope_m3s_per_hm3 must be finite for negative monthly evaporation"
                );
                assert!(
                    coeff.intercept_m3s.is_finite(),
                    "intercept_m3s must be finite for negative monthly evaporation"
                );
                // Negative monthly evaporation with positive dA/dv → negative slope.
                assert!(
                    coeff.volume_slope_m3s_per_hm3 < 0.0,
                    "volume_slope_m3s_per_hm3 must be negative for net precipitation scenario"
                );
            }
            other => panic!("expected Linearized, got {other:?}"),
        }
    }

    /// A hydro with evaporation coefficients but no geometry rows degrades to
    /// disabled evaporation (`DisabledNoArea`) instead of erroring.
    #[test]
    fn resolve_evaporation_missing_geometry_disables_evaporation() {
        let evap_mm = [5.0f64; 12];
        let hydro = make_hydro_with_evaporation(0, 100.0, 500.0, Some(evap_mm));

        // Geometry map has no entry for hydro 0.
        let geometry_map: HashMap<EntityId, Vec<&HydroGeometryRow>> = HashMap::new();

        let study_stages = [make_stage_with_month(0, 0)];
        let stage_refs: Vec<_> = study_stages.iter().collect();

        let (models, provenance, _ref_provenance) =
            super::resolve_evaporation_core(&[hydro], &geometry_map, &stage_refs)
                .expect("missing geometry must degrade, not error");

        assert!(
            matches!(models.model(0), EvaporationModel::None),
            "hydro 0 evaporation must be disabled (None)"
        );
        assert_eq!(
            provenance[0].1,
            EvaporationSource::DisabledNoArea,
            "provenance must record DisabledNoArea"
        );
    }

    /// A hydro with evaporation coefficients but a geometry whose areas are all
    /// zero — e.g. a new/being-filled reservoir with only a dead-volume point,
    /// as JURUENA in cobre_rodada_2001 — degrades to disabled evaporation
    /// instead of failing the whole run.
    #[test]
    fn resolve_evaporation_all_zero_area_disables_evaporation() {
        let evap_mm = [5.0f64; 12];
        let hydro = make_hydro_with_evaporation(0, 100.0, 500.0, Some(evap_mm));

        // Single dead-volume point with zero surface area (no area-volume curve).
        let geo_rows = make_geo_rows(&[(2.93, 0.0)]);
        let refs: Vec<_> = geo_rows.iter().collect();
        let mut geometry_map: HashMap<EntityId, Vec<&HydroGeometryRow>> = HashMap::new();
        geometry_map.insert(EntityId::from(0), refs);

        let study_stages = [make_stage_with_month(0, 0)];
        let stage_refs: Vec<_> = study_stages.iter().collect();

        let (models, provenance, _ref_provenance) =
            super::resolve_evaporation_core(&[hydro], &geometry_map, &stage_refs)
                .expect("all-zero area must degrade, not error");

        assert!(
            matches!(models.model(0), EvaporationModel::None),
            "hydro 0 evaporation must be disabled (None) when all areas are zero"
        );
        assert_eq!(
            provenance[0].1,
            EvaporationSource::DisabledNoArea,
            "provenance must record DisabledNoArea"
        );
    }

    /// resolve_evaporation_models core logic: 4 hydros where 2 have evaporation and 2 do not.
    ///
    /// Acceptance criterion 1: returns 2 Linearized and 2 None models.
    #[test]
    fn resolve_evaporation_mixed_system_returns_correct_model_mix() {
        let evap_mm = [5.0f64; 12];
        let hydros = vec![
            make_hydro_with_evaporation(0, 100.0, 500.0, Some(evap_mm)),
            make_hydro_with_evaporation(1, 200.0, 1000.0, None),
            make_hydro_with_evaporation(2, 50.0, 300.0, Some(evap_mm)),
            make_hydro_with_evaporation(3, 300.0, 2000.0, None),
        ];

        let geo_rows_h0 = make_geo_rows(&[(100.0, 1.0), (300.0, 2.0), (500.0, 3.0)]);
        let geo_rows_h2 = make_geo_rows(&[(50.0, 0.5), (175.0, 1.0), (300.0, 1.5)]);

        let refs_h0: Vec<_> = geo_rows_h0.iter().collect();
        let refs_h2: Vec<_> = geo_rows_h2.iter().collect();

        let mut geometry_map: HashMap<EntityId, Vec<&HydroGeometryRow>> = HashMap::new();
        geometry_map.insert(EntityId::from(0), refs_h0);
        geometry_map.insert(EntityId::from(2), refs_h2);

        let study_stages = [make_stage_with_month(0, 0)];
        let stage_refs: Vec<_> = study_stages.iter().collect();

        let (models, provenance, _ref_provenance) =
            super::resolve_evaporation_core(&hydros, &geometry_map, &stage_refs)
                .expect("should succeed");

        assert_eq!(models.n_hydros(), 4);
        assert!(
            matches!(models.model(0), EvaporationModel::Linearized { .. }),
            "hydro 0 must be Linearized"
        );
        assert!(
            matches!(models.model(1), EvaporationModel::None),
            "hydro 1 must be None"
        );
        assert!(
            matches!(models.model(2), EvaporationModel::Linearized { .. }),
            "hydro 2 must be Linearized"
        );
        assert!(
            matches!(models.model(3), EvaporationModel::None),
            "hydro 3 must be None"
        );

        // 2 Linearized, 2 NotModeled in provenance.
        let n_linearized = provenance
            .iter()
            .filter(|(_, s)| *s == EvaporationSource::LinearizedFromGeometry)
            .count();
        let n_not_modeled = provenance
            .iter()
            .filter(|(_, s)| *s == EvaporationSource::NotModeled)
            .count();
        assert_eq!(n_linearized, 2, "expected 2 LinearizedFromGeometry");
        assert_eq!(n_not_modeled, 2, "expected 2 NotModeled");
    }

    /// resolve_evaporation_core: a zero-duration stage is rejected by the
    /// explicit `stage_hours > 0` guard (the divisor is the calendar month now,
    /// so a zero-duration stage no longer surfaces as a non-finite coefficient).
    #[test]
    fn resolve_evaporation_zero_duration_stage_is_rejected() {
        let evap_mm = [5.0f64; 12];
        let hydro = make_hydro_with_evaporation(0, 100.0, 500.0, Some(evap_mm));

        let mut stage_zero_duration = make_stage_with_month(0, 0);
        stage_zero_duration.blocks = vec![Block {
            index: 0,
            name: "ZERO".to_string(),
            duration_hours: 0.0,
        }];

        let geo_rows = make_geo_rows(&[(100.0, 1.0), (200.0, 1.5), (300.0, 2.0)]);
        let geo_refs: Vec<_> = geo_rows.iter().collect();
        let mut geometry_map: HashMap<EntityId, Vec<&HydroGeometryRow>> = HashMap::new();
        geometry_map.insert(EntityId::from(0), geo_refs);

        let stage_refs = vec![&stage_zero_duration];

        let err = super::resolve_evaporation_core(&[hydro], &geometry_map, &stage_refs)
            .expect_err("a zero-duration stage must return an error");

        assert!(
            matches!(err, SddpError::Validation(_)),
            "expected Validation error for non-finite coefficients, got {err:?}"
        );
    }

    // ── Per-season reference volume tests ────────────────────────────────────

    /// resolve_evaporation_core: user-supplied per-season reference volumes produce
    /// stage coefficients derived from the month-specific reference_volume.
    ///
    /// Geometry: volumes [100, 200, 300, 400, 500], areas [1.0, 1.5, 2.0, 2.5, 3.0].
    /// ref_vols[0] = 200 (January), ref_vols[1] = 400 (February).
    /// Hydro: v_min=100, v_max=500.
    /// Stage 0: season_id=0, 744h. Stage 1: season_id=1, 672h.
    ///
    /// For stage 0 (reference_volume=200): A(200)=1.5, dA/dv=(2.0-1.5)/(300-200)=0.005
    /// For stage 1 (reference_volume=400): A(400)=2.5, dA/dv=(3.0-2.5)/(500-400)=0.005
    #[test]
    fn resolve_evaporation_per_season_ref_vols_produces_per_stage_coefficients() {
        let mut ref_vols = [0.0f64; 12];
        ref_vols[0] = 200.0; // January
        ref_vols[1] = 400.0; // February

        let mut hydro = make_hydro_with_evaporation(0, 100.0, 500.0, Some([5.0f64; 12]));
        hydro.evaporation_reference_volumes_hm3 = Some(ref_vols);

        let geo_rows = make_geo_rows(&[
            (100.0, 1.0),
            (200.0, 1.5),
            (300.0, 2.0),
            (400.0, 2.5),
            (500.0, 3.0),
        ]);
        let geo_refs: Vec<_> = geo_rows.iter().collect();
        let mut geometry_map: HashMap<EntityId, Vec<&HydroGeometryRow>> = HashMap::new();
        geometry_map.insert(EntityId::from(0), geo_refs);

        // Two stages: January (stage 744h) and February 2024 (stage 672h).
        // Evaporation divides by the CALENDAR month: Jan 744h, leap-Feb 696h.
        let stage_jan = make_stage_with_month(0, 0);
        let mut stage_feb = make_stage_with_month(1, 1);
        stage_feb.blocks = vec![Block {
            index: 0,
            name: "FEB".to_string(),
            duration_hours: 672.0,
        }];
        let stage_refs = vec![&stage_jan, &stage_feb];

        let (models, evap_provenance, ref_provenance) =
            super::resolve_evaporation_core(&[hydro], &geometry_map, &stage_refs)
                .expect("should succeed");

        assert_eq!(models.n_hydros(), 1);
        assert_eq!(
            evap_provenance[0].1,
            EvaporationSource::LinearizedFromGeometry
        );
        assert_eq!(
            ref_provenance[0].1,
            EvaporationReferenceSource::UserSupplied,
            "user-supplied volumes must produce UserSupplied provenance"
        );

        match models.model(0) {
            EvaporationModel::Linearized {
                coefficients,
                reference_volumes_hm3,
            } => {
                assert_eq!(coefficients.len(), 2, "must have 2 stage coefficients");
                assert_eq!(reference_volumes_hm3.len(), 2, "must have 2 ref volumes");

                // Stage 0: reference_volume=200
                assert!(
                    (reference_volumes_hm3[0] - 200.0).abs() < 1e-10,
                    "stage 0 ref vol must be 200, got {}",
                    reference_volumes_hm3[0]
                );

                // Stage 1: reference_volume=400
                assert!(
                    (reference_volumes_hm3[1] - 400.0).abs() < 1e-10,
                    "stage 1 ref vol must be 400, got {}",
                    reference_volumes_hm3[1]
                );

                // Verify stage 0 coefficients using reference_volume=200.
                let monthly_evaporation_mm = 5.0_f64;
                let da_dv = 0.005_f64; // same slope in both segments

                let mm_km2_to_m3s_jan = 1.0 / (3.6 * 744.0);
                let a_jan = 1.5_f64;
                let reference_volume_jan = 200.0_f64;
                let expected_slope_jan = mm_km2_to_m3s_jan * monthly_evaporation_mm * da_dv;
                let expected_intercept_jan = mm_km2_to_m3s_jan * monthly_evaporation_mm * a_jan
                    - expected_slope_jan * reference_volume_jan;
                assert!(
                    (coefficients[0].volume_slope_m3s_per_hm3 - expected_slope_jan).abs() < 1e-10,
                    "stage 0 volume_slope_m3s_per_hm3: expected {expected_slope_jan}, got {}",
                    coefficients[0].volume_slope_m3s_per_hm3
                );
                assert!(
                    (coefficients[0].intercept_m3s - expected_intercept_jan).abs() < 1e-10,
                    "stage 0 intercept_m3s: expected {expected_intercept_jan}, got {}",
                    coefficients[0].intercept_m3s
                );

                // Verify stage 1 coefficients using reference_volume=400.
                // Divisor is the calendar month, not the stage's 672 h:
                // February 2024 is a leap month, 29 · 24 = 696 h.
                let mm_km2_to_m3s_feb = 1.0 / (3.6 * 696.0);
                let a_feb = 2.5_f64;
                let reference_volume_feb = 400.0_f64;
                let expected_slope_feb = mm_km2_to_m3s_feb * monthly_evaporation_mm * da_dv;
                let expected_intercept_feb = mm_km2_to_m3s_feb * monthly_evaporation_mm * a_feb
                    - expected_slope_feb * reference_volume_feb;
                assert!(
                    (coefficients[1].volume_slope_m3s_per_hm3 - expected_slope_feb).abs() < 1e-10,
                    "stage 1 volume_slope_m3s_per_hm3: expected {expected_slope_feb}, got {}",
                    coefficients[1].volume_slope_m3s_per_hm3
                );
                assert!(
                    (coefficients[1].intercept_m3s - expected_intercept_feb).abs() < 1e-10,
                    "stage 1 intercept_m3s: expected {expected_intercept_feb}, got {}",
                    coefficients[1].intercept_m3s
                );
            }
            other => panic!("expected Linearized, got {other:?}"),
        }
    }

    /// resolve_evaporation_core: None reference volumes produce DefaultMidpoint provenance and
    /// all reference_volumes_hm3 entries equal (v_min + v_max) / 2.
    #[test]
    fn resolve_evaporation_none_ref_vols_produces_default_midpoint_provenance() {
        // `make_hydro_with_evaporation` already sets evaporation_reference_volumes_hm3 = None.
        let hydro = make_hydro_with_evaporation(0, 100.0, 500.0, Some([5.0f64; 12]));

        let geo_rows = make_geo_rows(&[
            (100.0, 1.0),
            (200.0, 1.5),
            (300.0, 2.0),
            (400.0, 2.5),
            (500.0, 3.0),
        ]);
        let geo_refs: Vec<_> = geo_rows.iter().collect();
        let mut geometry_map: HashMap<EntityId, Vec<&HydroGeometryRow>> = HashMap::new();
        geometry_map.insert(EntityId::from(0), geo_refs);

        // Two stages with different months (January = 0, June = 5).
        let stage_january = make_stage_with_month(0, 0);
        let stage_june = make_stage_with_month(1, 5);
        let stage_refs = vec![&stage_january, &stage_june];

        let (models, evap_provenance, ref_provenance) =
            super::resolve_evaporation_core(&[hydro], &geometry_map, &stage_refs)
                .expect("should succeed");

        assert_eq!(
            ref_provenance[0].1,
            EvaporationReferenceSource::DefaultMidpoint,
            "None reference volumes must produce DefaultMidpoint provenance"
        );
        assert_eq!(
            evap_provenance[0].1,
            EvaporationSource::LinearizedFromGeometry
        );

        let expected_reference_volume = f64::midpoint(100.0, 500.0); // 300.0

        match models.model(0) {
            EvaporationModel::Linearized {
                reference_volumes_hm3,
                ..
            } => {
                assert_eq!(
                    reference_volumes_hm3.len(),
                    2,
                    "must have 2 ref volumes (one per stage)"
                );
                for (s, &v) in reference_volumes_hm3.iter().enumerate() {
                    assert!(
                        (v - expected_reference_volume).abs() < 1e-10,
                        "stage {s} ref vol must be midpoint {expected_reference_volume}, got {v}"
                    );
                }
            }
            other => panic!("expected Linearized, got {other:?}"),
        }
    }

    /// resolve_evaporation_core: mixed hydro set (one with user-supplied, one without)
    /// produces correct per-hydro provenance.
    #[test]
    fn resolve_evaporation_mixed_ref_vol_provenance() {
        let mut ref_vols = [300.0f64; 12];
        ref_vols[0] = 200.0;

        let mut hydro_with = make_hydro_with_evaporation(0, 100.0, 500.0, Some([5.0f64; 12]));
        hydro_with.evaporation_reference_volumes_hm3 = Some(ref_vols);

        let hydro_without = make_hydro_with_evaporation(1, 100.0, 500.0, Some([5.0f64; 12]));
        // hydro_without.evaporation_reference_volumes_hm3 is already None.

        let geo_rows = make_geo_rows(&[(100.0, 1.0), (300.0, 2.0), (500.0, 3.0)]);
        let refs: Vec<_> = geo_rows.iter().collect();
        let mut geometry_map: HashMap<EntityId, Vec<&HydroGeometryRow>> = HashMap::new();
        geometry_map.insert(EntityId::from(0), refs.clone());
        geometry_map.insert(EntityId::from(1), refs);

        let stage = make_stage_with_month(0, 0);
        let stage_refs = vec![&stage];

        let (_, _, ref_provenance) = super::resolve_evaporation_core(
            &[hydro_with, hydro_without],
            &geometry_map,
            &stage_refs,
        )
        .expect("should succeed");

        assert_eq!(ref_provenance.len(), 2);
        assert_eq!(
            ref_provenance[0].1,
            EvaporationReferenceSource::UserSupplied,
            "hydro with ref vols must be UserSupplied"
        );
        assert_eq!(
            ref_provenance[1].1,
            EvaporationReferenceSource::DefaultMidpoint,
            "hydro without ref vols must be DefaultMidpoint"
        );
    }

    // ── Derive-correct month resolution (season_id decoupled from month) ──────

    /// resolve_evaporation_core: a Custom-cycle stage's `season_id` is a
    /// non-monthly bucket; evaporation must index `coefficients_mm` by the
    /// month `month_of` derives from `start_date`, not by `season_id`.
    #[test]
    fn resolve_evaporation_custom_cycle_indexes_by_derived_month_not_season_id() {
        let mut evap_mm = [0.0f64; 12];
        evap_mm[3] = 99.0; // April: season_id's month, must NOT be used
        evap_mm[5] = 7.0; // June: start_date's month, must be used
        let hydro = make_hydro_with_evaporation(0, 100.0, 500.0, Some(evap_mm));

        let geo_rows = make_geo_rows(&[
            (100.0, 1.0),
            (200.0, 1.5),
            (300.0, 2.0),
            (400.0, 2.5),
            (500.0, 3.0),
        ]);
        let geo_refs: Vec<_> = geo_rows.iter().collect();
        let mut geometry_map: HashMap<EntityId, Vec<&HydroGeometryRow>> = HashMap::new();
        geometry_map.insert(EntityId::from(0), geo_refs);

        // Custom cycle: season_id=3 (April, a non-monthly bucket) but start_date is in June.
        let june = NaiveDate::from_ymd_opt(2024, 6, 10).expect("valid date");
        let stage = make_stage_with_date_and_season(0, june, Some(3));
        let stage_refs = vec![&stage];

        let (models, _, _) = super::resolve_evaporation_core(&[hydro], &geometry_map, &stage_refs)
            .expect("should succeed");

        let reference_volume = 300.0_f64;
        let a_ref = 2.0_f64;
        let da_dv = 0.005_f64;
        // Divisor is June's calendar month: 30 · 24 = 720 h.
        let mm_km2_to_m3s = 1.0 / (3.6 * 720.0_f64);
        let expected_slope = mm_km2_to_m3s * 7.0 * da_dv;
        let expected_intercept = mm_km2_to_m3s * 7.0 * a_ref - expected_slope * reference_volume;

        match models.model(0) {
            EvaporationModel::Linearized { coefficients, .. } => {
                assert!(
                    (coefficients[0].volume_slope_m3s_per_hm3 - expected_slope).abs() < 1e-10,
                    "must use June's coefficient (7.0), not April's (99.0): expected slope \
                     {expected_slope}, got {}",
                    coefficients[0].volume_slope_m3s_per_hm3
                );
                assert!(
                    (coefficients[0].intercept_m3s - expected_intercept).abs() < 1e-10,
                    "must use June's coefficient (7.0), not April's (99.0): expected intercept \
                     {expected_intercept}, got {}",
                    coefficients[0].intercept_m3s
                );
            }
            other => panic!("expected Linearized, got {other:?}"),
        }
    }

    /// resolve_evaporation_core: a Weekly-cycle evaporating stage
    /// (`season_id >= 12`) no longer hard-errors — the month is derived from
    /// `start_date` regardless of `season_id`'s range.
    #[test]
    fn resolve_evaporation_weekly_cycle_no_longer_errors_on_season_id_ge_12() {
        let evap_mm = [5.0f64; 12];
        let hydro = make_hydro_with_evaporation(0, 100.0, 500.0, Some(evap_mm));

        let geo_rows = make_geo_rows(&[(100.0, 1.0), (300.0, 2.0), (500.0, 3.0)]);
        let geo_refs: Vec<_> = geo_rows.iter().collect();
        let mut geometry_map: HashMap<EntityId, Vec<&HydroGeometryRow>> = HashMap::new();
        geometry_map.insert(EntityId::from(0), geo_refs);

        // Weekly cycle: season_id=48 (>= 12), a week in late December.
        let late_december = NaiveDate::from_ymd_opt(2024, 12, 23).expect("valid date");
        let stage = make_stage_with_date_and_season(0, late_december, Some(48));
        let stage_refs = vec![&stage];

        let (models, provenance, _) =
            super::resolve_evaporation_core(&[hydro], &geometry_map, &stage_refs)
                .expect("season_id >= 12 (Weekly) must no longer error");

        assert_eq!(provenance[0].1, EvaporationSource::LinearizedFromGeometry);
        assert!(matches!(
            models.model(0),
            EvaporationModel::Linearized { .. }
        ));
    }
}
