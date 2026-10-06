//! Production model resolution: per-`(hydro, stage)` constant productivity or FPHA.
//!
//! Resolves each hydro's production function from the case directory: constant
//! productivity from the entity definition / parquet override, precomputed FPHA
//! hyperplanes, or FPHA hyperplanes fitted from reservoir geometry via the
//! `crate::fpha_fitting` pipeline. Produces the `ProductionModelSet`, the
//! per-hydro `ProductionModelSource` provenance, the `ρ_eq` override carried for
//! energy-conversion derivation, and the computed-FPHA export rows.

use std::collections::HashMap;

use rayon::prelude::*;

use cobre_core::temporal::Stage;
use cobre_core::{
    EntityId, Hydro, StageId, StudyPos, System, entities::hydro::HydroGenerationModel,
};
use cobre_io::CaseArtifacts;
use cobre_io::FphaDeviationPointRow;
use cobre_io::HydroReferenceVolumeFractions;
use cobre_io::extensions::PlaneReductionConfig;
use cobre_io::extensions::{
    FphaColumnLayout, FphaHyperplaneRow, HydroGeometryRow, ProductionModelConfig, ReferenceVolume,
    SeasonConfig, SelectionMode, StageRange, build_hydro_reference_volumes_resolved,
};

use super::types::{
    FphaFitDeviationEntry, FphaPlane, ProductionModelSet, ProductionModelSource,
    ResolvedProductionModel,
};
use crate::SddpError;
use crate::energy_conversion::{
    HydroEnergyProductivityOverride, build_hydro_energy_productivity_override,
};
use crate::fpha_fitting::{
    ForebayTable, FphaDeviationPoint, FphaFitDeviation, FphaFitResult, TailraceFamilies,
    TailraceSource, build_tailrace_families_map, fit_fpha_planes,
};
// ── FPHA production model resolution ─────────────────────────────────────────

/// Return type for [`resolve_production_models_from_artifacts`]. Export rows are non-empty only
/// when at least one hydro uses `source: "computed"`; this function never does I/O.
type ResolveProductionResult = (
    ProductionModelSet,
    HydroEnergyProductivityOverride,
    Vec<(EntityId, ProductionModelSource)>,
    Vec<FphaHyperplaneRow>,
    Vec<(EntityId, StudyPos, f64)>,
    Vec<FphaFitDeviationEntry>,
    Vec<FphaDeviationPointRow>,
);

/// Resolve per-hydro per-stage production models from a pre-parsed
/// [`cobre_io::CaseArtifacts`] bundle.
///
/// Absent a `hydro_production_models.json` entry, every hydro falls back to its
/// entity [`HydroGenerationModel`]. `collect_deviation_points` is the run-level
/// opt-in from `config.exports.fpha_deviation_points`: `false` leaves the
/// deviation rows empty and the fit bit-identical (zero collection overhead). The
/// provenance vector is in canonical hydro ID order.
///
/// # Errors
///
/// | Condition                                                       | Error variant              |
/// | --------------------------------------------------------------- | -------------------------- |
/// | `Fpha` entity model with no config entry                        | [`SddpError::Validation`]  |
/// | `source: "computed"` with missing tailrace/losses/efficiency    | [`SddpError::Validation`]  |
/// | `source: "computed"` with no geometry rows for the hydro        | [`SddpError::Validation`]  |
/// | FPHA fitting pipeline error                                     | [`SddpError::Validation`]  |
/// | `gamma_v < 0` for any precomputed hyperplane                    | [`SddpError::Validation`]  |
/// | `gamma_s > 0` for any precomputed hyperplane                    | [`SddpError::Validation`]  |
/// | `gamma_q < 0` for any precomputed hyperplane                    | [`SddpError::Validation`]  |
/// | `kappa` not in `(0, 1]` for precomputed hyperplane              | [`SddpError::Validation`]  |
/// | Zero hyperplanes for an FPHA hydro at any stage                 | [`SddpError::Validation`]  |
/// | No precomputed hyperplane with `gamma_q > 0` for an FPHA hydro at a stage | [`SddpError::Validation`]  |
pub fn resolve_production_models_from_artifacts(
    system: &System,
    artifacts: &CaseArtifacts,
    collect_deviation_points: bool,
) -> Result<ResolveProductionResult, SddpError> {
    let override_table =
        build_hydro_energy_productivity_override(&artifacts.hydro_energy_productivity)
            .map_err(|e| SddpError::Validation(e.to_string()))?;

    let prod_configs: &[ProductionModelConfig] = &artifacts.production_models;

    let plane_reduction: Option<&PlaneReductionConfig> = artifacts.plane_reduction.as_ref();

    let config_map: HashMap<EntityId, &ProductionModelConfig> =
        prod_configs.iter().map(|c| (c.hydro_id, c)).collect();

    let mut hyperplane_map: HashMap<(EntityId, Option<i32>), Vec<&FphaHyperplaneRow>> =
        HashMap::new();
    if prod_configs.iter().any(config_uses_precomputed_fpha) {
        for row in &artifacts.fpha_hyperplanes {
            hyperplane_map
                .entry((row.hydro_id, row.stage_id))
                .or_default()
                .push(row);
        }
    }

    let uses_computed_fpha = prod_configs.iter().any(config_uses_computed_fpha);

    let geometry_map: HashMap<EntityId, Vec<&HydroGeometryRow>> = if uses_computed_fpha {
        build_geometry_map(&artifacts.hydro_geometry)
    } else {
        HashMap::new()
    };

    let families_map: HashMap<EntityId, TailraceFamilies> =
        if uses_computed_fpha && !artifacts.tailrace_curves.is_empty() {
            build_tailrace_families_map(&artifacts.tailrace_curves)?
        } else {
            HashMap::new()
        };

    let long_term_mean_inflow_table: HashMap<EntityId, f64> = if uses_computed_fpha {
        build_long_term_mean_inflow_table(system)
    } else {
        HashMap::new()
    };

    let study_stages: Vec<&Stage> = system.stages().iter().filter(|s| s.id >= 0).collect();
    let n_stages = study_stages.len();
    let n_hydros = system.hydros().len();

    // The same resolved table feeds the energy-conversion reference in
    // `setup::build_energy_and_templates`, so the backwater level and the
    // productivity reference share one source of truth.
    let reference_volumes_hm3: Vec<(EntityId, StudyPos, f64)> = system
        .hydros()
        .iter()
        .flat_map(|hydro| {
            study_stages.iter().enumerate().map(|(stage_pos, stage)| {
                let rv = config_map
                    .get(&hydro.id)
                    .and_then(|config| find_reference_volume_for_stage(config, stage));
                let resolved =
                    resolve_reference_volume_hm3(rv, hydro.min_storage_hm3, hydro.max_storage_hm3);
                // Key by the 0-based `study_stages` position, NOT `stage.index`:
                // both consumers (`resolve_downstream_level`,
                // `build_energy_conversion_set`) query by study position;
                // `stage.index` shifts every key by the pre-study-stage count.
                (hydro.id, StudyPos(stage_pos), resolved)
            })
        })
        .collect();
    let reference_volume_fractions =
        build_hydro_reference_volumes_resolved(&reference_volumes_hm3, 0.0);

    let mut all_models: Vec<Vec<ResolvedProductionModel>> = Vec::with_capacity(n_hydros);
    let mut provenance: Vec<(EntityId, ProductionModelSource)> = Vec::with_capacity(n_hydros);
    let mut export_rows: Vec<FphaHyperplaneRow> = Vec::new();
    let mut fpha_fit_deviations: Vec<FphaFitDeviationEntry> = Vec::new();
    // Per-sampled-point deviation rows, concatenated below in the same sequential
    // canonical-order flatten as `export_rows`. Empty unless the opt-in is on.
    let mut fpha_deviation_point_rows: Vec<FphaDeviationPointRow> = Vec::new();

    // Determinism: `par_iter().collect()` reassembles the fits in canonical
    // hydro order regardless of thread scheduling, then the SEQUENTIAL flatten
    // below preserves `(hydro_id, stage_id, plane_id)` export ordering bit-for-bit.
    // Collecting into a shared `Mutex<Vec>`, pushing rows from worker threads, or
    // keying the reduction seed on merge history would reorder the stream and break
    // bit-determinism.
    let fits: Vec<PerHydroFit> = system
        .hydros()
        .par_iter()
        .map(|hydro| {
            fit_one_hydro(
                hydro,
                &config_map,
                &geometry_map,
                &families_map,
                &reference_volume_fractions,
                &hyperplane_map,
                &override_table,
                &study_stages,
                plane_reduction,
                system,
                n_stages,
                collect_deviation_points,
                &long_term_mean_inflow_table,
            )
        })
        .collect::<Result<Vec<_>, SddpError>>()?;
    for (hydro, fit) in system.hydros().iter().zip(fits) {
        if fit.provenance.1 == ProductionModelSource::NoTurbineCapacity {
            tracing::warn!(
                "hydro {} (id={}) requests FPHA but has no turbine capacity \
                 (max_turbined_m3s = {}); modeling it with zero productivity",
                hydro.name,
                hydro.id.0,
                hydro.max_turbined_m3s
            );
        }
        provenance.push(fit.provenance);
        export_rows.extend(fit.export_rows);
        fpha_deviation_point_rows.extend(fit.deviation_point_rows);
        all_models.push(fit.stage_models);
        for diag in fit.fpha_deviations {
            fpha_fit_deviations.push(FphaFitDeviationEntry {
                hydro_id: hydro.id,
                stage_id: diag.stage_id,
                mean_abs_mw: diag.deviation.mean_abs_mw,
                max_abs_mw: diag.deviation.max_abs_mw,
                mean_signed_mw: diag.deviation.mean_signed_mw,
                relative: diag.deviation.relative,
            });
            if diag.deviation.exceeds_warn_threshold() {
                tracing::warn!(
                    "FPHA fit for hydro {} (stage {}) deviates {:.1}% from the exact \
                     production function (mean |Δ| {:.1} MW, max {:.1} MW); the \
                     convex-hull approximation is poor here — typically a strongly \
                     non-concave production surface that no single α correction can track",
                    diag.hydro_name,
                    diag.stage_id,
                    diag.deviation.relative * 100.0,
                    diag.deviation.mean_abs_mw,
                    diag.deviation.max_abs_mw,
                );
            }
        }
    }

    let set = ProductionModelSet::new(all_models, system.hydros(), n_stages);
    Ok((
        set,
        override_table,
        provenance,
        export_rows,
        reference_volumes_hm3,
        fpha_fit_deviations,
        fpha_deviation_point_rows,
    ))
}

/// One computed-FPHA fit's deviation, recorded once per distinct fit (per
/// `SelectionMode` entry). `stage_id` is the first study stage the entry covers.
struct FphaDeviationDiagnostic {
    hydro_name: String,
    stage_id: i32,
    deviation: FphaFitDeviation,
}

/// One hydro's fit outputs, owned by value so the parallel fit shares no `&mut`
/// accumulator. Each `Vec` is in this hydro's stage order; the caller concatenates
/// them in `system.hydros()` order for a declaration-order-invariant result.
struct PerHydroFit {
    stage_models: Vec<ResolvedProductionModel>,
    provenance: (EntityId, ProductionModelSource),
    export_rows: Vec<FphaHyperplaneRow>,
    fpha_deviations: Vec<FphaDeviationDiagnostic>,
    /// Per-sampled-point rows; empty unless `collect_deviation_points` is on.
    deviation_point_rows: Vec<FphaDeviationPointRow>,
}

fn no_turbine_capacity_fit(hydro_id: EntityId, n_stages: usize) -> PerHydroFit {
    PerHydroFit {
        stage_models: vec![
            ResolvedProductionModel::ConstantProductivity { productivity: 0.0 };
            n_stages
        ],
        provenance: (hydro_id, ProductionModelSource::NoTurbineCapacity),
        export_rows: Vec::new(),
        fpha_deviations: Vec::new(),
        deviation_point_rows: Vec::new(),
    }
}

fn has_hyperplane_rows(
    hyperplane_map: &HashMap<(EntityId, Option<i32>), Vec<&FphaHyperplaneRow>>,
    hydro_id: EntityId,
) -> bool {
    hyperplane_map.keys().any(|(id, _)| *id == hydro_id)
}

/// Resolve every study-stage production model for ONE hydro, returning the
/// per-hydro result by value with no shared `&mut` capture.
///
/// # Errors
///
/// Propagates the first [`SddpError`] from `determine_source`,
/// `fit_computed_planes_per_stage`, or `resolve_stage_model` for this hydro.
// Rationale: each argument is a distinct already-resolved upstream datum; a
// context struct would only relocate the same fields.
#[allow(clippy::too_many_arguments)]
fn fit_one_hydro(
    hydro: &Hydro,
    config_map: &HashMap<EntityId, &ProductionModelConfig>,
    geometry_map: &HashMap<EntityId, Vec<&HydroGeometryRow>>,
    families_map: &HashMap<EntityId, TailraceFamilies>,
    reference_volume_fractions: &HydroReferenceVolumeFractions,
    hyperplane_map: &HashMap<(EntityId, Option<i32>), Vec<&FphaHyperplaneRow>>,
    override_table: &HydroEnergyProductivityOverride,
    study_stages: &[&Stage],
    plane_reduction: Option<&PlaneReductionConfig>,
    system: &System,
    n_stages: usize,
    collect_deviation_points: bool,
    long_term_mean_inflow_table: &HashMap<EntityId, f64>,
) -> Result<PerHydroFit, SddpError> {
    let config_entry = config_map.get(&hydro.id).copied();

    let source = determine_source(hydro, config_entry)?;

    if !hydro.has_turbine_capacity() {
        match source {
            ProductionModelSource::ComputedFromGeometry => {
                validate_computed_prerequisites(hydro, geometry_map)?;
                return Ok(no_turbine_capacity_fit(hydro.id, n_stages));
            }
            ProductionModelSource::PrecomputedHyperplanes
                if !has_hyperplane_rows(hyperplane_map, hydro.id) =>
            {
                return Ok(no_turbine_capacity_fit(hydro.id, n_stages));
            }
            _ => {}
        }
    }

    let mut export_rows: Vec<FphaHyperplaneRow> = Vec::new();
    let mut fpha_deviations: Vec<FphaDeviationDiagnostic> = Vec::new();
    let mut deviation_point_rows: Vec<FphaDeviationPointRow> = Vec::new();

    // Fit once per distinct `SelectionMode` entry (via the dedup in
    // `fit_computed_planes_per_stage`); each study stage carries the plane set of
    // the entry covering it.
    let computed_planes_per_stage: Option<Vec<Vec<FphaPlane>>> =
        if source == ProductionModelSource::ComputedFromGeometry {
            // Drives the lateral-secant `S_max = 2·long-term mean inflow`; a
            // history-less hydro yields `0.0`, falling back to `2 × max_turbined`.
            let long_term_mean_inflow_m3s = long_term_mean_inflow_table
                .get(&hydro.id)
                .copied()
                .unwrap_or(0.0);
            let per_stage = fit_computed_planes_per_stage(
                hydro,
                config_entry,
                geometry_map,
                families_map,
                reference_volume_fractions,
                system,
                study_stages,
                long_term_mean_inflow_m3s,
                plane_reduction,
                collect_deviation_points,
                &mut export_rows,
                &mut fpha_deviations,
                &mut deviation_point_rows,
            )?;
            Some(per_stage)
        } else {
            None
        };

    let mut stage_models: Vec<ResolvedProductionModel> = Vec::with_capacity(n_stages);
    for (stage_idx, stage) in study_stages.iter().enumerate() {
        let cached_stage_planes = computed_planes_per_stage
            .as_ref()
            .map(|per_stage| per_stage[stage_idx].as_slice());
        let model = resolve_stage_model(
            hydro,
            stage,
            config_entry,
            source,
            hyperplane_map,
            cached_stage_planes,
            Some(override_table),
        )?;
        stage_models.push(model);
    }

    Ok(PerHydroFit {
        stage_models,
        provenance: (hydro.id, source),
        export_rows,
        fpha_deviations,
        deviation_point_rows,
    })
}

/// O(1) lookup: `hydro_id` → geometry rows, sorted by volume.
fn build_geometry_map(
    geometry_rows: &[HydroGeometryRow],
) -> HashMap<EntityId, Vec<&HydroGeometryRow>> {
    let mut geometry_map: HashMap<EntityId, Vec<&HydroGeometryRow>> = HashMap::new();
    for row in geometry_rows {
        geometry_map.entry(row.hydro_id).or_default().push(row);
    }
    for rows in geometry_map.values_mut() {
        rows.sort_by(|a, b| a.volume_hm3.total_cmp(&b.volume_hm3));
    }
    geometry_map
}

/// Resolve a plant's downstream reservoir level (m): the forebay surface
/// elevation of the plant `hydro` discharges into, at that plant's stage
/// reference volume. `None` when there is no downstream plant, or the downstream
/// plant is absent / has no geometry.
///
/// `stage_pos` is keyed to match the reference-volume resolver and
/// `build_energy_conversion_set`. The resolver already holds `v_ref` resolved to
/// absolute hm³, so it is consumed verbatim — do NOT re-apply the
/// `v_min + fraction·(..)` span formula here.
fn resolve_downstream_level(
    hydro: &Hydro,
    stage_pos: StudyPos,
    system: &System,
    geometry_map: &HashMap<EntityId, Vec<&HydroGeometryRow>>,
    reference_volume_fractions: &HydroReferenceVolumeFractions,
) -> Option<f64> {
    let downstream_id = hydro.downstream_id?;
    let downstream = system.hydro(downstream_id)?;

    let geo_refs = geometry_map.get(&downstream_id)?;
    if geo_refs.is_empty() {
        return None;
    }
    let geo_rows: Vec<HydroGeometryRow> = geo_refs.iter().map(|r| (*r).clone()).collect();
    let forebay = ForebayTable::new(&geo_rows, &downstream.name).ok()?;

    let v_ref = reference_volume_fractions.get(downstream_id, stage_pos);

    Some(forebay.height(v_ref))
}

/// Long-term mean natural inflow (m³/s) for every hydro, keyed by canonical
/// [`EntityId`], from ONE sequential pass over `System::inflow_history()`.
/// Feeds the lateral-secant `S_max = 2·mean`; a hydro absent from the table
/// (no history) falls back to `0.0` at the lookup site, mapping to the
/// `2 × max_turbined` fallback in `resolve_s_max`.
///
/// # Determinism
///
/// One sequential pass in `inflow_history()`'s stored canonical order; each
/// hydro's `(sum, count)` accumulates in that same encounter order regardless
/// of hydro declaration order, so every mean stays bit-identical to a
/// per-hydro filtered scan. A partitioned-then-reduced parallel accumulator
/// would reorder the adds and break bit-determinism.
fn build_long_term_mean_inflow_table(system: &System) -> HashMap<EntityId, f64> {
    let mut acc: HashMap<EntityId, (f64, u64)> = HashMap::new();
    for row in system.inflow_history() {
        let entry = acc.entry(row.hydro_id).or_insert((0.0, 0));
        entry.0 += row.value_m3s;
        entry.1 += 1;
    }
    acc.into_iter()
        .map(|(hydro_id, (sum, count))| {
            #[allow(clippy::cast_precision_loss)]
            let mean = sum / (count as f64);
            (hydro_id, mean)
        })
        .collect()
}

/// Reference oracle for [`build_long_term_mean_inflow_table`]: the retired
/// per-hydro filtered scan, kept to prove the one-pass batch table's mean is
/// bit-identical to a direct per-hydro `inflow_history()` scan.
#[cfg(test)]
fn long_term_mean_inflow_reference(system: &System, hydro_id: EntityId) -> f64 {
    let mut sum = 0.0_f64;
    let mut count = 0_u64;
    for row in system.inflow_history() {
        if row.hydro_id == hydro_id {
            sum += row.value_m3s;
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        #[allow(clippy::cast_precision_loss)]
        let n = count as f64;
        sum / n
    }
}

/// Fit FPHA planes for one `SelectionMode` entry's `FphaColumnLayout`, after
/// validating prerequisites (tailrace, losses, efficiency present).
///
/// `tailrace_source` changes only the `tailrace_level` the secant reads — never
/// the hull/α/secant procedure. `plane_reduction` `None` skips the merge pass.
/// `entry_level_bits` seeds the `Distance` reduction arm (with `hydro.id.0`).
fn fit_planes_for_hydro(
    hydro: &Hydro,
    config: &FphaColumnLayout,
    geometry_map: &HashMap<EntityId, Vec<&HydroGeometryRow>>,
    long_term_mean_inflow_m3s: f64,
    tailrace_source: TailraceSource,
    plane_reduction: Option<&PlaneReductionConfig>,
    entry_level_bits: u64,
    collect_deviation_points: bool,
) -> Result<FphaFitResult, SddpError> {
    validate_computed_prerequisites(hydro, geometry_map)?;

    let geo_rows_owned: Vec<HydroGeometryRow> = geometry_map
        .get(&hydro.id)
        .map_or(&[][..], Vec::as_slice)
        .iter()
        .map(|r| (*r).clone())
        .collect();

    Ok(fit_fpha_planes(
        &geo_rows_owned,
        hydro,
        config,
        long_term_mean_inflow_m3s,
        tailrace_source,
        plane_reduction,
        hydro.id.0,
        entry_level_bits,
        collect_deviation_points,
    )?)
}

/// Resolve the [`TailraceSource`] for one (hydro, stage) pair: exact backwater
/// families coupled to the resolved downstream level when the plant is in
/// `families_map`, else the entity [`cobre_core::TailraceModel`] fallback.
fn resolve_tailrace_source(
    hydro: &Hydro,
    stage_pos: StudyPos,
    families_map: &HashMap<EntityId, TailraceFamilies>,
    geometry_map: &HashMap<EntityId, Vec<&HydroGeometryRow>>,
    reference_volume_fractions: &HydroReferenceVolumeFractions,
    system: &System,
) -> TailraceSource {
    if let Some(families) = families_map.get(&hydro.id) {
        let downstream_level_m = resolve_downstream_level(
            hydro,
            stage_pos,
            system,
            geometry_map,
            reference_volume_fractions,
        );
        TailraceSource::Families {
            families: families.clone(),
            downstream_level_m,
        }
    } else {
        TailraceSource::Entity(hydro.tailrace.clone())
    }
}

/// One per-hydro dedup-cache entry: the `(config, downstream-level bits)` key, the
/// fitted plane set, and that fit's per-sampled-point deviation points. The
/// `Option<u64>` is the resolved `downstream_level_m` as `f64::to_bits` (`None` for
/// the entity fallback / unresolved level).
///
/// The deviation points are cached beside the planes so a dedup'd stage reuses the
/// same per-point rows — they are a pure function of (config + level).
type FittedCacheEntry<'a> = (
    (&'a FphaColumnLayout, Option<u64>),
    Vec<FphaPlane>,
    Vec<FphaDeviationPoint>,
);

/// Fit computed-FPHA planes per study stage for one hydro, deduplicating
/// identical `SelectionMode` entries. Returns one plane set per study stage (in
/// `study_stages` order) and appends export rows, per-distinct-fit diagnostics,
/// and (opt-in) per-point rows in canonical `(stage_id, …)` order.
///
/// ## Contract — the dedup key MUST include the downstream level
///
/// The key is `(FphaColumnLayout, Option<u64>)`: the config paired with the
/// resolved `downstream_level_m` as `f64::to_bits`. Keying on the
/// `FphaColumnLayout` ALONE is the wrong-but-compiling alternative — two stages
/// with the same config but different backwater levels would collapse to one fit,
/// silently using one stage's tailrace for the other. `to_bits` makes the match
/// exact and order-invariant (no `f64` `PartialEq`).
///
/// # Errors
///
/// - A stage mapping to no `fpha_config` (coverage gap) → [`SddpError::Validation`].
/// - Fitting errors propagate via [`fit_planes_for_hydro`], naming the hydro.
// Rationale: each argument is a distinct already-resolved input; a context struct
// would only relocate the same fields.
#[allow(clippy::too_many_arguments)]
fn fit_computed_planes_per_stage(
    hydro: &Hydro,
    config_entry: Option<&ProductionModelConfig>,
    geometry_map: &HashMap<EntityId, Vec<&HydroGeometryRow>>,
    families_map: &HashMap<EntityId, TailraceFamilies>,
    reference_volume_fractions: &HydroReferenceVolumeFractions,
    system: &System,
    study_stages: &[&Stage],
    long_term_mean_inflow_m3s: f64,
    plane_reduction: Option<&PlaneReductionConfig>,
    collect_deviation_points: bool,
    export_rows: &mut Vec<FphaHyperplaneRow>,
    diagnostics: &mut Vec<FphaDeviationDiagnostic>,
    deviation_point_rows: &mut Vec<FphaDeviationPointRow>,
) -> Result<Vec<Vec<FphaPlane>>, SddpError> {
    // Linear scan over `PartialEq` rather than a hash map: `FphaColumnLayout` holds
    // f64 fields (no `Eq`/`Hash`) and the per-hydro entry count is small.
    let mut fitted: Vec<FittedCacheEntry> = Vec::new();
    let mut per_stage: Vec<Vec<FphaPlane>> = Vec::with_capacity(study_stages.len());

    for (stage_pos, stage) in study_stages.iter().enumerate() {
        let config = config_entry
            .and_then(|c| find_fpha_config_for_stage(c, stage))
            .ok_or_else(|| {
                SddpError::Validation(format!(
                    "hydro {} (id={}) has source: \"computed\" but no FphaColumnLayout \
                     covers stage {} in hydro_production_models.json",
                    hydro.name, hydro.id.0, stage.id
                ))
            })?;

        let tailrace_source = resolve_tailrace_source(
            hydro,
            StudyPos(stage_pos),
            families_map,
            geometry_map,
            reference_volume_fractions,
            system,
        );
        let level_bits = match &tailrace_source {
            TailraceSource::Families {
                downstream_level_m, ..
            } => downstream_level_m.map(f64::to_bits),
            TailraceSource::Entity(_) => None,
        };

        let key = (config, level_bits);
        let (planes, deviation_points) =
            if let Some((_, planes, points)) = fitted.iter().find(|(k, _, _)| *k == key) {
                (planes.clone(), points.clone())
            } else {
                let fit_result = fit_planes_for_hydro(
                    hydro,
                    config,
                    geometry_map,
                    long_term_mean_inflow_m3s,
                    tailrace_source,
                    plane_reduction,
                    // Level bits double as the `Distance`-arm seed; `None` → 0.
                    level_bits.unwrap_or(0),
                    collect_deviation_points,
                )?;
                // One push per distinct fit (the caller applies the warn threshold).
                diagnostics.push(FphaDeviationDiagnostic {
                    hydro_name: hydro.name.clone(),
                    stage_id: stage.id,
                    deviation: fit_result.deviation,
                });
                fitted.push((
                    key,
                    fit_result.planes.clone(),
                    fit_result.deviation_points.clone(),
                ));
                (fit_result.planes, fit_result.deviation_points)
            };

        for point in &deviation_points {
            deviation_point_rows.push(FphaDeviationPointRow {
                hydro_id: hydro.id,
                stage_id: Some(stage.id),
                v: point.v,
                q: point.q,
                fph_exact: point.fph_exact,
                fpha_fitted: point.fpha_fitted,
                deviation: point.deviation,
                relative: point.relative_to_peak,
            });
        }

        for (plane_id, plane) in planes.iter().enumerate() {
            // The in-memory plane already carries the α-scaled coefficients;
            // exporting with `kappa = 1.0` makes the precomputed read-back
            // (`intercept = gamma_0 * kappa`) reproduce them verbatim. `kappa` MUST
            // stay 1.0 — any other value re-scales already-corrected coefficients on
            // read-back (double-correction).
            //
            // Rationale: plane counts are bounded by max_planes_per_hydro (default
            // <= 30), far below i32::MAX, so truncation and wrap are unreachable.
            #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
            export_rows.push(FphaHyperplaneRow {
                hydro_id: hydro.id,
                stage_id: Some(stage.id),
                plane_id: plane_id as i32,
                gamma_0: plane.intercept,
                gamma_v: plane.gamma_v,
                gamma_q: plane.gamma_q,
                gamma_s: plane.gamma_s,
                kappa: 1.0,
                valid_v_min_hm3: None,
                valid_v_max_hm3: None,
                valid_q_max_m3s: None,
            });
        }

        per_stage.push(planes);
    }

    Ok(per_stage)
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Return `true` if the config entry uses `source: "precomputed"` FPHA in any
/// stage range or season entry, or falls back to a declared `"fpha"` model
/// with no `fpha_config` — the `Seasonal` `default_model` case, which reads
/// `fpha_hyperplanes.parquet` globally and so is precomputed by construction.
fn config_uses_precomputed_fpha(config: &ProductionModelConfig) -> bool {
    selection_entries(config).any(|entry| {
        entry.fpha_config.is_some_and(|f| f.source == "precomputed")
            || (entry.model == "fpha" && entry.fpha_config.is_none())
    })
}

/// Return `true` if the config entry uses `source: "computed"` FPHA in any
/// stage range or season entry.
fn config_uses_computed_fpha(config: &ProductionModelConfig) -> bool {
    selection_entries(config).any(|entry| entry.fpha_config.is_some_and(|f| f.source == "computed"))
}

/// Extract the [`FphaColumnLayout`] that applies to a given stage from a [`ProductionModelConfig`].
///
/// Returns `None` when no stage range or season entry covers the stage, or when
/// the matched entry has no `fpha_config` field.
fn find_fpha_config_for_stage<'a>(
    config: &'a ProductionModelConfig,
    stage: &Stage,
) -> Option<&'a FphaColumnLayout> {
    resolve_stage(config, stage).fpha_config
}

/// Default reference operating volume as a fraction of the `[v_min, v_max]`
/// operating band [dimensionless, in `[0, 1]`], applied when no entry declares a
/// `reference_volume`. The sole owner of this literal: changing it shifts every
/// undeclared plant's resolved reference volume.
pub(crate) const DEFAULT_REFERENCE_VOLUME_FRACTION: f64 = 0.65;

/// Extract the [`ReferenceVolume`] that applies to a given stage from a [`ProductionModelConfig`].
///
/// Mirrors [`find_fpha_config_for_stage`]: walks the selection mode, finds the
/// entry covering `stage`, and returns its `reference_volume`. Returns `None`
/// when no stage range or season entry covers the stage, or when the covering
/// entry has no `reference_volume`.
fn find_reference_volume_for_stage<'a>(
    config: &'a ProductionModelConfig,
    stage: &Stage,
) -> Option<&'a ReferenceVolume> {
    resolve_stage(config, stage).reference_volume
}

/// Resolve a [`ReferenceVolume`] to an absolute storage value (`hm³`) against the
/// plant's `[v_min, v_max]` operating band.
///
/// The percentile and default arms multiply the span `(v_max − v_min)` — never
/// divide — so a degenerate `v_max == v_min` band yields `v_min`, not a `0/0` NaN.
pub(crate) fn resolve_reference_volume_hm3(
    rv: Option<&ReferenceVolume>,
    v_min: f64,
    v_max: f64,
) -> f64 {
    match rv {
        Some(ReferenceVolume::AbsoluteHm3(volume_hm3)) => *volume_hm3,
        Some(ReferenceVolume::Percentile(percentile)) => v_min + percentile * (v_max - v_min),
        None => v_min + DEFAULT_REFERENCE_VOLUME_FRACTION * (v_max - v_min),
    }
}

/// Validate that a hydro with `source: "computed"` has `tailrace`,
/// `hydraulic_losses`, `efficiency`, and at least one geometry row.
///
/// # Policy rationale
///
/// The production-function math could default each missing field (zero tailrace,
/// lossless penstock, 100% efficiency), but requiring all three as `Some` forces
/// the operator to declare them explicitly rather than silently accept partial
/// geometry that yields physically inconsistent envelopes.
///
/// # Errors
///
/// Returns `SddpError::Validation` naming the first missing prerequisite and the hydro.
fn validate_computed_prerequisites(
    hydro: &Hydro,
    geometry_map: &HashMap<EntityId, Vec<&HydroGeometryRow>>,
) -> Result<(), SddpError> {
    let missing = if hydro.tailrace.is_none() {
        Some("tailrace")
    } else if hydro.hydraulic_losses.is_none() {
        Some("hydraulic_losses")
    } else if hydro.efficiency.is_none() {
        Some("efficiency")
    } else if geometry_map.get(&hydro.id).is_none_or(Vec::is_empty) {
        Some("geometry data")
    } else {
        None
    };

    if let Some(missing_item) = missing {
        return Err(SddpError::Validation(format!(
            "hydro {} (id={}) has source: \"computed\" but is missing {}. \
             Computed FPHA fitting requires tailrace, hydraulic_losses, \
             efficiency, and geometry data.",
            hydro.name, hydro.id.0, missing_item
        )));
    }

    Ok(())
}

/// Determine the [`ProductionModelSource`] for one hydro (the high-level
/// classification only), rejecting unsupported cases before any Parquet load.
fn determine_source(
    hydro: &Hydro,
    config_entry: Option<&ProductionModelConfig>,
) -> Result<ProductionModelSource, SddpError> {
    if let Some(config) = config_entry {
        let has_computed = selection_entries(config)
            .any(|entry| entry.fpha_config.is_some_and(|f| f.source == "computed"));
        if has_computed {
            return Ok(ProductionModelSource::ComputedFromGeometry);
        }
        let has_fpha = selection_entries(config).any(|entry| entry.model == "fpha");
        Ok(if has_fpha {
            ProductionModelSource::PrecomputedHyperplanes
        } else {
            ProductionModelSource::DefaultConstant
        })
    } else {
        match &hydro.generation_model {
            HydroGenerationModel::ConstantProductivity | HydroGenerationModel::LinearizedHead => {
                Ok(ProductionModelSource::DefaultConstant)
            }
            HydroGenerationModel::Fpha => Err(SddpError::Validation(format!(
                "hydro {} (id={}) has generation_model: \"fpha\" in hydros.json \
                 but no entry in hydro_production_models.json. \
                 Add an entry with source: \"precomputed\" to specify the hyperplane source.",
                hydro.name, hydro.id.0
            ))),
        }
    }
}

/// Resolve the production model for one (hydro, stage) pair.
///
/// `cached_computed_planes` carries planes fitted once per hydro by the outer
/// loop (when `source == ComputedFromGeometry`), so the fitting pipeline does not
/// re-run per stage.
fn resolve_stage_model(
    hydro: &Hydro,
    stage: &Stage,
    config_entry: Option<&ProductionModelConfig>,
    source: ProductionModelSource,
    hyperplane_map: &HashMap<(EntityId, Option<i32>), Vec<&FphaHyperplaneRow>>,
    cached_computed_planes: Option<&[FphaPlane]>,
    productivity_override: Option<&HydroEnergyProductivityOverride>,
) -> Result<ResolvedProductionModel, SddpError> {
    // `cobre_io::validation::productivity_resolution` rejects both-JSON-and-parquet
    // at load time, so this override lookup never silently masks a JSON value.
    let parquet_productivity =
        productivity_override.and_then(|o| o.equivalent_productivity(hydro.id, StageId(stage.id)));

    if let Some(config) = config_entry {
        let model_info = find_model_for_stage(config, stage);

        if model_info.as_ref().map(|(name, _)| name.as_str()) == Some("fpha") {
            if source == ProductionModelSource::ComputedFromGeometry {
                let planes = cached_computed_planes
                    .ok_or_else(|| {
                        SddpError::Validation(format!(
                            "hydro {} (id={}) is ComputedFromGeometry but no cached planes \
                             were provided to resolve_stage_model",
                            hydro.name, hydro.id.0
                        ))
                    })?
                    .to_vec();
                Ok(ResolvedProductionModel::Fpha { planes })
            } else {
                build_fpha_model(hydro, stage, source, hyperplane_map)
            }
        } else {
            // Resolution order (matches build_energy_conversion_set): parquet
            // override first, then JSON. The validator guarantees exactly one
            // source supplies the value, so a `None` here is a validator gap.
            let productivity = parquet_productivity
                .or_else(|| model_info.and_then(|(_, p)| p))
                .unwrap_or_else(|| {
                    debug_assert!(
                        false,
                        "non-FPHA {}/{} reached resolve_stage_model with productivity=None; \
                         see cobre_io::validation::productivity_resolution",
                        hydro.name, stage.id
                    );
                    0.0
                });
            Ok(ResolvedProductionModel::ConstantProductivity { productivity })
        }
    } else {
        // No JSON entry: the parquet override is the supplier; the sentinel is
        // unreachable in production (the validator rejects this case at load time).
        let productivity = parquet_productivity.unwrap_or_else(|| {
            debug_assert!(
                false,
                "non-FPHA {}/{} reached resolve_stage_model with productivity=None; \
                 see cobre_io::validation::productivity_resolution",
                hydro.name, stage.id
            );
            0.0
        });
        Ok(ResolvedProductionModel::ConstantProductivity { productivity })
    }
}

/// Find the model name and optional productivity override for a given stage.
///
/// Returns `None` when the config has no entry covering the given stage (gap in coverage).
/// For `StageRanges`, the match is `start_stage_id <= stage.id <= end_stage_id`.
/// For `Seasonal`, the match is by `season_id == stage.season_id`.
fn find_model_for_stage(
    config: &ProductionModelConfig,
    stage: &Stage,
) -> Option<(String, Option<f64>)> {
    let resolution = resolve_stage(config, stage);
    resolution
        .model
        .map(|model| (model.to_string(), resolution.productivity))
}

// ── Canonical per-stage resolver ─────────────────────────────────────────────

/// One `SelectionMode` entry's borrowed `(model, fpha_config)` view, yielded by
/// [`selection_entries`].
struct SelectionEntry<'a> {
    model: &'a str,
    fpha_config: Option<&'a FphaColumnLayout>,
}

/// Enum-dispatched iterator backing [`selection_entries`] over the two
/// `SelectionMode` arms, avoiding a `Box<dyn Iterator>`.
enum SelectionEntries<'a> {
    StageRanges(std::slice::Iter<'a, StageRange>),
    Seasonal {
        seasons: std::slice::Iter<'a, SeasonConfig>,
        default_model: Option<&'a str>,
    },
}

impl<'a> Iterator for SelectionEntries<'a> {
    type Item = SelectionEntry<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::StageRanges(ranges) => ranges.next().map(|range| SelectionEntry {
                model: range.model.as_str(),
                fpha_config: range.fpha_config.as_ref(),
            }),
            Self::Seasonal {
                seasons,
                default_model,
            } => seasons
                .next()
                .map(|season| SelectionEntry {
                    model: season.model.as_str(),
                    fpha_config: season.fpha_config.as_ref(),
                })
                .or_else(|| {
                    default_model.take().map(|model| SelectionEntry {
                        model,
                        fpha_config: None,
                    })
                }),
        }
    }
}

/// Enumerate every declared `SelectionMode` entry, plus — for `Seasonal` — one
/// synthetic `default_model` entry with `fpha_config: None`: the ONLY OTHER
/// config-level `SelectionMode` match in this module (mirrors
/// [`resolve_stage`]'s per-stage match).
///
/// The synthetic entry makes `default_model` participate in source
/// classification alongside the declared seasons: folding for `fpha_config`
/// with `source == "precomputed"` OR `model == "fpha"` with no `fpha_config`
/// (precomputed-by-parquet has none to declare) classifies a Seasonal config
/// with `default_model == "fpha"` as using precomputed FPHA even when every
/// listed season is non-FPHA.
fn selection_entries(config: &ProductionModelConfig) -> SelectionEntries<'_> {
    match &config.selection_mode {
        SelectionMode::StageRanges { ranges } => SelectionEntries::StageRanges(ranges.iter()),
        SelectionMode::Seasonal {
            default_model,
            seasons,
        } => SelectionEntries::Seasonal {
            seasons: seasons.iter(),
            default_model: Some(default_model.as_str()),
        },
    }
}

/// Canonical per-stage production-model resolution: the model name, FPHA
/// config, reference volume, and constant-productivity override for one
/// `(hydro, stage)` pair, borrowed from `config` with no owning clones.
struct StageProductionResolution<'a> {
    /// Model name, or `None` for a `StageRanges` gap — the sole case with no
    /// default (a `Seasonal` season miss always resolves to `default_model`).
    model: Option<&'a str>,
    fpha_config: Option<&'a FphaColumnLayout>,
    reference_volume: Option<&'a ReferenceVolume>,
    productivity: Option<f64>,
}

/// Resolve one `(hydro, stage)` production-model entry: the ONLY per-stage
/// `SelectionMode` match in this module.
///
/// `StageRanges`: the first range covering `stage.id`, else every field
/// `None`. `Seasonal`: the season matching `stage.season_id`, else
/// `default_model` with `fpha_config`, `reference_volume`, and `productivity`
/// all `None` — `default_model` is the config's declared fallback, not a
/// coverage gap, so it still resolves a model where a `StageRanges` gap would
/// not.
fn resolve_stage<'a>(
    config: &'a ProductionModelConfig,
    stage: &Stage,
) -> StageProductionResolution<'a> {
    match &config.selection_mode {
        SelectionMode::StageRanges { ranges } => {
            for range in ranges {
                let after_start = stage.id >= range.start_stage_id;
                let before_end = range.end_stage_id.is_none_or(|end| stage.id <= end);
                if after_start && before_end {
                    return StageProductionResolution {
                        model: Some(range.model.as_str()),
                        fpha_config: range.fpha_config.as_ref(),
                        reference_volume: range.reference_volume.as_ref(),
                        productivity: range.productivity_mw_per_m3s,
                    };
                }
            }
            StageProductionResolution {
                model: None,
                fpha_config: None,
                reference_volume: None,
                productivity: None,
            }
        }
        SelectionMode::Seasonal {
            default_model,
            seasons,
        } => {
            if let Some(season_id) = stage.season_id {
                for season in seasons {
                    if i32::try_from(season_id).is_ok_and(|sid| sid == season.season_id) {
                        return StageProductionResolution {
                            model: Some(season.model.as_str()),
                            fpha_config: season.fpha_config.as_ref(),
                            reference_volume: season.reference_volume.as_ref(),
                            productivity: season.productivity_mw_per_m3s,
                        };
                    }
                }
            }
            StageProductionResolution {
                model: Some(default_model.as_str()),
                fpha_config: None,
                reference_volume: None,
                productivity: None,
            }
        }
    }
}

/// Build an `Fpha` `ResolvedProductionModel` for one (hydro, stage) pair.
///
/// Stage-specific rows `(hydro_id, Some(stage.id))` take priority over the global
/// `(hydro_id, None)` all-stage rows. Each `FphaPlane` intercept is the pre-scaled
/// `gamma_0 * kappa`. At least one row must have `gamma_q > 0`, otherwise generation
/// would not depend on turbined flow.
fn build_fpha_model(
    hydro: &Hydro,
    stage: &Stage,
    _source: ProductionModelSource,
    hyperplane_map: &HashMap<(EntityId, Option<i32>), Vec<&FphaHyperplaneRow>>,
) -> Result<ResolvedProductionModel, SddpError> {
    let rows: &[&FphaHyperplaneRow] = hyperplane_map
        .get(&(hydro.id, Some(stage.id)))
        .or_else(|| hyperplane_map.get(&(hydro.id, None)))
        .ok_or_else(|| {
            SddpError::Validation(format!(
                "hydro {} (id={}) is configured as FPHA but has no hyperplane rows \
             in fpha_hyperplanes.parquet for stage {} (and no global all-stage rows).",
                hydro.name, hydro.id.0, stage.id
            ))
        })?;

    if rows.is_empty() {
        return Err(SddpError::Validation(format!(
            "hydro {} (id={}) has zero hyperplane rows for stage {}.",
            hydro.name, hydro.id.0, stage.id
        )));
    }

    let mut planes: Vec<FphaPlane> = Vec::with_capacity(rows.len());
    for row in rows {
        validate_hyperplane_row(hydro, stage, row)?;
        planes.push(FphaPlane {
            intercept: row.gamma_0 * row.kappa,
            gamma_v: row.gamma_v,
            gamma_q: row.gamma_q,
            gamma_s: row.gamma_s,
        });
    }

    if !rows.iter().any(|row| row.gamma_q > 0.0) {
        return Err(SddpError::Validation(format!(
            "hydro {} (id={}) stage {}: no hyperplane has gamma_q > 0, so generation \
             would not depend on turbined flow",
            hydro.name, hydro.id.0, stage.id
        )));
    }

    Ok(ResolvedProductionModel::Fpha { planes })
}

/// Validate the physical constraints for one `FphaHyperplaneRow`.
///
/// Returns `Err(SddpError::Validation(...))` when any constraint is violated.
///
/// Constraints:
///
/// - `gamma_v >= 0` — higher storage must not decrease generation; zero is valid
///   for constant-head plants where head does not depend on volume
/// - `gamma_s <= 0` — spillage reduces generation
/// - `gamma_q >= 0` — more turbined flow must not decrease generation; zero is valid
///   where the capacity ceiling flattens the surface
/// - `kappa ∈ (0, 1]` — correction factor range
fn validate_hyperplane_row(
    hydro: &Hydro,
    stage: &Stage,
    row: &FphaHyperplaneRow,
) -> Result<(), SddpError> {
    let ctx = format!(
        "hydro {} (id={}) plane {} stage {}",
        hydro.name, hydro.id.0, row.plane_id, stage.id
    );

    if row.gamma_v < 0.0 {
        return Err(SddpError::Validation(format!(
            "{ctx}: gamma_v must be >= 0 (higher storage must not decrease generation; \
             zero is valid for constant-head plants), got gamma_v = {}",
            row.gamma_v
        )));
    }

    if row.gamma_s > 0.0 {
        return Err(SddpError::Validation(format!(
            "{ctx}: gamma_s must be <= 0 (spillage reduces generation), \
             got gamma_s = {}",
            row.gamma_s
        )));
    }

    if row.gamma_q < 0.0 {
        return Err(SddpError::Validation(format!(
            "{ctx}: gamma_q must be >= 0 (more turbined flow must not decrease generation), \
             got gamma_q = {}",
            row.gamma_q
        )));
    }

    if row.kappa <= 0.0 || row.kappa > 1.0 {
        return Err(SddpError::Validation(format!(
            "{ctx}: kappa must be in (0, 1] (correction factor range), \
             got kappa = {}",
            row.kappa
        )));
    }

    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests;
