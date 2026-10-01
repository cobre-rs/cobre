//! Resolves the stage-LP builder's study inputs once, in setup:
//! [`resolve_lp_build_inputs`] is the single derivation of every
//! [`LpBuildInputs`] field.

use std::collections::{BTreeMap, HashMap, HashSet};

#[cfg(any(test, feature = "test-support"))]
use cobre_core::Stage;
use cobre_core::scenario::LoadModel;
use cobre_core::{EntityId, Hydro, ResolvedBounds, System};
use cobre_io::StageIdResolver;
#[cfg(any(test, feature = "test-support"))]
use cobre_stochastic::normal::precompute::PrecomputedNormal;
#[cfg(any(test, feature = "test-support"))]
use cobre_stochastic::par::precompute::PrecomputedPar;

use crate::block_clock::BlockClock;
#[cfg(any(test, feature = "test-support"))]
use crate::error::SddpError;
#[cfg(any(test, feature = "test-support"))]
use crate::hydro_models::EvaporationModelSet;
use crate::hydro_models::{ProductionModelSet, ResolvedProductionModel};
#[cfg(any(test, feature = "test-support"))]
use crate::inflow_method::InflowNonNegativityMethod;
use crate::lp::builder::LpBuildInputs;
#[cfg(any(test, feature = "test-support"))]
use crate::lp::builder::{StageTemplates, build_stage_templates};
use crate::lp::indexer::{EntityPositions, HydroCellIndex, StudyDimensions};
use crate::resolved_parameters::ResolvedParameters;
#[cfg(any(test, feature = "test-support"))]
use crate::time_value::DeliveryCalendar;
use crate::time_value::TimeValue;

/// Precompute the per-stage minimum target-storage trajectory `V_target[t]` for
/// every filling hydro, keyed `(hydro_idx, stage_id) → V_target` \[hm³\].
///
/// Computed ONCE here, where the full per-stage ζ·rate schedule is available (a
/// per-stage row-fill helper sees one stage and cannot reconstruct the fold). With
/// `L = entry_stage_id − 1` the last Filling stage, anchored on the dead volume and
/// folded backward:
///
/// ```text
/// V_target[L] = min_storage_hm3                          (at L's stage_idx)
/// V_target[t] = min( V_target[t+1] − ζ_{t+1}·rate[t+1], min_storage_hm3 )
/// ```
///
/// `ζ_t = stage_zetas[stage_idx]`; `rate`/`min_storage` are
/// the RESOLVED per-stage bounds. The clip at `min_storage` enforces that no floor
/// exceeds the dead volume — dropping it would let an over-provisioned schedule
/// demand a floor ABOVE the dead volume — the forbidden alternative.
/// The fold runs on the UNCLIPPED running value, clipping each stored `V_target[t]`
/// independently to mirror the closed form.
///
/// Hydros are iterated in canonical slot order into a `BTreeMap`, so the result is
/// declaration-order-invariant; a non-filling system yields an empty map.
pub(crate) fn build_filling_v_target(
    hydros: &[Hydro],
    bounds: &ResolvedBounds,
    stage_zetas: &[f64],
    stage_id_to_idx: &HashMap<i32, usize>,
) -> BTreeMap<(usize, i32), f64> {
    let mut v_target: BTreeMap<(usize, i32), f64> = BTreeMap::new();
    for (h_idx, hydro) in hydros.iter().enumerate() {
        let (Some(filling), Some(entry)) = (hydro.filling.as_ref(), hydro.entry_stage_id) else {
            continue;
        };
        let start = filling.start_stage_id;
        let last = entry - 1;
        // Guard a hypothetical inverted config (`start < entry` is validated
        // upstream) into an empty trajectory rather than a malformed loop.
        if last < start {
            continue;
        }
        let Some(&last_idx) = stage_id_to_idx.get(&last) else {
            continue;
        };
        let min_storage_at_last = bounds.hydro_bounds(h_idx, last_idx).min_storage_hm3;
        v_target.insert((h_idx, last), min_storage_at_last);
        let mut running = min_storage_at_last;
        let mut t = last;
        while t > start {
            if let Some(&t_idx) = stage_id_to_idx.get(&t) {
                let zeta_t = stage_zetas[t_idx];
                let rate_t = bounds.hydro_bounds(h_idx, t_idx).filling_min_rate_m3s;
                running -= zeta_t * rate_t;
            }
            let prev = t - 1;
            if let Some(&prev_idx) = stage_id_to_idx.get(&prev) {
                let min_storage_prev = bounds.hydro_bounds(h_idx, prev_idx).min_storage_hm3;
                v_target.insert((h_idx, prev), running.min(min_storage_prev));
            }
            t = prev;
        }
    }
    v_target
}

/// [`build_filling_v_target`]'s stage-clock/id-index inputs, resolved from
/// `system`'s own study stages.
fn resolve_filling_v_target(system: &System) -> BTreeMap<(usize, i32), f64> {
    let study_stages: Vec<_> = system.stages().iter().filter(|s| s.id >= 0).collect();
    let study_stage_ids: Vec<i32> = study_stages.iter().map(|s| s.id).collect();
    let stage_resolver = StageIdResolver::from_study_stage_ids(&study_stage_ids);
    let stage_zetas: Vec<f64> = study_stages
        .iter()
        .map(|s| BlockClock::new(s).zeta())
        .collect();
    build_filling_v_target(
        system.hydros(),
        system.bounds(),
        &stage_zetas,
        stage_resolver.index_map(),
    )
}

/// Declared load-balance rows for buses outside `load_bus_ids`. A member bus
/// has no model here: its template row is `0` and every solve patches it
/// through `StageSolvePrep`'s load patch.
fn resolve_deterministic_load_models(system: &System, load_bus_ids: &[EntityId]) -> Vec<LoadModel> {
    let member_ids: HashSet<EntityId> = load_bus_ids.iter().copied().collect();
    system
        .load_models()
        .iter()
        .filter(|lm| !member_ids.contains(&lm.bus_id))
        .cloned()
        .collect()
}

/// `load_bus_ids`' bus-slice positions, in `load_bus_ids`' own (`EntityId`-
/// sorted) order.
fn resolve_load_bus_indices(positions: &EntityPositions, load_bus_ids: &[EntityId]) -> Vec<usize> {
    load_bus_ids
        .iter()
        .filter_map(|id| positions.bus(*id))
        .collect()
}

/// Target hydro ID → system indices of hydros diverting to it, in
/// hydro-slice order.
fn resolve_diversion_upstream(hydros: &[Hydro]) -> HashMap<EntityId, Vec<usize>> {
    let mut diversion_upstream: HashMap<EntityId, Vec<usize>> = HashMap::new();
    for (h_idx, hydro) in hydros.iter().enumerate() {
        if let Some(ref div) = hydro.diversion {
            diversion_upstream
                .entry(div.downstream_id)
                .or_default()
                .push(h_idx);
        }
    }
    diversion_upstream
}

/// Per-stage hydro productivities (MW per m³/s), over `(0..n_study) ×
/// (0..n_hydros)`; FPHA hydros carry `0.0`.
fn resolve_hydro_productivities_per_stage(
    n_study: usize,
    n_hydros: usize,
    production_models: &ProductionModelSet,
) -> Vec<Vec<f64>> {
    (0..n_study)
        .map(|s| {
            (0..n_hydros)
                .map(|h| match production_models.model(h, s) {
                    ResolvedProductionModel::ConstantProductivity { productivity } => *productivity,
                    ResolvedProductionModel::Fpha { .. } => 0.0,
                })
                .collect()
        })
        .collect()
}

/// Resolve every [`LpBuildInputs`] field once, so the stage-LP builder
/// consumes already-resolved input instead of re-deriving it per stage.
/// `load_bus_ids` is the stochastic owner's load-noise membership list (the
/// caller's single authority — this function resolves no membership of its
/// own). `study_dims`/`time_value`/`hydro_cell_index`/`resolved_parameters`
/// are borrowed through unchanged, from the same `resolve_stage_data` step
/// that built them.
pub(crate) fn resolve_lp_build_inputs<'a>(
    system: &System,
    load_bus_ids: &[EntityId],
    production_models: &ProductionModelSet,
    study_dims: &'a StudyDimensions,
    time_value: &'a TimeValue,
    hydro_cell_index: &'a HydroCellIndex,
    resolved_parameters: &'a ResolvedParameters,
) -> LpBuildInputs<'a> {
    let n_study = system.stages().iter().filter(|s| s.id >= 0).count();
    let positions = EntityPositions::build(system);
    LpBuildInputs {
        load_bus_indices: resolve_load_bus_indices(&positions, load_bus_ids),
        positions,
        filling_v_target: resolve_filling_v_target(system),
        deterministic_load_models: resolve_deterministic_load_models(system, load_bus_ids),
        diversion_upstream: resolve_diversion_upstream(system.hydros()),
        hydro_productivities_per_stage: resolve_hydro_productivities_per_stage(
            n_study,
            system.hydros().len(),
            production_models,
        ),
        study_dims,
        time_value,
        hydro_cell_index,
        resolved_parameters,
    }
}

/// The in-sample load-noise membership list, for a caller with no stochastic
/// context to resolve it from directly ([`build_stage_templates_resolving_layout`]).
#[cfg(any(test, feature = "test-support"))]
pub(crate) fn resolve_in_sample_load_bus_ids(system: &System) -> Vec<EntityId> {
    system.load_noise_member_bus_ids(cobre_core::scenario::SamplingScheme::InSample)
}

/// Test/integration-only convenience wrapper over [`build_stage_templates`]:
/// resolves the state layout and bucket topology from `system`/`par_lp`
/// through the same setup entry point production uses
/// ([`super::resolve_state_and_topology`]), then delegates. Production
/// (`StudySetup`) always threads its own already-resolved
/// `StateSpace`/`per_stage_mask` directly through `build_stage_templates`
/// instead — this wrapper exists so test call sites that build templates from
/// a bare system do not each need to resolve the layout themselves.
///
/// # Errors
///
/// Propagates [`super::resolve_state_and_topology`]'s `LeadTime` fan-out
/// rejection.
#[cfg(any(test, feature = "test-support"))]
pub fn build_stage_templates_resolving_layout(
    system: &System,
    inflow_method: InflowNonNegativityMethod,
    par_lp: &PrecomputedPar,
    normal_lp: &PrecomputedNormal,
    production_models: &ProductionModelSet,
    evaporation_models: &EvaporationModelSet,
    resolved_parameters: &ResolvedParameters,
) -> Result<StageTemplates, SddpError> {
    let calendar = DeliveryCalendar::from_system(system);
    let (topology, layout) =
        super::resolve_state_and_topology(system, &calendar, par_lp, None, false)?;
    let hydro_cell_index = HydroCellIndex::build(system.hydros());
    let stages: Vec<Stage> = system
        .stages()
        .iter()
        .filter(|s| s.id >= 0)
        .cloned()
        .collect();
    let (downstream_par_order, _) = super::resolve_stage_lag_transitions(
        &stages,
        par_lp,
        system.policy_graph().season_map.as_ref(),
    );
    let study_dims = super::build_study_dimensions(
        system,
        inflow_method,
        layout.anticipated_plants.clone(),
        downstream_par_order,
    );
    let time_value = TimeValue::from_system(system, &study_dims.anticipated_plants, calendar);
    let inputs = resolve_lp_build_inputs(
        system,
        &resolve_in_sample_load_bus_ids(system),
        production_models,
        &study_dims,
        &time_value,
        &hydro_cell_index,
        resolved_parameters,
    );
    debug_assert_eq!(
        normal_lp.n_entities(),
        inputs.load_bus_indices.len(),
        "load noise model and LP disagree on the stochastic load buses"
    );
    Ok(build_stage_templates(
        system,
        par_lp,
        production_models,
        evaporation_models,
        &layout.state,
        &topology,
        inputs,
    ))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use chrono::NaiveDate;
    use cobre_core::{
        BoundsCountsSpec, BoundsDefaults, ContractBlockBounds, EntityId, FillingConfig, Hydro,
        HydroBlockBounds, HydroGenerationModel, HydroPenalties, HydroStageBounds, LineBlockBounds,
        PumpingBlockBounds, ResolvedBounds, ThermalBlockBounds, ThermalStageBounds,
    };

    use super::build_filling_v_target;
    use crate::block_clock::M3S_TO_HM3;

    fn default_hydro_bounds() -> HydroStageBounds {
        HydroStageBounds {
            min_storage_hm3: 0.0,
            max_storage_hm3: 200.0,
            filling_min_rate_m3s: 0.0,
            water_withdrawal_m3s: 0.0,
        }
    }

    fn default_hydro_block_bounds() -> HydroBlockBounds {
        HydroBlockBounds {
            max_turbined_m3s: 100.0,
            max_generation_mw: 250.0,
            ..Default::default()
        }
    }

    /// All-zero per-plant [`HydroPenalties`] for fixture hydros.
    fn hydro_penalties_zero() -> HydroPenalties {
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
            inflow_nonnegativity_cost: 0.0,
        }
    }

    // Deliberately non-binding, not an install capacity: the mirror group copies
    // this value and every cell column bound sums against it, so a realistic
    // number caps the cells below what `default_hydro_bounds()`-derived resolved
    // bounds resolve to.
    const FIXTURE_NONBINDING_MAX_TURBINED_M3S: f64 = 1_000_000.0;

    /// A single non-cascade hydro carrying a `FillingConfig`
    /// (`start_stage_id`/`entry_stage_id`), used by the `build_filling_v_target`
    /// fold tests. All other fields are inert.
    fn vtarget_filling_hydro(id: i32, start: i32, entry: i32) -> Hydro {
        let mut hydro = Hydro {
            unit_groups: Vec::new(),
            id: EntityId(id),
            name: format!("H{id}"),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            downstream_id: None,
            travel_time_hours: None,
            entry_stage_id: Some(entry),
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 100.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: FIXTURE_NONBINDING_MAX_TURBINED_M3S,
            specific_productivity_mw_per_m3s_per_m: None,
            min_generation_mw: 0.0,
            max_generation_mw: 1_000_000.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: Some(FillingConfig {
                start_stage_id: start,
                filling_min_rate_m3s: 0.0,
            }),
            penalties: hydro_penalties_zero(),
        };
        hydro.declare_mirror_unit_group(EntityId(1));
        hydro
    }

    /// A `ResolvedBounds` table for one hydro across `n_stages` stages, with every
    /// stage's `min_storage_hm3` and `filling_min_rate_m3s` set to the given values.
    fn vtarget_bounds(n_stages: usize, min_storage: f64, rate: f64) -> ResolvedBounds {
        let mut bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 1,
                n_thermals: 0,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages,
                k_max: 0,
            },
            &BoundsDefaults {
                hydro: default_hydro_bounds(),
                hydro_block: default_hydro_block_bounds(),
                thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
                thermal_block: ThermalBlockBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 0.0,
                },
                line_block: LineBlockBounds {
                    direct_mw: 0.0,
                    reverse_mw: 0.0,
                },
                pumping_block: PumpingBlockBounds {
                    min_flow_m3s: 0.0,
                    max_flow_m3s: 0.0,
                },
                contract_block: ContractBlockBounds {
                    min_mw: 0.0,
                    max_mw: 0.0,
                    price_per_mwh: 0.0,
                },
            },
        );
        for stage_idx in 0..n_stages {
            let hb = bounds.hydro_bounds_mut(0, stage_idx);
            hb.min_storage_hm3 = min_storage;
            hb.filling_min_rate_m3s = rate;
        }
        bounds
    }

    /// Identity `stage_id → stage_idx` map for `n_stages` study stages.
    fn vtarget_id_map(n_stages: usize) -> HashMap<i32, usize> {
        (0..n_stages).map(|i| (i as i32, i)).collect()
    }

    /// The fixture: `start = 2`, `entry = 4`, `min_storage = 60`, per-stage ζ = 2.592
    /// (`720 * M3S_TO_HM3`), `rate = 5`. The backward fold
    /// pins `V_target[3] = 60` (the dead-volume anchor at L = entry − 1) and
    /// `V_target[2] = 60 − 2.592·5 = 47.04` (one stage of minimum accumulation
    /// below the anchor). No `V_target` is emitted at `PreFilling` (ids 0, 1) or
    /// `Operating` (id ≥ 4).
    #[test]
    fn build_filling_v_target_backward_fold_ac_values() {
        let n_stages = 5;
        let hydros = vec![vtarget_filling_hydro(1, 2, 4)];
        let bounds = vtarget_bounds(n_stages, 60.0, 5.0);
        // ζ = 720·M3S_TO_HM3 = 2.592 at every stage.
        let stage_zetas = vec![720.0 * M3S_TO_HM3; n_stages];
        let v_target =
            build_filling_v_target(&hydros, &bounds, &stage_zetas, &vtarget_id_map(n_stages));

        // L = entry − 1 = 3: anchored at the dead volume.
        assert!(
            (v_target[&(0, 3)] - 60.0).abs() < 1e-9,
            "V_target[3] == min_storage == 60.0, got {}",
            v_target[&(0, 3)]
        );
        // Early Filling stage 2: 60 − 2.592·5 = 47.04.
        assert!(
            (v_target[&(0, 2)] - 47.04).abs() < 1e-9,
            "V_target[2] == 60 − 2.592·5 == 47.04, got {}",
            v_target[&(0, 2)]
        );
        // No entry outside the Filling window {2, 3}.
        assert!(
            !v_target.contains_key(&(0, 0)),
            "no V_target at PreFilling id 0"
        );
        assert!(
            !v_target.contains_key(&(0, 1)),
            "no V_target at PreFilling id 1"
        );
        assert!(
            !v_target.contains_key(&(0, 4)),
            "no V_target at Operating id 4"
        );
        assert_eq!(v_target.len(), 2, "exactly one V_target per Filling stage");
    }

    /// Over-provisioned schedule: a fill rate large enough that the backward fold's
    /// unclipped value would exceed `min_storage` is impossible (non-negative rate
    /// only lowers it); the contract is that EVERY `V_target[t] ≤ min_storage`. With
    /// a wide Filling window (ids 1..=5, entry = 6) and a high rate, every earliest
    /// floor sits strictly below the dead volume and the clip never raises one above
    /// it. The clip is verified to hold at every Filling stage.
    #[test]
    fn build_filling_v_target_clips_at_min_storage_when_over_provisioned() {
        let n_stages = 7;
        let min_storage = 30.0;
        let hydros = vec![vtarget_filling_hydro(1, 1, 6)]; // Filling ids {1,2,3,4,5}.
        // A high rate (50 m³/s over ζ = 2.592 ⇒ 129.6 hm³/stage) far exceeds the
        // 30 hm³ dead volume, so the unclipped earliest floors go deeply negative.
        let bounds = vtarget_bounds(n_stages, min_storage, 50.0);
        let stage_zetas = vec![720.0 * M3S_TO_HM3; n_stages];
        let v_target =
            build_filling_v_target(&hydros, &bounds, &stage_zetas, &vtarget_id_map(n_stages));

        for stage_id in 1..=5 {
            let v = v_target[&(0, stage_id)];
            assert!(
                v <= min_storage + 1e-12,
                "V_target[{stage_id}] = {v} must not exceed the dead volume {min_storage}"
            );
        }
        assert!(
            (v_target[&(0, 5)] - min_storage).abs() < 1e-9,
            "V_target[L] == min_storage (the clip is a no-op at the anchor)"
        );
        assert!(
            v_target[&(0, 1)] < v_target[&(0, 5)],
            "earliest floor strictly below the anchor"
        );
    }

    /// A zero fill rate makes the trajectory FLAT: every Filling stage's floor
    /// equals `min_storage` (the design's `rate == 0 ⇒ V_target[t] == V_target[t+1]`
    /// degenerate case). The clip is a no-op throughout.
    #[test]
    fn build_filling_v_target_flat_when_rate_is_zero() {
        let n_stages = 5;
        let hydros = vec![vtarget_filling_hydro(1, 1, 4)]; // Filling ids {1,2,3}.
        let bounds = vtarget_bounds(n_stages, 45.0, 0.0);
        let stage_zetas = vec![720.0 * M3S_TO_HM3; n_stages];
        let v_target =
            build_filling_v_target(&hydros, &bounds, &stage_zetas, &vtarget_id_map(n_stages));
        for stage_id in 1..=3 {
            assert!(
                (v_target[&(0, stage_id)] - 45.0).abs() < 1e-9,
                "flat trajectory: V_target[{stage_id}] == min_storage == 45.0"
            );
        }
    }

    /// A non-filling hydro (no `FillingConfig`) yields an EMPTY map — the
    /// parity-neutrality contract for the precompute itself.
    #[test]
    fn build_filling_v_target_empty_for_non_filling() {
        let n_stages = 3;
        let mut h = vtarget_filling_hydro(1, 1, 2);
        h.filling = None;
        h.entry_stage_id = None;
        let hydros = vec![h];
        let bounds = vtarget_bounds(n_stages, 50.0, 5.0);
        let stage_zetas = vec![720.0 * M3S_TO_HM3; n_stages];
        let v_target =
            build_filling_v_target(&hydros, &bounds, &stage_zetas, &vtarget_id_map(n_stages));
        assert!(
            v_target.is_empty(),
            "non-filling hydro ⇒ empty V_target map"
        );
    }
}
