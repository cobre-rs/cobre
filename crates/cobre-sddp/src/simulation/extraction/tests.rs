#![allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::too_many_lines,
    clippy::cast_precision_loss
)]

use std::collections::HashMap;

use super::{
    EntityCounts, HydroReverseLookup, SimulationHydroResult, SolutionView, StageExtractionSpec,
    accumulate_category_costs, assign_scenarios, extract_anticipated_lanes, extract_contracts,
    extract_generic_violations, extract_pumping_stations, extract_stage_result,
    extract_stub_collections,
};
use cobre_core::{BlockMode, ContractType};

use crate::energy_conversion::EnergyConversionSet;
use crate::horizon_mode::HorizonMode;
use crate::lead_time::{AnticipatedResolution, PointResolution};
use crate::lp::builder::{GenericConstraintRowEntry, StageGeometry, evaporation_slot_count};
use crate::lp::indexer::{
    AnticipatedPlants, BlockRowFamily, FillingTargetLocal, FloorLocal, FphaLocal, HydroCellIndex,
    HydroSys, StateSpace, StudyDimensions,
};
use crate::simulation::types::{ScenarioCategoryCosts, SimulationCostResult};
use crate::test_support;
use crate::test_support::anticipated_plants_at;

// -------------------------------------------------------------------------
// HydroReverseLookup per-stage membership
// -------------------------------------------------------------------------

/// Pins that `build_per_stage` reads each stage's own `fpha_hydro_indices`, not
/// a single global stage-0 list: hydro 1 is FPHA only at stage 1.
#[test]
fn build_per_stage_resolves_stage_varying_fpha_membership() {
    let n_hydros = 2;

    // Stage 0: only hydro 0 is FPHA. Stage 1: hydros 0 and 1 are both FPHA,
    // so hydro 1's FPHA-local slot is 1. The `generation` range is the stage's
    // FPHA-generation column block (one column per FPHA hydro here, n_blks = 1);
    // its base differs per stage but only the membership lists drive the lookup.
    let geom_stage0 = StageGeometry {
        fpha_hydro_indices: vec![HydroSys::new(0)],
        generation: 100..101,
        ..test_support::equipment_free_geometry(&[0]).remove(0)
    };
    let geom_stage1 = StageGeometry {
        fpha_hydro_indices: vec![HydroSys::new(0), HydroSys::new(1)],
        generation: 200..202,
        ..test_support::equipment_free_geometry(&[0]).remove(0)
    };
    let geometry_per_stage = vec![geom_stage0, geom_stage1];
    let hydro_cell_index = test_support::identity_hydro_cell_index(256);

    let hydro_per_stage =
        HydroReverseLookup::build_per_stage(&geometry_per_stage, &hydro_cell_index, n_hydros);
    assert_eq!(hydro_per_stage.len(), 2);

    // Hydro 1 is absent from stage 0's FPHA list and present at FPHA-local slot
    // 1 in stage 1's — the membership genuinely differs by stage.
    assert_eq!(hydro_per_stage[0].fpha[1], None);
    assert_eq!(hydro_per_stage[1].fpha[1], Some(FphaLocal::new(1)));

    // Hydro 0 is FPHA at both stages, always at FPHA-local slot 0.
    assert_eq!(hydro_per_stage[0].fpha[0], Some(FphaLocal::new(0)));
    assert_eq!(hydro_per_stage[1].fpha[0], Some(FphaLocal::new(0)));

    // The stage-1 FPHA-local slot for hydro 1 is the index an extraction read at
    // stage 1 uses into `geometry[1].generation`: slot 1 addresses column
    // `generation.start + 1` (= 201), the column the solved primal occupies. A
    // stage-0 lookup would have reported `None` and skipped this read entirely.
    let stage1_local = hydro_per_stage[1].fpha[1].expect("hydro 1 is FPHA at stage 1");
    assert_eq!(
        geometry_per_stage[1].generation.start + stage1_local.get(),
        201
    );
}

// -------------------------------------------------------------------------
// Filling-slack (σ_fill / σ^{v-}) reverse-lookup membership
// -------------------------------------------------------------------------

/// Pins that the SPARSE `σ_fill` / `σ^{v-}` families resolve through a
/// system→slot reverse map (filling hydro `Some(slot)`, others `None`) and are
/// INDEPENDENT: a hydro can own a `σ^{v-}` column without a `σ_fill` column.
#[test]
fn filling_reverse_lookup_resolves_sparse_membership() {
    let n_hydros = 3;
    // Hydro 2 owns the (single) terminal σ_fill column at this stage; hydro 0
    // owns the σ^{v-} operating-floor column. Hydro 1 owns neither — the common
    // non-filling case. The column ranges are sparse: one column each.
    let geom = StageGeometry {
        filling_target_hydro_indices: vec![HydroSys::new(2)],
        filling_target_col: 500..501,
        filled_min_storage_floor_hydro_indices: vec![HydroSys::new(0)],
        filled_min_storage_floor_col: 600..601,
        ..test_support::equipment_free_geometry(&[0]).remove(0)
    };

    let hydro_cell_index = test_support::identity_hydro_cell_index(256);
    let lookup = HydroReverseLookup::build(&geom, &hydro_cell_index, n_hydros);

    // σ_fill: only hydro 2, at slot 0 ⇒ column 500. Others absent.
    assert_eq!(lookup.filling_target[2], Some(FillingTargetLocal::new(0)));
    assert_eq!(lookup.filling_target[0], None);
    assert_eq!(lookup.filling_target[1], None);
    let target_local = lookup.filling_target[2].expect("hydro 2 owns a σ_fill column");
    assert_eq!(geom.filling_target_col.start + target_local.get(), 500);

    // σ^{v-}: only hydro 0, at slot 0 ⇒ column 600. Others absent. Independent of
    // the σ_fill family — hydro 0 has a floor column but no target column.
    assert_eq!(lookup.filled_min_storage_floor[0], Some(FloorLocal::new(0)));
    assert_eq!(lookup.filled_min_storage_floor[1], None);
    assert_eq!(lookup.filled_min_storage_floor[2], None);
    let floor_local = lookup.filled_min_storage_floor[0].expect("hydro 0 owns a σ^{v-} column");
    assert_eq!(
        geom.filled_min_storage_floor_col.start + floor_local.get(),
        600
    );
}

/// `read_filling_target_slack_primal`/`read_floor_slack_primal` return the solved
/// primal at `start + local.get()` for a present slot and `0.0` for an absent slot
/// (the sparse-family default), for both filling-slack families.
#[test]
fn filling_target_and_floor_slack_primal_present_and_absent() {
    let primal = vec![0.0, 0.0, 7.5, 11.0];
    let geometry = StageGeometry {
        filling_target_col: 2..4,
        filled_min_storage_floor_col: 2..4,
        ..test_support::equipment_free_geometry(&[0]).remove(0)
    };
    // Slot 0 ⇒ column 2 ⇒ 7.5; slot 1 ⇒ column 3 ⇒ 11.0.
    assert_eq!(
        super::read_filling_target_slack_primal(
            &primal,
            &geometry,
            Some(FillingTargetLocal::new(0))
        ),
        7.5
    );
    assert_eq!(
        super::read_filling_target_slack_primal(
            &primal,
            &geometry,
            Some(FillingTargetLocal::new(1))
        ),
        11.0
    );
    // Absent ⇒ 0.0 regardless of what the primal vector holds.
    assert_eq!(
        super::read_filling_target_slack_primal(&primal, &geometry, None),
        0.0
    );

    assert_eq!(
        super::read_floor_slack_primal(&primal, &geometry, Some(FloorLocal::new(0))),
        7.5
    );
    assert_eq!(
        super::read_floor_slack_primal(&primal, &geometry, Some(FloorLocal::new(1))),
        11.0
    );
    assert_eq!(
        super::read_floor_slack_primal(&primal, &geometry, None),
        0.0
    );
}

/// A filling hydro whose `σ_fill` slack BINDS surfaces the non-zero primal in
/// `filling_target_violation_hm3`, while a non-filling hydro stays `0.0`.
#[test]
fn extract_reads_binding_filling_target_slack() {
    let study_dims = test_support::study_dims();
    let state = test_support::state_layout(2, 1);
    let ec = zero_energy_conversion(2, 1);

    let dims = test_support::GeometryDims {
        hydro_count: 2,
        max_par_order: 1,
        n_thermals: 1,
        n_lines: 1,
        n_buses: 1,
        n_blks: 1,
        has_inflow_penalty: false,
        max_deficit_segments: 1,
        n_anticipated: 0,
        lead_stages: 0,
        anticipated_plants: AnticipatedPlants::default(),
    };
    let base = test_support::geometry(&dims, vec![], &[], vec![]);
    let n_cols = base.generation_below_slack.end;

    // Hydro 0 is the lone filling hydro at its terminal Filling stage; hydro 1
    // is non-filling. The σ_fill slack column is appended past the base
    // geometry's last column.
    let sigma_fill = 4.25_f64;
    let geom = StageGeometry {
        filling_target_hydro_indices: vec![HydroSys::new(0)],
        filling_target_col: n_cols..n_cols + 1,
        ..base
    };

    let mut primal = make_primal_2_1([100.0, 200.0], [50.0, 60.0], [90.0, 180.0], 0.0);
    primal.resize(n_cols, 0.0);
    primal.push(sigma_fill);
    let objective_coeffs = vec![0.0; primal.len()];
    let dual = vec![0.0; geometry_row_capacity(&geom)];
    let row_lower = vec![0.0; geometry_row_capacity(&geom)];

    let inflow_m3s_per_hydro = test_support::inflow_m3s_per_hydro_from_primal(&state, &primal, 2);
    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &dual,
            objective: 0.0,
            objective_coeffs: &objective_coeffs,
            row_lower: &row_lower,
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &geom,
            hydro_cell_index: &test_support::identity_hydro_cell_index(256),
            state: &state,
            entity_counts: &make_entity_counts_2_hydros(),
            inflow_m3s_per_hydro: &inflow_m3s_per_hydro,
            block_hours: &[100.0],
            generic_constraint_entries: &[],
            ncs_col_upper: &[],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_slots: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[1.0, 1.0],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            cost_scale_factor: 1_000_000.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0; 2],
            stage_index: 0,
            horizon: &HorizonMode::Finite { num_stages: 1 },
            anticipated_windows: &[],
            study_stage_ids: &[],
        },
        0,
    );

    // n_blks=1 ⇒ one row per (hydro, block), in hydro_ids order.
    assert_eq!(result.hydros.len(), 2);
    // Hydro 0 (filling, slack binds) reports the σ_fill primal — not 0.0.
    assert_eq!(result.hydros[0].filling_target_violation_hm3, sigma_fill);
    // Hydro 1 (non-filling, absent from the family) reports 0.0.
    assert_eq!(result.hydros[1].filling_target_violation_hm3, 0.0);
    // Neither hydro owns a σ^{v-} column here, so both floor fields stay 0.0.
    assert_eq!(result.hydros[0].storage_violation_below_hm3, 0.0);
    assert_eq!(result.hydros[1].storage_violation_below_hm3, 0.0);
}

/// A filling hydro whose `σ^{v-}` operating-floor slack BINDS surfaces the
/// non-zero primal in `storage_violation_below_hm3` on every per-block row
/// (the slack is stage-level), while a hydro absent from the family stays `0.0`.
#[test]
fn extract_reads_binding_filled_min_storage_floor_slack_per_block_branch() {
    let study_dims = test_support::study_dims();
    let state = test_support::state_layout(2, 1);
    let ec = zero_energy_conversion(2, 1);

    let dims = test_support::GeometryDims {
        hydro_count: 2,
        max_par_order: 1,
        n_thermals: 1,
        n_lines: 1,
        n_buses: 1,
        n_blks: 1,
        has_inflow_penalty: false,
        max_deficit_segments: 1,
        n_anticipated: 0,
        lead_stages: 0,
        anticipated_plants: AnticipatedPlants::default(),
    };
    let base = test_support::geometry(&dims, vec![], &[], vec![]);
    let n_cols = base.generation_below_slack.end;

    // Hydro 1 owns the lone σ^{v-} column at this Operating stage; hydro 0 is
    // absent from the floor family. The slack column is appended past the base
    // geometry's last column.
    let sigma_floor = 2.75_f64;
    let geom = StageGeometry {
        filled_min_storage_floor_hydro_indices: vec![HydroSys::new(1)],
        filled_min_storage_floor_col: n_cols..n_cols + 1,
        ..base
    };

    let mut primal = make_primal_2_1([100.0, 200.0], [50.0, 60.0], [90.0, 180.0], 0.0);
    primal.resize(n_cols, 0.0);
    primal[geom.turbine.start] = 1.0; // turbine h0 b0
    primal[geom.turbine.start + 1] = 2.0; // turbine h1 b0
    primal.push(sigma_floor);
    let objective_coeffs = vec![0.0; primal.len()];
    let dual = vec![0.0; geometry_row_capacity(&geom)];
    let row_lower = vec![0.0; geometry_row_capacity(&geom)];

    let inflow_m3s_per_hydro = test_support::inflow_m3s_per_hydro_from_primal(&state, &primal, 2);
    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &dual,
            objective: 0.0,
            objective_coeffs: &objective_coeffs,
            row_lower: &row_lower,
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &geom,
            hydro_cell_index: &test_support::identity_hydro_cell_index(256),
            state: &state,
            entity_counts: &make_entity_counts_2_hydros(),
            inflow_m3s_per_hydro: &inflow_m3s_per_hydro,
            block_hours: &[100.0],
            generic_constraint_entries: &[],
            ncs_col_upper: &[],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_slots: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[1.0, 1.0],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            cost_scale_factor: 1_000_000.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0; 2],
            stage_index: 0,
            horizon: &HorizonMode::Finite { num_stages: 1 },
            anticipated_windows: &[],
            study_stage_ids: &[],
        },
        0,
    );

    // Per-block branch ⇒ one row per (hydro, block); n_blks = 1 ⇒ one row each.
    assert_eq!(result.hydros.len(), 2);
    let hydro0 = result
        .hydros
        .iter()
        .find(|h| h.hydro_id == 10)
        .expect("hydro 10 present");
    let hydro1 = result
        .hydros
        .iter()
        .find(|h| h.hydro_id == 20)
        .expect("hydro 20 present");
    // Hydro 1 (filling-floor, slack binds) reports the σ^{v-} primal.
    assert_eq!(hydro1.storage_violation_below_hm3, sigma_floor);
    // Hydro 0 (absent from the floor family) reports 0.0.
    assert_eq!(hydro0.storage_violation_below_hm3, 0.0);
    // No σ_fill column here, so both target fields stay 0.0.
    assert_eq!(hydro0.filling_target_violation_hm3, 0.0);
    assert_eq!(hydro1.filling_target_violation_hm3, 0.0);
}

// -------------------------------------------------------------------------
// assign_scenarios
// -------------------------------------------------------------------------

#[test]
fn assign_scenarios_uneven_rank0() {
    // 10 % 3 = 1 fat rank → rank 0 gets ceil(10/3) = 4 scenarios
    assert_eq!(assign_scenarios(10, 0, 3), 0..4);
}

#[test]
fn assign_scenarios_uneven_rank2() {
    assert_eq!(assign_scenarios(10, 2, 3), 7..10);
}

#[test]
fn assign_scenarios_single_rank() {
    assert_eq!(assign_scenarios(7, 0, 1), 0..7);
}

#[test]
fn assign_scenarios_uneven_rank1() {
    // Derived from acceptance criteria: rank 1 is a lean rank with 3 scenarios
    // starting at offset 4 (end of rank 0's fat slice).
    assert_eq!(assign_scenarios(10, 1, 3), 4..7);
}

#[test]
fn assign_scenarios_exact_division() {
    // n=9, world=3: 9 % 3 = 0 fat ranks → all lean (3 each)
    assert_eq!(assign_scenarios(9, 0, 3), 0..3);
    assert_eq!(assign_scenarios(9, 1, 3), 3..6);
    assert_eq!(assign_scenarios(9, 2, 3), 6..9);
}

#[test]
fn assign_scenarios_zero_scenarios() {
    assert_eq!(assign_scenarios(0, 0, 1), 0..0);
    assert_eq!(assign_scenarios(0, 0, 4), 0..0);
    assert_eq!(assign_scenarios(0, 3, 4), 0..0);
}

#[test]
fn assign_scenarios_more_ranks_than_scenarios() {
    // n=2, world=5: ranks 0-1 get 1 scenario each; ranks 2-4 get empty.
    assert_eq!(assign_scenarios(2, 0, 5), 0..1);
    assert_eq!(assign_scenarios(2, 1, 5), 1..2);
    assert_eq!(assign_scenarios(2, 2, 5), 2..2);
    assert_eq!(assign_scenarios(2, 3, 5), 2..2);
    assert_eq!(assign_scenarios(2, 4, 5), 2..2);
}

#[test]
fn assign_scenarios_sum_equals_n_scenarios() {
    // Property test: for various (n, world_size) pairs, total assigned = n.
    for (n, world_size) in [(0_u32, 1_usize), (1, 1), (10, 3), (9, 3), (2, 5), (100, 7)] {
        let total: u32 = (0..world_size)
            .map(|rank| {
                let r = assign_scenarios(n, rank, world_size);
                r.end - r.start
            })
            .sum();
        assert_eq!(
            total, n,
            "total assigned {total} != n_scenarios {n} for world_size={world_size}"
        );
    }
}

// -------------------------------------------------------------------------
// extract_stage_result
// -------------------------------------------------------------------------

/// Build a zero-valued [`crate::energy_conversion::EnergyConversionSet`] for tests
/// that do not assert on energy fields.
fn zero_energy_conversion(n_hydros: usize, n_stages: usize) -> EnergyConversionSet {
    use crate::energy_conversion::EnergyConversion;
    let zero_ec = EnergyConversion {
        equivalent_productivity_mw_per_m3s: 0.0,
        reference_volume_hm3: 0.0,
        reference_outflow_m3s: 0.0,
    };
    EnergyConversionSet::new(
        vec![vec![zero_ec; n_stages]; n_hydros],
        vec![vec![0.0_f64; n_stages]; n_hydros],
        &test_support::minimal_hydros(n_hydros),
        n_stages,
    )
}

fn make_entity_counts_2_hydros() -> EntityCounts {
    EntityCounts {
        hydro_ids: vec![10, 20],
        hydro_productivities: vec![1.0, 1.0],
        thermal_ids: vec![1],
        line_ids: vec![5],
        bus_ids: vec![100],
        pumping_station_ids: vec![],
        contract_ids: vec![],
        non_controllable_ids: vec![],
    }
}

/// Build a primal vector for a stage geometry with `hydro_count=2`, `max_par_order=1`.
fn make_primal_2_1(
    storage: [f64; 2],
    lags: [f64; 2],
    storage_in: [f64; 2],
    theta: f64,
) -> Vec<f64> {
    // Layout: storage(2), lags(2), z_inflow(2), storage_in(2), theta(1)
    vec![
        storage[0],
        storage[1],
        lags[0],
        lags[1],
        0.0, // z_inflow[0]
        0.0, // z_inflow[1]
        storage_in[0],
        storage_in[1],
        theta,
    ]
}

/// Row-vector length covering every row family `extract_stage_result` reads
/// unconditionally: water-balance duals and load-balance duals/`row_lower`.
fn geometry_row_capacity(geom: &StageGeometry) -> usize {
    geom.water_balance.end().max(geom.load_balance.end())
}

#[test]
fn extract_costs_has_one_entry_matching_stage_id() {
    // Acceptance criterion: costs contains exactly one entry whose stage_id
    // matches the input stage and whose future_cost == primal[state.theta].
    let study_dims = test_support::study_dims();
    let state = test_support::state_layout(2, 1);
    let ws_start = state.control_region_start();
    let indexer = StageGeometry {
        withdrawal_slack_neg: ws_start..ws_start + 2,
        withdrawal_slack_pos: ws_start + 2..ws_start + 4,
        ..test_support::equipment_free_geometry(&[0]).remove(0)
    };
    let mut primal = make_primal_2_1([100.0, 200.0], [50.0, 60.0], [90.0, 180.0], 999.5);
    primal.resize(indexer.withdrawal_slack_pos.end, 0.0);
    // theta's coefficient must stay 1.0 (the `objective_coeffs.get(theta)`
    // default an empty slice used to give) for `future_cost` below to still
    // equal `primal[theta] * cost_scale_factor`.
    let mut objective_coeffs = vec![0.0; primal.len()];
    objective_coeffs[state.theta] = 1.0;
    let dual = vec![0.0; 4];
    let ec = zero_energy_conversion(2, 1);

    let inflow_m3s_per_hydro = test_support::inflow_m3s_per_hydro_from_primal(&state, &primal, 2);
    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &dual,
            objective: 1500.0,
            objective_coeffs: &objective_coeffs,
            row_lower: &[],
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &indexer,
            hydro_cell_index: &test_support::identity_hydro_cell_index(256),
            state: &state,
            entity_counts: &make_entity_counts_2_hydros(),
            inflow_m3s_per_hydro: &inflow_m3s_per_hydro,
            block_hours: &[],
            generic_constraint_entries: &[],
            ncs_col_upper: &[],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_slots: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[1.0, 1.0],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            cost_scale_factor: 1_000_000.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0; 2],
            stage_index: 0,
            horizon: &HorizonMode::Finite { num_stages: 1 },
            anticipated_windows: &[],
            study_stage_ids: &[],
        },
        3,
    );

    assert_eq!(result.costs.len(), 1);
    assert_eq!(result.costs[0].stage_id, 3);
    // future_cost = primal[theta] * COST_SCALE_FACTOR = 999.5 * 1_000_000 = 999_500_000.0
    assert_eq!(result.costs[0].future_cost, 999_500_000.0);
}

#[test]
fn extract_cost_splits_objective_correctly() {
    // objective = immediate_cost + future_cost
    let study_dims = test_support::study_dims();
    let state = test_support::state_layout(2, 1);
    let ws_start = state.control_region_start();
    let indexer = StageGeometry {
        withdrawal_slack_neg: ws_start..ws_start + 2,
        withdrawal_slack_pos: ws_start + 2..ws_start + 4,
        ..test_support::equipment_free_geometry(&[0]).remove(0)
    };
    let theta_val = 300.0;
    let objective = 800.0;
    let mut primal = make_primal_2_1([0.0; 2], [0.0; 2], [0.0; 2], theta_val);
    primal.resize(indexer.withdrawal_slack_pos.end, 0.0);
    // theta's coefficient must stay 1.0 (the `objective_coeffs.get(theta)`
    // default an empty slice used to give) so objective still splits into
    // immediate_cost + future_cost as asserted below.
    let mut objective_coeffs = vec![0.0; primal.len()];
    objective_coeffs[state.theta] = 1.0;
    let dual = vec![0.0; 4];
    let ec = zero_energy_conversion(2, 1);

    let inflow_m3s_per_hydro = test_support::inflow_m3s_per_hydro_from_primal(&state, &primal, 2);
    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &dual,
            objective,
            objective_coeffs: &objective_coeffs,
            row_lower: &[],
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &indexer,
            hydro_cell_index: &test_support::identity_hydro_cell_index(256),
            state: &state,
            entity_counts: &make_entity_counts_2_hydros(),
            inflow_m3s_per_hydro: &inflow_m3s_per_hydro,
            block_hours: &[],
            generic_constraint_entries: &[],
            ncs_col_upper: &[],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_slots: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[1.0, 1.0],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            cost_scale_factor: 1_000_000.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0; 2],
            stage_index: 0,
            horizon: &HorizonMode::Finite { num_stages: 1 },
            anticipated_windows: &[],
            study_stage_ids: &[],
        },
        0,
    );

    let cost = &result.costs[0];
    // All cost fields are in original units: multiply by COST_SCALE_FACTOR.
    assert_eq!(cost.future_cost, theta_val * 1_000_000.0);
    assert_eq!(cost.immediate_cost, (objective - theta_val) * 1_000_000.0);
    assert_eq!(cost.total_cost, objective * 1_000_000.0);
}

#[test]
fn extract_hydro_storage_values_from_primal() {
    // Hydro h=0: storage[0]=100, storage_in[4]=90
    // Hydro h=1: storage[1]=200, storage_in[5]=180
    let dims = test_support::GeometryDims {
        hydro_count: 2,
        max_par_order: 1,
        n_thermals: 1,
        n_lines: 1,
        n_buses: 1,
        n_blks: 1,
        has_inflow_penalty: false,
        max_deficit_segments: 1,
        n_anticipated: 0,
        lead_stages: 0,
        anticipated_plants: AnticipatedPlants::default(),
    };
    let indexer = test_support::geometry(&dims, vec![], &[], vec![]);
    let study_dims = test_support::study_dims();
    let state = test_support::state_layout(2, 1);
    let mut primal = make_primal_2_1([100.0, 200.0], [50.0, 60.0], [90.0, 180.0], 999.5);
    primal.resize(indexer.generation_below_slack.end, 0.0);
    let objective_coeffs = vec![0.0; primal.len()];
    let dual = vec![0.0; geometry_row_capacity(&indexer)];
    let row_lower = vec![0.0; geometry_row_capacity(&indexer)];
    let ec = zero_energy_conversion(2, 1);

    let inflow_m3s_per_hydro = test_support::inflow_m3s_per_hydro_from_primal(&state, &primal, 2);
    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &dual,
            objective: 1500.0,
            objective_coeffs: &objective_coeffs,
            row_lower: &row_lower,
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &indexer,
            hydro_cell_index: &test_support::identity_hydro_cell_index(256),
            state: &state,
            entity_counts: &make_entity_counts_2_hydros(),
            inflow_m3s_per_hydro: &inflow_m3s_per_hydro,
            block_hours: &[100.0],
            generic_constraint_entries: &[],
            ncs_col_upper: &[],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_slots: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[1.0, 1.0],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            cost_scale_factor: 1_000_000.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0; 2],
            stage_index: 0,
            horizon: &HorizonMode::Finite { num_stages: 1 },
            anticipated_windows: &[],
            study_stage_ids: &[],
        },
        0,
    );

    assert_eq!(result.hydros.len(), 2);
    assert_eq!(result.hydros[0].hydro_id, 10);
    assert_eq!(result.hydros[0].storage_initial_hm3, 90.0);
    assert_eq!(result.hydros[0].storage_final_hm3, 100.0);

    assert_eq!(result.hydros[1].hydro_id, 20);
    assert_eq!(result.hydros[1].storage_initial_hm3, 180.0);
    assert_eq!(result.hydros[1].storage_final_hm3, 200.0);
}

#[test]
fn extract_inflow_lag_values_from_primal() {
    // inflow_lags[2]=50.0 for hydro 0 lag 0, [3]=60.0 for hydro 1 lag 0
    let study_dims = test_support::study_dims();
    let state = test_support::state_layout(2, 1);
    let ws_start = state.control_region_start();
    let indexer = StageGeometry {
        withdrawal_slack_neg: ws_start..ws_start + 2,
        withdrawal_slack_pos: ws_start + 2..ws_start + 4,
        ..test_support::equipment_free_geometry(&[0]).remove(0)
    };
    let mut primal = make_primal_2_1([100.0, 200.0], [50.0, 60.0], [90.0, 180.0], 999.5);
    primal.resize(indexer.withdrawal_slack_pos.end, 0.0);
    let objective_coeffs = vec![0.0; primal.len()];
    let dual = vec![0.0; 4];
    let ec = zero_energy_conversion(2, 1);

    let inflow_m3s_per_hydro = test_support::inflow_m3s_per_hydro_from_primal(&state, &primal, 2);
    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &dual,
            objective: 1500.0,
            objective_coeffs: &objective_coeffs,
            row_lower: &[],
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &indexer,
            hydro_cell_index: &test_support::identity_hydro_cell_index(256),
            state: &state,
            entity_counts: &make_entity_counts_2_hydros(),
            inflow_m3s_per_hydro: &inflow_m3s_per_hydro,
            block_hours: &[],
            generic_constraint_entries: &[],
            ncs_col_upper: &[],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_slots: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[1.0, 1.0],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            cost_scale_factor: 1_000_000.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0; 2],
            stage_index: 0,
            horizon: &HorizonMode::Finite { num_stages: 1 },
            anticipated_windows: &[],
            study_stage_ids: &[],
        },
        0,
    );

    assert_eq!(result.inflow_lags.len(), 2); // 2 hydros × 1 lag each
    // Hydro 10, lag 0 → primal[2] = 50.0
    assert_eq!(result.inflow_lags[0].hydro_id, 10);
    assert_eq!(result.inflow_lags[0].lag_index, 0);
    assert_eq!(result.inflow_lags[0].inflow_m3s, 50.0);
    // Hydro 20, lag 0 → primal[3] = 60.0
    assert_eq!(result.inflow_lags[1].hydro_id, 20);
    assert_eq!(result.inflow_lags[1].lag_index, 0);
    assert_eq!(result.inflow_lags[1].inflow_m3s, 60.0);
}

#[test]
fn extract_no_lags_when_max_par_order_zero() {
    // Stage geometry (N=2, L=0): no inflow_lag columns → empty inflow_lags vec.
    let dims = test_support::GeometryDims {
        hydro_count: 2,
        max_par_order: 0,
        n_thermals: 0,
        n_lines: 0,
        n_buses: 0,
        n_blks: 1,
        has_inflow_penalty: false,
        max_deficit_segments: 1,
        n_anticipated: 0,
        lead_stages: 0,
        anticipated_plants: AnticipatedPlants::default(),
    };
    let indexer = test_support::geometry(&dims, vec![], &[], vec![]);
    let study_dims = test_support::study_dims();
    let state = test_support::state_layout(2, 0);
    // Layout: storage[0..2], z_inflow[2..4], storage_in[4..6], theta=6
    let mut primal = vec![100.0, 200.0, 0.0, 0.0, 90.0, 180.0, 500.0];
    primal.resize(indexer.generation_below_slack.end, 0.0);
    let objective_coeffs = vec![0.0; primal.len()];
    let dual = vec![0.0; geometry_row_capacity(&indexer)];
    let counts = EntityCounts {
        hydro_ids: vec![10, 20],
        hydro_productivities: vec![1.0, 1.0],
        thermal_ids: vec![],
        line_ids: vec![],
        bus_ids: vec![],
        pumping_station_ids: vec![],
        contract_ids: vec![],
        non_controllable_ids: vec![],
    };
    let ec = zero_energy_conversion(2, 1);

    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &dual,
            objective: 600.0,
            objective_coeffs: &objective_coeffs,
            row_lower: &[],
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &indexer,
            hydro_cell_index: &test_support::identity_hydro_cell_index(256),
            state: &state,
            entity_counts: &counts,
            inflow_m3s_per_hydro: &[0.0; 2],
            block_hours: &[],
            generic_constraint_entries: &[],
            ncs_col_upper: &[],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_slots: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[1.0, 1.0],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            cost_scale_factor: 1_000_000.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0; 2],
            stage_index: 0,
            horizon: &HorizonMode::Finite { num_stages: 1 },
            anticipated_windows: &[],
            study_stage_ids: &[],
        },
        2,
    );

    assert!(result.inflow_lags.is_empty());
    assert_eq!(result.hydros[0].incremental_inflow_m3s, 0.0);
}

#[test]
fn extract_stage_id_propagates_to_all_results() {
    let study_dims = test_support::study_dims();
    let state = test_support::state_layout(2, 1);
    let ws_start = state.control_region_start();
    let indexer = StageGeometry {
        withdrawal_slack_neg: ws_start..ws_start + 2,
        withdrawal_slack_pos: ws_start + 2..ws_start + 4,
        ..test_support::equipment_free_geometry(&[0]).remove(0)
    };
    let mut primal = make_primal_2_1([100.0, 200.0], [50.0, 60.0], [90.0, 180.0], 10.0);
    primal.resize(indexer.withdrawal_slack_pos.end, 0.0);
    let objective_coeffs = vec![0.0; primal.len()];
    let dual = vec![0.0; 4];
    let stage_id = 7_u32;
    let ec = zero_energy_conversion(2, 1);

    let inflow_m3s_per_hydro = test_support::inflow_m3s_per_hydro_from_primal(&state, &primal, 2);
    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &dual,
            objective: 110.0,
            objective_coeffs: &objective_coeffs,
            row_lower: &[],
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &indexer,
            hydro_cell_index: &test_support::identity_hydro_cell_index(256),
            state: &state,
            entity_counts: &make_entity_counts_2_hydros(),
            inflow_m3s_per_hydro: &inflow_m3s_per_hydro,
            block_hours: &[],
            generic_constraint_entries: &[],
            ncs_col_upper: &[],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_slots: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[1.0, 1.0],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            cost_scale_factor: 1_000_000.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0; 2],
            stage_index: 0,
            horizon: &HorizonMode::Finite { num_stages: 1 },
            anticipated_windows: &[],
            study_stage_ids: &[],
        },
        stage_id,
    );

    assert_eq!(result.stage_id, stage_id);
    assert_eq!(result.costs[0].stage_id, stage_id);
    assert!(result.hydros.iter().all(|h| h.stage_id == stage_id));
    assert!(result.thermals.iter().all(|t| t.stage_id == stage_id));
    assert!(result.exchanges.iter().all(|e| e.stage_id == stage_id));
    assert!(result.buses.iter().all(|b| b.stage_id == stage_id));
    assert!(result.inflow_lags.iter().all(|l| l.stage_id == stage_id));
}

#[test]
fn extract_equipment_zero_when_indexer_has_no_equipment_ranges() {
    let dims = test_support::GeometryDims {
        hydro_count: 2,
        max_par_order: 1,
        n_thermals: 1,
        n_lines: 1,
        n_buses: 1,
        n_blks: 1,
        has_inflow_penalty: false,
        max_deficit_segments: 1,
        n_anticipated: 0,
        lead_stages: 0,
        anticipated_plants: AnticipatedPlants::default(),
    };
    let indexer = test_support::geometry(&dims, vec![], &[], vec![]);
    let study_dims = test_support::study_dims();
    let state = test_support::state_layout(2, 1);
    let mut primal = make_primal_2_1([0.0; 2], [0.0; 2], [0.0; 2], 0.0);
    primal.resize(indexer.generation_below_slack.end, 0.0);
    let objective_coeffs = vec![0.0; primal.len()];
    let dual = vec![0.0; geometry_row_capacity(&indexer)];
    let row_lower = vec![0.0; geometry_row_capacity(&indexer)];
    let ec = zero_energy_conversion(2, 1);

    let inflow_m3s_per_hydro = test_support::inflow_m3s_per_hydro_from_primal(&state, &primal, 2);
    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &dual,
            objective: 0.0,
            objective_coeffs: &objective_coeffs,
            row_lower: &row_lower,
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &indexer,
            hydro_cell_index: &test_support::identity_hydro_cell_index(256),
            state: &state,
            entity_counts: &make_entity_counts_2_hydros(),
            inflow_m3s_per_hydro: &inflow_m3s_per_hydro,
            block_hours: &[100.0],
            generic_constraint_entries: &[],
            ncs_col_upper: &[],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_slots: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[1.0, 1.0],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            cost_scale_factor: 1_000_000.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0; 2],
            stage_index: 0,
            horizon: &HorizonMode::Finite { num_stages: 1 },
            anticipated_windows: &[],
            study_stage_ids: &[],
        },
        0,
    );

    // Thermal — one entry per (thermal, block); n_blks=1 ⇒ one entry, all zero.
    assert_eq!(result.thermals.len(), 1);
    assert_eq!(result.thermals[0].generation_mw, 0.0);
    assert_eq!(result.thermals[0].generation_cost, 0.0);
    assert_eq!(result.thermals[0].block_id, Some(0));

    // Exchange — one entry per (line, block); n_blks=1 ⇒ one entry, all zero.
    assert_eq!(result.exchanges.len(), 1);
    assert_eq!(result.exchanges[0].direct_flow_mw, 0.0);
    assert_eq!(result.exchanges[0].block_id, Some(0));

    // Bus — one entry per (bus, block); n_blks=1 ⇒ one entry, all zero.
    assert_eq!(result.buses.len(), 1);
    assert_eq!(result.buses[0].deficit_mw, 0.0);
    assert_eq!(result.buses[0].spot_price, 0.0);
    assert_eq!(result.buses[0].block_id, Some(0));
}

/// Verify that equipment columns are read from the primal vector when the
/// indexer was built via `with_equipment`.
///
/// Column layout for N=2 hydros, L=1 lag, T=1 thermal, Ln=1 line, B=1 bus, K=1 block:
///
/// ```text
/// theta = N*(3+L) = 2*(3+1) = 8
/// turbine:   [9, 11)   h0→9, h1→10
/// spillage:  [11, 13)  h0→11, h1→12
/// diversion: [13, 15)  h0→13, h1→14
/// thermal:   [15, 16)  t0→15
/// line_fwd:  [16, 17)  l0→16
/// line_rev:  [17, 18)  l0→17
/// deficit:   [18, 19)  b0→18
/// excess:    [19, 20)  b0→19
/// ```
#[test]
fn extract_equipment_reads_primal_when_with_equipment() {
    let eq_counts = test_support::GeometryDims {
        hydro_count: 2,
        max_par_order: 1,
        n_thermals: 1,
        n_lines: 1,
        n_buses: 1,
        n_blks: 1,
        has_inflow_penalty: false,
        max_deficit_segments: 1,
        n_anticipated: 0,
        lead_stages: 0,
        anticipated_plants: AnticipatedPlants::default(),
    };
    let indexer = test_support::geometry(&eq_counts, vec![], &[], vec![]);
    let study_dims = test_support::study_dims_for(&eq_counts);
    let state = test_support::state_layout_full(2, 1, vec![]);
    // theta = 8, equipment starts at 9
    assert_eq!(state.theta, 8);
    assert_eq!(indexer.turbine, 9..11);
    assert_eq!(indexer.spillage, 11..13);
    assert_eq!(indexer.diversion, 13..15);
    assert_eq!(indexer.thermal, 15..16);
    assert_eq!(indexer.line_fwd, 16..17);
    assert_eq!(indexer.line_rev, 17..18);
    assert_eq!(indexer.deficit, 18..19);
    assert_eq!(indexer.excess, 19..20);

    // Build a primal vector sized to include withdrawal_slack columns.
    // storage[0..2]=100,200  inflow_lags[2..4]=50,60  z_inflow[4..6]=0,0
    // storage_in[6..8]=90,180  theta[8]=500
    // turbine[9..11]=30.0,40.0   spillage[11..13]=5.0,0.0
    // diversion[13..15]=0.0,0.0
    // thermal[15]=80.0   line_fwd[16]=15.0   line_rev[17]=0.0
    // deficit[18]=10.0   excess[19]=2.0   withdrawal_slack[20..22]=0.0,0.0
    let n_cols = indexer.generation_below_slack.end;
    let mut primal = vec![0.0_f64; n_cols];
    primal[0] = 100.0; // storage h0
    primal[1] = 200.0; // storage h1
    primal[2] = 50.0; // lag h0
    primal[3] = 60.0; // lag h1
    // primal[4..6] = z_inflow (zeros)
    primal[6] = 90.0; // storage_in h0
    primal[7] = 180.0; // storage_in h1
    primal[8] = 500.0; // theta
    primal[9] = 30.0; // turbine h0 b0
    primal[10] = 40.0; // turbine h1 b0
    primal[11] = 5.0; // spillage h0 b0
    primal[12] = 0.0; // spillage h1 b0
    // primal[13..15] = diversion (zeros)
    primal[15] = 80.0; // thermal t0 b0
    primal[16] = 15.0; // line_fwd l0 b0
    primal[17] = 0.0; // line_rev l0 b0
    primal[18] = 10.0; // deficit b0 b0
    primal[19] = 2.0; // excess b0 b0

    // Objective coefficients: thermal cost=50/MWh, spillage cost=0.1, deficit=1000, excess=50
    let mut obj = vec![0.0_f64; n_cols];
    obj[8] = 1.0; // theta (objective = 1)
    obj[11] = 0.1; // spillage h0 penalty
    obj[15] = 50.0; // thermal cost per MW
    obj[16] = 5.0; // line_fwd cost per MW
    obj[18] = 1000.0; // deficit cost per MW
    obj[19] = 50.0; // excess cost per MW

    let counts = EntityCounts {
        hydro_ids: vec![10, 20],
        hydro_productivities: vec![1.0, 1.0],
        thermal_ids: vec![1],
        line_ids: vec![5],
        bus_ids: vec![100],
        pumping_station_ids: vec![],
        contract_ids: vec![],
        non_controllable_ids: vec![],
    };
    // Dual vector: water_value reads from dual[water_balance.start() + h],
    // load balance from dual[load_balance.start() + b*K + blk].
    // N=2, L=1: z_inflow rows [0,2), water_balance=[2,4), load_balance=[4,5).
    let mut dual = vec![0.0_f64; 5];
    dual[2] = -120.0; // water value h0 ($/hm³) — at water_balance.start()+0
    dual[3] = -95.0; // water value h1 ($/hm³) — at water_balance.start()+1
    dual[4] = 108_000.0; // raw load balance dual ($/MW); 150 $/MWh × 720 h

    // Build row_lower for the load balance row. load_balance.start()=4, K=1, B=1.
    let mut row_lower = vec![0.0_f64; 5]; // must be >= load_balance.end = 5
    row_lower[4] = 75.0; // load = 75 MW for bus 100
    let block_hours = [720.0_f64]; // one block, 30-day month
    let ec = zero_energy_conversion(2, 1);
    let inflow_m3s_per_hydro = test_support::inflow_m3s_per_hydro_from_primal(&state, &primal, 2);
    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &dual,
            objective: 600.0,
            objective_coeffs: &obj,
            row_lower: &row_lower,
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &indexer,
            hydro_cell_index: &test_support::identity_hydro_cell_index(256),
            state: &state,
            entity_counts: &counts,
            inflow_m3s_per_hydro: &inflow_m3s_per_hydro,
            block_hours: &block_hours,
            generic_constraint_entries: &[],
            ncs_col_upper: &[],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_slots: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[1.0, 1.0],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            cost_scale_factor: 1_000_000.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0; 2],
            stage_index: 0,
            horizon: &HorizonMode::Finite { num_stages: 1 },
            anticipated_windows: &[],
            study_stage_ids: &[],
        },
        0,
    );

    // Hydro: one entry per (hydro, block), block_id = Some(0)
    assert_eq!(result.hydros.len(), 2); // 2 hydros × 1 block
    assert_eq!(result.hydros[0].block_id, Some(0));
    assert_eq!(result.hydros[0].turbined_m3s, 30.0);
    assert_eq!(result.hydros[0].spillage_m3s, 5.0);
    // spillage_cost = 5.0 * 0.1 * COST_SCALE_FACTOR = 500_000.0
    assert!((result.hydros[0].spillage_cost - 500_000.0).abs() < 1e-12);

    // Hydro generation = turbined * productivity (1.0)
    assert_eq!(result.hydros[0].generation_mw, 30.0); // 30 * 1.0
    assert_eq!(result.hydros[1].generation_mw, 40.0); // 40 * 1.0

    // Hydro h=1 (no spillage)
    assert_eq!(result.hydros[1].block_id, Some(0));
    assert_eq!(result.hydros[1].turbined_m3s, 40.0);
    assert_eq!(result.hydros[1].spillage_m3s, 0.0);

    // Thermal: one entry per (thermal, block), block_id = Some(0)
    assert_eq!(result.thermals.len(), 1);
    assert_eq!(result.thermals[0].generation_mw, 80.0);
    // generation_cost = 80 * 50 * COST_SCALE_FACTOR = 4_000_000_000.0
    assert!((result.thermals[0].generation_cost - 4_000_000_000.0).abs() < 1e-3);
    assert_eq!(result.thermals[0].block_id, Some(0));

    // Exchange: one entry per (line, block)
    assert_eq!(result.exchanges.len(), 1);
    assert_eq!(result.exchanges[0].direct_flow_mw, 15.0);
    assert_eq!(result.exchanges[0].reverse_flow_mw, 0.0);
    // exchange_cost = 15 * 5 * COST_SCALE_FACTOR = 75_000_000.0
    assert!((result.exchanges[0].exchange_cost - 75_000_000.0).abs() < 1e-3);
    assert_eq!(result.exchanges[0].block_id, Some(0));

    // Bus: one entry per (bus, block)
    assert_eq!(result.buses.len(), 1);
    assert_eq!(result.buses[0].load_mw, 75.0); // from row_lower
    assert_eq!(result.buses[0].deficit_mw, 10.0);
    assert_eq!(result.buses[0].excess_mw, 2.0);
    assert_eq!(result.buses[0].block_id, Some(0));
    // spot_price = dual * COST_SCALE_FACTOR / hrs = 108_000 * 1_000_000 / 720 = 150_000_000.0 $/MWh
    assert!((result.buses[0].spot_price - 150_000_000.0).abs() < 1e-3);

    // water_value = dual[water_balance.start()+h] * COST_SCALE_FACTOR
    assert!((result.hydros[0].water_value_per_hm3 - (-120_000_000.0)).abs() < 1e-3);
    assert!((result.hydros[1].water_value_per_hm3 - (-95_000_000.0)).abs() < 1e-3);

    // Cost breakdown — all values multiplied by COST_SCALE_FACTOR = 1_000_000.
    let cost = &result.costs[0];
    assert!((cost.thermal_cost - 4_000_000_000.0).abs() < 1e-3); // 80 * 50 * 1_000_000
    assert!((cost.spillage_cost - 500_000.0).abs() < 1e-6); // 5 * 0.1 * 1_000_000
    assert!((cost.deficit_cost - 10_000_000_000.0).abs() < 1e-3); // 10 * 1000 * 1_000_000
    assert!((cost.excess_cost - 100_000_000.0).abs() < 1e-3); // 2 * 50 * 1_000_000
    assert!((cost.exchange_cost - 75_000_000.0).abs() < 1e-3); // 15 * 5 * 1_000_000
}

/// Verify that `is_anticipated` is set to `true` for thermals whose global
/// index appears in `anticipated_plants`, and `false` for all others.
///
/// Setup: 2 thermals (ids 10 and 20), 1 block. Thermal at global index 1
/// (id=20) is anticipated. The per-block branch is exercised by using
/// `n_blks=1` with a non-empty thermal range.
#[test]
fn extract_thermals_marks_anticipated_thermals_when_indices_nonempty() {
    // N=0 hydros, T=2 thermals, B=0 buses, K=1 block, n_anticipated=1 (index 1)
    let eq_counts = test_support::GeometryDims {
        hydro_count: 0,
        max_par_order: 0,
        n_thermals: 2,
        n_lines: 0,
        n_buses: 0,
        n_blks: 1,
        has_inflow_penalty: false,
        max_deficit_segments: 1,
        n_anticipated: 1,
        lead_stages: 1,
        anticipated_plants: anticipated_plants_at(&[1]),
    };
    let indexer = test_support::geometry(&eq_counts, vec![], &[], vec![]);
    let study_dims = test_support::study_dims_for(&eq_counts);
    let state = test_support::state_layout_full(0, 0, vec![1]);
    // With N=0, L=0, A=1, K_max=1:
    //   commit_out = [0, 1)  (outgoing ring, A*K_max = 1 slot)
    //   commit_in     = [1, 2)  (incoming, A*K_max = 1 slot)
    //   theta = N*(3+L) + 2*A*K_max = 0 + 2 = 2
    //   thermal = [theta+1, theta+1+T*K) = [3, 5)  — t0→3, t1→4
    assert_eq!(state.theta, 2);
    assert_eq!(indexer.thermal, 3..5);

    // n_cols must cover at least anticipated_decision.end (last column used)
    let n_cols = indexer.anticipated_decision.end.max(5);
    let mut primal = vec![0.0_f64; n_cols];
    primal[2] = 0.0; // theta
    primal[3] = 50.0; // thermal t0 b0 (gen_mw)
    primal[4] = 30.0; // thermal t1 b0 (gen_mw)

    let mut obj = vec![0.0_f64; n_cols];
    obj[3] = 10.0; // thermal t0 cost coefficient
    obj[4] = 20.0; // thermal t1 cost coefficient

    let counts = EntityCounts {
        hydro_ids: vec![],
        hydro_productivities: vec![],
        thermal_ids: vec![10, 20],
        line_ids: vec![],
        bus_ids: vec![],
        pumping_station_ids: vec![],
        contract_ids: vec![],
        non_controllable_ids: vec![],
    };

    let ec = zero_energy_conversion(0, 2);
    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &[],
            objective: 0.0,
            objective_coeffs: &obj,
            row_lower: &[],
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &indexer,
            hydro_cell_index: &test_support::identity_hydro_cell_index(256),
            state: &state,
            entity_counts: &counts,
            inflow_m3s_per_hydro: &[],
            block_hours: &[1.0],
            generic_constraint_entries: &[],
            ncs_col_upper: &[],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_slots: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            cost_scale_factor: 1_000_000.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[],
            stage_index: 0,
            horizon: &HorizonMode::Finite { num_stages: 2 },
            anticipated_windows: &[(None, None)],
            study_stage_ids: &[0, 1, 2, 3, 4, 5],
        },
        0,
    );

    // Per-block path: 2 thermals × 1 block = 2 entries
    assert_eq!(result.thermals.len(), 2);

    // Thermal 10 at global index 0: NOT anticipated
    assert_eq!(result.thermals[0].thermal_id, 10);
    assert!(
        !result.thermals[0].is_anticipated,
        "thermal 10 should not be anticipated"
    );
    assert_eq!(result.thermals[0].anticipated_committed_mw, None);
    assert_eq!(result.thermals[0].anticipated_decision_mw, None);

    // Thermal 20 at global index 1: IS anticipated
    assert_eq!(result.thermals[1].thermal_id, 20);
    assert!(
        result.thermals[1].is_anticipated,
        "thermal 20 should be anticipated"
    );
    // Always-active fishing: committed_mw reads slot 0 of commit_in
    // (primal[commit_in.start + 0] = 0.0 in this zero-initialised fixture).
    assert_eq!(result.thermals[1].anticipated_committed_mw, Some(0.0));
    // With stage_index=0, n_stages=2, K_i=1: t+K_i=1 <= n_stages -> decision is active.
    // primal[anticipated_decision.start + 0] defaults to 0.0 in this fixture.
    assert_eq!(result.thermals[1].anticipated_decision_mw, Some(0.0));
}

/// The re-sourced `extract_anticipated_lanes` emits one row per anticipated
/// plant's genuine decision at this stage whose delivery target is post-study,
/// reading the deposited value from the ring decision column
/// (`anticipated_decision.start + local`) and the carried value from the ring
/// slot the target lands in (`commit_out.start + commitment_hold_in_study_offset`),
/// and dating the row from the extended delivery-stage anchors — while a
/// decision targeting an in-study delivery emits nothing.
///
/// Fixture: one anticipated plant (`k_max = 2`) over a 2-stage study extended by
/// one post-study delivery stage (`n_delivery = 3`). Its genuine decisions are
/// an in-study self-delivery decided at stage 0 (`m = 1`) and a post-study
/// delivery decided at stage 1 (`m = 2`). Distinct primal values on the two ring
/// columns prove the extractor reads the two distinct columns; the post-study
/// target `m = 2` lands in slot `(2 % 2) * 1 + 0 = 0` and dates onto
/// `delivery_dates[2]`.
#[test]
fn extract_anticipated_lanes_reads_ring_decision_and_slot_for_post_study_decision() {
    let resolution = AnticipatedResolution {
        per_plant: vec![PointResolution {
            decider: vec![None, Some(0), Some(1)],
            decision_sets: vec![vec![1], vec![2]],
            depth: vec![0, 0],
            occupancy: vec![0, 0],
        }],
    };
    let state = test_support::state_layout_with_transit_buckets_and_resolution(
        0,
        0,
        Vec::new(),
        vec![2],
        resolution,
    );
    assert_eq!(state.commit_out.start, 0);

    let geometry = StageGeometry {
        anticipated_decision: 6..7,
        ..test_support::equipment_free_geometry(&[0]).remove(0)
    };
    let study_dims = StudyDimensions {
        anticipated_plants: anticipated_plants_at(&[0]),
        ..StudyDimensions::default()
    };
    let counts = EntityCounts {
        hydro_ids: vec![],
        hydro_productivities: vec![],
        thermal_ids: vec![7],
        line_ids: vec![],
        bus_ids: vec![],
        pumping_station_ids: vec![],
        contract_ids: vec![],
        non_controllable_ids: vec![],
    };
    let mut primal = vec![0.0_f64; 8];
    primal[state.commit_out.start] = 31.0; // ring slot the target lands in (carried)
    primal[geometry.anticipated_decision.start] = 30.0; // decision column (deposited)
    let delivery_dates = [2024_0101_i32, 2024_0201, 2024_0401];

    let hydro_cell_index = test_support::identity_hydro_cell_index(1);
    let ec = zero_energy_conversion(0, 2);
    let diversion = HashMap::new();
    let view = SolutionView {
        primal: &primal,
        dual: &[],
        objective: 0.0,
        objective_coeffs: &[],
        row_lower: &[],
    };
    let spec = StageExtractionSpec {
        state: &state,
        study_dims: &study_dims,
        geometry: &geometry,
        hydro_cell_index: &hydro_cell_index,
        entity_counts: &counts,
        inflow_m3s_per_hydro: &[],
        block_hours: &[],
        generic_constraint_entries: &[],
        ncs_col_upper: &[],
        pumping_consumption_mw_per_m3s: &[],
        contract_prices: &[],
        contract_slots: &[],
        diversion_upstream: &diversion,
        hydro_productivities: &[],
        col_scale: &[],
        row_scale: &[],
        cumulative_discount_factor: 1.0,
        cost_scale_factor: 1.0,
        energy_conversion: &ec,
        hydro_min_storage_hm3: &[],
        stage_index: 1,
        horizon: &HorizonMode::Finite { num_stages: 2 },
        anticipated_windows: &[],
        study_stage_ids: &[],
    };

    // Decider stage (m = 2 is post-study): exactly one row, reading the two
    // distinct ring columns and dating onto delivery_dates[2].
    let rows = extract_anticipated_lanes(&view, &spec, &delivery_dates, 1);
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].stage_id, 1);
    assert_eq!(rows[0].thermal_id, 7);
    assert_eq!(rows[0].delivery_date, 2024_0401);
    assert_eq!(rows[0].deposited_decision_mw, 30.0);
    assert_eq!(rows[0].carried_committed_mw, 31.0);

    // Stage 0's only genuine decision (m = 1) targets an in-study delivery, so
    // no lane row is emitted there.
    let spec_stage0 = StageExtractionSpec {
        stage_index: 0,
        ..spec
    };
    let rows_stage0 = extract_anticipated_lanes(&view, &spec_stage0, &delivery_dates, 0);
    assert!(
        rows_stage0.is_empty(),
        "an in-study-only decision emits no anticipated-lane row"
    );
}

/// Verify that `is_anticipated` is `false` for every thermal when
/// `anticipated_plants` is empty (no anticipated thermals configured).
#[test]
fn extract_thermals_marks_no_thermals_anticipated_when_indices_empty() {
    // N=0 hydros, T=2 thermals, B=0 buses, K=1 block, n_anticipated=0
    let eq_counts = test_support::GeometryDims {
        hydro_count: 0,
        max_par_order: 0,
        n_thermals: 2,
        n_lines: 0,
        n_buses: 0,
        n_blks: 1,
        has_inflow_penalty: false,
        max_deficit_segments: 1,
        n_anticipated: 0,
        lead_stages: 0,
        anticipated_plants: AnticipatedPlants::default(),
    };
    let indexer = test_support::geometry(&eq_counts, vec![], &[], vec![]);
    let study_dims = test_support::study_dims_for(&eq_counts);
    let state = test_support::state_layout_full(0, 0, vec![]);

    let n_cols = indexer.generation_below_slack.end.max(3);
    let primal = vec![0.0_f64; n_cols];
    let obj = vec![0.0_f64; n_cols];

    let counts = EntityCounts {
        hydro_ids: vec![],
        hydro_productivities: vec![],
        thermal_ids: vec![10, 20],
        line_ids: vec![],
        bus_ids: vec![],
        pumping_station_ids: vec![],
        contract_ids: vec![],
        non_controllable_ids: vec![],
    };

    let ec = zero_energy_conversion(0, 2);
    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &[],
            objective: 0.0,
            objective_coeffs: &obj,
            row_lower: &[],
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &indexer,
            hydro_cell_index: &test_support::identity_hydro_cell_index(256),
            state: &state,
            entity_counts: &counts,
            inflow_m3s_per_hydro: &[],
            block_hours: &[1.0],
            generic_constraint_entries: &[],
            ncs_col_upper: &[],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_slots: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            cost_scale_factor: 1_000_000.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[],
            stage_index: 0,
            horizon: &HorizonMode::Finite { num_stages: 2 },
            anticipated_windows: &[],
            study_stage_ids: &[],
        },
        0,
    );

    // Per-block path: 2 thermals × 1 block = 2 entries, both non-anticipated
    assert_eq!(result.thermals.len(), 2);
    assert!(
        !result.thermals[0].is_anticipated,
        "thermal 10 should not be anticipated"
    );
    assert!(
        !result.thermals[1].is_anticipated,
        "thermal 20 should not be anticipated"
    );
}

// -------------------------------------------------------------------------
// Tests for compute_anticipated_decision_mw
// -------------------------------------------------------------------------

/// Shared fixture builder for the anticipated-decision tests.
///
/// N=0 hydros, T=2 thermals (IDs 10 and 20), B=0 buses, K=1 block,
/// `n_anticipated=1` (global index 1, ID 20), `k_max=2`, `K_i=2`.
///
/// Layout:
///   `n_ant_state = 1*2 = 2`  →  theta = 2
///   thermal = [3, 5)         →  `anticipated_decision.start = 5`
/// The `GeometryDims` both [`make_anticipated_decision_indexer_k2`] and the
/// matching `study_dims` derive from, so the role-(b) geometry and the
/// non-state `StudyDimensions` stay aligned from one source.
fn anticipated_decision_counts_k2() -> test_support::GeometryDims {
    test_support::GeometryDims {
        n_thermals: 2,
        n_blks: 1,
        n_anticipated: 1,
        lead_stages: 2,
        anticipated_plants: anticipated_plants_at(&[1]),
        ..Default::default()
    }
}

fn make_anticipated_decision_indexer_k2() -> StageGeometry {
    test_support::geometry(&anticipated_decision_counts_k2(), vec![], &[], vec![])
}

/// Returns a primal vector sized to cover `anticipated_decision.end`, with
/// the decision column for local index 0 set to `sentinel`.
fn make_primal_with_decision_sentinel(geometry: &StageGeometry, sentinel: f64) -> Vec<f64> {
    let n_cols = geometry.anticipated_decision.end.max(geometry.thermal.end);
    let mut primal = vec![0.0_f64; n_cols];
    primal[geometry.anticipated_decision.start] = sentinel;
    primal
}

/// Single anticipated thermal (local index 0, global index 1, ID 20),
/// `K_i=2`, `stage_index=0`, `n_stages=3`.  `t+K_i = 2 <= 3` — decision is active.
///
/// Expects `anticipated_decision_mw == Some(123.5)` for the record with ID 20.
#[test]
fn extract_thermals_reads_anticipated_decision_when_in_horizon() {
    let indexer = make_anticipated_decision_indexer_k2();
    let study_dims = test_support::study_dims_for(&anticipated_decision_counts_k2());
    let state = test_support::state_layout_full(0, 0, vec![2]);
    let primal = make_primal_with_decision_sentinel(&indexer, 123.5);
    let obj = vec![0.0_f64; primal.len()];

    let counts = EntityCounts {
        hydro_ids: vec![],
        hydro_productivities: vec![],
        thermal_ids: vec![10, 20],
        line_ids: vec![],
        bus_ids: vec![],
        pumping_station_ids: vec![],
        contract_ids: vec![],
        non_controllable_ids: vec![],
    };

    let ec = zero_energy_conversion(0, 2);
    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &[],
            objective: 0.0,
            objective_coeffs: &obj,
            row_lower: &[],
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &indexer,
            hydro_cell_index: &test_support::identity_hydro_cell_index(256),
            state: &state,
            entity_counts: &counts,
            inflow_m3s_per_hydro: &[],
            block_hours: &[1.0],
            generic_constraint_entries: &[],
            ncs_col_upper: &[],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_slots: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            cost_scale_factor: 1_000_000.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[],
            stage_index: 0,
            horizon: &HorizonMode::Finite { num_stages: 3 },
            anticipated_windows: &[(None, None)],
            study_stage_ids: &[0, 1, 2, 3, 4, 5],
        },
        0,
    );

    // 2 thermals × 1 block = 2 records.  Thermal 20 is at index 1.
    assert_eq!(result.thermals.len(), 2);
    assert_eq!(result.thermals[1].thermal_id, 20);
    assert_eq!(
        result.thermals[1].anticipated_decision_mw,
        Some(123.5),
        "expected Some(123.5) for thermal 20 at stage 0 with K_i=2, n_stages=3"
    );
}

/// Same fixture, `stage_index=1`.  `t+K_i = 1+2 = 3 == n_stages` (boundary
/// is INACTIVE under the strict predicate `<`).
///
/// Expects `anticipated_decision_mw == None` — the LP has [0,0] bounds at
/// this boundary column and the extraction predicate must match.
#[test]
fn extract_thermals_emits_none_at_horizon_boundary() {
    let indexer = make_anticipated_decision_indexer_k2();
    let study_dims = test_support::study_dims_for(&anticipated_decision_counts_k2());
    let state = test_support::state_layout_full(0, 0, vec![2]);
    let primal = make_primal_with_decision_sentinel(&indexer, 123.5);
    let obj = vec![0.0_f64; primal.len()];

    let counts = EntityCounts {
        hydro_ids: vec![],
        hydro_productivities: vec![],
        thermal_ids: vec![10, 20],
        line_ids: vec![],
        bus_ids: vec![],
        pumping_station_ids: vec![],
        contract_ids: vec![],
        non_controllable_ids: vec![],
    };

    let ec = zero_energy_conversion(0, 2);
    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &[],
            objective: 0.0,
            objective_coeffs: &obj,
            row_lower: &[],
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &indexer,
            hydro_cell_index: &test_support::identity_hydro_cell_index(256),
            state: &state,
            entity_counts: &counts,
            inflow_m3s_per_hydro: &[],
            block_hours: &[1.0],
            generic_constraint_entries: &[],
            ncs_col_upper: &[],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_slots: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            cost_scale_factor: 1_000_000.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[],
            stage_index: 1,
            horizon: &HorizonMode::Finite { num_stages: 3 },
            anticipated_windows: &[(None, None)],
            study_stage_ids: &[0, 1, 2, 3, 4, 5],
        },
        0,
    );

    assert_eq!(result.thermals.len(), 2);
    assert_eq!(result.thermals[1].thermal_id, 20);
    assert_eq!(
        result.thermals[1].anticipated_decision_mw, None,
        "expected None for thermal 20 at boundary stage 1 (K_i=2, n_stages=3): \
             t+K_i = 3 >= n_stages = 3 under the strict predicate"
    );
}

/// Same fixture, `stage_index=2`.  `t+K_i = 2+2 = 4 > 3 = n_stages` — one past
/// the boundary.  Decision column is structurally absent.
///
/// Expects `anticipated_decision_mw == None`.
#[test]
fn extract_thermals_emits_none_one_past_horizon_boundary() {
    let indexer = make_anticipated_decision_indexer_k2();
    let study_dims = test_support::study_dims_for(&anticipated_decision_counts_k2());
    let state = test_support::state_layout_full(0, 0, vec![2]);
    let primal = make_primal_with_decision_sentinel(&indexer, 123.5);
    let obj = vec![0.0_f64; primal.len()];

    let counts = EntityCounts {
        hydro_ids: vec![],
        hydro_productivities: vec![],
        thermal_ids: vec![10, 20],
        line_ids: vec![],
        bus_ids: vec![],
        pumping_station_ids: vec![],
        contract_ids: vec![],
        non_controllable_ids: vec![],
    };

    let ec = zero_energy_conversion(0, 2);
    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &[],
            objective: 0.0,
            objective_coeffs: &obj,
            row_lower: &[],
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &indexer,
            hydro_cell_index: &test_support::identity_hydro_cell_index(256),
            state: &state,
            entity_counts: &counts,
            inflow_m3s_per_hydro: &[],
            block_hours: &[1.0],
            generic_constraint_entries: &[],
            ncs_col_upper: &[],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_slots: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            cost_scale_factor: 1_000_000.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[],
            stage_index: 2,
            horizon: &HorizonMode::Finite { num_stages: 3 },
            anticipated_windows: &[(None, None)],
            study_stage_ids: &[0, 1, 2, 3, 4, 5],
        },
        0,
    );

    assert_eq!(result.thermals.len(), 2);
    assert_eq!(result.thermals[1].thermal_id, 20);
    assert_eq!(
        result.thermals[1].anticipated_decision_mw, None,
        "expected None for thermal 20 at stage 2 with K_i=2, n_stages=3 (one past boundary)"
    );
}

/// Two thermals (IDs 10 and 20), only thermal 20 is anticipated (global
/// index 1, local index 0).  `K_i=1`, `stage_index=0`, `n_stages=2`.
///
/// Asserts that the record for thermal 10 (the regular thermal) has
/// `anticipated_decision_mw == None`.
#[test]
fn extract_thermals_emits_none_for_non_anticipated_thermals() {
    // N=0, T=2, n_blks=1, n_anticipated=1 (index 1), k_max=1, K_i=1
    // Layout: n_ant_state=1, theta=1, thermal=[2,4), anticipated_decision.start=4
    let eq_counts = test_support::GeometryDims {
        hydro_count: 0,
        max_par_order: 0,
        n_thermals: 2,
        n_lines: 0,
        n_buses: 0,
        n_blks: 1,
        has_inflow_penalty: false,
        max_deficit_segments: 1,
        n_anticipated: 1,
        lead_stages: 1,
        anticipated_plants: anticipated_plants_at(&[1]),
    };
    let indexer = test_support::geometry(&eq_counts, vec![], &[], vec![]);
    let study_dims = test_support::study_dims_for(&eq_counts);
    let state = test_support::state_layout_full(0, 0, vec![1]);

    let n_cols = indexer.anticipated_decision.end.max(indexer.thermal.end);
    let mut primal = vec![0.0_f64; n_cols];
    primal[indexer.anticipated_decision.start] = 123.5;
    let obj = vec![0.0_f64; n_cols];

    let counts = EntityCounts {
        hydro_ids: vec![],
        hydro_productivities: vec![],
        thermal_ids: vec![10, 20],
        line_ids: vec![],
        bus_ids: vec![],
        pumping_station_ids: vec![],
        contract_ids: vec![],
        non_controllable_ids: vec![],
    };

    let ec = zero_energy_conversion(0, 2);
    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &[],
            objective: 0.0,
            objective_coeffs: &obj,
            row_lower: &[],
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &indexer,
            hydro_cell_index: &test_support::identity_hydro_cell_index(256),
            state: &state,
            entity_counts: &counts,
            inflow_m3s_per_hydro: &[],
            block_hours: &[1.0],
            generic_constraint_entries: &[],
            ncs_col_upper: &[],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_slots: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            cost_scale_factor: 1_000_000.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[],
            stage_index: 0,
            horizon: &HorizonMode::Finite { num_stages: 2 },
            anticipated_windows: &[(None, None)],
            study_stage_ids: &[0, 1, 2, 3, 4, 5],
        },
        0,
    );

    assert_eq!(result.thermals.len(), 2);
    assert_eq!(result.thermals[0].thermal_id, 10);
    assert_eq!(
        result.thermals[0].anticipated_decision_mw, None,
        "thermal 10 is not anticipated; expected None"
    );
}

/// `n_blks=4`, one anticipated thermal (ID 30, global index 0, local index
/// 0).  `K_i=1`, `stage_index=0`, `n_stages=2`.  The decision column is
/// stage-level (not per-block), so the same value must appear in all 4
/// block records.
///
/// Layout: N=0, T=1, `n_blks=4`, `n_anticipated=1`, `k_max=1`:
///   `n_ant_state=1`, theta=1, thermal=[2,6), `anticipated_decision.start=6`
#[test]
fn extract_thermals_anticipated_decision_is_per_block_invariant() {
    let eq_counts = test_support::GeometryDims {
        hydro_count: 0,
        max_par_order: 0,
        n_thermals: 1,
        n_lines: 0,
        n_buses: 0,
        n_blks: 4,
        has_inflow_penalty: false,
        max_deficit_segments: 1,
        n_anticipated: 1,
        lead_stages: 1,
        anticipated_plants: anticipated_plants_at(&[0]),
    };
    let indexer = test_support::geometry(&eq_counts, vec![], &[], vec![]);
    let study_dims = test_support::study_dims_for(&eq_counts);
    let state = test_support::state_layout_full(0, 0, vec![1]);

    let n_cols = indexer.anticipated_decision.end.max(indexer.thermal.end);
    let mut primal = vec![0.0_f64; n_cols];
    primal[indexer.anticipated_decision.start] = 123.5;
    let obj = vec![0.0_f64; n_cols];

    let counts = EntityCounts {
        hydro_ids: vec![],
        hydro_productivities: vec![],
        thermal_ids: vec![30],
        line_ids: vec![],
        bus_ids: vec![],
        pumping_station_ids: vec![],
        contract_ids: vec![],
        non_controllable_ids: vec![],
    };

    let ec = zero_energy_conversion(0, 1);
    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &[],
            objective: 0.0,
            objective_coeffs: &obj,
            row_lower: &[],
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &indexer,
            hydro_cell_index: &test_support::identity_hydro_cell_index(256),
            state: &state,
            entity_counts: &counts,
            inflow_m3s_per_hydro: &[],
            block_hours: &[1.0, 1.0, 1.0, 1.0],
            generic_constraint_entries: &[],
            ncs_col_upper: &[],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_slots: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            cost_scale_factor: 1_000_000.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[],
            stage_index: 0,
            horizon: &HorizonMode::Finite { num_stages: 2 },
            anticipated_windows: &[(None, None)],
            study_stage_ids: &[0, 1, 2, 3, 4, 5],
        },
        0,
    );

    // 1 thermal × 4 blocks = 4 records, all for thermal 30.
    assert_eq!(
        result.thermals.len(),
        4,
        "expected 4 block records (1 thermal × 4 blocks)"
    );
    for (blk, rec) in result.thermals.iter().enumerate() {
        assert_eq!(rec.thermal_id, 30);
        assert_eq!(
            rec.anticipated_decision_mw,
            Some(123.5),
            "block {blk}: expected Some(123.5) for per-block invariance"
        );
    }
}

/// The delivery stage comes from the plant's attached `PointResolution`
/// (`PointResolution::genuine_decisions_at`), never `stage_idx +
/// anticipated_lead_stages` — the constant-lead fallback
/// `anticipated_resolution_for` uses only when no resolution is
/// attached.
///
/// `anticipated_lead_stages = [3]` (stale once a resolution is attached) would
/// place a naive delivery stage at `0 + 3 = 3 >= n_stages = 3` (inactive); the
/// attached resolution instead genuinely decides delivery stage 1 at stage 0,
/// well inside the horizon.
#[test]
fn extract_thermals_decision_uses_attached_resolution_delivery_stage() {
    let eq_counts = test_support::GeometryDims {
        hydro_count: 0,
        max_par_order: 0,
        n_thermals: 1,
        n_lines: 0,
        n_buses: 0,
        n_blks: 1,
        has_inflow_penalty: false,
        max_deficit_segments: 1,
        n_anticipated: 1,
        lead_stages: 3,
        anticipated_plants: anticipated_plants_at(&[0]),
    };
    let indexer = test_support::geometry(&eq_counts, vec![], &[], vec![]);
    let study_dims = test_support::study_dims_for(&eq_counts);
    let resolution = AnticipatedResolution {
        per_plant: vec![PointResolution {
            decider: vec![None, Some(0), None],
            decision_sets: vec![vec![1], vec![], vec![]],
            depth: vec![1, 0, 0],
            occupancy: vec![2, 1, 0],
        }],
    };
    let state = test_support::state_layout_with_transit_buckets_and_resolution(
        0,
        0,
        Vec::new(),
        vec![3],
        resolution,
    );

    let n_cols = indexer.anticipated_decision.end.max(indexer.thermal.end);
    let mut primal = vec![0.0_f64; n_cols];
    primal[indexer.anticipated_decision.start] = 77.0;
    let obj = vec![0.0_f64; n_cols];

    let counts = EntityCounts {
        hydro_ids: vec![],
        hydro_productivities: vec![],
        thermal_ids: vec![30],
        line_ids: vec![],
        bus_ids: vec![],
        pumping_station_ids: vec![],
        contract_ids: vec![],
        non_controllable_ids: vec![],
    };

    let ec = zero_energy_conversion(0, 1);
    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &[],
            objective: 0.0,
            objective_coeffs: &obj,
            row_lower: &[],
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &indexer,
            hydro_cell_index: &test_support::identity_hydro_cell_index(256),
            state: &state,
            entity_counts: &counts,
            inflow_m3s_per_hydro: &[],
            block_hours: &[1.0],
            generic_constraint_entries: &[],
            ncs_col_upper: &[],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_slots: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            cost_scale_factor: 1_000_000.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[],
            stage_index: 0,
            horizon: &HorizonMode::Finite { num_stages: 3 },
            anticipated_windows: &[(None, None)],
            study_stage_ids: &[0, 1, 2],
        },
        0,
    );

    assert_eq!(result.thermals.len(), 1);
    assert_eq!(
        result.thermals[0].anticipated_decision_mw,
        Some(77.0),
        "the attached resolution's genuine delivery stage (1) is inside the \
         horizon; the stale anticipated_lead_stages=[3] constant-lead fallback \
         would instead place delivery at stage 3 (>= n_stages) and read None"
    );
}

// Tests for compute_anticipated_committed_mw
// -------------------------------------------------------------------------

/// Build a `StageGeometry` for the anticipated-committed tests.
///
/// N=0 hydros, T=1 thermal (ID 10, global index 0), `n_blks=3`, `n_anticipated=1`
/// (global index 0, local index 0), `k_max=2`, `K_i=2`.
///
/// Layout: `n_ant_state = 1*2 = 2`, doubled for the outgoing+incoming ring
/// blocks → `theta = 4`.
///   `commit_out` (outgoing) = `[0, 2)`
///   `commit_in` (incoming)     = `[2, 4)`
///   `thermal` = `[5, 8)`          (1 thermal * 3 blocks)
///   `anticipated_decision.start = 8`
/// The `GeometryDims` both
/// [`make_anticipated_committed_indexer_k2_3blks`] and the matching
/// `study_dims` derive from, keeping the role-(b) geometry and the non-state
/// `StudyDimensions` aligned from one source.
fn anticipated_committed_counts_k2_3blks() -> test_support::GeometryDims {
    test_support::GeometryDims {
        n_thermals: 1,
        n_blks: 3,
        n_anticipated: 1,
        lead_stages: 2,
        anticipated_plants: anticipated_plants_at(&[0]),
        ..Default::default()
    }
}

fn make_anticipated_committed_indexer_k2_3blks() -> StageGeometry {
    test_support::geometry(
        &anticipated_committed_counts_k2_3blks(),
        vec![],
        &[],
        vec![],
    )
}

/// Per-block branch, K=2, `stage_index=2` (delivery stage).
///
/// The committed value is the slot-0 entry of the `commit_in` ring
/// buffer (per-plant, per-stage scalar), NOT the per-block thermal
/// generation. To guard against the regression where the helper
/// returned `primal[thermal_col]` (the per-block generation) instead of
/// `primal[commit_in.start + local_idx]`, this fixture uses three
/// distinct per-block generation values (50/60/70 MW) and a distinct
/// slot-0 value (42.0 MW). The fix must read 42.0 for every block; the
/// bug would read 50/60/70.
///
/// Per the fishing constraint
/// `sum_blk h_blk * g_blk = h_total * s_{i,0}`, a real LP would couple
/// these — but this is a synthetic primal vector, so the values are
/// independent and the test isolates which column the helper reads.
#[test]
fn extract_thermals_per_block_committed_at_delivery_stage() {
    let indexer = make_anticipated_committed_indexer_k2_3blks();
    let state = test_support::state_layout_full(0, 0, vec![2]);
    // thermal = [5, 8): col 5 = block 0, col 6 = block 1, col 7 = block 2
    // (theta = N*(3+L) + 2*n_ant_state = 0 + 2*2 = 4, control region starts at 5).
    assert_eq!(indexer.thermal.start, 5);
    // commit_in (the incoming, pinned block) = [2, 4): col 2 = slot 0,
    // col 3 = slot 1 (the outgoing `commit_out` ring occupies [0, 2)
    // instead; `compute_anticipated_committed_mw` reads the INCOMING slot).
    assert_eq!(state.commit_in.start, 2);
    let n_cols = indexer.anticipated_decision.end.max(indexer.thermal.end);
    let mut primal = vec![0.0_f64; n_cols];
    primal[2] = 42.0; // ant_state slot 0 (the committed scalar)
    primal[3] = 99.0; // ant_state slot 1 (unrelated; must not be read)
    primal[4] = 0.0; // theta
    primal[5] = 50.0; // thermal 10, block 0
    primal[6] = 60.0; // thermal 10, block 1
    primal[7] = 70.0; // thermal 10, block 2
    let obj = vec![0.0_f64; n_cols];

    let study_dims = test_support::study_dims_for(&anticipated_committed_counts_k2_3blks());
    let anticipated_plants = &study_dims.anticipated_plants;
    let spec = StageExtractionSpec {
        study_dims: &study_dims,
        geometry: &indexer,
        hydro_cell_index: &test_support::identity_hydro_cell_index(256),
        state: &state,
        entity_counts: &EntityCounts {
            hydro_ids: vec![],
            hydro_productivities: vec![],
            thermal_ids: vec![10],
            line_ids: vec![],
            bus_ids: vec![],
            pumping_station_ids: vec![],
            contract_ids: vec![],
            non_controllable_ids: vec![],
        },
        inflow_m3s_per_hydro: &[],
        block_hours: &[1.0, 1.0, 1.0],
        generic_constraint_entries: &[],
        ncs_col_upper: &[],
        pumping_consumption_mw_per_m3s: &[],
        contract_prices: &[],
        contract_slots: &[],
        diversion_upstream: &HashMap::new(),
        hydro_productivities: &[],
        col_scale: &[],
        row_scale: &[],
        cumulative_discount_factor: 1.0,
        cost_scale_factor: 1_000_000.0,
        energy_conversion: &zero_energy_conversion(0, 3),
        hydro_min_storage_hm3: &[],
        stage_index: 2,
        horizon: &HorizonMode::Finite { num_stages: 3 },
        anticipated_windows: &[(None, None)],
        study_stage_ids: &[0, 1, 2, 3, 4, 5],
    };
    let view = SolutionView {
        primal: &primal,
        dual: &[],
        objective: 0.0,
        objective_coeffs: &obj,
        row_lower: &[],
    };

    assert_eq!(
        super::compute_anticipated_committed_mw(&view, &spec, anticipated_plants, 0),
        Some(42.0),
        "helper: expected slot-0 value 42.0, NOT a per-block thermal value"
    );

    let result = extract_stage_result(&view, &spec, 0);

    assert_eq!(result.thermals.len(), 3);
    for (blk, rec) in result.thermals.iter().enumerate() {
        assert_eq!(
            rec.anticipated_committed_mw,
            Some(42.0),
            "block {blk}: must read slot-0 ant_state (42.0), not per-block gen"
        );
        assert_ne!(
            rec.anticipated_committed_mw,
            Some(rec.generation_mw),
            "block {blk}: committed_mw must NOT alias generation_mw"
        );
    }
}

/// Per-block branch, K=2, `stage_index=1`. The always-active fishing extraction
/// reads the maturing `commit_in` slot for THIS stage — `stage_index mod k_max =
/// 1 mod 2 = 1` (`commit_in` slot 1), never a fixed slot 0. A decoy in slot 0
/// (col 2) proves the read is stage-keyed; the maturing slot 1 (col 3) is left
/// zero-seeded, so every block reads `Some(0.0)`.
#[test]
fn extract_thermals_per_block_committed_reads_stage_maturing_slot_when_seed_zero() {
    let indexer = make_anticipated_committed_indexer_k2_3blks();
    let study_dims = test_support::study_dims_for(&anticipated_committed_counts_k2_3blks());
    let state = test_support::state_layout_full(0, 0, vec![2]);
    // commit_in = [2, 4): slot 0 = col 2, slot 1 = col 3. At stage 1 the maturing
    // slot is 1 mod 2 = 1 (col 3); slot 0 carries a decoy that must NOT be read.
    assert_eq!(state.commit_in.start, 2);
    let n_cols = indexer.anticipated_decision.end.max(indexer.thermal.end);
    let mut primal = vec![0.0_f64; n_cols];
    primal[2] = 88.0; // slot 0 decoy — not the maturing slot at stage 1
    // primal[3] (slot 1, the stage-1 maturing slot) stays 0.0.
    primal[5] = 70.0; // thermal block 0 (distinct, must not alias committed)
    let obj = vec![0.0_f64; n_cols];

    let counts = EntityCounts {
        hydro_ids: vec![],
        hydro_productivities: vec![],
        thermal_ids: vec![10],
        line_ids: vec![],
        bus_ids: vec![],
        pumping_station_ids: vec![],
        contract_ids: vec![],
        non_controllable_ids: vec![],
    };
    let ec = zero_energy_conversion(0, 3);
    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &[],
            objective: 0.0,
            objective_coeffs: &obj,
            row_lower: &[],
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &indexer,
            hydro_cell_index: &test_support::identity_hydro_cell_index(256),
            state: &state,
            entity_counts: &counts,
            inflow_m3s_per_hydro: &[],
            block_hours: &[1.0, 1.0, 1.0],
            generic_constraint_entries: &[],
            ncs_col_upper: &[],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_slots: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            cost_scale_factor: 1_000_000.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[],
            stage_index: 1,
            horizon: &HorizonMode::Finite { num_stages: 3 },
            anticipated_windows: &[(None, None)],
            study_stage_ids: &[0, 1, 2, 3, 4, 5],
        },
        0,
    );

    assert_eq!(result.thermals.len(), 3);
    for (blk, rec) in result.thermals.iter().enumerate() {
        assert_eq!(
            rec.anticipated_committed_mw,
            Some(0.0),
            "block {blk}: stage 1 maturing slot is 1 mod 2 = 1 (zero-seeded); the \
             slot-0 decoy (88.0) must not be read"
        );
    }
}

/// Per-block branch, K=2, `stage_index=2` (boundary: `k_i == stage_index`).
/// Expects every block to have `anticipated_committed_mw == Some(_)`.
#[test]
fn extract_thermals_per_block_committed_at_first_delivery_boundary() {
    let indexer = make_anticipated_committed_indexer_k2_3blks();
    let study_dims = test_support::study_dims_for(&anticipated_committed_counts_k2_3blks());
    let state = test_support::state_layout_full(0, 0, vec![2]);
    let n_cols = indexer.anticipated_decision.end.max(indexer.thermal.end);
    let mut primal = vec![0.0_f64; n_cols];
    primal[3] = 50.0;
    primal[4] = 60.0;
    primal[5] = 70.0;
    let obj = vec![0.0_f64; n_cols];

    let counts = EntityCounts {
        hydro_ids: vec![],
        hydro_productivities: vec![],
        thermal_ids: vec![10],
        line_ids: vec![],
        bus_ids: vec![],
        pumping_station_ids: vec![],
        contract_ids: vec![],
        non_controllable_ids: vec![],
    };
    let ec = zero_energy_conversion(0, 3);
    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &[],
            objective: 0.0,
            objective_coeffs: &obj,
            row_lower: &[],
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &indexer,
            hydro_cell_index: &test_support::identity_hydro_cell_index(256),
            state: &state,
            entity_counts: &counts,
            inflow_m3s_per_hydro: &[],
            block_hours: &[1.0, 1.0, 1.0],
            generic_constraint_entries: &[],
            ncs_col_upper: &[],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_slots: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            cost_scale_factor: 1_000_000.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[],
            stage_index: 2, // k_i == stage_index: boundary acceptance
            horizon: &HorizonMode::Finite { num_stages: 3 },
            anticipated_windows: &[(None, None)],
            study_stage_ids: &[0, 1, 2, 3, 4, 5],
        },
        0,
    );

    assert_eq!(result.thermals.len(), 3);
    for (blk, rec) in result.thermals.iter().enumerate() {
        assert!(
            rec.anticipated_committed_mw.is_some(),
            "block {blk}: expected Some(_) at first delivery boundary (k_i == stage_index)"
        );
    }
}

/// Two thermals, only thermal at global index 1 is anticipated.
/// Thermal at global index 0 must have `anticipated_committed_mw == None` for every block.
#[test]
fn extract_thermals_per_block_committed_none_for_non_anticipated() {
    // N=0, T=2, n_blks=3, n_anticipated=1 (global index 1), k_max=2, K_i=2
    let eq_counts = test_support::GeometryDims {
        hydro_count: 0,
        max_par_order: 0,
        n_thermals: 2,
        n_lines: 0,
        n_buses: 0,
        n_blks: 3,
        has_inflow_penalty: false,
        max_deficit_segments: 1,
        n_anticipated: 1,
        lead_stages: 2,
        anticipated_plants: anticipated_plants_at(&[1]),
    };
    let indexer = test_support::geometry(&eq_counts, vec![], &[], vec![]);
    let study_dims = test_support::study_dims_for(&eq_counts);
    let state = test_support::state_layout_full(0, 0, vec![2]);

    let n_cols = indexer.anticipated_decision.end.max(indexer.thermal.end);
    let mut primal = vec![0.0_f64; n_cols];
    // thermal 0 (non-anticipated): blocks at thermal.start + 0*3 + [0,1,2]
    primal[indexer.thermal.start] = 10.0;
    primal[indexer.thermal.start + 1] = 20.0;
    primal[indexer.thermal.start + 2] = 30.0;
    // thermal 1 (anticipated): blocks at thermal.start + 1*3 + [0,1,2]
    primal[indexer.thermal.start + 3] = 40.0;
    primal[indexer.thermal.start + 4] = 50.0;
    primal[indexer.thermal.start + 5] = 60.0;
    let obj = vec![0.0_f64; n_cols];

    let counts = EntityCounts {
        hydro_ids: vec![],
        hydro_productivities: vec![],
        thermal_ids: vec![10, 20],
        line_ids: vec![],
        bus_ids: vec![],
        pumping_station_ids: vec![],
        contract_ids: vec![],
        non_controllable_ids: vec![],
    };
    let ec = zero_energy_conversion(0, 3);
    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &[],
            objective: 0.0,
            objective_coeffs: &obj,
            row_lower: &[],
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &indexer,
            hydro_cell_index: &test_support::identity_hydro_cell_index(256),
            state: &state,
            entity_counts: &counts,
            inflow_m3s_per_hydro: &[],
            block_hours: &[1.0, 1.0, 1.0],
            generic_constraint_entries: &[],
            ncs_col_upper: &[],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_slots: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            cost_scale_factor: 1_000_000.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[],
            stage_index: 2, // delivery stage for thermal 20
            horizon: &HorizonMode::Finite { num_stages: 3 },
            anticipated_windows: &[(None, None)],
            study_stage_ids: &[0, 1, 2, 3, 4, 5],
        },
        0,
    );

    // 2 thermals * 3 blocks = 6 records; thermal 10 is first 3.
    assert_eq!(result.thermals.len(), 6);
    for blk in 0..3 {
        assert_eq!(result.thermals[blk].thermal_id, 10);
        assert_eq!(
            result.thermals[blk].anticipated_committed_mw, None,
            "block {blk}: non-anticipated thermal 10 must have None"
        );
    }
    // Thermal 20 (anticipated) at delivery stage must have Some(_).
    for blk in 0..3 {
        assert!(
            result.thermals[3 + blk].anticipated_committed_mw.is_some(),
            "block {blk}: anticipated thermal 20 must have Some(_) at delivery"
        );
    }
}

/// K=1, `stage_index=1`, `n_stages=2` (delivery).
/// Expects `anticipated_committed_mw == Some(0.0)`.
#[test]
fn extract_thermals_committed_at_delivery_is_zero_when_slot0_seeded() {
    // N=0, T=1, n_blks=1, n_anticipated=1, k_max=1, K_i=1
    let eq_counts = test_support::GeometryDims {
        hydro_count: 0,
        max_par_order: 0,
        n_thermals: 1,
        n_lines: 0,
        n_buses: 0,
        n_blks: 1,
        has_inflow_penalty: false,
        max_deficit_segments: 1,
        n_anticipated: 1,
        lead_stages: 1,
        anticipated_plants: anticipated_plants_at(&[0]),
    };
    let indexer = test_support::geometry(&eq_counts, vec![], &[], vec![]);
    let study_dims = test_support::study_dims_for(&eq_counts);
    let state = test_support::state_layout_full(0, 0, vec![1]);

    let anticipated_plants = &study_dims.anticipated_plants;
    let spec_delivery = StageExtractionSpec {
        study_dims: &study_dims,
        geometry: &indexer,
        hydro_cell_index: &test_support::identity_hydro_cell_index(256),
        state: &state,
        entity_counts: &EntityCounts {
            hydro_ids: vec![],
            hydro_productivities: vec![],
            thermal_ids: vec![10],
            line_ids: vec![],
            bus_ids: vec![],
            pumping_station_ids: vec![],
            contract_ids: vec![],
            non_controllable_ids: vec![],
        },
        inflow_m3s_per_hydro: &[],
        block_hours: &[],
        generic_constraint_entries: &[],
        ncs_col_upper: &[],
        pumping_consumption_mw_per_m3s: &[],
        contract_prices: &[],
        contract_slots: &[],
        diversion_upstream: &HashMap::new(),
        hydro_productivities: &[],
        col_scale: &[],
        row_scale: &[],
        cumulative_discount_factor: 1.0,
        cost_scale_factor: 1_000_000.0,
        energy_conversion: &zero_energy_conversion(0, 2),
        hydro_min_storage_hm3: &[],
        stage_index: 1,
        horizon: &HorizonMode::Finite { num_stages: 2 },
        anticipated_windows: &[(None, None)],
        study_stage_ids: &[0, 1, 2, 3, 4, 5],
    };
    // The commitment-hold ring's slot 0 (incoming) is zero-seeded here, so the
    // helper returns Some(0.0) regardless of the thermal-generation columns.
    let n_cols = indexer.anticipated_decision.end.max(1);
    let primal = vec![0.0_f64; n_cols];
    let obj = vec![0.0_f64; n_cols];
    let view_helper = SolutionView {
        primal: &primal,
        dual: &[],
        objective: 0.0,
        objective_coeffs: &[],
        row_lower: &[],
    };
    assert_eq!(
        super::compute_anticipated_committed_mw(
            &view_helper,
            &spec_delivery,
            anticipated_plants,
            0
        ),
        Some(0.0),
        "consolidated helper: expected Some(0.0) at delivery stage (slot-0 = incoming = 0.0)"
    );

    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &[],
            objective: 0.0,
            objective_coeffs: &obj,
            row_lower: &[],
        },
        &spec_delivery,
        0,
    );

    assert_eq!(result.thermals.len(), 1);
    assert_eq!(
        result.thermals[0].anticipated_committed_mw,
        Some(0.0),
        "delivery: expected Some(0.0)"
    );
}

/// K=1, `stage_index=0`, `n_stages=2`. Pre-delivery
/// under a maturity gate, but the always-active fishing predicate
/// reads slot 0 of `commit_in` regardless. Expects `Some(0.0)`
/// (zero-initialised slot 0).
#[test]
fn extract_thermals_committed_reads_slot0_when_seed_zero() {
    let eq_counts = test_support::GeometryDims {
        hydro_count: 0,
        max_par_order: 0,
        n_thermals: 1,
        n_lines: 0,
        n_buses: 0,
        n_blks: 1,
        has_inflow_penalty: false,
        max_deficit_segments: 1,
        n_anticipated: 1,
        lead_stages: 1,
        anticipated_plants: anticipated_plants_at(&[0]),
    };
    let indexer = test_support::geometry(&eq_counts, vec![], &[], vec![]);
    let study_dims = test_support::study_dims_for(&eq_counts);
    let state = test_support::state_layout_full(0, 0, vec![1]);

    let n_cols = indexer.anticipated_decision.end.max(1);
    let primal = vec![0.0_f64; n_cols];
    let obj = vec![0.0_f64; n_cols];

    let counts = EntityCounts {
        hydro_ids: vec![],
        hydro_productivities: vec![],
        thermal_ids: vec![10],
        line_ids: vec![],
        bus_ids: vec![],
        pumping_station_ids: vec![],
        contract_ids: vec![],
        non_controllable_ids: vec![],
    };
    let ec = zero_energy_conversion(0, 2);
    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &[],
            objective: 0.0,
            objective_coeffs: &obj,
            row_lower: &[],
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &indexer,
            hydro_cell_index: &test_support::identity_hydro_cell_index(256),
            state: &state,
            entity_counts: &counts,
            inflow_m3s_per_hydro: &[],
            block_hours: &[],
            generic_constraint_entries: &[],
            ncs_col_upper: &[],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_slots: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            cost_scale_factor: 1_000_000.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[],
            stage_index: 0, // before delivery
            horizon: &HorizonMode::Finite { num_stages: 2 },
            anticipated_windows: &[(None, None)],
            study_stage_ids: &[0, 1, 2, 3, 4, 5],
        },
        0,
    );

    assert_eq!(result.thermals.len(), 1);
    assert_eq!(
        result.thermals[0].anticipated_committed_mw,
        Some(0.0),
        "always-active: reads slot 0 = 0.0 regardless of stage"
    );
}

/// Regression guard: verify that using a pre-built [`HydroReverseLookup`] via
/// `extract_stage_result_with_lookups` produces bit-for-bit identical results
/// to the standard [`extract_stage_result`] path (which builds it internally).
#[test]
fn extract_stage_result_prebuilt_lookup_matches_standard_path() {
    use super::{HydroReverseLookup, extract_stage_result_with_lookups};
    use crate::setup::NodeId;

    let eq_counts = test_support::GeometryDims {
        hydro_count: 0,
        max_par_order: 0,
        n_thermals: 1,
        n_lines: 0,
        n_buses: 0,
        n_blks: 2,
        has_inflow_penalty: false,
        max_deficit_segments: 1,
        n_anticipated: 1,
        lead_stages: 1,
        anticipated_plants: anticipated_plants_at(&[0]),
    };
    let indexer = test_support::geometry(&eq_counts, vec![], &[], vec![]);
    let state = test_support::state_layout_full(0, 0, vec![1]);

    let n_cols = indexer
        .anticipated_decision
        .end
        .max(indexer.thermal.end)
        .max(state.commit_in.end)
        .max(state.theta + 1);
    let mut primal = vec![0.0_f64; n_cols];
    // Slot 0 of commit_in = committed MW scalar.
    primal[state.commit_in.start] = 37.5;
    // Anticipated decision column.
    primal[indexer.anticipated_decision.start] = 80.0;
    // Thermal columns: block 0, block 1.
    if !indexer.thermal.is_empty() {
        primal[indexer.thermal.start] = 40.0;
        primal[indexer.thermal.start + 1] = 50.0;
    }
    let obj = vec![0.01_f64; n_cols];
    let ec = zero_energy_conversion(0, 3);
    let counts = EntityCounts {
        hydro_ids: vec![],
        hydro_productivities: vec![],
        thermal_ids: vec![42],
        line_ids: vec![],
        bus_ids: vec![],
        pumping_station_ids: vec![],
        contract_ids: vec![],
        non_controllable_ids: vec![],
    };
    let view = SolutionView {
        primal: &primal,
        dual: &[],
        objective: 0.0,
        objective_coeffs: &obj,
        row_lower: &[],
    };
    let study_dims = test_support::study_dims_for(&eq_counts);
    let spec = StageExtractionSpec {
        study_dims: &study_dims,
        geometry: &indexer,
        hydro_cell_index: &test_support::identity_hydro_cell_index(256),
        state: &state,
        entity_counts: &counts,
        inflow_m3s_per_hydro: &[],
        block_hours: &[1.0, 1.0],
        generic_constraint_entries: &[],
        ncs_col_upper: &[],
        pumping_consumption_mw_per_m3s: &[],
        contract_prices: &[],
        contract_slots: &[],
        diversion_upstream: &HashMap::new(),
        hydro_productivities: &[],
        col_scale: &[],
        row_scale: &[],
        cumulative_discount_factor: 1.0,
        cost_scale_factor: 1_000_000.0,
        energy_conversion: &ec,
        hydro_min_storage_hm3: &[],
        stage_index: 2, // delivery stage (k_i=1, stage_index=2 > k_i)
        horizon: &HorizonMode::Finite { num_stages: 3 },
        anticipated_windows: &[(None, None)],
        study_stage_ids: &[0, 1, 2, 3, 4, 5],
    };

    let result_standard = extract_stage_result(&view, &spec, 2);

    let hydro_lookup =
        HydroReverseLookup::build(spec.geometry, spec.hydro_cell_index, counts.hydro_ids.len());
    let result_prebuilt = extract_stage_result_with_lookups(
        &view,
        &spec,
        2,
        NodeId(2),
        &hydro_lookup,
        &study_dims.anticipated_plants,
    );

    assert_eq!(
        result_standard.thermals.len(),
        result_prebuilt.thermals.len(),
        "thermal result count must match"
    );
    for (std_t, pre_t) in result_standard
        .thermals
        .iter()
        .zip(result_prebuilt.thermals.iter())
    {
        assert_eq!(
            std_t.is_anticipated, pre_t.is_anticipated,
            "is_anticipated must match"
        );
        assert_eq!(
            std_t.anticipated_committed_mw.map(f64::to_bits),
            pre_t.anticipated_committed_mw.map(f64::to_bits),
            "anticipated_committed_mw bits must match"
        );
        assert_eq!(
            std_t.anticipated_decision_mw.map(f64::to_bits),
            pre_t.anticipated_decision_mw.map(f64::to_bits),
            "anticipated_decision_mw bits must match"
        );
        assert_eq!(
            std_t.generation_mw.to_bits(),
            pre_t.generation_mw.to_bits(),
            "generation_mw bits must match"
        );
    }
}

#[test]
fn extract_optional_entity_types_are_empty_when_absent() {
    let study_dims = test_support::study_dims();
    let state = test_support::state_layout(1, 0);
    let ws_start = state.control_region_start();
    let indexer = StageGeometry {
        withdrawal_slack_neg: ws_start..ws_start + 1,
        withdrawal_slack_pos: ws_start + 1..ws_start + 2,
        ..test_support::equipment_free_geometry(&[0]).remove(0)
    };
    let mut primal = vec![50.0, 0.0, 40.0, 200.0]; // storage, z_inflow, storage_in, theta
    primal.resize(indexer.withdrawal_slack_pos.end, 0.0);
    let objective_coeffs = vec![0.0; primal.len()];
    let dual = vec![];
    let counts = EntityCounts {
        hydro_ids: vec![1],
        hydro_productivities: vec![1.0],
        thermal_ids: vec![],
        line_ids: vec![],
        bus_ids: vec![],
        pumping_station_ids: vec![],
        contract_ids: vec![],
        non_controllable_ids: vec![],
    };

    let ec = zero_energy_conversion(1, 1);
    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &dual,
            objective: 250.0,
            objective_coeffs: &objective_coeffs,
            row_lower: &[],
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &indexer,
            hydro_cell_index: &test_support::identity_hydro_cell_index(256),
            state: &state,
            entity_counts: &counts,
            inflow_m3s_per_hydro: &[0.0; 1],
            block_hours: &[],
            generic_constraint_entries: &[],
            ncs_col_upper: &[],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_slots: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[1.0],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            cost_scale_factor: 1_000_000.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0],
            stage_index: 0,
            horizon: &HorizonMode::Finite { num_stages: 1 },
            anticipated_windows: &[],
            study_stage_ids: &[],
        },
        0,
    );

    assert!(result.pumping_stations.is_empty());
    assert!(result.contracts.is_empty());
    assert!(result.non_controllables.is_empty());
    assert!(result.generic_violations.is_empty());
}

// -------------------------------------------------------------------------
// accumulate_category_costs
// -------------------------------------------------------------------------

// One positional argument per cost category, 1:1 with SimulationCostResult's
// fields; a builder or params struct would just restate the struct itself.
#[allow(clippy::too_many_arguments)]
fn make_cost(
    thermal: f64,
    contract: f64,
    deficit: f64,
    excess: f64,
    storage_violation: f64,
    filling: f64,
    hydro_violation: f64,
    inflow_penalty: f64,
    generic_violation: f64,
    spillage: f64,
    fpha: f64,
    curtailment: f64,
    exchange: f64,
    pumping: f64,
) -> SimulationCostResult {
    SimulationCostResult {
        stage_id: 0,
        block_id: None,
        total_cost: 0.0,
        immediate_cost: 0.0,
        future_cost: 0.0,
        discount_factor: 1.0,
        thermal_cost: thermal,
        // Held at zero so the `resource_cost` expectations (thermal + contract) stay exact.
        anticipated_thermal_cost: 0.0,
        contract_cost: contract,
        deficit_cost: deficit,
        excess_cost: excess,
        storage_violation_cost: storage_violation,
        filling_target_cost: filling,
        hydro_violation_cost: hydro_violation,
        outflow_violation_below_cost: 0.0,
        outflow_violation_above_cost: 0.0,
        turbined_violation_cost: 0.0,
        generation_violation_cost: 0.0,
        evaporation_violation_cost: 0.0,
        withdrawal_violation_cost: 0.0,
        inflow_penalty_cost: inflow_penalty,
        generic_violation_cost: generic_violation,
        spillage_cost: spillage,
        turbined_cost: fpha,
        curtailment_cost: curtailment,
        exchange_cost: exchange,
        pumping_cost: pumping,
    }
}

fn zero_accum() -> ScenarioCategoryCosts {
    ScenarioCategoryCosts {
        resource_cost: 0.0,
        recourse_cost: 0.0,
        violation_cost: 0.0,
        regularization_cost: 0.0,
        imputed_cost: 0.0,
    }
}

#[test]
fn accumulate_single_stage_all_categories() {
    let cost = make_cost(
        400.0, 100.0, // resource: 500
        50.0, 10.0, // recourse: 60
        20.0, 30.0, 5.0, 3.0, 2.0, // violation: 60
        1.0, 4.0, 7.0, 8.0,  // regularization: 20
        60.0, // imputed: 60
    );
    let mut accum = zero_accum();
    accumulate_category_costs(&cost, &mut accum);

    assert_eq!(accum.resource_cost, 500.0);
    assert_eq!(accum.recourse_cost, 60.0);
    assert_eq!(accum.violation_cost, 60.0);
    assert_eq!(accum.regularization_cost, 20.0);
    assert_eq!(accum.imputed_cost, 60.0);
}

#[test]
fn accumulate_two_consecutive_stages_sums_correctly() {
    let cost1 = make_cost(
        100.0, 0.0, // resource
        10.0, 0.0, // recourse
        0.0, 0.0, 0.0, 0.0, 0.0, // violation
        0.0, 0.0, 0.0, 0.0, // regularization
        5.0, // imputed
    );
    let cost2 = make_cost(
        200.0, 50.0, // resource
        20.0, 5.0, // recourse
        0.0, 0.0, 0.0, 0.0, 0.0, // violation
        0.0, 0.0, 0.0, 0.0,  // regularization
        10.0, // imputed
    );
    let mut accum = zero_accum();
    accumulate_category_costs(&cost1, &mut accum);
    accumulate_category_costs(&cost2, &mut accum);

    assert_eq!(accum.resource_cost, 100.0 + 200.0 + 50.0);
    assert_eq!(accum.recourse_cost, 10.0 + 20.0 + 5.0);
    assert_eq!(accum.violation_cost, 0.0);
    assert_eq!(accum.regularization_cost, 0.0);
    assert_eq!(accum.imputed_cost, 5.0 + 10.0);
}

#[test]
fn accumulate_all_zeros_leaves_accum_unchanged() {
    let cost = make_cost(
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    );
    let mut accum = ScenarioCategoryCosts {
        resource_cost: 1.0,
        recourse_cost: 2.0,
        violation_cost: 3.0,
        regularization_cost: 4.0,
        imputed_cost: 5.0,
    };
    accumulate_category_costs(&cost, &mut accum);

    assert_eq!(accum.resource_cost, 1.0);
    assert_eq!(accum.recourse_cost, 2.0);
    assert_eq!(accum.violation_cost, 3.0);
    assert_eq!(accum.regularization_cost, 4.0);
    assert_eq!(accum.imputed_cost, 5.0);
}

#[test]
fn accumulate_violation_all_five_components() {
    let cost = make_cost(
        0.0, 0.0, // resource
        0.0, 0.0, // recourse
        1.0, 2.0, 3.0, 4.0, 5.0, // violation: 15
        0.0, 0.0, 0.0, 0.0, // regularization
        0.0, // imputed
    );
    let mut accum = zero_accum();
    accumulate_category_costs(&cost, &mut accum);

    assert_eq!(accum.violation_cost, 15.0);
}

#[test]
fn accumulate_regularization_all_four_components() {
    let cost = make_cost(
        0.0, 0.0, // resource
        0.0, 0.0, // recourse
        0.0, 0.0, 0.0, 0.0, 0.0, // violation
        2.0, 3.0, 4.0, 5.0, // regularization: 14
        0.0, // imputed
    );
    let mut accum = zero_accum();
    accumulate_category_costs(&cost, &mut accum);

    assert_eq!(accum.regularization_cost, 14.0);
}

// ── test_slack_extraction_in_simulation ──────────────────────────────────

/// Verify that `inflow_nonnegativity_slack_m3s` is read from the primal
/// solution when `has_inflow_penalty == true`.
///
/// Column layout for N=2 hydros, L=1 lag, T=1 thermal, Ln=1 line, B=1 bus,
/// K=1 block, with penalty method active:
///
/// theta=6, `turbine`=[7,9), `spillage`=[9,11), `thermal`=[11,12),
/// `line_fwd`=[12,13), `line_rev`=[13,14), `deficit`=[14,15), `excess`=[15,16),
/// `inflow_slack`=[16,18)
#[test]
fn test_slack_extraction_with_penalty_active() {
    // N=2, L=1, T=1, Ln=1, B=1, K=1, has_inflow_penalty=true
    let eq_counts = test_support::GeometryDims {
        hydro_count: 2,
        max_par_order: 1,
        n_thermals: 1,
        n_lines: 1,
        n_buses: 1,
        n_blks: 1,
        has_inflow_penalty: true,
        max_deficit_segments: 1,
        n_anticipated: 0,
        lead_stages: 0,
        anticipated_plants: AnticipatedPlants::default(),
    };
    let indexer = test_support::geometry(&eq_counts, vec![], &[], vec![]);
    let study_dims = test_support::study_dims_for(&eq_counts);
    let state = test_support::state_layout_full(2, 1, vec![]);

    assert!(
        !indexer.inflow_slack.is_empty(),
        "inflow_slack must be non-empty"
    );

    let n_cols = indexer.generation_below_slack.end;
    let mut primal = vec![0.0_f64; n_cols];

    primal[0] = 100.0; // storage h0
    primal[1] = 200.0; // storage h1
    primal[2] = 50.0; // lag h0
    primal[3] = 60.0; // lag h1
    primal[4] = 90.0; // storage_in h0
    primal[5] = 180.0; // storage_in h1
    primal[6] = 500.0; // theta

    // Inflow slack values: hydro 0 has slack 7.5, hydro 1 has slack 0.0
    primal[indexer.inflow_slack.start] = 7.5; // slack h0
    primal[indexer.inflow_slack.start + 1] = 0.0; // slack h1

    let obj = vec![0.0_f64; n_cols];
    let dual = vec![0.0_f64; geometry_row_capacity(&indexer)];
    let row_lower = vec![0.0_f64; geometry_row_capacity(&indexer)];

    let counts = EntityCounts {
        hydro_ids: vec![10, 20],
        hydro_productivities: vec![1.0, 1.0],
        thermal_ids: vec![1],
        line_ids: vec![5],
        bus_ids: vec![100],
        pumping_station_ids: vec![],
        contract_ids: vec![],
        non_controllable_ids: vec![],
    };

    let ec = zero_energy_conversion(2, 1);
    let inflow_m3s_per_hydro = test_support::inflow_m3s_per_hydro_from_primal(&state, &primal, 2);
    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &dual,
            objective: 500.0,
            objective_coeffs: &obj,
            row_lower: &row_lower,
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &indexer,
            hydro_cell_index: &test_support::identity_hydro_cell_index(256),
            state: &state,
            entity_counts: &counts,
            inflow_m3s_per_hydro: &inflow_m3s_per_hydro,
            block_hours: &[100.0],
            generic_constraint_entries: &[],
            ncs_col_upper: &[],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_slots: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[1.0, 1.0],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            cost_scale_factor: 1_000_000.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0; 2],
            stage_index: 0,
            horizon: &HorizonMode::Finite { num_stages: 1 },
            anticipated_windows: &[],
            study_stage_ids: &[],
        },
        0,
    );

    // turbine is non-empty → per-(hydro, block) results, so 2 hydros × 1 block = 2 entries
    assert_eq!(result.hydros.len(), 2);

    // Slack for hydro 0 must equal the primal slack column value
    assert!(
        (result.hydros[0].inflow_nonnegativity_slack_m3s - 7.5).abs() < 1e-12,
        "hydro 0 slack should be 7.5, got {}",
        result.hydros[0].inflow_nonnegativity_slack_m3s
    );

    // Slack for hydro 1 must be 0.0
    assert_eq!(
        result.hydros[1].inflow_nonnegativity_slack_m3s, 0.0,
        "hydro 1 slack should be 0.0"
    );
}

/// Verify that `inflow_nonnegativity_slack_m3s` is zero when the penalty
/// method is inactive (`has_inflow_penalty == false`).
#[test]
fn test_slack_extraction_without_penalty_is_zero() {
    // N=2, L=1, T=1, Ln=1, B=1, K=1, has_inflow_penalty=false
    let eq_counts = test_support::GeometryDims {
        hydro_count: 2,
        max_par_order: 1,
        n_thermals: 1,
        n_lines: 1,
        n_buses: 1,
        n_blks: 1,
        has_inflow_penalty: false,
        max_deficit_segments: 1,
        n_anticipated: 0,
        lead_stages: 0,
        anticipated_plants: AnticipatedPlants::default(),
    };
    let indexer = test_support::geometry(&eq_counts, vec![], &[], vec![]);
    let study_dims = test_support::study_dims_for(&eq_counts);
    let state = test_support::state_layout_full(2, 1, vec![]);
    assert!(
        indexer.inflow_slack.is_empty(),
        "inflow_slack must be empty"
    );

    let n_cols = indexer.generation_below_slack.end; // includes withdrawal_slack columns
    let primal = vec![1.0_f64; n_cols]; // all ones
    let obj = vec![0.0_f64; n_cols];
    let dual = vec![0.0_f64; geometry_row_capacity(&indexer)];
    let row_lower = vec![0.0_f64; geometry_row_capacity(&indexer)];

    let counts = EntityCounts {
        hydro_ids: vec![10, 20],
        hydro_productivities: vec![1.0, 1.0],
        thermal_ids: vec![1],
        line_ids: vec![5],
        bus_ids: vec![100],
        pumping_station_ids: vec![],
        contract_ids: vec![],
        non_controllable_ids: vec![],
    };

    let ec = zero_energy_conversion(2, 1);
    let inflow_m3s_per_hydro = test_support::inflow_m3s_per_hydro_from_primal(&state, &primal, 2);
    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &dual,
            objective: 1.0,
            objective_coeffs: &obj,
            row_lower: &row_lower,
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &indexer,
            hydro_cell_index: &test_support::identity_hydro_cell_index(256),
            state: &state,
            entity_counts: &counts,
            inflow_m3s_per_hydro: &inflow_m3s_per_hydro,
            block_hours: &[100.0],
            generic_constraint_entries: &[],
            ncs_col_upper: &[],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_slots: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[1.0, 1.0],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            cost_scale_factor: 1_000_000.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0; 2],
            stage_index: 0,
            horizon: &HorizonMode::Finite { num_stages: 1 },
            anticipated_windows: &[],
            study_stage_ids: &[],
        },
        0,
    );

    for (h, hr) in result.hydros.iter().enumerate() {
        assert_eq!(
            hr.inflow_nonnegativity_slack_m3s, 0.0,
            "hydro {h} slack must be 0.0 when penalty is inactive"
        );
    }
}

// ── FPHA and Evaporation extraction tests ────────────────────────────────

/// Build a `StageGeometry` with 2 hydros (h0 = FPHA, h1 = constant-productivity),
/// 1 block, no thermals/lines/buses.
///
/// Column layout:
/// ```text
/// N=2, L=0, T=0, Ln=0, B=0, K=1, penalty=false, fpha=[0], planes=[2]
/// theta = N*(3+L) = 2*(3+0) = 6
/// turbine:    [7, 9)    h0→7, h1→8
/// spillage:   [9, 11)   h0→9, h1→10
/// diversion: [11, 13)   h0→11, h1→12
/// generation:[13, 14)   fpha h0 b0 → 13
/// ```
/// The `GeometryDims` both [`make_indexer_2h_1fpha_1blk`] and the matching
/// `study_dims` derive from, keeping the role-(b) geometry and the non-state
/// `StudyDimensions` aligned from one source.
fn counts_2h_1fpha_1blk() -> test_support::GeometryDims {
    test_support::GeometryDims {
        hydro_count: 2,
        n_blks: 1,
        ..Default::default()
    }
}

fn make_indexer_2h_1fpha_1blk() -> StageGeometry {
    // h0 is FPHA (system index 0), h1 is constant-productivity (system index 1)
    test_support::geometry(&counts_2h_1fpha_1blk(), vec![0], &[2], vec![])
}

/// Acceptance criterion: FPHA hydro's `generation_mw` equals the LP generation
/// variable (not turbined * productivity = 0).
#[test]
fn fpha_generation_read_from_lp_column() {
    let indexer = make_indexer_2h_1fpha_1blk();
    let study_dims = test_support::study_dims_for(&counts_2h_1fpha_1blk());
    let state = test_support::state_layout(2, 0);
    // generation.start should be after turbine(7..9) + spillage(9..11) + diversion(11..13) = 13
    // generation[0] = generation.start + 0 * 1 + 0 = 13
    assert_eq!(indexer.generation.start, 13, "generation starts at 13");
    assert_eq!(indexer.fpha_hydro_indices, vec![HydroSys::new(0)]);

    let n_cols = indexer.generation_below_slack.end;
    let mut primal = vec![0.0_f64; n_cols];
    primal[0] = 50.0; // storage h0
    primal[1] = 80.0; // storage h1
    // primal[2..4] = z_inflow (zeros)
    primal[4] = 45.0; // storage_in h0
    primal[5] = 75.0; // storage_in h1
    primal[6] = 0.0; // theta
    primal[7] = 20.0; // turbine h0 b0 (not used for FPHA gen)
    primal[8] = 30.0; // turbine h1 b0
    primal[9] = 0.0; // spillage h0 b0
    primal[10] = 0.0; // spillage h1 b0
    // primal[11..13] = diversion (zeros)
    primal[13] = 75.0; // FPHA generation h0 b0 — acceptance criterion value

    let obj = vec![0.0_f64; n_cols];
    let dual = vec![0.0_f64; geometry_row_capacity(&indexer)];
    let row_lower = vec![0.0_f64; geometry_row_capacity(&indexer)];

    let counts = EntityCounts {
        hydro_ids: vec![1, 2],
        hydro_productivities: vec![0.0, 1.5], // FPHA has 0.0, constant has 1.5
        thermal_ids: vec![],
        line_ids: vec![],
        bus_ids: vec![],
        pumping_station_ids: vec![],
        contract_ids: vec![],
        non_controllable_ids: vec![],
    };

    let ec = zero_energy_conversion(2, 1);
    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &dual,
            objective: 0.0,
            objective_coeffs: &obj,
            row_lower: &row_lower,
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &indexer,
            hydro_cell_index: &test_support::identity_hydro_cell_index(256),
            state: &state,
            entity_counts: &counts,
            inflow_m3s_per_hydro: &[0.0; 2],
            block_hours: &[],
            generic_constraint_entries: &[],
            ncs_col_upper: &[],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_slots: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[0.0, 1.5],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            cost_scale_factor: 1_000_000.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0; 2],
            stage_index: 0,
            horizon: &HorizonMode::Finite { num_stages: 1 },
            anticipated_windows: &[],
            study_stage_ids: &[],
        },
        0,
    );

    // 2 hydros × 1 block = 2 entries
    assert_eq!(result.hydros.len(), 2);

    // FPHA hydro (h0, block 0): generation from LP column 13 = 75.0
    assert!(
        (result.hydros[0].generation_mw - 75.0).abs() < 1e-12,
        "FPHA generation_mw should be 75.0, got {}",
        result.hydros[0].generation_mw
    );

    // Constant-productivity hydro (h1, block 0): generation = turbined * productivity
    // turbine h1 b0 = primal[8] = 30.0, productivity = 1.5 → 45.0
    assert!(
        (result.hydros[1].generation_mw - 45.0).abs() < 1e-12,
        "constant-productivity generation_mw should be 45.0, got {}",
        result.hydros[1].generation_mw
    );
}

/// Acceptance criterion: both FPHA and constant-productivity hydros report
/// the `equivalent_productivity_mw_per_m3s` produced by the supplied
/// `EnergyConversionSet`. With a zero-valued set the field is `0.0` for
/// every hydro regardless of generation model.
#[test]
fn fpha_productivity_placeholder_zero() {
    let indexer = make_indexer_2h_1fpha_1blk();
    let study_dims = test_support::study_dims_for(&counts_2h_1fpha_1blk());
    let state = test_support::state_layout(2, 0);
    let n_cols = indexer.generation_below_slack.end;
    let primal = vec![0.0_f64; n_cols];
    let obj = vec![0.0_f64; n_cols];
    let dual = vec![0.0_f64; geometry_row_capacity(&indexer)];
    let row_lower = vec![0.0_f64; geometry_row_capacity(&indexer)];

    let counts = EntityCounts {
        hydro_ids: vec![1, 2],
        hydro_productivities: vec![0.0, 1.5],
        thermal_ids: vec![],
        line_ids: vec![],
        bus_ids: vec![],
        pumping_station_ids: vec![],
        contract_ids: vec![],
        non_controllable_ids: vec![],
    };

    let ec = zero_energy_conversion(2, 1);
    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &dual,
            objective: 0.0,
            objective_coeffs: &obj,
            row_lower: &row_lower,
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &indexer,
            hydro_cell_index: &test_support::identity_hydro_cell_index(256),
            state: &state,
            entity_counts: &counts,
            inflow_m3s_per_hydro: &[0.0; 2],
            block_hours: &[],
            generic_constraint_entries: &[],
            ncs_col_upper: &[],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_slots: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[0.0, 1.5],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            cost_scale_factor: 1_000_000.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0; 2],
            stage_index: 0,
            horizon: &HorizonMode::Finite { num_stages: 1 },
            anticipated_windows: &[],
            study_stage_ids: &[],
        },
        0,
    );

    // Both hydros report the values supplied by the zero-valued EnergyConversionSet.
    assert_eq!(
        result.hydros[0].equivalent_productivity_mw_per_m3s, 0.0,
        "FPHA hydro must carry placeholder 0.0 for equivalent_productivity_mw_per_m3s"
    );
    assert_eq!(
        result.hydros[1].equivalent_productivity_mw_per_m3s, 0.0,
        "constant-productivity hydro must carry placeholder 0.0 for equivalent_productivity_mw_per_m3s"
    );
}

/// Build a `StageGeometry` with 1 hydro that has evaporation, 1 block.
///
/// Column layout:
/// ```text
/// N=1, L=0, T=0, Ln=0, B=0, K=1, penalty=false, fpha=[], evap=[0]
/// theta = 1*(3+0) = 3
/// turbine:    [4, 5)   h0→4
/// spillage:   [5, 6)   h0→5
/// diversion:  [6, 7)   h0→6
/// evap:       [7, 10)  evaporation_flow→7, f_plus→8, f_minus→9
/// ```
/// The `GeometryDims` both [`make_indexer_1h_evap_1blk`] and the matching
/// `study_dims` derive from, keeping the role-(b) geometry and the non-state
/// `StudyDimensions` aligned from one source.
fn counts_1h_evap_1blk() -> test_support::GeometryDims {
    test_support::GeometryDims {
        hydro_count: 1,
        n_blks: 1,
        ..Default::default()
    }
}

fn make_indexer_1h_evap_1blk() -> StageGeometry {
    test_support::geometry(&counts_1h_evap_1blk(), vec![], &[], vec![0])
}

/// Acceptance criterion: `evaporation_m3s` equals the LP evaporation-outflow variable value.
#[test]
fn evaporation_read_from_lp_column() {
    let indexer = make_indexer_1h_evap_1blk();
    let study_dims = test_support::study_dims_for(&counts_1h_evap_1blk());
    let state = test_support::state_layout(1, 0);
    assert_eq!(indexer.evap_hydro_indices, vec![HydroSys::new(0)]);
    let ei = &indexer.evap_indices[0];
    assert_eq!(ei.evaporation_flow_col, 7);
    assert_eq!(ei.f_evap_plus_col, 8);
    assert_eq!(ei.f_evap_minus_col, 9);

    let n_cols = indexer.generation_below_slack.end;
    let mut primal = vec![0.0_f64; n_cols];
    primal[0] = 200.0; // storage h0
    // primal[1] = z_inflow h0 (zero)
    primal[2] = 190.0; // storage_in h0
    primal[3] = 0.0; // theta
    primal[4] = 10.0; // turbine h0 b0
    primal[5] = 0.0; // spillage h0 b0
    // primal[6] = diversion h0 b0 (zero)
    primal[7] = 3.5; // evaporation outflow — acceptance criterion value

    let obj = vec![0.0_f64; n_cols];
    let dual = vec![0.0_f64; geometry_row_capacity(&indexer)];
    let row_lower = vec![0.0_f64; geometry_row_capacity(&indexer)];

    let counts = EntityCounts {
        hydro_ids: vec![1],
        hydro_productivities: vec![1.0],
        thermal_ids: vec![],
        line_ids: vec![],
        bus_ids: vec![],
        pumping_station_ids: vec![],
        contract_ids: vec![],
        non_controllable_ids: vec![],
    };

    let ec = zero_energy_conversion(1, 1);
    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &dual,
            objective: 0.0,
            objective_coeffs: &obj,
            row_lower: &row_lower,
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &indexer,
            hydro_cell_index: &test_support::identity_hydro_cell_index(256),
            state: &state,
            entity_counts: &counts,
            inflow_m3s_per_hydro: &[0.0; 1],
            block_hours: &[],
            generic_constraint_entries: &[],
            ncs_col_upper: &[],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_slots: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[1.0],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            cost_scale_factor: 1_000_000.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0],
            stage_index: 0,
            horizon: &HorizonMode::Finite { num_stages: 1 },
            anticipated_windows: &[],
            study_stage_ids: &[],
        },
        0,
    );

    assert_eq!(result.hydros.len(), 1);
    assert_eq!(
        result.hydros[0].evaporation_m3s,
        Some(3.5),
        "evaporation_m3s should be Some(3.5)"
    );
    assert!(
        result.hydros[0].evaporation_violation_pos_m3s.abs() < 1e-12,
        "evaporation_violation_pos_m3s should be 0.0"
    );
}

/// Acceptance criterion: directional evaporation violations are extracted
/// separately from the LP's `f_evap_plus` (under-evaporation / neg) and
/// `f_evap_minus` (over-evaporation / pos) columns.
#[test]
fn evaporation_violation_is_sum_of_slacks() {
    let indexer = make_indexer_1h_evap_1blk();
    let study_dims = test_support::study_dims_for(&counts_1h_evap_1blk());
    let state = test_support::state_layout(1, 0);
    let n_cols = indexer.generation_below_slack.end;
    let mut primal = vec![0.0_f64; n_cols];
    primal[0] = 200.0;
    // primal[1] = z_inflow h0 (zero)
    primal[2] = 190.0; // storage_in h0
    // primal[3] = theta = 0
    // primal[6] = diversion h0 b0 (zero)
    primal[7] = 2.0; // evaporation outflow
    primal[8] = 0.5; // f_evap_plus (under-evaporation -> neg)
    primal[9] = 0.0; // f_evap_minus (over-evaporation -> pos)

    let obj = vec![0.0_f64; n_cols];
    let dual = vec![0.0_f64; geometry_row_capacity(&indexer)];
    let row_lower = vec![0.0_f64; geometry_row_capacity(&indexer)];

    let counts = EntityCounts {
        hydro_ids: vec![1],
        hydro_productivities: vec![1.0],
        thermal_ids: vec![],
        line_ids: vec![],
        bus_ids: vec![],
        pumping_station_ids: vec![],
        contract_ids: vec![],
        non_controllable_ids: vec![],
    };

    let ec = zero_energy_conversion(1, 1);
    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &dual,
            objective: 0.0,
            objective_coeffs: &obj,
            row_lower: &row_lower,
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &indexer,
            hydro_cell_index: &test_support::identity_hydro_cell_index(256),
            state: &state,
            entity_counts: &counts,
            inflow_m3s_per_hydro: &[0.0; 1],
            block_hours: &[],
            generic_constraint_entries: &[],
            ncs_col_upper: &[],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_slots: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[1.0],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            cost_scale_factor: 1_000_000.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0],
            stage_index: 0,
            horizon: &HorizonMode::Finite { num_stages: 1 },
            anticipated_windows: &[],
            study_stage_ids: &[],
        },
        0,
    );

    // f_evap_plus (primal[8] = 0.5) maps to under-evaporation (neg).
    assert!(
        (result.hydros[0].evaporation_violation_neg_m3s - 0.5).abs() < 1e-12,
        "evaporation_violation_neg_m3s should be 0.5, got {}",
        result.hydros[0].evaporation_violation_neg_m3s
    );
    // f_evap_minus (primal[9] = 0.0) maps to over-evaporation (pos).
    assert!(
        result.hydros[0].evaporation_violation_pos_m3s.abs() < 1e-12,
        "evaporation_violation_pos_m3s should be 0.0, got {}",
        result.hydros[0].evaporation_violation_pos_m3s
    );
}

/// Acceptance criterion: `turbined_cost` equals the sum of primal * obj\_coeff
/// over turbine columns (every hydro, every block), multiplied by
/// `COST_SCALE_FACTOR`.
///
/// Setup: 2 hydros, 1 block. h0 turbine column: primal=30.0,
/// `objective_coeff`=0.01 → `scaled_cost`=0.3 → unscaled=300.0.
#[test]
fn turbined_cost_in_compute_cost_result() {
    let indexer = make_indexer_2h_1fpha_1blk();
    let study_dims = test_support::study_dims_for(&counts_2h_1fpha_1blk());
    let state = test_support::state_layout(2, 0);
    // turbine.start = 7 (h0 b0)
    let n_cols = indexer.generation_below_slack.end;
    let mut primal = vec![0.0_f64; n_cols];
    primal[6] = 500.0; // theta (at N*(3+L) = 2*3 = 6)

    // h0 turbine column 7: primal=30.0
    primal[7] = 30.0;

    let mut obj = vec![0.0_f64; n_cols];
    obj[6] = 1.0; // theta coefficient (undiscounted)
    // h0 turbine column 7: objective_coeff=0.01
    obj[7] = 0.01;

    let dual = vec![0.0_f64; geometry_row_capacity(&indexer)];
    let row_lower = vec![0.0_f64; geometry_row_capacity(&indexer)];

    let counts = EntityCounts {
        hydro_ids: vec![1, 2],
        hydro_productivities: vec![0.0, 1.5],
        thermal_ids: vec![],
        line_ids: vec![],
        bus_ids: vec![],
        pumping_station_ids: vec![],
        contract_ids: vec![],
        non_controllable_ids: vec![],
    };

    let ec = zero_energy_conversion(2, 1);
    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &dual,
            objective: 500.3, // theta + fpha cost
            objective_coeffs: &obj,
            row_lower: &row_lower,
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &indexer,
            hydro_cell_index: &test_support::identity_hydro_cell_index(256),
            state: &state,
            entity_counts: &counts,
            inflow_m3s_per_hydro: &[0.0; 2],
            block_hours: &[],
            generic_constraint_entries: &[],
            ncs_col_upper: &[],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_slots: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[0.0, 1.5],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            cost_scale_factor: 1_000_000.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0; 2],
            stage_index: 0,
            horizon: &HorizonMode::Finite { num_stages: 1 },
            anticipated_windows: &[],
            study_stage_ids: &[],
        },
        0,
    );

    let cost = &result.costs[0];
    assert!(
        (cost.turbined_cost - 300_000.0).abs() < 1e-9,
        "turbined_cost should be 300_000.0 (30.0 * 0.01 * COST_SCALE_FACTOR), got {}",
        cost.turbined_cost
    );
}

/// Verify that per-component cost breakdown sums to `immediate_cost` under
/// identity `col_scale`.
///
/// Setup: 2 hydros, 1 block. h0 turbine col 7: primal=30, `obj_coeff`=0.01.
/// Objective = `theta_scaled` + `turbine_scaled` = 500 + (30 * 0.01) = 500.3.
/// `immediate_cost` = (500.3 - 500) * `1_000_000` = `300_000`.
/// Per-component sum = `turbined_cost` = 30 * 0.01 * `1_000_000` = `300_000` (matches).
#[test]
fn cost_breakdown_sums_to_immediate_identity_scale() {
    let indexer = make_indexer_2h_1fpha_1blk();
    let study_dims = test_support::study_dims_for(&counts_2h_1fpha_1blk());
    let state = test_support::state_layout(2, 0);
    let n_cols = indexer.generation_below_slack.end;
    let mut primal = vec![0.0_f64; n_cols];
    primal[6] = 500.0; // theta
    primal[7] = 30.0; // h0 turbine column

    let mut obj = vec![0.0_f64; n_cols];
    obj[6] = 1.0; // theta coefficient (undiscounted)
    obj[7] = 0.01; // turbined cost (scaled)
    // objective in scaled space = theta_coeff * theta + turbine_coeff * turbine
    //                           = 1.0 * 500 + 0.01 * 30 = 500.3
    let objective_val = 500.3_f64;

    let dual = vec![0.0_f64; geometry_row_capacity(&indexer)];
    let row_lower = vec![0.0_f64; geometry_row_capacity(&indexer)];

    let counts = EntityCounts {
        hydro_ids: vec![1, 2],
        hydro_productivities: vec![0.0, 1.5],
        thermal_ids: vec![],
        line_ids: vec![],
        bus_ids: vec![],
        pumping_station_ids: vec![],
        contract_ids: vec![],
        non_controllable_ids: vec![],
    };

    let ec = zero_energy_conversion(2, 1);
    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &dual,
            objective: objective_val,
            objective_coeffs: &obj,
            row_lower: &row_lower,
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &indexer,
            hydro_cell_index: &test_support::identity_hydro_cell_index(256),
            state: &state,
            entity_counts: &counts,
            inflow_m3s_per_hydro: &[0.0; 2],
            block_hours: &[],
            generic_constraint_entries: &[],
            ncs_col_upper: &[],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_slots: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[0.0, 1.5],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            cost_scale_factor: 1_000_000.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0; 2],
            stage_index: 0,
            horizon: &HorizonMode::Finite { num_stages: 1 },
            anticipated_windows: &[],
            study_stage_ids: &[],
        },
        0,
    );

    let cost = &result.costs[0];
    // immediate_cost = (obj - theta) * K = (500.3 - 500.0) * 1_000_000 = 300_000.0
    assert!(
        (cost.immediate_cost - 300_000.0).abs() < 1.0,
        "immediate_cost should be 300_000.0, got {}",
        cost.immediate_cost
    );

    // Per-component sum: only turbined_cost is non-zero.
    let component_sum = cost.thermal_cost
        + cost.deficit_cost
        + cost.excess_cost
        + cost.exchange_cost
        + cost.spillage_cost
        + cost.generic_violation_cost
        + cost.turbined_cost
        + cost.curtailment_cost;

    assert!(
        (component_sum - cost.immediate_cost).abs() < 1.0,
        "per-component cost sum ({component_sum}) must equal immediate_cost ({})",
        cost.immediate_cost
    );
    assert_eq!(
        cost.contract_cost, 0.0,
        "contract-free stage books zero contract_cost"
    );
}

/// `contract_cost` for an active import contract equals `price * power * hours`.
/// The scaled LP objective coeff is `price * hours / COST_SCALE_FACTOR`; the
/// extractor recovers the original-unit cost. Exercises the geometry-range
/// cost path in `compute_cost_result` (`contract_ids: vec![]`), NOT the
/// `extract_contracts` primal read — `d41_energy_contracts_simulation` covers
/// that end to end.
#[test]
fn contract_cost_active_import_equals_price_power_hours_via_cost_result() {
    let indexer = make_indexer_2h_1fpha_1blk();
    let study_dims = test_support::study_dims_for(&counts_2h_1fpha_1blk());
    let state = test_support::state_layout(2, 0);
    let contract_col = indexer.generation_below_slack.end;
    let geometry = StageGeometry {
        contract_import: contract_col..contract_col + 1,
        contract_export: contract_col + 1..contract_col + 1,
        ..indexer
    };

    let n_cols = contract_col + 1;
    let mut primal = vec![0.0_f64; n_cols];
    primal[6] = 500.0; // theta
    primal[contract_col] = 40.0; // import power (MW)

    let mut obj = vec![0.0_f64; n_cols];
    obj[6] = 1.0;
    // scaled objective coeff = price * hours / COST_SCALE_FACTOR = 200 * 730 / 1e6
    obj[contract_col] = 200.0 * 730.0 / 1_000_000.0;
    // objective = theta + contract = 500 + 40 * 0.146 = 505.84
    let objective_val = 500.0 + 40.0 * (200.0 * 730.0 / 1_000_000.0);

    let dual = vec![0.0_f64; geometry_row_capacity(&geometry)];
    let row_lower = vec![0.0_f64; geometry_row_capacity(&geometry)];
    let counts = EntityCounts {
        hydro_ids: vec![1, 2],
        hydro_productivities: vec![0.0, 1.5],
        thermal_ids: vec![],
        line_ids: vec![],
        bus_ids: vec![],
        pumping_station_ids: vec![],
        contract_ids: vec![],
        non_controllable_ids: vec![],
    };
    let ec = zero_energy_conversion(2, 1);

    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &dual,
            objective: objective_val,
            objective_coeffs: &obj,
            row_lower: &row_lower,
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &geometry,
            hydro_cell_index: &test_support::identity_hydro_cell_index(256),
            state: &state,
            entity_counts: &counts,
            inflow_m3s_per_hydro: &[0.0; 2],
            block_hours: &[],
            generic_constraint_entries: &[],
            ncs_col_upper: &[],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_slots: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[0.0, 1.5],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            cost_scale_factor: 1_000_000.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0; 2],
            stage_index: 0,
            horizon: &HorizonMode::Finite { num_stages: 1 },
            anticipated_windows: &[],
            study_stage_ids: &[],
        },
        0,
    );

    let cost = &result.costs[0];
    assert!(
        (cost.contract_cost - 200.0 * 40.0 * 730.0).abs() < 1.0,
        "contract_cost should be 200 * 40 * 730, got {}",
        cost.contract_cost
    );
}

/// Σ(macro categories) == `immediate_cost` when a contract is active — the
/// load-bearing invariant. An export contract (negative price) nets negative
/// `contract_cost` (revenue), so the sum still matches `immediate_cost`.
/// Exercises the geometry-range cost path in `compute_cost_result`
/// (`contract_ids: vec![]`), NOT the `extract_contracts` primal read —
/// `d41_energy_contracts_simulation` covers that end to end.
#[test]
fn cost_breakdown_sums_to_immediate_with_active_export_contract_via_cost_result() {
    let indexer = make_indexer_2h_1fpha_1blk();
    let study_dims = test_support::study_dims_for(&counts_2h_1fpha_1blk());
    let state = test_support::state_layout(2, 0);
    let export_col = indexer.generation_below_slack.end;
    let geometry = StageGeometry {
        contract_import: export_col..export_col,
        contract_export: export_col..export_col + 1,
        ..indexer
    };

    let n_cols = export_col + 1;
    let mut primal = vec![0.0_f64; n_cols];
    primal[6] = 500.0; // theta
    primal[7] = 30.0; // h0 turbine column
    primal[export_col] = 30.0; // export power (MW)

    let mut obj = vec![0.0_f64; n_cols];
    obj[6] = 1.0;
    obj[7] = 0.01; // turbined cost (scaled)
    // export price < 0 (revenue): scaled coeff = -150 * 730 / 1e6
    obj[export_col] = -150.0 * 730.0 / 1_000_000.0;
    let objective_val = 500.0 + 30.0 * 0.01 + 30.0 * (-150.0 * 730.0 / 1_000_000.0);

    let dual = vec![0.0_f64; geometry_row_capacity(&geometry)];
    let row_lower = vec![0.0_f64; geometry_row_capacity(&geometry)];
    let counts = EntityCounts {
        hydro_ids: vec![1, 2],
        hydro_productivities: vec![0.0, 1.5],
        thermal_ids: vec![],
        line_ids: vec![],
        bus_ids: vec![],
        pumping_station_ids: vec![],
        contract_ids: vec![],
        non_controllable_ids: vec![],
    };
    let ec = zero_energy_conversion(2, 1);

    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &dual,
            objective: objective_val,
            objective_coeffs: &obj,
            row_lower: &row_lower,
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &geometry,
            hydro_cell_index: &test_support::identity_hydro_cell_index(256),
            state: &state,
            entity_counts: &counts,
            inflow_m3s_per_hydro: &[0.0; 2],
            block_hours: &[],
            generic_constraint_entries: &[],
            ncs_col_upper: &[],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_slots: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[0.0, 1.5],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            cost_scale_factor: 1_000_000.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0; 2],
            stage_index: 0,
            horizon: &HorizonMode::Finite { num_stages: 1 },
            anticipated_windows: &[],
            study_stage_ids: &[],
        },
        0,
    );

    let cost = &result.costs[0];
    assert!(
        cost.contract_cost < 0.0,
        "export contract nets negative contract_cost (revenue), got {}",
        cost.contract_cost
    );
    assert!(
        (cost.contract_cost - (-150.0 * 30.0 * 730.0)).abs() < 1.0,
        "contract_cost should be -150 * 30 * 730, got {}",
        cost.contract_cost
    );

    let mut accum = zero_accum();
    accumulate_category_costs(cost, &mut accum);
    let macro_sum = accum.resource_cost
        + accum.recourse_cost
        + accum.violation_cost
        + accum.regularization_cost
        + accum.imputed_cost;
    assert!(
        (macro_sum - cost.immediate_cost).abs() < 1.0,
        "Σ(macro categories) ({macro_sum}) must equal immediate_cost ({}) with an active contract",
        cost.immediate_cost
    );
}

/// Verify that per-component costs are correctly unscaled when non-trivial
/// `col_scale` is applied.
///
/// With `col_scale` = 2.0 on the h0 turbine column:
/// - `obj_coeff` in template = `c_orig` * `col_scale` / K = 0.005 * 2.0 = 0.01
/// - After unscaling: cost = primal * `obj_coeff` / `col_scale` * K = 30 * 0.01 / 2.0 * `1_000_000` = `150_000`.
#[test]
fn cost_unscaled_by_col_scale() {
    let indexer = make_indexer_2h_1fpha_1blk();
    let study_dims = test_support::study_dims_for(&counts_2h_1fpha_1blk());
    let state = test_support::state_layout(2, 0);
    let n_cols = indexer.generation_below_slack.end;
    let mut primal = vec![0.0_f64; n_cols];
    primal[6] = 500.0; // theta
    primal[7] = 30.0; // h0 turbine column

    let mut obj = vec![0.0_f64; n_cols];
    obj[6] = 1.0; // theta coefficient (undiscounted)
    // c_orig / K = 0.005.  With col_scale = 2.0: obj_coeff = 0.005 * 2.0 = 0.01.
    obj[7] = 0.01;

    // Build col_scale: all 1.0 except column 7 = 2.0.
    let mut col_scale = vec![1.0_f64; n_cols];
    col_scale[7] = 2.0;

    let dual = vec![0.0_f64; geometry_row_capacity(&indexer)];
    let row_lower = vec![0.0_f64; geometry_row_capacity(&indexer)];

    let counts = EntityCounts {
        hydro_ids: vec![1, 2],
        hydro_productivities: vec![0.0, 1.5],
        thermal_ids: vec![],
        line_ids: vec![],
        bus_ids: vec![],
        pumping_station_ids: vec![],
        contract_ids: vec![],
        non_controllable_ids: vec![],
    };

    let ec = zero_energy_conversion(2, 1);
    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &dual,
            objective: 500.3, // scaled space: theta + fpha_scaled
            objective_coeffs: &obj,
            row_lower: &row_lower,
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &indexer,
            hydro_cell_index: &test_support::identity_hydro_cell_index(256),
            state: &state,
            entity_counts: &counts,
            inflow_m3s_per_hydro: &[0.0; 2],
            block_hours: &[],
            generic_constraint_entries: &[],
            ncs_col_upper: &[],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_slots: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[0.0, 1.5],
            col_scale: &col_scale,
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            cost_scale_factor: 1_000_000.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0; 2],
            stage_index: 0,
            horizon: &HorizonMode::Finite { num_stages: 1 },
            anticipated_windows: &[],
            study_stage_ids: &[],
        },
        0,
    );

    let cost = &result.costs[0];
    // turbined_cost = primal * obj_coeff / col_scale * K
    //                    = 30 * 0.01 / 2.0 * 1_000_000 = 150_000.0
    assert!(
        (cost.turbined_cost - 150_000.0).abs() < 1e-6,
        "turbined_cost should be 150_000.0 (unscaled by col_scale=2.0), got {}",
        cost.turbined_cost
    );
}

/// Verify that `compute_cost_result` decomposes `hydro_violation_cost` into
/// the 6 per-constraint components, and that the sum invariant holds.
#[test]
fn hydro_violation_cost_decomposition() {
    let indexer = make_indexer_2h_1fpha_1blk();
    let study_dims = test_support::study_dims_for(&counts_2h_1fpha_1blk());
    let state = test_support::state_layout(2, 0);
    // Layout (see make_indexer_2h_1fpha_1blk):
    //   withdrawal_slack_neg:   14..16
    //   withdrawal_slack_pos:   16..18
    //   outflow_below_slack:    18..20
    //   outflow_above_slack:    20..22
    //   turbine_below_slack:    22..24
    //   generation_below_slack: 24..26
    assert_eq!(indexer.withdrawal_slack_neg, 14..16);
    assert_eq!(indexer.withdrawal_slack_pos, 16..18);
    assert_eq!(indexer.outflow_below_slack, 18..20);
    assert_eq!(indexer.outflow_above_slack, 20..22);
    assert_eq!(indexer.turbine_below_slack, 22..24);
    assert_eq!(indexer.generation_below_slack, 24..26);

    let n_cols = indexer.generation_below_slack.end;
    let mut primal = vec![0.0_f64; n_cols];
    let mut obj = vec![0.0_f64; n_cols];

    // Set theta so objective algebra works.
    primal[state.theta] = 0.0;

    // Assign known primal (slack) and objective (penalty) values per
    // constraint type. Each slack * penalty gives a known cost contribution.
    // COST_SCALE_FACTOR = 1_000_000.0 is applied inside the extraction.

    // outflow_below: h0=2.0 * 10.0, h1=3.0 * 10.0  => (20+30) * 1_000_000 = 50_000_000
    primal[18] = 2.0;
    obj[18] = 10.0;
    primal[19] = 3.0;
    obj[19] = 10.0;

    // outflow_above: h0=1.0 * 5.0, h1=0.0  => 5 * 1_000_000 = 5_000_000
    primal[20] = 1.0;
    obj[20] = 5.0;

    // turbine_below: h0=4.0 * 8.0, h1=0.0  => 32 * 1_000_000 = 32_000_000
    primal[22] = 4.0;
    obj[22] = 8.0;

    // generation_below: h0=0.0, h1=6.0 * 3.0  => 18 * 1_000_000 = 18_000_000
    primal[25] = 6.0;
    obj[25] = 3.0;

    // withdrawal (neg): h0=0.5 * 20.0, h1=0.0  => 10 * 1_000_000 = 10_000_000
    primal[14] = 0.5;
    obj[14] = 20.0;

    // withdrawal (pos): h0=0.0, h1=0.3 * 15.0  => 4.5 * 1_000_000 = 4_500_000
    primal[17] = 0.3;
    obj[17] = 15.0;

    // Total objective for the LP (sum of primal * obj):
    let total_obj: f64 = primal.iter().zip(obj.iter()).map(|(p, o)| p * o).sum();

    let dual = vec![0.0_f64; geometry_row_capacity(&indexer)];
    let row_lower = vec![0.0_f64; geometry_row_capacity(&indexer)];

    let counts = EntityCounts {
        hydro_ids: vec![1, 2],
        hydro_productivities: vec![0.0, 1.5],
        thermal_ids: vec![],
        line_ids: vec![],
        bus_ids: vec![],
        pumping_station_ids: vec![],
        contract_ids: vec![],
        non_controllable_ids: vec![],
    };

    let ec = zero_energy_conversion(2, 1);
    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &dual,
            objective: total_obj,
            objective_coeffs: &obj,
            row_lower: &row_lower,
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &indexer,
            hydro_cell_index: &test_support::identity_hydro_cell_index(256),
            state: &state,
            entity_counts: &counts,
            inflow_m3s_per_hydro: &[0.0; 2],
            block_hours: &[],
            generic_constraint_entries: &[],
            ncs_col_upper: &[],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_slots: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[0.0, 1.5],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            cost_scale_factor: 1_000_000.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0; 2],
            stage_index: 0,
            horizon: &HorizonMode::Finite { num_stages: 1 },
            anticipated_windows: &[],
            study_stage_ids: &[],
        },
        0,
    );

    let cost = &result.costs[0];

    // Expected values (primal * obj * COST_SCALE_FACTOR):
    let expected_outflow_below = (2.0 * 10.0 + 3.0 * 10.0) * 1_000_000.0;
    let expected_outflow_above = (1.0 * 5.0) * 1_000_000.0;
    let expected_turbined = (4.0 * 8.0) * 1_000_000.0;
    let expected_generation = (6.0 * 3.0) * 1_000_000.0;
    let expected_withdrawal = (0.5 * 20.0 + 0.3 * 15.0) * 1_000_000.0;
    let expected_evaporation = 0.0; // no evaporation hydros

    assert!(
        (cost.outflow_violation_below_cost - expected_outflow_below).abs() < 1e-6,
        "outflow_below: expected {expected_outflow_below}, got {}",
        cost.outflow_violation_below_cost
    );
    assert!(
        (cost.outflow_violation_above_cost - expected_outflow_above).abs() < 1e-6,
        "outflow_above: expected {expected_outflow_above}, got {}",
        cost.outflow_violation_above_cost
    );
    assert!(
        (cost.turbined_violation_cost - expected_turbined).abs() < 1e-6,
        "turbined: expected {expected_turbined}, got {}",
        cost.turbined_violation_cost
    );
    assert!(
        (cost.generation_violation_cost - expected_generation).abs() < 1e-6,
        "generation: expected {expected_generation}, got {}",
        cost.generation_violation_cost
    );
    assert!(
        (cost.evaporation_violation_cost - expected_evaporation).abs() < 1e-6,
        "evaporation: expected {expected_evaporation}, got {}",
        cost.evaporation_violation_cost
    );
    assert!(
        (cost.withdrawal_violation_cost - expected_withdrawal).abs() < 1e-6,
        "withdrawal: expected {expected_withdrawal}, got {}",
        cost.withdrawal_violation_cost
    );

    // Sum invariant: hydro_violation_cost == sum of all 6 components.
    let component_sum = cost.outflow_violation_below_cost
        + cost.outflow_violation_above_cost
        + cost.turbined_violation_cost
        + cost.generation_violation_cost
        + cost.evaporation_violation_cost
        + cost.withdrawal_violation_cost;
    assert!(
        (cost.hydro_violation_cost - component_sum).abs() < 1e-6,
        "hydro_violation_cost ({}) must equal sum of components ({component_sum})",
        cost.hydro_violation_cost
    );
}

/// Build an [`EnergyConversionSet`] for the (1 hydro, 1 stage) case with
/// explicit `ρ_eq` and `ρ_acum` values.
fn one_hydro_energy_set(rho_eq: f64, rho_acum: f64) -> EnergyConversionSet {
    use crate::energy_conversion::EnergyConversion;
    let cell = EnergyConversion {
        equivalent_productivity_mw_per_m3s: rho_eq,
        reference_volume_hm3: 0.0,
        reference_outflow_m3s: 0.0,
    };
    EnergyConversionSet::new(
        vec![vec![cell; 1]; 1],
        vec![vec![rho_acum; 1]; 1],
        &test_support::minimal_hydros(1),
        1,
    )
}

fn make_entity_counts_1_hydro() -> EntityCounts {
    EntityCounts {
        hydro_ids: vec![10],
        hydro_productivities: vec![1.0],
        thermal_ids: vec![],
        line_ids: vec![],
        bus_ids: vec![100],
        pumping_station_ids: vec![],
        contract_ids: vec![],
        non_controllable_ids: vec![],
    }
}

/// Build a primal vector for the N=1, L=1 state layout:
/// `[storage, lag, z_inflow, storage_in, theta]` (length 5).
fn make_primal_1_1(storage: f64, storage_in: f64, theta: f64) -> Vec<f64> {
    vec![storage, 0.0, 0.0, storage_in, theta]
}

#[test]
fn stored_energy_initial_uses_v_min_offset() {
    // storage_initial = 110, V_min = 100, ρ_acum_integrated = 4.0 (a plain
    // EnergyConversionSet::new with no with_integrated defaults the integrated
    // grid to a clone of the reference-point one, so this is the coinciding
    // case) → stored_energy_initial = (110 - 100) * 4 * 1e6 / 3600 ≈ 11_111.111…
    let dims = test_support::GeometryDims {
        hydro_count: 1,
        max_par_order: 1,
        n_thermals: 0,
        n_lines: 0,
        n_buses: 1,
        n_blks: 1,
        has_inflow_penalty: false,
        max_deficit_segments: 1,
        n_anticipated: 0,
        lead_stages: 0,
        anticipated_plants: AnticipatedPlants::default(),
    };
    let indexer = test_support::geometry(&dims, vec![], &[], vec![]);
    let study_dims = test_support::study_dims();
    let state = test_support::state_layout(1, 1);
    let mut primal = make_primal_1_1(120.0, 110.0, 0.0);
    primal.resize(indexer.generation_below_slack.end, 0.0);
    let objective_coeffs = vec![0.0; primal.len()];
    let dual = vec![0.0; geometry_row_capacity(&indexer)];
    let row_lower = vec![0.0; geometry_row_capacity(&indexer)];
    let ec = one_hydro_energy_set(0.9, 4.0);

    let inflow_m3s_per_hydro = test_support::inflow_m3s_per_hydro_from_primal(&state, &primal, 1);
    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &dual,
            objective: 0.0,
            objective_coeffs: &objective_coeffs,
            row_lower: &row_lower,
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &indexer,
            hydro_cell_index: &test_support::identity_hydro_cell_index(256),
            state: &state,
            entity_counts: &make_entity_counts_1_hydro(),
            inflow_m3s_per_hydro: &inflow_m3s_per_hydro,
            block_hours: &[100.0],
            generic_constraint_entries: &[],
            ncs_col_upper: &[],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_slots: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[1.0],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            cost_scale_factor: 1_000_000.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[100.0],
            stage_index: 0,
            horizon: &HorizonMode::Finite { num_stages: 1 },
            anticipated_windows: &[],
            study_stage_ids: &[],
        },
        0,
    );

    assert_eq!(result.hydros.len(), 1);
    let h = &result.hydros[0];
    assert!(
        (h.equivalent_productivity_mw_per_m3s - 0.9).abs() < 1e-12,
        "equivalent_productivity should be 0.9, got {}",
        h.equivalent_productivity_mw_per_m3s
    );
    assert!(
        (h.accumulated_productivity_mw_per_m3s - 4.0).abs() < 1e-12,
        "accumulated_productivity should be 4.0, got {}",
        h.accumulated_productivity_mw_per_m3s
    );
    // storage_initial_hm3 = 110.0 (from primal storage_in), V_min = 100.0, against
    // ρ_acum_integrated (coincides with ρ_acum here — see the comment above).
    let expected = (110.0_f64 - 100.0) * 4.0 * 1.0e6 / 3600.0;
    assert!(
        (h.stored_energy_initial_mwh - expected).abs() < 1e-6,
        "stored_energy_initial: expected {expected}, got {}",
        h.stored_energy_initial_mwh
    );
}

#[test]
fn incremental_inflow_energy_uses_rho_acum() {
    // ρ_acum = 4.0, incremental_inflow = 50.0 →
    // incremental_inflow_energy = 4.0 * 50.0 = 200.0 (exactly).
    let dims = test_support::GeometryDims {
        hydro_count: 1,
        max_par_order: 1,
        n_thermals: 0,
        n_lines: 0,
        n_buses: 1,
        n_blks: 1,
        has_inflow_penalty: false,
        max_deficit_segments: 1,
        n_anticipated: 0,
        lead_stages: 0,
        anticipated_plants: AnticipatedPlants::default(),
    };
    let indexer = test_support::geometry(&dims, vec![], &[], vec![]);
    let study_dims = test_support::study_dims();
    let state = test_support::state_layout(1, 1);
    let mut primal = make_primal_1_1(120.0, 110.0, 0.0);
    primal.resize(indexer.generation_below_slack.end, 0.0);
    let objective_coeffs = vec![0.0; primal.len()];
    let dual = vec![0.0; geometry_row_capacity(&indexer)];
    let row_lower = vec![0.0; geometry_row_capacity(&indexer)];
    let ec = one_hydro_energy_set(0.9, 4.0);

    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &dual,
            objective: 0.0,
            objective_coeffs: &objective_coeffs,
            row_lower: &row_lower,
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &indexer,
            hydro_cell_index: &test_support::identity_hydro_cell_index(256),
            state: &state,
            entity_counts: &make_entity_counts_1_hydro(),
            inflow_m3s_per_hydro: &[50.0],
            block_hours: &[100.0],
            generic_constraint_entries: &[],
            ncs_col_upper: &[],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_slots: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[1.0],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            cost_scale_factor: 1_000_000.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[100.0],
            stage_index: 0,
            horizon: &HorizonMode::Finite { num_stages: 1 },
            anticipated_windows: &[],
            study_stage_ids: &[],
        },
        0,
    );

    assert_eq!(result.hydros.len(), 1);
    assert!(
        (result.hydros[0].incremental_inflow_energy_mw - 200.0).abs() < 1e-12,
        "incremental_inflow_energy should be 200.0, got {}",
        result.hydros[0].incremental_inflow_energy_mw
    );
}

/// With `with_integrated` giving `integrated_accumulated_productivity` a value
/// distinct from `accumulated_productivity`, stored energy rides the integrated
/// grid while `incremental_inflow_energy_mw` and the reported accumulated column
/// stay on the reference-point grid — proven on both the builder geometry and
/// the hand-built Parallel geometry.
#[test]
fn stored_energy_rides_integrated_grid_distinct_from_reference_point() {
    let rho_acum = 4.0_f64;
    let rho_acum_integrated = 7.5_f64;
    let v_min = 100.0_f64;
    let incremental_inflow = 50.0_f64;
    let ec = one_hydro_energy_set(0.9, rho_acum)
        .with_integrated(vec![vec![0.9]], vec![vec![rho_acum_integrated]]);

    let dims = test_support::GeometryDims {
        hydro_count: 1,
        max_par_order: 1,
        n_thermals: 0,
        n_lines: 0,
        n_buses: 1,
        n_blks: 1,
        has_inflow_penalty: false,
        max_deficit_segments: 1,
        n_anticipated: 0,
        lead_stages: 0,
        anticipated_plants: AnticipatedPlants::default(),
    };
    let indexer = test_support::geometry(&dims, vec![], &[], vec![]);
    let study_dims = test_support::study_dims();
    let state = test_support::state_layout(1, 1);
    let mut primal = make_primal_1_1(120.0, 110.0, 0.0);
    primal.resize(indexer.generation_below_slack.end, 0.0);
    let objective_coeffs = vec![0.0; primal.len()];
    let dual = vec![0.0; geometry_row_capacity(&indexer)];
    let row_lower = vec![0.0; geometry_row_capacity(&indexer)];

    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &dual,
            objective: 0.0,
            objective_coeffs: &objective_coeffs,
            row_lower: &row_lower,
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &indexer,
            hydro_cell_index: &test_support::identity_hydro_cell_index(256),
            state: &state,
            entity_counts: &make_entity_counts_1_hydro(),
            inflow_m3s_per_hydro: &[incremental_inflow],
            block_hours: &[100.0],
            generic_constraint_entries: &[],
            ncs_col_upper: &[],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_slots: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[1.0],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            cost_scale_factor: 1_000_000.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[v_min],
            stage_index: 0,
            horizon: &HorizonMode::Finite { num_stages: 1 },
            anticipated_windows: &[],
            study_stage_ids: &[],
        },
        0,
    );

    assert_eq!(result.hydros.len(), 1);
    let row = &result.hydros[0];
    assert_eq!(
        row.accumulated_productivity_mw_per_m3s.to_bits(),
        rho_acum.to_bits(),
        "builder geometry: reported accumulated column must stay on the reference-point grid"
    );
    let expected_inflow_energy = rho_acum * incremental_inflow;
    assert_eq!(
        row.incremental_inflow_energy_mw.to_bits(),
        expected_inflow_energy.to_bits(),
        "builder geometry: inflow energy must stay on the reference-point grid"
    );
    let expected_initial =
        (110.0_f64 - v_min) * rho_acum_integrated * super::ENERGY_FACTOR_MWH_PER_HM3_PER_MW_PER_M3S;
    let expected_final =
        (120.0_f64 - v_min) * rho_acum_integrated * super::ENERGY_FACTOR_MWH_PER_HM3_PER_MW_PER_M3S;
    assert_eq!(
        row.stored_energy_initial_mwh.to_bits(),
        expected_initial.to_bits(),
        "builder geometry: stored energy must ride the integrated grid"
    );
    assert_eq!(
        row.stored_energy_final_mwh.to_bits(),
        expected_final.to_bits(),
        "builder geometry: stored energy must ride the integrated grid"
    );

    let k = 1_usize;
    let geom = single_hydro_block_geometry(BlockMode::Parallel, k);
    let pb_state = test_support::state_layout(1, 0);
    let n_cols = geom.generation_below_slack.end;
    let mut pb_primal = vec![0.0_f64; n_cols];
    pb_primal[0] = 120.0; // Sᴷ (outgoing)
    pb_primal[2] = 110.0; // S⁰ (incoming)
    let pb_dual = vec![0.0_f64; 4];

    let pb_spec = StageExtractionSpec {
        study_dims: &study_dims,
        geometry: &geom,
        hydro_cell_index: &test_support::identity_hydro_cell_index(256),
        state: &pb_state,
        entity_counts: &entity_counts_1_hydro(),
        inflow_m3s_per_hydro: &[incremental_inflow],
        block_hours: &[100.0],
        generic_constraint_entries: &[],
        ncs_col_upper: &[],
        pumping_consumption_mw_per_m3s: &[],
        contract_prices: &[],
        contract_slots: &[],
        diversion_upstream: &HashMap::new(),
        hydro_productivities: &[0.0],
        col_scale: &[],
        row_scale: &[],
        cumulative_discount_factor: 1.0,
        cost_scale_factor: 1_000_000.0,
        energy_conversion: &ec,
        hydro_min_storage_hm3: &[v_min],
        stage_index: 0,
        horizon: &HorizonMode::Finite { num_stages: 1 },
        anticipated_windows: &[],
        study_stage_ids: &[],
    };
    let pb_view = SolutionView {
        primal: &pb_primal,
        dual: &pb_dual,
        objective: 0.0,
        objective_coeffs: &vec![0.0_f64; n_cols],
        row_lower: &[],
    };

    let pb_result = extract_stage_result(&pb_view, &pb_spec, 0);
    assert_eq!(pb_result.hydros.len(), k);
    let pb_row = &pb_result.hydros[0];
    assert_eq!(
        pb_row.accumulated_productivity_mw_per_m3s.to_bits(),
        rho_acum.to_bits(),
        "hand-built Parallel geometry: reported accumulated column must stay on the reference-point grid"
    );
    assert_eq!(
        pb_row.incremental_inflow_energy_mw.to_bits(),
        expected_inflow_energy.to_bits(),
        "hand-built Parallel geometry: inflow energy must stay on the reference-point grid"
    );
    assert_eq!(
        pb_row.stored_energy_initial_mwh.to_bits(),
        expected_initial.to_bits(),
        "hand-built Parallel geometry: stored energy must ride the integrated grid"
    );
    assert_eq!(
        pb_row.stored_energy_final_mwh.to_bits(),
        expected_final.to_bits(),
        "hand-built Parallel geometry: stored energy must ride the integrated grid"
    );
}

/// `stored_energy_{initial,final}_mw` divide the corresponding `_mwh` value by
/// the STAGE's total hours (`Σ block_hours`), never a per-block hours — proven
/// on the builder geometry and, under Parallel block mode (storage held constant
/// across blocks), identical on every row of the hand-built Parallel geometry.
#[test]
fn stored_energy_mw_divides_by_stage_total_hours() {
    let rho_acum_integrated = 2.0_f64;
    let v_min = 5.0_f64;
    let storage_initial = 20.0_f64;
    let storage_final = 50.0_f64;
    let ec = one_hydro_energy_set(0.0, rho_acum_integrated);
    let block_hours = [100.0_f64, 200.0, 300.0];
    let stage_total_hours: f64 = block_hours.iter().sum();

    let dims = test_support::GeometryDims {
        hydro_count: 1,
        max_par_order: 1,
        n_thermals: 0,
        n_lines: 0,
        n_buses: 1,
        n_blks: 3,
        has_inflow_penalty: false,
        max_deficit_segments: 1,
        n_anticipated: 0,
        lead_stages: 0,
        anticipated_plants: AnticipatedPlants::default(),
    };
    let indexer = test_support::geometry(&dims, vec![], &[], vec![]);
    let study_dims = test_support::study_dims();
    let state = test_support::state_layout(1, 1);
    let mut primal = make_primal_1_1(storage_final, storage_initial, 0.0);
    primal.resize(indexer.generation_below_slack.end, 0.0);
    let objective_coeffs = vec![0.0; primal.len()];
    let dual = vec![0.0; geometry_row_capacity(&indexer)];
    let row_lower = vec![0.0; geometry_row_capacity(&indexer)];

    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &dual,
            objective: 0.0,
            objective_coeffs: &objective_coeffs,
            row_lower: &row_lower,
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &indexer,
            hydro_cell_index: &test_support::identity_hydro_cell_index(256),
            state: &state,
            entity_counts: &make_entity_counts_1_hydro(),
            inflow_m3s_per_hydro: &[0.0],
            block_hours: &block_hours,
            generic_constraint_entries: &[],
            ncs_col_upper: &[],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_slots: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[1.0],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            cost_scale_factor: 1_000_000.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[v_min],
            stage_index: 0,
            horizon: &HorizonMode::Finite { num_stages: 1 },
            anticipated_windows: &[],
            study_stage_ids: &[],
        },
        0,
    );

    assert_eq!(result.hydros.len(), 3);
    let expected_initial_mw = result.hydros[0].stored_energy_initial_mwh / stage_total_hours;
    let expected_final_mw = result.hydros[0].stored_energy_final_mwh / stage_total_hours;
    for row in &result.hydros {
        assert_eq!(
            row.stored_energy_initial_mw.to_bits(),
            (row.stored_energy_initial_mwh / stage_total_hours).to_bits(),
            "builder geometry: stored_energy_initial_mw must equal _mwh / stage total hours"
        );
        assert_eq!(
            row.stored_energy_final_mw.to_bits(),
            (row.stored_energy_final_mwh / stage_total_hours).to_bits(),
            "builder geometry: stored_energy_final_mw must equal _mwh / stage total hours"
        );
    }

    let k = 3_usize;
    let geom = single_hydro_block_geometry(BlockMode::Parallel, k);
    let pb_state = test_support::state_layout(1, 0);
    let n_cols = geom.generation_below_slack.end;
    let mut pb_primal = vec![0.0_f64; n_cols];
    pb_primal[0] = storage_final; // Sᴷ (outgoing)
    pb_primal[2] = storage_initial; // S⁰ (incoming)
    let pb_dual = vec![0.0_f64; 4];

    let pb_spec = StageExtractionSpec {
        study_dims: &study_dims,
        geometry: &geom,
        hydro_cell_index: &test_support::identity_hydro_cell_index(256),
        state: &pb_state,
        entity_counts: &entity_counts_1_hydro(),
        inflow_m3s_per_hydro: &[0.0],
        block_hours: &block_hours,
        generic_constraint_entries: &[],
        ncs_col_upper: &[],
        pumping_consumption_mw_per_m3s: &[],
        contract_prices: &[],
        contract_slots: &[],
        diversion_upstream: &HashMap::new(),
        hydro_productivities: &[0.0],
        col_scale: &[],
        row_scale: &[],
        cumulative_discount_factor: 1.0,
        cost_scale_factor: 1_000_000.0,
        energy_conversion: &ec,
        hydro_min_storage_hm3: &[v_min],
        stage_index: 0,
        horizon: &HorizonMode::Finite { num_stages: 1 },
        anticipated_windows: &[],
        study_stage_ids: &[],
    };
    let pb_view = SolutionView {
        primal: &pb_primal,
        dual: &pb_dual,
        objective: 0.0,
        objective_coeffs: &vec![0.0_f64; n_cols],
        row_lower: &[],
    };

    let pb_result = extract_stage_result(&pb_view, &pb_spec, 0);
    assert_eq!(pb_result.hydros.len(), k);
    for pb_row in &pb_result.hydros {
        assert_eq!(
            pb_row.stored_energy_initial_mw.to_bits(),
            expected_initial_mw.to_bits(),
            "hand-built Parallel geometry: stored_energy_initial_mw must divide by stage, not per-block, hours"
        );
        assert_eq!(
            pb_row.stored_energy_final_mw.to_bits(),
            expected_final_mw.to_bits(),
            "hand-built Parallel geometry: stored_energy_final_mw must divide by stage, not per-block, hours"
        );
    }
}

#[test]
fn stage_path_propagates_productivity_values() {
    // Must read ρ_eq and ρ_acum from the supplied EnergyConversionSet and
    // surface them on the result.
    let dims = test_support::GeometryDims {
        hydro_count: 1,
        max_par_order: 1,
        n_thermals: 0,
        n_lines: 0,
        n_buses: 1,
        n_blks: 1,
        has_inflow_penalty: false,
        max_deficit_segments: 1,
        n_anticipated: 0,
        lead_stages: 0,
        anticipated_plants: AnticipatedPlants::default(),
    };
    let indexer = test_support::geometry(&dims, vec![], &[], vec![]);
    let study_dims = test_support::study_dims();
    let state = test_support::state_layout(1, 1);
    let mut primal = make_primal_1_1(120.0, 110.0, 0.0);
    primal.resize(indexer.generation_below_slack.end, 0.0);
    let objective_coeffs = vec![0.0; primal.len()];
    let dual = vec![0.0; geometry_row_capacity(&indexer)];
    let row_lower = vec![0.0; geometry_row_capacity(&indexer)];
    let ec = one_hydro_energy_set(0.85, 3.5);

    let stage_result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &dual,
            objective: 0.0,
            objective_coeffs: &objective_coeffs,
            row_lower: &row_lower,
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &indexer,
            hydro_cell_index: &test_support::identity_hydro_cell_index(256),
            state: &state,
            entity_counts: &make_entity_counts_1_hydro(),
            inflow_m3s_per_hydro: &[10.0],
            block_hours: &[100.0],
            generic_constraint_entries: &[],
            ncs_col_upper: &[],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_slots: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[1.0],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            cost_scale_factor: 1_000_000.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[100.0],
            stage_index: 0,
            horizon: &HorizonMode::Finite { num_stages: 1 },
            anticipated_windows: &[],
            study_stage_ids: &[],
        },
        0,
    );

    let rho_eq = stage_result.hydros[0].equivalent_productivity_mw_per_m3s;
    let rho_acum = stage_result.hydros[0].accumulated_productivity_mw_per_m3s;
    assert!((rho_eq - 0.85).abs() < 1e-12, "stage rho_eq = {rho_eq}");
    assert!(
        (rho_acum - 3.5).abs() < 1e-12,
        "stage rho_acum = {rho_acum}"
    );
}

#[test]
fn integrated_productivity_columns_read_the_mean_evaluator_on_both_branches() {
    // Reference point ρ_eq = 0.9, ρ_acum = 4.0; the mean-evaluator grids are
    // installed distinct (1.5, 6.0) so a column reading the reference-point pair
    // instead of the integrated accessor would fail this test.
    let dims = test_support::GeometryDims {
        hydro_count: 1,
        max_par_order: 1,
        n_thermals: 0,
        n_lines: 0,
        n_buses: 1,
        n_blks: 1,
        has_inflow_penalty: false,
        max_deficit_segments: 1,
        n_anticipated: 0,
        lead_stages: 0,
        anticipated_plants: AnticipatedPlants::default(),
    };
    let indexer = test_support::geometry(&dims, vec![], &[], vec![]);
    let study_dims = test_support::study_dims();
    let state = test_support::state_layout(1, 1);
    let ec = one_hydro_energy_set(0.9, 4.0).with_integrated(vec![vec![1.5]], vec![vec![6.0]]);
    let dual = vec![0.0; geometry_row_capacity(&indexer)];
    let row_lower = vec![0.0; geometry_row_capacity(&indexer)];

    let assert_columns = |h: &SimulationHydroResult| {
        assert!(
            (h.equivalent_productivity_mw_per_m3s - 0.9).abs() < 1e-12,
            "reference ρ_eq must stay unchanged, got {}",
            h.equivalent_productivity_mw_per_m3s
        );
        assert!(
            (h.accumulated_productivity_mw_per_m3s - 4.0).abs() < 1e-12,
            "reference ρ_acum must stay unchanged, got {}",
            h.accumulated_productivity_mw_per_m3s
        );
        assert!(
            (h.integrated_equivalent_productivity_mw_per_m3s - 1.5).abs() < 1e-12,
            "integrated ρ_eq must equal the own-scope accessor, got {}",
            h.integrated_equivalent_productivity_mw_per_m3s
        );
        assert!(
            (h.integrated_accumulated_productivity_mw_per_m3s - 6.0).abs() < 1e-12,
            "integrated ρ_acum must equal the cascade-scope accessor, got {}",
            h.integrated_accumulated_productivity_mw_per_m3s
        );
    };

    // First independently-built call: zero turbine primal.
    let mut primal_a = make_primal_1_1(120.0, 110.0, 0.0);
    primal_a.resize(indexer.generation_below_slack.end, 0.0);
    let objective_coeffs_a = vec![0.0; primal_a.len()];
    let result_a = extract_stage_result(
        &SolutionView {
            primal: &primal_a,
            dual: &dual,
            objective: 0.0,
            objective_coeffs: &objective_coeffs_a,
            row_lower: &row_lower,
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &indexer,
            hydro_cell_index: &test_support::identity_hydro_cell_index(256),
            state: &state,
            entity_counts: &make_entity_counts_1_hydro(),
            inflow_m3s_per_hydro: &[10.0],
            block_hours: &[100.0],
            generic_constraint_entries: &[],
            ncs_col_upper: &[],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_slots: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[1.0],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            cost_scale_factor: 1_000_000.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[100.0],
            stage_index: 0,
            horizon: &HorizonMode::Finite { num_stages: 1 },
            anticipated_windows: &[],
            study_stage_ids: &[],
        },
        0,
    );
    assert_eq!(result_a.hydros.len(), 1);
    assert_columns(&result_a.hydros[0]);

    // Second independently-built call: a non-zero turbine primal and non-empty
    // block_hours, over the same real geometry — must not disagree with the first.
    let mut primal_b = make_primal_1_1(120.0, 110.0, 0.0);
    primal_b.resize(indexer.generation_below_slack.end, 0.0);
    primal_b[indexer.turbine.start] = 5.0;
    let objective_coeffs_b = vec![0.0; primal_b.len()];
    let result_b = extract_stage_result(
        &SolutionView {
            primal: &primal_b,
            dual: &dual,
            objective: 0.0,
            objective_coeffs: &objective_coeffs_b,
            row_lower: &row_lower,
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &indexer,
            hydro_cell_index: &test_support::identity_hydro_cell_index(256),
            state: &state,
            entity_counts: &make_entity_counts_1_hydro(),
            inflow_m3s_per_hydro: &[10.0],
            block_hours: &[720.0],
            generic_constraint_entries: &[],
            ncs_col_upper: &[],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_slots: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[1.0],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            cost_scale_factor: 1_000_000.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[100.0],
            stage_index: 0,
            horizon: &HorizonMode::Finite { num_stages: 1 },
            anticipated_windows: &[],
            study_stage_ids: &[],
        },
        0,
    );
    assert_eq!(result_b.hydros.len(), 1);
    assert_columns(&result_b.hydros[0]);
}

// -------------------------------------------------------------------------
// extract_pumping_stations
// -------------------------------------------------------------------------

/// `EntityCounts` carrying a single pumping station (id `42`) and no other
/// entities, for the pumping-extraction tests.
fn entity_counts_one_pumping_station() -> EntityCounts {
    EntityCounts {
        hydro_ids: vec![],
        hydro_productivities: vec![],
        thermal_ids: vec![],
        line_ids: vec![],
        bus_ids: vec![],
        pumping_station_ids: vec![42],
        contract_ids: vec![],
        non_controllable_ids: vec![],
    }
}

/// Build a `StageExtractionSpec` whose only meaningful fields are the
/// pumping inputs; all other fields are inert defaults.  The spec's `n_blks`
/// is taken from `geometry.n_blks`. Under the dense layout
/// `extract_pumping_stations` indexes the station id list directly by the
/// system column position, so the caller need only supply a matching
/// `entity_counts.pumping_station_ids`/`geometry.pumping_flow`.
// Rationale: this helper assembles a single inert-default StageExtractionSpec
// for the pumping tests; its parameter list mirrors the spec's borrowed
// inputs, so bundling them would obscure rather than clarify.
#[allow(clippy::too_many_arguments)]
fn pumping_only_spec<'a>(
    study_dims: &'a StudyDimensions,
    geometry: &'a StageGeometry,
    state: &'a StateSpace,
    entity_counts: &'a EntityCounts,
    consumption: &'a [f64],
    ec: &'a EnergyConversionSet,
    diversion: &'a HashMap<cobre_core::EntityId, Vec<usize>>,
) -> StageExtractionSpec<'a> {
    // Leaked so the borrow outlives this function call, matching the caller's own
    // `'a` — every call site here is single-bus, so an identity index is exact.
    let hydro_cell_index: &'a HydroCellIndex =
        Box::leak(Box::new(test_support::identity_hydro_cell_index(256)));
    StageExtractionSpec {
        study_dims,
        geometry,
        hydro_cell_index,
        state,
        entity_counts,
        inflow_m3s_per_hydro: &[],
        block_hours: &[],
        generic_constraint_entries: &[],
        ncs_col_upper: &[],
        pumping_consumption_mw_per_m3s: consumption,
        contract_prices: &[],
        contract_slots: &[],
        diversion_upstream: diversion,
        hydro_productivities: &[],
        col_scale: &[],
        row_scale: &[],
        cumulative_discount_factor: 1.0,
        cost_scale_factor: 1_000_000.0,
        energy_conversion: ec,
        hydro_min_storage_hm3: &[],
        stage_index: 0,
        horizon: &HorizonMode::Finite { num_stages: 1 },
        anticipated_windows: &[],
        study_stage_ids: &[],
    }
}

/// One station, two blocks, distinct flow primals
/// (`[7.0, 3.0]`) and a non-unit `consumption_mw_per_m3s = 0.5`.
///
/// The pumping columns are placed at `pumping_flow.start = 4` in the primal,
/// block-major (`col + p_idx * n_blks + blk`).  Each row's
/// `pumped_flow_m3s` is read directly from `view.primal` (already unscaled —
/// no `col_scale` division) and `power_consumption_mw = flow * consumption`.
#[test]
fn extract_pumping_two_blocks_reads_per_block_flow_and_power() {
    let state = test_support::state_layout(0, 0);
    let entity_counts = entity_counts_one_pumping_station();
    let ec = zero_energy_conversion(0, 1);
    let diversion = HashMap::new();
    let consumption = [0.5_f64];

    // Primal layout: filler columns 0..4, then pumping block at 4..6:
    //   col 4 = (station 0, block 0) = 7.0
    //   col 5 = (station 0, block 1) = 3.0
    let primal = vec![0.0, 0.0, 0.0, 0.0, 7.0, 3.0];
    let dual = vec![0.0; 1];
    let view = SolutionView {
        primal: &primal,
        dual: &dual,
        objective: 0.0,
        objective_coeffs: &[],
        row_lower: &[],
    };

    let equipment = StageGeometry {
        n_blks: 2,
        pumping_flow: 4..6,
        ..test_support::equipment_free_geometry(&[2]).remove(0)
    };
    let study_dims = test_support::study_dims();
    let spec = pumping_only_spec(
        &study_dims,
        &equipment,
        &state,
        &entity_counts,
        &consumption,
        &ec,
        &diversion,
    );

    let rows = extract_pumping_stations(&view, &spec, 9);

    assert_eq!(rows.len(), 2, "one station x two blocks = two rows");

    assert_eq!(rows[0].stage_id, 9);
    assert_eq!(rows[0].pumping_station_id, 42);
    assert_eq!(rows[0].block_id, Some(0));
    assert_eq!(rows[0].pumped_flow_m3s, 7.0);
    assert_eq!(rows[0].power_consumption_mw, 3.5); // 7.0 * 0.5
    assert_eq!(rows[0].pumping_cost, 0.0);
    assert_eq!(rows[0].operative_state_code, 1);

    assert_eq!(rows[1].block_id, Some(1));
    assert_eq!(rows[1].pumped_flow_m3s, 3.0);
    assert_eq!(rows[1].power_consumption_mw, 1.5); // 3.0 * 0.5
    assert_eq!(rows[1].pumping_station_id, 42);
}

/// Zero pumping stations yields an empty `Vec` (the block loop never
/// runs), independent of `n_blks`.
#[test]
fn extract_pumping_zero_stations_is_empty() {
    let state = test_support::state_layout(0, 0);
    let entity_counts = EntityCounts {
        hydro_ids: vec![],
        hydro_productivities: vec![],
        thermal_ids: vec![],
        line_ids: vec![],
        bus_ids: vec![],
        pumping_station_ids: vec![],
        contract_ids: vec![],
        non_controllable_ids: vec![],
    };
    let ec = zero_energy_conversion(0, 1);
    let diversion = HashMap::new();
    let primal = vec![0.0; 4];
    let dual = vec![0.0; 1];
    let view = SolutionView {
        primal: &primal,
        dual: &dual,
        objective: 0.0,
        objective_coeffs: &[],
        row_lower: &[],
    };

    let equipment = test_support::equipment_free_geometry(&[0]).remove(0);
    let study_dims = test_support::study_dims();
    let spec = pumping_only_spec(
        &study_dims,
        &equipment,
        &state,
        &entity_counts,
        &[],
        &ec,
        &diversion,
    );

    let rows = extract_pumping_stations(&view, &spec, 0);
    assert!(rows.is_empty(), "no stations => no rows");
}

/// `n_blks == 0` also yields an empty `Vec` even with an active station.
#[test]
fn extract_pumping_zero_blocks_is_empty() {
    let state = test_support::state_layout(0, 0);
    let entity_counts = entity_counts_one_pumping_station();
    let ec = zero_energy_conversion(0, 1);
    let diversion = HashMap::new();
    let consumption = [0.5_f64];
    let primal = vec![0.0; 4];
    let dual = vec![0.0; 1];
    let view = SolutionView {
        primal: &primal,
        dual: &dual,
        objective: 0.0,
        objective_coeffs: &[],
        row_lower: &[],
    };

    let equipment = test_support::equipment_free_geometry(&[0]).remove(0);
    let study_dims = test_support::study_dims();
    let spec = pumping_only_spec(
        &study_dims,
        &equipment,
        &state,
        &entity_counts,
        &consumption,
        &ec,
        &diversion,
    );

    let rows = extract_pumping_stations(&view, &spec, 0);
    assert!(rows.is_empty(), "n_blks == 0 => no rows");
}

// -------------------------------------------------------------------------
// extract_contracts
// -------------------------------------------------------------------------

fn entity_counts_contracts(contract_ids: Vec<i32>) -> EntityCounts {
    EntityCounts {
        hydro_ids: vec![],
        hydro_productivities: vec![],
        thermal_ids: vec![],
        line_ids: vec![],
        bus_ids: vec![],
        pumping_station_ids: vec![],
        contract_ids,
        non_controllable_ids: vec![],
    }
}

/// Build a `StageExtractionSpec` whose only meaningful fields are the contract
/// inputs; all other fields are inert defaults. `n_blks` is `geometry.n_blks`.
// Rationale: mirrors `pumping_only_spec`; the parameter list IS the spec's
// borrowed contract inputs, so bundling them would obscure rather than clarify.
#[allow(clippy::too_many_arguments)]
fn contract_only_spec<'a>(
    study_dims: &'a StudyDimensions,
    geometry: &'a StageGeometry,
    state: &'a StateSpace,
    entity_counts: &'a EntityCounts,
    block_hours: &'a [f64],
    contract_prices: &'a [f64],
    contract_slots: &'a [(ContractType, usize)],
    ec: &'a EnergyConversionSet,
    diversion: &'a HashMap<cobre_core::EntityId, Vec<usize>>,
) -> StageExtractionSpec<'a> {
    // Leaked so the borrow outlives this function call, matching the caller's own
    // `'a` — every call site here is single-bus, so an identity index is exact.
    let hydro_cell_index: &'a HydroCellIndex =
        Box::leak(Box::new(test_support::identity_hydro_cell_index(256)));
    StageExtractionSpec {
        study_dims,
        geometry,
        hydro_cell_index,
        state,
        entity_counts,
        inflow_m3s_per_hydro: &[],
        block_hours,
        generic_constraint_entries: &[],
        ncs_col_upper: &[],
        pumping_consumption_mw_per_m3s: &[],
        contract_prices,
        contract_slots,
        diversion_upstream: diversion,
        hydro_productivities: &[],
        col_scale: &[],
        row_scale: &[],
        cumulative_discount_factor: 1.0,
        cost_scale_factor: 1_000_000.0,
        energy_conversion: ec,
        hydro_min_storage_hm3: &[],
        stage_index: 0,
        horizon: &HorizonMode::Finite { num_stages: 1 },
        anticipated_windows: &[],
        study_stage_ids: &[],
    }
}

/// An import contract column holding `40.0` with `block_hours = 730` and
/// resolved price `200` yields `power_mw == 40.0`,
/// `total_cost == 200 * 40 * 730`, and `operative_state_code == 1`.
#[test]
fn extract_contract_import_reads_primal_and_cost() {
    let state = test_support::state_layout(0, 0);
    let entity_counts = entity_counts_contracts(vec![7]);
    let ec = zero_energy_conversion(0, 1);
    let diversion = HashMap::new();

    // Import block at column 4..5 (one import, one block); export block empty at 5..5.
    let primal = vec![0.0, 0.0, 0.0, 0.0, 40.0];
    let dual = vec![0.0; 1];
    let view = SolutionView {
        primal: &primal,
        dual: &dual,
        objective: 0.0,
        objective_coeffs: &[],
        row_lower: &[],
    };
    let geometry = StageGeometry {
        n_blks: 1,
        contract_import: 4..5,
        contract_export: 5..5,
        ..test_support::equipment_free_geometry(&[1]).remove(0)
    };
    let study_dims = test_support::study_dims();
    let block_hours = [730.0_f64];
    let prices = [200.0_f64];
    let contract_slots = [(ContractType::Import, 0)];
    let spec = contract_only_spec(
        &study_dims,
        &geometry,
        &state,
        &entity_counts,
        &block_hours,
        &prices,
        &contract_slots,
        &ec,
        &diversion,
    );

    let rows = extract_contracts(&view, &spec, 9);

    assert_eq!(rows.len(), 1, "one contract x one block = one row");
    assert_eq!(rows[0].stage_id, 9);
    assert_eq!(rows[0].contract_id, 7);
    assert_eq!(rows[0].block_id, Some(0));
    assert_eq!(rows[0].power_mw, 40.0);
    assert_eq!(rows[0].price_per_mwh, 200.0);
    assert_eq!(rows[0].total_cost, 200.0 * 40.0 * 730.0);
    assert_eq!(rows[0].operative_state_code, 1);
}

/// An export contract column holding `30.0` with resolved price `-150` and
/// `block_hours = 730` yields a negative `total_cost` (export revenue), addressed
/// from the export family base.
#[test]
fn extract_contract_export_yields_negative_cost() {
    let state = test_support::state_layout(0, 0);
    let entity_counts = entity_counts_contracts(vec![8]);
    let ec = zero_energy_conversion(0, 1);
    let diversion = HashMap::new();

    // No imports (empty 4..4); one export at column 4..5.
    let primal = vec![0.0, 0.0, 0.0, 0.0, 30.0];
    let dual = vec![0.0; 1];
    let view = SolutionView {
        primal: &primal,
        dual: &dual,
        objective: 0.0,
        objective_coeffs: &[],
        row_lower: &[],
    };
    let geometry = StageGeometry {
        n_blks: 1,
        contract_import: 4..4,
        contract_export: 4..5,
        ..test_support::equipment_free_geometry(&[1]).remove(0)
    };
    let study_dims = test_support::study_dims();
    let block_hours = [730.0_f64];
    let prices = [-150.0_f64];
    let contract_slots = [(ContractType::Export, 0)];
    let spec = contract_only_spec(
        &study_dims,
        &geometry,
        &state,
        &entity_counts,
        &block_hours,
        &prices,
        &contract_slots,
        &ec,
        &diversion,
    );

    let rows = extract_contracts(&view, &spec, 0);

    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].contract_id, 8);
    assert_eq!(rows[0].power_mw, 30.0);
    assert_eq!(rows[0].total_cost, -150.0 * 30.0 * 730.0);
    assert!(rows[0].total_cost < 0.0, "export revenue is negative");
    assert_eq!(rows[0].operative_state_code, 1);
}

/// A dormant contract whose column is pinned to `0.0` yields `power_mw == 0.0`,
/// `total_cost == 0.0`, and still `operative_state_code == 1` (never a
/// commissioning flag).
#[test]
fn extract_contract_dormant_zero_row_keeps_state_code_1() {
    let state = test_support::state_layout(0, 0);
    let entity_counts = entity_counts_contracts(vec![3]);
    let ec = zero_energy_conversion(0, 1);
    let diversion = HashMap::new();

    // Import column pinned to 0.0 (dormant) at 4..5.
    let primal = vec![0.0, 0.0, 0.0, 0.0, 0.0];
    let dual = vec![0.0; 1];
    let view = SolutionView {
        primal: &primal,
        dual: &dual,
        objective: 0.0,
        objective_coeffs: &[],
        row_lower: &[],
    };
    let geometry = StageGeometry {
        n_blks: 1,
        contract_import: 4..5,
        contract_export: 5..5,
        ..test_support::equipment_free_geometry(&[1]).remove(0)
    };
    let study_dims = test_support::study_dims();
    let block_hours = [730.0_f64];
    let prices = [200.0_f64];
    let contract_slots = [(ContractType::Import, 0)];
    let spec = contract_only_spec(
        &study_dims,
        &geometry,
        &state,
        &entity_counts,
        &block_hours,
        &prices,
        &contract_slots,
        &ec,
        &diversion,
    );

    let rows = extract_contracts(&view, &spec, 0);

    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].power_mw, 0.0);
    assert_eq!(rows[0].total_cost, 0.0);
    assert_eq!(rows[0].operative_state_code, 1);
}

/// Two import contracts on a three-block stage; contract 7's block 1
/// carries an overridden `120.0` price over its stage-wide `80.0`, while
/// contract 8's three blocks stay `80.0` — pins that the per-block price read
/// is indexed `c * n_blks + blk`, not `c` alone (which would misalign every
/// contract past the first against the flat per-block table).
#[test]
fn test_contract_extraction_uses_per_block_price() {
    let state = test_support::state_layout(0, 0);
    let entity_counts = entity_counts_contracts(vec![7, 8]);
    let ec = zero_energy_conversion(0, 1);
    let diversion = HashMap::new();

    // Both imports, 3 blocks each: contract 7 at columns 4..7 (10, 20, 30),
    // contract 8 at columns 7..10 (40, 50, 60); export block empty at 10..10.
    let primal = vec![0.0, 0.0, 0.0, 0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
    let dual = vec![0.0; 1];
    let view = SolutionView {
        primal: &primal,
        dual: &dual,
        objective: 0.0,
        objective_coeffs: &[],
        row_lower: &[],
    };
    let geometry = StageGeometry {
        n_blks: 3,
        contract_import: 4..10,
        contract_export: 10..10,
        ..test_support::equipment_free_geometry(&[3]).remove(0)
    };
    let study_dims = test_support::study_dims();
    let block_hours = [730.0_f64, 730.0, 730.0];
    // contract 7 (c=0): [80, 120, 80]; contract 8 (c=1): [80, 80, 80].
    let prices = [80.0_f64, 120.0, 80.0, 80.0, 80.0, 80.0];
    let contract_slots = [(ContractType::Import, 0), (ContractType::Import, 1)];
    let spec = contract_only_spec(
        &study_dims,
        &geometry,
        &state,
        &entity_counts,
        &block_hours,
        &prices,
        &contract_slots,
        &ec,
        &diversion,
    );

    let rows = extract_contracts(&view, &spec, 5);

    assert_eq!(rows.len(), 6, "2 contracts x 3 blocks = 6 rows");

    let c7 = &rows[0..3];
    assert_eq!(c7[0].price_per_mwh, 80.0);
    assert_eq!(c7[0].total_cost, 80.0 * 10.0 * 730.0);
    assert_eq!(
        c7[1].price_per_mwh, 120.0,
        "contract 7 block 1 carries the override"
    );
    assert_eq!(c7[1].total_cost, 120.0 * 20.0 * 730.0);
    assert_eq!(c7[2].price_per_mwh, 80.0);
    assert_eq!(c7[2].total_cost, 80.0 * 30.0 * 730.0);

    let c8 = &rows[3..6];
    assert_eq!(c8[0].price_per_mwh, 80.0);
    assert_eq!(c8[0].total_cost, 80.0 * 40.0 * 730.0);
    assert_eq!(
        c8[1].price_per_mwh, 80.0,
        "contract 8 block 1 must not read contract 7's override"
    );
    assert_eq!(c8[1].total_cost, 80.0 * 50.0 * 730.0);
    assert_eq!(c8[2].price_per_mwh, 80.0);
    assert_eq!(c8[2].total_cost, 80.0 * 60.0 * 730.0);
}

/// A `contract_prices` slice whose length is not `n_contracts * n_blks`
/// panics via the stride `debug_assert!`, naming both lengths.
/// `debug_assert!` compiles out in release, so this is gated on
/// `debug_assertions`.
#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "contract_prices stride mismatch")]
fn test_contract_price_stride_mismatch_asserts() {
    let state = test_support::state_layout(0, 0);
    let entity_counts = entity_counts_contracts(vec![7, 8]);
    let ec = zero_energy_conversion(0, 1);
    let diversion = HashMap::new();

    let primal = vec![0.0; 10];
    let dual = vec![0.0; 1];
    let view = SolutionView {
        primal: &primal,
        dual: &dual,
        objective: 0.0,
        objective_coeffs: &[],
        row_lower: &[],
    };
    let geometry = StageGeometry {
        n_blks: 3,
        contract_import: 4..10,
        contract_export: 10..10,
        ..test_support::equipment_free_geometry(&[3]).remove(0)
    };
    let study_dims = test_support::study_dims();
    let block_hours = [730.0_f64, 730.0, 730.0];
    // Wrong length: 5 instead of n_contracts * n_blks == 2 * 3 == 6.
    let prices = [80.0_f64, 80.0, 80.0, 80.0, 80.0];
    let contract_slots = [(ContractType::Import, 0), (ContractType::Import, 1)];
    let spec = contract_only_spec(
        &study_dims,
        &geometry,
        &state,
        &entity_counts,
        &block_hours,
        &prices,
        &contract_slots,
        &ec,
        &diversion,
    );

    let _ = extract_contracts(&view, &spec, 0);
}

/// A contract-free system yields an empty contracts vector from
/// `extract_stub_collections`.
#[test]
fn extract_stub_collections_contract_free_is_empty() {
    let state = test_support::state_layout(0, 0);
    let entity_counts = entity_counts_contracts(vec![]);
    let ec = zero_energy_conversion(0, 1);
    let diversion = HashMap::new();

    let primal = vec![0.0; 4];
    let dual = vec![0.0; 1];
    let view = SolutionView {
        primal: &primal,
        dual: &dual,
        objective: 0.0,
        objective_coeffs: &[],
        row_lower: &[],
    };
    let geometry = StageGeometry {
        n_blks: 1,
        ..test_support::equipment_free_geometry(&[1]).remove(0)
    };
    let study_dims = test_support::study_dims();
    let spec = contract_only_spec(
        &study_dims,
        &geometry,
        &state,
        &entity_counts,
        &[730.0],
        &[],
        &[],
        &ec,
        &diversion,
    );

    let (_inflow_lags, _pumping, contracts) = extract_stub_collections(&view, &spec, 0);
    assert!(contracts.is_empty(), "no contracts => empty vector");
}

// -------------------------------------------------------------------------
// Per-block chronological storage and evaporation
// -------------------------------------------------------------------------

/// One evaporating hydro, one operating block, `state_layout(1, 0)` gives
/// `storage_in.start == 2` and `storage.start == 0`.
fn entity_counts_1_hydro() -> EntityCounts {
    EntityCounts {
        hydro_ids: vec![7],
        hydro_productivities: vec![0.0],
        thermal_ids: vec![],
        line_ids: vec![],
        bus_ids: vec![],
        pumping_station_ids: vec![],
        contract_ids: vec![],
        non_controllable_ids: vec![],
    }
}

/// Build a single-hydro `K`-block `StageGeometry` in the given mode.
///
/// Control-region layout for `state_layout(1, 0)` (`control_region_start == 4`):
/// interior storage `S¹ … Sᴷ⁻¹` at `[4, 4 + (K−1))`, then turbine `[t0, t0 + K)`,
/// spillage `[t0 + K, t0 + 2K)`, then `evaporation_slot_count(block_mode, k)`
/// evaporation triples, then one withdrawal-slack column pair, then the four
/// `K`-wide operational-violation slack families. In parallel mode the
/// interior family is empty and turbine begins at 4.
fn single_hydro_block_geometry(block_mode: BlockMode, k: usize) -> StageGeometry {
    use crate::lp::indexer::EvaporationIndices;
    let n_interior = match block_mode {
        BlockMode::Chronological => k - 1,
        BlockMode::Parallel => 0,
    };
    let storage_internal_start = 4;
    let turbine_start = storage_internal_start + n_interior;
    let spillage_start = turbine_start + k;
    let evap_start = spillage_start + k;
    let n_evap_slots = evaporation_slot_count(block_mode, k);
    let evap_indices: Vec<EvaporationIndices> = (0..n_evap_slots)
        .map(|slot| {
            let base = evap_start + slot * 3;
            EvaporationIndices {
                evaporation_flow_col: base,
                f_evap_plus_col: base + 1,
                f_evap_minus_col: base + 2,
                evap_row: slot,
            }
        })
        .collect();
    // One hydro: a single stage-level water-balance row in Parallel mode
    // (every block reads it), one row per block in Chronological mode.
    let water_balance = match block_mode {
        BlockMode::Chronological => BlockRowFamily::per_block(0..k),
        BlockMode::Parallel => BlockRowFamily::one_per_entity(0..1),
    };
    let evap_end = evap_start + n_evap_slots * 3;
    let ws_end = evap_end + 2;
    StageGeometry {
        turbine: turbine_start..spillage_start,
        spillage: spillage_start..evap_start,
        n_blks: k,
        storage_internal_start,
        block_mode,
        evap_indices,
        evap_hydro_indices: vec![HydroSys::new(0)],
        water_balance,
        withdrawal_slack_neg: evap_end..evap_end + 1,
        withdrawal_slack_pos: evap_end + 1..ws_end,
        outflow_below_slack: ws_end..ws_end + k,
        outflow_above_slack: ws_end + k..ws_end + 2 * k,
        turbine_below_slack: ws_end + 2 * k..ws_end + 3 * k,
        generation_below_slack: ws_end + 3 * k..ws_end + 4 * k,
        ..test_support::equipment_free_geometry(&[k]).remove(0)
    }
}

/// Chronological block `b` reports `(Sᵇ, Sᵇ⁺¹)` read through the accessor, and the
/// boundary chain's endpoints coincide with the state columns (`S⁰`/`Sᴷ`).
#[test]
fn extract_chronological_per_block_storage() {
    let k = 3_usize;
    let geom = single_hydro_block_geometry(BlockMode::Chronological, k);
    let study_dims = test_support::study_dims();
    let state = test_support::state_layout(1, 0);
    let ec = zero_energy_conversion(1, 1);

    // Boundary values: S⁰=10 (storage_in col 2), S¹=20 (col 4), S²=30 (col 5),
    // S³=Sᴷ=40 (storage col 0). Turbine cols [6,9), evap triples at [12, ...).
    let n_cols = geom.generation_below_slack.end;
    let mut primal = vec![0.0_f64; n_cols];
    primal[0] = 40.0; // Sᴷ (outgoing state)
    primal[2] = 10.0; // S⁰ (incoming state)
    primal[4] = 20.0; // S¹ (interior)
    primal[5] = 30.0; // S² (interior)
    let dual = vec![0.0_f64; 4];

    let spec = StageExtractionSpec {
        study_dims: &study_dims,
        geometry: &geom,
        hydro_cell_index: &test_support::identity_hydro_cell_index(256),
        state: &state,
        entity_counts: &entity_counts_1_hydro(),
        inflow_m3s_per_hydro: &[0.0; 1],
        block_hours: &[100.0, 100.0, 100.0],
        generic_constraint_entries: &[],
        ncs_col_upper: &[],
        pumping_consumption_mw_per_m3s: &[],
        contract_prices: &[],
        contract_slots: &[],
        diversion_upstream: &HashMap::new(),
        hydro_productivities: &[0.0],
        col_scale: &[],
        row_scale: &[],
        cumulative_discount_factor: 1.0,
        cost_scale_factor: 1_000_000.0,
        energy_conversion: &ec,
        hydro_min_storage_hm3: &[0.0],
        stage_index: 0,
        horizon: &HorizonMode::Finite { num_stages: 1 },
        anticipated_windows: &[],
        study_stage_ids: &[],
    };
    let view = SolutionView {
        primal: &primal,
        dual: &dual,
        objective: 0.0,
        objective_coeffs: &vec![0.0_f64; n_cols],
        row_lower: &[],
    };

    let result = extract_stage_result(&view, &spec, 0);
    assert_eq!(result.hydros.len(), k);

    let boundaries = [10.0, 20.0, 30.0, 40.0];
    for b in 0..k {
        let row = &result.hydros[b];
        assert_eq!(row.block_id, Some(b as u32));
        assert_eq!(
            row.storage_initial_hm3, boundaries[b],
            "block {b} incoming == Sᵇ"
        );
        assert_eq!(
            row.storage_final_hm3,
            boundaries[b + 1],
            "block {b} outgoing == Sᵇ⁺¹"
        );
    }

    // Endpoints coincide with the state region.
    assert_eq!(
        result.hydros[0].storage_initial_hm3,
        primal[state.storage_in.start]
    );
    assert_eq!(
        result.hydros[k - 1].storage_final_hm3,
        primal[state.storage.start]
    );
}

/// Parallel mode: every block row reports the stage-level `(S⁰, Sᴷ)` state pair,
/// `.to_bits()`-identical to the direct state-column reads.
#[test]
fn extract_parallel_per_block_storage_byte_identical() {
    let k = 3_usize;
    let geom = single_hydro_block_geometry(BlockMode::Parallel, k);
    let study_dims = test_support::study_dims();
    let state = test_support::state_layout(1, 0);
    let ec = zero_energy_conversion(1, 1);

    let n_cols = geom.generation_below_slack.end;
    let mut primal = vec![0.0_f64; n_cols];
    primal[0] = 40.0; // Sᴷ (outgoing state)
    primal[2] = 10.0; // S⁰ (incoming state)
    let dual = vec![0.0_f64; 4];

    let spec = StageExtractionSpec {
        study_dims: &study_dims,
        geometry: &geom,
        hydro_cell_index: &test_support::identity_hydro_cell_index(256),
        state: &state,
        entity_counts: &entity_counts_1_hydro(),
        inflow_m3s_per_hydro: &[0.0; 1],
        block_hours: &[100.0, 100.0, 100.0],
        generic_constraint_entries: &[],
        ncs_col_upper: &[],
        pumping_consumption_mw_per_m3s: &[],
        contract_prices: &[],
        contract_slots: &[],
        diversion_upstream: &HashMap::new(),
        hydro_productivities: &[0.0],
        col_scale: &[],
        row_scale: &[],
        cumulative_discount_factor: 1.0,
        cost_scale_factor: 1_000_000.0,
        energy_conversion: &ec,
        hydro_min_storage_hm3: &[0.0],
        stage_index: 0,
        horizon: &HorizonMode::Finite { num_stages: 1 },
        anticipated_windows: &[],
        study_stage_ids: &[],
    };
    let view = SolutionView {
        primal: &primal,
        dual: &dual,
        objective: 0.0,
        objective_coeffs: &vec![0.0_f64; n_cols],
        row_lower: &[],
    };

    let result = extract_stage_result(&view, &spec, 0);
    assert_eq!(result.hydros.len(), k);
    let s0 = primal[state.storage_in.start];
    let sk = primal[state.storage.start];
    for row in &result.hydros {
        assert_eq!(row.storage_initial_hm3.to_bits(), s0.to_bits());
        assert_eq!(row.storage_final_hm3.to_bits(), sk.to_bits());
    }
}

/// Chronological block `b` reads its own water-balance row through
/// `water_balance_row`: `.to_bits()`-identical to `dual[1 + b] * cost_scale_factor`.
#[test]
fn extract_chronological_water_value_reads_each_block_row() {
    let k = 3_usize;
    let geom = StageGeometry {
        water_balance: BlockRowFamily::per_block(1..1 + k),
        ..single_hydro_block_geometry(BlockMode::Chronological, k)
    };
    let study_dims = test_support::study_dims();
    let state = test_support::state_layout(1, 0);
    let ec = zero_energy_conversion(1, 1);

    let n_cols = geom.generation_below_slack.end;
    let primal = vec![0.0_f64; n_cols];
    let dual = vec![0.0_f64, 10.0, 20.0, 30.0];

    let spec = StageExtractionSpec {
        study_dims: &study_dims,
        geometry: &geom,
        hydro_cell_index: &test_support::identity_hydro_cell_index(256),
        state: &state,
        entity_counts: &entity_counts_1_hydro(),
        inflow_m3s_per_hydro: &[0.0; 1],
        block_hours: &[100.0, 100.0, 100.0],
        generic_constraint_entries: &[],
        ncs_col_upper: &[],
        pumping_consumption_mw_per_m3s: &[],
        contract_prices: &[],
        contract_slots: &[],
        diversion_upstream: &HashMap::new(),
        hydro_productivities: &[0.0],
        col_scale: &[],
        row_scale: &[],
        cumulative_discount_factor: 1.0,
        cost_scale_factor: 1_000_000.0,
        energy_conversion: &ec,
        hydro_min_storage_hm3: &[0.0],
        stage_index: 0,
        horizon: &HorizonMode::Finite { num_stages: 1 },
        anticipated_windows: &[],
        study_stage_ids: &[],
    };
    let view = SolutionView {
        primal: &primal,
        dual: &dual,
        objective: 0.0,
        objective_coeffs: &vec![0.0_f64; n_cols],
        row_lower: &[],
    };

    let result = extract_stage_result(&view, &spec, 0);
    assert_eq!(result.hydros.len(), k);
    for (b, row) in result.hydros.iter().enumerate() {
        assert_eq!(
            row.water_value_per_hm3.to_bits(),
            (dual[1 + b] * spec.cost_scale_factor).to_bits(),
            "block {b} must report its own water-balance row"
        );
    }
}

/// Parallel mode: every block row repeats the single stage water-balance row,
/// `.to_bits()`-identical to `dual[1] * cost_scale_factor`.
#[test]
fn extract_parallel_water_value_repeats_the_stage_row() {
    let k = 3_usize;
    let geom = StageGeometry {
        water_balance: BlockRowFamily::one_per_entity(1..2),
        ..single_hydro_block_geometry(BlockMode::Parallel, k)
    };
    let study_dims = test_support::study_dims();
    let state = test_support::state_layout(1, 0);
    let ec = zero_energy_conversion(1, 1);

    let n_cols = geom.generation_below_slack.end;
    let primal = vec![0.0_f64; n_cols];
    let dual = vec![0.0_f64, 10.0, 20.0, 30.0];

    let spec = StageExtractionSpec {
        study_dims: &study_dims,
        geometry: &geom,
        hydro_cell_index: &test_support::identity_hydro_cell_index(256),
        state: &state,
        entity_counts: &entity_counts_1_hydro(),
        inflow_m3s_per_hydro: &[0.0; 1],
        block_hours: &[100.0, 100.0, 100.0],
        generic_constraint_entries: &[],
        ncs_col_upper: &[],
        pumping_consumption_mw_per_m3s: &[],
        contract_prices: &[],
        contract_slots: &[],
        diversion_upstream: &HashMap::new(),
        hydro_productivities: &[0.0],
        col_scale: &[],
        row_scale: &[],
        cumulative_discount_factor: 1.0,
        cost_scale_factor: 1_000_000.0,
        energy_conversion: &ec,
        hydro_min_storage_hm3: &[0.0],
        stage_index: 0,
        horizon: &HorizonMode::Finite { num_stages: 1 },
        anticipated_windows: &[],
        study_stage_ids: &[],
    };
    let view = SolutionView {
        primal: &primal,
        dual: &dual,
        objective: 0.0,
        objective_coeffs: &vec![0.0_f64; n_cols],
        row_lower: &[],
    };

    let result = extract_stage_result(&view, &spec, 0);
    assert_eq!(result.hydros.len(), k);
    let expected = (dual[1] * spec.cost_scale_factor).to_bits();
    for row in &result.hydros {
        assert_eq!(row.water_value_per_hm3.to_bits(), expected);
    }
}

/// Chronological per-block stored energy derives from each block's own boundary:
/// `(S − V_min) · ρ_acum_integrated · ENERGY_FACTOR`, with `V_min` /
/// `ρ_acum_integrated` block-invariant. A plain `EnergyConversionSet::new` with no
/// `with_integrated` defaults the integrated grid to a clone of the
/// reference-point one, so `ρ_acum_integrated` coincides with `ρ_acum` here.
#[test]
fn extract_chronological_per_block_stored_energy() {
    use crate::energy_conversion::EnergyConversion;
    let k = 3_usize;
    let geom = single_hydro_block_geometry(BlockMode::Chronological, k);
    let study_dims = test_support::study_dims();
    let state = test_support::state_layout(1, 0);

    let rho_acum_integrated = 2.0_f64;
    let v_min = 5.0_f64;
    let ec = EnergyConversionSet::new(
        vec![vec![EnergyConversion {
            equivalent_productivity_mw_per_m3s: 0.0,
            reference_volume_hm3: 0.0,
            reference_outflow_m3s: 0.0,
        }]],
        vec![vec![rho_acum_integrated]],
        &test_support::minimal_hydros(1),
        1,
    );

    let n_cols = geom.generation_below_slack.end;
    let mut primal = vec![0.0_f64; n_cols];
    primal[0] = 40.0; // Sᴷ
    primal[2] = 10.0; // S⁰
    primal[4] = 20.0; // S¹
    primal[5] = 30.0; // S²
    let dual = vec![0.0_f64; 4];

    let spec = StageExtractionSpec {
        study_dims: &study_dims,
        geometry: &geom,
        hydro_cell_index: &test_support::identity_hydro_cell_index(256),
        state: &state,
        entity_counts: &entity_counts_1_hydro(),
        inflow_m3s_per_hydro: &[0.0; 1],
        block_hours: &[100.0, 100.0, 100.0],
        generic_constraint_entries: &[],
        ncs_col_upper: &[],
        pumping_consumption_mw_per_m3s: &[],
        contract_prices: &[],
        contract_slots: &[],
        diversion_upstream: &HashMap::new(),
        hydro_productivities: &[0.0],
        col_scale: &[],
        row_scale: &[],
        cumulative_discount_factor: 1.0,
        cost_scale_factor: 1_000_000.0,
        energy_conversion: &ec,
        hydro_min_storage_hm3: &[v_min],
        stage_index: 0,
        horizon: &HorizonMode::Finite { num_stages: 1 },
        anticipated_windows: &[],
        study_stage_ids: &[],
    };
    let view = SolutionView {
        primal: &primal,
        dual: &dual,
        objective: 0.0,
        objective_coeffs: &vec![0.0_f64; n_cols],
        row_lower: &[],
    };

    let result = extract_stage_result(&view, &spec, 0);
    let boundaries = [10.0, 20.0, 30.0, 40.0];
    let expected = |s: f64| -> f64 {
        (s - v_min) * rho_acum_integrated * super::ENERGY_FACTOR_MWH_PER_HM3_PER_MW_PER_M3S
    };
    for b in 0..k {
        let row = &result.hydros[b];
        assert!((row.stored_energy_initial_mwh - expected(boundaries[b])).abs() < 1e-9);
        assert!((row.stored_energy_final_mwh - expected(boundaries[b + 1])).abs() < 1e-9);
    }
}

/// Chronological block `b` reads its OWN evaporation triple
/// `evap_indices[local * n_blks + b]`, not block 0's.
#[test]
fn extract_chronological_per_block_evaporation() {
    let k = 3_usize;
    let geom = single_hydro_block_geometry(BlockMode::Chronological, k);
    let study_dims = test_support::study_dims();
    let state = test_support::state_layout(1, 0);
    let ec = zero_energy_conversion(1, 1);

    let n_cols = geom.generation_below_slack.end;
    let mut primal = vec![0.0_f64; n_cols];
    primal[0] = 40.0;
    primal[2] = 10.0;
    primal[4] = 20.0;
    primal[5] = 30.0;
    // Distinct evaporation values per block: block b flow = 1.0 + b, neg = 0.1 + b,
    // pos = 0.2 + b, laid out block-major at evap_indices[b].
    for b in 0..k {
        let ei = &geom.evap_indices[b];
        primal[ei.evaporation_flow_col] = 1.0 + b as f64;
        primal[ei.f_evap_plus_col] = 0.1 + b as f64;
        primal[ei.f_evap_minus_col] = 0.2 + b as f64;
    }
    let dual = vec![0.0_f64; 4];

    let spec = StageExtractionSpec {
        study_dims: &study_dims,
        geometry: &geom,
        hydro_cell_index: &test_support::identity_hydro_cell_index(256),
        state: &state,
        entity_counts: &entity_counts_1_hydro(),
        inflow_m3s_per_hydro: &[0.0; 1],
        block_hours: &[100.0, 100.0, 100.0],
        generic_constraint_entries: &[],
        ncs_col_upper: &[],
        pumping_consumption_mw_per_m3s: &[],
        contract_prices: &[],
        contract_slots: &[],
        diversion_upstream: &HashMap::new(),
        hydro_productivities: &[0.0],
        col_scale: &[],
        row_scale: &[],
        cumulative_discount_factor: 1.0,
        cost_scale_factor: 1_000_000.0,
        energy_conversion: &ec,
        hydro_min_storage_hm3: &[0.0],
        stage_index: 0,
        horizon: &HorizonMode::Finite { num_stages: 1 },
        anticipated_windows: &[],
        study_stage_ids: &[],
    };
    let view = SolutionView {
        primal: &primal,
        dual: &dual,
        objective: 0.0,
        objective_coeffs: &vec![0.0_f64; n_cols],
        row_lower: &[],
    };

    let result = extract_stage_result(&view, &spec, 0);
    for b in 0..k {
        let row = &result.hydros[b];
        assert_eq!(row.evaporation_m3s, Some(1.0 + b as f64), "block {b} flow");
        assert!(
            (row.evaporation_violation_neg_m3s - (0.1 + b as f64)).abs() < 1e-12,
            "block {b} neg violation reads its own triple"
        );
        assert!(
            (row.evaporation_violation_pos_m3s - (0.2 + b as f64)).abs() < 1e-12,
            "block {b} pos violation reads its own triple"
        );
    }
}

/// Parallel mode: every block row's evaporation fields are `.to_bits()`-identical to
/// the single stage-level slot's read.
#[test]
fn extract_parallel_per_block_evaporation_byte_identical() {
    let k = 3_usize;
    let geom = single_hydro_block_geometry(BlockMode::Parallel, k);
    let study_dims = test_support::study_dims();
    let state = test_support::state_layout(1, 0);
    let ec = zero_energy_conversion(1, 1);

    assert_eq!(
        geom.evap_indices.len(),
        1,
        "a parallel stage reserves exactly one evaporation slot"
    );
    let n_cols = geom.generation_below_slack.end;
    let mut primal = vec![0.0_f64; n_cols];
    primal[0] = 40.0;
    primal[2] = 10.0;
    let ei = &geom.evap_indices[0];
    primal[ei.evaporation_flow_col] = 1.0;
    primal[ei.f_evap_plus_col] = 0.1;
    primal[ei.f_evap_minus_col] = 0.2;
    let dual = vec![0.0_f64; 4];

    let spec = StageExtractionSpec {
        study_dims: &study_dims,
        geometry: &geom,
        hydro_cell_index: &test_support::identity_hydro_cell_index(256),
        state: &state,
        entity_counts: &entity_counts_1_hydro(),
        inflow_m3s_per_hydro: &[0.0; 1],
        block_hours: &[100.0, 100.0, 100.0],
        generic_constraint_entries: &[],
        ncs_col_upper: &[],
        pumping_consumption_mw_per_m3s: &[],
        contract_prices: &[],
        contract_slots: &[],
        diversion_upstream: &HashMap::new(),
        hydro_productivities: &[0.0],
        col_scale: &[],
        row_scale: &[],
        cumulative_discount_factor: 1.0,
        cost_scale_factor: 1_000_000.0,
        energy_conversion: &ec,
        hydro_min_storage_hm3: &[0.0],
        stage_index: 0,
        horizon: &HorizonMode::Finite { num_stages: 1 },
        anticipated_windows: &[],
        study_stage_ids: &[],
    };
    let view = SolutionView {
        primal: &primal,
        dual: &dual,
        objective: 0.0,
        objective_coeffs: &vec![0.0_f64; n_cols],
        row_lower: &[],
    };

    let result = extract_stage_result(&view, &spec, 0);
    // The single stage-level slot's triple is what every block row reads.
    let flow0 = 1.0_f64;
    let neg0 = 0.1_f64;
    let pos0 = 0.2_f64;
    for row in &result.hydros {
        assert_eq!(row.evaporation_m3s.map(f64::to_bits), Some(flow0.to_bits()));
        assert_eq!(row.evaporation_violation_neg_m3s.to_bits(), neg0.to_bits());
        assert_eq!(row.evaporation_violation_pos_m3s.to_bits(), pos0.to_bits());
    }
}

// -------------------------------------------------------------------------
// Travel-time in-transit bucket extraction
// -------------------------------------------------------------------------

/// Primal for a (`hydro_count = 2`, `max_par_order = 1`) bucketed layout with
/// `transit_buckets_out.len()` buckets. Layout: `storage`[0,2) `lags`[2,4)
/// `transit_buckets_out`[4,4+B) `z_inflow`[4+B,6+B) `storage_in`[6+B,8+B)
/// `transit_buckets_in`[8+B,8+2B) `theta`=8+2B.
fn make_transit_bucket_primal(transit_buckets_out: &[f64], transit_buckets_in: &[f64]) -> Vec<f64> {
    assert_eq!(transit_buckets_out.len(), transit_buckets_in.len());
    let b = transit_buckets_out.len();
    let theta = 8 + 2 * b;
    let mut p = vec![0.0_f64; theta + 1];
    p[0] = 100.0;
    p[1] = 200.0;
    p[2] = 50.0;
    p[3] = 60.0;
    for (i, &v) in transit_buckets_out.iter().enumerate() {
        p[4 + i] = v;
    }
    p[6 + b] = 90.0;
    p[7 + b] = 180.0;
    for (i, &v) in transit_buckets_in.iter().enumerate() {
        p[8 + b + i] = v;
    }
    p[theta] = 999.5;
    p
}

/// Shape + canonical order + delayed-arrival: one downstream plant (state index
/// 0 = `hydro_id` 10) with two maturity buckets. `in_transit_volume_hm3` reads
/// the outgoing bucket state; `delayed_arrival_hm3` reads the incoming `b_1^in`
/// only at `lag == 1`.
#[test]
fn extract_transit_buckets_shape_canonical_order_and_delayed_arrival() {
    let study_dims = test_support::study_dims();
    let state = test_support::state_layout_with_transit_buckets(
        2,
        1,
        vec![(HydroSys::new(0), 1), (HydroSys::new(0), 2)],
        vec![],
    );
    let ws_start = state.control_region_start();
    let geometry = StageGeometry {
        withdrawal_slack_neg: ws_start..ws_start + 2,
        withdrawal_slack_pos: ws_start + 2..ws_start + 4,
        ..test_support::equipment_free_geometry(&[0]).remove(0)
    };
    let mut primal = make_transit_bucket_primal(&[11.0, 22.0], &[7.0, 8.0]);
    primal.resize(geometry.withdrawal_slack_pos.end, 0.0);
    let objective_coeffs = vec![0.0; primal.len()];
    let dual = vec![0.0; 4];
    let ec = zero_energy_conversion(2, 1);
    let inflow_m3s_per_hydro = test_support::inflow_m3s_per_hydro_from_primal(&state, &primal, 2);

    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &dual,
            objective: 0.0,
            objective_coeffs: &objective_coeffs,
            row_lower: &[],
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &geometry,
            hydro_cell_index: &test_support::identity_hydro_cell_index(256),
            state: &state,
            entity_counts: &make_entity_counts_2_hydros(),
            inflow_m3s_per_hydro: &inflow_m3s_per_hydro,
            block_hours: &[],
            generic_constraint_entries: &[],
            ncs_col_upper: &[],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_slots: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[1.0, 1.0],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            cost_scale_factor: 1_000_000.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0; 2],
            stage_index: 0,
            horizon: &HorizonMode::Finite { num_stages: 1 },
            anticipated_windows: &[],
            study_stage_ids: &[],
        },
        5,
    );

    let tb = &result.transit_buckets;
    assert_eq!(tb.len(), 2, "one row per declared bucket");
    assert_eq!(tb[0].stage_id, 5);
    assert_eq!(tb[0].hydro_id, 10);
    assert_eq!(tb[0].lag, 1);
    assert_eq!(tb[0].in_transit_volume_hm3, 11.0);
    assert_eq!(tb[0].delayed_arrival_hm3, 7.0);
    assert_eq!(tb[1].hydro_id, 10);
    assert_eq!(tb[1].lag, 2);
    assert_eq!(tb[1].in_transit_volume_hm3, 22.0);
    assert_eq!(
        tb[1].delayed_arrival_hm3, 0.0,
        "delayed arrival is non-zero only at the maturing lag 1"
    );
}

/// Absent when undeclared: `n_buckets == 0` produces no in-transit rows, keeping
/// the whole table off for a non-travel-time study.
#[test]
fn extract_transit_buckets_absent_when_n_buckets_zero() {
    let study_dims = test_support::study_dims();
    let state = test_support::state_layout(2, 1);
    let ws_start = state.control_region_start();
    let geometry = StageGeometry {
        withdrawal_slack_neg: ws_start..ws_start + 2,
        withdrawal_slack_pos: ws_start + 2..ws_start + 4,
        ..test_support::equipment_free_geometry(&[0]).remove(0)
    };
    let mut primal = make_primal_2_1([100.0, 200.0], [50.0, 60.0], [90.0, 180.0], 999.5);
    primal.resize(geometry.withdrawal_slack_pos.end, 0.0);
    let objective_coeffs = vec![0.0; primal.len()];
    let dual = vec![0.0; 4];
    let ec = zero_energy_conversion(2, 1);

    let inflow_m3s_per_hydro = test_support::inflow_m3s_per_hydro_from_primal(&state, &primal, 2);
    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &dual,
            objective: 0.0,
            objective_coeffs: &objective_coeffs,
            row_lower: &[],
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &geometry,
            hydro_cell_index: &test_support::identity_hydro_cell_index(256),
            state: &state,
            entity_counts: &make_entity_counts_2_hydros(),
            inflow_m3s_per_hydro: &inflow_m3s_per_hydro,
            block_hours: &[],
            generic_constraint_entries: &[],
            ncs_col_upper: &[],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_slots: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[1.0, 1.0],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            cost_scale_factor: 1_000_000.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0; 2],
            stage_index: 0,
            horizon: &HorizonMode::Finite { num_stages: 1 },
            anticipated_windows: &[],
            study_stage_ids: &[],
        },
        0,
    );

    assert!(result.transit_buckets.is_empty());
}

/// Row order follows the canonical `transit_bucket_column_order` (grouped by plant,
/// ascending lag) with each plant's `b_1^in` reported only at its lag-1 row.
/// Together with `bucket_topology`'s declaration-order-invariant column order,
/// this pins the output table to a deterministic row/column order regardless of
/// hydro input ordering.
#[test]
fn extract_transit_buckets_rows_follow_canonical_column_order() {
    let study_dims = test_support::study_dims();
    // Plant 0 (hydro_id 10) depth 2, plant 1 (hydro_id 20) depth 1.
    let state = test_support::state_layout_with_transit_buckets(
        2,
        1,
        vec![
            (HydroSys::new(0), 1),
            (HydroSys::new(0), 2),
            (HydroSys::new(1), 1),
        ],
        vec![],
    );
    let ws_start = state.control_region_start();
    let geometry = StageGeometry {
        withdrawal_slack_neg: ws_start..ws_start + 2,
        withdrawal_slack_pos: ws_start + 2..ws_start + 4,
        ..test_support::equipment_free_geometry(&[0]).remove(0)
    };
    let mut primal = make_transit_bucket_primal(&[11.0, 22.0, 33.0], &[7.0, 8.0, 9.0]);
    primal.resize(geometry.withdrawal_slack_pos.end, 0.0);
    let objective_coeffs = vec![0.0; primal.len()];
    let dual = vec![0.0; 4];
    let ec = zero_energy_conversion(2, 1);
    let inflow_m3s_per_hydro = test_support::inflow_m3s_per_hydro_from_primal(&state, &primal, 2);

    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &dual,
            objective: 0.0,
            objective_coeffs: &objective_coeffs,
            row_lower: &[],
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &geometry,
            hydro_cell_index: &test_support::identity_hydro_cell_index(256),
            state: &state,
            entity_counts: &make_entity_counts_2_hydros(),
            inflow_m3s_per_hydro: &inflow_m3s_per_hydro,
            block_hours: &[],
            generic_constraint_entries: &[],
            ncs_col_upper: &[],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_slots: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[1.0, 1.0],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            cost_scale_factor: 1_000_000.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0; 2],
            stage_index: 0,
            horizon: &HorizonMode::Finite { num_stages: 1 },
            anticipated_windows: &[],
            study_stage_ids: &[],
        },
        0,
    );

    let seq: Vec<(i32, u32)> = result
        .transit_buckets
        .iter()
        .map(|r| (r.hydro_id, r.lag))
        .collect();
    assert_eq!(seq, vec![(10, 1), (10, 2), (20, 1)]);
    // Each plant's delayed arrival lands only on its lag-1 row.
    assert_eq!(result.transit_buckets[0].delayed_arrival_hm3, 7.0);
    assert_eq!(result.transit_buckets[1].delayed_arrival_hm3, 0.0);
    assert_eq!(result.transit_buckets[2].delayed_arrival_hm3, 9.0);
    assert_eq!(result.transit_buckets[2].in_transit_volume_hm3, 33.0);
}

// -------------------------------------------------------------------------
// turbined/generation are sums over a plant's cells
// -------------------------------------------------------------------------

/// Plant 0 is single-bus (one cell); plant 1 splits across two buses (two
/// cells, `bus_id` 10 then 11 — ascending, so cell 1 is the bus-10 group and
/// cell 2 the bus-11 group) and is the sole FPHA plant. `n_cells = 3`,
/// `n_fpha_cells = 2`. Column layout (equipment starts at `theta + 1 = 7` for
/// `state_layout(2, 0)`): `turbine` = cell-major `[7, 13)` (3 cells x 2 blks),
/// `spillage` = plant-major `[13, 17)` (2 hydros x 2 blks — spillage stays
/// plant-keyed), `generation` = FPHA-cell-major `[17, 21)` (2 cells x 2 blks).
fn split_plant_multi_bus_extraction_fixture() -> (StateSpace, StageGeometry, HydroCellIndex) {
    let state = test_support::state_layout(2, 0);
    let hydro_cell_index = HydroCellIndex::build(&[
        test_support::geometry_hydro(0),
        test_support::geometry_hydro_with_groups(
            1,
            vec![
                test_support::make_unit_group(
                    cobre_core::EntityId(110),
                    cobre_core::EntityId(10),
                    0.0,
                    10.0,
                    0.0,
                    10.0,
                ),
                test_support::make_unit_group(
                    cobre_core::EntityId(111),
                    cobre_core::EntityId(11),
                    0.0,
                    10.0,
                    0.0,
                    10.0,
                ),
            ],
            cobre_core::HydroGenerationModel::Fpha,
        ),
    ]);
    assert_eq!(hydro_cell_index.n_cells(), 3);
    assert_eq!(hydro_cell_index.cells_of(HydroSys::new(0)), 0..1);
    assert_eq!(hydro_cell_index.cells_of(HydroSys::new(1)), 1..3);
    let geometry = StageGeometry {
        turbine: 7..13,
        spillage: 13..17,
        generation: 17..21,
        fpha_hydro_indices: vec![HydroSys::new(1)],
        n_blks: 2,
        water_balance: BlockRowFamily::one_per_entity(0..2),
        withdrawal_slack_neg: 21..23,
        withdrawal_slack_pos: 23..25,
        outflow_below_slack: 25..29,
        outflow_above_slack: 29..33,
        turbine_below_slack: 33..39,
        generation_below_slack: 39..45,
        ..test_support::equipment_free_geometry(&[2]).remove(0)
    };
    (state, geometry, hydro_cell_index)
}

/// A split plant's reported `turbined_m3s` and `generation_mw` are the SUM
/// over its cells, not a single-cell read — exact because every cell of a
/// plant shares that plant's one production model and objective coefficients
/// (no group- or cell-level production field exists to price them apart).
/// Cells carry deliberately UNEQUAL flows (100/101 vs 5/7, and 50/51 vs 3/4)
/// so a first-cell-only regression is visible; an even split would be
/// absorbed by construction.
#[test]
fn extract_hydro_turbined_and_generation_sum_over_a_split_plants_cells() {
    let (state, geometry, hydro_cell_index) = split_plant_multi_bus_extraction_fixture();
    let study_dims = test_support::study_dims();
    let ec = zero_energy_conversion(2, 1);

    let mut primal = vec![0.0_f64; 7]; // state region [0, 7), unused by this test
    primal.extend_from_slice(&[
        10.0, 11.0, // turbine[cell 0] (plant 0, blk 0/1)
        100.0, 101.0, // turbine[cell 1] (plant 1's bus-10 group, blk 0/1)
        5.0, 7.0, // turbine[cell 2] (plant 1's bus-11 group, blk 0/1)
        0.0, 0.0, // spillage[hydro 0] (plant-keyed, not per cell)
        0.0, 0.0, // spillage[hydro 1]
        50.0, 51.0, // generation[fpha-cell-local 0 = cell 1], blk 0/1
        3.0, 4.0, // generation[fpha-cell-local 1 = cell 2], blk 0/1
    ]);
    // Pads withdrawal-slack and the four operational-violation slack families
    // (this test asserts none of them) with always-zero columns.
    primal.resize(geometry.generation_below_slack.end, 0.0);
    let dual = vec![0.0; 2];
    let objective_coeffs = vec![0.0; primal.len()];

    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &dual,
            objective: 0.0,
            objective_coeffs: &objective_coeffs,
            row_lower: &[],
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &geometry,
            hydro_cell_index: &hydro_cell_index,
            state: &state,
            entity_counts: &EntityCounts {
                hydro_ids: vec![100, 200],
                hydro_productivities: vec![0.0; 2],
                thermal_ids: vec![],
                line_ids: vec![],
                bus_ids: vec![],
                pumping_station_ids: vec![],
                contract_ids: vec![],
                non_controllable_ids: vec![],
            },
            inflow_m3s_per_hydro: &[0.0, 0.0],
            block_hours: &[100.0, 100.0],
            generic_constraint_entries: &[],
            ncs_col_upper: &[],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_slots: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[2.0, 0.0], // plant 0 constant-productivity; plant 1 is FPHA
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            cost_scale_factor: 1_000_000.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0; 2],
            stage_index: 0,
            horizon: &HorizonMode::Finite { num_stages: 1 },
            anticipated_windows: &[],
            study_stage_ids: &[],
        },
        0,
    );

    assert_eq!(result.hydros.len(), 4, "2 hydros x 2 blocks");
    let plant0: Vec<_> = result.hydros.iter().filter(|h| h.hydro_id == 100).collect();
    let plant1: Vec<_> = result.hydros.iter().filter(|h| h.hydro_id == 200).collect();

    // Plant 0 (single cell): turbined passes through; generation is the
    // constant-productivity product (turbined * 2.0), not a cell sum.
    assert_eq!(plant0[0].turbined_m3s, 10.0);
    assert_eq!(plant0[1].turbined_m3s, 11.0);
    assert_eq!(plant0[0].generation_mw, 20.0);
    assert_eq!(plant0[1].generation_mw, 22.0);

    // Plant 1 (two cells): both turbined_m3s and generation_mw are the SUM
    // over cells 1 and 2, ascending — 100+5=105 / 101+7=108, 50+3=53 / 51+4=55.
    assert_eq!(plant1[0].turbined_m3s, 105.0);
    assert_eq!(plant1[1].turbined_m3s, 108.0);
    assert_eq!(plant1[0].generation_mw, 53.0);
    assert_eq!(plant1[1].generation_mw, 55.0);
}

// -------------------------------------------------------------------------
// hydro_bus_generation: one row per (hydro, block, cell)
// -------------------------------------------------------------------------

/// Three plants: position 0 (bus 41, one cell) and position 2 (bus 53, one
/// cell) bracket the split plant at position 1 (buses 31 then 37, two
/// cells) — a non-zero position preceded by a plant with a different cell
/// count. Bus ids are far from every hydro-system-position/cell-index/block
/// value so a mutation confusing any two of them is visible.
fn split_middle_plant_fixture() -> (StateSpace, StageGeometry, HydroCellIndex, Vec<f64>) {
    let state = test_support::state_layout(3, 0);
    let hydro_cell_index = HydroCellIndex::build(&[
        test_support::geometry_hydro_with_groups(
            0,
            vec![test_support::make_unit_group(
                cobre_core::EntityId(1),
                cobre_core::EntityId(41),
                0.0,
                10.0,
                0.0,
                10.0,
            )],
            cobre_core::HydroGenerationModel::ConstantProductivity,
        ),
        test_support::geometry_hydro_with_groups(
            1,
            vec![
                test_support::make_unit_group(
                    cobre_core::EntityId(2),
                    cobre_core::EntityId(31),
                    0.0,
                    10.0,
                    0.0,
                    10.0,
                ),
                test_support::make_unit_group(
                    cobre_core::EntityId(3),
                    cobre_core::EntityId(37),
                    0.0,
                    10.0,
                    0.0,
                    10.0,
                ),
            ],
            cobre_core::HydroGenerationModel::ConstantProductivity,
        ),
        test_support::geometry_hydro_with_groups(
            2,
            vec![test_support::make_unit_group(
                cobre_core::EntityId(4),
                cobre_core::EntityId(53),
                0.0,
                10.0,
                0.0,
                10.0,
            )],
            cobre_core::HydroGenerationModel::ConstantProductivity,
        ),
    ]);
    assert_eq!(hydro_cell_index.n_cells(), 4, "1 + 2 + 1 cells");
    assert_eq!(hydro_cell_index.cells_of(HydroSys::new(0)), 0..1);
    assert_eq!(hydro_cell_index.cells_of(HydroSys::new(1)), 1..3);
    assert_eq!(hydro_cell_index.cells_of(HydroSys::new(2)), 3..4);

    let geometry = StageGeometry {
        turbine: 10..18,
        spillage: 18..24,
        n_blks: 2,
        water_balance: BlockRowFamily::one_per_entity(0..3),
        withdrawal_slack_neg: 24..27,
        withdrawal_slack_pos: 27..30,
        outflow_below_slack: 30..36,
        outflow_above_slack: 36..42,
        turbine_below_slack: 42..50,
        generation_below_slack: 50..58,
        ..test_support::equipment_free_geometry(&[2]).remove(0)
    };
    // theta = 3*(3+0) = 9, equipment starts at 10 (state_layout(3, 0)).
    let mut primal = vec![0.0_f64; 58];
    let turbine_values: [(usize, f64, f64); 4] = [
        (0, 10.0, 11.0),
        (1, 100.0, 101.0),
        (2, 200.0, 201.0),
        (3, 300.0, 301.0),
    ];
    for (c, v0, v1) in turbine_values {
        primal[10 + c * 2] = v0;
        primal[10 + c * 2 + 1] = v1;
    }
    (state, geometry, hydro_cell_index, primal)
}

/// One row per `(hydro, block, cell)`, ordered hydro-major/block-middle/
/// cell-minor. Row index 3 (`hydro_id` 200, block 0, cell 2, bus 37) is the
/// asserted row where hydro-system-position (1), cell index (2), bus id (37)
/// and block index (0) are mutually distinct, so a turbine read indexed by
/// `h` instead of `c` is visible there even though plant 1's OWN system
/// position (1) coincides with its first cell's index (1).
#[test]
fn extract_hydro_bus_generation_emits_one_row_per_cell_per_block() {
    let (state, geometry, hydro_cell_index, primal) = split_middle_plant_fixture();
    let study_dims = test_support::study_dims();
    let ec = zero_energy_conversion(3, 1);
    let dual: Vec<f64> = vec![0.0; geometry_row_capacity(&geometry)];
    let objective_coeffs = vec![0.0; primal.len()];

    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &dual,
            objective: 0.0,
            objective_coeffs: &objective_coeffs,
            row_lower: &[],
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &geometry,
            hydro_cell_index: &hydro_cell_index,
            state: &state,
            entity_counts: &EntityCounts {
                hydro_ids: vec![100, 200, 300],
                hydro_productivities: vec![0.0; 3],
                thermal_ids: vec![],
                line_ids: vec![],
                bus_ids: vec![],
                pumping_station_ids: vec![],
                contract_ids: vec![],
                non_controllable_ids: vec![],
            },
            inflow_m3s_per_hydro: &[0.0, 0.0, 0.0],
            block_hours: &[100.0, 100.0],
            generic_constraint_entries: &[],
            ncs_col_upper: &[],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_slots: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[2.0, 3.0, 5.0],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            cost_scale_factor: 1_000_000.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0; 3],
            stage_index: 0,
            horizon: &HorizonMode::Finite { num_stages: 1 },
            anticipated_windows: &[],
            study_stage_ids: &[],
        },
        0,
    );

    let rows = &result.hydro_bus_generation;
    assert_eq!(rows.len(), 8, "4 cells x 2 blocks");

    let seq: Vec<(i32, Option<u32>, i32, f64)> = rows
        .iter()
        .map(|r| (r.hydro_id, r.block_id, r.bus_id, r.turbined_m3s))
        .collect();
    assert_eq!(
        seq,
        vec![
            (100, Some(0), 41, 10.0),
            (100, Some(1), 41, 11.0),
            (200, Some(0), 31, 100.0),
            (200, Some(0), 37, 200.0),
            (200, Some(1), 31, 101.0),
            (200, Some(1), 37, 201.0),
            (300, Some(0), 53, 300.0),
            (300, Some(1), 53, 301.0),
        ],
        "hydro-major, block-middle, cell-minor order"
    );

    // Plant 1 (hydro_id 200) at block 1: its two cells' bus ids in ascending order.
    assert_eq!(rows[4].bus_id, 31);
    assert_eq!(rows[5].bus_id, 37);

    // hydro system position 1, cell index 2, bus 37, block 0 — mutually distinct.
    assert_eq!(rows[3].hydro_id, 200);
    assert_eq!(rows[3].bus_id, 37);
    assert_eq!(rows[3].block_id, Some(0));
    assert_eq!(rows[3].turbined_m3s, 200.0);
}

/// Every plant in [`split_middle_plant_fixture`] is `ConstantProductivity`
/// (`geometry.fpha_hydro_indices` is empty, so `lookup.fpha[h]` is `None`
/// everywhere): each row's `generation_mw` must be its OWN cell's
/// `turbined_m3s * hydro_productivities[h]`, never a different hydro's
/// productivity. `[2.0, 3.0, 5.0]` are mutually distinct so a read from the
/// wrong hydro index is visible, and no productivity is `999.0` so a wrong
/// multiplier is visible on every row.
#[test]
fn extract_hydro_bus_generation_constant_productivity_uses_own_hydro_productivity() {
    let (state, geometry, hydro_cell_index, primal) = split_middle_plant_fixture();
    let study_dims = test_support::study_dims();
    let ec = zero_energy_conversion(3, 1);
    let dual: Vec<f64> = vec![0.0; geometry_row_capacity(&geometry)];
    let objective_coeffs = vec![0.0; primal.len()];

    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &dual,
            objective: 0.0,
            objective_coeffs: &objective_coeffs,
            row_lower: &[],
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &geometry,
            hydro_cell_index: &hydro_cell_index,
            state: &state,
            entity_counts: &EntityCounts {
                hydro_ids: vec![100, 200, 300],
                hydro_productivities: vec![0.0; 3],
                thermal_ids: vec![],
                line_ids: vec![],
                bus_ids: vec![],
                pumping_station_ids: vec![],
                contract_ids: vec![],
                non_controllable_ids: vec![],
            },
            inflow_m3s_per_hydro: &[0.0, 0.0, 0.0],
            block_hours: &[100.0, 100.0],
            generic_constraint_entries: &[],
            ncs_col_upper: &[],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_slots: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[2.0, 3.0, 5.0],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            cost_scale_factor: 1_000_000.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0; 3],
            stage_index: 0,
            horizon: &HorizonMode::Finite { num_stages: 1 },
            anticipated_windows: &[],
            study_stage_ids: &[],
        },
        0,
    );

    let productivity_of = |hydro_id: i32| match hydro_id {
        100 => 2.0,
        200 => 3.0,
        300 => 5.0,
        other => panic!("unexpected hydro_id {other} in split_middle_plant_fixture"),
    };

    let rows = &result.hydro_bus_generation;
    assert_eq!(rows.len(), 8, "4 cells x 2 blocks");
    for row in rows {
        let expected = row.turbined_m3s * productivity_of(row.hydro_id);
        assert_eq!(
            row.generation_mw.to_bits(),
            expected.to_bits(),
            "hydro {} bus {} block {:?}: generation_mw {} must equal turbined_m3s \
             ({}) * hydro_productivities[h] ({})",
            row.hydro_id,
            row.bus_id,
            row.block_id,
            row.generation_mw,
            row.turbined_m3s,
            productivity_of(row.hydro_id),
        );
    }
}

/// Plant 0 (bus 9, one cell) precedes the 3-cell split plant 1 (buses 5/6/7)
/// — different cell counts, as the fixture-discipline rule requires. Turbine
/// values `[1e16, 1.0, 1.0]` in ascending-cell order are order-sensitive at
/// the bit level: summing ascending (`(0.0 + 1e16) + 1.0) + 1.0`) ties to
/// even and rounds each `+1.0` away, while summing descending
/// (`(0.0 + 1.0) + 1.0) + 1e16`) computes the exact `2.0 + 1e16` (`2.0` is a
/// multiple of the ULP at that magnitude) — the two differ, which is exactly
/// what pins the row-emission order.
fn order_sensitive_split_plant_fixture() -> (StateSpace, StageGeometry, HydroCellIndex, Vec<f64>) {
    let state = test_support::state_layout(2, 0);
    let hydro_cell_index = HydroCellIndex::build(&[
        test_support::geometry_hydro_with_groups(
            0,
            vec![test_support::make_unit_group(
                cobre_core::EntityId(1),
                cobre_core::EntityId(9),
                0.0,
                10.0,
                0.0,
                10.0,
            )],
            cobre_core::HydroGenerationModel::ConstantProductivity,
        ),
        test_support::geometry_hydro_with_groups(
            1,
            vec![
                test_support::make_unit_group(
                    cobre_core::EntityId(2),
                    cobre_core::EntityId(5),
                    0.0,
                    10.0,
                    0.0,
                    10.0,
                ),
                test_support::make_unit_group(
                    cobre_core::EntityId(3),
                    cobre_core::EntityId(6),
                    0.0,
                    10.0,
                    0.0,
                    10.0,
                ),
                test_support::make_unit_group(
                    cobre_core::EntityId(4),
                    cobre_core::EntityId(7),
                    0.0,
                    10.0,
                    0.0,
                    10.0,
                ),
            ],
            cobre_core::HydroGenerationModel::ConstantProductivity,
        ),
    ]);
    assert_eq!(hydro_cell_index.n_cells(), 4, "1 + 3 cells");

    let geometry = StageGeometry {
        turbine: 7..11,
        spillage: 11..13,
        n_blks: 1,
        water_balance: BlockRowFamily::one_per_entity(0..2),
        withdrawal_slack_neg: 13..15,
        withdrawal_slack_pos: 15..17,
        outflow_below_slack: 17..19,
        outflow_above_slack: 19..21,
        turbine_below_slack: 21..25,
        generation_below_slack: 25..29,
        ..test_support::equipment_free_geometry(&[1]).remove(0)
    };
    // theta = 2*(3+0) = 6, equipment starts at 7 (state_layout(2, 0)).
    let mut primal = vec![0.0_f64; 29];
    primal[7] = 42.0; // cell 0 (plant 0)
    primal[8] = 1.0e16; // cell 1 (plant 1, bus 5)
    primal[9] = 1.0; // cell 2 (plant 1, bus 6)
    primal[10] = 1.0; // cell 3 (plant 1, bus 7)
    (state, geometry, hydro_cell_index, primal)
}

/// Summing the split plant's `hydro_bus_generation` rows in ROW order equals
/// its own `hydros` row's `turbined_m3s` bit-for-bit — the same ascending-cell
/// fold `extract_hydro_per_block`'s `.sum()` already performs.
#[test]
fn hydro_bus_rows_sum_bit_exactly_to_the_plant_turbined_row() {
    let (state, geometry, hydro_cell_index, primal) = order_sensitive_split_plant_fixture();
    let study_dims = test_support::study_dims();
    let ec = zero_energy_conversion(2, 1);
    let dual: Vec<f64> = vec![0.0; geometry_row_capacity(&geometry)];
    let objective_coeffs = vec![0.0; primal.len()];

    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &dual,
            objective: 0.0,
            objective_coeffs: &objective_coeffs,
            row_lower: &[],
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &geometry,
            hydro_cell_index: &hydro_cell_index,
            state: &state,
            entity_counts: &EntityCounts {
                hydro_ids: vec![100, 200],
                hydro_productivities: vec![0.0; 2],
                thermal_ids: vec![],
                line_ids: vec![],
                bus_ids: vec![],
                pumping_station_ids: vec![],
                contract_ids: vec![],
                non_controllable_ids: vec![],
            },
            inflow_m3s_per_hydro: &[0.0, 0.0],
            block_hours: &[100.0],
            generic_constraint_entries: &[],
            ncs_col_upper: &[],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_slots: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[1.0, 1.0],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            cost_scale_factor: 1_000_000.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0; 2],
            stage_index: 0,
            horizon: &HorizonMode::Finite { num_stages: 1 },
            anticipated_windows: &[],
            study_stage_ids: &[],
        },
        0,
    );

    let plant1_bus_sum: f64 = result
        .hydro_bus_generation
        .iter()
        .filter(|r| r.hydro_id == 200 && r.block_id == Some(0))
        .map(|r| r.turbined_m3s)
        .sum();
    let plant1_row = result
        .hydros
        .iter()
        .find(|h| h.hydro_id == 200 && h.block_id == Some(0))
        .expect("plant 1 must have a block-0 hydros row");

    assert_eq!(
        plant1_bus_sum.to_bits(),
        plant1_row.turbined_m3s.to_bits(),
        "bus-row sum ({plant1_bus_sum}) must equal the plant's own turbined_m3s \
         ({}) bit-for-bit",
        plant1_row.turbined_m3s
    );
}

/// Plant 0 (position 0, one cell) is NOT FPHA, so `fpha_cell_local_start` is a
/// prefix sum over plant 1 alone — using the ABSOLUTE cell index instead of the
/// plant-relative offset would alias plant 0's cell (index 0) into plant 1's
/// generation read. Reuses `split_plant_multi_bus_extraction_fixture`, whose
/// plant 1 is FPHA with two cells (bus 10 then bus 11).
#[test]
fn extract_hydro_bus_generation_maps_fpha_cells_by_plant_relative_offset() {
    let (state, geometry, hydro_cell_index) = split_plant_multi_bus_extraction_fixture();
    let study_dims = test_support::study_dims();
    let ec = zero_energy_conversion(2, 1);

    let mut primal = vec![0.0_f64; 7]; // state region [0, 7), unused by this test
    primal.extend_from_slice(&[
        10.0, 11.0, // turbine[cell 0] (plant 0, blk 0/1)
        100.0, 101.0, // turbine[cell 1] (plant 1's bus-10 group, blk 0/1)
        5.0, 7.0, // turbine[cell 2] (plant 1's bus-11 group, blk 0/1)
        0.0, 0.0, // spillage[hydro 0]
        0.0, 0.0, // spillage[hydro 1]
        50.0, 51.0, // generation[fpha-cell-local 0 = cell 1], blk 0/1
        3.0, 4.0, // generation[fpha-cell-local 1 = cell 2], blk 0/1
    ]);
    // Pads withdrawal-slack and the four operational-violation slack families
    // (this test asserts none of them) with always-zero columns.
    primal.resize(geometry.generation_below_slack.end, 0.0);
    let dual = vec![0.0; 2];
    let objective_coeffs = vec![0.0; primal.len()];

    let result = extract_stage_result(
        &SolutionView {
            primal: &primal,
            dual: &dual,
            objective: 0.0,
            objective_coeffs: &objective_coeffs,
            row_lower: &[],
        },
        &StageExtractionSpec {
            study_dims: &study_dims,
            geometry: &geometry,
            hydro_cell_index: &hydro_cell_index,
            state: &state,
            entity_counts: &EntityCounts {
                hydro_ids: vec![100, 200],
                hydro_productivities: vec![0.0; 2],
                thermal_ids: vec![],
                line_ids: vec![],
                bus_ids: vec![],
                pumping_station_ids: vec![],
                contract_ids: vec![],
                non_controllable_ids: vec![],
            },
            inflow_m3s_per_hydro: &[0.0, 0.0],
            block_hours: &[100.0, 100.0],
            generic_constraint_entries: &[],
            ncs_col_upper: &[],
            pumping_consumption_mw_per_m3s: &[],
            contract_prices: &[],
            contract_slots: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities: &[2.0, 0.0],
            col_scale: &[],
            row_scale: &[],
            cumulative_discount_factor: 1.0,
            cost_scale_factor: 1_000_000.0,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0; 2],
            stage_index: 0,
            horizon: &HorizonMode::Finite { num_stages: 1 },
            anticipated_windows: &[],
            study_stage_ids: &[],
        },
        0,
    );

    let plant1_rows: Vec<_> = result
        .hydro_bus_generation
        .iter()
        .filter(|r| r.hydro_id == 200)
        .collect();
    assert_eq!(plant1_rows.len(), 4, "2 cells x 2 blocks");

    // Block 0: bus 10 (cell 1, fpha-local 0) then bus 11 (cell 2, fpha-local 1).
    assert_eq!(plant1_rows[0].bus_id, 10);
    assert_eq!(plant1_rows[0].generation_mw, 50.0);
    assert_eq!(plant1_rows[1].bus_id, 11);
    assert_eq!(plant1_rows[1].generation_mw, 3.0);
    // Block 1.
    assert_eq!(plant1_rows[2].generation_mw, 51.0);
    assert_eq!(plant1_rows[3].generation_mw, 4.0);

    let sum_blk0 = plant1_rows[0].generation_mw + plant1_rows[1].generation_mw;
    let sum_blk1 = plant1_rows[2].generation_mw + plant1_rows[3].generation_mw;
    let plant1_hydros: Vec<_> = result.hydros.iter().filter(|h| h.hydro_id == 200).collect();
    assert_eq!(sum_blk0.to_bits(), plant1_hydros[0].generation_mw.to_bits());
    assert_eq!(sum_blk1.to_bits(), plant1_hydros[1].generation_mw.to_bits());
}

// -------------------------------------------------------------------------
// extract_generic_violations
// -------------------------------------------------------------------------

/// Build a [`GenericConstraintRowEntry`] for the two-sided (both `bound_lower`
/// and `bound_upper` present) net-report arm, plus its matching
/// `[s_plus, s_minus]` primal pair at columns `[0, 1]`.
fn two_slack_row_entry(
    is_stage_level: bool,
    block_idx: usize,
    penalty: f64,
    s_plus: f64,
    s_minus: f64,
) -> (GenericConstraintRowEntry, Vec<f64>) {
    let entry = GenericConstraintRowEntry {
        constraint_idx: 0,
        entity_id: 42,
        block_idx,
        is_stage_level,
        bound_lower: Some(0.0),
        bound_upper: Some(0.0),
        slack_enabled: true,
        slack_penalty: penalty,
        slack_plus_col: Some(0),
        slack_minus_col: Some(1),
    };
    (entry, vec![s_plus, s_minus])
}

/// Build a `StageExtractionSpec` whose only meaningful fields are
/// `generic_constraint_entries` and `block_hours` — `extract_generic_violations`
/// reads nothing else from the spec, mirroring `contract_only_spec`'s /
/// `pumping_only_spec`'s inert-default pattern.
fn generic_only_spec<'a>(
    entries: &'a [GenericConstraintRowEntry],
    block_hours: &'a [f64],
) -> StageExtractionSpec<'a> {
    // Leaked so the borrow outlives this function call, matching the caller's own
    // `'a` — none of these fields are read by `extract_generic_violations`, only
    // constructed to satisfy the spec's shape.
    let study_dims: &'a StudyDimensions = Box::leak(Box::new(test_support::study_dims()));
    let state: &'a StateSpace = Box::leak(Box::new(test_support::state_layout(0, 0)));
    let geometry: &'a StageGeometry = Box::leak(Box::new(
        test_support::equipment_free_geometry(&[0]).remove(0),
    ));
    let entity_counts: &'a EntityCounts = Box::leak(Box::new(EntityCounts {
        hydro_ids: vec![],
        hydro_productivities: vec![],
        thermal_ids: vec![],
        line_ids: vec![],
        bus_ids: vec![],
        pumping_station_ids: vec![],
        contract_ids: vec![],
        non_controllable_ids: vec![],
    }));
    let ec: &'a EnergyConversionSet = Box::leak(Box::new(zero_energy_conversion(0, 1)));
    let diversion: &'a HashMap<cobre_core::EntityId, Vec<usize>> =
        Box::leak(Box::new(HashMap::new()));
    let hydro_cell_index: &'a HydroCellIndex =
        Box::leak(Box::new(test_support::identity_hydro_cell_index(0)));

    StageExtractionSpec {
        study_dims,
        geometry,
        hydro_cell_index,
        state,
        entity_counts,
        inflow_m3s_per_hydro: &[],
        block_hours,
        generic_constraint_entries: entries,
        ncs_col_upper: &[],
        pumping_consumption_mw_per_m3s: &[],
        contract_prices: &[],
        contract_slots: &[],
        diversion_upstream: diversion,
        hydro_productivities: &[],
        col_scale: &[],
        row_scale: &[],
        cumulative_discount_factor: 1.0,
        cost_scale_factor: 1_000_000.0,
        energy_conversion: ec,
        hydro_min_storage_hm3: &[],
        stage_index: 0,
        horizon: &HorizonMode::Finite { num_stages: 1 },
        anticipated_windows: &[],
        study_stage_ids: &[],
    }
}

/// A two-sided entry with `slack_enabled = true`, `slack_penalty = 100.0`,
/// `block_hours = 730.0`, `s_plus = 3.0`, `s_minus = 1.0` reports the signed net
/// (`2.0`) as `slack_value` while charging BOTH slacks
/// (`4.0 * 100.0 * 730.0`) as `slack_cost`.
#[test]
fn two_sided_net_report_reads_signed_net_and_charges_both_slacks() {
    let (entry, primal) = two_slack_row_entry(false, 0, 100.0, 3.0, 1.0);
    let block_hours = [730.0_f64];
    let dual: Vec<f64> = Vec::new();
    let view = SolutionView {
        primal: &primal,
        dual: &dual,
        objective: 0.0,
        objective_coeffs: &[],
        row_lower: &[],
    };
    let entries = [entry];
    let spec = generic_only_spec(&entries, &block_hours);

    let (results, total_cost) = extract_generic_violations(&view, &spec, 3);

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].stage_id, 3);
    assert_eq!(results[0].block_id, Some(0));
    assert_eq!(results[0].constraint_id, 42);
    assert_eq!(results[0].slack_value, 2.0);
    assert_eq!(results[0].slack_cost, 4.0 * 100.0 * 730.0);
    assert_eq!(total_cost, 4.0 * 100.0 * 730.0);
}

/// A stage-level two-sided entry (`is_stage_level = true`) over a 2-block
/// stage with block hours `[400.0, 330.0]` reports `block_id = None` and prices
/// its slack cost on the SUMMED stage hours (`730.0`), not either individual
/// block's.
#[test]
fn stage_level_two_sided_prices_summed_stage_hours() {
    let (entry, primal) = two_slack_row_entry(true, 0, 1.0, 1.0, 0.0);
    let block_hours = [400.0_f64, 330.0_f64];
    let dual: Vec<f64> = Vec::new();
    let view = SolutionView {
        primal: &primal,
        dual: &dual,
        objective: 0.0,
        objective_coeffs: &[],
        row_lower: &[],
    };
    let entries = [entry];
    let spec = generic_only_spec(&entries, &block_hours);

    let (results, _) = extract_generic_violations(&view, &spec, 0);

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].block_id, None);
    assert_eq!(results[0].slack_cost, 1.0 * 1.0 * 730.0);
}

/// A per-block two-sided entry (`is_stage_level = false`, `block_idx = 1`) over a
/// 2-block stage reports `block_id = Some(1)` and prices its slack cost on THAT
/// block's own duration (`330.0`) — not block 0's `400.0` nor the summed `730.0`.
#[test]
fn per_block_two_sided_prices_its_own_block_duration() {
    let (entry, primal) = two_slack_row_entry(false, 1, 10.0, 2.0, 0.5);
    let block_hours = [400.0_f64, 330.0_f64];
    let dual: Vec<f64> = Vec::new();
    let view = SolutionView {
        primal: &primal,
        dual: &dual,
        objective: 0.0,
        objective_coeffs: &[],
        row_lower: &[],
    };
    let entries = [entry];
    let spec = generic_only_spec(&entries, &block_hours);

    let (results, _) = extract_generic_violations(&view, &spec, 0);

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].block_id, Some(1));
    assert_eq!(results[0].slack_value, 1.5);
    assert_eq!(results[0].slack_cost, 2.5 * 10.0 * 330.0);
}

/// A two-sided entry with `slack_enabled = false` reports `slack_value ==
/// 0.0` and `slack_cost == 0.0` and contributes nothing to the returned total
/// cost — the `map_or(0.0, ...)` fallback path, never reading the (absent)
/// slack columns.
#[test]
fn two_sided_slack_disabled_reports_zero_and_contributes_nothing() {
    let entry = GenericConstraintRowEntry {
        constraint_idx: 0,
        entity_id: 42,
        block_idx: 0,
        is_stage_level: false,
        bound_lower: Some(0.0),
        bound_upper: Some(0.0),
        slack_enabled: false,
        slack_penalty: 0.0,
        slack_plus_col: None,
        slack_minus_col: None,
    };
    let primal: Vec<f64> = Vec::new();
    let dual: Vec<f64> = Vec::new();
    let view = SolutionView {
        primal: &primal,
        dual: &dual,
        objective: 0.0,
        objective_coeffs: &[],
        row_lower: &[],
    };
    let block_hours = [500.0_f64];
    let entries = [entry];
    let spec = generic_only_spec(&entries, &block_hours);

    let (results, total_cost) = extract_generic_violations(&view, &spec, 0);

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].slack_value, 0.0);
    assert_eq!(results[0].slack_cost, 0.0);
    assert_eq!(total_cost, 0.0);
}
