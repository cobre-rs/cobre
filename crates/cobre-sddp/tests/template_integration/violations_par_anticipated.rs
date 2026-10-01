//! `violations_par_anticipated` section tests.

use super::*;

use super::common::in_code_studies::{discounted_anticipated_study, mixed_lead_anticipated_study};
use super::common::{build_setup_in_code, run_simulation};
use cobre_io::Config;
use cobre_io::config::{SimulationConfig as IoSimulationConfig, SimulationSelection};
use cobre_sddp::test_support::template_structure::{
    RingLaneKind, RowOwner, UnscaledMatrix, geometry_row_families, ring_lanes, row_owners,
};
use cobre_sddp::test_support::{constant_lead_resolution, equipment_free_geometry};

#[test]
fn min_outflow_active_col_bounds() {
    let result = build_active_violations_template();
    let t = &result.templates[0];
    let indexer = &result.geometry_per_stage[0];

    let col = indexer.outflow_below_slack.start;
    assert_eq!(t.col_lower[col], 0.0, "outflow_below lower must be 0");
    assert_eq!(
        t.col_upper[col],
        f64::INFINITY,
        "outflow_below upper must be +inf when active"
    );
}

#[test]
fn max_outflow_active_col_bounds() {
    let result = build_active_violations_template();
    let t = &result.templates[0];
    let indexer = &result.geometry_per_stage[0];

    let col = indexer.outflow_above_slack.start;
    assert_eq!(t.col_lower[col], 0.0, "outflow_above lower must be 0");
    assert_eq!(
        t.col_upper[col],
        f64::INFINITY,
        "outflow_above upper must be +inf when max_outflow is Some"
    );
}

#[test]
fn operational_violation_inactive_pinned() {
    let system = one_hydro_system(1, 0); // default: all violation bounds = 0
    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("base ok");
    let t = &result.templates[0];
    let indexer = &result.geometry_per_stage[0];

    for &col in &[
        indexer.outflow_below_slack.start,
        indexer.outflow_above_slack.start,
        indexer.turbine_below_slack.start,
        indexer.generation_below_slack.start,
    ] {
        assert_eq!(t.col_lower[col], 0.0, "inactive col {col} lower != 0");
        assert_eq!(
            t.col_upper[col], 0.0,
            "inactive col {col} upper != 0 (should be pinned)"
        );
    }
}

#[test]
fn operational_violation_objective_costs() {
    // Per-block: penalty * block_hours / COST_SCALE_FACTOR.
    let result = build_active_violations_template();
    let t = &result.templates[0];
    let indexer = &result.geometry_per_stage[0];

    let block_hours = [720.0, 48.0];
    for (blk, &hours) in block_hours.iter().enumerate() {
        let expected = 1000.0 * hours / COST_SCALE_FACTOR;
        for &start in &[
            indexer.outflow_below_slack.start,
            indexer.outflow_above_slack.start,
            indexer.turbine_below_slack.start,
            indexer.generation_below_slack.start,
        ] {
            // Column for hydro 0, block `blk`: start + 0 * n_blks + blk.
            let col = start + blk;
            assert!(
                (t.objective[col] - expected).abs() < 1e-10,
                "col {col} (block {blk}): objective = {}, expected = {}",
                t.objective[col],
                expected
            );
        }
    }
}

#[test]
fn turbine_column_lower_bound_is_zero() {
    // Turbine column lower bound is 0.0, NOT min_turbined_m3s (the min is enforced
    // by the turbine_below slack, not a hard column bound).
    let result = build_active_violations_template();
    let t = &result.templates[0];
    let indexer = &result.geometry_per_stage[0];

    assert_eq!(
        t.col_lower[indexer.turbine.start], 0.0,
        "turbine blk0 lower bound must be 0.0"
    );
    assert_eq!(
        t.col_lower[indexer.turbine.start + 1],
        0.0,
        "turbine blk1 lower bound must be 0.0"
    );
}

/// When an annual component is present, `max_par_order` is widened to 12
/// regardless of the classical AR order.
#[test]
fn max_par_order_uses_par_lp_when_annual_present() {
    use cobre_core::scenario::{AnnualComponent, InflowModel};

    let ar_coeffs: Vec<f64> = vec![0.3, 0.2];
    let ann = AnnualComponent {
        coefficient: 0.5,
        mean_m3s: 80.0,
        std_m3s: 20.0,
    };
    let inflow_models = vec![
        InflowModel {
            hydro_id: EntityId(2),
            stage_id: 0,
            mean_m3s: 80.0,
            std_m3s: 20.0,
            ar_coefficients: ar_coeffs.clone(),
            residual_std_ratio: 1.0,
            annual: Some(ann),
        },
        InflowModel {
            hydro_id: EntityId(3),
            stage_id: 0,
            mean_m3s: 60.0,
            std_m3s: 15.0,
            ar_coefficients: ar_coeffs.clone(),
            residual_std_ratio: 1.0,
            annual: None,
        },
    ];

    let system = two_hydro_par_system(2, inflow_models.clone());
    let stages = system.stages().to_vec();
    let hydro_ids: Vec<EntityId> = system.hydros().iter().map(|h| h.id).collect();
    let par_lp =
        PrecomputedPar::build(&inflow_models, &stages, &hydro_ids, None).expect("par build ok");

    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &par_lp,
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("build_stage_templates_resolving_layout ok");

    assert_eq!(
        result.templates[0].n_state,
        2 * (1 + 12),
        "annual component must widen the PAR order to 12, so n_state == N*(1+12); got {}",
        result.templates[0].n_state
    );
}

/// Classical PAR systems are unaffected: `max_par_order` equals the AR order.
#[test]
fn max_par_order_classical_unchanged() {
    use cobre_core::scenario::InflowModel;

    let ar_coeffs: Vec<f64> = vec![0.3, 0.2, 0.1];
    let inflow_models = vec![
        InflowModel {
            hydro_id: EntityId(2),
            stage_id: 0,
            mean_m3s: 80.0,
            std_m3s: 20.0,
            ar_coefficients: ar_coeffs.clone(),
            residual_std_ratio: 1.0,
            annual: None,
        },
        InflowModel {
            hydro_id: EntityId(3),
            stage_id: 0,
            mean_m3s: 60.0,
            std_m3s: 15.0,
            ar_coefficients: ar_coeffs.clone(),
            residual_std_ratio: 1.0,
            annual: None,
        },
    ];

    let system = two_hydro_par_system(3, inflow_models.clone());
    let stages = system.stages().to_vec();
    let hydro_ids: Vec<EntityId> = system.hydros().iter().map(|h| h.id).collect();
    let par_lp =
        PrecomputedPar::build(&inflow_models, &stages, &hydro_ids, None).expect("par build ok");

    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &par_lp,
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("build_stage_templates_resolving_layout ok");

    assert_eq!(
        result.templates[0].n_state,
        2 * (1 + 3),
        "classical PAR order must remain 3, so n_state == N*(1+3); got {}",
        result.templates[0].n_state
    );
}

/// When `max_par_order == 12`, the z-inflow definition row for hydro 0
/// has exactly 12 nonzero lag-column entries (one per lag in 0..12).
///
/// The `+1.0` entry on `col_z_inflow_start + 0` is excluded from the count.
#[allow(clippy::cast_sign_loss)]
#[test]
fn max_par_order_z_inflow_row_has_twelve_lag_entries() {
    use cobre_core::scenario::{AnnualComponent, InflowModel};

    let ar_coeffs: Vec<f64> = vec![0.3, 0.2];
    let ann = AnnualComponent {
        coefficient: 0.5,
        mean_m3s: 80.0,
        std_m3s: 20.0,
    };
    let inflow_models = vec![
        InflowModel {
            hydro_id: EntityId(2),
            stage_id: 0,
            mean_m3s: 80.0,
            std_m3s: 20.0,
            ar_coefficients: ar_coeffs.clone(),
            residual_std_ratio: 1.0,
            annual: Some(ann),
        },
        InflowModel {
            hydro_id: EntityId(3),
            stage_id: 0,
            mean_m3s: 60.0,
            std_m3s: 15.0,
            ar_coefficients: ar_coeffs.clone(),
            residual_std_ratio: 1.0,
            annual: None,
        },
    ];

    let system = two_hydro_par_system(2, inflow_models.clone());
    let stages = system.stages().to_vec();
    let hydro_ids: Vec<EntityId> = system.hydros().iter().map(|h| h.id).collect();
    let par_lp =
        PrecomputedPar::build(&inflow_models, &stages, &hydro_ids, None).expect("par build ok");

    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &par_lp,
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("build_stage_templates_resolving_layout ok");

    let t = &result.templates[0];
    let n_h = 2_usize;
    let l = 12_usize;
    assert_eq!(
        t.n_state,
        n_h * (1 + l),
        "precondition: the PAR order must be 12, so n_state == N*(1+12)"
    );

    let row_z_inflow_h0 = 0_usize; // z_inflow rows start at 0

    // z_inflow column for hydro 0: col_z_inflow_start = N*(1+L) = 2*13 = 26.
    let col_z_inflow_h0 = n_h * (1 + l); // = 26
    let mut lag_entry_count = 0usize;
    for col in 0..t.num_cols {
        let start = t.col_starts[col] as usize;
        let end = t.col_starts[col + 1] as usize;
        for pos in start..end {
            if t.row_indices[pos] as usize == row_z_inflow_h0 && col != col_z_inflow_h0 {
                lag_entry_count += 1;
            }
        }
    }

    assert_eq!(
        lag_entry_count, 12,
        "z-inflow definition row for hydro 0 must have exactly 12 lag-column entries \
         when max_par_order == 12, got {lag_entry_count}"
    );
}

/// Regression guard: [`PatchBuffer`] must never grow to include generic-constraint rows.
///
/// The only row categories `PatchBuffer` mutates at solve time are AR dynamics /
/// noise, load-balance, and z-inflow definition; incoming state is pinned via
/// column bounds, not patched rows. Generic-constraint coefficients are immutable
/// after construction.
#[test]
#[allow(clippy::cast_precision_loss)] // fixture values are small integers; no precision is lost
fn parameter_coefficient_persists_across_stage_template_uses() {
    // Realistic-scale system: N=3, L=2, M=2, B_max=3.
    // Row capacity = M*B_max + N = 2*3 + 3 = 9.
    let n: usize = 3;
    let l: usize = 2;
    let m: usize = 2;
    let b_max: usize = 3;

    // `load_bus_indices`/`geometry_per_stage` are owners `PatchBuffer::new` reads
    // only through `.len()`/`StageGeometry::max_blocks` (i.e. only `n_blks`);
    // their content need not correspond to a real M=2-load-bus, N=3-hydro
    // system — an equipment-free, `b_max`-block geometry supplies the block
    // count, and an arbitrary `m`-long index slice supplies M.
    let load_bus_indices: Vec<usize> = (0..m).collect();
    let geometry_per_stage = equipment_free_geometry(&[b_max]);
    let state = StateSpace::new(
        n,
        l,
        Vec::new(),
        vec![],
        AnticipatedResolution::default(),
        &vec![l; n],
    );

    let capacity_formula = m * b_max + n;
    let mut buf = PatchBuffer::new(&state, &load_bus_indices, &geometry_per_stage);

    assert_eq!(
        buf.indices.len(),
        capacity_formula,
        "PatchBuffer capacity must equal M*B_max + N; \
         formula change indicates new patch categories were added"
    );

    // Load — 2 load buses, 2 active blocks (< max 3). The per-stage grid
    // carries `b_active`, NOT `b_max`, so the load-balance row stride matches the
    // per-stage LP (a global grid striding by `b_max` would address the wrong row).
    let b_active: usize = 2;
    let load_rhs: Vec<f64> = (0..m * b_active).map(|i| 100.0 + i as f64).collect();
    let bus_positions: Vec<usize> = (0..m).collect();
    let load_start: usize = 200; // arbitrary LP row offset
    let load_rows = BlockRowFamily::per_block(load_start..load_start + m * b_active);
    buf.fill_load_patches(
        load_rows,
        BlockGrid::new(b_active, 1),
        &load_rhs,
        &bus_positions,
        &[],
    );

    let z_inflow_rhs: Vec<f64> = (0..n).map(|h| 80.0 + h as f64).collect();
    buf.fill_z_inflow_patches(&state, &z_inflow_rhs, &[]);

    // The count uses b_active, not B_max: any generic-constraint patching would push
    // it past the M*B_max + N capacity into an out-of-bounds write.
    let expected_count = m * b_active + n;
    assert_eq!(
        buf.forward_patch_count(),
        expected_count,
        "forward_patch_count must equal M*b_active + N; \
         any generic-constraint patching would alter this count"
    );

    assert!(
        buf.forward_patch_count() < buf.indices.len(),
        "forward_patch_count {} must be < capacity {} when b_active < b_max",
        buf.forward_patch_count(),
        buf.indices.len(),
    );

    // Two builds of the same system must yield bit-identical CSC arrays —
    // the matrix is not mutated by the solver loop (determinism).
    let system = one_hydro_system(2, l);
    let result_a = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("build ok");
    let result_b = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("build ok");

    for (s, (ta, tb)) in result_a
        .templates
        .iter()
        .zip(&result_b.templates)
        .enumerate()
    {
        assert_eq!(
            ta.values, tb.values,
            "stage {s}: CSC values differ between two builds of the same system; \
             stage-template matrix must be deterministic and immutable"
        );
        assert_eq!(
            ta.row_indices, tb.row_indices,
            "stage {s}: CSC row_indices differ between two builds"
        );
    }
}

/// When `t + K_i < n_stages`, the anticipated-decision column takes bounds from
/// `thermal_block_base(thermal_idx, t + K_i)`. With `n_stages = 4`, `K_i = 2`,
/// min/max = 10.0/100.0, stage `t = 0` (delivery 2, active) → col bounds [10, 100].
#[test]
fn test_anticipated_decision_bounds_at_active_stage() {
    let system = one_anticipated_thermal_system(4, 2, 10.0, 100.0);
    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("build ok");

    let col = anticipated_decision_col(2);
    let t = &result.templates[0];
    assert_eq!(
        t.col_lower[col], 10.0,
        "stage 0: anticipated-decision col_lower must equal min_generation_mw=10.0 \
         (delivery stage = 0+2=2, active)"
    );
    assert_eq!(
        t.col_upper[col], 100.0,
        "stage 0: anticipated-decision col_upper must equal max_generation_mw=100.0 \
         (delivery stage = 0+2=2, active)"
    );
}

/// When `t + K_i > n_stages` the anticipated-decision column is gated inactive
/// with bounds [0.0, 0.0]. With `n_stages = 4`, `K_i = 2`, stage `t = 3`
/// (delivery 5 > 4) → [0, 0].
#[test]
fn test_anticipated_decision_bounds_inactive_when_beyond_horizon() {
    let system = one_anticipated_thermal_system(4, 2, 10.0, 100.0);
    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("build ok");

    let col = anticipated_decision_col(2);
    let t = &result.templates[3];
    assert_eq!(
        t.col_lower[col], 0.0,
        "stage 3: anticipated-decision col_lower must be 0.0 \
         (delivery stage = 3+2=5 > n_stages=4, inactive)"
    );
    assert_eq!(
        t.col_upper[col], 0.0,
        "stage 3: anticipated-decision col_upper must be 0.0 \
         (delivery stage = 3+2=5 > n_stages=4, inactive)"
    );
}

/// The horizon boundary `t + K_i == n_stages` is REJECTED (inactive)
/// under the strict predicate `t + K_i < n_stages`.
///
/// Setup: `n_stages = 4`, `K_i = 2`, `min_generation_mw = 10.0`,
/// `max_generation_mw = 100.0`.
/// At stage `t = 2`: delivery stage = `2 + 2 = 4 == n_stages` → inactive
/// (the strict predicate excludes equality; no delivery LP exists at
/// `delivery_stage == n_stages` because the per-stage loop iterates
/// `[0, n_stages)`).
/// Expected: `col_lower = 0.0`, `col_upper = 0.0`.
#[test]
fn test_anticipated_decision_inactive_at_horizon_boundary() {
    let system = one_anticipated_thermal_system(4, 2, 10.0, 100.0);
    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("build ok");

    let col = anticipated_decision_col(2);
    let t = &result.templates[2];
    assert_eq!(
        t.col_lower[col], 0.0,
        "stage 2: anticipated-decision col_lower must be 0.0 \
         (delivery stage = 2+2=4 == n_stages=4, strict predicate excludes boundary)"
    );
    assert_eq!(
        t.col_upper[col], 0.0,
        "stage 2: anticipated-decision col_upper must be 0.0 \
         (delivery stage = 2+2=4 == n_stages=4, strict predicate excludes boundary)"
    );
}

/// One-past-boundary `t + K_i == n_stages + 1` is also inactive. With
/// `n_stages = 3`, `K_i = 2`, stage `t = 2` (delivery 4) → [0, 0].
#[test]
fn test_anticipated_decision_inactive_one_past_horizon_boundary() {
    let system = one_anticipated_thermal_system(3, 2, 10.0, 100.0);
    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("build ok");

    let col = anticipated_decision_col(2);
    let t = &result.templates[2];
    assert_eq!(
        t.col_lower[col], 0.0,
        "stage 2: anticipated-decision col_lower must be 0.0 \
         (delivery stage = 2+2=4 = n_stages+1=4, one-past-boundary inactive)"
    );
    assert_eq!(
        t.col_upper[col], 0.0,
        "stage 2: anticipated-decision col_upper must be 0.0 \
         (delivery stage = 2+2=4 = n_stages+1=4, one-past-boundary inactive)"
    );
}

/// The decision-stage objective uses the DELIVERY stage's cost, hours, and
/// discount factor. With K_i=2, cost=50.0, no discount, 744h blocks, stage t=0
/// (delivery 2) → 50.0 * 744.0 * 1.0 / COST_SCALE_FACTOR = 37.2.
#[test]
fn test_anticipated_decision_objective_uses_delivery_stage_factors() {
    // System: n_stages=4, K_i=2, cost_per_mwh=50.0, 744h blocks, no discounting.
    // At stage t=0: delivery=2, d_factor=1.0, delivery_hours=744.0.
    // objective (pre-scale) = 50.0 * 744.0 * 1.0 = 37200.0.
    // After /COST_SCALE_FACTOR: 37200.0 / 1000.0 = 37.2.
    let system = one_anticipated_thermal_system(4, 2, 0.0, 100.0);
    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("build ok");

    let col = anticipated_decision_col(2);
    let tmpl = &result.templates[0];

    let cost_per_mwh = 50.0_f64;
    let delivery_hours = 744.0_f64; // all stages have 744h single block
    let d_factor = 1.0_f64; // no discount
    let expected = cost_per_mwh * delivery_hours * d_factor / COST_SCALE_FACTOR;
    assert_eq!(
        tmpl.objective[col], expected,
        "stage 0: anticipated-decision objective must equal 50*744*1/1000 = {expected}"
    );
}

/// Objective at boundary stage t + K_i == n_stages is REJECTED (zero)
/// under the strict predicate `t + K_i < n_stages`.
///
/// System: n_stages=4, K_i=2. At stage t=2: delivery_stage=4==n_stages → inactive
/// (the strict predicate excludes equality; no delivery LP exists at
/// `delivery_stage == n_stages`).
/// Expected: `objective[col] == 0.0`.
#[test]
fn test_anticipated_decision_objective_zero_at_horizon_boundary() {
    let system = one_anticipated_thermal_system(4, 2, 0.0, 100.0);
    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("build ok");

    let col = anticipated_decision_col(2);
    let tmpl = &result.templates[2]; // t=2, delivery=4==n_stages, strict-predicate-inactive

    assert_eq!(
        tmpl.objective[col], 0.0,
        "stage 2 (t+K==n_stages): anticipated-decision objective must be 0.0 \
         (strict predicate excludes boundary; no delivery LP exists at n_stages)"
    );
}

/// The objective is zero when the plant is inactive (delivery_stage > n_stages):
/// `n_stages=4`, `K_i=2`, stage t=3 (delivery 5) → objective 0.0.
#[test]
fn test_anticipated_decision_objective_zero_when_inactive() {
    let system = one_anticipated_thermal_system(4, 2, 0.0, 100.0);
    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("build ok");

    let col = anticipated_decision_col(2);
    let tmpl = &result.templates[3]; // t=3, delivery=5>n_stages=4

    assert_eq!(
        tmpl.objective[col], 0.0,
        "stage 3: anticipated-decision objective must be 0.0 (inactive beyond horizon)"
    );
}

/// The objective is zero one past the boundary (`t + K_i == n_stages + 1`):
/// `n_stages=3`, `K_i=2`, stage t=2 (delivery 4) → objective 0.0.
#[test]
fn test_anticipated_decision_objective_zero_one_past_boundary() {
    let system = one_anticipated_thermal_system(3, 2, 0.0, 100.0);
    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("build ok");

    let col = anticipated_decision_col(2);
    let tmpl = &result.templates[2]; // t=2, delivery=4=n_stages+1=4

    assert_eq!(
        tmpl.objective[col], 0.0,
        "stage 2 (t+K=n_stages+1): anticipated-decision objective must be 0.0"
    );
}

/// Regression: at `stage_idx = n_stages - K_i`, no anticipated-decision
/// column is emitted (bounds `[0,0]` and objective `0.0`) for any K and any
/// n_stages such that K < n_stages.
///
/// Under the strict predicate `stage_idx + K_i < n_stages`, the boundary stage
/// `stage_idx = n_stages - K_i` would produce `delivery_stage = n_stages` —
/// outside the study horizon `[0, n_stages)`. The decision column at that
/// stage must be gated out so the LP never pays for an undelivered commitment.
///
/// This is the multi-K sweep guarding the strict-predicate semantics across
/// realistic horizons (n_stages in {3, 4, 5, 6}) and lead times (K in {1, 2, 3}).
#[test]
fn test_anticipated_decision_no_column_at_boundary_stage_strict_predicate() {
    for n_stages in 3_usize..=6 {
        for k in 1_usize..=3 {
            if k >= n_stages {
                // K must be strictly less than n_stages for the boundary stage
                // `n_stages - K` to be a valid non-negative index.
                continue;
            }
            let boundary_stage = n_stages - k;
            #[allow(clippy::cast_possible_truncation)]
            let k_u32 = k as u32;
            let system = one_anticipated_thermal_system(n_stages, k_u32, 10.0, 100.0);
            let result = build_stage_templates_resolving_layout(
                &system,
                no_penalty_config(),
                &PrecomputedPar::default(),
                &PrecomputedNormal::default(),
                &default_production(&system),
                &default_evaporation(&system),
                &ResolvedParameters::default(),
            )
            .expect("build ok");

            let col = anticipated_decision_col(k);
            let tmpl = &result.templates[boundary_stage];

            assert_eq!(
                tmpl.col_lower[col], 0.0,
                "n_stages={n_stages}, K={k}, boundary_stage={boundary_stage}: \
                 col_lower must be 0.0 (delivery_stage = n_stages excluded by strict predicate)"
            );
            assert_eq!(
                tmpl.col_upper[col], 0.0,
                "n_stages={n_stages}, K={k}, boundary_stage={boundary_stage}: \
                 col_upper must be 0.0 (delivery_stage = n_stages excluded by strict predicate)"
            );
            assert_eq!(
                tmpl.objective[col], 0.0,
                "n_stages={n_stages}, K={k}, boundary_stage={boundary_stage}: \
                 objective must be 0.0 (delivery_stage = n_stages excluded by strict predicate)"
            );
        }
    }
}

/// The anticipated thermal's per-block cost is zero at its delivery stages. With
/// `n_stages=4`, `K_i=2`, thermal 0's delivery stages are {2, 3} → objective 0.0.
#[test]
fn test_anticipated_delivery_thermal_cost_is_zero() {
    let lead_stages = 2_usize;
    let system = two_thermal_one_anticipated_system(4, lead_stages as u32);
    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("build ok");

    let col_thermal_0 = two_thermal_col_thermal_start(lead_stages); // thermal 0, blk 0

    // Delivery stages for K_i=2: stage_idx in {2, 3}.
    for stage_idx in [2_usize, 3] {
        let obj = result.templates[stage_idx].objective[col_thermal_0];
        assert_eq!(
            obj, 0.0,
            "stage {stage_idx}: anticipated thermal 0 per-block cost must be 0.0 (delivery stage)"
        );
    }
}

/// The anticipated thermal's per-block cost is 0.0 at EVERY stage. The fishing
/// constraint is always active for an anticipated plant, so `fill_thermal_columns`
/// skips its per-block objective (via `AnticipatedPlants::local_of`) at every
/// stage — including pre-horizon stages before K_i matures — leaving the cost at
/// its 0.0 initialization default.
#[test]
fn test_anticipated_pre_delivery_thermal_cost_unchanged() {
    let lead_stages = 2_usize;
    let system = two_thermal_one_anticipated_system(4, lead_stages as u32);
    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("build ok");

    let col_thermal_0 = two_thermal_col_thermal_start(lead_stages); // thermal 0, blk 0

    // Always-active predicate: cost is zeroed at all stages, including pre-delivery.
    for stage_idx in [0_usize, 1] {
        let obj = result.templates[stage_idx].objective[col_thermal_0];
        assert_eq!(
            obj, 0.0,
            "stage {stage_idx}: anticipated thermal 0 cost must be 0.0 (always-active zeroing)"
        );
    }
}

/// The non-anticipated thermal carries `cost_per_mwh * block_hours / COST_SCALE`
/// at every stage, unaffected by the anticipated thermal's cost zeroing.
#[test]
fn test_non_anticipated_thermal_cost_unchanged_under_anticipated_zero_out() {
    let lead_stages = 2_usize;
    let system = two_thermal_one_anticipated_system(4, lead_stages as u32);
    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("build ok");

    // Thermal 1 is at index 1 in the thermals slice; col = col_thermal_start + 1 * n_blks + 0.
    let col_thermal_1 = two_thermal_col_thermal_start(lead_stages) + 1; // offset 1 for thermal 1, blk 0
    let expected = 50.0 * 744.0 / COST_SCALE_FACTOR;

    for stage_idx in 0..4 {
        let obj = result.templates[stage_idx].objective[col_thermal_1];
        assert_eq!(
            obj, expected,
            "stage {stage_idx}: non-anticipated thermal 1 cost must equal {expected} at every stage"
        );
    }
}

/// The set of stages where the anticipated thermal's per-block cost is zero equals
/// ALL stages: the fishing constraint is always active, so `fill_thermal_columns`
/// always skips its per-block objective (K_i=2, n_stages=4 → zero at {0,1,2,3}).
#[test]
fn test_zero_out_and_fishing_active_predicate_align() {
    let lead_stages = 2_usize;
    let n_stages = 4_usize;
    let system = two_thermal_one_anticipated_system(n_stages, lead_stages as u32);
    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("build ok");

    let col_thermal_0 = two_thermal_col_thermal_start(lead_stages);

    let zeroed_stages: Vec<usize> = (0..n_stages)
        .filter(|&s| result.templates[s].objective[col_thermal_0] == 0.0)
        .collect();
    let all_stages: Vec<usize> = (0..n_stages).collect();

    assert_eq!(
        zeroed_stages, all_stages,
        "stages with zero-out thermal cost must equal all stages under always-active predicate. \
         Got zeroed={zeroed_stages:?}, expected={all_stages:?}"
    );
}

/// With two anticipated thermals (K_0=1, K_1=2) and n_stages=4, fishing rows are
/// always 2 per stage (always-active = `n_anticipated`); the `state_out_def`
/// (deposit/latch) count varies with the strict `stage + K_i < 4` gate, and the
/// interior carry (hold) definition row count varies with how many in-flight,
/// not-yet-due commitments the plants hold. Under the HOLD geometry a commitment
/// is carried in place (same-slot `out - in = 0`) until it matures — the interior
/// row COUNT is a permutation-invariant of the retired shift geometry.
///
/// State_out_def rows per stage:
///   stage 0: K_0=1: 0+1=1 < 4 ✓, K_1=2: 0+2=2 < 4 ✓ → 2 rows
///   stage 1: K_0=1: 1+1=2 < 4 ✓, K_1=2: 1+2=3 < 4 ✓ → 2 rows
///   stage 2: K_0=1: 2+1=3 < 4 ✓, K_1=2: 2+2=4 < 4 ✗ → 1 row
///   stage 3: K_0=1: 3+1=4 < 4 ✗, K_1=2: 3+2=5 < 4 ✗ → 0 rows
///
/// Interior carry rows per stage: K_0=1 has none (its commitment matures the very
/// next stage — always a deposit-or-fish, never carried); K_1=2 carries its single
/// in-flight not-yet-due delivery (`m = stage + 1`) while it lands inside the
/// horizon (`stage + 1 < 4`):
///   stage 0: 1 row; stage 1: 1 row; stage 2: 1 row; stage 3: 0 (m=4 past horizon).
///
/// Combined fishing + state_out_def + interior counts
/// (base = stage 0: 2 fishing + 2 def + 1 interior):
///   stage 1: 2 fishing + 2 state_out_def + 1 interior = base (same as stage 0)
///   stage 2: 2 fishing + 1 state_out_def + 1 interior = base - 1
///   stage 3: 2 fishing + 0 state_out_def + 0 interior = base - 3
#[test]
fn test_anticipated_fishing_rows_count_by_stage() {
    let system = two_anticipated_thermal_system(4);
    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("build ok");

    // base = stage 0: 2 fishing rows (always-active) + 2 state_out_def rows
    // + 1 interior carry row (K_1's single in-flight not-yet-due commitment).
    let base = result.templates[0].num_rows;
    assert_eq!(
        result.templates[1].num_rows, base,
        "stage 1: 2 fishing + 2 state_out_def + 1 interior = base (equal to stage 0)"
    );
    assert_eq!(
        result.templates[2].num_rows,
        base - 1,
        "stage 2: 2 fishing + 1 state_out_def + 1 interior = base - 1 (K_1=2 def inactive: 2+2=4)"
    );
    assert_eq!(
        result.templates[3].num_rows,
        base - 3,
        "stage 3: 2 fishing + 0 state_out_def + 0 interior = base - 3 \
         (both def inactive, K_1's interior slot beyond horizon cap)"
    );
}

/// The fishing constraint is always active for every anticipated plant, so each
/// emits exactly one fishing row at every stage in `[0, n_stages)` — `num_rows` is
/// stage-invariant for this single-anticipated-thermal fixture (K=2, n_stages=4).
#[test]
fn test_anticipated_fishing_same_count_both_stages() {
    let system = one_anticipated_thermal_system(4, 2, 0.0, 100.0);
    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("build ok");

    let rows_stage_0 = result.templates[0].num_rows;
    let rows_stage_1 = result.templates[1].num_rows;
    assert_eq!(
        rows_stage_1, rows_stage_0,
        "stage 1 and stage 0 must have identical row counts (always-active fishing emits one row per anticipated plant at every stage)"
    );
}

/// With two anticipated thermals (K_0=1, K_1=2) sharing one `k_max=2` ring at
/// stage 0 (`n_stages=4`), commit_out columns are unbounded (-INF, +INF) at
/// every slot a plant reaches under the HOLD geometry (interior carry OR active
/// deposit), and frozen `[0, 0]` at a padding slot. Deposit slots are modular:
/// slot `delivery mod k_max` (delivery = 0 + K_i). Slot-major layout:
/// `col = StateSpace::commit_out.start + slot * n_anticipated + plant`
/// (`DeliveryRing::anticipated`'s `out_col(slot, plant)`).
///
/// - col 0 (slot 0, plant 0): PADDING — plant 0 (K=1) delivers at stage 1
///   (slot `1 mod 2 = 1`); its delivery-2 (slot 0) is not yet decided at stage 0.
///   frozen `[0, 0]`.
/// - col 1 (slot 0, plant 1 = K_1=2's deposit slot `2 mod 2 = 0`): free (active: 0+2<4).
/// - col 2 (slot 1, plant 0 = K_0=1's deposit slot `1 mod 2 = 1`): free (active: 0+1<4).
/// - col 3 (slot 1, plant 1's interior carry, the delivery-1 pre-study seed): free.
#[test]
fn test_anticipated_state_columns_unconstrained() {
    let system = two_anticipated_thermal_system(4);
    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("build ok");

    let t = &result.templates[0];

    for &col in &[1_usize, 2, 3] {
        assert!(
            t.col_lower[col].is_infinite() && t.col_lower[col] < 0.0,
            "col {col}: col_lower must be -INF, got {}",
            t.col_lower[col]
        );
        assert!(
            t.col_upper[col].is_infinite() && t.col_upper[col] > 0.0,
            "col {col}: col_upper must be +INF, got {}",
            t.col_upper[col]
        );
    }

    assert_eq!(
        t.col_lower[0], 0.0,
        "col 0 (plant 0's padding slot 0 — its delivery-2 not yet decided at stage 0) must be frozen"
    );
    assert_eq!(
        t.col_upper[0], 0.0,
        "col 0 (plant 0's padding slot 0 — its delivery-2 not yet decided at stage 0) must be frozen"
    );
}

/// One anticipated thermal with K=2, n_stages=4: the incoming state is
/// pinned by column bounds, so no row pins it — the decision column's one
/// structural entry is its own deposit row in the delivery ring, never a
/// row addressing `commit_in`/`storage_in`/`transit_buckets_in`/
/// `inflow_lags` by equality.
#[test]
fn test_anticipated_decision_write_to_state_out_def_row() {
    let system = one_anticipated_thermal_system(4, 2, 0.0, 100.0);
    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("build ok");

    let t = &result.templates[0]; // stage 0: plant active (0+2<4)
    let col_dec = anticipated_decision_col(2);
    let lead_stages = vec![2_usize];
    let resolution = constant_lead_resolution(&lead_stages, 4);
    let state = StateSpace::new(0, 0, Vec::new(), lead_stages, resolution, &[]);
    let geom = &result.geometry_per_stage[0];
    let matrix = UnscaledMatrix::of(t);
    let owners = row_owners(&system, geom, &state);

    for (family, rows) in geometry_row_families(geom, &state) {
        for row in rows {
            let entries = matrix.row(row);
            if entries.len() == 1 {
                let (col, _) = entries[0];
                let pinned = state.commit_in.contains(&col)
                    || state.storage_in.contains(&col)
                    || state.transit_buckets_in.contains(&col)
                    || state.inflow_lags.contains(&col);
                assert!(
                    !(pinned && t.row_lower[row] == t.row_upper[row]),
                    "stage 0, active plant K=2: {family} row {row}: incoming \
                     state is pinned by column bounds; no row pins it"
                );
            }
            if entries.iter().any(|&(c, _)| c == col_dec) {
                assert!(
                    matches!(owners.get(&row), Some(RowOwner::Load { .. })),
                    "stage 0, active plant K=2: {family} row {row}: decision \
                     column must enter only its own bus's load-balance rows, \
                     got owner {:?}",
                    owners.get(&row)
                );
            }
        }
    }

    // Positive check: the decision's one structural entry is its own deposit
    // row — the row where the matching ring lane's out_col also carries
    // +1.0 (DeliveryRing::emit_deposit).
    let decision_entries = matrix.col(col_dec);
    assert_eq!(
        decision_entries.len(),
        1,
        "stage 0, active plant K=2: decision column must have exactly one \
         structural entry (its own deposit row), got {decision_entries:?}"
    );
    let (deposit_row, value) = decision_entries[0];
    assert_eq!(
        value, -1.0,
        "stage 0, active plant K=2: decision column's deposit-row coefficient must be -1.0"
    );
    let decision_lane = ring_lanes(&state, geom)
        .into_iter()
        .find(|l| matches!(l.kind, RingLaneKind::Anticipated { lane: 0 }))
        .expect("anticipated lane 0 must exist");
    assert_eq!(decision_lane.decision_col, Some(col_dec));
    assert!(
        decision_lane.out_cols.iter().any(|&c| matrix
            .col(c)
            .iter()
            .any(|&(r, v)| r == deposit_row && v == 1.0)),
        "stage 0, active plant K=2: row {deposit_row} must also carry +1.0 on the \
         lane's own out_col (the deposit row identity)"
    );
}

/// At an inactive stage (K=2, n_stages=4, stage 3: 3+2=5 > 4) the incoming
/// state is pinned by column bounds, so no row pins it: the
/// anticipated-decision column has no entry on any row addressing
/// `commit_in`/`storage_in`/`transit_buckets_in`/`inflow_lags`, nor on any
/// row outside its own bus's load-balance rows.
#[test]
fn test_anticipated_decision_inactive_no_state_write() {
    let system = one_anticipated_thermal_system(4, 2, 0.0, 100.0);
    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("build ok");

    let t = &result.templates[3]; // stage 3: 3+2=5 > 4 → inactive
    let col_dec = anticipated_decision_col(2);
    let lead_stages = vec![2_usize];
    let resolution = constant_lead_resolution(&lead_stages, 4);
    let state = StateSpace::new(0, 0, Vec::new(), lead_stages, resolution, &[]);
    let geom = &result.geometry_per_stage[3];
    let matrix = UnscaledMatrix::of(t);
    let owners = row_owners(&system, geom, &state);

    for (family, rows) in geometry_row_families(geom, &state) {
        for row in rows {
            let entries = matrix.row(row);
            if entries.len() == 1 {
                let (col, _) = entries[0];
                let pinned = state.commit_in.contains(&col)
                    || state.storage_in.contains(&col)
                    || state.transit_buckets_in.contains(&col)
                    || state.inflow_lags.contains(&col);
                assert!(
                    !(pinned && t.row_lower[row] == t.row_upper[row]),
                    "stage 3, inactive plant K=2: {family} row {row}: incoming \
                     state is pinned by column bounds; no row pins it"
                );
            }
            if entries.iter().any(|&(c, _)| c == col_dec) {
                assert!(
                    matches!(owners.get(&row), Some(RowOwner::Load { .. })),
                    "stage 3, inactive plant K=2: {family} row {row}: decision \
                     column must enter only its own bus's load-balance rows, \
                     got owner {:?}",
                    owners.get(&row)
                );
            }
        }
    }
}

/// With 1 hydro (max_par_order=1) and 1 anticipated thermal (K=2),
/// `n_state = N*(1+L) + n_ant_state = 1*(1+1) + 1*2 = 4`. `one_hydro_one_ant_system`
/// keeps the hydro term N*(1+L) non-zero so the full formula is exercised.
#[test]
fn test_n_state_includes_n_ant_state() {
    let system = one_hydro_one_ant_system(4);
    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("build ok");

    let t = &result.templates[0];
    // n_hydros=1, max_par_order=1 → N*(1+L) = 1*(1+1) = 2.
    // n_anticipated=1, k_max=2 → n_ant_state = 2.
    // Expected n_state = N*(1+L) + n_ant_state = 2 + 2 = 4.
    let expected_n_state = 4_usize;
    assert_eq!(
        t.n_state, expected_n_state,
        "n_state must equal N*(1+L) + n_ant_state = {expected_n_state}, got {}",
        t.n_state
    );
}

/// K=1 LP roundtrip: N=1 hydro, T=1 anticipated thermal (K=1), B=1 bus,
/// 2 blocks × 360h, n_stages=4, no discounting.
///
/// Verifies simultaneously:
/// - `n_state == 2` for all stages.
/// - `num_cols == 29` and `num_rows` per stage match the K=1 formula.
/// - anticipated_decision bounds: `[0,100]` when active (`t+1 < 4`), which
///   is stages 0..2 for K=1; INACTIVE at boundary stage 3 (`3+1=4==n_stages`,
///   excluded by the strict predicate).
/// - NPV objective coefficient at stage 0 (no discount): `50*720/1000 = 36.0`.
/// - Fishing row CSC at stage 1 (first stage with K=1 <= stage_idx=1).
/// - Fishing row equality bounds 0==0.
#[test]
fn test_anticipated_thermals_lp_roundtrip_k1() {
    let k = 1_usize;
    let n_stages = 4_usize;
    let block_hours = 360.0_f64;
    let total_hours = 2.0 * block_hours; // 720.0
    let system = build_k1_system();
    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("K=1 build ok");

    let col_ant_state = rt_col_ant_state_incoming_start(k); // fishing couples the INCOMING slot
    let col_ant_dec = rt_col_ant_dec_start(k); // 13
    let col_thermal = rt_col_thermal_start(k); // 11
    let row_fish_start = rt_row_ant_fishing_start(k); // 12

    // ── n_state ─────────────────────────────────────────────────────
    for t in 0..n_stages {
        assert_eq!(
            result.templates[t].n_state,
            1 + k,
            "K=1, stage {t}: n_state must be {} (1 hydro + 1 ant-slot), got {}",
            1 + k,
            result.templates[t].n_state
        );
    }

    // ── num_cols ────────────────────────────────────────────────────
    let expected_cols = rt_expected_num_cols(k);
    for t in 0..n_stages {
        assert_eq!(
            result.templates[t].num_cols, expected_cols,
            "K=1, stage {t}: num_cols must be {expected_cols}, got {}",
            result.templates[t].num_cols
        );
    }

    // ── num_rows ────────────────────────────────────────────────────
    for t in 0..n_stages {
        let expected_rows = rt_expected_num_rows(k, t);
        assert_eq!(
            result.templates[t].num_rows, expected_rows,
            "K=1, stage {t}: num_rows must be {expected_rows}, got {}",
            result.templates[t].num_rows
        );
    }

    // ── anticipated_decision bounds ────────────────────────────────
    // Active: t in 0..3 (t+1 < 4: 1, 2, 3 all < 4 under strict predicate).
    for t in 0..(n_stages - k) {
        let tmpl = &result.templates[t];
        assert_eq!(
            tmpl.col_lower[col_ant_dec], 0.0,
            "K=1, stage {t}: anticipated_decision col_lower must be 0.0 (active)"
        );
        assert_eq!(
            tmpl.col_upper[col_ant_dec], 100.0,
            "K=1, stage {t}: anticipated_decision col_upper must be 100.0 (active)"
        );
    }
    // Inactive: boundary stage t=3 (3+1=4 NOT < 4 under strict predicate).
    {
        let tmpl = &result.templates[n_stages - k];
        assert_eq!(
            tmpl.col_lower[col_ant_dec],
            0.0,
            "K=1, boundary stage {}: anticipated_decision col_lower must be 0.0 (strict predicate excludes)",
            n_stages - k
        );
        assert_eq!(
            tmpl.col_upper[col_ant_dec],
            0.0,
            "K=1, boundary stage {}: anticipated_decision col_upper must be 0.0 (strict predicate excludes; t+K=n_stages)",
            n_stages - k
        );
    }

    // ── NPV objective at stage 0 ────────────────────────────────────
    // delivery_stage = 0+1 = 1; cumulative_factor[1] = 1.0 (no discount).
    let expected_obj = 50.0 * total_hours * 1.0 / COST_SCALE_FACTOR; // 36.0
    assert!(
        (result.templates[0].objective[col_ant_dec] - expected_obj).abs() < 1e-12,
        "K=1, stage 0: anticipated_decision objective must be {expected_obj:.6}, got {:.6}",
        result.templates[0].objective[col_ant_dec]
    );

    // ── Fishing row CSC at stage 1 (K=1 <= 1) ──────────────────────
    {
        let t = &result.templates[1]; // stage 1: K=1 <= stage_idx=1 → fishing active
        let row_fish = row_fish_start;
        // Thermal generation columns: col_thermal + blk for blk in 0..2.
        for blk in 0..2 {
            let col = col_thermal + blk;
            let entries = csc_entries_at(t, col, row_fish);
            assert_eq!(
                entries,
                vec![block_hours],
                "K=1, stage 1: fishing CSC at thermal col (blk={blk}) must be [+{block_hours}], \
                 got {entries:?}"
            );
        }
        // Slot-0 anticipated-state column: -total_hours.
        let col_state_slot0 = col_ant_state; // slot 0, plant 0
        let entries = csc_entries_at(t, col_state_slot0, row_fish);
        let expected_neg = -total_hours; // -(360+360) = -720.0
        assert_eq!(
            entries,
            vec![expected_neg],
            "K=1, stage 1: fishing CSC at ant_state slot 0 must be [{expected_neg}], \
             got {entries:?}"
        );
    }

    // ── Fishing row equality bounds 0==0 at stage 1 ─────────────────
    {
        let t = &result.templates[1];
        let row_fish = row_fish_start;
        assert_eq!(
            t.row_lower[row_fish], 0.0,
            "K=1, stage 1: fishing row_lower must be 0.0"
        );
        assert_eq!(
            t.row_upper[row_fish], 0.0,
            "K=1, stage 1: fishing row_upper must be 0.0"
        );
    }

    // Fishing is present at every stage; state_out_def is active at stages 0,1,2
    // (0+1,1+1,2+1 all < 4) but not stage 3 — so 0,1,2 share num_rows, stage 3 has one fewer.
    {
        assert_eq!(
            result.templates[0].num_rows, result.templates[1].num_rows,
            "K=1: stage 0 and stage 1 must have equal row count (fishing always-active)"
        );
        assert_eq!(
            result.templates[1].num_rows, result.templates[2].num_rows,
            "K=1: stage 1 and stage 2 must have equal row count (state_out_def still active)"
        );
        assert_eq!(
            result.templates[3].num_rows + 1,
            result.templates[2].num_rows,
            "K=1: stage 3 must have 1 fewer row than stage 2 (state_out_def inactive at stage 3)"
        );
    }
}

/// K=2 LP roundtrip: N=1 hydro, T=1 anticipated thermal (K=2), B=1 bus,
/// 2×360h, n_stages=4.
///
/// Verifies:
/// - `n_state == 3` for all stages.
/// - `num_cols == 30` and `num_rows` per stage match K=2 formula.
/// - Bounds: active at t=0 (`0+2=2<4`), active at t=1 (`1+2=3<4`),
///   INACTIVE at boundary t=2 (`2+2=4 NOT < 4`) and t=3 (`3+2=5>4`).
/// - Interior carry row: `out(slot 1)` and `in(slot 1)` hold the same-slot
///   carry identity (+1.0/-1.0), never the retired shift target `in(slot+1)`.
/// - Fishing row active at stage 2 (K=2 <= 2), absent at stage 1 (K=2 > 1).
/// - Fishing row CSC pattern at stage 2.
#[test]
fn test_anticipated_thermals_lp_roundtrip_k2() {
    let k = 2_usize;
    let n_stages = 4_usize;
    let block_hours = 360.0_f64;
    let total_hours = 2.0 * block_hours; // 720.0
    let system = build_k2_system();
    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("K=2 build ok");

    let col_ant_state = rt_col_ant_state_incoming_start(k); // fishing couples the INCOMING slot
    let col_ant_dec = rt_col_ant_dec_start(k); // 14
    let col_thermal = rt_col_thermal_start(k); // 12
    let row_fish_start = rt_row_ant_fishing_start(k); // 12

    // ── n_state ─────────────────────────────────────────────────────
    for t in 0..n_stages {
        assert_eq!(
            result.templates[t].n_state,
            1 + k,
            "K=2, stage {t}: n_state must be {}, got {}",
            1 + k,
            result.templates[t].n_state
        );
    }

    // ── num_cols / num_rows ─────────────────────────────────────────
    let expected_cols = rt_expected_num_cols(k);
    for t in 0..n_stages {
        assert_eq!(
            result.templates[t].num_cols, expected_cols,
            "K=2, stage {t}: num_cols must be {expected_cols}, got {}",
            result.templates[t].num_cols
        );
        let expected_rows = rt_expected_num_rows(k, t);
        assert_eq!(
            result.templates[t].num_rows, expected_rows,
            "K=2, stage {t}: num_rows must be {expected_rows}, got {}",
            result.templates[t].num_rows
        );
    }

    // ── anticipated_decision bounds ─────────────────────────────────
    // Active under strict predicate: t=0 (0+2=2 < 4), t=1 (1+2=3 < 4).
    for t in 0..=1 {
        let tmpl = &result.templates[t];
        assert_eq!(
            tmpl.col_lower[col_ant_dec], 0.0,
            "K=2, stage {t}: anticipated_decision col_lower must be 0.0 (active)"
        );
        assert_eq!(
            tmpl.col_upper[col_ant_dec], 100.0,
            "K=2, stage {t}: anticipated_decision col_upper must be 100.0 (active)"
        );
    }
    // Inactive under strict predicate: boundary t=2 (2+2=4 NOT < 4) and t=3.
    for t in 2..n_stages {
        let tmpl = &result.templates[t];
        assert_eq!(
            tmpl.col_lower[col_ant_dec], 0.0,
            "K=2, stage {t}: anticipated_decision col_lower must be 0.0 (inactive)"
        );
        assert_eq!(
            tmpl.col_upper[col_ant_dec], 0.0,
            "K=2, stage {t}: anticipated_decision col_upper must be 0.0 (inactive under strict predicate; t+K >= n_stages)"
        );
    }

    // ── NPV objective at stage 0 ────────────────────────────────────
    // delivery_stage=2; cumulative_factor[2]=1.0 (no discount).
    let expected_obj = 50.0 * total_hours * 1.0 / COST_SCALE_FACTOR; // 36.0
    assert!(
        (result.templates[0].objective[col_ant_dec] - expected_obj).abs() < 1e-12,
        "K=2, stage 0: anticipated_decision objective must be {expected_obj:.6}, got {:.6}",
        result.templates[0].objective[col_ant_dec]
    );

    // ── Fishing row CSC at stage 2 (K=2 <= 2) ──────────────────────
    {
        let t = &result.templates[2];
        let row_fish = row_fish_start;
        for blk in 0..2 {
            let col = col_thermal + blk;
            let entries = csc_entries_at(t, col, row_fish);
            assert_eq!(
                entries,
                vec![block_hours],
                "K=2, stage 2: fishing CSC thermal col (blk={blk}) must be [+{block_hours}], \
                 got {entries:?}"
            );
        }
        let col_state_slot0 = col_ant_state;
        let expected_neg = -total_hours;
        let entries = csc_entries_at(t, col_state_slot0, row_fish);
        assert_eq!(
            entries,
            vec![expected_neg],
            "K=2, stage 2: fishing CSC ant_state slot 0 must be [{expected_neg}], got {entries:?}"
        );
    }

    // ── Fishing row equality bounds 0==0 at stage 2 ─────────────────
    {
        let t = &result.templates[2];
        let row_fish = row_fish_start;
        assert_eq!(
            t.row_lower[row_fish], 0.0,
            "K=2, stage 2: fishing row_lower must be 0.0"
        );
        assert_eq!(
            t.row_upper[row_fish], 0.0,
            "K=2, stage 2: fishing row_upper must be 0.0"
        );
    }

    // K=2, n_stages=4: fishing is 1 row at every stage; state_out_def (s+K<4) is
    // active at 0,1 but absent at 2,3. The single interior carry row (one
    // in-flight not-yet-due commitment) is present at 0,1,2 but not at 3 (its
    // delivery target m=4 is past the horizon), so stage 2 keeps one more row
    // than stage 3 despite both lacking state_out_def.
    {
        assert_eq!(
            result.templates[1].num_rows, result.templates[0].num_rows,
            "K=2: stage 1 must have same row count as stage 0 \
             (both have fishing + state_out_def + the interior carry row active)"
        );
        assert_eq!(
            result.templates[2].num_rows,
            result.templates[3].num_rows + 1,
            "K=2: stage 2 must have exactly 1 more row than stage 3 \
             (both lack state_out_def, but stage 2's interior carry commitment is \
             still inside the horizon while stage 3's is not)"
        );
        assert_eq!(
            result.templates[0].num_rows,
            result.templates[2].num_rows + 1,
            "K=2: stage 0 must have exactly 1 more row than stage 2 \
             (state_out_def active at stage 0, absent at stage 2)"
        );
    }

    // ── Interior carry (hold) definition row: the same-slot carry identity ──
    // At stage 0 the single in-flight not-yet-due commitment is delivery-1 (held
    // at its modular slot `1 mod k_max = 1`). Its carry row pins
    // `out(slot 1) - in(slot 1) = 0` — the SAME slot on both columns; the retired
    // shift geometry instead targeted `in(slot + 1)`. The outgoing anticipated
    // block is `[n_state - k, n_state)` (storage occupies `[0, n_hydros)` first);
    // the incoming block starts at `rt_col_ant_state_incoming_start(k)`.
    {
        let t0 = &result.templates[0];
        let out_slot1 = (t0.n_state - k) + 1; // outgoing slot 1
        let in_slot1 = col_ant_state + 1; // incoming slot 1
        let in_slot0 = col_ant_state; // incoming slot 0 (the retired shift target)
        let carry_row = (0..t0.num_rows)
            .find(|&r| csc_entries_at(t0, out_slot1, r) == vec![1.0])
            .expect("K=2: a carry definition row with out(slot 1) = +1.0 must exist");
        assert_eq!(
            csc_entries_at(t0, in_slot1, carry_row),
            vec![-1.0],
            "K=2 carry row: in(slot 1) must carry -1.0 — the SAME slot as out (hold identity)"
        );
        assert!(
            csc_entries_at(t0, in_slot0, carry_row).is_empty(),
            "K=2 carry row must NOT touch in(slot 0) — the retired shift target in(slot+1)"
        );
    }
}

/// K=3 LP roundtrip: N=1 hydro, T=1 anticipated thermal (K=3), B=1 bus,
/// 2×360h, n_stages=4.
///
/// Verifies:
/// - `n_state == 4` for all stages.
/// - `num_cols == 31` and `num_rows` per stage match K=3 formula.
/// - Bounds: active at t=0 (`0+3=3 < 4`), INACTIVE at boundary t=1
///   (`1+3=4 NOT < 4`), t=2 (`2+3=5>4`), and t=3.
/// - Interior carry rows: `out(slot)` and `in(slot)` hold the same-slot
///   carry identity (+1.0/-1.0) for slots 1 and 2.
/// - Fishing rows: absent at t=0,1,2; present at t=3 (K=3 <= 3).
/// - Fishing row CSC pattern at stage 3.
#[test]
fn test_anticipated_thermals_lp_roundtrip_k3() {
    let k = 3_usize;
    let n_stages = 4_usize;
    let block_hours = 360.0_f64;
    let total_hours = 2.0 * block_hours; // 720.0
    let system = build_k3_system();
    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("K=3 build ok");

    let col_ant_state = rt_col_ant_state_incoming_start(k); // fishing couples the INCOMING slot
    let col_ant_dec = rt_col_ant_dec_start(k); // 15
    let col_thermal = rt_col_thermal_start(k); // 13
    let row_fish_start = rt_row_ant_fishing_start(k); // 12 (K-independent)

    // ── n_state ─────────────────────────────────────────────────────
    for t in 0..n_stages {
        assert_eq!(
            result.templates[t].n_state,
            1 + k,
            "K=3, stage {t}: n_state must be {}, got {}",
            1 + k,
            result.templates[t].n_state
        );
    }

    // ── num_cols / num_rows ─────────────────────────────────────────
    let expected_cols = rt_expected_num_cols(k);
    for t in 0..n_stages {
        assert_eq!(
            result.templates[t].num_cols, expected_cols,
            "K=3, stage {t}: num_cols must be {expected_cols}, got {}",
            result.templates[t].num_cols
        );
        let expected_rows = rt_expected_num_rows(k, t);
        assert_eq!(
            result.templates[t].num_rows, expected_rows,
            "K=3, stage {t}: num_rows must be {expected_rows}, got {}",
            result.templates[t].num_rows
        );
    }

    // ── anticipated_decision bounds ─────────────────────────────────
    // Active under strict predicate: t=0 only (0+3=3 < 4).
    {
        let tmpl = &result.templates[0];
        assert_eq!(
            tmpl.col_lower[col_ant_dec], 0.0,
            "K=3, stage 0: anticipated_decision col_lower must be 0.0 (active)"
        );
        assert_eq!(
            tmpl.col_upper[col_ant_dec], 100.0,
            "K=3, stage 0: anticipated_decision col_upper must be 100.0 (active)"
        );
    }
    // Inactive under strict predicate: boundary t=1 (1+3=4 NOT < 4), t=2, t=3.
    for t in 1..n_stages {
        let tmpl = &result.templates[t];
        assert_eq!(
            tmpl.col_lower[col_ant_dec], 0.0,
            "K=3, stage {t}: anticipated_decision col_lower must be 0.0 (inactive)"
        );
        assert_eq!(
            tmpl.col_upper[col_ant_dec], 0.0,
            "K=3, stage {t}: anticipated_decision col_upper must be 0.0 (inactive under strict predicate; t+3 >= n_stages)"
        );
    }

    // ── NPV objective at stage 0 ────────────────────────────────────
    // delivery_stage=3; cumulative_factor[3]=1.0 (no discount).
    let expected_obj = 50.0 * total_hours * 1.0 / COST_SCALE_FACTOR; // 36.0
    assert!(
        (result.templates[0].objective[col_ant_dec] - expected_obj).abs() < 1e-12,
        "K=3, stage 0: anticipated_decision objective must be {expected_obj:.6}, got {:.6}",
        result.templates[0].objective[col_ant_dec]
    );

    // K=3, n_stages=4: fishing is always-active (1 row/stage); state_out_def
    // (s+K<4) is active at stage 0 only. Under the HOLD geometry the interior
    // carry rows count the in-flight not-yet-due commitments whose delivery
    // target `m in (stage, stage+3]` lands inside the horizon: 2 at stages 0-1,
    // 1 at stage 2, 0 at stage 3 — a count invariant of the retired shift slots.
    {
        assert_eq!(
            result.templates[1].num_rows,
            result.templates[2].num_rows + 1,
            "K=3: stage 1 must have 1 more row than stage 2 \
             (both lack state_out_def, but stage 2 has one fewer interior carry \
             commitment inside the horizon)"
        );
        assert_eq!(
            result.templates[0].num_rows,
            result.templates[1].num_rows + 1,
            "K=3: stage 0 must have 1 more row than stage 1 (state_out_def active at stage 0)"
        );
    }

    // ── Fishing row CSC at stage 3 (K=3 <= 3) ──────────────────────
    {
        let t = &result.templates[3];
        let row_fish = row_fish_start;
        for blk in 0..2 {
            let col = col_thermal + blk;
            let entries = csc_entries_at(t, col, row_fish);
            assert_eq!(
                entries,
                vec![block_hours],
                "K=3, stage 3: fishing CSC thermal col (blk={blk}) must be [+{block_hours}], \
                 got {entries:?}"
            );
        }
        let col_state_slot0 = col_ant_state;
        let expected_neg = -total_hours;
        let entries = csc_entries_at(t, col_state_slot0, row_fish);
        assert_eq!(
            entries,
            vec![expected_neg],
            "K=3, stage 3: fishing CSC ant_state slot 0 must be [{expected_neg}], got {entries:?}"
        );
    }

    // ── Fishing row equality bounds 0==0 at stage 3 ─────────────────
    {
        let t = &result.templates[3];
        let row_fish = row_fish_start;
        assert_eq!(
            t.row_lower[row_fish], 0.0,
            "K=3, stage 3: fishing row_lower must be 0.0"
        );
        assert_eq!(
            t.row_upper[row_fish], 0.0,
            "K=3, stage 3: fishing row_upper must be 0.0"
        );
    }

    // ── Row-count invariants under always-active fishing (K=3) ───────────────
    // For K=3, n_stages=4:
    //   Fishing: 1 row per stage at every stage (always-active predicate).
    //   State_out_def (s+K<4): active at 0; absent at 1,2,3.
    //   Interior carry rows: 2 at stages 0-1, 1 at stage 2, 0 at stage 3 (the
    //   in-flight not-yet-due commitments whose delivery target stays inside the
    //   horizon) — so stage 2 has one more row than stage 3, not an equal count.
    {
        assert_eq!(
            result.templates[3].num_rows + 1,
            result.templates[2].num_rows,
            "K=3: stage 2 must have 1 more row than stage 3 \
             (fishing always-active and state_out_def absent at both, but \
             stage 2 still has one interior carry commitment inside the \
             horizon while stage 3 has none)"
        );
    }

    // ── Interior carry (hold) definition rows: the same-slot carry identity ──
    // At stage 0 the two in-flight not-yet-due commitments are delivery-1 (slot
    // `1 mod 3 = 1`) and delivery-2 (slot `2 mod 3 = 2`); delivery-3 is the fresh
    // decision (deposit slot `3 mod 3 = 0`, not a carry). Each carry row pins
    // `out(slot) - in(slot) = 0` on the SAME slot. The outgoing anticipated block
    // is `[n_state - k, n_state)`; the incoming block starts at
    // `rt_col_ant_state_incoming_start(k)`.
    {
        let t0 = &result.templates[0];
        let out_start = t0.n_state - k; // outgoing anticipated block start
        for slot in [1_usize, 2] {
            let out_col = out_start + slot;
            let in_col = col_ant_state + slot;
            let carry_row = (0..t0.num_rows)
                .find(|&r| csc_entries_at(t0, out_col, r) == vec![1.0])
                .unwrap_or_else(|| {
                    panic!("K=3: a carry definition row with out(slot {slot}) = +1.0 must exist")
                });
            assert_eq!(
                csc_entries_at(t0, in_col, carry_row),
                vec![-1.0],
                "K=3 carry row (slot {slot}): in(slot) must carry -1.0 — the SAME slot (hold identity)"
            );
        }
    }
}

/// A thermal with `anticipated_config: None` (K=0) yields `n_anticipated=0` and a
/// layout identical to the pre-anticipated baseline: no anticipated columns, no
/// fishing rows. Same geometry as the K-cases (N=1 hydro, T=1 thermal, B=1,
/// 2×360h, n_stages=4).
#[test]
fn test_anticipated_thermals_lp_roundtrip_k0_baseline_parity() {
    let system_baseline = build_k0_baseline_system();
    let result_baseline = build_stage_templates_resolving_layout(
        &system_baseline,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system_baseline),
        &default_evaporation(&system_baseline),
        &ResolvedParameters::default(),
    )
    .expect("baseline build ok");

    // A second identical build must be bit-identical (determinism).
    let result_baseline2 = build_stage_templates_resolving_layout(
        &system_baseline,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system_baseline),
        &default_evaporation(&system_baseline),
        &ResolvedParameters::default(),
    )
    .expect("baseline build2 ok");

    let n_stages = result_baseline.templates.len();
    assert_eq!(n_stages, 4, "baseline must have 4 templates");

    for s in 0..n_stages {
        let ta = &result_baseline.templates[s];
        let tb = &result_baseline2.templates[s];
        assert_eq!(
            ta.num_cols, tb.num_cols,
            "parity: stage {s} num_cols must match ({} vs {})",
            ta.num_cols, tb.num_cols
        );
        assert_eq!(
            ta.num_rows, tb.num_rows,
            "parity: stage {s} num_rows must match ({} vs {})",
            ta.num_rows, tb.num_rows
        );
        assert_eq!(
            ta.n_state, tb.n_state,
            "parity: stage {s} n_state must match ({} vs {})",
            ta.n_state, tb.n_state
        );
        assert_eq!(
            ta.col_starts, tb.col_starts,
            "parity: stage {s} col_starts differ between two builds"
        );
        assert_eq!(
            ta.row_indices, tb.row_indices,
            "parity: stage {s} row_indices differ between two builds"
        );
        assert_eq!(
            ta.values, tb.values,
            "parity: stage {s} CSC values differ between two builds"
        );
        assert_eq!(
            ta.col_lower, tb.col_lower,
            "parity: stage {s} col_lower differs between two builds"
        );
        assert_eq!(
            ta.col_upper, tb.col_upper,
            "parity: stage {s} col_upper differs between two builds"
        );
        assert_eq!(
            ta.objective, tb.objective,
            "parity: stage {s} objective differs between two builds"
        );
        assert_eq!(
            ta.row_lower, tb.row_lower,
            "parity: stage {s} row_lower differs between two builds"
        );
        assert_eq!(
            ta.row_upper, tb.row_upper,
            "parity: stage {s} row_upper differs between two builds"
        );
    }

    // n_state = N*(1+L) = 1 (no anticipated term).
    assert_eq!(
        result_baseline.templates[0].n_state, 1,
        "K=0 baseline: n_state must be 1 (no anticipated state)"
    );
    // num_cols = 26: the anticipated geometry's 28+K drops both K-only columns
    // (1 commitment-hold slot + 1 anticipated_decision) when K=0.
    assert_eq!(
        result_baseline.templates[0].num_cols, 26,
        "K=0 baseline: num_cols must be 26 (no anticipated state or decision columns)"
    );
    assert_eq!(
        result_baseline.templates[0].num_rows, 12,
        "K=0 baseline: num_rows must be 12 (no rows pin incoming state)"
    );
}

/// K=2 roundtrip with 6% annual discount: the stage-0 anticipated-decision
/// objective uses the DELIVERY stage (2) factors, `50 * total_hours *
/// cumulative_discount_factors[2] / COST_SCALE_FACTOR`. With 31-day stages,
/// `per_stage_factor = 1 / (1.06)^(31/365.25)` and `cumulative[2] = per_stage_factor^2`.
#[test]
fn test_anticipated_thermals_lp_roundtrip_k2_with_discount_rate() {
    let k = 2_usize;
    let annual_rate = 0.06_f64;
    let block_hours = 360.0_f64;
    let total_hours = 2.0 * block_hours; // 720.0

    let system = build_hydro_one_ant_system(4, k as u32, annual_rate);
    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("K=2 discount build ok");

    let col_ant_dec = rt_col_ant_dec_start(k); // 14

    // cumulative_discount_factors[delivery=2], computed from 31-day stages.
    let dt_days = 31.0_f64;
    let per_stage_factor = 1.0 / (1.0 + annual_rate).powf(dt_days / 365.25);
    let cumulative_at_delivery = per_stage_factor * per_stage_factor;

    let expected_obj = 50.0 * total_hours * cumulative_at_delivery / COST_SCALE_FACTOR;

    let actual_obj = result.templates[0].objective[col_ant_dec];
    let rel_err = (actual_obj - expected_obj).abs() / expected_obj.abs().max(f64::EPSILON);
    assert!(
        rel_err < 1e-12,
        "K=2 with 6% discount: stage 0 anticipated_decision objective must be {expected_obj:.15} \
         (rel_err={rel_err:.2e}), got {actual_obj:.15}"
    );
}

/// A decision taken at stage 1 is priced in stage-1 units, like every other
/// stage-1 cost: `D(3) / D(1)`, so the θ cascade's `D(1)` brings it to `D(3)` at
/// the root instead of `D(1)·D(3)`.
#[test]
fn test_anticipated_decision_after_stage_zero_is_priced_relative_to_its_own_stage() {
    let k = 2_usize;
    let annual_rate = 0.06_f64;
    let total_hours = 2.0 * 360.0_f64;

    let system = build_hydro_one_ant_system(4, k as u32, annual_rate);
    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("K=2 discount build ok");

    let per_stage_factor = 1.0 / (1.0 + annual_rate).powf(31.0 / 365.25);
    let decision_stage = 1;
    let delivery_stage = decision_stage + k;
    let relative_discount = per_stage_factor.powi((delivery_stage - decision_stage) as i32);
    let expected_obj = 50.0 * total_hours * relative_discount / COST_SCALE_FACTOR;

    let actual_obj = result.templates[decision_stage].objective[rt_col_ant_dec_start(k)];
    let rel_err = (actual_obj - expected_obj).abs() / expected_obj.abs().max(f64::EPSILON);
    assert!(
        rel_err < 1e-12,
        "stage {decision_stage} anticipated_decision objective must be \
         50 * {total_hours} * D({delivery_stage})/D({decision_stage}) / COST_SCALE_FACTOR = \
         {expected_obj:.15} (rel_err={rel_err:.2e}), got {actual_obj:.15}"
    );
}

/// Every active in-study anticipated decision is priced relative to its own
/// decision stage's discount, `D(t + 2)/D(t)` — never the absolute delivery
/// discount `D(t + 2)`.
#[test]
fn discounted_anticipated_study_prices_each_decision_relative_to_its_stage() {
    let (system, config) = discounted_anticipated_study();
    let setup = build_setup_in_code(system, &config);
    let stage_ctx = setup.stage_ctx();
    let d = stage_ctx.cumulative_discount_factors;

    let mut checked_after_stage_zero = false;
    for t in 0..stage_ctx.templates.len() {
        let decision_col = stage_ctx.geometry_per_stage[t].anticipated_decision.start;
        let template = &stage_ctx.templates[t];
        if template.col_upper[decision_col] == 0.0 {
            continue;
        }
        let unscaled = template.objective[decision_col] * stage_ctx.cost_scale_factor
            / template.col_scale[decision_col];
        let expected = 50.0 * 720.0 * d[t + 2] / d[t];
        let rel_err = (unscaled - expected).abs() / expected.abs().max(1e-12);
        assert!(
            rel_err < 1e-12,
            "stage {t}: unscaled anticipated decision objective must be \
             50 * 720 * D({})/D({t}) = {expected:.15}, got {unscaled:.15} (rel_err={rel_err:.2e})",
            t + 2,
        );
        checked_after_stage_zero |= t >= 1;
    }
    assert!(
        checked_after_stage_zero,
        "vacuity guard: no active anticipated decision at t >= 1 was checked"
    );
}

/// The discounted-anticipated fixture's stage-1 anticipated decision is
/// costed: `LeadStages(2)` on a 4-stage horizon decides stages 0 and 1, both
/// delivering after stage 0, and the decision column's objective coefficient
/// must be non-zero for the discount path to be exercised at all.
#[test]
fn discounted_anticipated_fixture_decides_after_stage_zero() {
    let (system, config) = discounted_anticipated_study();
    let setup = build_setup_in_code(system, &config);

    let geometry = &setup.inputs.stage_data.stage_templates.geometry_per_stage[1];
    assert!(
        !geometry.anticipated_decision.is_empty(),
        "stage 1 must have an active anticipated-decision column"
    );
    let template = &setup.inputs.stage_data.stage_templates.templates[1];
    assert!(
        template.objective[geometry.anticipated_decision.start] > 0.0,
        "stage 1's anticipated decision must carry a nonzero costed objective coefficient"
    );
}

/// The mixed-lead fixture's `LeadStages(3)` thermal has no in-study delivery
/// target left at decision stages 2-4 (`n_stages == 5`), so each of those
/// decision columns must be costed against the declared post-study calendar
/// — the coverage the fixture's own doc comment claims, pinned as a fact
/// rather than left as a doc-only claim.
#[test]
fn mixed_lead_long_lead_late_decisions_target_post_study_delivery() {
    use cobre_sddp::indexer::AnticipatedLocal;

    let (system, config) = mixed_lead_anticipated_study(false);
    let setup = build_setup_in_code(system, &config);

    // Canonical anticipated-local order is ascending EntityId: the short lead
    // (id 10) is local 0, the long lead (id 20) is local 1.
    let long_lead_local = AnticipatedLocal::new(1);
    for stage_idx in 2..5 {
        let geometry = &setup.inputs.stage_data.stage_templates.geometry_per_stage[stage_idx];
        let template = &setup.inputs.stage_data.stage_templates.templates[stage_idx];
        let col = geometry.anticipated_decision_col(long_lead_local);
        assert!(
            template.objective[col] > 0.0,
            "stage {stage_idx}'s long-lead decision must carry a nonzero costed \
             objective coefficient (a post-study delivery target), got {}",
            template.objective[col]
        );
    }
}

/// The simulation books the anticipated decision's present value at its
/// delivery stage: `anticipated_thermal_cost * discount_factor` (both read at
/// the decision stage) equals `cost * hours * D(delivery) * decision_mw`.
#[test]
fn discounted_anticipated_simulation_books_present_value_at_delivery() {
    let (system, config) = discounted_anticipated_study();
    // The fixture's own config leaves simulation disabled; enable it here,
    // sampling a single deterministic scenario (std_m3s == 0.0,
    // branching_factor == 1).
    let config = Config {
        simulation: IoSimulationConfig {
            enabled: true,
            io_channel_capacity: 8,
            selection: Some(SimulationSelection::Sampled { num_scenarios: 1 }),
            ..IoSimulationConfig::default()
        },
        ..config
    };
    let mut setup = build_setup_in_code(system, &config);
    let scenario_results = run_simulation(&mut setup, 5);
    let scenario = &scenario_results[0];
    let d: Vec<f64> = setup.stage_ctx().cumulative_discount_factors.to_vec();

    let mut positive_after_stage_zero = false;
    for (t, stage) in scenario.stages.iter().enumerate() {
        let Some(thermal) = stage.thermals.iter().find(|th| th.is_anticipated) else {
            continue;
        };
        let Some(x) = thermal.anticipated_decision_mw else {
            continue;
        };
        let cost = &stage.costs[0];
        let lhs = cost.anticipated_thermal_cost * cost.discount_factor;
        let rhs = 50.0 * 720.0 * d[t + 2] * x;
        let tol = 1e-9 * rhs.abs().max(1.0);
        assert!(
            (lhs - rhs).abs() < tol,
            "stage {t}: anticipated_thermal_cost * discount_factor must book the present \
             value at delivery: 50 * 720 * D({}) * {x} = {rhs}, got {lhs}",
            t + 2,
        );
        if t >= 1 && x > 0.0 {
            positive_after_stage_zero = true;
        }
    }
    assert!(
        positive_after_stage_zero,
        "vacuity guard: no stage t >= 1 has a positive anticipated decision"
    );
}
