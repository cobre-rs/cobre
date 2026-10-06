//! `turbined_cost` section tests.

use super::*;

#[test]
fn turbined_cost_applied_to_fpha_turbine_column() {
    // turbined_cost = 0.5 $/(m³/s·h) over a 720h block → turbine objective 0.5*720 = 360.0
    // (then scaled by 1/COST_SCALE_FACTOR).
    let (system, production) = fpha_system_with_turbined_cost(3, 0.5, &[720.0]);
    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &production,
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("fpha system builds ok");
    let t = &result.templates[0];
    let turbine_col = 4_usize;
    let expected = 0.5 * 720.0 / COST_SCALE_FACTOR;
    assert!(
        (t.objective[turbine_col] - expected).abs() < 1e-15,
        "FPHA turbine col objective: expected {expected}, got {}",
        t.objective[turbine_col]
    );
}

#[test]
fn turbined_cost_multi_block_uses_per_block_hours() {
    // turbined_cost = 1.0 $/(m³/s·h); each turbine column carries cost * its own
    // block_hours (block 0 = 300h, block 1 = 420h), not the stage total.
    let (system, production) = fpha_system_with_turbined_cost(3, 1.0, &[300.0, 420.0]);
    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &production,
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("fpha multi-block system builds ok");
    let t = &result.templates[0];
    let col_blk0 = 4_usize;
    let col_blk1 = 5_usize;
    assert!(
        (t.objective[col_blk0] - 300.0 / COST_SCALE_FACTOR).abs() < 1e-15,
        "block 0 turbine objective: expected {}, got {}",
        300.0 / COST_SCALE_FACTOR,
        t.objective[col_blk0]
    );
    assert!(
        (t.objective[col_blk1] - 420.0 / COST_SCALE_FACTOR).abs() < 1e-15,
        "block 1 turbine objective: expected {}, got {}",
        420.0 / COST_SCALE_FACTOR,
        t.objective[col_blk1]
    );
}

#[test]
fn turbined_cost_mixed_system_all_hydros_carry_cost() {
    let (system, production) = four_hydro_mixed_system();

    let hydro_pen = HydroPenalties {
        spillage_cost: 0.01,
        diversion_cost: 0.0,
        turbined_cost: 1.0,
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
    };
    let penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: 4,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages: 1,
        },
        &PenaltiesDefaults {
            hydro: hydro_pen,
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    );

    let system = SystemBuilder::new()
        .buses(system.buses().to_vec())
        .hydros(system.hydros().to_vec())
        .stages(system.stages().to_vec())
        .inflow_models(system.inflow_models().to_vec())
        .load_models(system.load_models().to_vec())
        .bounds(system.bounds().clone())
        .penalties(penalties)
        .build()
        .expect("mixed system with turbined cost");

    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &production,
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("mixed system builds");
    let t = &result.templates[0];

    // N=4, L=0, K=1: theta=12, decision_start=13
    // col_turbine_start=13, turbine cols: h0=13, h1=14, h2=15, h3=16
    // spillage cols: h0=17, h1=18, h2=19, h3=20
    let col_turbine_start = 13;
    let block_hours = 744.0;

    let expected = block_hours / COST_SCALE_FACTOR;
    for h in 0..4 {
        assert!(
            (t.objective[col_turbine_start + h] - expected).abs() < 1e-15,
            "hydro {h} turbine objective should be {expected}, got {}",
            t.objective[col_turbine_start + h]
        );
    }
}

#[test]
fn load_balance_rhs_matches_load_model_mean_mw() {
    // Fixture LoadModel mean_mw = 100.0, so the load-balance RHS is 100.0.
    let system = one_bus_system(1);
    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("constant productivity ok");
    let t = &result.templates[0];
    // No hydros → water_balance_rows=0, load_balance at row 0, blk 0
    let load_row = 0;
    assert_eq!(
        t.row_lower[load_row], 100.0,
        "row_lower for load balance must be mean_mw"
    );
    assert_eq!(
        t.row_upper[load_row], 100.0,
        "row_upper for load balance must be mean_mw"
    );
}

#[test]
fn multiple_stages_produce_same_count_templates_and_block_hours() {
    let system = one_hydro_system(3, 1);
    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("constant productivity ok");
    assert_eq!(result.templates.len(), 3);
    assert_eq!(result.block_hours_per_stage.len(), 3);
}

#[test]
fn stage_templates_clone_and_debug() {
    let system = one_hydro_system(1, 0);
    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("constant productivity ok");
    let cloned = result.clone();
    assert_eq!(cloned.templates.len(), result.templates.len());
    let s = format!("{result:?}");
    assert!(s.contains("StageTemplates"));
}
