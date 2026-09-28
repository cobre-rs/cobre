//! Terminal boundary-FCF pricing of a post-study-targeted delivery on the
//! unified anticipated ring: a loaded boundary cut carrying coefficient `β` on
//! the ring slot the delivery's modular residue resolves to reaches the
//! terminal `θ`, which responds to the carried committed MW by `β·x`, and the
//! backward pass propagates that `β` inward to the deciding stage's cut. The
//! one shared boundary prices every terminal fan leaf against its OWN ring
//! state (the shared-fan injection topology). `β` values the carried STATE;
//! the delivery-anchored fuel is booked separately on the decision column —
//! the two are disjoint columns, so they compose without double-counting.
//!
//! The fixture is deliberately minimal: zero hydros, one bus, one anticipated
//! thermal declared `LeadTime`, two study stages, and one declared
//! `post_study_stages` stage. The lead is long enough that the thermal's own
//! stage-1 delivery decides PRE-STUDY (dormant, no genuine in-study decision)
//! while the post-study target's decider lands at stage 0 — so the ring sizes
//! non-degenerately (`k_max == 2`, avoiding the untested `k_max == 0` path)
//! and the terminal's own carried slot is a pure CARRY (`out = in`), never a
//! fresh deposit — pinning the incoming ring column via
//! `patch_backward_opening_for_probe` controls the terminal's priced
//! committed MW directly. With `hydro_count = 0` and `max_par_order = 0`, the
//! state vector is exactly the two-slot ring (`n_state == 2`), so a boundary
//! priced ONLY on the post-study-targeted slot isolates the pricing
//! arithmetic to that slot.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::too_many_lines
)]

use std::path::Path;

use chrono::NaiveDate;
use cobre_core::entities::thermal::AnticipatedConfig;
use cobre_core::temporal::{Node as PolicyNode, PolicyGraphType, StageStateConfig, Transition};
use cobre_core::{
    AnticipatedCommitmentHistory, BoundsCountsSpec, BoundsDefaults, ContractBlockBounds, EntityId,
    HorizonGraph, HydroBlockBounds, HydroStageBounds, InitialConditions, LineBlockBounds,
    PostStudyStage, PostStudyStages, PostStudyThermalBound, PumpingBlockBounds, ResolvedBounds,
    System, SystemBuilder, ThermalBlockBounds, ThermalStageBounds,
};
use cobre_sddp::indexer::{CutStateProjection, StateDim};
use cobre_sddp::setup::{NodeGraph, NodeId, NodePos, StageIdx};
use cobre_sddp::test_support::{patch_backward_opening_for_probe, solve_stage_for_probe};
use cobre_sddp::workspace::SolverWorkspace;
use cobre_sddp::{BoundaryLoadRequest, inject_boundary_cuts, load_boundary_cuts};
use cobre_solver::{
    ActiveSolver, FreezeScratch, RowBatch, SolverInterface, StageTemplate,
    freeze_rows_into_template,
};

mod common;
use common::StubComm;
use common::build_setup_in_code;
use common::builders::{BusSpec, StageSpec, ThermalSpec, make_bus, make_stage, make_thermal};

const BUS_ID: EntityId = EntityId(1);
const THERMAL_ID: EntityId = EntityId(2);
/// `LeadTime` delta (hours), ~62.5 days. Resolves the thermal's own stage-1
/// delivery to a PRE-STUDY decider (`None` — dormant, no genuine decision),
/// while the post-study target resolves to a stage-0 decider — the terminal
/// slot is a carry, not the latch.
const DELTA_HOURS: f64 = 1500.0;
/// Boundary cut intercept `α`, positive so the single injected cut binds `θ`
/// above its zero floor for every committed MW `x >= 0`.
const ALPHA: f64 = 5.0;
/// Boundary cut coefficient `β` on the post-study-targeted ring slot (zero on
/// the dormant padding slot), so `θ = α + β·x` isolates that slot.
const BETA: f64 = 3.0;
/// Post-study fuel `$/MWh` at the resolved cell — booked on the decision
/// column, disjoint from the terminal `β·state` valuation this file pins.
const COST_PER_MWH: f64 = 37.5;
/// Post-study stage duration \[h\]; the study carries a `0.0` discount rate.
const POST_STUDY_HOURS: f64 = 720.0;
const TOL: f64 = 1e-6;

fn study_start() -> NaiveDate {
    NaiveDate::from_ymd_opt(2024, 1, 1).expect("valid date")
}

/// Study end date (`stage 1` end) — the first `post_study_stages` stage and the
/// post-horizon delivery window both anchor here, making the segment
/// date-contiguous with the horizon (the first-post-study-start == study-end
/// rule).
fn study_end() -> NaiveDate {
    study_start() + chrono::TimeDelta::days(61)
}

/// Two stages matching their real calendar span exactly (`StageCalendar`
/// coverage): stage 0 January (744h), stage 1 the following 30 days (720h).
fn stages() -> Vec<cobre_core::temporal::Stage> {
    let start = study_start();
    let stage0_end = start + chrono::TimeDelta::days(31);
    vec![
        make_stage(
            0,
            StageSpec {
                start_date: start,
                end_date: stage0_end,
                blocks: vec![cobre_core::temporal::Block {
                    index: 0,
                    name: "S0".to_string(),
                    duration_hours: 744.0,
                }],
                ..Default::default()
            },
        ),
        make_stage(
            1,
            StageSpec {
                start_date: stage0_end,
                end_date: study_end(),
                blocks: vec![cobre_core::temporal::Block {
                    index: 0,
                    name: "S1".to_string(),
                    duration_hours: 720.0,
                }],
                ..Default::default()
            },
        ),
    ]
}

/// The one declared post-study stage: the span the anticipated
/// `LeadTime(DELTA_HOURS)` thermal's post-study-targeted delivery resolves onto,
/// so `StageCalendar::resolve_window` covers it at destination index 0.
fn post_study_stages() -> PostStudyStages {
    PostStudyStages {
        stages: vec![PostStudyStage {
            start_date: study_end(),
            duration_hours: POST_STUDY_HOURS,
        }],
        thermal_bounds: vec![PostStudyThermalBound {
            thermal_id: THERMAL_ID,
            post_study_stage_index: 0,
            cost_per_mwh: COST_PER_MWH,
            min_mw: 0.0,
            max_mw: 100.0,
        }],
    }
}

fn bounds() -> ResolvedBounds {
    ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 0,
            n_thermals: 1,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: 2,
            k_max: 1,
        },
        &BoundsDefaults {
            hydro: HydroStageBounds {
                min_storage_hm3: 0.0,
                max_storage_hm3: 0.0,
                filling_min_rate_m3s: 0.0,
                water_withdrawal_m3s: 0.0,
            },
            hydro_block: HydroBlockBounds::default(),
            thermal: ThermalStageBounds { cost_per_mwh: 1.0 },
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
    )
}

fn penalties() -> cobre_core::resolved::ResolvedPenalties {
    use cobre_core::HydroPenalties;
    use cobre_core::resolved::{
        BusStagePenalties, LineStagePenalties, NcsStagePenalties, PenaltiesCountsSpec,
        PenaltiesDefaults, ResolvedPenalties,
    };
    ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: 0,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages: 2,
        },
        &PenaltiesDefaults {
            hydro: HydroPenalties {
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
            },
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    )
}

/// Two-leaf terminal fan: root (id 0, stage 0) branches into leaves (ids 1 and
/// 2, stage 1) — both terminal, sharing ONE pool (`build_node_graph`'s
/// leaf-sharing rule).
fn two_leaf_fan_graph() -> HorizonGraph {
    HorizonGraph {
        graph_type: PolicyGraphType::FiniteHorizon,
        annual_discount_rate: 0.0,
        transitions: vec![
            Transition {
                source_id: 0,
                target_id: 1,
                probability: 0.5,
                annual_discount_rate_override: None,
            },
            Transition {
                source_id: 0,
                target_id: 2,
                probability: 0.5,
                annual_discount_rate_override: None,
            },
        ],
        nodes: vec![
            PolicyNode {
                id: 0,
                stage_id: 0,
                scenario_id: None,
                label: None,
            },
            PolicyNode {
                id: 1,
                stage_id: 1,
                scenario_id: None,
                label: None,
            },
            PolicyNode {
                id: 2,
                stage_id: 1,
                scenario_id: None,
                label: None,
            },
        ],
        stage_discount_rate_overrides: std::collections::BTreeMap::new(),
        season_map: None,
    }
}

/// `with_window` toggles the covering `post_study_stages` — which the
/// anticipated `LeadTime(DELTA_HOURS)` thermal resolves a post-study-targeted
/// delivery onto; `fanned` toggles the two-leaf terminal fan (chain otherwise).
fn build_system(with_window: bool, fanned: bool) -> System {
    let bus = make_bus(BUS_ID, BusSpec::default());
    let thermal = make_thermal(
        THERMAL_ID,
        ThermalSpec {
            bus_id: BUS_ID,
            cost_per_mwh: 1.0,
            min_generation_mw: 0.0,
            max_generation_mw: 0.0,
            anticipated_config: Some(AnticipatedConfig::LeadTime(DELTA_HOURS)),
            ..Default::default()
        },
    );
    let mut builder = SystemBuilder::new()
        .buses(vec![bus])
        .thermals(vec![thermal])
        .stages(stages())
        .bounds(bounds())
        .penalties(penalties())
        .initial_conditions(InitialConditions::default());
    if with_window {
        builder = builder.post_study_stages(Some(post_study_stages()));
    }
    if fanned {
        builder = builder.policy_graph(two_leaf_fan_graph());
    }
    builder.build().expect("fixture System must build")
}

fn config() -> cobre_io::config::Config {
    use cobre_io::config::{
        Config, EstimationConfig, ExportsConfig, InflowNonNegativityConfig,
        InflowNonNegativityMethod, ModelingConfig, ParallelismConfig, PolicyConfig,
        RowSelectionConfig, SimulationConfig, StoppingMode, StoppingRuleConfig, TrainingConfig,
        TrainingSelection, TrainingSolverConfig, UpperBoundEvaluationConfig,
    };
    Config {
        schema: None,
        modeling: ModelingConfig {
            inflow_non_negativity: InflowNonNegativityConfig {
                method: InflowNonNegativityMethod::Penalty,
            },
            // Unscaled: the priced `θ` and the propagated cut coefficient
            // compare directly against `α`/`β` with no COST_SCALE_FACTOR back-out.
            cost_scale_factor: Some(1.0),
        },
        training: TrainingConfig {
            enabled: true,
            tree_seed: Some(42),
            stopping_rules: Some(vec![StoppingRuleConfig::IterationLimit { limit: 1 }]),
            stopping_mode: StoppingMode::Any,
            cut_selection: RowSelectionConfig::default(),
            solver: TrainingSolverConfig::default(),
            parallelism: ParallelismConfig::default(),
            scenario_source: None,
            selection: Some(TrainingSelection::Sampled { forward_passes: 1 }),
        },
        upper_bound_evaluation: UpperBoundEvaluationConfig::default(),
        policy: PolicyConfig::default(),
        simulation: SimulationConfig::default(),
        exports: ExportsConfig::default(),
        estimation: EstimationConfig::default(),
    }
}

/// State-vector index of the ring slot carrying the post-study-targeted
/// delivery: `commitment_hold_in_study_offset(plant, m)` for the sole
/// anticipated plant and delivery target `m = num_stages()` (the first
/// post-study stage) — mirrors `StateSpace::commitment_hold_in_study_offset`
/// byte-for-byte (`pub(crate)`, unreachable from this integration-test
/// binary), never a hand-rolled `commit_out.end`-relative guess.
fn post_study_ring_slot(setup: &cobre_sddp::StudySetup) -> usize {
    let state = setup.stage_state();
    let m = setup.num_stages();
    state.commit_out.start + (m % state.k_max) * state.n_anticipated
}

/// Pool `pool`'s fixture `priced_state_date`: `2030-01-01` plus `pool`
/// months.
fn fixture_priced_date(pool: u32) -> NaiveDate {
    cobre_sddp::test_support::fixture_priced_date(cobre_sddp::test_support::ymd(2030, 1, 1), pool)
}

/// Load a boundary carrying `β` on [`post_study_ring_slot`] alone, zero
/// elsewhere, intercept `α`, and inject it into `setup`'s terminal pool.
fn inject_ring_boundary(setup: &mut cobre_sddp::StudySetup, dir: &Path) {
    let state_dimension = setup.fcf.state_dimension as u32;
    let slot = post_study_ring_slot(setup);
    let mut coefficients = vec![0.0_f64; state_dimension as usize];
    coefficients[slot] = BETA;
    cobre_sddp::test_support::write_synthetic_boundary(
        dir,
        state_dimension,
        ALPHA,
        &coefficients,
        fixture_priced_date(0),
    );

    let boundary_cuts = load_boundary_cuts(&BoundaryLoadRequest::new(
        dir,
        fixture_priced_date(0),
        state_dimension,
        &[],
        1.0,
    ))
    .expect("boundary cut must load");
    inject_boundary_cuts(setup, &boundary_cuts).unwrap();
}

/// Build the terminal pool's FROZEN LP template: the base structural template
/// plus one literal row per active cut in the pool, exactly as production
/// freezes a pool once per iteration (`freeze_rows_into_template`).
/// `solve_stage_for_probe`'s `pool` argument only drives basis reconstruction
/// on a stored warm start — it never appends cut rows to a cold solve, so
/// pricing an injected cut requires freezing it into the loaded model first.
fn freeze_terminal_template(setup: &cobre_sddp::StudySetup, pool_id: usize) -> StageTemplate {
    let state = setup.stage_state();
    let terminal_stage = setup.num_stages() - 1;
    let ctx = setup.stage_ctx();
    let base = &ctx.templates[terminal_stage];

    // The terminal pool projects the full global state, so the all-enabled
    // projection reproduces exactly the coefficients the boundary cut carries.
    let cut_state = CutStateProjection::new(
        state,
        StageStateConfig {
            storage: true,
            inflow_lags: true,
        },
    );

    let mut batch = RowBatch {
        num_rows: 0,
        row_starts: Vec::new(),
        col_indices: Vec::new(),
        values: Vec::new(),
        row_lower: Vec::new(),
        row_upper: Vec::new(),
    };
    cobre_sddp::build_cut_row_batch_into(
        &mut batch,
        &setup.fcf,
        pool_id,
        state,
        &cut_state,
        &base.col_scale,
    );

    let mut frozen = StageTemplate::empty();
    let mut scratch = FreezeScratch::new();
    freeze_rows_into_template(base, &batch, &mut frozen, &mut scratch);
    frozen
}

/// Solve the terminal stage's frozen `template` against `pool` with the state
/// vector pinned to `pinned_state` (length `n_state`), returning `θ`'s primal
/// value. `raw_noise = &[]`: `hydro_count == 0`, so the noise transform has no
/// dimension to iterate.
fn terminal_theta(
    setup: &cobre_sddp::StudySetup,
    template: &StageTemplate,
    pool: &cobre_sddp::CutPool,
    node_id: NodeId,
    pinned_state: &[f64],
) -> f64 {
    let comm = StubComm;
    let mut workspace_pool = setup
        .create_workspace_pool(&comm, 1, ActiveSolver::new)
        .expect("create_workspace_pool");
    let ws: &mut SolverWorkspace<ActiveSolver> = &mut workspace_pool.workspaces[0];
    let ctx = setup.stage_ctx();
    let training_ctx = setup.training_ctx();
    let theta_col = setup.stage_state().theta;
    let terminal_stage = setup.num_stages() - 1;

    ws.solver.reset_solver_state();
    ws.solver.load_model(template);
    patch_backward_opening_for_probe(
        ws,
        &ctx,
        &training_ctx,
        StageIdx(terminal_stage),
        pinned_state,
        &[],
    );

    let view = solve_stage_for_probe(ws, &ctx, pool, None, StageIdx(terminal_stage), 0, node_id)
        .expect("terminal stage solve must not error");
    view.primal[theta_col]
}

/// Pin the state vector with `x` on [`post_study_ring_slot`] and `0.0` on
/// every other component (the dormant padding slot carries a zero boundary
/// coefficient, so its pinned value never reaches `θ`).
fn ring_pin(setup: &cobre_sddp::StudySetup, x: f64) -> Vec<f64> {
    let mut pin = vec![0.0_f64; setup.stage_state().n_state];
    pin[post_study_ring_slot(setup)] = x;
    pin
}

/// Every leaf `NodePos` in `graph` — a node with no successors.
fn leaf_positions(graph: &NodeGraph) -> Vec<NodePos> {
    graph
        .nodes
        .iter_indexed()
        .filter(|&(pos, _)| graph.successors[pos].is_empty())
        .map(|(pos, _)| pos)
        .collect()
}

// ── θ prices the carried committed MW by β·x, and β propagates inward ──

/// The terminal value responds to the outstanding committed MW on the
/// post-study-targeted ring slot by exactly `β·Δx`: with the single injected
/// cut binding (`α > 0`), `θ = α + β·x` on that slot, so `θ(x1) − θ(x0) =
/// β·(x1 − x0)` — closed-form for the single-slot single-coefficient case.
/// Repeating the solve at the same pin reproduces `θ` bit-for-bit
/// (run-to-run reproducibility).
#[test]
fn boundary_cut_prices_the_committed_mw_by_beta_times_x() {
    let mut setup = build_setup_in_code(build_system(true, false), &config());
    assert_eq!(
        setup.fcf.state_dimension, 2,
        "fixture must be a two-slot ring (the post-study-targeted slot and the dormant padding slot)"
    );

    let tmp = tempfile::tempdir().expect("tempdir");
    inject_ring_boundary(&mut setup, &tmp.path().join("boundary"));

    let terminal_pool_id = setup
        .inputs
        .node_graph
        .terminal_pool(setup.num_stages())
        .unwrap();
    let template = freeze_terminal_template(&setup, terminal_pool_id);
    let pool = &setup.fcf.pools[terminal_pool_id];

    let x0 = 0.0;
    let x1 = 2.0;
    let theta0 = terminal_theta(&setup, &template, pool, NodeId(1), &ring_pin(&setup, x0));
    let theta1 = terminal_theta(&setup, &template, pool, NodeId(1), &ring_pin(&setup, x1));

    assert!(
        theta0 > 0.0 && theta1 > 0.0,
        "both solves must land on the injected cut's binding regime (theta0={theta0}, \
         theta1={theta1})"
    );

    let expected_delta = BETA * (x1 - x0);
    let actual_delta = theta1 - theta0;
    assert!(
        (actual_delta - expected_delta).abs() < TOL,
        "theta must respond to the committed MW by beta*x: expected delta {expected_delta}, \
         got {actual_delta}"
    );
    // The closed form is exact: theta = alpha + beta*x on the priced slot.
    assert!(
        (theta0 - ALPHA).abs() < TOL && (theta1 - (ALPHA + BETA * x1)).abs() < TOL,
        "theta must equal alpha + beta*x: theta0={theta0} (alpha={ALPHA}), \
         theta1={theta1} (alpha+beta*x1={})",
        ALPHA + BETA * x1
    );

    let theta1_again = terminal_theta(&setup, &template, pool, NodeId(1), &ring_pin(&setup, x1));
    assert_eq!(
        theta1.to_bits(),
        theta1_again.to_bits(),
        "the priced terminal value must reproduce bit-for-bit across a repeated solve"
    );
}

/// The backward pass propagates the boundary coefficient inward: with the
/// ring-only boundary injected into the terminal pool, one training iteration
/// leaves the deciding stage's generated cut carrying a nonzero coefficient on
/// the post-study-targeted slot. Because the terminal row is a pure carry
/// (`out = in`) with unit `col_scale` and unit `cost_scale`, the propagated
/// coefficient is exactly `β` (analytical).
#[test]
fn backward_pass_propagates_boundary_coefficient_to_the_deciding_stage() {
    let mut setup = build_setup_in_code(build_system(true, false), &config());
    // The post-study target decides at stage 0; pool 0 is the FCF the
    // deciding stage carries, populated by the backward pass solving the
    // terminal. The stage-0 anticipated-decision column is active (nonzero
    // fuel objective) — the decider's own sanity check, since the range
    // itself is dense (one column per anticipated plant at EVERY stage).
    let deciding_pool = 0;
    let decision_col = setup.stage_ctx().geometry_per_stage[deciding_pool]
        .anticipated_decision
        .start;
    assert!(
        setup.stage_ctx().templates[deciding_pool].objective[decision_col].abs() > TOL,
        "the post-study target must decide at stage 0 (a priced, active decision column)"
    );
    let slot = post_study_ring_slot(&setup);

    let tmp = tempfile::tempdir().expect("tempdir");
    inject_ring_boundary(&mut setup, &tmp.path().join("boundary"));

    let comm = StubComm;
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new");
    let outcome = setup
        .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
        .expect("train must not return Err");
    assert!(
        outcome.error.is_none(),
        "training error must be None; got {:?}",
        outcome.error
    );

    let (_slot, _intercept, coefficients) = setup
        .fcf
        .active_cuts(deciding_pool)
        .next()
        .expect("the deciding stage's pool must carry a generated cut after training");
    let slot_coeff = coefficients[slot];
    assert!(
        slot_coeff.abs() > TOL,
        "the deciding stage's cut must carry a nonzero coefficient on the post-study-targeted \
         slot; got {slot_coeff}"
    );
    assert!(
        (slot_coeff - BETA).abs() < TOL,
        "the carried terminal boundary must propagate the coefficient unchanged (out = in, \
         unit scale): expected {BETA}, got {slot_coeff}"
    );
}

// ── Fuel-exclusive composition: β prices the state, the decision books fuel ──

/// The terminal `β·state` valuation and the delivery-anchored fuel
/// booking live on DIFFERENT LP columns, so they compose additively with no
/// double-count: at the deciding stage the decision column carries the fuel in
/// its objective while the ring slot's own `commit_out` column carries no
/// objective at all (β reaches it through the boundary cut ROW, never the
/// objective).
#[test]
fn terminal_valuation_and_decision_fuel_are_disjoint_columns() {
    let setup = build_setup_in_code(build_system(true, false), &config());
    let deciding_stage = 0;

    let decision_col = setup.stage_ctx().geometry_per_stage[deciding_stage]
        .anticipated_decision
        .start;

    let slot = post_study_ring_slot(&setup);
    let slot_out_col = setup
        .stage_state()
        .state_to_lp_column(StateDim::new(slot))
        .get();

    assert_ne!(
        decision_col, slot_out_col,
        "the fuel-bearing decision column and the beta-priced ring-slot column must be distinct"
    );

    let template = &setup.stage_ctx().templates[deciding_stage];
    assert!(
        template.objective[decision_col].abs() > TOL,
        "the decision column must book the delivery-anchored fuel (nonzero objective); got {}",
        template.objective[decision_col]
    );
    assert_eq!(
        template.objective[slot_out_col], 0.0,
        "the ring slot's commit_out column must carry no fuel term — beta prices it via the \
         cut row"
    );
}

// ── one shared boundary prices each fanned leaf by its own ring state ──

/// Extends the shared-boundary fan to the `commit_out`/`commit_in` ring:
/// both terminal fan leaves resolve to ONE shared pool carrying the single
/// injected boundary, and each leaf's terminal value is priced by ITS OWN
/// committed MW from that one shared `β` — `θ = α + β·x_leaf`, distinct
/// across leaves committing distinct MW.
#[test]
fn shared_boundary_prices_each_fanned_leaf_by_its_own_ring_state() {
    let mut setup = build_setup_in_code(build_system(true, true), &config());

    let leaves = leaf_positions(&setup.inputs.node_graph);
    assert_eq!(
        leaves.len(),
        2,
        "fixture must declare a 2-leaf terminal fan"
    );
    let leaf_pool_ids: Vec<usize> = leaves
        .iter()
        .map(|&pos| setup.inputs.node_graph.nodes[pos].pool_id)
        .collect();
    assert_eq!(
        leaf_pool_ids[0], leaf_pool_ids[1],
        "both terminal fan leaves must share ONE pool"
    );

    let tmp = tempfile::tempdir().expect("tempdir");
    inject_ring_boundary(&mut setup, &tmp.path().join("boundary"));

    let shared_pool_id = leaf_pool_ids[0];
    assert_eq!(
        shared_pool_id,
        setup
            .inputs
            .node_graph
            .terminal_pool(setup.num_stages())
            .unwrap(),
        "the shared leaf pool is the terminal pool the boundary overwrites"
    );
    let pool = &setup.fcf.pools[shared_pool_id];
    let injected: Vec<(f64, Vec<f64>)> = pool
        .active_cuts()
        .map(|(_, intercept, coeffs)| (intercept, coeffs.to_vec()))
        .collect();
    let slot = post_study_ring_slot(&setup);
    let mut expected_coeffs = vec![0.0_f64; setup.fcf.state_dimension];
    expected_coeffs[slot] = BETA;
    assert_eq!(
        injected,
        vec![(ALPHA, expected_coeffs)],
        "the shared pool must carry exactly the one injected ring boundary"
    );

    let template = freeze_terminal_template(&setup, shared_pool_id);
    let x_leaf1 = 1.5;
    let x_leaf2 = 4.0;
    let theta_leaf1 = terminal_theta(
        &setup,
        &template,
        pool,
        NodeId(1),
        &ring_pin(&setup, x_leaf1),
    );
    let theta_leaf2 = terminal_theta(
        &setup,
        &template,
        pool,
        NodeId(2),
        &ring_pin(&setup, x_leaf2),
    );

    for (label, x, theta) in [
        ("leaf1", x_leaf1, theta_leaf1),
        ("leaf2", x_leaf2, theta_leaf2),
    ] {
        let expected = ALPHA + BETA * x;
        assert!(
            (theta - expected).abs() < TOL,
            "{label}: the shared beta must price this leaf by its OWN committed MW: \
             expected {expected}, got {theta}"
        );
    }
    assert!(
        (theta_leaf1 - theta_leaf2).abs() > TOL,
        "leaves committing different MW must be priced to different terminal values"
    );
}

// ── Inert/bootstrap: no boundary → θ bounded at zero, ring columns present ──

/// Without an injected boundary the terminal carries no future cost, so `θ` is
/// bounded at zero regardless of the committed MW — yet the ring columns are
/// structurally present (kept open at every stage, per the keep-live
/// contract), ready to be priced once a boundary is injected.
#[test]
fn no_boundary_leaves_theta_zero_with_ring_columns_present() {
    let setup = build_setup_in_code(build_system(true, false), &config());

    assert_eq!(
        setup.fcf.state_dimension, 2,
        "the post-study-targeted ring slot must be present in the state dimension"
    );

    let terminal_pool_id = setup
        .inputs
        .node_graph
        .terminal_pool(setup.num_stages())
        .unwrap();
    let pool = &setup.fcf.pools[terminal_pool_id];
    assert_eq!(pool.populated(), 0, "no boundary cut must be loaded");

    let template = freeze_terminal_template(&setup, terminal_pool_id);
    let theta = terminal_theta(&setup, &template, pool, NodeId(1), &ring_pin(&setup, 2.0));
    assert!(
        theta.abs() < 1e-9,
        "with no boundary cut, theta must be 0 regardless of the committed MW: got {theta}"
    );
}

// ── accessor: the study's fixed post-horizon (class-4) commitment windows ──

/// Second anticipated thermal, for the declaration-order-invariance fixture.
const THERMAL_ID_B: EntityId = EntityId(3);
/// Generation cap admitting an in-study (class-2) commitment seed.
const CAPABILITY_MAX_MW: f64 = 100.0;

/// Zero-default bounds parameterized only by `n_thermals`; the per-block
/// thermal cap is [`CAPABILITY_MAX_MW`] so an in-study class-2 seed is
/// admissible.
fn bounds_for(n_thermals: usize) -> ResolvedBounds {
    ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 0,
            n_thermals,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: 2,
            k_max: 1,
        },
        &BoundsDefaults {
            hydro: HydroStageBounds {
                min_storage_hm3: 0.0,
                max_storage_hm3: 0.0,
                filling_min_rate_m3s: 0.0,
                water_withdrawal_m3s: 0.0,
            },
            hydro_block: HydroBlockBounds::default(),
            thermal: ThermalStageBounds { cost_per_mwh: 1.0 },
            thermal_block: ThermalBlockBounds {
                min_generation_mw: 0.0,
                max_generation_mw: CAPABILITY_MAX_MW,
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
    )
}

/// Build an accessor-test system: each id in `thermal_ids` (given declaration
/// order — the builder canonicalizes) is a `LeadStages(1)` anticipated thermal
/// on the shared bus; `commitments` seeds `past_anticipated_commitments`.
/// Independent of the ring-pricing fixtures above (its own thermal count and
/// bounds).
fn build_accessor_system(
    thermal_ids: &[EntityId],
    commitments: Vec<AnticipatedCommitmentHistory>,
) -> System {
    let bus = make_bus(BUS_ID, BusSpec::default());
    let thermals: Vec<_> = thermal_ids
        .iter()
        .map(|&id| {
            make_thermal(
                id,
                ThermalSpec {
                    bus_id: BUS_ID,
                    cost_per_mwh: 1.0,
                    min_generation_mw: 0.0,
                    max_generation_mw: CAPABILITY_MAX_MW,
                    anticipated_config: Some(AnticipatedConfig::LeadStages(1)),
                    ..Default::default()
                },
            )
        })
        .collect();
    let initial_conditions = InitialConditions {
        past_anticipated_commitments: commitments,
        ..Default::default()
    };
    SystemBuilder::new()
        .buses(vec![bus])
        .thermals(thermals)
        .stages(stages())
        .bounds(bounds_for(thermal_ids.len()))
        .penalties(penalties())
        .initial_conditions(initial_conditions)
        .build()
        .expect("accessor fixture System must build")
}

/// An in-study (class-2) commitment window covering stage 0 for `thermal_id`.
fn class_two_window(thermal_id: EntityId, value_mw: f64) -> AnticipatedCommitmentHistory {
    AnticipatedCommitmentHistory {
        thermal_id,
        start_date: study_start(),
        end_date: study_start() + chrono::TimeDelta::days(31),
        value_mw,
    }
}

/// A post-horizon (class-4) commitment window starting at the horizon end for
/// `thermal_id`.
fn class_four_window(thermal_id: EntityId, value_mw: f64) -> AnticipatedCommitmentHistory {
    AnticipatedCommitmentHistory {
        thermal_id,
        start_date: study_end(),
        end_date: study_end() + chrono::TimeDelta::days(30),
        value_mw,
    }
}

/// An anticipated thermal declaring both an in-study (class-2) and a
/// post-horizon (class-4) commitment window: the accessor returns only the
/// class-4 window (`start_date >= horizon_end`), omitting the class-2 one.
#[test]
fn terminal_fixed_post_horizon_windows_selects_only_class_four() {
    let class_two = class_two_window(THERMAL_ID, 40.0);
    let class_four = class_four_window(THERMAL_ID, 55.0);
    let commitments = vec![class_two.clone(), class_four.clone()];

    let setup = build_setup_in_code(
        build_accessor_system(&[THERMAL_ID], commitments.clone()),
        &config(),
    );
    let system = build_accessor_system(&[THERMAL_ID], commitments);

    let windows = setup.build_terminal_fixed_post_horizon_windows(&system);

    assert_eq!(
        windows,
        vec![class_four],
        "only the post-horizon (class-4) window is returned"
    );
    assert!(
        !windows.contains(&class_two),
        "the in-study (class-2) window must be omitted"
    );
}

/// Two systems differing only in the declaration order of their anticipated
/// thermals return identical class-4 window lists (declaration-order
/// invariance): the accessor iterates `thermals()`, which the builder sorts
/// canonically, so the emitted order tracks canonical plant order, never the
/// declaration order.
#[test]
fn terminal_fixed_post_horizon_windows_declaration_order_invariant() {
    let commitments = vec![
        class_four_window(THERMAL_ID, 55.0),
        class_four_window(THERMAL_ID_B, 70.0),
    ];

    let setup_ab = build_setup_in_code(
        build_accessor_system(&[THERMAL_ID, THERMAL_ID_B], commitments.clone()),
        &config(),
    );
    let system_ab = build_accessor_system(&[THERMAL_ID, THERMAL_ID_B], commitments.clone());
    let windows_ab = setup_ab.build_terminal_fixed_post_horizon_windows(&system_ab);

    let setup_ba = build_setup_in_code(
        build_accessor_system(&[THERMAL_ID_B, THERMAL_ID], commitments.clone()),
        &config(),
    );
    let system_ba = build_accessor_system(&[THERMAL_ID_B, THERMAL_ID], commitments);
    let windows_ba = setup_ba.build_terminal_fixed_post_horizon_windows(&system_ba);

    assert_eq!(windows_ab, windows_ba, "declaration-order invariant");
    assert_eq!(
        windows_ab.len(),
        2,
        "both plants' class-4 windows are returned"
    );
    assert_eq!(
        windows_ab[0].thermal_id, THERMAL_ID,
        "canonical plant order: thermal 2 before thermal 3"
    );
    assert_eq!(windows_ab[1].thermal_id, THERMAL_ID_B);
}
