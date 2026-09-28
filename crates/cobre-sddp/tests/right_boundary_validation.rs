//! Right-boundary validation suite: consolidates the unified
//! hold-by-identity family's core guarantees, on minimal hermetic
//! in-code fixtures mirroring the pattern in `right_boundary_pricing.rs`,
//! `right_boundary_cost_semantics.rs`, and `right_boundary_output.rs`: zero
//! hydros, one bus, one anticipated thermal, two study stages matching
//! their real calendar span.
//!
//! - `inert_regression` — a study declaring no `post_study_stages` behaves as
//!   if the post-horizon machinery were absent (purely additive).
//! - `fcf_prices_committed_mw` — the reported lower bound responds to the
//!   outstanding committed MW by `beta*x`, in contrast to the frozen
//!   `beta*0` baseline.
//! - `left_right_symmetry` — a LEFT (`past_anticipated_commitments`, fished
//!   in-study) and a RIGHT (a post-study-targeted delivery,
//!   held-to-terminal + FCF-priced) commitment of the same magnitude
//!   transport that magnitude identically: value-conservation +
//!   mechanism-parity, never objective/`final_lb` bit-identity — the two
//!   directions value the commitment through different mechanisms
//!   (fuel-at-delivery vs FCF-at-terminal), so a bit-identity claim would
//!   contradict that design.

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

use chrono::{NaiveDate, TimeDelta};
use cobre_core::entities::thermal::AnticipatedConfig;
use cobre_core::temporal::StageStateConfig;
use cobre_core::{
    AnticipatedCommitmentHistory, BoundsCountsSpec, BoundsDefaults, ContractBlockBounds, EntityId,
    HydroBlockBounds, HydroStageBounds, InitialConditions, LineBlockBounds, PostStudyStage,
    PostStudyStages, PostStudyThermalBound, PumpingBlockBounds, ResolvedBounds, System,
    SystemBuilder, ThermalBlockBounds, ThermalStageBounds,
};
use cobre_sddp::indexer::CutStateProjection;
use cobre_sddp::setup::{NodeId, StageIdx};
use cobre_sddp::test_support::{
    patch_backward_opening_for_counterfactual_probe, solve_stage_for_probe,
};
use cobre_sddp::workspace::SolverWorkspace;
use cobre_sddp::{
    BoundaryLoadRequest, CutPool, StudySetup, build_cut_row_batch_into, inject_boundary_cuts,
    load_boundary_cuts,
};
use cobre_solver::{
    ActiveSolver, FreezeScratch, RowBatch, SolverInterface, StageTemplate,
    freeze_rows_into_template,
};

mod common;
use common::builders::{BusSpec, StageSpec, ThermalSpec, make_bus, make_stage, make_thermal};
use common::{StubComm, build_setup_in_code, run_simulation};

const BUS_ID: EntityId = EntityId(1);
const THERMAL_ID: EntityId = EntityId(2);
/// RIGHT-boundary `LeadTime` delta (hours), ~62.5 days: the thermal's own
/// stage-1 delivery decides PRE-STUDY (`None` — dormant, no genuine
/// decision), while the post-study target resolves to a stage-0 decider —
/// the pricing fixture's calendar (`right_boundary_pricing.rs`).
const DELTA_HOURS: f64 = 1500.0;
/// Boundary cut intercept, positive so an injected cut binds `theta` above
/// its zero floor for every committed MW `x >= 0`.
const ALPHA: f64 = 5.0;
/// Shared capability envelope: the post-study destination's max MW and the
/// LEFT fixture's thermal capability — the same magnitude on both boundaries.
const CAPABILITY_MAX_MW: f64 = 100.0;
/// Post-study fuel cost `$/MWh`, booked on the RIGHT decision column,
/// disjoint from the terminal `beta*state` valuation.
const COST_PER_MWH: f64 = 37.5;
/// Post-study stage duration \[h\]; the study carries a `0.0` discount rate.
const POST_STUDY_HOURS: f64 = 720.0;
const TOL: f64 = 1e-6;

fn study_start() -> NaiveDate {
    NaiveDate::from_ymd_opt(2024, 1, 1).expect("valid date")
}

/// Stage 0's end date — also the LEFT fixture's leading-delivery-stage
/// commitment window's end.
fn stage0_end() -> NaiveDate {
    study_start() + TimeDelta::days(31)
}

/// Study end date (stage 1's end) — the RIGHT fixture's post-horizon window
/// and post-study stage both anchor here.
fn study_end() -> NaiveDate {
    stage0_end() + TimeDelta::days(30)
}

/// Two stages matching their real calendar span exactly: stage 0 January
/// (744h), stage 1 the following 30 days (720h). Shared by every fixture in
/// this suite — the LEFT/RIGHT "calendar reflection" is the same horizon,
/// read from its two opposite edges.
fn stages() -> Vec<cobre_core::temporal::Stage> {
    let start = study_start();
    let s0_end = stage0_end();
    vec![
        make_stage(
            0,
            StageSpec {
                start_date: start,
                end_date: s0_end,
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
                start_date: s0_end,
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

/// The RIGHT fixture's declared post-study stage: `[study_end(), study_end()
/// + 30 days)`, pinned to `v` (`min_mw == max_mw`) — the ring's post-study
/// bound comes SOLELY from `post_study_stages.json`'s own table.
fn post_study_stages(v: f64) -> PostStudyStages {
    PostStudyStages {
        stages: vec![PostStudyStage {
            start_date: study_end(),
            duration_hours: POST_STUDY_HOURS,
        }],
        thermal_bounds: vec![PostStudyThermalBound {
            thermal_id: THERMAL_ID,
            post_study_stage_index: 0,
            cost_per_mwh: COST_PER_MWH,
            min_mw: v,
            max_mw: v,
        }],
    }
}

/// Shared zero-default bounds, parameterized only by the per-block thermal
/// generation cap: `0.0` for the RIGHT fixture (its in-study self-delivery
/// needs no in-study generation) and [`CAPABILITY_MAX_MW`] for the LEFT
/// fixture (its fished generation must reach the committed value).
fn bounds(thermal_block_max_mw: f64) -> ResolvedBounds {
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
                max_generation_mw: thermal_block_max_mw,
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

/// RIGHT boundary: `LeadTime` thermal, `with_window` toggles the declared
/// `post_study_stages` destination pinning the post-study-targeted delivery
/// to `v`; `v` is read only when `with_window` is `true`.
fn build_system_right(with_window: bool, v: f64) -> System {
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
        .bounds(bounds(0.0))
        .penalties(penalties());
    if with_window {
        builder = builder.post_study_stages(Some(post_study_stages(v)));
    }
    builder.build().expect("fixture System must build")
}

/// LEFT boundary: the calendar reflection of [`build_system_right`] — a
/// `LeadStages(1)` thermal whose ONE leading delivery stage (stage 0) is
/// seeded via `past_anticipated_commitments` to `v`, instead of a
/// post-study-targeted delivery past the horizon.
fn build_system_left(v: f64) -> System {
    let bus = make_bus(BUS_ID, BusSpec::default());
    let thermal = make_thermal(
        THERMAL_ID,
        ThermalSpec {
            bus_id: BUS_ID,
            cost_per_mwh: 1.0,
            min_generation_mw: 0.0,
            max_generation_mw: CAPABILITY_MAX_MW,
            anticipated_config: Some(AnticipatedConfig::LeadStages(1)),
            ..Default::default()
        },
    );
    let initial_conditions = InitialConditions {
        past_anticipated_commitments: vec![AnticipatedCommitmentHistory {
            thermal_id: THERMAL_ID,
            start_date: study_start(),
            end_date: stage0_end(),
            value_mw: v,
        }],
        ..Default::default()
    };

    SystemBuilder::new()
        .buses(vec![bus])
        .thermals(vec![thermal])
        .stages(stages())
        .bounds(bounds(CAPABILITY_MAX_MW))
        .penalties(penalties())
        .initial_conditions(initial_conditions)
        .build()
        .expect("fixture System must build")
}

fn config() -> cobre_io::config::Config {
    use cobre_io::config::{
        Config, EstimationConfig, ExportsConfig, InflowNonNegativityConfig,
        InflowNonNegativityMethod, ModelingConfig, ParallelismConfig, PolicyConfig,
        RowSelectionConfig, SimulationConfig, SimulationSelection, StoppingMode,
        StoppingRuleConfig, TrainingConfig, TrainingSelection, TrainingSolverConfig,
        UpperBoundEvaluationConfig,
    };
    Config {
        schema: None,
        modeling: ModelingConfig {
            inflow_non_negativity: InflowNonNegativityConfig {
                method: InflowNonNegativityMethod::Penalty,
            },
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
        simulation: SimulationConfig {
            enabled: true,
            io_channel_capacity: 16,
            selection: Some(SimulationSelection::Sampled { num_scenarios: 1 }),
            ..SimulationConfig::default()
        },
        exports: ExportsConfig::default(),
        estimation: EstimationConfig::default(),
    }
}

/// State-vector index of the ring slot carrying the post-study-targeted
/// delivery: mirrors `StateSpace::commitment_hold_in_study_offset`
/// byte-for-byte (`pub(crate)`, unreachable from this integration-test
/// binary), never a hand-rolled `commit_out.end`-relative guess.
fn post_study_ring_slot(setup: &StudySetup) -> usize {
    let state = setup.stage_state();
    let m = setup.num_stages();
    state.commit_out.start + (m % state.k_max) * state.n_anticipated
}

/// Pool `pool`'s fixture `priced_state_date`: `2030-01-01` plus `pool`
/// months.
fn fixture_priced_date(pool: u32) -> NaiveDate {
    cobre_sddp::test_support::fixture_priced_date(cobre_sddp::test_support::ymd(2030, 1, 1), pool)
}

/// Load a boundary carrying `beta` on [`post_study_ring_slot`] alone, zero
/// elsewhere, intercept [`ALPHA`], and inject it into `setup`'s terminal
/// pool.
fn inject_ring_boundary(setup: &mut StudySetup, dir: &Path, beta: f64) {
    let state_dimension = setup.fcf.state_dimension as u32;
    let slot = post_study_ring_slot(setup);
    let mut coefficients = vec![0.0_f64; state_dimension as usize];
    coefficients[slot] = beta;
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
/// freezes a pool once per iteration.
fn freeze_terminal_template(setup: &StudySetup, pool_id: usize) -> StageTemplate {
    let state = setup.stage_state();
    let terminal_stage = setup.num_stages() - 1;
    let ctx = setup.stage_ctx();
    let base = &ctx.templates[terminal_stage];

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
    build_cut_row_batch_into(
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

/// Solve the terminal stage's frozen `template` against `pool` with the
/// state vector pinned to `pinned_state`, returning `theta`'s primal value.
fn terminal_theta(
    setup: &StudySetup,
    template: &StageTemplate,
    pool: &CutPool,
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
    patch_backward_opening_for_counterfactual_probe(
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
/// every other component.
fn ring_pin(setup: &StudySetup, x: f64) -> Vec<f64> {
    let mut pin = vec![0.0_f64; setup.stage_state().n_state];
    pin[post_study_ring_slot(setup)] = x;
    pin
}

// ── A dormant post-horizon carrier is structurally and numerically inert ──

mod inert_regression {
    //! A study declaring no `post_study_stages` behaves as if the post-horizon
    //! machinery were absent. The pre-declaration OUTPUT golden is delegated to
    //! the tier-1 parity golden `parity_hash_d34` (both backends) and the shift-to-hold
    //! byte-stability verdict `k1_byte_stability_verdict` — this module pins
    //! only that the dormant machinery introduces no nondeterminism of its
    //! own, never a new golden.

    use super::*;

    /// `pool_id`'s active cuts, as `to_bits()` so two independent runs
    /// compare exactly rather than by float tolerance.
    fn active_cut_bits(setup: &StudySetup, pool_id: usize) -> Vec<(u64, Vec<u64>)> {
        setup
            .fcf
            .active_cuts(pool_id)
            .map(|(_, intercept, coeffs)| {
                (
                    intercept.to_bits(),
                    coeffs.iter().map(|c| c.to_bits()).collect(),
                )
            })
            .collect()
    }

    #[test]
    fn structural_inertness_when_no_window_declared() {
        let setup = build_setup_in_code(build_system_right(false, 0.0), &config());
        let state = setup.stage_state();
        assert_eq!(
            state.commit_out.len(),
            state.n_anticipated * state.k_max,
            "without a declared window, commit_out must be exactly the in-study ring's width \
             (no post-horizon lane appended)"
        );
    }

    #[test]
    fn dormant_machinery_trains_bit_reproducibly() {
        let deciding_pool = 0;
        let comm = StubComm;

        let mut setup_a = build_setup_in_code(build_system_right(false, 0.0), &config());
        let mut solver_a = ActiveSolver::new().expect("ActiveSolver::new");
        let outcome_a = setup_a
            .train(&mut solver_a, &comm, 1, ActiveSolver::new, None, None)
            .expect("train must not return Err");
        assert!(
            outcome_a.error.is_none(),
            "training error: {:?}",
            outcome_a.error
        );

        let mut setup_b = build_setup_in_code(build_system_right(false, 0.0), &config());
        let mut solver_b = ActiveSolver::new().expect("ActiveSolver::new");
        let outcome_b = setup_b
            .train(&mut solver_b, &comm, 1, ActiveSolver::new, None, None)
            .expect("train must not return Err");
        assert!(
            outcome_b.error.is_none(),
            "training error: {:?}",
            outcome_b.error
        );

        assert_eq!(
            outcome_a.result.final_lb.to_bits(),
            outcome_b.result.final_lb.to_bits(),
            "final_lb must reproduce bit-for-bit across two independent runs of the identical \
             dormant fixture"
        );
        assert_eq!(
            active_cut_bits(&setup_a, deciding_pool),
            active_cut_bits(&setup_b, deciding_pool),
            "every active cut's coefficient slice must reproduce bit-for-bit across two \
             independent runs of the identical dormant fixture"
        );
    }
}

// ── The FCF prices the outstanding committed MW at the bound ──

mod fcf_prices_committed_mw {
    use super::*;

    /// Pinned committed MW, inside the post-study capability
    /// `[0, CAPABILITY_MAX_MW]`.
    const V: f64 = 20.0;
    /// The nonzero boundary coefficient contrasted against the frozen
    /// `beta == 0.0` baseline.
    const BETA_B: f64 = 3.0;

    fn train_with_beta(beta: f64) -> f64 {
        let mut setup = build_setup_in_code(build_system_right(true, V), &config());
        let tmp = tempfile::tempdir().expect("tempdir");
        inject_ring_boundary(&mut setup, &tmp.path().join("boundary"), beta);

        let comm = StubComm;
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new");
        let outcome = setup
            .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
            .expect("train must not return Err");
        assert!(
            outcome.error.is_none(),
            "training error: {:?}",
            outcome.error
        );
        outcome.result.final_lb
    }

    #[test]
    fn final_lb_responds_to_the_committed_mw_by_beta_times_v() {
        let lb_at_zero = train_with_beta(0.0);
        let lb_at_b = train_with_beta(BETA_B);

        let tol = TOL * lb_at_b.abs().max(1.0);
        let expected_delta = BETA_B * V;
        let actual_delta = lb_at_b - lb_at_zero;
        assert!(
            (actual_delta - expected_delta).abs() < tol,
            "final_lb must respond to the committed MW by beta*V: expected delta \
             {expected_delta}, got {actual_delta}"
        );
    }

    #[test]
    fn final_lb_at_a_fixed_beta_reproduces_bit_for_bit() {
        let first = train_with_beta(BETA_B);
        let second = train_with_beta(BETA_B);
        assert_eq!(
            first.to_bits(),
            second.to_bits(),
            "final_lb must reproduce bit-for-bit across two independent trainings at the same \
             beta"
        );
    }
}

// ── Left/right symmetry — value-conservation + mechanism-parity ──

mod left_right_symmetry {
    //! "Left/right behaviour matches under reflection" is value-conservation
    //! and mechanism-parity — the committed magnitude `V` is transported
    //! without loss in BOTH directions — never objective/`final_lb`
    //! bit-identity, since the two directions value the commitment through
    //! different mechanisms (fuel-at-delivery vs FCF-at-terminal). Asserting
    //! objective bit-identity here would contradict that design; do not
    //! "simplify" this suite toward it.

    use super::*;

    /// The committed magnitude transported by both boundaries.
    const V: f64 = 25.0;
    /// The boundary coefficient pricing the RIGHT terminal lane.
    const BETA: f64 = 2.5;

    /// LEFT: a `past_anticipated_commitments` seed covering the plant's
    /// leading delivery stage (stage 0) fishes exactly the committed value —
    /// the always-fish contract pins delivery-stage generation to the
    /// matured commitment.
    #[test]
    fn left_boundary_fished_generation_equals_the_committed_value() {
        let mut setup = build_setup_in_code(build_system_left(V), &config());
        let results = run_simulation(&mut setup, 1);
        assert_eq!(results.len(), 1, "must produce exactly one scenario");

        let scenario = &results[0];
        let leading_stage = scenario
            .stages
            .iter()
            .find(|s| s.stage_id == 0)
            .expect("the plant's leading delivery stage must be present");
        let committed = leading_stage
            .thermals
            .iter()
            .find(|th| th.thermal_id == THERMAL_ID.0)
            .and_then(|th| th.anticipated_committed_mw)
            .expect("anticipated_committed_mw must be Some at the leading delivery stage");

        assert!(
            (committed - V).abs() < TOL,
            "the LEFT delivery-stage fished generation must equal the committed value V: \
             expected {V}, got {committed}"
        );
    }

    /// RIGHT: a post-study destination pinned to the SAME magnitude is
    /// deposited into the ring, carried to the terminal slot, and priced
    /// there by an injected boundary at exactly `beta*V` — the mirror
    /// mechanism at the right boundary.
    #[test]
    fn right_boundary_carries_and_prices_the_committed_value() {
        let setup_struct = build_setup_in_code(build_system_right(true, V), &config());
        let decision_col = setup_struct.stage_ctx().geometry_per_stage[0]
            .anticipated_decision
            .start;
        let decider_template = &setup_struct.stage_ctx().templates[0];
        assert_eq!(
            (
                decider_template.col_lower[decision_col],
                decider_template.col_upper[decision_col]
            ),
            (V, V),
            "the RIGHT decision column must pin the committed value V (the post-study table's \
             own [min_mw, max_mw] interval)"
        );

        let mut setup = setup_struct;
        let tmp = tempfile::tempdir().expect("tempdir");
        inject_ring_boundary(&mut setup, &tmp.path().join("boundary"), BETA);

        let terminal_pool_id = setup
            .inputs
            .node_graph
            .terminal_pool(setup.num_stages())
            .unwrap();
        let template = freeze_terminal_template(&setup, terminal_pool_id);
        let pool = &setup.fcf.pools[terminal_pool_id];
        let theta_at_v = terminal_theta(&setup, &template, pool, NodeId(1), &ring_pin(&setup, V));
        let theta_at_zero =
            terminal_theta(&setup, &template, pool, NodeId(1), &ring_pin(&setup, 0.0));

        let expected_delta = BETA * V;
        let actual_delta = theta_at_v - theta_at_zero;
        assert!(
            (actual_delta - expected_delta).abs() < TOL,
            "the RIGHT terminal theta must respond to the carried committed value by beta*V: \
             expected {expected_delta}, got {actual_delta}"
        );
    }
}
