//! End-to-end horizon-reduction regression: a full-horizon study trains and
//! checkpoints through the real producer, then a study truncated to a
//! shorter horizon — its removed stages replayed as a post-study calendar —
//! loads that checkpoint at its own boundary date and reconciles every
//! family with zero dropped source couplings.
//!
//! Both studies declare the same two-hydro travel-time cascade and the same
//! `LeadStages` anticipated thermal; the truncated study's `PostStudyStages`
//! reproduce the removed stages' calendar exactly, so every dated fan-out
//! join lands at full coverage and the load is a faithful reduction rather
//! than a differing-shape reconciliation.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::doc_markdown,
    clippy::too_many_lines
)]

use chrono::{Duration, NaiveDate};
use cobre_core::entities::hydro::HydroGenerationModel;
use cobre_core::entities::thermal::AnticipatedConfig;
use cobre_core::resolved::{
    BusStagePenalties, LineStagePenalties, NcsStagePenalties, PenaltiesCountsSpec,
    PenaltiesDefaults,
};
use cobre_core::scenario::InflowModel;
use cobre_core::temporal::{
    Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig, StageStateConfig,
};
use cobre_core::{
    BoundsCountsSpec, BoundsDefaults, ContractBlockBounds, EntityId, HorizonGraph,
    HydroBlockBounds, HydroPastDefluence, HydroPenalties, HydroStageBounds, HydroStorage,
    InitialConditions, LineBlockBounds, PostStudyStage, PostStudyStages, PostStudyThermalBound,
    PumpingBlockBounds, ResolvedBounds, ResolvedPenalties, SeasonCycleType, SeasonDefinition,
    SeasonMap, System, SystemBuilder, ThermalBlockBounds, ThermalStageBounds,
};
use cobre_io::config::{
    BoundaryPolicy, Config, EstimationConfig, ExportsConfig, InflowNonNegativityConfig,
    InflowNonNegativityMethod, ModelingConfig, ParallelismConfig, PolicyConfig, RowSelectionConfig,
    SimulationConfig, StoppingMode, StoppingRuleConfig, TrainingConfig, TrainingSelection,
    TrainingSolverConfig, UpperBoundEvaluationConfig,
};
use cobre_io::{SEASON_CYCLE_CODE_MONTHLY, encode_slot_date, read_policy_checkpoint};
use cobre_sddp::policy::orchestration::{
    CheckpointParams, build_season_manifest, write_checkpoint,
};
use cobre_sddp::{
    BoundaryLoadRequest, FamilyTally, ValidatedBoundaryCuts, load_boundary_cuts, study_horizon_end,
};
use cobre_solver::ActiveSolver;
use tempfile::TempDir;

mod common;
use common::StubComm;
use common::build_setup_in_code;
use common::builders::{
    BusSpec, HydroSpec, StageSpec, ThermalSpec, make_bus, make_hydro, make_stage, make_thermal,
};

const BUS_ID: EntityId = EntityId(1);
const DOWNSTREAM_ID: EntityId = EntityId(2);
const UPSTREAM_ID: EntityId = EntityId(3);
const THERMAL_ID: EntityId = EntityId(4);

const N_FULL_STAGES: usize = 4;
const N_TRUNCATED_STAGES: usize = 2;
const LEAD_STAGES: u32 = 2;
const TRAVEL_TIME_HOURS: f64 = 900.0;
const COST_SCALE_FACTOR: f64 = 1.0;
const N_ITERATIONS: u32 = 2;

/// The season-map coverage the fixture's `System` declares.
#[derive(Clone, Copy)]
enum SeasonCoverage {
    /// No season map declared — `build_season_manifest` returns the absent
    /// descriptor and the boundary-load season/PAR-identity gate is skipped.
    Absent,
    /// A twelve-season monthly cycle — the boundary-load gate runs.
    MonthlyCycle,
}

/// A twelve-season monthly `SeasonMap`, `id: i` / `month_start: i + 1`.
fn monthly_season_map() -> SeasonMap {
    let seasons = (0..12)
        .map(|id| SeasonDefinition {
            id,
            label: format!("Month{}", id + 1),
            month_start: u32::try_from(id + 1).expect("month fits u32"),
            day_start: None,
            month_end: None,
            day_end: None,
        })
        .collect();
    SeasonMap {
        cycle_type: SeasonCycleType::Monthly,
        seasons,
    }
}

/// The five calendar boundaries of the full study's four stages: `bounds[i]`
/// is stage `i`'s `start_date`, `bounds[i + 1]` its `end_date`.
fn calendar_bounds() -> [NaiveDate; 5] {
    [
        NaiveDate::from_ymd_opt(2024, 1, 1).expect("valid date"),
        NaiveDate::from_ymd_opt(2024, 2, 1).expect("valid date"),
        NaiveDate::from_ymd_opt(2024, 3, 1).expect("valid date"),
        NaiveDate::from_ymd_opt(2024, 4, 1).expect("valid date"),
        NaiveDate::from_ymd_opt(2024, 5, 1).expect("valid date"),
    ]
}

fn hydro_penalties() -> HydroPenalties {
    HydroPenalties {
        spillage_cost: 0.01,
        diversion_cost: 0.0,
        turbined_cost: 0.0,
        storage_violation_below_cost: 500.0,
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
        inflow_nonnegativity_cost: 1_000.0,
    }
}

/// Study stage `i`, dated from [`calendar_bounds`] — the single calendar
/// source both the full study's real stages and the truncated study's
/// `PostStudyStages` read, so the two can never drift apart.
fn build_stage(i: usize) -> Stage {
    let bounds = calendar_bounds();
    let start = bounds[i];
    let end = bounds[i + 1];
    let hours = (end - start).num_hours() as f64;
    make_stage(
        i,
        StageSpec {
            start_date: start,
            end_date: end,
            season_id: Some(i),
            blocks: vec![Block {
                index: 0,
                name: "S".to_string(),
                duration_hours: hours,
            }],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: true,
                inflow_lags: true,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        },
    )
}

/// The truncated study's post-study calendar: stages 2 and 3's own start
/// dates and durations, replayed exactly — the faithful-reduction condition
/// that makes every dated fan-out join land at full coverage.
fn truncated_post_study_stages() -> PostStudyStages {
    let bounds = calendar_bounds();
    let stage = |i: usize| PostStudyStage {
        start_date: bounds[i],
        duration_hours: (bounds[i + 1] - bounds[i]).num_hours() as f64,
    };
    PostStudyStages {
        stages: vec![stage(2), stage(3)],
        thermal_bounds: vec![
            PostStudyThermalBound {
                thermal_id: THERMAL_ID,
                post_study_stage_index: 0,
                cost_per_mwh: 37.5,
                min_mw: 0.0,
                max_mw: 100.0,
            },
            PostStudyThermalBound {
                thermal_id: THERMAL_ID,
                post_study_stage_index: 1,
                cost_per_mwh: 37.5,
                min_mw: 0.0,
                max_mw: 100.0,
            },
        ],
    }
}

fn inflow_models(n_stages: usize) -> Vec<InflowModel> {
    let mut models = Vec::with_capacity(2 * n_stages);
    for hydro_id in [DOWNSTREAM_ID, UPSTREAM_ID] {
        for i in 0..n_stages {
            models.push(InflowModel {
                hydro_id,
                stage_id: i32::try_from(i).expect("stage index fits i32"),
                mean_m3s: 80.0,
                std_m3s: 20.0,
                ar_coefficients: vec![0.3],
                residual_std_ratio: 1.0,
                annual: None,
            });
        }
    }
    models
}

fn bounds(n_stages: usize) -> ResolvedBounds {
    ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 2,
            n_thermals: 1,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages,
            k_max: LEAD_STAGES as usize,
        },
        &BoundsDefaults {
            hydro: HydroStageBounds {
                min_storage_hm3: 0.0,
                max_storage_hm3: 1_000.0,
                filling_min_rate_m3s: 0.0,
                water_withdrawal_m3s: 0.0,
            },
            hydro_block: HydroBlockBounds {
                max_turbined_m3s: 200.0,
                max_generation_mw: 500.0,
                ..Default::default()
            },
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

fn penalties(n_stages: usize) -> ResolvedPenalties {
    ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: 2,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages,
        },
        &PenaltiesDefaults {
            hydro: hydro_penalties(),
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    )
}

/// A two-hydro travel-time cascade plus a `LeadStages` anticipated thermal,
/// over `n_stages` real study stages and an optional post-study calendar.
fn build_system(
    n_stages: usize,
    post_study_stages: Option<PostStudyStages>,
    coverage: SeasonCoverage,
) -> System {
    let bus = make_bus(BUS_ID, BusSpec::default());

    let downstream = make_hydro(
        DOWNSTREAM_ID,
        HydroSpec {
            bus_id: BUS_ID,
            max_storage_hm3: 1_000.0,
            max_turbined_m3s: 200.0,
            max_generation_mw: 500.0,
            generation_model: HydroGenerationModel::ConstantProductivity,
            ..Default::default()
        },
    );
    let upstream = make_hydro(
        UPSTREAM_ID,
        HydroSpec {
            bus_id: BUS_ID,
            downstream_id: Some(DOWNSTREAM_ID),
            travel_time_hours: Some(TRAVEL_TIME_HOURS),
            min_outflow_m3s: 50.0,
            max_storage_hm3: 1_000.0,
            max_turbined_m3s: 200.0,
            max_generation_mw: 500.0,
            generation_model: HydroGenerationModel::ConstantProductivity,
            ..Default::default()
        },
    );
    let thermal = make_thermal(
        THERMAL_ID,
        ThermalSpec {
            bus_id: BUS_ID,
            cost_per_mwh: 1.0,
            min_generation_mw: 0.0,
            max_generation_mw: 0.0,
            anticipated_config: Some(AnticipatedConfig::LeadStages(LEAD_STAGES)),
            ..Default::default()
        },
    );

    let study_start = calendar_bounds()[0];
    let initial_conditions = InitialConditions {
        storage: vec![
            HydroStorage {
                hydro_id: DOWNSTREAM_ID,
                value_hm3: 100.0,
            },
            HydroStorage {
                hydro_id: UPSTREAM_ID,
                value_hm3: 100.0,
            },
        ],
        past_defluences: vec![HydroPastDefluence {
            hydro_id: UPSTREAM_ID,
            start_date: study_start - Duration::days(38),
            end_date: study_start,
            value_m3s: 50.0,
        }],
        ..Default::default()
    };

    let season_map = match coverage {
        SeasonCoverage::Absent => None,
        SeasonCoverage::MonthlyCycle => Some(monthly_season_map()),
    };

    SystemBuilder::new()
        .buses(vec![bus])
        .hydros(vec![downstream, upstream])
        .thermals(vec![thermal])
        .stages((0..n_stages).map(build_stage).collect())
        .inflow_models(inflow_models(n_stages))
        .bounds(bounds(n_stages))
        .penalties(penalties(n_stages))
        .initial_conditions(initial_conditions)
        .post_study_stages(post_study_stages)
        .policy_graph(HorizonGraph {
            season_map,
            ..HorizonGraph::default()
        })
        .build()
        .expect("horizon-reduction fixture: valid two-hydro cascade with a LeadStages thermal")
}

fn full_system(coverage: SeasonCoverage) -> System {
    build_system(N_FULL_STAGES, None, coverage)
}

fn truncated_system(coverage: SeasonCoverage) -> System {
    build_system(
        N_TRUNCATED_STAGES,
        Some(truncated_post_study_stages()),
        coverage,
    )
}

fn full_config() -> Config {
    Config {
        schema: None,
        modeling: ModelingConfig {
            inflow_non_negativity: InflowNonNegativityConfig {
                method: InflowNonNegativityMethod::Penalty,
            },
            cost_scale_factor: Some(COST_SCALE_FACTOR),
        },
        training: TrainingConfig {
            enabled: true,
            tree_seed: Some(42),
            stopping_rules: Some(vec![StoppingRuleConfig::IterationLimit {
                limit: N_ITERATIONS,
            }]),
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

/// The truncated study's config: identical to [`full_config`] except it
/// declares `policy.boundary`, which is what keeps its own terminal pool's
/// water-bucket and anticipated ring state live rather than horizon-capped —
/// the state a boundary injection then needs to reconcile against.
fn truncated_config() -> Config {
    let mut config = full_config();
    config.policy.boundary = Some(BoundaryPolicy {
        path: "unused-boundary-checkpoint".to_string(),
        strict: false,
    });
    config
}

/// One horizon-reduction load: the full study trained and checkpointed
/// through the real producer, then the truncated study's boundary load
/// against that checkpoint at its own `study_horizon_end`.
struct ReductionFixture {
    policy_dir: TempDir,
    validated: ValidatedBoundaryCuts,
    boundary_date: NaiveDate,
    full_cost_scale_factor: f64,
    truncated_cost_scale_factor: f64,
}

fn build_reduction_fixture(coverage: SeasonCoverage) -> ReductionFixture {
    let full_config = full_config();
    let mut full_setup = build_setup_in_code(full_system(coverage), &full_config);
    let comm = StubComm;
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new");
    let outcome = full_setup
        .train(
            &mut solver,
            &comm,
            N_ITERATIONS as usize,
            ActiveSolver::new,
            None,
            None,
        )
        .expect("training the full study must not return Err");
    assert!(
        outcome.error.is_none(),
        "full-study training error: {:?}",
        outcome.error
    );
    assert!(
        outcome.result.final_lb.is_finite(),
        "the full study's final lower bound must be finite, got {}",
        outcome.result.final_lb
    );

    // `System` is not `Clone`; rebuild the identical study for the checkpoint
    // writer, matching `deterministic.rs`'s `boundary_season_gate_round_trip`.
    let full_system_for_checkpoint = full_system(coverage);
    let policy_dir = tempfile::tempdir().expect("tempdir");
    let params = CheckpointParams {
        max_iterations: full_setup.loop_params.max_iterations,
        forward_passes: full_setup.loop_params.forward_passes,
        seed: full_setup.loop_params.seed,
        export_states: full_config.exports.states,
    };
    write_checkpoint(
        policy_dir.path(),
        &full_setup,
        &full_system_for_checkpoint,
        &outcome.result,
        &params,
    )
    .expect("write_checkpoint must succeed");

    let truncated_config = truncated_config();
    let truncated_setup = build_setup_in_code(truncated_system(coverage), &truncated_config);
    let truncated_system_for_load = truncated_system(coverage);

    let state_dim = truncated_setup.fcf.state_dimension as u32;
    let current_manifest =
        truncated_setup.build_terminal_entity_manifest(&truncated_system_for_load);
    let boundary_date = study_horizon_end(&truncated_system_for_load)
        .expect("the truncated study declares a non-negative stage");
    let study_seasons = build_season_manifest(&truncated_system_for_load);

    let validated = load_boundary_cuts(
        &BoundaryLoadRequest::new(
            policy_dir.path(),
            boundary_date,
            state_dim,
            &current_manifest,
            truncated_setup
                .inputs
                .stage_data
                .stage_templates
                .cost_scale_factor,
        )
        .with_inflow_lag_depth(truncated_setup.boundary_requirements().inflow_lag_depth())
        .with_study_seasons(&study_seasons),
    )
    .expect("a faithful horizon reduction must reconcile cleanly");

    ReductionFixture {
        policy_dir,
        validated,
        boundary_date,
        full_cost_scale_factor: full_setup
            .inputs
            .stage_data
            .stage_templates
            .cost_scale_factor,
        truncated_cost_scale_factor: truncated_setup
            .inputs
            .stage_data
            .stage_templates
            .cost_scale_factor,
    }
}

#[test]
fn horizon_reduction_reconciles_with_zero_dropped_couplings() {
    let fixture = build_reduction_fixture(SeasonCoverage::Absent);
    let report = fixture.validated.report();

    assert!(
        report.reconciled,
        "a faithful horizon reduction must reconcile per-slot: {report:?}"
    );
    assert!(
        report.dropped_source_slots.is_empty(),
        "expected zero dropped source slots on a faithful reduction, found: {:#?}",
        report.dropped_source_slots
    );
    assert!(
        report.straddling_slots.is_empty(),
        "expected zero straddling slots on a faithful reduction (every dated join is full \
         coverage), found: {:#?}",
        report.straddling_slots
    );

    for (label, tally) in [
        ("storage", report.storage),
        ("inflow_lag", report.inflow_lag),
        ("transit_bucket", report.transit_bucket),
        ("anticipated", report.anticipated),
        ("other_identity", report.other_identity),
    ] {
        assert_eq!(
            tally.dropped_source, 0,
            "family {label} must have zero dropped_source, found tally {tally:?}"
        );
    }
}

/// The boundary pool is interior (neither first nor last) and the loaded
/// record count equals its active cut count, so selection is by date rather
/// than by position. Each modeled family has non-zero `copy + fan_out` while
/// `other_identity` stays zero, proving the fixture didn't lose a family.
#[test]
fn horizon_reduction_selects_the_pool_priced_at_the_boundary_date() {
    let fixture = build_reduction_fixture(SeasonCoverage::Absent);
    let report = fixture.validated.report();

    for (label, tally) in [
        ("storage", report.storage),
        ("inflow_lag", report.inflow_lag),
        ("transit_bucket", report.transit_bucket),
        ("anticipated", report.anticipated),
    ] {
        assert!(
            tally.copy + tally.fan_out > 0,
            "family {label} must have a non-zero copy+fan_out total, found tally {tally:?}"
        );
    }
    let FamilyTally {
        copy,
        fan_out,
        straddling,
        default_zero,
        dropped_source,
    } = report.other_identity;
    assert_eq!(
        (copy, fan_out, straddling, default_zero, dropped_source),
        (0, 0, 0, 0, 0),
        "other_identity must be entirely zero (this fixture declares no entity outside the \
         four modeled families), found tally {:?}",
        report.other_identity
    );

    let checkpoint = read_policy_checkpoint(fixture.policy_dir.path())
        .expect("the checkpoint this fixture just wrote must read back");
    assert_eq!(
        checkpoint.stage_cuts.len(),
        N_FULL_STAGES,
        "the full study writes one pool per study stage"
    );

    let target = encode_slot_date(fixture.boundary_date);
    let matched: Vec<_> = checkpoint
        .stage_cuts
        .iter()
        .filter(|sr| sr.priced_state_date == target)
        .collect();
    assert_eq!(
        matched.len(),
        1,
        "exactly one pool must be priced at the boundary date {}, found: {:?}",
        fixture.boundary_date,
        matched.iter().map(|sr| sr.stage_id).collect::<Vec<_>>()
    );
    let pool = matched[0];

    let first_stage_id = checkpoint
        .stage_cuts
        .first()
        .expect("the checkpoint carries at least one pool")
        .stage_id;
    let last_stage_id = checkpoint
        .stage_cuts
        .last()
        .expect("the checkpoint carries at least one pool")
        .stage_id;
    assert_ne!(
        pool.stage_id, first_stage_id,
        "the pool priced at the boundary date must not be the first pool"
    );
    assert_ne!(
        pool.stage_id, last_stage_id,
        "the pool priced at the boundary date must not be the last pool"
    );

    let active_count = pool.cuts.iter().filter(|c| c.is_active).count();
    assert_eq!(
        fixture.validated.len(),
        active_count,
        "the loaded record count must equal the selected pool's own active cut count"
    );
}

/// With `fixed_windows` left at the default empty slice, the constant
/// intercept fold contributes nothing, so every reconciled record's
/// intercept must stay bit-identical to its source cut's intercept —
/// provided the two studies share one `cost_scale_factor`, asserted first.
#[test]
fn horizon_reduction_leaves_intercepts_untouched_by_an_empty_fold() {
    let fixture = build_reduction_fixture(SeasonCoverage::Absent);

    assert_eq!(
        fixture.full_cost_scale_factor.to_bits(),
        fixture.truncated_cost_scale_factor.to_bits(),
        "the full and truncated studies must resolve the same cost_scale_factor, or the \
         intercept bit-identity comparison below is meaningless: full {} vs truncated {}",
        fixture.full_cost_scale_factor,
        fixture.truncated_cost_scale_factor
    );

    let checkpoint = read_policy_checkpoint(fixture.policy_dir.path())
        .expect("the checkpoint this fixture just wrote must read back");
    let target = encode_slot_date(fixture.boundary_date);
    let pool = checkpoint
        .stage_cuts
        .iter()
        .find(|sr| sr.priced_state_date == target)
        .expect("the boundary pool must be present in the checkpoint just written");

    assert_eq!(
        fixture.validated.len(),
        pool.cuts.len(),
        "every populated source cut must reach the loaded record set (no cut is dropped before \
         rebind under the default, non-deactivating cut-selection config)"
    );
    for (loaded, source) in fixture.validated.iter().zip(pool.cuts.iter()) {
        assert_eq!(
            loaded.intercept.to_bits(),
            source.intercept.to_bits(),
            "an empty fixed_windows fold under a matching cost_scale_factor must leave every \
             intercept bit-identical: loaded {} vs source {}",
            loaded.intercept,
            source.intercept
        );
    }
}

/// The truncated study references only seasons 0 and 1, so its descriptor
/// holds `None` at seasons 2 and 3 for hydro 2 even though the source
/// checkpoint genuinely priced order 1 there — the season-gate relaxation to
/// unreferenced seasons. Zero dropped couplings across every family confirms
/// this loads correctly.
#[test]
fn horizon_reduction_under_a_real_season_map_reconciles_with_zero_dropped_couplings() {
    let truncated = truncated_system(SeasonCoverage::MonthlyCycle);
    let manifest = build_season_manifest(&truncated);
    assert_eq!(manifest.cycle_code, SEASON_CYCLE_CODE_MONTHLY);
    assert_eq!(manifest.n_seasons, 12);
    let hydro_2 = manifest
        .hydro_orders
        .iter()
        .find(|h| h.hydro_id == DOWNSTREAM_ID.0)
        .expect("the truncated study models hydro 2's inflow");
    assert_eq!(hydro_2.orders[0], Some(1));
    assert_eq!(hydro_2.orders[1], Some(1));
    assert_eq!(hydro_2.orders[2], None);
    assert_eq!(hydro_2.orders[3], None);

    let fixture = build_reduction_fixture(SeasonCoverage::MonthlyCycle);
    let checkpoint = read_policy_checkpoint(fixture.policy_dir.path())
        .expect("the checkpoint this fixture just wrote must read back");
    let source_hydro_2 = checkpoint
        .metadata
        .season_manifest
        .hydro_orders
        .iter()
        .find(|h| h.hydro_id == DOWNSTREAM_ID.0)
        .expect("the source checkpoint models hydro 2's inflow");
    assert_eq!(
        source_hydro_2.orders[2], 1,
        "the source (full) study genuinely priced order 1 at season 2"
    );
    assert_eq!(
        source_hydro_2.orders[3], 1,
        "the source (full) study genuinely priced order 1 at season 3"
    );

    let report = fixture.validated.report();
    assert!(
        report.reconciled,
        "a faithful horizon reduction must reconcile per-slot under a real season map: {report:?}"
    );
    assert!(
        report.dropped_source_slots.is_empty(),
        "expected zero dropped source slots under a real season map, found: {:#?}",
        report.dropped_source_slots
    );

    for (label, tally) in [
        ("storage", report.storage),
        ("inflow_lag", report.inflow_lag),
        ("transit_bucket", report.transit_bucket),
        ("anticipated", report.anticipated),
        ("other_identity", report.other_identity),
    ] {
        assert_eq!(
            tally.dropped_source, 0,
            "family {label} must have zero dropped_source under a real season map, found tally \
             {tally:?}"
        );
    }
}
