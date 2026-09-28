//! End-to-end NEWAVE-to-DECOMP boundary reconciliation regression. A
//! NEWAVE-shaped SOURCE checkpoint (one storage slot, twelve monthly
//! `HydroInflowLag` slots, two dated monthly `AnticipatedThermalState` slots,
//! no transit buckets; dimension 15) reconciles into a DECOMP-shaped CURRENT
//! terminal manifest (the same storage and twelve lags as `Copy` targets, an
//! hour-based anticipated ring `Blend`ing the source months, and two
//! `HydroTransitBucket` `Zero` targets; a larger dimension). The reconciled
//! cuts, injected into a runnable DECOMP-shaped study (a water-arc plus
//! anticipated-post-study cascade), then train to a finite, bit-reproducible
//! lower bound. Proves the relaxed `state_dimension` gate lets the
//! differing-dimension load reach per-slot reconciliation and run.

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

use std::collections::BTreeSet;

use chrono::{Duration, NaiveDate};
use cobre_core::entities::hydro::HydroGenerationModel;
use cobre_core::entities::thermal::AnticipatedConfig;
use cobre_core::scenario::InflowModel;
use cobre_core::temporal::{Block, Stage};
use cobre_core::{
    BoundsCountsSpec, BoundsDefaults, ContractBlockBounds, EntityId, HydroBlockBounds,
    HydroPastDefluence, HydroStageBounds, HydroStorage, InitialConditions, LineBlockBounds,
    PostStudyStage, PostStudyStages, PostStudyThermalBound, PumpingBlockBounds, ResolvedBounds,
    System, SystemBuilder, ThermalBlockBounds, ThermalStageBounds,
};
use cobre_io::config::{
    BoundaryPolicy, Config, EstimationConfig, ExportsConfig, InflowNonNegativityConfig,
    InflowNonNegativityMethod, ModelingConfig, ParallelismConfig, PolicyConfig, RowSelectionConfig,
    SimulationConfig, StoppingMode, StoppingRuleConfig, TrainingConfig, TrainingSelection,
    TrainingSolverConfig, UpperBoundEvaluationConfig,
};
use cobre_io::{
    ENTITY_SLOT_DATE_SENTINEL, EntitySlot, PolicyCutRecord, ProducerBlock, StageCutsPayload,
    StateFamily, encode_slot_date, write_policy_checkpoint,
};
use cobre_sddp::test_support::{
    anticipated_slot_at, chain_graph_manifest, inflow_lag_slot, storage_slot, transit_bucket_slot,
    ymd,
};
use cobre_sddp::{BoundaryLoadRequest, inject_boundary_cuts, load_boundary_cuts};
use cobre_solver::ActiveSolver;

mod common;
use common::StubComm;
use common::build_setup_in_code;
use common::builders::{
    BusSpec, HydroSpec, StageSpec, ThermalSpec, make_bus, make_hydro, make_stage, make_thermal,
};

// ── shared source-checkpoint helpers ────────────────────────────────────────

/// Pool `pool`'s fixture `priced_state_date`: `2026-04-01` plus `pool`
/// months — at or before the April/May 2026 span the NEWAVE source fixture
/// prices, so the boundary-date-driven anticipated predicate never zeroes
/// the fan-out coefficients `newave_source_reconciles_into_decomp_current`
/// asserts on.
fn fixture_priced_date(pool: u32) -> NaiveDate {
    cobre_sddp::test_support::fixture_priced_date(ymd(2026, 4, 1), pool)
}

fn producer_block() -> ProducerBlock {
    ProducerBlock {
        completed_iterations: 10,
        max_iterations: 50,
        forward_passes: 1,
        ..cobre_sddp::test_support::producer_block()
    }
}

/// Write a single-stage, single-cut source checkpoint whose one cut carries
/// `coefficients`, one per `manifest` slot in the same order, at rest under
/// `cost_scale_factor` (`None` = legacy).
fn write_source_checkpoint(
    dir: &std::path::Path,
    manifest: &[EntitySlot],
    coefficients: &[f64],
    cost_scale_factor: Option<f64>,
) {
    let state_dimension = u32::try_from(coefficients.len()).expect("small coefficient count");
    let cut = PolicyCutRecord {
        cut_id: 0,
        slot_index: 0,
        iteration: 0,
        forward_pass_index: 0,
        intercept: 1.0,
        coefficients,
        is_active: true,
    };
    let cuts = vec![cut];
    let payload = StageCutsPayload {
        stage_id: 0,
        state_dimension,
        capacity: 1,
        warm_start_count: 0,
        cuts: &cuts,
        active_cut_indices: &[0],
        populated_count: 1,
        entity_manifest: manifest,
        cost_scale_factor: cost_scale_factor.unwrap_or(1_000_000.0),
        node_id: 0,
        graph_stage_id: -1,
        priced_state_date: encode_slot_date(fixture_priced_date(0)),
    };
    let metadata = cobre_sddp::test_support::checkpoint_metadata(
        1,
        chain_graph_manifest(1),
        ProducerBlock {
            cost_scale_factor,
            ..producer_block()
        },
    );
    write_policy_checkpoint(dir, &[payload], &[], &metadata, &[]).expect("write checkpoint");
}

// ── reconcile: NEWAVE source -> hand-built DECOMP current manifest ──────────

/// NEWAVE source hydro id (storage + inflow lags).
const HYDRO_ID: i32 = 5;
/// NEWAVE source anticipated-thermal id.
const THERMAL_ID: i32 = 9;
/// DECOMP-only downstream hydro id carrying the transit buckets the source lacks.
const DOWNSTREAM_ID: i32 = 7;
/// The NEWAVE source's inflow-lag depth (12 monthly lags, subindex 1..=12).
const LAG_DEPTH: u32 = 12;

/// The 15-slot NEWAVE source manifest: 1 storage + 12 monthly inflow lags + 2
/// dated anticipated slots on consecutive months (April, May 2026), NO transit
/// buckets.
fn newave_source_manifest() -> Vec<EntitySlot> {
    let mut manifest = Vec::with_capacity(15);
    manifest.push(storage_slot(HYDRO_ID));
    for lag in 1..=LAG_DEPTH {
        manifest.push(inflow_lag_slot(HYDRO_ID, lag));
    }
    manifest.push(anticipated_slot_at(THERMAL_ID, 0, 20_260_401));
    manifest.push(anticipated_slot_at(THERMAL_ID, 1, 20_260_501));
    manifest
}

/// Distinguishable nonzero source coefficients aligned 1:1 with
/// [`newave_source_manifest`]: storage `100.0`, lag `k` -> `k`, April `300.0`,
/// May `400.0`.
fn newave_source_coefficients() -> Vec<f64> {
    let mut coeffs = Vec::with_capacity(15);
    coeffs.push(100.0);
    for lag in 1..=LAG_DEPTH {
        coeffs.push(f64::from(lag));
    }
    coeffs.push(300.0);
    coeffs.push(400.0);
    coeffs
}

/// The DECOMP-shaped current terminal manifest: same storage + 12 lags (`Copy`
/// targets), 2 hour-based anticipated ring slots dated to overlap the source
/// months (`Blend` targets), and 2 DATED transit-bucket slots the source has
/// no transit arc at all to overlap (`Zero` targets) — 17 slots, a larger
/// `state_dimension` than the source's 15.
fn decomp_current_manifest() -> Vec<EntitySlot> {
    let mut manifest = Vec::with_capacity(17);
    manifest.push(storage_slot(HYDRO_ID));
    for lag in 1..=LAG_DEPTH {
        manifest.push(inflow_lag_slot(HYDRO_ID, lag));
    }
    manifest.push(
        anticipated_slot_at(THERMAL_ID, 100, 20_260_401).with_interval(
            encode_slot_date(ymd(2026, 4, 1)),
            encode_slot_date(ymd(2026, 4, 8)),
        ),
    );
    manifest.push(
        anticipated_slot_at(THERMAL_ID, 101, 20_260_501).with_interval(
            encode_slot_date(ymd(2026, 5, 1)),
            encode_slot_date(ymd(2026, 5, 8)),
        ),
    );
    manifest.push(transit_bucket_slot(DOWNSTREAM_ID, 1).with_interval(
        encode_slot_date(ymd(2026, 4, 8)),
        encode_slot_date(ymd(2026, 4, 15)),
    ));
    manifest.push(transit_bucket_slot(DOWNSTREAM_ID, 2).with_interval(
        encode_slot_date(ymd(2026, 4, 15)),
        encode_slot_date(ymd(2026, 4, 22)),
    ));
    manifest
}

/// Given the 15-dim NEWAVE source and the 17-dim DECOMP current manifest, the
/// relaxed `state_dimension` gate lets the differing-dimension load reconcile:
/// storage + 12 lags `Copy` (13 total), the two ring slots `Blend` the source
/// months, the two transit slots resolve to `Zero`, and every returned record
/// projects to `current_state_dimension`.
#[test]
fn newave_source_reconciles_into_decomp_current() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let source_manifest = newave_source_manifest();
    let source_coefficients = newave_source_coefficients();
    assert_eq!(source_manifest.len(), 15, "NEWAVE source state dimension");
    write_source_checkpoint(
        tmp.path(),
        &source_manifest,
        &source_coefficients,
        Some(RUN_LOADING_FACTOR),
    );

    let current = decomp_current_manifest();
    let current_state_dimension = u32::try_from(current.len()).expect("small dimension");
    assert_eq!(
        current_state_dimension, 17,
        "DECOMP current state dimension"
    );

    let cuts = load_boundary_cuts(&BoundaryLoadRequest::new(
        tmp.path(),
        fixture_priced_date(0),
        current_state_dimension,
        &current,
        RUN_LOADING_FACTOR,
    ))
    .expect("the differing-dimension NEWAVE source must reconcile into the DECOMP current");

    for record in cuts.iter() {
        assert_eq!(
            record.coefficients.len(),
            current_state_dimension as usize,
            "every reconciled record projects to the target state dimension"
        );
    }

    let report = cuts.report();
    assert!(
        report.reconciled,
        "a verifiable manifest reconciles per-slot"
    );
    assert_eq!(report.storage.copy, 1, "the one storage slot copies");
    assert_eq!(report.inflow_lag.copy, 12, "all 12 lag slots copy");
    assert_eq!(
        report.tally_totals().0,
        13,
        "copy == 13 (1 storage + 12 lags)"
    );
    assert_eq!(
        report.anticipated.fan_out, 2,
        "both source months blend onto their overlapping ring targets"
    );
    assert_eq!(
        report.anticipated.straddling, 0,
        "each week is fully covered by its month, so Blend not Renormalize"
    );
    assert_eq!(
        report.anticipated_coverage.source_interval_count, 2,
        "two live dated source anticipated delivery intervals"
    );
    assert_eq!(
        report.transit_bucket.default_zero, 2,
        "the two transit-bucket targets the source lacks default to Zero"
    );

    for record in cuts.iter() {
        assert_eq!(
            record.coefficients[15], 0.0,
            "transit-bucket target slot 1 carries 0.0 in every reconciled vector"
        );
        assert_eq!(
            record.coefficients[16], 0.0,
            "transit-bucket target slot 2 carries 0.0 in every reconciled vector"
        );
    }

    let record = &cuts[0];
    assert_eq!(record.coefficients[0], 100.0, "storage copies verbatim");
    for lag in 1..=LAG_DEPTH as usize {
        assert_eq!(
            record.coefficients[lag], lag as f64,
            "inflow lag {lag} copies verbatim"
        );
    }
    let expected_april = 300.0 * (168.0 / 720.0);
    assert!(
        (record.coefficients[13] - expected_april).abs() < expected_april.abs() * 1e-9,
        "April ring slot blends to pi_M * overlap/H_M = {expected_april}, got {}",
        record.coefficients[13]
    );
    let expected_may = 400.0 * (168.0 / 744.0);
    assert!(
        (record.coefficients[14] - expected_may).abs() < expected_may.abs() * 1e-9,
        "May ring slot blends to pi_M * overlap/H_M = {expected_may}, got {}",
        record.coefficients[14]
    );
    assert_ne!(
        record.coefficients[13], 0.0,
        "the anticipated fan-out contributes a nonzero coefficient"
    );
}

// ── run: reconciled cuts injected into a runnable DECOMP-shaped study ───────
//
// The runnable DECOMP-shaped study is a two-hydro cascade with a water
// travel-time arc (2 transit buckets) and a `LeadTime` anticipated thermal
// delivering onto a declared post-study stage — the coexistence fixture from
// `hydro_sim.rs`'s `water_arc_and_post_study_anticipated_coexist_on_extended_layout`,
// reproduced here as local helpers.

const RUN_BUS_ID: EntityId = EntityId(1);
const RUN_THERMAL_ID: EntityId = EntityId(2);
const RUN_DOWNSTREAM_ID: EntityId = EntityId(3);
const RUN_UPSTREAM_ID: EntityId = EntityId(4);
const RUN_N_STAGES: usize = 2;
/// Bounded iteration count keeping the run in the default (non-`slow-tests`)
/// tier: a two-stage single-block study converges in milliseconds.
const RUN_MAX_ITERATIONS: u32 = 3;
/// Travel time (hours) producing exactly two water transit buckets.
const RUN_TRAVEL_TIME_HOURS: f64 = 900.0;
/// `LeadTime` delta (hours): the thermal's post-study target resolves to a
/// stage-0 decider, ring-sizing `k_max == 2`.
const RUN_DELTA_HOURS: f64 = 1_500.0;
const RUN_POST_STUDY_HOURS: f64 = 720.0;
/// The neutral loading factor: paired with a same-scale-marked source so
/// `rescale_cut_records_for_load` is a no-op — the study's own cost scale
/// (`config().modeling.cost_scale_factor` below) and the boundary loads in
/// both the reconcile and the run tests all share it.
const RUN_LOADING_FACTOR: f64 = 1.0;

fn run_study_start() -> NaiveDate {
    NaiveDate::from_ymd_opt(2024, 1, 1).expect("valid date")
}

fn run_study_end() -> NaiveDate {
    run_study_start() + Duration::days(61)
}

fn run_stages() -> Vec<Stage> {
    let start = run_study_start();
    let stage0_end = start + Duration::days(31);
    vec![
        make_stage(
            0,
            StageSpec {
                start_date: start,
                end_date: stage0_end,
                blocks: vec![Block {
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
                end_date: run_study_end(),
                blocks: vec![Block {
                    index: 0,
                    name: "S1".to_string(),
                    duration_hours: 720.0,
                }],
                ..Default::default()
            },
        ),
    ]
}

fn run_post_study_stages() -> PostStudyStages {
    PostStudyStages {
        stages: vec![PostStudyStage {
            start_date: run_study_end(),
            duration_hours: RUN_POST_STUDY_HOURS,
        }],
        thermal_bounds: vec![PostStudyThermalBound {
            thermal_id: RUN_THERMAL_ID,
            post_study_stage_index: 0,
            cost_per_mwh: 37.5,
            min_mw: 0.0,
            max_mw: 100.0,
        }],
    }
}

fn run_bounds() -> ResolvedBounds {
    ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 2,
            n_thermals: 1,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: RUN_N_STAGES,
            k_max: 1,
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

fn run_penalties() -> cobre_core::resolved::ResolvedPenalties {
    use cobre_core::HydroPenalties;
    use cobre_core::resolved::{
        BusStagePenalties, LineStagePenalties, NcsStagePenalties, PenaltiesCountsSpec,
        PenaltiesDefaults, ResolvedPenalties,
    };
    ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: 2,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages: RUN_N_STAGES,
        },
        &PenaltiesDefaults {
            hydro: HydroPenalties {
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
            },
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    )
}

/// A water travel-time arc (`RUN_UPSTREAM_ID` -> `RUN_DOWNSTREAM_ID`), a
/// `LeadTime` anticipated thermal with a post-study target, and the
/// `post_study_stages` calendar that target resolves into.
fn run_build_system() -> System {
    let bus = make_bus(RUN_BUS_ID, BusSpec::default());

    let downstream = make_hydro(
        RUN_DOWNSTREAM_ID,
        HydroSpec {
            bus_id: RUN_BUS_ID,
            max_storage_hm3: 1_000.0,
            max_turbined_m3s: 200.0,
            max_generation_mw: 500.0,
            generation_model: HydroGenerationModel::ConstantProductivity,
            ..Default::default()
        },
    );
    let upstream = make_hydro(
        RUN_UPSTREAM_ID,
        HydroSpec {
            bus_id: RUN_BUS_ID,
            downstream_id: Some(RUN_DOWNSTREAM_ID),
            travel_time_hours: Some(RUN_TRAVEL_TIME_HOURS),
            min_outflow_m3s: 50.0,
            max_storage_hm3: 1_000.0,
            max_turbined_m3s: 200.0,
            max_generation_mw: 500.0,
            generation_model: HydroGenerationModel::ConstantProductivity,
            ..Default::default()
        },
    );
    let thermal = make_thermal(
        RUN_THERMAL_ID,
        ThermalSpec {
            bus_id: RUN_BUS_ID,
            cost_per_mwh: 1.0,
            min_generation_mw: 0.0,
            max_generation_mw: 0.0,
            anticipated_config: Some(AnticipatedConfig::LeadTime(RUN_DELTA_HOURS)),
            ..Default::default()
        },
    );

    let inflow_models: Vec<InflowModel> = (0..RUN_N_STAGES)
        .map(|i| InflowModel {
            hydro_id: RUN_UPSTREAM_ID,
            stage_id: i32::try_from(i).unwrap_or(0),
            mean_m3s: 80.0,
            std_m3s: 20.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        })
        .collect();

    let initial_conditions = InitialConditions {
        storage: vec![
            HydroStorage {
                hydro_id: RUN_DOWNSTREAM_ID,
                value_hm3: 100.0,
            },
            HydroStorage {
                hydro_id: RUN_UPSTREAM_ID,
                value_hm3: 100.0,
            },
        ],
        past_defluences: vec![HydroPastDefluence {
            hydro_id: RUN_UPSTREAM_ID,
            start_date: run_study_start() - Duration::days(38),
            end_date: run_study_start(),
            value_m3s: 50.0,
        }],
        ..Default::default()
    };

    SystemBuilder::new()
        .buses(vec![bus])
        .hydros(vec![downstream, upstream])
        .thermals(vec![thermal])
        .stages(run_stages())
        .inflow_models(inflow_models)
        .bounds(run_bounds())
        .penalties(run_penalties())
        .initial_conditions(initial_conditions)
        .post_study_stages(Some(run_post_study_stages()))
        .build()
        .expect("combined deck: water arc + post-study anticipated thermal must build")
}

fn run_config() -> Config {
    Config {
        schema: None,
        modeling: ModelingConfig {
            inflow_non_negativity: InflowNonNegativityConfig {
                method: InflowNonNegativityMethod::Penalty,
            },
            cost_scale_factor: Some(RUN_LOADING_FACTOR),
        },
        training: TrainingConfig {
            enabled: true,
            tree_seed: Some(42),
            stopping_rules: Some(vec![StoppingRuleConfig::IterationLimit {
                limit: RUN_MAX_ITERATIONS,
            }]),
            stopping_mode: StoppingMode::Any,
            cut_selection: RowSelectionConfig::default(),
            solver: TrainingSolverConfig::default(),
            parallelism: ParallelismConfig::default(),
            scenario_source: None,
            selection: Some(TrainingSelection::Sampled { forward_passes: 1 }),
        },
        upper_bound_evaluation: UpperBoundEvaluationConfig::default(),
        policy: PolicyConfig {
            boundary: Some(BoundaryPolicy {
                path: "unused-boundary-checkpoint".to_string(),
                strict: false,
            }),
            ..PolicyConfig::default()
        },
        simulation: SimulationConfig::default(),
        exports: ExportsConfig::default(),
        estimation: EstimationConfig::default(),
    }
}

/// Derive a NEWAVE-shaped source manifest + distinguishable nonzero coefficients
/// from the study's own terminal `manifest`: every storage/inflow-lag target
/// slot becomes a same-identity source slot (`Copy` core), one anticipated
/// source slot per distinct dated post-study delivery month (so each dated ring
/// target `Blend`s), and NO transit slot (so every transit target resolves to
/// `Zero`).
fn derive_newave_source(manifest: &[EntitySlot]) -> (Vec<EntitySlot>, Vec<f64>) {
    let mut source = Vec::new();
    let mut coefficients = Vec::new();
    let mut next = 1.0_f64;

    for slot in manifest {
        if slot.entity_type == StateFamily::HydroStorage.code()
            || slot.entity_type == StateFamily::HydroInflowLag.code()
        {
            source.push(slot.clone());
            coefficients.push(next);
            next += 1.0;
        }
    }

    let mut months: BTreeSet<(i32, i32)> = BTreeSet::new();
    for slot in manifest {
        if slot.entity_type == StateFamily::AnticipatedThermalState.code()
            && slot.interval_start != ENTITY_SLOT_DATE_SENTINEL
            && months.insert((slot.entity_id, slot.interval_start))
        {
            source.push(anticipated_slot_at(slot.entity_id, 0, slot.interval_start));
            coefficients.push(next);
            next += 1.0;
        }
    }

    (source, coefficients)
}

/// Build the DECOMP study, reconcile a derived NEWAVE source into its terminal
/// manifest, inject the reconciled cuts, run a bounded training, and return the
/// final lower bound after asserting the run's invariants (`Ok`, terminal pool
/// boundary-loaded, finite bound).
fn run_injected_decomp() -> f64 {
    let tmp = tempfile::tempdir().expect("tempdir");
    let system = run_build_system();
    let mut setup = build_setup_in_code(run_build_system(), &run_config());

    let manifest = setup.build_terminal_entity_manifest(&system);
    let fixed = setup.build_terminal_fixed_post_horizon_windows(&system);

    let n_storage = manifest
        .iter()
        .filter(|s| s.entity_type == StateFamily::HydroStorage.code())
        .count();
    let n_transit = manifest
        .iter()
        .filter(|s| s.entity_type == StateFamily::HydroTransitBucket.code())
        .count();
    let n_dated_anticipated = manifest
        .iter()
        .filter(|s| {
            s.entity_type == StateFamily::AnticipatedThermalState.code()
                && s.interval_start != ENTITY_SLOT_DATE_SENTINEL
        })
        .count();
    assert!(
        n_storage >= 1,
        "the DECOMP run study must carry storage state"
    );
    assert_eq!(
        n_transit, 2,
        "the DECOMP run study must carry exactly two transit-bucket Zero targets"
    );
    assert!(
        n_dated_anticipated >= 1,
        "the DECOMP run study must carry a dated post-study anticipated Blend target"
    );
    assert_eq!(
        manifest.len(),
        setup.fcf.state_dimension,
        "the terminal pool projects the full state (all-enabled leaf), so the boundary cut \
         aligns 1:1 with the injected pool's state dimension"
    );

    let (source_manifest, source_coefficients) = derive_newave_source(&manifest);
    write_source_checkpoint(
        tmp.path(),
        &source_manifest,
        &source_coefficients,
        Some(RUN_LOADING_FACTOR),
    );

    let cuts = load_boundary_cuts(
        &BoundaryLoadRequest::new(
            tmp.path(),
            fixture_priced_date(0),
            setup.fcf.state_dimension as u32,
            &manifest,
            RUN_LOADING_FACTOR,
        )
        .with_fixed_windows(&fixed),
    )
    .expect("the derived NEWAVE source must reconcile into the DECOMP terminal manifest");
    for record in cuts.iter() {
        assert_eq!(
            record.coefficients.len(),
            setup.fcf.state_dimension,
            "every reconciled record projects to the terminal state dimension"
        );
    }

    inject_boundary_cuts(&mut setup, &cuts).unwrap();
    let terminal = setup
        .inputs
        .node_graph
        .terminal_pool(setup.num_stages())
        .unwrap();
    assert!(
        setup.fcf.pools[terminal].has_warm_start_cuts(),
        "injecting the boundary must make the terminal pool boundary-loaded"
    );

    let comm = StubComm;
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new");
    let outcome = setup
        .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
        .expect("training the injected DECOMP study must not return Err");
    assert!(
        outcome.error.is_none(),
        "training error: {:?}",
        outcome.error
    );
    assert!(
        outcome.result.final_lb.is_finite(),
        "the final lower bound must be finite, got {}",
        outcome.result.final_lb
    );
    outcome.result.final_lb
}

/// The reconciled NEWAVE boundary injected into a runnable DECOMP-shaped study
/// trains to a finite lower bound, and two identical runs reproduce it
/// bit-for-bit (run-to-run reproducibility per the determinism contract — never
/// a hot-vs-cold claim).
#[test]
fn newave_boundary_injected_decomp_run_converges() {
    let final_lb_a = run_injected_decomp();
    let final_lb_b = run_injected_decomp();
    assert_eq!(
        final_lb_a.to_bits(),
        final_lb_b.to_bits(),
        "two identical injected DECOMP runs must reproduce the final lower bound bit-for-bit"
    );
}
