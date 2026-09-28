//! Guard test for two wiring assumptions the terminal (post-horizon)
//! commitment-hold lane depends on: the anticipated-thermal manifest's
//! slot-identity match keys on a generic `subindex`, and the commitment-hold
//! ring's LP columns / stored cuts ride `col_scale`/`cost_scale` exactly like
//! every other state family.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap
)]

mod common;

use chrono::{NaiveDate, TimeDelta};
use cobre_core::entities::thermal::AnticipatedConfig;
use cobre_core::{
    BoundsCountsSpec, BoundsDefaults, BusStagePenalties, ContractBlockBounds, EntityId,
    HydroBlockBounds, HydroPenalties, HydroStageBounds, LineBlockBounds, LineStagePenalties,
    NcsStagePenalties, PenaltiesCountsSpec, PenaltiesDefaults, PumpingBlockBounds, ResolvedBounds,
    ResolvedPenalties, System, SystemBuilder, ThermalBlockBounds, ThermalStageBounds,
};
use cobre_io::config::{
    Config, EstimationConfig, ExportsConfig, InflowNonNegativityConfig, InflowNonNegativityMethod,
    ModelingConfig, ParallelismConfig, PolicyConfig, RowSelectionConfig, SimulationConfig,
    StoppingMode, StoppingRuleConfig, TrainingConfig, TrainingSelection, TrainingSolverConfig,
    UpperBoundEvaluationConfig,
};
use cobre_io::{
    EntitySlot, OwnedPolicyCutRecord, STAGE_CUTS_PRICED_STATE_DATE_SENTINEL, StageCutsReadResult,
};
use cobre_sddp::{
    LEGACY_COST_SCALE_FACTOR, SddpError, compare_manifest_slot_identity,
    rescale_checkpoint_cuts_for_load,
};

use common::build_setup_in_code;
use common::builders::{BusSpec, StageSpec, ThermalSpec, make_bus, make_stage, make_thermal};

const K_MAX: usize = 2;
const N_STAGES: usize = 3;
const COST_PER_MWH: f64 = 42.0;
const MAX_GEN_MW: f64 = 100.0;

/// One bus, one anticipated thermal (`lead_stages = K_MAX`), zero load, zero
/// hydros — the minimal fixture that gives the anticipated ring a non-empty
/// `n_state` without exercising any other state family.
fn build_system() -> System {
    let bus = make_bus(EntityId(1), BusSpec::default());

    let thermal = make_thermal(
        EntityId(2),
        ThermalSpec {
            name: "T_ant".to_string(),
            bus_id: EntityId(1),
            anticipated_config: Some(AnticipatedConfig::LeadStages(K_MAX as u32)),
            ..Default::default()
        },
    );

    let stage_start = |offset_days: i64| {
        NaiveDate::from_ymd_opt(2024, 1, 1).unwrap() + TimeDelta::days(offset_days)
    };
    let stages = (0..N_STAGES)
        .map(|i| {
            make_stage(
                i,
                StageSpec {
                    start_date: stage_start(i as i64),
                    end_date: stage_start(i as i64 + 1),
                    ..Default::default()
                },
            )
        })
        .collect();

    let bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 0,
            n_thermals: 1,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: N_STAGES,
            k_max: K_MAX,
        },
        &BoundsDefaults {
            hydro: HydroStageBounds {
                min_storage_hm3: 0.0,
                max_storage_hm3: 0.0,
                filling_min_rate_m3s: 0.0,
                water_withdrawal_m3s: 0.0,
            },
            hydro_block: HydroBlockBounds::default(),
            thermal: ThermalStageBounds {
                cost_per_mwh: COST_PER_MWH,
            },
            thermal_block: ThermalBlockBounds {
                min_generation_mw: 0.0,
                max_generation_mw: MAX_GEN_MW,
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

    let penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: 0,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages: N_STAGES,
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
    );

    SystemBuilder::new()
        .buses(vec![bus])
        .thermals(vec![thermal])
        .stages(stages)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("fixture System must build")
}

fn build_config() -> Config {
    Config {
        schema: None,
        modeling: ModelingConfig {
            inflow_non_negativity: InflowNonNegativityConfig {
                method: InflowNonNegativityMethod::Penalty,
            },
            cost_scale_factor: None,
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

/// Invariant 1: `compare_manifest_slot_identity` keys on the plain
/// `(entity_type, entity_id, subindex)` triple — `subindex` is opaque to it, so
/// a window-index slot the commitment block assigns (a numbering unrelated to
/// the ring's own `[0, k_max)` domain) binds under the identical mechanism the
/// anticipated ring already relies on, with no schema change.
#[test]
fn manifest_slot_identity_binds_on_subindex_with_no_schema_change() {
    let system = build_system();
    let setup = build_setup_in_code(build_system(), &build_config());
    let manifest = setup.build_terminal_entity_manifest(&system);

    assert!(
        !manifest.is_empty(),
        "fixture's terminal manifest must be non-empty (n_anticipated=1, k_max={K_MAX})",
    );
    // n_hydros = 0 and no travel-time buckets are declared, so every terminal
    // manifest slot on this fixture belongs to the anticipated-thermal family —
    // there is no storage/lag/bucket slot competing for the entity-type byte.
    let real_slot = manifest[0].clone();

    let window_indexed_slot = EntitySlot::anticipated(real_slot.entity_id, 7, real_slot.was_active)
        .with_interval(real_slot.interval_start, real_slot.interval_end);
    let source = vec![window_indexed_slot.clone()];
    let matching_current = vec![window_indexed_slot.clone()];
    compare_manifest_slot_identity(&source, &matching_current, &mut |_| {})
        .expect("identical (entity_type, entity_id, subindex) slots must bind");

    let mut mismatched_current = matching_current;
    mismatched_current[0].subindex = window_indexed_slot.subindex + 1;
    let err = compare_manifest_slot_identity(&source, &mismatched_current, &mut |_| {}).expect_err(
        "a subindex-only mismatch must be rejected, or a boundary-injected cut could \
             silently attach to the wrong state variable",
    );
    assert!(
        matches!(err, SddpError::Validation(_)),
        "subindex mismatch must reject as SddpError::Validation; got {err:?}",
    );
}

/// Invariant 2: `commit_out` and `commit_in` carry
/// `col_scale == 1.0` on every stage template after construction-time LP
/// postprocessing — the ring's pin/re-read round-trip depends on this.
#[test]
fn anticipated_ring_columns_carry_unit_col_scale() {
    let setup = build_setup_in_code(build_system(), &build_config());
    let state = setup.stage_state();
    let commit_out = state.commit_out.clone();
    let commit_in = state.commit_in.clone();

    assert!(
        !commit_out.is_empty() && !commit_in.is_empty(),
        "fixture must carry a non-empty anticipated ring (n_anticipated=1, k_max={K_MAX})",
    );

    for (stage_idx, template) in setup
        .inputs
        .stage_data
        .stage_templates
        .templates
        .iter()
        .enumerate()
    {
        for c in commit_out.clone() {
            assert_eq!(
                template.col_scale[c], 1.0,
                "stage {stage_idx}: commit_out column {c} must carry col_scale == 1.0",
            );
        }
        for c in commit_in.clone() {
            assert_eq!(
                template.col_scale[c], 1.0,
                "stage {stage_idx}: commit_in column {c} must carry col_scale == 1.0",
            );
        }
    }
}

fn sample_cut_records() -> Vec<OwnedPolicyCutRecord> {
    vec![
        OwnedPolicyCutRecord {
            cut_id: 1,
            slot_index: 0,
            iteration: 0,
            forward_pass_index: 0,
            intercept: 10.0,
            coefficients: vec![1.0, -2.0, 3.5],
            is_active: true,
        },
        OwnedPolicyCutRecord {
            cut_id: 2,
            slot_index: 1,
            iteration: 0,
            forward_pass_index: 0,
            intercept: -4.0,
            coefficients: vec![0.0, 7.25, -1.0],
            is_active: true,
        },
    ]
}

fn sample_stage_cuts() -> Vec<StageCutsReadResult> {
    vec![StageCutsReadResult {
        stage_id: 0,
        state_dimension: 3,
        capacity: 2,
        warm_start_count: 2,
        populated_count: 2,
        cuts: sample_cut_records(),
        entity_manifest: Vec::new(),
        cost_scale_factor: None,
        node_id: -1,
        graph_stage_id: -1,
        priced_state_date: STAGE_CUTS_PRICED_STATE_DATE_SENTINEL,
    }]
}

/// Invariant 3 (marked-source path): every coefficient AND the intercept scale
/// by the identical `1 / loading_cost_scale_factor` factor — no per-family
/// branch a commitment-block coefficient could fall outside of.
#[test]
fn rescale_checkpoint_cuts_scales_every_coefficient_uniformly_marked_source() {
    let before = sample_cut_records();
    let mut stage_cuts = sample_stage_cuts();
    let loading_cost_scale_factor = 2_000_000.0;

    rescale_checkpoint_cuts_for_load(
        &mut stage_cuts,
        Some(1_000_000.0),
        loading_cost_scale_factor,
    );

    for (b, a) in before.iter().zip(&stage_cuts[0].cuts) {
        assert_eq!(a.intercept, b.intercept / loading_cost_scale_factor);
        for (bc, ac) in b.coefficients.iter().zip(&a.coefficients) {
            assert_eq!(
                *ac,
                bc / loading_cost_scale_factor,
                "every coefficient must scale by the identical factor as the intercept",
            );
        }
    }
}

/// Invariant 3 (legacy-source path): the alternate branch of the same function
/// also scales the intercept and every coefficient by one identical ratio.
#[test]
fn rescale_checkpoint_cuts_scales_every_coefficient_uniformly_legacy_source() {
    let before = sample_cut_records();
    let mut stage_cuts = sample_stage_cuts();
    let loading_cost_scale_factor = 2_000_000.0;
    let ratio = LEGACY_COST_SCALE_FACTOR / loading_cost_scale_factor;

    rescale_checkpoint_cuts_for_load(&mut stage_cuts, None, loading_cost_scale_factor);

    for (b, a) in before.iter().zip(&stage_cuts[0].cuts) {
        assert_eq!(a.intercept, b.intercept * ratio);
        for (bc, ac) in b.coefficients.iter().zip(&a.coefficients) {
            assert_eq!(
                *ac,
                bc * ratio,
                "every coefficient must scale by the identical factor as the intercept",
            );
        }
    }
}
