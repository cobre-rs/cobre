#![expect(
    clippy::too_many_lines,
    clippy::cast_sign_loss,
    clippy::needless_range_loop,
    clippy::doc_markdown,
    reason = "the fixture spells out one complete study inline so each assertion traces to a literal, the test reads non-negative CSC offsets, the loop index addresses parallel arrays, and test docs name LP symbols that are not code identifiers"
)]

use chrono::NaiveDate;
use cobre_core::scenario::SamplingScheme;
use cobre_core::{
    AnticipatedConfig, Block, BlockMode, BoundsCountsSpec, BoundsDefaults, Bus, BusStagePenalties,
    ContractBlockBounds, ContractType, DeficitSegment, EnergyContract, EntityId, FillingConfig,
    Hydro, HydroBlockBounds, HydroGenerationModel, HydroPenalties, HydroStageBounds,
    LineBlockBounds, LineStagePenalties, LoadModel, NcsStagePenalties, NoiseMethod,
    NonControllableSource, PenaltiesCountsSpec, PenaltiesDefaults, PostStudyStage, PostStudyStages,
    PostStudyThermalBound, PumpingBlockBounds, PumpingStation, ResolvedBounds, ResolvedNcsBounds,
    ResolvedNcsFactors, ResolvedPenalties, ScenarioSourceConfig, Stage, StageRiskConfig,
    StageStateConfig, SystemBuilder, Thermal, ThermalBlockBounds, ThermalStageBounds,
};
use cobre_stochastic::PrecomputedNormal;
use cobre_stochastic::par::precompute::PrecomputedPar;
use cobre_stochastic::season_cast::post_study_calendar_stages;

use crate::block_clock::M3S_TO_HM3;
use crate::hydro_models::PrepareHydroModelsResult;
use crate::indexer::{
    AnticipatedLocal, AnticipatedPlants, BlockIdx, Boundary, HydroCell, HydroCellIndex, HydroSys,
    NcsSys, PumpingSys, StateSpace, StudyDimensions, ThermalSys, anticipated_resolution_for,
};
use crate::inflow_method::InflowNonNegativityMethod;
use crate::lead_time::AnticipatedResolution;
use crate::resolved_parameters::ResolvedParameters;
use crate::test_support::{
    assert_templates_byte_identical, resolve_anticipated_commitments, state_layout_full,
};
use crate::time_value::{
    DeliveryCalendar, PostStudyResolved, TimeValue, compute_cumulative_discount_factors,
    compute_per_stage_discount_factors, resolve_post_study_artifacts,
};

/// The value `build_template_build_ctx`'s own `time_value` parameter takes at
/// every direct test call site — resolved through the same production entry
/// point (`TimeValue::from_system`) rather than hand-assembled.
fn build_time_value_for(system: &cobre_core::System) -> TimeValue {
    let anticipated_plants = AnticipatedPlants::build(system.thermals());
    TimeValue::from_system(
        system,
        &anticipated_plants,
        DeliveryCalendar::from_system(system),
    )
}

// ── Fixtures ─────────────────────────────────────────────────────────────

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

fn default_hydro_penalties() -> HydroPenalties {
    HydroPenalties {
        spillage_cost: 0.01,
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
        inflow_nonnegativity_cost: 1000.0,
    }
}

fn fixture_bus() -> Bus {
    Bus {
        id: EntityId(1),
        name: "B1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 500.0,
        }],
        excess_cost: 0.0,
    }
}

/// Build a one-bus system with exactly the thermals provided.
///
/// Uses one study stage with a single block of 744 hours and no hydros.
fn system_with_thermals(thermals: Vec<Thermal>) -> cobre_core::System {
    let n_thermals = thermals.len();
    let n_stages = 1_usize;

    let bus = fixture_bus();

    let stages: Vec<Stage> = vec![Stage {
        index: 0,
        id: 0,
        start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
        season_id: Some(0),
        blocks: vec![Block {
            index: 0,
            name: "BLK0".to_string(),
            duration_hours: 744.0,
        }],
        block_mode: BlockMode::Parallel,
        state_config: StageStateConfig {
            storage: false,
            inflow_lags: false,
        },
        risk_config: StageRiskConfig::Expectation,
        scenario_config: ScenarioSourceConfig {
            branching_factor: 1,
            noise_method: NoiseMethod::Saa,
        },
    }];

    let load_models = vec![LoadModel {
        bus_id: EntityId(1),
        stage_id: 0,
        mean_mw: 100.0,
        std_mw: 0.0,
    }];

    let k_max = thermals
        .iter()
        .filter_map(|t| t.anticipated_config.as_ref())
        .map(|c| c.lead_stages().unwrap() as usize)
        .max()
        .unwrap_or(0);

    let resolved_bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 0,
            n_thermals,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages,
            k_max,
        },
        &BoundsDefaults {
            hydro: default_hydro_bounds(),
            hydro_block: default_hydro_block_bounds(),
            thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
            thermal_block: ThermalBlockBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 100.0,
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
            n_stages,
        },
        &PenaltiesDefaults {
            hydro: default_hydro_penalties(),
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    );

    SystemBuilder::new()
        .buses(vec![bus])
        .thermals(thermals)
        .stages(stages)
        .load_models(load_models)
        .bounds(resolved_bounds)
        .penalties(penalties)
        .build()
        .expect("system_with_thermals: valid system")
}

/// `system_with_thermals`'s bus carries a `std_mw == 0.0` load model; its slot
/// is admitted under `SamplingScheme::External` and excluded under
/// `SamplingScheme::InSample` (the byte-neutral default every other caller
/// threads). The scheme resolves the membership list the caller passes into
/// `resolve_lp_build_inputs` — its own `load_bus_indices` derivation is a
/// pure id-to-position map, no scheme of its own.
#[test]
fn resolve_lp_build_inputs_load_bus_indices_honors_threaded_scheme() {
    let system = system_with_thermals(vec![]);
    let production_models = ProductionModelSet::new(Vec::new(), &[], 0);
    let study_dims = StudyDimensions::default();
    let time_value = build_time_value_for(&system);
    let hydro_cell_index = HydroCellIndex::build(system.hydros());
    let resolved_params = empty_resolved_params();

    let external = crate::test_support::resolve_lp_build_inputs(
        &system,
        &system.load_noise_member_bus_ids(SamplingScheme::External),
        &production_models,
        &study_dims,
        &time_value,
        &hydro_cell_index,
        &resolved_params,
    );
    assert_eq!(external.load_bus_indices, vec![0]);

    let in_sample = crate::test_support::resolve_lp_build_inputs(
        &system,
        &system.load_noise_member_bus_ids(SamplingScheme::InSample),
        &production_models,
        &study_dims,
        &time_value,
        &hydro_cell_index,
        &resolved_params,
    );
    assert!(in_sample.load_bus_indices.is_empty());
}

/// Build empty [`ResolvedParameters`] (no parameters).
fn empty_resolved_params() -> ResolvedParameters {
    ResolvedParameters {
        per_param: vec![],
        id_to_slot: vec![],
        cost_scale_factor: 1_000_000.0,
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

/// Minimal independent (no-downstream) hydro for pumping-station refs.
fn fixture_hydro(id: i32) -> Hydro {
    let mut hydro = Hydro {
        unit_groups: Vec::new(),
        id: EntityId(id),
        name: format!("H{id}"),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        downstream_id: None,
        travel_time_hours: None,
        entry_stage_id: None,
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
        filling: None,
        penalties: hydro_penalties_zero(),
    };
    hydro.declare_mirror_unit_group(EntityId(1));
    hydro
}

/// Build a one-bus, two-hydro system with the supplied pumping stations.
///
/// `SystemBuilder::build` sorts every entity Vec by `id.0`, so passing
/// stations out of declaration order exercises the canonical-ordering
/// guarantee that `build_template_build_ctx` relies on when threading the
/// slice into `ctx.pumping_stations`/`ctx.positions`. The two hydros and bus
/// exist solely to satisfy pumping-station reference validation.
fn system_with_pumping_stations(stations: Vec<PumpingStation>) -> cobre_core::System {
    let n_pumping = stations.len();
    let n_hydros = 2_usize;
    let n_stages = 1_usize;

    let bus = fixture_bus();

    let hydros = vec![fixture_hydro(1), fixture_hydro(2)];

    let stages: Vec<Stage> = vec![Stage {
        index: 0,
        id: 0,
        start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
        season_id: Some(0),
        blocks: vec![Block {
            index: 0,
            name: "BLK0".to_string(),
            duration_hours: 744.0,
        }],
        block_mode: BlockMode::Parallel,
        state_config: StageStateConfig {
            storage: false,
            inflow_lags: false,
        },
        risk_config: StageRiskConfig::Expectation,
        scenario_config: ScenarioSourceConfig {
            branching_factor: 1,
            noise_method: NoiseMethod::Saa,
        },
    }];

    let load_models = vec![LoadModel {
        bus_id: EntityId(1),
        stage_id: 0,
        mean_mw: 100.0,
        std_mw: 0.0,
    }];

    let resolved_bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros,
            n_thermals: 0,
            n_lines: 0,
            n_pumping,
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
                max_generation_mw: 100.0,
            },
            line_block: LineBlockBounds {
                direct_mw: 0.0,
                reverse_mw: 0.0,
            },
            pumping_block: PumpingBlockBounds {
                min_flow_m3s: 0.0,
                max_flow_m3s: 100.0,
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
            n_hydros,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages,
        },
        &PenaltiesDefaults {
            hydro: default_hydro_penalties(),
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    );

    SystemBuilder::new()
        .buses(vec![bus])
        .hydros(hydros)
        .pumping_stations(stations)
        .stages(stages)
        .load_models(load_models)
        .bounds(resolved_bounds)
        .penalties(penalties)
        .build()
        .expect("system_with_pumping_stations: valid system")
}

/// Build a pumping station with the given id (bus/hydro refs fixed to the
/// fixture entities; flow window and consumption are non-degenerate).
fn fixture_pumping_station(id: i32) -> PumpingStation {
    PumpingStation {
        id: EntityId(id),
        name: format!("P{id}"),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        source_hydro_id: EntityId(1),
        destination_hydro_id: EntityId(2),
        entry_stage_id: None,
        exit_stage_id: None,
        consumption_mw_per_m3s: 0.5,
        min_flow_m3s: 0.0,
        max_flow_m3s: 100.0,
    }
}

// ── Pumping data threaded into TemplateBuildCtx ────────────────────────────

/// Stations declared out of ID order are exposed ID-sorted on the ctx, and
/// `ctx.positions.pumping` maps each station id to its slot in that sorted
/// slice.
///
/// Declaration order `[30, 10, 20]` must become `[10, 20, 30]` on the ctx
/// (the canonical sort applied by `SystemBuilder::build`), with
/// `positions.pumping = {10->Some(0), 20->Some(1), 30->Some(2)}`.
#[test]
fn build_template_build_ctx_pumping_stations_id_sorted_and_pos_mapped() {
    let stations = vec![
        fixture_pumping_station(30),
        fixture_pumping_station(10),
        fixture_pumping_station(20),
    ];
    let system = system_with_pumping_stations(stations);
    let hydro_result = PrepareHydroModelsResult::default_from_system(&system);
    let par_lp = PrecomputedPar::default();
    let resolved_params = empty_resolved_params();

    let (topology, resolved) = crate::test_support::resolved_layout_for(&system, &par_lp);
    let hydro_cell_index = HydroCellIndex::build(system.hydros());
    let time_value = build_time_value_for(&system);
    let study_dims = crate::test_support::build_study_dimensions(
        &system,
        InflowNonNegativityMethod::None,
        resolved.anticipated_plants.clone(),
        0,
    );
    let inputs = crate::test_support::resolve_lp_build_inputs(
        &system,
        &[],
        &hydro_result.production,
        &study_dims,
        &time_value,
        &hydro_cell_index,
        &resolved_params,
    );
    let ctx = super::build_template_build_ctx(
        &system,
        &par_lp,
        &hydro_result.production,
        &hydro_result.evaporation,
        &resolved.state,
        &topology,
        &inputs,
    );

    let ids: Vec<i32> = ctx.pumping_stations.iter().map(|p| p.id.0).collect();
    assert_eq!(
        ids,
        vec![10, 20, 30],
        "ctx.pumping_stations must be ID-sorted regardless of declaration order"
    );

    assert_eq!(ctx.positions.pumping(EntityId(10)), Some(0));
    assert_eq!(ctx.positions.pumping(EntityId(20)), Some(1));
    assert_eq!(ctx.positions.pumping(EntityId(30)), Some(2));

    for (slot, station) in ctx.pumping_stations.iter().enumerate() {
        assert_eq!(
            ctx.positions.pumping(station.id),
            Some(slot),
            "positions.pumping({:?}) must equal its slot in the sorted slice",
            station.id
        );
    }
}

/// `ctx.pumping_stations.len()` equals the resolved-bounds station count,
/// and that count is what `StageLayout` reserves pumping-flow columns for.
#[test]
fn build_template_build_ctx_n_pumping_matches_slice_and_bounds() {
    let stations = vec![fixture_pumping_station(7), fixture_pumping_station(3)];
    let system = system_with_pumping_stations(stations);
    let hydro_result = PrepareHydroModelsResult::default_from_system(&system);
    let par_lp = PrecomputedPar::default();
    let resolved_params = empty_resolved_params();

    let (topology, resolved) = crate::test_support::resolved_layout_for(&system, &par_lp);
    let hydro_cell_index = HydroCellIndex::build(system.hydros());
    let time_value = build_time_value_for(&system);
    let study_dims = crate::test_support::build_study_dimensions(
        &system,
        InflowNonNegativityMethod::None,
        resolved.anticipated_plants.clone(),
        0,
    );
    let inputs = crate::test_support::resolve_lp_build_inputs(
        &system,
        &[],
        &hydro_result.production,
        &study_dims,
        &time_value,
        &hydro_cell_index,
        &resolved_params,
    );
    let ctx = super::build_template_build_ctx(
        &system,
        &par_lp,
        &hydro_result.production,
        &hydro_result.evaporation,
        &resolved.state,
        &topology,
        &inputs,
    );

    assert_eq!(ctx.pumping_stations.len(), 2, "two stations were declared");
    assert_eq!(
        ctx.pumping_stations.len(),
        ctx.resolved.bounds.n_pumping(),
        "ctx.pumping_stations.len() must agree with the resolved-bounds station count"
    );

    // Block-major column reservation is pinned separately by the layout-module
    // test `pumping_layout_reserves_block_major_columns`.
    let stage = system
        .stages()
        .iter()
        .find(|s| s.id >= 0)
        .expect("one study stage");
    let layout = super::super::layout::StageLayout::new(&ctx, stage, 0);
    assert_eq!(
        layout.geometry.pumping_flow.len(),
        ctx.pumping_stations.len() * layout.clock.n_blks(),
        "StageLayout must reserve exactly one pumping-flow column per station per block"
    );
}

/// `build_stage_templates` records the layout-owned pumping-flow range for
/// every stage: `geometry_per_stage[t].pumping_flow` equals
/// `StageLayout::new(..).geometry.pumping_flow`, with the station count
/// constant across stages under the dense layout.
///
/// This pins the threading contract the simulation extraction pipeline reads
/// from: the column base is sourced from the layout, the sole owner of the
/// pumping-flow column base.
#[test]
fn build_stage_templates_records_the_layout_pumping_flow_range_per_stage() {
    let stations = vec![fixture_pumping_station(5), fixture_pumping_station(2)];
    let system = system_with_pumping_stations(stations);
    let hydro_result = PrepareHydroModelsResult::default_from_system(&system);
    let par_lp = PrecomputedPar::default();
    let normal_lp = PrecomputedNormal::default();
    let resolved_params = empty_resolved_params();

    let templates = crate::build_stage_templates_resolving_layout(
        &system,
        InflowNonNegativityMethod::None,
        &par_lp,
        &normal_lp,
        &hydro_result.production,
        &hydro_result.evaporation,
        &resolved_params,
    )
    .expect("build_stage_templates: valid system");

    let (topology, resolved) = crate::test_support::resolved_layout_for(&system, &par_lp);
    let hydro_cell_index = HydroCellIndex::build(system.hydros());
    let time_value = build_time_value_for(&system);
    let study_dims = crate::test_support::build_study_dimensions(
        &system,
        InflowNonNegativityMethod::None,
        resolved.anticipated_plants.clone(),
        0,
    );
    let inputs = crate::test_support::resolve_lp_build_inputs(
        &system,
        &[],
        &hydro_result.production,
        &study_dims,
        &time_value,
        &hydro_cell_index,
        &resolved_params,
    );
    let ctx = super::build_template_build_ctx(
        &system,
        &par_lp,
        &hydro_result.production,
        &hydro_result.evaporation,
        &resolved.state,
        &topology,
        &inputs,
    );
    let study_stages: Vec<_> = system.stages().iter().filter(|s| s.id >= 0).collect();

    assert_eq!(templates.geometry_per_stage.len(), study_stages.len());
    for (t, stage) in study_stages.iter().enumerate() {
        let layout = super::super::layout::StageLayout::new(&ctx, stage, t);
        assert_eq!(
            ctx.pumping_stations.len(),
            2,
            "stage {t}: two stations were declared"
        );
        let geom = &templates.geometry_per_stage[t];
        assert_eq!(
            geom.pumping_flow,
            layout.geometry.pumping_flow.start
                ..layout.geometry.pumping_flow.start + ctx.pumping_stations.len() * geom.n_blks,
            "stage {t}: geometry.pumping_flow must equal the layout's own pumping range"
        );
    }
}

/// `StageGeometry::pumping_flow` spans exactly `n_pumping * n_blks` columns at
/// every stage, block-major over the station count, and
/// `StageGeometry::pumping_flow_col` addresses it by
/// `pumping_flow.start + p * n_blks + blk`, on the pumping-station fixture
/// study.
#[test]
fn geometry_pumping_family_is_block_major_over_the_station_count() {
    let stations = vec![fixture_pumping_station(5), fixture_pumping_station(2)];
    let system = system_with_pumping_stations(stations);
    let hydro_result = PrepareHydroModelsResult::default_from_system(&system);
    let par_lp = PrecomputedPar::default();
    let normal_lp = PrecomputedNormal::default();
    let resolved_params = empty_resolved_params();

    let templates = crate::build_stage_templates_resolving_layout(
        &system,
        InflowNonNegativityMethod::None,
        &par_lp,
        &normal_lp,
        &hydro_result.production,
        &hydro_result.evaporation,
        &resolved_params,
    )
    .expect("build_stage_templates: valid system");

    for t in 0..templates.geometry_per_stage.len() {
        let geom = &templates.geometry_per_stage[t];
        let n_pumping = system.pumping_stations().len();
        assert_eq!(
            geom.pumping_flow.len(),
            n_pumping * geom.n_blks,
            "stage {t}: pumping_flow must span n_pumping*n_blks columns"
        );
        for p in 0..n_pumping {
            for blk in 0..geom.n_blks {
                assert_eq!(
                    geom.pumping_flow_col(PumpingSys::new(p), BlockIdx::new(blk)),
                    geom.pumping_flow.start + p * geom.n_blks + blk,
                    "stage {t}: pumping_flow_col({p}, {blk}) must equal \
                     pumping_flow.start + p*n_blks+blk"
                );
            }
        }
    }
}

// ── NCS data threaded into TemplateBuildCtx and StageGeometry ─────────────

/// Build a non-controllable source with the given id (bus ref fixed to the
/// fixture bus; a non-degenerate installed capacity, curtailment allowed).
fn fixture_non_controllable_source(id: i32) -> NonControllableSource {
    NonControllableSource {
        id: EntityId(id),
        name: format!("W{id}"),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        entry_stage_id: None,
        exit_stage_id: None,
        max_generation_mw: 100.0,
        allow_curtailment: true,
        curtailment_cost: 5.0,
    }
}

/// Build a one-bus, two-hydro system with the supplied non-controllable
/// sources over `n_blks` blocks. Mirrors `system_with_pumping_stations`; the
/// two hydros and bus exist solely to satisfy NCS bus-reference validation.
fn system_with_non_controllable_sources(
    sources: Vec<NonControllableSource>,
    n_blks: usize,
) -> cobre_core::System {
    let n_ncs = sources.len();
    let n_hydros = 2_usize;
    let n_stages = 1_usize;

    let bus = fixture_bus();
    let hydros = vec![fixture_hydro(1), fixture_hydro(2)];

    let blocks: Vec<Block> = (0..n_blks)
        .map(|b| Block {
            index: b,
            name: format!("BLK{b}"),
            duration_hours: 372.0,
        })
        .collect();

    let stages: Vec<Stage> = vec![Stage {
        index: 0,
        id: 0,
        start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
        season_id: Some(0),
        blocks,
        block_mode: BlockMode::Parallel,
        state_config: StageStateConfig {
            storage: false,
            inflow_lags: false,
        },
        risk_config: StageRiskConfig::Expectation,
        scenario_config: ScenarioSourceConfig {
            branching_factor: 1,
            noise_method: NoiseMethod::Saa,
        },
    }];

    let load_models = vec![LoadModel {
        bus_id: EntityId(1),
        stage_id: 0,
        mean_mw: 100.0,
        std_mw: 0.0,
    }];

    let resolved_bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros,
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
                max_generation_mw: 100.0,
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
            n_hydros,
            n_buses: 1,
            n_lines: 0,
            n_ncs,
            n_stages,
        },
        &PenaltiesDefaults {
            hydro: default_hydro_penalties(),
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    );
    let max_gen: Vec<f64> = sources.iter().map(|s| s.max_generation_mw).collect();

    SystemBuilder::new()
        .buses(vec![bus])
        .hydros(hydros)
        .non_controllable_sources(sources)
        .stages(stages)
        .load_models(load_models)
        .bounds(resolved_bounds)
        .penalties(penalties)
        .resolved_ncs_bounds(ResolvedNcsBounds::new(n_ncs, n_stages, &max_gen))
        .resolved_ncs_factors(ResolvedNcsFactors::new(n_ncs, n_stages, n_blks))
        .build()
        .expect("system_with_non_controllable_sources: valid system")
}

/// `StageGeometry::ncs_generation` spans exactly `n_ncs * n_blks` columns at
/// every stage, matches the stage's own `StageLayout`, and
/// `StageGeometry::ncs_generation_col` addresses it by
/// `ncs_generation.start + sys_idx * n_blks + blk`, on a deck that models NCS.
#[test]
fn geometry_ncs_family_matches_the_stage_layout() {
    let sources = vec![
        fixture_non_controllable_source(5),
        fixture_non_controllable_source(2),
    ];
    let system = system_with_non_controllable_sources(sources, 2);
    let hydro_result = PrepareHydroModelsResult::default_from_system(&system);
    let par_lp = PrecomputedPar::default();
    let normal_lp = PrecomputedNormal::default();
    let resolved_params = empty_resolved_params();

    let templates = crate::build_stage_templates_resolving_layout(
        &system,
        InflowNonNegativityMethod::None,
        &par_lp,
        &normal_lp,
        &hydro_result.production,
        &hydro_result.evaporation,
        &resolved_params,
    )
    .expect("build_stage_templates: valid system");

    let (topology, resolved) = crate::test_support::resolved_layout_for(&system, &par_lp);
    let hydro_cell_index = HydroCellIndex::build(system.hydros());
    let time_value = build_time_value_for(&system);
    let study_dims = crate::test_support::build_study_dimensions(
        &system,
        InflowNonNegativityMethod::None,
        resolved.anticipated_plants.clone(),
        0,
    );
    let inputs = crate::test_support::resolve_lp_build_inputs(
        &system,
        &[],
        &hydro_result.production,
        &study_dims,
        &time_value,
        &hydro_cell_index,
        &resolved_params,
    );
    let ctx = super::build_template_build_ctx(
        &system,
        &par_lp,
        &hydro_result.production,
        &hydro_result.evaporation,
        &resolved.state,
        &topology,
        &inputs,
    );
    let study_stages: Vec<_> = system.stages().iter().filter(|s| s.id >= 0).collect();

    for (t, stage) in study_stages.iter().enumerate() {
        let layout = super::super::layout::StageLayout::new(&ctx, stage, t);
        assert_eq!(
            ctx.non_controllable_sources.len(),
            2,
            "stage {t}: two NCS sources were declared"
        );
        let geom = &templates.geometry_per_stage[t];
        let n_ncs = ctx.non_controllable_sources.len();
        assert_eq!(
            geom.ncs_generation.len(),
            n_ncs * geom.n_blks,
            "stage {t}: ncs_generation must span n_ncs*n_blks columns"
        );
        assert_eq!(
            geom.ncs_generation,
            layout.geometry.ncs_generation.start
                ..layout.geometry.ncs_generation.start + n_ncs * geom.n_blks,
            "stage {t}: geometry.ncs_generation must equal the layout's own NCS range"
        );
        for sys_idx in 0..n_ncs {
            for blk in 0..geom.n_blks {
                assert_eq!(
                    geom.ncs_generation_col(NcsSys::new(sys_idx), BlockIdx::new(blk)),
                    geom.ncs_generation.start + sys_idx * geom.n_blks + blk,
                    "stage {t}: ncs_generation_col({sys_idx}, {blk}) must equal \
                     ncs_generation.start + sys_idx*n_blks+blk"
                );
            }
        }
    }
}

/// `StageGeometry::ncs_generation`/`pumping_flow` span exactly the system's
/// own NCS/pumping-station count times `n_blks` at every stage, on the NCS
/// and pumping-station fixture decks above.
#[test]
fn ncs_and_pumping_families_span_the_system_count_per_block_at_every_stage() {
    let sources = vec![
        fixture_non_controllable_source(5),
        fixture_non_controllable_source(2),
    ];
    let ncs_system = system_with_non_controllable_sources(sources, 2);
    let hydro_result = PrepareHydroModelsResult::default_from_system(&ncs_system);
    let par_lp = PrecomputedPar::default();
    let normal_lp = PrecomputedNormal::default();
    let resolved_params = empty_resolved_params();
    let ncs_templates = crate::build_stage_templates_resolving_layout(
        &ncs_system,
        InflowNonNegativityMethod::None,
        &par_lp,
        &normal_lp,
        &hydro_result.production,
        &hydro_result.evaporation,
        &resolved_params,
    )
    .expect("build_stage_templates: valid system");
    for geom in &ncs_templates.geometry_per_stage {
        assert_eq!(
            geom.ncs_generation.len(),
            ncs_system.non_controllable_sources().len() * geom.n_blks,
            "ncs_generation must span the system NCS count * n_blks columns"
        );
    }

    let stations = vec![fixture_pumping_station(5), fixture_pumping_station(2)];
    let pumping_system = system_with_pumping_stations(stations);
    let hydro_result = PrepareHydroModelsResult::default_from_system(&pumping_system);
    let pumping_templates = crate::build_stage_templates_resolving_layout(
        &pumping_system,
        InflowNonNegativityMethod::None,
        &par_lp,
        &normal_lp,
        &hydro_result.production,
        &hydro_result.evaporation,
        &resolved_params,
    )
    .expect("build_stage_templates: valid system");
    for geom in &pumping_templates.geometry_per_stage {
        assert_eq!(
            geom.pumping_flow.len(),
            pumping_system.pumping_stations().len() * geom.n_blks,
            "pumping_flow must span the system pumping-station count * n_blks columns"
        );
    }
}

/// A system with zero study stages returns empty templates — the early
/// return `build_stage_templates` takes before touching `state_layout`.
#[test]
fn build_stage_templates_empty_system_yields_no_templates() {
    let bus = fixture_bus();
    let system = SystemBuilder::new()
        .buses(vec![bus])
        .build()
        .expect("bus-only system must build");
    let hydro_result = PrepareHydroModelsResult::default_from_system(&system);
    let par_lp = PrecomputedPar::default();
    let normal_lp = PrecomputedNormal::default();
    let resolved_params = empty_resolved_params();

    let result = crate::build_stage_templates_resolving_layout(
        &system,
        InflowNonNegativityMethod::None,
        &par_lp,
        &normal_lp,
        &hydro_result.production,
        &hydro_result.evaporation,
        &resolved_params,
    )
    .expect("zero-stage system must resolve");

    assert!(result.templates.is_empty());
}

// ── Contract data threaded into TemplateBuildCtx and StageGeometry ─────────

/// Build an energy contract with the given id and direction (bus fixed to the
/// fixture bus; a non-degenerate price/power window).
fn fixture_contract(id: i32, contract_type: ContractType) -> EnergyContract {
    EnergyContract {
        id: EntityId(id),
        name: format!("C{id}"),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        contract_type,
        entry_stage_id: None,
        exit_stage_id: None,
        price_per_mwh: 100.0,
        min_mw: 0.0,
        max_mw: 500.0,
    }
}

/// Build a one-bus, two-hydro system with the supplied contracts and a single
/// `n_blks`-block study stage. `n_contracts` matches the slice so the
/// resolved-bounds count check holds; the two hydros and bus exist solely to
/// satisfy contract bus-reference validation and give the layout pumping-end
/// anchor a non-trivial column prefix.
fn system_with_contracts(contracts: Vec<EnergyContract>, n_blks: usize) -> cobre_core::System {
    let n_contracts = contracts.len();
    let n_hydros = 2_usize;
    let n_stages = 1_usize;

    let bus = fixture_bus();

    let hydros = vec![fixture_hydro(1), fixture_hydro(2)];

    let blocks: Vec<Block> = (0..n_blks)
        .map(|b| Block {
            index: b,
            name: format!("BLK{b}"),
            duration_hours: 372.0,
        })
        .collect();

    let stages: Vec<Stage> = vec![Stage {
        index: 0,
        id: 0,
        start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
        season_id: Some(0),
        blocks,
        block_mode: BlockMode::Parallel,
        state_config: StageStateConfig {
            storage: false,
            inflow_lags: false,
        },
        risk_config: StageRiskConfig::Expectation,
        scenario_config: ScenarioSourceConfig {
            branching_factor: 1,
            noise_method: NoiseMethod::Saa,
        },
    }];

    let load_models = vec![LoadModel {
        bus_id: EntityId(1),
        stage_id: 0,
        mean_mw: 100.0,
        std_mw: 0.0,
    }];

    let resolved_bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros,
            n_thermals: 0,
            n_lines: 0,
            n_pumping: 0,
            n_contracts,
            n_stages,
            k_max: 0,
        },
        &BoundsDefaults {
            hydro: default_hydro_bounds(),
            hydro_block: default_hydro_block_bounds(),
            thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
            thermal_block: ThermalBlockBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 100.0,
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
                max_mw: 500.0,
                price_per_mwh: 100.0,
            },
        },
    );
    let penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages,
        },
        &PenaltiesDefaults {
            hydro: default_hydro_penalties(),
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    );

    SystemBuilder::new()
        .buses(vec![bus])
        .hydros(hydros)
        .contracts(contracts)
        .stages(stages)
        .load_models(load_models)
        .bounds(resolved_bounds)
        .penalties(penalties)
        .build()
        .expect("system_with_contracts: valid system")
}

/// One import + one export contract (declared out of ID order) are exposed
/// ID-sorted on the ctx; `ctx.positions.contract` maps each id to its slot, and the
/// per-direction counts are derived by `contract_type`.
#[test]
fn build_template_build_ctx_contracts_counted_and_pos_mapped() {
    let contracts = vec![
        fixture_contract(20, ContractType::Export),
        fixture_contract(10, ContractType::Import),
    ];
    let system = system_with_contracts(contracts, 1);
    let hydro_result = PrepareHydroModelsResult::default_from_system(&system);
    let par_lp = PrecomputedPar::default();
    let resolved_params = empty_resolved_params();

    let (topology, resolved) = crate::test_support::resolved_layout_for(&system, &par_lp);
    let hydro_cell_index = HydroCellIndex::build(system.hydros());
    let time_value = build_time_value_for(&system);
    let study_dims = crate::test_support::build_study_dimensions(
        &system,
        InflowNonNegativityMethod::None,
        resolved.anticipated_plants.clone(),
        0,
    );
    let inputs = crate::test_support::resolve_lp_build_inputs(
        &system,
        &[],
        &hydro_result.production,
        &study_dims,
        &time_value,
        &hydro_cell_index,
        &resolved_params,
    );
    let ctx = super::build_template_build_ctx(
        &system,
        &par_lp,
        &hydro_result.production,
        &hydro_result.evaporation,
        &resolved.state,
        &topology,
        &inputs,
    );

    assert_eq!(ctx.contracts.len(), 2);
    let ids: Vec<i32> = ctx.contracts.iter().map(|c| c.id.0).collect();
    assert_eq!(
        ids,
        vec![10, 20],
        "ctx.contracts must be ID-sorted regardless of declaration order"
    );
    let n_import = ctx
        .contracts
        .iter()
        .filter(|c| c.contract_type == ContractType::Import)
        .count();
    let n_export = ctx
        .contracts
        .iter()
        .filter(|c| c.contract_type == ContractType::Export)
        .count();
    assert_eq!(n_import, 1);
    assert_eq!(n_export, 1);
    assert_eq!(ctx.positions.contract(EntityId(10)), Some(0));
    assert_eq!(ctx.positions.contract(EntityId(20)), Some(1));
    for (slot, contract) in ctx.contracts.iter().enumerate() {
        assert_eq!(
            ctx.positions.contract(contract.id),
            Some(slot),
            "positions.contract({:?}) must equal its slot in the sorted slice",
            contract.id
        );
    }
}

/// With `n_blks == 2` and one import + one export contract, `StageLayout::geometry`
/// populates each contract column range with `n_contracts * n_blks` columns:
/// import follows pumping, export follows import.
#[test]
fn stage_layout_geometry_populates_contract_ranges() {
    let contracts = vec![
        fixture_contract(10, ContractType::Import),
        fixture_contract(20, ContractType::Export),
    ];
    let system = system_with_contracts(contracts, 2);
    let hydro_result = PrepareHydroModelsResult::default_from_system(&system);
    let par_lp = PrecomputedPar::default();
    let resolved_params = empty_resolved_params();

    let (topology, resolved) = crate::test_support::resolved_layout_for(&system, &par_lp);
    let hydro_cell_index = HydroCellIndex::build(system.hydros());
    let time_value = build_time_value_for(&system);
    let study_dims = crate::test_support::build_study_dimensions(
        &system,
        InflowNonNegativityMethod::None,
        resolved.anticipated_plants.clone(),
        0,
    );
    let inputs = crate::test_support::resolve_lp_build_inputs(
        &system,
        &[],
        &hydro_result.production,
        &study_dims,
        &time_value,
        &hydro_cell_index,
        &resolved_params,
    );
    let ctx = super::build_template_build_ctx(
        &system,
        &par_lp,
        &hydro_result.production,
        &hydro_result.evaporation,
        &resolved.state,
        &topology,
        &inputs,
    );
    let stage = system
        .stages()
        .iter()
        .find(|s| s.id >= 0)
        .expect("one study stage");
    let layout = super::super::layout::StageLayout::new(&ctx, stage, 0);
    let geometry = layout.geometry.clone();

    assert_eq!(geometry.contract_import.len(), 2, "1 import * 2 blocks");
    assert_eq!(geometry.contract_export.len(), 2, "1 export * 2 blocks");
    assert_eq!(
        geometry.contract_import.start, layout.geometry.contract_import.start,
        "import range anchored at the layout import-block start"
    );
    assert_eq!(
        geometry.contract_export.start, geometry.contract_import.end,
        "export block immediately follows the import block"
    );
}

/// A contract-free system yields empty contract ranges anchored at the
/// pumping-end column (`start..start`, not `0..0`), leaving the prior column
/// layout byte-identical (parity-neutral).
#[test]
fn stage_layout_geometry_empty_contracts_are_pumping_end_anchored() {
    let system = system_with_contracts(vec![], 2);
    let hydro_result = PrepareHydroModelsResult::default_from_system(&system);
    let par_lp = PrecomputedPar::default();
    let resolved_params = empty_resolved_params();

    let (topology, resolved) = crate::test_support::resolved_layout_for(&system, &par_lp);
    let hydro_cell_index = HydroCellIndex::build(system.hydros());
    let time_value = build_time_value_for(&system);
    let study_dims = crate::test_support::build_study_dimensions(
        &system,
        InflowNonNegativityMethod::None,
        resolved.anticipated_plants.clone(),
        0,
    );
    let inputs = crate::test_support::resolve_lp_build_inputs(
        &system,
        &[],
        &hydro_result.production,
        &study_dims,
        &time_value,
        &hydro_cell_index,
        &resolved_params,
    );
    let ctx = super::build_template_build_ctx(
        &system,
        &par_lp,
        &hydro_result.production,
        &hydro_result.evaporation,
        &resolved.state,
        &topology,
        &inputs,
    );
    let stage = system
        .stages()
        .iter()
        .find(|s| s.id >= 0)
        .expect("one study stage");
    let layout = super::super::layout::StageLayout::new(&ctx, stage, 0);
    let col_pumping_end = layout.geometry.pumping_flow.end;
    let geometry = layout.geometry.clone();

    assert!(geometry.contract_import.is_empty());
    assert!(geometry.contract_export.is_empty());
    assert_eq!(
        geometry.contract_import.start, col_pumping_end,
        "empty import range anchors at the pumping-end column, not 0"
    );
    assert_eq!(
        geometry.contract_export.start, col_pumping_end,
        "empty export range anchors at the pumping-end column, not 0"
    );
}

/// A resolved-bounds count divergence (one contract entity, `n_contracts: 0`
/// in the bounds table) trips the `debug_assert_eq!` in
/// `build_template_build_ctx`.
#[test]
#[should_panic(expected = "resolved-bounds")]
fn build_template_build_ctx_contract_count_divergence_panics() {
    let bus = fixture_bus();
    let stages: Vec<Stage> = vec![Stage {
        index: 0,
        id: 0,
        start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
        season_id: Some(0),
        blocks: vec![Block {
            index: 0,
            name: "BLK0".to_string(),
            duration_hours: 744.0,
        }],
        block_mode: BlockMode::Parallel,
        state_config: StageStateConfig {
            storage: false,
            inflow_lags: false,
        },
        risk_config: StageRiskConfig::Expectation,
        scenario_config: ScenarioSourceConfig {
            branching_factor: 1,
            noise_method: NoiseMethod::Saa,
        },
    }];
    let load_models = vec![LoadModel {
        bus_id: EntityId(1),
        stage_id: 0,
        mean_mw: 100.0,
        std_mw: 0.0,
    }];
    let resolved_bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 0,
            n_thermals: 0,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: 1,
            k_max: 0,
        },
        &BoundsDefaults {
            hydro: default_hydro_bounds(),
            hydro_block: default_hydro_block_bounds(),
            thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
            thermal_block: ThermalBlockBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 100.0,
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
            n_stages: 1,
        },
        &PenaltiesDefaults {
            hydro: default_hydro_penalties(),
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    );
    let system = SystemBuilder::new()
        .buses(vec![bus])
        .contracts(vec![fixture_contract(1, ContractType::Import)])
        .stages(stages)
        .load_models(load_models)
        .bounds(resolved_bounds)
        .penalties(penalties)
        .build()
        .expect("valid system; the count mismatch is caught downstream");
    let hydro_result = PrepareHydroModelsResult::default_from_system(&system);
    let par_lp = PrecomputedPar::default();
    let resolved_params = empty_resolved_params();

    let (topology, resolved) = crate::test_support::resolved_layout_for(&system, &par_lp);
    let hydro_cell_index = HydroCellIndex::build(system.hydros());
    let time_value = build_time_value_for(&system);
    let study_dims = crate::test_support::build_study_dimensions(
        &system,
        InflowNonNegativityMethod::None,
        resolved.anticipated_plants.clone(),
        0,
    );
    let inputs = crate::test_support::resolve_lp_build_inputs(
        &system,
        &[],
        &hydro_result.production,
        &study_dims,
        &time_value,
        &hydro_cell_index,
        &resolved_params,
    );
    let _ = super::build_template_build_ctx(
        &system,
        &par_lp,
        &hydro_result.production,
        &hydro_result.evaporation,
        &resolved.state,
        &topology,
        &inputs,
    );
}

/// `build_template_build_ctx` populates anticipated metadata for a
/// system with `T_a`(K=2), `T_b`(no anticipated), `T_c`(K=3).
///
/// Expected: `n_anticipated`=2, `k_max`=3, `anticipated_lead_stages`=[2,3],
/// `anticipated_plants`=[0,2].
#[test]
fn build_template_build_ctx_populates_anticipated_metadata() {
    let thermals = vec![
        Thermal {
            id: EntityId(1),
            name: "T_a".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(1),
            entry_stage_id: None,
            exit_stage_id: None,
            cost_per_mwh: 10.0,
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            anticipated_config: Some(AnticipatedConfig::LeadStages(2)),
        },
        Thermal {
            id: EntityId(2),
            name: "T_b".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(1),
            entry_stage_id: None,
            exit_stage_id: None,
            cost_per_mwh: 20.0,
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            anticipated_config: None,
        },
        Thermal {
            id: EntityId(3),
            name: "T_c".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(1),
            entry_stage_id: None,
            exit_stage_id: None,
            cost_per_mwh: 30.0,
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            anticipated_config: Some(AnticipatedConfig::LeadStages(3)),
        },
    ];
    let system = system_with_thermals(thermals);
    let hydro_result = PrepareHydroModelsResult::default_from_system(&system);
    let par_lp = PrecomputedPar::default();
    let resolved_params = empty_resolved_params();

    let (topology, resolved) = crate::test_support::resolved_layout_for(&system, &par_lp);
    let hydro_cell_index = HydroCellIndex::build(system.hydros());
    let time_value = build_time_value_for(&system);
    let study_dims = crate::test_support::build_study_dimensions(
        &system,
        InflowNonNegativityMethod::None,
        resolved.anticipated_plants.clone(),
        0,
    );
    let inputs = crate::test_support::resolve_lp_build_inputs(
        &system,
        &[],
        &hydro_result.production,
        &study_dims,
        &time_value,
        &hydro_cell_index,
        &resolved_params,
    );
    let ctx = super::build_template_build_ctx(
        &system,
        &par_lp,
        &hydro_result.production,
        &hydro_result.evaporation,
        &resolved.state,
        &topology,
        &inputs,
    );

    assert_eq!(ctx.study_dims.anticipated_plants.len(), 2, "n_anticipated");
    assert_eq!(
        ctx.state
            .anticipated_resolution
            .ring_size(&ctx.state.anticipated_lead_stages),
        3,
        "k_max"
    );
    assert_eq!(
        ctx.state.anticipated_lead_stages,
        vec![2, 3],
        "anticipated_lead_stages"
    );
    assert_eq!(
        ctx.study_dims
            .anticipated_plants
            .thermals()
            .collect::<Vec<_>>(),
        vec![ThermalSys::new(0), ThermalSys::new(2)],
        "anticipated_plants"
    );
}

/// `build_template_build_ctx` returns zeroed metadata when no
/// thermal has `anticipated_config`.
#[test]
fn build_template_build_ctx_zero_anticipated_when_none() {
    let thermals = vec![
        Thermal {
            id: EntityId(1),
            name: "T1".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(1),
            entry_stage_id: None,
            exit_stage_id: None,
            cost_per_mwh: 10.0,
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            anticipated_config: None,
        },
        Thermal {
            id: EntityId(2),
            name: "T2".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(1),
            entry_stage_id: None,
            exit_stage_id: None,
            cost_per_mwh: 20.0,
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            anticipated_config: None,
        },
    ];
    let system = system_with_thermals(thermals);
    let hydro_result = PrepareHydroModelsResult::default_from_system(&system);
    let par_lp = PrecomputedPar::default();
    let resolved_params = empty_resolved_params();

    let (topology, resolved) = crate::test_support::resolved_layout_for(&system, &par_lp);
    let hydro_cell_index = HydroCellIndex::build(system.hydros());
    let time_value = build_time_value_for(&system);
    let study_dims = crate::test_support::build_study_dimensions(
        &system,
        InflowNonNegativityMethod::None,
        resolved.anticipated_plants.clone(),
        0,
    );
    let inputs = crate::test_support::resolve_lp_build_inputs(
        &system,
        &[],
        &hydro_result.production,
        &study_dims,
        &time_value,
        &hydro_cell_index,
        &resolved_params,
    );
    let ctx = super::build_template_build_ctx(
        &system,
        &par_lp,
        &hydro_result.production,
        &hydro_result.evaporation,
        &resolved.state,
        &topology,
        &inputs,
    );

    assert_eq!(ctx.study_dims.anticipated_plants.len(), 0, "n_anticipated");
    assert_eq!(
        ctx.state
            .anticipated_resolution
            .ring_size(&ctx.state.anticipated_lead_stages),
        0,
        "k_max"
    );
    assert!(
        ctx.state.anticipated_lead_stages.is_empty(),
        "anticipated_lead_stages"
    );
    assert!(
        ctx.study_dims.anticipated_plants.len() == 0,
        "anticipated_plants"
    );
}

// ── Real declaration-order-invariance probe ──

/// Build a 5-stage 3-thermal system used by the order-invariance probe.
///
/// Three thermals (canonical EntityId order, since `SystemBuilder::build`
/// sorts by `EntityId`):
/// - `id=1`: anticipated K=2, max=120 MW, cost=50 $/MWh
/// - `id=2`: anticipated K=3, max=80 MW, cost=40 $/MWh
/// - `id=3`: standard thermal (no anticipation), max=200 MW, cost=500 $/MWh
///
/// `ResolvedBounds` is populated with per-thermal stage costs/limits matching
/// the per-thermal declarations (the default `BoundsDefaults::thermal` is uniform,
/// so a probe that relied on defaults would be trivial — distinct per-thermal
/// stage data is required to expose any latent order-dependence in the LP fill).
///
/// `n_stages = 5` ensures both anticipated decisions are active at `stage_idx=0`
/// (strict gate `t + K_i < n_stages` -> `2 < 5` and `3 < 5`).
fn anticipated_invariance_system() -> cobre_core::System {
    let thermals = vec![
        Thermal {
            id: EntityId(1),
            name: "T_ant_k2".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(1),
            entry_stage_id: None,
            exit_stage_id: None,
            cost_per_mwh: 50.0,
            min_generation_mw: 0.0,
            max_generation_mw: 120.0,
            anticipated_config: Some(AnticipatedConfig::LeadStages(2)),
        },
        Thermal {
            id: EntityId(2),
            name: "T_ant_k3".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(1),
            entry_stage_id: None,
            exit_stage_id: None,
            cost_per_mwh: 40.0,
            min_generation_mw: 0.0,
            max_generation_mw: 80.0,
            anticipated_config: Some(AnticipatedConfig::LeadStages(3)),
        },
        Thermal {
            id: EntityId(3),
            name: "T_backup".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(1),
            entry_stage_id: None,
            exit_stage_id: None,
            cost_per_mwh: 500.0,
            min_generation_mw: 0.0,
            max_generation_mw: 200.0,
            anticipated_config: None,
        },
    ];

    let n_thermals = thermals.len();
    let n_stages = 5_usize;
    let k_max = 3_usize;

    let bus = Bus {
        id: EntityId(1),
        name: "B1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 1000.0,
        }],
        excess_cost: 0.0,
    };

    let stages: Vec<Stage> = (0..n_stages)
        .map(|i| Stage {
            index: i,
            id: i as i32,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: Some(0),
            blocks: vec![Block {
                index: 0,
                name: "BLK0".to_string(),
                duration_hours: 744.0,
            }],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: false,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        })
        .collect();

    let load_models: Vec<LoadModel> = (0..n_stages)
        .map(|s| LoadModel {
            bus_id: EntityId(1),
            stage_id: s as i32,
            mean_mw: 150.0,
            std_mw: 0.0,
        })
        .collect();

    let mut resolved_bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 0,
            n_thermals,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages,
            k_max,
        },
        &BoundsDefaults {
            hydro: default_hydro_bounds(),
            hydro_block: default_hydro_block_bounds(),
            thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
            thermal_block: ThermalBlockBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 100.0,
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

    // The bounds table is indexed [thermal_idx][stage_idx] with a stage axis of
    // length `n_stages + k_max` (delivery-stage padding).
    let stage_axis_len = resolved_bounds.thermal_stage_axis_len();
    for t_idx in 0..n_thermals {
        for s_idx in 0..stage_axis_len {
            let (max_generation_mw, cost_per_mwh) = match t_idx {
                0 => (120.0, 50.0),
                1 => (80.0, 40.0),
                2 => (200.0, 500.0),
                _ => unreachable!("only 3 thermals"),
            };
            resolved_bounds
                .thermal_block_base_mut(t_idx, s_idx)
                .max_generation_mw = max_generation_mw;
            resolved_bounds
                .thermal_bounds_mut(t_idx, s_idx)
                .cost_per_mwh = cost_per_mwh;
        }
    }

    let penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: 0,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages,
        },
        &PenaltiesDefaults {
            hydro: default_hydro_penalties(),
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    );

    SystemBuilder::new()
        .buses(vec![bus])
        .thermals(thermals)
        .stages(stages)
        .load_models(load_models)
        .bounds(resolved_bounds)
        .penalties(penalties)
        .build()
        .expect("anticipated_invariance_system: valid system")
}

/// Assert two `StageTemplate`s are bit-for-bit equivalent under the swap-(0,1)
/// permutation on anticipated-decision, anticipated-state (slot-major), and
/// anticipated-fishing columns/rows.
#[expect(
    clippy::too_many_arguments,
    reason = "the equivalence check takes each varied input explicitly"
)]
fn assert_lp_equivalence_after_anticipated_swap(
    tpl_a: &cobre_solver::StageTemplate,
    tpl_b: &cobre_solver::StageTemplate,
    dec_start_a: usize,
    dec_start_b: usize,
    state_start_a: usize,
    state_start_b: usize,
    slots_out_start_a: usize,
    slots_out_start_b: usize,
    n_ant: usize,
    k_max: usize,
    fish_start_a: usize,
    fish_start_b: usize,
    n_fish_rows: usize,
    def_row_start_a: usize,
    def_row_start_b: usize,
    n_def_rows: usize,
    slot_def_row_start_a: usize,
    slot_def_row_start_b: usize,
    slot_row_pos_a: &[Option<usize>],
    slot_row_pos_b: &[Option<usize>],
    stage_idx: usize,
) {
    assert_eq!(
        tpl_a.num_cols, tpl_b.num_cols,
        "stage {stage_idx}: num_cols"
    );
    assert_eq!(
        tpl_a.num_rows, tpl_b.num_rows,
        "stage {stage_idx}: num_rows"
    );
    assert_eq!(tpl_a.num_nz, tpl_b.num_nz, "stage {stage_idx}: num_nz");
    assert_eq!(n_ant, 2, "this helper requires n_ant == 2");

    // col_perm[j] = i: tpl_a column i corresponds to tpl_b column j.
    let mut col_perm: Vec<usize> = (0..tpl_a.num_cols).collect();
    col_perm[dec_start_b] = dec_start_a + 1;
    col_perm[dec_start_b + 1] = dec_start_a;
    // Slot-major layout: column for slot s, plant p = state_start + s * n_ant + p.
    for s in 0..k_max {
        col_perm[state_start_b + s * n_ant] = state_start_a + s * n_ant + 1;
        col_perm[state_start_b + s * n_ant + 1] = state_start_a + s * n_ant;
    }
    for s in 0..k_max {
        col_perm[slots_out_start_b + s * n_ant] = slots_out_start_a + s * n_ant + 1;
        col_perm[slots_out_start_b + s * n_ant + 1] = slots_out_start_a + s * n_ant;
    }

    // State pinning uses column bounds, not equality rows, so no state-fixing
    // rows are permuted.
    let mut row_perm: Vec<usize> = (0..tpl_a.num_rows).collect();
    if n_fish_rows == 2 {
        row_perm[fish_start_b] = fish_start_a + 1;
        row_perm[fish_start_b + 1] = fish_start_a;
    }
    if n_fish_rows == 1 {
        row_perm[fish_start_b] = fish_start_a;
    }
    if n_def_rows == 2 {
        row_perm[def_row_start_b] = def_row_start_a + 1;
        row_perm[def_row_start_b + 1] = def_row_start_a;
    }
    if n_def_rows == 1 {
        row_perm[def_row_start_b] = def_row_start_a;
    }
    // slot_row_pos_b's global index slot*n_ant + plant maps to slot_row_pos_a's
    // slot*n_ant + (1 - plant) (the column loops' plant-swap); both must agree on
    // reachability — swapping local labels never changes which physical
    // (slot, plant) pair is in-horizon.
    for (g_b, pos_b) in slot_row_pos_b.iter().enumerate() {
        let Some(pos_b) = *pos_b else { continue };
        let slot = g_b / n_ant;
        let plant = g_b % n_ant;
        let g_a = slot * n_ant + (1 - plant);
        let pos_a = slot_row_pos_a[g_a].unwrap_or_else(|| {
            panic!(
                "stage {stage_idx}: slot_row_pos_a[{g_a}] must be reachable to \
                 match slot_row_pos_b[{g_b}]"
            )
        });
        row_perm[slot_def_row_start_b + pos_b] = slot_def_row_start_a + pos_a;
    }

    for j in 0..tpl_a.num_cols {
        let a = col_perm[j];
        assert_eq!(
            tpl_a.col_lower[a].to_bits(),
            tpl_b.col_lower[j].to_bits(),
            "stage {stage_idx}: col_lower mismatch at permuted col {j} <- {a}"
        );
        assert_eq!(
            tpl_a.col_upper[a].to_bits(),
            tpl_b.col_upper[j].to_bits(),
            "stage {stage_idx}: col_upper mismatch at permuted col {j} <- {a}"
        );
        assert_eq!(
            tpl_a.objective[a].to_bits(),
            tpl_b.objective[j].to_bits(),
            "stage {stage_idx}: objective mismatch at permuted col {j} <- {a}"
        );
    }
    for i in 0..tpl_a.num_rows {
        let ra = row_perm[i];
        assert_eq!(
            tpl_a.row_lower[ra].to_bits(),
            tpl_b.row_lower[i].to_bits(),
            "stage {stage_idx}: row_lower mismatch at permuted row {i} <- {ra}"
        );
        assert_eq!(
            tpl_a.row_upper[ra].to_bits(),
            tpl_b.row_upper[i].to_bits(),
            "stage {stage_idx}: row_upper mismatch at permuted row {i} <- {ra}"
        );
    }

    let dense_a = csc_to_dense(tpl_a);
    let dense_b = csc_to_dense(tpl_b);
    for i in 0..tpl_a.num_rows {
        for j in 0..tpl_a.num_cols {
            let va = dense_a[row_perm[i]][col_perm[j]];
            let vb = dense_b[i][j];
            assert_eq!(
                va.to_bits(),
                vb.to_bits(),
                "stage {stage_idx}: coefficient mismatch at row {i} col {j} \
                     (permuted from row {} col {} in tpl_a)",
                row_perm[i],
                col_perm[j],
            );
        }
    }
}

/// Expand a CSC `StageTemplate` to a dense `Vec<Vec<f64>>`.
fn csc_to_dense(tpl: &cobre_solver::StageTemplate) -> Vec<Vec<f64>> {
    let mut dense = vec![vec![0.0_f64; tpl.num_cols]; tpl.num_rows];
    for j in 0..tpl.num_cols {
        let start = tpl.col_starts[j] as usize;
        let end = tpl.col_starts[j + 1] as usize;
        for k in start..end {
            let row = tpl.row_indices[k] as usize;
            dense[row][j] = tpl.values[k];
        }
    }
    dense
}

/// Invariance probe at the LP-construction layer: the templates from
/// [`build_single_stage_template`] are equivalent under a permutation of the
/// `anticipated_plants` / `anticipated_lead_stages` arrays.
///
/// A full-`System` declaration-order test is a tautology here — `SystemBuilder::build`
/// sorts by `EntityId`, so both orderings present identical canonical input and
/// prove only determinism, not invariance (that canonicalization is covered by the
/// `cobre-core` proptest `build_canonical_order_invariant_under_input_permutation`).
/// This test constructs the permuted `TemplateBuildCtx` directly, hitting the path
/// the canonical sort otherwise masks.
#[test]
fn lp_template_invariant_under_anticipated_index_permutation() {
    let system = anticipated_invariance_system();
    assert_eq!(system.thermals().len(), 3);
    assert_eq!(system.thermals()[0].id.0, 1);
    assert_eq!(system.thermals()[1].id.0, 2);
    assert_eq!(system.thermals()[2].id.0, 3);

    let hydro_result = PrepareHydroModelsResult::default_from_system(&system);
    let par_lp = PrecomputedPar::default();
    let resolved_params = empty_resolved_params();

    let (topology, resolved) = crate::test_support::resolved_layout_for(&system, &par_lp);
    let hydro_cell_index = HydroCellIndex::build(system.hydros());
    let time_value = build_time_value_for(&system);
    let study_dims_a = crate::test_support::build_study_dimensions(
        &system,
        InflowNonNegativityMethod::None,
        resolved.anticipated_plants.clone(),
        0,
    );
    let inputs_a = crate::test_support::resolve_lp_build_inputs(
        &system,
        &[],
        &hydro_result.production,
        &study_dims_a,
        &time_value,
        &hydro_cell_index,
        &resolved_params,
    );
    let ctx_a = super::build_template_build_ctx(
        &system,
        &par_lp,
        &hydro_result.production,
        &hydro_result.evaporation,
        &resolved.state,
        &topology,
        &inputs_a,
    );

    assert_eq!(ctx_a.study_dims.anticipated_plants.len(), 2);
    assert_eq!(
        ctx_a
            .state
            .anticipated_resolution
            .ring_size(&ctx_a.state.anticipated_lead_stages),
        3
    );
    assert_eq!(
        ctx_a
            .study_dims
            .anticipated_plants
            .thermals()
            .collect::<Vec<_>>(),
        vec![ThermalSys::new(0), ThermalSys::new(1)]
    );
    assert_eq!(ctx_a.state.anticipated_lead_stages, vec![2, 3]);

    // Both anticipated arrays must be permuted in lockstep to preserve the
    // (thermal_idx, K_i) pairing.
    let ctx_b_anticipated_plants = AnticipatedPlants::from_positions_for_test(
        vec![
            ctx_a
                .study_dims
                .anticipated_plants
                .thermal_of(AnticipatedLocal::new(1)),
            ctx_a
                .study_dims
                .anticipated_plants
                .thermal_of(AnticipatedLocal::new(0)),
        ],
        vec![
            ctx_a.study_dims.anticipated_plants.windows()[1],
            ctx_a.study_dims.anticipated_plants.windows()[0],
        ],
    );
    let ctx_b_lead_stages = vec![
        ctx_a.state.anticipated_lead_stages[1],
        ctx_a.state.anticipated_lead_stages[0],
    ];
    let ctx_b_resolution = AnticipatedResolution {
        per_plant: vec![
            ctx_a.state.anticipated_resolution.per_plant[1].clone(),
            ctx_a.state.anticipated_resolution.per_plant[0].clone(),
        ],
    };
    let max_par_order = par_lp.max_order();
    let effective_lag_counts: Vec<usize> = if max_par_order > 0 {
        (0..resolved.state.hydro_count)
            .map(|h| {
                if h < par_lp.n_hydros() {
                    par_lp.effective_lag_count(h)
                } else {
                    max_par_order
                }
            })
            .collect()
    } else {
        vec![0; resolved.state.hydro_count]
    };
    let ctx_b_state = StateSpace::new(
        resolved.state.hydro_count,
        max_par_order,
        topology.column_order.clone(),
        ctx_b_lead_stages,
        ctx_b_resolution,
        &effective_lag_counts,
    );
    let study_dims_b = crate::test_support::build_study_dimensions(
        &system,
        InflowNonNegativityMethod::None,
        ctx_b_anticipated_plants.clone(),
        0,
    );
    let inputs_b = crate::test_support::resolve_lp_build_inputs(
        &system,
        &[],
        &hydro_result.production,
        &study_dims_b,
        &time_value,
        &hydro_cell_index,
        &resolved_params,
    );
    let ctx_b = super::build_template_build_ctx(
        &system,
        &par_lp,
        &hydro_result.production,
        &hydro_result.evaporation,
        &ctx_b_state,
        &topology,
        &inputs_b,
    );

    assert_eq!(
        ctx_b
            .study_dims
            .anticipated_plants
            .thermals()
            .collect::<Vec<_>>(),
        vec![ThermalSys::new(1), ThermalSys::new(0)]
    );
    assert_eq!(ctx_b.state.anticipated_lead_stages, vec![3, 2]);

    let study_stages: Vec<_> = system.stages().iter().filter(|s| s.id >= 0).collect();

    // Stages [0, 2, 3] straddle the active-decision boundary; the always-active
    // fishing predicate keeps the fishing-row count constant across them.
    for stage_idx in [0_usize, 2, 3] {
        let stage = study_stages[stage_idx];

        let tpl_a = super::build_single_stage_template(&ctx_a, stage, stage_idx).template;
        let tpl_b = super::build_single_stage_template(&ctx_b, stage, stage_idx).template;

        // Both templates share num_cols/num_rows: the layout depends only on
        // n_anticipated and k_max, unchanged by the swap.
        let layout_a = super::super::layout::StageLayout::new(&ctx_a, stage, stage_idx);
        let layout_b = super::super::layout::StageLayout::new(&ctx_b, stage, stage_idx);

        assert_eq!(
            layout_a.geometry.anticipated_decision.start,
            layout_b.geometry.anticipated_decision.start,
            "stage {stage_idx}: dec_start"
        );
        assert_eq!(
            layout_a.state.commit_in.start, layout_b.state.commit_in.start,
            "stage {stage_idx}: state_start"
        );
        assert_eq!(
            layout_a.anticipated.fishing_rows.start, layout_b.anticipated.fishing_rows.start,
            "stage {stage_idx}: fish_start"
        );
        assert_eq!(
            layout_a.anticipated.fishing_rows.len(),
            layout_b.anticipated.fishing_rows.len(),
            "stage {stage_idx}: n_fish_rows"
        );

        assert_lp_equivalence_after_anticipated_swap(
            &tpl_a,
            &tpl_b,
            layout_a.geometry.anticipated_decision.start,
            layout_b.geometry.anticipated_decision.start,
            layout_a.state.commit_in.start,
            layout_b.state.commit_in.start,
            layout_a.state.commit_out.start,
            layout_b.state.commit_out.start,
            ctx_a.study_dims.anticipated_plants.len(),
            ctx_a
                .state
                .anticipated_resolution
                .ring_size(&ctx_a.state.anticipated_lead_stages),
            layout_a.anticipated.fishing_rows.start,
            layout_b.anticipated.fishing_rows.start,
            layout_a.anticipated.fishing_rows.len(),
            layout_a.anticipated.state_out_def_rows.start,
            layout_b.anticipated.state_out_def_rows.start,
            layout_a.anticipated.state_out_def_rows.len(),
            layout_a.anticipated.slot_definition_rows.start,
            layout_b.anticipated.slot_definition_rows.start,
            &layout_a.anticipated.anticipated_slot_row_pos,
            &layout_b.anticipated.anticipated_slot_row_pos,
            stage_idx,
        );
    }
}

// ── StageTemplates::empty ──────────────────────────────────────────────────

/// Pins the all-empty shape the empty-study early return relies on.
#[test]
fn stage_templates_empty_is_all_empty() {
    let empty = super::StageTemplates::empty(DEFAULT_COST_SCALE_FACTOR);

    assert_eq!(empty.n_load_buses(), 0, "n_load_buses must be 0");

    assert!(empty.templates.is_empty(), "templates");
    assert!(empty.state_boxes().is_empty(), "state_boxes");
    assert!(
        empty.block_hours_per_stage.is_empty(),
        "block_hours_per_stage"
    );
    assert!(empty.load_bus_indices.is_empty(), "load_bus_indices");
    assert!(
        empty.generic_constraint_row_entries.is_empty(),
        "generic_constraint_row_entries"
    );
    assert!(empty.diversion_upstream.is_empty(), "diversion_upstream");
    assert!(
        empty.hydro_productivities_per_stage.is_empty(),
        "hydro_productivities_per_stage"
    );
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "state_boxes read before postprocess_templates filled them")]
fn state_boxes_read_before_postprocess_panics() {
    let mut unfilled = super::StageTemplates::empty(DEFAULT_COST_SCALE_FACTOR);
    unfilled
        .templates
        .push(crate::test_support::transit_bucket_only_template(1, 1));

    let _ = unfilled.state_boxes();
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "set_state_boxes needs one state box per stage")]
fn set_state_boxes_rejects_a_box_count_other_than_the_stage_count() {
    let mut templates = super::StageTemplates::empty(DEFAULT_COST_SCALE_FACTOR);
    templates
        .templates
        .push(crate::test_support::transit_bucket_only_template(1, 1));

    templates.set_state_boxes(Vec::new());
}

// ── theta's coefficient is the stage's one-step discount factor ────────────

/// Build a 3-stage thermals-only system carrying a non-zero global annual
/// discount rate. Empty `transitions` means every stage falls back to the
/// global rate, so the per-stage factors are all < 1.0 and the cumulative
/// vector compounds below 1.0.
fn discounted_multi_stage_system() -> cobre_core::System {
    discounted_multi_stage_system_with_post_study(None)
}

/// [`discounted_multi_stage_system`], optionally attaching `post_study`
/// stages — the single owner of this fixture's shape so the delivery-axis
/// tests share the same non-trivial discount rate.
fn discounted_multi_stage_system_with_post_study(
    post_study: Option<PostStudyStages>,
) -> cobre_core::System {
    use cobre_core::{HorizonGraph, PolicyGraphType};

    let n_stages = 3_usize;
    let thermals = vec![Thermal {
        id: EntityId(1),
        name: "T1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        entry_stage_id: None,
        exit_stage_id: None,
        cost_per_mwh: 10.0,
        min_generation_mw: 0.0,
        max_generation_mw: 100.0,
        anticipated_config: None,
    }];
    let n_thermals = thermals.len();

    let bus = fixture_bus();

    let stages: Vec<Stage> = (0..n_stages)
        .map(|i| Stage {
            index: i,
            id: i as i32,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: Some(0),
            blocks: vec![Block {
                index: 0,
                name: "BLK0".to_string(),
                duration_hours: 744.0,
            }],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: false,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        })
        .collect();

    let load_models: Vec<LoadModel> = (0..n_stages)
        .map(|s| LoadModel {
            bus_id: EntityId(1),
            stage_id: s as i32,
            mean_mw: 100.0,
            std_mw: 0.0,
        })
        .collect();

    let resolved_bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 0,
            n_thermals,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages,
            k_max: 0,
        },
        &BoundsDefaults {
            hydro: default_hydro_bounds(),
            hydro_block: default_hydro_block_bounds(),
            thermal: ThermalStageBounds { cost_per_mwh: 10.0 },
            thermal_block: ThermalBlockBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 100.0,
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
            n_stages,
        },
        &PenaltiesDefaults {
            hydro: default_hydro_penalties(),
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    );

    let policy_graph = HorizonGraph {
        stage_discount_rate_overrides: std::collections::BTreeMap::new(),
        graph_type: PolicyGraphType::FiniteHorizon,
        annual_discount_rate: 0.10,
        transitions: Vec::new(),
        nodes: Vec::new(),
        season_map: None,
    };

    SystemBuilder::new()
        .buses(vec![bus])
        .thermals(thermals)
        .stages(stages)
        .load_models(load_models)
        .bounds(resolved_bounds)
        .penalties(penalties)
        .policy_graph(policy_graph)
        .post_study_stages(post_study)
        .build()
        .expect("discounted_multi_stage_system: valid system")
}

/// Building the templates gives every stage's theta coefficient the one-step
/// discount factor `TimeValue` resolves from the system's non-zero annual
/// discount rate.
#[test]
fn built_stage_templates_carry_the_one_step_discount_on_theta() {
    let system = discounted_multi_stage_system();
    let hydro_result = PrepareHydroModelsResult::default_from_system(&system);
    let par_lp = PrecomputedPar::default();
    let resolved_params = empty_resolved_params();
    let (topology, layout) = crate::test_support::resolved_layout_for(&system, &par_lp);
    let hydro_cell_index = HydroCellIndex::build(system.hydros());

    let time_value = build_time_value_for(&system);
    let study_dims = crate::test_support::build_study_dimensions(
        &system,
        InflowNonNegativityMethod::None,
        layout.anticipated_plants.clone(),
        0,
    );
    let inputs = crate::test_support::resolve_lp_build_inputs(
        &system,
        &[],
        &hydro_result.production,
        &study_dims,
        &time_value,
        &hydro_cell_index,
        &resolved_params,
    );
    let templates = super::build_stage_templates(
        &system,
        &par_lp,
        &hydro_result.production,
        &hydro_result.evaporation,
        &layout.state,
        &topology,
        inputs,
    );

    let discount_factors = time_value.discount_factors();
    assert!(
        discount_factors.iter().any(|&d| d < 1.0),
        "the 0.10 annual rate must give at least one one-step factor below 1.0, got {discount_factors:?}"
    );
    for (t, template) in templates.templates.iter().enumerate() {
        assert_eq!(
            template.objective[layout.state.theta].to_bits(),
            discount_factors[t].to_bits(),
            "stage {t}: theta's objective must be the stage's one-step discount factor"
        );
    }

    let cumulative = time_value.cumulative_discount_factors();
    assert_eq!(
        cumulative.len(),
        templates.templates.len(),
        "cumulative_discount_factors length must equal templates.len()"
    );
    assert_eq!(
        cumulative[0], 1.0,
        "cumulative_discount_factors[0] is the present value (1.0)"
    );
    assert!(
        cumulative.iter().any(|&d| d < 1.0),
        "cumulative factors must drop below 1.0, got {cumulative:?}"
    );
    assert!(
        cumulative[cumulative.len() - 1] < 1.0,
        "the final cumulative factor must be discounted below 1.0, got {}",
        cumulative[cumulative.len() - 1]
    );
}

/// Theta's coefficient lands on `state.theta`, never a hand re-derivation from
/// `n_state`/`n_hydros`: this fixture's commitment-hold region (one anticipated
/// thermal, `k_max = 1`) shifts `theta` off both (`n_state == 1`,
/// `n_hydros == 0`, `theta == 2`), so a wrong re-derivation would silently
/// discount the wrong column. Two builds that differ only in the one-step
/// factors may differ in theta and in the anticipated decision's price (which
/// carries its relative delivery discount), and nowhere else.
#[test]
fn theta_discount_lands_on_the_state_theta_column_with_anticipated_thermals() {
    let system = anticipated_lead_config_system(2, 744.0, AnticipatedConfig::LeadStages(1), 1);
    let hydro_result = PrepareHydroModelsResult::default_from_system(&system);
    let par_lp = PrecomputedPar::default();
    let resolved_params = empty_resolved_params();
    let (topology, layout) = crate::test_support::resolved_layout_for(&system, &par_lp);
    let hydro_cell_index = HydroCellIndex::build(system.hydros());
    let study_dims = crate::test_support::build_study_dimensions(
        &system,
        InflowNonNegativityMethod::None,
        layout.anticipated_plants.clone(),
        0,
    );
    assert_eq!(
        layout.state.theta, 2,
        "fixture sanity: theta must sit past commit_out/commit_in"
    );
    assert_eq!(
        layout.state.n_state, 1,
        "fixture sanity: one commitment slot"
    );

    let build = |discount_factors: &[f64]| {
        let time_value = TimeValue::from_parts(
            discount_factors.to_vec(),
            compute_cumulative_discount_factors(discount_factors),
            vec![744.0; discount_factors.len()],
            vec![0, 1],
            PostStudyResolved::default(),
        );
        let inputs = crate::test_support::resolve_lp_build_inputs(
            &system,
            &[],
            &hydro_result.production,
            &study_dims,
            &time_value,
            &hydro_cell_index,
            &resolved_params,
        );
        super::build_stage_templates(
            &system,
            &par_lp,
            &hydro_result.production,
            &hydro_result.evaporation,
            &layout.state,
            &topology,
            inputs,
        )
    };

    let discount_factors = [0.6_f64, 0.3_f64];
    let undiscounted = build(&[1.0, 1.0]);
    let discounted = build(&discount_factors);

    let theta = layout.state.theta;
    for (t, &d) in discount_factors.iter().enumerate() {
        assert_eq!(
            discounted.templates[t].objective[theta].to_bits(),
            d.to_bits(),
            "stage {t}: theta's objective must be the stage's one-step discount factor"
        );
        let decision_col =
            discounted.geometry_per_stage[t].anticipated_decision_col(AnticipatedLocal::new(0));
        for j in 0..discounted.templates[t].num_cols {
            if j == theta || j == decision_col {
                continue;
            }
            assert_eq!(
                discounted.templates[t].objective[j].to_bits(),
                undiscounted.templates[t].objective[j].to_bits(),
                "stage {t} col {j}: only theta and the decision's delivery discount may move with the factors"
            );
        }
    }
}

// ── Delivery-axis extended vectors (delivery_stage_ids / delivery_total_hours /
// time_value) ────────────────────────────────────────────────────────────

/// Two post-study stages following the discounted 3-stage study horizon,
/// mirroring [`cobre_core::model::post_study`]'s doc fixture. No thermal
/// bounds: these tests exercise only the delivery-vector concatenation, not
/// the per-thermal cost/bounds lookup.
fn two_post_study_stages() -> PostStudyStages {
    PostStudyStages {
        stages: vec![
            PostStudyStage {
                start_date: NaiveDate::from_ymd_opt(2024, 4, 1).unwrap(),
                duration_hours: 720.0,
            },
            PostStudyStage {
                start_date: NaiveDate::from_ymd_opt(2024, 5, 1).unwrap(),
                duration_hours: 744.0,
            },
        ],
        thermal_bounds: vec![],
    }
}

/// With no `post_study_stages` declared, `delivery_stage_ids` is element-wise
/// identical to `study_stage_ids` — `n_post == 0` collapses the concatenation
/// to a no-op.
#[test]
fn delivery_stage_ids_equals_study_stage_ids_with_no_post_study() {
    let system = discounted_multi_stage_system();
    let hydro_result = PrepareHydroModelsResult::default_from_system(&system);
    let par_lp = PrecomputedPar::default();
    let resolved_params = empty_resolved_params();
    let (topology, resolved) = crate::test_support::resolved_layout_for(&system, &par_lp);
    let hydro_cell_index = HydroCellIndex::build(system.hydros());
    let time_value = build_time_value_for(&system);
    let study_dims = crate::test_support::build_study_dimensions(
        &system,
        InflowNonNegativityMethod::None,
        resolved.anticipated_plants.clone(),
        0,
    );
    let inputs = crate::test_support::resolve_lp_build_inputs(
        &system,
        &[],
        &hydro_result.production,
        &study_dims,
        &time_value,
        &hydro_cell_index,
        &resolved_params,
    );
    let ctx = super::build_template_build_ctx(
        &system,
        &par_lp,
        &hydro_result.production,
        &hydro_result.evaporation,
        &resolved.state,
        &topology,
        &inputs,
    );

    let study_stage_ids: Vec<i32> = system
        .stages()
        .iter()
        .filter(|s| s.id >= 0)
        .map(|s| s.id)
        .collect();
    assert_eq!(ctx.time_value.delivery_stage_ids(), study_stage_ids);
}

/// Three study stages (ids `[0, 1, 2]`) plus two post-study stages continue
/// `delivery_stage_ids` with a synthetic, strictly increasing tail
/// (`max(study_stage_ids) + 1 ..`), never `post_study_calendar_stages`'s own
/// `Stage::id` (which restarts at `0`).
#[test]
fn delivery_stage_ids_continue_the_horizon_with_synthetic_ids() {
    let system = discounted_multi_stage_system_with_post_study(Some(two_post_study_stages()));
    let hydro_result = PrepareHydroModelsResult::default_from_system(&system);
    let par_lp = PrecomputedPar::default();
    let resolved_params = empty_resolved_params();
    let (topology, resolved) = crate::test_support::resolved_layout_for(&system, &par_lp);
    let hydro_cell_index = HydroCellIndex::build(system.hydros());
    let time_value = build_time_value_for(&system);
    let study_dims = crate::test_support::build_study_dimensions(
        &system,
        InflowNonNegativityMethod::None,
        resolved.anticipated_plants.clone(),
        0,
    );
    let inputs = crate::test_support::resolve_lp_build_inputs(
        &system,
        &[],
        &hydro_result.production,
        &study_dims,
        &time_value,
        &hydro_cell_index,
        &resolved_params,
    );
    let ctx = super::build_template_build_ctx(
        &system,
        &par_lp,
        &hydro_result.production,
        &hydro_result.evaporation,
        &resolved.state,
        &topology,
        &inputs,
    );

    let study_stage_ids: Vec<i32> = system
        .stages()
        .iter()
        .filter(|s| s.id >= 0)
        .map(|s| s.id)
        .collect();
    assert_eq!(study_stage_ids, vec![0, 1, 2]);
    assert_eq!(ctx.time_value.delivery_stage_ids(), vec![0, 1, 2, 3, 4]);
    assert!(
        ctx.time_value
            .delivery_stage_ids()
            .windows(2)
            .all(|w| w[0] < w[1]),
        "delivery_stage_ids must be strictly increasing, got {:?}",
        ctx.time_value.delivery_stage_ids()
    );
}

/// The post-study delivery index reads that post-study stage's own hours and
/// its continued cumulative discount factor.
#[test]
fn delivery_vectors_read_the_post_study_element_at_its_delivery_index() {
    let post_study = two_post_study_stages();
    let system = discounted_multi_stage_system_with_post_study(Some(post_study.clone()));
    let hydro_result = PrepareHydroModelsResult::default_from_system(&system);
    let par_lp = PrecomputedPar::default();
    let resolved_params = empty_resolved_params();
    let (topology, resolved) = crate::test_support::resolved_layout_for(&system, &par_lp);
    let hydro_cell_index = HydroCellIndex::build(system.hydros());
    let time_value = build_time_value_for(&system);
    let study_dims = crate::test_support::build_study_dimensions(
        &system,
        InflowNonNegativityMethod::None,
        resolved.anticipated_plants.clone(),
        0,
    );
    let inputs = crate::test_support::resolve_lp_build_inputs(
        &system,
        &[],
        &hydro_result.production,
        &study_dims,
        &time_value,
        &hydro_cell_index,
        &resolved_params,
    );
    let ctx = super::build_template_build_ctx(
        &system,
        &par_lp,
        &hydro_result.production,
        &hydro_result.evaporation,
        &resolved.state,
        &topology,
        &inputs,
    );

    assert_eq!(
        ctx.time_value.delivery_total_hours(3),
        post_study.stages[0].duration_hours
    );
    assert_eq!(
        ctx.time_value.relative_delivery_discount(0, 3),
        ctx.time_value.post_study().cumulative_discount_factors[0]
    );
}

/// `TimeValue`'s delivery-axis factors are bit-identical to
/// `compute_cumulative_discount_factors` run over the study's own per-stage
/// factors concatenated with the post-study per-stage factors — the same
/// identity `continued_cumulative_discount_matches_extended_horizon`
/// (`crate::time_value`) already pins for the post-study half alone.
#[test]
fn delivery_cumulative_discount_matches_recomputed_extended_horizon() {
    let post_study = two_post_study_stages();
    let system = discounted_multi_stage_system_with_post_study(Some(post_study.clone()));
    let hydro_result = PrepareHydroModelsResult::default_from_system(&system);
    let par_lp = PrecomputedPar::default();
    let resolved_params = empty_resolved_params();
    let (topology, resolved) = crate::test_support::resolved_layout_for(&system, &par_lp);
    let hydro_cell_index = HydroCellIndex::build(system.hydros());
    let time_value = build_time_value_for(&system);
    let study_dims = crate::test_support::build_study_dimensions(
        &system,
        InflowNonNegativityMethod::None,
        resolved.anticipated_plants.clone(),
        0,
    );
    let inputs = crate::test_support::resolve_lp_build_inputs(
        &system,
        &[],
        &hydro_result.production,
        &study_dims,
        &time_value,
        &hydro_cell_index,
        &resolved_params,
    );
    let ctx = super::build_template_build_ctx(
        &system,
        &par_lp,
        &hydro_result.production,
        &hydro_result.evaporation,
        &resolved.state,
        &topology,
        &inputs,
    );

    let study_stages: Vec<_> = system.stages().iter().filter(|s| s.id >= 0).collect();
    let study_per_stage = compute_per_stage_discount_factors(&study_stages, system.policy_graph());

    // Mirrors `resolve_post_study_artifacts`'s own deliberately stripped rate
    // graph — a full `system.policy_graph()` would let a real stage's
    // discount-rate override leak onto a synthetic post-study stage.
    let rate_graph = cobre_core::HorizonGraph {
        annual_discount_rate: system.policy_graph().annual_discount_rate,
        ..cobre_core::HorizonGraph::default()
    };
    let calendar_stages = post_study_calendar_stages(&post_study.stages);
    let calendar_stage_refs: Vec<_> = calendar_stages.iter().collect();
    let per_stage_post = compute_per_stage_discount_factors(&calendar_stage_refs, &rate_graph);

    let mut extended_per_stage = study_per_stage;
    extended_per_stage.extend_from_slice(&per_stage_post);
    let extended_cumulative = compute_cumulative_discount_factors(&extended_per_stage);

    let recomputed: Vec<f64> = (0..extended_cumulative.len())
        .map(|m| ctx.time_value.relative_delivery_discount(0, m))
        .collect();
    assert_eq!(
        recomputed, extended_cumulative,
        "TimeValue's delivery-axis factors must be bit-identical to \
         compute_cumulative_discount_factors over study++post per-stage factors"
    );
}

// ── Post-study anticipated bounds table (dense [anticipated_local][stage]) ─

/// One `LeadStages(1)` anticipated thermal per id, distinct `operational_start_date`s
/// so canonical `(operational_start_date, id)` order places `EntityId(7)` at
/// anticipated-local `0` and `EntityId(3)` at anticipated-local `1` — the
/// declared-id-descending order the sort would NOT produce on its own.
fn two_anticipated_thermals(ids: [i32; 2]) -> Vec<Thermal> {
    ids.into_iter()
        .enumerate()
        .map(|(i, id)| Thermal {
            id: EntityId(id),
            name: format!("T{id}"),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1 + i as u32, 1).unwrap(),
            bus_id: EntityId(1),
            entry_stage_id: None,
            exit_stage_id: None,
            cost_per_mwh: 10.0,
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            anticipated_config: Some(AnticipatedConfig::LeadStages(1)),
        })
        .collect()
}

/// [`system_with_thermals`]'s one-stage, one-bus shape plus `post_study`
/// attached — a dedicated fixture rather than widening `system_with_thermals`
/// itself, whose existing callers are outside this table's scope.
fn system_with_anticipated_thermals_and_post_study(
    thermals: Vec<Thermal>,
    post_study: PostStudyStages,
) -> cobre_core::System {
    let n_thermals = thermals.len();
    let n_stages = 1_usize;
    let bus = fixture_bus();

    let stages: Vec<Stage> = vec![Stage {
        index: 0,
        id: 0,
        start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
        season_id: Some(0),
        blocks: vec![Block {
            index: 0,
            name: "BLK0".to_string(),
            duration_hours: 744.0,
        }],
        block_mode: BlockMode::Parallel,
        state_config: StageStateConfig {
            storage: false,
            inflow_lags: false,
        },
        risk_config: StageRiskConfig::Expectation,
        scenario_config: ScenarioSourceConfig {
            branching_factor: 1,
            noise_method: NoiseMethod::Saa,
        },
    }];

    let load_models = vec![LoadModel {
        bus_id: EntityId(1),
        stage_id: 0,
        mean_mw: 100.0,
        std_mw: 0.0,
    }];

    let resolved_bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 0,
            n_thermals,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages,
            k_max: 1,
        },
        &BoundsDefaults {
            hydro: default_hydro_bounds(),
            hydro_block: default_hydro_block_bounds(),
            thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
            thermal_block: ThermalBlockBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 100.0,
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
            n_stages,
        },
        &PenaltiesDefaults {
            hydro: default_hydro_penalties(),
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    );

    SystemBuilder::new()
        .buses(vec![bus])
        .thermals(thermals)
        .stages(stages)
        .load_models(load_models)
        .bounds(resolved_bounds)
        .penalties(penalties)
        .post_study_stages(Some(post_study))
        .build()
        .expect("system_with_anticipated_thermals_and_post_study: valid system")
}

/// Three post-study stages, matching [`two_anticipated_thermals`]'s pair —
/// used by every table-shape test below.
fn three_post_study_stages() -> Vec<PostStudyStage> {
    vec![
        PostStudyStage {
            start_date: NaiveDate::from_ymd_opt(2024, 4, 1).unwrap(),
            duration_hours: 720.0,
        },
        PostStudyStage {
            start_date: NaiveDate::from_ymd_opt(2024, 5, 1).unwrap(),
            duration_hours: 744.0,
        },
        PostStudyStage {
            start_date: NaiveDate::from_ymd_opt(2024, 6, 1).unwrap(),
            duration_hours: 720.0,
        },
    ]
}

/// Build `ctx.time_value.post_study()` for a two-anticipated-thermal,
/// three-post-study-stage system declaring exactly `thermal_bounds`.
fn build_post_study_resolved_for(
    ids: [i32; 2],
    thermal_bounds: Vec<PostStudyThermalBound>,
) -> PostStudyResolved {
    let system = system_with_anticipated_thermals_and_post_study(
        two_anticipated_thermals(ids),
        PostStudyStages {
            stages: three_post_study_stages(),
            thermal_bounds,
        },
    );
    let hydro_result = PrepareHydroModelsResult::default_from_system(&system);
    let par_lp = PrecomputedPar::default();
    let resolved_params = empty_resolved_params();
    let (topology, resolved) = crate::test_support::resolved_layout_for(&system, &par_lp);
    let hydro_cell_index = HydroCellIndex::build(system.hydros());
    let time_value = build_time_value_for(&system);
    let study_dims = crate::test_support::build_study_dimensions(
        &system,
        InflowNonNegativityMethod::None,
        resolved.anticipated_plants.clone(),
        0,
    );
    let inputs = crate::test_support::resolve_lp_build_inputs(
        &system,
        &[],
        &hydro_result.production,
        &study_dims,
        &time_value,
        &hydro_cell_index,
        &resolved_params,
    );
    let ctx = super::build_template_build_ctx(
        &system,
        &par_lp,
        &hydro_result.production,
        &hydro_result.evaporation,
        &resolved.state,
        &topology,
        &inputs,
    );
    ctx.time_value.post_study().clone()
}

/// Given `post_study` is `None`, `resolve_post_study_artifacts` returns
/// [`PostStudyResolved::default`] and the accessor returns `None` on the empty
/// table, never a panic.
#[test]
fn post_study_artifacts_none_returns_default_and_empty_table() {
    let resolved = resolve_post_study_artifacts(
        None,
        &[],
        &cobre_core::HorizonGraph::default(),
        1.0,
        1.0,
        &[],
    );

    assert_eq!(resolved, PostStudyResolved::default());
    assert_eq!(
        resolved.anticipated_bound(AnticipatedLocal::new(0), 0),
        None
    );
}

/// Two anticipated plants in canonical anticipated-local order
/// `[EntityId(7), EntityId(3)]`, a deck declaring a bound only at
/// `(EntityId(7), post-study stage 1)`: `anticipated_bound` returns that bound
/// for local `0` at stage `1` and `None` for local `1` at the same stage.
#[test]
fn anticipated_bound_matches_the_one_declared_cell_only() {
    let resolved = build_post_study_resolved_for(
        [7, 3],
        vec![PostStudyThermalBound {
            thermal_id: EntityId(7),
            post_study_stage_index: 1,
            cost_per_mwh: 42.0,
            min_mw: 5.0,
            max_mw: 50.0,
        }],
    );

    assert_eq!(
        resolved.anticipated_bound(AnticipatedLocal::new(0), 1),
        Some((42.0, 5.0, 50.0)),
        "local 0 (EntityId(7)) must carry the declared bound at stage 1"
    );
    assert_eq!(
        resolved.anticipated_bound(AnticipatedLocal::new(1), 1),
        None,
        "local 1 (EntityId(3)) has no declared bound at stage 1"
    );
}

/// Distinct `(cost, min, max)` triples per `(local, stage)`, local-major —
/// shared by every table-shape test below so a stride/row mix-up surfaces as
/// a mismatch against a specific expected value, never a coincidental match.
const FULL_TABLE_VALUES: [[(f64, f64, f64); 3]; 2] = [
    [(10.0, 1.0, 11.0), (20.0, 2.0, 22.0), (30.0, 3.0, 33.0)],
    [(40.0, 4.0, 44.0), (50.0, 5.0, 55.0), (60.0, 6.0, 66.0)],
];

/// Every `(local, stage)` cell of `ids` x 3 stages, valued from
/// [`FULL_TABLE_VALUES`], sorted canonically by `(thermal_id,
/// post_study_stage_index)` — [`PostStudyThermalLookup::new`](crate::time_value::PostStudyThermalLookup::new)'s
/// own precondition, which anticipated-local order (`ids` here) does not
/// generally satisfy.
fn full_two_plant_three_stage_bounds(ids: [i32; 2]) -> Vec<PostStudyThermalBound> {
    let mut bounds: Vec<PostStudyThermalBound> = ids
        .iter()
        .enumerate()
        .flat_map(|(local, &id)| {
            FULL_TABLE_VALUES[local].iter().enumerate().map(
                move |(stage, &(cost_per_mwh, min_mw, max_mw))| PostStudyThermalBound {
                    thermal_id: EntityId(id),
                    post_study_stage_index: stage,
                    cost_per_mwh,
                    min_mw,
                    max_mw,
                },
            )
        })
        .collect();
    bounds.sort_by_key(|b| (b.thermal_id, b.post_study_stage_index));
    bounds
}

/// A fully-declared 2-plant x 3-stage deck: every cell of the dense table
/// agrees exactly with [`PostStudyThermalLookup::lookup`](crate::time_value::PostStudyThermalLookup::lookup)
/// for the corresponding `EntityId` — pins the table as a faithful projection
/// of the lookup, not merely a plausible one.
#[test]
fn anticipated_bound_agrees_with_lookup_on_every_cell() {
    let ids = [7, 3];
    let resolved = build_post_study_resolved_for(ids, full_two_plant_three_stage_bounds(ids));

    for (local, &id) in ids.iter().enumerate() {
        for stage in 0..3 {
            assert_eq!(
                resolved.anticipated_bound(AnticipatedLocal::new(local), stage),
                resolved.thermal_bounds.lookup(EntityId(id), stage),
                "local {local} (EntityId({id})) stage {stage} must agree with lookup"
            );
        }
    }
}

/// The same fully-declared 2-plant x 3-stage deck: the flattened table holds
/// exactly `2 * 3 = 6` cells. Every in-range `(local, stage)` combination
/// resolves to its own distinct declared value (so a stride/row mix-up would
/// surface as a mismatch, not a coincidental match), and both boundaries —
/// one plant past the row count, one stage past the stride — return `None`
/// rather than wrapping into a neighboring row, pinning the row-count and
/// stride the builder's `debug_assert`s enforce.
#[test]
fn anticipated_bounds_table_shape_is_two_plants_by_three_stages() {
    let ids = [7, 3];
    let resolved = build_post_study_resolved_for(ids, full_two_plant_three_stage_bounds(ids));

    for local in 0..2 {
        for stage in 0..3 {
            assert_eq!(
                resolved.anticipated_bound(AnticipatedLocal::new(local), stage),
                Some(FULL_TABLE_VALUES[local][stage]),
                "local {local} stage {stage}"
            );
        }
    }

    assert_eq!(
        resolved.anticipated_bound(AnticipatedLocal::new(2), 0),
        None,
        "one plant past the row count must return None, not wrap into a nonexistent row"
    );
    assert_eq!(
        resolved.anticipated_bound(AnticipatedLocal::new(0), 3),
        None,
        "one stage past the stride must return None, not shift into the next row"
    );
}

// ── Operational-violation RHS & matrix-coefficient verification ──────────

use super::super::layout::StageLayout;
use crate::DEFAULT_COST_SCALE_FACTOR;
use crate::hydro_models::{ProductionModelSet, ResolvedProductionModel};
use cobre_core::System;
use cobre_solver::StageTemplate;

/// One-hydro system with all operational-violation bounds active (min/max
/// outflow, min turbine, min generation > 0), two blocks per stage, and
/// `1000.0` violation penalties — the fixture the operational-violation
/// builder tests exercise.
fn one_hydro_active_violations(n_stages: usize) -> System {
    use cobre_core::scenario::InflowModel;

    let bus = fixture_bus();

    let mut hydro = Hydro {
        unit_groups: Vec::new(),
        id: EntityId(2),
        name: "H1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        downstream_id: None,
        travel_time_hours: None,
        entry_stage_id: None,
        exit_stage_id: None,
        min_storage_hm3: 0.0,
        max_storage_hm3: 200.0,
        min_outflow_m3s: 50.0,
        max_outflow_m3s: Some(800.0),
        generation_model: HydroGenerationModel::ConstantProductivity,
        min_turbined_m3s: 10.0,
        max_turbined_m3s: 100.0,
        specific_productivity_mw_per_m3s_per_m: None,
        min_generation_mw: 5.0,
        max_generation_mw: 250.0,
        tailrace: None,
        hydraulic_losses: None,
        efficiency: None,
        evaporation_coefficients_mm: None,
        evaporation_reference_volumes_hm3: None,
        diversion: None,
        filling: None,
        penalties: HydroPenalties {
            spillage_cost: 0.01,
            diversion_cost: 0.0,
            turbined_cost: 0.0,
            storage_violation_below_cost: 0.0,
            filling_target_violation_cost: 0.0,
            turbined_violation_below_cost: 1000.0,
            outflow_violation_below_cost: 1000.0,
            outflow_violation_above_cost: 1000.0,
            generation_violation_below_cost: 1000.0,
            evaporation_violation_cost: 0.0,
            water_withdrawal_violation_cost: 0.0,
            water_withdrawal_violation_pos_cost: 0.0,
            water_withdrawal_violation_neg_cost: 0.0,
            evaporation_violation_pos_cost: 0.0,
            evaporation_violation_neg_cost: 0.0,
            inflow_nonnegativity_cost: 1000.0,
        },
    };
    hydro.declare_mirror_unit_group(EntityId(1));

    let stages: Vec<Stage> = (0..n_stages)
        .map(|i| Stage {
            index: i,
            id: i as i32,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: None,
            blocks: vec![
                Block {
                    index: 0,
                    name: "Heavy".to_string(),
                    duration_hours: 720.0,
                },
                Block {
                    index: 1,
                    name: "Light".to_string(),
                    duration_hours: 48.0,
                },
            ],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: true,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        })
        .collect();

    let inflow_models: Vec<InflowModel> = (0..n_stages)
        .map(|i| InflowModel {
            hydro_id: EntityId(2),
            stage_id: i as i32,
            mean_m3s: 80.0,
            std_m3s: 20.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        })
        .collect();

    let load_models: Vec<LoadModel> = (0..n_stages)
        .map(|i| LoadModel {
            bus_id: EntityId(1),
            stage_id: i as i32,
            mean_mw: 100.0,
            std_mw: 0.0,
        })
        .collect();

    let n_st = n_stages.max(1);
    let bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 1,
            n_thermals: 0,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: n_st,
            k_max: 0,
        },
        &BoundsDefaults {
            hydro: HydroStageBounds {
                min_storage_hm3: 0.0,
                max_storage_hm3: 200.0,
                filling_min_rate_m3s: 0.0,
                water_withdrawal_m3s: 0.0,
            },
            hydro_block: HydroBlockBounds {
                min_turbined_m3s: 10.0,
                max_turbined_m3s: 100.0,
                min_outflow_m3s: 50.0,
                max_outflow_m3s: Some(800.0),
                min_generation_mw: 5.0,
                max_generation_mw: 250.0,
                ..Default::default()
            },
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
    let penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: 1,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages: n_st,
        },
        &PenaltiesDefaults {
            hydro: HydroPenalties {
                spillage_cost: 0.01,
                diversion_cost: 0.0,
                turbined_cost: 0.0,
                storage_violation_below_cost: 0.0,
                filling_target_violation_cost: 0.0,
                turbined_violation_below_cost: 1000.0,
                outflow_violation_below_cost: 1000.0,
                outflow_violation_above_cost: 1000.0,
                generation_violation_below_cost: 1000.0,
                evaporation_violation_cost: 0.0,
                water_withdrawal_violation_cost: 0.0,
                water_withdrawal_violation_pos_cost: 0.0,
                water_withdrawal_violation_neg_cost: 0.0,
                evaporation_violation_pos_cost: 0.0,
                evaporation_violation_neg_cost: 0.0,
                inflow_nonnegativity_cost: 1000.0,
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
        .hydros(vec![hydro])
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("one_hydro_active_violations: valid")
}

/// Get CSC entries for column `col` of a built `StageTemplate` as
/// `(row, value)` pairs.
fn csc_entries_for_col(t: &StageTemplate, col: usize) -> Vec<(usize, f64)> {
    let start = t.col_starts[col] as usize;
    let end = t.col_starts[col + 1] as usize;
    (start..end)
        .map(|nz| (t.row_indices[nz] as usize, t.values[nz]))
        .collect()
}

/// Sum of column `col`'s CSC entries that land on row `row`.
fn csc_entry_sum(t: &StageTemplate, col: usize, row: usize) -> f64 {
    csc_entries_for_col(t, col)
        .iter()
        .filter(|(r, _)| *r == row)
        .map(|(_, v)| *v)
        .sum()
}

/// Build the active-violations stage-0 `StageLayout` (the owner of the
/// op-violation row/column ranges) and the matching `StageTemplate` (RHS,
/// bounds, objective, CSC) from one shared `TemplateBuildCtx`, so the row
/// ranges and the template the tests query agree by construction.
///
/// Productivity is `0.5` so the per-block min-generation row carries a
/// `0.5` turbine coefficient (asserted by
/// [`relocated_min_generation_constant_productivity_coefficients`]).
fn build_active_violations_layout_and_template() -> (StageLayout<'static>, StageTemplate) {
    let system = Box::leak(Box::new(one_hydro_active_violations(1)));
    let par_lp = Box::leak(Box::new(PrecomputedPar::default()));
    let production = Box::leak(Box::new(ProductionModelSet::new(
        vec![vec![ResolvedProductionModel::ConstantProductivity {
            productivity: 0.5,
        }]],
        system.hydros(),
        1,
    )));
    let hydro_models = Box::leak(Box::new(PrepareHydroModelsResult::default_from_system(
        system,
    )));
    let resolved_params = Box::leak(Box::new(empty_resolved_params()));

    let (topology, resolved) = crate::test_support::resolved_layout_for(system, par_lp);
    let topology = Box::leak(Box::new(topology));
    let resolved = Box::leak(Box::new(resolved));
    let hydro_cell_index = Box::leak(Box::new(HydroCellIndex::build(system.hydros())));
    let time_value = Box::leak(Box::new(build_time_value_for(system)));
    let study_dims = Box::leak(Box::new(crate::test_support::build_study_dimensions(
        system,
        InflowNonNegativityMethod::None,
        resolved.anticipated_plants.clone(),
        0,
    )));
    let inputs = Box::leak(Box::new(crate::test_support::resolve_lp_build_inputs(
        system,
        &[],
        production,
        study_dims,
        time_value,
        hydro_cell_index,
        resolved_params,
    )));
    let ctx = super::build_template_build_ctx(
        system,
        par_lp,
        production,
        &hydro_models.evaporation,
        &resolved.state,
        topology,
        inputs,
    );
    let ctx = Box::leak(Box::new(ctx));
    let stage = &system.stages()[0];

    let template = super::build_single_stage_template(ctx, stage, 0).template;
    let layout = StageLayout::new(ctx, stage, 0);
    (layout, template)
}

#[test]
fn relocated_operational_violation_row_counts() {
    let (layout, t) = build_active_violations_layout_and_template();

    // 4 row ranges each contain n_hydros * n_blks = 1 * 2 = 2 rows.
    assert_eq!(layout.oper_violation.min_outflow.len(), 2);
    assert_eq!(layout.oper_violation.max_outflow.len(), 2);
    assert_eq!(layout.oper_violation.min_turbine.len(), 2);
    assert_eq!(layout.oper_violation.min_generation.len(), 2);

    assert!(
        layout.oper_violation.min_generation.end <= t.num_rows,
        "operational violation rows exceed num_rows"
    );
}

#[test]
fn relocated_min_outflow_row_bounds() {
    // Per-block: RHS in rate units (m3/s), not volume.
    let (layout, t) = build_active_violations_layout_and_template();
    let expected_lower = 50.0; // min_outflow_m3s

    for blk in 0..2 {
        let row = layout.oper_violation.min_outflow.start + blk;
        assert!(
            (t.row_lower[row] - expected_lower).abs() < 1e-10,
            "min_outflow row_lower (block {blk}) = {}, expected {}",
            t.row_lower[row],
            expected_lower
        );
        assert_eq!(
            t.row_upper[row],
            f64::INFINITY,
            "min_outflow row_upper must be +inf"
        );
    }
}

#[test]
fn relocated_max_outflow_row_bounds() {
    // Per-block: RHS in rate units (m3/s).
    let (layout, t) = build_active_violations_layout_and_template();
    let expected_upper = 800.0; // max_outflow_m3s

    for blk in 0..2 {
        let row = layout.oper_violation.max_outflow.start + blk;
        assert_eq!(
            t.row_lower[row],
            f64::NEG_INFINITY,
            "max_outflow row_lower must be -inf"
        );
        assert!(
            (t.row_upper[row] - expected_upper).abs() < 1e-10,
            "max_outflow row_upper (block {blk}) = {}, expected {}",
            t.row_upper[row],
            expected_upper
        );
    }
}

#[test]
fn relocated_min_turbine_row_bounds() {
    // Per-block: RHS in rate units (m3/s).
    let (layout, t) = build_active_violations_layout_and_template();
    let expected_lower = 10.0; // min_turbined_m3s

    for blk in 0..2 {
        let row = layout.oper_violation.min_turbine.start + blk;
        assert!(
            (t.row_lower[row] - expected_lower).abs() < 1e-10,
            "min_turbine row_lower (block {blk}) = {}, expected {}",
            t.row_lower[row],
            expected_lower
        );
        assert_eq!(
            t.row_upper[row],
            f64::INFINITY,
            "min_turbine row_upper must be +inf"
        );
    }
}

#[test]
fn relocated_min_generation_row_bounds() {
    // Per-block: RHS in rate units (MW), not MWh.
    let (layout, t) = build_active_violations_layout_and_template();
    let expected_lower = 5.0; // min_generation_mw

    for blk in 0..2 {
        let row = layout.oper_violation.min_generation.start + blk;
        assert!(
            (t.row_lower[row] - expected_lower).abs() < 1e-10,
            "min_generation row_lower (block {blk}) = {}, expected {}",
            t.row_lower[row],
            expected_lower
        );
        assert_eq!(
            t.row_upper[row],
            f64::INFINITY,
            "min_generation row_upper must be +inf"
        );
    }
}

#[test]
fn relocated_min_outflow_matrix_coefficients() {
    // Per-block min outflow: q + s + slack = 1.0 per block-row. Diversion `d`
    // is EXCLUDED (the floor binds the non-diverted river-remnant flow);
    // `min_outflow_row_excludes_diversion_but_max_includes_it` pins that both
    // ways against the deliberate min/max asymmetry.
    let (layout, t) = build_active_violations_layout_and_template();
    let n_blks = 2;

    for blk in 0..n_blks {
        let row = layout.oper_violation.min_outflow.start + blk;

        let entries = csc_entries_for_col(&t, layout.geometry.turbine.start + blk);
        let v = entries.iter().find(|e| e.0 == row).map(|e| e.1);
        assert!(
            v.is_some() && (v.unwrap() - 1.0).abs() < 1e-15,
            "turbine blk{blk} entry for min_outflow row: {v:?}"
        );

        let entries = csc_entries_for_col(&t, layout.geometry.spillage.start + blk);
        let v = entries.iter().find(|e| e.0 == row).map(|e| e.1);
        assert!(
            v.is_some() && (v.unwrap() - 1.0).abs() < 1e-15,
            "spillage blk{blk} entry for min_outflow row: {v:?}"
        );

        let entries = csc_entries_for_col(&t, layout.geometry.outflow_below_slack.start + blk);
        let v = entries.iter().find(|e| e.0 == row).map(|e| e.1);
        assert!(
            v.is_some() && (v.unwrap() - 1.0).abs() < 1e-15,
            "outflow_below slack blk{blk}: {v:?}"
        );
    }
}

/// The diversion column is coupled into NEITHER outflow row: both the minimum
/// and the maximum bind the non-diverted `q + s`, leaving diversion to its own
/// channel cap. Re-adding `d` to either row fails.
#[test]
fn both_outflow_rows_exclude_diversion() {
    let (layout, t) = build_active_violations_layout_and_template();
    let n_blks = 2;

    for blk in 0..n_blks {
        let div_col = layout.geometry.diversion.start + blk;
        let entries = csc_entries_for_col(&t, div_col);

        for (label, start) in [
            ("min_outflow", layout.oper_violation.min_outflow.start),
            ("max_outflow", layout.oper_violation.max_outflow.start),
        ] {
            let row = start + blk;
            let entry = entries.iter().find(|e| e.0 == row);
            assert!(
                entry.is_none(),
                "diversion col must have NO entry in the {label} row (blk {blk}), got {entry:?}"
            );
        }
    }
}

#[test]
fn relocated_max_outflow_matrix_slack_is_negative() {
    let (layout, t) = build_active_violations_layout_and_template();
    let n_blks = 2;

    for blk in 0..n_blks {
        let row = layout.oper_violation.max_outflow.start + blk;
        let entries = csc_entries_for_col(&t, layout.geometry.outflow_above_slack.start + blk);
        let v = entries.iter().find(|e| e.0 == row).map(|e| e.1);
        assert!(
            v.is_some() && (v.unwrap() - (-1.0)).abs() < 1e-15,
            "outflow_above slack blk{blk} must be -1.0, got {v:?}"
        );
    }
}

#[test]
fn relocated_min_turbine_matrix_only_turbine_cols() {
    // Per-block min turbine: only turbine columns (no spillage), coefficient 1.0.
    let (layout, t) = build_active_violations_layout_and_template();
    let n_blks = 2;

    for blk in 0..n_blks {
        let row = layout.oper_violation.min_turbine.start + blk;

        let entries = csc_entries_for_col(&t, layout.geometry.turbine.start + blk);
        let v = entries.iter().find(|e| e.0 == row).map(|e| e.1);
        assert!(
            v.is_some() && (v.unwrap() - 1.0).abs() < 1e-15,
            "turbine blk{blk} min_turbine: {v:?}"
        );

        let entries_spill = csc_entries_for_col(&t, layout.geometry.spillage.start + blk);
        let v_spill = entries_spill.iter().find(|e| e.0 == row);
        assert!(
            v_spill.is_none(),
            "spillage should not appear in min_turbine row (blk {blk})"
        );

        let entries = csc_entries_for_col(&t, layout.geometry.turbine_below_slack.start + blk);
        let v = entries.iter().find(|e| e.0 == row).map(|e| e.1);
        assert!(
            v.is_some() && (v.unwrap() - 1.0).abs() < 1e-15,
            "turbine_below slack blk{blk}: {v:?}"
        );
    }
}

#[test]
fn relocated_min_generation_constant_productivity_coefficients() {
    // Per-block constant productivity: coefficient = rho = 0.5 per block-row.
    let (layout, t) = build_active_violations_layout_and_template();
    let n_blks = 2;
    let rho = 0.5;

    for blk in 0..n_blks {
        let row = layout.oper_violation.min_generation.start + blk;

        let entries = csc_entries_for_col(&t, layout.geometry.turbine.start + blk);
        let v = entries.iter().find(|e| e.0 == row).map(|e| e.1);
        assert!(
            v.is_some() && (v.unwrap() - rho).abs() < 1e-10,
            "turbine blk{blk} min_gen coeff: {v:?}, expected {rho}"
        );

        let entries_s = csc_entries_for_col(&t, layout.geometry.generation_below_slack.start + blk);
        let vs = entries_s.iter().find(|e| e.0 == row).map(|e| e.1);
        assert!(
            vs.is_some() && (vs.unwrap() - 1.0).abs() < 1e-15,
            "generation_below slack blk{blk}: {vs:?}"
        );
    }
}

#[test]
fn relocated_diagnostic_template_operational_violation_correctness() {
    let (layout, t) = build_active_violations_layout_and_template();

    assert!(
        !layout.geometry.outflow_below_slack.is_empty(),
        "operational-violation slack columns must be present when hydros exist"
    );

    // Per-block formulation: RHS is in rate units (m3/s or MW), not volume/energy.
    let block_hours_0 = 720.0;

    let row = layout.oper_violation.min_outflow.start;
    assert!(
        (t.row_lower[row] - 50.0).abs() < 1e-10,
        "min_outflow row_lower = {}, expected 50.0 (rate units m3/s)",
        t.row_lower[row],
    );
    assert_eq!(
        t.row_upper[row],
        f64::INFINITY,
        "min_outflow row_upper must be +inf for >= constraint"
    );

    let col = layout.geometry.outflow_below_slack.start;
    assert_eq!(
        t.col_lower[col], 0.0,
        "outflow_below_slack col_lower must be 0"
    );
    assert_eq!(
        t.col_upper[col],
        f64::INFINITY,
        "outflow_below_slack col_upper must be +inf when min_outflow > 0"
    );

    let expected_objective = 1000.0 * block_hours_0 / DEFAULT_COST_SCALE_FACTOR;
    assert!(
        t.objective[col] > 0.0,
        "outflow_below_slack objective must be positive (penalty), got {}",
        t.objective[col]
    );
    assert!(
        (t.objective[col] - expected_objective).abs() < 1e-10,
        "outflow_below_slack objective = {}, expected {} (= 1000 * {} / {})",
        t.objective[col],
        expected_objective,
        block_hours_0,
        DEFAULT_COST_SCALE_FACTOR
    );

    let col_above = layout.geometry.outflow_above_slack.start;
    assert_eq!(t.col_upper[col_above], f64::INFINITY);
    assert!(t.objective[col_above] > 0.0);

    let col_turb = layout.geometry.turbine_below_slack.start;
    assert_eq!(t.col_upper[col_turb], f64::INFINITY);
    assert!(t.objective[col_turb] > 0.0);

    let col_gen = layout.geometry.generation_below_slack.start;
    assert_eq!(t.col_upper[col_gen], f64::INFINITY);
    assert!(t.objective[col_gen] > 0.0);

    let min_turb_row = layout.oper_violation.min_turbine.start;
    assert!(
        (t.row_lower[min_turb_row] - 10.0).abs() < 1e-10,
        "min_turbine row_lower = {}, expected 10.0 (rate units m3/s)",
        t.row_lower[min_turb_row],
    );

    let min_gen_row = layout.oper_violation.min_generation.start;
    assert!(
        (t.row_lower[min_gen_row] - 5.0).abs() < 1e-10,
        "min_generation row_lower = {}, expected 5.0 (rate units MW)",
        t.row_lower[min_gen_row],
    );

    let max_outflow_row = layout.oper_violation.max_outflow.start;
    assert!(
        (t.row_upper[max_outflow_row] - 800.0).abs() < 1e-10,
        "max_outflow row_upper = {}, expected 800.0 (rate units m3/s)",
        t.row_upper[max_outflow_row],
    );
}

/// One-bus, one-hydro FPHA system whose single stage carries `n_blks` blocks
/// under `block_mode`. The FPHA generation rows put the average-storage `γᵥ/2`
/// coefficient on both the incoming and outgoing storage columns, so the
/// byte-identity check actually exercises the storage-bearing rows.
fn one_hydro_block_system(block_mode: BlockMode, n_blks: usize) -> System {
    use cobre_core::scenario::InflowModel;

    let bus = fixture_bus();

    let hydro = fixture_hydro(2);

    let blocks: Vec<Block> = (0..n_blks)
        .map(|b| Block {
            index: b,
            name: format!("BLK{b}"),
            duration_hours: 300.0 + 100.0 * f64::from(u32::try_from(b).unwrap_or(0)),
        })
        .collect();

    let stages: Vec<Stage> = vec![Stage {
        index: 0,
        id: 0,
        start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
        season_id: Some(0),
        blocks,
        block_mode,
        state_config: StageStateConfig {
            storage: true,
            inflow_lags: false,
        },
        risk_config: StageRiskConfig::Expectation,
        scenario_config: ScenarioSourceConfig {
            branching_factor: 1,
            noise_method: NoiseMethod::Saa,
        },
    }];

    let inflow_models = vec![InflowModel {
        hydro_id: EntityId(2),
        stage_id: 0,
        mean_m3s: 80.0,
        std_m3s: 20.0,
        ar_coefficients: vec![],
        residual_std_ratio: 1.0,
        annual: None,
    }];

    let load_models = vec![LoadModel {
        bus_id: EntityId(1),
        stage_id: 0,
        mean_mw: 100.0,
        std_mw: 0.0,
    }];

    let bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 1,
            n_thermals: 0,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: 1,
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
    let penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: 1,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages: 1,
        },
        &PenaltiesDefaults {
            hydro: default_hydro_penalties(),
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    );

    SystemBuilder::new()
        .buses(vec![bus])
        .hydros(vec![hydro])
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("one_hydro_block_system: valid")
}

/// Build the full stage-0 `StageTemplate` for the one-hydro FPHA study under
/// `block_mode` with `n_blks` blocks, using a single FPHA plane so the
/// generation row carries the average-storage anchor.
fn block_template(block_mode: BlockMode, n_blks: usize) -> StageTemplate {
    use crate::hydro_models::FphaPlane;

    let system = one_hydro_block_system(block_mode, n_blks);
    let par_lp = PrecomputedPar::default();
    let production = ProductionModelSet::new(
        vec![vec![ResolvedProductionModel::Fpha {
            planes: vec![FphaPlane {
                intercept: 1.0,
                gamma_v: 0.2,
                gamma_q: 0.5,
                gamma_s: 0.05,
            }],
        }]],
        system.hydros(),
        1,
    );
    let hydro_models = PrepareHydroModelsResult::default_from_system(&system);
    let resolved_params = empty_resolved_params();

    let (topology, resolved) = crate::test_support::resolved_layout_for(&system, &par_lp);
    let hydro_cell_index = HydroCellIndex::build(system.hydros());
    let time_value = build_time_value_for(&system);
    let study_dims = crate::test_support::build_study_dimensions(
        &system,
        InflowNonNegativityMethod::None,
        resolved.anticipated_plants.clone(),
        0,
    );
    let inputs = crate::test_support::resolve_lp_build_inputs(
        &system,
        &[],
        &production,
        &study_dims,
        &time_value,
        &hydro_cell_index,
        &resolved_params,
    );
    let ctx = super::build_template_build_ctx(
        &system,
        &par_lp,
        &production,
        &hydro_models.evaporation,
        &resolved.state,
        &topology,
        &inputs,
    );
    let stage = &system.stages()[0];
    super::build_single_stage_template(&ctx, stage, 0).template
}

/// `K = 1` chronological build collapses to the parallel LP: the interior
/// storage-boundary column family is empty, there is one water row,
/// and FPHA rides the single incoming/outgoing storage pair — so the two
/// templates are byte-identical. This anchors the layout half of
/// the chronological feature against any regression that perturbs the `K = 1`
/// column/row/value layout.
#[test]
fn chronological_k1_byte_identical_to_parallel() {
    let parallel = block_template(BlockMode::Parallel, 1);
    let chronological = block_template(BlockMode::Chronological, 1);
    assert_templates_byte_identical(&parallel, &chronological, "parallel vs chronological");
}

/// `theta` and `n_state` are pure functions of `(N, L, A, k_max)` and are
/// `n_blks`-free by construction (`StateSpace::new` never sees `block_mode` or
/// `n_blks`): per-block storage lives strictly in the control region, never in
/// the state region. Building a chronological `K ≥ 2` stage therefore
/// changes neither — only the control-region column count grows, by exactly
/// `n_h * (n_blks − 1)` interior storage columns.
#[test]
fn theta_and_n_state_invariant_to_block_mode() {
    let hydro_count = 1_usize;
    let max_par_order = 0_usize;
    let n_blks = 3_usize;

    let state = state_layout_full(hydro_count, max_par_order, vec![]);
    let parallel_theta = state.theta;
    let parallel_n_state = state.n_state;

    let parallel = block_template(BlockMode::Parallel, n_blks);
    let chronological = block_template(BlockMode::Chronological, n_blks);

    assert_eq!(
        parallel.n_state, parallel_n_state,
        "parallel template n_state must equal StateSpace n_state"
    );
    assert_eq!(
        chronological.n_state, parallel_n_state,
        "chronological build must not change n_state"
    );
    assert_eq!(
        parallel_theta, state.theta,
        "building chronological must not change theta"
    );

    assert_eq!(
        chronological.num_cols,
        parallel.num_cols + hydro_count * (n_blks - 1),
        "chronological adds exactly n_h*(n_blks-1) interior storage columns"
    );
}

/// Build the one-hydro FPHA study's stage-0 `StageLayout` AND `StageTemplate` from
/// one shared `TemplateBuildCtx`, so the column/row accessors the chronological
/// water tests call (`block_storage_col`, `turbine_col`, `row_water_balance_start`)
/// agree with the template they query by construction.
fn block_layout_and_template(
    block_mode: BlockMode,
    n_blks: usize,
) -> (StageLayout<'static>, StageTemplate, Vec<f64>) {
    use crate::hydro_models::FphaPlane;

    let system = Box::leak(Box::new(one_hydro_block_system(block_mode, n_blks)));
    let par_lp = Box::leak(Box::new(PrecomputedPar::default()));
    let production = Box::leak(Box::new(ProductionModelSet::new(
        vec![vec![ResolvedProductionModel::Fpha {
            planes: vec![FphaPlane {
                intercept: 1.0,
                gamma_v: 0.2,
                gamma_q: 0.5,
                gamma_s: 0.05,
            }],
        }]],
        system.hydros(),
        1,
    )));
    let hydro_models = Box::leak(Box::new(PrepareHydroModelsResult::default_from_system(
        system,
    )));
    let resolved_params = Box::leak(Box::new(empty_resolved_params()));

    let (topology, resolved) = crate::test_support::resolved_layout_for(system, par_lp);
    let topology = Box::leak(Box::new(topology));
    let resolved = Box::leak(Box::new(resolved));
    let hydro_cell_index = Box::leak(Box::new(HydroCellIndex::build(system.hydros())));
    let time_value = Box::leak(Box::new(build_time_value_for(system)));
    let study_dims = Box::leak(Box::new(crate::test_support::build_study_dimensions(
        system,
        InflowNonNegativityMethod::None,
        resolved.anticipated_plants.clone(),
        0,
    )));
    let inputs = Box::leak(Box::new(crate::test_support::resolve_lp_build_inputs(
        system,
        &[],
        production,
        study_dims,
        time_value,
        hydro_cell_index,
        resolved_params,
    )));
    let ctx = super::build_template_build_ctx(
        system,
        par_lp,
        production,
        &hydro_models.evaporation,
        &resolved.state,
        topology,
        inputs,
    );
    let ctx = Box::leak(Box::new(ctx));
    let stage = &system.stages()[0];

    let template = super::build_single_stage_template(ctx, stage, 0).template;
    let layout = StageLayout::new(ctx, stage, 0);
    let tau: Vec<f64> = stage
        .blocks
        .iter()
        .map(|b| b.duration_hours * M3S_TO_HM3)
        .collect();
    (layout, template, tau)
}

/// A chronological `K = 2` Operating hydro emits two chained rows; block
/// `k`'s row carries `+1.0` on `Sᵏ`, `−1.0` on `Sᵏ⁻¹`, and `+τ_k` on that block's
/// turbine column.
#[test]
fn chronological_water_balance_chained_rows() {
    let (layout, t, tau) = block_layout_and_template(BlockMode::Chronological, 2);
    let h = 0_usize;
    let row0 = layout.geometry.water_balance.start() + h * 2;
    let row1 = layout.geometry.water_balance.start() + h * 2 + 1;

    let entry = |col: usize, row: usize| -> f64 {
        let es = csc_entries_for_col(&t, col);
        let vals: Vec<f64> = es
            .iter()
            .filter(|(r, _)| *r == row)
            .map(|(_, v)| *v)
            .collect();
        assert_eq!(vals.len(), 1, "exactly one entry at (col {col}, row {row})");
        vals[0]
    };

    assert_eq!(
        entry(
            layout.block_storage_col(HydroSys::new(h), Boundary::Interior(1)),
            row0
        ),
        1.0
    );
    assert_eq!(
        entry(
            layout.block_storage_col(HydroSys::new(h), Boundary::Incoming),
            row0
        ),
        -1.0
    );
    assert_eq!(
        entry(
            layout
                .geometry
                .turbine_col(HydroCell::new(h), BlockIdx::new(0)),
            row0
        ),
        tau[0]
    );

    assert_eq!(
        entry(
            layout.block_storage_col(HydroSys::new(h), Boundary::Outgoing),
            row1
        ),
        1.0
    );
    assert_eq!(
        entry(
            layout.block_storage_col(HydroSys::new(h), Boundary::Interior(1)),
            row1
        ),
        -1.0
    );
    assert_eq!(
        entry(
            layout
                .geometry
                .turbine_col(HydroCell::new(h), BlockIdx::new(1)),
            row1
        ),
        tau[1]
    );
}

/// Summing the `K` chronological water rows coefficient-wise (over every
/// column) reproduces the parallel single-row build. The interior `Sⁱ` terms cancel
/// to `+1.0` on `Sᴷ` and `−1.0` on `S⁰`, and every `τ_k`-scaled flow term sums to
/// its `ζ`-scaled parallel coefficient (`Σ_k τ_k = ζ`).
#[test]
fn chronological_water_balance_telescopes_to_parallel() {
    let n_blks = 3_usize;
    let h = 0_usize;
    let (par_layout, par_t, _) = block_layout_and_template(BlockMode::Parallel, n_blks);
    let (chr_layout, chr_t, _) = block_layout_and_template(BlockMode::Chronological, n_blks);

    let dense_par = csc_to_dense(&par_t);
    let dense_chr = csc_to_dense(&chr_t);

    // Interior storage columns are shifted into chronological's control region, so
    // parallel and chronological do NOT share control-region column indices; compare
    // per SEMANTIC column via each layout's accessors.
    let par_row = par_layout.geometry.water_balance.start() + h;
    let chr_sum = |chr_col: usize| -> f64 {
        (0..n_blks)
            .map(|k| dense_chr[chr_layout.geometry.water_balance.start() + h * n_blks + k][chr_col])
            .sum()
    };
    let assert_telescopes = |par_col: usize, chr_col: usize, label: &str| {
        let summed = chr_sum(chr_col);
        let expected = dense_par[par_row][par_col];
        assert!(
            (summed - expected).abs() < 1e-12,
            "{label}: chronological telescoped sum {summed} != parallel {expected}"
        );
    };

    // Storage endpoints: Sᴷ (outgoing) telescopes to +1, S⁰ (incoming) to −1.
    assert_telescopes(
        h,
        chr_layout.block_storage_col(HydroSys::new(h), Boundary::Outgoing),
        "outgoing storage Sᴷ",
    );
    assert_telescopes(
        par_layout.state.storage_in.start + h,
        chr_layout.block_storage_col(HydroSys::new(h), Boundary::Incoming),
        "incoming storage S⁰",
    );

    // Per-block flow columns share their semantic accessor (same block index); each
    // block's τ_k sum reproduces the parallel ζ-scaled flow coefficient.
    for blk in 0..n_blks {
        assert_telescopes(
            par_layout
                .geometry
                .turbine_col(HydroCell::new(h), BlockIdx::new(blk)),
            chr_layout
                .geometry
                .turbine_col(HydroCell::new(h), BlockIdx::new(blk)),
            "turbine",
        );
        assert_telescopes(
            par_layout
                .geometry
                .spillage_col(HydroSys::new(h), BlockIdx::new(blk)),
            chr_layout
                .geometry
                .spillage_col(HydroSys::new(h), BlockIdx::new(blk)),
            "spillage",
        );
        assert_telescopes(
            par_layout
                .geometry
                .diversion_col(HydroSys::new(h), BlockIdx::new(blk)),
            chr_layout
                .geometry
                .diversion_col(HydroSys::new(h), BlockIdx::new(blk)),
            "diversion",
        );
    }

    // Withdrawal slacks: parallel applies ±ζ once; chronological's per-block ±τ_k sum
    // recovers ±ζ.
    assert_telescopes(
        par_layout.geometry.withdrawal_slack_neg.start + h,
        chr_layout.geometry.withdrawal_slack_neg.start + h,
        "withdrawal neg",
    );
    assert_telescopes(
        par_layout.geometry.withdrawal_slack_pos.start + h,
        chr_layout.geometry.withdrawal_slack_pos.start + h,
        "withdrawal pos",
    );

    // Interior boundaries Sⁱ (chronological-only) appear +1 in row i−1 and −1 in
    // row i, so they net to zero across the K rows.
    for k in 1..n_blks {
        let summed = chr_sum(chr_layout.block_storage_col(HydroSys::new(h), Boundary::Interior(k)));
        assert!(
            summed.abs() < 1e-12,
            "interior boundary S{k} must cancel across the K rows, got {summed}"
        );
    }

    // The telescoped RHS recovers the parallel RHS: Σ_k −(τ_k·withdrawal) =
    // −(ζ·withdrawal).
    let chr_rhs_sum: f64 = (0..n_blks)
        .map(|k| chr_t.row_lower[chr_layout.geometry.water_balance.start() + h * n_blks + k])
        .sum();
    let par_rhs = par_t.row_lower[par_row];
    assert!(
        (chr_rhs_sum - par_rhs).abs() < 1e-12,
        "telescoped RHS {chr_rhs_sum} != parallel RHS {par_rhs}"
    );
}

/// `StageGeometry::block_storage_col` resolves S⁰, every interior boundary, and
/// Sᴷ to hand-computed expectations sourced independently of
/// `StorageBoundaryGrid::col` itself — `StateSpace`'s own `storage_in`/`storage`
/// ranges for the endpoints, the equipment cursor's `storage_internal_start` for
/// the interior — so a regression in `col`'s match arms fails this test, not just
/// an identity of the owner with itself. Also pins the open-coded parallel
/// endpoint pair (`entries.rs`'s `fill_fpha_entries`/`fill_evaporation_entries`,
/// `BlockMode::Parallel` arm) to the `k = 0`/`k = K` formula arms.
///
/// Seam A of the geometry cross-check guard — pairs with
/// `hydro_storage_boundary_resolves_each_boundary` (Seam B, in
/// `generic_constraints::tests`), each independently anchored against its own
/// hand-computed oracle rather than compared to each other.
#[test]
fn stage_geometry_block_storage_col_matches_layout() {
    let n_blks = 3_usize;
    let (layout, _, _) = block_layout_and_template(BlockMode::Chronological, n_blks);
    let geometry = layout.geometry.clone();
    let storage_in_start = layout.state.storage_in.start;
    let storage_internal_start = layout.geometry.storage_internal_start;
    let storage_final_start = layout.state.storage.start;

    for h in 0..layout.state.hydro_count {
        assert_eq!(
            geometry.block_storage_col(layout.state, HydroSys::new(h), Boundary::Incoming),
            storage_in_start + h,
            "S⁰ endpoint (the parallel-fill open-coded pair) must resolve to \
             storage_in_start + h at hydro {h}"
        );
        for k in 1..n_blks {
            assert_eq!(
                geometry.block_storage_col(layout.state, HydroSys::new(h), Boundary::Interior(k)),
                storage_internal_start + h * (n_blks - 1) + (k - 1),
                "interior boundary S{k} must resolve to storage_internal_start + \
                 h * (n_blks - 1) + (k - 1) at hydro {h}"
            );
        }
        assert_eq!(
            geometry.block_storage_col(layout.state, HydroSys::new(h), Boundary::Outgoing),
            storage_final_start + h,
            "Sᴷ endpoint (the parallel-fill open-coded pair) must resolve to \
             storage_final_start + h at hydro {h}"
        );
    }
}

/// `StageLayout::geometry` field-equals every range/scalar its `StageLayout`
/// source produces, on a `K = 3` fixture (`block_storage_col` agreement is
/// Seam A above; `evap_indices` is empty here — no evaporation model
/// configured, so an emptiness check is the meaningful comparison).
#[test]
fn stage_layout_geometry_field_equals_layout_source_at_k3() {
    let n_blks = 3_usize;
    let (layout, _, _) = block_layout_and_template(BlockMode::Chronological, n_blks);
    let geometry = layout.geometry.clone();

    assert_eq!(geometry.turbine, layout.geometry.turbine, "turbine");
    assert_eq!(geometry.spillage, layout.geometry.spillage, "spillage");
    assert_eq!(geometry.diversion, layout.geometry.diversion, "diversion");
    assert_eq!(geometry.thermal, layout.geometry.thermal, "thermal");
    assert_eq!(
        geometry.anticipated_decision,
        layout.geometry.anticipated_decision.clone(),
        "anticipated_decision"
    );
    assert_eq!(geometry.line_fwd, layout.geometry.line_fwd, "line_fwd");
    assert_eq!(geometry.line_rev, layout.geometry.line_rev, "line_rev");
    assert_eq!(geometry.deficit, layout.geometry.deficit, "deficit");
    assert_eq!(geometry.excess, layout.geometry.excess, "excess");
    assert_eq!(
        geometry.generation, layout.geometry.generation,
        "generation"
    );
    assert_eq!(
        geometry.evap_indices.is_empty(),
        layout.geometry.evap_indices.is_empty(),
        "evap_indices emptiness"
    );
    assert_eq!(
        geometry.inflow_slack, layout.geometry.inflow_slack,
        "inflow_slack"
    );
    assert_eq!(
        geometry.withdrawal_slack_neg, layout.geometry.withdrawal_slack_neg,
        "withdrawal_slack_neg"
    );
    assert_eq!(
        geometry.withdrawal_slack_pos, layout.geometry.withdrawal_slack_pos,
        "withdrawal_slack_pos"
    );
    assert_eq!(
        geometry.outflow_below_slack, layout.geometry.outflow_below_slack,
        "outflow_below_slack"
    );
    assert_eq!(
        geometry.outflow_above_slack, layout.geometry.outflow_above_slack,
        "outflow_above_slack"
    );
    assert_eq!(
        geometry.turbine_below_slack, layout.geometry.turbine_below_slack,
        "turbine_below_slack"
    );
    assert_eq!(
        geometry.generation_below_slack, layout.geometry.generation_below_slack,
        "generation_below_slack"
    );
    assert_eq!(
        geometry.contract_import, layout.geometry.contract_import,
        "contract_import"
    );
    assert_eq!(
        geometry.contract_export, layout.geometry.contract_export,
        "contract_export"
    );
    assert_eq!(
        geometry.water_balance, layout.geometry.water_balance,
        "water_balance"
    );
    assert_eq!(
        geometry.load_balance, layout.geometry.load_balance,
        "load_balance"
    );
    assert_eq!(
        geometry.filling_target,
        layout.geometry.filling_target.clone(),
        "filling_target"
    );
    assert_eq!(
        geometry.filling_target_col,
        layout.geometry.filling_target_col.clone(),
        "filling_target_col"
    );
    assert_eq!(
        geometry.filled_min_storage_floor,
        layout.geometry.filled_min_storage_floor.clone(),
        "filled_min_storage_floor"
    );
    assert_eq!(
        geometry.filled_min_storage_floor_col,
        layout.geometry.filled_min_storage_floor_col.clone(),
        "filled_min_storage_floor_col"
    );
    assert_eq!(geometry.n_blks, layout.clock.n_blks(), "n_blks");
    assert_eq!(geometry.block_mode, BlockMode::Chronological, "block_mode");
    assert_eq!(
        geometry.fpha_hydro_indices, layout.geometry.fpha_hydro_indices,
        "fpha_hydro_indices"
    );
    assert_eq!(
        geometry.evap_hydro_indices, layout.geometry.evap_hydro_indices,
        "evap_hydro_indices"
    );
    assert_eq!(
        geometry.filling_target_hydro_indices, layout.geometry.filling_target_hydro_indices,
        "filling_target_hydro_indices"
    );
    assert_eq!(
        geometry.filled_min_storage_floor_hydro_indices,
        layout.geometry.filled_min_storage_floor_hydro_indices,
        "filled_min_storage_floor_hydro_indices"
    );
}

/// `StageGeometry::water_balance` is the layout's own family, kind included, on
/// a Parallel stage too (the K3 test above covers Chronological).
#[test]
fn stage_layout_geometry_water_balance_family_matches_layout_source_in_parallel_mode() {
    let n_blks = 3_usize;
    let (layout, _, _) = block_layout_and_template(BlockMode::Parallel, n_blks);
    let geometry = layout.geometry.clone();

    assert_eq!(
        geometry.water_balance, layout.geometry.water_balance,
        "water_balance"
    );
}

/// Four-stage, one-bus system combining an import contract, an export contract,
/// a `FillingConfig` hydro (`start_stage_id=1`, `entry_stage_id=3`: PreFilling at
/// stage 0, Filling at stages 1-2, Operating at stage 3), and a `LeadStages(1)`
/// anticipated thermal — so every rerouted `StageGeometry` range is non-trivial
/// and the filling families are exercised both populated and empty across stages.
fn system_with_contracts_filling_and_anticipated() -> cobre_core::System {
    let n_stages = 4_usize;

    let bus = fixture_bus();

    let mut hydro = Hydro {
        unit_groups: Vec::new(),
        id: EntityId(1),
        name: "H1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        downstream_id: None,
        travel_time_hours: None,
        entry_stage_id: Some(3),
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
            start_stage_id: 1,
            filling_min_rate_m3s: 0.0,
        }),
        penalties: hydro_penalties_zero(),
    };
    hydro.declare_mirror_unit_group(EntityId(1));

    let thermal = Thermal {
        id: EntityId(2),
        name: "T1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        entry_stage_id: None,
        exit_stage_id: None,
        cost_per_mwh: 50.0,
        min_generation_mw: 0.0,
        max_generation_mw: 100.0,
        anticipated_config: Some(AnticipatedConfig::LeadStages(1)),
    };

    let contracts = vec![
        fixture_contract(10, ContractType::Import),
        fixture_contract(20, ContractType::Export),
    ];

    let stages: Vec<Stage> = (0..n_stages)
        .map(|i| Stage {
            index: i,
            id: i as i32,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: Some(0),
            blocks: vec![Block {
                index: 0,
                name: "BLK0".to_string(),
                duration_hours: 744.0,
            }],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: false,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        })
        .collect();

    let load_models: Vec<LoadModel> = (0..n_stages)
        .map(|s| LoadModel {
            bus_id: EntityId(1),
            stage_id: s as i32,
            mean_mw: 100.0,
            std_mw: 0.0,
        })
        .collect();

    let resolved_bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 1,
            n_thermals: 1,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 2,
            n_stages,
            k_max: 1,
        },
        &BoundsDefaults {
            hydro: default_hydro_bounds(),
            hydro_block: default_hydro_block_bounds(),
            thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
            thermal_block: ThermalBlockBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 100.0,
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
                max_mw: 500.0,
                price_per_mwh: 100.0,
            },
        },
    );
    let penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: 1,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages,
        },
        &PenaltiesDefaults {
            hydro: default_hydro_penalties(),
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    );

    SystemBuilder::new()
        .buses(vec![bus])
        .hydros(vec![hydro])
        .thermals(vec![thermal])
        .contracts(contracts)
        .stages(stages)
        .load_models(load_models)
        .bounds(resolved_bounds)
        .penalties(penalties)
        .build()
        .expect("system_with_contracts_filling_and_anticipated: valid system")
}

/// Range-equality regression for the seven rerouted `StageGeometry` fields
/// (`contract_import`, `contract_export`, `anticipated_decision`,
/// `filling_target`, `filling_target_col`, `filled_min_storage_floor`,
/// `filled_min_storage_floor_col`): each must equal its `StageLayout` source
/// range at every stage of a fixture combining contracts, a filling hydro, and
/// an anticipated thermal — a future hand-derivation reintroduced into
/// `StageLayout::geometry` that silently drifts from the `StageLayout` accessor
/// would fail this before it fails a parity digest.
#[test]
fn stage_geometry_rerouted_ranges_match_layout_source_at_every_stage() {
    let system = system_with_contracts_filling_and_anticipated();
    let hydro_result = PrepareHydroModelsResult::default_from_system(&system);
    let par_lp = PrecomputedPar::default();
    let resolved_params = empty_resolved_params();

    let (topology, resolved) = crate::test_support::resolved_layout_for(&system, &par_lp);
    let hydro_cell_index = HydroCellIndex::build(system.hydros());
    let time_value = build_time_value_for(&system);
    let study_dims = crate::test_support::build_study_dimensions(
        &system,
        InflowNonNegativityMethod::None,
        resolved.anticipated_plants.clone(),
        0,
    );
    let inputs = crate::test_support::resolve_lp_build_inputs(
        &system,
        &[],
        &hydro_result.production,
        &study_dims,
        &time_value,
        &hydro_cell_index,
        &resolved_params,
    );
    let ctx = super::build_template_build_ctx(
        &system,
        &par_lp,
        &hydro_result.production,
        &hydro_result.evaporation,
        &resolved.state,
        &topology,
        &inputs,
    );

    let mut saw_populated_filling_target = false;
    let mut saw_empty_filling_target = false;
    let mut saw_populated_filled_floor = false;
    let mut saw_empty_filled_floor = false;

    for (stage_idx, stage) in system.stages().iter().enumerate() {
        let layout = super::super::layout::StageLayout::new(&ctx, stage, stage_idx);
        let geometry = layout.geometry.clone();

        assert_eq!(
            geometry.contract_import, layout.geometry.contract_import,
            "stage {stage_idx}: contract_import"
        );
        assert_eq!(
            geometry.contract_export, layout.geometry.contract_export,
            "stage {stage_idx}: contract_export"
        );
        assert_eq!(
            geometry.anticipated_decision,
            layout.geometry.anticipated_decision.clone(),
            "stage {stage_idx}: anticipated_decision"
        );
        assert_eq!(
            geometry.filling_target,
            layout.geometry.filling_target.clone(),
            "stage {stage_idx}: filling_target"
        );
        assert_eq!(
            geometry.filling_target_col,
            layout.geometry.filling_target_col.clone(),
            "stage {stage_idx}: filling_target_col"
        );
        assert_eq!(
            geometry.filled_min_storage_floor,
            layout.geometry.filled_min_storage_floor.clone(),
            "stage {stage_idx}: filled_min_storage_floor"
        );
        assert_eq!(
            geometry.filled_min_storage_floor_col,
            layout.geometry.filled_min_storage_floor_col.clone(),
            "stage {stage_idx}: filled_min_storage_floor_col"
        );

        assert!(
            !geometry.contract_import.is_empty(),
            "stage {stage_idx}: import contract range must be non-empty"
        );
        assert!(
            !geometry.contract_export.is_empty(),
            "stage {stage_idx}: export contract range must be non-empty"
        );
        assert!(
            !geometry.anticipated_decision.is_empty(),
            "stage {stage_idx}: anticipated_decision range must be non-empty (n_anticipated=1)"
        );

        if geometry.filling_target.is_empty() {
            saw_empty_filling_target = true;
        } else {
            saw_populated_filling_target = true;
        }
        if geometry.filled_min_storage_floor.is_empty() {
            saw_empty_filled_floor = true;
        } else {
            saw_populated_filled_floor = true;
        }
    }

    assert!(
        saw_populated_filling_target,
        "fixture must exercise a Filling stage"
    );
    assert!(
        saw_empty_filling_target,
        "fixture must exercise a non-Filling stage"
    );
    assert!(
        saw_populated_filled_floor,
        "fixture must exercise an Operating filling-hydro stage"
    );
    assert!(
        saw_empty_filled_floor,
        "fixture must exercise a non-Operating stage"
    );
}

/// A chronological `K = 1` build's water-balance row is byte-identical to the
/// parallel build's — the single chained row IS the parallel row (`τ_1 = ζ`, no
/// interior boundary). The full-template anchor is
/// [`chronological_k1_byte_identical_to_parallel`]; this isolates the water row.
#[test]
fn chronological_k1_water_row_byte_identical() {
    let parallel = block_template(BlockMode::Parallel, 1);
    let (chrono_layout, chrono, _tau) = block_layout_and_template(BlockMode::Chronological, 1);
    let row = chrono_layout.geometry.water_balance.start();

    let dense_par = csc_to_dense(&parallel);
    let dense_chr = csc_to_dense(&chrono);
    assert_eq!(parallel.num_cols, chrono.num_cols, "K=1 column count");
    for col in 0..parallel.num_cols {
        assert_eq!(
            dense_par[row][col].to_bits(),
            dense_chr[row][col].to_bits(),
            "K=1 water row coefficient mismatch at col {col}"
        );
    }
    assert_eq!(
        parallel.row_lower[row].to_bits(),
        chrono.row_lower[row].to_bits(),
        "K=1 water row_lower"
    );
    assert_eq!(
        parallel.row_upper[row].to_bits(),
        chrono.row_upper[row].to_bits(),
        "K=1 water row_upper"
    );
}

/// D06 preserved per block: at `K ≥ 2` each per-block FPHA plane row
/// carries `−γᵥ/2` on BOTH block-local storage columns `block_storage_col(h, k−1)`
/// (`Sᵏ⁻¹`) and `block_storage_col(h, k)` (`Sᵏ`) — the FPHA average-storage rule
/// applied to the block's own `(Sᵏ⁻¹, Sᵏ)` pair. Placing it on the outgoing column
/// alone (or on the stage endpoints instead of the block boundaries) understates the
/// head term and is the wrong-but-compiling alternative D06 pins against.
#[test]
fn chronological_d06_gamma_v_on_both_block_columns() {
    let n_blks = 2_usize;
    let (layout, t, _tau) = block_layout_and_template(BlockMode::Chronological, n_blks);
    let h = 0_usize;
    // `block_template`/`block_layout_and_template` fix the single FPHA plane's
    // gamma_v; the row coefficient is `−gamma_v/2` after the objective-neutral
    // matrix fill (matrix values are not cost-scaled).
    let half_gamma_v = -0.2 / 2.0;

    let entry = |col: usize, row: usize| csc_entry_sum(&t, col, row);

    for k in 1..=n_blks {
        let blk = k - 1;
        let row = layout.geometry.fpha.start + blk;
        assert_eq!(
            entry(
                layout.block_storage_col(HydroSys::new(h), Boundary::from_index(k - 1, n_blks)),
                row
            ),
            half_gamma_v,
            "block {k}: −γᵥ/2 on Sᵏ⁻¹ (D06 both-columns)"
        );
        assert_eq!(
            entry(
                layout.block_storage_col(HydroSys::new(h), Boundary::from_index(k, n_blks)),
                row
            ),
            half_gamma_v,
            "block {k}: −γᵥ/2 on Sᵏ (D06 both-columns)"
        );
    }
}

/// Cross-mode cut-row byte-comparability invariant: for a chronological `K ≥ 2`
/// FPHA study, the matrix-derived column scale at an interior `block_storage_col(h,
/// k)` equals the endpoint storage-column scale. Identical state-column scaling
/// across the storage family is what keeps rendered cut rows (`−coeff·col_scale[col]`)
/// byte-comparable; a divergent interior scale would silently desynchronise the cut
/// rendering.
#[test]
fn chronological_interior_storage_scale_matches_endpoint() {
    let n_blks = 3_usize;
    let (layout, t, _tau) = block_layout_and_template(BlockMode::Chronological, n_blks);
    let h = 0_usize;

    let col_scale = super::super::compute_col_scale(t.num_cols, &t.col_starts, &t.values);

    let endpoint_scale = col_scale[layout.block_storage_col(HydroSys::new(h), Boundary::Outgoing)];
    for k in 1..n_blks {
        let interior_col = layout.block_storage_col(HydroSys::new(h), Boundary::Interior(k));
        assert_eq!(
            col_scale[interior_col].to_bits(),
            endpoint_scale.to_bits(),
            "interior boundary S{k} scale must equal the endpoint Sᴷ scale (cut-row \
             byte-comparability); divergence signals FPHA/evap coefficients differ \
             between interior and endpoint storage columns"
        );
    }
}

// ── Filling / PreFilling block anchors (D38–D42, σ_fill on Sᴷ) ─────────────

const FILL_N_STAGES: usize = 5;
const FILL_ENTRY_ID: i32 = 4;
const FILL_PRE_START_ID: i32 = 3;
const FILL_MIN_STORAGE_HM3: f64 = 60.0;
const FILL_RATE_M3S: f64 = 5.0;
const FILL_PRE_HYDRO_ID: i32 = 2;
const FILL_FILL_HYDRO_ID: i32 = 3;

/// A `FILL_N_STAGES`-stage, one-bus cascade under `block_mode` with `n_blks` blocks.
/// H2 (`FILL_PRE_HYDRO_ID`) is a filling hydro whose `start_stage_id` sits at
/// `FILL_PRE_START_ID`, so at stage id 0 it is `PreFilling`; H3
/// (`FILL_FILL_HYDRO_ID`) is a filling hydro with `start_stage_id = 0`, so at stage
/// id 0 it is `Filling`. Both share `entry = FILL_ENTRY_ID`. A backup thermal and a
/// bus deficit segment keep the LP feasible regardless of the frozen filling storage.
fn filling_block_system(block_mode: BlockMode, n_blks: usize) -> System {
    use cobre_core::scenario::InflowModel;

    let bus = fixture_bus();

    let filling_hydro = |id: i32, downstream: Option<i32>, start: i32| {
        let mut hydro = Hydro {
            unit_groups: Vec::new(),
            id: EntityId(id),
            name: format!("H{id}"),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            downstream_id: downstream.map(EntityId),
            travel_time_hours: None,
            entry_stage_id: Some(FILL_ENTRY_ID),
            exit_stage_id: None,
            min_storage_hm3: FILL_MIN_STORAGE_HM3,
            max_storage_hm3: 200.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 100.0,
            specific_productivity_mw_per_m3s_per_m: None,
            min_generation_mw: 0.0,
            max_generation_mw: 250.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: Some(FillingConfig {
                start_stage_id: start,
                filling_min_rate_m3s: FILL_RATE_M3S,
            }),
            penalties: hydro_penalties_zero(),
        };
        hydro.declare_mirror_unit_group(EntityId(1));
        hydro
    };

    let hydros = vec![
        filling_hydro(
            FILL_PRE_HYDRO_ID,
            Some(FILL_FILL_HYDRO_ID),
            FILL_PRE_START_ID,
        ),
        filling_hydro(FILL_FILL_HYDRO_ID, None, 0),
    ];

    let blocks: Vec<Block> = (0..n_blks)
        .map(|b| Block {
            index: b,
            name: format!("BLK{b}"),
            duration_hours: 360.0 + 24.0 * f64::from(u32::try_from(b).unwrap_or(0)),
        })
        .collect();

    let stages: Vec<Stage> = (0..FILL_N_STAGES)
        .map(|i| Stage {
            index: i,
            id: i as i32,
            start_date: NaiveDate::from_ymd_opt(2024, (i % 12 + 1) as u32, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, ((i % 12 + 1) % 12 + 1) as u32, 1).unwrap(),
            season_id: Some(0),
            blocks: blocks.clone(),
            block_mode,
            state_config: StageStateConfig {
                storage: true,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        })
        .collect();

    let inflow_models: Vec<InflowModel> = [FILL_PRE_HYDRO_ID, FILL_FILL_HYDRO_ID]
        .into_iter()
        .flat_map(|hid| {
            (0..FILL_N_STAGES).map(move |i| InflowModel {
                hydro_id: EntityId(hid),
                stage_id: i as i32,
                mean_m3s: 80.0,
                std_m3s: 0.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
                annual: None,
            })
        })
        .collect();

    let load_models: Vec<LoadModel> = (0..FILL_N_STAGES)
        .map(|i| LoadModel {
            bus_id: EntityId(1),
            stage_id: i as i32,
            mean_mw: 100.0,
            std_mw: 0.0,
        })
        .collect();

    let mut bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 2,
            n_thermals: 0,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: FILL_N_STAGES,
            k_max: 0,
        },
        &BoundsDefaults {
            hydro: HydroStageBounds {
                min_storage_hm3: FILL_MIN_STORAGE_HM3,
                max_storage_hm3: 200.0,
                filling_min_rate_m3s: FILL_RATE_M3S,
                water_withdrawal_m3s: 0.0,
            },
            hydro_block: HydroBlockBounds {
                max_turbined_m3s: 100.0,
                max_generation_mw: 250.0,
                ..Default::default()
            },
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
    for h_idx in [0_usize, 1] {
        for stage_idx in 0..FILL_N_STAGES {
            let hb = bounds.hydro_bounds_mut(h_idx, stage_idx);
            hb.min_storage_hm3 = FILL_MIN_STORAGE_HM3;
            hb.filling_min_rate_m3s = FILL_RATE_M3S;
        }
    }

    let penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: 2,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages: FILL_N_STAGES,
        },
        &PenaltiesDefaults {
            hydro: default_hydro_penalties(),
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    );

    SystemBuilder::new()
        .buses(vec![bus])
        .hydros(hydros)
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("filling_block_system: valid filling cascade")
}

/// Build the stage-0 `StageLayout` and `StageTemplate` for [`filling_block_system`]
/// under `block_mode` with `n_blks` blocks, alongside the resolved `filling_v_target`
/// fold the σ_fill row RHS reads. Stage 0 places H2 in `PreFilling` and H3 in
/// `Filling`. `Box::leak` matches [`block_layout_and_template`]: the ctx/state must
/// outlive the borrowed `StageLayout`.
fn filling_block_layout_and_template(
    block_mode: BlockMode,
    n_blks: usize,
) -> (
    StageLayout<'static>,
    StageTemplate,
    std::collections::BTreeMap<(usize, i32), f64>,
) {
    let system = Box::leak(Box::new(filling_block_system(block_mode, n_blks)));
    let par_lp = Box::leak(Box::new(PrecomputedPar::default()));
    let production = Box::leak(Box::new(ProductionModelSet::new(
        vec![
            vec![
                ResolvedProductionModel::ConstantProductivity { productivity: 1.0 };
                FILL_N_STAGES
            ];
            2
        ],
        system.hydros(),
        FILL_N_STAGES,
    )));
    let hydro_models = Box::leak(Box::new(PrepareHydroModelsResult::default_from_system(
        system,
    )));
    let resolved_params = Box::leak(Box::new(empty_resolved_params()));

    let (topology, resolved) = crate::test_support::resolved_layout_for(system, par_lp);
    let topology = Box::leak(Box::new(topology));
    let resolved = Box::leak(Box::new(resolved));
    let hydro_cell_index = Box::leak(Box::new(HydroCellIndex::build(system.hydros())));
    let time_value = Box::leak(Box::new(build_time_value_for(system)));
    let study_dims = Box::leak(Box::new(crate::test_support::build_study_dimensions(
        system,
        InflowNonNegativityMethod::None,
        resolved.anticipated_plants.clone(),
        0,
    )));
    let inputs = Box::leak(Box::new(crate::test_support::resolve_lp_build_inputs(
        system,
        &[],
        production,
        study_dims,
        time_value,
        hydro_cell_index,
        resolved_params,
    )));
    let ctx = super::build_template_build_ctx(
        system,
        par_lp,
        production,
        &hydro_models.evaporation,
        &resolved.state,
        topology,
        inputs,
    );
    let ctx = Box::leak(Box::new(ctx));
    let stage = &system.stages()[0];

    let template = super::build_single_stage_template(ctx, stage, 0).template;
    let layout = StageLayout::new(ctx, stage, 0);
    (layout, template, (*ctx.filling_v_target).clone())
}

/// D38–D42 preserved per block: at `K ≥ 2` a `PreFilling` hydro's `K`
/// water rows are frozen identities (`Sᵏ − Sᵏ⁻¹ = 0`) and its spillage AND turbine
/// columns are frozen `[0, 0]` on every block (no dam, no machinery). A `Filling`
/// hydro at the same stage keeps its per-block spillage FREE (the D40 over-dam relief
/// valve) while its turbine stays frozen `[0, 0]` (no machinery until entry).
#[test]
fn chronological_prefilling_d38_d42_per_block() {
    let n_blks = 2_usize;
    let (layout, t, _v_target) =
        filling_block_layout_and_template(BlockMode::Chronological, n_blks);

    // Positional indices after the SystemBuilder id-sort: H2 → 0 (PreFilling at
    // stage 0), H3 → 1 (Filling at stage 0).
    let h_pre = 0_usize;
    let h_fill = 1_usize;

    let entry = |col: usize, row: usize| csc_entry_sum(&t, col, row);

    for k in 1..=n_blks {
        let blk = k - 1;
        let row = layout.geometry.water_balance.start() + h_pre * n_blks + blk;
        assert_eq!(
            entry(
                layout.block_storage_col(HydroSys::new(h_pre), Boundary::from_index(k, n_blks)),
                row
            ),
            1.0,
            "PreFilling block {k}: +1 on Sᵏ (frozen identity, D38/D39/D42)"
        );
        assert_eq!(
            entry(
                layout.block_storage_col(HydroSys::new(h_pre), Boundary::from_index(k - 1, n_blks)),
                row
            ),
            -1.0,
            "PreFilling block {k}: −1 on Sᵏ⁻¹ (frozen identity, D38/D39/D42)"
        );
        assert_eq!(
            t.row_lower[row], 0.0,
            "PreFilling block {k}: frozen-identity RHS lower == 0"
        );
        assert_eq!(
            t.row_upper[row], 0.0,
            "PreFilling block {k}: frozen-identity RHS upper == 0"
        );

        let spill_pre = layout
            .geometry
            .spillage_col(HydroSys::new(h_pre), BlockIdx::new(blk));
        assert_eq!(
            (t.col_lower[spill_pre], t.col_upper[spill_pre]),
            (0.0, 0.0),
            "PreFilling block {k}: spillage frozen [0,0] (no dam to spill from, D38/D39/D42)"
        );
        let turb_pre = layout
            .geometry
            .turbine_col(HydroCell::new(h_pre), BlockIdx::new(blk));
        assert_eq!(
            (t.col_lower[turb_pre], t.col_upper[turb_pre]),
            (0.0, 0.0),
            "PreFilling block {k}: turbine frozen [0,0] (no machinery, D38/D39/D42)"
        );

        // A Filling hydro's spillage is the legitimate D40 relief valve: free upward.
        let spill_fill = layout
            .geometry
            .spillage_col(HydroSys::new(h_fill), BlockIdx::new(blk));
        assert_eq!(
            t.col_lower[spill_fill], 0.0,
            "Filling block {k}: spillage lower == 0"
        );
        assert_eq!(
            t.col_upper[spill_fill],
            f64::INFINITY,
            "Filling block {k}: spillage FREE (D40 relief valve), not frozen"
        );
    }
}

/// Filling-phase target on `Sᴷ`: at `K ≥ 2` a `Filling`-phase hydro's
/// `σ_fill` row references the stage-final storage `block_storage_col(h, K)` (= `Sᴷ`,
/// which aliases the outgoing endpoint `h`), its `V_target` fold value is UNCHANGED
/// from the parallel build (`build_filling_v_target` is keyed `(hydro, stage)` and
/// `ζ`-scaled, so the τ_k-replaces-ζ water change leaves it untouched), and its
/// per-block spillage stays FREE (D40).
#[test]
fn chronological_filling_target_on_final_storage() {
    let n_blks = 2_usize;
    let (chr_layout, chr_t, chr_v_target) =
        filling_block_layout_and_template(BlockMode::Chronological, n_blks);
    let (_par_layout, _par_t, par_v_target) =
        filling_block_layout_and_template(BlockMode::Parallel, n_blks);

    // H3 is the Filling hydro at stage 0 (positional index 1 after the id-sort).
    let h_fill = 1_usize;
    assert_eq!(
        chr_layout.geometry.filling_target_hydro_indices,
        vec![HydroSys::new(h_fill)],
        "exactly the Filling hydro H3 emits a σ_fill target at stage 0"
    );

    let sk_col = chr_layout.block_storage_col(HydroSys::new(h_fill), Boundary::Outgoing);
    assert_eq!(
        sk_col, h_fill,
        "block_storage_col(h, K) aliases the outgoing endpoint (= dense hydro index)"
    );
    let row = chr_layout.geometry.filling_target.start;
    let entry = |col: usize| csc_entry_sum(&chr_t, col, row);
    assert_eq!(
        entry(sk_col),
        1.0,
        "σ_fill row references Sᴷ (block_storage_col(h, K)), the stage-final storage"
    );
    assert_eq!(
        entry(chr_layout.geometry.filling_target_col.start),
        1.0,
        "σ_fill row carries +1 on its σ_fill slack column"
    );

    // The ζ-scaled V_target fold is mode-independent: build_filling_v_target never
    // sees block_mode, so replacing ζ with per-block τ_k in the water rows must not
    // move the target RHS.
    let stage0_id = 0_i32;
    let chr_target = chr_v_target[&(h_fill, stage0_id)];
    let par_target = par_v_target[&(h_fill, stage0_id)];
    assert_eq!(
        chr_target.to_bits(),
        par_target.to_bits(),
        "Filling V_target fold must be byte-identical across modes (keyed (hydro, \
         stage), ζ-scaled)"
    );
    assert_eq!(
        chr_t.row_lower[row].to_bits(),
        chr_target.to_bits(),
        "σ_fill row RHS (≥ lower) equals the V_target fold value"
    );

    for blk in 0..n_blks {
        let spill = chr_layout
            .geometry
            .spillage_col(HydroSys::new(h_fill), BlockIdx::new(blk));
        assert_eq!(
            (chr_t.col_lower[spill], chr_t.col_upper[spill]),
            (0.0, f64::INFINITY),
            "Filling block {blk}: per-block spillage FREE (D40), not frozen"
        );
    }
}

// ── Anticipated-resolution threading (build_template_build_ctx ↔ setup) ──
//
// `build_template_build_ctx`'s threaded `anticipated_resolution` /
// `anticipated_lead_stages` params must carry the same delivery-anchored
// `AnticipatedResolution` setup's `resolve_state_layout` resolves, not the
// constant-lead fallback `anticipated_resolution_for` would otherwise
// reconstruct from `anticipated_lead_stages` alone.

/// One-bus, no-hydro, `n_stages`-stage system with a single thermal carrying
/// `anticipated_config`, each stage a single `stage_hours`-hour block.
/// `k_max_bounds` sizes `BoundsCountsSpec::k_max` for the delivery-stage
/// padding the thermal's per-stage bounds axis needs.
fn anticipated_lead_config_system(
    n_stages: usize,
    stage_hours: f64,
    anticipated_config: AnticipatedConfig,
    k_max_bounds: usize,
) -> cobre_core::System {
    let bus = fixture_bus();

    let thermal = Thermal {
        id: EntityId(1),
        name: "T1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        entry_stage_id: None,
        exit_stage_id: None,
        cost_per_mwh: 50.0,
        min_generation_mw: 0.0,
        max_generation_mw: 100.0,
        anticipated_config: Some(anticipated_config),
    };

    let stages: Vec<Stage> = (0..n_stages)
        .map(|i| Stage {
            index: i,
            id: i as i32,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: Some(0),
            blocks: vec![Block {
                index: 0,
                name: "BLK0".to_string(),
                duration_hours: stage_hours,
            }],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: false,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        })
        .collect();

    let load_models: Vec<LoadModel> = (0..n_stages)
        .map(|i| LoadModel {
            bus_id: EntityId(1),
            stage_id: i as i32,
            mean_mw: 100.0,
            std_mw: 0.0,
        })
        .collect();

    let resolved_bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 0,
            n_thermals: 1,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages,
            k_max: k_max_bounds,
        },
        &BoundsDefaults {
            hydro: default_hydro_bounds(),
            hydro_block: default_hydro_block_bounds(),
            thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
            thermal_block: ThermalBlockBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 100.0,
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
            n_stages,
        },
        &PenaltiesDefaults {
            hydro: default_hydro_penalties(),
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
        .load_models(load_models)
        .bounds(resolved_bounds)
        .penalties(penalties)
        .build()
        .expect("anticipated_lead_config_system: valid system")
}

/// On a uniform 3×744h calendar, a `LeadTime(744.0)` plant resolves
/// `c(m) = [None, Some(0), Some(1)]` (hand-derived: `resolve_decider_physical`
/// against boundaries `[0, 744, 1488, 2232]`, target `= boundaries[m+1] - 744`
/// lands one boundary before `m` at every `m > 0`), giving `depth = [1, 1,
/// 0]` and `k_max = 1`. `build_template_build_ctx`'s resolution-derived
/// `k_max`/`anticipated_lead_stages` and the template `StateSpace`'s
/// threaded resolution must match this and setup's own
/// `resolve_anticipated_commitments` byte-for-byte — not the constant-lead
/// fallback a `Stages(1)`-equivalent reconstruction happens to coincide with
/// here only because the physical lead equals exactly one stage length.
#[test]
fn template_anticipated_resolution_matches_setup_lead_time() {
    let system = anticipated_lead_config_system(3, 744.0, AnticipatedConfig::LeadTime(744.0), 1);

    let hydro_result = PrepareHydroModelsResult::default_from_system(&system);
    let par_lp = PrecomputedPar::default();
    let resolved_params = empty_resolved_params();

    let (topology, resolved) = crate::test_support::resolved_layout_for(&system, &par_lp);
    let hydro_cell_index = HydroCellIndex::build(system.hydros());
    let time_value = build_time_value_for(&system);
    let study_dims = crate::test_support::build_study_dimensions(
        &system,
        InflowNonNegativityMethod::None,
        resolved.anticipated_plants.clone(),
        0,
    );
    let inputs = crate::test_support::resolve_lp_build_inputs(
        &system,
        &[],
        &hydro_result.production,
        &study_dims,
        &time_value,
        &hydro_cell_index,
        &resolved_params,
    );
    let ctx = super::build_template_build_ctx(
        &system,
        &par_lp,
        &hydro_result.production,
        &hydro_result.evaporation,
        &resolved.state,
        &topology,
        &inputs,
    );
    assert_eq!(
        ctx.state
            .anticipated_resolution
            .ring_size(&ctx.state.anticipated_lead_stages),
        1,
        "ctx.state ring size"
    );
    assert_eq!(
        ctx.state.anticipated_lead_stages,
        vec![1],
        "ctx.state.anticipated_lead_stages"
    );

    let template_state = ctx.state;
    assert_eq!(template_state.k_max, 1, "template StateSpace k_max");
    assert_eq!(
        template_state.anticipated_lead_stages,
        vec![1],
        "template StateSpace anticipated_lead_stages"
    );
    let expected_decider = vec![None, Some(0), Some(1)];
    assert_eq!(
        anticipated_resolution_for(template_state, AnticipatedLocal::new(0)).decider,
        expected_decider,
        "template's threaded resolution must resolve the calendar-derived decider"
    );

    let (setup_resolution, setup_lead_stages) = resolve_anticipated_commitments(
        &system,
        &DeliveryCalendar::from_system(&system),
        &AnticipatedPlants::build(system.thermals()),
    );
    assert_eq!(
        setup_lead_stages, ctx.state.anticipated_lead_stages,
        "setup vs template anticipated_lead_stages"
    );
    assert_eq!(
        setup_resolution.anchored_depth(),
        ctx.state
            .anticipated_resolution
            .ring_size(&ctx.state.anticipated_lead_stages),
        "setup vs template k_max"
    );
    assert_eq!(
        setup_resolution.per_plant[0].decider, expected_decider,
        "setup vs template decider"
    );
}

/// A `LeadStages(1)` plant on the same calendar keeps the fallback
/// byte-identical to the threaded resolution — the LeadStages behaviour must
/// stay unchanged (d34/d37 parity).
#[test]
fn template_leadstages_byte_identical_to_setup_and_fallback() {
    let system = anticipated_lead_config_system(3, 744.0, AnticipatedConfig::LeadStages(1), 1);

    let hydro_result = PrepareHydroModelsResult::default_from_system(&system);
    let par_lp = PrecomputedPar::default();
    let resolved_params = empty_resolved_params();

    let (topology, resolved) = crate::test_support::resolved_layout_for(&system, &par_lp);
    let hydro_cell_index = HydroCellIndex::build(system.hydros());
    let time_value = build_time_value_for(&system);
    let study_dims = crate::test_support::build_study_dimensions(
        &system,
        InflowNonNegativityMethod::None,
        resolved.anticipated_plants.clone(),
        0,
    );
    let inputs = crate::test_support::resolve_lp_build_inputs(
        &system,
        &[],
        &hydro_result.production,
        &study_dims,
        &time_value,
        &hydro_cell_index,
        &resolved_params,
    );
    let ctx = super::build_template_build_ctx(
        &system,
        &par_lp,
        &hydro_result.production,
        &hydro_result.evaporation,
        &resolved.state,
        &topology,
        &inputs,
    );
    assert_eq!(ctx.state.anticipated_lead_stages, vec![1]);

    let template_state = ctx.state;
    let template_decider = anticipated_resolution_for(template_state, AnticipatedLocal::new(0))
        .decider
        .clone();

    let (setup_resolution, setup_lead_stages) = resolve_anticipated_commitments(
        &system,
        &DeliveryCalendar::from_system(&system),
        &AnticipatedPlants::build(system.thermals()),
    );
    assert_eq!(setup_lead_stages, ctx.state.anticipated_lead_stages);
    assert_eq!(setup_resolution.per_plant[0].decider, template_decider);

    let fallback_decider = anticipated_resolution_for(ctx.state, AnticipatedLocal::new(0))
        .decider
        .clone();
    assert_eq!(
        fallback_decider, template_decider,
        "LeadStages fallback must stay byte-identical to the threaded resolution"
    );
    assert_eq!(template_decider, vec![None, Some(0), Some(1)]);
}

/// Minimal WARN-capturing `tracing::Subscriber`, mirroring setup's own
/// `WarnRecorder` test fixture (the established setup-time advisory-test
/// pattern for this crate).
struct WarnRecorder {
    messages: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
}

impl WarnRecorder {
    fn new() -> (Self, std::sync::Arc<std::sync::Mutex<Vec<String>>>) {
        let messages = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        (
            Self {
                messages: std::sync::Arc::clone(&messages),
            },
            messages,
        )
    }
}

impl tracing::Subscriber for WarnRecorder {
    fn enabled(&self, metadata: &tracing::Metadata<'_>) -> bool {
        *metadata.level() <= tracing::Level::WARN
    }

    fn new_span(&self, _attrs: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }

    fn record(&self, _span: &tracing::span::Id, _values: &tracing::span::Record<'_>) {}

    fn record_follows_from(&self, _span: &tracing::span::Id, _follows: &tracing::span::Id) {}

    fn event(&self, event: &tracing::Event<'_>) {
        if *event.metadata().level() == tracing::Level::WARN {
            struct MessageVisitor(String);
            impl tracing::field::Visit for MessageVisitor {
                fn record_debug(
                    &mut self,
                    field: &tracing::field::Field,
                    value: &dyn std::fmt::Debug,
                ) {
                    if field.name() == "message" {
                        self.0 = format!("{value:?}");
                    }
                }

                fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
                    if field.name() == "message" {
                        self.0 = value.to_string();
                    }
                }
            }
            let mut visitor = MessageVisitor(String::new());
            event.record(&mut visitor);
            self.messages.lock().unwrap().push(visitor.0);
        }
    }

    fn enter(&self, _span: &tracing::span::Id) {}

    fn exit(&self, _span: &tracing::span::Id) {}
}

/// `build_stage_templates` does not resolve anticipated commitments itself
/// — that responsibility belongs solely to setup's `resolve_state_layout` —
/// so, given an already-resolved `state_layout`/`per_stage_mask`, running it
/// under a WARN-capturing subscriber emits no `K = 0` advisory, even for a
/// system whose sub-stage-lead deliveries would otherwise trigger one.
#[test]
fn build_stage_templates_never_emits_k0_advisory_itself() {
    let system = anticipated_lead_config_system(4, 744.0, AnticipatedConfig::LeadTime(720.0), 0);

    let hydro_result = PrepareHydroModelsResult::default_from_system(&system);
    let par_lp = PrecomputedPar::default();
    let resolved_params = empty_resolved_params();
    let (topology, layout) = crate::test_support::resolved_layout_for(&system, &par_lp);
    let hydro_cell_index = HydroCellIndex::build(system.hydros());
    let study_dims = crate::test_support::build_study_dimensions(
        &system,
        InflowNonNegativityMethod::None,
        layout.anticipated_plants.clone(),
        0,
    );

    let (subscriber, messages) = WarnRecorder::new();
    tracing::subscriber::with_default(subscriber, || {
        let time_value = build_time_value_for(&system);
        let inputs = crate::test_support::resolve_lp_build_inputs(
            &system,
            &[],
            &hydro_result.production,
            &study_dims,
            &time_value,
            &hydro_cell_index,
            &resolved_params,
        );
        let _ = super::build_stage_templates(
            &system,
            &par_lp,
            &hydro_result.production,
            &hydro_result.evaporation,
            &layout.state,
            &topology,
            inputs,
        );
    });

    let recorded = messages.lock().unwrap();
    let relevant: Vec<&str> = recorded
        .iter()
        .filter(|msg| msg.contains("lead_stages == 0"))
        .map(std::string::String::as_str)
        .collect();
    assert!(
        relevant.is_empty(),
        "build_stage_templates must not itself emit the K=0 advisory — that \
         responsibility belongs solely to setup's resolve_state_layout, got: {recorded:?}"
    );
}
