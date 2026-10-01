#![expect(
    clippy::too_many_lines,
    reason = "the fixture spells out one complete study inline so each assertion traces to a literal"
)]

use std::collections::HashMap;
use std::ops::Range;

use chrono::NaiveDate;
use cobre_core::{
    AffineBound, Block, BlockMode, BoundsCountsSpec, BoundsDefaults, Bus, CascadeTopology,
    ConstraintExpression, ContractBlockBounds, ContractType, EnergyContract, EntityId,
    FillingConfig, GenericConstraint, Hydro, HydroBlockBounds, HydroGenerationModel,
    HydroStageBounds, LineBlockBounds, LinearTerm, NoiseMethod, NonControllableSource,
    PumpingBlockBounds, PumpingStation, ResolvedBounds, ResolvedGenericConstraintBounds,
    ScenarioSourceConfig, SlackConfig, Stage, StageRiskConfig, StageStateConfig, Thermal,
    ThermalBlockBounds, ThermalStageBounds, VariableRef,
};

use crate::hydro_models::{EvaporationModelSet, ProductionModelSet};
use crate::indexer::{
    AnticipatedLocal, BlockIdx, BlockRowFamily, Boundary, BusSys, CutStateProjection, EvapLocal,
    FillingTargetLocal, FloorLocal, FphaCellLocal, FphaLocal, HydroCell, HydroCellIndex, HydroSys,
    NcsSys, PumpingSys, StateDim, StateRegion, ThermalSys, anticipated_resolution_for,
};
use crate::lead_time::{AnticipatedResolution, DeliveryAxis, LeadTime, PointResolution};
use crate::resolved_parameters::ResolvedParameters;
use crate::test_support::ctx_fixture::CtxFixture;
use crate::test_support::{
    anticipated_plants_at, constant_lead_resolution, identity_hydro_cell_index, make_unit_group,
    state_layout, state_layout_with_transit_buckets,
    state_layout_with_transit_buckets_and_resolution,
};
use crate::time_value::{PostStudyResolved, TimeValue};

use super::super::DeliveryRing;
use super::super::entries::build_stage_matrix_entries;
use super::super::test_support::zero_hydro_penalties;
use super::{
    RangeCursor, StageLayout, StateSpace, TemplateBuildCtx, build_anticipated_decision_row_pos,
    build_anticipated_fishing_row_pos, build_anticipated_slot_row_pos,
    build_transit_bucket_row_pos, entity_flat, fold_endpoint, variable_ref_is_block_independent,
};

// ── RangeCursor ──────────────────────────────────────────────────────────

/// Consecutive `alloc` calls return adjacent ranges (`r1.end == r2.start`),
/// `alloc(0)` returns `pos..pos` (never `0..0`), and `pos()` reads the running
/// cursor without advancing it.
#[test]
fn range_cursor_adjacency_and_empty_alloc_carries_position() {
    let mut cursor = RangeCursor::new(10);
    assert_eq!(cursor.pos(), 10);

    let r1 = cursor.alloc(3);
    assert_eq!(r1, 10..13);
    assert_eq!(cursor.pos(), 13);

    let r2 = cursor.alloc(5);
    assert_eq!(r2, 13..18);
    assert_eq!(r1.end, r2.start, "consecutive allocations must be adjacent");

    let peeked = cursor.pos();
    let empty = cursor.alloc(0);
    assert_eq!(empty, 18..18, "alloc(0) must return pos..pos, never 0..0");
    assert_eq!(empty.start, empty.end);
    assert_eq!(cursor.pos(), peeked, "alloc(0) must not advance the cursor");
}

// ── Fixture helpers ───────────────────────────────────────────────────────

/// Owns all data needed to construct a zero-entity `TemplateBuildCtx`.
///
/// Fields are kept together so that references into them share a single
/// lifetime `'_`, avoiding the 16-argument helper that clippy flags.
struct ZeroEntityFixtures {
    base: CtxFixture,
}

impl ZeroEntityFixtures {
    fn new() -> Self {
        Self {
            base: CtxFixture::default(),
        }
    }

    /// Install one generic constraint (id 5, slack enabled) whose UPPER bound is a
    /// symbolic reference to a `PerStageBlock` parameter (id 42) carrying two
    /// distinct values `[100.0, 200.0]` at stage 0, and whose LOWER bound is a
    /// numeric parquet endpoint `5.0`. The activation row is `block_id = None`, both
    /// numeric columns null on the upper side. Exercises the effective-endpoint
    /// resolution, the block-varying collapse suppression, and the two-sided slack
    /// shape in one fixture.
    fn install_symbolic_upper_bound(&mut self) {
        self.base.generic_constraints = vec![GenericConstraint {
            id: EntityId(5),
            name: "demand_cap".to_string(),
            description: None,
            expression: ConstraintExpression { terms: vec![] },
            slack: SlackConfig {
                enabled: true,
                penalty: Some(10.0),
            },
            bound_lower_affine: None,
            bound_upper_affine: Some(AffineBound::single(EntityId(42))),
        }];
        let id_map: HashMap<i32, usize> = [(5, 0)].into_iter().collect();
        self.base.resolved_generic_bounds = ResolvedGenericConstraintBounds::new(
            &id_map,
            std::iter::once((5i32, 0i32, None::<i32>, Some(5.0f64), None::<f64>)),
        );
        self.base.resolved_parameters = ResolvedParameters {
            per_param: vec![vec![vec![100.0, 200.0]]],
            id_to_slot: vec![(42, 0)],
            cost_scale_factor: 1_000_000.0,
        };
    }

    /// Install one generic constraint (id 5, no slack) whose UPPER bound carries
    /// BOTH a numeric parquet base (`100.0`) and a constant-only affine remainder
    /// (`-5.0`): the fold arm `fold_endpoint` newly reaches, `base + R`.
    fn install_folded_upper_bound_constant(&mut self) {
        self.base.generic_constraints = vec![GenericConstraint {
            id: EntityId(5),
            name: "folded_cap".to_string(),
            description: None,
            expression: ConstraintExpression { terms: vec![] },
            slack: SlackConfig {
                enabled: false,
                penalty: None,
            },
            bound_lower_affine: None,
            bound_upper_affine: Some(AffineBound {
                constant: -5.0,
                terms: vec![],
            }),
        }];
        let id_map: HashMap<i32, usize> = [(5, 0)].into_iter().collect();
        self.base.resolved_generic_bounds = ResolvedGenericConstraintBounds::new(
            &id_map,
            std::iter::once((5i32, 0i32, None::<i32>, None::<f64>, Some(100.0f64))),
        );
    }

    /// A zero-anticipated `TemplateBuildCtx` that carries the fixture's own
    /// generic constraints (rather than the empty slice `make_ctx` installs).
    fn make_ctx_generic(&mut self) -> TemplateBuildCtx<'_> {
        self.base.anticipated_plants = anticipated_plants_at(&[]);
        self.base.anticipated_lead_stages = vec![];
        self.base.ctx()
    }

    /// Build a zero-entity `TemplateBuildCtx` with the supplied
    /// anticipated-metadata overrides.
    ///
    /// All slice fields are empty; all scalar entity counts are zero except
    /// the anticipated fields provided by the caller. `anticipated_positions`
    /// must be strictly ascending (`test_support::anticipated_plants_at`),
    /// and its length is the resulting `n_anticipated`.
    fn make_ctx(
        &mut self,
        anticipated_lead_stages: Vec<usize>,
        anticipated_positions: &[usize],
    ) -> TemplateBuildCtx<'_> {
        self.base.anticipated_plants = anticipated_plants_at(anticipated_positions);
        self.build_ctx(anticipated_lead_stages)
    }

    /// The shared half of `make_ctx`, reading the already-set
    /// `anticipated_plants` field.
    fn build_ctx(&mut self, anticipated_lead_stages: Vec<usize>) -> TemplateBuildCtx<'_> {
        self.base.anticipated_lead_stages = anticipated_lead_stages;
        let mut ctx = self.base.ctx();
        ctx.generic_constraints = &[];
        ctx
    }
}

/// Build a minimal `Stage` with one block of 744 hours.
fn minimal_stage() -> Stage {
    stage_with_id(0)
}

/// Build a one-block `Stage` whose `id` (the study stage id `filling_phase`
/// keys on) equals `stage_id`. `index` is held at `0` because the per-stage
/// FPHA/evaporation/bounds lookups in these fixtures are indexed by
/// `stage_idx = 0`, while the phase gate reads `stage.id` alone — decoupling
/// the two lets one bounds/model row serve every phase under test.
fn stage_with_id(stage_id: i32) -> Stage {
    Stage {
        index: 0,
        id: stage_id,
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
    }
}

/// Build a single hydro for the FPHA/evaporation membership fixtures.
///
/// `filling`/`entry` drive the [`filling_phase`] gate; `generation_model`
/// follows `fpha`. All other fields are inert defaults — these fixtures
/// exercise per-stage row *membership* (`identify_fpha_hydros` /
/// `identify_evap_hydros`), not column values.
fn membership_hydro(
    id: i32,
    fpha: bool,
    filling: Option<FillingConfig>,
    entry: Option<i32>,
) -> Hydro {
    let mut hydro = Hydro {
        unit_groups: Vec::new(),
        id: EntityId(id),
        name: format!("H{id}"),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        downstream_id: None,
        travel_time_hours: None,
        entry_stage_id: entry,
        exit_stage_id: None,
        min_storage_hm3: 0.0,
        max_storage_hm3: 100.0,
        min_outflow_m3s: 0.0,
        max_outflow_m3s: None,
        generation_model: if fpha {
            HydroGenerationModel::Fpha
        } else {
            HydroGenerationModel::ConstantProductivity
        },
        min_turbined_m3s: 0.0,
        max_turbined_m3s: 50.0,
        specific_productivity_mw_per_m3s_per_m: None,
        min_generation_mw: 0.0,
        max_generation_mw: 45.0,
        tailrace: None,
        hydraulic_losses: None,
        efficiency: None,
        evaporation_coefficients_mm: None,
        evaporation_reference_volumes_hm3: None,
        diversion: None,
        filling,
        penalties: zero_hydro_penalties(),
    };
    hydro.declare_mirror_unit_group(EntityId(1));
    hydro
}

/// Placeholder thermal at position `idx`, inert past its `id`: these tests
/// need only the count `ctx.thermals.len()` reserves in `StageLayout`, never
/// a bound or cost.
fn dormant_thermal(idx: usize) -> Thermal {
    Thermal {
        id: EntityId(i32::try_from(idx).unwrap_or(i32::MAX)),
        name: String::new(),
        operational_start_date: NaiveDate::default(),
        bus_id: EntityId(0),
        entry_stage_id: None,
        exit_stage_id: None,
        cost_per_mwh: 0.0,
        min_generation_mw: 0.0,
        max_generation_mw: 0.0,
        anticipated_config: None,
    }
}

/// Placeholder bus at position `idx`, inert past its `id`: these tests need
/// only the count `ctx.buses.len()` reserves in `StageLayout`, never a
/// deficit segment or excess cost.
fn dormant_bus(idx: usize) -> Bus {
    Bus {
        id: EntityId(i32::try_from(idx).unwrap_or(i32::MAX)),
        name: String::new(),
        operational_start_date: NaiveDate::default(),
        deficit_segments: Vec::new(),
        excess_cost: 0.0,
    }
}

/// Placeholder contract at position `idx`, inert past its `id` and
/// `contract_type`: these tests need only the per-direction counts
/// `StageLayout` derives from `ctx.contracts`, never a price or MW bound.
fn dormant_contract(idx: usize, contract_type: ContractType) -> EnergyContract {
    EnergyContract {
        id: EntityId(i32::try_from(idx).unwrap_or(i32::MAX)),
        name: String::new(),
        operational_start_date: NaiveDate::default(),
        bus_id: EntityId(0),
        contract_type,
        entry_stage_id: None,
        exit_stage_id: None,
        price_per_mwh: 0.0,
        min_mw: 0.0,
        max_mw: 0.0,
    }
}

/// `StageLayout` built from a context with `n_anticipated == 0` has
/// `n_ant_state == 0`, `n_anticipated == 0`, `k_max == 0`, and
/// `col_turbine_start == idx.theta + 1` where `idx` is the N=0, L=0 state
/// layout (zero hydros, zero lag order).
///
/// This verifies that the decision-region offset before the anticipated-ring
/// insertion is preserved when no anticipated thermals are present.
#[test]
fn stage_layout_zero_anticipated_matches_pre_anticipated_offsets() {
    let mut fixtures = ZeroEntityFixtures::new();
    let ctx = fixtures.make_ctx(vec![], &[]);
    let stage = minimal_stage();
    let layout = StageLayout::new(&ctx, &stage, 0);

    assert_eq!(layout.state.commit_out.len(), 0, "n_ant_state");
    assert_eq!(layout.state.n_anticipated, 0, "n_anticipated");
    assert_eq!(layout.state.k_max, 0, "k_max");

    let idx = state_layout(ctx.hydros.len(), ctx.par_lp.max_order());
    assert_eq!(
        layout.geometry.turbine.start,
        idx.theta + 1,
        "col_turbine_start must equal idx.theta + 1 with zero anticipated"
    );
}

/// A symbolic upper bound resolves per `(stage, block)` through the referenced
/// `PerStageBlock` parameter, and the block-varying reference suppresses the
/// stage-level collapse: one row per block, each carrying the parameter's own
/// block value, distinct between blocks.
#[test]
fn symbolic_upper_bound_resolves_per_block_and_suppresses_collapse() {
    let mut fixtures = ZeroEntityFixtures::new();
    fixtures.install_symbolic_upper_bound();
    let ctx = fixtures.make_ctx_generic();
    let stage = stage_with_blocks(BlockMode::Parallel, 2);
    let layout = StageLayout::new(&ctx, &stage, 0);

    assert_eq!(
        layout.generic_constraint_rows.len(),
        2,
        "a block-varying bound reference must not collapse to a single stage-level row"
    );
    let b0 = &layout.generic_constraint_rows[0];
    let b1 = &layout.generic_constraint_rows[1];
    assert_eq!((b0.block_idx, b1.block_idx), (0, 1));
    assert!(!b0.is_stage_level && !b1.is_stage_level);

    assert_eq!(
        b0.bound_upper.expect("upper present").to_bits(),
        100.0_f64.to_bits(),
        "block 0 upper must equal get(42, 0, 0)"
    );
    assert_eq!(
        b1.bound_upper.expect("upper present").to_bits(),
        200.0_f64.to_bits(),
        "block 1 upper must equal get(42, 0, 1)"
    );
    assert_ne!(
        b0.bound_upper.expect("upper present").to_bits(),
        b1.bound_upper.expect("upper present").to_bits(),
        "distinct per-block parameter values must produce distinct row_upper"
    );

    // The numeric parquet lower endpoint flows through unchanged on both rows.
    assert_eq!(b0.bound_lower, Some(5.0));
    assert_eq!(b1.bound_lower, Some(5.0));
}

/// A symbolic endpoint counts as present when shaping the slack: the fixture's
/// numeric lower and symbolic upper make each row two-sided, so an enabled slack
/// gets both a plus and a minus column.
#[test]
fn symbolic_endpoint_makes_row_two_sided_for_slack() {
    let mut fixtures = ZeroEntityFixtures::new();
    fixtures.install_symbolic_upper_bound();
    let ctx = fixtures.make_ctx_generic();
    let stage = stage_with_blocks(BlockMode::Parallel, 2);
    let layout = StageLayout::new(&ctx, &stage, 0);

    for row in &layout.generic_constraint_rows {
        assert!(
            row.slack_plus_col.is_some() && row.slack_minus_col.is_some(),
            "a numeric-lower + symbolic-upper row is two-sided, so slack needs both columns"
        );
    }
}

/// A parquet upper base composed with a constant-only affine remainder folds
/// (`base + R`) into one row bound, byte-identical to a hand-flattened literal.
#[test]
fn folded_upper_bound_constant_shifts_parquet_base() {
    let mut fixtures = ZeroEntityFixtures::new();
    fixtures.install_folded_upper_bound_constant();
    let ctx = fixtures.make_ctx_generic();
    let stage = minimal_stage();
    let layout = StageLayout::new(&ctx, &stage, 0);

    assert_eq!(layout.generic_constraint_rows.len(), 1);
    let row = &layout.generic_constraint_rows[0];
    assert_eq!(row.bound_lower, None);
    assert_eq!(
        row.bound_upper.expect("upper present").to_bits(),
        95.0_f64.to_bits(),
        "row_upper = 100 + (-5) = 95, to_bits() equality"
    );
}

// ── fold_endpoint ─────────────────────────────────────────────────────────

/// A `ResolvedParameters` fixture with one `PerStageBlock` parameter at the
/// given id, one stage of per-block values.
fn resolved_with_param(id: i32, values: Vec<f64>) -> ResolvedParameters {
    ResolvedParameters {
        per_param: vec![vec![values]],
        id_to_slot: vec![(id, 0)],
        ..ResolvedParameters::default()
    }
}

/// `(None, None)`: an endpoint neither side targets stays untargeted.
#[test]
fn fold_endpoint_both_absent_stays_none() {
    let resolved = ResolvedParameters::default();
    assert_eq!(fold_endpoint(None, None, &resolved, 0, 0), None);
}

/// `(Some(base), None)`: a pure literal passes through unchanged.
#[test]
fn fold_endpoint_pure_literal_is_unchanged() {
    let resolved = ResolvedParameters::default();
    assert_eq!(fold_endpoint(Some(95.0), None, &resolved, 0, 0), Some(95.0));
}

/// `(None, Some(bound))`: with no parquet base, the remainder alone
/// establishes the endpoint.
#[test]
fn fold_endpoint_pure_affine_establishes_endpoint() {
    let resolved = resolved_with_param(42, vec![7.0]);
    let bound = AffineBound::single(EntityId(42));
    assert_eq!(
        fold_endpoint(None, Some(&bound), &resolved, 0, 0),
        Some(7.0)
    );
}

/// `(Some(base), Some(bound))` with a constant-only remainder shifts the base
/// by the constant — the fold is `base + R`, never `R` replacing `base`.
#[test]
fn fold_endpoint_literal_and_constant_affine_composes() {
    let resolved = ResolvedParameters::default();
    let bound = AffineBound {
        constant: -5.0,
        terms: vec![],
    };
    assert_eq!(
        fold_endpoint(Some(100.0), Some(&bound), &resolved, 0, 0),
        Some(95.0)
    );
}

/// `(Some(base), Some(bound))` with a `@param`-bearing remainder folds the base
/// with the resolved parameter value at `(stage_idx, block_idx)`, exactly as a
/// constant-only remainder does — the fold does not distinguish the shape.
#[test]
fn fold_endpoint_literal_and_param_affine_composes() {
    let resolved = resolved_with_param(42, vec![3.0, 4.0]);
    let bound = AffineBound::single(EntityId(42));
    assert_eq!(
        fold_endpoint(Some(100.0), Some(&bound), &resolved, 0, 1),
        Some(104.0)
    );
}

/// An `==`-normalized remainder assigned to both endpoints folds each
/// independently against its own parquet base.
#[test]
fn fold_endpoint_equals_normalized_folds_both_endpoints() {
    let resolved = ResolvedParameters::default();
    let bound = AffineBound {
        constant: 2.0,
        terms: vec![],
    };
    assert_eq!(
        fold_endpoint(Some(10.0), Some(&bound), &resolved, 0, 0),
        Some(12.0),
        "lower endpoint folds against its own base"
    );
    assert_eq!(
        fold_endpoint(Some(50.0), Some(&bound), &resolved, 0, 0),
        Some(52.0),
        "upper endpoint folds against its own base"
    );
}

// ── useful_volume_bound_shift (via `enumerate_generic_constraint_rows`) ────

/// One `ResolvedGenericConstraintBounds::new` raw row, constraint id fixed by the
/// caller: `(stage_id, block_id, bound_lower, bound_upper)`.
type RawBoundRow = (i32, Option<i32>, Option<f64>, Option<f64>);

/// Owns a `TemplateBuildCtx` whose generic constraints reference
/// `HydroUsefulVolume{Initial,Final}` terms. The fold reads only `hydro_pos` and
/// `resolved.bounds`, so — like `PumpingFixtures` — `hydros`/`n_hydros` stay
/// empty/zero; every hydro-count-driven column/row family in `StageLayout::new`
/// then allocates zero entries, leaving only the generic-constraint row family
/// under test.
struct UsefulVolumeFixtures {
    base: CtxFixture,
}

impl UsefulVolumeFixtures {
    /// `n_hydros` hydros at ids `1..=n_hydros` (positions `0..n_hydros`), `n_stages`
    /// stages, every entity `min_storage_hm3` defaulted to `0.0` — set per test via
    /// `hydros[pos].min_storage_hm3` (the useful-volume fold's source); `bounds`
    /// stays available to set a differing per-stage operative value.
    /// `production_models`/`evaporation_models`/`cascade`/`hydro_cell_index` are
    /// sized to the real `hydros` slice (constant-productivity, no evaporation,
    /// no cascade links) purely so `StageLayout::new`'s hydro-column families
    /// stay safe to allocate; the fold itself reads `ctx.hydros`/`ctx.positions`
    /// directly and these tests never assert on hydro-column values.
    fn new(n_hydros: usize, n_stages: usize) -> Self {
        use crate::hydro_models::{EvaporationModel, ResolvedProductionModel};

        let hydros: Vec<Hydro> = (0..n_hydros)
            .map(|i| {
                membership_hydro(
                    i32::try_from(i + 1).expect("small test id"),
                    false,
                    None,
                    None,
                )
            })
            .collect();
        let cascade = CascadeTopology::build(&hydros);
        let hydro_cell_index = HydroCellIndex::build(&hydros);
        let constant = ResolvedProductionModel::ConstantProductivity { productivity: 0.0 };
        let production_models =
            ProductionModelSet::new(vec![vec![constant; n_stages]; n_hydros], &hydros, n_stages);
        let evaporation_models = EvaporationModelSet::new(vec![EvaporationModel::None; n_hydros]);
        Self {
            base: CtxFixture {
                hydros,
                cascade,
                hydro_cell_index,
                production_models,
                evaporation_models,
                bounds: ResolvedBounds::new(
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
                        hydro: HydroStageBounds {
                            min_storage_hm3: 0.0,
                            max_storage_hm3: 0.0,
                            filling_min_rate_m3s: 0.0,
                            water_withdrawal_m3s: 0.0,
                        },
                        hydro_block: HydroBlockBounds::default(),
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
                ),
                time_value: TimeValue::from_parts(
                    vec![],
                    vec![1.0; n_stages],
                    vec![744.0; n_stages],
                    (0..i32::try_from(n_stages).unwrap_or(0)).collect(),
                    PostStudyResolved::default(),
                ),
                ..CtxFixture::default()
            },
        }
    }

    /// Install one generic constraint (id 5) with the given expression `terms` and
    /// resolved bound rows (`(stage_id, block_id, bound_lower, bound_upper)`).
    fn install_constraint(&mut self, terms: Vec<LinearTerm>, raw_bounds: Vec<RawBoundRow>) {
        self.base.generic_constraints = vec![GenericConstraint {
            id: EntityId(5),
            name: "useful_volume".to_string(),
            description: None,
            expression: ConstraintExpression { terms },
            slack: SlackConfig {
                enabled: false,
                penalty: None,
            },
            bound_lower_affine: None,
            bound_upper_affine: None,
        }];
        let id_map: HashMap<i32, usize> = [(5, 0)].into_iter().collect();
        self.base.resolved_generic_bounds = ResolvedGenericConstraintBounds::new(
            &id_map,
            raw_bounds
                .into_iter()
                .map(|(stage_id, block_id, lo, hi)| (5i32, stage_id, block_id, lo, hi)),
        );
    }

    fn make_ctx(&mut self) -> TemplateBuildCtx<'_> {
        self.base.ctx()
    }
}

/// `1.0 * hydro_useful_volume_final(h) >= B` resolves to `B + 1.0*V_lo(h,t)`.
#[test]
fn useful_volume_single_term_lower_bound_folds_v_lo() {
    let mut fixtures = UsefulVolumeFixtures::new(1, 1);
    let h = EntityId(1);
    fixtures.base.hydros[0].min_storage_hm3 = 12.5;
    fixtures.install_constraint(
        vec![LinearTerm::literal(
            1.0,
            VariableRef::HydroUsefulVolumeFinal {
                hydro_id: h,
                block_id: None,
            },
        )],
        vec![(0, None, Some(20.0), None)],
    );
    let ctx = fixtures.make_ctx();
    let stage = minimal_stage();
    let layout = StageLayout::new(&ctx, &stage, 0);

    assert_eq!(layout.generic_constraint_rows.len(), 1);
    let row = &layout.generic_constraint_rows[0];
    assert_eq!(
        row.bound_lower.expect("lower present").to_bits(),
        32.5_f64.to_bits(),
        "B + 1.0 * V_lo = 20.0 + 12.5"
    );
    assert_eq!(row.bound_upper, None);
}

/// A negative useful-volume coefficient SUBTRACTS `|coef| * V_lo` from the fold
/// — every other fold test here exercises a positive coefficient only.
#[test]
fn useful_volume_negative_coefficient_lower_bound_subtracts_v_lo() {
    let mut fixtures = UsefulVolumeFixtures::new(1, 1);
    let h = EntityId(1);
    fixtures.base.hydros[0].min_storage_hm3 = 12.5;
    fixtures.install_constraint(
        vec![LinearTerm::literal(
            -1.0,
            VariableRef::HydroUsefulVolumeFinal {
                hydro_id: h,
                block_id: None,
            },
        )],
        vec![(0, None, Some(20.0), None)],
    );
    let ctx = fixtures.make_ctx();
    let stage = minimal_stage();
    let layout = StageLayout::new(&ctx, &stage, 0);

    assert_eq!(layout.generic_constraint_rows.len(), 1);
    let row = &layout.generic_constraint_rows[0];
    assert_eq!(
        row.bound_lower.expect("lower present").to_bits(),
        7.5_f64.to_bits(),
        "B + (-1.0) * V_lo = 20.0 - 12.5"
    );
    assert_eq!(row.bound_upper, None);
}

/// `c1*ufv(h1) + c2*ufv(h2) >= B` resolves to `B + c1*V_lo(h1,t) + c2*V_lo(h2,t)`.
#[test]
fn useful_volume_multi_term_lower_bound_sums_each_hydros_v_lo() {
    let mut fixtures = UsefulVolumeFixtures::new(2, 1);
    let h1 = EntityId(1);
    let h2 = EntityId(2);
    fixtures.base.hydros[0].min_storage_hm3 = 10.0;
    fixtures.base.hydros[1].min_storage_hm3 = 4.0;
    fixtures.install_constraint(
        vec![
            LinearTerm::literal(
                2.0,
                VariableRef::HydroUsefulVolumeFinal {
                    hydro_id: h1,
                    block_id: None,
                },
            ),
            LinearTerm::literal(
                3.0,
                VariableRef::HydroUsefulVolumeFinal {
                    hydro_id: h2,
                    block_id: None,
                },
            ),
        ],
        vec![(0, None, Some(50.0), None)],
    );
    let ctx = fixtures.make_ctx();
    let stage = minimal_stage();
    let layout = StageLayout::new(&ctx, &stage, 0);

    assert_eq!(layout.generic_constraint_rows.len(), 1);
    let row = &layout.generic_constraint_rows[0];
    assert_eq!(
        row.bound_lower.expect("lower present").to_bits(),
        82.0_f64.to_bits(),
        "50 + 2*10 + 3*4 = 82"
    );
}

/// A constraint with no useful-volume term is bit-identical to the pre-fold
/// value — proven with a `-0.0` endpoint, since `-0.0 + 0.0` would flip the sign
/// bit, catching an implementation that always adds the (possibly zero) shift.
#[test]
fn useful_volume_fold_inert_for_non_useful_volume_constraint() {
    let mut fixtures = UsefulVolumeFixtures::new(1, 1);
    fixtures.base.hydros[0].min_storage_hm3 = 99.0;
    fixtures.install_constraint(vec![], vec![(0, None, Some(-0.0), None)]);
    let ctx = fixtures.make_ctx();
    let stage = minimal_stage();
    let layout = StageLayout::new(&ctx, &stage, 0);

    let row = &layout.generic_constraint_rows[0];
    assert_eq!(
        row.bound_lower.expect("lower present").to_bits(),
        (-0.0_f64).to_bits(),
        "no useful-volume term: the endpoint must not even add 0.0"
    );
}

/// Orchestrator clarification: the fold shifts only the endpoint(s)
/// `fold_endpoint` already resolved to `Some`; an untargeted `None` endpoint
/// stays `None`. Also exercises `HydroUsefulVolumeInitial` on an upper-bounded
/// constraint (the endpoint-symmetry requirement).
#[test]
fn useful_volume_fold_leaves_untargeted_lower_endpoint_as_none() {
    let mut fixtures = UsefulVolumeFixtures::new(1, 1);
    let h = EntityId(1);
    fixtures.base.hydros[0].min_storage_hm3 = 7.0;
    fixtures.install_constraint(
        vec![LinearTerm::literal(
            1.0,
            VariableRef::HydroUsefulVolumeInitial {
                hydro_id: h,
                block_id: None,
            },
        )],
        vec![(0, None, None, Some(40.0))],
    );
    let ctx = fixtures.make_ctx();
    let stage = minimal_stage();
    let layout = StageLayout::new(&ctx, &stage, 0);

    let row = &layout.generic_constraint_rows[0];
    assert_eq!(
        row.bound_lower, None,
        "untargeted lower endpoint stays None"
    );
    assert_eq!(
        row.bound_upper.expect("upper present").to_bits(),
        47.0_f64.to_bits(),
        "40 + 1.0 * 7.0 = 47"
    );
}

/// The fold does not defeat the stage-level collapse: a block-independent
/// useful-volume expression still resolves to one stage-level row across
/// `n_blks` blocks, carrying the correctly folded value.
#[test]
fn useful_volume_fold_collapses_block_independent_expression_to_one_row() {
    let mut fixtures = UsefulVolumeFixtures::new(1, 1);
    let h = EntityId(1);
    fixtures.base.hydros[0].min_storage_hm3 = 5.0;
    fixtures.install_constraint(
        vec![LinearTerm::literal(
            1.0,
            VariableRef::HydroUsefulVolumeFinal {
                hydro_id: h,
                block_id: None,
            },
        )],
        vec![(0, None, Some(10.0), None)],
    );
    let ctx = fixtures.make_ctx();
    let stage = stage_with_blocks(BlockMode::Parallel, 3);
    let layout = StageLayout::new(&ctx, &stage, 0);

    assert_eq!(
        layout.generic_constraint_rows.len(),
        1,
        "block-independent useful-volume term must still collapse to one stage-level row"
    );
    let row = &layout.generic_constraint_rows[0];
    assert!(row.is_stage_level);
    assert_eq!(
        row.bound_lower.expect("lower present").to_bits(),
        15.0_f64.to_bits(),
        "10 + 1.0*5.0 = 15"
    );
}

/// The fold reads the ENTITY physical `min_storage_hm3` — stage-invariant —
/// never the per-stage resolved `HydroStageBounds.min_storage_hm3`, which can
/// carry an operative floor (flood control, DECOMP RHV) diverging from it.
#[test]
fn useful_volume_fold_uses_entity_physical_v_lo_not_per_stage_operative_bounds() {
    let mut fixtures = UsefulVolumeFixtures::new(1, 2);
    let h = EntityId(1);
    // Per-stage operative bounds differ from each other and from the entity
    // physical value; the fold must track neither.
    fixtures.base.bounds.hydro_bounds_mut(0, 0).min_storage_hm3 = 5.0;
    fixtures.base.bounds.hydro_bounds_mut(0, 1).min_storage_hm3 = 8.0;
    fixtures.base.hydros[0].min_storage_hm3 = 20.0;
    fixtures.install_constraint(
        vec![LinearTerm::literal(
            1.0,
            VariableRef::HydroUsefulVolumeFinal {
                hydro_id: h,
                block_id: None,
            },
        )],
        vec![(0, None, Some(10.0), None), (1, None, Some(10.0), None)],
    );
    let ctx = fixtures.make_ctx();

    let stage0 = stage_with_id(0);
    let layout0 = StageLayout::new(&ctx, &stage0, 0);
    assert_eq!(
        layout0.generic_constraint_rows[0]
            .bound_lower
            .expect("present")
            .to_bits(),
        30.0_f64.to_bits(),
        "stage 0: 10 + 20.0 (entity physical value, not the operative 5.0)"
    );

    let stage1 = stage_with_id(1);
    let layout1 = StageLayout::new(&ctx, &stage1, 1);
    assert_eq!(
        layout1.generic_constraint_rows[0]
            .bound_lower
            .expect("present")
            .to_bits(),
        30.0_f64.to_bits(),
        "stage 1: same entity value 20.0 — stage-invariant, not the operative 8.0"
    );
}

/// The fold uses the EFFECTIVE coefficient `resolved_coeff * term.scale` — the
/// same value `fill_generic_constraint_entries` prices the storage column at
/// (`resolve_variable_ref` resolves a useful-volume variant at multiplier
/// `1.0`), not the raw resolved coefficient alone.
#[test]
fn useful_volume_fold_effective_coefficient_includes_term_scale() {
    let mut fixtures = UsefulVolumeFixtures::new(1, 1);
    let h = EntityId(1);
    fixtures.base.hydros[0].min_storage_hm3 = 3.0;
    fixtures.base.resolved_parameters = ResolvedParameters {
        per_param: vec![vec![vec![4.0]]],
        id_to_slot: vec![(9, 0)],
        cost_scale_factor: 1_000_000.0,
    };
    fixtures.install_constraint(
        vec![LinearTerm::parameter(
            EntityId(9),
            2.5,
            VariableRef::HydroUsefulVolumeFinal {
                hydro_id: h,
                block_id: None,
            },
        )],
        vec![(0, None, Some(1.0), None)],
    );
    let ctx = fixtures.make_ctx();
    let stage = minimal_stage();
    let layout = StageLayout::new(&ctx, &stage, 0);

    let row = &layout.generic_constraint_rows[0];
    assert_eq!(
        row.bound_lower.expect("lower present").to_bits(),
        31.0_f64.to_bits(),
        "1.0 + (4.0 resolved * 2.5 scale) * 3.0 V_lo = 1 + 10*3 = 31"
    );
}

/// Referential validation (`validate_variable_ref_entity`) guarantees every
/// useful-volume `hydro_id` resolves in `hydro_pos`; a miss fires the same
/// test-loud, production-safe `debug_assert!` `ResolvedParameters::get` uses
/// for its own unreachable-post-validation misses.
#[test]
#[should_panic(expected = "useful-volume term references unknown hydro")]
fn useful_volume_fold_unresolvable_hydro_id_fires_debug_assert() {
    let mut fixtures = UsefulVolumeFixtures::new(0, 1);
    fixtures.install_constraint(
        vec![LinearTerm::literal(
            1.0,
            VariableRef::HydroUsefulVolumeFinal {
                hydro_id: EntityId(99),
                block_id: None,
            },
        )],
        vec![(0, None, Some(1.0), None)],
    );
    let ctx = fixtures.make_ctx();
    let stage = minimal_stage();
    let _ = StageLayout::new(&ctx, &stage, 0);
}

// ── interior storage-boundary sizing ─────────────────────────────────────

/// Owns a two-hydro, constant-productivity `TemplateBuildCtx` for the
/// interior storage-boundary sizing assertions. No FPHA/filling/evaporation, so
/// only the block geometry (`n_blks`, `block_mode`) and `n_hydros` drive the family.
struct TwoHydroFixtures {
    base: CtxFixture,
}

impl TwoHydroFixtures {
    fn new() -> Self {
        use crate::hydro_models::{EvaporationModel, ResolvedProductionModel};

        let constant = ResolvedProductionModel::ConstantProductivity { productivity: 0.0 };
        let models = vec![vec![constant.clone()], vec![constant]];
        let hydros = vec![
            membership_hydro(1, false, None, None),
            membership_hydro(2, false, None, None),
        ];
        let cascade = CascadeTopology::build(&hydros);
        let hydro_cell_index = HydroCellIndex::build(&hydros);
        let production_models = ProductionModelSet::new(models, &hydros, 1);
        Self {
            base: CtxFixture {
                hydros,
                hydro_cell_index,
                cascade,
                production_models,
                evaporation_models: EvaporationModelSet::new(vec![
                    EvaporationModel::None,
                    EvaporationModel::None,
                ]),
                ..CtxFixture::default()
            },
        }
    }

    fn make_ctx(&mut self) -> TemplateBuildCtx<'_> {
        self.base.ctx()
    }
}

/// Build a `Stage` with `n_blks` equal-duration blocks under `block_mode`.
fn stage_with_blocks(block_mode: BlockMode, n_blks: usize) -> Stage {
    let mut stage = minimal_stage();
    stage.block_mode = block_mode;
    stage.blocks = (0..n_blks)
        .map(|index| Block {
            index,
            name: format!("BLK{index}"),
            duration_hours: 744.0,
        })
        .collect();
    stage
}

/// The interior storage-boundary family spans the `K − 1` interior boundaries
/// per hydro only in chronological mode with `K ≥ 2`: empty in parallel mode
/// and at `K = 1`, with `turbine.start` re-anchored to the family's end.
#[test]
fn chronological_interior_storage_boundary_sizing() {
    let mut fixtures = TwoHydroFixtures::new();
    let ctx = fixtures.make_ctx();
    let anchor = ctx.state.control_region_start();

    let stage_parallel = stage_with_blocks(BlockMode::Parallel, 3);
    let parallel = StageLayout::new(&ctx, &stage_parallel, 0);
    assert_eq!(
        parallel.geometry.storage_internal_start, parallel.geometry.turbine.start,
        "parallel K=3 interior storage-boundary family is empty"
    );
    assert_eq!(
        parallel.geometry.storage_internal_start, anchor,
        "parallel storage_internal_start anchors at control_region_start()"
    );
    assert_eq!(
        parallel.geometry.turbine.start, anchor,
        "parallel turbine.start anchors at control_region_start()"
    );

    let stage_chrono_k1 = stage_with_blocks(BlockMode::Chronological, 1);
    let chrono_k1 = StageLayout::new(&ctx, &stage_chrono_k1, 0);
    assert_eq!(
        chrono_k1.geometry.storage_internal_start, chrono_k1.geometry.turbine.start,
        "chronological K=1 interior storage-boundary family is empty"
    );
    assert_eq!(
        chrono_k1.geometry.storage_internal_start, anchor,
        "chronological K=1 storage_internal_start anchors at control_region_start()"
    );
    assert_eq!(
        chrono_k1.geometry.turbine.start, anchor,
        "chronological K=1 turbine.start anchors at control_region_start()"
    );

    let stage_chrono_k3 = stage_with_blocks(BlockMode::Chronological, 3);
    let chrono_k3 = StageLayout::new(&ctx, &stage_chrono_k3, 0);
    assert_eq!(
        chrono_k3.geometry.storage_internal_start, anchor,
        "chronological K=3 storage_internal_start anchors at control_region_start()"
    );
    assert_eq!(
        chrono_k3.geometry.turbine.start - chrono_k3.geometry.storage_internal_start,
        4,
        "chronological K=3 interior storage-boundary family spans n_h * (K - 1) = 2 * 2 columns"
    );
    assert_eq!(
        chrono_k3.geometry.turbine.start,
        chrono_k3.geometry.storage_internal_start + 4,
        "chronological K=3 turbine.start re-anchors 4 columns after storage_internal_start"
    );
}

/// `block_storage_col` resolves all `K + 1` boundaries: the two endpoints to the
/// state columns (`k = 0 → storage_in[h]`, `k = K → storage[h] = h`) and the
/// `K − 1` interiors into the interior storage-boundary family at stride
/// `n_blks − 1`. At `K = 1` only the two endpoints resolve (no interior column
/// is addressed).
#[test]
fn block_storage_col_resolves_all_boundaries() {
    let mut fixtures = TwoHydroFixtures::new();
    let ctx = fixtures.make_ctx();

    let stage_chrono_k3 = stage_with_blocks(BlockMode::Chronological, 3);
    let chrono_k3 = StageLayout::new(&ctx, &stage_chrono_k3, 0);
    let h = 1;
    assert_eq!(
        chrono_k3.block_storage_col(HydroSys::new(h), Boundary::Incoming),
        chrono_k3.state.storage_in.start + h,
        "k = 0 resolves to the incoming-state column storage_in[h]"
    );
    assert_eq!(
        chrono_k3.block_storage_col(HydroSys::new(h), Boundary::Outgoing),
        h,
        "k = K resolves to the outgoing-state column storage[h] = storage.start + h = h"
    );
    let interior_1 = chrono_k3.block_storage_col(HydroSys::new(h), Boundary::Interior(1));
    let interior_2 = chrono_k3.block_storage_col(HydroSys::new(h), Boundary::Interior(2));
    assert_eq!(
        interior_1,
        chrono_k3.geometry.storage_internal_start + h * 2,
        "k = 1 resolves to storage_internal_start + h * (K - 1) + 0"
    );
    assert_eq!(
        interior_2,
        chrono_k3.geometry.storage_internal_start + h * 2 + 1,
        "k = 2 resolves to storage_internal_start + h * (K - 1) + 1"
    );
    let interior_range =
        chrono_k3.geometry.storage_internal_start..chrono_k3.geometry.turbine.start;
    assert!(
        interior_range.contains(&interior_1) && interior_range.contains(&interior_2),
        "both interior columns lie within the interior storage-boundary range"
    );

    let stage_chrono_k1 = stage_with_blocks(BlockMode::Chronological, 1);
    let chrono_k1 = StageLayout::new(&ctx, &stage_chrono_k1, 0);
    assert_eq!(
        chrono_k1.geometry.storage_internal_start, chrono_k1.geometry.turbine.start,
        "K = 1 has no interior storage columns"
    );
    for h in 0..ctx.hydros.len() {
        assert_eq!(
            chrono_k1.block_storage_col(HydroSys::new(h), Boundary::Incoming),
            chrono_k1.state.storage_in.start + h,
            "K = 1 endpoint k = 0 resolves to storage_in[h]"
        );
        assert_eq!(
            chrono_k1.block_storage_col(HydroSys::new(h), Boundary::Outgoing),
            h,
            "K = 1 endpoint k = K = 1 resolves to storage[h] = h"
        );
    }
}

/// `StageLayout::block_storage_col` and `StageGeometry::block_storage_col` both
/// resolve their endpoint arms to the exact same columns `StateSpace`'s own
/// `storage_incoming_col`/`storage_outgoing_col` accessors return, for every
/// hydro — the migration-proof pin for routing `StorageBoundaryGrid`'s endpoint
/// arms through those accessors instead of its own copied state bases.
#[test]
fn storage_boundary_endpoints_match_state_space_accessors() {
    let mut fixtures = TwoHydroFixtures::new();
    let ctx = fixtures.make_ctx();
    let stage = stage_with_blocks(BlockMode::Chronological, 3);
    let layout = StageLayout::new(&ctx, &stage, 0);
    let geometry = layout.geometry.clone();
    assert!(ctx.state.hydro_count >= 2);
    let mut compared = 0;
    for h in 0..ctx.state.hydro_count {
        let incoming = ctx.state.storage_incoming_col(HydroSys::new(h)).get();
        let outgoing = ctx.state.storage_outgoing_col(HydroSys::new(h)).get();
        assert_eq!(
            layout.block_storage_col(HydroSys::new(h), Boundary::Incoming),
            incoming
        );
        assert_eq!(
            layout.block_storage_col(HydroSys::new(h), Boundary::Outgoing),
            outgoing
        );
        assert_eq!(
            geometry.block_storage_col(ctx.state, HydroSys::new(h), Boundary::Incoming),
            incoming
        );
        assert_eq!(
            geometry.block_storage_col(ctx.state, HydroSys::new(h), Boundary::Outgoing),
            outgoing
        );
        compared += 4;
    }
    assert_eq!(compared, 4 * ctx.state.hydro_count);
}

/// The water-balance block spans `n_h` rows in parallel mode and `n_h * n_blks`
/// in chronological mode (the `K` chained per-hydro rows), with `K = 1`
/// chronological collapsing to the parallel count. `load_balance.start()` chains off
/// `water_balance.end()` in every case.
#[test]
fn chronological_water_balance_row_count() {
    let mut fixtures = TwoHydroFixtures::new();
    let ctx = fixtures.make_ctx();

    let stage_parallel = stage_with_blocks(BlockMode::Parallel, 3);
    let parallel = StageLayout::new(&ctx, &stage_parallel, 0);
    assert_eq!(
        parallel.geometry.water_balance.end() - parallel.geometry.water_balance.start(),
        2,
        "parallel n_h=2 n_blks=3 water_balance spans n_h = 2 rows"
    );
    assert_eq!(
        parallel.geometry.load_balance.start(),
        parallel.geometry.water_balance.end(),
        "parallel load_balance.start() chains off water_balance.end()"
    );

    let stage_chrono_k3 = stage_with_blocks(BlockMode::Chronological, 3);
    let chrono_k3 = StageLayout::new(&ctx, &stage_chrono_k3, 0);
    assert_eq!(
        chrono_k3.geometry.water_balance.end() - chrono_k3.geometry.water_balance.start(),
        6,
        "chronological n_h=2 n_blks=3 water_balance spans n_h * n_blks = 6 rows"
    );
    assert_eq!(
        chrono_k3.geometry.load_balance.start(),
        chrono_k3.geometry.water_balance.end(),
        "chronological K=3 load_balance.start() chains off water_balance.end()"
    );

    let stage_chrono_k1 = stage_with_blocks(BlockMode::Chronological, 1);
    let chrono_k1 = StageLayout::new(&ctx, &stage_chrono_k1, 0);
    assert_eq!(
        chrono_k1.geometry.water_balance.end() - chrono_k1.geometry.water_balance.start(),
        2,
        "chronological n_h=2 n_blks=1 water_balance spans n_h = 2 rows, identical to parallel"
    );
    assert_eq!(
        chrono_k1.geometry.load_balance.start(),
        chrono_k1.geometry.water_balance.end(),
        "chronological K=1 load_balance.start() chains off water_balance.end()"
    );
}

/// In parallel mode `push_z_inflow_coupling` loops
/// `0..water_balance.rows_per_entity(n_blks) == 1`, so `z_h`'s column carries
/// exactly one WATER-ROW entry per target hydro regardless of `n_blks` — a
/// `Σ_k` per-block loop would inflate the routed-entry count instead. `z_h`
/// also carries a second entry on its own z-inflow definition row
/// ([`super::super::entries::fill_z_inflow_entries`]), which this test does
/// not count.
#[test]
fn parallel_z_inflow_column_enters_each_target_water_row_once() {
    let mut fixtures = TwoHydroFixtures::new();
    let ctx = fixtures.make_ctx();
    let stage = stage_with_blocks(BlockMode::Parallel, 3);
    let layout = StageLayout::new(&ctx, &stage, 0);

    let col_entries = build_stage_matrix_entries(&ctx, &stage, 0, &layout);
    let water_rows = layout.geometry.water_balance.range();

    for h in 0..ctx.hydros.len() {
        let z_h = layout.state.z_inflow.start + h;
        let water_row_entries = col_entries[z_h]
            .iter()
            .filter(|&&(row, _)| water_rows.contains(&row))
            .count();
        assert_eq!(
            water_row_entries, 1,
            "hydro {h}'s z-inflow column must carry exactly one water-row entry in parallel mode"
        );
    }
}

/// `StageGeometry::water_balance_row` collapses every block onto the single
/// stage row on a parallel stage, and strides `n_blks` block-major rows per
/// hydro on a chronological stage.
#[test]
fn water_balance_row_collapses_parallel_blocks_and_strides_chronological_blocks() {
    use super::StageGeometry;

    let parallel = StageGeometry {
        water_balance: BlockRowFamily::one_per_entity(2..4),
        n_blks: 3,
        block_mode: BlockMode::Parallel,
        ..crate::test_support::equipment_free_geometry(&[3]).remove(0)
    };
    assert_eq!(
        parallel.water_balance_row(HydroSys::new(1), BlockIdx::new(2)),
        3
    );

    let chronological = StageGeometry {
        water_balance: BlockRowFamily::per_block(2..8),
        n_blks: 3,
        block_mode: BlockMode::Chronological,
        ..crate::test_support::equipment_free_geometry(&[3]).remove(0)
    };
    assert_eq!(
        chronological.water_balance_row(HydroSys::new(1), BlockIdx::new(2)),
        7
    );
    for h in 0..2 {
        for k in 0..3 {
            assert_eq!(
                chronological.water_balance_row(HydroSys::new(h), BlockIdx::new(k)),
                2 + h * 3 + k
            );
        }
    }
}

/// `StageGeometry::load_balance_row` strides buses by the block count,
/// regardless of `block_mode`.
#[test]
fn load_balance_row_strides_buses_by_the_block_count() {
    use super::StageGeometry;

    let geometry = StageGeometry {
        load_balance: BlockRowFamily::per_block(10..22),
        n_blks: 4,
        block_mode: BlockMode::Parallel,
        ..crate::test_support::equipment_free_geometry(&[4]).remove(0)
    };
    for bus in 0..3 {
        for k in 0..4 {
            assert_eq!(
                geometry.load_balance_row(BusSys::new(bus), BlockIdx::new(k)),
                10 + bus * 4 + k
            );
        }
    }
}

/// Each `StageGeometry` one-per-entity column accessor resolves to
/// `family.start + local`.
#[test]
fn stage_geometry_entity_col_accessors_match_hand_offsets() {
    use super::StageGeometry;

    let geometry = StageGeometry {
        anticipated_decision: 40..43,
        inflow_slack: 10..13,
        withdrawal_slack_neg: 13..16,
        withdrawal_slack_pos: 16..19,
        filling_target_col: 30..32,
        filled_min_storage_floor_col: 32..33,
        ..crate::test_support::equipment_free_geometry(&[0]).remove(0)
    };
    assert_eq!(
        geometry.anticipated_decision_col(AnticipatedLocal::new(2)),
        42
    );
    assert_eq!(geometry.inflow_slack_col(HydroSys::new(1)), 11);
    assert_eq!(geometry.withdrawal_slack_neg_col(HydroSys::new(2)), 15);
    assert_eq!(geometry.withdrawal_slack_pos_col(HydroSys::new(0)), 16);
    assert_eq!(
        geometry.filling_target_slack_col(FillingTargetLocal::new(1)),
        31
    );
    assert_eq!(
        geometry.filled_min_storage_floor_slack_col(FloorLocal::new(0)),
        32
    );
}

/// `inflow_slack_col` debug-asserts the hydro is inside the family.
#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "outside")]
fn inflow_slack_col_rejects_a_hydro_past_the_family() {
    use super::StageGeometry;

    let geometry = StageGeometry {
        inflow_slack: 10..13,
        ..crate::test_support::equipment_free_geometry(&[0]).remove(0)
    };
    let _ = geometry.inflow_slack_col(HydroSys::new(3));
}

// ── FPHA-local inverse map ───────────────────────────────────────────────

/// Owns the data needed to construct a three-hydro `TemplateBuildCtx` with a
/// single FPHA hydro at system index 1 (the other two use constant
/// productivity), so `StageLayout::new` derives `fpha_hydro_indices == [1]`.
struct FphaMixFixtures {
    base: CtxFixture,
}

impl FphaMixFixtures {
    fn new() -> Self {
        use crate::hydro_models::{EvaporationModel, FphaPlane, ResolvedProductionModel};

        let constant = ResolvedProductionModel::ConstantProductivity { productivity: 0.0 };
        let fpha = ResolvedProductionModel::Fpha {
            planes: vec![FphaPlane {
                intercept: 0.0,
                gamma_v: 0.0,
                gamma_q: 0.0,
                gamma_s: 0.0,
            }],
        };
        // models[hydro][stage]: hydro 1 is FPHA, hydros 0 and 2 are constant.
        let models = vec![vec![constant.clone()], vec![fpha], vec![constant]];
        // All three hydros are non-filling: `filling_phase` is `Operating` at
        // every stage, so the filling exclusion never fires and these fixtures
        // assert the same FPHA membership a pre-gate build would (parity-neutral).
        let hydros = vec![
            membership_hydro(1, false, None, None),
            membership_hydro(2, true, None, None),
            membership_hydro(3, false, None, None),
        ];
        let cascade = CascadeTopology::build(&hydros);
        let hydro_cell_index = HydroCellIndex::build(&hydros);
        let production_models = ProductionModelSet::new(models, &hydros, 1);
        Self {
            base: CtxFixture {
                hydros,
                hydro_cell_index,
                cascade,
                production_models,
                evaporation_models: EvaporationModelSet::new(vec![
                    EvaporationModel::None,
                    EvaporationModel::None,
                    EvaporationModel::None,
                ]),
                ..CtxFixture::default()
            },
        }
    }

    fn make_ctx(&mut self) -> TemplateBuildCtx<'_> {
        self.base.ctx()
    }
}

/// `StageLayout::new` inverts `fpha_hydro_indices` into `fpha_local_index`:
/// the FPHA hydro at system index 1 of three maps to local index 0, and the
/// two non-FPHA hydros stay `None`, giving `[None, Some(0), None]`.
#[test]
fn stage_layout_populates_fpha_local_index_inverse_map() {
    let mut fixtures = FphaMixFixtures::new();
    let ctx = fixtures.make_ctx();
    let stage = minimal_stage();
    let layout = StageLayout::new(&ctx, &stage, 0);

    assert_eq!(
        layout.geometry.fpha_hydro_indices,
        vec![HydroSys::new(1)],
        "only the system-index-1 hydro uses FPHA"
    );
    assert_eq!(
        layout.fpha_local_index,
        vec![None, Some(FphaLocal::new(0)), None],
        "fpha_local_index inverts fpha_hydro_indices over n_h = 3"
    );
}

// ── Per-stage FPHA / evaporation filling exclusion ───────────────────────

/// Owns a two-hydro `TemplateBuildCtx` for the filling-phase membership
/// tests. Hydro 0 is an FPHA **filling** hydro; hydro 1 is a non-FPHA
/// **filling** hydro carrying a linearized evaporation model. Both share the
/// filling window `start_stage_id = 1`, `entry_stage_id = 3`, so a single
/// fixture exercises every phase by varying only `stage.id`:
/// `0` ⇒ `PreFilling`, `1`/`2` ⇒ `Filling`, `≥ 3` ⇒ `Operating`.
struct FillingMembershipFixtures {
    base: CtxFixture,
}

impl FillingMembershipFixtures {
    const START_STAGE_ID: i32 = 1;
    const ENTRY_STAGE_ID: i32 = 3;

    fn new() -> Self {
        use crate::hydro_models::{
            EvaporationModel, FphaPlane, LinearizedEvaporation, ResolvedProductionModel,
        };

        let filling = || {
            Some(FillingConfig {
                start_stage_id: Self::START_STAGE_ID,
                filling_min_rate_m3s: 0.0,
            })
        };
        let entry = Some(Self::ENTRY_STAGE_ID);
        let hydros = vec![
            membership_hydro(1, true, filling(), entry),
            membership_hydro(2, false, filling(), entry),
        ];
        let cascade = CascadeTopology::build(&hydros);
        let hydro_cell_index = HydroCellIndex::build(&hydros);

        // Production: hydro 0 is FPHA at stage 0; hydro 1 is constant.
        let fpha = ResolvedProductionModel::Fpha {
            planes: vec![FphaPlane {
                intercept: 0.0,
                gamma_v: 0.0,
                gamma_q: 0.0,
                gamma_s: 0.0,
            }],
        };
        let constant = ResolvedProductionModel::ConstantProductivity { productivity: 0.0 };
        let models = vec![vec![fpha], vec![constant]];

        // Evaporation: hydro 0 has none; hydro 1 is linearized. The
        // `Linearized` variant is per-hydro, so membership does not depend on
        // `stage_idx`.
        let evaporation_models = EvaporationModelSet::new(vec![
            EvaporationModel::None,
            EvaporationModel::Linearized {
                coefficients: vec![LinearizedEvaporation {
                    intercept_m3s: 0.0,
                    volume_slope_m3s_per_hm3: 0.0,
                }],
                reference_volumes_hm3: vec![0.0],
            },
        ]);

        let production_models = ProductionModelSet::new(models, &hydros, 1);
        Self {
            base: CtxFixture {
                hydros,
                cascade,
                hydro_cell_index,
                production_models,
                evaporation_models,
                ..CtxFixture::default()
            },
        }
    }

    fn make_ctx(&mut self) -> TemplateBuildCtx<'_> {
        self.base.ctx()
    }

    /// `fpha_hydro_indices` for a stage built at `stage_id` (`stage_idx` held
    /// at 0 so the single FPHA/evaporation model row serves every phase).
    fn fpha_indices_at(&mut self, stage_id: i32) -> Vec<HydroSys> {
        let ctx = self.make_ctx();
        let stage = stage_with_id(stage_id);
        StageLayout::new(&ctx, &stage, 0)
            .geometry
            .fpha_hydro_indices
    }

    /// `evap_hydro_indices` for a stage built at `stage_id`.
    fn evap_indices_at(&mut self, stage_id: i32) -> Vec<HydroSys> {
        let ctx = self.make_ctx();
        let stage = stage_with_id(stage_id);
        StageLayout::new(&ctx, &stage, 0)
            .geometry
            .evap_hydro_indices
    }

    /// `filling_target_hydro_indices` for a stage built at `stage_id`.
    fn filling_target_indices_at(&mut self, stage_id: i32) -> Vec<HydroSys> {
        let ctx = self.make_ctx();
        let stage = stage_with_id(stage_id);
        StageLayout::new(&ctx, &stage, 0)
            .geometry
            .filling_target_hydro_indices
    }

    /// `filled_min_storage_floor_hydro_indices` for a stage built at `stage_id`.
    fn filled_min_storage_floor_indices_at(&mut self, stage_id: i32) -> Vec<HydroSys> {
        let ctx = self.make_ctx();
        let stage = stage_with_id(stage_id);
        StageLayout::new(&ctx, &stage, 0)
            .geometry
            .filled_min_storage_floor_hydro_indices
    }
}

/// The per-stage `σ_fill` target is emitted at EVERY Filling stage, not only at
/// `entry − 1`. Both filling hydros share `start = 1`, `entry = 3`, so the
/// Filling stages are `{1, 2}`; both carry the target at BOTH. `PreFilling` (id 0)
/// and Operating (id ≥ 3) emit none. The wrong-but-compiling alternative is the
/// v1 terminal-only rule (`entry − 1 == stage_id`), which would drop the id-1
/// floor; this test pins per-stage Filling membership.
#[test]
fn filling_target_emitted_at_every_filling_stage() {
    let mut fixtures = FillingMembershipFixtures::new();

    // Filling stages 1 and 2 (start = 1, entry = 3): both filling hydros
    // (system indices 0, 1) carry the target at every Filling stage.
    for stage_id in [1, 2] {
        assert_eq!(
            fixtures.filling_target_indices_at(stage_id),
            vec![HydroSys::new(0), HydroSys::new(1)],
            "both filling hydros carry the σ_fill target at Filling id {stage_id}"
        );
    }

    // PreFilling (id 0) and Operating (id ≥ entry = 3) emit NO target.
    for stage_id in [0, 3, 4] {
        assert_eq!(
            fixtures.filling_target_indices_at(stage_id),
            Vec::<HydroSys>::new(),
            "no σ_fill target at non-Filling id {stage_id}"
        );
    }
}

/// Parity-neutrality: a non-filling system never emits a `σ_fill` target, so
/// `num_rows` is bit-identical across every stage id (the cut-row region anchor
/// is unmoved). The forbidden alternative — reserving a target row for every
/// hydro unconditionally — would shift `num_rows` and alias the append-only cut
/// rows for the existing non-filling deterministic cases.
#[test]
fn non_filling_system_no_filling_target_num_rows_unchanged() {
    // `FphaMixFixtures` hydros are all non-filling.
    let mut fixtures = FphaMixFixtures::new();
    let mut layout_at = |stage_id: i32| {
        let ctx = fixtures.make_ctx();
        let stage = stage_with_id(stage_id);
        let layout = StageLayout::new(&ctx, &stage, 0);
        (
            layout.geometry.filling_target_hydro_indices.clone(),
            layout.rows.num_rows,
        )
    };
    let (reference_targets, reference_num_rows) = layout_at(0);
    assert_eq!(
        reference_targets,
        Vec::<HydroSys>::new(),
        "non-filling system emits no σ_fill target"
    );
    for stage_id in [1, 2, 3, 7] {
        let (targets, num_rows) = layout_at(stage_id);
        assert_eq!(
            targets,
            Vec::<HydroSys>::new(),
            "non-filling σ_fill target empty at id {stage_id}"
        );
        assert_eq!(
            num_rows, reference_num_rows,
            "non-filling num_rows unchanged at id {stage_id}"
        );
    }
}

/// The `σ_fill` row block lands STRICTLY BELOW `num_rows` (the pre-cut
/// region), ahead of the append-only cut rows that begin at `num_rows`. A row
/// at index `>= num_rows` would alias a cut row and corrupt slot-identity
/// warm-start reconstruction. The `σ_fill` column likewise lands strictly below
/// `num_cols`. The `filling_target` block is the FIRST pre-cut filling-row
/// family, so it follows the operational-violation rows directly (no retention
/// block precedes it).
#[test]
fn filling_target_row_and_col_below_structural_bounds() {
    let mut fixtures = FillingMembershipFixtures::new();
    let ctx = fixtures.make_ctx();
    let stage = stage_with_id(2); // entry − 1: the terminal stage.
    let layout = StageLayout::new(&ctx, &stage, 0);

    let n_targets = layout.geometry.filling_target_hydro_indices.len();
    assert_eq!(n_targets, 2, "both filling hydros carry the target at id 2");

    let row_start = layout.geometry.filling_target.start;
    for local_idx in 0..n_targets {
        assert!(
            row_start + local_idx < layout.rows.num_rows,
            "σ_fill row {} must be < num_rows {}",
            row_start + local_idx,
            layout.rows.num_rows
        );
    }
    assert_eq!(
        row_start, layout.oper_violation.min_generation.end,
        "σ_fill rows follow the operational-violation rows directly"
    );

    let col_start = layout.geometry.filling_target_col.start;
    for local_idx in 0..n_targets {
        assert!(
            col_start + local_idx < layout.num_cols,
            "σ_fill col {} must be < num_cols {}",
            col_start + local_idx,
            layout.num_cols
        );
    }
    // At the terminal Filling stage (id 2) no hydro is Operating, so the
    // sibling σ^{v-} `filled_min_storage_floor` column block (the true last column family)
    // is empty: its start coincides with num_cols and the σ_fill block is the
    // last occupied family, so num_cols = col_filling_target_start + n_targets.
    assert_eq!(
        layout.geometry.filled_min_storage_floor_col.start, layout.num_cols,
        "σ^{{v-}} column block empty at the terminal Filling stage"
    );
    assert_eq!(
        layout.num_cols,
        col_start + n_targets,
        "num_cols = col_filling_target_start + n_targets (σ^{{v-}} block empty here)"
    );
}

/// The `σ_fill` target family adds rows at EVERY Filling stage (ids 1, 2 here)
/// and NONE at `PreFilling` (id 0) or Operating (id ≥ 3). The non-Filling stages
/// keep an empty target block (the fishing-row start coincides with the
/// target-row start), isolating the per-stage target rows to the Filling window.
#[test]
fn filling_target_adds_rows_at_every_filling_stage() {
    let mut fixtures = FillingMembershipFixtures::new();
    // PreFilling (id 0) and Operating (id 3, 4): the σ_fill TARGET adds no rows.
    for stage_id in [0, 3, 4] {
        let ctx = fixtures.make_ctx();
        let stage = stage_with_id(stage_id);
        let layout = StageLayout::new(&ctx, &stage, 0);
        assert!(
            layout.geometry.filling_target_hydro_indices.is_empty(),
            "no σ_fill target rows at non-Filling id {stage_id}"
        );
    }
    // Every Filling stage (ids 1, 2) adds exactly 2 target rows (one per hydro).
    for stage_id in [1, 2] {
        assert_eq!(
            fixtures.filling_target_indices_at(stage_id).len(),
            2,
            "Filling id {stage_id} adds one σ_fill target row per filling hydro"
        );
    }
}

/// The soft `σ^{v-}` operating floor is emitted at EVERY `Operating` stage of a
/// filling hydro (id ≥ entry = 3), for BOTH filling hydros — distinct from the
/// every-Filling-stage `σ_fill` target. `PreFilling` (id 0) and `Filling` (id 1, 2)
/// emit none. This pins the `Operating`-only scope and the `σ^{v-}`/`σ_fill`
/// stage split.
#[test]
fn filled_min_storage_floor_emitted_at_every_operating_stage() {
    let mut fixtures = FillingMembershipFixtures::new();

    // Operating (id >= entry = 3): both filling hydros carry the floor at every
    // stage, not just one terminal stage.
    for stage_id in [3, 4, 7] {
        assert_eq!(
            fixtures.filled_min_storage_floor_indices_at(stage_id),
            vec![HydroSys::new(0), HydroSys::new(1)],
            "both filling hydros carry σ^{{v-}} at Operating id {stage_id}"
        );
    }

    // PreFilling (id 0) and Filling (id 1, 2 = the σ_fill terminal): no floor.
    for stage_id in [0, 1, 2] {
        assert_eq!(
            fixtures.filled_min_storage_floor_indices_at(stage_id),
            Vec::<HydroSys>::new(),
            "no σ^{{v-}} at non-operating id {stage_id}"
        );
    }

    // Mutual exclusivity at the boundary: id 2 (entry − 1) carries σ_fill but
    // NOT σ^{v-}; id 3 (entry) carries σ^{v-} but NOT σ_fill.
    assert_eq!(
        fixtures.filling_target_indices_at(2),
        vec![HydroSys::new(0), HydroSys::new(1)]
    );
    assert!(fixtures.filled_min_storage_floor_indices_at(2).is_empty());
    assert!(fixtures.filling_target_indices_at(3).is_empty());
    assert_eq!(
        fixtures.filled_min_storage_floor_indices_at(3),
        vec![HydroSys::new(0), HydroSys::new(1)]
    );
}

/// Parity-neutrality: a non-filling system never emits a `σ^{v-}` floor, so
/// `num_rows` is bit-identical across every stage id. The forbidden GLOBAL soft
/// floor — reserving a floor row for every Operating hydro regardless of
/// `filling` — would shift `num_rows` and alias the append-only cut rows for the
/// existing deterministic cases.
#[test]
fn non_filling_system_no_filled_min_storage_floor_num_rows_unchanged() {
    let mut fixtures = FphaMixFixtures::new();
    let mut layout_at = |stage_id: i32| {
        let ctx = fixtures.make_ctx();
        let stage = stage_with_id(stage_id);
        let layout = StageLayout::new(&ctx, &stage, 0);
        (
            layout
                .geometry
                .filled_min_storage_floor_hydro_indices
                .clone(),
            layout.rows.num_rows,
        )
    };
    let (reference_floors, reference_num_rows) = layout_at(0);
    assert_eq!(
        reference_floors,
        Vec::<HydroSys>::new(),
        "non-filling system emits no σ^{{v-}} floor"
    );
    for stage_id in [1, 2, 3, 7] {
        let (floors, num_rows) = layout_at(stage_id);
        assert_eq!(
            floors,
            Vec::<HydroSys>::new(),
            "non-filling σ^{{v-}} floor empty at id {stage_id}"
        );
        assert_eq!(
            num_rows, reference_num_rows,
            "non-filling num_rows unchanged at id {stage_id}"
        );
    }
}

/// A filling FPHA hydro is excluded from `fpha_hydro_indices` while
/// `Filling` (its FPHA fit is invalid below `min_storage`), and re-included
/// once `Operating`. The forbidden alternative — leaving it in the set during
/// filling — would emit an FPHA production row over an invalid operating-range
/// fit and a generation column with no constraining row.
#[test]
fn filling_fpha_hydro_excluded_while_filling_present_when_operating() {
    let mut fixtures = FillingMembershipFixtures::new();

    // Filling (stage_id 1 and 2 are in `[start_stage_id, entry_stage_id)`):
    // hydro 0 (the FPHA hydro) is absent.
    assert_eq!(
        fixtures.fpha_indices_at(1),
        Vec::<HydroSys>::new(),
        "FPHA filling hydro absent from fpha_hydro_indices during Filling"
    );
    assert_eq!(
        fixtures.fpha_indices_at(2),
        Vec::<HydroSys>::new(),
        "FPHA filling hydro absent at the last Filling stage"
    );

    // Operating (stage_id >= entry_stage_id): hydro 0 re-enters.
    assert_eq!(
        fixtures.fpha_indices_at(3),
        vec![HydroSys::new(0)],
        "FPHA filling hydro present from the first Operating stage"
    );
    assert_eq!(
        fixtures.fpha_indices_at(4),
        vec![HydroSys::new(0)],
        "FPHA filling hydro present at later Operating stages"
    );

    // PreFilling (stage_id < start_stage_id): the dam does not exist yet, so
    // the FPHA hydro is also excluded.
    assert_eq!(
        fixtures.fpha_indices_at(0),
        Vec::<HydroSys>::new(),
        "FPHA filling hydro absent during PreFilling"
    );
}

/// A filling hydro with evaporation is excluded from `evap_hydro_indices`
/// only during `PreFilling` (no reservoir surface), and present during
/// `Filling` and `Operating` (the impounding reservoir has a surface). This
/// is the opposite of the FPHA rule, which also excludes during `Filling` —
/// the two exclusions must not be unified.
#[test]
fn filling_evap_hydro_excluded_only_in_prefilling() {
    let mut fixtures = FillingMembershipFixtures::new();

    // PreFilling (stage_id < start_stage_id): hydro 1 (evaporation) is absent.
    assert_eq!(
        fixtures.evap_indices_at(0),
        Vec::<HydroSys>::new(),
        "evaporation filling hydro absent during PreFilling (no reservoir surface)"
    );

    // Filling: evaporation is normal — the reservoir already has a surface.
    assert_eq!(
        fixtures.evap_indices_at(1),
        vec![HydroSys::new(1)],
        "evaporation filling hydro present during Filling"
    );
    assert_eq!(
        fixtures.evap_indices_at(2),
        vec![HydroSys::new(1)],
        "evaporation filling hydro present at the last Filling stage"
    );

    // Operating: evaporation remains normal.
    assert_eq!(
        fixtures.evap_indices_at(3),
        vec![HydroSys::new(1)],
        "evaporation filling hydro present once Operating"
    );
}

/// Parity-neutrality contract: a non-filling hydro is `Operating` at every
/// stage, so neither exclusion fires — its membership in both
/// `fpha_hydro_indices` and `evap_hydro_indices` is bit-identical across all
/// stages, matching a build without the filling gate.
#[test]
fn non_filling_hydro_membership_bit_identical_across_stages() {
    // The `FphaMixFixtures` hydros are all non-filling (one FPHA at system
    // index 1, two constant), so its membership must be invariant to stage_id.
    let mut fixtures = FphaMixFixtures::new();
    let (reference_fpha, reference_evap) = {
        let ctx = fixtures.make_ctx();
        let stage = stage_with_id(0);
        let layout = StageLayout::new(&ctx, &stage, 0);
        (
            layout.geometry.fpha_hydro_indices,
            layout.geometry.evap_hydro_indices,
        )
    };

    assert_eq!(reference_fpha, vec![HydroSys::new(1)]);
    assert_eq!(reference_evap, Vec::<HydroSys>::new());

    for stage_id in [1, 2, 3, 7] {
        let ctx = fixtures.make_ctx();
        let stage = stage_with_id(stage_id);
        let layout = StageLayout::new(&ctx, &stage, 0);
        assert_eq!(
            layout.geometry.fpha_hydro_indices, reference_fpha,
            "non-filling fpha_hydro_indices must be stage-invariant (stage_id {stage_id})"
        );
        assert_eq!(
            layout.geometry.evap_hydro_indices, reference_evap,
            "non-filling evap_hydro_indices must be stage-invariant (stage_id {stage_id})"
        );
    }
}

// ── Operational-violation row ranges ─────────────────────────────────────

/// The four operational-violation row families (`min_outflow`,
/// `max_outflow`, `min_turbine`, `min_generation`) are
/// contiguous, in that order, each spanning exactly `n_h * n_blks` rows, and
/// the block starts immediately after the post-equipment row cursor
/// (`evap_rows_end`), which equals `min_outflow.start`. The owning
/// arithmetic lives in [`StageLayout::new`]; this pins it at the internal
/// layer where the row ranges are visible. The forbidden alternative is a
/// stale or transposed base — placing `max_outflow` before `min_outflow`, or
/// striding any family by something other than `n_h * n_blks`, addresses the
/// wrong constraint rows and silently mis-bounds the operational violations.
///
/// Fixture: `n_h = 3`, one block (`n_blks = 1`), so `n_op = 3` per family.
#[test]
fn stage_layout_operational_violation_rows_are_contiguous_blocks() {
    let mut fixtures = FphaMixFixtures::new();
    let ctx = fixtures.make_ctx();
    let stage = minimal_stage();
    let layout = StageLayout::new(&ctx, &stage, 0);

    let n_op = ctx.hydros.len(); // n_h * n_blks with n_blks == 1
    assert!(
        n_op > 0,
        "fixture must have hydros so the rows are non-empty"
    );

    assert_eq!(
        layout.oper_violation.min_outflow.len(),
        n_op,
        "min_outflow row count"
    );
    assert_eq!(
        layout.oper_violation.max_outflow.len(),
        n_op,
        "max_outflow row count"
    );
    assert_eq!(
        layout.oper_violation.min_turbine.len(),
        n_op,
        "min_turbine row count"
    );
    assert_eq!(
        layout.oper_violation.min_generation.len(),
        n_op,
        "min_generation row count"
    );

    assert_eq!(
        layout.oper_violation.max_outflow.start, layout.oper_violation.min_outflow.end,
        "max_outflow must follow min_outflow contiguously"
    );
    assert_eq!(
        layout.oper_violation.min_turbine.start, layout.oper_violation.max_outflow.end,
        "min_turbine must follow max_outflow contiguously"
    );
    assert_eq!(
        layout.oper_violation.min_generation.start,
        layout.oper_violation.max_outflow.end + n_op,
        "min_generation must start one min_turbine block (n_op rows) after max_outflow ends"
    );
}

// ── Block-strided address pin (rows, thermal and excess columns) ────────

/// Every position `k` in a block-major row or column family resolves to
/// `range.start + k` via `BlockGrid::flat` at `(k / n_blks, k % n_blks)` — a
/// transposed stride fails the moment `n_blks >= 2`. Returns each family's
/// count, in this order: min outflow, max outflow, min turbine, min
/// generation, thermal, excess.
fn assert_block_strided_addresses(layout: &StageLayout) -> [usize; 6] {
    fn check(
        n_blks: usize,
        range: &Range<usize>,
        label: &str,
        addr: impl Fn(usize, BlockIdx) -> usize,
    ) -> usize {
        assert_eq!(
            range.len() % n_blks,
            0,
            "{label} family length not a multiple of n_blks"
        );
        for k in 0..range.len() {
            let (i, blk) = (k / n_blks, k % n_blks);
            assert_eq!(
                range.start + k,
                addr(i, BlockIdx::new(blk)),
                "{label} address mismatch at k={k}"
            );
        }
        range.len()
    }

    let n_blks = layout.clock.n_blks();
    let geometry = layout.geometry.clone();
    let oper = &layout.oper_violation;

    [
        check(n_blks, &oper.min_outflow, "min outflow", |i, blk| {
            layout.min_outflow_row(HydroSys::new(i), blk)
        }),
        check(n_blks, &oper.max_outflow, "max outflow", |i, blk| {
            layout.max_outflow_row(HydroSys::new(i), blk)
        }),
        check(n_blks, &oper.min_turbine, "min turbine", |i, blk| {
            layout.min_turbine_row(HydroCell::new(i), blk)
        }),
        check(n_blks, &oper.min_generation, "min generation", |i, blk| {
            layout.min_generation_row(HydroCell::new(i), blk)
        }),
        check(n_blks, &geometry.thermal, "thermal", |i, blk| {
            layout.geometry.thermal_col(ThermalSys::new(i), blk)
        }),
        check(n_blks, &geometry.excess, "excess", |i, blk| {
            layout.geometry.excess_col(BusSys::new(i), blk)
        }),
    ]
}

/// Runs [`assert_block_strided_addresses`] over `FphaMixFixtures` (hydros and
/// cells, one block) and over a `ZeroEntityFixtures` copy with
/// `n_thermals`/`n_buses` set and four blocks, so every family's summed count
/// is nonzero and at least one layout is multi-block.
#[test]
fn block_strided_addresses_match_their_family_ranges() {
    let mut fpha_fixtures = FphaMixFixtures::new();
    let fpha_ctx = fpha_fixtures.make_ctx();
    let fpha_stage = minimal_stage();
    let fpha_layout = StageLayout::new(&fpha_ctx, &fpha_stage, 0);
    let fpha_counts = assert_block_strided_addresses(&fpha_layout);

    let mut zero_fixtures = ZeroEntityFixtures::new();
    zero_fixtures.base.thermals = vec![dormant_thermal(0), dormant_thermal(1)];
    zero_fixtures.base.buses = vec![dormant_bus(0), dormant_bus(1)];
    let thermal_ctx = zero_fixtures.make_ctx(vec![], &[]);
    let thermal_stage = stage_with_blocks(BlockMode::Parallel, 4);
    let thermal_layout = StageLayout::new(&thermal_ctx, &thermal_stage, 0);
    assert_eq!(
        thermal_layout.clock.n_blks(),
        4,
        "fixture must build a 4-block layout"
    );
    let thermal_counts = assert_block_strided_addresses(&thermal_layout);

    for (idx, (fpha, thermal)) in fpha_counts.iter().zip(thermal_counts).enumerate() {
        assert!(
            fpha + thermal > 0,
            "family {idx} has zero rows/cols across both layouts"
        );
    }
}

// ── Anticipated-decision column positioning ──────────────────────────────

/// `geometry.anticipated_decision.start` falls between thermal end and
/// `geometry.line_fwd.start` when `n_anticipated=2, n_thermals=3, n_blks=4`.
///
/// The control region is `thermal` then `anticipated_decision` (2 cols) then
/// `line_fwd` — the anticipated ring's outgoing slots live entirely in the
/// state region. So `geometry.line_fwd.start` equals
/// `geometry.anticipated_decision.start + n_anticipated`, and the ring's own
/// out-block start (`StateSpace::commit_out.start`) is sourced from the
/// state-region position (immediately after `transit_buckets_out`), not from
/// the control region.
#[test]
fn anticipated_decision_columns_placed_between_thermal_and_line_fwd() {
    let mut fixtures = ZeroEntityFixtures::new();
    // ZeroEntityFixtures builds n_thermals=0, so the thermal per-block block is
    // empty and anticipated_decision.start == thermal.start.
    let n_anticipated = 2_usize;
    let ctx = fixtures.make_ctx(vec![1, 1], &[0, 1]);

    let mut stage = minimal_stage();
    stage.blocks = (0..4)
        .map(|index| Block {
            index,
            name: format!("B{index}"),
            duration_hours: 186.0,
        })
        .collect();
    let layout = StageLayout::new(&ctx, &stage, 0);

    assert_eq!(
        layout.geometry.anticipated_decision.start, layout.geometry.thermal.start,
        "anticipated_decision.start must equal thermal.start \
             when n_thermals=0 (no thermal per-block cols)"
    );
    assert_eq!(
        layout.geometry.line_fwd.start,
        layout.geometry.anticipated_decision.start + n_anticipated,
        "line_fwd.start == anticipated_decision.start + n_anticipated \
             (state_out relocated out of the control region)"
    );
    // The outgoing ring start equals the indexer's state-region position:
    // immediately after `transit_buckets_out` (N*(1+L) + B). Here N=0, L=0,
    // B=0 → the ring starts at 0.
    assert_eq!(
        layout.state.commit_out.start, 0,
        "commit_out.start must equal the state-region offset N*(1+L) + B"
    );
    assert_eq!(
        layout.geometry.line_fwd.start - layout.geometry.thermal.start,
        n_anticipated,
        "gap from thermal_start to line_fwd_start must be exactly n_anticipated \
             (only the anticipated_decision block remains in the control region)"
    );
}

/// `StageLayout` with `n_anticipated=2, k_max=3, n_hydros=0,
/// max_par_order=0` has `col_turbine_start == 0*(3+0) + 2*6 + 1 == 13`.
///
/// `n_ant_state = n_anticipated * k_max = 2 * 3 = 6` and the in-LP ring's TWO
/// `n_ant_state`-wide blocks (`commit_out` outgoing +
/// `commit_in` incoming) together shift `theta` from the legacy
/// `N*(3+L) = 0` to `0 + 2*6 = 12`, so decisions begin at 13.
///
/// The general formula (any N, L, B) is
/// `N*(3+L) + 2*B + 2*n_ant_state + 1`.
#[test]
fn stage_layout_with_anticipated_shifts_decision_region() {
    let n_hydros = 0_usize;
    let max_par_order = 0_usize;
    let n_anticipated = 2_usize;
    let k_max = 3_usize;

    let mut fixtures = ZeroEntityFixtures::new();
    let ctx = fixtures.make_ctx(
        vec![2, 3], // anticipated_lead_stages
        &[0, 2],    // anticipated_positions (arbitrary; layout doesn't inspect them)
    );
    let stage = minimal_stage();
    let layout = StageLayout::new(&ctx, &stage, 0);

    let expected_n_ant_state = n_anticipated * k_max;
    assert_eq!(
        layout.state.commit_out.len(),
        expected_n_ant_state,
        "n_ant_state"
    );

    let expected_col_turbine_start = n_hydros * (3 + max_par_order) + 2 * expected_n_ant_state + 1;
    assert_eq!(
        layout.geometry.turbine.start, expected_col_turbine_start,
        "col_turbine_start == N*(3+L) + 2*B + 2*n_ant_state + 1"
    );
}

// ── Anticipated-fishing row positioning ──────────────────────────────────

/// [`super::AnticipatedLayout::fishing_rows`] immediately follows the operational
/// violation row block, i.e. its start equals `row_min_generation_start +
/// n_op_rows`.
///
/// Uses a zero-hydro context so `n_op_rows == 0`, which means the fishing
/// start equals `row_min_generation_start` exactly. The algebraic identity
/// `fishing_rows.start == row_min_generation_start + n_op_rows`
/// is verified for the general formula; the case `n_op_rows > 0` is covered by
/// the production code path (`n_hydros * n_blks` counts operational violation rows).
///
/// Setup: `n_anticipated=2`, `k_max=2`, `anticipated_lead_stages=[1,2]`,
/// zero hydros, one block, `n_stages=4` (`AntFixturesWithNStages`, not the
/// study-stage-count-0 `ZeroEntityFixtures`: `build_anticipated_fishing_row_pos`'s
/// in-study guard reads the fixture's own `n_stages`, so a `stage_idx=1` probe
/// needs a real study-stage count to stay in-study). At `stage_idx=1`:
/// - `n_op_rows = 0 * 1 = 0` (no hydros)
/// - `fishing_rows.start` must equal `row_min_generation_start + 0`
#[test]
fn anticipated_fishing_row_offset_after_operational_violations() {
    let mut fixtures = AntFixturesWithNStages::new(4);
    let ctx = fixtures.make_ctx(
        vec![1, 2], // K_0=1, K_1=2
        &[0, 1],    // arbitrary thermal indices
    );
    let stage = minimal_stage(); // 1 block
    let layout = StageLayout::new(&ctx, &stage, 1);

    // n_op_rows = n_hydros * n_blks = 0 * 1 = 0
    let n_op_rows = 0_usize;
    assert_eq!(
        layout.anticipated.fishing_rows.start,
        layout.oper_violation.min_generation.start + n_op_rows,
        "fishing_rows.start must equal row_min_generation_start + n_op_rows"
    );
    assert_eq!(
        layout.anticipated.fishing_rows.len(),
        2,
        "fishing_rows.len() must equal n_anticipated (2) under always-active predicate"
    );
}

/// [`super::AnticipatedLayout::fishing_rows`]'s length equals `n_anticipated` at
/// every stage under the always-active predicate. With `K_i=[1,2]` and
/// `n_anticipated=2`, the count is 2 at every stage in `[0, 1, 2, 3]`.
/// `n_stages=4` covers the probed range (`AntFixturesWithNStages`, not the
/// study-stage-count-0 `ZeroEntityFixtures` — see the sibling test above for
/// why).
#[test]
fn anticipated_fishing_row_count_grows_with_stage() {
    let mut fixtures = AntFixturesWithNStages::new(4);
    let ctx = fixtures.make_ctx(
        vec![1, 2], // K_0=1, K_1=2
        &[0, 1],    // arbitrary thermal indices
    );
    let stage = minimal_stage(); // 1 block

    for (stage_idx, expected) in [(0_usize, 2), (1, 2), (2, 2), (3, 2)] {
        let layout = StageLayout::new(&ctx, &stage, stage_idx);
        assert_eq!(
            layout.anticipated.fishing_rows.len(),
            expected,
            "fishing_rows.len() must equal {expected} at stage_idx={stage_idx}"
        );
    }
}

/// `num_rows` does not include state-fixing rows; the LP row layout starts
/// directly with `z_inflow_rows` at row 0.
///
/// State pinning uses column bounds, so there is no `[0, n_state)` row
/// prefix. `num_rows` equals the count of structural rows only (`z_inflow`,
/// water balance, load balance, FPHA, evap, operational, fishing,
/// `anticipated_state_out_def`, generic).
///
/// `AntFixturesWithNStages`, not the study-stage-count-0 `ZeroEntityFixtures`:
/// `build_anticipated_fishing_row_pos`'s in-study guard reads the fixture's
/// own `n_stages`, so even `stage_idx=0` needs a real study-stage count.
#[test]
fn num_rows_drops_by_n_state_with_anticipated_thermals() {
    let n_anticipated = 2_usize;
    let k_max = 3_usize;

    let mut fixtures = AntFixturesWithNStages::new(1);
    let ctx = fixtures.make_ctx(vec![3, 2], &[0, 1]);
    let stage = minimal_stage();
    let layout = StageLayout::new(&ctx, &stage, 0);

    // n_state for this fixture: N*(1+L) + A*K = 0 + 2*3 = 6.
    let n_state = ctx.hydros.len() * (1 + ctx.par_lp.max_order()) + n_anticipated * k_max;
    assert_eq!(n_state, 6);

    // num_rows for this zero-hydro fixture: only the anticipated_fishing
    // block contributes (2 active plants at stage 0). All other row blocks
    // are 0 (no hydros, no buses, no FPHA, no evap).
    let observed = layout.rows.num_rows;
    assert_eq!(
        observed, 2,
        "num_rows equals anticipated_fishing_rows (2) for this fixture"
    );

    // Reference value: if state-fixing rows existed, num_rows would be observed + n_state.
    let num_rows_if_state_rows_existed = observed + n_state;
    assert_eq!(
        num_rows_if_state_rows_existed, 8,
        "observed + n_state is 8 for this fixture"
    );
    // Structural invariant proving the reduction: row_water_balance_start
    // equals ctx.hydros.len() (no n_state offset). With state-fixing rows it
    // would be n_state + ctx.hydros.len().
    assert_eq!(
        layout.geometry.water_balance.start(),
        ctx.hydros.len(),
        "row_water_balance_start does not include the n_state offset"
    );
}

// ── Delivery-axis generalization: row positions & masking ────────────────

/// Build a one-plant `StateSpace` carrying an attached delivery-anchored
/// `AnticipatedResolution` directly, not the context's constant-lead
/// resolution; these tests need the real resolved axis.
fn state_with_attached_resolution(k_max: usize, resolution: AnticipatedResolution) -> StateSpace {
    let n_anticipated = resolution.per_plant.len();
    StateSpace::new(
        0,
        0,
        Vec::new(),
        vec![k_max; n_anticipated],
        resolution,
        &[],
    )
}

/// Study-only axis (`state.n_delivery() == 3`, the same width as the study
/// horizon): every returned position must be byte-identical to the
/// pre-generalization `m >= n_stages` skip. `k_max = 2`, `LeadTime::Stages(2)`
/// over 3 study stages, `stage_idx = 1`. Depths `[2, 3]` reach delivery
/// targets `m = [2, 3]`: `m=2` (`decider[2] = Some(0)`) is ready at stage 1
/// and not a fresh deposit there, so slot `2 % 2 = 0` is an interior carry;
/// `m=3 >= n_delivery(3)` is masked. Hand-written, not recomputed, so a
/// regression in the residue arithmetic is caught rather than reproduced.
#[test]
fn build_anticipated_slot_row_pos_study_only_byte_identity() {
    let k_max = 2;
    let resolution = AnticipatedResolution::resolve(
        &[LeadTime::Stages(2)],
        DeliveryAxis {
            study_stage_hours: &[720.0; 3],
            post_study_stage_hours: &[],
        },
    );
    assert_eq!(
        resolution.anchored_depth(),
        k_max,
        "fixture must realize k_max == 2"
    );
    let state = state_with_attached_resolution(k_max, resolution);

    let (row_pos, n_reachable) = build_anticipated_slot_row_pos(&state, 1);

    assert_eq!(
        row_pos,
        vec![Some(0), None],
        "study-only axis must reproduce the exact pre-generalization positions"
    );
    assert_eq!(n_reachable, 1);
}

/// Extended axis (four study stages, four post-study stages: `n_decision =
/// 4`, `n_delivery = 8`), `k_max = 4`, `LeadTime::Stages(4)`. At `stage_idx =
/// 2`, depth `2` reaches delivery target `m = 5` — a POST-STUDY stage
/// (`m >= n_decision`). `decider[5] = Some(1)`: not a fresh deposit at stage
/// 2 (`Some(1) != Some(2)`) and ready (`1 <= 2`), so slot `5 % 4 = 1` must be
/// `Some` — an interior carry — rather than masked by the retired
/// `m >= n_stages` bound.
#[test]
fn build_anticipated_slot_row_pos_extended_axis_carries_post_study_target_m5() {
    let k_max = 4;
    let resolution = AnticipatedResolution::resolve(
        &[LeadTime::Stages(4)],
        DeliveryAxis {
            study_stage_hours: &[720.0; 4],
            post_study_stage_hours: &[720.0; 4],
        },
    );
    assert_eq!(
        resolution.anchored_depth(),
        k_max,
        "fixture must realize k_max == 4"
    );
    let state = state_with_attached_resolution(k_max, resolution);

    let (row_pos, _n_reachable) = build_anticipated_slot_row_pos(&state, 2);

    let slot = 5 % k_max;
    assert!(
        row_pos[slot].is_some(),
        "post-study delivery target m=5 (slot {slot}) must carry as an interior \
         position, not be masked by the retired study-horizon bound"
    );
}

// ── Ring-axis excision: per-plant physical-target mapping ────────────────

/// Hand-derived raw delivery-axis reference: `n_anticipated = 2`, leads
/// `[1, 2]`, 5 study stages, an identity resolution (`n_decision ==
/// n_delivery == 5`, so `g == 0` for both plants), so `k_max = ring_size(&[1,
/// 2]) == 2`. `g == 0` collapses `ring_index`/`physical_target` to the
/// identity, so the ring-axis mapping must equal the raw delivery-axis sweep
/// derived below.
///
/// Derivation (`decider_0 = [None, 0, 1, 2, 3]` for lead 1,
/// `decider_1 = [None, None, 0, 1, 2]` for lead 2; `row_pos` indexed
/// `slot * 2 + plant`; a residue is a deposit iff `decider[m] ==
/// Some(stage_idx)`, else a carry when ready, else absent):
///
/// | `stage_idx` | plant 0 (`m = stage_idx+1`) | plant 1 (`m = stage_idx+1`) | plant 0 (`m = stage_idx+2`) | plant 1 (`m = stage_idx+2`) |
/// | --- | --- | --- | --- | --- |
/// | 0 | `m=1` deposit | `m=1` carry → slot 1 | `m=2` not ready | `m=2` deposit |
/// | 1 | `m=2` deposit | `m=2` carry → slot 0 | `m=3` not ready | `m=3` deposit |
/// | 2 | `m=3` deposit | `m=3` carry → slot 1 | `m=4` not ready | `m=4` deposit |
/// | 3 | `m=4` deposit | `m=4` carry → slot 0 | `m=5` beyond `n_delivery` | `m=5` beyond `n_delivery` |
/// | 4 | `m=5` beyond `n_delivery` | `m=5` beyond `n_delivery` | `m=6` beyond `n_delivery` | `m=6` beyond `n_delivery` |
///
/// Each carry is the sole reachable entry at its stage (`n_reachable == 1`,
/// `0` at stage 4), landing at `slot * 2 + 1` (plant 1 is always the carry).
#[test]
fn anticipated_slot_row_pos_identity_axis_matches_the_recorded_pre_excision_mapping() {
    let leads = vec![1, 2];
    let resolution = constant_lead_resolution(&leads, 5);
    let state = StateSpace::new(0, 0, Vec::new(), leads, resolution, &[]);
    assert_eq!(
        state.k_max, 2,
        "fixture sanity: ring_size(&[1, 2]) must be 2"
    );

    let expected: [(Vec<Option<usize>>, usize); 5] = [
        (vec![None, None, None, Some(0)], 1),
        (vec![None, Some(0), None, None], 1),
        (vec![None, None, None, Some(0)], 1),
        (vec![None, Some(0), None, None], 1),
        (vec![None, None, None, None], 0),
    ];

    for (stage_idx, (expected_row_pos, expected_n_reachable)) in expected.into_iter().enumerate() {
        let (row_pos, n_reachable) = build_anticipated_slot_row_pos(&state, stage_idx);
        assert_eq!(
            row_pos, expected_row_pos,
            "stage_idx={stage_idx}: row_pos must match the recorded pre-excision mapping"
        );
        assert_eq!(
            n_reachable, expected_n_reachable,
            "stage_idx={stage_idx}: n_reachable must match the recorded pre-excision mapping"
        );
    }
}

/// Two-plant fixture: `k_max = 4`, four study stages, seven post-study stages
/// (`n_delivery = 11`). Plant 0's decider carries a 3-wide excised fixed
/// post-horizon window (`decider[4..7) == None`, so `g == 3`); plant 1's
/// post-study slice is `Some` immediately (`g == 0`). `PointResolution`
/// literals are built directly rather than through `resolve_point`, whose
/// monotonic deciders cannot produce a `None` island bracketed by `Some` on
/// both sides — the same pattern `lead_time::tests`'s `ring_index`/
/// `physical_target` fixtures use.
fn two_plant_excised_window_fixture() -> StateSpace {
    let plant0 = PointResolution {
        decider: vec![
            None,
            None,
            None,
            None, // m = 0..3 (in-study)
            None,
            None,
            None, // m = 4..6 (excised window, g = 3)
            Some(0),
            Some(1),
            Some(2),
            Some(3), // m = 7..10
        ],
        decision_sets: vec![Vec::new(); 4],
        depth: vec![0; 4],
        occupancy: vec![0; 4],
    };
    let plant1 = PointResolution {
        decider: vec![
            None,
            None,
            None,
            None, // m = 0..3 (in-study)
            Some(0),
            Some(1),
            Some(2),
            Some(3),
            Some(4),
            Some(5),
            Some(6), // m = 4..10, g = 0
        ],
        decision_sets: vec![Vec::new(); 4],
        depth: vec![0; 4],
        occupancy: vec![0; 4],
    };
    let resolution = AnticipatedResolution {
        per_plant: vec![plant0, plant1],
    };
    state_with_attached_resolution(4, resolution)
}

/// At `stage_idx = 0`, ring-axis depths `0..k_max` give `r = 1, 2, 3, 4`. For
/// `r < n_decision(4)` (`r = 1, 2, 3`) `physical_target` is the identity for
/// both plants — both fixture deciders mark those `None` (interior carries).
/// At `r = 4` the two plants diverge: plant 0's excised window shifts it to
/// `m = 7`, plant 1's `g = 0` leaves it at `m = 4`. Both fixture deciders mark
/// their own `r = 4` physical target a deposit at stage 0 (`Some(0)`).
/// Reading the WRONG, raw ring-axis `m = 4` for plant 0 instead would hit its
/// excised window's `None` entry (always "ready, not a deposit" — `None`
/// never equals a deposit's `Some(stage_idx)`), producing `Some` where the
/// correct sweep produces `None`.
#[test]
fn anticipated_slot_row_pos_walks_physical_targets_across_the_excised_window() {
    let state = two_plant_excised_window_fixture();

    let (row_pos, n_reachable) = build_anticipated_slot_row_pos(&state, 0);

    assert_eq!(
        row_pos,
        vec![
            None,
            None,
            Some(0),
            Some(1),
            Some(2),
            Some(3),
            Some(4),
            Some(5),
        ],
        "plant 0 must be classified at its ring-shifted physical target (7), \
         not the raw ring index (4)"
    );
    assert_eq!(n_reachable, 6);
}

/// The same fixture as
/// `anticipated_slot_row_pos_walks_physical_targets_across_the_excised_window`,
/// swept across every `stage_idx` from which ring-axis index `r = 4` is
/// reachable (`0..=3`, `depth = 3 - stage_idx`). Plant 0's physical target at
/// `r = 4` is `m = 7` (`decider[7] == Some(0)`): a deposit — excluded — only
/// at `stage_idx = 0`, an interior carry at `stage_idx = 1, 2, 3`. Reading the
/// excised window's own `m = 4` instead (`None`, forced by the `g = 3`
/// contiguity invariant) would classify every `stage_idx` identically as an
/// interior carry, since `None` never equals a deposit's `Some(stage_idx)` —
/// the stage-dependent split below is possible only because the sweep reads
/// `m = 7`, never a delivery inside the excised window `[4, 7)`.
#[test]
fn anticipated_slot_row_pos_gives_no_row_to_an_excised_delivery() {
    let state = two_plant_excised_window_fixture();
    let slot = 4 % state.k_max;

    for stage_idx in 0..=3 {
        let (row_pos, _n_reachable) = build_anticipated_slot_row_pos(&state, stage_idx);
        let plant0_at_r4 = row_pos[slot * state.n_anticipated];
        if stage_idx == 0 {
            assert!(
                plant0_at_r4.is_none(),
                "stage_idx=0: physical target 7 is a deposit, so r=4's slot must \
                 carry no row"
            );
        } else {
            assert!(
                plant0_at_r4.is_some(),
                "stage_idx={stage_idx}: physical target 7 is ready and not a \
                 deposit, so r=4's slot must carry an interior-carry row"
            );
        }
    }
}

/// A two-plant fixture where plant A's fixed window pushes its physical
/// target past `n_delivery` at `depth = 1` (`k_max = 3`, two study stages,
/// two post-study stages, `n_delivery = 4`; plant A's `g = 1` shifts
/// `r = 3` to `m = 4 >= n_delivery`), while zero-width sibling plant B's own
/// `r = 3` maps to `m = 3 < n_delivery` and is still classified. The mask
/// must be evaluated per plant, after resolving each plant's own physical
/// target — not once against the shared ring-axis `r`.
#[test]
fn anticipated_slot_row_pos_masks_per_plant_at_the_extended_axis_bound() {
    let plant_a = PointResolution {
        decider: vec![None, None, None, Some(0)],
        decision_sets: vec![Vec::new(); 2],
        depth: vec![0; 2],
        occupancy: vec![0; 2],
    };
    let plant_b = PointResolution {
        decider: vec![None, None, Some(0), Some(0)],
        decision_sets: vec![Vec::new(); 2],
        depth: vec![0; 2],
        occupancy: vec![0; 2],
    };
    let resolution = AnticipatedResolution {
        per_plant: vec![plant_a, plant_b],
    };
    let state = state_with_attached_resolution(3, resolution);

    let (row_pos, n_reachable) = build_anticipated_slot_row_pos(&state, 1);

    assert_eq!(
        row_pos,
        vec![None, Some(2), None, None, Some(0), Some(1)],
        "plant A's slot at depth=1 (physical target 4, at n_delivery=4) must be \
         masked while plant B's own physical target (3) is still classified"
    );
    assert_eq!(n_reachable, 3);
}

// ── Mixed-lead reachability: mask vs. the LP's own latch set ─────────────

/// Leads `(1, 3)`, `k_max = 3`, 4 study stages, no post-study calendar. At
/// every stage the mask's commitment-hold tail must equal the union, over
/// every stage, of the LP's own carry map ([`build_anticipated_slot_row_pos`])
/// plus its deposit map ([`build_anticipated_decision_row_pos`]) — never the
/// retired per-plant-lead-bounded rule, under which the lead-1 plant's slot 1
/// at stage 0 would be excluded even though the LP latches it (a fresh
/// deposit).
#[test]
fn mixed_lead_nonzero_mask_covers_every_slot_the_lp_latches() {
    let resolution = AnticipatedResolution::resolve(
        &[LeadTime::Stages(1), LeadTime::Stages(3)],
        DeliveryAxis {
            study_stage_hours: &[720.0; 4],
            post_study_stage_hours: &[],
        },
    );
    let state =
        state_layout_with_transit_buckets_and_resolution(1, 1, Vec::new(), vec![1, 3], resolution);

    let expected_by_stage: [&[usize]; 4] = [&[2, 3, 5, 1], &[4, 5, 1], &[0, 1], &[]];

    let mut latched: Vec<usize> = Vec::new();
    for (t, &expected) in expected_by_stage.iter().enumerate() {
        let mut this_stage: Vec<usize> = build_anticipated_slot_row_pos(&state, t)
            .0
            .iter()
            .enumerate()
            .filter_map(|(o, pos)| pos.is_some().then_some(o))
            .collect();

        let (decision_pos, _) =
            build_anticipated_decision_row_pos(&state, t, &[(None, None); 2], &[0, 1, 2, 3]);
        for (p, pos) in decision_pos.iter().enumerate() {
            if pos.is_some() {
                let m = anticipated_resolution_for(&state, AnticipatedLocal::new(p))
                    .genuine_decisions_at(t)
                    .next()
                    .expect("a decision-row position implies a genuine decision this stage");
                this_stage.push(state.commitment_hold_in_study_offset(p, m));
            }
        }

        this_stage.sort_unstable();
        let mut expected = expected.to_vec();
        expected.sort_unstable();
        assert_eq!(this_stage, expected, "stage {t}: latched offsets");

        latched.extend_from_slice(&this_stage);
    }
    latched.sort_unstable();
    latched.dedup();

    let start = state.state_dim_range(StateRegion::CommitmentHold).start;
    for &o in &latched {
        assert!(
            state
                .nonzero_state_indices
                .contains(&StateDim::new(start + o)),
            "offset {o} is latched by the LP but missing from the mask"
        );
    }

    let projection = CutStateProjection::new(
        &state,
        StageStateConfig {
            storage: true,
            inflow_lags: true,
        },
    );
    assert_eq!(
        projection.render_len(),
        state.nonzero_state_indices.len(),
        "an all-enabled projection must render exactly the global mask"
    );
}

/// Fishing-count invariance under the same extended axis as the carry test
/// above: `build_anticipated_fishing_row_pos` run at every in-study
/// `stage_idx` (`0..4`) must produce the SAME active count whether the
/// attached resolution's delivery axis is study-only (`n_delivery = 4`) or
/// extended (`n_delivery = 8`) — no post-study maturity ever produces a
/// fishing row, since a maturity is always checked against `stage_idx`
/// itself, which the caller (and the function's own explicit in-study guard)
/// keeps in `[0, n_stages)`.
#[test]
fn build_anticipated_fishing_row_pos_extended_axis_matches_study_only_count() {
    let k_max = 4;
    let lead = LeadTime::Stages(4);
    let n_stages = 4;

    let study_stage_hours = vec![720.0; n_stages];
    let study_only_resolution = AnticipatedResolution::resolve(
        &[lead],
        DeliveryAxis {
            study_stage_hours: &study_stage_hours,
            post_study_stage_hours: &[],
        },
    );
    let post_study_stage_hours = vec![720.0; n_stages];
    let extended_resolution = AnticipatedResolution::resolve(
        &[lead],
        DeliveryAxis {
            study_stage_hours: &study_stage_hours,
            post_study_stage_hours: &post_study_stage_hours,
        },
    );
    let study_only_state = state_with_attached_resolution(k_max, study_only_resolution);
    let extended_state = state_with_attached_resolution(k_max, extended_resolution);

    for stage_idx in 0..n_stages {
        let (_, study_only_count) =
            build_anticipated_fishing_row_pos(&study_only_state, n_stages, stage_idx);
        let (_, extended_count) =
            build_anticipated_fishing_row_pos(&extended_state, n_stages, stage_idx);
        assert_eq!(
            extended_count, study_only_count,
            "extended-axis fishing count at stage_idx={stage_idx} must match the \
             study-only count — no post-study maturity produces a fishing row"
        );
    }
}

// ── Anticipated-decision range tests ──────────────────────────────────────

/// Build a `ResolvedBounds` with zero entities but the given `n_stages`.
///
/// Used to exercise the `is_anticipated_decision_active` gate
/// in [`super::AnticipatedLayout::state_out_def_rows`] without needing real entity data.
fn bounds_with_n_stages(n_stages: usize) -> ResolvedBounds {
    bounds_with_pumping(0, n_stages)
}

/// Builds a fixture struct owning all data for a context with anticipated
/// thermals and a known `n_stages` for the `state_out_def` predicate.
struct AntFixturesWithNStages {
    base: CtxFixture,
}

impl AntFixturesWithNStages {
    fn new(n_stages: usize) -> Self {
        Self {
            base: CtxFixture {
                bounds: bounds_with_n_stages(n_stages),
                time_value: TimeValue::from_parts(
                    vec![],
                    vec![1.0; n_stages],
                    vec![744.0; n_stages],
                    (0..i32::try_from(n_stages).unwrap_or(0)).collect(),
                    PostStudyResolved::default(),
                ),
                ..CtxFixture::default()
            },
        }
    }

    /// `anticipated_positions` must be strictly ascending
    /// (`test_support::anticipated_plants_at`), and its length is the
    /// resulting `n_anticipated`.
    fn make_ctx(
        &mut self,
        anticipated_lead_stages: Vec<usize>,
        anticipated_positions: &[usize],
    ) -> TemplateBuildCtx<'_> {
        self.base.anticipated_plants = anticipated_plants_at(anticipated_positions);
        self.base.anticipated_lead_stages = anticipated_lead_stages;
        self.base.ctx()
    }
}

/// The ring's own out-block start (`StateSpace::commit_out.start`) is sourced
/// from the state-region position immediately after `transit_buckets_out`
/// (before `z_inflow`), `col_line_fwd_start` follows `anticipated_decision`
/// directly, and [`super::AnticipatedLayout::state_out_def_rows`] counts both active
/// plants at stage 0.
///
/// Fixture: `n_anticipated=2`, `K=[2,3]`, `k_max=3`, `n_stages=6`,
/// `stage_idx=0`, `N=0`, `L=0`, `B=0`. Both plants are active: `0+2=2 < 6` and
/// `0+3=3 < 6`. State-region offset = `N*(1+L) + B = 0`.
#[test]
fn test_layout_state_out_block_adjacent_to_decision() {
    let mut fixtures = AntFixturesWithNStages::new(6);
    let ctx = fixtures.make_ctx(
        vec![2, 3], // K_0=2, K_1=3
        &[0, 1],
    );
    let stage = minimal_stage();
    let layout = StageLayout::new(&ctx, &stage, 0);

    // The outgoing ring sits in the state region: N*(1+L) + B.
    assert_eq!(
        layout.state.commit_out.start, 0,
        "outgoing-ring columns must be sourced from the state-region offset \
             N*(1+L) + B"
    );
    assert_eq!(
        layout.geometry.line_fwd.start,
        layout.geometry.anticipated_decision.start + 2,
        "line_fwd must be immediately after the anticipated_decision block"
    );
    assert_eq!(layout.anticipated.state_out_def_rows.len(), 2);
    assert_eq!(
        layout.anticipated.state_out_def_rows.start,
        layout.anticipated.fishing_rows.end
    );
}

/// [`super::AnticipatedLayout::state_out_def_rows`] is empty when all plants are
/// inactive at the given stage, but the column block stays allocated.
///
/// Fixture: `n_anticipated=2`, `K=[2,3]`, `n_stages=6`, `stage_idx=5`.
/// Both inactive: `5+2=7 >= 6` and `5+3=8 >= 6`.
#[test]
fn test_layout_state_out_def_rows_zero_when_all_inactive() {
    let mut fixtures = AntFixturesWithNStages::new(6);
    let ctx = fixtures.make_ctx(
        vec![2, 3], // K_0=2, K_1=3
        &[0, 1],
    );
    let stage = minimal_stage();
    let layout = StageLayout::new(&ctx, &stage, 5);

    assert_eq!(layout.anticipated.state_out_def_rows.len(), 0);
    // Column block stays allocated at the state-region offset regardless of
    // activity: N*(1+L) + B = 0.
    assert_eq!(layout.state.commit_out.start, 0);
}

/// Zero-anticipated layouts emit no `anticipated_state_out_def` rows.
#[test]
fn test_layout_no_anticipated_unchanged_num_cols() {
    let mut fixtures = ZeroEntityFixtures::new();
    let ctx = fixtures.make_ctx(vec![], &[]);
    let stage = minimal_stage();
    let layout = StageLayout::new(&ctx, &stage, 0);

    assert_eq!(layout.anticipated.state_out_def_rows.len(), 0);
}

// ── Pumping-flow column region ─────────────────────────────────────────────

/// Build a `ResolvedBounds` with the given pumping-station count and stage
/// count (all other entity tables empty). `table.n_pumping()` recovers
/// `n_pumping` from the `pumping` Vec length divided by `n_stages`.
fn bounds_with_pumping(n_pumping: usize, n_stages: usize) -> ResolvedBounds {
    ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 0,
            n_thermals: 0,
            n_lines: 0,
            n_pumping,
            n_contracts: 0,
            n_stages,
            k_max: 0,
        },
        &BoundsDefaults {
            hydro: HydroStageBounds {
                min_storage_hm3: 0.0,
                max_storage_hm3: 0.0,
                filling_min_rate_m3s: 0.0,
                water_withdrawal_m3s: 0.0,
            },
            hydro_block: HydroBlockBounds::default(),
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
    )
}

/// Owns the data for a `TemplateBuildCtx` whose `bounds` report a non-zero
/// `n_pumping()`. Mirrors `ZeroEntityFixtures` but injects a pumping-aware
/// `ResolvedBounds` so `StageLayout::new` reserves the `pumping_flow` block.
struct PumpingFixtures {
    base: CtxFixture,
}

impl PumpingFixtures {
    fn new(n_pumping: usize, n_stages: usize) -> Self {
        let stations = (0..n_pumping)
            .map(|i| PumpingStation {
                id: EntityId(i as i32),
                name: format!("P{i}"),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                bus_id: EntityId(0),
                source_hydro_id: EntityId(0),
                destination_hydro_id: EntityId(1),
                entry_stage_id: None,
                exit_stage_id: None,
                consumption_mw_per_m3s: 0.5,
                min_flow_m3s: 0.0,
                max_flow_m3s: 10.0,
            })
            .collect();
        Self {
            base: CtxFixture {
                bounds: bounds_with_pumping(n_pumping, n_stages),
                pumping_stations: stations,
                time_value: TimeValue::from_parts(
                    vec![],
                    vec![1.0; n_stages],
                    vec![744.0; n_stages],
                    (0..i32::try_from(n_stages).unwrap_or(0)).collect(),
                    PostStudyResolved::default(),
                ),
                ..CtxFixture::default()
            },
        }
    }

    fn make_ctx(&mut self) -> TemplateBuildCtx<'_> {
        self.base.ctx()
    }

    /// Build a stage with `n_blks` equal-duration blocks.
    fn stage_with_blocks(n_blks: usize) -> Stage {
        let mut stage = minimal_stage();
        stage.blocks = (0..n_blks)
            .map(|b| Block {
                index: b,
                name: format!("B{b}"),
                duration_hours: 248.0,
            })
            .collect();
        stage
    }
}

/// Inert-layout invariant: with `n_pumping == 0` the `pumping_flow` block
/// collapses, so `col_pumping_start` sits exactly where the generic-slack
/// columns begin (`col_ncs_end`, which equals `col_ncs_start` when no NCS are
/// active) and `num_cols` is unshifted. For this zero-entity one-block system
/// the entire column count is the single theta column.
///
/// Pinning `n_pumping == 0`, `col_pumping_start == col_ncs_start`, and the
/// exact `num_cols`/equipment starts proves that reserving the pumping region
/// does not move any pre-existing column when there are no stations.
#[test]
fn pumping_layout_inert_when_no_stations() {
    let mut fixtures = ZeroEntityFixtures::new();
    let ctx = fixtures.make_ctx(vec![], &[]);
    let stage = minimal_stage();
    let layout = StageLayout::new(&ctx, &stage, 0);

    assert_eq!(
        ctx.resolved.bounds.n_pumping(),
        0,
        "fixture has no pumping stations"
    );
    assert_eq!(
        ctx.pumping_stations.len(),
        0,
        "ctx.pumping_stations must be empty"
    );

    // The empty pumping block does not advance the cursor: its start equals
    // the NCS-region end. With zero active NCS, col_ncs_end == col_ncs_start.
    assert_eq!(
        layout.geometry.pumping_flow.start, layout.geometry.ncs_generation.start,
        "col_pumping_start must equal col_ncs_start (col_ncs_end) when no stations"
    );

    // Pre-existing column starts for the zero-entity, single-block layout:
    // theta == 0, every equipment/slack/NCS region empty starting at theta+1.
    let idx = state_layout(ctx.hydros.len(), ctx.par_lp.max_order());
    let expected_start = idx.theta + 1;
    assert_eq!(layout.geometry.turbine.start, expected_start);
    assert_eq!(layout.geometry.thermal.start, expected_start);
    assert_eq!(layout.geometry.line_fwd.start, expected_start);
    assert_eq!(layout.geometry.deficit.start, expected_start);
    assert_eq!(layout.geometry.excess.start, expected_start);
    assert_eq!(layout.geometry.ncs_generation.start, expected_start);
    assert_eq!(layout.geometry.pumping_flow.start, expected_start);
    assert_eq!(
        layout.num_cols, expected_start,
        "num_cols must be unshifted"
    );
}

/// `n_pumping == 2`, `n_blks == 3` ⇒ a 6-column `pumping_flow` block at
/// `col_pumping_start`, block-major, and `num_cols` increased by exactly 6
/// relative to the otherwise-identical station-free layout.
#[test]
fn pumping_layout_reserves_block_major_columns() {
    let n_pumping = 2_usize;
    let n_blks = 3_usize;

    let mut baseline_fixtures = PumpingFixtures::new(0, 3);
    let baseline_ctx = baseline_fixtures.make_ctx();
    let stage = PumpingFixtures::stage_with_blocks(n_blks);
    let baseline = StageLayout::new(&baseline_ctx, &stage, 0);
    assert_eq!(baseline_ctx.pumping_stations.len(), 0);

    let mut fixtures = PumpingFixtures::new(n_pumping, 3);
    let ctx = fixtures.make_ctx();
    assert_eq!(
        ctx.resolved.bounds.n_pumping(),
        n_pumping,
        "fixture bounds must report n_pumping() == 2"
    );
    let layout = StageLayout::new(&ctx, &stage, 0);

    assert_eq!(
        ctx.pumping_stations.len(),
        n_pumping,
        "ctx.pumping_stations.len() == 2"
    );
    assert_eq!(
        layout.geometry.pumping_flow.start, layout.geometry.ncs_generation.start,
        "col_pumping_start must follow the NCS region"
    );
    // Block-major width: n_pumping * n_blks == 2 * 3 == 6.
    assert_eq!(
        layout.num_cols - baseline.num_cols,
        n_pumping * n_blks,
        "num_cols must grow by exactly n_pumping * n_blks == 6"
    );
    assert_eq!(
        layout.num_cols,
        layout.geometry.pumping_flow.start + n_pumping * n_blks,
        "the 6-column block ends at num_cols (no generic-slack columns here)"
    );
}

/// With both contract counts 0 the import/export blocks collapse onto
/// `col_pumping_end` and the generic-slack start (here surfaced as
/// `col_filling_target_start()`, since no generic-slack columns exist) is
/// unshifted — the contract-free parity guarantee.
#[test]
fn contract_columns_empty_keep_generic_slack_at_pumping_end() {
    let n_pumping = 2_usize;
    let n_blks = 3_usize;
    let mut fixtures = PumpingFixtures::new(n_pumping, 3);
    let ctx = fixtures.make_ctx();
    assert_eq!(ctx.contracts.len(), 0);

    let stage = PumpingFixtures::stage_with_blocks(n_blks);
    let layout = StageLayout::new(&ctx, &stage, 0);

    let col_pumping_end = layout.geometry.pumping_flow.end;
    assert_eq!(
        layout.geometry.contract_import.start, col_pumping_end,
        "empty import block starts at col_pumping_end"
    );
    assert_eq!(
        layout.geometry.contract_export.start, col_pumping_end,
        "empty export block collapses onto col_pumping_end"
    );
    assert_eq!(
        layout.geometry.filling_target_col.start, col_pumping_end,
        "generic-slack start is unshifted for a contract-free system"
    );
}

/// `n_contract_import == 2`, `n_contract_export == 1`, `n_blks == 3`: the import
/// block (6 columns) starts at `col_pumping_end`, the export block (3 columns)
/// follows it, and the generic-slack start (`col_filling_target_start()` with no
/// generic-slack columns) shifts by `(2 + 1) * 3 == 9`.
#[test]
fn contract_columns_reserve_import_then_export_blocks() {
    let n_pumping = 2_usize;
    let n_blks = 3_usize;
    let mut fixtures = PumpingFixtures::new(n_pumping, 3);
    fixtures.base.contracts = vec![
        dormant_contract(0, ContractType::Import),
        dormant_contract(1, ContractType::Import),
        dormant_contract(2, ContractType::Export),
    ];
    let ctx = fixtures.make_ctx();

    let stage = PumpingFixtures::stage_with_blocks(n_blks);
    let layout = StageLayout::new(&ctx, &stage, 0);

    let col_pumping_end = layout.geometry.pumping_flow.end;
    assert_eq!(
        layout.geometry.contract_import.start, col_pumping_end,
        "import block starts at col_pumping_end"
    );
    assert_eq!(
        layout.geometry.contract_export.start,
        col_pumping_end + 6,
        "export block follows the 6-column import block"
    );
    assert_eq!(
        layout.geometry.filling_target_col.start,
        col_pumping_end + 9,
        "generic-slack start shifts by (2 + 1) * 3 == 9"
    );
}

/// Every contract column, import and export, is addressed exactly once by the
/// block-major oracle `range.start + slot * n_blks + blk`.
#[test]
fn contract_col_covers_each_contract_column_once() {
    let n_pumping = 2_usize;
    let n_blks = 3_usize;
    let mut fixtures = PumpingFixtures::new(n_pumping, 3);
    fixtures.base.contracts = vec![
        dormant_contract(0, ContractType::Import),
        dormant_contract(1, ContractType::Import),
        dormant_contract(2, ContractType::Export),
    ];
    let ctx = fixtures.make_ctx();

    let stage = PumpingFixtures::stage_with_blocks(n_blks);
    let layout = StageLayout::new(&ctx, &stage, 0);

    let mut hits = vec![0_usize; layout.num_cols];
    let mut compared = 0_usize;
    for (contract_type, range, n) in [
        (
            ContractType::Import,
            layout.geometry.contract_import.clone(),
            2,
        ),
        (
            ContractType::Export,
            layout.geometry.contract_export.clone(),
            1,
        ),
    ] {
        for slot in 0..n {
            for blk in 0..n_blks {
                let col = layout
                    .geometry
                    .contract_col(contract_type, slot, BlockIdx::new(blk));
                let oracle = range.start + slot * n_blks + blk;
                assert_eq!(
                    col, oracle,
                    "{contract_type:?} slot {slot} blk {blk}: builder address must match the oracle"
                );
                assert!(
                    range.contains(&col),
                    "{contract_type:?} slot {slot} blk {blk}: address must lie in its own range"
                );
                hits[col] += 1;
                compared += 1;
            }
        }
    }

    for (c, &hit) in hits.iter().enumerate() {
        let expected = usize::from(
            layout.geometry.contract_import.contains(&c)
                || layout.geometry.contract_export.contains(&c),
        );
        assert_eq!(hit, expected, "column {c}");
    }
    assert_eq!(
        compared,
        layout.geometry.contract_import.len() + layout.geometry.contract_export.len()
    );
    assert_eq!(compared, 9);
}

/// The shared `commissioning_active` predicate gates on
/// `entry_stage_id <= stage_id < exit_stage_id` and is the single owner of
/// commissioning activity for every equipment family (NCS, pumping, and the
/// later thermal/line/hydro). Covers the five window shapes — no window
/// (always active), entry-only, exit-only, both, and a stage outside the
/// window. The forbidden alternative — a non-strict upper bound
/// (`stage_id <= exit`) — would keep a decommissioned entity active at its
/// exit stage.
#[test]
fn commissioning_active_gates_on_stage_id_with_half_open_window() {
    use cobre_core::commissioning::commissioning_active;
    // p0 no window: active at every stage.
    for id in [0, 1, 2, 3, 4, 100] {
        assert!(
            commissioning_active(None, None, id),
            "no window active at {id}"
        );
    }
    // entry=2: active iff id >= 2.
    assert!(!commissioning_active(Some(2), None, 1));
    assert!(commissioning_active(Some(2), None, 2));
    // exit=3: active iff id < 3 (strict upper).
    assert!(commissioning_active(None, Some(3), 2));
    assert!(!commissioning_active(None, Some(3), 3));
    // window [1, 4): active iff 1 <= id < 4.
    assert!(!commissioning_active(Some(1), Some(4), 0));
    assert!(commissioning_active(Some(1), Some(4), 1));
    assert!(commissioning_active(Some(1), Some(4), 3));
    assert!(!commissioning_active(Some(1), Some(4), 4));
}

// ── post-equipment column cursor (no-hydro fork fallback) ───────────────────

/// With `n_hydros == 0` every withdrawal / operational-violation / NCS column
/// region is empty, so `RangeCursor::alloc(0)` leaves all their starts at the
/// single post-equipment column cursor `col_evap_start`. A multi-block stage
/// keeps that cursor non-trivial (not the degenerate one-column case), so a
/// hand-computed offset that reintroduces a `0..0` empty-range fallback would
/// shift these starts off `col_evap_start` and fail here.
#[test]
fn withdrawal_and_operational_columns_collapse_onto_evap_col_start_when_no_hydros() {
    let mut fixtures = ZeroEntityFixtures::new();
    let ctx = fixtures.make_ctx(vec![], &[]);
    let stage = PumpingFixtures::stage_with_blocks(4);
    let layout = StageLayout::new(&ctx, &stage, 0);

    assert_eq!(ctx.hydros.len(), 0, "fixture must have zero hydros");
    assert_eq!(
        layout.clock.n_blks(),
        4,
        "fixture must build a 4-block layout"
    );

    let post_equipment = layout.equipment.evap_col_start;
    assert_eq!(
        layout.geometry.ncs_generation.start, post_equipment,
        "col_ncs_start must collapse onto col_evap_start when n_hydros == 0"
    );
    assert_eq!(
        layout.geometry.withdrawal_slack_neg.start, post_equipment,
        "col_withdrawal_neg_start"
    );
    assert_eq!(
        layout.geometry.withdrawal_slack_pos.start, post_equipment,
        "col_withdrawal_pos_start"
    );
    assert_eq!(
        layout.geometry.outflow_below_slack.start, post_equipment,
        "col_outflow_below_start"
    );
    assert_eq!(
        layout.geometry.outflow_above_slack.start, post_equipment,
        "col_outflow_above_start"
    );
    assert_eq!(
        layout.geometry.turbine_below_slack.start, post_equipment,
        "col_turbine_below_start"
    );
    assert_eq!(
        layout.geometry.generation_below_slack.start, post_equipment,
        "col_generation_below_start"
    );
}

// ── post-equipment row cursor (no-hydro fork fallback) ──────────────────────

/// With `n_hydros == 0` every operational-violation row block is empty, so
/// `RangeCursor::alloc(0)` leaves all four row starts at the shared
/// post-equipment row cursor, which with zero evap hydros equals
/// `row_evap_start()`. A multi-block stage keeps that cursor non-trivial (not
/// the degenerate one-row case), so a hand-computed offset that reintroduces a
/// `0..0` empty-range fallback would shift these starts off it and fail here.
#[test]
fn operational_violation_rows_collapse_onto_row_evap_start_when_no_hydros() {
    let mut fixtures = ZeroEntityFixtures::new();
    let ctx = fixtures.make_ctx(vec![], &[]);
    let stage = PumpingFixtures::stage_with_blocks(4);
    let layout = StageLayout::new(&ctx, &stage, 0);

    assert_eq!(ctx.hydros.len(), 0, "fixture must have zero hydros");
    assert_eq!(
        layout.clock.n_blks(),
        4,
        "fixture must build a 4-block layout"
    );

    let post_equipment = layout.row_evap_start();
    assert_eq!(
        layout.oper_violation.min_outflow.start, post_equipment,
        "row_min_outflow_start must collapse onto the post-equipment row cursor when n_hydros == 0"
    );
    assert_eq!(
        layout.oper_violation.max_outflow.start, post_equipment,
        "row_max_outflow_start"
    );
    assert_eq!(
        layout.oper_violation.min_turbine.start, post_equipment,
        "row_min_turbine_start"
    );
    assert_eq!(
        layout.oper_violation.min_generation.start, post_equipment,
        "row_min_generation_start"
    );
}

// ── Group-2 accessors: hydro-free divergence guard ──────────────────────────

/// With `n_hydros == 0`, every Group-2 accessor must return the shared
/// post-equipment cursor — `evap_col_start` for the eight column accessors,
/// `row_evap_start()` for the five row accessors. Each accessor is a
/// bare `self.<range>.start`/`.end`, correct only because `StageLayout::new`
/// allocates every one of these families through `RangeCursor::alloc`:
/// `alloc(0)` returns `pos..pos`, so an empty family's `.start` already equals
/// the post-equipment cursor. A hand-computed offset that reintroduces a
/// `0..0` fallback (losing the cursor position) would fail these equality
/// assertions.
///
/// The column cursor is additionally asserted `!= 0`: the theta and state columns
/// always precede the equipment/slack region, so `evap_col_start` is
/// provably positive and a spurious `0` is directly detectable. The row cursor is
/// NOT asserted `!= 0`: with zero hydros AND zero buses no rows precede the
/// operational-violation block, so `row_evap_start()` is legitimately `0`
/// here (asserting `!= 0` would test a false invariant). The non-zero-row
/// divergence is covered end-to-end by the D01 hydro-free parity case, whose
/// load-balance rows make the row cursor positive.
#[test]
fn group2_accessors_return_post_equipment_cursor_when_no_hydros() {
    let mut fixtures = ZeroEntityFixtures::new();
    let ctx = fixtures.make_ctx(vec![], &[]);
    let stage = PumpingFixtures::stage_with_blocks(4);
    let layout = StageLayout::new(&ctx, &stage, 0);

    assert_eq!(ctx.hydros.len(), 0, "fixture must have zero hydros");
    assert_eq!(
        layout.clock.n_blks(),
        4,
        "fixture must build a 4-block layout"
    );

    // Column cursor: the eight column accessors collapse onto `evap_col_start`
    // with no hydros, and that cursor is provably positive (theta + state
    // columns precede it).
    let post_col = layout.equipment.evap_col_start;
    assert_ne!(post_col, 0, "post-equipment column cursor must not be 0");
    for (value, name) in [
        (layout.geometry.generation.start, "col_generation_start"),
        (layout.equipment.evap_col_start, "col_evap_start"),
        (
            layout.geometry.withdrawal_slack_neg.start,
            "col_withdrawal_neg_start",
        ),
        (
            layout.geometry.withdrawal_slack_pos.start,
            "col_withdrawal_pos_start",
        ),
        (
            layout.geometry.outflow_below_slack.start,
            "col_outflow_below_start",
        ),
        (
            layout.geometry.outflow_above_slack.start,
            "col_outflow_above_start",
        ),
        (
            layout.geometry.turbine_below_slack.start,
            "col_turbine_below_start",
        ),
        (
            layout.geometry.generation_below_slack.start,
            "col_generation_below_start",
        ),
    ] {
        assert_eq!(
            value, post_col,
            "{name} must equal evap_col_start (not 0) when n_hydros == 0"
        );
    }

    // Row cursor: `row_evap_start()` is `fpha_rows_end`. The four
    // operational-violation row accessors collapse onto that same cursor when
    // n_evap_hydros == 0. Each must equal it, never a bare `.start`.
    let post_row = layout.row_evap_start();
    for (value, name) in [
        (layout.row_evap_start(), "row_evap_start"),
        (
            layout.oper_violation.min_outflow.start,
            "row_min_outflow_start",
        ),
        (
            layout.oper_violation.max_outflow.start,
            "row_max_outflow_start",
        ),
        (
            layout.oper_violation.min_turbine.start,
            "row_min_turbine_start",
        ),
        (
            layout.oper_violation.min_generation.start,
            "row_min_generation_start",
        ),
    ] {
        assert_eq!(
            value, post_row,
            "{name} must equal row_evap_start() when n_hydros == 0"
        );
    }
}

/// Consumption side of `crate::bucket_topology`'s
/// `test_horizon_cap_drops_lag_targeting_past_last_stage` (that test pins the
/// MASK — `per_stage_mask == [2, 1, 0]` for a depth-3 plant over 3
/// stages; this pins that `build_transit_bucket_row_pos` actually gates row emission
/// from it): stage 0 already drops the deepest lag (cap = 2), stage 1 drops
/// the two deepest (cap = 1), and stage 2 (the terminal stage, cap = 0) drops
/// all three.
#[test]
fn build_bucket_row_pos_gates_fewer_rows_as_horizon_cap_shrinks() {
    let column_order = vec![
        (HydroSys::new(0), 1_usize),
        (HydroSys::new(0), 2),
        (HydroSys::new(0), 3),
    ];
    let state = state_layout_with_transit_buckets(1, 0, column_order, vec![]);
    let per_stage_mask = vec![vec![2], vec![1], vec![0]];

    let (pos_stage0, n_stage0) = build_transit_bucket_row_pos(&state, &per_stage_mask, 0);
    assert_eq!(
        pos_stage0,
        vec![Some(0), Some(1), None],
        "stage 0: cap = 2, lags 1 and 2 keep a row, lag 3 does not"
    );
    assert_eq!(n_stage0, 2);

    let (pos_stage1, n_stage1) = build_transit_bucket_row_pos(&state, &per_stage_mask, 1);
    assert_eq!(
        pos_stage1,
        vec![Some(0), None, None],
        "stage 1: cap = 1, only lag 1 keeps a row"
    );
    assert_eq!(n_stage1, 1);

    let (pos_stage2, n_stage2) = build_transit_bucket_row_pos(&state, &per_stage_mask, 2);
    assert_eq!(
        pos_stage2,
        vec![None, None, None],
        "stage 2 (terminal): cap = 0, no lag keeps a row"
    );
    assert_eq!(
        n_stage2, 0,
        "the terminal stage emits zero bucket-definition rows"
    );
}

/// `column_order.is_empty()` (B==0) short-circuits before indexing
/// `per_stage_mask`, so an empty mask vec (a fixture that never builds one) is
/// safe — the B==0 byte-identity anchor at the `build_transit_bucket_row_pos` level.
#[test]
fn build_bucket_row_pos_b_zero_short_circuits_without_indexing_mask() {
    let state = state_layout_with_transit_buckets(0, 0, vec![], vec![]);
    let (pos, n) = build_transit_bucket_row_pos(&state, &[], 0);
    assert!(pos.is_empty());
    assert_eq!(n, 0);
}

// ── turbine/generation families sized and addressed by cell ────────────────

/// Owns a two-hydro `TemplateBuildCtx` where plant 1 (system index 1) may
/// split into two cells. `split == true` declares its two unit groups on
/// distinct buses (`n_cells == 3`); `split == false` declares them on the
/// SAME bus (the identity, `n_cells == 2`). Every other fixture in this file
/// is single-bus and therefore blind to the distinction this one exists to
/// exercise.
struct TwoHydroMultiBusFixtures {
    base: CtxFixture,
}

impl TwoHydroMultiBusFixtures {
    fn new(split: bool) -> Self {
        use crate::hydro_models::{EvaporationModel, ResolvedProductionModel};

        let (bus_a, bus_b) = if split {
            (EntityId(50), EntityId(51))
        } else {
            (EntityId(50), EntityId(50))
        };
        let plant0 = membership_hydro(1, false, None, None);
        let mut plant1 = membership_hydro(2, false, None, None);
        plant1.unit_groups = vec![
            make_unit_group(EntityId(10), bus_a, 0.0, 10.0, 0.0, 10.0),
            make_unit_group(EntityId(11), bus_b, 0.0, 10.0, 0.0, 10.0),
        ];
        let hydros = vec![plant0, plant1];
        let cascade = CascadeTopology::build(&hydros);
        let hydro_cell_index = HydroCellIndex::build(&hydros);

        let constant = ResolvedProductionModel::ConstantProductivity { productivity: 0.0 };
        let models = vec![vec![constant.clone()], vec![constant]];
        let production_models = ProductionModelSet::new(models, &hydros, 1);
        Self {
            base: CtxFixture {
                hydros,
                hydro_cell_index,
                cascade,
                production_models,
                evaporation_models: EvaporationModelSet::new(vec![
                    EvaporationModel::None,
                    EvaporationModel::None,
                ]),
                ..CtxFixture::default()
            },
        }
    }

    fn make_ctx(&mut self) -> TemplateBuildCtx<'_> {
        self.base.ctx()
    }
}

/// `equipment.turbine` is sized `n_cells * n_blks`, never `n_hydros *
/// n_blks`. A two-bus plant 1 yields `n_cells == 3` (`turbine.len() == 9`); the
/// same fixture with plant 1's groups collapsed onto one bus yields the
/// identity (`n_cells == 2`, `turbine.len() == 6`), with every later family
/// shifted down by exactly `n_blks`.
#[test]
fn test_turbine_family_is_sized_by_cell_not_by_plant() {
    let n_blks = 3;
    let stage = stage_with_blocks(BlockMode::Parallel, n_blks);

    let mut split_fixtures = TwoHydroMultiBusFixtures::new(true);
    let split_ctx = split_fixtures.make_ctx();
    assert_eq!(split_ctx.hydro_cell_index.n_cells(), 3);
    let split_layout = StageLayout::new(&split_ctx, &stage, 0);
    assert_eq!(split_layout.geometry.turbine.len(), 9, "3 cells * 3 blocks");
    assert_eq!(
        split_layout.geometry.spillage.start, split_layout.geometry.turbine.end,
        "spillage follows turbine directly, with no gap"
    );

    let mut same_bus_fixtures = TwoHydroMultiBusFixtures::new(false);
    let same_bus_ctx = same_bus_fixtures.make_ctx();
    assert_eq!(same_bus_ctx.hydro_cell_index.n_cells(), 2);
    let same_bus_layout = StageLayout::new(&same_bus_ctx, &stage, 0);
    assert_eq!(
        same_bus_layout.geometry.turbine.len(),
        6,
        "2 cells * 3 blocks under the identity"
    );
    assert_eq!(
        same_bus_layout.geometry.spillage.start,
        split_layout.geometry.spillage.start - n_blks,
        "one fewer cell shifts every later family down by exactly n_blks"
    );
}

/// `turbine_col` addresses each of a split plant's cells at a distinct
/// column, exactly `n_blks` apart, and every returned column lies inside
/// `equipment.turbine`. The final assertion's triple — `hydro_idx = 1`
/// (plant 1, the split plant), `cell_idx = 2` (its SECOND cell), `block_idx =
/// 0` — is mutually distinct on every axis, so confusing any two of them
/// changes the addressed column.
#[test]
fn test_turbine_col_addresses_each_cell_of_a_split_plant() {
    let n_blks = 3;
    let mut fixtures = TwoHydroMultiBusFixtures::new(true);
    let ctx = fixtures.make_ctx();
    assert_eq!(
        ctx.hydro_cell_index.cells_of(HydroSys::new(1)),
        1..3,
        "plant 1 owns cells 1 and 2"
    );

    let stage = stage_with_blocks(BlockMode::Parallel, n_blks);
    let layout = StageLayout::new(&ctx, &stage, 0);

    let mut columns = Vec::with_capacity(9);
    for cell in 0..3 {
        for blk in 0..n_blks {
            let col = layout
                .geometry
                .turbine_col(HydroCell::new(cell), BlockIdx::new(blk));
            assert!(
                layout.geometry.turbine.contains(&col),
                "cell {cell} block {blk}: column {col} must lie inside equipment.turbine"
            );
            columns.push(col);
        }
    }
    columns.sort_unstable();
    columns.dedup();
    assert_eq!(
        columns.len(),
        9,
        "every (cell, block) pair must resolve to a distinct column"
    );

    for blk in 0..n_blks {
        let cell1 = layout
            .geometry
            .turbine_col(HydroCell::new(1), BlockIdx::new(blk));
        let cell2 = layout
            .geometry
            .turbine_col(HydroCell::new(2), BlockIdx::new(blk));
        assert_eq!(
            cell2 - cell1,
            n_blks,
            "block {blk}: plant 1's two cells (1 and 2) must differ by exactly n_blks"
        );
    }

    let hydro_idx = 1_usize;
    let cell_idx = 2_usize;
    let block_idx = 0_usize;
    assert_ne!(
        cell_idx, block_idx,
        "cell and block indices must differ or the next assertion goes blind: \
         `cell * n_blks + blk` equals its own transposition whenever they are equal"
    );
    let asserted_col = layout
        .geometry
        .turbine_col(HydroCell::new(cell_idx), BlockIdx::new(block_idx));
    assert_eq!(
        asserted_col,
        layout.geometry.turbine.start + cell_idx * n_blks + block_idx
    );
    assert_ne!(
        asserted_col,
        layout
            .geometry
            .turbine_col(HydroCell::new(hydro_idx), BlockIdx::new(block_idx)),
        "cell {cell_idx}'s column must differ from the column at raw hydro index {hydro_idx}"
    );
}

/// Owns a three-hydro `TemplateBuildCtx` where plant 0 and plant 2 are FPHA
/// and non-FPHA plant 1 sits between them; plant 2 additionally splits into
/// two cells (two buses). Plant 2's three index families — `HydroSys` (2),
/// `FphaLocal` (1, its position among FPHA plants 0 and 2), and its two
/// `FphaCellLocal` values (1 and 2, its position among FPHA CELLS) — take
/// values that let a mixup between any two families surface as a wrong
/// column.
struct FphaMultiBusFixtures {
    base: CtxFixture,
}

impl FphaMultiBusFixtures {
    fn new() -> Self {
        use crate::hydro_models::{EvaporationModel, FphaPlane, ResolvedProductionModel};

        let fpha = ResolvedProductionModel::Fpha {
            planes: vec![FphaPlane {
                intercept: 0.0,
                gamma_v: 0.0,
                gamma_q: 0.0,
                gamma_s: 0.0,
            }],
        };
        let constant = ResolvedProductionModel::ConstantProductivity { productivity: 0.0 };
        // models[hydro][stage]: hydros 0 and 2 are FPHA, hydro 1 is constant.
        let models = vec![vec![fpha.clone()], vec![constant], vec![fpha]];

        let plant0 = membership_hydro(1, true, None, None);
        let plant1 = membership_hydro(2, false, None, None);
        let mut plant2 = membership_hydro(3, true, None, None);
        plant2.unit_groups = vec![
            make_unit_group(EntityId(20), EntityId(60), 0.0, 10.0, 0.0, 10.0),
            make_unit_group(EntityId(21), EntityId(61), 0.0, 10.0, 0.0, 10.0),
        ];

        let hydros = vec![plant0, plant1, plant2];
        let cascade = CascadeTopology::build(&hydros);
        let hydro_cell_index = HydroCellIndex::build(&hydros);
        let production_models = ProductionModelSet::new(models, &hydros, 1);

        Self {
            base: CtxFixture {
                hydros,
                hydro_cell_index,
                cascade,
                production_models,
                evaporation_models: EvaporationModelSet::new(vec![
                    EvaporationModel::None,
                    EvaporationModel::None,
                    EvaporationModel::None,
                ]),
                ..CtxFixture::default()
            },
        }
    }

    fn make_ctx(&mut self) -> TemplateBuildCtx<'_> {
        self.base.ctx()
    }
}

/// The FPHA-generation family is sized by FPHA CELL, never FPHA plant.
/// `n_fpha_cells == 3` (plant 0's one cell + plant 2's two cells; non-FPHA
/// plant 1 contributes none, even though it owns a hydro-cell of its own).
#[test]
fn test_generation_family_is_sized_by_fpha_cell() {
    let n_blks = 2;
    let mut fixtures = FphaMultiBusFixtures::new();
    let ctx = fixtures.make_ctx();

    // The fixture's three index families genuinely diverge for plant 2:
    // HydroSys = 2, FphaLocal = 1 (below), and its two FphaCellLocal values
    // (1 and 2, asserted below) are neither uniformly equal to 2 nor to 1.
    assert_eq!(ctx.hydro_cell_index.cells_of(HydroSys::new(2)), 2..4);

    let stage = stage_with_blocks(BlockMode::Parallel, n_blks);
    let layout = StageLayout::new(&ctx, &stage, 0);

    assert_eq!(
        layout.geometry.fpha_hydro_indices,
        vec![HydroSys::new(0), HydroSys::new(2)],
        "plant 2 is FPHA-local index 1"
    );
    assert_eq!(
        layout.geometry.generation.len(),
        3 * n_blks,
        "n_fpha_cells == 3: plant 0's one cell + plant 2's two cells"
    );

    for blk in 0..n_blks {
        let col1 = layout
            .geometry
            .generation_col(FphaCellLocal::new(1), BlockIdx::new(blk));
        let col2 = layout
            .geometry
            .generation_col(FphaCellLocal::new(2), BlockIdx::new(blk));
        assert!(
            layout.geometry.generation.contains(&col2),
            "block {blk}: column {col2} must lie inside equipment.generation"
        );
        assert_ne!(
            col1, col2,
            "block {blk}: plant 2's two FPHA cells must be distinct columns"
        );
    }
}

// ── Column address pins (builder raw arithmetic vs. layout accessors) ────

/// Minimal non-controllable source for the equipment column pins.
fn make_ncs(id: i32) -> NonControllableSource {
    NonControllableSource {
        id: EntityId(id),
        name: format!("W{id}"),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        entry_stage_id: None,
        exit_stage_id: None,
        max_generation_mw: 100.0,
        allow_curtailment: true,
        curtailment_cost: 0.0,
    }
}

/// Minimal pumping station for the equipment column pins.
fn make_pumping_station(id: i32) -> PumpingStation {
    PumpingStation {
        id: EntityId(id),
        name: format!("P{id}"),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(1),
        source_hydro_id: EntityId(1),
        destination_hydro_id: EntityId(2),
        entry_stage_id: None,
        exit_stage_id: None,
        consumption_mw_per_m3s: 0.0,
        min_flow_m3s: 0.0,
        max_flow_m3s: 0.0,
    }
}

/// Per-family count of addresses [`compare_column_addresses`] compared, so a
/// caller combining several fixtures can confirm every family was exercised.
#[derive(Default)]
struct ColumnAddressCounts {
    inflow_slack: usize,
    withdrawal_slack_neg: usize,
    withdrawal_slack_pos: usize,
    anticipated_decision: usize,
    filling_target_slack: usize,
    filled_min_storage_floor_slack: usize,
    ncs_generation: usize,
    pumping_flow: usize,
}

/// Compares every index of the builder's eight one-per-entity/NCS/pumping
/// column families on `layout` against `layout.geometry`'s own accessor
/// (job 3, `docs/design/lp-builder-contract.md`: one address formula per
/// family). Returns how many addresses each family compared; a family empty
/// on this particular `layout` compares zero.
fn compare_column_addresses(
    ctx: &TemplateBuildCtx<'_>,
    layout: &StageLayout,
) -> ColumnAddressCounts {
    let geom = layout.geometry.clone();
    let mut counts = ColumnAddressCounts::default();

    for h in 0..geom.inflow_slack.len() {
        assert_eq!(
            geom.inflow_slack_col(HydroSys::new(h)),
            layout.geometry.inflow_slack_col(HydroSys::new(h)),
            "inflow_slack_col mismatch at h={h}"
        );
        counts.inflow_slack += 1;
    }
    for h in 0..geom.withdrawal_slack_neg.len() {
        assert_eq!(
            geom.withdrawal_slack_neg_col(HydroSys::new(h)),
            layout.geometry.withdrawal_slack_neg_col(HydroSys::new(h)),
            "withdrawal_slack_neg_col mismatch at h={h}"
        );
        counts.withdrawal_slack_neg += 1;
    }
    for h in 0..geom.withdrawal_slack_pos.len() {
        assert_eq!(
            geom.withdrawal_slack_pos_col(HydroSys::new(h)),
            layout.geometry.withdrawal_slack_pos_col(HydroSys::new(h)),
            "withdrawal_slack_pos_col mismatch at h={h}"
        );
        counts.withdrawal_slack_pos += 1;
    }
    for i in 0..geom.anticipated_decision.len() {
        assert_eq!(
            geom.anticipated_decision_col(AnticipatedLocal::new(i)),
            layout
                .geometry
                .anticipated_decision_col(AnticipatedLocal::new(i)),
            "anticipated_decision_col mismatch at i={i}"
        );
        counts.anticipated_decision += 1;
    }
    for i in 0..geom.filling_target_col.len() {
        assert_eq!(
            geom.filling_target_slack_col(FillingTargetLocal::new(i)),
            layout
                .geometry
                .filling_target_slack_col(FillingTargetLocal::new(i)),
            "filling_target_slack_col mismatch at i={i}"
        );
        counts.filling_target_slack += 1;
    }
    for i in 0..geom.filled_min_storage_floor_col.len() {
        assert_eq!(
            geom.filled_min_storage_floor_slack_col(FloorLocal::new(i)),
            layout
                .geometry
                .filled_min_storage_floor_slack_col(FloorLocal::new(i)),
            "filled_min_storage_floor_slack_col mismatch at i={i}"
        );
        counts.filled_min_storage_floor_slack += 1;
    }
    for i in 0..ctx.non_controllable_sources.len() {
        for blk in 0..layout.clock.n_blks() {
            let blk = BlockIdx::new(blk);
            assert_eq!(
                geom.ncs_generation_col(NcsSys::new(i), blk),
                layout.geometry.ncs_generation_col(NcsSys::new(i), blk),
                "ncs_generation_col mismatch at i={i}"
            );
            counts.ncs_generation += 1;
        }
    }
    for i in 0..ctx.pumping_stations.len() {
        for blk in 0..layout.clock.n_blks() {
            let blk = BlockIdx::new(blk);
            assert_eq!(
                geom.pumping_flow_col(PumpingSys::new(i), blk),
                layout.geometry.pumping_flow_col(PumpingSys::new(i), blk),
                "pumping_flow_col mismatch at i={i}"
            );
            counts.pumping_flow += 1;
        }
    }

    counts
}

/// Pins every one-per-entity/NCS/pumping column family's raw builder address
/// against `StageGeometry`'s own accessor. No single fixture in this file
/// populates every family at once, so this runs [`compare_column_addresses`]
/// over several and sums the per-family counts.
#[test]
fn column_address_pins_cover_every_family() {
    let mut totals = ColumnAddressCounts::default();

    let mut filling_fixtures = FillingMembershipFixtures::new();
    filling_fixtures.base.has_penalty = true;
    let ctx = filling_fixtures.make_ctx();

    let filling_stage = stage_with_id(1);
    let filling_layout = StageLayout::new(&ctx, &filling_stage, 0);
    let filling_counts = compare_column_addresses(&ctx, &filling_layout);
    totals.inflow_slack += filling_counts.inflow_slack;
    totals.withdrawal_slack_neg += filling_counts.withdrawal_slack_neg;
    totals.withdrawal_slack_pos += filling_counts.withdrawal_slack_pos;
    totals.filling_target_slack += filling_counts.filling_target_slack;

    let operating_stage = stage_with_id(3);
    let operating_layout = StageLayout::new(&ctx, &operating_stage, 0);
    let operating_counts = compare_column_addresses(&ctx, &operating_layout);
    totals.filled_min_storage_floor_slack += operating_counts.filled_min_storage_floor_slack;

    let mut anticipated_fixtures = ZeroEntityFixtures::new();
    let anticipated_ctx = anticipated_fixtures.make_ctx(vec![1, 1], &[0, 1]);
    let anticipated_stage = minimal_stage();
    let anticipated_layout = StageLayout::new(&anticipated_ctx, &anticipated_stage, 0);
    let anticipated_counts = compare_column_addresses(&anticipated_ctx, &anticipated_layout);
    totals.anticipated_decision += anticipated_counts.anticipated_decision;

    let mut equipment_fixtures = ZeroEntityFixtures::new();
    let ncs = vec![make_ncs(1), make_ncs(2)];
    let pumping = vec![make_pumping_station(1)];
    let mut equipment_ctx = equipment_fixtures.make_ctx(vec![], &[]);
    equipment_ctx.non_controllable_sources = &ncs;
    equipment_ctx.pumping_stations = &pumping;
    let equipment_stage = stage_with_blocks(BlockMode::Parallel, 2);
    let equipment_layout = StageLayout::new(&equipment_ctx, &equipment_stage, 0);
    let equipment_counts = compare_column_addresses(&equipment_ctx, &equipment_layout);
    totals.ncs_generation += equipment_counts.ncs_generation;
    totals.pumping_flow += equipment_counts.pumping_flow;

    assert!(totals.inflow_slack > 0, "inflow_slack never compared");
    assert!(
        totals.withdrawal_slack_neg > 0,
        "withdrawal_slack_neg never compared"
    );
    assert!(
        totals.withdrawal_slack_pos > 0,
        "withdrawal_slack_pos never compared"
    );
    assert!(
        totals.anticipated_decision > 0,
        "anticipated_decision never compared"
    );
    assert!(
        totals.filling_target_slack > 0,
        "filling_target_slack never compared"
    );
    assert!(
        totals.filled_min_storage_floor_slack > 0,
        "filled_min_storage_floor_slack never compared"
    );
    assert!(totals.ncs_generation > 0, "ncs_generation never compared");
    assert!(totals.pumping_flow > 0, "pumping_flow never compared");
}

/// Pins the travel-time bucket ring's own addressing against `StateSpace`'s
/// element accessors: a plant's bucket sub-range is addressed through a ring
/// built at its own local offset, and that offset must equal the family's own
/// outgoing/incoming column. This test's left side never changes.
#[test]
fn transit_bucket_addressing_matches_state_space_bucket_accessors() {
    let state = StateSpace::new(
        0,
        0,
        vec![
            (HydroSys::new(0), 0),
            (HydroSys::new(0), 1),
            (HydroSys::new(1), 0),
        ],
        vec![],
        AnticipatedResolution::default(),
        &[],
    );
    let mut compared = 0usize;
    for bucket in DeliveryRing::transit_buckets(&state) {
        for slot in 0..bucket.local.len() {
            assert_eq!(
                bucket.ring.out_col(slot, 0),
                state.bucket_outgoing_col(bucket.local.start + slot).get(),
                "out_col mismatch at slot={slot} local={:?}",
                bucket.local
            );
            assert_eq!(
                bucket.ring.in_col(slot, 0),
                state.bucket_incoming_col(bucket.local.start + slot).get(),
                "in_col mismatch at slot={slot} local={:?}",
                bucket.local
            );
            compared += 1;
        }
    }
    assert!(compared > 0, "must compare at least one bucket");
}

/// Compares every index of the builder's eight hand-built row families on
/// `layout` against a frozen reference formula (job 3,
/// `docs/design/lp-builder-contract.md`): the left side below is written
/// once and never changes across this migration's commits; the right side
/// tracks each site's own production expression, so this test pins the new
/// accessor or primitive against the old derivation as production moves onto
/// it. Returns how many addresses each family compared, in family order
/// (fishing, state-out-def, slot definition, transit definition, filling
/// target, floor, evaporation, generic); a family empty on this particular
/// `layout` compares zero.
fn assert_row_addresses(layout: &StageLayout) -> [usize; 8] {
    let geom = layout.geometry.clone();
    let anticipated = &layout.anticipated;
    let mut counts = [0usize; 8];

    let mut fishing_rows = Vec::new();
    for i in 0..layout.state.n_anticipated {
        let expected = anticipated
            .anticipated_fishing_row_pos
            .get(i)
            .copied()
            .flatten()
            .map(|pos| entity_flat(&anticipated.fishing_rows, pos));
        let actual = layout.anticipated_fishing_row(AnticipatedLocal::new(i));
        assert_eq!(actual, expected, "fishing row disagreement at local {i}");
        if let Some(row) = actual {
            fishing_rows.push(row);
        }
    }
    fishing_rows.sort_unstable();
    assert_eq!(
        fishing_rows,
        anticipated.fishing_rows.clone().collect::<Vec<_>>(),
        "fishing rows must exactly cover their allocated range"
    );
    counts[0] = fishing_rows.len();

    let mut state_out_def_rows = Vec::new();
    for i in 0..layout.state.n_anticipated {
        let expected = anticipated
            .anticipated_decision_row_pos
            .get(i)
            .copied()
            .flatten()
            .map(|pos| entity_flat(&anticipated.state_out_def_rows, pos));
        let actual = layout.anticipated_state_out_def_row(AnticipatedLocal::new(i));
        assert_eq!(
            actual, expected,
            "state-out-def row disagreement at local {i}"
        );
        if let Some(row) = actual {
            state_out_def_rows.push(row);
        }
    }
    state_out_def_rows.sort_unstable();
    assert_eq!(
        state_out_def_rows,
        anticipated.state_out_def_rows.clone().collect::<Vec<_>>(),
        "state-out-def rows must exactly cover their allocated range"
    );
    counts[1] = state_out_def_rows.len();

    let ring = DeliveryRing::anticipated(layout.state);
    let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];
    ring.emit_carry_rows(
        &anticipated.anticipated_slot_row_pos,
        anticipated.slot_definition_rows.start,
        &mut col_entries,
    );

    let mut slot_definition_rows = Vec::new();
    for i in 0..anticipated.anticipated_slot_row_pos.len() {
        let expected = anticipated
            .anticipated_slot_row_pos
            .get(i)
            .copied()
            .flatten()
            .map(|pos| entity_flat(&anticipated.slot_definition_rows, pos));
        let (slot, lane) = ring.slot_lane_at(i);
        let actual = col_entries[ring.out_col(slot, lane)]
            .first()
            .map(|&(row, _)| row);
        assert_eq!(
            actual, expected,
            "slot definition row disagreement at flat {i}"
        );
        if let Some(row) = actual {
            slot_definition_rows.push(row);
        }
    }
    slot_definition_rows.sort_unstable();
    assert_eq!(
        slot_definition_rows,
        anticipated.slot_definition_rows.clone().collect::<Vec<_>>(),
        "slot definition rows must exactly cover their allocated range"
    );
    counts[2] = slot_definition_rows.len();

    let mut transit_rows = Vec::new();
    for bucket in DeliveryRing::transit_buckets(layout.state) {
        let row_pos = &layout.rows.transit_bucket_row_pos[bucket.local.clone()];
        for slot in 0..bucket.local.len() {
            let expected = row_pos
                .get(slot)
                .copied()
                .flatten()
                .map(|pos| layout.rows.transit_bucket_definition.start + pos);
            let actual = layout.transit_bucket_definition_row(&bucket.local, slot);
            assert_eq!(
                actual, expected,
                "transit definition row disagreement at local={:?} slot={slot}",
                bucket.local
            );
            if let Some(row) = actual {
                transit_rows.push(row);
            }
        }
    }
    transit_rows.sort_unstable();
    assert_eq!(
        transit_rows,
        layout
            .rows
            .transit_bucket_definition
            .clone()
            .collect::<Vec<_>>(),
        "transit definition rows must exactly cover their allocated range"
    );
    counts[3] = transit_rows.len();

    for local_idx in 0..layout.geometry.filling_target_hydro_indices.len() {
        let expected = geom
            .filling_target
            .clone()
            .nth(local_idx)
            .expect("local_idx within filling_target range");
        let actual = layout.filling_target_row(FillingTargetLocal::new(local_idx));
        assert_eq!(
            actual, expected,
            "filling target row disagreement at local {local_idx}"
        );
        counts[4] += 1;
    }

    for local_idx in 0..layout.geometry.filled_min_storage_floor_hydro_indices.len() {
        let expected = geom
            .filled_min_storage_floor
            .clone()
            .nth(local_idx)
            .expect("local_idx within filled_min_storage_floor range");
        let actual = layout.filled_min_storage_floor_row(FloorLocal::new(local_idx));
        assert_eq!(
            actual, expected,
            "floor row disagreement at local {local_idx}"
        );
        counts[5] += 1;
    }

    for k in 0..geom.evap_indices.len() {
        let expected = geom.evap_indices[k].evap_row;
        let l = k / layout.n_evap_slots;
        let s = k % layout.n_evap_slots;
        let actual = layout.evap_row(EvapLocal::new(l), BlockIdx::new(s));
        assert_eq!(actual, expected, "evap row disagreement at k={k}");
        counts[6] += 1;
    }

    assert_eq!(
        layout.rows.n_generic_rows,
        layout.generic_constraint_rows.len(),
        "n_generic_rows must equal generic_constraint_rows.len()"
    );
    for i in 0..layout.generic_constraint_rows.len() {
        let expected = layout.rows.row_generic_start + i;
        let actual = layout.generic_row(i);
        assert_eq!(actual, expected, "generic row disagreement at i={i}");
        counts[7] += 1;
    }

    counts
}

/// Pins every hand-built row family's address against a frozen reference
/// formula. No single fixture in this file populates every family at once,
/// so this runs [`assert_row_addresses`] over several, in both block modes,
/// and sums the per-family counts.
#[test]
fn row_address_pins_cover_every_family() {
    let mut totals = [0usize; 8];

    // Fishing + state-out-def + slot-definition (carry): two anticipated
    // plants with heterogeneous leads (2, 3) sharing a depth-3 ring — the
    // short-lead plant's deposit sits in flight (a carry row) the stage after
    // its own deposit, before the long-lead plant's own next deposit reaches
    // that residue.
    let mut ant_fixtures = AntFixturesWithNStages::new(6);
    let ant_ctx = ant_fixtures.make_ctx(vec![2, 3], &[0, 1]);
    let ant_stage = minimal_stage();
    for stage_idx in [0, 1] {
        let layout = StageLayout::new(&ant_ctx, &ant_stage, stage_idx);
        let counts = assert_row_addresses(&layout);
        for (total, count) in totals.iter_mut().zip(counts) {
            *total += count;
        }
    }

    // Filling target + evaporation (Parallel).
    let mut filling_fixtures = FillingMembershipFixtures::new();
    let filling_ctx = filling_fixtures.make_ctx();
    let filling_stage = stage_with_id(1);
    let filling_layout = StageLayout::new(&filling_ctx, &filling_stage, 0);
    let filling_counts = assert_row_addresses(&filling_layout);
    for (total, count) in totals.iter_mut().zip(filling_counts) {
        *total += count;
    }

    // Floor + evaporation (Chronological — exercises the other block mode).
    let mut operating_stage = stage_with_id(3);
    operating_stage.block_mode = BlockMode::Chronological;
    operating_stage.blocks = (0..2)
        .map(|index| Block {
            index,
            name: format!("BLK{index}"),
            duration_hours: 372.0,
        })
        .collect();
    let operating_layout = StageLayout::new(&filling_ctx, &operating_stage, 0);
    let operating_counts = assert_row_addresses(&operating_layout);
    for (total, count) in totals.iter_mut().zip(operating_counts) {
        *total += count;
    }

    // Generic: one block-varying symbolic-upper-bound constraint, two blocks.
    let mut generic_fixtures = ZeroEntityFixtures::new();
    generic_fixtures.install_symbolic_upper_bound();
    let generic_ctx = generic_fixtures.make_ctx_generic();
    let generic_stage = stage_with_blocks(BlockMode::Parallel, 2);
    let generic_layout = StageLayout::new(&generic_ctx, &generic_stage, 0);
    let generic_counts = assert_row_addresses(&generic_layout);
    for (total, count) in totals.iter_mut().zip(generic_counts) {
        *total += count;
    }

    // Transit-bucket definition: one downstream plant, one reachable lag.
    let mut transit_fixtures = ZeroEntityFixtures::new();
    transit_fixtures.base.topology.column_order = vec![(HydroSys::new(0), 1)];
    transit_fixtures.base.topology.per_stage_mask = vec![vec![1]];
    let transit_ctx = transit_fixtures.make_ctx(vec![], &[]);
    assert_eq!(
        transit_ctx.state.n_buckets, 1,
        "the context's own state must carry the bucket this case addresses"
    );
    let transit_stage = minimal_stage();
    let transit_layout = StageLayout::new(&transit_ctx, &transit_stage, 0);
    let transit_counts = assert_row_addresses(&transit_layout);
    for (total, count) in totals.iter_mut().zip(transit_counts) {
        *total += count;
    }

    let family_names = [
        "fishing",
        "state-out-def",
        "slot definition",
        "transit definition",
        "filling target",
        "floor",
        "evaporation",
        "generic",
    ];
    for (name, total) in family_names.iter().zip(totals) {
        assert!(total > 0, "{name} never compared");
    }
}

/// `PumpingFlow` and `PumpingPower` are block-DEPENDENT — per-block columns,
/// so the single-row collapse must NOT apply (they stay in the `false` arm).
#[test]
fn pumping_variants_are_block_dependent() {
    assert!(!variable_ref_is_block_independent(
        &VariableRef::PumpingFlow {
            station_id: EntityId(10),
            block_id: None,
        }
    ));
    assert!(!variable_ref_is_block_independent(
        &VariableRef::PumpingPower {
            station_id: EntityId(10),
            block_id: None,
        }
    ));
}

/// `HydroStorage`, `HydroEvaporation`, and `AnticipatedDecision` are
/// block-INDEPENDENT — stage-level stock variables whose resolver ignores
/// `block_idx`, so the single-row collapse is sound. This is the `true`-arm
/// counterpart to `pumping_variants_are_block_dependent` /
/// `hydro_inflow_is_block_dependent`: dropping any of these three from the
/// `true` branch of `variable_ref_is_block_independent` would silently expand
/// a per-stage stock variable into per-block rows.
#[test]
fn block_independent_kinds_classify_true() {
    assert!(variable_ref_is_block_independent(
        &VariableRef::HydroStorage {
            hydro_id: EntityId(10),
        }
    ));
    assert!(variable_ref_is_block_independent(
        &VariableRef::HydroEvaporation {
            hydro_id: EntityId(10),
            block_id: None,
        }
    ));
    assert!(variable_ref_is_block_independent(
        &VariableRef::AnticipatedDecision {
            thermal_id: EntityId(6),
        }
    ));
}

/// `HydroInflow` is block-DEPENDENT — its upstream releases are per-block
/// columns, so the single-row collapse must NOT apply.
#[test]
fn hydro_inflow_is_block_dependent() {
    assert!(!variable_ref_is_block_independent(
        &VariableRef::HydroInflow {
            hydro_id: EntityId(0),
            block_id: None,
        }
    ));
}

/// Both storage boundary variants resolve to a fixed column (a stage endpoint or a
/// named boundary), so they are block-INDEPENDENT (`true`) for `None` and `Some`
/// alike, like the stage-final alias `HydroStorage`.
#[test]
fn storage_boundary_variants_are_block_independent() {
    for block_id in [None, Some(1)] {
        assert!(variable_ref_is_block_independent(
            &VariableRef::HydroStorageInitial {
                hydro_id: EntityId(10),
                block_id,
            }
        ));
        assert!(variable_ref_is_block_independent(
            &VariableRef::HydroStorageFinal {
                hydro_id: EntityId(10),
                block_id,
            }
        ));
    }
    assert!(variable_ref_is_block_independent(
        &VariableRef::HydroStorage {
            hydro_id: EntityId(10),
        }
    ));
}

// ── Generic-constraint slack allocation ─────────────────────────────────────

/// Regression guard for the one-slack-allocation defect: a two-sided entry
/// produced by the REAL layout path (`allocate_generic_slack_cols`, via
/// `StageLayout::new`) must carry `slack_minus_col == Some(_)`. Asserts the
/// CORRECT case, not the broken one, so this fails if
/// `allocate_generic_slack_cols` ever regresses to allocating a minus column
/// only for a specific label instead of deriving two-sidedness from the row's
/// own endpoint pair. Sourced from the real layout rather than hand-built: a
/// hand-built entry would assert nothing about allocation.
#[test]
fn two_sided_real_layout_allocates_minus_slack_column() {
    let constraint = GenericConstraint {
        id: cobre_core::EntityId(1),
        name: "gc_range_test".to_string(),
        description: None,
        expression: ConstraintExpression { terms: vec![] },
        slack: SlackConfig {
            enabled: true,
            penalty: Some(10.0),
        },
        bound_lower_affine: None,
        bound_upper_affine: None,
    };

    let id_map: HashMap<i32, usize> = [(1, 0)].into_iter().collect();
    let raw_bounds = vec![(1i32, 0i32, Some(0i32), Some(5.0_f64), Some(20.0_f64))];
    let resolved_generic_bounds =
        ResolvedGenericConstraintBounds::new(&id_map, raw_bounds.into_iter());

    let mut fixture = CtxFixture {
        hydro_cell_index: identity_hydro_cell_index(0),
        production_models: ProductionModelSet::new(Vec::new(), &[], 1),
        evaporation_models: EvaporationModelSet::new(Vec::new()),
        generic_constraints: vec![constraint],
        resolved_generic_bounds,
        time_value: TimeValue::from_parts(
            vec![],
            vec![1.0],
            vec![730.0],
            vec![0],
            PostStudyResolved::default(),
        ),
        ..CtxFixture::default()
    };
    let ctx = fixture.ctx();

    let stage = Stage {
        index: 0,
        id: 0,
        start_date: NaiveDate::default(),
        end_date: NaiveDate::default(),
        season_id: Some(0),
        blocks: vec![Block {
            index: 0,
            name: "BLK0".to_string(),
            duration_hours: 730.0,
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
    };

    let layout = StageLayout::new(&ctx, &stage, 0);

    assert_eq!(
        layout.generic_constraint_rows.len(),
        1,
        "one active (constraint, block) row"
    );
    let entry = &layout.generic_constraint_rows[0];
    assert_eq!(entry.bound_lower, Some(5.0));
    assert_eq!(entry.bound_upper, Some(20.0));
    assert!(entry.slack_plus_col.is_some());
    assert!(
        entry.slack_minus_col.is_some(),
        "a two-sided row with slack enabled must allocate a DISTINCT minus-slack column"
    );
}
