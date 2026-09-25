#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::too_many_lines,
    clippy::similar_names
)]

use std::collections::{BTreeMap, HashMap};

use chrono::NaiveDate;
use cobre_core::{
    AffineBound, Block, BlockMode, BoundsCountsSpec, BoundsDefaults, CascadeTopology,
    ConstraintExpression, ContractBlockBounds, EntityId, FillingConfig, GenericConstraint, Hydro,
    HydroBlockBounds, HydroGenerationModel, HydroStageBounds, LineBlockBounds, LinearTerm,
    NoiseMethod, PumpingBlockBounds, PumpingStation, ResolvedBounds,
    ResolvedGenericConstraintBounds, ResolvedLoadFactors, ResolvedNcsBounds, ResolvedNcsFactors,
    ResolvedPenalties, ScenarioSourceConfig, SlackConfig, Stage, StageRiskConfig, StageStateConfig,
    ThermalBlockBounds, ThermalStageBounds, VariableRef,
};
use cobre_stochastic::par::precompute::PrecomputedPar;

use crate::hydro_models::{EvaporationModelSet, ProductionModelSet};
use crate::indexer::{
    BlockIdx, Boundary, EvapLocal, FphaCellLocal, FphaLocal, HydroCell, HydroCellIndex, HydroSys,
    LineSys, ThermalSys,
};
use crate::lead_time::{AnticipatedResolution, DeliveryAxis, LeadTime, PointResolution};
use crate::resolved_parameters::ResolvedParameters;
use crate::setup::PostStudyResolved;
use crate::test_support::{make_unit_group, state_layout};
use crate::time_value::TimeValue;

use super::super::test_support::{state_layout_for, zero_hydro_penalties};
use super::{
    EVAP_COLS_PER_HYDRO, EVAP_F_MINUS_OFFSET, EVAP_F_PLUS_OFFSET, EVAP_FLOW_OFFSET, RangeCursor,
    ResolvedTables, StageLayout, StateSpace, TemplateBuildCtx, build_anticipated_fishing_row_pos,
    build_anticipated_slot_row_pos, build_transit_bucket_row_pos, fold_endpoint,
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
    par_lp: PrecomputedPar,
    cascade: CascadeTopology,
    hydro_cell_index: HydroCellIndex,
    bounds: ResolvedBounds,
    penalties: ResolvedPenalties,
    resolved_generic_bounds: ResolvedGenericConstraintBounds,
    resolved_load_factors: ResolvedLoadFactors,
    resolved_ncs_bounds: ResolvedNcsBounds,
    resolved_ncs_factors: ResolvedNcsFactors,
    resolved_parameters: ResolvedParameters,
    production_models: ProductionModelSet,
    evaporation_models: EvaporationModelSet,
    generic_constraints: Vec<GenericConstraint>,
}

impl ZeroEntityFixtures {
    fn new() -> Self {
        Self {
            par_lp: PrecomputedPar::default(),
            cascade: CascadeTopology::build(&[]),
            hydro_cell_index: HydroCellIndex::build(&[]),
            bounds: ResolvedBounds::empty(),
            penalties: ResolvedPenalties::empty(),
            resolved_generic_bounds: ResolvedGenericConstraintBounds::empty(),
            resolved_load_factors: ResolvedLoadFactors::empty(),
            resolved_ncs_bounds: ResolvedNcsBounds::empty(),
            resolved_ncs_factors: ResolvedNcsFactors::empty(),
            resolved_parameters: ResolvedParameters {
                per_param: vec![],
                id_to_slot: vec![],
                cost_scale_factor: 1_000_000.0,
            },
            production_models: ProductionModelSet::new(vec![], 0, 1),
            evaporation_models: EvaporationModelSet::new(vec![]),
            generic_constraints: Vec::new(),
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
        self.generic_constraints = vec![GenericConstraint {
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
        self.resolved_generic_bounds = ResolvedGenericConstraintBounds::new(
            &id_map,
            std::iter::once((5i32, 0i32, None::<i32>, Some(5.0f64), None::<f64>)),
        );
        self.resolved_parameters = ResolvedParameters {
            per_param: vec![vec![vec![100.0, 200.0]]],
            id_to_slot: vec![(42, 0)],
            cost_scale_factor: 1_000_000.0,
        };
    }

    /// Install one generic constraint (id 5, no slack) whose UPPER bound carries
    /// BOTH a numeric parquet base (`100.0`) and a constant-only affine remainder
    /// (`-5.0`): the fold arm `fold_endpoint` newly reaches, `base + R`.
    fn install_folded_upper_bound_constant(&mut self) {
        self.generic_constraints = vec![GenericConstraint {
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
        self.resolved_generic_bounds = ResolvedGenericConstraintBounds::new(
            &id_map,
            std::iter::once((5i32, 0i32, None::<i32>, None::<f64>, Some(100.0f64))),
        );
    }

    /// A zero-anticipated `TemplateBuildCtx` that carries the fixture's own
    /// generic constraints (rather than the empty slice `make_ctx` installs).
    fn make_ctx_generic(&self) -> TemplateBuildCtx<'_> {
        let mut ctx = self.make_ctx(0, 0, vec![], vec![]);
        ctx.generic_constraints = &self.generic_constraints;
        ctx
    }

    /// Build a zero-entity `TemplateBuildCtx` with the supplied
    /// anticipated-metadata overrides.
    ///
    /// All slice fields are empty; all scalar entity counts are zero except
    /// the anticipated fields provided by the caller.
    fn make_ctx(
        &self,
        n_anticipated: usize,
        k_max: usize,
        anticipated_lead_stages: Vec<usize>,
        anticipated_thermal_indices: Vec<usize>,
    ) -> TemplateBuildCtx<'_> {
        TemplateBuildCtx {
            hydros: &[],
            thermals: &[],
            lines: &[],
            buses: &[],
            load_models: &[],
            cascade: &self.cascade,
            hydro_cell_index: &self.hydro_cell_index,
            resolved: ResolvedTables {
                bounds: &self.bounds,
                penalties: &self.penalties,
                resolved_generic_bounds: &self.resolved_generic_bounds,
                resolved_load_factors: &self.resolved_load_factors,
                resolved_ncs_bounds: &self.resolved_ncs_bounds,
                resolved_ncs_factors: &self.resolved_ncs_factors,
                resolved_parameters: &self.resolved_parameters,
            },
            hydro_pos: BTreeMap::new(),
            thermal_pos: BTreeMap::new(),
            line_pos: BTreeMap::new(),
            bus_pos: BTreeMap::new(),
            par_lp: &self.par_lp,
            production_models: &self.production_models,
            evaporation_models: &self.evaporation_models,
            generic_constraints: &[],
            non_controllable_sources: &[],
            pumping_stations: &[],
            pumping_pos: BTreeMap::new(),
            n_pumping: 0,
            contracts: &[],
            contract_pos: BTreeMap::new(),
            n_contract_import: 0,
            n_contract_export: 0,
            diversion_upstream: HashMap::new(),
            arc_stage_weights: HashMap::new(),
            arc_spread_chrono: HashMap::new(),
            arc_arrival_density: HashMap::new(),
            per_stage_mask: Vec::new(),
            post_study_resolved: PostStudyResolved::default(),
            n_hydros: 0,
            n_thermals: 0,
            n_lines: 0,
            n_buses: 0,
            max_par_order: 0,
            n_anticipated,
            k_max,
            anticipated_lead_stages,
            // Windowless: one `(None, None)` per anticipated plant. With no
            // window the operation-window clause is identically true, so the
            // decision gate reduces to the strict horizon clause — the
            // behaviour these layout tests assert. `study_stage_ids` is sized
            // to the bounds' study-stage count so the gate's in-range
            // delivery-stage lookup never indexes out of bounds.
            anticipated_windows: vec![(None, None); n_anticipated],
            anticipated_resolution: AnticipatedResolution::default(),
            study_stage_ids: (0..i32::try_from(self.bounds.n_stages()).unwrap_or(0)).collect(),
            delivery_stage_ids: (0..i32::try_from(self.bounds.n_stages()).unwrap_or(0)).collect(),
            anticipated_thermal_indices: anticipated_thermal_indices
                .into_iter()
                .map(ThermalSys::new)
                .collect(),
            has_penalty: false,
            // Tests that use ZeroEntityFixtures don't exercise discount
            // factors; provide n_stages = 1 element vecs that won't panic.
            time_value: TimeValue::new(vec![1.0]),
            delivery_total_hours: vec![744.0],
            filling_v_target: BTreeMap::new(),
        }
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

/// `StageLayout` built from a context with `n_anticipated == 0` has
/// `n_ant_state == 0`, `n_anticipated == 0`, `k_max == 0`, and
/// `col_turbine_start == idx.theta + 1` where `idx` is the N=0, L=0 state
/// layout (zero hydros, zero lag order).
///
/// This verifies that the decision-region offset before the anticipated-ring
/// insertion is preserved when no anticipated thermals are present.
#[test]
fn stage_layout_zero_anticipated_matches_pre_anticipated_offsets() {
    let fixtures = ZeroEntityFixtures::new();
    let ctx = fixtures.make_ctx(0, 0, vec![], vec![]);
    let stage = minimal_stage();
    let state = state_layout_for(&ctx);
    let layout = StageLayout::new(&ctx, &state, &stage, 0);

    assert_eq!(layout.n_ant_state, 0, "n_ant_state");
    assert_eq!(layout.n_anticipated, 0, "n_anticipated");
    assert_eq!(layout.k_max, 0, "k_max");

    let idx = state_layout(ctx.n_hydros, ctx.max_par_order);
    assert_eq!(
        layout.equipment.turbine.start,
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
    let state = state_layout_for(&ctx);
    let stage = stage_with_blocks(BlockMode::Parallel, 2);
    let layout = StageLayout::new(&ctx, &state, &stage, 0);

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
    let state = state_layout_for(&ctx);
    let stage = stage_with_blocks(BlockMode::Parallel, 2);
    let layout = StageLayout::new(&ctx, &state, &stage, 0);

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
    let state = state_layout_for(&ctx);
    let stage = minimal_stage();
    let layout = StageLayout::new(&ctx, &state, &stage, 0);

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
    par_lp: PrecomputedPar,
    hydros: Vec<Hydro>,
    cascade: CascadeTopology,
    hydro_cell_index: HydroCellIndex,
    bounds: ResolvedBounds,
    penalties: ResolvedPenalties,
    resolved_generic_bounds: ResolvedGenericConstraintBounds,
    resolved_load_factors: ResolvedLoadFactors,
    resolved_ncs_bounds: ResolvedNcsBounds,
    resolved_ncs_factors: ResolvedNcsFactors,
    resolved_parameters: ResolvedParameters,
    production_models: ProductionModelSet,
    evaporation_models: EvaporationModelSet,
    hydro_pos: BTreeMap<EntityId, usize>,
    generic_constraints: Vec<GenericConstraint>,
}

impl UsefulVolumeFixtures {
    /// `n_hydros` hydros at ids `1..=n_hydros` (positions `0..n_hydros`), `n_stages`
    /// stages, every entity `min_storage_hm3` defaulted to `0.0` — set per test via
    /// `hydros[pos].min_storage_hm3` (the useful-volume fold's source); `bounds`
    /// stays available to set a differing per-stage operative value.
    fn new(n_hydros: usize, n_stages: usize) -> Self {
        let hydro_pos = (0..n_hydros)
            .map(|i| (EntityId(i32::try_from(i + 1).expect("small test id")), i))
            .collect();
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
        Self {
            par_lp: PrecomputedPar::default(),
            hydros,
            cascade: CascadeTopology::build(&[]),
            hydro_cell_index: HydroCellIndex::build(&[]),
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
            penalties: ResolvedPenalties::empty(),
            resolved_generic_bounds: ResolvedGenericConstraintBounds::empty(),
            resolved_load_factors: ResolvedLoadFactors::empty(),
            resolved_ncs_bounds: ResolvedNcsBounds::empty(),
            resolved_ncs_factors: ResolvedNcsFactors::empty(),
            resolved_parameters: ResolvedParameters {
                per_param: vec![],
                id_to_slot: vec![],
                cost_scale_factor: 1_000_000.0,
            },
            production_models: ProductionModelSet::new(vec![], 0, 1),
            evaporation_models: EvaporationModelSet::new(vec![]),
            hydro_pos,
            generic_constraints: Vec::new(),
        }
    }

    /// Install one generic constraint (id 5) with the given expression `terms` and
    /// resolved bound rows (`(stage_id, block_id, bound_lower, bound_upper)`).
    fn install_constraint(&mut self, terms: Vec<LinearTerm>, raw_bounds: Vec<RawBoundRow>) {
        self.generic_constraints = vec![GenericConstraint {
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
        self.resolved_generic_bounds = ResolvedGenericConstraintBounds::new(
            &id_map,
            raw_bounds
                .into_iter()
                .map(|(stage_id, block_id, lo, hi)| (5i32, stage_id, block_id, lo, hi)),
        );
    }

    fn make_ctx(&self) -> TemplateBuildCtx<'_> {
        let n_stages = self.bounds.n_stages();
        TemplateBuildCtx {
            hydros: &self.hydros,
            thermals: &[],
            lines: &[],
            buses: &[],
            load_models: &[],
            cascade: &self.cascade,
            hydro_cell_index: &self.hydro_cell_index,
            resolved: ResolvedTables {
                bounds: &self.bounds,
                penalties: &self.penalties,
                resolved_generic_bounds: &self.resolved_generic_bounds,
                resolved_load_factors: &self.resolved_load_factors,
                resolved_ncs_bounds: &self.resolved_ncs_bounds,
                resolved_ncs_factors: &self.resolved_ncs_factors,
                resolved_parameters: &self.resolved_parameters,
            },
            hydro_pos: self.hydro_pos.clone(),
            thermal_pos: BTreeMap::new(),
            line_pos: BTreeMap::new(),
            bus_pos: BTreeMap::new(),
            par_lp: &self.par_lp,
            production_models: &self.production_models,
            evaporation_models: &self.evaporation_models,
            generic_constraints: &self.generic_constraints,
            non_controllable_sources: &[],
            pumping_stations: &[],
            pumping_pos: BTreeMap::new(),
            n_pumping: 0,
            contracts: &[],
            contract_pos: BTreeMap::new(),
            n_contract_import: 0,
            n_contract_export: 0,
            diversion_upstream: HashMap::new(),
            arc_stage_weights: HashMap::new(),
            arc_spread_chrono: HashMap::new(),
            arc_arrival_density: HashMap::new(),
            per_stage_mask: Vec::new(),
            post_study_resolved: PostStudyResolved::default(),
            n_hydros: 0,
            n_thermals: 0,
            n_lines: 0,
            n_buses: 0,
            max_par_order: 0,
            n_anticipated: 0,
            k_max: 0,
            anticipated_lead_stages: vec![],
            anticipated_thermal_indices: vec![],
            anticipated_windows: vec![],
            anticipated_resolution: AnticipatedResolution::default(),
            study_stage_ids: (0..i32::try_from(n_stages).unwrap_or(0)).collect(),
            delivery_stage_ids: (0..i32::try_from(n_stages).unwrap_or(0)).collect(),
            has_penalty: false,
            time_value: TimeValue::new(vec![1.0; n_stages]),
            delivery_total_hours: vec![744.0; n_stages],
            filling_v_target: BTreeMap::new(),
        }
    }
}

/// `1.0 * hydro_useful_volume_final(h) >= B` resolves to `B + 1.0*V_lo(h,t)`.
#[test]
fn useful_volume_single_term_lower_bound_folds_v_lo() {
    let mut fixtures = UsefulVolumeFixtures::new(1, 1);
    let h = EntityId(1);
    fixtures.hydros[0].min_storage_hm3 = 12.5;
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
    let state = state_layout_for(&ctx);
    let stage = minimal_stage();
    let layout = StageLayout::new(&ctx, &state, &stage, 0);

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
    fixtures.hydros[0].min_storage_hm3 = 12.5;
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
    let state = state_layout_for(&ctx);
    let stage = minimal_stage();
    let layout = StageLayout::new(&ctx, &state, &stage, 0);

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
    fixtures.hydros[0].min_storage_hm3 = 10.0;
    fixtures.hydros[1].min_storage_hm3 = 4.0;
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
    let state = state_layout_for(&ctx);
    let stage = minimal_stage();
    let layout = StageLayout::new(&ctx, &state, &stage, 0);

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
    fixtures.hydros[0].min_storage_hm3 = 99.0;
    fixtures.install_constraint(vec![], vec![(0, None, Some(-0.0), None)]);
    let ctx = fixtures.make_ctx();
    let state = state_layout_for(&ctx);
    let stage = minimal_stage();
    let layout = StageLayout::new(&ctx, &state, &stage, 0);

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
    fixtures.hydros[0].min_storage_hm3 = 7.0;
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
    let state = state_layout_for(&ctx);
    let stage = minimal_stage();
    let layout = StageLayout::new(&ctx, &state, &stage, 0);

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
    fixtures.hydros[0].min_storage_hm3 = 5.0;
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
    let state = state_layout_for(&ctx);
    let stage = stage_with_blocks(BlockMode::Parallel, 3);
    let layout = StageLayout::new(&ctx, &state, &stage, 0);

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

/// C5: the fold reads the ENTITY physical `min_storage_hm3` — stage-invariant —
/// never the per-stage resolved `HydroStageBounds.min_storage_hm3`, which can
/// carry an operative floor (flood control, DECOMP RHV) diverging from it.
#[test]
fn useful_volume_fold_uses_entity_physical_v_lo_not_per_stage_operative_bounds() {
    let mut fixtures = UsefulVolumeFixtures::new(1, 2);
    let h = EntityId(1);
    // Per-stage operative bounds differ from each other and from the entity
    // physical value; the fold must track neither.
    fixtures.bounds.hydro_bounds_mut(0, 0).min_storage_hm3 = 5.0;
    fixtures.bounds.hydro_bounds_mut(0, 1).min_storage_hm3 = 8.0;
    fixtures.hydros[0].min_storage_hm3 = 20.0;
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
    let state = state_layout_for(&ctx);

    let stage0 = stage_with_id(0);
    let layout0 = StageLayout::new(&ctx, &state, &stage0, 0);
    assert_eq!(
        layout0.generic_constraint_rows[0]
            .bound_lower
            .expect("present")
            .to_bits(),
        30.0_f64.to_bits(),
        "stage 0: 10 + 20.0 (entity physical value, not the operative 5.0)"
    );

    let stage1 = stage_with_id(1);
    let layout1 = StageLayout::new(&ctx, &state, &stage1, 1);
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
    fixtures.hydros[0].min_storage_hm3 = 3.0;
    fixtures.resolved_parameters = ResolvedParameters {
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
    let state = state_layout_for(&ctx);
    let stage = minimal_stage();
    let layout = StageLayout::new(&ctx, &state, &stage, 0);

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
    let state = state_layout_for(&ctx);
    let stage = minimal_stage();
    let _ = StageLayout::new(&ctx, &state, &stage, 0);
}

// ── storage_internal interior-boundary sizing ────────────────────────────

/// Owns a two-hydro, constant-productivity `TemplateBuildCtx` for the
/// `storage_internal` sizing assertions. No FPHA/filling/evaporation, so only
/// the block geometry (`n_blks`, `block_mode`) and `n_hydros` drive the family.
struct TwoHydroFixtures {
    par_lp: PrecomputedPar,
    hydros: Vec<Hydro>,
    hydro_cell_index: HydroCellIndex,
    cascade: CascadeTopology,
    bounds: ResolvedBounds,
    penalties: ResolvedPenalties,
    resolved_generic_bounds: ResolvedGenericConstraintBounds,
    resolved_load_factors: ResolvedLoadFactors,
    resolved_ncs_bounds: ResolvedNcsBounds,
    resolved_ncs_factors: ResolvedNcsFactors,
    resolved_parameters: ResolvedParameters,
    production_models: ProductionModelSet,
    evaporation_models: EvaporationModelSet,
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
        Self {
            par_lp: PrecomputedPar::default(),
            hydros,
            hydro_cell_index,
            cascade,
            bounds: ResolvedBounds::empty(),
            penalties: ResolvedPenalties::empty(),
            resolved_generic_bounds: ResolvedGenericConstraintBounds::empty(),
            resolved_load_factors: ResolvedLoadFactors::empty(),
            resolved_ncs_bounds: ResolvedNcsBounds::empty(),
            resolved_ncs_factors: ResolvedNcsFactors::empty(),
            resolved_parameters: ResolvedParameters {
                per_param: vec![],
                id_to_slot: vec![],
                cost_scale_factor: 1_000_000.0,
            },
            production_models: ProductionModelSet::new(models, 2, 1),
            evaporation_models: EvaporationModelSet::new(vec![
                EvaporationModel::None,
                EvaporationModel::None,
            ]),
        }
    }

    fn make_ctx(&self) -> TemplateBuildCtx<'_> {
        TemplateBuildCtx {
            hydros: &self.hydros,
            thermals: &[],
            lines: &[],
            buses: &[],
            load_models: &[],
            cascade: &self.cascade,
            hydro_cell_index: &self.hydro_cell_index,
            resolved: ResolvedTables {
                bounds: &self.bounds,
                penalties: &self.penalties,
                resolved_generic_bounds: &self.resolved_generic_bounds,
                resolved_load_factors: &self.resolved_load_factors,
                resolved_ncs_bounds: &self.resolved_ncs_bounds,
                resolved_ncs_factors: &self.resolved_ncs_factors,
                resolved_parameters: &self.resolved_parameters,
            },
            hydro_pos: BTreeMap::new(),
            thermal_pos: BTreeMap::new(),
            line_pos: BTreeMap::new(),
            bus_pos: BTreeMap::new(),
            par_lp: &self.par_lp,
            production_models: &self.production_models,
            evaporation_models: &self.evaporation_models,
            generic_constraints: &[],
            non_controllable_sources: &[],
            pumping_stations: &[],
            pumping_pos: BTreeMap::new(),
            n_pumping: 0,
            contracts: &[],
            contract_pos: BTreeMap::new(),
            n_contract_import: 0,
            n_contract_export: 0,
            diversion_upstream: HashMap::new(),
            arc_stage_weights: HashMap::new(),
            arc_spread_chrono: HashMap::new(),
            arc_arrival_density: HashMap::new(),
            per_stage_mask: Vec::new(),
            post_study_resolved: PostStudyResolved::default(),
            n_hydros: 2,
            n_thermals: 0,
            n_lines: 0,
            n_buses: 0,
            max_par_order: 0,
            n_anticipated: 0,
            k_max: 0,
            anticipated_lead_stages: vec![],
            anticipated_thermal_indices: vec![],
            anticipated_windows: vec![],
            anticipated_resolution: AnticipatedResolution::default(),
            study_stage_ids: vec![],
            delivery_stage_ids: vec![],
            has_penalty: false,
            time_value: TimeValue::new(vec![1.0]),
            delivery_total_hours: vec![744.0],
            filling_v_target: BTreeMap::new(),
        }
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

/// `storage_internal` spans the `K − 1` interior boundaries per hydro only in
/// chronological mode with `K ≥ 2`: empty in parallel mode and at `K = 1`, with
/// `turbine.start` re-anchored to `storage_internal.end`.
#[test]
fn chronological_storage_internal_sizing() {
    let fixtures = TwoHydroFixtures::new();
    let ctx = fixtures.make_ctx();
    let state = state_layout_for(&ctx);
    let anchor = state.control_region_start();

    let stage_parallel = stage_with_blocks(BlockMode::Parallel, 3);
    let parallel = StageLayout::new(&ctx, &state, &stage_parallel, 0);
    assert_eq!(
        parallel.equipment.storage_internal.start, parallel.equipment.storage_internal.end,
        "parallel K=3 storage_internal is empty"
    );
    assert_eq!(
        parallel.equipment.storage_internal_start, anchor,
        "parallel storage_internal_start anchors at control_region_start()"
    );
    assert_eq!(
        parallel.equipment.turbine.start, anchor,
        "parallel turbine.start anchors at control_region_start()"
    );

    let stage_chrono_k1 = stage_with_blocks(BlockMode::Chronological, 1);
    let chrono_k1 = StageLayout::new(&ctx, &state, &stage_chrono_k1, 0);
    assert_eq!(
        chrono_k1.equipment.storage_internal.start, chrono_k1.equipment.storage_internal.end,
        "chronological K=1 storage_internal is empty"
    );
    assert_eq!(
        chrono_k1.equipment.storage_internal_start, anchor,
        "chronological K=1 storage_internal_start anchors at control_region_start()"
    );
    assert_eq!(
        chrono_k1.equipment.turbine.start, anchor,
        "chronological K=1 turbine.start anchors at control_region_start()"
    );

    let stage_chrono_k3 = stage_with_blocks(BlockMode::Chronological, 3);
    let chrono_k3 = StageLayout::new(&ctx, &state, &stage_chrono_k3, 0);
    assert_eq!(
        chrono_k3.equipment.storage_internal_start, anchor,
        "chronological K=3 storage_internal_start anchors at control_region_start()"
    );
    assert_eq!(
        chrono_k3.equipment.storage_internal.end - chrono_k3.equipment.storage_internal.start,
        4,
        "chronological K=3 storage_internal spans n_h * (K - 1) = 2 * 2 columns"
    );
    assert_eq!(
        chrono_k3.equipment.turbine.start, chrono_k3.equipment.storage_internal.end,
        "chronological K=3 turbine.start re-anchors to storage_internal.end"
    );
}

/// `block_storage_col` resolves all `K + 1` boundaries: the two endpoints to the
/// state columns (`k = 0 → storage_in[h]`, `k = K → storage[h] = h`) and the
/// `K − 1` interiors into the `storage_internal` family at stride `n_blks − 1`. At
/// `K = 1` only the two endpoints resolve (no interior column is addressed).
#[test]
fn block_storage_col_resolves_all_boundaries() {
    let fixtures = TwoHydroFixtures::new();
    let ctx = fixtures.make_ctx();
    let state = state_layout_for(&ctx);

    let stage_chrono_k3 = stage_with_blocks(BlockMode::Chronological, 3);
    let chrono_k3 = StageLayout::new(&ctx, &state, &stage_chrono_k3, 0);
    let h = 1;
    assert_eq!(
        chrono_k3.block_storage_col(HydroSys::new(h), Boundary::Incoming),
        chrono_k3.col_storage_in_start() + h,
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
        chrono_k3.equipment.storage_internal_start + h * 2,
        "k = 1 resolves to storage_internal_start + h * (K - 1) + 0"
    );
    assert_eq!(
        interior_2,
        chrono_k3.equipment.storage_internal_start + h * 2 + 1,
        "k = 2 resolves to storage_internal_start + h * (K - 1) + 1"
    );
    assert!(
        chrono_k3.equipment.storage_internal.contains(&interior_1)
            && chrono_k3.equipment.storage_internal.contains(&interior_2),
        "both interior columns lie within the storage_internal range"
    );

    let stage_chrono_k1 = stage_with_blocks(BlockMode::Chronological, 1);
    let chrono_k1 = StageLayout::new(&ctx, &state, &stage_chrono_k1, 0);
    assert!(
        chrono_k1.equipment.storage_internal.is_empty(),
        "K = 1 has no interior storage columns"
    );
    for h in 0..ctx.n_hydros {
        assert_eq!(
            chrono_k1.block_storage_col(HydroSys::new(h), Boundary::Incoming),
            chrono_k1.col_storage_in_start() + h,
            "K = 1 endpoint k = 0 resolves to storage_in[h]"
        );
        assert_eq!(
            chrono_k1.block_storage_col(HydroSys::new(h), Boundary::Outgoing),
            h,
            "K = 1 endpoint k = K = 1 resolves to storage[h] = h"
        );
    }
}

/// The water-balance block spans `n_h` rows in parallel mode and `n_h * n_blks`
/// in chronological mode (the `K` chained per-hydro rows), with `K = 1`
/// chronological collapsing to the parallel count. `load_balance.start` chains off
/// `water_balance.end` in every case.
#[test]
fn chronological_water_balance_row_count() {
    let fixtures = TwoHydroFixtures::new();
    let ctx = fixtures.make_ctx();
    let state = state_layout_for(&ctx);

    let stage_parallel = stage_with_blocks(BlockMode::Parallel, 3);
    let parallel = StageLayout::new(&ctx, &state, &stage_parallel, 0);
    assert_eq!(
        parallel.rows.water_balance.end - parallel.rows.water_balance.start,
        2,
        "parallel n_h=2 n_blks=3 water_balance spans n_h = 2 rows"
    );
    assert_eq!(
        parallel.rows.load_balance.start, parallel.rows.water_balance.end,
        "parallel load_balance.start chains off water_balance.end"
    );

    let stage_chrono_k3 = stage_with_blocks(BlockMode::Chronological, 3);
    let chrono_k3 = StageLayout::new(&ctx, &state, &stage_chrono_k3, 0);
    assert_eq!(
        chrono_k3.rows.water_balance.end - chrono_k3.rows.water_balance.start,
        6,
        "chronological n_h=2 n_blks=3 water_balance spans n_h * n_blks = 6 rows"
    );
    assert_eq!(
        chrono_k3.rows.load_balance.start, chrono_k3.rows.water_balance.end,
        "chronological K=3 load_balance.start chains off water_balance.end"
    );

    let stage_chrono_k1 = stage_with_blocks(BlockMode::Chronological, 1);
    let chrono_k1 = StageLayout::new(&ctx, &state, &stage_chrono_k1, 0);
    assert_eq!(
        chrono_k1.rows.water_balance.end - chrono_k1.rows.water_balance.start,
        2,
        "chronological n_h=2 n_blks=1 water_balance spans n_h = 2 rows, identical to parallel"
    );
    assert_eq!(
        chrono_k1.rows.load_balance.start, chrono_k1.rows.water_balance.end,
        "chronological K=1 load_balance.start chains off water_balance.end"
    );
}

// ── FPHA-local inverse map ───────────────────────────────────────────────

/// Owns the data needed to construct a three-hydro `TemplateBuildCtx` with a
/// single FPHA hydro at system index 1 (the other two use constant
/// productivity), so `StageLayout::new` derives `fpha_hydro_indices == [1]`.
struct FphaMixFixtures {
    par_lp: PrecomputedPar,
    hydros: Vec<Hydro>,
    hydro_cell_index: HydroCellIndex,
    cascade: CascadeTopology,
    bounds: ResolvedBounds,
    penalties: ResolvedPenalties,
    resolved_generic_bounds: ResolvedGenericConstraintBounds,
    resolved_load_factors: ResolvedLoadFactors,
    resolved_ncs_bounds: ResolvedNcsBounds,
    resolved_ncs_factors: ResolvedNcsFactors,
    resolved_parameters: ResolvedParameters,
    production_models: ProductionModelSet,
    evaporation_models: EvaporationModelSet,
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
        Self {
            par_lp: PrecomputedPar::default(),
            hydros,
            hydro_cell_index,
            cascade,
            bounds: ResolvedBounds::empty(),
            penalties: ResolvedPenalties::empty(),
            resolved_generic_bounds: ResolvedGenericConstraintBounds::empty(),
            resolved_load_factors: ResolvedLoadFactors::empty(),
            resolved_ncs_bounds: ResolvedNcsBounds::empty(),
            resolved_ncs_factors: ResolvedNcsFactors::empty(),
            resolved_parameters: ResolvedParameters {
                per_param: vec![],
                id_to_slot: vec![],
                cost_scale_factor: 1_000_000.0,
            },
            production_models: ProductionModelSet::new(models, 3, 1),
            evaporation_models: EvaporationModelSet::new(vec![
                EvaporationModel::None,
                EvaporationModel::None,
                EvaporationModel::None,
            ]),
        }
    }

    fn make_ctx(&self) -> TemplateBuildCtx<'_> {
        TemplateBuildCtx {
            hydros: &self.hydros,
            thermals: &[],
            lines: &[],
            buses: &[],
            load_models: &[],
            cascade: &self.cascade,
            hydro_cell_index: &self.hydro_cell_index,
            resolved: ResolvedTables {
                bounds: &self.bounds,
                penalties: &self.penalties,
                resolved_generic_bounds: &self.resolved_generic_bounds,
                resolved_load_factors: &self.resolved_load_factors,
                resolved_ncs_bounds: &self.resolved_ncs_bounds,
                resolved_ncs_factors: &self.resolved_ncs_factors,
                resolved_parameters: &self.resolved_parameters,
            },
            hydro_pos: BTreeMap::new(),
            thermal_pos: BTreeMap::new(),
            line_pos: BTreeMap::new(),
            bus_pos: BTreeMap::new(),
            par_lp: &self.par_lp,
            production_models: &self.production_models,
            evaporation_models: &self.evaporation_models,
            generic_constraints: &[],
            non_controllable_sources: &[],
            pumping_stations: &[],
            pumping_pos: BTreeMap::new(),
            n_pumping: 0,
            contracts: &[],
            contract_pos: BTreeMap::new(),
            n_contract_import: 0,
            n_contract_export: 0,
            diversion_upstream: HashMap::new(),
            arc_stage_weights: HashMap::new(),
            arc_spread_chrono: HashMap::new(),
            arc_arrival_density: HashMap::new(),
            per_stage_mask: Vec::new(),
            post_study_resolved: PostStudyResolved::default(),
            n_hydros: 3,
            n_thermals: 0,
            n_lines: 0,
            n_buses: 0,
            max_par_order: 0,
            n_anticipated: 0,
            k_max: 0,
            anticipated_lead_stages: vec![],
            anticipated_thermal_indices: vec![],
            anticipated_windows: vec![],
            anticipated_resolution: AnticipatedResolution::default(),
            study_stage_ids: vec![],
            delivery_stage_ids: vec![],
            has_penalty: false,
            time_value: TimeValue::new(vec![1.0]),
            delivery_total_hours: vec![744.0],
            filling_v_target: BTreeMap::new(),
        }
    }
}

/// `StageLayout::new` inverts `fpha_hydro_indices` into `fpha_local_index`:
/// the FPHA hydro at system index 1 of three maps to local index 0, and the
/// two non-FPHA hydros stay `None`, giving `[None, Some(0), None]`.
#[test]
fn stage_layout_populates_fpha_local_index_inverse_map() {
    let fixtures = FphaMixFixtures::new();
    let ctx = fixtures.make_ctx();
    let stage = minimal_stage();
    let state = state_layout_for(&ctx);
    let layout = StageLayout::new(&ctx, &state, &stage, 0);

    assert_eq!(
        layout.fpha_hydro_indices,
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
    par_lp: PrecomputedPar,
    hydros: Vec<Hydro>,
    hydro_cell_index: HydroCellIndex,
    cascade: CascadeTopology,
    bounds: ResolvedBounds,
    penalties: ResolvedPenalties,
    resolved_generic_bounds: ResolvedGenericConstraintBounds,
    resolved_load_factors: ResolvedLoadFactors,
    resolved_ncs_bounds: ResolvedNcsBounds,
    resolved_ncs_factors: ResolvedNcsFactors,
    resolved_parameters: ResolvedParameters,
    production_models: ProductionModelSet,
    evaporation_models: EvaporationModelSet,
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

        Self {
            par_lp: PrecomputedPar::default(),
            hydros,
            hydro_cell_index,
            cascade,
            bounds: ResolvedBounds::empty(),
            penalties: ResolvedPenalties::empty(),
            resolved_generic_bounds: ResolvedGenericConstraintBounds::empty(),
            resolved_load_factors: ResolvedLoadFactors::empty(),
            resolved_ncs_bounds: ResolvedNcsBounds::empty(),
            resolved_ncs_factors: ResolvedNcsFactors::empty(),
            resolved_parameters: ResolvedParameters {
                per_param: vec![],
                id_to_slot: vec![],
                cost_scale_factor: 1_000_000.0,
            },
            production_models: ProductionModelSet::new(models, 2, 1),
            evaporation_models,
        }
    }

    fn make_ctx(&self) -> TemplateBuildCtx<'_> {
        TemplateBuildCtx {
            hydros: &self.hydros,
            thermals: &[],
            lines: &[],
            buses: &[],
            load_models: &[],
            cascade: &self.cascade,
            hydro_cell_index: &self.hydro_cell_index,
            resolved: ResolvedTables {
                bounds: &self.bounds,
                penalties: &self.penalties,
                resolved_generic_bounds: &self.resolved_generic_bounds,
                resolved_load_factors: &self.resolved_load_factors,
                resolved_ncs_bounds: &self.resolved_ncs_bounds,
                resolved_ncs_factors: &self.resolved_ncs_factors,
                resolved_parameters: &self.resolved_parameters,
            },
            hydro_pos: BTreeMap::new(),
            thermal_pos: BTreeMap::new(),
            line_pos: BTreeMap::new(),
            bus_pos: BTreeMap::new(),
            par_lp: &self.par_lp,
            production_models: &self.production_models,
            evaporation_models: &self.evaporation_models,
            generic_constraints: &[],
            non_controllable_sources: &[],
            pumping_stations: &[],
            pumping_pos: BTreeMap::new(),
            n_pumping: 0,
            contracts: &[],
            contract_pos: BTreeMap::new(),
            n_contract_import: 0,
            n_contract_export: 0,
            diversion_upstream: HashMap::new(),
            arc_stage_weights: HashMap::new(),
            arc_spread_chrono: HashMap::new(),
            arc_arrival_density: HashMap::new(),
            per_stage_mask: Vec::new(),
            post_study_resolved: PostStudyResolved::default(),
            n_hydros: 2,
            n_thermals: 0,
            n_lines: 0,
            n_buses: 0,
            max_par_order: 0,
            n_anticipated: 0,
            k_max: 0,
            anticipated_lead_stages: vec![],
            anticipated_thermal_indices: vec![],
            anticipated_windows: vec![],
            anticipated_resolution: AnticipatedResolution::default(),
            study_stage_ids: vec![],
            delivery_stage_ids: vec![],
            has_penalty: false,
            time_value: TimeValue::new(vec![1.0]),
            delivery_total_hours: vec![744.0],
            filling_v_target: BTreeMap::new(),
        }
    }

    /// `fpha_hydro_indices` for a stage built at `stage_id` (`stage_idx` held
    /// at 0 so the single FPHA/evaporation model row serves every phase).
    fn fpha_indices_at(&self, stage_id: i32) -> Vec<HydroSys> {
        let ctx = self.make_ctx();
        let stage = stage_with_id(stage_id);
        let state = state_layout_for(&ctx);
        StageLayout::new(&ctx, &state, &stage, 0).fpha_hydro_indices
    }

    /// `evap_hydro_indices` for a stage built at `stage_id`.
    fn evap_indices_at(&self, stage_id: i32) -> Vec<HydroSys> {
        let ctx = self.make_ctx();
        let stage = stage_with_id(stage_id);
        let state = state_layout_for(&ctx);
        StageLayout::new(&ctx, &state, &stage, 0).evap_hydro_indices
    }

    /// `filling_target_hydro_indices` for a stage built at `stage_id`.
    fn filling_target_indices_at(&self, stage_id: i32) -> Vec<HydroSys> {
        let ctx = self.make_ctx();
        let stage = stage_with_id(stage_id);
        let state = state_layout_for(&ctx);
        StageLayout::new(&ctx, &state, &stage, 0)
            .filling
            .filling_target_hydro_indices
    }

    /// `filled_min_storage_floor_hydro_indices` for a stage built at `stage_id`.
    fn filled_min_storage_floor_indices_at(&self, stage_id: i32) -> Vec<HydroSys> {
        let ctx = self.make_ctx();
        let stage = stage_with_id(stage_id);
        let state = state_layout_for(&ctx);
        StageLayout::new(&ctx, &state, &stage, 0)
            .filling
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
    let fixtures = FillingMembershipFixtures::new();

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
    let fixtures = FphaMixFixtures::new();
    let layout_at = |stage_id: i32| {
        let ctx = fixtures.make_ctx();
        let stage = stage_with_id(stage_id);
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, 0);
        (
            layout.filling.filling_target_hydro_indices.clone(),
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
    let fixtures = FillingMembershipFixtures::new();
    let ctx = fixtures.make_ctx();
    let stage = stage_with_id(2); // entry − 1: the terminal stage.
    let state = state_layout_for(&ctx);
    let layout = StageLayout::new(&ctx, &state, &stage, 0);

    let n_targets = layout.filling.filling_target_hydro_indices.len();
    assert_eq!(n_targets, 2, "both filling hydros carry the target at id 2");

    let row_start = layout.filling.row_filling_target_start;
    for local_idx in 0..n_targets {
        assert!(
            row_start + local_idx < layout.rows.num_rows,
            "σ_fill row {} must be < num_rows {}",
            row_start + local_idx,
            layout.rows.num_rows
        );
    }
    assert_eq!(
        row_start, layout.slack.oper_violation.min_generation_rows.end,
        "σ_fill rows follow the operational-violation rows directly"
    );

    let col_start = layout.filling.col_filling_target_start;
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
        layout.filling.col_filled_min_storage_floor_start, layout.num_cols,
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
    let fixtures = FillingMembershipFixtures::new();
    // PreFilling (id 0) and Operating (id 3, 4): the σ_fill TARGET adds no rows.
    for stage_id in [0, 3, 4] {
        let ctx = fixtures.make_ctx();
        let stage = stage_with_id(stage_id);
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, 0);
        assert!(
            layout.filling.filling_target_hydro_indices.is_empty(),
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
    let fixtures = FillingMembershipFixtures::new();

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
    let fixtures = FphaMixFixtures::new();
    let layout_at = |stage_id: i32| {
        let ctx = fixtures.make_ctx();
        let stage = stage_with_id(stage_id);
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, 0);
        (
            layout
                .filling
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
    let fixtures = FillingMembershipFixtures::new();

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
    let fixtures = FillingMembershipFixtures::new();

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
    let fixtures = FphaMixFixtures::new();
    let (reference_fpha, reference_evap) = {
        let ctx = fixtures.make_ctx();
        let stage = stage_with_id(0);
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, 0);
        (layout.fpha_hydro_indices, layout.evap_hydro_indices)
    };

    assert_eq!(reference_fpha, vec![HydroSys::new(1)]);
    assert_eq!(reference_evap, Vec::<HydroSys>::new());

    for stage_id in [1, 2, 3, 7] {
        let ctx = fixtures.make_ctx();
        let stage = stage_with_id(stage_id);
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, 0);
        assert_eq!(
            layout.fpha_hydro_indices, reference_fpha,
            "non-filling fpha_hydro_indices must be stage-invariant (stage_id {stage_id})"
        );
        assert_eq!(
            layout.evap_hydro_indices, reference_evap,
            "non-filling evap_hydro_indices must be stage-invariant (stage_id {stage_id})"
        );
    }
}

// ── Operational-violation row ranges ─────────────────────────────────────

/// The four operational-violation row families (`min_outflow_rows`,
/// `max_outflow_rows`, `min_turbine_rows`, `min_generation_rows`) are
/// contiguous, in that order, each spanning exactly `n_h * n_blks` rows, and
/// the block starts immediately after the post-equipment row cursor
/// (`evap_rows_end`), which equals `min_outflow_rows.start`. The owning
/// arithmetic lives in [`StageLayout::new`]; this pins it at the internal
/// layer where the row ranges are visible. The forbidden alternative is a
/// stale or transposed base — placing `max_outflow` before `min_outflow`, or
/// striding any family by something other than `n_h * n_blks`, addresses the
/// wrong constraint rows and silently mis-bounds the operational violations.
///
/// Fixture: `n_h = 3`, one block (`n_blks = 1`), so `n_op = 3` per family.
#[test]
fn stage_layout_operational_violation_rows_are_contiguous_blocks() {
    let fixtures = FphaMixFixtures::new();
    let ctx = fixtures.make_ctx();
    let stage = minimal_stage();
    let state = state_layout_for(&ctx);
    let layout = StageLayout::new(&ctx, &state, &stage, 0);

    let n_op = ctx.n_hydros; // n_h * n_blks with n_blks == 1
    assert!(
        n_op > 0,
        "fixture must have hydros so the rows are non-empty"
    );

    assert_eq!(
        layout.slack.oper_violation.min_outflow_rows.len(),
        n_op,
        "min_outflow row count"
    );
    assert_eq!(
        layout.slack.oper_violation.max_outflow_rows.len(),
        n_op,
        "max_outflow row count"
    );
    assert_eq!(
        layout.slack.oper_violation.min_turbine_rows.len(),
        n_op,
        "min_turbine row count"
    );
    assert_eq!(
        layout.slack.oper_violation.min_generation_rows.len(),
        n_op,
        "min_generation row count"
    );

    assert_eq!(
        layout.slack.oper_violation.max_outflow_rows.start,
        layout.slack.oper_violation.min_outflow_rows.end,
        "max_outflow must follow min_outflow contiguously"
    );
    assert_eq!(
        layout.slack.oper_violation.min_turbine_rows.start,
        layout.slack.oper_violation.max_outflow_rows.end,
        "min_turbine must follow max_outflow contiguously"
    );
    assert_eq!(
        layout.slack.oper_violation.min_generation_rows.start,
        layout.slack.oper_violation.max_outflow_rows.end + n_op,
        "min_generation must start one min_turbine block (n_op rows) after max_outflow ends"
    );
}

// ── Anticipated-decision column positioning ──────────────────────────────

/// `col_anticipated_decision_start` falls between thermal end and
/// `col_line_fwd_start` when `n_anticipated=2, n_thermals=3, n_blks=4`.
///
/// The control region is `thermal` then `anticipated_decision` (2 cols) then
/// `line_fwd` — the anticipated ring's outgoing slots live entirely in the
/// state region. So `col_line_fwd_start` equals
/// `col_anticipated_decision_start + n_anticipated`, and
/// `col_anticipated_slots_out_start` is sourced from the state-region
/// position (immediately after `transit_buckets_out`), not from the control
/// region.
#[test]
fn anticipated_decision_columns_placed_between_thermal_and_line_fwd() {
    let fixtures = ZeroEntityFixtures::new();
    // ZeroEntityFixtures builds n_thermals=0, so the thermal per-block block is
    // empty and col_anticipated_decision_start == col_thermal_start.
    let n_anticipated = 2_usize;
    let k_max = 1_usize;
    let ctx = fixtures.make_ctx(n_anticipated, k_max, vec![1, 1], vec![0, 0]);

    let mut stage = minimal_stage();
    stage.blocks = (0..4)
        .map(|index| Block {
            index,
            name: format!("B{index}"),
            duration_hours: 186.0,
        })
        .collect();
    let state = state_layout_for(&ctx);
    let layout = StageLayout::new(&ctx, &state, &stage, 0);

    assert_eq!(
        layout.anticipated.col_anticipated_decision_start, layout.equipment.thermal.start,
        "col_anticipated_decision_start must equal col_thermal_start \
             when n_thermals=0 (no thermal per-block cols)"
    );
    assert_eq!(
        layout.equipment.line_fwd.start,
        layout.anticipated.col_anticipated_decision_start + n_anticipated,
        "col_line_fwd_start == col_anticipated_decision_start + n_anticipated \
             (state_out relocated out of the control region)"
    );
    // The outgoing ring start equals the indexer's state-region position:
    // immediately after `transit_buckets_out` (N*(1+L) + B). Here N=0, L=0,
    // B=0 → the ring starts at 0.
    assert_eq!(
        layout.anticipated.col_anticipated_slots_out_start, 0,
        "col_anticipated_slots_out_start must equal the state-region offset \
             N*(1+L) + B"
    );
    assert_eq!(
        layout.equipment.line_fwd.start - layout.equipment.thermal.start,
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

    let fixtures = ZeroEntityFixtures::new();
    let ctx = fixtures.make_ctx(
        n_anticipated,
        k_max,
        vec![2, 3], // anticipated_lead_stages
        vec![0, 2], // anticipated_thermal_indices (arbitrary; layout doesn't inspect them)
    );
    let stage = minimal_stage();
    let state = state_layout_for(&ctx);
    let layout = StageLayout::new(&ctx, &state, &stage, 0);

    let expected_n_ant_state = n_anticipated * k_max;
    assert_eq!(layout.n_ant_state, expected_n_ant_state, "n_ant_state");

    let expected_col_turbine_start = n_hydros * (3 + max_par_order) + 2 * expected_n_ant_state + 1;
    assert_eq!(
        layout.equipment.turbine.start, expected_col_turbine_start,
        "col_turbine_start == N*(3+L) + 2*B + 2*n_ant_state + 1"
    );
}

// ── Anticipated-fishing row positioning ──────────────────────────────────

/// `row_anticipated_fishing_start` immediately follows the operational
/// violation row block, i.e. equals `row_min_generation_start + n_op_rows`.
///
/// Uses a zero-hydro context so `n_op_rows == 0`, which means the fishing
/// start equals `row_min_generation_start` exactly. The algebraic identity
/// `row_anticipated_fishing_start == row_min_generation_start + n_op_rows`
/// is verified for the general formula; the case `n_op_rows > 0` is covered by
/// the production code path (`n_hydros * n_blks` counts operational violation rows).
///
/// Setup: `n_anticipated=2`, `k_max=2`, `anticipated_lead_stages=[1,2]`,
/// zero hydros, one block, `n_stages=4` (`AntFixturesWithNStages`, not the
/// study-stage-count-0 `ZeroEntityFixtures`: `build_anticipated_fishing_row_pos`'s
/// in-study guard reads the fixture's own `n_stages`, so a `stage_idx=1` probe
/// needs a real study-stage count to stay in-study). At `stage_idx=1`:
/// - `n_op_rows = 0 * 1 = 0` (no hydros)
/// - `row_anticipated_fishing_start` must equal `row_min_generation_start + 0`
#[test]
fn anticipated_fishing_row_offset_after_operational_violations() {
    let n_anticipated = 2_usize;
    let k_max = 2_usize;

    let fixtures = AntFixturesWithNStages::new(4);
    let ctx = fixtures.make_ctx(
        n_anticipated,
        k_max,
        vec![1, 2], // K_0=1, K_1=2
        vec![0, 1], // arbitrary thermal indices
    );
    let stage = minimal_stage(); // 1 block
    let state = state_layout_for(&ctx);
    let layout = StageLayout::new(&ctx, &state, &stage, 1);

    // n_op_rows = n_hydros * n_blks = 0 * 1 = 0
    let n_op_rows = 0_usize;
    assert_eq!(
        layout.anticipated.row_anticipated_fishing_start,
        layout.slack.oper_violation.min_generation_rows.start + n_op_rows,
        "row_anticipated_fishing_start must equal row_min_generation_start + n_op_rows"
    );
    assert_eq!(
        layout.anticipated.n_anticipated_fishing_rows, 2,
        "n_anticipated_fishing_rows must equal n_anticipated (2) under always-active predicate"
    );
}

/// `n_anticipated_fishing_rows` equals `n_anticipated` at every stage under
/// the always-active predicate. With `K_i=[1,2]` and `n_anticipated=2`, the
/// count is 2 at every stage in `[0, 1, 2, 3]`. `n_stages=4` covers the
/// probed range (`AntFixturesWithNStages`, not the study-stage-count-0
/// `ZeroEntityFixtures` — see the sibling test above for why).
#[test]
fn anticipated_fishing_row_count_grows_with_stage() {
    let n_anticipated = 2_usize;
    let k_max = 2_usize;

    let fixtures = AntFixturesWithNStages::new(4);
    let ctx = fixtures.make_ctx(
        n_anticipated,
        k_max,
        vec![1, 2], // K_0=1, K_1=2
        vec![0, 1], // arbitrary thermal indices
    );
    let stage = minimal_stage(); // 1 block
    let state = state_layout_for(&ctx);

    for (stage_idx, expected) in [(0_usize, 2), (1, 2), (2, 2), (3, 2)] {
        let layout = StageLayout::new(&ctx, &state, &stage, stage_idx);
        assert_eq!(
            layout.anticipated.n_anticipated_fishing_rows, expected,
            "n_anticipated_fishing_rows must equal {expected} at stage_idx={stage_idx}"
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

    let fixtures = AntFixturesWithNStages::new(1);
    let ctx = fixtures.make_ctx(n_anticipated, k_max, vec![3, 2], vec![0, 1]);
    let stage = minimal_stage();
    let state = state_layout_for(&ctx);
    let layout = StageLayout::new(&ctx, &state, &stage, 0);

    // n_state for this fixture: N*(1+L) + A*K = 0 + 2*3 = 6.
    let n_state = ctx.n_hydros * (1 + ctx.max_par_order) + n_anticipated * k_max;
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
    // equals ctx.n_hydros (no n_state offset). With state-fixing rows it
    // would be n_state + ctx.n_hydros.
    assert_eq!(
        layout.rows.water_balance.start, ctx.n_hydros,
        "row_water_balance_start does not include the n_state offset"
    );
}

// ── Delivery-axis generalization: row positions & masking ────────────────

/// Build a one-plant `StateSpace` carrying an attached delivery-anchored
/// `AnticipatedResolution` directly — never the constant-lead fallback
/// `anticipated_resolution_for` falls back to when none is attached
/// (`state_layout_for` leaves that fallback active; these tests need the
/// real resolved axis instead).
fn state_with_attached_resolution(k_max: usize, resolution: AnticipatedResolution) -> StateSpace {
    let n_anticipated = resolution.per_plant.len();
    let mut state = StateSpace::new(
        0,
        0,
        0,
        Vec::new(),
        n_anticipated,
        k_max,
        vec![k_max; n_anticipated],
        &[],
    );
    state.set_anticipated_resolution(resolution);
    state
}

/// Study-only axis (`delivery_stage_count(n_stages) == n_stages`): every
/// returned position must be byte-identical to the pre-generalization
/// `m >= n_stages` skip. `k_max = 2`, `LeadTime::Stages(2)` over 3 study
/// stages, `stage_idx = 1`. Depths `[2, 3]` reach delivery targets `m = [2,
/// 3]`: `m=2` (`decider[2] = Some(0)`) is ready at stage 1 and not a fresh
/// deposit there, so slot `2 % 2 = 0` is an interior carry; `m=3 >=
/// n_delivery(3)` is masked. Hand-written, not recomputed, so a regression in
/// the residue arithmetic is caught rather than reproduced.
#[test]
fn build_anticipated_slot_row_pos_study_only_byte_identity() {
    let k_max = 2;
    let resolution = AnticipatedResolution::resolve(
        &[LeadTime::Stages(2)],
        DeliveryAxis {
            stage_lengths_hours: &[],
            n_decision: 3,
            n_delivery: 3,
        },
    );
    assert_eq!(resolution.k_max, k_max, "fixture must realize k_max == 2");
    let state = state_with_attached_resolution(k_max, resolution);

    let (row_pos, n_reachable) = build_anticipated_slot_row_pos(&state, 3, 1);

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
            stage_lengths_hours: &[],
            n_decision: 4,
            n_delivery: 8,
        },
    );
    assert_eq!(resolution.k_max, k_max, "fixture must realize k_max == 4");
    let state = state_with_attached_resolution(k_max, resolution);

    let (row_pos, _n_reachable) = build_anticipated_slot_row_pos(&state, 4, 2);

    let slot = 5 % k_max;
    assert!(
        row_pos[slot].is_some(),
        "post-study delivery target m=5 (slot {slot}) must carry as an interior \
         position, not be masked by the retired study-horizon bound"
    );
}

// ── Ring-axis excision: per-plant physical-target mapping ────────────────

/// Hand-derived pre-excision reference: `n_anticipated = 2`, `k_max = 3`,
/// leads `[1, 2]`, 5 study stages, no attached resolution — the fallback
/// `anticipated_resolution_for` resolves an identity delivery axis
/// (`n_decision == n_delivery`, so `g == 0` for both plants), matching the
/// retired raw-delivery-axis sweep exactly. Pinned per `stage_idx` as a
/// literal, not recomputed by the ring-axis formula under test.
#[test]
fn anticipated_slot_row_pos_identity_axis_matches_the_recorded_pre_excision_mapping() {
    let state = StateSpace::new(0, 0, 0, Vec::new(), 2, 3, vec![1, 2], &[]);

    let expected: [(Vec<Option<usize>>, usize); 5] = [
        (vec![None, None, None, Some(0), None, None], 1),
        (vec![None, None, None, None, None, Some(0)], 1),
        (vec![None, Some(0), None, None, None, None], 1),
        (vec![None, None, None, Some(0), None, None], 1),
        (vec![None, None, None, None, None, None], 0),
    ];

    for (stage_idx, (expected_row_pos, expected_n_reachable)) in expected.into_iter().enumerate() {
        let (row_pos, n_reachable) = build_anticipated_slot_row_pos(&state, 5, stage_idx);
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
        k_max: 4,
        max_fanout: 1,
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

    let (row_pos, n_reachable) = build_anticipated_slot_row_pos(&state, 4, 0);

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
        let (row_pos, _n_reachable) = build_anticipated_slot_row_pos(&state, 4, stage_idx);
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
        k_max: 3,
        max_fanout: 1,
    };
    let state = state_with_attached_resolution(3, resolution);

    let (row_pos, n_reachable) = build_anticipated_slot_row_pos(&state, 2, 1);

    assert_eq!(
        row_pos,
        vec![None, Some(2), None, None, Some(0), Some(1)],
        "plant A's slot at depth=1 (physical target 4, at n_delivery=4) must be \
         masked while plant B's own physical target (3) is still classified"
    );
    assert_eq!(n_reachable, 3);
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

    let study_only_resolution = AnticipatedResolution::resolve(
        &[lead],
        DeliveryAxis {
            stage_lengths_hours: &[],
            n_decision: n_stages,
            n_delivery: n_stages,
        },
    );
    let extended_resolution = AnticipatedResolution::resolve(
        &[lead],
        DeliveryAxis {
            stage_lengths_hours: &[],
            n_decision: n_stages,
            n_delivery: 2 * n_stages,
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
/// in `n_anticipated_state_out_def_rows` without needing real entity data.
fn bounds_with_n_stages(n_stages: usize) -> ResolvedBounds {
    bounds_with_pumping(0, n_stages)
}

/// Builds a fixture struct owning all data for a context with anticipated
/// thermals and a known `n_stages` for the `state_out_def` predicate.
struct AntFixturesWithNStages {
    par_lp: PrecomputedPar,
    cascade: CascadeTopology,
    hydro_cell_index: HydroCellIndex,
    bounds: ResolvedBounds,
    penalties: ResolvedPenalties,
    resolved_generic_bounds: ResolvedGenericConstraintBounds,
    resolved_load_factors: ResolvedLoadFactors,
    resolved_ncs_bounds: ResolvedNcsBounds,
    resolved_ncs_factors: ResolvedNcsFactors,
    resolved_parameters: ResolvedParameters,
    production_models: ProductionModelSet,
    evaporation_models: EvaporationModelSet,
}

impl AntFixturesWithNStages {
    fn new(n_stages: usize) -> Self {
        Self {
            par_lp: PrecomputedPar::default(),
            cascade: CascadeTopology::build(&[]),
            hydro_cell_index: HydroCellIndex::build(&[]),
            bounds: bounds_with_n_stages(n_stages),
            penalties: ResolvedPenalties::empty(),
            resolved_generic_bounds: ResolvedGenericConstraintBounds::empty(),
            resolved_load_factors: ResolvedLoadFactors::empty(),
            resolved_ncs_bounds: ResolvedNcsBounds::empty(),
            resolved_ncs_factors: ResolvedNcsFactors::empty(),
            resolved_parameters: ResolvedParameters {
                per_param: vec![],
                id_to_slot: vec![],
                cost_scale_factor: 1_000_000.0,
            },
            production_models: ProductionModelSet::new(vec![], 0, 1),
            evaporation_models: EvaporationModelSet::new(vec![]),
        }
    }

    fn make_ctx(
        &self,
        n_anticipated: usize,
        k_max: usize,
        anticipated_lead_stages: Vec<usize>,
        anticipated_thermal_indices: Vec<usize>,
    ) -> TemplateBuildCtx<'_> {
        let n_stages = self.bounds.n_stages();
        TemplateBuildCtx {
            hydros: &[],
            thermals: &[],
            lines: &[],
            buses: &[],
            load_models: &[],
            cascade: &self.cascade,
            hydro_cell_index: &self.hydro_cell_index,
            resolved: ResolvedTables {
                bounds: &self.bounds,
                penalties: &self.penalties,
                resolved_generic_bounds: &self.resolved_generic_bounds,
                resolved_load_factors: &self.resolved_load_factors,
                resolved_ncs_bounds: &self.resolved_ncs_bounds,
                resolved_ncs_factors: &self.resolved_ncs_factors,
                resolved_parameters: &self.resolved_parameters,
            },
            hydro_pos: BTreeMap::new(),
            thermal_pos: BTreeMap::new(),
            line_pos: BTreeMap::new(),
            bus_pos: BTreeMap::new(),
            par_lp: &self.par_lp,
            production_models: &self.production_models,
            evaporation_models: &self.evaporation_models,
            generic_constraints: &[],
            non_controllable_sources: &[],
            pumping_stations: &[],
            pumping_pos: BTreeMap::new(),
            n_pumping: 0,
            contracts: &[],
            contract_pos: BTreeMap::new(),
            n_contract_import: 0,
            n_contract_export: 0,
            diversion_upstream: HashMap::new(),
            arc_stage_weights: HashMap::new(),
            arc_spread_chrono: HashMap::new(),
            arc_arrival_density: HashMap::new(),
            per_stage_mask: Vec::new(),
            post_study_resolved: PostStudyResolved::default(),
            n_hydros: 0,
            n_thermals: 0,
            n_lines: 0,
            n_buses: 0,
            max_par_order: 0,
            n_anticipated,
            k_max,
            anticipated_lead_stages,
            anticipated_thermal_indices: anticipated_thermal_indices
                .into_iter()
                .map(ThermalSys::new)
                .collect(),
            // Windowless: one `(None, None)` per plant, so the decision gate
            // reduces to the strict horizon clause. `study_stage_ids` covers
            // the study-stage count so the in-range delivery lookup is safe.
            anticipated_windows: vec![(None, None); n_anticipated],
            anticipated_resolution: AnticipatedResolution::default(),
            study_stage_ids: (0..i32::try_from(n_stages).unwrap_or(0)).collect(),
            delivery_stage_ids: (0..i32::try_from(n_stages).unwrap_or(0)).collect(),
            has_penalty: false,
            time_value: TimeValue::new(vec![1.0; n_stages]),
            delivery_total_hours: vec![744.0; n_stages],
            filling_v_target: BTreeMap::new(),
        }
    }
}

/// `col_anticipated_slots_out_start` is sourced from the state-region
/// position immediately after `transit_buckets_out` (before `z_inflow`),
/// `col_line_fwd_start` follows `anticipated_decision` directly, and
/// `n_anticipated_state_out_def_rows` counts both active plants at stage 0.
///
/// Fixture: `n_anticipated=2`, `K=[2,3]`, `k_max=3`, `n_stages=6`,
/// `stage_idx=0`, `N=0`, `L=0`, `B=0`. Both plants are active: `0+2=2 < 6` and
/// `0+3=3 < 6`. State-region offset = `N*(1+L) + B = 0`.
#[test]
fn test_layout_state_out_block_adjacent_to_decision() {
    let fixtures = AntFixturesWithNStages::new(6);
    let ctx = fixtures.make_ctx(
        2,          // n_anticipated
        3,          // k_max
        vec![2, 3], // K_0=2, K_1=3
        vec![0, 1],
    );
    let stage = minimal_stage();
    let state = state_layout_for(&ctx);
    let layout = StageLayout::new(&ctx, &state, &stage, 0);

    // The outgoing ring sits in the state region: N*(1+L) + B.
    assert_eq!(
        layout.anticipated.col_anticipated_slots_out_start, 0,
        "outgoing-ring columns must be sourced from the state-region offset \
             N*(1+L) + B"
    );
    assert_eq!(
        layout.equipment.line_fwd.start,
        layout.anticipated.col_anticipated_decision_start + 2,
        "line_fwd must be immediately after the anticipated_decision block"
    );
    assert_eq!(layout.anticipated.n_anticipated_state_out_def_rows, 2);
    assert_eq!(
        layout.anticipated.row_anticipated_state_out_def_start,
        layout.anticipated.row_anticipated_fishing_start
            + layout.anticipated.n_anticipated_fishing_rows
    );
}

/// `n_anticipated_state_out_def_rows == 0` when all plants are inactive at
/// the given stage, but the column block stays allocated.
///
/// Fixture: `n_anticipated=2`, `K=[2,3]`, `n_stages=6`, `stage_idx=5`.
/// Both inactive: `5+2=7 >= 6` and `5+3=8 >= 6`.
#[test]
fn test_layout_state_out_def_rows_zero_when_all_inactive() {
    let fixtures = AntFixturesWithNStages::new(6);
    let ctx = fixtures.make_ctx(
        2,          // n_anticipated
        3,          // k_max
        vec![2, 3], // K_0=2, K_1=3
        vec![0, 1],
    );
    let stage = minimal_stage();
    let state = state_layout_for(&ctx);
    let layout = StageLayout::new(&ctx, &state, &stage, 5);

    assert_eq!(layout.anticipated.n_anticipated_state_out_def_rows, 0);
    // Column block stays allocated at the state-region offset regardless of
    // activity: N*(1+L) + B = 0.
    assert_eq!(layout.anticipated.col_anticipated_slots_out_start, 0);
}

/// Zero-anticipated layouts must not grow `num_cols` or emit def rows.
///
/// `col_anticipated_slots_out_start` must equal `col_anticipated_decision_start`
/// when `n_anticipated == 0` (empty block; both starts coincide).
#[test]
fn test_layout_no_anticipated_unchanged_num_cols() {
    let fixtures = ZeroEntityFixtures::new();
    let ctx = fixtures.make_ctx(0, 0, vec![], vec![]);
    let stage = minimal_stage();
    let state = state_layout_for(&ctx);
    let layout = StageLayout::new(&ctx, &state, &stage, 0);

    assert_eq!(
        layout.anticipated.col_anticipated_slots_out_start,
        layout.anticipated.col_anticipated_decision_start,
        "col_anticipated_slots_out_start must equal col_anticipated_decision_start when n_anticipated=0"
    );
    assert_eq!(layout.anticipated.n_anticipated_state_out_def_rows, 0);
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
    par_lp: PrecomputedPar,
    cascade: CascadeTopology,
    hydro_cell_index: HydroCellIndex,
    bounds: ResolvedBounds,
    penalties: ResolvedPenalties,
    resolved_generic_bounds: ResolvedGenericConstraintBounds,
    resolved_load_factors: ResolvedLoadFactors,
    resolved_ncs_bounds: ResolvedNcsBounds,
    resolved_ncs_factors: ResolvedNcsFactors,
    resolved_parameters: ResolvedParameters,
    production_models: ProductionModelSet,
    evaporation_models: EvaporationModelSet,
    /// Windowless stations (always commissioning-active); the
    /// column-reservation probe exercises the dense per-station arithmetic the
    /// production builder runs.
    stations: Vec<PumpingStation>,
}

impl PumpingFixtures {
    fn new(n_pumping: usize, n_stages: usize) -> Self {
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
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
            par_lp: PrecomputedPar::default(),
            cascade: CascadeTopology::build(&[]),
            hydro_cell_index: HydroCellIndex::build(&[]),
            bounds: bounds_with_pumping(n_pumping, n_stages),
            penalties: ResolvedPenalties::empty(),
            resolved_generic_bounds: ResolvedGenericConstraintBounds::empty(),
            resolved_load_factors: ResolvedLoadFactors::empty(),
            resolved_ncs_bounds: ResolvedNcsBounds::empty(),
            resolved_ncs_factors: ResolvedNcsFactors::empty(),
            resolved_parameters: ResolvedParameters {
                per_param: vec![],
                id_to_slot: vec![],
                cost_scale_factor: 1_000_000.0,
            },
            production_models: ProductionModelSet::new(vec![], 0, 1),
            evaporation_models: EvaporationModelSet::new(vec![]),
            stations,
        }
    }

    fn make_ctx(&self) -> TemplateBuildCtx<'_> {
        let n_stages = self.bounds.n_stages();
        TemplateBuildCtx {
            hydros: &[],
            thermals: &[],
            lines: &[],
            buses: &[],
            load_models: &[],
            cascade: &self.cascade,
            hydro_cell_index: &self.hydro_cell_index,
            resolved: ResolvedTables {
                bounds: &self.bounds,
                penalties: &self.penalties,
                resolved_generic_bounds: &self.resolved_generic_bounds,
                resolved_load_factors: &self.resolved_load_factors,
                resolved_ncs_bounds: &self.resolved_ncs_bounds,
                resolved_ncs_factors: &self.resolved_ncs_factors,
                resolved_parameters: &self.resolved_parameters,
            },
            hydro_pos: BTreeMap::new(),
            thermal_pos: BTreeMap::new(),
            line_pos: BTreeMap::new(),
            bus_pos: BTreeMap::new(),
            par_lp: &self.par_lp,
            production_models: &self.production_models,
            evaporation_models: &self.evaporation_models,
            generic_constraints: &[],
            non_controllable_sources: &[],
            // The slice/`pumping_pos` threading is covered by the
            // `build_template_build_ctx` tests in `template.rs`.
            pumping_stations: &self.stations,
            pumping_pos: BTreeMap::new(),
            n_pumping: self.bounds.n_pumping(),
            contracts: &[],
            contract_pos: BTreeMap::new(),
            n_contract_import: 0,
            n_contract_export: 0,
            diversion_upstream: HashMap::new(),
            arc_stage_weights: HashMap::new(),
            arc_spread_chrono: HashMap::new(),
            arc_arrival_density: HashMap::new(),
            per_stage_mask: Vec::new(),
            post_study_resolved: PostStudyResolved::default(),
            n_hydros: 0,
            n_thermals: 0,
            n_lines: 0,
            n_buses: 0,
            max_par_order: 0,
            n_anticipated: 0,
            k_max: 0,
            anticipated_lead_stages: vec![],
            anticipated_thermal_indices: vec![],
            anticipated_windows: vec![],
            anticipated_resolution: AnticipatedResolution::default(),
            study_stage_ids: vec![],
            delivery_stage_ids: vec![],
            has_penalty: false,
            time_value: TimeValue::new(vec![1.0; n_stages]),
            delivery_total_hours: vec![744.0; n_stages],
            filling_v_target: BTreeMap::new(),
        }
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
    let fixtures = ZeroEntityFixtures::new();
    let ctx = fixtures.make_ctx(0, 0, vec![], vec![]);
    let stage = minimal_stage();
    let state = state_layout_for(&ctx);
    let layout = StageLayout::new(&ctx, &state, &stage, 0);

    assert_eq!(
        ctx.resolved.bounds.n_pumping(),
        0,
        "fixture has no pumping stations"
    );
    assert_eq!(layout.equipment.n_pumping, 0, "layout.n_pumping must be 0");

    // The empty pumping block does not advance the cursor: its start equals
    // the NCS-region end. With zero active NCS, col_ncs_end == col_ncs_start.
    assert_eq!(
        layout.equipment.col_pumping_start, layout.equipment.col_ncs_start,
        "col_pumping_start must equal col_ncs_start (col_ncs_end) when no stations"
    );

    // Pre-existing column starts for the zero-entity, single-block layout:
    // theta == 0, every equipment/slack/NCS region empty starting at theta+1.
    let idx = state_layout(ctx.n_hydros, ctx.max_par_order);
    let expected_start = idx.theta + 1;
    assert_eq!(layout.equipment.turbine.start, expected_start);
    assert_eq!(layout.equipment.thermal.start, expected_start);
    assert_eq!(layout.equipment.line_fwd.start, expected_start);
    assert_eq!(layout.equipment.deficit.start, expected_start);
    assert_eq!(layout.equipment.excess.start, expected_start);
    assert_eq!(layout.equipment.col_ncs_start, expected_start);
    assert_eq!(layout.equipment.col_pumping_start, expected_start);
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

    let baseline_fixtures = PumpingFixtures::new(0, 3);
    let baseline_ctx = baseline_fixtures.make_ctx();
    let stage = PumpingFixtures::stage_with_blocks(n_blks);
    let state = state_layout_for(&baseline_ctx);
    let baseline = StageLayout::new(&baseline_ctx, &state, &stage, 0);
    assert_eq!(baseline.equipment.n_pumping, 0);

    let fixtures = PumpingFixtures::new(n_pumping, 3);
    let ctx = fixtures.make_ctx();
    assert_eq!(
        ctx.resolved.bounds.n_pumping(),
        n_pumping,
        "fixture bounds must report n_pumping() == 2"
    );
    let state = state_layout_for(&ctx);
    let layout = StageLayout::new(&ctx, &state, &stage, 0);

    assert_eq!(
        layout.equipment.n_pumping, n_pumping,
        "layout.n_pumping == 2"
    );
    assert_eq!(
        layout.equipment.col_pumping_start, layout.equipment.col_ncs_start,
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
        layout.equipment.col_pumping_start + n_pumping * n_blks,
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
    let fixtures = PumpingFixtures::new(n_pumping, 3);
    let ctx = fixtures.make_ctx();
    assert_eq!(ctx.n_contract_import, 0);
    assert_eq!(ctx.n_contract_export, 0);

    let stage = PumpingFixtures::stage_with_blocks(n_blks);
    let state = state_layout_for(&ctx);
    let layout = StageLayout::new(&ctx, &state, &stage, 0);

    let col_pumping_end = layout.equipment.col_pumping_start + layout.equipment.n_pumping * n_blks;
    assert_eq!(
        layout.equipment.col_contract_import_start, col_pumping_end,
        "empty import block starts at col_pumping_end"
    );
    assert_eq!(
        layout.equipment.col_contract_export_start, col_pumping_end,
        "empty export block collapses onto col_pumping_end"
    );
    assert_eq!(
        layout.filling.col_filling_target_start, col_pumping_end,
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
    let fixtures = PumpingFixtures::new(n_pumping, 3);
    let ctx = TemplateBuildCtx {
        n_contract_import: 2,
        n_contract_export: 1,
        ..fixtures.make_ctx()
    };

    let stage = PumpingFixtures::stage_with_blocks(n_blks);
    let state = state_layout_for(&ctx);
    let layout = StageLayout::new(&ctx, &state, &stage, 0);

    let col_pumping_end = layout.equipment.col_pumping_start + layout.equipment.n_pumping * n_blks;
    assert_eq!(
        layout.equipment.col_contract_import_start, col_pumping_end,
        "import block starts at col_pumping_end"
    );
    assert_eq!(
        layout.equipment.col_contract_export_start,
        col_pumping_end + 6,
        "export block follows the 6-column import block"
    );
    assert_eq!(
        layout.filling.col_filling_target_start,
        col_pumping_end + 9,
        "generic-slack start shifts by (2 + 1) * 3 == 9"
    );
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

// ── block-major column-accessor arithmetic equivalence ─────────────────────

/// Every `#[inline]` column accessor returns the exact `usize` its open-coded
/// formula returned. Built over a `n_blks = 4` layout and probed with
/// `entity >= 1`, `blk >= 1`, and `seg >= 1` so a transposed stride
/// (`blk * n_entities + entity`) or a swapped evap offset would differ from
/// the open-coded expression and fail the assertion. `turbine_col` and
/// `generation_col` address a CELL (`HydroCell`/`FphaCellLocal`), not a plant
/// (`HydroSys`/`FphaLocal`) — the arithmetic `block_col` delegates to is
/// unchanged, only the meaning of `entity` for these two accessors is.
#[test]
fn column_accessors_match_open_coded_formulas() {
    // Multi-block, zero-entity layout: the block-major `col_*_start` fields
    // and `n_blks` are populated; the accessor reads the same fields the
    // open-coded formula reads, so this pins each accessor's arithmetic.
    // Chronological so `n_evap_slots == n_blks`, giving the evaporation probe
    // below the same multi-slot coverage as every other block-major family.
    let fixtures = ZeroEntityFixtures::new();
    let ctx = fixtures.make_ctx(0, 0, vec![], vec![]);
    let stage = Stage {
        block_mode: BlockMode::Chronological,
        ..PumpingFixtures::stage_with_blocks(4)
    };
    let state = state_layout_for(&ctx);
    let layout = StageLayout::new(&ctx, &state, &stage, 0);
    let n_blks = layout.n_blks;
    assert_eq!(n_blks, 4, "fixture must build a 4-block layout");

    // Generic block_col against its definition, across a grid that makes the
    // entity (outer) and block (inner) factors distinguishable.
    for entity in [0_usize, 1, 2, 5] {
        for blk in 0..n_blks {
            assert_eq!(
                layout.block_col(layout.equipment.turbine.start, entity, BlockIdx::new(blk)),
                layout.equipment.turbine.start + entity * n_blks + blk,
                "block_col(entity={entity}, blk={blk})"
            );
        }
    }

    for entity in [0_usize, 1, 3] {
        for blk in 0..n_blks {
            assert_eq!(
                layout.turbine_col(HydroCell::new(entity), BlockIdx::new(blk)),
                layout.equipment.turbine.start + entity * n_blks + blk,
                "turbine_col"
            );
            assert_eq!(
                layout.spillage_col(HydroSys::new(entity), BlockIdx::new(blk)),
                layout.equipment.spillage.start + entity * n_blks + blk,
                "spillage_col"
            );
            assert_eq!(
                layout.diversion_col(HydroSys::new(entity), BlockIdx::new(blk)),
                layout.equipment.diversion.start + entity * n_blks + blk,
                "diversion_col"
            );
            assert_eq!(
                layout.generation_col(FphaCellLocal::new(entity), BlockIdx::new(blk)),
                layout.equipment.generation_col_start + entity * n_blks + blk,
                "generation_col"
            );
            assert_eq!(
                layout.line_fwd_col(LineSys::new(entity), BlockIdx::new(blk)),
                layout.equipment.line_fwd.start + entity * n_blks + blk,
                "line_fwd_col"
            );
            assert_eq!(
                layout.line_rev_col(LineSys::new(entity), BlockIdx::new(blk)),
                layout.equipment.line_rev.start + entity * n_blks + blk,
                "line_rev_col"
            );
            assert_eq!(
                layout.outflow_below_col(HydroSys::new(entity), BlockIdx::new(blk)),
                layout.slack.oper_violation.outflow_below_slack.start + entity * n_blks + blk,
                "outflow_below_col"
            );
            assert_eq!(
                layout.outflow_above_col(HydroSys::new(entity), BlockIdx::new(blk)),
                layout.slack.oper_violation.outflow_above_slack.start + entity * n_blks + blk,
                "outflow_above_col"
            );
            assert_eq!(
                layout.turbine_below_col(HydroCell::new(entity), BlockIdx::new(blk)),
                layout.slack.oper_violation.turbine_below_slack.start + entity * n_blks + blk,
                "turbine_below_col"
            );
            assert_eq!(
                layout.generation_below_col(HydroCell::new(entity), BlockIdx::new(blk)),
                layout.slack.oper_violation.generation_below_slack.start + entity * n_blks + blk,
                "generation_below_col"
            );
        }
    }

    // Evaporation accessors: slot-major (`local * n_evap_slots + slot`) triple,
    // EVAP_COLS_PER_HYDRO-strided (chronological, so `n_evap_slots == n_blks`).
    // The three within-triple offsets must map flow→0, f_plus→1, f_minus→2.
    let n_evap_slots = layout.n_evap_slots;
    assert_eq!(
        n_evap_slots, n_blks,
        "chronological: one evap slot per block"
    );
    for local_idx in [0_usize, 1, 4] {
        let local = EvapLocal::new(local_idx);
        for blk in 0..n_blks {
            let triple_base = layout.equipment.evap_col_start
                + (local_idx * n_evap_slots + blk) * EVAP_COLS_PER_HYDRO;
            assert_eq!(
                layout.evap_flow_col(local, BlockIdx::new(blk)),
                triple_base + EVAP_FLOW_OFFSET,
                "evap_flow_col"
            );
            assert_eq!(
                layout.evap_f_plus_col(local, BlockIdx::new(blk)),
                triple_base + EVAP_F_PLUS_OFFSET,
                "evap_f_plus_col"
            );
            assert_eq!(
                layout.evap_f_minus_col(local, BlockIdx::new(blk)),
                triple_base + EVAP_F_MINUS_OFFSET,
                "evap_f_minus_col"
            );
            // The three columns are consecutive and ordered flow < plus < minus.
            assert_eq!(
                layout.evap_f_plus_col(local, BlockIdx::new(blk)),
                layout.evap_flow_col(local, BlockIdx::new(blk)) + 1
            );
            assert_eq!(
                layout.evap_f_minus_col(local, BlockIdx::new(blk)),
                layout.evap_flow_col(local, BlockIdx::new(blk)) + 2
            );
        }
    }
    assert_eq!(EVAP_FLOW_OFFSET, 0);
    assert_eq!(EVAP_F_PLUS_OFFSET, 1);
    assert_eq!(EVAP_F_MINUS_OFFSET, 2);
    assert_eq!(EVAP_COLS_PER_HYDRO, 3);

    // Deficit three-term stride: per-bus span, per-segment span, then block.
    for b_idx in [0_usize, 1, 2] {
        for seg_idx in [0_usize, 1] {
            for blk in 0..n_blks {
                assert_eq!(
                    layout.deficit_col(b_idx, seg_idx, BlockIdx::new(blk)),
                    layout.equipment.deficit.start
                        + b_idx * layout.equipment.max_deficit_segments * n_blks
                        + seg_idx * n_blks
                        + blk,
                    "deficit_col(b={b_idx}, seg={seg_idx}, blk={blk})"
                );
            }
        }
    }
}

// ── post-equipment column cursor (no-hydro fork fallback) ───────────────────

/// With `n_hydros == 0` every withdrawal / operational-violation / NCS column
/// region is empty, so `RangeCursor::alloc(0)` leaves all their starts at the
/// single post-equipment column cursor `col_evap_start`. A multi-block stage
/// keeps that cursor non-trivial (not the degenerate one-column case), so a
/// hand-computed offset that reintroduces a `0..0` empty-range fallback would
/// shift these starts off `col_evap_start` and fail here.
#[test]
fn post_equipment_col_start_matches_evap_col_start_when_no_hydros() {
    let fixtures = ZeroEntityFixtures::new();
    let ctx = fixtures.make_ctx(0, 0, vec![], vec![]);
    let stage = PumpingFixtures::stage_with_blocks(4);
    let state = state_layout_for(&ctx);
    let layout = StageLayout::new(&ctx, &state, &stage, 0);

    assert_eq!(ctx.n_hydros, 0, "fixture must have zero hydros");
    assert_eq!(layout.n_blks, 4, "fixture must build a 4-block layout");

    let post_equipment = layout.equipment.evap_col_start;
    assert_eq!(
        layout.equipment.col_ncs_start, post_equipment,
        "col_ncs_start must collapse onto col_evap_start when n_hydros == 0"
    );
    assert_eq!(
        layout.slack.withdrawal_slack_neg.start, post_equipment,
        "col_withdrawal_neg_start"
    );
    assert_eq!(
        layout.slack.withdrawal_slack_pos.start, post_equipment,
        "col_withdrawal_pos_start"
    );
    assert_eq!(
        layout.slack.oper_violation.outflow_below_slack.start, post_equipment,
        "col_outflow_below_start"
    );
    assert_eq!(
        layout.slack.oper_violation.outflow_above_slack.start, post_equipment,
        "col_outflow_above_start"
    );
    assert_eq!(
        layout.slack.oper_violation.turbine_below_slack.start, post_equipment,
        "col_turbine_below_start"
    );
    assert_eq!(
        layout.slack.oper_violation.generation_below_slack.start, post_equipment,
        "col_generation_below_start"
    );
}

// ── post-equipment row cursor (no-hydro fork fallback) ──────────────────────

/// With `n_hydros == 0` every operational-violation row block is empty, so
/// `RangeCursor::alloc(0)` leaves all four row starts at the single
/// post-equipment row cursor `fpha_rows_end() + n_evap_hydros`. A multi-block
/// stage keeps that cursor non-trivial (not the degenerate one-row case), so a
/// hand-computed offset that reintroduces a `0..0` empty-range fallback would
/// shift these starts off the shared post-equipment row cursor and fail here.
#[test]
fn post_equipment_row_start_matches_evap_rows_end_when_no_hydros() {
    let fixtures = ZeroEntityFixtures::new();
    let ctx = fixtures.make_ctx(0, 0, vec![], vec![]);
    let stage = PumpingFixtures::stage_with_blocks(4);
    let state = state_layout_for(&ctx);
    let layout = StageLayout::new(&ctx, &state, &stage, 0);

    assert_eq!(ctx.n_hydros, 0, "fixture must have zero hydros");
    assert_eq!(layout.n_blks, 4, "fixture must build a 4-block layout");

    let post_equipment = layout.rows.post_equipment_row_start;
    assert_eq!(
        layout.slack.oper_violation.min_outflow_rows.start, post_equipment,
        "row_min_outflow_start must collapse onto the post-equipment row cursor when n_hydros == 0"
    );
    assert_eq!(
        layout.slack.oper_violation.max_outflow_rows.start, post_equipment,
        "row_max_outflow_start"
    );
    assert_eq!(
        layout.slack.oper_violation.min_turbine_rows.start, post_equipment,
        "row_min_turbine_start"
    );
    assert_eq!(
        layout.slack.oper_violation.min_generation_rows.start, post_equipment,
        "row_min_generation_start"
    );
}

// ── Group-2 accessors: hydro-free divergence guard ──────────────────────────

/// With `n_hydros == 0`, every Group-2 accessor must return the post-equipment
/// cursor — `post_equipment_col_start` for the eight column accessors,
/// `post_equipment_row_start` for the five row accessors. Each accessor is a
/// bare `self.<range>.start`/`.end`, correct only because `StageLayout::new`
/// allocates every one of these families through `RangeCursor::alloc`:
/// `alloc(0)` returns `pos..pos`, so an empty family's `.start` already equals
/// the post-equipment cursor. A hand-computed offset that reintroduces a
/// `0..0` fallback (losing the cursor position) would fail these equality
/// assertions.
///
/// The column cursor is additionally asserted `!= 0`: the theta and state columns
/// always precede the equipment/slack region, so `post_equipment_col_start` is
/// provably positive and a spurious `0` is directly detectable. The row cursor is
/// NOT asserted `!= 0`: with zero hydros AND zero buses no rows precede the
/// operational-violation block, so `post_equipment_row_start` is legitimately `0`
/// here (asserting `!= 0` would test a false invariant). The non-zero-row
/// divergence is covered end-to-end by the D01 hydro-free parity case, whose
/// load-balance rows make the row cursor positive.
#[test]
fn group2_accessors_return_post_equipment_cursor_when_no_hydros() {
    let fixtures = ZeroEntityFixtures::new();
    let ctx = fixtures.make_ctx(0, 0, vec![], vec![]);
    let stage = PumpingFixtures::stage_with_blocks(4);
    let state = state_layout_for(&ctx);
    let layout = StageLayout::new(&ctx, &state, &stage, 0);

    assert_eq!(ctx.n_hydros, 0, "fixture must have zero hydros");
    assert_eq!(layout.n_blks, 4, "fixture must build a 4-block layout");

    // Column cursor: the eight column accessors collapse onto
    // `post_equipment_col_start` (== `col_evap_start()`) with no hydros, and
    // that cursor is provably positive (theta + state columns precede it).
    let post_col = layout.equipment.post_equipment_col_start;
    assert_ne!(post_col, 0, "post-equipment column cursor must not be 0");
    for (value, name) in [
        (
            layout.equipment.generation_col_start,
            "col_generation_start",
        ),
        (layout.equipment.evap_col_start, "col_evap_start"),
        (
            layout.slack.withdrawal_slack_neg.start,
            "col_withdrawal_neg_start",
        ),
        (
            layout.slack.withdrawal_slack_pos.start,
            "col_withdrawal_pos_start",
        ),
        (
            layout.slack.oper_violation.outflow_below_slack.start,
            "col_outflow_below_start",
        ),
        (
            layout.slack.oper_violation.outflow_above_slack.start,
            "col_outflow_above_start",
        ),
        (
            layout.slack.oper_violation.turbine_below_slack.start,
            "col_turbine_below_start",
        ),
        (
            layout.slack.oper_violation.generation_below_slack.start,
            "col_generation_below_start",
        ),
    ] {
        assert_eq!(
            value, post_col,
            "{name} must equal post_equipment_col_start() (not 0) when n_hydros == 0"
        );
    }

    // Row cursor: `row_evap_start()` is `fpha_rows_end`, which equals
    // `post_equipment_row_start` (= `fpha_rows_end + n_evap_hydros`) when
    // `n_evap_hydros == 0`. The four operational-violation row accessors collapse
    // onto that same cursor. Each must equal it, never a bare `.start`.
    let post_row = layout.rows.post_equipment_row_start;
    for (value, name) in [
        (layout.row_evap_start(), "row_evap_start"),
        (
            layout.slack.oper_violation.min_outflow_rows.start,
            "row_min_outflow_start",
        ),
        (
            layout.slack.oper_violation.max_outflow_rows.start,
            "row_max_outflow_start",
        ),
        (
            layout.slack.oper_violation.min_turbine_rows.start,
            "row_min_turbine_start",
        ),
        (
            layout.slack.oper_violation.min_generation_rows.start,
            "row_min_generation_start",
        ),
    ] {
        assert_eq!(
            value, post_row,
            "{name} must equal post_equipment_row_start() when n_hydros == 0"
        );
    }
}

/// Consumption side of `setup::bucket_topology`'s
/// `test_horizon_cap_drops_lag_targeting_past_last_stage` (that test pins the
/// MASK — `per_stage_mask == [2, 1, 0]` for a depth-3 plant over 3
/// stages; this pins that `build_transit_bucket_row_pos` actually gates row emission
/// from it): stage 0 already drops the deepest lag (cap = 2), stage 1 drops
/// the two deepest (cap = 1), and stage 2 (the terminal stage, cap = 0) drops
/// all three.
#[test]
fn build_bucket_row_pos_gates_fewer_rows_as_horizon_cap_shrinks() {
    let column_order = vec![(0_usize, 1_usize), (0, 2), (0, 3)];
    let per_stage_mask = vec![vec![2], vec![1], vec![0]];

    let (pos_stage0, n_stage0) = build_transit_bucket_row_pos(&column_order, &per_stage_mask, 0);
    assert_eq!(
        pos_stage0,
        vec![Some(0), Some(1), None],
        "stage 0: cap = 2, lags 1 and 2 keep a row, lag 3 does not"
    );
    assert_eq!(n_stage0, 2);

    let (pos_stage1, n_stage1) = build_transit_bucket_row_pos(&column_order, &per_stage_mask, 1);
    assert_eq!(
        pos_stage1,
        vec![Some(0), None, None],
        "stage 1: cap = 1, only lag 1 keeps a row"
    );
    assert_eq!(n_stage1, 1);

    let (pos_stage2, n_stage2) = build_transit_bucket_row_pos(&column_order, &per_stage_mask, 2);
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
    let (pos, n) = build_transit_bucket_row_pos(&[], &[], 0);
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
    par_lp: PrecomputedPar,
    hydros: Vec<Hydro>,
    hydro_cell_index: HydroCellIndex,
    cascade: CascadeTopology,
    bounds: ResolvedBounds,
    penalties: ResolvedPenalties,
    resolved_generic_bounds: ResolvedGenericConstraintBounds,
    resolved_load_factors: ResolvedLoadFactors,
    resolved_ncs_bounds: ResolvedNcsBounds,
    resolved_ncs_factors: ResolvedNcsFactors,
    resolved_parameters: ResolvedParameters,
    production_models: ProductionModelSet,
    evaporation_models: EvaporationModelSet,
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
        Self {
            par_lp: PrecomputedPar::default(),
            hydros,
            hydro_cell_index,
            cascade,
            bounds: ResolvedBounds::empty(),
            penalties: ResolvedPenalties::empty(),
            resolved_generic_bounds: ResolvedGenericConstraintBounds::empty(),
            resolved_load_factors: ResolvedLoadFactors::empty(),
            resolved_ncs_bounds: ResolvedNcsBounds::empty(),
            resolved_ncs_factors: ResolvedNcsFactors::empty(),
            resolved_parameters: ResolvedParameters {
                per_param: vec![],
                id_to_slot: vec![],
                cost_scale_factor: 1_000_000.0,
            },
            production_models: ProductionModelSet::new(models, 2, 1),
            evaporation_models: EvaporationModelSet::new(vec![
                EvaporationModel::None,
                EvaporationModel::None,
            ]),
        }
    }

    fn make_ctx(&self) -> TemplateBuildCtx<'_> {
        TemplateBuildCtx {
            hydros: &self.hydros,
            thermals: &[],
            lines: &[],
            buses: &[],
            load_models: &[],
            cascade: &self.cascade,
            hydro_cell_index: &self.hydro_cell_index,
            resolved: ResolvedTables {
                bounds: &self.bounds,
                penalties: &self.penalties,
                resolved_generic_bounds: &self.resolved_generic_bounds,
                resolved_load_factors: &self.resolved_load_factors,
                resolved_ncs_bounds: &self.resolved_ncs_bounds,
                resolved_ncs_factors: &self.resolved_ncs_factors,
                resolved_parameters: &self.resolved_parameters,
            },
            hydro_pos: BTreeMap::new(),
            thermal_pos: BTreeMap::new(),
            line_pos: BTreeMap::new(),
            bus_pos: BTreeMap::new(),
            par_lp: &self.par_lp,
            production_models: &self.production_models,
            evaporation_models: &self.evaporation_models,
            generic_constraints: &[],
            non_controllable_sources: &[],
            pumping_stations: &[],
            pumping_pos: BTreeMap::new(),
            n_pumping: 0,
            contracts: &[],
            contract_pos: BTreeMap::new(),
            n_contract_import: 0,
            n_contract_export: 0,
            diversion_upstream: HashMap::new(),
            arc_stage_weights: HashMap::new(),
            arc_spread_chrono: HashMap::new(),
            arc_arrival_density: HashMap::new(),
            per_stage_mask: Vec::new(),
            post_study_resolved: PostStudyResolved::default(),
            n_hydros: 2,
            n_thermals: 0,
            n_lines: 0,
            n_buses: 0,
            max_par_order: 0,
            n_anticipated: 0,
            k_max: 0,
            anticipated_lead_stages: vec![],
            anticipated_thermal_indices: vec![],
            anticipated_windows: vec![],
            anticipated_resolution: AnticipatedResolution::default(),
            study_stage_ids: vec![],
            delivery_stage_ids: vec![],
            has_penalty: false,
            time_value: TimeValue::new(vec![1.0]),
            delivery_total_hours: vec![744.0],
            filling_v_target: BTreeMap::new(),
        }
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

    let split_fixtures = TwoHydroMultiBusFixtures::new(true);
    let split_ctx = split_fixtures.make_ctx();
    assert_eq!(split_ctx.hydro_cell_index.n_cells(), 3);
    let split_state = state_layout_for(&split_ctx);
    let split_layout = StageLayout::new(&split_ctx, &split_state, &stage, 0);
    assert_eq!(
        split_layout.equipment.turbine.len(),
        9,
        "3 cells * 3 blocks"
    );
    assert_eq!(
        split_layout.equipment.spillage.start, split_layout.equipment.turbine.end,
        "spillage follows turbine directly, with no gap"
    );

    let same_bus_fixtures = TwoHydroMultiBusFixtures::new(false);
    let same_bus_ctx = same_bus_fixtures.make_ctx();
    assert_eq!(same_bus_ctx.hydro_cell_index.n_cells(), 2);
    let same_bus_state = state_layout_for(&same_bus_ctx);
    let same_bus_layout = StageLayout::new(&same_bus_ctx, &same_bus_state, &stage, 0);
    assert_eq!(
        same_bus_layout.equipment.turbine.len(),
        6,
        "2 cells * 3 blocks under the identity"
    );
    assert_eq!(
        same_bus_layout.equipment.spillage.start,
        split_layout.equipment.spillage.start - n_blks,
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
    let fixtures = TwoHydroMultiBusFixtures::new(true);
    let ctx = fixtures.make_ctx();
    assert_eq!(
        ctx.hydro_cell_index.cells_of(HydroSys::new(1)),
        1..3,
        "plant 1 owns cells 1 and 2"
    );

    let stage = stage_with_blocks(BlockMode::Parallel, n_blks);
    let state = state_layout_for(&ctx);
    let layout = StageLayout::new(&ctx, &state, &stage, 0);

    let mut columns = Vec::with_capacity(9);
    for cell in 0..3 {
        for blk in 0..n_blks {
            let col = layout.turbine_col(HydroCell::new(cell), BlockIdx::new(blk));
            assert!(
                layout.equipment.turbine.contains(&col),
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
        let cell1 = layout.turbine_col(HydroCell::new(1), BlockIdx::new(blk));
        let cell2 = layout.turbine_col(HydroCell::new(2), BlockIdx::new(blk));
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
    let asserted_col = layout.turbine_col(HydroCell::new(cell_idx), BlockIdx::new(block_idx));
    assert_eq!(
        asserted_col,
        layout.equipment.turbine.start + cell_idx * n_blks + block_idx
    );
    assert_ne!(
        asserted_col,
        layout.turbine_col(HydroCell::new(hydro_idx), BlockIdx::new(block_idx)),
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
    par_lp: PrecomputedPar,
    hydros: Vec<Hydro>,
    hydro_cell_index: HydroCellIndex,
    cascade: CascadeTopology,
    bounds: ResolvedBounds,
    penalties: ResolvedPenalties,
    resolved_generic_bounds: ResolvedGenericConstraintBounds,
    resolved_load_factors: ResolvedLoadFactors,
    resolved_ncs_bounds: ResolvedNcsBounds,
    resolved_ncs_factors: ResolvedNcsFactors,
    resolved_parameters: ResolvedParameters,
    production_models: ProductionModelSet,
    evaporation_models: EvaporationModelSet,
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

        Self {
            par_lp: PrecomputedPar::default(),
            hydros,
            hydro_cell_index,
            cascade,
            bounds: ResolvedBounds::empty(),
            penalties: ResolvedPenalties::empty(),
            resolved_generic_bounds: ResolvedGenericConstraintBounds::empty(),
            resolved_load_factors: ResolvedLoadFactors::empty(),
            resolved_ncs_bounds: ResolvedNcsBounds::empty(),
            resolved_ncs_factors: ResolvedNcsFactors::empty(),
            resolved_parameters: ResolvedParameters {
                per_param: vec![],
                id_to_slot: vec![],
                cost_scale_factor: 1_000_000.0,
            },
            production_models: ProductionModelSet::new(models, 3, 1),
            evaporation_models: EvaporationModelSet::new(vec![
                EvaporationModel::None,
                EvaporationModel::None,
                EvaporationModel::None,
            ]),
        }
    }

    fn make_ctx(&self) -> TemplateBuildCtx<'_> {
        TemplateBuildCtx {
            hydros: &self.hydros,
            thermals: &[],
            lines: &[],
            buses: &[],
            load_models: &[],
            cascade: &self.cascade,
            hydro_cell_index: &self.hydro_cell_index,
            resolved: ResolvedTables {
                bounds: &self.bounds,
                penalties: &self.penalties,
                resolved_generic_bounds: &self.resolved_generic_bounds,
                resolved_load_factors: &self.resolved_load_factors,
                resolved_ncs_bounds: &self.resolved_ncs_bounds,
                resolved_ncs_factors: &self.resolved_ncs_factors,
                resolved_parameters: &self.resolved_parameters,
            },
            hydro_pos: BTreeMap::new(),
            thermal_pos: BTreeMap::new(),
            line_pos: BTreeMap::new(),
            bus_pos: BTreeMap::new(),
            par_lp: &self.par_lp,
            production_models: &self.production_models,
            evaporation_models: &self.evaporation_models,
            generic_constraints: &[],
            non_controllable_sources: &[],
            pumping_stations: &[],
            pumping_pos: BTreeMap::new(),
            n_pumping: 0,
            contracts: &[],
            contract_pos: BTreeMap::new(),
            n_contract_import: 0,
            n_contract_export: 0,
            diversion_upstream: HashMap::new(),
            arc_stage_weights: HashMap::new(),
            arc_spread_chrono: HashMap::new(),
            arc_arrival_density: HashMap::new(),
            per_stage_mask: Vec::new(),
            post_study_resolved: PostStudyResolved::default(),
            n_hydros: 3,
            n_thermals: 0,
            n_lines: 0,
            n_buses: 0,
            max_par_order: 0,
            n_anticipated: 0,
            k_max: 0,
            anticipated_lead_stages: vec![],
            anticipated_thermal_indices: vec![],
            anticipated_windows: vec![],
            anticipated_resolution: AnticipatedResolution::default(),
            study_stage_ids: vec![],
            delivery_stage_ids: vec![],
            has_penalty: false,
            time_value: TimeValue::new(vec![1.0]),
            delivery_total_hours: vec![744.0],
            filling_v_target: BTreeMap::new(),
        }
    }
}

/// The FPHA-generation family is sized by FPHA CELL, never FPHA plant.
/// `n_fpha_cells == 3` (plant 0's one cell + plant 2's two cells; non-FPHA
/// plant 1 contributes none, even though it owns a hydro-cell of its own).
#[test]
fn test_generation_family_is_sized_by_fpha_cell() {
    let n_blks = 2;
    let fixtures = FphaMultiBusFixtures::new();
    let ctx = fixtures.make_ctx();

    // The fixture's three index families genuinely diverge for plant 2:
    // HydroSys = 2, FphaLocal = 1 (below), and its two FphaCellLocal values
    // (1 and 2, asserted below) are neither uniformly equal to 2 nor to 1.
    assert_eq!(ctx.hydro_cell_index.cells_of(HydroSys::new(2)), 2..4);

    let stage = stage_with_blocks(BlockMode::Parallel, n_blks);
    let state = state_layout_for(&ctx);
    let layout = StageLayout::new(&ctx, &state, &stage, 0);

    assert_eq!(
        layout.fpha_hydro_indices,
        vec![HydroSys::new(0), HydroSys::new(2)],
        "plant 2 is FPHA-local index 1"
    );
    assert_eq!(
        layout.equipment.generation.len(),
        3 * n_blks,
        "n_fpha_cells == 3: plant 0's one cell + plant 2's two cells"
    );

    for blk in 0..n_blks {
        let col1 = layout.generation_col(FphaCellLocal::new(1), BlockIdx::new(blk));
        let col2 = layout.generation_col(FphaCellLocal::new(2), BlockIdx::new(blk));
        assert!(
            layout.equipment.generation.contains(&col2),
            "block {blk}: column {col2} must lie inside equipment.generation"
        );
        assert_ne!(
            col1, col2,
            "block {blk}: plant 2's two FPHA cells must be distinct columns"
        );
    }
}
