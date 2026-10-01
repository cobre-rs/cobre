//! Scenario distribution and result extraction for the SDDP simulation phase.
//!
//! ## Column layout
//!
//! The state-region column layout is defined by [`StateSpace`]:
//!
//! ```text
//! [0, N)             storage      — outgoing storage volumes
//! [N, N*(1+L))       inflow_lags  — AR lag variables (hydro-major order)
//! [N*(1+L), N*(2+L)) z_inflow     — realized inflow (auxiliary, not state)
//! [N*(2+L), N*(3+L)) storage_in   — incoming storage volumes (fixed vars)
//! N*(3+L)            theta        — future cost variable
//! [theta+1, ...)     equipment    — turbine, spillage, thermal, lines, deficit, excess
//! ```
//!
//! The equipment column layout is defined per stage by `StageLayout`, threaded
//! into extraction as [`StageGeometry`].

use std::collections::HashMap;
use std::ops::Range;

use chrono::NaiveDate;
use cobre_core::BlockMode;
use cobre_core::ContractType;
use cobre_core::EntityId;
use cobre_core::HydroPastDefluence;

use crate::energy_conversion::EnergyConversionSet;
use crate::horizon_mode::HorizonMode;
use crate::lp::builder::{
    GenericConstraintRowEntry, StageGeometry, evaporation_slot, evaporation_slot_count,
};
use crate::lp::indexer::{
    AnticipatedLocal, AnticipatedPlants, BlockIdx, Boundary, BusSys, EvapLocal, FillingTargetLocal,
    FloorLocal, FphaCellLocal, FphaLocal, HydroCell, HydroCellIndex, HydroSys, LineSys, NcsSys,
    PumpingSys, StateSpace, StudyDimensions, ThermalSys, anticipated_resolution_for,
    is_anticipated_decision_active_for_delivery,
};
use crate::setup::NodeId;
use crate::simulation::types::{
    ScenarioCategoryCosts, SimulationAnticipatedLaneResult, SimulationBusResult,
    SimulationContractResult, SimulationCostResult, SimulationExchangeResult,
    SimulationGenericViolationResult, SimulationHydroBusResult, SimulationHydroResult,
    SimulationInflowLagResult, SimulationNonControllableResult, SimulationPumpingResult,
    SimulationStageResult, SimulationThermalResult, SimulationTransitBucketResult,
    SimulationTransitSeedResult,
};

/// Reverse lookups from system hydro index to local FPHA/evaporation/filling-slack
/// slot, for **one stage**.
///
/// Membership is per-`(hydro, stage)`: a hydro can be FPHA at one stage and not
/// another, `σ_fill` exists only at a filling hydro's terminal Filling stage, and
/// `σ^{v-}` only at its Operating stages. A single global stage-0 list would
/// misclassify any stage whose membership differs. Each entry is `Some(slot)` /
/// `None`.
pub(crate) struct HydroReverseLookup {
    /// FPHA-local slot per hydro, `None` if not FPHA at this stage.
    pub(crate) fpha: Vec<Option<FphaLocal>>,
    /// FPHA-cell-local start per FPHA-local slot, parallel to
    /// `geometry.fpha_hydro_indices`: `fpha_cell_local_start[local]` is the
    /// position, among the cells of FPHA plants, of that FPHA-local plant's first
    /// cell. Mirrors `StageLayout`'s own same-named field; computed here (once per
    /// stage, alongside `fpha`) rather than read from `StageGeometry` so the O(1)
    /// per-`(scenario, block)` extraction read never re-derives it.
    pub(crate) fpha_cell_local_start: Vec<usize>,
    /// Evaporation-local slot per hydro, `None` if no evaporation at this stage.
    pub(crate) evap: Vec<Option<EvapLocal>>,
    /// `σ_fill`-target slot per hydro, `None` if it owns no target column at this stage.
    pub(crate) filling_target: Vec<Option<FillingTargetLocal>>,
    /// `σ^{v-}` operating-floor slot per hydro, `None` if it owns no floor column at this stage.
    pub(crate) filled_min_storage_floor: Vec<Option<FloorLocal>>,
}

/// Map each system hydro index in `indices` to its local slot via `make`;
/// `None` for a hydro not in `indices`.
fn build_reverse_slots<T: Copy>(
    n_hydros: usize,
    indices: &[HydroSys],
    make: impl Fn(usize) -> T,
) -> Vec<Option<T>> {
    let mut slots = vec![None; n_hydros];
    for (local, &sys) in indices.iter().enumerate() {
        slots[sys.get()] = Some(make(local));
    }
    slots
}

impl HydroReverseLookup {
    /// Build the reverse lookup for one stage from its [`StageGeometry`] and the
    /// study-scope [`HydroCellIndex`].
    pub(crate) fn build(
        geometry: &StageGeometry,
        hydro_cell_index: &HydroCellIndex,
        n_hydros: usize,
    ) -> Self {
        let mut fpha = vec![None; n_hydros];
        let mut fpha_cell_local_start = Vec::with_capacity(geometry.fpha_hydro_indices.len());
        let mut n_fpha_cells = 0_usize;
        for (local, &sys) in geometry.fpha_hydro_indices.iter().enumerate() {
            fpha[sys.get()] = Some(FphaLocal::new(local));
            fpha_cell_local_start.push(n_fpha_cells);
            n_fpha_cells += hydro_cell_index.cells_of(sys).len();
        }
        let evap = build_reverse_slots(n_hydros, &geometry.evap_hydro_indices, EvapLocal::new);
        let filling_target = build_reverse_slots(
            n_hydros,
            &geometry.filling_target_hydro_indices,
            FillingTargetLocal::new,
        );
        let filled_min_storage_floor = build_reverse_slots(
            n_hydros,
            &geometry.filled_min_storage_floor_hydro_indices,
            FloorLocal::new,
        );
        Self {
            fpha,
            fpha_cell_local_start,
            evap,
            filling_target,
            filled_min_storage_floor,
        }
    }

    /// Build one [`HydroReverseLookup`] per stage from the per-stage geometry table,
    /// once per simulation run so per-`(scenario, stage)` extraction never reallocates.
    pub(crate) fn build_per_stage(
        geometry_per_stage: &[StageGeometry],
        hydro_cell_index: &HydroCellIndex,
        n_hydros: usize,
    ) -> Vec<Self> {
        geometry_per_stage
            .iter()
            .map(|g| Self::build(g, hydro_cell_index, n_hydros))
            .collect()
    }
}

/// Read the primal of the sparse `σ_fill` filling-target-slack column, or `0.0`
/// when `local` is `None` (the hydro owns no column in that family at this stage).
#[inline]
fn read_filling_target_slack_primal(
    primal: &[f64],
    geometry: &StageGeometry,
    local: Option<FillingTargetLocal>,
) -> f64 {
    let Some(local) = local else { return 0.0 };
    let col = geometry.filling_target_slack_col(local);
    debug_assert!(
        col < primal.len(),
        "filling-slack col {col} out of primal len {}",
        primal.len(),
    );
    primal[col]
}

/// Read the primal of the sparse `σ^{v-}` operating-floor-slack column, or `0.0`
/// when `local` is `None` (the hydro owns no column in that family at this stage).
#[inline]
fn read_floor_slack_primal(
    primal: &[f64],
    geometry: &StageGeometry,
    local: Option<FloorLocal>,
) -> f64 {
    let Some(local) = local else { return 0.0 };
    let col = geometry.filled_min_storage_floor_slack_col(local);
    debug_assert!(
        col < primal.len(),
        "floor-slack col {col} out of primal len {}",
        primal.len(),
    );
    primal[col]
}

/// Primal of a thermal's anticipated-decision column, or `None` when the
/// thermal is not anticipated, has no in-study decision at this stage, or the
/// decision is inactive at its delivery stage.
///
/// The delivery stage comes from `anticipated_resolution_for`'s
/// delivery-anchored resolution (`PointResolution::genuine_decisions_at`),
/// never `stage_idx + lead_stages` — the constant-lead shortcut mis-resolves a
/// calendar-anchored lead on a non-uniform study. Only the single-decider case
/// is read: at most one genuine decision per plant per stage. A plant fanning
/// out several deliveries from one decision stage needs a per-delivery-stage
/// output extraction this helper does not provide, so `resolve_state_layout`
/// rejects `AnticipatedResolution::max_fanout > 1` at setup — a fanned study
/// never reaches here, and the `debug_assert` below is defence-in-depth against
/// that guard regressing, not a reachable panic.
///
/// Gates on `is_anticipated_decision_active_for_delivery` rather
/// than reading then checking: an inactive column is pinned to `[0, 0]` and the
/// predicate is the canonical single-owner test.
#[inline]
fn compute_anticipated_decision_mw(
    view: &SolutionView<'_>,
    spec: &StageExtractionSpec<'_>,
    anticipated_plants: &AnticipatedPlants,
    thermal_local: usize,
) -> Option<f64> {
    let local_idx = anticipated_plants.local_of(ThermalSys::new(thermal_local))?;
    let resolution = anticipated_resolution_for(spec.state, local_idx);
    let mut genuine = resolution.genuine_decisions_at(spec.stage_index);
    let delivery_stage = genuine.next()?;
    // TODO(anticipated-fanout-output): gated by resolve_state_layout's max_fanout > 1 reject
    debug_assert!(
        genuine.next().is_none(),
        "compute_anticipated_decision_mw reads a single delivery stage per plant \
         per stage; a fanned-out decision needs per-delivery-stage output \
         extraction, not implemented here"
    );
    if !is_anticipated_decision_active_for_delivery(
        local_idx,
        delivery_stage,
        spec.horizon.num_stages(),
        spec.anticipated_windows,
        spec.study_stage_ids,
    ) {
        return None;
    }
    // Base is the per-stage `thermal.end` (n_blks-dependent), so use `spec.geometry`,
    // never the global stage-0 indexer — that addresses the wrong column off stage 0.
    let col = spec.geometry.anticipated_decision_col(local_idx);
    debug_assert!(
        col < view.primal.len(),
        "anticipated_decision col {col} out of primal bounds {}",
        view.primal.len(),
    );
    Some(view.primal[col])
}

/// Committed MW for an anticipated thermal, or `None` when not anticipated.
///
/// The committed scalar is the commitment-hold ring's maturing in-study slot
/// for this stage — delivery target `m = spec.stage_index`'s modular slot
/// (`m mod k_max`, [`StateSpace::commitment_hold_in_study_offset`]), NOT a
/// per-block thermal generation column: those differ when block hours or
/// generations are non-uniform, and the fishing constraint pins that slot to
/// the block-hours-weighted average. The read applies unconditionally for any
/// anticipated plant.
#[inline]
fn compute_anticipated_committed_mw(
    view: &SolutionView<'_>,
    spec: &StageExtractionSpec<'_>,
    anticipated_plants: &AnticipatedPlants,
    thermal_local: usize,
) -> Option<f64> {
    let local_idx = anticipated_plants.local_of(ThermalSys::new(thermal_local))?;
    // Ring buffer lives in the stage-invariant state region, so the base is the
    // role-(a) `StateSpace`, not the geometry indexer.
    let col = spec
        .state
        .commitment_hold_incoming_col(local_idx.get(), spec.stage_index)
        .get();
    debug_assert!(
        col < view.primal.len(),
        "commitment-hold maturing-slot col {col} out of primal bounds {}",
        view.primal.len(),
    );
    Some(view.primal[col])
}

/// Extract one `anticipated_lanes` row per anticipated plant's genuine decision
/// at THIS stage whose delivery target `m` is post-study (`m >= n_stages`).
///
/// Sparse by construction: a plant with no genuine decision this stage, or one
/// targeting an in-study delivery, contributes nothing, so the partition is
/// empty at every non-decider stage. `deposited_decision_mw` reads the plant's
/// ring decision column ([`StageGeometry::anticipated_decision_col`]);
/// `carried_committed_mw` reads the ring slot the target lands in
/// ([`StateSpace::commitment_hold_outgoing_col`]) — the SAME
/// slot the deposit latches (`fill_anticipated_state_out_def_entries`), so the
/// deposit row pins the two equal at the decider stage. `delivery_dates` is the
/// extended delivery-stage anchor array indexed by delivery target `m` (study
/// stages then the post-study continuation), the same calendar the policy
/// manifest dates ring slots against via `delivery_anchor_at`. `thermal_id`
/// resolves anticipated-local `local` through
/// [`StudyDimensions::anticipated_plants`] into the system thermals, the
/// canonical anticipated order the ring and manifest share.
///
/// Iterating every genuine post-study decision (not `.next()`) keeps one row per
/// decision, so a future multi-decider fill fans out rather than silently
/// keeping the first; today's single-decider fill makes this at most one row per
/// plant per stage.
pub(crate) fn extract_anticipated_lanes(
    view: &SolutionView<'_>,
    spec: &StageExtractionSpec<'_>,
    delivery_dates: &[i32],
    stage_id: u32,
) -> Vec<SimulationAnticipatedLaneResult> {
    let state = spec.state;
    let mut results = Vec::new();
    for local in 0..state.n_anticipated {
        let local_idx = AnticipatedLocal::new(local);
        let resolution = anticipated_resolution_for(state, local_idx);
        for m in resolution.genuine_decisions_at(spec.stage_index) {
            if m < spec.horizon.num_stages() {
                continue;
            }
            let decision_col = spec.geometry.anticipated_decision_col(local_idx);
            let carried_col = state.commitment_hold_outgoing_col(local, m).get();
            debug_assert!(
                decision_col < view.primal.len() && carried_col < view.primal.len(),
                "anticipated-lane ring cols {decision_col}/{carried_col} out of primal bounds {}",
                view.primal.len(),
            );
            debug_assert!(
                m < delivery_dates.len(),
                "delivery target {m} out of delivery_dates bounds {}",
                delivery_dates.len(),
            );
            let sys_thermal = spec
                .study_dims
                .anticipated_plants
                .thermal_of(local_idx)
                .get();
            results.push(SimulationAnticipatedLaneResult {
                stage_id,
                thermal_id: spec.entity_counts.thermal_ids[sys_thermal],
                delivery_date: delivery_dates[m],
                deposited_decision_mw: view.primal[decision_col],
                carried_committed_mw: view.primal[carried_col],
            });
        }
    }
    results
}

/// Extract the travel-time in-transit bucket records for one stage, in the
/// canonical [`StateSpace::transit_bucket_column_order`] `(downstream plant, lag)`
/// order — the same column order the LP fill and cut projection use, so the
/// output row/column order is declaration-order invariant.
///
/// Empty when `state.n_buckets == 0` (no arc declared), which keeps the table
/// absent for a non-travel-time study. The in-transit volume is the outgoing
/// bucket state `transit_buckets_out`; the delayed-arrival delivery is the incoming
/// lag-1 bucket `b_1^in` (`transit_buckets_in` at the plant's first bucket), reported
/// only at `lag == 1` where the water matures onto the balance row.
fn extract_transit_buckets(
    view: &SolutionView<'_>,
    spec: &StageExtractionSpec<'_>,
    stage_id: u32,
) -> Vec<SimulationTransitBucketResult> {
    let state = spec.state;
    if state.n_buckets == 0 {
        return Vec::new();
    }
    debug_assert!(
        state.transit_buckets_out.end <= view.primal.len()
            && state.transit_buckets_in.end <= view.primal.len(),
        "bucket primal out of bounds: n_buckets {}, primal len {}",
        state.n_buckets,
        view.primal.len(),
    );
    let mut results = Vec::with_capacity(state.n_buckets);
    for (b, &(plant_idx, lag)) in state.transit_bucket_column_order.iter().enumerate() {
        debug_assert!(
            plant_idx.get() < spec.entity_counts.hydro_ids.len(),
            "bucket plant index {} out of bounds for hydro_ids len {}",
            plant_idx.get(),
            spec.entity_counts.hydro_ids.len(),
        );
        let hydro_id = spec.entity_counts.hydro_ids[plant_idx.get()];
        let in_transit_volume_hm3 = view.primal[state.bucket_outgoing_col(b).get()];
        let delayed_arrival_hm3 = if lag == 1 {
            view.primal[state.bucket_incoming_col(b).get()]
        } else {
            0.0
        };
        #[allow(clippy::cast_possible_truncation)]
        results.push(SimulationTransitBucketResult {
            stage_id,
            hydro_id,
            lag: lag as u32,
            in_transit_volume_hm3,
            delayed_arrival_hm3,
        });
    }
    results
}

/// One declared travel-time arc's upstream hydro identity, resolved once at
/// setup time from [`cobre_core::System::hydros`] and threaded into the
/// simulation pipeline — the rolling-seed emitter never re-derives it from
/// `System`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransitSeedArc {
    /// Upstream hydro plant entity ID whose release feeds the arc.
    pub upstream_hydro_id: i32,
    /// Travel time on the arc, in hours (`> 0.0`).
    pub travel_time_hours: f64,
}

/// Hours between `study_end` and `end_date` (`study_end − end_date`), mirroring
/// `setup::hours_between`'s convention — positive when `end_date` precedes
/// `study_end`.
#[allow(clippy::cast_precision_loss)]
fn hours_before(study_end: NaiveDate, end_date: NaiveDate) -> f64 {
    (study_end - end_date).num_hours() as f64
}

/// Duration-weighted mean of (turbined + spillage) across this stage's blocks
/// for one hydro, in m³/s — the same quantity `fill_arc_release_block_entries`
/// (`lp/builder/entries.rs`) deposits into the arc's `k_d`-weighted rows
/// (`push_plant_release`'s `Σ_c q_c + s`), so re-seeding from it reproduces the
/// ring's in-transit distribution by construction. Diverted outflow is
/// deliberately excluded: it leaves via the plant's own `diversion_col` to a
/// different downstream target and never feeds this arc's deposit — including
/// it would overstate the seed whenever the same hydro also declares a
/// diversion channel. `0.0` if the hydro has no block rows at this stage or the
/// stage's total block hours are `0.0`.
fn stage_release_rate_m3s(
    stage_result: &SimulationStageResult,
    hydro_id: i32,
    block_hours: &[f64],
) -> f64 {
    let mut weighted_sum = 0.0;
    let mut total_hours = 0.0;
    for hydro in stage_result
        .hydros
        .iter()
        .filter(|h| h.hydro_id == hydro_id)
    {
        let Some(b) = hydro.block_id else { continue };
        let hours = block_hours[b as usize];
        weighted_sum += hours * (hydro.turbined_m3s + hydro.spillage_m3s);
        total_hours += hours;
    }
    if total_hours > 0.0 {
        weighted_sum / total_hours
    } else {
        0.0
    }
}

/// Build one scenario's rolling-seed windows, reconstructed from realized
/// releases rather than the LP's masked terminal bucket state, so this
/// emitter stays independent of the terminal-state FCF valuation.
///
/// For each declared arc, emits one window per in-study stage whose own
/// `[start_date, end_date)` overlaps the trailing `[study_end −
/// travel_time_hours, study_end)` span — `value_m3s` the stage's
/// [`stage_release_rate_m3s`] — followed by the stitched pre-study windows
/// sliced from `past_defluences` that overlap the same trailing span
/// (nonempty only when `travel_time_hours` exceeds the study horizon).
/// `stage_results` and `study_stage_dates` are parallel, one entry per
/// in-study stage. Empty when `declared_arcs` is empty (no declared
/// travel-time arc), keeping the `transit_seed` partition absent.
pub(crate) fn build_transit_seed(
    stage_results: &[SimulationStageResult],
    study_stage_dates: &[(NaiveDate, NaiveDate)],
    declared_arcs: &[TransitSeedArc],
    past_defluences: &[HydroPastDefluence],
    block_hours_per_stage: &[Vec<f64>],
) -> Vec<SimulationTransitSeedResult> {
    if declared_arcs.is_empty() {
        return Vec::new();
    }
    debug_assert_eq!(
        stage_results.len(),
        study_stage_dates.len(),
        "stage_results and study_stage_dates must be parallel, one entry per in-study stage"
    );
    let Some(&(_, study_end)) = study_stage_dates.last() else {
        return Vec::new();
    };

    let mut windows = Vec::new();
    for arc in declared_arcs {
        for (stage_result, &(start_date, end_date)) in stage_results.iter().zip(study_stage_dates) {
            if hours_before(study_end, end_date) >= arc.travel_time_hours {
                continue;
            }
            let value_m3s = stage_release_rate_m3s(
                stage_result,
                arc.upstream_hydro_id,
                block_hours_per_stage
                    .get(stage_result.stage_id as usize)
                    .map_or(&[][..], Vec::as_slice),
            );
            windows.push(SimulationTransitSeedResult {
                hydro_id: arc.upstream_hydro_id,
                start_date,
                end_date,
                value_m3s,
            });
        }

        for window in past_defluences
            .iter()
            .filter(|w| w.hydro_id.0 == arc.upstream_hydro_id)
        {
            if hours_before(study_end, window.end_date) >= arc.travel_time_hours {
                continue;
            }
            windows.push(SimulationTransitSeedResult {
                hydro_id: arc.upstream_hydro_id,
                start_date: window.start_date,
                end_date: window.end_date,
                value_m3s: window.value_m3s,
            });
        }
    }
    windows
}

/// System entity counts needed to populate per-entity result [`Vec`]s. Every ID
/// list is in canonical ID-sorted order. Entity types that contribute no columns
/// at a stage (e.g. contracts) still carry counts so stub zero-valued entries
/// preserve entity ordering for the output writer.
#[derive(Debug, Clone)]
pub struct EntityCounts {
    /// Operating hydro plant IDs.
    pub hydro_ids: Vec<i32>,
    /// Thermal unit IDs.
    pub thermal_ids: Vec<i32>,
    /// Transmission line IDs.
    pub line_ids: Vec<i32>,
    /// Bus IDs.
    pub bus_ids: Vec<i32>,
    /// Length must equal `indexer.hydro_count`. Values are unused — per-stage
    /// productivity is read through `StageExtractionSpec::hydro_productivities`;
    /// retained for the `debug_assert!` length invariant.
    pub hydro_productivities: Vec<f64>,
    /// Pumping station IDs (empty if none).
    pub pumping_station_ids: Vec<i32>,
    /// Contract IDs (empty if none).
    pub contract_ids: Vec<i32>,
    /// Non-controllable source IDs (empty if none).
    pub non_controllable_ids: Vec<i32>,
}

/// Return the 0-based scenario ID range assigned to `rank` out of `world_size` ranks.
///
/// Uses a two-level distribution: the first `n_scenarios % world_size` ranks
/// receive one extra scenario (the "fat" group), and the remaining ranks receive
/// the floor. This matches the distribution strategy from
/// simulation-architecture.md SS3.1.
///
/// The sum of all ranks' range lengths equals `n_scenarios`.
///
/// # Panics
///
/// Panics in debug builds when `world_size == 0`.
///
/// # Examples
///
/// ```
/// use cobre_sddp::simulation::extraction::assign_scenarios;
///
/// // 10 scenarios, 3 ranks:
/// //   10 % 3 = 1  → rank 0 gets ceil(10/3) = 4 scenarios
/// //   ranks 1-2 get floor(10/3) = 3 scenarios
/// assert_eq!(assign_scenarios(10, 0, 3), 0..4);
/// assert_eq!(assign_scenarios(10, 1, 3), 4..7);
/// assert_eq!(assign_scenarios(10, 2, 3), 7..10);
///
/// // Single rank: all scenarios assigned to rank 0.
/// assert_eq!(assign_scenarios(7, 0, 1), 0..7);
/// ```
#[must_use]
pub fn assign_scenarios(n_scenarios: u32, rank: usize, world_size: usize) -> Range<u32> {
    debug_assert!(world_size > 0, "world_size must be > 0");

    let n = n_scenarios as usize;
    let r = world_size;

    let fat_count = n % r;
    let fat_size = n / r + 1;
    let lean_size = n / r;

    let (start, size): (usize, usize) = if rank < fat_count {
        (rank * fat_size, fat_size)
    } else {
        (
            fat_count * fat_size + (rank - fat_count) * lean_size,
            lean_size,
        )
    };
    let end = start + size;

    #[allow(clippy::cast_possible_truncation)]
    {
        (start as u32)..(end as u32)
    }
}

/// LP solution view passed to result extraction helpers.
pub struct SolutionView<'a> {
    /// Primal variable values from the LP solve.
    pub primal: &'a [f64],
    /// Dual variable values (shadow prices) from the LP solve.
    pub dual: &'a [f64],
    /// LP objective value.
    pub objective: f64,
    /// Objective coefficient vector from the stage template.
    pub objective_coeffs: &'a [f64],
    /// Row lower bounds from the stage template (may be patched for load noise).
    pub row_lower: &'a [f64],
}

/// Conversion factor from `hm³ · MW/(m³/s)` to `MWh`.
///
/// Unit cancellation: `hm³ × 10⁶ m³/hm³ ÷ 3600 s/h × MW/(m³/s) = MWh`.
pub const ENERGY_FACTOR_MWH_PER_HM3_PER_MW_PER_M3S: f64 = 1.0e6 / 3600.0;

/// `(storage - v_min) * rho_acum * ENERGY_FACTOR` — stored energy above the
/// minimum operable volume, shared by the initial- and final-storage reads.
#[inline]
fn stored_energy_mwh(storage_hm3: f64, v_min_hm3: f64, rho_acum: f64) -> f64 {
    (storage_hm3 - v_min_hm3) * rho_acum * ENERGY_FACTOR_MWH_PER_HM3_PER_MW_PER_M3S
}

/// Block `blk`'s own water-balance dual, in currency units: hydro `h`'s row
/// resolved through [`StageGeometry::water_balance_row`], times `cost_scale_factor`.
#[inline]
fn water_value_per_hm3(
    view: &SolutionView<'_>,
    spec: &StageExtractionSpec<'_>,
    h: usize,
    blk: BlockIdx,
) -> f64 {
    view.dual[spec.geometry.water_balance_row(HydroSys::new(h), blk)] * spec.cost_scale_factor
}

/// Extraction parameters bundled for a single stage.
///
/// **Per-stage geometry contract.** Every block-major equipment read must take its
/// base AND length from `n_blks` / `geometry` (this stage's values), never a global
/// stage-0 geometry: under a non-uniform block schedule (e.g. `[1, 3, 2]`) a stage-0
/// base/stride addresses the WRONG primal columns, silently misreporting equipment
/// and the cost breakdown. `state` (role-(a), pure function of `(N, L, A, k_max)`)
/// and `study_dims` (study-invariant non-state shape) instead resolve at every
/// stage. For uniform-block studies the per-stage and stage-0 reads coincide.
pub struct StageExtractionSpec<'a> {
    /// Role-(a) state layout: source of the state-region column reads (`storage`,
    /// `storage_in`, `inflow_lags`, `commit_in`, `max_par_order`).
    pub state: &'a StateSpace,
    /// Single owner of the study-invariant, non-state LP shape (entity counts and
    /// optional-column presence flags).
    pub study_dims: &'a StudyDimensions,
    /// Stage-correct equipment geometry, resolved per stage from `StageLayout`
    /// (via `StageTemplates::geometry_per_stage`).
    pub geometry: &'a StageGeometry,
    /// Study-scope hydro-cell partition: a plant's reported `turbined_m3s`,
    /// FPHA `generation_mw`, `turbined_slack_m3s`, and `generation_slack_mw`
    /// are each sums over [`HydroCellIndex::cells_of`] — exact because every
    /// cell of a plant shares that plant's one production model, and, for the
    /// two slacks, because each cell owns its own min-floor row and column.
    pub hydro_cell_index: &'a HydroCellIndex,
    /// Entity ID lists and productivities needed to build result records.
    pub entity_counts: &'a EntityCounts,
    /// Volumetric inflow per hydro (m³/s), one entry per hydro plant.
    pub inflow_m3s_per_hydro: &'a [f64],
    /// Block hours per dispatch block, used to convert duals to spot prices.
    pub block_hours: &'a [f64],
    /// Per-row metadata for active generic constraint rows at this stage.
    pub generic_constraint_entries: &'a [GenericConstraintRowEntry],
    /// Per-(ncs, block) column upper bounds, `available_gen * factor`. Same
    /// block-major layout as the NCS columns: one run of `n_blks` per NCS entity.
    pub ncs_col_upper: &'a [f64],
    /// Per-station pumping power-consumption rate \[MW/(m³/s)\]. ID-sorted, indexed
    /// by SYSTEM station index — which under the dense layout IS the column-block
    /// position, so extraction reads it at the enumeration index.
    pub pumping_consumption_mw_per_m3s: &'a [f64],
    /// RESOLVED per-(contract, block) price \[$/`MWh`\] for THIS stage, flat with
    /// stride `n_blks`: index `c * n_blks + blk`, `c` ID-sorted parallel to
    /// `entity_counts.contract_ids`. The unscaled
    /// `contract_bounds_at_block(c, t, blk).price_per_mwh`, NOT the
    /// `col_scale`-scaled LP objective; `total_cost = price * power * hours`. Length
    /// must equal `entity_counts.contract_ids.len() * n_blks` (debug-asserted by the
    /// contract extractor).
    pub contract_prices: &'a [f64],
    /// Per-contract `(ContractType, per-family slot)`, ID-sorted parallel to
    /// `entity_counts.contract_ids`, from [`contract_family_slot`](crate::lp::builder::contract_family_slot).
    pub contract_slots: &'a [(ContractType, usize)],
    /// Map from target hydro ID to source hydro indices that divert to it.
    pub diversion_upstream: &'a HashMap<EntityId, Vec<usize>>,
    /// Per-hydro productivity at this stage. `0.0` for FPHA hydros (generation is
    /// read from the LP column instead). Length equals `indexer.hydro_count`.
    pub hydro_productivities: &'a [f64],
    /// Column scaling factors. Unscale per-variable cost: `c_orig = c_scaled / col_scale[j]`.
    pub col_scale: &'a [f64],
    /// Row scaling factors, to unscale row bounds at the extraction boundary.
    pub row_scale: &'a [f64],
    /// Product of one-step discount factors for transitions before this stage; `1.0` for stage 0.
    pub cumulative_discount_factor: f64,
    /// Resolved objective cost-scale factor (`modeling.cost_scale_factor`).
    /// Multiplies a scaled-objective quantity back to currency units.
    pub cost_scale_factor: f64,
    /// `ρ_eq` and `ρ_acum` scalars per `(hydro, stage)` via [`EnergyConversionSet`].
    pub energy_conversion: &'a EnergyConversionSet,
    /// `V_min` per hydro (hm³), in `entity_counts.hydro_ids` order. Feeds
    /// `stored_energy_mwh = (V - V_min) · ρ_acum_integrated · ENERGY_FACTOR`.
    pub hydro_min_storage_hm3: &'a [f64],
    /// Stage index within the planning horizon (0-based).
    pub stage_index: usize,
    /// Horizon mode; evaluates the horizon-boundary predicate `t + K_i <=
    /// horizon.num_stages()`.
    pub horizon: &'a HorizonMode,
    /// Per-plant commissioning window `(entry_stage_id, exit_stage_id)` for
    /// anticipated thermals, by anticipated-local position. Gates the
    /// anticipated-decision read via `is_anticipated_decision_active`
    /// on the same predicate the LP builder used — reading when the gate is `false`
    /// reports a decision for a `[0, 0]`-pinned column. Empty when none.
    pub anticipated_windows: &'a [(Option<i32>, Option<i32>)],
    /// Study-stage commissioning id per stage index (`study_stage_ids[t] = stage.id`).
    /// The gate keys its operation-window clause on the DELIVERY stage's id
    /// (`t + K_i`). Length equals `horizon.num_stages()`.
    pub study_stage_ids: &'a [i32],
}

impl StageExtractionSpec<'_> {
    /// `col_scale_factor_at` for this stage's `col_scale`.
    #[inline]
    fn col_scale_factor(&self, col: usize) -> f64 {
        col_scale_factor_at(self.col_scale, col)
    }
}

/// `col_scale[col]` when in range and non-zero; `1.0` otherwise.
#[inline]
fn col_scale_factor_at(col_scale: &[f64], col: usize) -> f64 {
    if col < col_scale.len() {
        let d = col_scale[col];
        if d == 0.0 { 1.0 } else { d }
    } else {
        1.0
    }
}

/// Sum a CELL-keyed operational-violation slack family (`turbine_below_slack`/
/// `generation_below_slack`) over plant `h`'s own cells at block `b`: the
/// plant-level report for a per-cell-keyed LP family, mirroring the
/// `turbined`/`generation_mw` cell-sum reads elsewhere in this module.
fn sum_cell_slack(
    view: &SolutionView<'_>,
    spec: &StageExtractionSpec<'_>,
    col: fn(&StageGeometry, HydroCell, BlockIdx) -> usize,
    h: usize,
    b: usize,
) -> f64 {
    spec.hydro_cell_index
        .cells_of(HydroSys::new(h))
        .map(|c| view.primal[col(spec.geometry, HydroCell::new(c), BlockIdx::new(b))])
        .sum()
}

/// The four operational-violation slack values for plant `h` at block `b`:
/// `(turbined_slack, outflow_slack_below, outflow_slack_above, generation_slack)`.
/// `turbine_below_slack`/`generation_below_slack` are CELL-keyed, so those two
/// sum `h`'s own cells via [`sum_cell_slack`]; the two outflow families stay
/// hydro-keyed.
fn hydro_operational_slacks(
    view: &SolutionView<'_>,
    spec: &StageExtractionSpec<'_>,
    h: usize,
    b: usize,
) -> (f64, f64, f64, f64) {
    let blk = BlockIdx::new(b);
    (
        sum_cell_slack(view, spec, StageGeometry::turbine_below_col, h, b),
        view.primal[spec.geometry.outflow_below_col(HydroSys::new(h), blk)],
        view.primal[spec.geometry.outflow_above_col(HydroSys::new(h), blk)],
        sum_cell_slack(view, spec, StageGeometry::generation_below_col, h, b),
    )
}

/// Stage-level (non-per-block) data extracted for one hydro plant.
///
/// Captures values that are constant across all blocks within a stage so that
/// the per-block closure in [`extract_hydro_per_block`] only needs to read
/// per-block columns.
struct HydroStageContext {
    storage_final: f64,
    storage_initial: f64,
    incremental_inflow: f64,
    inflow_slack: f64,
    withdrawal_neg: f64,
    withdrawal_pos: f64,
    fpha_local: Option<FphaLocal>,
    /// Evaporation-local slot, `None` for a hydro with no evaporation at this stage;
    /// the closure reads `evap_indices[evap_local * n_evap_slots + slot]` per block.
    evap_local: Option<EvapLocal>,
    equivalent_productivity_mw_per_m3s: f64,
    accumulated_productivity_mw_per_m3s: f64,
    integrated_equivalent_productivity_mw_per_m3s: f64,
    integrated_accumulated_productivity_mw_per_m3s: f64,
    incremental_inflow_energy_mw: f64,
    /// `Σ block_hours` — the `stored_energy_*_mw` divisor; never a per-block hours.
    stage_total_hours: f64,
    /// `V_min` (hm³), block-invariant, retained so the per-block closure derives
    /// each boundary's stored energy without re-querying conversions.
    v_min: f64,
    /// Cascade mean-evaluator grid (`integrated_accumulated_productivity`) stored
    /// energy rides — `incremental_inflow_energy_mw` stays on the reference-point
    /// grid instead; repointing it here is the forbidden alternative.
    rho_acum_integrated: f64,
    evaporation_m3s: Option<f64>,
    evaporation_violation_neg_m3s: f64,
    evaporation_violation_pos_m3s: f64,
    /// `σ_fill` terminal-target slack (hm³); read once, repeated across per-block rows.
    filling_target_violation: f64,
    /// `σ^{v-}` operating-floor slack (hm³); read once, repeated across per-block rows.
    storage_violation_below: f64,
}

impl HydroStageContext {
    /// Read all stage-level scalars for hydro at system index `h`.
    fn new(
        view: &SolutionView<'_>,
        spec: &StageExtractionSpec<'_>,
        lookup: &HydroReverseLookup,
        h: usize,
    ) -> Self {
        let state = spec.state;
        let storage_final = view.primal[state.storage_outgoing_col(HydroSys::new(h)).get()];
        let storage_initial = view.primal[state.storage_incoming_col(HydroSys::new(h)).get()];
        let incremental_inflow = spec.inflow_m3s_per_hydro[h];
        let inflow_slack = if spec.geometry.inflow_slack.is_empty() {
            0.0
        } else {
            view.primal[spec.geometry.inflow_slack_col(HydroSys::new(h))]
        };
        let withdrawal_neg = view.primal[spec.geometry.withdrawal_slack_neg_col(HydroSys::new(h))];
        let withdrawal_pos = view.primal[spec.geometry.withdrawal_slack_pos_col(HydroSys::new(h))];
        let fpha_local = lookup.fpha[h];
        let evap_local = lookup.evap[h];
        let (evaporation_m3s, evaporation_violation_neg_m3s, evaporation_violation_pos_m3s) =
            if let Some(lei) = evap_local {
                // Slot-major `evap_indices`; this reads the stage-level slot 0.
                // `extract_hydro_per_block` resolves each chronological block's own
                // triple; a parallel block routes through this same read.
                let n_evap_slots =
                    evaporation_slot_count(spec.geometry.block_mode, spec.geometry.n_blks);
                let ei = &spec.geometry.evap_indices[lei.get() * n_evap_slots];
                let evaporation_flow = view.primal[ei.evaporation_flow_col];
                let neg = view.primal[ei.f_evap_plus_col]; // f_evap_plus = under-evaporation
                let pos = view.primal[ei.f_evap_minus_col]; // f_evap_minus = over-evaporation
                (Some(evaporation_flow), neg, pos)
            } else {
                (Some(0.0), 0.0, 0.0)
            };
        let filling_target_violation =
            read_filling_target_slack_primal(view.primal, spec.geometry, lookup.filling_target[h]);
        let storage_violation_below = read_floor_slack_primal(
            view.primal,
            spec.geometry,
            lookup.filled_min_storage_floor[h],
        );
        let conv = spec.energy_conversion.conversion(h, spec.stage_index);
        let rho_acum = spec
            .energy_conversion
            .accumulated_productivity(h, spec.stage_index);
        let integrated_equivalent = spec
            .energy_conversion
            .integrated_equivalent_productivity(h, spec.stage_index);
        let integrated_accumulated = spec
            .energy_conversion
            .integrated_accumulated_productivity(h, spec.stage_index);
        let v_min = spec.hydro_min_storage_hm3[h];
        let stage_total_hours: f64 = spec.block_hours.iter().sum();
        Self {
            storage_final,
            storage_initial,
            incremental_inflow,
            inflow_slack,
            withdrawal_neg,
            withdrawal_pos,
            fpha_local,
            evap_local,
            equivalent_productivity_mw_per_m3s: conv.equivalent_productivity_mw_per_m3s,
            accumulated_productivity_mw_per_m3s: rho_acum,
            integrated_equivalent_productivity_mw_per_m3s: integrated_equivalent,
            integrated_accumulated_productivity_mw_per_m3s: integrated_accumulated,
            incremental_inflow_energy_mw: rho_acum * incremental_inflow,
            stage_total_hours,
            v_min,
            rho_acum_integrated: integrated_accumulated,
            evaporation_m3s,
            evaporation_violation_neg_m3s,
            evaporation_violation_pos_m3s,
            filling_target_violation,
            storage_violation_below,
        }
    }
}

/// Extract per-block hydro results for one hydro plant (turbined/spillage branch).
fn extract_hydro_per_block<'a>(
    view: &'a SolutionView<'a>,
    spec: &'a StageExtractionSpec<'a>,
    lookup: &'a HydroReverseLookup,
    h: usize,
    hydro_id: i32,
    stage_id: u32,
) -> impl Iterator<Item = SimulationHydroResult> + 'a {
    let n_blks = spec.geometry.n_blks;

    let ctx = HydroStageContext::new(view, spec, lookup, h);

    let hydro_entity_id = EntityId(hydro_id);
    let div_sources = spec.diversion_upstream.get(&hydro_entity_id);

    (0..n_blks).map(move |b| {
        let blk = BlockIdx::new(b);
        // Plant `h`'s turbined flow is the sum over its cells (ascending, so the
        // sum is reproducible); under single-bus identity staging this is the
        // one-term sum the pre-cell code always computed.
        let turbined: f64 = spec
            .hydro_cell_index
            .cells_of(HydroSys::new(h))
            .map(|c| view.primal[spec.geometry.turbine_col(HydroCell::new(c), blk)])
            .sum();
        let s_col = spec.geometry.spillage_col(HydroSys::new(h), blk);
        let spillage = view.primal[s_col];

        let diverted_outflow = if spec.geometry.diversion.is_empty() {
            0.0
        } else {
            view.primal[spec.geometry.diversion_col(HydroSys::new(h), blk)]
        };

        let diverted_inflow = if let Some(sources) = div_sources {
            let mut total = 0.0;
            for &d_idx in sources {
                total += view.primal[spec.geometry.diversion_col(HydroSys::new(d_idx), blk)];
            }
            total
        } else {
            0.0
        };

        // FPHA hydros sum the LP `g_{c,k}` column over the plant's cells (same
        // ascending-order/single-term-under-identity reasoning as `turbined`
        // above); constant-productivity hydros compute it as turbined * productivity.
        let generation_mw = if let Some(local_fpha_idx) = ctx.fpha_local {
            let cell_start = lookup.fpha_cell_local_start[local_fpha_idx.get()];
            let n_cells = spec.hydro_cell_index.cells_of(HydroSys::new(h)).len();
            (0..n_cells)
                .map(|i| {
                    view.primal[spec
                        .geometry
                        .generation_col(FphaCellLocal::new(cell_start + i), blk)]
                })
                .sum::<f64>()
        } else {
            turbined * spec.hydro_productivities[h]
        };

        let (turbined_slack, outflow_slack_below, outflow_slack_above, generation_slack) =
            hydro_operational_slacks(view, spec, h, b);

        // Chronological block `b` reports its own boundary pair `(Sᵇ, Sᵇ⁺¹)` via the
        // accessor (interior columns stride `n_blks − 1`, so the read cannot go
        // through `flat`); parallel keeps the stage-level `(S⁰, Sᴷ)`. The
        // endpoints coincide with the state region: block 0 incoming == `ctx.storage_initial`
        // (`S⁰`), block `K−1` outgoing == `ctx.storage_final` (`Sᴷ`).
        let (storage_initial, storage_final) = match spec.geometry.block_mode {
            BlockMode::Chronological => {
                let hydro = HydroSys::new(h);
                let storage_col =
                    |boundary| spec.geometry.block_storage_col(spec.state, hydro, boundary);
                let in_col = storage_col(Boundary::from_index(b, n_blks));
                let out_col = storage_col(Boundary::from_index(b + 1, n_blks));
                debug_assert!(
                    in_col < view.primal.len() && out_col < view.primal.len(),
                    "per-block storage cols {in_col}/{out_col} out of primal bounds {}",
                    view.primal.len(),
                );
                (view.primal[in_col], view.primal[out_col])
            }
            BlockMode::Parallel => (ctx.storage_initial, ctx.storage_final),
        };
        let stored_energy_initial_mwh =
            stored_energy_mwh(storage_initial, ctx.v_min, ctx.rho_acum_integrated);
        let stored_energy_final_mwh =
            stored_energy_mwh(storage_final, ctx.v_min, ctx.rho_acum_integrated);
        let stored_energy_initial_mw = stored_energy_initial_mwh / ctx.stage_total_hours;
        let stored_energy_final_mw = stored_energy_final_mwh / ctx.stage_total_hours;

        // Chronological block `b` reports its own slot's evaporation triple; parallel
        // routes every block through the stage-level slot already resolved into
        // `ctx` (`HydroStageContext::new`). A hydro with no evaporation slot stays
        // at the `ctx` defaults.
        let (evaporation_m3s, evaporation_violation_neg_m3s, evaporation_violation_pos_m3s) =
            match (spec.geometry.block_mode, ctx.evap_local) {
                (BlockMode::Chronological, Some(local)) => {
                    let n_evap_slots =
                        evaporation_slot_count(spec.geometry.block_mode, spec.geometry.n_blks);
                    let slot = evaporation_slot(n_evap_slots, blk);
                    let ei = &spec.geometry.evap_indices[local.get() * n_evap_slots + slot.get()];
                    debug_assert!(
                        ei.evaporation_flow_col < view.primal.len()
                            && ei.f_evap_plus_col < view.primal.len()
                            && ei.f_evap_minus_col < view.primal.len(),
                        "per-block evaporation cols out of primal bounds {}",
                        view.primal.len(),
                    );
                    (
                        Some(view.primal[ei.evaporation_flow_col]),
                        view.primal[ei.f_evap_plus_col], // f_evap_plus = under-evaporation (neg)
                        view.primal[ei.f_evap_minus_col], // f_evap_minus = over-evaporation (pos)
                    )
                }
                (BlockMode::Parallel, _) | (BlockMode::Chronological, None) => (
                    ctx.evaporation_m3s,
                    ctx.evaporation_violation_neg_m3s,
                    ctx.evaporation_violation_pos_m3s,
                ),
            };

        #[allow(clippy::cast_possible_truncation)]
        SimulationHydroResult {
            stage_id,
            block_id: Some(b as u32),
            hydro_id,
            turbined_m3s: turbined,
            spillage_m3s: spillage,
            evaporation_m3s,
            diverted_inflow_m3s: Some(diverted_inflow),
            diverted_outflow_m3s: Some(diverted_outflow),
            incremental_inflow_m3s: ctx.incremental_inflow,
            inflow_m3s: ctx.incremental_inflow,
            storage_initial_hm3: storage_initial,
            storage_final_hm3: storage_final,
            generation_mw,
            equivalent_productivity_mw_per_m3s: ctx.equivalent_productivity_mw_per_m3s,
            accumulated_productivity_mw_per_m3s: ctx.accumulated_productivity_mw_per_m3s,
            incremental_inflow_energy_mw: ctx.incremental_inflow_energy_mw,
            stored_energy_initial_mwh,
            stored_energy_final_mwh,
            spillage_cost: spillage * view.objective_coeffs[s_col] / spec.col_scale_factor(s_col)
                * spec.cost_scale_factor,
            water_value_per_hm3: water_value_per_hm3(view, spec, h, blk),
            storage_binding_code: 0,
            operative_state_code: 1,
            turbined_slack_m3s: turbined_slack,
            outflow_slack_below_m3s: outflow_slack_below,
            outflow_slack_above_m3s: outflow_slack_above,
            generation_slack_mw: generation_slack,
            storage_violation_below_hm3: ctx.storage_violation_below,
            filling_target_violation_hm3: ctx.filling_target_violation,
            evaporation_violation_pos_m3s,
            evaporation_violation_neg_m3s,
            inflow_nonnegativity_slack_m3s: ctx.inflow_slack,
            water_withdrawal_violation_pos_m3s: ctx.withdrawal_pos,
            water_withdrawal_violation_neg_m3s: ctx.withdrawal_neg,
            integrated_equivalent_productivity_mw_per_m3s: ctx
                .integrated_equivalent_productivity_mw_per_m3s,
            integrated_accumulated_productivity_mw_per_m3s: ctx
                .integrated_accumulated_productivity_mw_per_m3s,
            stored_energy_initial_mw,
            stored_energy_final_mw,
        }
    })
}

fn extract_hydros(
    view: &SolutionView<'_>,
    spec: &StageExtractionSpec<'_>,
    stage_id: u32,
    lookup: &HydroReverseLookup,
) -> Vec<SimulationHydroResult> {
    let mut results = Vec::with_capacity(spec.entity_counts.hydro_ids.len() * spec.geometry.n_blks);
    results.extend(
        spec.entity_counts
            .hydro_ids
            .iter()
            .enumerate()
            .flat_map(|(h, &hydro_id)| {
                extract_hydro_per_block(view, spec, lookup, h, hydro_id, stage_id)
            }),
    );
    results
}

/// Extract one row per `(hydro, block, cell)`, hydro-major/block-middle/
/// cell-minor. `cells_of` is iterated in the SAME ascending order
/// `extract_hydro_per_block`'s per-hydro `.sum()` consumes, so summing one
/// `(hydro, block)`'s `turbined_m3s` rows reproduces that plant's `hydros` row
/// bit-for-bit.
fn extract_hydro_bus_generation(
    view: &SolutionView<'_>,
    spec: &StageExtractionSpec<'_>,
    stage_id: u32,
    lookup: &HydroReverseLookup,
) -> Vec<SimulationHydroBusResult> {
    let n_cells = spec.hydro_cell_index.n_cells();
    let n_blks = spec.geometry.n_blks;
    let mut results = Vec::with_capacity(n_cells * n_blks);
    for (h, &hydro_id) in spec.entity_counts.hydro_ids.iter().enumerate() {
        let cells = spec.hydro_cell_index.cells_of(HydroSys::new(h));
        let fpha_local = lookup.fpha[h];
        for b in 0..n_blks {
            let blk = BlockIdx::new(b);
            for c in cells.clone() {
                let turbined_m3s = view.primal[spec.geometry.turbine_col(HydroCell::new(c), blk)];
                let generation_mw = if let Some(local) = fpha_local {
                    let cell_start = lookup.fpha_cell_local_start[local.get()];
                    view.primal[spec
                        .geometry
                        .generation_col(FphaCellLocal::new(cell_start + (c - cells.start)), blk)]
                } else {
                    turbined_m3s * spec.hydro_productivities[h]
                };
                #[allow(clippy::cast_possible_truncation)]
                results.push(SimulationHydroBusResult {
                    stage_id,
                    block_id: Some(b as u32),
                    hydro_id,
                    bus_id: i32::from(spec.hydro_cell_index.bus_of(HydroCell::new(c))),
                    turbined_m3s,
                    generation_mw,
                });
            }
        }
    }
    results
}

/// Extract thermal results from a raw LP solution view.
fn extract_thermals(
    view: &SolutionView<'_>,
    spec: &StageExtractionSpec<'_>,
    stage_id: u32,
    anticipated_plants: &AnticipatedPlants,
) -> Vec<SimulationThermalResult> {
    let n_blks = spec.geometry.n_blks;
    let mut results = Vec::with_capacity(spec.entity_counts.thermal_ids.len() * n_blks);
    for (t, &thermal_id) in spec.entity_counts.thermal_ids.iter().enumerate() {
        let is_anticipated = anticipated_plants.local_of(ThermalSys::new(t)).is_some();
        let anticipated_decision_mw =
            compute_anticipated_decision_mw(view, spec, anticipated_plants, t);
        let anticipated_committed_mw =
            compute_anticipated_committed_mw(view, spec, anticipated_plants, t);
        for b in 0..n_blks {
            let col = spec
                .geometry
                .thermal_col(ThermalSys::new(t), BlockIdx::new(b));
            let gen_mw = view.primal[col];
            #[allow(clippy::cast_possible_truncation)]
            results.push(SimulationThermalResult {
                stage_id,
                block_id: Some(b as u32),
                thermal_id,
                generation_mw: gen_mw,
                generation_cost: gen_mw * view.objective_coeffs[col] / spec.col_scale_factor(col)
                    * spec.cost_scale_factor,
                is_anticipated,
                anticipated_committed_mw,
                anticipated_decision_mw,
                operative_state_code: 1,
            });
        }
    }
    results
}

/// Extract exchange (line flow) results from a raw LP solution view.
fn extract_exchanges(
    view: &SolutionView<'_>,
    spec: &StageExtractionSpec<'_>,
    stage_id: u32,
) -> Vec<SimulationExchangeResult> {
    let n_blks = spec.geometry.n_blks;
    let mut results = Vec::with_capacity(spec.entity_counts.line_ids.len() * n_blks);
    results.extend(spec.entity_counts.line_ids.iter().enumerate().flat_map(
        move |(l, &line_id)| {
            (0..n_blks).map(move |b| {
                let blk = BlockIdx::new(b);
                let fwd_col = spec.geometry.line_fwd_col(LineSys::new(l), blk);
                let rev_col = spec.geometry.line_rev_col(LineSys::new(l), blk);
                let fwd = view.primal[fwd_col];
                let rev = view.primal[rev_col];
                #[allow(clippy::cast_possible_truncation)]
                SimulationExchangeResult {
                    stage_id,
                    block_id: Some(b as u32),
                    line_id,
                    direct_flow_mw: fwd,
                    reverse_flow_mw: rev,
                    exchange_cost: (fwd * view.objective_coeffs[fwd_col]
                        / spec.col_scale_factor(fwd_col)
                        + rev * view.objective_coeffs[rev_col] / spec.col_scale_factor(rev_col))
                        * spec.cost_scale_factor,
                    operative_state_code: 2,
                }
            })
        },
    ));
    results
}

/// Extract bus results from a raw LP solution view.
fn extract_buses(
    view: &SolutionView<'_>,
    spec: &StageExtractionSpec<'_>,
    stage_id: u32,
) -> Vec<SimulationBusResult> {
    let n_blks = spec.geometry.n_blks;
    let max_segs = spec.study_dims.max_deficit_segments;
    let mut results = Vec::with_capacity(spec.entity_counts.bus_ids.len() * n_blks);
    results.extend(spec.entity_counts.bus_ids.iter().enumerate().flat_map(
        move |(bus_idx, &bus_id)| {
            (0..n_blks).map(move |b| {
                let blk = BlockIdx::new(b);
                let bus = BusSys::new(bus_idx);
                let deficit_mw: f64 = (0..max_segs)
                    .map(|s| view.primal[spec.geometry.deficit_col(bus, s, blk, max_segs)])
                    .sum();
                let excess_col = spec.geometry.excess_col(bus, blk);
                let load_row = spec.geometry.load_balance_row(bus, blk);
                let raw_dual = view.dual[load_row];
                let hrs = spec.block_hours[b];
                #[allow(clippy::cast_possible_truncation)]
                SimulationBusResult {
                    stage_id,
                    block_id: Some(b as u32),
                    bus_id,
                    load_mw: view.row_lower[load_row],
                    deficit_mw,
                    excess_mw: view.primal[excess_col],
                    spot_price: if hrs > 0.0 {
                        raw_dual * spec.cost_scale_factor / hrs
                    } else {
                        0.0
                    },
                }
            })
        },
    ));
    results
}

/// Extract a [`SimulationStageResult`] from a raw LP solution at one stage.
///
/// Reads role-(b) equipment column values from `view.primal` using the ranges
/// stored in `spec.geometry` (the per-stage [`StageGeometry`]);
/// role-(a) state columns resolve via `spec.state` ([`StateSpace`]). When a
/// family has zero entities its range is empty (`0..0`) and that result defaults
/// to zero.
///
/// The LP objective is split into `future_cost = primal[spec.state.theta]` and
/// `stage_cost = objective - future_cost`, following the same convention as the
/// training forward pass.
///
/// # Preconditions
///
/// - `view.primal.len() >= spec.state.theta + 1`
/// - `spec.entity_counts.hydro_ids.len() == spec.state.hydro_count`
/// - `spec.entity_counts.hydro_productivities.len() == spec.state.hydro_count`
/// - `view.objective_coeffs.len() >= view.primal.len()` when equipment ranges are non-empty
/// - `view.row_lower.len() >= spec.geometry.load_balance.end()` when `load_balance` is non-empty
/// - `stage_id` is 0-based
///
/// Violations are caught by `debug_assert!` in debug builds.
///
/// # Performance
///
/// Builds the hydro reverse-lookup table on every call. On the hot path use
/// `extract_stage_result_with_lookups` with a pre-built `hydro_lookup` instead.
///
/// The visited node id defaults to `stage_id` — the chain-degenerate node id
/// (`node_graph.node_ids[t] == t` on a chain). A branching walk supplies its own
/// node id through [`extract_stage_result_with_lookups`] (the hot path).
#[must_use]
#[allow(clippy::cast_possible_wrap)]
pub fn extract_stage_result(
    view: &SolutionView<'_>,
    spec: &StageExtractionSpec<'_>,
    stage_id: u32,
) -> SimulationStageResult {
    let n_hydros = spec.entity_counts.hydro_ids.len();
    let hydro_lookup = HydroReverseLookup::build(spec.geometry, spec.hydro_cell_index, n_hydros);
    extract_stage_result_with_lookups(
        view,
        spec,
        stage_id,
        NodeId(stage_id as i32),
        &hydro_lookup,
        &spec.study_dims.anticipated_plants,
    )
}

/// Extract a [`SimulationStageResult`] using a pre-built hydro reverse-lookup
/// table.
///
/// Identical to [`extract_stage_result`] but avoids building the
/// [`HydroReverseLookup`] table on every call.
///
/// `anticipated_plants` is the study-invariant anticipated-plant set
/// (typically `spec.study_dims.anticipated_plants`); `hydro_lookup` is the
/// lookup for **this stage** (FPHA/evap membership is per-`(hydro, stage)`).
/// Build the per-stage hydro lookups once per simulation run (or per worker
/// thread) and pass them by reference here to eliminate per-`(scenario,
/// stage)` allocations on the hot path.
///
/// # Preconditions
///
/// Same as [`extract_stage_result`] plus:
/// - `hydro_lookup` was built from this stage's [`StageGeometry`] and `n_hydros`.
pub(crate) fn extract_stage_result_with_lookups(
    view: &SolutionView<'_>,
    spec: &StageExtractionSpec<'_>,
    stage_id: u32,
    node_id: NodeId,
    hydro_lookup: &HydroReverseLookup,
    anticipated_plants: &AnticipatedPlants,
) -> SimulationStageResult {
    let state = spec.state;
    debug_assert!(
        view.primal.len() > state.theta,
        "primal vector too short: len={}, need > theta={}",
        view.primal.len(),
        state.theta
    );
    debug_assert!(
        spec.entity_counts.hydro_ids.len() == state.hydro_count,
        "hydro_ids length {} does not match state.hydro_count {}",
        spec.entity_counts.hydro_ids.len(),
        state.hydro_count
    );
    // Bounds guard against the per-stage geometry's excess end, NOT the global
    // stage-0 `indexer.excess.end` (n_blks-dependent), so a non-uniform stage with
    // fewer blocks than stage 0 cannot spuriously trip this on a stale stage-0 end.
    debug_assert!(
        spec.geometry.excess.is_empty() || view.objective_coeffs.len() >= spec.geometry.excess.end,
        "objective_coeffs too short: len={}, need >= excess.end={}",
        view.objective_coeffs.len(),
        spec.geometry.excess.end
    );
    debug_assert!(
        spec.entity_counts.hydro_productivities.len() == state.hydro_count,
        "hydro_productivities length {} does not match state.hydro_count {}",
        spec.entity_counts.hydro_productivities.len(),
        state.hydro_count
    );
    let load_balance_end = spec.geometry.load_balance.end();
    debug_assert!(
        spec.geometry.load_balance.range().is_empty() || view.row_lower.len() >= load_balance_end,
        "row_lower too short: len={}, need >= load_balance_end={load_balance_end}",
        view.row_lower.len(),
    );

    let (generic_violations, generic_violation_cost) =
        extract_generic_violations(view, spec, stage_id);
    let (non_controllables, ncs_curtailment_cost) = extract_non_controllables(view, spec, stage_id);
    let costs = vec![compute_cost_result(
        view,
        spec.geometry,
        spec.state,
        spec.col_scale,
        generic_violation_cost,
        spec.cumulative_discount_factor,
        spec.cost_scale_factor,
        ncs_curtailment_cost,
        stage_id,
    )];
    let (inflow_lags, pumping_stations, contracts) = extract_stub_collections(view, spec, stage_id);

    SimulationStageResult {
        stage_id,
        node_id,
        costs,
        hydros: extract_hydros(view, spec, stage_id, hydro_lookup),
        hydro_bus_generation: extract_hydro_bus_generation(view, spec, stage_id, hydro_lookup),
        thermals: extract_thermals(view, spec, stage_id, anticipated_plants),
        exchanges: extract_exchanges(view, spec, stage_id),
        buses: extract_buses(view, spec, stage_id),
        pumping_stations,
        contracts,
        non_controllables,
        inflow_lags,
        transit_buckets: extract_transit_buckets(view, spec, stage_id),
        generic_violations,
        // The window-indexed delivery-date array lives outside `StageExtractionSpec`
        // (`SimulationOutputSpec::extended_delivery_anchors`); the hot path
        // (`extract_sim_stage_result`) populates this field as a post-step via
        // `extract_anticipated_lanes` instead of through this shared builder.
        anticipated_lanes: Vec::new(),
    }
}

/// Per-constraint hydro violation costs extracted from a solution view.
struct HydroViolationCosts {
    evaporation: f64,
    withdrawal: f64,
    outflow_below: f64,
    outflow_above: f64,
    turbined: f64,
    generation: f64,
}

impl HydroViolationCosts {
    fn total(&self) -> f64 {
        self.evaporation
            + self.withdrawal
            + self.outflow_below
            + self.outflow_above
            + self.turbined
            + self.generation
    }
}

/// Compute the 6 per-constraint hydro violation costs from a solution view.
fn compute_hydro_violation_costs(
    equipment: &StageGeometry,
    col_cost: impl Fn(usize) -> f64,
    range_sum: impl Fn(Range<usize>) -> f64,
    cost_scale_factor: f64,
) -> HydroViolationCosts {
    let evaporation = equipment
        .evap_indices
        .iter()
        .map(|ei| col_cost(ei.f_evap_plus_col) + col_cost(ei.f_evap_minus_col))
        .sum::<f64>()
        * cost_scale_factor;

    let withdrawal = if equipment.withdrawal_slack_neg.is_empty() {
        0.0
    } else {
        (range_sum(equipment.withdrawal_slack_neg.clone())
            + range_sum(equipment.withdrawal_slack_pos.clone()))
            * cost_scale_factor
    };

    let (outflow_below, outflow_above, turbined, generation) =
        if equipment.outflow_below_slack.is_empty() {
            (0.0, 0.0, 0.0, 0.0)
        } else {
            (
                range_sum(equipment.outflow_below_slack.clone()) * cost_scale_factor,
                range_sum(equipment.outflow_above_slack.clone()) * cost_scale_factor,
                range_sum(equipment.turbine_below_slack.clone()) * cost_scale_factor,
                range_sum(equipment.generation_below_slack.clone()) * cost_scale_factor,
            )
        };

    HydroViolationCosts {
        evaporation,
        withdrawal,
        outflow_below,
        outflow_above,
        turbined,
        generation,
    }
}

/// Compute the single-stage cost breakdown from an LP solution view.
///
/// All cost fields are returned in original monetary units. The LP operates in
/// scaled cost space (objective coefficients divided by `cost_scale_factor` at
/// template build time); this function multiplies back by `cost_scale_factor`
/// at the reporting boundary to recover original units.
// Rationale: each parameter is an independently-sourced per-stage scalar/slice
// the single-solve cost breakdown needs once; a wrapper struct would just move
// the arity to the one call site that already builds this from `spec` fields.
#[allow(clippy::too_many_arguments)]
fn compute_cost_result(
    view: &SolutionView<'_>,
    equipment: &StageGeometry,
    state: &StateSpace,
    col_scale: &[f64],
    generic_violation_cost: f64,
    cumulative_discount_factor: f64,
    cost_scale_factor: f64,
    ncs_curtailment_cost: f64,
    stage_id: u32,
) -> SimulationCostResult {
    let scale_factor = |col: usize| col_scale_factor_at(col_scale, col);
    let col_cost = |col: usize| view.primal[col] * view.objective_coeffs[col] / scale_factor(col);
    let range_sum = |r: Range<usize>| -> f64 { r.map(col_cost).sum() };

    let theta_obj_coeff = view
        .objective_coeffs
        .get(state.theta)
        .copied()
        .unwrap_or(1.0);
    let theta_contribution = view.primal[state.theta] * theta_obj_coeff;
    let future_cost = theta_contribution * cost_scale_factor;
    let immediate_cost = (view.objective - theta_contribution) * cost_scale_factor;

    // Every range summed below must sum the whole per-stage `equipment` family, not
    // just active columns: this is what keeps `Σ(breakdown) == immediate_cost`.
    let family_cost = |r: &Range<usize>| -> f64 {
        if r.is_empty() {
            0.0
        } else {
            range_sum(r.clone()) * cost_scale_factor
        }
    };
    let thermal_cost = family_cost(&equipment.thermal);
    // Inactive anticipated-decision columns are `[0, 0]`-pinned (primal 0), so
    // summing the whole range books the fuel only where the decision is live —
    // matching `immediate_cost`.
    let anticipated_thermal_cost = family_cost(&equipment.anticipated_decision);
    // Contract objective coeff is `price_per_mwh * block_hours`, so `col_cost`
    // sums `power * price * hours` with the stored sign (export price < 0 nets
    // negative). The objective term is in `immediate_cost`; booking it here keeps
    // `Σ(macro categories) == immediate_cost` when contracts are active.
    let contract_cost =
        if equipment.contract_import.is_empty() && equipment.contract_export.is_empty() {
            0.0
        } else {
            equipment
                .contract_import
                .clone()
                .chain(equipment.contract_export.clone())
                .map(col_cost)
                .sum::<f64>()
                * cost_scale_factor
        };
    let spillage_cost = family_cost(&equipment.spillage);
    let exchange_cost = if equipment.line_fwd.is_empty() {
        0.0
    } else {
        equipment
            .line_fwd
            .clone()
            .chain(equipment.line_rev.clone())
            .map(col_cost)
            .sum::<f64>()
            * cost_scale_factor
    };
    let deficit_cost = family_cost(&equipment.deficit);
    let excess_cost = family_cost(&equipment.excess);
    let turbined_cost = family_cost(&equipment.turbine);
    let inflow_penalty_cost = family_cost(&equipment.inflow_slack);
    let diversion_cost = family_cost(&equipment.diversion);

    let hv = compute_hydro_violation_costs(equipment, col_cost, range_sum, cost_scale_factor);

    SimulationCostResult {
        stage_id,
        block_id: None,
        total_cost: view.objective * cost_scale_factor,
        immediate_cost,
        future_cost,
        discount_factor: cumulative_discount_factor,
        thermal_cost,
        anticipated_thermal_cost,
        contract_cost,
        deficit_cost,
        excess_cost,
        storage_violation_cost: 0.0,
        filling_target_cost: 0.0,
        hydro_violation_cost: hv.total(),
        outflow_violation_below_cost: hv.outflow_below,
        outflow_violation_above_cost: hv.outflow_above,
        turbined_violation_cost: hv.turbined,
        generation_violation_cost: hv.generation,
        evaporation_violation_cost: hv.evaporation,
        withdrawal_violation_cost: hv.withdrawal,
        inflow_penalty_cost,
        generic_violation_cost,
        spillage_cost: spillage_cost + diversion_cost,
        turbined_cost,
        curtailment_cost: ncs_curtailment_cost,
        exchange_cost,
        pumping_cost: 0.0,
    }
}

/// Extract generic constraint violation results from a solved LP.
///
/// For a two-sided row (`slack_minus_col` present) the reported `slack_value` is
/// the net violation `s_plus - s_minus`, while its cost charges both
/// (`s_plus + s_minus`) — one violation record per row either way.
fn extract_generic_violations(
    view: &SolutionView<'_>,
    spec: &StageExtractionSpec<'_>,
    stage_id: u32,
) -> (Vec<SimulationGenericViolationResult>, f64) {
    let entries = spec.generic_constraint_entries;
    if entries.is_empty() {
        return (Vec::new(), 0.0);
    }

    let mut results = Vec::with_capacity(entries.len());
    let mut total_cost = 0.0;

    for entry in entries {
        // A stage-level row is priced by total stage hours (matching the LP
        // objective in `fill_generic_constraint_entries`); a per-block row by its block's.
        let block_hours = if entry.is_stage_level {
            spec.block_hours.iter().sum()
        } else {
            spec.block_hours
                .get(entry.block_idx)
                .copied()
                .unwrap_or(0.0)
        };
        let (slack_value, slack_cost) = if entry.slack_enabled {
            if let Some(minus_col) = entry.slack_minus_col {
                let s_plus = entry.slack_plus_col.map_or(0.0, |col| view.primal[col]);
                let s_minus = view.primal[minus_col];
                let net = s_plus - s_minus;
                let cost = (s_plus + s_minus) * entry.slack_penalty * block_hours;
                (net, cost)
            } else {
                let s = entry.slack_plus_col.map_or(0.0, |col| view.primal[col]);
                let cost = s * entry.slack_penalty * block_hours;
                (s, cost)
            }
        } else {
            (0.0, 0.0)
        };

        total_cost += slack_cost;

        results.push(SimulationGenericViolationResult {
            stage_id,
            // SAFETY: block_idx is a stage block index, always < n_blocks which is << 2^32.
            #[allow(clippy::cast_possible_truncation)]
            block_id: if entry.is_stage_level {
                None
            } else {
                Some(entry.block_idx as u32)
            },
            constraint_id: entry.entity_id,
            slack_value,
            slack_cost,
        });
    }

    (results, total_cost)
}

/// Extract NCS generation results from a solved LP — dense, one row per system
/// NCS at every stage.
///
/// The total curtailment cost is negated so it is positive in the breakdown. A
/// commissioning-dormant NCS (`[0, 0]`-pinned column) emits a ZERO row rather than
/// being absent — uniform with how thermal/line report a zeroed entity.
fn extract_non_controllables(
    view: &SolutionView<'_>,
    spec: &StageExtractionSpec<'_>,
    stage_id: u32,
) -> (Vec<SimulationNonControllableResult>, f64) {
    if spec.geometry.ncs_generation.is_empty() {
        return (Vec::new(), 0.0);
    }

    let n_blks = spec.geometry.n_blks;
    let mut results = Vec::with_capacity(spec.geometry.ncs_generation.len());
    let mut total_curtailment_cost = 0.0;

    for (ncs_sys, &ncs_id) in spec.entity_counts.non_controllable_ids.iter().enumerate() {
        for blk in 0..n_blks {
            let col = spec
                .geometry
                .ncs_generation_col(NcsSys::new(ncs_sys), BlockIdx::new(blk));
            let generation_mw = view.primal[col];
            let col_upper_offset = col - spec.geometry.ncs_generation.start;
            debug_assert!(
                col_upper_offset < spec.ncs_col_upper.len(),
                "NCS col_upper out of bounds: offset {col_upper_offset}, len {}",
                spec.ncs_col_upper.len()
            );
            let available_mw = spec.ncs_col_upper[col_upper_offset];
            let curtailment_mw = available_mw - generation_mw;
            // NCS obj coefficient is negative, so negate to report a positive cost.
            let col_cost = -(curtailment_mw * view.objective_coeffs[col]
                / spec.col_scale_factor(col))
                * spec.cost_scale_factor;
            total_curtailment_cost += col_cost;

            #[allow(clippy::cast_possible_truncation)]
            results.push(SimulationNonControllableResult {
                stage_id,
                block_id: Some(blk as u32),
                non_controllable_id: ncs_id,
                generation_mw,
                available_mw,
                curtailment_mw,
                curtailment_cost: col_cost,
                operative_state_code: 1,
            });
        }
    }

    (results, total_curtailment_cost)
}

/// Extract one [`SimulationPumpingResult`] per (station, block) from the solved
/// pumping-flow primals — dense, one row per system station at every stage.
///
/// The flow is NOT divided by `col_scale` — `view.primal` is already unscaled —
/// and `power_consumption_mw = pumped_flow_m3s * consumption[p_sys]` reuses the
/// same coefficient the `PumpingPower` resolver applies on the bus load-balance
/// row. Under the dense layout the enumeration index IS the SYSTEM index, so a
/// commissioning-dormant station emits a ZERO row rather than being absent.
/// `pumping_cost` is imputed `0.0` here, finalized by the output writer.
fn extract_pumping_stations(
    view: &SolutionView<'_>,
    spec: &StageExtractionSpec<'_>,
    stage_id: u32,
) -> Vec<SimulationPumpingResult> {
    let n_blks = spec.geometry.n_blks;

    debug_assert!(
        view.primal.len() >= spec.geometry.pumping_flow.end,
        "pumping primal out of bounds: need {}, have {}",
        spec.geometry.pumping_flow.end,
        view.primal.len()
    );

    let mut results = Vec::with_capacity(spec.geometry.pumping_flow.len());
    for (p_sys, &pumping_station_id) in spec.entity_counts.pumping_station_ids.iter().enumerate() {
        let consumption = spec.pumping_consumption_mw_per_m3s[p_sys];
        for blk in 0..n_blks {
            let col = spec
                .geometry
                .pumping_flow_col(PumpingSys::new(p_sys), BlockIdx::new(blk));
            let pumped_flow_m3s = view.primal[col];
            #[allow(clippy::cast_possible_truncation)]
            results.push(SimulationPumpingResult {
                stage_id,
                block_id: Some(blk as u32),
                pumping_station_id,
                pumped_flow_m3s,
                power_consumption_mw: pumped_flow_m3s * consumption,
                pumping_cost: 0.0,
                operative_state_code: 1,
            });
        }
    }
    results
}

/// Extract one [`SimulationContractResult`] per (contract, block) from the solved
/// dispatch primals — dense, one row per system contract at every stage.
///
/// `spec.contract_slots[c]` gives the `(ContractType, family_slot)`
/// [`StageGeometry::contract_col`] addresses. `power_mw` is read directly from
/// `view.primal` (already unscaled). `price` is read PER BLOCK from `spec.contract_prices[c *
/// n_blks + blk]`, inside the `for blk` loop — hoisting it to `spec.contract_prices[c]`
/// above the loop compiles but silently misaligns every cell against the flat
/// per-block table. `total_cost = price * power_mw * block_hours` uses the
/// RESOLVED price, not the `col_scale`-scaled LP objective. A dormant `[0, 0]`-pinned
/// contract emits a ZERO row with `operative_state_code = 1`.
fn extract_contracts(
    view: &SolutionView<'_>,
    spec: &StageExtractionSpec<'_>,
    stage_id: u32,
) -> Vec<SimulationContractResult> {
    let n_contracts = spec.entity_counts.contract_ids.len();
    let n_blks = spec.geometry.n_blks;
    if n_contracts == 0 || n_blks == 0 {
        return Vec::new();
    }

    let import_end = spec.geometry.contract_import.end;
    let export_end = spec.geometry.contract_export.end;
    debug_assert!(
        view.primal.len() >= import_end && view.primal.len() >= export_end,
        "contract primal out of bounds: need import_end {import_end} / export_end {export_end}, have {}",
        view.primal.len()
    );
    debug_assert!(
        spec.contract_prices.len() == n_contracts * n_blks,
        "contract_prices stride mismatch: expected n_contracts * n_blks = {} ({n_contracts} * {n_blks}), got len {}",
        n_contracts * n_blks,
        spec.contract_prices.len()
    );

    let mut results = Vec::with_capacity(n_contracts * n_blks);
    for (c, &contract_id) in spec.entity_counts.contract_ids.iter().enumerate() {
        let (contract_type, family_slot) = spec.contract_slots[c];
        for blk in 0..n_blks {
            let col = spec
                .geometry
                .contract_col(contract_type, family_slot, BlockIdx::new(blk));
            let power_mw = view.primal[col];
            let dur = spec.block_hours[blk];
            let energy_mwh = power_mw * dur;
            let price = spec.contract_prices[c * n_blks + blk];
            let total_cost = price * energy_mwh;
            #[allow(clippy::cast_possible_truncation)]
            results.push(SimulationContractResult {
                stage_id,
                block_id: Some(blk as u32),
                contract_id,
                power_mw,
                price_per_mwh: price,
                total_cost,
                operative_state_code: 1,
            });
        }
    }
    results
}

/// Extract per-entity result collections grouped by their shared iteration pattern.
///
/// Inflow lags, pumping stations, and contracts read real primal values; the
/// pumping and contract reads are delegated to [`extract_pumping_stations`] and
/// [`extract_contracts`].
fn extract_stub_collections(
    view: &SolutionView<'_>,
    spec: &StageExtractionSpec<'_>,
    stage_id: u32,
) -> (
    Vec<SimulationInflowLagResult>,
    Vec<SimulationPumpingResult>,
    Vec<SimulationContractResult>,
) {
    let state = spec.state;
    let mut inflow_lags =
        Vec::with_capacity(spec.entity_counts.hydro_ids.len() * state.max_par_order);
    inflow_lags.extend(spec.entity_counts.hydro_ids.iter().enumerate().flat_map(
        |(h, &hydro_id)| {
            (0..state.max_par_order).map(move |l| {
                #[allow(clippy::cast_possible_truncation)]
                SimulationInflowLagResult {
                    stage_id,
                    hydro_id,
                    lag_index: l as u32,
                    inflow_m3s: view.primal[state.lag_incoming_col(l, HydroSys::new(h)).get()],
                }
            })
        },
    ));
    let pumping_stations = extract_pumping_stations(view, spec, stage_id);
    let contracts = extract_contracts(view, spec, stage_id);
    (inflow_lags, pumping_stations, contracts)
}

/// Add one stage's cost breakdown into a running per-category accumulator.
///
/// The five categories follow the breakdown in `ScenarioCategoryCosts`:
///
/// | Field              | Sum expression                                        |
/// |--------------------|-------------------------------------------------------|
/// | `resource_cost`    | `thermal_cost + anticipated_thermal_cost + contract_cost` |
/// | `recourse_cost`    | `deficit_cost + excess_cost`                          |
/// | `violation_cost`   | `storage_violation_cost + filling_target_cost`        |
/// |                    | `+ hydro_violation_cost + inflow_penalty_cost`        |
/// |                    | `+ generic_violation_cost`                            |
/// | `regularization_cost` | `spillage_cost + turbined_cost`               |
/// |                    | `+ curtailment_cost + exchange_cost`                  |
/// | `imputed_cost`     | `pumping_cost`                                        |
///
/// # Examples
///
/// ```
/// use cobre_sddp::simulation::types::{ScenarioCategoryCosts, SimulationCostResult};
/// use cobre_sddp::simulation::extraction::accumulate_category_costs;
///
/// let cost = SimulationCostResult {
///     stage_id: 0,
///     block_id: None,
///     total_cost: 1000.0,
///     immediate_cost: 800.0,
///     future_cost: 200.0,
///     discount_factor: 1.0,
///     thermal_cost: 400.0,
///     anticipated_thermal_cost: 0.0,
///     contract_cost: 100.0,
///     deficit_cost: 50.0,
///     excess_cost: 10.0,
///     storage_violation_cost: 20.0,
///     filling_target_cost: 30.0,
///     hydro_violation_cost: 5.0,
///     outflow_violation_below_cost: 0.0,
///     outflow_violation_above_cost: 0.0,
///     turbined_violation_cost: 0.0,
///     generation_violation_cost: 0.0,
///     evaporation_violation_cost: 0.0,
///     withdrawal_violation_cost: 0.0,
///     inflow_penalty_cost: 3.0,
///     generic_violation_cost: 2.0,
///     spillage_cost: 1.0,
///     turbined_cost: 4.0,
///     curtailment_cost: 7.0,
///     exchange_cost: 8.0,
///     pumping_cost: 60.0,
/// };
///
/// let mut accum = ScenarioCategoryCosts {
///     resource_cost: 0.0,
///     recourse_cost: 0.0,
///     violation_cost: 0.0,
///     regularization_cost: 0.0,
///     imputed_cost: 0.0,
/// };
///
/// accumulate_category_costs(&cost, &mut accum);
/// assert_eq!(accum.resource_cost, 500.0);       // 400 + 0 + 100
/// assert_eq!(accum.recourse_cost, 60.0);         // 50 + 10
/// assert_eq!(accum.violation_cost, 60.0);        // 20 + 30 + 5 + 3 + 2
/// assert_eq!(accum.regularization_cost, 20.0);   // 1 + 4 + 7 + 8
/// assert_eq!(accum.imputed_cost, 60.0);          // 60
/// ```
pub fn accumulate_category_costs(cost: &SimulationCostResult, accum: &mut ScenarioCategoryCosts) {
    // Anticipated thermal fuel rolls up as a resource cost; this is what keeps
    // Σ(macro categories) == immediate_cost.
    accum.resource_cost += cost.thermal_cost + cost.anticipated_thermal_cost + cost.contract_cost;
    accum.recourse_cost += cost.deficit_cost + cost.excess_cost;
    accum.violation_cost += cost.storage_violation_cost
        + cost.filling_target_cost
        + cost.hydro_violation_cost
        + cost.inflow_penalty_cost
        + cost.generic_violation_cost;
    accum.regularization_cost +=
        cost.spillage_cost + cost.turbined_cost + cost.curtailment_cost + cost.exchange_cost;
    accum.imputed_cost += cost.pumping_cost;
}

#[cfg(test)]
mod transit_seed_tests {
    use chrono::NaiveDate;
    use cobre_core::{EntityId, HydroPastDefluence};

    use super::{
        SimulationHydroResult, SimulationStageResult, SimulationTransitSeedResult, TransitSeedArc,
        build_transit_seed,
    };
    use crate::setup::NodeId;

    fn date(y: i32, m: u32, d: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, d).unwrap_or_else(|| unreachable!("hardcoded date is valid"))
    }

    /// Three 30-day stages: `[2024-01-01, 2024-01-31)`, `[2024-01-31,
    /// 2024-03-01)`, `[2024-03-01, 2024-03-31)` — 2160h horizon.
    fn three_stage_dates() -> Vec<(NaiveDate, NaiveDate)> {
        vec![
            (date(2024, 1, 1), date(2024, 1, 31)),
            (date(2024, 1, 31), date(2024, 3, 1)),
            (date(2024, 3, 1), date(2024, 3, 31)),
        ]
    }

    fn hydro_row(
        stage_id: u32,
        block_id: u32,
        hydro_id: i32,
        turbined: f64,
        spillage: f64,
    ) -> SimulationHydroResult {
        SimulationHydroResult {
            stage_id,
            block_id: Some(block_id),
            hydro_id,
            turbined_m3s: turbined,
            spillage_m3s: spillage,
            evaporation_m3s: None,
            diverted_inflow_m3s: None,
            diverted_outflow_m3s: None,
            incremental_inflow_m3s: 0.0,
            inflow_m3s: 0.0,
            storage_initial_hm3: 0.0,
            storage_final_hm3: 0.0,
            generation_mw: 0.0,
            equivalent_productivity_mw_per_m3s: 0.0,
            accumulated_productivity_mw_per_m3s: 0.0,
            incremental_inflow_energy_mw: 0.0,
            stored_energy_initial_mwh: 0.0,
            stored_energy_final_mwh: 0.0,
            spillage_cost: 0.0,
            water_value_per_hm3: 0.0,
            storage_binding_code: 0,
            operative_state_code: 0,
            turbined_slack_m3s: 0.0,
            outflow_slack_below_m3s: 0.0,
            outflow_slack_above_m3s: 0.0,
            generation_slack_mw: 0.0,
            storage_violation_below_hm3: 0.0,
            filling_target_violation_hm3: 0.0,
            evaporation_violation_pos_m3s: 0.0,
            evaporation_violation_neg_m3s: 0.0,
            inflow_nonnegativity_slack_m3s: 0.0,
            water_withdrawal_violation_pos_m3s: 0.0,
            water_withdrawal_violation_neg_m3s: 0.0,
            integrated_equivalent_productivity_mw_per_m3s: 0.0,
            integrated_accumulated_productivity_mw_per_m3s: 0.0,
            stored_energy_initial_mw: 0.0,
            stored_energy_final_mw: 0.0,
        }
    }

    /// One stage result carrying a single-block hydro row (`turbined +
    /// spillage` given directly as the whole-stage rate; block hours supplied
    /// separately via `block_hours_per_stage`).
    fn stage_with_hydro(stage_id: u32, hydro_id: i32, rate_m3s: f64) -> SimulationStageResult {
        SimulationStageResult {
            stage_id,
            node_id: NodeId(i32::try_from(stage_id).unwrap_or(0)),
            costs: vec![],
            hydros: vec![hydro_row(stage_id, 0, hydro_id, rate_m3s, 0.0)],
            hydro_bus_generation: vec![],
            thermals: vec![],
            exchanges: vec![],
            buses: vec![],
            pumping_stations: vec![],
            contracts: vec![],
            non_controllables: vec![],
            inflow_lags: vec![],
            transit_buckets: vec![],
            generic_violations: vec![],
            anticipated_lanes: vec![],
        }
    }

    const HYDRO_ID: i32 = 5;

    /// `t_v <= horizon`: the trailing window excludes a stage whose release
    /// matured EXACTLY at `t_v` (delivered in full, zero remaining in-transit
    /// mass) — only stages 1 and 2 overlap `[study_end - 1440h, study_end)`.
    #[test]
    fn in_study_only_excludes_the_stage_that_fully_matured_at_t_v() {
        let dates = three_stage_dates();
        let stages = vec![
            stage_with_hydro(0, HYDRO_ID, 10.0),
            stage_with_hydro(1, HYDRO_ID, 20.0),
            stage_with_hydro(2, HYDRO_ID, 30.0),
        ];
        let arcs = [TransitSeedArc {
            upstream_hydro_id: HYDRO_ID,
            travel_time_hours: 1440.0,
        }];
        let block_hours = vec![vec![720.0]; 3];

        let windows = build_transit_seed(&stages, &dates, &arcs, &[], &block_hours);

        assert_eq!(
            windows,
            vec![
                SimulationTransitSeedResult {
                    hydro_id: HYDRO_ID,
                    start_date: dates[1].0,
                    end_date: dates[1].1,
                    value_m3s: 20.0,
                },
                SimulationTransitSeedResult {
                    hydro_id: HYDRO_ID,
                    start_date: dates[2].0,
                    end_date: dates[2].1,
                    value_m3s: 30.0,
                },
            ],
            "stage 0 matured exactly at t_v and must be excluded; stages 1-2 must carry their \
             own release rate verbatim"
        );
    }

    /// `t_v > horizon` pulls in every in-study stage AND stitches the
    /// pre-study `past_defluences` tail — two additive window sources, never
    /// a `.find()` that would silently keep only the first past-defluence
    /// window.
    #[test]
    fn wide_t_v_stitches_every_in_study_stage_and_the_past_defluence_tail() {
        let dates = three_stage_dates();
        let stages = vec![
            stage_with_hydro(0, HYDRO_ID, 10.0),
            stage_with_hydro(1, HYDRO_ID, 20.0),
            stage_with_hydro(2, HYDRO_ID, 30.0),
        ];
        let arcs = [TransitSeedArc {
            upstream_hydro_id: HYDRO_ID,
            travel_time_hours: 3000.0,
        }];
        let block_hours = vec![vec![720.0]; 3];
        let past_defluences = vec![
            HydroPastDefluence {
                hydro_id: EntityId(HYDRO_ID),
                start_date: date(2023, 12, 2),
                end_date: date(2024, 1, 1),
                value_m3s: 99.0,
            },
            HydroPastDefluence {
                hydro_id: EntityId(HYDRO_ID),
                start_date: date(2023, 11, 1),
                end_date: date(2023, 12, 2),
                value_m3s: 88.0,
            },
            // A window ending well before the trailing span must be dropped.
            HydroPastDefluence {
                hydro_id: EntityId(HYDRO_ID),
                start_date: date(2020, 1, 1),
                end_date: date(2020, 2, 1),
                value_m3s: 1.0,
            },
        ];

        let windows = build_transit_seed(&stages, &dates, &arcs, &past_defluences, &block_hours);

        assert_eq!(
            windows,
            vec![
                SimulationTransitSeedResult {
                    hydro_id: HYDRO_ID,
                    start_date: dates[0].0,
                    end_date: dates[0].1,
                    value_m3s: 10.0,
                },
                SimulationTransitSeedResult {
                    hydro_id: HYDRO_ID,
                    start_date: dates[1].0,
                    end_date: dates[1].1,
                    value_m3s: 20.0,
                },
                SimulationTransitSeedResult {
                    hydro_id: HYDRO_ID,
                    start_date: dates[2].0,
                    end_date: dates[2].1,
                    value_m3s: 30.0,
                },
                SimulationTransitSeedResult {
                    hydro_id: HYDRO_ID,
                    start_date: date(2023, 12, 2),
                    end_date: date(2024, 1, 1),
                    value_m3s: 99.0,
                },
                SimulationTransitSeedResult {
                    hydro_id: HYDRO_ID,
                    start_date: date(2023, 11, 1),
                    end_date: date(2023, 12, 2),
                    value_m3s: 88.0,
                },
            ],
            "a wider t_v pulls in stage 0 too, and both non-contiguous past_defluence windows \
             must contribute independently; the stale pre-2020 window must be dropped"
        );
    }

    #[test]
    fn no_declared_arc_emits_no_windows() {
        let dates = three_stage_dates();
        let stages = vec![stage_with_hydro(0, HYDRO_ID, 10.0)];
        let past_defluences = vec![HydroPastDefluence {
            hydro_id: EntityId(HYDRO_ID),
            start_date: date(2023, 11, 1),
            end_date: date(2024, 1, 1),
            value_m3s: 99.0,
        }];

        let windows =
            build_transit_seed(&stages, &dates[..1], &[], &past_defluences, &[vec![720.0]]);

        assert!(
            windows.is_empty(),
            "no declared arc must emit no windows even with populated past_defluences"
        );
    }

    /// [`super::stage_release_rate_m3s`] duration-weights turbined + spillage
    /// across blocks, excluding diverted outflow — diverted water leaves via
    /// a different downstream target and never feeds this arc's deposit.
    #[test]
    fn stage_release_rate_is_duration_weighted_mean_excluding_diversion() {
        let mut stage = stage_with_hydro(0, HYDRO_ID, 0.0);
        stage.hydros = vec![
            hydro_row(0, 0, HYDRO_ID, 10.0, 5.0),
            hydro_row(0, 1, HYDRO_ID, 20.0, 0.0),
        ];
        stage.hydros[0].diverted_outflow_m3s = Some(1_000.0);
        let block_hours = [100.0, 300.0];

        let rate = super::stage_release_rate_m3s(&stage, HYDRO_ID, &block_hours);

        assert!(
            (rate - 18.75).abs() < 1e-9,
            "expected duration-weighted mean (100*15 + 300*20)/400 = 18.75, got {rate}"
        );
    }
}

#[cfg(test)]
mod tests;
