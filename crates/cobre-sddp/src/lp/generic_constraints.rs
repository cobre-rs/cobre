//! Variable reference to LP column index mapping for generic constraints.
//!
//! `resolve_variable_ref` maps a [`VariableRef`] and block index to a list of
//! `(column_index, coefficient_multiplier)` pairs; the LP builder calls it for each
//! [`cobre_core::LinearTerm`] of a generic-constraint expression to produce CSC
//! entries. Column offsets come from the [`GenericResolverGeom`] view (role-(a)
//! state region through its [`StateSpace`] handle, role-(b) equipment ranges
//! directly), with all block-stride arithmetic routed through the single-owner
//! [`BlockGrid`] primitive.
//!
//! For a block-level variable with `block_id = None`, the resolver returns the
//! column for the *current* `block_idx`; the caller loops over blocks and calls once
//! per block, so per-block expansion happens in the caller, not here.
//!
//! `PumpingPower` aliases the SAME flow column as `PumpingFlow`, scaled by the
//! station's `consumption_mw_per_m3s` — power is affine in flow, so a column of its
//! own would be an unconstrained free variable. Variables referencing entity types
//! with no LP columns (contracts, non-controllable sources, withdrawal) return an
//! empty vec.

use std::collections::{BTreeMap, HashMap};
use std::ops::Range;

use cobre_core::{
    CascadeTopology, ConstraintExpression, ContractType, EnergyContract, EntityId, PumpingStation,
    VariableRef,
};

use crate::hydro_models::{ProductionModelSet, ResolvedProductionModel};
use crate::indexer::{
    BlockGrid, BlockIdx, Boundary, EvaporationIndices, HydroCellIndex, HydroSys, StateSpace,
    StorageBoundaryGrid,
};
use crate::lp::builder::evaporation_slot;

/// Borrowed LP-column geometry the generic-constraint resolver reads — the
/// resolver's window onto a `StageLayout` (private to `builder`) without exposing it.
///
/// ## Role split (the load-bearing distinction for the cut path)
///
/// - **Role (a)** — `state`: storage and z-inflow columns, owned by [`StateSpace`].
///   Resolving a `HydroStorage`/`HydroInflow` term through the handle is what keeps a
///   generic constraint's storage/inflow column landing on the same column the cut
///   path reads.
/// - **Role (b)** — every other field: per-stage equipment/slack column ranges, the
///   block-stride constants, the FPHA/evaporation local maps, and the
///   anticipated-decision base + reverse map, riding this stage's own block count.
pub(crate) struct GenericResolverGeom<'a> {
    /// Role-(a) state-region handle (storage + z-inflow column owner).
    pub state: &'a StateSpace,
    /// Role-(a)-adjacent: storage-boundary address primitive, feeding
    /// [`Self::block_storage_col`].
    pub storage_boundary_grid: StorageBoundaryGrid,
    /// Study-scope hydro-cell partition, needed because the `Turbine` and FPHA
    /// `generation` families are sized and addressed by cell: a plant-level
    /// `VariableRef` (no group selector) resolves to the sum over the plant's
    /// cells, one `(col, coefficient)` pair per cell.
    pub hydro_cell_index: &'a HydroCellIndex,
    /// Turbine column range (one per cell per block).
    pub turbine: &'a Range<usize>,
    /// Spillage column range.
    pub spillage: &'a Range<usize>,
    /// Diversion column range.
    pub diversion: &'a Range<usize>,
    /// Thermal column range.
    pub thermal: &'a Range<usize>,
    /// Forward line-flow column range.
    pub line_fwd: &'a Range<usize>,
    /// Reverse line-flow column range.
    pub line_rev: &'a Range<usize>,
    /// Bus-excess column range.
    pub excess: &'a Range<usize>,
    /// Import-contract column range (one per import contract per block).
    pub contract_import: &'a Range<usize>,
    /// Export-contract column range (one per export contract per block).
    pub contract_export: &'a Range<usize>,
    /// FPHA generation column range.
    pub generation: &'a Range<usize>,
    /// Borrowed from `StageLayout`, which owns this prefix sum: FPHA-local plant
    /// index → that plant's first cell's FPHA-cell-local index. Borrowed rather
    /// than recomputed here because `resolver_geom` is a `StageLayout` method, so
    /// the owner is in scope; the independent copy in `HydroReverseLookup` exists
    /// only because extraction has no `StageLayout`.
    pub fpha_cell_local_start: &'a [usize],
    /// Bus-deficit column range.
    pub deficit: &'a Range<usize>,
    /// Deficit-stride constant (`S`).
    pub max_deficit_segments: usize,
    /// Per-stage block count (`K`); the `BlockGrid` flat/​deficit stride.
    pub n_blks: usize,
    /// Evaporation slots per evaporating hydro at this stage
    /// (`evaporation_slot_count`); the stride [`Self::evap_indices`] uses.
    pub n_evap_slots: usize,
    /// Per-evaporation-hydro column indices, parallel to
    /// [`Self::evap_hydro_indices`].
    pub evap_indices: &'a [EvaporationIndices],
    /// System hydro indices of the evaporation hydros at this stage.
    pub evap_hydro_indices: &'a [HydroSys],
    /// System hydro indices of the FPHA hydros at this stage.
    pub fpha_hydro_indices: &'a [HydroSys],
    /// First anticipated-decision column (`anticipated_decision.start`).
    pub anticipated_decision_start: usize,
    /// Reverse map: global thermal position → anticipated-local index.
    pub anticipated_local_by_sys_pos: &'a HashMap<usize, usize>,
}

impl GenericResolverGeom<'_> {
    /// The [`BlockGrid`] for this stage, built from the role-(b) stride constants.
    #[inline]
    fn block_grid(&self) -> BlockGrid {
        BlockGrid::new(self.n_blks, self.max_deficit_segments)
    }

    /// Storage column at chronological `boundary` for hydro `h`, so the
    /// resolver reaches per-block boundaries without a `StageLayout`;
    /// delegates to
    /// [`StorageBoundaryGrid::col`](crate::indexer::StorageBoundaryGrid::col),
    /// the single owner of the endpoints-vs-interior split.
    #[inline]
    fn block_storage_col(&self, h: HydroSys, boundary: Boundary) -> usize {
        self.storage_boundary_grid.col(self.state, h, boundary)
    }
}

/// Position maps for entity types, mapping entity IDs to their index in
/// the system's entity arrays.
///
/// Used by [`resolve_variable_ref`] to translate `VariableRef` entity IDs
/// into LP column offsets.
pub(crate) struct EntityPositionMaps<'a> {
    pub hydro: &'a BTreeMap<EntityId, usize>,
    pub thermal: &'a BTreeMap<EntityId, usize>,
    pub bus: &'a BTreeMap<EntityId, usize>,
    pub line: &'a BTreeMap<EntityId, usize>,
}

/// Borrowed cascade context for resolving the total-inflow expression. Consulted
/// only by the `HydroInflow` arm; grouped to keep [`resolve_variable_ref`] under the
/// `too_many_arguments` threshold.
pub(crate) struct CascadeRefs<'a> {
    /// Immediately-upstream cascade adjacency. `upstream(h)` returns
    /// `&[EntityId]` sorted by `EntityId.0` at build time, so the resolver
    /// iterates it in a fixed, input-ordering-independent sequence.
    pub cascade: &'a CascadeTopology,
    /// Target hydro id to the **system indices** of plants diverting into it,
    /// built in canonical hydro order (the same representation
    /// `fill_state_and_water_entries` iterates for the storage-balance
    /// diversion-inflow term).
    pub diversion_upstream: &'a HashMap<EntityId, Vec<usize>>,
}

/// Borrowed pumping context for resolving `PumpingFlow`/`PumpingPower`. Consulted
/// only by those two arms; grouped to keep [`resolve_variable_ref`] under the
/// `too_many_arguments` threshold.
pub(crate) struct PumpingRefs<'a> {
    /// First pumping-flow column (from `StageLayout::col_pumping_start`); meaningless
    /// when `pumping_stations` is empty, so the resolver guards on the `pumping_pos`
    /// lookup first.
    pub col_pumping_start: usize,
    /// Pumping stations in canonical ID-sorted slot order, indexed by the `p_idx`
    /// from [`PumpingRefs::pumping_pos`]; each entry's `consumption_mw_per_m3s` is the
    /// `PumpingPower` coefficient.
    pub pumping_stations: &'a [PumpingStation],
    /// Station id → local index (`p_idx`) into [`PumpingRefs::pumping_stations`].
    pub pumping_pos: &'a BTreeMap<EntityId, usize>,
}

/// Borrowed contract context for resolving `ContractImport`/`ContractExport`.
/// Consulted only by those two arms; grouped to keep [`resolve_variable_ref`] under
/// the `too_many_arguments` threshold.
pub(crate) struct ContractRefs<'a> {
    /// Energy contracts in canonical ID-sorted slot order (one slice for both
    /// directions); [`contract_family_slot`] derives a contract's per-family slot by
    /// counting same-direction contracts that precede it.
    pub contracts: &'a [EnergyContract],
    /// Contract id → combined slot into [`ContractRefs::contracts`].
    pub contract_pos: &'a BTreeMap<EntityId, usize>,
}

/// Map a [`VariableRef`] and block index to LP column indices with multipliers.
///
/// Returns a `Vec<(column_index, coefficient_multiplier)>`; the caller scales each
/// entry by the `LinearTerm::coefficient` for the final CSC value. `block_idx` is
/// ignored for stage-level variables and overridden by `block_id = Some(b)` for
/// block-level ones.
///
/// # Returns
///
/// An empty vec when the entity ID is absent from the relevant position map
/// (defense-in-depth past referential validation), or when the variable references
/// a stub entity with no LP columns (contracts, non-controllable sources,
/// withdrawal).
#[must_use]
pub(crate) fn resolve_variable_ref(
    var_ref: &VariableRef,
    block_idx: usize,
    stage_idx: usize,
    geom: &GenericResolverGeom<'_>,
    production_models: &ProductionModelSet,
    positions: &EntityPositionMaps<'_>,
    cascade_refs: &CascadeRefs<'_>,
    pumping_refs: &PumpingRefs<'_>,
    contract_refs: &ContractRefs<'_>,
) -> Vec<(usize, f64)> {
    let hydro_pos = positions.hydro;
    let thermal_pos = positions.thermal;
    let bus_pos = positions.bus;
    let line_pos = positions.line;
    let grid = geom.block_grid();
    match var_ref {
        VariableRef::HydroStorage { hydro_id } => resolve_hydro_storage(*hydro_id, geom, hydro_pos),

        VariableRef::HydroStorageInitial { hydro_id, block_id }
        | VariableRef::HydroUsefulVolumeInitial { hydro_id, block_id } => {
            resolve_hydro_storage_boundary(*hydro_id, *block_id, 0, geom, hydro_pos)
        }

        VariableRef::HydroStorageFinal { hydro_id, block_id }
        | VariableRef::HydroUsefulVolumeFinal { hydro_id, block_id } => {
            resolve_hydro_storage_boundary(*hydro_id, *block_id, 1, geom, hydro_pos)
        }

        VariableRef::HydroEvaporation { hydro_id, block_id } => {
            resolve_hydro_evaporation(*hydro_id, *block_id, geom, hydro_pos)
        }

        VariableRef::HydroInflow { hydro_id, block_id } => resolve_hydro_inflow(
            *hydro_id,
            *block_id,
            block_idx,
            grid,
            geom,
            hydro_pos,
            cascade_refs,
        ),

        VariableRef::HydroTurbined {
            hydro_id,
            block_id,
            bus_id,
        } => resolve_turbine_cells(
            *hydro_id, *block_id, *bus_id, block_idx, grid, geom, hydro_pos, 1.0,
        ),

        VariableRef::HydroSpillage { hydro_id, block_id } => {
            resolve_hydro_spillage(*hydro_id, *block_id, block_idx, grid, geom, hydro_pos)
        }

        VariableRef::HydroOutflow { hydro_id, block_id } => {
            resolve_hydro_outflow(*hydro_id, *block_id, block_idx, grid, geom, hydro_pos)
        }

        VariableRef::HydroGeneration {
            hydro_id,
            block_id,
            bus_id,
        } => resolve_hydro_generation(
            *hydro_id,
            *block_id,
            *bus_id,
            block_idx,
            grid,
            stage_idx,
            geom,
            production_models,
            hydro_pos,
        ),

        VariableRef::ThermalGeneration {
            thermal_id,
            block_id,
        } => resolve_thermal_generation(*thermal_id, *block_id, block_idx, grid, geom, thermal_pos),

        VariableRef::LineDirect { line_id, block_id } => {
            resolve_line_direct(*line_id, *block_id, block_idx, grid, geom, line_pos)
        }

        VariableRef::LineReverse { line_id, block_id } => {
            resolve_line_reverse(*line_id, *block_id, block_idx, grid, geom, line_pos)
        }

        VariableRef::LineExchange { line_id, block_id } => {
            resolve_line_exchange(*line_id, *block_id, block_idx, grid, geom, line_pos)
        }

        VariableRef::BusDeficit { bus_id, block_id } => {
            resolve_bus_deficit(*bus_id, *block_id, block_idx, grid, geom, bus_pos)
        }

        VariableRef::BusExcess { bus_id, block_id } => {
            resolve_bus_excess(*bus_id, *block_id, block_idx, grid, geom, bus_pos)
        }

        VariableRef::HydroDiversion { hydro_id, block_id } => {
            resolve_hydro_diversion(*hydro_id, *block_id, block_idx, grid, geom, hydro_pos)
        }

        VariableRef::AnticipatedDecision { thermal_id } => {
            resolve_anticipated_decision(*thermal_id, geom, thermal_pos)
        }

        VariableRef::PumpingFlow {
            station_id,
            block_id,
        } => resolve_pumping_column(
            *station_id,
            *block_id,
            block_idx,
            grid,
            pumping_refs,
            |_| 1.0,
        ),

        VariableRef::PumpingPower {
            station_id,
            block_id,
        } => resolve_pumping_column(
            *station_id,
            *block_id,
            block_idx,
            grid,
            pumping_refs,
            |station| station.consumption_mw_per_m3s,
        ),

        VariableRef::ContractImport {
            contract_id,
            block_id,
        } => resolve_contract_column(
            *contract_id,
            *block_id,
            block_idx,
            grid,
            block_col_range(geom, ElementKind::ContractImport).start,
            ContractType::Import,
            contract_refs,
        ),

        VariableRef::ContractExport {
            contract_id,
            block_id,
        } => resolve_contract_column(
            *contract_id,
            *block_id,
            block_idx,
            grid,
            block_col_range(geom, ElementKind::ContractExport).start,
            ContractType::Export,
            contract_refs,
        ),

        // Registered in the data model but no LP decision column: withdrawal is a
        // schedule fixed by bounds; non-controllable sources carry no decision column.
        VariableRef::HydroWithdrawal { .. }
        | VariableRef::NonControllableGeneration { .. }
        | VariableRef::NonControllableCurtailment { .. } => vec![],
    }
}

/// Whether a single [`VariableRef`] resolves to the *same* LP column(s) regardless
/// of `block_idx` — **block-independent** ("stock"). Seven kinds qualify:
/// [`VariableRef::HydroStorage`] (stage-final alias `Sᴷ`),
/// [`VariableRef::AnticipatedDecision`], [`VariableRef::HydroEvaporation`] (a fixed
/// single-block column or the all-block sum — both `block_idx`-independent), and the
/// four storage/useful-volume-boundary variants [`VariableRef::HydroStorageInitial`] /
/// [`VariableRef::HydroStorageFinal`] / [`VariableRef::HydroUsefulVolumeInitial`] /
/// [`VariableRef::HydroUsefulVolumeFinal`], each resolving to a fixed boundary column
/// (`Sᵏ` / `S⁰` / `Sᴷ`) that does not follow the materialized row's block.
///
/// [`VariableRef::HydroInflow`] is block-DEPENDENT: its upstream-release terms are
/// per-block columns. Classifying it "stock" would collapse a multi-block expression
/// to one mis-priced stage-level row reading upstream columns at a single arbitrary
/// block, silently dropping the other blocks. [`VariableRef::PumpingFlow`] /
/// [`VariableRef::PumpingPower`] are block-level for the same reason. The stub kinds
/// (withdrawal, contracts, non-controllable) resolve to no columns and are
/// conservatively block-level, so only *provably* stock variables enable the
/// single-row collapse.
///
/// The match is exhaustive (no wildcard): a future variant forces a compile error
/// here, defaulting nothing to "stock" by omission.
#[must_use]
fn variable_ref_is_block_independent(var_ref: &VariableRef) -> bool {
    match var_ref {
        VariableRef::HydroStorage { .. }
        | VariableRef::HydroStorageInitial { .. }
        | VariableRef::HydroStorageFinal { .. }
        | VariableRef::HydroUsefulVolumeInitial { .. }
        | VariableRef::HydroUsefulVolumeFinal { .. }
        | VariableRef::HydroEvaporation { .. }
        | VariableRef::AnticipatedDecision { .. } => true,
        VariableRef::HydroInflow { .. }
        | VariableRef::HydroTurbined { .. }
        | VariableRef::HydroSpillage { .. }
        | VariableRef::HydroDiversion { .. }
        | VariableRef::HydroOutflow { .. }
        | VariableRef::HydroGeneration { .. }
        | VariableRef::ThermalGeneration { .. }
        | VariableRef::LineDirect { .. }
        | VariableRef::LineReverse { .. }
        | VariableRef::LineExchange { .. }
        | VariableRef::BusDeficit { .. }
        | VariableRef::BusExcess { .. }
        | VariableRef::HydroWithdrawal { .. }
        | VariableRef::PumpingFlow { .. }
        | VariableRef::PumpingPower { .. }
        | VariableRef::ContractImport { .. }
        | VariableRef::ContractExport { .. }
        | VariableRef::NonControllableGeneration { .. }
        | VariableRef::NonControllableCurtailment { .. } => false,
    }
}

/// Whether **every** term of a generic-constraint expression is block-independent
/// (see [`variable_ref_is_block_independent`]), letting a `block_id = None` bound
/// collapse its per-block replication into one stage-level row. Any block-level term
/// forces `false`. An empty expression is vacuously true.
#[must_use]
pub(crate) fn expression_is_block_independent(expression: &ConstraintExpression) -> bool {
    expression
        .terms
        .iter()
        .all(|term| variable_ref_is_block_independent(&term.variable))
}

/// One `(column, multiplier)` pair per cell of the plant addressed by
/// `hydro_id` in the `Turbine` family, or exactly one pair when `bus_id` names
/// one of the plant's cells — the correct plant-level resolution now that the
/// family is sized by cell, not by plant: a `VariableRef` with `bus_id: None`
/// means the whole plant, i.e. every cell's column, matching today's single
/// pair under the identity partition. `Some(b)` resolves through
/// [`HydroCellIndex::cell_of_bus`], never `Hydro::bus_id`: the cell's bus comes
/// from a unit group's `bus_id`, an independent value the plant's own field
/// need not match. Returns an empty vec on a `hydro_pos` miss (mirrors every
/// other resolver's guard) or a `bus_id` naming no cell of the plant.
fn resolve_turbine_cells(
    hydro_id: EntityId,
    block_id: Option<usize>,
    bus_id: Option<EntityId>,
    block_idx: usize,
    grid: BlockGrid,
    geom: &GenericResolverGeom<'_>,
    hydro_pos: &BTreeMap<EntityId, usize>,
    multiplier: f64,
) -> Vec<(usize, f64)> {
    let Some(&pos) = hydro_pos.get(&hydro_id) else {
        return vec![];
    };
    let effective_blk = block_id.unwrap_or(block_idx);
    let turbine_start = block_col_range(geom, ElementKind::Turbine).start;
    let sys = HydroSys::new(pos);
    let flat = |c: usize| {
        (
            grid.flat(turbine_start, c, BlockIdx::new(effective_blk)),
            multiplier,
        )
    };
    match bus_id {
        Some(b) => geom
            .hydro_cell_index
            .cell_of_bus(sys, b)
            .map(|cell| vec![flat(cell.get())])
            .unwrap_or_default(),
        None => geom.hydro_cell_index.cells_of(sys).map(flat).collect(),
    }
}

/// Resolve `HydroStorage` to its stage-level outgoing storage column.
///
/// Role (a): the storage column is `state.storage_outgoing_col(h)`, read
/// through the state handle. Returns empty vec when the hydro ID is not found
/// in `hydro_pos`.
fn resolve_hydro_storage(
    hydro_id: EntityId,
    geom: &GenericResolverGeom<'_>,
    hydro_pos: &BTreeMap<EntityId, usize>,
) -> Vec<(usize, f64)> {
    if let Some(&pos) = hydro_pos.get(&hydro_id) {
        vec![(
            geom.state.storage_outgoing_col(HydroSys::new(pos)).get(),
            1.0,
        )]
    } else {
        vec![]
    }
}

/// Resolve `HydroStorageInitial`/`HydroStorageFinal` to a single fixed storage
/// boundary column via [`GenericResolverGeom::block_storage_col`].
/// `boundary_offset = 0` (initial): `Some(k)` → boundary `k`, `None` → stage-initial
/// `S⁰` (boundary `0`). `boundary_offset = 1` (final): `Some(k)` → boundary `k + 1`,
/// `None` → stage-final `Sᴷ` (boundary `K`). Both are stage-level stocks (fixed
/// column, no per-block expansion). Returns an empty vec on a `hydro_pos` miss
/// (mirrors [`resolve_hydro_storage`]).
fn resolve_hydro_storage_boundary(
    hydro_id: EntityId,
    block_id: Option<usize>,
    boundary_offset: usize,
    geom: &GenericResolverGeom<'_>,
    hydro_pos: &BTreeMap<EntityId, usize>,
) -> Vec<(usize, f64)> {
    if let Some(&pos) = hydro_pos.get(&hydro_id) {
        let k = match block_id {
            Some(k) => k + boundary_offset,
            None => boundary_offset * geom.n_blks,
        };
        let boundary = Boundary::from_index(k, geom.n_blks);
        vec![(geom.block_storage_col(HydroSys::new(pos), boundary), 1.0)]
    } else {
        vec![]
    }
}

/// Resolve `HydroInflow` to the cascade total-inflow expression at `eff_blk`: the
/// incremental (local) `z_inflow` column plus immediately-upstream releases (turbine
/// + spillage) plus plants diverting into `h`, all coefficient `+1.0`.
///
/// This is an instantaneous **rate** identity (m³/s), **not** the `−τ`-weighted (hm³)
/// storage-balance row — the `−τ` sign and `τ` weighting belong to storage balance
/// and must not be copied here. `h`'s own outflows, evaporation, withdrawal slacks,
/// AR-lag-`ψ`, and pumped transfer are excluded (storage-balance / loss / outflow
/// terms, or no LP column).
///
/// Upstream releases iterate `cascade.upstream(h)`; diverted inflow iterates
/// `diversion_upstream[h]` (values already system indices). Both are canonically
/// ordered at build time, so emitted pairs are input-ordering-independent with no
/// extra sort.
///
/// The `z_inflow.is_empty()` guard is load-bearing: `z_inflow` is empty when
/// `hydro_count == 0` (unlike `storage`), so `z_inflow_col` would be meaningless.
/// Returns an empty vec when `hydro_count == 0` or `hydro_id` is unknown.
fn resolve_hydro_inflow(
    hydro_id: EntityId,
    block_id: Option<usize>,
    block_idx: usize,
    grid: BlockGrid,
    geom: &GenericResolverGeom<'_>,
    hydro_pos: &BTreeMap<EntityId, usize>,
    cascade_refs: &CascadeRefs<'_>,
) -> Vec<(usize, f64)> {
    if geom.state.z_inflow.is_empty() {
        return vec![];
    }
    let Some(&pos_h) = hydro_pos.get(&hydro_id) else {
        return vec![];
    };

    let eff_blk = block_id.unwrap_or(block_idx);
    let upstream = cascade_refs.cascade.upstream(hydro_id);
    let diversion_into = cascade_refs
        .diversion_upstream
        .get(&hydro_id)
        .map_or(&[][..], Vec::as_slice);

    let mut result = Vec::with_capacity(1 + 2 * upstream.len() + diversion_into.len());

    result.push((geom.state.z_inflow_col(HydroSys::new(pos_h)).get(), 1.0));

    // Upstream releases (turbine + spillage): same column set as the storage-balance
    // inflow side but coefficient +1.0 (rate), not −τ (volume). Turbine sums every
    // one of the upstream plant's cells; spillage stays plant-keyed (unsplit).
    let turbine = block_col_range(geom, ElementKind::Turbine);
    let spillage = block_col_range(geom, ElementKind::Spillage);
    if !turbine.is_empty() && !spillage.is_empty() {
        for &up_id in upstream {
            if let Some(&pos_up) = hydro_pos.get(&up_id) {
                for cell in geom.hydro_cell_index.cells_of(HydroSys::new(pos_up)) {
                    result.push((grid.flat(turbine.start, cell, BlockIdx::new(eff_blk)), 1.0));
                }
                result.push((
                    grid.flat(spillage.start, pos_up, BlockIdx::new(eff_blk)),
                    1.0,
                ));
            }
        }
    }

    // `diversion_upstream[h]` already holds system indices, so no `hydro_pos` lookup
    // (mirrors the `fill_state_and_water_entries` diversion-inflow loop).
    let diversion = block_col_range(geom, ElementKind::Diversion);
    if !diversion.is_empty() {
        for &d_idx in diversion_into {
            result.push((
                grid.flat(diversion.start, d_idx, BlockIdx::new(eff_blk)),
                1.0,
            ));
        }
    }

    result
}

/// Resolve `HydroEvaporation` to the evaporation-outflow column for the matching
/// hydro; empty vec when the hydro has no linearized evaporation at this stage, or
/// when `block_id` names a block `>= n_blks`. `None` maps to block 0. On a
/// chronological stage each block resolves to its own slot; `None` in `K > 1`
/// (where blocks differ) is rejected upstream by generic-constraint validation, so
/// it is not reached here for a valid study. On a parallel stage `None`/`Some(0)`
/// resolve to the one stage-level slot; `Some(k >= 1)` is rejected by the same
/// validation, so the collapse below onto that slot for `k >= 1`
/// (`evaporation_slot`/`evaporation_slot_count`) is reached only by a study that
/// bypassed validation.
fn resolve_hydro_evaporation(
    hydro_id: EntityId,
    block_id: Option<usize>,
    geom: &GenericResolverGeom<'_>,
    hydro_pos: &BTreeMap<EntityId, usize>,
) -> Vec<(usize, f64)> {
    let Some(&sys_pos) = hydro_pos.get(&hydro_id) else {
        return vec![];
    };
    // Linear scan: cold template-build path over a handful of evap hydros, so an
    // O(1) reverse map is not warranted (unlike `resolve_anticipated_decision`).
    let Some(local_idx) = geom
        .evap_hydro_indices
        .iter()
        .position(|&p| p.get() == sys_pos)
    else {
        return vec![];
    };
    let blk = block_id.unwrap_or(0);
    if blk >= geom.n_blks {
        return vec![];
    }
    let slot = evaporation_slot(geom.n_evap_slots, BlockIdx::new(blk));
    let base = local_idx * geom.n_evap_slots;
    vec![(
        geom.evap_indices[base + slot.get()].evaporation_flow_col,
        1.0,
    )]
}

/// Resolve `HydroOutflow` to turbine (every cell of the plant, summed) plus
/// spillage (one plant-keyed column). A `hydro_pos` miss returns an empty vec,
/// never a partial reading.
fn resolve_hydro_outflow(
    hydro_id: EntityId,
    block_id: Option<usize>,
    block_idx: usize,
    grid: BlockGrid,
    geom: &GenericResolverGeom<'_>,
    hydro_pos: &BTreeMap<EntityId, usize>,
) -> Vec<(usize, f64)> {
    let Some(&pos) = hydro_pos.get(&hydro_id) else {
        return vec![];
    };
    let effective_blk = block_id.unwrap_or(block_idx);
    let mut result = resolve_turbine_cells(
        hydro_id, block_id, None, block_idx, grid, geom, hydro_pos, 1.0,
    );
    let spillage_col = grid.flat(
        block_col_range(geom, ElementKind::Spillage).start,
        pos,
        BlockIdx::new(effective_blk),
    );
    result.push((spillage_col, 1.0));
    result
}

/// Resolve `HydroGeneration` by dispatching on the production model.
///
/// - FPHA hydros: maps to the generation column at
///   `grid.flat(generation.start, fpha_local_idx, blk)` (via [`BlockGrid::flat`]),
///   one pair per cell, or exactly one when `bus_id` names a cell.
/// - Constant-productivity hydros: maps to the turbine column(s) scaled by
///   productivity, threading `bus_id` through [`resolve_turbine_cells`] — one
///   plant, one productivity (`ProductionModelSet::model` carries no group or
///   cell axis), so scaling the selected cell alone is exact.
fn resolve_hydro_generation(
    hydro_id: EntityId,
    block_id: Option<usize>,
    bus_id: Option<EntityId>,
    block_idx: usize,
    grid: BlockGrid,
    stage_idx: usize,
    geom: &GenericResolverGeom<'_>,
    production_models: &ProductionModelSet,
    hydro_pos: &BTreeMap<EntityId, usize>,
) -> Vec<(usize, f64)> {
    let Some(&sys_pos) = hydro_pos.get(&hydro_id) else {
        return vec![];
    };
    match production_models.model(sys_pos, stage_idx) {
        ResolvedProductionModel::Fpha { .. } => {
            // Linear scan: cold template-build path over a handful of FPHA hydros, so
            // an O(1) reverse map is not warranted (see `resolve_hydro_evaporation`).
            let Some(fpha_local_idx) = geom
                .fpha_hydro_indices
                .iter()
                .position(|&p| p.get() == sys_pos)
            else {
                return vec![];
            };
            let sys = HydroSys::new(sys_pos);
            let fpha_cell_start = geom.fpha_cell_local_start[fpha_local_idx];
            let effective_blk = block_id.unwrap_or(block_idx);
            let flat = |fpha_idx: usize| {
                (
                    grid.flat(
                        geom.generation.start,
                        fpha_idx,
                        BlockIdx::new(effective_blk),
                    ),
                    1.0,
                )
            };
            if let Some(b) = bus_id {
                geom.hydro_cell_index
                    .cell_of_bus(sys, b)
                    .map(|cell| {
                        // Offset within the PLANT'S OWN cell range, never the absolute
                        // cell index: `fpha_cell_local_start` prefixes FPHA plants only,
                        // so a leading non-FPHA plant makes the two diverge.
                        let offset = cell.get() - geom.hydro_cell_index.cells_of(sys).start;
                        vec![flat(fpha_cell_start + offset)]
                    })
                    .unwrap_or_default()
            } else {
                let n_cells = geom.hydro_cell_index.cells_of(sys).len();
                (0..n_cells).map(|i| flat(fpha_cell_start + i)).collect()
            }
        }
        ResolvedProductionModel::ConstantProductivity { productivity } => resolve_turbine_cells(
            hydro_id,
            block_id,
            bus_id,
            block_idx,
            grid,
            geom,
            hydro_pos,
            *productivity,
        ),
    }
}

/// Resolve `LineExchange` (net = forward − reverse) to two columns with signs.
fn resolve_line_exchange(
    line_id: EntityId,
    block_id: Option<usize>,
    block_idx: usize,
    grid: BlockGrid,
    geom: &GenericResolverGeom<'_>,
    line_pos: &BTreeMap<EntityId, usize>,
) -> Vec<(usize, f64)> {
    if let Some(&pos) = line_pos.get(&line_id) {
        let effective_blk = block_id.unwrap_or(block_idx);
        let fwd_col = grid.flat(
            block_col_range(geom, ElementKind::LineFwd).start,
            pos,
            BlockIdx::new(effective_blk),
        );
        let rev_col = grid.flat(
            block_col_range(geom, ElementKind::LineRev).start,
            pos,
            BlockIdx::new(effective_blk),
        );
        vec![(fwd_col, 1.0), (rev_col, -1.0)]
    } else {
        vec![]
    }
}

/// Resolve `BusDeficit` to one column per deficit segment via the 3-term
/// [`BlockGrid::deficit`] address. The segment count `S` comes from
/// `geom.max_deficit_segments` (the grid exposes no accessor for it).
fn resolve_bus_deficit(
    bus_id: EntityId,
    block_id: Option<usize>,
    block_idx: usize,
    grid: BlockGrid,
    geom: &GenericResolverGeom<'_>,
    bus_pos: &BTreeMap<EntityId, usize>,
) -> Vec<(usize, f64)> {
    if let Some(&b_pos) = bus_pos.get(&bus_id) {
        let effective_blk = block_id.unwrap_or(block_idx);
        (0..geom.max_deficit_segments)
            .map(|seg| {
                (
                    grid.deficit(geom.deficit.start, b_pos, seg, BlockIdx::new(effective_blk)),
                    1.0,
                )
            })
            .collect()
    } else {
        vec![]
    }
}

/// Resolve `AnticipatedDecision` to `anticipated_decision_start + local_idx`, the
/// per-plant stage-level decision column.
///
/// Returns an empty vec when `thermal_id` is not in `thermal_pos`, or the thermal's
/// position is not in `anticipated_local_by_sys_pos` (the thermal is not anticipated)
/// — both defense-in-depth past semantic validation
/// (`check_anticipated_decision_target_is_anticipated`).
fn resolve_anticipated_decision(
    thermal_id: EntityId,
    geom: &GenericResolverGeom<'_>,
    thermal_pos: &BTreeMap<EntityId, usize>,
) -> Vec<(usize, f64)> {
    let Some(&sys_pos) = thermal_pos.get(&thermal_id) else {
        return vec![];
    };
    if let Some(&local_idx) = geom.anticipated_local_by_sys_pos.get(&sys_pos) {
        vec![(geom.anticipated_decision_start + local_idx, 1.0)]
    } else {
        vec![]
    }
}

/// Resolve `PumpingFlow`/`PumpingPower` to the block-major pumping-flow column.
///
/// Both variants resolve to the SAME flow column — `PumpingPower` has no column of
/// its own; resolving it to a separate column would create an unconstrained free
/// variable. The coefficient comes from `coeff_fn`: `|_| 1.0` for flow,
/// `|s| s.consumption_mw_per_m3s` for power (power is affine in flow).
///
/// The address `grid.flat(col_pumping_start, p_idx, eff_blk)` uses the station's
/// SYSTEM index `p_idx`: under the dense layout the column block is system-indexed, so
/// the system index IS the correct column-block position at every stage (a dormant
/// station keeps its zeroed column).
///
/// Returns an empty vec on an unknown station or no stations (`pumping_pos` miss);
/// `n_pumping == 0` is handled by the same guard. No panic.
fn resolve_pumping_column(
    station_id: EntityId,
    block_id: Option<usize>,
    block_idx: usize,
    grid: BlockGrid,
    pumping_refs: &PumpingRefs<'_>,
    coeff_fn: impl Fn(&PumpingStation) -> f64,
) -> Vec<(usize, f64)> {
    let Some(&p_idx) = pumping_refs.pumping_pos.get(&station_id) else {
        return vec![];
    };
    // Guard rather than index to uphold no-panic if `pumping_pos` and
    // `pumping_stations` ever diverge (both built from the same ID-sorted slice).
    let Some(station) = pumping_refs.pumping_stations.get(p_idx) else {
        return vec![];
    };
    let eff_blk = block_id.unwrap_or(block_idx);
    let col = grid.flat(
        pumping_refs.col_pumping_start,
        p_idx,
        BlockIdx::new(eff_blk),
    );
    vec![(col, coeff_fn(station))]
}

/// A contract's [`ContractType`] and its PER-FAMILY slot — the count of
/// same-direction contracts that precede `c_sys` in the id-sorted `contracts` slice.
///
/// The dense column layout addresses each family by this per-family slot (the running
/// position within its own direction), NOT the combined slot, so both the LP-column
/// fill and the resolver must agree on it; sharing this one derivation keeps them
/// consistent.
pub(crate) fn contract_family_slot(
    contracts: &[EnergyContract],
    c_sys: usize,
) -> (ContractType, usize) {
    let contract_type = contracts[c_sys].contract_type;
    let family_slot = contracts[..c_sys]
        .iter()
        .filter(|c| c.contract_type == contract_type)
        .count();
    (contract_type, family_slot)
}

/// Resolve `ContractImport`/`ContractExport` to the block-major contract column.
///
/// The injection/withdrawal LOAD-BALANCE sign is owned by the load-balance fill, not
/// here: the resolved coefficient is the variable's own unit `+1.0` (the
/// generic-constraint coefficient is the user's), matching `resolve_pumping_column`'s
/// `|_| 1.0`.
///
/// The address `grid.flat(base, family_slot, eff_blk)` uses the contract's per-family
/// slot ([`contract_family_slot`]) so the import block precedes the export block under
/// the dense layout. A dormant (commissioning-window-inactive) contract keeps its
/// `[0, 0]` column, so the column always exists.
///
/// Returns an empty vec on an unknown contract id (`contract_pos` miss) or a
/// direction mismatch (the referenced family differs from the contract's
/// `contract_type` — a referential-validation gap), mirroring the pumping precedent.
/// No panic.
fn resolve_contract_column(
    contract_id: EntityId,
    block_id: Option<usize>,
    block_idx: usize,
    grid: BlockGrid,
    base: usize,
    family: ContractType,
    contract_refs: &ContractRefs<'_>,
) -> Vec<(usize, f64)> {
    let Some(&c_sys) = contract_refs.contract_pos.get(&contract_id) else {
        return vec![];
    };
    let Some(contract) = contract_refs.contracts.get(c_sys) else {
        return vec![];
    };
    if contract.contract_type != family {
        return vec![];
    }
    let (_, family_slot) = contract_family_slot(contract_refs.contracts, c_sys);
    let eff_blk = block_id.unwrap_or(block_idx);
    let col = grid.flat(base, family_slot, BlockIdx::new(eff_blk));
    vec![(col, 1.0)]
}

/// Resolve `HydroSpillage` via the single-column dispatcher.
fn resolve_hydro_spillage(
    hydro_id: EntityId,
    block_id: Option<usize>,
    block_idx: usize,
    grid: BlockGrid,
    geom: &GenericResolverGeom<'_>,
    hydro_pos: &BTreeMap<EntityId, usize>,
) -> Vec<(usize, f64)> {
    resolve_block_variable(
        hydro_id,
        block_id,
        block_idx,
        grid,
        block_col_range(geom, ElementKind::Spillage).start,
        hydro_pos,
    )
}

/// Resolve `HydroDiversion` via the single-column dispatcher.
fn resolve_hydro_diversion(
    hydro_id: EntityId,
    block_id: Option<usize>,
    block_idx: usize,
    grid: BlockGrid,
    geom: &GenericResolverGeom<'_>,
    hydro_pos: &BTreeMap<EntityId, usize>,
) -> Vec<(usize, f64)> {
    resolve_block_variable(
        hydro_id,
        block_id,
        block_idx,
        grid,
        block_col_range(geom, ElementKind::Diversion).start,
        hydro_pos,
    )
}

/// Resolve `ThermalGeneration` via the single-column dispatcher.
fn resolve_thermal_generation(
    thermal_id: EntityId,
    block_id: Option<usize>,
    block_idx: usize,
    grid: BlockGrid,
    geom: &GenericResolverGeom<'_>,
    thermal_pos: &BTreeMap<EntityId, usize>,
) -> Vec<(usize, f64)> {
    resolve_block_variable(
        thermal_id,
        block_id,
        block_idx,
        grid,
        block_col_range(geom, ElementKind::Thermal).start,
        thermal_pos,
    )
}

/// Resolve `LineDirect` via the single-column dispatcher.
fn resolve_line_direct(
    line_id: EntityId,
    block_id: Option<usize>,
    block_idx: usize,
    grid: BlockGrid,
    geom: &GenericResolverGeom<'_>,
    line_pos: &BTreeMap<EntityId, usize>,
) -> Vec<(usize, f64)> {
    resolve_block_variable(
        line_id,
        block_id,
        block_idx,
        grid,
        block_col_range(geom, ElementKind::LineFwd).start,
        line_pos,
    )
}

/// Resolve `LineReverse` via the single-column dispatcher.
fn resolve_line_reverse(
    line_id: EntityId,
    block_id: Option<usize>,
    block_idx: usize,
    grid: BlockGrid,
    geom: &GenericResolverGeom<'_>,
    line_pos: &BTreeMap<EntityId, usize>,
) -> Vec<(usize, f64)> {
    resolve_block_variable(
        line_id,
        block_id,
        block_idx,
        grid,
        block_col_range(geom, ElementKind::LineRev).start,
        line_pos,
    )
}

/// Resolve `BusExcess` via the single-column dispatcher.
fn resolve_bus_excess(
    bus_id: EntityId,
    block_id: Option<usize>,
    block_idx: usize,
    grid: BlockGrid,
    geom: &GenericResolverGeom<'_>,
    bus_pos: &BTreeMap<EntityId, usize>,
) -> Vec<(usize, f64)> {
    resolve_block_variable(
        bus_id,
        block_id,
        block_idx,
        grid,
        block_col_range(geom, ElementKind::Excess).start,
        bus_pos,
    )
}

/// Resolve a block-level LP variable to its `(column_index, 1.0)` pair via the
/// single-owner [`BlockGrid::flat`] address (`eff_blk = ref_block_id.unwrap_or(...)`);
/// empty vec on a `pos_map` miss.
fn resolve_block_variable(
    entity_id: EntityId,
    ref_block_id: Option<usize>,
    current_block_idx: usize,
    grid: BlockGrid,
    col_start: usize,
    pos_map: &BTreeMap<EntityId, usize>,
) -> Vec<(usize, f64)> {
    if let Some(&pos) = pos_map.get(&entity_id) {
        let effective_blk = ref_block_id.unwrap_or(current_block_idx);
        vec![(grid.flat(col_start, pos, BlockIdx::new(effective_blk)), 1.0)]
    } else {
        vec![]
    }
}

/// A block-major equipment/line/contract column family, each mapping to exactly one
/// range in [`block_col_range`].
///
/// Exhaustively matched there (no `_` arm): a new family is a compile error until its
/// range source is named, rather than silently resolving to whichever field a
/// hand-written `.start` read happened to pick.
#[derive(Clone, Copy)]
enum ElementKind {
    Turbine,
    Spillage,
    Diversion,
    Thermal,
    LineFwd,
    LineRev,
    Excess,
    ContractImport,
    ContractExport,
}

/// Map an [`ElementKind`] to its block-major column range on `geom` — the single
/// point pairing a family with its `StageLayout` range, so a wrong arm mapping (e.g.
/// `geom.spillage` for `Turbine`) is caught here once instead of open-coded and
/// silently wrong at each `col_start` read.
///
/// Returns an **owned** `Range<usize>` so an empty range is returnable without tying
/// the result's lifetime to `geom`; do NOT change this to `&Range<usize>`.
#[must_use]
fn block_col_range(geom: &GenericResolverGeom<'_>, kind: ElementKind) -> Range<usize> {
    match kind {
        ElementKind::Turbine => geom.turbine.clone(),
        ElementKind::Spillage => geom.spillage.clone(),
        ElementKind::Diversion => geom.diversion.clone(),
        ElementKind::Thermal => geom.thermal.clone(),
        ElementKind::LineFwd => geom.line_fwd.clone(),
        ElementKind::LineRev => geom.line_rev.clone(),
        ElementKind::Excess => geom.excess.clone(),
        ElementKind::ContractImport => geom.contract_import.clone(),
        ElementKind::ContractExport => geom.contract_export.clone(),
    }
}

#[cfg(test)]
mod tests;
