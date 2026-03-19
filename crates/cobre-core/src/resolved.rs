//! Pre-resolved penalty and bound containers for O(1) solver lookup.
//!
//! During input loading, the three-tier cascade (global defaults → entity overrides
//! → stage overrides) is evaluated once and the results stored in these containers.
//! Solvers and LP builders then query resolved values in constant time via direct
//! array indexing — no re-evaluation of the cascade at solve time.
//!
//! The storage layout for both [`ResolvedPenalties`] and [`ResolvedBounds`] uses a
//! flat `Vec<T>` with manual 2D indexing:
//! `data[entity_idx * n_stages + stage_idx]`
//!
//! This gives cache-friendly sequential access when iterating over stages for a
//! fixed entity (the inner loop pattern used by LP builders).
//!
//! # Population
//!
//! These containers are populated by `cobre-io` during the penalty/bound resolution
//! step. They are never modified after construction.
//!
//! # Note on deficit segments
//!
//! Bus deficit segments are **not** stage-varying (see Penalty System spec SS3).
//! The piecewise structure is too complex for per-stage override. Therefore
//! [`BusStagePenalties`] contains only `excess_cost`.

use std::collections::HashMap;
use std::ops::Range;

// ─── Per-(entity, stage) penalty structs ─────────────────────────────────────

/// All 11 hydro penalty values for a given (hydro, stage) pair.
///
/// This is the stage-resolved form of [`crate::HydroPenalties`]. All fields hold
/// the final effective penalty after the full three-tier cascade has been applied.
///
/// # Examples
///
/// ```
/// use cobre_core::resolved::HydroStagePenalties;
///
/// let p = HydroStagePenalties {
///     spillage_cost: 0.01,
///     diversion_cost: 0.02,
///     fpha_turbined_cost: 0.03,
///     storage_violation_below_cost: 1000.0,
///     filling_target_violation_cost: 5000.0,
///     turbined_violation_below_cost: 500.0,
///     outflow_violation_below_cost: 500.0,
///     outflow_violation_above_cost: 500.0,
///     generation_violation_below_cost: 500.0,
///     evaporation_violation_cost: 500.0,
///     water_withdrawal_violation_cost: 500.0,
/// };
/// // Copy-semantics: can be passed by value
/// let q = p;
/// assert!((q.spillage_cost - 0.01).abs() < f64::EPSILON);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct HydroStagePenalties {
    /// Spillage regularization cost \[$/m³/s\]. Prefer turbining over spilling.
    pub spillage_cost: f64,
    /// Diversion regularization cost \[$/m³/s\]. Prefer main-channel flow.
    pub diversion_cost: f64,
    /// FPHA turbined regularization cost \[$/`MWh`\]. Prevents interior FPHA solutions.
    /// Must be `> spillage_cost` for FPHA hydros.
    pub fpha_turbined_cost: f64,
    /// Constraint-violation cost for storage below dead volume \[$/hm³\].
    pub storage_violation_below_cost: f64,
    /// Constraint-violation cost for missing the dead-volume filling target \[$/hm³\].
    /// Must be the highest penalty in the system.
    pub filling_target_violation_cost: f64,
    /// Constraint-violation cost for turbined flow below minimum \[$/m³/s\].
    pub turbined_violation_below_cost: f64,
    /// Constraint-violation cost for outflow below environmental minimum \[$/m³/s\].
    pub outflow_violation_below_cost: f64,
    /// Constraint-violation cost for outflow above flood-control limit \[$/m³/s\].
    pub outflow_violation_above_cost: f64,
    /// Constraint-violation cost for generation below contractual minimum \[$/MW\].
    pub generation_violation_below_cost: f64,
    /// Constraint-violation cost for evaporation constraint violation \[$/mm\].
    pub evaporation_violation_cost: f64,
    /// Constraint-violation cost for unmet water withdrawal \[$/m³/s\].
    pub water_withdrawal_violation_cost: f64,
}

/// Bus penalty values for a given (bus, stage) pair.
///
/// Contains only `excess_cost` because deficit segments are **not** stage-varying
/// (Penalty System spec SS3). The piecewise-linear deficit structure is fixed at
/// the entity or global level and applies uniformly across all stages.
///
/// # Examples
///
/// ```
/// use cobre_core::resolved::BusStagePenalties;
///
/// let p = BusStagePenalties { excess_cost: 0.01 };
/// let q = p; // Copy
/// assert!((q.excess_cost - 0.01).abs() < f64::EPSILON);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BusStagePenalties {
    /// Excess generation absorption cost \[$/`MWh`\].
    pub excess_cost: f64,
}

/// Line penalty values for a given (line, stage) pair.
///
/// # Examples
///
/// ```
/// use cobre_core::resolved::LineStagePenalties;
///
/// let p = LineStagePenalties { exchange_cost: 0.5 };
/// let q = p; // Copy
/// assert!((q.exchange_cost - 0.5).abs() < f64::EPSILON);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LineStagePenalties {
    /// Flow regularization cost \[$/`MWh`\]. Discourages unnecessary exchange.
    pub exchange_cost: f64,
}

/// Non-controllable source penalty values for a given (source, stage) pair.
///
/// # Examples
///
/// ```
/// use cobre_core::resolved::NcsStagePenalties;
///
/// let p = NcsStagePenalties { curtailment_cost: 10.0 };
/// let q = p; // Copy
/// assert!((q.curtailment_cost - 10.0).abs() < f64::EPSILON);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NcsStagePenalties {
    /// Curtailment regularization cost \[$/`MWh`\]. Penalizes curtailing available generation.
    pub curtailment_cost: f64,
}

// ─── Per-(entity, stage) bound structs ───────────────────────────────────────

/// All hydro bound values for a given (hydro, stage) pair.
///
/// The 11 fields match the 11 rows in spec SS11 hydro bounds table. These are
/// the fully resolved bounds after base values from `hydros.json` have been
/// overlaid with any stage-specific overrides from `constraints/hydro_bounds.parquet`.
///
/// `max_outflow_m3s` is `Option<f64>` because the outflow upper bound may be absent
/// (unbounded above) when no flood-control limit is defined for the hydro.
/// `water_withdrawal_m3s` defaults to `0.0` when no per-stage override is present.
///
/// # Examples
///
/// ```
/// use cobre_core::resolved::HydroStageBounds;
///
/// let b = HydroStageBounds {
///     min_storage_hm3: 10.0,
///     max_storage_hm3: 200.0,
///     min_turbined_m3s: 0.0,
///     max_turbined_m3s: 500.0,
///     min_outflow_m3s: 5.0,
///     max_outflow_m3s: None,
///     min_generation_mw: 0.0,
///     max_generation_mw: 100.0,
///     max_diversion_m3s: None,
///     filling_inflow_m3s: 0.0,
///     water_withdrawal_m3s: 0.0,
/// };
/// assert!((b.min_storage_hm3 - 10.0).abs() < f64::EPSILON);
/// assert!(b.max_outflow_m3s.is_none());
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct HydroStageBounds {
    /// Minimum reservoir storage — dead volume \[hm³\]. Soft lower bound;
    /// violation uses `storage_violation_below` slack.
    pub min_storage_hm3: f64,
    /// Maximum reservoir storage — physical capacity \[hm³\]. Hard upper bound;
    /// emergency spillage handles excess.
    pub max_storage_hm3: f64,
    /// Minimum turbined flow \[m³/s\]. Soft lower bound;
    /// violation uses `turbined_violation_below` slack.
    pub min_turbined_m3s: f64,
    /// Maximum turbined flow \[m³/s\]. Hard upper bound.
    pub max_turbined_m3s: f64,
    /// Minimum outflow — environmental flow requirement \[m³/s\]. Soft lower bound;
    /// violation uses `outflow_violation_below` slack.
    pub min_outflow_m3s: f64,
    /// Maximum outflow — flood-control limit \[m³/s\]. Soft upper bound;
    /// violation uses `outflow_violation_above` slack. `None` = unbounded.
    pub max_outflow_m3s: Option<f64>,
    /// Minimum generation \[MW\]. Soft lower bound;
    /// violation uses `generation_violation_below` slack.
    pub min_generation_mw: f64,
    /// Maximum generation \[MW\]. Hard upper bound.
    pub max_generation_mw: f64,
    /// Maximum diversion flow \[m³/s\]. Hard upper bound. `None` = no diversion channel.
    pub max_diversion_m3s: Option<f64>,
    /// Filling inflow retained for dead-volume filling during filling stages \[m³/s\].
    /// Resolved from entity default → stage override cascade. Default `0.0`.
    pub filling_inflow_m3s: f64,
    /// Water withdrawal from reservoir per stage \[m³/s\]. Positive = water removed;
    /// negative = external addition. Default `0.0`.
    pub water_withdrawal_m3s: f64,
}

/// Thermal bound values for a given (thermal, stage) pair.
///
/// Resolved from base values in `thermals.json` with optional per-stage overrides
/// from `constraints/thermal_bounds.parquet`.
///
/// # Examples
///
/// ```
/// use cobre_core::resolved::ThermalStageBounds;
///
/// let b = ThermalStageBounds { min_generation_mw: 50.0, max_generation_mw: 400.0 };
/// let c = b; // Copy
/// assert!((c.max_generation_mw - 400.0).abs() < f64::EPSILON);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ThermalStageBounds {
    /// Minimum stable generation \[MW\]. Hard lower bound.
    pub min_generation_mw: f64,
    /// Maximum generation capacity \[MW\]. Hard upper bound.
    pub max_generation_mw: f64,
}

/// Transmission line bound values for a given (line, stage) pair.
///
/// Resolved from base values in `lines.json` with optional per-stage overrides
/// from `constraints/line_bounds.parquet`. Note that block-level exchange factors
/// (per-block capacity multipliers) are stored separately and applied on top of
/// these stage-level bounds at LP construction time.
///
/// # Examples
///
/// ```
/// use cobre_core::resolved::LineStageBounds;
///
/// let b = LineStageBounds { direct_mw: 1000.0, reverse_mw: 800.0 };
/// let c = b; // Copy
/// assert!((c.direct_mw - 1000.0).abs() < f64::EPSILON);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LineStageBounds {
    /// Maximum direct flow capacity \[MW\]. Hard upper bound.
    pub direct_mw: f64,
    /// Maximum reverse flow capacity \[MW\]. Hard upper bound.
    pub reverse_mw: f64,
}

/// Pumping station bound values for a given (pumping, stage) pair.
///
/// Resolved from base values in `pumping_stations.json` with optional per-stage
/// overrides from `constraints/pumping_bounds.parquet`.
///
/// # Examples
///
/// ```
/// use cobre_core::resolved::PumpingStageBounds;
///
/// let b = PumpingStageBounds { min_flow_m3s: 0.0, max_flow_m3s: 50.0 };
/// let c = b; // Copy
/// assert!((c.max_flow_m3s - 50.0).abs() < f64::EPSILON);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PumpingStageBounds {
    /// Minimum pumped flow \[m³/s\]. Hard lower bound.
    pub min_flow_m3s: f64,
    /// Maximum pumped flow \[m³/s\]. Hard upper bound.
    pub max_flow_m3s: f64,
}

/// Energy contract bound values for a given (contract, stage) pair.
///
/// Resolved from base values in `energy_contracts.json` with optional per-stage
/// overrides from `constraints/contract_bounds.parquet`. The price field can also
/// be stage-varying.
///
/// # Examples
///
/// ```
/// use cobre_core::resolved::ContractStageBounds;
///
/// let b = ContractStageBounds { min_mw: 0.0, max_mw: 200.0, price_per_mwh: 80.0 };
/// let c = b; // Copy
/// assert!((c.max_mw - 200.0).abs() < f64::EPSILON);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ContractStageBounds {
    /// Minimum contract usage \[MW\]. Hard lower bound.
    pub min_mw: f64,
    /// Maximum contract usage \[MW\]. Hard upper bound.
    pub max_mw: f64,
    /// Effective contract price \[$/`MWh`\]. May differ from base when a stage override
    /// supplies a per-stage price.
    pub price_per_mwh: f64,
}

// ─── Pre-resolved containers ──────────────────────────────────────────────────

/// Pre-resolved penalty table for all entities across all stages.
///
/// Populated by `cobre-io` after the three-tier penalty cascade is applied.
/// Provides O(1) lookup via direct array indexing.
///
/// Internal layout: `data[entity_idx * n_stages + stage_idx]` — iterating
/// stages for a fixed entity accesses a contiguous memory region.
///
/// # Construction
///
/// Use [`ResolvedPenalties::new`] to allocate the table with a given default
/// value, then populate by writing into the flat slice returned by the internal
/// accessors. `cobre-io` is responsible for filling the data.
///
/// # Examples
///
/// ```
/// use cobre_core::resolved::{
///     BusStagePenalties, HydroStagePenalties, LineStagePenalties,
///     NcsStagePenalties, ResolvedPenalties,
/// };
///
/// let hydro_default = HydroStagePenalties {
///     spillage_cost: 0.01,
///     diversion_cost: 0.02,
///     fpha_turbined_cost: 0.03,
///     storage_violation_below_cost: 1000.0,
///     filling_target_violation_cost: 5000.0,
///     turbined_violation_below_cost: 500.0,
///     outflow_violation_below_cost: 500.0,
///     outflow_violation_above_cost: 500.0,
///     generation_violation_below_cost: 500.0,
///     evaporation_violation_cost: 500.0,
///     water_withdrawal_violation_cost: 500.0,
/// };
/// let bus_default = BusStagePenalties { excess_cost: 100.0 };
/// let line_default = LineStagePenalties { exchange_cost: 5.0 };
/// let ncs_default = NcsStagePenalties { curtailment_cost: 50.0 };
///
/// let table = ResolvedPenalties::new(
///     3, 2, 1, 4, 5,
///     hydro_default, bus_default, line_default, ncs_default,
/// );
///
/// // Hydro 1, stage 2 returns the default penalties.
/// let p = table.hydro_penalties(1, 2);
/// assert!((p.spillage_cost - 0.01).abs() < f64::EPSILON);
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ResolvedPenalties {
    /// Total number of stages. Used to compute flat indices.
    n_stages: usize,
    /// Flat `n_hydros * n_stages` array indexed `[hydro_idx * n_stages + stage_idx]`.
    hydro: Vec<HydroStagePenalties>,
    /// Flat `n_buses * n_stages` array indexed `[bus_idx * n_stages + stage_idx]`.
    bus: Vec<BusStagePenalties>,
    /// Flat `n_lines * n_stages` array indexed `[line_idx * n_stages + stage_idx]`.
    line: Vec<LineStagePenalties>,
    /// Flat `n_ncs * n_stages` array indexed `[ncs_idx * n_stages + stage_idx]`.
    ncs: Vec<NcsStagePenalties>,
}

impl ResolvedPenalties {
    /// Return an empty penalty table with zero entities and zero stages.
    ///
    /// Used as the default value in [`System`](crate::System) when no penalty
    /// resolution has been performed yet (e.g., when building a `System` from
    /// raw entity collections without `cobre-io`).
    ///
    /// # Examples
    ///
    /// ```
    /// use cobre_core::ResolvedPenalties;
    ///
    /// let empty = ResolvedPenalties::empty();
    /// assert_eq!(empty.n_stages(), 0);
    /// ```
    #[must_use]
    pub fn empty() -> Self {
        Self {
            n_stages: 0,
            hydro: Vec::new(),
            bus: Vec::new(),
            line: Vec::new(),
            ncs: Vec::new(),
        }
    }

    /// Allocate a new resolved-penalties table filled with the given defaults.
    ///
    /// `n_stages` must be `> 0`. Entity counts may be `0` (empty vectors are valid).
    ///
    /// # Arguments
    ///
    /// * `n_hydros` — number of hydro plants
    /// * `n_buses` — number of buses
    /// * `n_lines` — number of transmission lines
    /// * `n_ncs` — number of non-controllable sources
    /// * `n_stages` — number of study stages
    /// * `hydro_default` — initial value for all (hydro, stage) cells
    /// * `bus_default` — initial value for all (bus, stage) cells
    /// * `line_default` — initial value for all (line, stage) cells
    /// * `ncs_default` — initial value for all (ncs, stage) cells
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n_hydros: usize,
        n_buses: usize,
        n_lines: usize,
        n_ncs: usize,
        n_stages: usize,
        hydro_default: HydroStagePenalties,
        bus_default: BusStagePenalties,
        line_default: LineStagePenalties,
        ncs_default: NcsStagePenalties,
    ) -> Self {
        Self {
            n_stages,
            hydro: vec![hydro_default; n_hydros * n_stages],
            bus: vec![bus_default; n_buses * n_stages],
            line: vec![line_default; n_lines * n_stages],
            ncs: vec![ncs_default; n_ncs * n_stages],
        }
    }

    /// Return the resolved penalties for a hydro plant at a specific stage.
    #[inline]
    #[must_use]
    pub fn hydro_penalties(&self, hydro_index: usize, stage_index: usize) -> HydroStagePenalties {
        self.hydro[hydro_index * self.n_stages + stage_index]
    }

    /// Return the resolved penalties for a bus at a specific stage.
    #[inline]
    #[must_use]
    pub fn bus_penalties(&self, bus_index: usize, stage_index: usize) -> BusStagePenalties {
        self.bus[bus_index * self.n_stages + stage_index]
    }

    /// Return the resolved penalties for a line at a specific stage.
    #[inline]
    #[must_use]
    pub fn line_penalties(&self, line_index: usize, stage_index: usize) -> LineStagePenalties {
        self.line[line_index * self.n_stages + stage_index]
    }

    /// Return the resolved penalties for a non-controllable source at a specific stage.
    #[inline]
    #[must_use]
    pub fn ncs_penalties(&self, ncs_index: usize, stage_index: usize) -> NcsStagePenalties {
        self.ncs[ncs_index * self.n_stages + stage_index]
    }

    /// Return a mutable reference to the hydro penalty cell for in-place update.
    ///
    /// Used by `cobre-io` during penalty cascade resolution to set resolved values.
    #[inline]
    pub fn hydro_penalties_mut(
        &mut self,
        hydro_index: usize,
        stage_index: usize,
    ) -> &mut HydroStagePenalties {
        let idx = hydro_index * self.n_stages + stage_index;
        &mut self.hydro[idx]
    }

    /// Return a mutable reference to the bus penalty cell for in-place update.
    #[inline]
    pub fn bus_penalties_mut(
        &mut self,
        bus_index: usize,
        stage_index: usize,
    ) -> &mut BusStagePenalties {
        let idx = bus_index * self.n_stages + stage_index;
        &mut self.bus[idx]
    }

    /// Return a mutable reference to the line penalty cell for in-place update.
    #[inline]
    pub fn line_penalties_mut(
        &mut self,
        line_index: usize,
        stage_index: usize,
    ) -> &mut LineStagePenalties {
        let idx = line_index * self.n_stages + stage_index;
        &mut self.line[idx]
    }

    /// Return a mutable reference to the NCS penalty cell for in-place update.
    #[inline]
    pub fn ncs_penalties_mut(
        &mut self,
        ncs_index: usize,
        stage_index: usize,
    ) -> &mut NcsStagePenalties {
        let idx = ncs_index * self.n_stages + stage_index;
        &mut self.ncs[idx]
    }

    /// Return the number of stages in this table.
    #[inline]
    #[must_use]
    pub fn n_stages(&self) -> usize {
        self.n_stages
    }
}

/// Pre-resolved bound table for all entities across all stages.
///
/// Populated by `cobre-io` after base bounds are overlaid with stage-specific
/// overrides. Provides O(1) lookup via direct array indexing.
///
/// Internal layout: `data[entity_idx * n_stages + stage_idx]`.
///
/// # Examples
///
/// ```
/// use cobre_core::resolved::{
///     ContractStageBounds, HydroStageBounds, LineStageBounds,
///     PumpingStageBounds, ResolvedBounds, ThermalStageBounds,
/// };
///
/// let hydro_default = HydroStageBounds {
///     min_storage_hm3: 0.0, max_storage_hm3: 100.0,
///     min_turbined_m3s: 0.0, max_turbined_m3s: 50.0,
///     min_outflow_m3s: 0.0, max_outflow_m3s: None,
///     min_generation_mw: 0.0, max_generation_mw: 30.0,
///     max_diversion_m3s: None,
///     filling_inflow_m3s: 0.0, water_withdrawal_m3s: 0.0,
/// };
/// let thermal_default = ThermalStageBounds { min_generation_mw: 0.0, max_generation_mw: 100.0 };
/// let line_default = LineStageBounds { direct_mw: 500.0, reverse_mw: 500.0 };
/// let pumping_default = PumpingStageBounds { min_flow_m3s: 0.0, max_flow_m3s: 20.0 };
/// let contract_default = ContractStageBounds { min_mw: 0.0, max_mw: 50.0, price_per_mwh: 80.0 };
///
/// let table = ResolvedBounds::new(
///     2, 1, 1, 1, 1, 3,
///     hydro_default, thermal_default, line_default, pumping_default, contract_default,
/// );
///
/// let b = table.hydro_bounds(0, 2);
/// assert!((b.max_storage_hm3 - 100.0).abs() < f64::EPSILON);
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ResolvedBounds {
    /// Total number of stages. Used to compute flat indices.
    n_stages: usize,
    /// Flat `n_hydros * n_stages` array indexed `[hydro_idx * n_stages + stage_idx]`.
    hydro: Vec<HydroStageBounds>,
    /// Flat `n_thermals * n_stages` array indexed `[thermal_idx * n_stages + stage_idx]`.
    thermal: Vec<ThermalStageBounds>,
    /// Flat `n_lines * n_stages` array indexed `[line_idx * n_stages + stage_idx]`.
    line: Vec<LineStageBounds>,
    /// Flat `n_pumping * n_stages` array indexed `[pumping_idx * n_stages + stage_idx]`.
    pumping: Vec<PumpingStageBounds>,
    /// Flat `n_contracts * n_stages` array indexed `[contract_idx * n_stages + stage_idx]`.
    contract: Vec<ContractStageBounds>,
}

impl ResolvedBounds {
    /// Return an empty bounds table with zero entities and zero stages.
    ///
    /// Used as the default value in [`System`](crate::System) when no bound
    /// resolution has been performed yet (e.g., when building a `System` from
    /// raw entity collections without `cobre-io`).
    ///
    /// # Examples
    ///
    /// ```
    /// use cobre_core::ResolvedBounds;
    ///
    /// let empty = ResolvedBounds::empty();
    /// assert_eq!(empty.n_stages(), 0);
    /// ```
    #[must_use]
    pub fn empty() -> Self {
        Self {
            n_stages: 0,
            hydro: Vec::new(),
            thermal: Vec::new(),
            line: Vec::new(),
            pumping: Vec::new(),
            contract: Vec::new(),
        }
    }

    /// Allocate a new resolved-bounds table filled with the given defaults.
    ///
    /// `n_stages` must be `> 0`. Entity counts may be `0`.
    ///
    /// # Arguments
    ///
    /// * `n_hydros` — number of hydro plants
    /// * `n_thermals` — number of thermal units
    /// * `n_lines` — number of transmission lines
    /// * `n_pumping` — number of pumping stations
    /// * `n_contracts` — number of energy contracts
    /// * `n_stages` — number of study stages
    /// * `hydro_default` — initial value for all (hydro, stage) cells
    /// * `thermal_default` — initial value for all (thermal, stage) cells
    /// * `line_default` — initial value for all (line, stage) cells
    /// * `pumping_default` — initial value for all (pumping, stage) cells
    /// * `contract_default` — initial value for all (contract, stage) cells
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n_hydros: usize,
        n_thermals: usize,
        n_lines: usize,
        n_pumping: usize,
        n_contracts: usize,
        n_stages: usize,
        hydro_default: HydroStageBounds,
        thermal_default: ThermalStageBounds,
        line_default: LineStageBounds,
        pumping_default: PumpingStageBounds,
        contract_default: ContractStageBounds,
    ) -> Self {
        Self {
            n_stages,
            hydro: vec![hydro_default; n_hydros * n_stages],
            thermal: vec![thermal_default; n_thermals * n_stages],
            line: vec![line_default; n_lines * n_stages],
            pumping: vec![pumping_default; n_pumping * n_stages],
            contract: vec![contract_default; n_contracts * n_stages],
        }
    }

    /// Return the resolved bounds for a hydro plant at a specific stage.
    ///
    /// Returns a shared reference to avoid copying the 11-field struct on hot paths.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `hydro_index >= n_hydros` or `stage_index >= n_stages`.
    #[inline]
    #[must_use]
    pub fn hydro_bounds(&self, hydro_index: usize, stage_index: usize) -> &HydroStageBounds {
        &self.hydro[hydro_index * self.n_stages + stage_index]
    }

    /// Return the resolved bounds for a thermal unit at a specific stage.
    #[inline]
    #[must_use]
    pub fn thermal_bounds(&self, thermal_index: usize, stage_index: usize) -> ThermalStageBounds {
        self.thermal[thermal_index * self.n_stages + stage_index]
    }

    /// Return the resolved bounds for a transmission line at a specific stage.
    #[inline]
    #[must_use]
    pub fn line_bounds(&self, line_index: usize, stage_index: usize) -> LineStageBounds {
        self.line[line_index * self.n_stages + stage_index]
    }

    /// Return the resolved bounds for a pumping station at a specific stage.
    #[inline]
    #[must_use]
    pub fn pumping_bounds(&self, pumping_index: usize, stage_index: usize) -> PumpingStageBounds {
        self.pumping[pumping_index * self.n_stages + stage_index]
    }

    /// Return the resolved bounds for an energy contract at a specific stage.
    #[inline]
    #[must_use]
    pub fn contract_bounds(
        &self,
        contract_index: usize,
        stage_index: usize,
    ) -> ContractStageBounds {
        self.contract[contract_index * self.n_stages + stage_index]
    }

    /// Return a mutable reference to the hydro bounds cell for in-place update.
    ///
    /// Used by `cobre-io` during bound resolution to set stage-specific overrides.
    #[inline]
    pub fn hydro_bounds_mut(
        &mut self,
        hydro_index: usize,
        stage_index: usize,
    ) -> &mut HydroStageBounds {
        let idx = hydro_index * self.n_stages + stage_index;
        &mut self.hydro[idx]
    }

    /// Return a mutable reference to the thermal bounds cell for in-place update.
    #[inline]
    pub fn thermal_bounds_mut(
        &mut self,
        thermal_index: usize,
        stage_index: usize,
    ) -> &mut ThermalStageBounds {
        let idx = thermal_index * self.n_stages + stage_index;
        &mut self.thermal[idx]
    }

    /// Return a mutable reference to the line bounds cell for in-place update.
    #[inline]
    pub fn line_bounds_mut(
        &mut self,
        line_index: usize,
        stage_index: usize,
    ) -> &mut LineStageBounds {
        let idx = line_index * self.n_stages + stage_index;
        &mut self.line[idx]
    }

    /// Return a mutable reference to the pumping bounds cell for in-place update.
    #[inline]
    pub fn pumping_bounds_mut(
        &mut self,
        pumping_index: usize,
        stage_index: usize,
    ) -> &mut PumpingStageBounds {
        let idx = pumping_index * self.n_stages + stage_index;
        &mut self.pumping[idx]
    }

    /// Return a mutable reference to the contract bounds cell for in-place update.
    #[inline]
    pub fn contract_bounds_mut(
        &mut self,
        contract_index: usize,
        stage_index: usize,
    ) -> &mut ContractStageBounds {
        let idx = contract_index * self.n_stages + stage_index;
        &mut self.contract[idx]
    }

    /// Return the number of stages in this table.
    #[inline]
    #[must_use]
    pub fn n_stages(&self) -> usize {
        self.n_stages
    }
}

// ─── Generic constraint bounds ────────────────────────────────────────────────

/// Pre-resolved RHS bound table for user-defined generic linear constraints.
///
/// Indexed by `(constraint_index, stage_id)` using a sparse `HashMap`. Provides O(1)
/// lookup of the active bounds for LP row construction.
///
/// Entries are stored in a flat `Vec<(Option<i32>, f64)>` of `(block_id, bound)` pairs.
/// Each `(constraint_index, stage_id)` key maps to a contiguous `Range<usize>` slice
/// within that flat vec.
///
/// When no bounds exist for a `(constraint_index, stage_id)` pair, [`is_active`]
/// returns `false` and [`bounds_for_stage`] returns an empty slice — there is no
/// panic or error.
///
/// # Construction
///
/// Use [`ResolvedGenericConstraintBounds::empty`] as the default (no generic constraints),
/// or [`ResolvedGenericConstraintBounds::new`] to build from parsed bound rows.
/// `cobre-io` is responsible for populating the table.
///
/// # Examples
///
/// ```
/// use cobre_core::ResolvedGenericConstraintBounds;
///
/// let empty = ResolvedGenericConstraintBounds::empty();
/// assert!(!empty.is_active(0, 0));
/// assert!(empty.bounds_for_stage(0, 0).is_empty());
/// ```
///
/// [`is_active`]: ResolvedGenericConstraintBounds::is_active
/// [`bounds_for_stage`]: ResolvedGenericConstraintBounds::bounds_for_stage
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedGenericConstraintBounds {
    /// Sparse index: maps `(constraint_idx, stage_id)` to a range in `entries`.
    ///
    /// Using `i32` for `stage_id` because domain-level stage IDs are `i32` and may
    /// be negative for pre-study stages (though generic constraint bounds should only
    /// reference study stages).
    index: HashMap<(usize, i32), Range<usize>>,
    /// Flat storage of `(block_id, bound)` pairs, grouped by `(constraint_idx, stage_id)`.
    ///
    /// Entries for each key occupy a contiguous region; the [`index`] map provides
    /// the `Range<usize>` slice boundaries.
    ///
    /// [`index`]: Self::index
    entries: Vec<(Option<i32>, f64)>,
}

#[cfg(feature = "serde")]
mod serde_generic_bounds {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    use super::ResolvedGenericConstraintBounds;

    /// Wire format for serde: a list of `(constraint_idx, stage_id, pairs)` groups.
    ///
    /// JSON/bincode cannot serialize `HashMap<(usize, i32), Range<usize>>` directly
    /// because composite tuple keys are not strings. This wire format avoids that
    /// by encoding each group as a tagged list of entries.
    #[derive(Serialize, Deserialize)]
    struct WireEntry {
        constraint_idx: usize,
        stage_id: i32,
        pairs: Vec<(Option<i32>, f64)>,
    }

    impl Serialize for ResolvedGenericConstraintBounds {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            // Collect all keys from the index, sort for deterministic output, then
            // emit each group as a `WireEntry`.
            let mut keys: Vec<(usize, i32)> = self.index.keys().copied().collect();
            keys.sort_unstable();

            let wire: Vec<WireEntry> = keys
                .into_iter()
                .map(|(constraint_idx, stage_id)| {
                    let range = self.index[&(constraint_idx, stage_id)].clone();
                    WireEntry {
                        constraint_idx,
                        stage_id,
                        pairs: self.entries[range].to_vec(),
                    }
                })
                .collect();

            wire.serialize(serializer)
        }
    }

    impl<'de> Deserialize<'de> for ResolvedGenericConstraintBounds {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            let wire = Vec::<WireEntry>::deserialize(deserializer)?;

            let mut index = std::collections::HashMap::new();
            let mut entries = Vec::new();

            for entry in wire {
                let start = entries.len();
                entries.extend_from_slice(&entry.pairs);
                let end = entries.len();
                if end > start {
                    index.insert((entry.constraint_idx, entry.stage_id), start..end);
                }
            }

            Ok(ResolvedGenericConstraintBounds { index, entries })
        }
    }
}

impl ResolvedGenericConstraintBounds {
    /// Return an empty table with no constraints and no bounds.
    ///
    /// Used as the default value in [`System`](crate::System) when no generic constraints
    /// are loaded. All queries on the empty table return `false` / empty slices.
    ///
    /// # Examples
    ///
    /// ```
    /// use cobre_core::ResolvedGenericConstraintBounds;
    ///
    /// let t = ResolvedGenericConstraintBounds::empty();
    /// assert!(!t.is_active(0, 0));
    /// assert!(t.bounds_for_stage(99, 5).is_empty());
    /// ```
    #[must_use]
    pub fn empty() -> Self {
        Self {
            index: HashMap::new(),
            entries: Vec::new(),
        }
    }

    /// Build a resolved table from sorted bound rows.
    ///
    /// `constraint_id_to_idx` maps domain-level `constraint_id: i32` values to
    /// positional indices in the constraint collection. Rows whose `constraint_id`
    /// is not present in that map are silently skipped (they would have been caught
    /// by referential validation upstream).
    ///
    /// `raw_bounds` must be sorted by `(constraint_id, stage_id, block_id)` ascending
    /// (the ordering produced by `parse_generic_constraint_bounds`).
    ///
    /// # Arguments
    ///
    /// * `constraint_id_to_idx` — maps domain `constraint_id` to positional index
    /// * `raw_bounds` — sorted rows from `constraints/generic_constraint_bounds.parquet`
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashMap;
    /// use cobre_core::ResolvedGenericConstraintBounds;
    ///
    /// // Two constraints with IDs 10 and 20, mapped to positions 0 and 1.
    /// let id_map: HashMap<i32, usize> = [(10, 0), (20, 1)].into_iter().collect();
    ///
    /// // One bound row: constraint 10 at stage 3, block_id = None, bound = 500.0.
    /// let rows = vec![(10i32, 3i32, None::<i32>, 500.0f64)];
    ///
    /// let table = ResolvedGenericConstraintBounds::new(
    ///     &id_map,
    ///     rows.iter().map(|(cid, sid, bid, b)| (*cid, *sid, *bid, *b)),
    /// );
    ///
    /// assert!(table.is_active(0, 3));
    /// assert!(!table.is_active(1, 3));
    ///
    /// let slice = table.bounds_for_stage(0, 3);
    /// assert_eq!(slice.len(), 1);
    /// assert_eq!(slice[0], (None, 500.0));
    /// ```
    pub fn new<I>(constraint_id_to_idx: &HashMap<i32, usize>, raw_bounds: I) -> Self
    where
        I: Iterator<Item = (i32, i32, Option<i32>, f64)>,
    {
        let mut index: HashMap<(usize, i32), Range<usize>> = HashMap::new();
        let mut entries: Vec<(Option<i32>, f64)> = Vec::new();

        // The input rows are sorted by (constraint_id, stage_id, block_id).
        // We group consecutive rows with the same (constraint_idx, stage_id) key
        // into a contiguous range in `entries`.

        let mut current_key: Option<(usize, i32)> = None;
        let mut range_start: usize = 0;

        for (constraint_id, stage_id, block_id, bound) in raw_bounds {
            let Some(&constraint_idx) = constraint_id_to_idx.get(&constraint_id) else {
                // Unknown constraint ID — silently skip (referential validation concern).
                continue;
            };

            let key = (constraint_idx, stage_id);

            // When the key changes, commit the range for the previous key.
            if current_key != Some(key) {
                if let Some(prev_key) = current_key {
                    let range_end = entries.len();
                    if range_end > range_start {
                        index.insert(prev_key, range_start..range_end);
                    }
                }
                range_start = entries.len();
                current_key = Some(key);
            }

            entries.push((block_id, bound));
        }

        // Commit the final key.
        if let Some(last_key) = current_key {
            let range_end = entries.len();
            if range_end > range_start {
                index.insert(last_key, range_start..range_end);
            }
        }

        Self { index, entries }
    }

    /// Return `true` if at least one bound entry exists for this constraint at the given stage.
    ///
    /// Returns `false` for any unknown `(constraint_idx, stage_id)` pair.
    ///
    /// # Examples
    ///
    /// ```
    /// use cobre_core::ResolvedGenericConstraintBounds;
    ///
    /// let empty = ResolvedGenericConstraintBounds::empty();
    /// assert!(!empty.is_active(0, 0));
    /// ```
    #[inline]
    #[must_use]
    pub fn is_active(&self, constraint_idx: usize, stage_id: i32) -> bool {
        self.index.contains_key(&(constraint_idx, stage_id))
    }

    /// Return the `(block_id, bound)` pairs for a constraint at the given stage.
    ///
    /// Returns an empty slice when no bounds exist for the `(constraint_idx, stage_id)` pair.
    ///
    /// # Examples
    ///
    /// ```
    /// use cobre_core::ResolvedGenericConstraintBounds;
    ///
    /// let empty = ResolvedGenericConstraintBounds::empty();
    /// assert!(empty.bounds_for_stage(0, 0).is_empty());
    /// ```
    #[inline]
    #[must_use]
    pub fn bounds_for_stage(&self, constraint_idx: usize, stage_id: i32) -> &[(Option<i32>, f64)] {
        match self.index.get(&(constraint_idx, stage_id)) {
            Some(range) => &self.entries[range.clone()],
            None => &[],
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_hydro_penalties() -> HydroStagePenalties {
        HydroStagePenalties {
            spillage_cost: 0.01,
            diversion_cost: 0.02,
            fpha_turbined_cost: 0.03,
            storage_violation_below_cost: 1000.0,
            filling_target_violation_cost: 5000.0,
            turbined_violation_below_cost: 500.0,
            outflow_violation_below_cost: 400.0,
            outflow_violation_above_cost: 300.0,
            generation_violation_below_cost: 200.0,
            evaporation_violation_cost: 150.0,
            water_withdrawal_violation_cost: 100.0,
        }
    }

    fn make_hydro_bounds() -> HydroStageBounds {
        HydroStageBounds {
            min_storage_hm3: 10.0,
            max_storage_hm3: 200.0,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 500.0,
            min_outflow_m3s: 5.0,
            max_outflow_m3s: None,
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            max_diversion_m3s: None,
            filling_inflow_m3s: 0.0,
            water_withdrawal_m3s: 0.0,
        }
    }

    #[test]
    fn test_hydro_stage_penalties_copy() {
        let p = make_hydro_penalties();
        let q = p;
        let r = p;
        assert_eq!(q, r);
        assert!((q.spillage_cost - p.spillage_cost).abs() < f64::EPSILON);
    }

    #[test]
    fn test_all_penalty_structs_are_copy() {
        let bp = BusStagePenalties { excess_cost: 1.0 };
        let lp = LineStagePenalties { exchange_cost: 2.0 };
        let np = NcsStagePenalties {
            curtailment_cost: 3.0,
        };

        assert_eq!(bp, bp);
        assert_eq!(lp, lp);
        assert_eq!(np, np);
        let bp2 = bp;
        let lp2 = lp;
        let np2 = np;
        assert!((bp2.excess_cost - 1.0).abs() < f64::EPSILON);
        assert!((lp2.exchange_cost - 2.0).abs() < f64::EPSILON);
        assert!((np2.curtailment_cost - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_all_bound_structs_are_copy() {
        let hb = make_hydro_bounds();
        let tb = ThermalStageBounds {
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
        };
        let lb = LineStageBounds {
            direct_mw: 500.0,
            reverse_mw: 500.0,
        };
        let pb = PumpingStageBounds {
            min_flow_m3s: 0.0,
            max_flow_m3s: 20.0,
        };
        let cb = ContractStageBounds {
            min_mw: 0.0,
            max_mw: 50.0,
            price_per_mwh: 80.0,
        };

        let hb2 = hb;
        let tb2 = tb;
        let lb2 = lb;
        let pb2 = pb;
        let cb2 = cb;
        assert_eq!(hb, hb2);
        assert_eq!(tb, tb2);
        assert_eq!(lb, lb2);
        assert_eq!(pb, pb2);
        assert_eq!(cb, cb2);
    }

    #[test]
    fn test_resolved_penalties_construction() {
        // 2 hydros, 1 bus, 1 line, 1 ncs, 3 stages
        let hp = make_hydro_penalties();
        let bp = BusStagePenalties { excess_cost: 100.0 };
        let lp = LineStagePenalties { exchange_cost: 5.0 };
        let np = NcsStagePenalties {
            curtailment_cost: 50.0,
        };

        let table = ResolvedPenalties::new(2, 1, 1, 1, 3, hp, bp, lp, np);

        for hydro_idx in 0..2 {
            for stage_idx in 0..3 {
                let p = table.hydro_penalties(hydro_idx, stage_idx);
                assert!((p.spillage_cost - 0.01).abs() < f64::EPSILON);
                assert!((p.storage_violation_below_cost - 1000.0).abs() < f64::EPSILON);
            }
        }

        assert!((table.bus_penalties(0, 0).excess_cost - 100.0).abs() < f64::EPSILON);
        assert!((table.line_penalties(0, 1).exchange_cost - 5.0).abs() < f64::EPSILON);
        assert!((table.ncs_penalties(0, 2).curtailment_cost - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_resolved_penalties_indexed_access() {
        let hp = make_hydro_penalties();
        let bp = BusStagePenalties { excess_cost: 10.0 };
        let lp = LineStagePenalties { exchange_cost: 1.0 };
        let np = NcsStagePenalties {
            curtailment_cost: 5.0,
        };

        let table = ResolvedPenalties::new(3, 0, 0, 0, 5, hp, bp, lp, np);
        assert_eq!(table.n_stages(), 5);

        let p = table.hydro_penalties(1, 3);
        assert!((p.diversion_cost - 0.02).abs() < f64::EPSILON);
        assert!((p.filling_target_violation_cost - 5000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_resolved_penalties_mutable_update() {
        let hp = make_hydro_penalties();
        let bp = BusStagePenalties { excess_cost: 10.0 };
        let lp = LineStagePenalties { exchange_cost: 1.0 };
        let np = NcsStagePenalties {
            curtailment_cost: 5.0,
        };

        let mut table = ResolvedPenalties::new(2, 2, 1, 1, 3, hp, bp, lp, np);

        table.hydro_penalties_mut(0, 1).spillage_cost = 99.0;

        assert!((table.hydro_penalties(0, 1).spillage_cost - 99.0).abs() < f64::EPSILON);
        assert!((table.hydro_penalties(0, 0).spillage_cost - 0.01).abs() < f64::EPSILON);
        assert!((table.hydro_penalties(1, 1).spillage_cost - 0.01).abs() < f64::EPSILON);

        table.bus_penalties_mut(1, 2).excess_cost = 999.0;
        assert!((table.bus_penalties(1, 2).excess_cost - 999.0).abs() < f64::EPSILON);
        assert!((table.bus_penalties(0, 2).excess_cost - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_resolved_bounds_construction() {
        let hb = make_hydro_bounds();
        let tb = ThermalStageBounds {
            min_generation_mw: 50.0,
            max_generation_mw: 400.0,
        };
        let lb = LineStageBounds {
            direct_mw: 1000.0,
            reverse_mw: 800.0,
        };
        let pb = PumpingStageBounds {
            min_flow_m3s: 0.0,
            max_flow_m3s: 20.0,
        };
        let cb = ContractStageBounds {
            min_mw: 0.0,
            max_mw: 100.0,
            price_per_mwh: 80.0,
        };

        let table = ResolvedBounds::new(1, 2, 1, 1, 1, 3, hb, tb, lb, pb, cb);

        let b = table.hydro_bounds(0, 2);
        assert!((b.min_storage_hm3 - 10.0).abs() < f64::EPSILON);
        assert!((b.max_storage_hm3 - 200.0).abs() < f64::EPSILON);
        assert!(b.max_outflow_m3s.is_none());
        assert!(b.max_diversion_m3s.is_none());

        let t0 = table.thermal_bounds(0, 0);
        let t1 = table.thermal_bounds(1, 2);
        assert!((t0.max_generation_mw - 400.0).abs() < f64::EPSILON);
        assert!((t1.min_generation_mw - 50.0).abs() < f64::EPSILON);

        assert!((table.line_bounds(0, 1).direct_mw - 1000.0).abs() < f64::EPSILON);
        assert!((table.pumping_bounds(0, 0).max_flow_m3s - 20.0).abs() < f64::EPSILON);
        assert!((table.contract_bounds(0, 2).price_per_mwh - 80.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_resolved_bounds_mutable_update() {
        let hb = make_hydro_bounds();
        let tb = ThermalStageBounds {
            min_generation_mw: 0.0,
            max_generation_mw: 200.0,
        };
        let lb = LineStageBounds {
            direct_mw: 500.0,
            reverse_mw: 500.0,
        };
        let pb = PumpingStageBounds {
            min_flow_m3s: 0.0,
            max_flow_m3s: 30.0,
        };
        let cb = ContractStageBounds {
            min_mw: 0.0,
            max_mw: 50.0,
            price_per_mwh: 60.0,
        };

        let mut table = ResolvedBounds::new(2, 1, 1, 1, 1, 3, hb, tb, lb, pb, cb);

        let cell = table.hydro_bounds_mut(1, 0);
        cell.min_storage_hm3 = 25.0;
        cell.max_outflow_m3s = Some(1000.0);

        assert!((table.hydro_bounds(1, 0).min_storage_hm3 - 25.0).abs() < f64::EPSILON);
        assert_eq!(table.hydro_bounds(1, 0).max_outflow_m3s, Some(1000.0));
        assert!((table.hydro_bounds(0, 0).min_storage_hm3 - 10.0).abs() < f64::EPSILON);
        assert!(table.hydro_bounds(1, 1).max_outflow_m3s.is_none());

        table.thermal_bounds_mut(0, 2).max_generation_mw = 150.0;
        assert!((table.thermal_bounds(0, 2).max_generation_mw - 150.0).abs() < f64::EPSILON);
        assert!((table.thermal_bounds(0, 0).max_generation_mw - 200.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_hydro_stage_bounds_has_eleven_fields() {
        let b = HydroStageBounds {
            min_storage_hm3: 1.0,
            max_storage_hm3: 2.0,
            min_turbined_m3s: 3.0,
            max_turbined_m3s: 4.0,
            min_outflow_m3s: 5.0,
            max_outflow_m3s: Some(6.0),
            min_generation_mw: 7.0,
            max_generation_mw: 8.0,
            max_diversion_m3s: Some(9.0),
            filling_inflow_m3s: 10.0,
            water_withdrawal_m3s: 11.0,
        };
        assert!((b.min_storage_hm3 - 1.0).abs() < f64::EPSILON);
        assert!((b.water_withdrawal_m3s - 11.0).abs() < f64::EPSILON);
        assert_eq!(b.max_outflow_m3s, Some(6.0));
        assert_eq!(b.max_diversion_m3s, Some(9.0));
    }

    #[test]
    #[cfg(feature = "serde")]
    fn test_resolved_penalties_serde_roundtrip() {
        let hp = make_hydro_penalties();
        let bp = BusStagePenalties { excess_cost: 100.0 };
        let lp = LineStagePenalties { exchange_cost: 5.0 };
        let np = NcsStagePenalties {
            curtailment_cost: 50.0,
        };

        let original = ResolvedPenalties::new(2, 1, 1, 1, 3, hp, bp, lp, np);
        let json = serde_json::to_string(&original).expect("serialize");
        let restored: ResolvedPenalties = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(original, restored);
    }

    #[test]
    #[cfg(feature = "serde")]
    fn test_resolved_bounds_serde_roundtrip() {
        let hb = make_hydro_bounds();
        let tb = ThermalStageBounds {
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
        };
        let lb = LineStageBounds {
            direct_mw: 500.0,
            reverse_mw: 500.0,
        };
        let pb = PumpingStageBounds {
            min_flow_m3s: 0.0,
            max_flow_m3s: 20.0,
        };
        let cb = ContractStageBounds {
            min_mw: 0.0,
            max_mw: 50.0,
            price_per_mwh: 80.0,
        };

        let original = ResolvedBounds::new(1, 1, 1, 1, 1, 3, hb, tb, lb, pb, cb);
        let json = serde_json::to_string(&original).expect("serialize");
        let restored: ResolvedBounds = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(original, restored);
    }

    // ─── ResolvedGenericConstraintBounds tests ────────────────────────────────

    /// `empty()` returns a table where all queries return false/empty.
    #[test]
    fn test_generic_bounds_empty() {
        let t = ResolvedGenericConstraintBounds::empty();
        assert!(!t.is_active(0, 0));
        assert!(!t.is_active(99, -1));
        assert!(t.bounds_for_stage(0, 0).is_empty());
        assert!(t.bounds_for_stage(99, 5).is_empty());
    }

    /// `new()` with 2 constraints, sparse bounds: constraint 0 at stage 0 is active;
    /// constraint 1 at stage 0 is not active.
    #[test]
    fn test_generic_bounds_sparse_active() {
        let id_map: HashMap<i32, usize> = [(0, 0), (1, 1)].into_iter().collect();

        // One row: constraint_id=0, stage_id=0, block_id=None, bound=100.0
        let rows = vec![(0i32, 0i32, None::<i32>, 100.0f64)];
        let t = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

        assert!(t.is_active(0, 0), "constraint 0 at stage 0 must be active");
        assert!(
            !t.is_active(1, 0),
            "constraint 1 at stage 0 must not be active"
        );
        assert!(
            !t.is_active(0, 1),
            "constraint 0 at stage 1 must not be active"
        );
    }

    /// `bounds_for_stage()` with block_id=None returns the correct single-entry slice.
    #[test]
    fn test_generic_bounds_single_block_none() {
        let id_map: HashMap<i32, usize> = [(0, 0)].into_iter().collect();
        let rows = vec![(0i32, 0i32, None::<i32>, 100.0f64)];
        let t = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

        let slice = t.bounds_for_stage(0, 0);
        assert_eq!(slice.len(), 1);
        assert_eq!(slice[0], (None, 100.0));
    }

    /// Multiple (block_id, bound) pairs for the same (constraint, stage).
    #[test]
    fn test_generic_bounds_multiple_blocks() {
        let id_map: HashMap<i32, usize> = [(0, 0)].into_iter().collect();
        // Three rows for constraint 0 at stage 2: block None, block 0, block 1.
        let rows = vec![
            (0i32, 2i32, None::<i32>, 50.0f64),
            (0i32, 2i32, Some(0i32), 60.0f64),
            (0i32, 2i32, Some(1i32), 70.0f64),
        ];
        let t = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

        assert!(t.is_active(0, 2));
        let slice = t.bounds_for_stage(0, 2);
        assert_eq!(slice.len(), 3);
        assert_eq!(slice[0], (None, 50.0));
        assert_eq!(slice[1], (Some(0), 60.0));
        assert_eq!(slice[2], (Some(1), 70.0));
    }

    /// Rows with unknown constraint_id are silently skipped.
    #[test]
    fn test_generic_bounds_unknown_constraint_id_skipped() {
        let id_map: HashMap<i32, usize> = [(0, 0)].into_iter().collect();
        // Row with constraint_id=99 not in id_map.
        let rows = vec![(99i32, 0i32, None::<i32>, 1000.0f64)];
        let t = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

        assert!(!t.is_active(0, 0), "unknown constraint_id must be skipped");
        assert!(t.bounds_for_stage(0, 0).is_empty());
    }

    /// Empty raw_bounds produces a table identical to `empty()`.
    #[test]
    fn test_generic_bounds_no_rows() {
        let id_map: HashMap<i32, usize> = [(0, 0), (1, 1)].into_iter().collect();
        let t = ResolvedGenericConstraintBounds::new(&id_map, std::iter::empty());

        assert!(!t.is_active(0, 0));
        assert!(!t.is_active(1, 0));
        assert!(t.bounds_for_stage(0, 0).is_empty());
    }

    /// Bounds for constraint 0 at stages 0 and 1; constraint 1 has no bounds.
    #[test]
    fn test_generic_bounds_two_stages_one_constraint() {
        let id_map: HashMap<i32, usize> = [(0, 0), (1, 1)].into_iter().collect();
        let rows = vec![
            (0i32, 0i32, None::<i32>, 100.0f64),
            (0i32, 1i32, None::<i32>, 200.0f64),
        ];
        let t = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

        assert!(t.is_active(0, 0));
        assert!(t.is_active(0, 1));
        assert!(!t.is_active(1, 0));
        assert!(!t.is_active(1, 1));

        let s0 = t.bounds_for_stage(0, 0);
        assert_eq!(s0.len(), 1);
        assert!((s0[0].1 - 100.0).abs() < f64::EPSILON);

        let s1 = t.bounds_for_stage(0, 1);
        assert_eq!(s1.len(), 1);
        assert!((s1[0].1 - 200.0).abs() < f64::EPSILON);
    }

    #[test]
    #[cfg(feature = "serde")]
    fn test_generic_bounds_serde_roundtrip() {
        let id_map: HashMap<i32, usize> = [(0, 0), (1, 1)].into_iter().collect();
        let rows = vec![
            (0i32, 0i32, None::<i32>, 100.0f64),
            (0i32, 0i32, Some(1i32), 150.0f64),
            (1i32, 2i32, None::<i32>, 300.0f64),
        ];
        let original = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());
        let json = serde_json::to_string(&original).expect("serialize");
        let restored: ResolvedGenericConstraintBounds =
            serde_json::from_str(&json).expect("deserialize");
        assert_eq!(original, restored);
    }
}
