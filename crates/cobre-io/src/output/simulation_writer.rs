//! Parquet writer for simulation pipeline output.
//!
//! [`SimulationParquetWriter`] writes Hive-partitioned Parquet files for every
//! entity type produced by the simulation forward pass. Each scenario produces
//! one partition directory per entity type:
//!
//! ```text
//! simulation/
//!   costs/scenario_id=0000/data.parquet
//!   hydros/scenario_id=0000/data.parquet
//!   hydro_bus_generation/scenario_id=0000/data.parquet
//!   thermals/scenario_id=0000/data.parquet
//!   exchanges/scenario_id=0000/data.parquet
//!   buses/scenario_id=0000/data.parquet
//!   pumping_stations/scenario_id=0000/data.parquet
//!   contracts/scenario_id=0000/data.parquet
//!   non_controllables/scenario_id=0000/data.parquet
//!   inflow_lags/scenario_id=0000/data.parquet
//!   in_transit/scenario_id=0000/data.parquet
//!   transit_seed/scenario_id=0000/data.parquet
//!   violations/generic/scenario_id=0000/data.parquet
//!   anticipated_lanes/scenario_id=0000/data.parquet
//! ```
//!
//! The `in_transit/` and `transit_seed/` partitions are present only when the
//! system declares a travel-time arc; a non-travel-time study writes neither
//! directory nor file. The `anticipated_lanes/` partition is present only when
//! the system declares post-study stages; without them, no such directory or
//! file is written. [`SIMULATION_FAMILIES`] is the single table of every
//! family and the condition that declares it.
//!
//! Every record's `node_id` below is the declared node visited at that stage —
//! the degenerate per-stage id on a chain — and must never be gated on whether
//! the system declared `nodes[]`.
//!
//! ## Circular-dependency mitigation
//!
//! The crate-local [`ScenarioWritePayload`] mirrors the caller's result layout
//! so this crate need not depend on the calling crate (which already depends on
//! `cobre-io`); conversion is the caller's responsibility.

use std::collections::HashMap;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{
    BooleanBuilder, Date32Builder, Float64Builder, Int8Builder, Int32Builder, RecordBatch,
};

use chrono::NaiveDate;

use super::date32_days;
use cobre_core::System;

use crate::MetadataSimulationSolveStats;
use crate::output::SimulationOutput;
use crate::output::atomic::write_parquet_atomic;
use crate::output::error::OutputError;
use crate::output::results_writer::SIMULATION_METADATA_FILE;
use crate::output::schemas::{
    anticipated_lanes_schema, buses_schema, contracts_schema, costs_schema, exchanges_schema,
    generic_violations_schema, hydro_bus_generation_schema, hydros_schema, in_transit_schema,
    inflow_lags_schema, non_controllables_schema, paths_schema, pumping_stations_schema,
    scenario_summary_schema, thermals_schema, transit_seed_schema,
};
use crate::output::solver_stats_writer::{
    SIMULATION_SOLVER_DIR, SOLVER_ITERATIONS_FILE, SOLVER_RETRY_HISTOGRAM_FILE,
};

// Payload types (mirrors solver simulation result types)

/// Cost breakdown for one (stage, block) pair.
///
/// Conversion to this type from algorithm-specific cost results is handled by
/// the calling solver.
#[derive(Debug)]
pub struct CostWriteRecord {
    /// Stage index (0-based).
    pub stage_id: u32,
    /// Declared node id visited at this stage.
    pub node_id: i32,
    /// Block index, or `None` for stage-level aggregates.
    pub block_id: Option<u32>,
    /// Total discounted stage cost.
    pub total_cost: f64,
    /// Undiscounted immediate cost.
    pub immediate_cost: f64,
    /// Future cost function value.
    pub future_cost: f64,
    /// Cumulative discount factor.
    pub discount_factor: f64,
    /// Thermal generation cost.
    pub thermal_cost: f64,
    /// Anticipated (forward-committed) thermal generation cost, booked on the
    /// decision-stage commitment column. Zero when no anticipated thermals exist.
    pub anticipated_thermal_cost: f64,
    /// Contract energy cost.
    pub contract_cost: f64,
    /// Load deficit cost.
    pub deficit_cost: f64,
    /// Load excess cost.
    pub excess_cost: f64,
    /// Storage bound violation cost.
    pub storage_violation_cost: f64,
    /// Filling target violation cost.
    pub filling_target_cost: f64,
    /// Hydro operational violation cost.
    pub hydro_violation_cost: f64,
    /// Cost of minimum outflow violations.
    pub outflow_violation_below_cost: f64,
    /// Cost of maximum outflow violations.
    pub outflow_violation_above_cost: f64,
    /// Cost of minimum turbining violations.
    pub turbined_violation_cost: f64,
    /// Cost of minimum generation violations.
    pub generation_violation_cost: f64,
    /// Cost of evaporation constraint violations.
    pub evaporation_violation_cost: f64,
    /// Cost of water withdrawal constraint violations.
    pub withdrawal_violation_cost: f64,
    /// Inflow non-negativity violation cost.
    pub inflow_penalty_cost: f64,
    /// Generic constraint violation cost.
    pub generic_violation_cost: f64,
    /// Spillage regularization cost.
    pub spillage_cost: f64,
    /// Turbining regularization cost (applied to every hydro's turbine flow).
    pub turbined_cost: f64,
    /// Curtailment regularization cost.
    pub curtailment_cost: f64,
    /// Exchange regularization cost.
    pub exchange_cost: f64,
    /// Pumping imputed cost.
    pub pumping_cost: f64,
}

/// Hydro plant result for one (stage, block, hydro) tuple.
#[derive(Debug)]
pub struct HydroWriteRecord {
    /// Stage index (0-based).
    pub stage_id: u32,
    /// Declared node id visited at this stage.
    pub node_id: i32,
    /// Block index, or `None` for stage-level rows.
    pub block_id: Option<u32>,
    /// Hydro plant entity ID.
    pub hydro_id: i32,
    /// Turbined flow in m³/s.
    pub turbined_m3s: f64,
    /// Spilled flow in m³/s.
    pub spillage_m3s: f64,
    /// Evaporation loss in m³/s, or `None` if not modeled.
    pub evaporation_m3s: Option<f64>,
    /// Diverted inflow in m³/s, or `None` if no diversion.
    pub diverted_inflow_m3s: Option<f64>,
    /// Diverted outflow in m³/s, or `None` if no diversion.
    pub diverted_outflow_m3s: Option<f64>,
    /// Incremental natural inflow in m³/s.
    pub incremental_inflow_m3s: f64,
    /// Incremental (local) natural inflow in m³/s; the same value as
    /// `incremental_inflow_m3s`.
    pub inflow_m3s: f64,
    /// Reservoir storage at block start in hm³.
    pub storage_initial_hm3: f64,
    /// Reservoir storage at block end in hm³.
    pub storage_final_hm3: f64,
    /// Active power generation in MW.
    pub generation_mw: f64,
    /// Equivalent productivity (FPHA `ρ_eq`) in MW/(m³/s).
    pub equivalent_productivity_mw_per_m3s: f64,
    /// Accumulated productivity (VHA `ρ_acc`) in MW/(m³/s).
    pub accumulated_productivity_mw_per_m3s: f64,
    /// Incremental inflow energy equivalent in MW.
    pub incremental_inflow_energy_mw: f64,
    /// Stored energy at block start in `MWh`.
    pub stored_energy_initial_mwh: f64,
    /// Stored energy at block end in `MWh`.
    pub stored_energy_final_mwh: f64,
    /// Spillage regularization cost.
    pub spillage_cost: f64,
    /// Marginal water value: the dual of the row's own water-balance constraint
    /// (each block's own row in chronological stages; the stage row, repeated on
    /// every block row, in parallel stages), in cost/hm³.
    pub water_value_per_hm3: f64,
    /// Storage binding code.
    pub storage_binding_code: i8,
    /// Operative state code.
    pub operative_state_code: i8,
    /// Turbining capacity slack in m³/s.
    pub turbined_slack_m3s: f64,
    /// Minimum outflow violation slack in m³/s.
    pub outflow_slack_below_m3s: f64,
    /// Maximum outflow violation slack in m³/s.
    pub outflow_slack_above_m3s: f64,
    /// Generation capacity violation slack in MW.
    pub generation_slack_mw: f64,
    /// Storage below minimum bound violation in hm³.
    pub storage_violation_below_hm3: f64,
    /// Filling target violation in hm³.
    pub filling_target_violation_hm3: f64,
    /// Over-evaporation violation in m³/s.
    pub evaporation_violation_pos_m3s: f64,
    /// Under-evaporation violation in m³/s.
    pub evaporation_violation_neg_m3s: f64,
    /// Inflow non-negativity slack in m³/s.
    pub inflow_nonnegativity_slack_m3s: f64,
    /// Over-withdrawal violation in m³/s.
    pub water_withdrawal_violation_pos_m3s: f64,
    /// Under-withdrawal violation in m³/s.
    pub water_withdrawal_violation_neg_m3s: f64,
    /// Storage-range mean equivalent productivity in MW/(m³/s).
    pub integrated_equivalent_productivity_mw_per_m3s: f64,
    /// Storage-range mean accumulated productivity along the downstream cascade in MW/(m³/s).
    pub integrated_accumulated_productivity_mw_per_m3s: f64,
    /// Stored energy at block start, averaged over the stage's total hours, in MW.
    pub stored_energy_initial_mw: f64,
    /// Stored energy at block end, averaged over the stage's total hours, in MW.
    pub stored_energy_final_mw: f64,
}

/// Thermal unit result for one (stage, block, thermal) tuple.
#[derive(Debug)]
pub struct ThermalWriteRecord {
    /// Stage index (0-based).
    pub stage_id: u32,
    /// Declared node id visited at this stage.
    pub node_id: i32,
    /// Block index, or `None` for stage-level rows.
    pub block_id: Option<u32>,
    /// Thermal unit entity ID.
    pub thermal_id: i32,
    /// Active power generation in MW.
    pub generation_mw: f64,
    /// Variable generation cost.
    pub generation_cost: f64,
    /// Whether this unit uses anticipated dispatch.
    pub is_anticipated: bool,
    /// Realised delivery commitment in MW, or `None`.
    pub anticipated_committed_mw: Option<f64>,
    /// New anticipated commitment decided at this stage in MW, or `None`.
    pub anticipated_decision_mw: Option<f64>,
    /// Operative state code.
    pub operative_state_code: i8,
}

/// Exchange (transmission line) result for one (stage, block, line) tuple.
#[derive(Debug)]
pub struct ExchangeWriteRecord {
    /// Stage index (0-based).
    pub stage_id: u32,
    /// Declared node id visited at this stage.
    pub node_id: i32,
    /// Block index, or `None` for stage-level rows.
    pub block_id: Option<u32>,
    /// Transmission line entity ID.
    pub line_id: i32,
    /// Forward direction flow in MW.
    pub direct_flow_mw: f64,
    /// Reverse direction flow in MW.
    pub reverse_flow_mw: f64,
    /// Exchange regularization cost.
    pub exchange_cost: f64,
    /// Operative state code.
    pub operative_state_code: i8,
}

/// Bus result for one (stage, block, bus) tuple.
#[derive(Debug)]
pub struct BusWriteRecord {
    /// Stage index (0-based).
    pub stage_id: u32,
    /// Declared node id visited at this stage.
    pub node_id: i32,
    /// Block index, or `None` for stage-level rows.
    pub block_id: Option<u32>,
    /// Bus entity ID.
    pub bus_id: i32,
    /// Total demand in MW.
    pub load_mw: f64,
    /// Load deficit in MW.
    pub deficit_mw: f64,
    /// Load excess in MW.
    pub excess_mw: f64,
    /// Marginal cost of energy (spot price) in cost/MWh.
    pub spot_price: f64,
}

/// Pumping station result for one (stage, block, station) tuple.
#[derive(Debug)]
pub struct PumpingWriteRecord {
    /// Stage index (0-based).
    pub stage_id: u32,
    /// Declared node id visited at this stage.
    pub node_id: i32,
    /// Block index, or `None` for stage-level rows.
    pub block_id: Option<u32>,
    /// Pumping station entity ID.
    pub pumping_station_id: i32,
    /// Pumped flow rate in m³/s.
    pub pumped_flow_m3s: f64,
    /// Active power consumed in MW.
    pub power_consumption_mw: f64,
    /// Pumping imputed cost.
    pub pumping_cost: f64,
    /// Operative state code.
    pub operative_state_code: i8,
}

/// Contract result for one (stage, block, contract) tuple.
#[derive(Debug)]
pub struct ContractWriteRecord {
    /// Stage index (0-based).
    pub stage_id: u32,
    /// Declared node id visited at this stage.
    pub node_id: i32,
    /// Block index, or `None` for stage-level rows.
    pub block_id: Option<u32>,
    /// Contract entity ID.
    pub contract_id: i32,
    /// Contracted power in MW.
    pub power_mw: f64,
    /// Contract price in cost/MWh.
    pub price_per_mwh: f64,
    /// Total cost for this contract at this block.
    pub total_cost: f64,
    /// Operative state code.
    pub operative_state_code: i8,
}

/// Non-controllable source result for one (stage, block, source) tuple.
#[derive(Debug)]
pub struct NonControllableWriteRecord {
    /// Stage index (0-based).
    pub stage_id: u32,
    /// Declared node id visited at this stage.
    pub node_id: i32,
    /// Block index, or `None` for stage-level rows.
    pub block_id: Option<u32>,
    /// Non-controllable source entity ID.
    pub non_controllable_id: i32,
    /// Active power injected in MW.
    pub generation_mw: f64,
    /// Maximum available power in MW.
    pub available_mw: f64,
    /// Curtailed power in MW.
    pub curtailment_mw: f64,
    /// Curtailment regularization cost.
    pub curtailment_cost: f64,
    /// Operative state code.
    pub operative_state_code: i8,
}

/// Inflow lag state for one (stage, hydro, `lag_index`) tuple.
#[derive(Debug)]
pub struct InflowLagWriteRecord {
    /// Stage index (0-based).
    pub stage_id: u32,
    /// Declared node id visited at this stage.
    pub node_id: i32,
    /// Hydro plant entity ID.
    pub hydro_id: i32,
    /// Lag index within the AR model (0 = most recent past period).
    pub lag_index: u32,
    /// Observed inflow at this lag in m³/s.
    pub inflow_m3s: f64,
}

/// Travel-time in-transit water state for one (stage, downstream-plant, lag)
/// tuple.
#[derive(Debug)]
pub struct TransitBucketWriteRecord {
    /// Stage index (0-based).
    pub stage_id: u32,
    /// Declared node id visited at this stage.
    pub node_id: i32,
    /// Downstream hydro plant entity ID the arc feeds.
    pub hydro_id: i32,
    /// Maturity bucket index (1-based).
    pub lag: u32,
    /// Outgoing in-transit water volume at this maturity in hm³.
    pub in_transit_volume_hm3: f64,
    /// Delivered volume in hm³ that matured this stage; non-zero only at
    /// `lag == 1`.
    pub delayed_arrival_hm3: f64,
}

/// One windowed release record for an upstream entity — a rolling-seed
/// artifact, not a per-stage diagnostic: the record carries no stage or node
/// axis, only a `[start_date, end_date)` release window and rate.
#[derive(Debug)]
pub struct TransitSeedWriteRecord {
    /// Upstream entity identifier whose release the window covers.
    pub hydro_id: i32,
    /// Start of the release window (inclusive).
    pub start_date: NaiveDate,
    /// End of the release window (exclusive).
    pub end_date: NaiveDate,
    /// Mean release rate over the window, in m³/s.
    pub value_m3s: f64,
}

/// Per-cell hydro dispatch result for one (stage, block, hydro, bus) tuple —
/// one LP cell.
#[derive(Debug)]
pub struct HydroBusWriteRecord {
    /// Stage index (0-based).
    pub stage_id: u32,
    /// Declared node id visited at this stage.
    pub node_id: i32,
    /// Block index, or `None` for stage-level rows.
    pub block_id: Option<u32>,
    /// Hydro plant entity ID.
    pub hydro_id: i32,
    /// Bus entity ID owning this cell.
    pub bus_id: i32,
    /// Turbined flow in m³/s.
    pub turbined_m3s: f64,
    /// Active power generation in MW.
    pub generation_mw: f64,
}

/// Generic constraint violation for one (stage, block, constraint) tuple.
#[derive(Debug)]
pub struct GenericViolationWriteRecord {
    /// Stage index (0-based).
    pub stage_id: u32,
    /// Declared node id visited at this stage.
    pub node_id: i32,
    /// Block index, or `None` for stage-level rows.
    pub block_id: Option<u32>,
    /// Generic constraint entity ID.
    pub constraint_id: i32,
    /// Violation slack value.
    pub slack_value: f64,
    /// Cost for this violation.
    pub slack_cost: f64,
}

/// One post-horizon commitment lane result for one declared window, one
/// terminal scenario — keyed `(thermal_id, delivery_date)`.
#[derive(Debug)]
pub struct AnticipatedLaneWriteRecord {
    /// Stage index (0-based) — the lane's own in-study decider stage.
    pub stage_id: u32,
    /// Declared node id visited at this stage.
    pub node_id: i32,
    /// Thermal unit entity ID owning this lane.
    pub thermal_id: i32,
    /// `YYYYMM01` anchor of the lane's resolved post-study delivery stage.
    pub delivery_date: i32,
    /// Deposited decision in MW.
    pub deposited_decision_mw: f64,
    /// Carried committed value in MW.
    pub carried_committed_mw: f64,
}

/// All simulation results for one stage within one scenario.
#[derive(Debug)]
pub struct StageWritePayload {
    /// Stage index (0-based).
    pub stage_id: u32,
    /// Declared node id visited at this stage.
    pub node_id: i32,
    /// Cost breakdown records for this stage.
    pub costs: Vec<CostWriteRecord>,
    /// Hydro plant records for this stage.
    pub hydros: Vec<HydroWriteRecord>,
    /// Per-cell hydro dispatch records for this stage.
    pub hydro_bus_generation: Vec<HydroBusWriteRecord>,
    /// Thermal unit records for this stage.
    pub thermals: Vec<ThermalWriteRecord>,
    /// Exchange records for this stage.
    pub exchanges: Vec<ExchangeWriteRecord>,
    /// Bus records for this stage.
    pub buses: Vec<BusWriteRecord>,
    /// Pumping station records for this stage.
    pub pumping_stations: Vec<PumpingWriteRecord>,
    /// Contract records for this stage.
    pub contracts: Vec<ContractWriteRecord>,
    /// Non-controllable source records for this stage.
    pub non_controllables: Vec<NonControllableWriteRecord>,
    /// Inflow lag state records for this stage.
    pub inflow_lags: Vec<InflowLagWriteRecord>,
    /// Travel-time in-transit bucket records for this stage.
    pub transit_buckets: Vec<TransitBucketWriteRecord>,
    /// Generic constraint violation records for this stage.
    pub generic_violations: Vec<GenericViolationWriteRecord>,
    /// Post-horizon commitment lane records for this stage.
    pub anticipated_lanes: Vec<AnticipatedLaneWriteRecord>,
}

/// Complete simulation result for one scenario, ready for Parquet writing.
///
/// This is the local counterpart of solver-specific simulation result types.
/// Conversion to this payload is handled by the solver's output integration layer.
#[derive(Debug)]
pub struct ScenarioWritePayload {
    /// 0-based scenario identifier. Determines the Hive partition
    /// (`{entity}/scenario_id={scenario_id:04d}/data.parquet`) and is also written
    /// as a real `scenario_id` column on every entity row.
    pub scenario_id: u32,

    /// Per-stage detailed results.
    pub stages: Vec<StageWritePayload>,

    /// Rolling-seed release windows, scenario-level (no stage/node axis).
    /// Empty when the system declares no upstream travel-time arc.
    pub transit_seed: Vec<TransitSeedWriteRecord>,
}

/// One row of `simulation/paths.parquet`: the node visited at one
/// `(scenario, stage)` of the sampled walk.
///
/// Accumulated by [`SimulationParquetWriter::write_scenario`] and serialized by
/// the single-owner [`write_paths`]; the wire form the CLI gathers across MPI
/// ranks is three `i32`s in this field order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SimulationPathRecord {
    /// 0-based scenario identifier.
    pub scenario_id: i32,
    /// Stage index (0-based).
    pub stage_id: i32,
    /// Declared node id visited at this stage.
    pub node_id: i32,
}

// ---------------------------------------------------------------------------
// SimulationParquetWriter
// ---------------------------------------------------------------------------

/// Writes simulation results to Hive-partitioned Parquet files.
///
/// Designed to run on a dedicated I/O thread: it implements [`Send`] and
/// is moved to the background writer thread during the simulation pipeline.
///
/// # Construction
///
/// ```no_run
/// use cobre_io::output::simulation_writer::SimulationParquetWriter;
/// use std::path::Path;
///
/// # fn main() -> Result<(), cobre_io::OutputError> {
/// # let system = unimplemented!();
/// let writer = SimulationParquetWriter::new(Path::new("/tmp/out"), system)?;
/// # Ok(())
/// # }
/// ```
pub struct SimulationParquetWriter {
    output_dir: PathBuf,
    /// Hours, indexed `[stage_position][block_index]`; `stage_position` is the
    /// 0-based index into `system.stages()` (sorted by stage ID), which the
    /// simulation's 0-based `stage_id` values match for study stages (ID >= 0).
    block_durations: Vec<Vec<f64>>,
    /// Keyed by line entity ID, not indexed by position — line IDs are
    /// non-contiguous.
    loss_factors: HashMap<i32, f64>,
    scenarios_written: u32,
    /// One row per `(scenario, stage)` visited, accumulated across every
    /// `write_scenario` call; drained by [`Self::path_rows`] into the run-level
    /// `paths.parquet` after the scenario stream closes.
    path_rows: Vec<SimulationPathRecord>,
}

// Must stay Send: moved to a background I/O thread.
const _: fn() = || {
    fn assert_send<T: Send>() {}
    assert_send::<SimulationParquetWriter>();
};

impl SimulationParquetWriter {
    /// Create a new writer targeting `output_dir`.
    ///
    /// Creates the `simulation/` subdirectory and, for each family in
    /// [`SIMULATION_FAMILIES`] the system declares (see
    /// [`SimulationFamily::declared`]), that family's top-level directory.
    ///
    /// # Errors
    ///
    /// - [`OutputError::IoError`] if any directory cannot be created.
    pub fn new(output_dir: &Path, system: &System) -> Result<Self, OutputError> {
        let sim_dir = output_dir.join("simulation");

        let block_durations: Vec<Vec<f64>> = system
            .stages()
            .iter()
            .map(|s| s.blocks.iter().map(|b| b.duration_hours).collect())
            .collect();

        let loss_factors: HashMap<i32, f64> = system
            .lines()
            .iter()
            .map(|l| (l.id.0, 1.0 - l.losses_percent / 100.0))
            .collect();

        for family in SIMULATION_FAMILIES {
            if (family.declared)(system) {
                let dir = sim_dir.join(family.subpath);
                std::fs::create_dir_all(&dir).map_err(|e| OutputError::io(&dir, e))?;
            }
        }

        Ok(Self {
            output_dir: output_dir.to_path_buf(),
            block_durations,
            loss_factors,
            scenarios_written: 0,
            path_rows: Vec::new(),
        })
    }

    /// Create the partition directory and write `batch` as `data.parquet` —
    /// the single owner of the create-dir / write tail every entity type in
    /// [`Self::write_scenario`] shares. `subpath` is the entity's directory
    /// under `simulation/` (`"costs"`, `"violations/generic"`, ...).
    ///
    /// # Errors
    ///
    /// - [`OutputError::SerializationError`] if a `RecordBatch` cannot be
    ///   constructed (array length mismatch).
    /// - [`OutputError::IoError`] if any filesystem operation fails.
    fn write_partition(
        &self,
        subpath: &str,
        suffix: &str,
        batch: &RecordBatch,
    ) -> Result<(), OutputError> {
        let part_dir = self
            .output_dir
            .join("simulation")
            .join(subpath)
            .join(suffix);
        std::fs::create_dir_all(&part_dir).map_err(|e| OutputError::io(&part_dir, e))?;
        let file_path = part_dir.join("data.parquet");
        write_parquet_atomic(&file_path, batch)?;
        Ok(())
    }

    /// Write one scenario's results to Hive-partitioned Parquet files.
    ///
    /// Iterates [`SIMULATION_FAMILIES`]: a partition is written for a family
    /// only when this scenario's payload carries at least one row for it (see
    /// [`SimulationFamily`] for the full directory-vs-partition contract).
    ///
    /// # Errors
    ///
    /// - [`OutputError::SerializationError`] if a `RecordBatch` cannot be
    ///   constructed (array length mismatch).
    /// - [`OutputError::IoError`] if any filesystem operation fails.
    #[allow(clippy::needless_pass_by_value)] // consuming by value is intentional: payload drives output
    #[allow(clippy::cast_possible_wrap)] // scenario/stage ids are small non-negative indices
    pub fn write_scenario(&mut self, result: ScenarioWritePayload) -> Result<(), OutputError> {
        let id = result.scenario_id;
        let scenario_id = id as i32;
        let partition_suffix = format!("scenario_id={id:04}");

        // One path row per visited (scenario, stage); node_id is stage-uniform.
        self.path_rows
            .extend(result.stages.iter().map(|s| SimulationPathRecord {
                scenario_id,
                stage_id: s.stage_id as i32,
                node_id: s.node_id,
            }));

        for family in SIMULATION_FAMILIES {
            if let Some(batch) = (family.build)(self, &result, scenario_id)? {
                self.write_partition(family.subpath, &partition_suffix, &batch)?;
            }
        }

        self.scenarios_written += 1;
        Ok(())
    }

    /// The `(scenario_id, stage_id, node_id)` rows accumulated across every
    /// [`Self::write_scenario`] call, in arrival order.
    ///
    /// The caller is the run-level owner of `paths.parquet`: it feeds these rows
    /// (gathered across MPI ranks in the distributed CLI, taken directly in the
    /// single-process bindings) to [`write_paths`], which fixes the canonical
    /// order. Kept separate from [`Self::finalize`] so the rows survive the move
    /// that consumes the writer.
    #[must_use]
    pub fn path_rows(&self) -> &[SimulationPathRecord] {
        &self.path_rows
    }

    /// Finalize writing and return the [`SimulationOutput`] summary.
    ///
    /// `total_time_ms` is the caller-measured wall-clock run duration; it is
    /// written to the simulation manifest as `duration_seconds`.
    #[must_use]
    pub fn finalize(self, total_time_ms: u64) -> SimulationOutput {
        SimulationOutput {
            n_scenarios: self.scenarios_written,
            completed: self.scenarios_written,
            failed: 0,
            total_time_ms,
            cost: None,
            solve_stats: MetadataSimulationSolveStats::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// Entity family table
// ---------------------------------------------------------------------------

/// One row of the entity-family table: the single owner of which
/// Hive-partitioned entity types this writer knows about.
///
/// [`SimulationParquetWriter::new`] iterates [`SIMULATION_FAMILIES`] to create
/// each declared family's top-level directory; [`SimulationParquetWriter::write_scenario`]
/// iterates the same table to build and write each family's per-scenario
/// partition. A family's directory is created once, at construction, when
/// `declared` is true for the system — independent of any one scenario's
/// payload; a partition is written only when `build` returns `Some`, i.e. this
/// scenario's payload carries at least one row for the family. A declared
/// family can therefore end up with an empty top-level directory (no
/// partitions at all) if every scenario's payload happens to carry no rows for
/// it — nothing downstream of this writer assumes otherwise.
struct SimulationFamily {
    /// Directory subpath under `simulation/` (`"costs"`, `"violations/generic"`, …).
    subpath: &'static str,
    /// Whether the system declares this family at all; gates directory
    /// creation in [`SimulationParquetWriter::new`].
    declared: fn(&System) -> bool,
    /// Builds this scenario's batch for the family, or `None` when the
    /// scenario's payload carries no rows for it.
    build: fn(
        &SimulationParquetWriter,
        &ScenarioWritePayload,
        i32,
    ) -> Result<Option<RecordBatch>, OutputError>,
}

fn family_always_declared(_system: &System) -> bool {
    true
}

fn family_has_hydros(system: &System) -> bool {
    system.n_hydros() > 0
}

fn family_has_thermals(system: &System) -> bool {
    system.n_thermals() > 0
}

fn family_has_lines(system: &System) -> bool {
    system.n_lines() > 0
}

fn family_has_buses(system: &System) -> bool {
    system.n_buses() > 0
}

fn family_has_pumping_stations(system: &System) -> bool {
    system.n_pumping_stations() > 0
}

fn family_has_contracts(system: &System) -> bool {
    system.n_contracts() > 0
}

fn family_has_non_controllables(system: &System) -> bool {
    system.n_non_controllable_sources() > 0
}

/// A declared travel-time arc: `travel_time_hours > 0` and a downstream
/// target. Mirrors `bucket_topology`'s arc predicate.
fn family_declares_travel_time(system: &System) -> bool {
    system
        .hydros()
        .iter()
        .any(|h| h.travel_time_hours.is_some_and(|t| t > 0.0) && h.downstream_id.is_some())
}

fn family_has_generic_constraints(system: &System) -> bool {
    !system.generic_constraints().is_empty()
}

fn family_declares_post_study(system: &System) -> bool {
    system
        .post_study_stages()
        .is_some_and(|ps| !ps.stages.is_empty())
}

fn scenario_batch_costs(
    _writer: &SimulationParquetWriter,
    result: &ScenarioWritePayload,
    scenario_id: i32,
) -> Result<Option<RecordBatch>, OutputError> {
    let n: usize = result.stages.iter().map(|s| s.costs.len()).sum();
    if n == 0 {
        return Ok(None);
    }
    let batch = build_costs_batch(
        result.stages.iter().flat_map(|s| s.costs.iter()),
        scenario_id,
        n,
    )?;
    Ok(Some(batch))
}

fn scenario_batch_hydros(
    writer: &SimulationParquetWriter,
    result: &ScenarioWritePayload,
    scenario_id: i32,
) -> Result<Option<RecordBatch>, OutputError> {
    let n: usize = result.stages.iter().map(|s| s.hydros.len()).sum();
    if n == 0 {
        return Ok(None);
    }
    let batch = build_hydros_batch(
        result.stages.iter().flat_map(|s| s.hydros.iter()),
        scenario_id,
        &writer.block_durations,
        n,
    )?;
    Ok(Some(batch))
}

fn scenario_batch_hydro_bus_generation(
    writer: &SimulationParquetWriter,
    result: &ScenarioWritePayload,
    scenario_id: i32,
) -> Result<Option<RecordBatch>, OutputError> {
    let n: usize = result
        .stages
        .iter()
        .map(|s| s.hydro_bus_generation.len())
        .sum();
    if n == 0 {
        return Ok(None);
    }
    let batch = build_hydro_bus_generation_batch(
        result
            .stages
            .iter()
            .flat_map(|s| s.hydro_bus_generation.iter()),
        scenario_id,
        &writer.block_durations,
        n,
    )?;
    Ok(Some(batch))
}

fn scenario_batch_thermals(
    writer: &SimulationParquetWriter,
    result: &ScenarioWritePayload,
    scenario_id: i32,
) -> Result<Option<RecordBatch>, OutputError> {
    let n: usize = result.stages.iter().map(|s| s.thermals.len()).sum();
    if n == 0 {
        return Ok(None);
    }
    let batch = build_thermals_batch(
        result.stages.iter().flat_map(|s| s.thermals.iter()),
        scenario_id,
        &writer.block_durations,
        n,
    )?;
    Ok(Some(batch))
}

fn scenario_batch_exchanges(
    writer: &SimulationParquetWriter,
    result: &ScenarioWritePayload,
    scenario_id: i32,
) -> Result<Option<RecordBatch>, OutputError> {
    let n: usize = result.stages.iter().map(|s| s.exchanges.len()).sum();
    if n == 0 {
        return Ok(None);
    }
    let batch = build_exchanges_batch(
        result.stages.iter().flat_map(|s| s.exchanges.iter()),
        scenario_id,
        &writer.block_durations,
        &writer.loss_factors,
        n,
    )?;
    Ok(Some(batch))
}

fn scenario_batch_buses(
    writer: &SimulationParquetWriter,
    result: &ScenarioWritePayload,
    scenario_id: i32,
) -> Result<Option<RecordBatch>, OutputError> {
    let n: usize = result.stages.iter().map(|s| s.buses.len()).sum();
    if n == 0 {
        return Ok(None);
    }
    let batch = build_buses_batch(
        result.stages.iter().flat_map(|s| s.buses.iter()),
        scenario_id,
        &writer.block_durations,
        n,
    )?;
    Ok(Some(batch))
}

fn scenario_batch_pumping_stations(
    writer: &SimulationParquetWriter,
    result: &ScenarioWritePayload,
    scenario_id: i32,
) -> Result<Option<RecordBatch>, OutputError> {
    let n: usize = result.stages.iter().map(|s| s.pumping_stations.len()).sum();
    if n == 0 {
        return Ok(None);
    }
    let batch = build_pumping_batch(
        result.stages.iter().flat_map(|s| s.pumping_stations.iter()),
        scenario_id,
        &writer.block_durations,
        n,
    )?;
    Ok(Some(batch))
}

fn scenario_batch_contracts(
    writer: &SimulationParquetWriter,
    result: &ScenarioWritePayload,
    scenario_id: i32,
) -> Result<Option<RecordBatch>, OutputError> {
    let n: usize = result.stages.iter().map(|s| s.contracts.len()).sum();
    if n == 0 {
        return Ok(None);
    }
    let batch = build_contracts_batch(
        result.stages.iter().flat_map(|s| s.contracts.iter()),
        scenario_id,
        &writer.block_durations,
        n,
    )?;
    Ok(Some(batch))
}

fn scenario_batch_non_controllables(
    writer: &SimulationParquetWriter,
    result: &ScenarioWritePayload,
    scenario_id: i32,
) -> Result<Option<RecordBatch>, OutputError> {
    let n: usize = result
        .stages
        .iter()
        .map(|s| s.non_controllables.len())
        .sum();
    if n == 0 {
        return Ok(None);
    }
    let batch = build_non_controllables_batch(
        result
            .stages
            .iter()
            .flat_map(|s| s.non_controllables.iter()),
        scenario_id,
        &writer.block_durations,
        n,
    )?;
    Ok(Some(batch))
}

// Gated on hydro count, not a lag-specific predicate.
fn scenario_batch_inflow_lags(
    _writer: &SimulationParquetWriter,
    result: &ScenarioWritePayload,
    scenario_id: i32,
) -> Result<Option<RecordBatch>, OutputError> {
    let n: usize = result.stages.iter().map(|s| s.inflow_lags.len()).sum();
    if n == 0 {
        return Ok(None);
    }
    let batch = build_inflow_lags_batch(
        result.stages.iter().flat_map(|s| s.inflow_lags.iter()),
        scenario_id,
        n,
    )?;
    Ok(Some(batch))
}

fn scenario_batch_in_transit(
    _writer: &SimulationParquetWriter,
    result: &ScenarioWritePayload,
    scenario_id: i32,
) -> Result<Option<RecordBatch>, OutputError> {
    let n: usize = result.stages.iter().map(|s| s.transit_buckets.len()).sum();
    if n == 0 {
        return Ok(None);
    }
    let batch = build_in_transit_batch(
        result.stages.iter().flat_map(|s| s.transit_buckets.iter()),
        scenario_id,
        n,
    )?;
    Ok(Some(batch))
}

/// Scenario-level, not per-stage: unlike every other family, `transit_seed`'s
/// payload carries no stage axis, so its own predicate reads `result`
/// directly rather than folding over `result.stages`.
fn scenario_batch_transit_seed(
    _writer: &SimulationParquetWriter,
    result: &ScenarioWritePayload,
    scenario_id: i32,
) -> Result<Option<RecordBatch>, OutputError> {
    if result.transit_seed.is_empty() {
        return Ok(None);
    }
    let batch = build_transit_seed_batch(
        result.transit_seed.iter(),
        scenario_id,
        result.transit_seed.len(),
    )?;
    Ok(Some(batch))
}

fn scenario_batch_generic_violations(
    _writer: &SimulationParquetWriter,
    result: &ScenarioWritePayload,
    scenario_id: i32,
) -> Result<Option<RecordBatch>, OutputError> {
    let n: usize = result
        .stages
        .iter()
        .map(|s| s.generic_violations.len())
        .sum();
    if n == 0 {
        return Ok(None);
    }
    let batch = build_generic_violations_batch(
        result
            .stages
            .iter()
            .flat_map(|s| s.generic_violations.iter()),
        scenario_id,
        n,
    )?;
    Ok(Some(batch))
}

fn scenario_batch_anticipated_lanes(
    _writer: &SimulationParquetWriter,
    result: &ScenarioWritePayload,
    scenario_id: i32,
) -> Result<Option<RecordBatch>, OutputError> {
    let n: usize = result
        .stages
        .iter()
        .map(|s| s.anticipated_lanes.len())
        .sum();
    if n == 0 {
        return Ok(None);
    }
    let batch = build_anticipated_lanes_batch(
        result
            .stages
            .iter()
            .flat_map(|s| s.anticipated_lanes.iter()),
        scenario_id,
        n,
    )?;
    Ok(Some(batch))
}

/// The single owner of the entity-family set: every family this writer knows
/// about — directory creation ([`SimulationParquetWriter::new`]) and
/// per-scenario partitioning ([`SimulationParquetWriter::write_scenario`]) both
/// iterate this table rather than keeping their own family lists.
const SIMULATION_FAMILIES: &[SimulationFamily] = &[
    SimulationFamily {
        subpath: "costs",
        declared: family_always_declared,
        build: scenario_batch_costs,
    },
    SimulationFamily {
        subpath: "hydros",
        declared: family_has_hydros,
        build: scenario_batch_hydros,
    },
    SimulationFamily {
        subpath: "hydro_bus_generation",
        declared: family_has_hydros,
        build: scenario_batch_hydro_bus_generation,
    },
    SimulationFamily {
        subpath: "thermals",
        declared: family_has_thermals,
        build: scenario_batch_thermals,
    },
    SimulationFamily {
        subpath: "exchanges",
        declared: family_has_lines,
        build: scenario_batch_exchanges,
    },
    SimulationFamily {
        subpath: "buses",
        declared: family_has_buses,
        build: scenario_batch_buses,
    },
    SimulationFamily {
        subpath: "pumping_stations",
        declared: family_has_pumping_stations,
        build: scenario_batch_pumping_stations,
    },
    SimulationFamily {
        subpath: "contracts",
        declared: family_has_contracts,
        build: scenario_batch_contracts,
    },
    SimulationFamily {
        subpath: "non_controllables",
        declared: family_has_non_controllables,
        build: scenario_batch_non_controllables,
    },
    SimulationFamily {
        subpath: "inflow_lags",
        declared: family_has_hydros,
        build: scenario_batch_inflow_lags,
    },
    SimulationFamily {
        subpath: "in_transit",
        declared: family_declares_travel_time,
        build: scenario_batch_in_transit,
    },
    SimulationFamily {
        subpath: "transit_seed",
        declared: family_declares_travel_time,
        build: scenario_batch_transit_seed,
    },
    SimulationFamily {
        subpath: "violations/generic",
        declared: family_has_generic_constraints,
        build: scenario_batch_generic_violations,
    },
    SimulationFamily {
        subpath: "anticipated_lanes",
        declared: family_declares_post_study,
        build: scenario_batch_anticipated_lanes,
    },
];

/// The `simulation/` subpath of every family the writer emits, in
/// [`SIMULATION_FAMILIES`] order — the single owner, so a consumer that iterates
/// this sees each new family without keeping a second list to drift.
pub fn simulation_family_subpaths() -> impl Iterator<Item = &'static str> {
    SIMULATION_FAMILIES.iter().map(|family| family.subpath)
}

const PATHS_FILE: &str = "paths.parquet";
const SCENARIO_SUMMARY_FILE: &str = "scenario_summary.parquet";

/// Write the run-level, unpartitioned `simulation/paths.parquet` from the
/// per-`(scenario, stage)` node-path rows.
///
/// Single owner of the file: it fixes the canonical `(scenario_id, stage_id)`
/// order so the output is identical regardless of the order scenarios drained or
/// which MPI rank contributed which scenario (declaration-order-invariant and
/// rank-invariant). Uses the crate-default Parquet codec, matching the other
/// run-level simulation files.
///
/// # Errors
///
/// - [`OutputError::IoError`] if the `simulation/` directory or file cannot be
///   written.
/// - [`OutputError::SerializationError`] if the `RecordBatch` cannot be built.
pub fn write_paths(
    output_dir: &Path,
    mut rows: Vec<SimulationPathRecord>,
) -> Result<(), OutputError> {
    rows.sort_by_key(|r| (r.scenario_id, r.stage_id));

    let n = rows.len();
    let mut scenario_id = Int32Builder::with_capacity(n);
    let mut stage_id = Int32Builder::with_capacity(n);
    let mut node_id = Int32Builder::with_capacity(n);
    for r in &rows {
        scenario_id.append_value(r.scenario_id);
        stage_id.append_value(r.stage_id);
        node_id.append_value(r.node_id);
    }

    let batch = RecordBatch::try_new(
        Arc::new(paths_schema()),
        vec![
            Arc::new(scenario_id.finish()),
            Arc::new(stage_id.finish()),
            Arc::new(node_id.finish()),
        ],
    )
    .map_err(|e| OutputError::serialization("paths", e.to_string()))?;

    let sim_dir = output_dir.join("simulation");
    std::fs::create_dir_all(&sim_dir).map_err(|e| OutputError::io(&sim_dir, e))?;
    write_parquet_atomic(&sim_dir.join(PATHS_FILE), &batch)
}

/// Write the run-level, unpartitioned `simulation/scenario_summary.parquet` from
/// the gathered per-scenario `(scenario_id, probability, discounted_immediate_cost)`
/// rows.
///
/// Single owner of the file. `rows` arrive in ascending `scenario_id` (the
/// canonical order the caller's cross-rank gather already fixed) and are written
/// verbatim — never re-sorted or re-reduced here — so the output is identical
/// across rank and thread shapes. `probability` is `Some` per row only under a
/// declared census and `None` under sampled selection.
///
/// # Errors
///
/// - [`OutputError::IoError`] if the `simulation/` directory or file cannot be
///   written.
/// - [`OutputError::SerializationError`] if the `RecordBatch` cannot be built.
#[allow(clippy::cast_possible_wrap)] // scenario ids are small non-negative indices
pub fn write_scenario_summary(
    output_dir: &Path,
    rows: &[(u32, Option<f64>, f64)],
) -> Result<(), OutputError> {
    let n = rows.len();
    let mut scenario_id = Int32Builder::with_capacity(n);
    let mut probability = Float64Builder::with_capacity(n);
    let mut discounted_immediate_cost = Float64Builder::with_capacity(n);
    for &(id, prob, cost) in rows {
        scenario_id.append_value(id as i32);
        probability.append_option(prob);
        discounted_immediate_cost.append_value(cost);
    }

    let batch = RecordBatch::try_new(
        Arc::new(scenario_summary_schema()),
        vec![
            Arc::new(scenario_id.finish()),
            Arc::new(probability.finish()),
            Arc::new(discounted_immediate_cost.finish()),
        ],
    )
    .map_err(|e| OutputError::serialization("scenario_summary", e.to_string()))?;

    let sim_dir = output_dir.join("simulation");
    std::fs::create_dir_all(&sim_dir).map_err(|e| OutputError::io(&sim_dir, e))?;
    write_parquet_atomic(&sim_dir.join(SCENARIO_SUMMARY_FILE), &batch)
}

/// Remove an earlier run's simulation outputs under `output_dir`: every family's
/// per-scenario partition tree, `solver/iterations.parquet`,
/// `solver/retry_histogram.parquet`, `paths.parquet`, `scenario_summary.parquet`
/// and `metadata.json`, plus `solver/` and a nested family's parents when that
/// leaves them empty. Callers invoke it before the phase's first write, right
/// after removing the stale `_SUCCESS` marker.
///
/// # Errors
///
/// Returns [`OutputError::IoError`] for the first entry that exists but cannot be
/// removed, such as a regular file at a family path or a directory at a file
/// path: the entry is not one this writer wrote.
pub fn remove_simulation_outputs(output_dir: &Path) -> Result<(), OutputError> {
    let sim_dir = output_dir.join("simulation");
    for subpath in simulation_family_subpaths() {
        let family_dir = sim_dir.join(subpath);
        match std::fs::remove_dir_all(&family_dir) {
            Ok(()) => {}
            Err(e) if e.kind() == ErrorKind::NotFound => {}
            Err(e) => return Err(OutputError::io(&family_dir, e)),
        }
        for parent in family_dir
            .ancestors()
            .skip(1)
            .take_while(|dir| *dir != sim_dir)
        {
            remove_dir_if_empty(parent)?;
        }
    }

    let solver_dir = output_dir.join(SIMULATION_SOLVER_DIR);
    for file in [SOLVER_ITERATIONS_FILE, SOLVER_RETRY_HISTOGRAM_FILE] {
        remove_file_if_present(&solver_dir.join(file))?;
    }
    remove_dir_if_empty(&solver_dir)?;

    for file in [
        sim_dir.join(PATHS_FILE),
        sim_dir.join(SCENARIO_SUMMARY_FILE),
        output_dir.join(SIMULATION_METADATA_FILE),
    ] {
        remove_file_if_present(&file)?;
    }
    Ok(())
}

pub(super) fn remove_file_if_present(path: &Path) -> Result<(), OutputError> {
    match std::fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == ErrorKind::NotFound => Ok(()),
        Err(e) => Err(OutputError::io(path, e)),
    }
}

pub(super) fn remove_dir_if_empty(dir: &Path) -> Result<(), OutputError> {
    match std::fs::remove_dir(dir) {
        Ok(()) => Ok(()),
        Err(e) if matches!(e.kind(), ErrorKind::NotFound | ErrorKind::DirectoryNotEmpty) => Ok(()),
        Err(e) => Err(OutputError::io(dir, e)),
    }
}

// ---------------------------------------------------------------------------
// Block duration lookup helper
// ---------------------------------------------------------------------------

/// Look up the duration in hours for the block identified by `(stage_id, block_id)`.
///
/// Returns `1.0` for a `None` `block_id` (stage-level rows) and as an
/// out-of-range fallback, so the duration-scaled columns stay identity.
fn block_duration(block_durations: &[Vec<f64>], stage_id: u32, block_id: Option<u32>) -> f64 {
    let Some(block_idx) = block_id else {
        return 1.0;
    };
    let stage_idx = stage_id as usize;
    block_durations
        .get(stage_idx)
        .and_then(|blocks| blocks.get(block_idx as usize))
        .copied()
        .unwrap_or(1.0)
}

// ---------------------------------------------------------------------------
// RecordBatch builders
// ---------------------------------------------------------------------------

#[allow(clippy::cast_possible_wrap)]
fn build_costs_batch<'a>(
    records: impl IntoIterator<Item = &'a CostWriteRecord>,
    scenario_id: i32,
    n: usize,
) -> Result<RecordBatch, OutputError> {
    let schema = costs_schema();

    let mut scenario_id_col = Int32Builder::with_capacity(n);
    let mut stage_id = Int32Builder::with_capacity(n);
    let mut node_id = Int32Builder::with_capacity(n);
    let mut block_id = Int32Builder::with_capacity(n);
    let mut total_cost = Float64Builder::with_capacity(n);
    let mut immediate_cost = Float64Builder::with_capacity(n);
    let mut future_cost = Float64Builder::with_capacity(n);
    let mut discount_factor = Float64Builder::with_capacity(n);
    let mut thermal_cost = Float64Builder::with_capacity(n);
    let mut anticipated_thermal_cost = Float64Builder::with_capacity(n);
    let mut contract_cost = Float64Builder::with_capacity(n);
    let mut deficit_cost = Float64Builder::with_capacity(n);
    let mut excess_cost = Float64Builder::with_capacity(n);
    let mut storage_violation_cost = Float64Builder::with_capacity(n);
    let mut filling_target_cost = Float64Builder::with_capacity(n);
    let mut hydro_violation_cost = Float64Builder::with_capacity(n);
    let mut outflow_violation_below_cost = Float64Builder::with_capacity(n);
    let mut outflow_violation_above_cost = Float64Builder::with_capacity(n);
    let mut turbined_violation_cost = Float64Builder::with_capacity(n);
    let mut generation_violation_cost = Float64Builder::with_capacity(n);
    let mut evaporation_violation_cost = Float64Builder::with_capacity(n);
    let mut withdrawal_violation_cost = Float64Builder::with_capacity(n);
    let mut inflow_penalty_cost = Float64Builder::with_capacity(n);
    let mut generic_violation_cost = Float64Builder::with_capacity(n);
    let mut spillage_cost = Float64Builder::with_capacity(n);
    let mut turbined_cost = Float64Builder::with_capacity(n);
    let mut curtailment_cost = Float64Builder::with_capacity(n);
    let mut exchange_cost = Float64Builder::with_capacity(n);
    let mut pumping_cost = Float64Builder::with_capacity(n);

    for r in records {
        scenario_id_col.append_value(scenario_id);
        stage_id.append_value(r.stage_id as i32);
        node_id.append_value(r.node_id);
        block_id.append_option(r.block_id.map(|b| b as i32));
        total_cost.append_value(r.total_cost);
        immediate_cost.append_value(r.immediate_cost);
        future_cost.append_value(r.future_cost);
        discount_factor.append_value(r.discount_factor);
        thermal_cost.append_value(r.thermal_cost);
        anticipated_thermal_cost.append_value(r.anticipated_thermal_cost);
        contract_cost.append_value(r.contract_cost);
        deficit_cost.append_value(r.deficit_cost);
        excess_cost.append_value(r.excess_cost);
        storage_violation_cost.append_value(r.storage_violation_cost);
        filling_target_cost.append_value(r.filling_target_cost);
        hydro_violation_cost.append_value(r.hydro_violation_cost);
        outflow_violation_below_cost.append_value(r.outflow_violation_below_cost);
        outflow_violation_above_cost.append_value(r.outflow_violation_above_cost);
        turbined_violation_cost.append_value(r.turbined_violation_cost);
        generation_violation_cost.append_value(r.generation_violation_cost);
        evaporation_violation_cost.append_value(r.evaporation_violation_cost);
        withdrawal_violation_cost.append_value(r.withdrawal_violation_cost);
        inflow_penalty_cost.append_value(r.inflow_penalty_cost);
        generic_violation_cost.append_value(r.generic_violation_cost);
        spillage_cost.append_value(r.spillage_cost);
        turbined_cost.append_value(r.turbined_cost);
        curtailment_cost.append_value(r.curtailment_cost);
        exchange_cost.append_value(r.exchange_cost);
        pumping_cost.append_value(r.pumping_cost);
    }

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(scenario_id_col.finish()),
            Arc::new(stage_id.finish()),
            Arc::new(node_id.finish()),
            Arc::new(block_id.finish()),
            Arc::new(total_cost.finish()),
            Arc::new(immediate_cost.finish()),
            Arc::new(future_cost.finish()),
            Arc::new(discount_factor.finish()),
            Arc::new(thermal_cost.finish()),
            Arc::new(anticipated_thermal_cost.finish()),
            Arc::new(contract_cost.finish()),
            Arc::new(deficit_cost.finish()),
            Arc::new(excess_cost.finish()),
            Arc::new(storage_violation_cost.finish()),
            Arc::new(filling_target_cost.finish()),
            Arc::new(hydro_violation_cost.finish()),
            Arc::new(outflow_violation_below_cost.finish()),
            Arc::new(outflow_violation_above_cost.finish()),
            Arc::new(turbined_violation_cost.finish()),
            Arc::new(generation_violation_cost.finish()),
            Arc::new(evaporation_violation_cost.finish()),
            Arc::new(withdrawal_violation_cost.finish()),
            Arc::new(inflow_penalty_cost.finish()),
            Arc::new(generic_violation_cost.finish()),
            Arc::new(spillage_cost.finish()),
            Arc::new(turbined_cost.finish()),
            Arc::new(curtailment_cost.finish()),
            Arc::new(exchange_cost.finish()),
            Arc::new(pumping_cost.finish()),
        ],
    )
    .map_err(|e| OutputError::serialization("costs", e.to_string()))
}

/// Arrow column builders for the hydros `RecordBatch`, grouped into one struct
/// so [`fill_hydro_builders`] takes a single argument rather than one per column.
struct HydroBuilders {
    scenario_id: Int32Builder,
    stage_id: Int32Builder,
    node_id: Int32Builder,
    block_id: Int32Builder,
    hydro_id: Int32Builder,
    turbined_m3s: Float64Builder,
    spillage_m3s: Float64Builder,
    outflow_m3s: Float64Builder,
    evaporation_m3s: Float64Builder,
    diverted_inflow_m3s: Float64Builder,
    diverted_outflow_m3s: Float64Builder,
    incremental_inflow_m3s: Float64Builder,
    inflow_m3s: Float64Builder,
    storage_initial_hm3: Float64Builder,
    storage_final_hm3: Float64Builder,
    generation_mw: Float64Builder,
    generation_mwh: Float64Builder,
    equivalent_productivity_mw_per_m3s: Float64Builder,
    accumulated_productivity_mw_per_m3s: Float64Builder,
    incremental_inflow_energy_mw: Float64Builder,
    stored_energy_initial_mwh: Float64Builder,
    stored_energy_final_mwh: Float64Builder,
    spillage_cost: Float64Builder,
    water_value_per_hm3: Float64Builder,
    storage_binding_code: Int8Builder,
    operative_state_code: Int8Builder,
    turbined_slack_m3s: Float64Builder,
    outflow_slack_below_m3s: Float64Builder,
    outflow_slack_above_m3s: Float64Builder,
    generation_slack_mw: Float64Builder,
    storage_violation_below_hm3: Float64Builder,
    filling_target_violation_hm3: Float64Builder,
    evaporation_violation_pos_m3s: Float64Builder,
    evaporation_violation_neg_m3s: Float64Builder,
    inflow_nonnegativity_slack_m3s: Float64Builder,
    water_withdrawal_violation_pos_m3s: Float64Builder,
    water_withdrawal_violation_neg_m3s: Float64Builder,
    integrated_equivalent_productivity_mw_per_m3s: Float64Builder,
    integrated_accumulated_productivity_mw_per_m3s: Float64Builder,
    stored_energy_initial_mw: Float64Builder,
    stored_energy_final_mw: Float64Builder,
}

impl HydroBuilders {
    fn with_capacity(n: usize) -> Self {
        Self {
            scenario_id: Int32Builder::with_capacity(n),
            stage_id: Int32Builder::with_capacity(n),
            node_id: Int32Builder::with_capacity(n),
            block_id: Int32Builder::with_capacity(n),
            hydro_id: Int32Builder::with_capacity(n),
            turbined_m3s: Float64Builder::with_capacity(n),
            spillage_m3s: Float64Builder::with_capacity(n),
            outflow_m3s: Float64Builder::with_capacity(n),
            evaporation_m3s: Float64Builder::with_capacity(n),
            diverted_inflow_m3s: Float64Builder::with_capacity(n),
            diverted_outflow_m3s: Float64Builder::with_capacity(n),
            incremental_inflow_m3s: Float64Builder::with_capacity(n),
            inflow_m3s: Float64Builder::with_capacity(n),
            storage_initial_hm3: Float64Builder::with_capacity(n),
            storage_final_hm3: Float64Builder::with_capacity(n),
            generation_mw: Float64Builder::with_capacity(n),
            generation_mwh: Float64Builder::with_capacity(n),
            equivalent_productivity_mw_per_m3s: Float64Builder::with_capacity(n),
            accumulated_productivity_mw_per_m3s: Float64Builder::with_capacity(n),
            incremental_inflow_energy_mw: Float64Builder::with_capacity(n),
            stored_energy_initial_mwh: Float64Builder::with_capacity(n),
            stored_energy_final_mwh: Float64Builder::with_capacity(n),
            spillage_cost: Float64Builder::with_capacity(n),
            water_value_per_hm3: Float64Builder::with_capacity(n),
            storage_binding_code: Int8Builder::with_capacity(n),
            operative_state_code: Int8Builder::with_capacity(n),
            turbined_slack_m3s: Float64Builder::with_capacity(n),
            outflow_slack_below_m3s: Float64Builder::with_capacity(n),
            outflow_slack_above_m3s: Float64Builder::with_capacity(n),
            generation_slack_mw: Float64Builder::with_capacity(n),
            storage_violation_below_hm3: Float64Builder::with_capacity(n),
            filling_target_violation_hm3: Float64Builder::with_capacity(n),
            evaporation_violation_pos_m3s: Float64Builder::with_capacity(n),
            evaporation_violation_neg_m3s: Float64Builder::with_capacity(n),
            inflow_nonnegativity_slack_m3s: Float64Builder::with_capacity(n),
            water_withdrawal_violation_pos_m3s: Float64Builder::with_capacity(n),
            water_withdrawal_violation_neg_m3s: Float64Builder::with_capacity(n),
            integrated_equivalent_productivity_mw_per_m3s: Float64Builder::with_capacity(n),
            integrated_accumulated_productivity_mw_per_m3s: Float64Builder::with_capacity(n),
            stored_energy_initial_mw: Float64Builder::with_capacity(n),
            stored_energy_final_mw: Float64Builder::with_capacity(n),
        }
    }
}

/// Append one row per `HydroWriteRecord` into `b`. Derived columns:
/// - `outflow_m3s = turbined_m3s + spillage_m3s`
/// - `generation_mwh = generation_mw * block_duration_hours`
#[allow(clippy::cast_possible_wrap)]
fn fill_hydro_builders<'a>(
    records: impl IntoIterator<Item = &'a HydroWriteRecord>,
    scenario_id: i32,
    block_durations: &[Vec<f64>],
    b: &mut HydroBuilders,
) {
    for r in records {
        let dur = block_duration(block_durations, r.stage_id, r.block_id);
        b.scenario_id.append_value(scenario_id);
        b.stage_id.append_value(r.stage_id as i32);
        b.node_id.append_value(r.node_id);
        b.block_id.append_option(r.block_id.map(|v| v as i32));
        b.hydro_id.append_value(r.hydro_id);
        b.turbined_m3s.append_value(r.turbined_m3s);
        b.spillage_m3s.append_value(r.spillage_m3s);
        b.outflow_m3s.append_value(r.turbined_m3s + r.spillage_m3s);
        b.evaporation_m3s.append_option(r.evaporation_m3s);
        b.diverted_inflow_m3s.append_option(r.diverted_inflow_m3s);
        b.diverted_outflow_m3s.append_option(r.diverted_outflow_m3s);
        b.incremental_inflow_m3s
            .append_value(r.incremental_inflow_m3s);
        b.inflow_m3s.append_value(r.inflow_m3s);
        b.storage_initial_hm3.append_value(r.storage_initial_hm3);
        b.storage_final_hm3.append_value(r.storage_final_hm3);
        b.generation_mw.append_value(r.generation_mw);
        b.generation_mwh.append_value(r.generation_mw * dur);
        b.equivalent_productivity_mw_per_m3s
            .append_value(r.equivalent_productivity_mw_per_m3s);
        b.accumulated_productivity_mw_per_m3s
            .append_value(r.accumulated_productivity_mw_per_m3s);
        b.incremental_inflow_energy_mw
            .append_value(r.incremental_inflow_energy_mw);
        b.stored_energy_initial_mwh
            .append_value(r.stored_energy_initial_mwh);
        b.stored_energy_final_mwh
            .append_value(r.stored_energy_final_mwh);
        b.spillage_cost.append_value(r.spillage_cost);
        b.water_value_per_hm3.append_value(r.water_value_per_hm3);
        b.storage_binding_code.append_value(r.storage_binding_code);
        b.operative_state_code.append_value(r.operative_state_code);
        b.turbined_slack_m3s.append_value(r.turbined_slack_m3s);
        b.outflow_slack_below_m3s
            .append_value(r.outflow_slack_below_m3s);
        b.outflow_slack_above_m3s
            .append_value(r.outflow_slack_above_m3s);
        b.generation_slack_mw.append_value(r.generation_slack_mw);
        b.storage_violation_below_hm3
            .append_value(r.storage_violation_below_hm3);
        b.filling_target_violation_hm3
            .append_value(r.filling_target_violation_hm3);
        b.evaporation_violation_pos_m3s
            .append_value(r.evaporation_violation_pos_m3s);
        b.evaporation_violation_neg_m3s
            .append_value(r.evaporation_violation_neg_m3s);
        b.inflow_nonnegativity_slack_m3s
            .append_value(r.inflow_nonnegativity_slack_m3s);
        b.water_withdrawal_violation_pos_m3s
            .append_value(r.water_withdrawal_violation_pos_m3s);
        b.water_withdrawal_violation_neg_m3s
            .append_value(r.water_withdrawal_violation_neg_m3s);
        b.integrated_equivalent_productivity_mw_per_m3s
            .append_value(r.integrated_equivalent_productivity_mw_per_m3s);
        b.integrated_accumulated_productivity_mw_per_m3s
            .append_value(r.integrated_accumulated_productivity_mw_per_m3s);
        b.stored_energy_initial_mw
            .append_value(r.stored_energy_initial_mw);
        b.stored_energy_final_mw
            .append_value(r.stored_energy_final_mw);
    }
}

/// Build the hydros `RecordBatch`; derived columns are filled by
/// [`fill_hydro_builders`].
fn build_hydros_batch<'a>(
    records: impl IntoIterator<Item = &'a HydroWriteRecord>,
    scenario_id: i32,
    block_durations: &[Vec<f64>],
    n: usize,
) -> Result<RecordBatch, OutputError> {
    let schema = hydros_schema();
    let mut b = HydroBuilders::with_capacity(n);
    fill_hydro_builders(records, scenario_id, block_durations, &mut b);
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(b.scenario_id.finish()),
            Arc::new(b.stage_id.finish()),
            Arc::new(b.node_id.finish()),
            Arc::new(b.block_id.finish()),
            Arc::new(b.hydro_id.finish()),
            Arc::new(b.turbined_m3s.finish()),
            Arc::new(b.spillage_m3s.finish()),
            Arc::new(b.outflow_m3s.finish()),
            Arc::new(b.evaporation_m3s.finish()),
            Arc::new(b.diverted_inflow_m3s.finish()),
            Arc::new(b.diverted_outflow_m3s.finish()),
            Arc::new(b.incremental_inflow_m3s.finish()),
            Arc::new(b.inflow_m3s.finish()),
            Arc::new(b.storage_initial_hm3.finish()),
            Arc::new(b.storage_final_hm3.finish()),
            Arc::new(b.generation_mw.finish()),
            Arc::new(b.generation_mwh.finish()),
            Arc::new(b.equivalent_productivity_mw_per_m3s.finish()),
            Arc::new(b.accumulated_productivity_mw_per_m3s.finish()),
            Arc::new(b.incremental_inflow_energy_mw.finish()),
            Arc::new(b.stored_energy_initial_mwh.finish()),
            Arc::new(b.stored_energy_final_mwh.finish()),
            Arc::new(b.spillage_cost.finish()),
            Arc::new(b.water_value_per_hm3.finish()),
            Arc::new(b.storage_binding_code.finish()),
            Arc::new(b.operative_state_code.finish()),
            Arc::new(b.turbined_slack_m3s.finish()),
            Arc::new(b.outflow_slack_below_m3s.finish()),
            Arc::new(b.outflow_slack_above_m3s.finish()),
            Arc::new(b.generation_slack_mw.finish()),
            Arc::new(b.storage_violation_below_hm3.finish()),
            Arc::new(b.filling_target_violation_hm3.finish()),
            Arc::new(b.evaporation_violation_pos_m3s.finish()),
            Arc::new(b.evaporation_violation_neg_m3s.finish()),
            Arc::new(b.inflow_nonnegativity_slack_m3s.finish()),
            Arc::new(b.water_withdrawal_violation_pos_m3s.finish()),
            Arc::new(b.water_withdrawal_violation_neg_m3s.finish()),
            Arc::new(b.integrated_equivalent_productivity_mw_per_m3s.finish()),
            Arc::new(b.integrated_accumulated_productivity_mw_per_m3s.finish()),
            Arc::new(b.stored_energy_initial_mw.finish()),
            Arc::new(b.stored_energy_final_mw.finish()),
        ],
    )
    .map_err(|e| OutputError::serialization("hydros", e.to_string()))
}

/// Build the `hydro_bus_generation` `RecordBatch`. Derived column:
/// - `generation_mwh = generation_mw * block_duration_hours`
#[allow(clippy::cast_possible_wrap)]
fn build_hydro_bus_generation_batch<'a>(
    records: impl IntoIterator<Item = &'a HydroBusWriteRecord>,
    scenario_id: i32,
    block_durations: &[Vec<f64>],
    n: usize,
) -> Result<RecordBatch, OutputError> {
    let schema = hydro_bus_generation_schema();

    let mut scenario_id_col = Int32Builder::with_capacity(n);
    let mut stage_id = Int32Builder::with_capacity(n);
    let mut node_id = Int32Builder::with_capacity(n);
    let mut block_id = Int32Builder::with_capacity(n);
    let mut hydro_id = Int32Builder::with_capacity(n);
    let mut bus_id = Int32Builder::with_capacity(n);
    let mut turbined_m3s = Float64Builder::with_capacity(n);
    let mut generation_mw = Float64Builder::with_capacity(n);
    let mut generation_mwh = Float64Builder::with_capacity(n);

    for r in records {
        let dur = block_duration(block_durations, r.stage_id, r.block_id);
        scenario_id_col.append_value(scenario_id);
        stage_id.append_value(r.stage_id as i32);
        node_id.append_value(r.node_id);
        block_id.append_option(r.block_id.map(|b| b as i32));
        hydro_id.append_value(r.hydro_id);
        bus_id.append_value(r.bus_id);
        turbined_m3s.append_value(r.turbined_m3s);
        generation_mw.append_value(r.generation_mw);
        generation_mwh.append_value(r.generation_mw * dur);
    }

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(scenario_id_col.finish()),
            Arc::new(stage_id.finish()),
            Arc::new(node_id.finish()),
            Arc::new(block_id.finish()),
            Arc::new(hydro_id.finish()),
            Arc::new(bus_id.finish()),
            Arc::new(turbined_m3s.finish()),
            Arc::new(generation_mw.finish()),
            Arc::new(generation_mwh.finish()),
        ],
    )
    .map_err(|e| OutputError::serialization("hydro_bus_generation", e.to_string()))
}

/// Build the thermals `RecordBatch`. Derived column:
/// - `generation_mwh = generation_mw * block_duration_hours`
#[allow(clippy::cast_possible_wrap)]
fn build_thermals_batch<'a>(
    records: impl IntoIterator<Item = &'a ThermalWriteRecord>,
    scenario_id: i32,
    block_durations: &[Vec<f64>],
    n: usize,
) -> Result<RecordBatch, OutputError> {
    let schema = thermals_schema();

    let mut scenario_id_col = Int32Builder::with_capacity(n);
    let mut stage_id = Int32Builder::with_capacity(n);
    let mut node_id = Int32Builder::with_capacity(n);
    let mut block_id = Int32Builder::with_capacity(n);
    let mut thermal_id = Int32Builder::with_capacity(n);
    let mut generation_mw = Float64Builder::with_capacity(n);
    let mut generation_mwh = Float64Builder::with_capacity(n);
    let mut generation_cost = Float64Builder::with_capacity(n);
    let mut is_anticipated = BooleanBuilder::with_capacity(n);
    let mut anticipated_committed_mw = Float64Builder::with_capacity(n);
    let mut anticipated_decision_mw = Float64Builder::with_capacity(n);
    let mut operative_state_code = Int8Builder::with_capacity(n);

    for r in records {
        let dur = block_duration(block_durations, r.stage_id, r.block_id);
        scenario_id_col.append_value(scenario_id);
        stage_id.append_value(r.stage_id as i32);
        node_id.append_value(r.node_id);
        block_id.append_option(r.block_id.map(|b| b as i32));
        thermal_id.append_value(r.thermal_id);
        generation_mw.append_value(r.generation_mw);
        generation_mwh.append_value(r.generation_mw * dur);
        generation_cost.append_value(r.generation_cost);
        is_anticipated.append_value(r.is_anticipated);
        anticipated_committed_mw.append_option(r.anticipated_committed_mw);
        anticipated_decision_mw.append_option(r.anticipated_decision_mw);
        operative_state_code.append_value(r.operative_state_code);
    }

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(scenario_id_col.finish()),
            Arc::new(stage_id.finish()),
            Arc::new(node_id.finish()),
            Arc::new(block_id.finish()),
            Arc::new(thermal_id.finish()),
            Arc::new(generation_mw.finish()),
            Arc::new(generation_mwh.finish()),
            Arc::new(generation_cost.finish()),
            Arc::new(is_anticipated.finish()),
            Arc::new(anticipated_committed_mw.finish()),
            Arc::new(anticipated_decision_mw.finish()),
            Arc::new(operative_state_code.finish()),
        ],
    )
    .map_err(|e| OutputError::serialization("thermals", e.to_string()))
}

/// Build the exchanges `RecordBatch`. Derived columns:
/// - `net_flow_mw = direct_flow_mw - reverse_flow_mw`
/// - `net_flow_mwh = net_flow_mw * block_duration_hours`
/// - `losses_mw = (1.0 - loss_factor) * (direct_flow_mw + reverse_flow_mw)`
/// - `losses_mwh = losses_mw * block_duration_hours`
///
/// A `line_id` absent from `loss_factors` defaults to loss factor `1.0`
/// (zero losses).
#[allow(
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::similar_names // MW / MWh builder pairs are semantically paired and intentionally similar
)]
fn build_exchanges_batch<'a>(
    records: impl IntoIterator<Item = &'a ExchangeWriteRecord>,
    scenario_id: i32,
    block_durations: &[Vec<f64>],
    loss_factors: &HashMap<i32, f64>,
    n: usize,
) -> Result<RecordBatch, OutputError> {
    let schema = exchanges_schema();

    let mut scenario_id_col = Int32Builder::with_capacity(n);
    let mut stage_id = Int32Builder::with_capacity(n);
    let mut node_id = Int32Builder::with_capacity(n);
    let mut block_id = Int32Builder::with_capacity(n);
    let mut line_id = Int32Builder::with_capacity(n);
    let mut direct_flow_mw = Float64Builder::with_capacity(n);
    let mut reverse_flow_mw = Float64Builder::with_capacity(n);
    let mut net_flow_mw_col = Float64Builder::with_capacity(n);
    let mut net_flow_mwh_col = Float64Builder::with_capacity(n);
    let mut losses_mw_col = Float64Builder::with_capacity(n);
    let mut losses_mwh_col = Float64Builder::with_capacity(n);
    let mut exchange_cost = Float64Builder::with_capacity(n);
    let mut operative_state_code = Int8Builder::with_capacity(n);

    for r in records {
        let dur = block_duration(block_durations, r.stage_id, r.block_id);

        let lf = loss_factors.get(&r.line_id).copied().unwrap_or(1.0);

        let net = r.direct_flow_mw - r.reverse_flow_mw;
        let total_flow = r.direct_flow_mw + r.reverse_flow_mw;
        let losses = (1.0 - lf) * total_flow;

        scenario_id_col.append_value(scenario_id);
        stage_id.append_value(r.stage_id as i32);
        node_id.append_value(r.node_id);
        block_id.append_option(r.block_id.map(|b| b as i32));
        line_id.append_value(r.line_id);
        direct_flow_mw.append_value(r.direct_flow_mw);
        reverse_flow_mw.append_value(r.reverse_flow_mw);
        net_flow_mw_col.append_value(net);
        net_flow_mwh_col.append_value(net * dur);
        losses_mw_col.append_value(losses);
        losses_mwh_col.append_value(losses * dur);
        exchange_cost.append_value(r.exchange_cost);
        operative_state_code.append_value(r.operative_state_code);
    }

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(scenario_id_col.finish()),
            Arc::new(stage_id.finish()),
            Arc::new(node_id.finish()),
            Arc::new(block_id.finish()),
            Arc::new(line_id.finish()),
            Arc::new(direct_flow_mw.finish()),
            Arc::new(reverse_flow_mw.finish()),
            Arc::new(net_flow_mw_col.finish()),
            Arc::new(net_flow_mwh_col.finish()),
            Arc::new(losses_mw_col.finish()),
            Arc::new(losses_mwh_col.finish()),
            Arc::new(exchange_cost.finish()),
            Arc::new(operative_state_code.finish()),
        ],
    )
    .map_err(|e| OutputError::serialization("exchanges", e.to_string()))
}

/// Build the buses `RecordBatch`. Derived columns:
/// - `load_mwh = load_mw * block_duration_hours`
/// - `deficit_mwh = deficit_mw * block_duration_hours`
/// - `excess_mwh = excess_mw * block_duration_hours`
#[allow(clippy::cast_possible_wrap)]
fn build_buses_batch<'a>(
    records: impl IntoIterator<Item = &'a BusWriteRecord>,
    scenario_id: i32,
    block_durations: &[Vec<f64>],
    n: usize,
) -> Result<RecordBatch, OutputError> {
    let schema = buses_schema();

    let mut scenario_id_col = Int32Builder::with_capacity(n);
    let mut stage_id = Int32Builder::with_capacity(n);
    let mut node_id = Int32Builder::with_capacity(n);
    let mut block_id = Int32Builder::with_capacity(n);
    let mut bus_id = Int32Builder::with_capacity(n);
    let mut load_mw = Float64Builder::with_capacity(n);
    let mut load_mwh = Float64Builder::with_capacity(n);
    let mut deficit_mw = Float64Builder::with_capacity(n);
    let mut deficit_mwh = Float64Builder::with_capacity(n);
    let mut excess_mw = Float64Builder::with_capacity(n);
    let mut excess_mwh = Float64Builder::with_capacity(n);
    let mut spot_price = Float64Builder::with_capacity(n);

    for r in records {
        let dur = block_duration(block_durations, r.stage_id, r.block_id);
        scenario_id_col.append_value(scenario_id);
        stage_id.append_value(r.stage_id as i32);
        node_id.append_value(r.node_id);
        block_id.append_option(r.block_id.map(|b| b as i32));
        bus_id.append_value(r.bus_id);
        load_mw.append_value(r.load_mw);
        load_mwh.append_value(r.load_mw * dur);
        deficit_mw.append_value(r.deficit_mw);
        deficit_mwh.append_value(r.deficit_mw * dur);
        excess_mw.append_value(r.excess_mw);
        excess_mwh.append_value(r.excess_mw * dur);
        spot_price.append_value(r.spot_price);
    }

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(scenario_id_col.finish()),
            Arc::new(stage_id.finish()),
            Arc::new(node_id.finish()),
            Arc::new(block_id.finish()),
            Arc::new(bus_id.finish()),
            Arc::new(load_mw.finish()),
            Arc::new(load_mwh.finish()),
            Arc::new(deficit_mw.finish()),
            Arc::new(deficit_mwh.finish()),
            Arc::new(excess_mw.finish()),
            Arc::new(excess_mwh.finish()),
            Arc::new(spot_price.finish()),
        ],
    )
    .map_err(|e| OutputError::serialization("buses", e.to_string()))
}

/// Build the `pumping_stations` `RecordBatch`. The writer is the only layer
/// holding per-block `block_duration_hours`, so the two duration-scaled columns
/// are derived here, **never** at extraction:
///
/// - `pumped_volume_hm3 = pumped_flow_m3s * block_duration_hours * 3600.0 / 1e6`
///   — do **not** substitute `M3S_TO_HM3`, which carries its own duration factor
///   and would double-count it.
/// - `energy_consumption_mwh = power_consumption_mw * block_duration_hours`.
///
/// `power_consumption_mw` and `pumping_cost` are **not** recomputed here —
/// `power_consumption_mw` is not a function of block duration, and `pumping_cost`
/// preserves the single imputed-`0.0` default owned at extraction.
#[allow(clippy::cast_possible_wrap)]
fn build_pumping_batch<'a>(
    records: impl IntoIterator<Item = &'a PumpingWriteRecord>,
    scenario_id: i32,
    block_durations: &[Vec<f64>],
    n: usize,
) -> Result<RecordBatch, OutputError> {
    let schema = pumping_stations_schema();

    let mut scenario_id_col = Int32Builder::with_capacity(n);
    let mut stage_id = Int32Builder::with_capacity(n);
    let mut node_id = Int32Builder::with_capacity(n);
    let mut block_id = Int32Builder::with_capacity(n);
    let mut pumping_station_id = Int32Builder::with_capacity(n);
    let mut pumped_flow_m3s = Float64Builder::with_capacity(n);
    let mut pumped_volume_hm3 = Float64Builder::with_capacity(n);
    let mut power_consumption_mw = Float64Builder::with_capacity(n);
    let mut energy_consumption_mwh = Float64Builder::with_capacity(n);
    let mut pumping_cost = Float64Builder::with_capacity(n);
    let mut operative_state_code = Int8Builder::with_capacity(n);

    for r in records {
        let dur = block_duration(block_durations, r.stage_id, r.block_id);
        scenario_id_col.append_value(scenario_id);
        stage_id.append_value(r.stage_id as i32);
        node_id.append_value(r.node_id);
        block_id.append_option(r.block_id.map(|b| b as i32));
        pumping_station_id.append_value(r.pumping_station_id);
        pumped_flow_m3s.append_value(r.pumped_flow_m3s);
        pumped_volume_hm3.append_value(r.pumped_flow_m3s * dur * 3600.0 / 1_000_000.0);
        power_consumption_mw.append_value(r.power_consumption_mw);
        energy_consumption_mwh.append_value(r.power_consumption_mw * dur);
        pumping_cost.append_value(r.pumping_cost);
        operative_state_code.append_value(r.operative_state_code);
    }

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(scenario_id_col.finish()),
            Arc::new(stage_id.finish()),
            Arc::new(node_id.finish()),
            Arc::new(block_id.finish()),
            Arc::new(pumping_station_id.finish()),
            Arc::new(pumped_flow_m3s.finish()),
            Arc::new(pumped_volume_hm3.finish()),
            Arc::new(power_consumption_mw.finish()),
            Arc::new(energy_consumption_mwh.finish()),
            Arc::new(pumping_cost.finish()),
            Arc::new(operative_state_code.finish()),
        ],
    )
    .map_err(|e| OutputError::serialization("pumping_stations", e.to_string()))
}

/// Build the contracts `RecordBatch`. Derived column:
/// - `energy_mwh = power_mw * block_duration_hours`
#[allow(clippy::cast_possible_wrap)]
fn build_contracts_batch<'a>(
    records: impl IntoIterator<Item = &'a ContractWriteRecord>,
    scenario_id: i32,
    block_durations: &[Vec<f64>],
    n: usize,
) -> Result<RecordBatch, OutputError> {
    let schema = contracts_schema();

    let mut scenario_id_col = Int32Builder::with_capacity(n);
    let mut stage_id = Int32Builder::with_capacity(n);
    let mut node_id = Int32Builder::with_capacity(n);
    let mut block_id = Int32Builder::with_capacity(n);
    let mut contract_id = Int32Builder::with_capacity(n);
    let mut power_mw = Float64Builder::with_capacity(n);
    let mut energy_mwh = Float64Builder::with_capacity(n);
    let mut price_per_mwh = Float64Builder::with_capacity(n);
    let mut total_cost = Float64Builder::with_capacity(n);
    let mut operative_state_code = Int8Builder::with_capacity(n);

    for r in records {
        let dur = block_duration(block_durations, r.stage_id, r.block_id);
        scenario_id_col.append_value(scenario_id);
        stage_id.append_value(r.stage_id as i32);
        node_id.append_value(r.node_id);
        block_id.append_option(r.block_id.map(|b| b as i32));
        contract_id.append_value(r.contract_id);
        power_mw.append_value(r.power_mw);
        energy_mwh.append_value(r.power_mw * dur);
        price_per_mwh.append_value(r.price_per_mwh);
        total_cost.append_value(r.total_cost);
        operative_state_code.append_value(r.operative_state_code);
    }

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(scenario_id_col.finish()),
            Arc::new(stage_id.finish()),
            Arc::new(node_id.finish()),
            Arc::new(block_id.finish()),
            Arc::new(contract_id.finish()),
            Arc::new(power_mw.finish()),
            Arc::new(energy_mwh.finish()),
            Arc::new(price_per_mwh.finish()),
            Arc::new(total_cost.finish()),
            Arc::new(operative_state_code.finish()),
        ],
    )
    .map_err(|e| OutputError::serialization("contracts", e.to_string()))
}

/// Build the `non_controllables` `RecordBatch`. Derived columns:
/// - `generation_mwh = generation_mw * block_duration_hours`
/// - `curtailment_mwh = curtailment_mw * block_duration_hours`
#[allow(clippy::cast_possible_wrap)]
fn build_non_controllables_batch<'a>(
    records: impl IntoIterator<Item = &'a NonControllableWriteRecord>,
    scenario_id: i32,
    block_durations: &[Vec<f64>],
    n: usize,
) -> Result<RecordBatch, OutputError> {
    let schema = non_controllables_schema();

    let mut scenario_id_col = Int32Builder::with_capacity(n);
    let mut stage_id = Int32Builder::with_capacity(n);
    let mut node_id = Int32Builder::with_capacity(n);
    let mut block_id = Int32Builder::with_capacity(n);
    let mut non_controllable_id = Int32Builder::with_capacity(n);
    let mut generation_mw = Float64Builder::with_capacity(n);
    let mut generation_mwh = Float64Builder::with_capacity(n);
    let mut available_mw = Float64Builder::with_capacity(n);
    let mut curtailment_mw = Float64Builder::with_capacity(n);
    let mut curtailment_mwh = Float64Builder::with_capacity(n);
    let mut curtailment_cost = Float64Builder::with_capacity(n);
    let mut operative_state_code = Int8Builder::with_capacity(n);

    for r in records {
        let dur = block_duration(block_durations, r.stage_id, r.block_id);
        scenario_id_col.append_value(scenario_id);
        stage_id.append_value(r.stage_id as i32);
        node_id.append_value(r.node_id);
        block_id.append_option(r.block_id.map(|b| b as i32));
        non_controllable_id.append_value(r.non_controllable_id);
        generation_mw.append_value(r.generation_mw);
        generation_mwh.append_value(r.generation_mw * dur);
        available_mw.append_value(r.available_mw);
        curtailment_mw.append_value(r.curtailment_mw);
        curtailment_mwh.append_value(r.curtailment_mw * dur);
        curtailment_cost.append_value(r.curtailment_cost);
        operative_state_code.append_value(r.operative_state_code);
    }

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(scenario_id_col.finish()),
            Arc::new(stage_id.finish()),
            Arc::new(node_id.finish()),
            Arc::new(block_id.finish()),
            Arc::new(non_controllable_id.finish()),
            Arc::new(generation_mw.finish()),
            Arc::new(generation_mwh.finish()),
            Arc::new(available_mw.finish()),
            Arc::new(curtailment_mw.finish()),
            Arc::new(curtailment_mwh.finish()),
            Arc::new(curtailment_cost.finish()),
            Arc::new(operative_state_code.finish()),
        ],
    )
    .map_err(|e| OutputError::serialization("non_controllables", e.to_string()))
}

#[allow(clippy::cast_possible_wrap)]
fn build_inflow_lags_batch<'a>(
    records: impl IntoIterator<Item = &'a InflowLagWriteRecord>,
    scenario_id: i32,
    n: usize,
) -> Result<RecordBatch, OutputError> {
    let schema = inflow_lags_schema();

    let mut scenario_id_col = Int32Builder::with_capacity(n);
    let mut stage_id = Int32Builder::with_capacity(n);
    let mut node_id = Int32Builder::with_capacity(n);
    let mut hydro_id = Int32Builder::with_capacity(n);
    let mut lag_index = Int32Builder::with_capacity(n);
    let mut inflow_m3s = Float64Builder::with_capacity(n);

    for r in records {
        scenario_id_col.append_value(scenario_id);
        stage_id.append_value(r.stage_id as i32);
        node_id.append_value(r.node_id);
        hydro_id.append_value(r.hydro_id);
        lag_index.append_value(r.lag_index as i32);
        inflow_m3s.append_value(r.inflow_m3s);
    }

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(scenario_id_col.finish()),
            Arc::new(stage_id.finish()),
            Arc::new(node_id.finish()),
            Arc::new(hydro_id.finish()),
            Arc::new(lag_index.finish()),
            Arc::new(inflow_m3s.finish()),
        ],
    )
    .map_err(|e| OutputError::serialization("inflow_lags", e.to_string()))
}

#[allow(clippy::cast_possible_wrap)]
fn build_in_transit_batch<'a>(
    records: impl IntoIterator<Item = &'a TransitBucketWriteRecord>,
    scenario_id: i32,
    n: usize,
) -> Result<RecordBatch, OutputError> {
    let schema = in_transit_schema();

    let mut scenario_id_col = Int32Builder::with_capacity(n);
    let mut stage_id = Int32Builder::with_capacity(n);
    let mut node_id = Int32Builder::with_capacity(n);
    let mut hydro_id = Int32Builder::with_capacity(n);
    let mut lag = Int32Builder::with_capacity(n);
    let mut in_transit_volume_hm3 = Float64Builder::with_capacity(n);
    let mut delayed_arrival_hm3 = Float64Builder::with_capacity(n);

    for r in records {
        scenario_id_col.append_value(scenario_id);
        stage_id.append_value(r.stage_id as i32);
        node_id.append_value(r.node_id);
        hydro_id.append_value(r.hydro_id);
        lag.append_value(r.lag as i32);
        in_transit_volume_hm3.append_value(r.in_transit_volume_hm3);
        delayed_arrival_hm3.append_value(r.delayed_arrival_hm3);
    }

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(scenario_id_col.finish()),
            Arc::new(stage_id.finish()),
            Arc::new(node_id.finish()),
            Arc::new(hydro_id.finish()),
            Arc::new(lag.finish()),
            Arc::new(in_transit_volume_hm3.finish()),
            Arc::new(delayed_arrival_hm3.finish()),
        ],
    )
    .map_err(|e| OutputError::serialization("in_transit", e.to_string()))
}

#[allow(clippy::cast_possible_wrap)]
fn build_transit_seed_batch<'a>(
    records: impl IntoIterator<Item = &'a TransitSeedWriteRecord>,
    scenario_id: i32,
    n: usize,
) -> Result<RecordBatch, OutputError> {
    let schema = transit_seed_schema();

    let mut scenario_id_col = Int32Builder::with_capacity(n);
    let mut hydro_id = Int32Builder::with_capacity(n);
    let mut start_date = Date32Builder::with_capacity(n);
    let mut end_date = Date32Builder::with_capacity(n);
    let mut value_m3s = Float64Builder::with_capacity(n);

    for r in records {
        scenario_id_col.append_value(scenario_id);
        hydro_id.append_value(r.hydro_id);
        start_date.append_value(date32_days(r.start_date));
        end_date.append_value(date32_days(r.end_date));
        value_m3s.append_value(r.value_m3s);
    }

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(scenario_id_col.finish()),
            Arc::new(hydro_id.finish()),
            Arc::new(start_date.finish()),
            Arc::new(end_date.finish()),
            Arc::new(value_m3s.finish()),
        ],
    )
    .map_err(|e| OutputError::serialization("transit_seed", e.to_string()))
}

#[allow(clippy::cast_possible_wrap)]
fn build_generic_violations_batch<'a>(
    records: impl IntoIterator<Item = &'a GenericViolationWriteRecord>,
    scenario_id: i32,
    n: usize,
) -> Result<RecordBatch, OutputError> {
    let schema = generic_violations_schema();

    let mut scenario_id_col = Int32Builder::with_capacity(n);
    let mut stage_id = Int32Builder::with_capacity(n);
    let mut node_id = Int32Builder::with_capacity(n);
    let mut block_id = Int32Builder::with_capacity(n);
    let mut constraint_id = Int32Builder::with_capacity(n);
    let mut slack_value = Float64Builder::with_capacity(n);
    let mut slack_cost = Float64Builder::with_capacity(n);

    for r in records {
        scenario_id_col.append_value(scenario_id);
        stage_id.append_value(r.stage_id as i32);
        node_id.append_value(r.node_id);
        block_id.append_option(r.block_id.map(|b| b as i32));
        constraint_id.append_value(r.constraint_id);
        slack_value.append_value(r.slack_value);
        slack_cost.append_value(r.slack_cost);
    }

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(scenario_id_col.finish()),
            Arc::new(stage_id.finish()),
            Arc::new(node_id.finish()),
            Arc::new(block_id.finish()),
            Arc::new(constraint_id.finish()),
            Arc::new(slack_value.finish()),
            Arc::new(slack_cost.finish()),
        ],
    )
    .map_err(|e| OutputError::serialization("generic_violations", e.to_string()))
}

#[allow(clippy::cast_possible_wrap)]
fn build_anticipated_lanes_batch<'a>(
    records: impl IntoIterator<Item = &'a AnticipatedLaneWriteRecord>,
    scenario_id: i32,
    n: usize,
) -> Result<RecordBatch, OutputError> {
    let schema = anticipated_lanes_schema();

    let mut scenario_id_col = Int32Builder::with_capacity(n);
    let mut stage_id = Int32Builder::with_capacity(n);
    let mut node_id = Int32Builder::with_capacity(n);
    let mut thermal_id = Int32Builder::with_capacity(n);
    let mut delivery_date = Int32Builder::with_capacity(n);
    let mut deposited_decision_mw = Float64Builder::with_capacity(n);
    let mut carried_committed_mw = Float64Builder::with_capacity(n);

    for r in records {
        scenario_id_col.append_value(scenario_id);
        stage_id.append_value(r.stage_id as i32);
        node_id.append_value(r.node_id);
        thermal_id.append_value(r.thermal_id);
        delivery_date.append_value(r.delivery_date);
        deposited_decision_mw.append_value(r.deposited_decision_mw);
        carried_committed_mw.append_value(r.carried_committed_mw);
    }

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(scenario_id_col.finish()),
            Arc::new(stage_id.finish()),
            Arc::new(node_id.finish()),
            Arc::new(thermal_id.finish()),
            Arc::new(delivery_date.finish()),
            Arc::new(deposited_decision_mw.finish()),
            Arc::new(carried_committed_mw.finish()),
        ],
    )
    .map_err(|e| OutputError::serialization("anticipated_lanes", e.to_string()))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::float_cmp,
    clippy::panic
)]
mod tests {
    use super::*;
    use crate::test_support::output::read_first_batch;
    use chrono::NaiveDate;
    use cobre_core::{
        Block, BlockMode, Bus, DeficitSegment, EntityId, Hydro, HydroGenerationModel,
        HydroPenalties, Line, NoiseMethod, PumpingStation, ScenarioSourceConfig, Stage,
        StageRiskConfig, StageStateConfig, SystemBuilder, Thermal,
    };

    // -----------------------------------------------------------------------
    // Test fixture helpers
    // -----------------------------------------------------------------------

    fn make_hydro_penalties_zero() -> HydroPenalties {
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
            inflow_nonnegativity_cost: 1000.0,
        }
    }

    fn make_hydro(id: i32) -> Hydro {
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
            max_storage_hm3: 1000.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 1000.0,
            specific_productivity_mw_per_m3s_per_m: None,
            min_generation_mw: 0.0,
            max_generation_mw: 900.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: make_hydro_penalties_zero(),
        };
        hydro.declare_mirror_unit_group(EntityId(1));
        hydro
    }

    fn make_stage(id: i32, duration_hours: f64) -> Stage {
        Stage {
            index: u32::try_from(id.max(0)).unwrap_or(0) as usize,
            id,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: Some(0),
            blocks: vec![Block {
                index: 0,
                name: "SINGLE".to_string(),
                duration_hours,
            }],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: true,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 10,
                noise_method: NoiseMethod::Saa,
            },
        }
    }

    /// A minimal two-stage, one-block-per-stage system with 2 hydros,
    /// 1 thermal, 1 bus, and 1 line (2.5% losses).
    fn make_test_system() -> System {
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

        let line = Line {
            id: EntityId(1),
            name: "L1".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            source_bus_id: EntityId(1),
            target_bus_id: EntityId(1),
            entry_stage_id: None,
            exit_stage_id: None,
            direct_capacity_mw: 500.0,
            reverse_capacity_mw: 500.0,
            losses_percent: 2.5,
            exchange_cost: 0.0,
        };

        let hydro1 = make_hydro(1);
        let hydro2 = make_hydro(2);

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
            anticipated_config: None,
        };

        // Stage 0: duration 720h; Stage 1: duration 744h.
        let stage0 = make_stage(0, 720.0);
        let stage1 = make_stage(1, 744.0);

        SystemBuilder::new()
            .buses(vec![bus])
            .lines(vec![line])
            .hydros(vec![hydro1, hydro2])
            .thermals(vec![thermal])
            .stages(vec![stage0, stage1])
            .build()
            .expect("test system must be valid")
    }

    /// The minimal [`make_test_system`] topology plus one pumping station, so
    /// `n_pumping_stations() > 0` and the writer's directory gate fires. The
    /// station references bus 1 and hydros 1/2, all present in the base system.
    fn make_test_system_with_pumping() -> System {
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

        let line = Line {
            id: EntityId(1),
            name: "L1".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            source_bus_id: EntityId(1),
            target_bus_id: EntityId(1),
            entry_stage_id: None,
            exit_stage_id: None,
            direct_capacity_mw: 500.0,
            reverse_capacity_mw: 500.0,
            losses_percent: 2.5,
            exchange_cost: 0.0,
        };

        let pumping_station = PumpingStation {
            id: EntityId(1),
            name: "P1".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(1),
            source_hydro_id: EntityId(1),
            destination_hydro_id: EntityId(2),
            entry_stage_id: None,
            exit_stage_id: None,
            consumption_mw_per_m3s: 0.5,
            min_flow_m3s: 0.0,
            max_flow_m3s: 150.0,
        };

        SystemBuilder::new()
            .buses(vec![bus])
            .lines(vec![line])
            .hydros(vec![make_hydro(1), make_hydro(2)])
            .pumping_stations(vec![pumping_station])
            .stages(vec![make_stage(0, 720.0), make_stage(1, 744.0)])
            .build()
            .expect("test system with pumping must be valid")
    }

    fn make_pumping_record(
        stage_id: u32,
        block_id: Option<u32>,
        pumping_station_id: i32,
        pumped_flow_m3s: f64,
        power_consumption_mw: f64,
    ) -> PumpingWriteRecord {
        PumpingWriteRecord {
            stage_id,
            node_id: stage_id as i32,
            block_id,
            pumping_station_id,
            pumped_flow_m3s,
            power_consumption_mw,
            pumping_cost: 0.0,
            operative_state_code: 1,
        }
    }

    fn make_cost_record(stage_id: u32, block_id: Option<u32>) -> CostWriteRecord {
        CostWriteRecord {
            stage_id,
            node_id: stage_id as i32,
            block_id,
            total_cost: 1000.0,
            immediate_cost: 800.0,
            future_cost: 200.0,
            discount_factor: 0.95,
            thermal_cost: 400.0,
            anticipated_thermal_cost: 0.0,
            contract_cost: 0.0,
            deficit_cost: 100.0,
            excess_cost: 0.0,
            storage_violation_cost: 0.0,
            filling_target_cost: 0.0,
            hydro_violation_cost: 0.0,
            outflow_violation_below_cost: 0.0,
            outflow_violation_above_cost: 0.0,
            turbined_violation_cost: 0.0,
            generation_violation_cost: 0.0,
            evaporation_violation_cost: 0.0,
            withdrawal_violation_cost: 0.0,
            inflow_penalty_cost: 0.0,
            generic_violation_cost: 0.0,
            spillage_cost: 5.0,
            turbined_cost: 3.0,
            curtailment_cost: 0.0,
            exchange_cost: 2.0,
            pumping_cost: 0.0,
        }
    }

    fn make_hydro_record(stage_id: u32, block_id: Option<u32>, hydro_id: i32) -> HydroWriteRecord {
        HydroWriteRecord {
            stage_id,
            node_id: stage_id as i32,
            block_id,
            hydro_id,
            turbined_m3s: 80.0,
            spillage_m3s: 10.0,
            evaporation_m3s: None,
            diverted_inflow_m3s: None,
            diverted_outflow_m3s: None,
            incremental_inflow_m3s: 100.0,
            inflow_m3s: 100.0,
            storage_initial_hm3: 500.0,
            storage_final_hm3: 495.0,
            generation_mw: 50.0,
            equivalent_productivity_mw_per_m3s: 0.9,
            accumulated_productivity_mw_per_m3s: 2.7,
            incremental_inflow_energy_mw: 135.0,
            stored_energy_initial_mwh: 1234.5,
            stored_energy_final_mwh: 1240.0,
            spillage_cost: 10.0,
            water_value_per_hm3: 5.0,
            storage_binding_code: 0,
            operative_state_code: 1,
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
            integrated_equivalent_productivity_mw_per_m3s: 1.5,
            integrated_accumulated_productivity_mw_per_m3s: 6.0,
            stored_energy_initial_mw: 1.7145,
            stored_energy_final_mw: 1.7222,
        }
    }

    fn make_scenario_payload(scenario_id: u32, n_stages: usize) -> ScenarioWritePayload {
        let stages = (0..n_stages as u32)
            .map(|s| StageWritePayload {
                stage_id: s,
                node_id: s as i32,
                costs: vec![make_cost_record(s, Some(0))],
                hydros: vec![
                    make_hydro_record(s, Some(0), 1),
                    make_hydro_record(s, Some(0), 2),
                ],
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
            })
            .collect();
        ScenarioWritePayload {
            scenario_id,
            stages,
            transit_seed: vec![],
        }
    }

    // -----------------------------------------------------------------------
    // Unit tests: batch builders
    // -----------------------------------------------------------------------

    #[test]
    fn build_costs_batch_from_two_stages() {
        let r0 = make_cost_record(0, Some(0));
        let r1 = make_cost_record(1, Some(0));
        let records = [&r0, &r1];
        let batch = build_costs_batch(records.iter().copied(), 0, records.len())
            .expect("costs batch must build");

        assert_eq!(batch.num_rows(), 2, "must have 2 rows");
        assert_eq!(batch.num_columns(), 29, "costs schema has 29 columns");

        let expected = costs_schema();
        assert_eq!(
            batch.schema().fields(),
            expected.fields(),
            "schema must match costs_schema()"
        );
    }

    #[test]
    fn build_hydros_batch_derived_columns() {
        // Stage 0 has block 0 with duration 720h; stage 1 has block 0 with 744h.
        let block_durations = vec![vec![720.0_f64], vec![744.0_f64]];

        let mut r0 = make_hydro_record(0, Some(0), 1); // generation_mw = 50.0
        r0.water_withdrawal_violation_pos_m3s = 2.5; // nonzero for round-trip test
        let r1 = make_hydro_record(1, Some(0), 2); // generation_mw = 50.0
        let records = [&r0, &r1];

        let batch = build_hydros_batch(records.iter().copied(), 0, &block_durations, records.len())
            .expect("hydros batch must build");
        assert_eq!(batch.num_rows(), 2);
        assert_eq!(batch.num_columns(), 41, "hydros schema has 41 columns");

        let gen_mwh_col = batch
            .column_by_name("generation_mwh")
            .expect("generation_mwh column must exist");
        let gen_mwh_arr = gen_mwh_col
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .expect("generation_mwh must be Float64Array");

        // row 0: 50.0 * 720.0 = 36_000.0
        assert_eq!(
            gen_mwh_arr.value(0),
            50.0 * 720.0,
            "generation_mwh row 0 must equal generation_mw * duration"
        );
        // row 1: 50.0 * 744.0 = 37_200.0
        assert_eq!(
            gen_mwh_arr.value(1),
            50.0 * 744.0,
            "generation_mwh row 1 must equal generation_mw * duration"
        );

        let outflow_col = batch
            .column_by_name("outflow_m3s")
            .expect("outflow_m3s column must exist");
        let outflow_arr = outflow_col
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .expect("outflow_m3s must be Float64Array");
        // outflow = turbined (80.0) + spillage (10.0) = 90.0
        assert_eq!(
            outflow_arr.value(0),
            90.0,
            "outflow_m3s must equal turbined + spillage"
        );
        assert_eq!(outflow_arr.value(1), 90.0);

        let ww_col = batch
            .column_by_name("water_withdrawal_violation_pos_m3s")
            .expect("water_withdrawal_violation_pos_m3s column must exist");
        let ww_arr = ww_col
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .expect("water_withdrawal_violation_pos_m3s must be Float64Array");
        assert_eq!(
            ww_arr.value(0),
            2.5,
            "row 0 withdrawal violation must be 2.5"
        );
        assert_eq!(
            ww_arr.value(1),
            0.0,
            "row 1 withdrawal violation must be 0.0"
        );
    }

    #[test]
    fn build_pumping_batch_derived_columns_and_schema() {
        // One stage, one block, 730 hours.
        let block_durations = vec![vec![730.0_f64]];

        let r = PumpingWriteRecord {
            stage_id: 0,
            node_id: 0,
            block_id: Some(0),
            pumping_station_id: 1,
            pumped_flow_m3s: 10.0,
            power_consumption_mw: 5.0,
            pumping_cost: 0.0, // imputed default owned by SimulationPumpingResult::pumping_cost
            operative_state_code: 1,
        };
        let records = [&r];

        let batch =
            build_pumping_batch(records.iter().copied(), 0, &block_durations, records.len())
                .expect("pumping batch must build");
        assert_eq!(batch.num_rows(), 1);

        // Schema is field-for-field equal to pumping_stations_schema() (11 fields).
        let expected = pumping_stations_schema();
        assert_eq!(
            batch.schema().fields(),
            expected.fields(),
            "schema must match pumping_stations_schema()"
        );
        assert_eq!(batch.num_columns(), 11, "pumping schema has 11 columns");

        let f64_col = |name: &str| {
            batch
                .column_by_name(name)
                .unwrap_or_else(|| panic!("{name} column must exist"))
                .as_any()
                .downcast_ref::<arrow::array::Float64Array>()
                .unwrap_or_else(|| panic!("{name} must be Float64Array"))
                .value(0)
        };

        // pumped_volume_hm3 = pumped_flow_m3s * dur * 3600.0 / 1e6 (writer-derived).
        assert_eq!(
            f64_col("pumped_volume_hm3"),
            10.0 * 730.0 * 3600.0 / 1_000_000.0,
            "pumped_volume_hm3 must equal pumped_flow_m3s * dur * 3600 / 1e6"
        );
        // energy_consumption_mwh = power_consumption_mw * dur (writer-derived).
        assert_eq!(
            f64_col("energy_consumption_mwh"),
            5.0 * 730.0,
            "energy_consumption_mwh must equal power_consumption_mw * dur"
        );
        // power_consumption_mw forwarded verbatim (set at extraction, not recomputed).
        assert_eq!(
            f64_col("power_consumption_mw"),
            5.0,
            "power_consumption_mw must be forwarded verbatim"
        );
        // pumping_cost is the imputed 0.0 default, forwarded verbatim.
        assert_eq!(
            f64_col("pumping_cost"),
            0.0,
            "pumping_cost must be the imputed 0.0 default"
        );
    }

    #[test]
    fn build_exchanges_batch_net_flow_and_losses() {
        // One stage, one block, 720 hours.
        let block_durations = vec![vec![720.0_f64]];
        // Line 1 has losses_percent=2.5 → loss_factor=0.975 → (1-lf)=0.025
        let loss_factors = HashMap::from([(1, 0.975_f64)]);

        let r = ExchangeWriteRecord {
            stage_id: 0,
            node_id: 0,
            block_id: Some(0),
            line_id: 1,
            direct_flow_mw: 100.0,
            reverse_flow_mw: 0.0,
            exchange_cost: 5.0,
            operative_state_code: 1,
        };
        let records = [&r];

        let batch = build_exchanges_batch(
            records.iter().copied(),
            0,
            &block_durations,
            &loss_factors,
            records.len(),
        )
        .expect("exchanges batch must build");
        assert_eq!(batch.num_rows(), 1);
        assert_eq!(batch.num_columns(), 13, "exchanges schema has 13 columns");

        let net_flow_col = batch
            .column_by_name("net_flow_mw")
            .expect("net_flow_mw column must exist");
        let net_flow_arr = net_flow_col
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .expect("net_flow_mw must be Float64Array");
        assert_eq!(net_flow_arr.value(0), 100.0, "net_flow_mw must be 100.0");

        let losses_col = batch
            .column_by_name("losses_mw")
            .expect("losses_mw column must exist");
        let losses_arr = losses_col
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .expect("losses_mw must be Float64Array");
        // (1 - 0.975) * (100 + 0) = 0.025 * 100 = 2.5
        // Use approximate comparison to handle IEEE 754 rounding.
        assert!(
            (losses_arr.value(0) - 2.5).abs() < 1e-10,
            "losses_mw must equal 2.5, got {}",
            losses_arr.value(0)
        );
    }

    #[test]
    fn build_costs_batch_block_id_nullable() {
        // block_id=None must produce a null value in the Arrow array.
        let r_with = make_cost_record(0, Some(0));
        let r_without = make_cost_record(1, None);
        let records = [&r_with, &r_without];

        let batch = build_costs_batch(records.iter().copied(), 0, records.len())
            .expect("costs batch must build");
        let block_col = batch
            .column_by_name("block_id")
            .expect("block_id column must exist");

        assert!(!block_col.is_null(0), "row 0: Some(0) must not be null");
        assert!(block_col.is_null(1), "row 1: None must be null");
    }

    // -----------------------------------------------------------------------
    // Unit tests: writer integration
    // -----------------------------------------------------------------------

    #[test]
    fn simulation_parquet_writer_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<SimulationParquetWriter>();
    }

    #[test]
    fn write_scenario_creates_hive_partitions() {
        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        std::fs::create_dir_all(tmp.path().join("simulation")).unwrap();

        let system = make_test_system();

        let mut writer =
            SimulationParquetWriter::new(tmp.path(), &system).expect("new must succeed");

        let payload = make_scenario_payload(0, 2);
        writer
            .write_scenario(payload)
            .expect("write_scenario must succeed");

        // costs partition
        assert!(
            tmp.path()
                .join("simulation/costs/scenario_id=0000/data.parquet")
                .exists(),
            "simulation/costs/scenario_id=0000/data.parquet must exist"
        );

        // hydros partition
        assert!(
            tmp.path()
                .join("simulation/hydros/scenario_id=0000/data.parquet")
                .exists(),
            "simulation/hydros/scenario_id=0000/data.parquet must exist"
        );
    }

    #[test]
    fn entity_rows_carry_node_id_and_scenario_id_columns() {
        use arrow::array::Array;

        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        std::fs::create_dir_all(tmp.path().join("simulation")).unwrap();

        // make_test_system() declares no nodes[]; make_scenario_payload stamps the
        // degenerate per-stage node id (node_id == stage_id). scenario_id = 3.
        let system = make_test_system();
        let mut writer =
            SimulationParquetWriter::new(tmp.path(), &system).expect("new must succeed");
        writer
            .write_scenario(make_scenario_payload(3, 2))
            .expect("write_scenario must succeed");

        let path = tmp
            .path()
            .join("simulation/hydros/scenario_id=0003/data.parquet");
        let batch = read_first_batch(&path);

        for col in &["scenario_id", "stage_id", "node_id"] {
            let field = batch
                .schema()
                .field_with_name(col)
                .unwrap_or_else(|_| panic!("{col} must be a real column"))
                .clone();
            assert_eq!(field.data_type(), &arrow::datatypes::DataType::Int32);
            assert!(!field.is_nullable(), "{col} must be non-null");
        }

        let i32_col = |name: &str| {
            batch
                .column_by_name(name)
                .unwrap_or_else(|| panic!("{name} column must exist"))
                .as_any()
                .downcast_ref::<arrow::array::Int32Array>()
                .unwrap_or_else(|| panic!("{name} must be Int32Array"))
                .clone()
        };
        let scenario = i32_col("scenario_id");
        let stage = i32_col("stage_id");
        let node = i32_col("node_id");
        assert_eq!(scenario.null_count(), 0, "scenario_id must be non-null");
        assert_eq!(node.null_count(), 0, "node_id must be non-null");
        for row in 0..batch.num_rows() {
            assert_eq!(
                scenario.value(row),
                3,
                "scenario_id column equals the partition"
            );
            // Degenerate chain node id equals the stage id on every row.
            assert_eq!(node.value(row), stage.value(row));
        }
    }

    #[test]
    fn row_schema_is_invariant_to_graph_shape() {
        // The schema depends only on the entity type, never on the visited node's
        // id — a chain (node_id == stage_id) and a branching walk (a distinct
        // node id) emit byte-identical columns.
        let chain = make_hydro_record(0, Some(0), 1); // node_id 0 (degenerate)
        let mut branching = make_hydro_record(0, Some(0), 1);
        branching.node_id = 42; // a branching node id unrelated to the stage
        let chain_batch = build_hydros_batch([&chain], 0, &[vec![720.0]], 1).unwrap();
        let branching_batch = build_hydros_batch([&branching], 0, &[vec![720.0]], 1).unwrap();
        assert_eq!(
            chain_batch.schema().fields(),
            branching_batch.schema().fields(),
            "row schema must not branch on graph shape"
        );
    }

    #[test]
    fn write_paths_is_three_int32_columns_sorted_canonically() {
        let tmp = tempfile::tempdir().expect("tempdir must succeed");

        // Deliberately out of (scenario_id, stage_id) order to pin the canonical sort.
        let rows = vec![
            SimulationPathRecord {
                scenario_id: 1,
                stage_id: 1,
                node_id: 5,
            },
            SimulationPathRecord {
                scenario_id: 0,
                stage_id: 1,
                node_id: 3,
            },
            SimulationPathRecord {
                scenario_id: 0,
                stage_id: 0,
                node_id: 2,
            },
        ];
        write_paths(tmp.path(), rows).expect("write_paths must succeed");

        let path = tmp.path().join("simulation/paths.parquet");
        assert!(
            path.exists(),
            "simulation/paths.parquet must exist (unpartitioned)"
        );
        let batch = read_first_batch(&path);

        let schema = batch.schema();
        let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
        assert_eq!(names, vec!["scenario_id", "stage_id", "node_id"]);
        assert_eq!(batch.num_columns(), 3);

        let col = |name: &str| {
            batch
                .column_by_name(name)
                .unwrap()
                .as_any()
                .downcast_ref::<arrow::array::Int32Array>()
                .unwrap()
                .clone()
        };
        let scenario = col("scenario_id");
        let stage = col("stage_id");
        let node = col("node_id");
        // Canonical (scenario_id, stage_id) order regardless of insertion order.
        assert_eq!(
            (0..3).map(|i| scenario.value(i)).collect::<Vec<_>>(),
            vec![0, 0, 1]
        );
        assert_eq!(
            (0..3).map(|i| stage.value(i)).collect::<Vec<_>>(),
            vec![0, 1, 1]
        );
        assert_eq!(
            (0..3).map(|i| node.value(i)).collect::<Vec<_>>(),
            vec![2, 3, 5]
        );
    }

    fn read_scenario_summary(
        dir: &Path,
    ) -> (
        arrow::array::Int32Array,
        arrow::array::Float64Array,
        arrow::array::Float64Array,
    ) {
        let path = dir.join("simulation/scenario_summary.parquet");
        assert!(
            path.exists(),
            "simulation/scenario_summary.parquet must exist (unpartitioned)"
        );
        let batch = read_first_batch(&path);

        let schema = batch.schema();
        let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
        assert_eq!(
            names,
            vec!["scenario_id", "probability", "discounted_immediate_cost"]
        );

        let scenario_id = batch
            .column_by_name("scenario_id")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow::array::Int32Array>()
            .unwrap()
            .clone();
        let probability = batch
            .column_by_name("probability")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .unwrap()
            .clone();
        let cost = batch
            .column_by_name("discounted_immediate_cost")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .unwrap()
            .clone();
        (scenario_id, probability, cost)
    }

    #[test]
    fn write_scenario_summary_sampled_has_all_null_probability() {
        use arrow::array::Array;
        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        let rows: Vec<(u32, Option<f64>, f64)> =
            vec![(0, None, 100.0), (1, None, 250.0), (2, None, 175.0)];
        write_scenario_summary(tmp.path(), &rows).expect("write_scenario_summary must succeed");

        let (scenario_id, probability, cost) = read_scenario_summary(tmp.path());
        assert_eq!(scenario_id.null_count(), 0, "scenario_id must be non-null");
        assert_eq!(
            probability.null_count(),
            3,
            "sampled runs leave every probability NULL"
        );
        assert_eq!(
            (0..3).map(|i| scenario_id.value(i)).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert_eq!(
            (0..3).map(|i| cost.value(i)).collect::<Vec<_>>(),
            vec![100.0, 250.0, 175.0]
        );
    }

    #[test]
    fn write_scenario_summary_census_populates_probability() {
        use arrow::array::Array;
        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        let rows: Vec<(u32, Option<f64>, f64)> = vec![(0, Some(0.25), 10.0), (1, Some(0.75), 30.0)];
        write_scenario_summary(tmp.path(), &rows).expect("write_scenario_summary must succeed");

        let (scenario_id, probability, cost) = read_scenario_summary(tmp.path());
        assert_eq!(
            probability.null_count(),
            0,
            "a declared census populates every probability"
        );
        assert_eq!(
            (0..2).map(|i| scenario_id.value(i)).collect::<Vec<_>>(),
            vec![0, 1]
        );
        assert_eq!(
            (0..2).map(|i| probability.value(i)).collect::<Vec<_>>(),
            vec![0.25, 0.75]
        );
        assert_eq!(
            (0..2).map(|i| cost.value(i)).collect::<Vec<_>>(),
            vec![10.0, 30.0]
        );
    }

    #[test]
    fn write_scenario_summary_writes_rows_verbatim_and_is_byte_deterministic() {
        // The gather (owned upstream) fixes canonical ascending scenario_id order;
        // the writer must preserve it verbatim and be a pure function of its rows,
        // so identical gathered rows produce byte-identical files across rank and
        // thread shapes.
        let rows: Vec<(u32, Option<f64>, f64)> = vec![(0, Some(0.5), 12.0), (1, Some(0.5), 8.0)];

        let a = tempfile::tempdir().expect("tempdir must succeed");
        let b = tempfile::tempdir().expect("tempdir must succeed");
        write_scenario_summary(a.path(), &rows).expect("write must succeed");
        write_scenario_summary(b.path(), &rows).expect("write must succeed");

        let (scenario_id, _prob, _cost) = read_scenario_summary(a.path());
        assert_eq!(
            (0..2).map(|i| scenario_id.value(i)).collect::<Vec<_>>(),
            vec![0, 1],
            "rows are written in the ascending order supplied, never re-sorted"
        );

        let bytes_a = std::fs::read(a.path().join("simulation/scenario_summary.parquet")).unwrap();
        let bytes_b = std::fs::read(b.path().join("simulation/scenario_summary.parquet")).unwrap();
        assert_eq!(
            bytes_a, bytes_b,
            "identical rows must serialize to byte-identical files"
        );
    }

    #[test]
    fn write_scenario_skips_empty_entity_types() {
        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        std::fs::create_dir_all(tmp.path().join("simulation")).unwrap();

        // System with no contracts, pumping stations, non-controllables, or generics.
        let system = make_test_system();

        let mut writer =
            SimulationParquetWriter::new(tmp.path(), &system).expect("new must succeed");

        let payload = make_scenario_payload(0, 2);
        writer
            .write_scenario(payload)
            .expect("write_scenario must succeed");

        // contracts/ directory must not exist (zero contracts in system).
        assert!(
            !tmp.path().join("simulation/contracts").exists(),
            "simulation/contracts/ must not exist when system has 0 contracts"
        );

        // pumping_stations/ directory must not exist.
        assert!(
            !tmp.path().join("simulation/pumping_stations").exists(),
            "simulation/pumping_stations/ must not exist when system has 0 pumping stations"
        );

        // non_controllables/ directory must not exist.
        assert!(
            !tmp.path().join("simulation/non_controllables").exists(),
            "simulation/non_controllables/ must not exist when system has 0 non-controllables"
        );
    }

    #[test]
    fn write_scenario_writes_pumping_partition_for_populated_system() {
        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        std::fs::create_dir_all(tmp.path().join("simulation")).unwrap();

        // Directory gate: n_pumping_stations() > 0 (one station in this system).
        let system = make_test_system_with_pumping();
        assert!(
            system.n_pumping_stations() > 0,
            "fixture must have a pumping station so the directory gate fires"
        );

        let mut writer =
            SimulationParquetWriter::new(tmp.path(), &system).expect("new must succeed");

        // One stage (stage 0, block 0, duration 720h) with two populated pumping
        // rows. block_id = Some(0) so the writer's block-duration lookup resolves
        // to 720.0, enabling exact derived-column assertions. Partition-write gate:
        // this non-empty pumping_stations vector is what triggers the write.
        let stage0 = StageWritePayload {
            stage_id: 0,
            node_id: 0,
            costs: vec![],
            hydros: vec![],
            hydro_bus_generation: vec![],
            thermals: vec![],
            exchanges: vec![],
            buses: vec![],
            pumping_stations: vec![
                make_pumping_record(0, Some(0), 1, 10.0, 5.0),
                make_pumping_record(0, Some(0), 2, 20.0, 8.0),
            ],
            contracts: vec![],
            non_controllables: vec![],
            inflow_lags: vec![],
            transit_buckets: vec![],
            generic_violations: vec![],
            anticipated_lanes: vec![],
        };
        let payload = ScenarioWritePayload {
            scenario_id: 0,
            stages: vec![stage0],
            transit_seed: vec![],
        };
        writer
            .write_scenario(payload)
            .expect("write_scenario must succeed");

        // File exists at the Hive partition path.
        let path = tmp
            .path()
            .join("simulation/pumping_stations/scenario_id=0000/data.parquet");
        assert!(
            path.exists(),
            "simulation/pumping_stations/scenario_id=0000/data.parquet must exist"
        );

        // Read the written Parquet back with the crate's existing reader helper.
        let batch = read_first_batch(&path);

        // Written schema is field-for-field equal to pumping_stations_schema().
        let expected = pumping_stations_schema();
        assert_eq!(
            batch.schema().fields(),
            expected.fields(),
            "written schema must match pumping_stations_schema()"
        );
        assert_eq!(batch.num_columns(), 11, "pumping schema has 11 columns");

        // Row count == number of (stage, block, station) tuples (2).
        assert_eq!(
            batch.num_rows(),
            2,
            "row count must equal the number of populated (stage, block, station) tuples"
        );

        let f64_col = |name: &str, row: usize| {
            batch
                .column_by_name(name)
                .unwrap_or_else(|| panic!("{name} column must exist"))
                .as_any()
                .downcast_ref::<arrow::array::Float64Array>()
                .unwrap_or_else(|| panic!("{name} must be Float64Array"))
                .value(row)
        };

        // pumped_flow_m3s / power_consumption_mw forwarded verbatim.
        assert_eq!(
            f64_col("pumped_flow_m3s", 0),
            10.0,
            "pumped_flow_m3s row 0 must be forwarded verbatim"
        );
        assert_eq!(
            f64_col("pumped_flow_m3s", 1),
            20.0,
            "pumped_flow_m3s row 1 must be forwarded verbatim"
        );
        assert_eq!(
            f64_col("power_consumption_mw", 0),
            5.0,
            "power_consumption_mw row 0 must be forwarded verbatim"
        );
        assert_eq!(
            f64_col("power_consumption_mw", 1),
            8.0,
            "power_consumption_mw row 1 must be forwarded verbatim"
        );

        // Derived columns for row 0 with block_duration = 720.0:
        // pumped_volume_hm3 = pumped_flow_m3s * dur * 3600.0 / 1e6
        assert_eq!(
            f64_col("pumped_volume_hm3", 0),
            10.0 * 720.0 * 3600.0 / 1_000_000.0,
            "pumped_volume_hm3 row 0 must equal pumped_flow_m3s * dur * 3600 / 1e6"
        );
        // energy_consumption_mwh = power_consumption_mw * dur
        assert_eq!(
            f64_col("energy_consumption_mwh", 0),
            5.0 * 720.0,
            "energy_consumption_mwh row 0 must equal power_consumption_mw * dur"
        );
    }

    #[test]
    fn finalize_returns_correct_counts() {
        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        std::fs::create_dir_all(tmp.path().join("simulation")).unwrap();

        let system = make_test_system();

        let mut writer =
            SimulationParquetWriter::new(tmp.path(), &system).expect("new must succeed");

        writer
            .write_scenario(make_scenario_payload(0, 1))
            .expect("write scenario 0 must succeed");
        writer
            .write_scenario(make_scenario_payload(1, 1))
            .expect("write scenario 1 must succeed");

        let output = writer.finalize(0);
        assert_eq!(output.n_scenarios, 2, "n_scenarios must be 2");
        assert_eq!(output.completed, 2, "completed must be 2");
        assert_eq!(output.failed, 0, "failed must be 0");
    }

    #[test]
    fn write_scenario_parquet_roundtrip_costs_row_count() {
        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        std::fs::create_dir_all(tmp.path().join("simulation")).unwrap();

        let system = make_test_system();

        let mut writer =
            SimulationParquetWriter::new(tmp.path(), &system).expect("new must succeed");

        // 2 stages, 1 cost record per stage → 2 rows
        let payload = make_scenario_payload(0, 2);
        writer
            .write_scenario(payload)
            .expect("write_scenario must succeed");

        let path = tmp
            .path()
            .join("simulation/costs/scenario_id=0000/data.parquet");
        let batch = read_first_batch(&path);
        assert_eq!(batch.num_rows(), 2, "costs parquet must have 2 rows");
        assert_eq!(batch.num_columns(), 29, "costs schema has 29 columns");
    }

    #[test]
    fn write_scenario_parquet_roundtrip_hydros_derived_mwh() {
        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        std::fs::create_dir_all(tmp.path().join("simulation")).unwrap();

        let system = make_test_system();

        let mut writer =
            SimulationParquetWriter::new(tmp.path(), &system).expect("new must succeed");

        // 2 stages x 2 hydros = 4 rows in hydros parquet.
        let payload = make_scenario_payload(0, 2);
        writer
            .write_scenario(payload)
            .expect("write_scenario must succeed");

        let path = tmp
            .path()
            .join("simulation/hydros/scenario_id=0000/data.parquet");
        let batch = read_first_batch(&path);
        assert_eq!(
            batch.num_rows(),
            4,
            "hydros parquet must have 4 rows (2 stages * 2 hydros)"
        );

        let gen_mwh_col = batch
            .column_by_name("generation_mwh")
            .expect("generation_mwh column must exist");
        let gen_mwh_arr = gen_mwh_col
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .expect("generation_mwh must be Float64Array");

        // Rows 0,1: stage 0, block 0, duration 720h → 50.0 * 720.0 = 36_000.0
        assert_eq!(
            gen_mwh_arr.value(0),
            50.0 * 720.0,
            "generation_mwh at row 0 (stage 0) must equal generation_mw * 720"
        );
        assert_eq!(
            gen_mwh_arr.value(1),
            50.0 * 720.0,
            "generation_mwh at row 1 (stage 0) must equal generation_mw * 720"
        );
        // Rows 2,3: stage 1, block 0, duration 744h → 50.0 * 744.0 = 37_200.0
        assert_eq!(
            gen_mwh_arr.value(2),
            50.0 * 744.0,
            "generation_mwh at row 2 (stage 1) must equal generation_mw * 744"
        );
        assert_eq!(
            gen_mwh_arr.value(3),
            50.0 * 744.0,
            "generation_mwh at row 3 (stage 1) must equal generation_mw * 744"
        );

        // The four tail-appended integrated-productivity and stored-energy-power
        // columns round-trip as non-nullable Float64, at their declared tail
        // positions — CLI and Python write through this one shared writer
        // (python_parity_script_passes), so a reordered column would mislabel
        // both front ends' output identically.
        let schema = batch.schema();
        for (col, expected_index) in [
            ("integrated_equivalent_productivity_mw_per_m3s", 37),
            ("integrated_accumulated_productivity_mw_per_m3s", 38),
            ("stored_energy_initial_mw", 39),
            ("stored_energy_final_mw", 40),
        ] {
            let field = schema
                .field_with_name(col)
                .unwrap_or_else(|_| panic!("{col} column must exist in schema"));
            assert_eq!(
                field.data_type(),
                &arrow::datatypes::DataType::Float64,
                "{col} must be Float64"
            );
            assert!(!field.is_nullable(), "{col} must be non-nullable");
            assert_eq!(
                schema.index_of(col).expect("column must exist"),
                expected_index,
                "{col} must sit at its declared tail position"
            );
        }
        let read_f64 = |col: &str| -> f64 {
            batch
                .column_by_name(col)
                .unwrap_or_else(|| panic!("column {col} must exist"))
                .as_any()
                .downcast_ref::<arrow::array::Float64Array>()
                .unwrap_or_else(|| panic!("column {col} must be Float64Array"))
                .value(0)
        };
        assert_eq!(
            read_f64("integrated_equivalent_productivity_mw_per_m3s"),
            1.5,
            "integrated_equivalent_productivity_mw_per_m3s must round-trip"
        );
        assert_eq!(
            read_f64("integrated_accumulated_productivity_mw_per_m3s"),
            6.0,
            "integrated_accumulated_productivity_mw_per_m3s must round-trip"
        );
        assert_eq!(
            read_f64("stored_energy_initial_mw"),
            1.7145,
            "stored_energy_initial_mw must round-trip"
        );
        assert_eq!(
            read_f64("stored_energy_final_mw"),
            1.7222,
            "stored_energy_final_mw must round-trip"
        );
    }

    #[test]
    fn write_scenario_atomic_no_tmp_file_remaining() {
        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        std::fs::create_dir_all(tmp.path().join("simulation")).unwrap();

        let system = make_test_system();

        let mut writer =
            SimulationParquetWriter::new(tmp.path(), &system).expect("new must succeed");

        let payload = make_scenario_payload(0, 1);
        writer
            .write_scenario(payload)
            .expect("write_scenario must succeed");

        let tmp_file = tmp
            .path()
            .join("simulation/costs/scenario_id=0000/data.parquet.tmp");
        assert!(
            !tmp_file.exists(),
            ".tmp file must not remain after successful atomic write"
        );
    }

    /// Verifies that the iterator-based `write_scenario` path runs correctly with
    /// 3 stages and 2 entity types (costs + hydros) — no flat Vec materialisation.
    #[test]
    fn write_scenario_does_not_materialize_flat_vecs() {
        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        std::fs::create_dir_all(tmp.path().join("simulation")).unwrap();

        let system = make_test_system();

        let mut writer =
            SimulationParquetWriter::new(tmp.path(), &system).expect("new must succeed");

        // 3 stages × 1 cost + 2 hydros each.
        let payload = make_scenario_payload(0, 3);
        writer
            .write_scenario(payload)
            .expect("write_scenario must succeed without panicking");

        // 3 stages × 1 cost record each → 3 rows in costs parquet.
        let costs_path = tmp
            .path()
            .join("simulation/costs/scenario_id=0000/data.parquet");
        assert!(costs_path.exists(), "costs parquet must be written");

        // 3 stages × 2 hydro records each → 6 rows in hydros parquet.
        let hydros_path = tmp
            .path()
            .join("simulation/hydros/scenario_id=0000/data.parquet");
        assert!(hydros_path.exists(), "hydros parquet must be written");

        let costs_batch = read_first_batch(&costs_path);
        assert_eq!(
            costs_batch.num_rows(),
            3,
            "costs must have 3 rows (3 stages)"
        );

        let hydros_batch = read_first_batch(&hydros_path);
        assert_eq!(
            hydros_batch.num_rows(),
            6,
            "hydros must have 6 rows (3 stages × 2 hydros)"
        );
    }

    #[test]
    fn hydros_schema_has_forty_one_fields() {
        let schema = hydros_schema();
        assert_eq!(
            schema.fields().len(),
            41,
            "hydros_schema must have 41 fields (39 + scenario_id + node_id)"
        );
    }

    #[test]
    fn hydros_schema_drops_old_productivity_field() {
        let schema = hydros_schema();
        assert!(
            schema.field_with_name("productivity_mw_per_m3s").is_err(),
            "productivity_mw_per_m3s must not exist in the updated schema"
        );
    }

    #[test]
    fn hydros_schema_has_equivalent_productivity_column() {
        use arrow::datatypes::DataType;
        let schema = hydros_schema();
        let field = schema
            .field_with_name("equivalent_productivity_mw_per_m3s")
            .expect("equivalent_productivity_mw_per_m3s must exist in schema");
        assert_eq!(
            field.data_type(),
            &DataType::Float64,
            "equivalent_productivity_mw_per_m3s must be Float64"
        );
        assert!(
            !field.is_nullable(),
            "equivalent_productivity_mw_per_m3s must be non-nullable"
        );
    }

    #[test]
    fn hydros_batch_round_trips_new_columns() {
        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        std::fs::create_dir_all(tmp.path().join("simulation")).unwrap();

        let system = make_test_system();

        let mut writer =
            SimulationParquetWriter::new(tmp.path(), &system).expect("new must succeed");

        // Single payload using make_hydro_record which sets distinct values:
        // equivalent_productivity_mw_per_m3s = 0.9
        // accumulated_productivity_mw_per_m3s = 2.7
        // incremental_inflow_energy_mw        = 135.0
        // stored_energy_initial_mwh           = 1234.5
        // stored_energy_final_mwh             = 1240.0
        let payload = make_scenario_payload(0, 1);
        writer
            .write_scenario(payload)
            .expect("write_scenario must succeed");

        let path = tmp
            .path()
            .join("simulation/hydros/scenario_id=0000/data.parquet");
        let batch = read_first_batch(&path);

        let read_f64 = |col_name: &str| -> f64 {
            batch
                .column_by_name(col_name)
                .unwrap_or_else(|| panic!("column {col_name} must exist"))
                .as_any()
                .downcast_ref::<arrow::array::Float64Array>()
                .unwrap_or_else(|| panic!("column {col_name} must be Float64Array"))
                .value(0)
        };

        assert_eq!(
            read_f64("equivalent_productivity_mw_per_m3s"),
            0.9,
            "equivalent_productivity_mw_per_m3s must round-trip"
        );
        assert_eq!(
            read_f64("accumulated_productivity_mw_per_m3s"),
            2.7,
            "accumulated_productivity_mw_per_m3s must round-trip"
        );
        assert_eq!(
            read_f64("incremental_inflow_energy_mw"),
            135.0,
            "incremental_inflow_energy_mw must round-trip"
        );
        assert_eq!(
            read_f64("stored_energy_initial_mwh"),
            1234.5,
            "stored_energy_initial_mwh must round-trip"
        );
        assert_eq!(
            read_f64("stored_energy_final_mwh"),
            1240.0,
            "stored_energy_final_mwh must round-trip"
        );

        // The moved pair sits at its mid-schema position — CLI and Python write
        // through this one shared writer (python_parity_script_passes), so a
        // reordered column would mislabel both front ends' output identically.
        let schema = batch.schema();
        assert_eq!(
            schema
                .index_of("stored_energy_initial_mwh")
                .expect("column must exist"),
            20,
            "stored_energy_initial_mwh must sit at its declared mid-schema position"
        );
        assert_eq!(
            schema
                .index_of("stored_energy_final_mwh")
                .expect("column must exist"),
            21,
            "stored_energy_final_mwh must sit at its declared mid-schema position"
        );
    }

    /// [`make_test_system`] plus a declared travel-time arc (hydro 2 → hydro 1,
    /// 720 h), so the writer's `in_transit` directory gate fires.
    fn make_test_system_with_travel_time() -> System {
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
        let downstream = make_hydro(1);
        let mut upstream = make_hydro(2);
        upstream.downstream_id = Some(EntityId(1));
        upstream.travel_time_hours = Some(720.0);

        SystemBuilder::new()
            .buses(vec![bus])
            .hydros(vec![downstream, upstream])
            .stages(vec![make_stage(0, 720.0), make_stage(1, 744.0)])
            .build()
            .expect("travel-time test system must be valid")
    }

    /// Absent when undeclared: a non-travel-time system creates no `in_transit`
    /// directory, and a scenario with no transit-bucket rows writes no file —
    /// byte-neutral for existing studies.
    #[test]
    fn no_in_transit_partition_for_non_travel_time_system() {
        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        std::fs::create_dir_all(tmp.path().join("simulation")).unwrap();

        let system = make_test_system();
        let mut writer =
            SimulationParquetWriter::new(tmp.path(), &system).expect("new must succeed");

        assert!(
            !tmp.path().join("simulation/in_transit").exists(),
            "in_transit/ must not exist when no travel-time arc is declared"
        );

        let stage0 = StageWritePayload {
            stage_id: 0,
            node_id: 0,
            costs: vec![],
            hydros: vec![],
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
        };
        writer
            .write_scenario(ScenarioWritePayload {
                scenario_id: 0,
                stages: vec![stage0],
                transit_seed: vec![],
            })
            .expect("write_scenario must succeed");

        assert!(
            !tmp.path().join("simulation/in_transit").exists(),
            "in_transit/ must stay absent after writing a bucket-free scenario"
        );
    }

    #[test]
    fn write_scenario_writes_in_transit_partition_round_trip() {
        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        std::fs::create_dir_all(tmp.path().join("simulation")).unwrap();

        let system = make_test_system_with_travel_time();
        let mut writer =
            SimulationParquetWriter::new(tmp.path(), &system).expect("new must succeed");

        assert!(
            tmp.path().join("simulation/in_transit").exists(),
            "in_transit/ directory must be created for a travel-time system"
        );

        // Plant hydro_id 1 with two maturity buckets; delayed arrival lands only
        // on lag 1.
        let stage0 = StageWritePayload {
            stage_id: 0,
            node_id: 0,
            costs: vec![],
            hydros: vec![],
            hydro_bus_generation: vec![],
            thermals: vec![],
            exchanges: vec![],
            buses: vec![],
            pumping_stations: vec![],
            contracts: vec![],
            non_controllables: vec![],
            inflow_lags: vec![],
            transit_buckets: vec![
                TransitBucketWriteRecord {
                    stage_id: 0,
                    node_id: 0,
                    hydro_id: 1,
                    lag: 1,
                    in_transit_volume_hm3: 11.0,
                    delayed_arrival_hm3: 7.0,
                },
                TransitBucketWriteRecord {
                    stage_id: 0,
                    node_id: 0,
                    hydro_id: 1,
                    lag: 2,
                    in_transit_volume_hm3: 22.0,
                    delayed_arrival_hm3: 0.0,
                },
            ],
            generic_violations: vec![],
            anticipated_lanes: vec![],
        };
        writer
            .write_scenario(ScenarioWritePayload {
                scenario_id: 0,
                stages: vec![stage0],
                transit_seed: vec![],
            })
            .expect("write_scenario must succeed");

        let path = tmp
            .path()
            .join("simulation/in_transit/scenario_id=0000/data.parquet");
        assert!(path.exists(), "in_transit parquet must exist");

        let batch = read_first_batch(&path);

        assert_eq!(
            batch.schema().fields(),
            in_transit_schema().fields(),
            "written schema must match in_transit_schema()"
        );
        assert_eq!(batch.num_rows(), 2, "one row per declared bucket");

        let i32_col = |name: &str, row: usize| {
            batch
                .column_by_name(name)
                .unwrap_or_else(|| panic!("{name} column must exist"))
                .as_any()
                .downcast_ref::<arrow::array::Int32Array>()
                .unwrap_or_else(|| panic!("{name} must be Int32Array"))
                .value(row)
        };
        let f64_col = |name: &str, row: usize| {
            batch
                .column_by_name(name)
                .unwrap_or_else(|| panic!("{name} column must exist"))
                .as_any()
                .downcast_ref::<arrow::array::Float64Array>()
                .unwrap_or_else(|| panic!("{name} must be Float64Array"))
                .value(row)
        };

        assert_eq!((i32_col("hydro_id", 0), i32_col("lag", 0)), (1, 1));
        assert_eq!(f64_col("in_transit_volume_hm3", 0), 11.0);
        assert_eq!(f64_col("delayed_arrival_hm3", 0), 7.0);
        assert_eq!((i32_col("hydro_id", 1), i32_col("lag", 1)), (1, 2));
        assert_eq!(f64_col("in_transit_volume_hm3", 1), 22.0);
        assert_eq!(f64_col("delayed_arrival_hm3", 1), 0.0);
    }

    #[test]
    fn no_hydro_bus_generation_partition_when_records_are_empty() {
        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        std::fs::create_dir_all(tmp.path().join("simulation")).unwrap();

        let system = make_test_system();
        let mut writer =
            SimulationParquetWriter::new(tmp.path(), &system).expect("new must succeed");

        writer
            .write_scenario(make_scenario_payload(0, 2))
            .expect("write_scenario must succeed");

        assert!(
            !tmp.path()
                .join("simulation/hydro_bus_generation/scenario_id=0000")
                .exists(),
            "no empty hydro_bus_generation per-scenario partition must ship when \
             a scenario's stages carry no cell records"
        );
    }

    #[test]
    fn hydro_bus_generation_directory_created_for_a_hydro_system() {
        use arrow::array::Array;

        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        std::fs::create_dir_all(tmp.path().join("simulation")).unwrap();

        let system = make_test_system();
        assert!(
            system.n_hydros() > 0,
            "fixture must have a hydro so the directory gate fires"
        );
        let mut writer =
            SimulationParquetWriter::new(tmp.path(), &system).expect("new must succeed");

        assert!(
            tmp.path().join("simulation/hydro_bus_generation").is_dir(),
            "simulation/hydro_bus_generation must exist as a directory for a hydro system"
        );

        // Single-bus end-to-end run: one cell per plant (the identity
        // partition), so hydro_bus_generation's row count matches hydros'.
        let stage0 = StageWritePayload {
            stage_id: 0,
            node_id: 0,
            costs: vec![],
            hydros: vec![
                make_hydro_record(0, Some(0), 1),
                make_hydro_record(0, Some(0), 2),
            ],
            hydro_bus_generation: vec![
                HydroBusWriteRecord {
                    stage_id: 0,
                    node_id: 0,
                    block_id: Some(0),
                    hydro_id: 1,
                    bus_id: 1,
                    turbined_m3s: 80.0,
                    generation_mw: 50.0,
                },
                HydroBusWriteRecord {
                    stage_id: 0,
                    node_id: 0,
                    block_id: Some(0),
                    hydro_id: 2,
                    bus_id: 1,
                    turbined_m3s: 80.0,
                    generation_mw: 50.0,
                },
            ],
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
        };
        writer
            .write_scenario(ScenarioWritePayload {
                scenario_id: 0,
                stages: vec![stage0],
                transit_seed: vec![],
            })
            .expect("write_scenario must succeed");

        let hydros_batch = read_first_batch(
            &tmp.path()
                .join("simulation/hydros/scenario_id=0000/data.parquet"),
        );
        let bus_path = tmp
            .path()
            .join("simulation/hydro_bus_generation/scenario_id=0000/data.parquet");
        assert!(bus_path.exists(), "hydro_bus_generation parquet must exist");
        let bus_batch = read_first_batch(&bus_path);

        assert_eq!(
            bus_batch.num_rows(),
            hydros_batch.num_rows(),
            "hydro_bus_generation row count must equal the hydros row count \
             under the identity partition"
        );

        let bus_id_col = bus_batch
            .column_by_name("bus_id")
            .expect("bus_id column must exist");
        assert_eq!(
            bus_id_col.null_count(),
            0,
            "every hydro_bus_generation row's bus_id must be non-null"
        );
    }

    #[test]
    fn write_scenario_writes_hydro_bus_generation_partition_round_trip() {
        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        std::fs::create_dir_all(tmp.path().join("simulation")).unwrap();

        // Stage 0 (block 0, duration 720h): same hydro_id (7) on two distinct
        // bus_ids (11, 22). Stage 1 (block 0, duration 744h): a row whose
        // stage_id/block_id/hydro_id/bus_id are mutually distinct (1, 0, 5, 9).
        let system = make_test_system();
        let mut writer =
            SimulationParquetWriter::new(tmp.path(), &system).expect("new must succeed");

        let stage0 = StageWritePayload {
            stage_id: 0,
            node_id: 0,
            costs: vec![],
            hydros: vec![],
            hydro_bus_generation: vec![
                HydroBusWriteRecord {
                    stage_id: 0,
                    node_id: 0,
                    block_id: Some(0),
                    hydro_id: 7,
                    bus_id: 11,
                    turbined_m3s: 30.0,
                    generation_mw: 12.0,
                },
                HydroBusWriteRecord {
                    stage_id: 0,
                    node_id: 0,
                    block_id: Some(0),
                    hydro_id: 7,
                    bus_id: 22,
                    turbined_m3s: 45.0,
                    generation_mw: 18.0,
                },
            ],
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
        };
        let stage1 = StageWritePayload {
            stage_id: 1,
            node_id: 1,
            costs: vec![],
            hydros: vec![],
            hydro_bus_generation: vec![HydroBusWriteRecord {
                stage_id: 1,
                node_id: 1,
                block_id: Some(0),
                hydro_id: 5,
                bus_id: 9,
                turbined_m3s: 13.0,
                generation_mw: 1.0,
            }],
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
        };
        writer
            .write_scenario(ScenarioWritePayload {
                scenario_id: 0,
                stages: vec![stage0, stage1],
                transit_seed: vec![],
            })
            .expect("write_scenario must succeed");

        let path = tmp
            .path()
            .join("simulation/hydro_bus_generation/scenario_id=0000/data.parquet");
        assert!(path.exists(), "hydro_bus_generation parquet must exist");

        let batch = read_first_batch(&path);

        assert_eq!(
            batch.schema().fields(),
            hydro_bus_generation_schema().fields(),
            "written schema must match hydro_bus_generation_schema()"
        );
        assert_eq!(batch.num_rows(), 3, "one row per declared cell");

        let i32_col = |name: &str, row: usize| {
            batch
                .column_by_name(name)
                .unwrap_or_else(|| panic!("{name} column must exist"))
                .as_any()
                .downcast_ref::<arrow::array::Int32Array>()
                .unwrap_or_else(|| panic!("{name} must be Int32Array"))
                .value(row)
        };
        let f64_col = |name: &str, row: usize| {
            batch
                .column_by_name(name)
                .unwrap_or_else(|| panic!("{name} column must exist"))
                .as_any()
                .downcast_ref::<arrow::array::Float64Array>()
                .unwrap_or_else(|| panic!("{name} must be Float64Array"))
                .value(row)
        };

        assert_eq!(
            (i32_col("bus_id", 0), i32_col("bus_id", 1)),
            (11, 22),
            "the two stage-0 rows must carry their two distinct bus_id values"
        );
        assert_eq!(
            f64_col("generation_mwh", 0),
            12.0 * 720.0,
            "row 0 generation_mwh must equal generation_mw * this row's stage duration"
        );
        assert_eq!(
            f64_col("generation_mwh", 1),
            18.0 * 720.0,
            "row 1 generation_mwh must equal generation_mw * this row's stage duration"
        );
        assert_eq!(
            f64_col("generation_mwh", 2),
            1.0 * 744.0,
            "row 2 generation_mwh must equal generation_mw * its OWN (stage 1) duration"
        );
    }

    /// `SIMULATION_FAMILIES` must list every entity family exactly once: the
    /// 13 `Vec` fields of `StageWritePayload` plus `ScenarioWritePayload`'s own
    /// scenario-level `transit_seed`. This count is a manual pin, not a
    /// reflection-derived count (Rust has no runtime enumeration of a struct's
    /// fields) — adding or removing a family must add or remove its row and
    /// update this count in the same change.
    #[test]
    fn simulation_family_table_has_no_duplicate_or_missing_rows() {
        const EXPECTED_FAMILY_COUNT: usize = 14;
        assert_eq!(
            SIMULATION_FAMILIES.len(),
            EXPECTED_FAMILY_COUNT,
            "SIMULATION_FAMILIES must list every entity family exactly once"
        );

        let mut seen_subpaths: Vec<&str> = Vec::with_capacity(SIMULATION_FAMILIES.len());
        for family in SIMULATION_FAMILIES {
            assert!(
                !seen_subpaths.contains(&family.subpath),
                "subpath '{}' appears more than once in SIMULATION_FAMILIES",
                family.subpath
            );
            seen_subpaths.push(family.subpath);
        }
    }

    #[test]
    fn simulation_family_subpaths_lists_every_family() {
        let subpaths: Vec<&str> = simulation_family_subpaths().collect();
        assert_eq!(
            subpaths.len(),
            SIMULATION_FAMILIES.len(),
            "the public subpath accessor must yield one entry per family"
        );
        for expected in [
            "hydro_bus_generation",
            "in_transit",
            "transit_seed",
            "anticipated_lanes",
        ] {
            assert!(
                subpaths.contains(&expected),
                "simulation_family_subpaths() must include '{expected}'"
            );
        }
    }

    fn seed_file(root: &Path, relative: &str) {
        let path = root.join(relative);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, b"stale").unwrap();
    }

    #[test]
    fn simulation_output_remover_clears_partitions_and_run_level_files() {
        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        let sim = tmp.path().join("simulation");
        let removed = [
            "costs/scenario_id=0000/data.parquet",
            "pumping_stations/scenario_id=0003/data.parquet",
            "violations/generic/scenario_id=0001/data.parquet",
            "solver/iterations.parquet",
            "solver/retry_histogram.parquet",
            "paths.parquet",
            "scenario_summary.parquet",
            "metadata.json",
        ];
        let kept = [
            "_SUCCESS",
            "metadata.json.tmp",
            "scenario_summary.parquet.tmp",
            "solver/stale.txt",
            "notes.txt",
        ];
        for relative in removed.iter().chain(&kept) {
            seed_file(&sim, relative);
        }

        remove_simulation_outputs(tmp.path()).expect("every seeded output is removable");

        for relative in removed
            .iter()
            .chain(&["costs", "pumping_stations", "violations"])
        {
            assert!(
                !sim.join(relative).exists(),
                "simulation/{relative} must be removed"
            );
        }
        for relative in kept {
            assert!(
                sim.join(relative).is_file(),
                "simulation/{relative} must be kept"
            );
        }
    }

    #[test]
    fn simulation_output_remover_clears_what_the_writers_wrote() {
        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        write_paths(tmp.path(), Vec::new()).unwrap();
        write_scenario_summary(tmp.path(), &[]).unwrap();
        crate::output::write_simulation_solver_stats(tmp.path(), &[]).unwrap();

        remove_simulation_outputs(tmp.path()).expect("the writers' outputs are removable");

        let left: Vec<_> = std::fs::read_dir(tmp.path().join("simulation"))
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect();
        assert!(
            left.is_empty(),
            "simulation/ must be left empty, found {left:?}"
        );
    }

    #[test]
    fn simulation_output_remover_accepts_a_missing_simulation_dir() {
        let tmp = tempfile::tempdir().expect("tempdir must succeed");

        remove_simulation_outputs(tmp.path()).expect("a missing simulation/ has nothing to remove");
    }

    #[test]
    fn simulation_output_remover_reports_an_entry_it_cannot_remove() {
        for (blocker, blocker_is_dir, reported) in [
            ("costs", false, "costs"),
            ("paths.parquet", true, "paths.parquet"),
            ("solver", false, "solver/iterations.parquet"),
        ] {
            let tmp = tempfile::tempdir().expect("tempdir must succeed");
            let sim = tmp.path().join("simulation");
            if blocker_is_dir {
                std::fs::create_dir_all(sim.join(blocker)).unwrap();
            } else {
                seed_file(&sim, blocker);
            }

            match remove_simulation_outputs(tmp.path()) {
                Err(OutputError::IoError { path, .. }) => assert!(
                    path.ends_with(reported),
                    "a {blocker} blocker must be reported at {reported}, got {}",
                    path.display()
                ),
                other => panic!("a {blocker} blocker must give an IoError, got {other:?}"),
            }
            assert!(
                sim.join(blocker).exists(),
                "the {blocker} blocker must be left in place"
            );
        }
    }
}
