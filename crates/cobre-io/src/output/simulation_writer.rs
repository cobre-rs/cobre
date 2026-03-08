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
//!   thermals/scenario_id=0000/data.parquet
//!   exchanges/scenario_id=0000/data.parquet
//!   buses/scenario_id=0000/data.parquet
//!   pumping_stations/scenario_id=0000/data.parquet
//!   contracts/scenario_id=0000/data.parquet
//!   non_controllables/scenario_id=0000/data.parquet
//!   inflow_lags/scenario_id=0000/data.parquet
//!   violations/generic/scenario_id=0000/data.parquet
//! ```
//!
//! The writer runs on a dedicated I/O thread. It receives
//! [`ScenarioWritePayload`] values, converts the nested per-entity-type
//! [`Vec`]s into columnar Arrow [`RecordBatch`] format, computing derived
//! columns (`MWh` energy, net flow, losses) from system metadata stored at
//! construction time.
//!
//! ## Circular-dependency mitigation
//!
//! The solver crate depends on `cobre-io` (not the other way around). Rather than
//! creating a circular dependency, this module defines a crate-local
//! [`ScenarioWritePayload`] that mirrors the simulation result data layout exactly.
//! Conversion from solver-specific types to this payload type is handled by the
//! solver's output integration layer.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{BooleanBuilder, Float64Builder, Int32Builder, Int8Builder, RecordBatch};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;

use cobre_core::System;

use crate::output::error::OutputError;
use crate::output::parquet_config::ParquetWriterConfig;
use crate::output::schemas::{
    buses_schema, contracts_schema, costs_schema, exchanges_schema, generic_violations_schema,
    hydros_schema, inflow_lags_schema, non_controllables_schema, pumping_stations_schema,
    thermals_schema,
};
use crate::output::SimulationOutput;

// Payload types (mirrors solver simulation result types)

/// Cost breakdown for one (stage, block) pair.
///
/// Conversion to this type from algorithm-specific cost results is handled by
/// the calling solver.
#[derive(Debug)]
pub struct CostWriteRecord {
    /// Stage index (0-based).
    pub stage_id: u32,
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
    /// Inflow non-negativity violation cost.
    pub inflow_penalty_cost: f64,
    /// Generic constraint violation cost.
    pub generic_violation_cost: f64,
    /// Spillage regularization cost.
    pub spillage_cost: f64,
    /// FPHA turbining regularization cost.
    pub fpha_turbined_cost: f64,
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
    /// Total inflow to the reservoir in m³/s.
    pub inflow_m3s: f64,
    /// Reservoir storage at block start in hm³.
    pub storage_initial_hm3: f64,
    /// Reservoir storage at block end in hm³.
    pub storage_final_hm3: f64,
    /// Active power generation in MW.
    pub generation_mw: f64,
    /// Plant productivity in MW/(m³/s), or `None` for tabular production.
    pub productivity_mw_per_m3s: Option<f64>,
    /// Spillage regularization cost.
    pub spillage_cost: f64,
    /// Water value (storage balance dual) in cost/hm³.
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
    /// Evaporation constraint violation in m³/s.
    pub evaporation_violation_m3s: f64,
    /// Inflow non-negativity slack in m³/s.
    pub inflow_nonnegativity_slack_m3s: f64,
}

/// Thermal unit result for one (stage, block, thermal) tuple.
#[derive(Debug)]
pub struct ThermalWriteRecord {
    /// Stage index (0-based).
    pub stage_id: u32,
    /// Block index, or `None` for stage-level rows.
    pub block_id: Option<u32>,
    /// Thermal unit entity ID.
    pub thermal_id: i32,
    /// Active power generation in MW.
    pub generation_mw: f64,
    /// Variable generation cost.
    pub generation_cost: f64,
    /// Whether this unit uses the GNL model.
    pub is_gnl: bool,
    /// Committed capacity under GNL in MW, or `None`.
    pub gnl_committed_mw: Option<f64>,
    /// Decision capacity under GNL in MW, or `None`.
    pub gnl_decision_mw: Option<f64>,
    /// Operative state code.
    pub operative_state_code: i8,
}

/// Exchange (transmission line) result for one (stage, block, line) tuple.
///
#[derive(Debug)]
pub struct ExchangeWriteRecord {
    /// Stage index (0-based).
    pub stage_id: u32,
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
///
#[derive(Debug)]
pub struct BusWriteRecord {
    /// Stage index (0-based).
    pub stage_id: u32,
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
///
#[derive(Debug)]
pub struct PumpingWriteRecord {
    /// Stage index (0-based).
    pub stage_id: u32,
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
///
#[derive(Debug)]
pub struct ContractWriteRecord {
    /// Stage index (0-based).
    pub stage_id: u32,
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
///
#[derive(Debug)]
pub struct NonControllableWriteRecord {
    /// Stage index (0-based).
    pub stage_id: u32,
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
///
#[derive(Debug)]
pub struct InflowLagWriteRecord {
    /// Stage index (0-based).
    pub stage_id: u32,
    /// Hydro plant entity ID.
    pub hydro_id: i32,
    /// Lag index within the AR model (0 = most recent past period).
    pub lag_index: u32,
    /// Observed inflow at this lag in m³/s.
    pub inflow_m3s: f64,
}

/// Generic constraint violation for one (stage, block, constraint) tuple.
///
#[derive(Debug)]
pub struct GenericViolationWriteRecord {
    /// Stage index (0-based).
    pub stage_id: u32,
    /// Block index, or `None` for stage-level rows.
    pub block_id: Option<u32>,
    /// Generic constraint entity ID.
    pub constraint_id: i32,
    /// Violation slack value.
    pub slack_value: f64,
    /// Cost for this violation.
    pub slack_cost: f64,
}

/// All simulation results for one stage within one scenario.
///
#[derive(Debug)]
pub struct StageWritePayload {
    /// Stage index (0-based).
    pub stage_id: u32,
    /// Cost breakdown records for this stage.
    pub costs: Vec<CostWriteRecord>,
    /// Hydro plant records for this stage.
    pub hydros: Vec<HydroWriteRecord>,
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
    /// Generic constraint violation records for this stage.
    pub generic_violations: Vec<GenericViolationWriteRecord>,
}

/// Complete simulation result for one scenario, ready for Parquet writing.
///
/// This is the local counterpart of solver-specific simulation result types.
/// Conversion to this payload is handled by the solver's output integration layer.
#[derive(Debug)]
pub struct ScenarioWritePayload {
    /// 0-based scenario identifier. Determines the Hive partition:
    /// `{entity}/scenario_id={scenario_id:04d}/data.parquet`.
    pub scenario_id: u32,

    /// Per-stage detailed results.
    pub stages: Vec<StageWritePayload>,
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
/// use cobre_io::ParquetWriterConfig;
/// use std::path::Path;
///
/// # fn main() -> Result<(), cobre_io::OutputError> {
/// # let system = unimplemented!();
/// let config = ParquetWriterConfig::default();
/// let writer = SimulationParquetWriter::new(Path::new("/tmp/out"), system, &config)?;
/// # Ok(())
/// # }
/// ```
pub struct SimulationParquetWriter {
    /// Root output directory (contains the `simulation/` sub-tree).
    output_dir: PathBuf,
    /// Parquet encoding configuration.
    config: ParquetWriterConfig,
    /// Block durations indexed by `[stage_position][block_index]` in hours.
    ///
    /// `stage_position` is the 0-based position of the stage in
    /// `system.stages()`, which is sorted by stage ID. The simulation
    /// passes 0-based `stage_id` values that map to this same ordering
    /// for study stages (IDs >= 0).
    block_durations: Vec<Vec<f64>>,
    /// Per-line loss factors (`1.0 - losses_percent / 100.0`), keyed by
    /// line entity ID for safe lookup with non-contiguous IDs.
    loss_factors: HashMap<i32, f64>,
    /// Number of scenarios written so far.
    scenarios_written: u32,
    /// Relative partition paths written (one per entity type per scenario).
    partitions_written: Vec<String>,
}

// Compile-time Send assertion: SimulationParquetWriter must be Send because
// it is moved to a background I/O thread.
const _: fn() = || {
    fn assert_send<T: Send>() {}
    assert_send::<SimulationParquetWriter>();
};

impl SimulationParquetWriter {
    /// Create a new writer targeting `output_dir`.
    ///
    /// Extracts block durations and line loss factors from `system`.
    /// Creates the `simulation/` subdirectory and one entity subdirectory
    /// for every entity type with a non-zero entity count.
    ///
    /// # Errors
    ///
    /// - [`OutputError::IoError`] if any directory cannot be created.
    pub fn new(
        output_dir: &Path,
        system: &System,
        config: &ParquetWriterConfig,
    ) -> Result<Self, OutputError> {
        let sim_dir = output_dir.join("simulation");

        // Extract block durations: one inner Vec per stage, indexed by block position.
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

        // costs: always present (every scenario has at least one stage with cost data).
        // We create the directory unconditionally for costs because the system always
        // has stages. All other entity-type directories are gated on count > 0.
        std::fs::create_dir_all(sim_dir.join("costs"))
            .map_err(|e| OutputError::io(sim_dir.join("costs"), e))?;

        if system.n_hydros() > 0 {
            std::fs::create_dir_all(sim_dir.join("hydros"))
                .map_err(|e| OutputError::io(sim_dir.join("hydros"), e))?;
            // inflow_lags shares the same hydro gate.
            std::fs::create_dir_all(sim_dir.join("inflow_lags"))
                .map_err(|e| OutputError::io(sim_dir.join("inflow_lags"), e))?;
        }
        if system.n_thermals() > 0 {
            std::fs::create_dir_all(sim_dir.join("thermals"))
                .map_err(|e| OutputError::io(sim_dir.join("thermals"), e))?;
        }
        if system.n_lines() > 0 {
            std::fs::create_dir_all(sim_dir.join("exchanges"))
                .map_err(|e| OutputError::io(sim_dir.join("exchanges"), e))?;
        }
        if system.n_buses() > 0 {
            std::fs::create_dir_all(sim_dir.join("buses"))
                .map_err(|e| OutputError::io(sim_dir.join("buses"), e))?;
        }
        if system.n_pumping_stations() > 0 {
            std::fs::create_dir_all(sim_dir.join("pumping_stations"))
                .map_err(|e| OutputError::io(sim_dir.join("pumping_stations"), e))?;
        }
        if system.n_contracts() > 0 {
            std::fs::create_dir_all(sim_dir.join("contracts"))
                .map_err(|e| OutputError::io(sim_dir.join("contracts"), e))?;
        }
        if system.n_non_controllable_sources() > 0 {
            std::fs::create_dir_all(sim_dir.join("non_controllables"))
                .map_err(|e| OutputError::io(sim_dir.join("non_controllables"), e))?;
        }
        if !system.generic_constraints().is_empty() {
            std::fs::create_dir_all(sim_dir.join("violations/generic"))
                .map_err(|e| OutputError::io(sim_dir.join("violations/generic"), e))?;
        }

        Ok(Self {
            output_dir: output_dir.to_path_buf(),
            config: config.clone(),
            block_durations,
            loss_factors,
            scenarios_written: 0,
            partitions_written: Vec::new(),
        })
    }

    /// Write one scenario's results to Hive-partitioned Parquet files.
    ///
    /// For each entity type with non-empty result data: creates the partition
    /// directory `simulation/{entity}/scenario_id={id:04d}/`, builds the
    /// Arrow `RecordBatch`, and writes a Parquet file atomically.
    ///
    /// Entity types with empty Vecs (zero entities in the system) are skipped
    /// entirely — no directory is created and no file is written.
    ///
    /// # Errors
    ///
    /// - [`OutputError::SerializationError`] if a `RecordBatch` cannot be
    ///   constructed (array length mismatch).
    /// - [`OutputError::IoError`] if any filesystem operation fails.
    #[allow(clippy::too_many_lines)] // 10 entity types × ~10 lines each is inherently long
    #[allow(clippy::needless_pass_by_value)] // consuming by value is intentional: payload drives output
    pub fn write_scenario(&mut self, result: ScenarioWritePayload) -> Result<(), OutputError> {
        let id = result.scenario_id;
        let sim_dir = self.output_dir.join("simulation");
        let partition_suffix = format!("scenario_id={id:04}");

        // Collect all records across stages.
        let mut all_costs: Vec<&CostWriteRecord> = Vec::new();
        let mut all_hydros: Vec<&HydroWriteRecord> = Vec::new();
        let mut all_thermals: Vec<&ThermalWriteRecord> = Vec::new();
        let mut all_exchanges: Vec<&ExchangeWriteRecord> = Vec::new();
        let mut all_buses: Vec<&BusWriteRecord> = Vec::new();
        let mut all_pumping: Vec<&PumpingWriteRecord> = Vec::new();
        let mut all_contracts: Vec<&ContractWriteRecord> = Vec::new();
        let mut all_non_controllables: Vec<&NonControllableWriteRecord> = Vec::new();
        let mut all_inflow_lags: Vec<&InflowLagWriteRecord> = Vec::new();
        let mut all_violations: Vec<&GenericViolationWriteRecord> = Vec::new();

        for stage in &result.stages {
            all_costs.extend(stage.costs.iter());
            all_hydros.extend(stage.hydros.iter());
            all_thermals.extend(stage.thermals.iter());
            all_exchanges.extend(stage.exchanges.iter());
            all_buses.extend(stage.buses.iter());
            all_pumping.extend(stage.pumping_stations.iter());
            all_contracts.extend(stage.contracts.iter());
            all_non_controllables.extend(stage.non_controllables.iter());
            all_inflow_lags.extend(stage.inflow_lags.iter());
            all_violations.extend(stage.generic_violations.iter());
        }

        // costs — always written when there are records.
        if !all_costs.is_empty() {
            let part_dir = sim_dir.join("costs").join(&partition_suffix);
            std::fs::create_dir_all(&part_dir).map_err(|e| OutputError::io(&part_dir, e))?;
            let batch = build_costs_batch(&all_costs)?;
            let file_path = part_dir.join("data.parquet");
            write_parquet_atomic(&file_path, &batch, &self.config)?;
            self.partitions_written
                .push(format!("simulation/costs/{partition_suffix}/data.parquet"));
        }

        // hydros
        if !all_hydros.is_empty() {
            let part_dir = sim_dir.join("hydros").join(&partition_suffix);
            std::fs::create_dir_all(&part_dir).map_err(|e| OutputError::io(&part_dir, e))?;
            let batch = build_hydros_batch(&all_hydros, &self.block_durations)?;
            let file_path = part_dir.join("data.parquet");
            write_parquet_atomic(&file_path, &batch, &self.config)?;
            self.partitions_written
                .push(format!("simulation/hydros/{partition_suffix}/data.parquet"));
        }

        // thermals
        if !all_thermals.is_empty() {
            let part_dir = sim_dir.join("thermals").join(&partition_suffix);
            std::fs::create_dir_all(&part_dir).map_err(|e| OutputError::io(&part_dir, e))?;
            let batch = build_thermals_batch(&all_thermals, &self.block_durations)?;
            let file_path = part_dir.join("data.parquet");
            write_parquet_atomic(&file_path, &batch, &self.config)?;
            self.partitions_written.push(format!(
                "simulation/thermals/{partition_suffix}/data.parquet"
            ));
        }

        // exchanges
        if !all_exchanges.is_empty() {
            let part_dir = sim_dir.join("exchanges").join(&partition_suffix);
            std::fs::create_dir_all(&part_dir).map_err(|e| OutputError::io(&part_dir, e))?;
            let batch =
                build_exchanges_batch(&all_exchanges, &self.block_durations, &self.loss_factors)?;
            let file_path = part_dir.join("data.parquet");
            write_parquet_atomic(&file_path, &batch, &self.config)?;
            self.partitions_written.push(format!(
                "simulation/exchanges/{partition_suffix}/data.parquet"
            ));
        }

        // buses
        if !all_buses.is_empty() {
            let part_dir = sim_dir.join("buses").join(&partition_suffix);
            std::fs::create_dir_all(&part_dir).map_err(|e| OutputError::io(&part_dir, e))?;
            let batch = build_buses_batch(&all_buses, &self.block_durations)?;
            let file_path = part_dir.join("data.parquet");
            write_parquet_atomic(&file_path, &batch, &self.config)?;
            self.partitions_written
                .push(format!("simulation/buses/{partition_suffix}/data.parquet"));
        }

        // pumping_stations
        if !all_pumping.is_empty() {
            let part_dir = sim_dir.join("pumping_stations").join(&partition_suffix);
            std::fs::create_dir_all(&part_dir).map_err(|e| OutputError::io(&part_dir, e))?;
            let batch = build_pumping_batch(&all_pumping, &self.block_durations)?;
            let file_path = part_dir.join("data.parquet");
            write_parquet_atomic(&file_path, &batch, &self.config)?;
            self.partitions_written.push(format!(
                "simulation/pumping_stations/{partition_suffix}/data.parquet"
            ));
        }

        // contracts
        if !all_contracts.is_empty() {
            let part_dir = sim_dir.join("contracts").join(&partition_suffix);
            std::fs::create_dir_all(&part_dir).map_err(|e| OutputError::io(&part_dir, e))?;
            let batch = build_contracts_batch(&all_contracts, &self.block_durations)?;
            let file_path = part_dir.join("data.parquet");
            write_parquet_atomic(&file_path, &batch, &self.config)?;
            self.partitions_written.push(format!(
                "simulation/contracts/{partition_suffix}/data.parquet"
            ));
        }

        // non_controllables
        if !all_non_controllables.is_empty() {
            let part_dir = sim_dir.join("non_controllables").join(&partition_suffix);
            std::fs::create_dir_all(&part_dir).map_err(|e| OutputError::io(&part_dir, e))?;
            let batch =
                build_non_controllables_batch(&all_non_controllables, &self.block_durations)?;
            let file_path = part_dir.join("data.parquet");
            write_parquet_atomic(&file_path, &batch, &self.config)?;
            self.partitions_written.push(format!(
                "simulation/non_controllables/{partition_suffix}/data.parquet"
            ));
        }

        // inflow_lags
        if !all_inflow_lags.is_empty() {
            let part_dir = sim_dir.join("inflow_lags").join(&partition_suffix);
            std::fs::create_dir_all(&part_dir).map_err(|e| OutputError::io(&part_dir, e))?;
            let batch = build_inflow_lags_batch(&all_inflow_lags)?;
            let file_path = part_dir.join("data.parquet");
            write_parquet_atomic(&file_path, &batch, &self.config)?;
            self.partitions_written.push(format!(
                "simulation/inflow_lags/{partition_suffix}/data.parquet"
            ));
        }

        // violations/generic
        if !all_violations.is_empty() {
            let part_dir = sim_dir.join("violations/generic").join(&partition_suffix);
            std::fs::create_dir_all(&part_dir).map_err(|e| OutputError::io(&part_dir, e))?;
            let batch = build_generic_violations_batch(&all_violations)?;
            let file_path = part_dir.join("data.parquet");
            write_parquet_atomic(&file_path, &batch, &self.config)?;
            self.partitions_written.push(format!(
                "simulation/violations/generic/{partition_suffix}/data.parquet"
            ));
        }

        self.scenarios_written += 1;
        Ok(())
    }

    /// Finalize writing and return the [`SimulationOutput`] summary.
    ///
    /// Consumes the writer. Returns a [`SimulationOutput`] with the total
    /// number of scenarios written and the list of Hive partition paths.
    #[must_use]
    pub fn finalize(self) -> SimulationOutput {
        SimulationOutput {
            n_scenarios: self.scenarios_written,
            completed: self.scenarios_written,
            failed: 0,
            partitions_written: self.partitions_written,
        }
    }
}

// ---------------------------------------------------------------------------
// Block duration lookup helper
// ---------------------------------------------------------------------------

/// Look up the duration in hours for the block identified by `(stage_id, block_id)`.
///
/// `stage_id` is the 0-based stage position used by the simulation, which
/// corresponds to the index in `block_durations`. Returns `1.0` as a safe
/// fallback when the stage or block index is out of range (should not occur
/// in well-formed simulation output).
fn block_duration(block_durations: &[Vec<f64>], stage_id: u32, block_id: Option<u32>) -> f64 {
    let Some(block_idx) = block_id else {
        // Stage-level aggregate rows have no block_id — duration is not applicable
        // for energy conversion on these rows; use 1.0 so multiplications are identity.
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

/// Build the costs `RecordBatch` from a slice of cost records.
#[allow(clippy::cast_possible_wrap)]
fn build_costs_batch(records: &[&CostWriteRecord]) -> Result<RecordBatch, OutputError> {
    let schema = Arc::new(costs_schema());
    let n = records.len();

    let mut stage_id = Int32Builder::with_capacity(n);
    let mut block_id = Int32Builder::with_capacity(n);
    let mut total_cost = Float64Builder::with_capacity(n);
    let mut immediate_cost = Float64Builder::with_capacity(n);
    let mut future_cost = Float64Builder::with_capacity(n);
    let mut discount_factor = Float64Builder::with_capacity(n);
    let mut thermal_cost = Float64Builder::with_capacity(n);
    let mut contract_cost = Float64Builder::with_capacity(n);
    let mut deficit_cost = Float64Builder::with_capacity(n);
    let mut excess_cost = Float64Builder::with_capacity(n);
    let mut storage_violation_cost = Float64Builder::with_capacity(n);
    let mut filling_target_cost = Float64Builder::with_capacity(n);
    let mut hydro_violation_cost = Float64Builder::with_capacity(n);
    let mut inflow_penalty_cost = Float64Builder::with_capacity(n);
    let mut generic_violation_cost = Float64Builder::with_capacity(n);
    let mut spillage_cost = Float64Builder::with_capacity(n);
    let mut fpha_turbined_cost = Float64Builder::with_capacity(n);
    let mut curtailment_cost = Float64Builder::with_capacity(n);
    let mut exchange_cost = Float64Builder::with_capacity(n);
    let mut pumping_cost = Float64Builder::with_capacity(n);

    for r in records {
        stage_id.append_value(r.stage_id as i32);
        block_id.append_option(r.block_id.map(|b| b as i32));
        total_cost.append_value(r.total_cost);
        immediate_cost.append_value(r.immediate_cost);
        future_cost.append_value(r.future_cost);
        discount_factor.append_value(r.discount_factor);
        thermal_cost.append_value(r.thermal_cost);
        contract_cost.append_value(r.contract_cost);
        deficit_cost.append_value(r.deficit_cost);
        excess_cost.append_value(r.excess_cost);
        storage_violation_cost.append_value(r.storage_violation_cost);
        filling_target_cost.append_value(r.filling_target_cost);
        hydro_violation_cost.append_value(r.hydro_violation_cost);
        inflow_penalty_cost.append_value(r.inflow_penalty_cost);
        generic_violation_cost.append_value(r.generic_violation_cost);
        spillage_cost.append_value(r.spillage_cost);
        fpha_turbined_cost.append_value(r.fpha_turbined_cost);
        curtailment_cost.append_value(r.curtailment_cost);
        exchange_cost.append_value(r.exchange_cost);
        pumping_cost.append_value(r.pumping_cost);
    }

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(stage_id.finish()),
            Arc::new(block_id.finish()),
            Arc::new(total_cost.finish()),
            Arc::new(immediate_cost.finish()),
            Arc::new(future_cost.finish()),
            Arc::new(discount_factor.finish()),
            Arc::new(thermal_cost.finish()),
            Arc::new(contract_cost.finish()),
            Arc::new(deficit_cost.finish()),
            Arc::new(excess_cost.finish()),
            Arc::new(storage_violation_cost.finish()),
            Arc::new(filling_target_cost.finish()),
            Arc::new(hydro_violation_cost.finish()),
            Arc::new(inflow_penalty_cost.finish()),
            Arc::new(generic_violation_cost.finish()),
            Arc::new(spillage_cost.finish()),
            Arc::new(fpha_turbined_cost.finish()),
            Arc::new(curtailment_cost.finish()),
            Arc::new(exchange_cost.finish()),
            Arc::new(pumping_cost.finish()),
        ],
    )
    .map_err(|e| OutputError::serialization("costs", e.to_string()))
}

/// Build the hydros `RecordBatch`, computing derived columns.
///
/// Derived columns:
/// - `generation_mwh = generation_mw * block_duration_hours`
/// - `outflow_m3s = turbined_m3s + spillage_m3s`
#[allow(clippy::cast_possible_wrap)]
fn build_hydros_batch(
    records: &[&HydroWriteRecord],
    block_durations: &[Vec<f64>],
) -> Result<RecordBatch, OutputError> {
    let schema = Arc::new(hydros_schema());
    let n = records.len();

    let mut stage_id = Int32Builder::with_capacity(n);
    let mut block_id = Int32Builder::with_capacity(n);
    let mut hydro_id = Int32Builder::with_capacity(n);
    let mut turbined_m3s = Float64Builder::with_capacity(n);
    let mut spillage_m3s = Float64Builder::with_capacity(n);
    let mut outflow_m3s = Float64Builder::with_capacity(n);
    let mut evaporation_m3s = Float64Builder::with_capacity(n);
    let mut diverted_inflow_m3s = Float64Builder::with_capacity(n);
    let mut diverted_outflow_m3s = Float64Builder::with_capacity(n);
    let mut incremental_inflow_m3s = Float64Builder::with_capacity(n);
    let mut inflow_m3s = Float64Builder::with_capacity(n);
    let mut storage_initial_hm3 = Float64Builder::with_capacity(n);
    let mut storage_final_hm3 = Float64Builder::with_capacity(n);
    let mut generation_mw = Float64Builder::with_capacity(n);
    let mut generation_mwh = Float64Builder::with_capacity(n);
    let mut productivity_mw_per_m3s = Float64Builder::with_capacity(n);
    let mut spillage_cost = Float64Builder::with_capacity(n);
    let mut water_value_per_hm3 = Float64Builder::with_capacity(n);
    let mut storage_binding_code = Int8Builder::with_capacity(n);
    let mut operative_state_code = Int8Builder::with_capacity(n);
    let mut turbined_slack_m3s = Float64Builder::with_capacity(n);
    let mut outflow_slack_below_m3s = Float64Builder::with_capacity(n);
    let mut outflow_slack_above_m3s = Float64Builder::with_capacity(n);
    let mut generation_slack_mw = Float64Builder::with_capacity(n);
    let mut storage_violation_below_hm3 = Float64Builder::with_capacity(n);
    let mut filling_target_violation_hm3 = Float64Builder::with_capacity(n);
    let mut evaporation_violation_m3s = Float64Builder::with_capacity(n);
    let mut inflow_nonnegativity_slack_m3s = Float64Builder::with_capacity(n);

    for r in records {
        let dur = block_duration(block_durations, r.stage_id, r.block_id);
        stage_id.append_value(r.stage_id as i32);
        block_id.append_option(r.block_id.map(|b| b as i32));
        hydro_id.append_value(r.hydro_id);
        turbined_m3s.append_value(r.turbined_m3s);
        spillage_m3s.append_value(r.spillage_m3s);
        // Derived: outflow_m3s = turbined_m3s + spillage_m3s
        outflow_m3s.append_value(r.turbined_m3s + r.spillage_m3s);
        evaporation_m3s.append_option(r.evaporation_m3s);
        diverted_inflow_m3s.append_option(r.diverted_inflow_m3s);
        diverted_outflow_m3s.append_option(r.diverted_outflow_m3s);
        incremental_inflow_m3s.append_value(r.incremental_inflow_m3s);
        inflow_m3s.append_value(r.inflow_m3s);
        storage_initial_hm3.append_value(r.storage_initial_hm3);
        storage_final_hm3.append_value(r.storage_final_hm3);
        generation_mw.append_value(r.generation_mw);
        // Derived: generation_mwh = generation_mw * block_duration_hours
        generation_mwh.append_value(r.generation_mw * dur);
        productivity_mw_per_m3s.append_option(r.productivity_mw_per_m3s);
        spillage_cost.append_value(r.spillage_cost);
        water_value_per_hm3.append_value(r.water_value_per_hm3);
        storage_binding_code.append_value(r.storage_binding_code);
        operative_state_code.append_value(r.operative_state_code);
        turbined_slack_m3s.append_value(r.turbined_slack_m3s);
        outflow_slack_below_m3s.append_value(r.outflow_slack_below_m3s);
        outflow_slack_above_m3s.append_value(r.outflow_slack_above_m3s);
        generation_slack_mw.append_value(r.generation_slack_mw);
        storage_violation_below_hm3.append_value(r.storage_violation_below_hm3);
        filling_target_violation_hm3.append_value(r.filling_target_violation_hm3);
        evaporation_violation_m3s.append_value(r.evaporation_violation_m3s);
        inflow_nonnegativity_slack_m3s.append_value(r.inflow_nonnegativity_slack_m3s);
    }

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(stage_id.finish()),
            Arc::new(block_id.finish()),
            Arc::new(hydro_id.finish()),
            Arc::new(turbined_m3s.finish()),
            Arc::new(spillage_m3s.finish()),
            Arc::new(outflow_m3s.finish()),
            Arc::new(evaporation_m3s.finish()),
            Arc::new(diverted_inflow_m3s.finish()),
            Arc::new(diverted_outflow_m3s.finish()),
            Arc::new(incremental_inflow_m3s.finish()),
            Arc::new(inflow_m3s.finish()),
            Arc::new(storage_initial_hm3.finish()),
            Arc::new(storage_final_hm3.finish()),
            Arc::new(generation_mw.finish()),
            Arc::new(generation_mwh.finish()),
            Arc::new(productivity_mw_per_m3s.finish()),
            Arc::new(spillage_cost.finish()),
            Arc::new(water_value_per_hm3.finish()),
            Arc::new(storage_binding_code.finish()),
            Arc::new(operative_state_code.finish()),
            Arc::new(turbined_slack_m3s.finish()),
            Arc::new(outflow_slack_below_m3s.finish()),
            Arc::new(outflow_slack_above_m3s.finish()),
            Arc::new(generation_slack_mw.finish()),
            Arc::new(storage_violation_below_hm3.finish()),
            Arc::new(filling_target_violation_hm3.finish()),
            Arc::new(evaporation_violation_m3s.finish()),
            Arc::new(inflow_nonnegativity_slack_m3s.finish()),
        ],
    )
    .map_err(|e| OutputError::serialization("hydros", e.to_string()))
}

/// Build the thermals `RecordBatch`, computing the derived column.
///
/// Derived column:
/// - `generation_mwh = generation_mw * block_duration_hours`
#[allow(clippy::cast_possible_wrap)]
fn build_thermals_batch(
    records: &[&ThermalWriteRecord],
    block_durations: &[Vec<f64>],
) -> Result<RecordBatch, OutputError> {
    let schema = Arc::new(thermals_schema());
    let n = records.len();

    let mut stage_id = Int32Builder::with_capacity(n);
    let mut block_id = Int32Builder::with_capacity(n);
    let mut thermal_id = Int32Builder::with_capacity(n);
    let mut generation_mw = Float64Builder::with_capacity(n);
    let mut generation_mwh = Float64Builder::with_capacity(n);
    let mut generation_cost = Float64Builder::with_capacity(n);
    let mut is_gnl = BooleanBuilder::with_capacity(n);
    let mut gnl_committed_mw = Float64Builder::with_capacity(n);
    let mut gnl_decision_mw = Float64Builder::with_capacity(n);
    let mut operative_state_code = Int8Builder::with_capacity(n);

    for r in records {
        let dur = block_duration(block_durations, r.stage_id, r.block_id);
        stage_id.append_value(r.stage_id as i32);
        block_id.append_option(r.block_id.map(|b| b as i32));
        thermal_id.append_value(r.thermal_id);
        generation_mw.append_value(r.generation_mw);
        // Derived: generation_mwh = generation_mw * block_duration_hours
        generation_mwh.append_value(r.generation_mw * dur);
        generation_cost.append_value(r.generation_cost);
        is_gnl.append_value(r.is_gnl);
        gnl_committed_mw.append_option(r.gnl_committed_mw);
        gnl_decision_mw.append_option(r.gnl_decision_mw);
        operative_state_code.append_value(r.operative_state_code);
    }

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(stage_id.finish()),
            Arc::new(block_id.finish()),
            Arc::new(thermal_id.finish()),
            Arc::new(generation_mw.finish()),
            Arc::new(generation_mwh.finish()),
            Arc::new(generation_cost.finish()),
            Arc::new(is_gnl.finish()),
            Arc::new(gnl_committed_mw.finish()),
            Arc::new(gnl_decision_mw.finish()),
            Arc::new(operative_state_code.finish()),
        ],
    )
    .map_err(|e| OutputError::serialization("thermals", e.to_string()))
}

/// Build the exchanges `RecordBatch`, computing derived columns.
///
/// Derived columns:
/// - `net_flow_mw = direct_flow_mw - reverse_flow_mw`
/// - `net_flow_mwh = net_flow_mw * block_duration_hours`
/// - `losses_mw = (1.0 - loss_factor) * (direct_flow_mw + reverse_flow_mw)`
///   where `loss_factor = 1.0 - losses_percent / 100.0`
/// - `losses_mwh = losses_mw * block_duration_hours`
///
/// `loss_factors` maps line entity ID to its loss factor. Unknown IDs
/// default to `1.0` (zero losses).
#[allow(
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::similar_names // MW / MWh builder pairs are semantically paired and intentionally similar
)]
fn build_exchanges_batch(
    records: &[&ExchangeWriteRecord],
    block_durations: &[Vec<f64>],
    loss_factors: &HashMap<i32, f64>,
) -> Result<RecordBatch, OutputError> {
    let schema = Arc::new(exchanges_schema());
    let n = records.len();

    let mut stage_id = Int32Builder::with_capacity(n);
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
        // losses_mw = (1.0 - loss_factor) * total_flow
        // loss_factor = 1.0 - losses_percent/100.0
        // so (1.0 - loss_factor) = losses_percent/100.0
        let losses = (1.0 - lf) * total_flow;

        stage_id.append_value(r.stage_id as i32);
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
            Arc::new(stage_id.finish()),
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

/// Build the buses `RecordBatch`, computing derived columns.
///
/// Derived columns:
/// - `load_mwh = load_mw * block_duration_hours`
/// - `deficit_mwh = deficit_mw * block_duration_hours`
/// - `excess_mwh = excess_mw * block_duration_hours`
#[allow(clippy::cast_possible_wrap)]
fn build_buses_batch(
    records: &[&BusWriteRecord],
    block_durations: &[Vec<f64>],
) -> Result<RecordBatch, OutputError> {
    let schema = Arc::new(buses_schema());
    let n = records.len();

    let mut stage_id = Int32Builder::with_capacity(n);
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
        stage_id.append_value(r.stage_id as i32);
        block_id.append_option(r.block_id.map(|b| b as i32));
        bus_id.append_value(r.bus_id);
        load_mw.append_value(r.load_mw);
        // Derived
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
            Arc::new(stage_id.finish()),
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

/// Build the `pumping_stations` `RecordBatch`, computing derived columns.
///
/// Derived columns:
/// - `pumped_volume_hm3 = pumped_flow_m3s * block_duration_hours * 3600.0 / 1e6`
/// - `energy_consumption_mwh = power_consumption_mw * block_duration_hours`
#[allow(clippy::cast_possible_wrap)]
fn build_pumping_batch(
    records: &[&PumpingWriteRecord],
    block_durations: &[Vec<f64>],
) -> Result<RecordBatch, OutputError> {
    let schema = Arc::new(pumping_stations_schema());
    let n = records.len();

    let mut stage_id = Int32Builder::with_capacity(n);
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
        stage_id.append_value(r.stage_id as i32);
        block_id.append_option(r.block_id.map(|b| b as i32));
        pumping_station_id.append_value(r.pumping_station_id);
        pumped_flow_m3s.append_value(r.pumped_flow_m3s);
        // Derived: pumped_volume_hm3 = pumped_flow_m3s * hours * 3600 / 1e6
        pumped_volume_hm3.append_value(r.pumped_flow_m3s * dur * 3600.0 / 1_000_000.0);
        power_consumption_mw.append_value(r.power_consumption_mw);
        // Derived: energy_consumption_mwh = power_consumption_mw * hours
        energy_consumption_mwh.append_value(r.power_consumption_mw * dur);
        pumping_cost.append_value(r.pumping_cost);
        operative_state_code.append_value(r.operative_state_code);
    }

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(stage_id.finish()),
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

/// Build the contracts `RecordBatch`, computing the derived column.
///
/// Derived column:
/// - `energy_mwh = power_mw * block_duration_hours`
#[allow(clippy::cast_possible_wrap)]
fn build_contracts_batch(
    records: &[&ContractWriteRecord],
    block_durations: &[Vec<f64>],
) -> Result<RecordBatch, OutputError> {
    let schema = Arc::new(contracts_schema());
    let n = records.len();

    let mut stage_id = Int32Builder::with_capacity(n);
    let mut block_id = Int32Builder::with_capacity(n);
    let mut contract_id = Int32Builder::with_capacity(n);
    let mut power_mw = Float64Builder::with_capacity(n);
    let mut energy_mwh = Float64Builder::with_capacity(n);
    let mut price_per_mwh = Float64Builder::with_capacity(n);
    let mut total_cost = Float64Builder::with_capacity(n);
    let mut operative_state_code = Int8Builder::with_capacity(n);

    for r in records {
        let dur = block_duration(block_durations, r.stage_id, r.block_id);
        stage_id.append_value(r.stage_id as i32);
        block_id.append_option(r.block_id.map(|b| b as i32));
        contract_id.append_value(r.contract_id);
        power_mw.append_value(r.power_mw);
        // Derived: energy_mwh = power_mw * block_duration_hours
        energy_mwh.append_value(r.power_mw * dur);
        price_per_mwh.append_value(r.price_per_mwh);
        total_cost.append_value(r.total_cost);
        operative_state_code.append_value(r.operative_state_code);
    }

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(stage_id.finish()),
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

/// Build the `non_controllables` `RecordBatch`, computing derived columns.
///
/// Derived columns:
/// - `generation_mwh = generation_mw * block_duration_hours`
/// - `curtailment_mwh = curtailment_mw * block_duration_hours`
#[allow(clippy::cast_possible_wrap)]
fn build_non_controllables_batch(
    records: &[&NonControllableWriteRecord],
    block_durations: &[Vec<f64>],
) -> Result<RecordBatch, OutputError> {
    let schema = Arc::new(non_controllables_schema());
    let n = records.len();

    let mut stage_id = Int32Builder::with_capacity(n);
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
        stage_id.append_value(r.stage_id as i32);
        block_id.append_option(r.block_id.map(|b| b as i32));
        non_controllable_id.append_value(r.non_controllable_id);
        generation_mw.append_value(r.generation_mw);
        // Derived
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
            Arc::new(stage_id.finish()),
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

/// Build the `inflow_lags` `RecordBatch`.
///
/// No derived columns — all four fields are stored directly.
#[allow(clippy::cast_possible_wrap)]
fn build_inflow_lags_batch(records: &[&InflowLagWriteRecord]) -> Result<RecordBatch, OutputError> {
    let schema = Arc::new(inflow_lags_schema());
    let n = records.len();

    let mut stage_id = Int32Builder::with_capacity(n);
    let mut hydro_id = Int32Builder::with_capacity(n);
    let mut lag_index = Int32Builder::with_capacity(n);
    let mut inflow_m3s = Float64Builder::with_capacity(n);

    for r in records {
        stage_id.append_value(r.stage_id as i32);
        hydro_id.append_value(r.hydro_id);
        lag_index.append_value(r.lag_index as i32);
        inflow_m3s.append_value(r.inflow_m3s);
    }

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(stage_id.finish()),
            Arc::new(hydro_id.finish()),
            Arc::new(lag_index.finish()),
            Arc::new(inflow_m3s.finish()),
        ],
    )
    .map_err(|e| OutputError::serialization("inflow_lags", e.to_string()))
}

/// Build the generic violations `RecordBatch`.
///
/// No derived columns.
#[allow(clippy::cast_possible_wrap)]
fn build_generic_violations_batch(
    records: &[&GenericViolationWriteRecord],
) -> Result<RecordBatch, OutputError> {
    let schema = Arc::new(generic_violations_schema());
    let n = records.len();

    let mut stage_id = Int32Builder::with_capacity(n);
    let mut block_id = Int32Builder::with_capacity(n);
    let mut constraint_id = Int32Builder::with_capacity(n);
    let mut slack_value = Float64Builder::with_capacity(n);
    let mut slack_cost = Float64Builder::with_capacity(n);

    for r in records {
        stage_id.append_value(r.stage_id as i32);
        block_id.append_option(r.block_id.map(|b| b as i32));
        constraint_id.append_value(r.constraint_id);
        slack_value.append_value(r.slack_value);
        slack_cost.append_value(r.slack_cost);
    }

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(stage_id.finish()),
            Arc::new(block_id.finish()),
            Arc::new(constraint_id.finish()),
            Arc::new(slack_value.finish()),
            Arc::new(slack_cost.finish()),
        ],
    )
    .map_err(|e| OutputError::serialization("generic_violations", e.to_string()))
}

// ---------------------------------------------------------------------------
// Atomic Parquet write helper
// ---------------------------------------------------------------------------

/// Write a `RecordBatch` to `path` as a Parquet file, atomically.
///
/// Writes to `{path}.tmp` first, then renames to `path`. If any step fails
/// the `.tmp` file may remain on disk but the final path is never partially
/// written.
fn write_parquet_atomic(
    path: &Path,
    batch: &RecordBatch,
    config: &ParquetWriterConfig,
) -> Result<(), OutputError> {
    let tmp_path = path.with_extension(path.extension().map_or_else(
        || "tmp".to_string(),
        |ext| format!("{}.tmp", ext.to_string_lossy()),
    ));

    let props = WriterProperties::builder()
        .set_compression(config.compression)
        .set_max_row_group_row_count(Some(config.row_group_size))
        .set_dictionary_enabled(config.dictionary_encoding)
        .build();

    let file = std::fs::File::create(&tmp_path).map_err(|e| OutputError::io(&tmp_path, e))?;

    let mut writer = ArrowWriter::try_new(file, batch.schema(), Some(props))
        .map_err(|e| OutputError::serialization("parquet_writer", e.to_string()))?;

    writer
        .write(batch)
        .map_err(|e| OutputError::serialization("parquet_writer", e.to_string()))?;

    writer
        .close()
        .map_err(|e| OutputError::serialization("parquet_writer", e.to_string()))?;

    std::fs::rename(&tmp_path, path).map_err(|e| OutputError::io(path, e))?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::cast_possible_truncation,
    clippy::float_cmp,
    clippy::panic
)]
mod tests {
    use super::*;
    use chrono::NaiveDate;
    use cobre_core::{
        Block, BlockMode, Bus, DeficitSegment, EntityId, Hydro, HydroGenerationModel,
        HydroPenalties, Line, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig, SystemBuilder, Thermal, ThermalCostSegment,
    };

    // -----------------------------------------------------------------------
    // Test fixture helpers
    // -----------------------------------------------------------------------

    fn make_hydro_penalties_zero() -> HydroPenalties {
        HydroPenalties {
            spillage_cost: 0.0,
            diversion_cost: 0.0,
            fpha_turbined_cost: 0.0,
            storage_violation_below_cost: 0.0,
            filling_target_violation_cost: 0.0,
            turbined_violation_below_cost: 0.0,
            outflow_violation_below_cost: 0.0,
            outflow_violation_above_cost: 0.0,
            generation_violation_below_cost: 0.0,
            evaporation_violation_cost: 0.0,
            water_withdrawal_violation_cost: 0.0,
        }
    }

    fn make_hydro(id: i32) -> Hydro {
        Hydro {
            id: EntityId(id),
            name: format!("H{id}"),
            bus_id: EntityId(1),
            downstream_id: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 1000.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity {
                productivity_mw_per_m3s: 0.9,
            },
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 1000.0,
            min_generation_mw: 0.0,
            max_generation_mw: 900.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            diversion: None,
            filling: None,
            penalties: make_hydro_penalties_zero(),
        }
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
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 1000.0,
            }],
            excess_cost: 0.0,
        };

        let line = Line {
            id: EntityId(1),
            name: "L1".to_string(),
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
            bus_id: EntityId(1),
            entry_stage_id: None,
            exit_stage_id: None,
            cost_segments: vec![ThermalCostSegment {
                capacity_mw: 100.0,
                cost_per_mwh: 50.0,
            }],
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            gnl_config: None,
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

    fn make_cost_record(stage_id: u32, block_id: Option<u32>) -> CostWriteRecord {
        CostWriteRecord {
            stage_id,
            block_id,
            total_cost: 1000.0,
            immediate_cost: 800.0,
            future_cost: 200.0,
            discount_factor: 0.95,
            thermal_cost: 400.0,
            contract_cost: 0.0,
            deficit_cost: 100.0,
            excess_cost: 0.0,
            storage_violation_cost: 0.0,
            filling_target_cost: 0.0,
            hydro_violation_cost: 0.0,
            inflow_penalty_cost: 0.0,
            generic_violation_cost: 0.0,
            spillage_cost: 5.0,
            fpha_turbined_cost: 3.0,
            curtailment_cost: 0.0,
            exchange_cost: 2.0,
            pumping_cost: 0.0,
        }
    }

    fn make_hydro_record(stage_id: u32, block_id: Option<u32>, hydro_id: i32) -> HydroWriteRecord {
        HydroWriteRecord {
            stage_id,
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
            productivity_mw_per_m3s: Some(0.9),
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
            evaporation_violation_m3s: 0.0,
            inflow_nonnegativity_slack_m3s: 0.0,
        }
    }

    fn make_scenario_payload(scenario_id: u32, n_stages: usize) -> ScenarioWritePayload {
        let stages = (0..n_stages as u32)
            .map(|s| StageWritePayload {
                stage_id: s,
                costs: vec![make_cost_record(s, Some(0))],
                hydros: vec![
                    make_hydro_record(s, Some(0), 1),
                    make_hydro_record(s, Some(0), 2),
                ],
                thermals: vec![],
                exchanges: vec![],
                buses: vec![],
                pumping_stations: vec![],
                contracts: vec![],
                non_controllables: vec![],
                inflow_lags: vec![],
                generic_violations: vec![],
            })
            .collect();
        ScenarioWritePayload {
            scenario_id,
            stages,
        }
    }

    // -----------------------------------------------------------------------
    // Unit tests: batch builders
    // -----------------------------------------------------------------------

    #[test]
    fn build_costs_batch_from_two_stages() {
        let r0 = make_cost_record(0, Some(0));
        let r1 = make_cost_record(1, Some(0));
        let records = vec![&r0, &r1];
        let batch = build_costs_batch(&records).expect("costs batch must build");

        assert_eq!(batch.num_rows(), 2, "must have 2 rows");
        assert_eq!(batch.num_columns(), 20, "costs schema has 20 columns");

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

        let r0 = make_hydro_record(0, Some(0), 1); // generation_mw = 50.0
        let r1 = make_hydro_record(1, Some(0), 2); // generation_mw = 50.0
        let records = vec![&r0, &r1];

        let batch =
            build_hydros_batch(&records, &block_durations).expect("hydros batch must build");
        assert_eq!(batch.num_rows(), 2);
        assert_eq!(batch.num_columns(), 28, "hydros schema has 28 columns");

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
    }

    #[test]
    fn build_exchanges_batch_net_flow_and_losses() {
        // One stage, one block, 720 hours.
        let block_durations = vec![vec![720.0_f64]];
        // Line 1 has losses_percent=2.5 → loss_factor=0.975 → (1-lf)=0.025
        let loss_factors = HashMap::from([(1, 0.975_f64)]);

        let r = ExchangeWriteRecord {
            stage_id: 0,
            block_id: Some(0),
            line_id: 1,
            direct_flow_mw: 100.0,
            reverse_flow_mw: 0.0,
            exchange_cost: 5.0,
            operative_state_code: 1,
        };
        let records = vec![&r];

        let batch = build_exchanges_batch(&records, &block_durations, &loss_factors)
            .expect("exchanges batch must build");
        assert_eq!(batch.num_rows(), 1);
        assert_eq!(batch.num_columns(), 11, "exchanges schema has 11 columns");

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
        let records = vec![&r_with, &r_without];

        let batch = build_costs_batch(&records).expect("costs batch must build");
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
        let config = ParquetWriterConfig::default();

        let mut writer =
            SimulationParquetWriter::new(tmp.path(), &system, &config).expect("new must succeed");

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
    fn write_scenario_skips_empty_entity_types() {
        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        std::fs::create_dir_all(tmp.path().join("simulation")).unwrap();

        // System with no contracts, pumping stations, non-controllables, or generics.
        let system = make_test_system();
        let config = ParquetWriterConfig::default();

        let mut writer =
            SimulationParquetWriter::new(tmp.path(), &system, &config).expect("new must succeed");

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
    fn finalize_returns_correct_counts() {
        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        std::fs::create_dir_all(tmp.path().join("simulation")).unwrap();

        let system = make_test_system();
        let config = ParquetWriterConfig::default();

        let mut writer =
            SimulationParquetWriter::new(tmp.path(), &system, &config).expect("new must succeed");

        writer
            .write_scenario(make_scenario_payload(0, 1))
            .expect("write scenario 0 must succeed");
        writer
            .write_scenario(make_scenario_payload(1, 1))
            .expect("write scenario 1 must succeed");

        let output = writer.finalize();
        assert_eq!(output.n_scenarios, 2, "n_scenarios must be 2");
        assert_eq!(output.completed, 2, "completed must be 2");
        assert_eq!(output.failed, 0, "failed must be 0");
    }

    #[test]
    fn finalize_partitions_written_contains_all_paths() {
        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        std::fs::create_dir_all(tmp.path().join("simulation")).unwrap();

        let system = make_test_system();
        let config = ParquetWriterConfig::default();

        let mut writer =
            SimulationParquetWriter::new(tmp.path(), &system, &config).expect("new must succeed");

        writer
            .write_scenario(make_scenario_payload(0, 1))
            .expect("write scenario 0 must succeed");

        let output = writer.finalize();
        // The test system has hydros, so at minimum costs and hydros partitions.
        assert!(
            output.partitions_written.len() >= 2,
            "partitions_written must include costs and hydros partitions"
        );
        assert!(
            output
                .partitions_written
                .iter()
                .any(|p| p.contains("simulation/costs/scenario_id=0000")),
            "partitions_written must contain costs partition for scenario 0"
        );
        assert!(
            output
                .partitions_written
                .iter()
                .any(|p| p.contains("simulation/hydros/scenario_id=0000")),
            "partitions_written must contain hydros partition for scenario 0"
        );
    }

    #[test]
    fn write_scenario_parquet_roundtrip_costs_row_count() {
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        std::fs::create_dir_all(tmp.path().join("simulation")).unwrap();

        let system = make_test_system();
        let config = ParquetWriterConfig::default();

        let mut writer =
            SimulationParquetWriter::new(tmp.path(), &system, &config).expect("new must succeed");

        // 2 stages, 1 cost record per stage → 2 rows
        let payload = make_scenario_payload(0, 2);
        writer
            .write_scenario(payload)
            .expect("write_scenario must succeed");

        let path = tmp
            .path()
            .join("simulation/costs/scenario_id=0000/data.parquet");
        let file = std::fs::File::open(&path).expect("parquet file must exist");
        let mut reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .expect("reader builder must succeed")
            .build()
            .expect("reader must build");

        let batch = reader
            .next()
            .expect("must have rows")
            .expect("batch must be Ok");
        assert_eq!(batch.num_rows(), 2, "costs parquet must have 2 rows");
        assert_eq!(batch.num_columns(), 20, "costs schema has 20 columns");
    }

    #[test]
    fn write_scenario_parquet_roundtrip_hydros_derived_mwh() {
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        std::fs::create_dir_all(tmp.path().join("simulation")).unwrap();

        let system = make_test_system();
        let config = ParquetWriterConfig::default();

        let mut writer =
            SimulationParquetWriter::new(tmp.path(), &system, &config).expect("new must succeed");

        // 2 stages x 2 hydros = 4 rows in hydros parquet.
        let payload = make_scenario_payload(0, 2);
        writer
            .write_scenario(payload)
            .expect("write_scenario must succeed");

        let path = tmp
            .path()
            .join("simulation/hydros/scenario_id=0000/data.parquet");
        let file = std::fs::File::open(&path).expect("hydros parquet must exist");
        let mut reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .expect("reader builder must succeed")
            .build()
            .expect("reader must build");

        let batch = reader
            .next()
            .expect("must have rows")
            .expect("batch must be Ok");
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
    }

    #[test]
    fn write_scenario_atomic_no_tmp_file_remaining() {
        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        std::fs::create_dir_all(tmp.path().join("simulation")).unwrap();

        let system = make_test_system();
        let config = ParquetWriterConfig::default();

        let mut writer =
            SimulationParquetWriter::new(tmp.path(), &system, &config).expect("new must succeed");

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
}
