//! Simulation I/O thread: receives scenario results from the solver and
//! writes them to Hive-partitioned Parquet files via [`SimulationParquetWriter`].
//!
//! This module also provides the type conversion from solver-specific
//! [`SimulationScenarioResult`] to the I/O-layer [`ScenarioWritePayload`].
//! The two type hierarchies are structurally identical (field-for-field) but
//! live in separate crates to avoid circular dependencies.

#![allow(clippy::needless_pass_by_value)]

use std::sync::mpsc::Receiver;

use cobre_io::SimulationOutput;
use cobre_io::output::simulation_writer::{
    BusWriteRecord, ContractWriteRecord, CostWriteRecord, ExchangeWriteRecord,
    GenericViolationWriteRecord, HydroWriteRecord, InflowLagWriteRecord,
    NonControllableWriteRecord, PumpingWriteRecord, ScenarioWritePayload, SimulationParquetWriter,
    StageWritePayload, ThermalWriteRecord,
};
use cobre_sddp::{
    SimulationBusResult, SimulationContractResult, SimulationCostResult, SimulationExchangeResult,
    SimulationGenericViolationResult, SimulationHydroResult, SimulationInflowLagResult,
    SimulationNonControllableResult, SimulationPumpingResult, SimulationScenarioResult,
    SimulationStageResult, SimulationThermalResult,
};

/// I/O thread loop: receives scenario results, converts them, and writes
/// Parquet files. The writer must be created by the caller (on the main
/// thread, where `&System` is available) and moved in.
///
/// The `Receiver` is consumed by value because it is moved into the thread.
#[allow(clippy::needless_pass_by_value)]
pub fn drain_results(
    rx: Receiver<SimulationScenarioResult>,
    mut writer: SimulationParquetWriter,
) -> SimulationOutput {
    let mut failed: u32 = 0;
    while let Ok(scenario_result) = rx.recv() {
        let payload = convert_scenario(scenario_result);
        if let Err(e) = writer.write_scenario(payload) {
            tracing::error!("simulation write error: {e}");
            failed += 1;
        }
    }

    let mut output = writer.finalize();
    output.failed = failed;
    output
}

// ---------------------------------------------------------------------------
// Type conversions: cobre-sddp → cobre-io
//
// These functions consume the source structs by value (moved from the channel)
// and produce the I/O-layer equivalents. Clippy flags them as
// "needless_pass_by_value" because the structs only have Copy fields, but
// taking ownership is intentional: the source is dropped after conversion.
// ---------------------------------------------------------------------------

fn convert_scenario(src: SimulationScenarioResult) -> ScenarioWritePayload {
    ScenarioWritePayload {
        scenario_id: src.scenario_id,
        stages: src.stages.into_iter().map(convert_stage).collect(),
    }
}

fn convert_stage(src: SimulationStageResult) -> StageWritePayload {
    StageWritePayload {
        stage_id: src.stage_id,
        costs: src.costs.into_iter().map(convert_cost).collect(),
        hydros: src.hydros.into_iter().map(convert_hydro).collect(),
        thermals: src.thermals.into_iter().map(convert_thermal).collect(),
        exchanges: src.exchanges.into_iter().map(convert_exchange).collect(),
        buses: src.buses.into_iter().map(convert_bus).collect(),
        pumping_stations: src
            .pumping_stations
            .into_iter()
            .map(convert_pumping)
            .collect(),
        contracts: src.contracts.into_iter().map(convert_contract).collect(),
        non_controllables: src
            .non_controllables
            .into_iter()
            .map(convert_non_controllable)
            .collect(),
        inflow_lags: src
            .inflow_lags
            .into_iter()
            .map(convert_inflow_lag)
            .collect(),
        generic_violations: src
            .generic_violations
            .into_iter()
            .map(convert_generic_violation)
            .collect(),
    }
}

fn convert_cost(s: SimulationCostResult) -> CostWriteRecord {
    CostWriteRecord {
        stage_id: s.stage_id,
        block_id: s.block_id,
        total_cost: s.total_cost,
        immediate_cost: s.immediate_cost,
        future_cost: s.future_cost,
        discount_factor: s.discount_factor,
        thermal_cost: s.thermal_cost,
        contract_cost: s.contract_cost,
        deficit_cost: s.deficit_cost,
        excess_cost: s.excess_cost,
        storage_violation_cost: s.storage_violation_cost,
        filling_target_cost: s.filling_target_cost,
        hydro_violation_cost: s.hydro_violation_cost,
        inflow_penalty_cost: s.inflow_penalty_cost,
        generic_violation_cost: s.generic_violation_cost,
        spillage_cost: s.spillage_cost,
        fpha_turbined_cost: s.fpha_turbined_cost,
        curtailment_cost: s.curtailment_cost,
        exchange_cost: s.exchange_cost,
        pumping_cost: s.pumping_cost,
    }
}

fn convert_hydro(s: SimulationHydroResult) -> HydroWriteRecord {
    HydroWriteRecord {
        stage_id: s.stage_id,
        block_id: s.block_id,
        hydro_id: s.hydro_id,
        turbined_m3s: s.turbined_m3s,
        spillage_m3s: s.spillage_m3s,
        evaporation_m3s: s.evaporation_m3s,
        diverted_inflow_m3s: s.diverted_inflow_m3s,
        diverted_outflow_m3s: s.diverted_outflow_m3s,
        incremental_inflow_m3s: s.incremental_inflow_m3s,
        inflow_m3s: s.inflow_m3s,
        storage_initial_hm3: s.storage_initial_hm3,
        storage_final_hm3: s.storage_final_hm3,
        generation_mw: s.generation_mw,
        productivity_mw_per_m3s: s.productivity_mw_per_m3s,
        spillage_cost: s.spillage_cost,
        water_value_per_hm3: s.water_value_per_hm3,
        storage_binding_code: s.storage_binding_code,
        operative_state_code: s.operative_state_code,
        turbined_slack_m3s: s.turbined_slack_m3s,
        outflow_slack_below_m3s: s.outflow_slack_below_m3s,
        outflow_slack_above_m3s: s.outflow_slack_above_m3s,
        generation_slack_mw: s.generation_slack_mw,
        storage_violation_below_hm3: s.storage_violation_below_hm3,
        filling_target_violation_hm3: s.filling_target_violation_hm3,
        evaporation_violation_m3s: s.evaporation_violation_m3s,
        inflow_nonnegativity_slack_m3s: s.inflow_nonnegativity_slack_m3s,
    }
}

fn convert_thermal(s: SimulationThermalResult) -> ThermalWriteRecord {
    ThermalWriteRecord {
        stage_id: s.stage_id,
        block_id: s.block_id,
        thermal_id: s.thermal_id,
        generation_mw: s.generation_mw,
        generation_cost: s.generation_cost,
        is_gnl: s.is_gnl,
        gnl_committed_mw: s.gnl_committed_mw,
        gnl_decision_mw: s.gnl_decision_mw,
        operative_state_code: s.operative_state_code,
    }
}

fn convert_exchange(s: SimulationExchangeResult) -> ExchangeWriteRecord {
    ExchangeWriteRecord {
        stage_id: s.stage_id,
        block_id: s.block_id,
        line_id: s.line_id,
        direct_flow_mw: s.direct_flow_mw,
        reverse_flow_mw: s.reverse_flow_mw,
        exchange_cost: s.exchange_cost,
        operative_state_code: s.operative_state_code,
    }
}

fn convert_bus(s: SimulationBusResult) -> BusWriteRecord {
    BusWriteRecord {
        stage_id: s.stage_id,
        block_id: s.block_id,
        bus_id: s.bus_id,
        load_mw: s.load_mw,
        deficit_mw: s.deficit_mw,
        excess_mw: s.excess_mw,
        spot_price: s.spot_price,
    }
}

fn convert_pumping(s: SimulationPumpingResult) -> PumpingWriteRecord {
    PumpingWriteRecord {
        stage_id: s.stage_id,
        block_id: s.block_id,
        pumping_station_id: s.pumping_station_id,
        pumped_flow_m3s: s.pumped_flow_m3s,
        power_consumption_mw: s.power_consumption_mw,
        pumping_cost: s.pumping_cost,
        operative_state_code: s.operative_state_code,
    }
}

fn convert_contract(s: SimulationContractResult) -> ContractWriteRecord {
    ContractWriteRecord {
        stage_id: s.stage_id,
        block_id: s.block_id,
        contract_id: s.contract_id,
        power_mw: s.power_mw,
        price_per_mwh: s.price_per_mwh,
        total_cost: s.total_cost,
        operative_state_code: s.operative_state_code,
    }
}

fn convert_non_controllable(s: SimulationNonControllableResult) -> NonControllableWriteRecord {
    NonControllableWriteRecord {
        stage_id: s.stage_id,
        block_id: s.block_id,
        non_controllable_id: s.non_controllable_id,
        generation_mw: s.generation_mw,
        available_mw: s.available_mw,
        curtailment_mw: s.curtailment_mw,
        curtailment_cost: s.curtailment_cost,
        operative_state_code: s.operative_state_code,
    }
}

fn convert_inflow_lag(s: SimulationInflowLagResult) -> InflowLagWriteRecord {
    InflowLagWriteRecord {
        stage_id: s.stage_id,
        hydro_id: s.hydro_id,
        lag_index: s.lag_index,
        inflow_m3s: s.inflow_m3s,
    }
}

fn convert_generic_violation(s: SimulationGenericViolationResult) -> GenericViolationWriteRecord {
    GenericViolationWriteRecord {
        stage_id: s.stage_id,
        block_id: s.block_id,
        constraint_id: s.constraint_id,
        slack_value: s.slack_value,
        slack_cost: s.slack_cost,
    }
}
