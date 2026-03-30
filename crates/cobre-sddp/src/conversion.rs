//! Field-for-field conversions from `Simulation*Result` to write-payload types.

use cobre_io::output::simulation_writer::{
    BusWriteRecord, ContractWriteRecord, CostWriteRecord, ExchangeWriteRecord,
    GenericViolationWriteRecord, HydroWriteRecord, InflowLagWriteRecord,
    NonControllableWriteRecord, PumpingWriteRecord, ScenarioWritePayload, StageWritePayload,
    ThermalWriteRecord,
};

use crate::{
    SimulationBusResult, SimulationContractResult, SimulationCostResult, SimulationExchangeResult,
    SimulationGenericViolationResult, SimulationHydroResult, SimulationInflowLagResult,
    SimulationNonControllableResult, SimulationPumpingResult, SimulationScenarioResult,
    SimulationStageResult, SimulationThermalResult,
};

impl From<SimulationCostResult> for CostWriteRecord {
    fn from(s: SimulationCostResult) -> Self {
        Self {
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
            outflow_violation_below_cost: s.outflow_violation_below_cost,
            outflow_violation_above_cost: s.outflow_violation_above_cost,
            turbined_violation_cost: s.turbined_violation_cost,
            generation_violation_cost: s.generation_violation_cost,
            evaporation_violation_cost: s.evaporation_violation_cost,
            withdrawal_violation_cost: s.withdrawal_violation_cost,
            inflow_penalty_cost: s.inflow_penalty_cost,
            generic_violation_cost: s.generic_violation_cost,
            spillage_cost: s.spillage_cost,
            fpha_turbined_cost: s.fpha_turbined_cost,
            curtailment_cost: s.curtailment_cost,
            exchange_cost: s.exchange_cost,
            pumping_cost: s.pumping_cost,
        }
    }
}

impl From<SimulationHydroResult> for HydroWriteRecord {
    fn from(s: SimulationHydroResult) -> Self {
        Self {
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
            water_withdrawal_violation_m3s: s.water_withdrawal_violation_m3s,
        }
    }
}

impl From<SimulationThermalResult> for ThermalWriteRecord {
    fn from(s: SimulationThermalResult) -> Self {
        Self {
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
}

impl From<SimulationExchangeResult> for ExchangeWriteRecord {
    fn from(s: SimulationExchangeResult) -> Self {
        Self {
            stage_id: s.stage_id,
            block_id: s.block_id,
            line_id: s.line_id,
            direct_flow_mw: s.direct_flow_mw,
            reverse_flow_mw: s.reverse_flow_mw,
            exchange_cost: s.exchange_cost,
            operative_state_code: s.operative_state_code,
        }
    }
}

impl From<SimulationBusResult> for BusWriteRecord {
    fn from(s: SimulationBusResult) -> Self {
        Self {
            stage_id: s.stage_id,
            block_id: s.block_id,
            bus_id: s.bus_id,
            load_mw: s.load_mw,
            deficit_mw: s.deficit_mw,
            excess_mw: s.excess_mw,
            spot_price: s.spot_price,
        }
    }
}

impl From<SimulationPumpingResult> for PumpingWriteRecord {
    fn from(s: SimulationPumpingResult) -> Self {
        Self {
            stage_id: s.stage_id,
            block_id: s.block_id,
            pumping_station_id: s.pumping_station_id,
            pumped_flow_m3s: s.pumped_flow_m3s,
            power_consumption_mw: s.power_consumption_mw,
            pumping_cost: s.pumping_cost,
            operative_state_code: s.operative_state_code,
        }
    }
}

impl From<SimulationContractResult> for ContractWriteRecord {
    fn from(s: SimulationContractResult) -> Self {
        Self {
            stage_id: s.stage_id,
            block_id: s.block_id,
            contract_id: s.contract_id,
            power_mw: s.power_mw,
            price_per_mwh: s.price_per_mwh,
            total_cost: s.total_cost,
            operative_state_code: s.operative_state_code,
        }
    }
}

impl From<SimulationNonControllableResult> for NonControllableWriteRecord {
    fn from(s: SimulationNonControllableResult) -> Self {
        Self {
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
}

impl From<SimulationInflowLagResult> for InflowLagWriteRecord {
    fn from(s: SimulationInflowLagResult) -> Self {
        Self {
            stage_id: s.stage_id,
            hydro_id: s.hydro_id,
            lag_index: s.lag_index,
            inflow_m3s: s.inflow_m3s,
        }
    }
}

impl From<SimulationGenericViolationResult> for GenericViolationWriteRecord {
    fn from(s: SimulationGenericViolationResult) -> Self {
        Self {
            stage_id: s.stage_id,
            block_id: s.block_id,
            constraint_id: s.constraint_id,
            slack_value: s.slack_value,
            slack_cost: s.slack_cost,
        }
    }
}

impl From<SimulationStageResult> for StageWritePayload {
    fn from(src: SimulationStageResult) -> Self {
        Self {
            stage_id: src.stage_id,
            costs: src.costs.into_iter().map(Into::into).collect(),
            hydros: src.hydros.into_iter().map(Into::into).collect(),
            thermals: src.thermals.into_iter().map(Into::into).collect(),
            exchanges: src.exchanges.into_iter().map(Into::into).collect(),
            buses: src.buses.into_iter().map(Into::into).collect(),
            pumping_stations: src.pumping_stations.into_iter().map(Into::into).collect(),
            contracts: src.contracts.into_iter().map(Into::into).collect(),
            non_controllables: src.non_controllables.into_iter().map(Into::into).collect(),
            inflow_lags: src.inflow_lags.into_iter().map(Into::into).collect(),
            generic_violations: src.generic_violations.into_iter().map(Into::into).collect(),
        }
    }
}

impl From<SimulationScenarioResult> for ScenarioWritePayload {
    fn from(src: SimulationScenarioResult) -> Self {
        Self {
            scenario_id: src.scenario_id,
            stages: src.stages.into_iter().map(Into::into).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ScenarioCategoryCosts;
    use cobre_io::output::simulation_writer::ScenarioWritePayload;

    fn make_cost(stage_id: u32, block_id: u32) -> SimulationCostResult {
        SimulationCostResult {
            stage_id,
            block_id: Some(block_id),
            total_cost: 1.0,
            immediate_cost: 2.0,
            future_cost: 3.0,
            discount_factor: 0.95,
            thermal_cost: 4.0,
            contract_cost: 5.0,
            deficit_cost: 6.0,
            excess_cost: 7.0,
            storage_violation_cost: 8.0,
            filling_target_cost: 9.0,
            hydro_violation_cost: 10.0,
            outflow_violation_below_cost: 0.0,
            outflow_violation_above_cost: 0.0,
            turbined_violation_cost: 0.0,
            generation_violation_cost: 0.0,
            evaporation_violation_cost: 0.0,
            withdrawal_violation_cost: 0.0,
            inflow_penalty_cost: 11.0,
            generic_violation_cost: 12.0,
            spillage_cost: 13.0,
            fpha_turbined_cost: 14.0,
            curtailment_cost: 15.0,
            exchange_cost: 16.0,
            pumping_cost: 17.0,
        }
    }

    fn make_hydro(stage_id: u32, block_id: u32) -> SimulationHydroResult {
        SimulationHydroResult {
            stage_id,
            block_id: Some(block_id),
            hydro_id: 1,
            turbined_m3s: 100.0,
            spillage_m3s: 10.0,
            evaporation_m3s: Some(1.0),
            diverted_inflow_m3s: Some(2.0),
            diverted_outflow_m3s: Some(3.0),
            incremental_inflow_m3s: 50.0,
            inflow_m3s: 55.0,
            storage_initial_hm3: 500.0,
            storage_final_hm3: 490.0,
            generation_mw: 200.0,
            productivity_mw_per_m3s: Some(2.0),
            spillage_cost: 0.0,
            water_value_per_hm3: 30.0,
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
            water_withdrawal_violation_m3s: 0.0,
        }
    }

    fn make_thermal(stage_id: u32, block_id: u32) -> SimulationThermalResult {
        SimulationThermalResult {
            stage_id,
            block_id: Some(block_id),
            thermal_id: 1,
            generation_mw: 150.0,
            generation_cost: 300.0,
            is_gnl: false,
            gnl_committed_mw: None,
            gnl_decision_mw: None,
            operative_state_code: 1,
        }
    }

    fn make_exchange(stage_id: u32, block_id: u32) -> SimulationExchangeResult {
        SimulationExchangeResult {
            stage_id,
            block_id: Some(block_id),
            line_id: 1,
            direct_flow_mw: 50.0,
            reverse_flow_mw: 0.0,
            exchange_cost: 5.0,
            operative_state_code: 1,
        }
    }

    fn make_bus(stage_id: u32, block_id: u32) -> SimulationBusResult {
        SimulationBusResult {
            stage_id,
            block_id: Some(block_id),
            bus_id: 1,
            load_mw: 400.0,
            deficit_mw: 0.0,
            excess_mw: 0.0,
            spot_price: 75.0,
        }
    }

    fn make_pumping(stage_id: u32, block_id: u32) -> SimulationPumpingResult {
        SimulationPumpingResult {
            stage_id,
            block_id: Some(block_id),
            pumping_station_id: 1,
            pumped_flow_m3s: 20.0,
            power_consumption_mw: 10.0,
            pumping_cost: 50.0,
            operative_state_code: 1,
        }
    }

    fn make_contract(stage_id: u32, block_id: u32) -> SimulationContractResult {
        SimulationContractResult {
            stage_id,
            block_id: Some(block_id),
            contract_id: 1,
            power_mw: 100.0,
            price_per_mwh: 80.0,
            total_cost: 192.0,
            operative_state_code: 1,
        }
    }

    fn make_non_controllable(stage_id: u32, block_id: u32) -> SimulationNonControllableResult {
        SimulationNonControllableResult {
            stage_id,
            block_id: Some(block_id),
            non_controllable_id: 1,
            generation_mw: 60.0,
            available_mw: 70.0,
            curtailment_mw: 10.0,
            curtailment_cost: 100.0,
            operative_state_code: 1,
        }
    }

    fn make_inflow_lag(stage_id: u32) -> SimulationInflowLagResult {
        SimulationInflowLagResult {
            stage_id,
            hydro_id: 1,
            lag_index: 0,
            inflow_m3s: 45.0,
        }
    }

    fn make_generic_violation(stage_id: u32, block_id: u32) -> SimulationGenericViolationResult {
        SimulationGenericViolationResult {
            stage_id,
            block_id: Some(block_id),
            constraint_id: 1,
            slack_value: 0.5,
            slack_cost: 1000.0,
        }
    }

    fn make_stage(stage_id: u32) -> SimulationStageResult {
        SimulationStageResult {
            stage_id,
            costs: vec![make_cost(stage_id, 0)],
            hydros: vec![make_hydro(stage_id, 0)],
            thermals: vec![make_thermal(stage_id, 0)],
            exchanges: vec![make_exchange(stage_id, 0)],
            buses: vec![make_bus(stage_id, 0)],
            pumping_stations: vec![make_pumping(stage_id, 0)],
            contracts: vec![make_contract(stage_id, 0)],
            non_controllables: vec![make_non_controllable(stage_id, 0)],
            inflow_lags: vec![make_inflow_lag(stage_id)],
            generic_violations: vec![make_generic_violation(stage_id, 0)],
        }
    }

    fn make_category_costs() -> ScenarioCategoryCosts {
        ScenarioCategoryCosts {
            resource_cost: 100.0,
            recourse_cost: 50.0,
            violation_cost: 10.0,
            regularization_cost: 5.0,
            imputed_cost: 2.0,
        }
    }

    #[test]
    fn convert_scenario_result_to_write_payload_round_trip() {
        let scenario = SimulationScenarioResult {
            scenario_id: 7,
            total_cost: 167.0,
            per_category_costs: make_category_costs(),
            stages: vec![make_stage(0), make_stage(1)],
        };

        let payload = ScenarioWritePayload::from(scenario);

        assert_eq!(payload.scenario_id, 7);
        assert_eq!(payload.stages.len(), 2);

        let stage0 = &payload.stages[0];
        assert_eq!(stage0.stage_id, 0);
        assert_eq!(stage0.costs[0].total_cost, 1.0);
        assert_eq!(stage0.costs[0].discount_factor, 0.95);
        assert_eq!(stage0.hydros[0].turbined_m3s, 100.0);
        assert_eq!(stage0.hydros[0].storage_initial_hm3, 500.0);
        assert_eq!(stage0.thermals[0].generation_mw, 150.0);
        assert_eq!(stage0.exchanges[0].direct_flow_mw, 50.0);
        assert_eq!(stage0.buses[0].spot_price, 75.0);
        assert_eq!(stage0.pumping_stations[0].pumped_flow_m3s, 20.0);
        assert_eq!(stage0.contracts[0].price_per_mwh, 80.0);
        assert_eq!(stage0.non_controllables[0].curtailment_mw, 10.0);
        assert_eq!(stage0.inflow_lags[0].inflow_m3s, 45.0);
        assert_eq!(stage0.generic_violations[0].slack_cost, 1000.0);

        let stage1 = &payload.stages[1];
        assert_eq!(stage1.stage_id, 1);
    }

    #[test]
    fn convert_stage_result_preserves_all_entity_types() {
        let stage = make_stage(3);
        let payload = StageWritePayload::from(stage);

        assert_eq!(payload.stage_id, 3);
        assert!(!payload.costs.is_empty(), "costs must be non-empty");
        assert!(!payload.hydros.is_empty(), "hydros must be non-empty");
        assert!(!payload.thermals.is_empty(), "thermals must be non-empty");
        assert!(!payload.exchanges.is_empty(), "exchanges must be non-empty");
        assert!(!payload.buses.is_empty(), "buses must be non-empty");
        assert!(
            !payload.pumping_stations.is_empty(),
            "pumping_stations must be non-empty"
        );
        assert!(!payload.contracts.is_empty(), "contracts must be non-empty");
        assert!(
            !payload.non_controllables.is_empty(),
            "non_controllables must be non-empty"
        );
        assert!(
            !payload.inflow_lags.is_empty(),
            "inflow_lags must be non-empty"
        );
        assert!(
            !payload.generic_violations.is_empty(),
            "generic_violations must be non-empty"
        );

        // Spot-check field values for each entity type
        assert_eq!(payload.costs[0].stage_id, 3);
        assert_eq!(payload.hydros[0].hydro_id, 1);
        assert_eq!(payload.thermals[0].thermal_id, 1);
        assert_eq!(payload.exchanges[0].line_id, 1);
        assert_eq!(payload.buses[0].bus_id, 1);
        assert_eq!(payload.pumping_stations[0].pumping_station_id, 1);
        assert_eq!(payload.contracts[0].contract_id, 1);
        assert_eq!(payload.non_controllables[0].non_controllable_id, 1);
        assert_eq!(payload.inflow_lags[0].lag_index, 0);
        assert_eq!(payload.generic_violations[0].constraint_id, 1);
    }
}
