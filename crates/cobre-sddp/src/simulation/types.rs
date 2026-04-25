//! Simulation result types produced by the SDDP simulation forward pass.
//!
//! These types form the data contract between the simulation pipeline and the
//! output writer. The simulation loop produces one [`SimulationScenarioResult`]
//! per completed scenario; each scenario result is sent through a bounded
//! channel to a background I/O thread that writes Parquet output files.
//!
//! ## Layout
//!
//! The types use a nested design: per-entity-type [`Vec`]s grouped by stage
//! inside [`SimulationStageResult`], then stages grouped by scenario inside
//! [`SimulationScenarioResult`]. The output writer is responsible for the
//! columnar transpose when writing Parquet.
//!
//! ## Derived columns
//!
//! Fields present in the Parquet output schemas but absent from these structs
//! are derived by the output writer from the fields present here together with
//! block duration metadata. Examples: `generation_mwh`, `net_flow_mw`,
//! `losses_mw`, `outflow_m3s`. See simulation-architecture.md SS3.4 for the
//! complete list.

/// Cost breakdown for one (stage, block) pair.
///
/// Corresponds to one row in the costs output schema
/// (output-schemas.md SS5.1). Contains both aggregate totals and
/// per-category breakdowns used for cost statistics (SS4.2).
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct SimulationCostResult {
    /// Stage index (0-based).
    pub stage_id: u32,
    /// Block index within the stage, or [`None`] for stage-level aggregates.
    pub block_id: Option<u32>,
    /// Total discounted stage cost: `immediate_cost + future_cost`.
    pub total_cost: f64,
    /// Undiscounted stage immediate cost (before applying `discount_factor`).
    pub immediate_cost: f64,
    /// Future cost function (theta variable value) at this stage.
    pub future_cost: f64,
    /// Cumulative discount factor at this stage for present-value reporting.
    ///
    /// `D_0 = 1.0`, `D_t = D_{t-1} * d_{t-1}` where `d_t` is the one-step
    /// discount factor for the transition departing stage `t`. The present
    /// value of `immediate_cost` is `discount_factor * immediate_cost`.
    /// When `annual_discount_rate == 0.0`, this field is `1.0` for all stages.
    pub discount_factor: f64,
    // Resource costs
    /// Cost attributed to thermal generation dispatch.
    pub thermal_cost: f64,
    /// Cost attributed to contract energy delivery.
    pub contract_cost: f64,
    /// Cost of load deficit (emergency energy).
    pub deficit_cost: f64,
    /// Cost of load excess (excess energy penalty).
    pub excess_cost: f64,
    /// Cost of reservoir storage bound violations.
    pub storage_violation_cost: f64,
    /// Cost of filling target violations.
    pub filling_target_cost: f64,
    /// Cost of hydro operational constraint violations.
    pub hydro_violation_cost: f64,
    /// Cost of minimum outflow violations: sum of `outflow_below_slack` * penalty.
    pub outflow_violation_below_cost: f64,
    /// Cost of maximum outflow violations: sum of `outflow_above_slack` * penalty.
    pub outflow_violation_above_cost: f64,
    /// Cost of minimum turbining violations: sum of `turbine_below_slack` * penalty.
    pub turbined_violation_cost: f64,
    /// Cost of minimum generation violations: sum of `generation_below_slack` * penalty.
    pub generation_violation_cost: f64,
    /// Cost of evaporation constraint violations.
    pub evaporation_violation_cost: f64,
    /// Cost of water withdrawal constraint violations.
    pub withdrawal_violation_cost: f64,
    /// Cost of inflow non-negativity constraint violations.
    pub inflow_penalty_cost: f64,
    /// Cost of generic constraint violations.
    pub generic_violation_cost: f64,
    /// Regularization cost for reservoir spillage.
    pub spillage_cost: f64,
    /// Regularization cost for FPHA turbining.
    pub fpha_turbined_cost: f64,
    /// Regularization cost for non-controllable source curtailment.
    pub curtailment_cost: f64,
    /// Regularization cost for transmission exchange.
    pub exchange_cost: f64,
    /// Imputed cost for pumping station operation.
    pub pumping_cost: f64,
}

/// Hydro plant result for one (stage, block, hydro) tuple.
///
/// Corresponds to one row in the hydros output schema
/// (output-schemas.md SS5.2). Derived columns (`generation_mwh`,
/// `outflow_m3s`) are computed by the output writer.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct SimulationHydroResult {
    /// Stage index (0-based).
    pub stage_id: u32,
    /// Block index within the stage.
    pub block_id: Option<u32>,
    /// Hydro plant entity ID.
    pub hydro_id: i32,
    /// Turbined flow in m³/s.
    pub turbined_m3s: f64,
    /// Spilled flow in m³/s.
    pub spillage_m3s: f64,
    /// Evaporation loss in m³/s, or [`None`] if evaporation is not modeled.
    pub evaporation_m3s: Option<f64>,
    /// Diverted inflow from upstream diversion in m³/s, or [`None`] if no
    /// diversion exists.
    pub diverted_inflow_m3s: Option<f64>,
    /// Diverted outflow to downstream diversion in m³/s, or [`None`] if no
    /// diversion exists.
    pub diverted_outflow_m3s: Option<f64>,
    /// Incremental (local) natural inflow in m³/s.
    pub incremental_inflow_m3s: f64,
    /// Total inflow to the reservoir in m³/s (incremental + upstream).
    pub inflow_m3s: f64,
    /// Reservoir storage at the start of the block in hm³.
    pub storage_initial_hm3: f64,
    /// Reservoir storage at the end of the block in hm³.
    pub storage_final_hm3: f64,
    /// Active power generation in MW.
    pub generation_mw: f64,
    /// Plant productivity in MW/(m³/s), or [`None`] if using a tabular
    /// production function.
    pub productivity_mw_per_m3s: Option<f64>,
    /// Regularization cost for spillage at this plant.
    pub spillage_cost: f64,
    /// Water value (dual of the storage balance constraint) in cost/hm³.
    pub water_value_per_hm3: f64,
    /// Storage binding code: indicates which storage bound is active.
    /// Encoded as `i8` matching the Parquet schema.
    pub storage_binding_code: i8,
    /// Operative state code for this hydro plant at this block.
    pub operative_state_code: i8,
    // Violation slacks
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
    /// Over-evaporation violation in m³/s (evaporated more than target).
    pub evaporation_violation_pos_m3s: f64,
    /// Under-evaporation violation in m³/s (evaporated less than target).
    pub evaporation_violation_neg_m3s: f64,
    /// Inflow non-negativity constraint slack in m³/s.
    pub inflow_nonnegativity_slack_m3s: f64,
    /// Over-withdrawal violation in m³/s (withdrew more than target).
    /// Zero when no withdrawal is modeled.
    pub water_withdrawal_violation_pos_m3s: f64,
    /// Under-withdrawal violation in m³/s (withdrew less than target).
    /// Zero when no withdrawal is modeled or withdrawal is fully sustained.
    pub water_withdrawal_violation_neg_m3s: f64,
}

/// Thermal unit result for one (stage, block, thermal) tuple.
///
/// Corresponds to one row in the thermals output schema
/// (output-schemas.md SS5.3). The derived column `generation_mwh`
/// is computed by the output writer.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct SimulationThermalResult {
    /// Stage index (0-based).
    pub stage_id: u32,
    /// Block index within the stage.
    pub block_id: Option<u32>,
    /// Thermal unit entity ID.
    pub thermal_id: i32,
    /// Active power generation in MW.
    pub generation_mw: f64,
    /// Variable generation cost for this dispatch.
    pub generation_cost: f64,
    /// Whether this unit participates in a GNL (Gas Natural Liquefied) model.
    pub is_gnl: bool,
    /// Committed capacity under the GNL model in MW, or [`None`] if
    /// `is_gnl` is false.
    pub gnl_committed_mw: Option<f64>,
    /// Decision (contracted) capacity under the GNL model in MW, or [`None`]
    /// if `is_gnl` is false.
    pub gnl_decision_mw: Option<f64>,
    /// Operative state code for this thermal unit at this block.
    pub operative_state_code: i8,
}

/// Exchange (transmission line) result for one (stage, block, line) tuple.
///
/// Corresponds to one row in the exchanges output schema
/// (output-schemas.md SS5.4). Derived columns (`net_flow_mw`,
/// `losses_mw`, and all `MWh` energy columns) are computed by the output
/// writer.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct SimulationExchangeResult {
    /// Stage index (0-based).
    pub stage_id: u32,
    /// Block index within the stage.
    pub block_id: Option<u32>,
    /// Transmission line entity ID.
    pub line_id: i32,
    /// Power flow in the direct (forward) direction in MW.
    pub direct_flow_mw: f64,
    /// Power flow in the reverse direction in MW.
    pub reverse_flow_mw: f64,
    /// Exchange regularization cost for this line.
    pub exchange_cost: f64,
    /// Operative state code for this line at this block.
    pub operative_state_code: i8,
}

/// Bus result for one (stage, block, bus) tuple.
///
/// Corresponds to one row in the buses output schema
/// (output-schemas.md SS5.5). Derived columns (`load_mwh`,
/// `deficit_mwh`, `excess_mwh`) are computed by the output writer.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct SimulationBusResult {
    /// Stage index (0-based).
    pub stage_id: u32,
    /// Block index within the stage.
    pub block_id: Option<u32>,
    /// Bus entity ID.
    pub bus_id: i32,
    /// Total demand (load) at this bus in MW.
    pub load_mw: f64,
    /// Load deficit (unserved demand) at this bus in MW.
    pub deficit_mw: f64,
    /// Load excess at this bus in MW.
    pub excess_mw: f64,
    /// Marginal cost of energy (spot price) at this bus in cost/MWh.
    pub spot_price: f64,
}

/// Pumping station result for one (stage, block, station) tuple.
///
/// Corresponds to one row in the `pumping_stations` output schema
/// (output-schemas.md SS5.6). Derived columns (`pumped_volume_hm3`,
/// `energy_consumption_mwh`) are computed by the output writer.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct SimulationPumpingResult {
    /// Stage index (0-based).
    pub stage_id: u32,
    /// Block index within the stage.
    pub block_id: Option<u32>,
    /// Pumping station entity ID.
    pub pumping_station_id: i32,
    /// Pumped flow rate in m³/s.
    pub pumped_flow_m3s: f64,
    /// Active power consumed by pumping in MW.
    pub power_consumption_mw: f64,
    /// Imputed pumping cost.
    pub pumping_cost: f64,
    /// Operative state code for this station at this block.
    pub operative_state_code: i8,
}

/// Contract result for one (stage, block, contract) tuple.
///
/// Corresponds to one row in the contracts output schema
/// (output-schemas.md SS5.7). The derived column `energy_mwh` is
/// computed by the output writer.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct SimulationContractResult {
    /// Stage index (0-based).
    pub stage_id: u32,
    /// Block index within the stage.
    pub block_id: Option<u32>,
    /// Contract entity ID.
    pub contract_id: i32,
    /// Contracted power in MW.
    pub power_mw: f64,
    /// Contract price in cost/MWh.
    pub price_per_mwh: f64,
    /// Total cost for this contract at this block.
    pub total_cost: f64,
    /// Operative state code for this contract at this block.
    pub operative_state_code: i8,
}

/// Non-controllable source result for one (stage, block, source) tuple.
///
/// Corresponds to one row in the `non_controllables` output schema
/// (output-schemas.md SS5.8). Derived columns (`generation_mwh`,
/// `curtailment_mwh`) are computed by the output writer.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct SimulationNonControllableResult {
    /// Stage index (0-based).
    pub stage_id: u32,
    /// Block index within the stage.
    pub block_id: Option<u32>,
    /// Non-controllable source entity ID.
    pub non_controllable_id: i32,
    /// Active power injected into the grid in MW.
    pub generation_mw: f64,
    /// Maximum available power from this source in MW.
    pub available_mw: f64,
    /// Curtailed power (available minus injected) in MW.
    pub curtailment_mw: f64,
    /// Curtailment regularization cost.
    pub curtailment_cost: f64,
    /// Operative state code for this source at this block.
    pub operative_state_code: i8,
}

/// Inflow lag state for one (stage, hydro, `lag_index`) tuple.
///
/// Corresponds to one row in the `inflow_lags` output schema
/// (output-schemas.md SS5.10). Only populated for hydro plants whose
/// PAR(p) model has AR order > 0.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct SimulationInflowLagResult {
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
/// Corresponds to one row in the violations/generic output schema
/// (output-schemas.md SS5.11). Only entries with non-zero slack values
/// are included.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct SimulationGenericViolationResult {
    /// Stage index (0-based).
    pub stage_id: u32,
    /// Block index within the stage.
    pub block_id: Option<u32>,
    /// Generic constraint entity ID.
    pub constraint_id: i32,
    /// Violation slack value (non-negative).
    pub slack_value: f64,
    /// Cost incurred for this violation.
    pub slack_cost: f64,
}

/// All simulation results for a single stage within one scenario.
///
/// The simulation loop produces one [`SimulationStageResult`] per stage per
/// scenario. Each per-entity-type [`Vec`] holds one entry per (block, entity)
/// pair within the stage. Entity types that are absent from the system or that
/// produce no violations result in empty [`Vec`]s.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct SimulationStageResult {
    /// Stage index (0-based).
    pub stage_id: u32,
    /// Cost breakdown results for this stage (one entry per block).
    pub costs: Vec<SimulationCostResult>,
    /// Hydro plant results for this stage.
    pub hydros: Vec<SimulationHydroResult>,
    /// Thermal unit results for this stage.
    pub thermals: Vec<SimulationThermalResult>,
    /// Transmission line (exchange) results for this stage.
    pub exchanges: Vec<SimulationExchangeResult>,
    /// Bus results for this stage.
    pub buses: Vec<SimulationBusResult>,
    /// Pumping station results for this stage.
    /// Empty if no pumping stations exist in the system.
    pub pumping_stations: Vec<SimulationPumpingResult>,
    /// Contract results for this stage.
    /// Empty if no contracts exist in the system.
    pub contracts: Vec<SimulationContractResult>,
    /// Non-controllable source results for this stage.
    /// Empty if no non-controllable sources exist in the system.
    pub non_controllables: Vec<SimulationNonControllableResult>,
    /// Inflow lag state records for this stage.
    /// Empty if no hydros have AR order > 0.
    pub inflow_lags: Vec<SimulationInflowLagResult>,
    /// Generic constraint violation records for this stage.
    /// Empty if no generic constraints exist or no violations occurred.
    /// Only non-zero violations are included.
    pub generic_violations: Vec<SimulationGenericViolationResult>,
}

/// Per-category cost totals for one scenario, summed across all stages.
///
/// Matches the category breakdown in SS4.2 and is retained in the compact
/// cost buffer even after per-stage detail is streamed to the output writer.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ScenarioCategoryCosts {
    /// Sum of thermal and contract costs: `thermal_cost + contract_cost`.
    pub resource_cost: f64,
    /// Sum of deficit and excess costs: `deficit_cost + excess_cost`.
    pub recourse_cost: f64,
    /// Sum of all violation costs: `storage_violation_cost +
    /// filling_target_cost + hydro_violation_cost + inflow_penalty_cost +
    /// generic_violation_cost`.
    pub violation_cost: f64,
    /// Sum of regularization costs: `spillage_cost + fpha_turbined_cost +
    /// curtailment_cost + exchange_cost`.
    pub regularization_cost: f64,
    /// Imputed pumping cost: `pumping_cost`.
    pub imputed_cost: f64,
}

/// Complete simulation result for one scenario.
///
/// This is the payload type of the bounded channel connecting simulation
/// threads to the background I/O thread (SS6.1).
///
/// # Send bound
///
/// [`SimulationScenarioResult`] implements `Send` because it is transferred
/// across a thread boundary: the simulation thread that produces it sends it
/// through the channel to the dedicated I/O thread. All constituent types are
/// `Send`-safe (plain data, no `Rc`, no raw pointers, no non-`Send` interior
/// mutability).
///
/// # Memory lifetime
///
/// Each instance is produced by a simulation thread, sent through the channel,
/// consumed by the I/O thread for Parquet writing, and then dropped. At most
/// `channel_capacity` instances exist simultaneously (bounded by channel
/// backpressure). See SS3.3.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct SimulationScenarioResult {
    /// 0-based scenario identifier, unique across all MPI ranks.
    /// Determines the Hive partition path:
    /// `{entity}/scenario_id={scenario_id:04d}/data.parquet`.
    pub scenario_id: u32,

    /// Total discounted cost for this scenario, summed across all stages.
    /// Computed as the sum over stages of the cumulative discount factor
    /// times the stage immediate cost.
    pub total_cost: f64,

    /// Per-category cost components for this scenario, summed across all
    /// stages. Used for per-category statistics (SS4.2) and retained in the
    /// compact cost buffer (SS3.3) even after per-stage detail is streamed to
    /// the output writer.
    pub per_category_costs: ScenarioCategoryCosts,

    /// Per-stage detailed results. Present when the output detail level
    /// (SS6.2) is Stage-level or Full. An empty [`Vec`] when detail level is
    /// Summary — in that case only `scenario_id`, `total_cost`, and
    /// `per_category_costs` are populated.
    pub stages: Vec<SimulationStageResult>,
}

/// Aggregate simulation statistics computed after all scenarios complete.
///
/// Produced by MPI aggregation on rank 0 (simulation-architecture.md SS4.4)
/// and returned as the `Ok` value of `fn simulate()`.
///
/// On non-rank-0 processes, `mean_cost`, `std_cost`, `cvar`, and
/// `category_stats` reflect only locally computed partial data; the
/// authoritative values are on rank 0 (SS4.4).
#[derive(Debug)]
pub struct SimulationSummary {
    /// Mean total cost across all scenarios: `mean_cost = (1/S) * sum(C_s for s in 1..=S)`.
    pub mean_cost: f64,
    /// Sample standard deviation of total cost: `std_cost = sqrt((1/(S-1)) * sum((C_s - mean_cost)^2 for s in 1..=S))`.
    pub std_cost: f64,
    /// Minimum total cost across all scenarios.
    pub min_cost: f64,
    /// Maximum total cost across all scenarios.
    pub max_cost: f64,
    /// `CVaR` (Conditional Value-at-Risk) at the configured confidence level `cvar_alpha`.
    /// Mean of the worst `(1 - cvar_alpha)` fraction of scenario costs. See simulation-architecture.md SS4.1.
    pub cvar: f64,
    /// Confidence level used for `CVaR` computation. Must be in `(0, 1)`.
    pub cvar_alpha: f64,
    /// Per-category cost statistics (mean, max, frequency) for each of the five cost categories.
    pub category_stats: Vec<CategoryCostStats>,
    /// Fraction of scenarios with at least one stage having deficit > 0.
    pub deficit_frequency: f64,
    /// Total deficit energy (`MWh`) summed across all scenarios and stages.
    pub total_deficit_mwh: f64,
    /// Total spillage energy (`MWh`) summed across all scenarios and stages.
    pub total_spillage_mwh: f64,
    /// Number of scenarios simulated (across all ranks).
    pub n_scenarios: u32,
}

/// Per-category cost statistics for one cost category (SS4.2).
///
/// Each of the five cost categories (resource, recourse, violation,
/// regularization, imputed) produces one `CategoryCostStats` entry in
/// [`SimulationSummary::category_stats`].
#[derive(Debug)]
pub struct CategoryCostStats {
    /// Category name. Matches the SS4.2 table:
    /// `"resource"`, `"recourse"`, `"violation"`, `"regularization"`,
    /// `"imputed"`.
    pub category: String,

    /// Mean cost for this category across all scenarios.
    pub mean: f64,

    /// Maximum cost for this category across all scenarios.
    pub max: f64,

    /// Fraction of scenarios where the category cost is non-zero.
    ///
    /// Particularly relevant for deficit (recourse) and constraint
    /// violations.
    pub frequency: f64,
}

const _: fn() = || {
    fn assert_send<T: Send>() {}
    assert_send::<SimulationScenarioResult>();
};

#[cfg(test)]
mod tests {
    use super::{
        CategoryCostStats, ScenarioCategoryCosts, SimulationBusResult, SimulationContractResult,
        SimulationCostResult, SimulationExchangeResult, SimulationGenericViolationResult,
        SimulationHydroResult, SimulationInflowLagResult, SimulationNonControllableResult,
        SimulationPumpingResult, SimulationScenarioResult, SimulationStageResult,
        SimulationSummary, SimulationThermalResult,
    };

    #[test]
    fn cost_result_construction_all_fields() {
        let r = SimulationCostResult {
            stage_id: 0,
            block_id: Some(0),
            total_cost: 1000.0,
            immediate_cost: 800.0,
            future_cost: 200.0,
            discount_factor: 0.95,
            thermal_cost: 500.0,
            contract_cost: 100.0,
            deficit_cost: 50.0,
            excess_cost: 10.0,
            storage_violation_cost: 20.0,
            filling_target_cost: 30.0,
            hydro_violation_cost: 5.0,
            outflow_violation_below_cost: 0.0,
            outflow_violation_above_cost: 0.0,
            turbined_violation_cost: 0.0,
            generation_violation_cost: 0.0,
            evaporation_violation_cost: 0.0,
            withdrawal_violation_cost: 0.0,
            inflow_penalty_cost: 3.0,
            generic_violation_cost: 2.0,
            spillage_cost: 1.0,
            fpha_turbined_cost: 4.0,
            curtailment_cost: 7.0,
            exchange_cost: 8.0,
            pumping_cost: 60.0,
        };

        assert_eq!(r.stage_id, 0);
        assert_eq!(r.block_id, Some(0));
        assert_eq!(r.total_cost, 1000.0);
        assert_eq!(r.immediate_cost, 800.0);
        assert_eq!(r.future_cost, 200.0);
        assert_eq!(r.discount_factor, 0.95);
        assert_eq!(r.thermal_cost, 500.0);
        assert_eq!(r.contract_cost, 100.0);
        assert_eq!(r.deficit_cost, 50.0);
        assert_eq!(r.excess_cost, 10.0);
        assert_eq!(r.storage_violation_cost, 20.0);
        assert_eq!(r.filling_target_cost, 30.0);
        assert_eq!(r.hydro_violation_cost, 5.0);
        assert_eq!(r.inflow_penalty_cost, 3.0);
        assert_eq!(r.generic_violation_cost, 2.0);
        assert_eq!(r.spillage_cost, 1.0);
        assert_eq!(r.fpha_turbined_cost, 4.0);
        assert_eq!(r.curtailment_cost, 7.0);
        assert_eq!(r.exchange_cost, 8.0);
        assert_eq!(r.pumping_cost, 60.0);
    }

    #[test]
    fn hydro_result_optional_fields() {
        let r = SimulationHydroResult {
            stage_id: 1,
            block_id: Some(0),
            hydro_id: 5,
            turbined_m3s: 100.0,
            spillage_m3s: 0.0,
            evaporation_m3s: None,
            diverted_inflow_m3s: None,
            diverted_outflow_m3s: None,
            incremental_inflow_m3s: 200.0,
            inflow_m3s: 200.0,
            storage_initial_hm3: 500.0,
            storage_final_hm3: 480.0,
            generation_mw: 50.0,
            productivity_mw_per_m3s: Some(0.5),
            spillage_cost: 0.0,
            water_value_per_hm3: 10.0,
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
        };

        assert_eq!(r.hydro_id, 5);
        assert_eq!(r.turbined_m3s, 100.0);
        assert_eq!(r.evaporation_m3s, None);
        assert_eq!(r.diverted_inflow_m3s, None);
        assert_eq!(r.diverted_outflow_m3s, None);
        assert_eq!(r.productivity_mw_per_m3s, Some(0.5));
    }

    #[test]
    fn thermal_result_gnl_fields_nullable() {
        let gnl = SimulationThermalResult {
            stage_id: 2,
            block_id: Some(1),
            thermal_id: 10,
            generation_mw: 200.0,
            generation_cost: 5000.0,
            is_gnl: true,
            gnl_committed_mw: Some(250.0),
            gnl_decision_mw: Some(200.0),
            operative_state_code: 1,
        };
        assert!(gnl.is_gnl);
        assert_eq!(gnl.gnl_committed_mw, Some(250.0));
        assert_eq!(gnl.gnl_decision_mw, Some(200.0));

        let non_gnl = SimulationThermalResult {
            stage_id: 2,
            block_id: Some(1),
            thermal_id: 11,
            generation_mw: 100.0,
            generation_cost: 3000.0,
            is_gnl: false,
            gnl_committed_mw: None,
            gnl_decision_mw: None,
            operative_state_code: 1,
        };
        assert!(!non_gnl.is_gnl);
        assert_eq!(non_gnl.gnl_committed_mw, None);
        assert_eq!(non_gnl.gnl_decision_mw, None);
    }

    #[test]
    fn exchange_result_construction() {
        let r = SimulationExchangeResult {
            stage_id: 0,
            block_id: Some(0),
            line_id: 3,
            direct_flow_mw: 150.0,
            reverse_flow_mw: 0.0,
            exchange_cost: 10.0,
            operative_state_code: 1,
        };

        assert_eq!(r.stage_id, 0);
        assert_eq!(r.block_id, Some(0));
        assert_eq!(r.line_id, 3);
        assert_eq!(r.direct_flow_mw, 150.0);
        assert_eq!(r.reverse_flow_mw, 0.0);
        assert_eq!(r.exchange_cost, 10.0);
        assert_eq!(r.operative_state_code, 1);
    }

    #[test]
    fn bus_result_construction() {
        let r = SimulationBusResult {
            stage_id: 0,
            block_id: Some(0),
            bus_id: 1,
            load_mw: 300.0,
            deficit_mw: 0.0,
            excess_mw: 0.0,
            spot_price: 120.0,
        };

        assert_eq!(r.stage_id, 0);
        assert_eq!(r.block_id, Some(0));
        assert_eq!(r.bus_id, 1);
        assert_eq!(r.load_mw, 300.0);
        assert_eq!(r.deficit_mw, 0.0);
        assert_eq!(r.excess_mw, 0.0);
        assert_eq!(r.spot_price, 120.0);
    }

    #[test]
    fn pumping_result_construction() {
        let r = SimulationPumpingResult {
            stage_id: 0,
            block_id: Some(0),
            pumping_station_id: 2,
            pumped_flow_m3s: 50.0,
            power_consumption_mw: 25.0,
            pumping_cost: 500.0,
            operative_state_code: 1,
        };

        assert_eq!(r.stage_id, 0);
        assert_eq!(r.block_id, Some(0));
        assert_eq!(r.pumping_station_id, 2);
        assert_eq!(r.pumped_flow_m3s, 50.0);
        assert_eq!(r.power_consumption_mw, 25.0);
        assert_eq!(r.pumping_cost, 500.0);
        assert_eq!(r.operative_state_code, 1);
    }

    #[test]
    fn contract_result_construction() {
        let r = SimulationContractResult {
            stage_id: 0,
            block_id: Some(0),
            contract_id: 7,
            power_mw: 80.0,
            price_per_mwh: 200.0,
            total_cost: 16000.0,
            operative_state_code: 1,
        };

        assert_eq!(r.stage_id, 0);
        assert_eq!(r.block_id, Some(0));
        assert_eq!(r.contract_id, 7);
        assert_eq!(r.power_mw, 80.0);
        assert_eq!(r.price_per_mwh, 200.0);
        assert_eq!(r.total_cost, 16000.0);
        assert_eq!(r.operative_state_code, 1);
    }

    #[test]
    fn non_controllable_result_construction() {
        let r = SimulationNonControllableResult {
            stage_id: 0,
            block_id: Some(0),
            non_controllable_id: 4,
            generation_mw: 60.0,
            available_mw: 80.0,
            curtailment_mw: 20.0,
            curtailment_cost: 200.0,
            operative_state_code: 1,
        };

        assert_eq!(r.stage_id, 0);
        assert_eq!(r.block_id, Some(0));
        assert_eq!(r.non_controllable_id, 4);
        assert_eq!(r.generation_mw, 60.0);
        assert_eq!(r.available_mw, 80.0);
        assert_eq!(r.curtailment_mw, 20.0);
        assert_eq!(r.curtailment_cost, 200.0);
        assert_eq!(r.operative_state_code, 1);
    }

    #[test]
    fn inflow_lag_result_construction() {
        let r = SimulationInflowLagResult {
            stage_id: 5,
            hydro_id: 2,
            lag_index: 1,
            inflow_m3s: 350.0,
        };

        assert_eq!(r.stage_id, 5);
        assert_eq!(r.hydro_id, 2);
        assert_eq!(r.lag_index, 1);
        assert_eq!(r.inflow_m3s, 350.0);
    }

    #[test]
    fn generic_violation_result_construction() {
        let r = SimulationGenericViolationResult {
            stage_id: 3,
            block_id: Some(2),
            constraint_id: 15,
            slack_value: 5.0,
            slack_cost: 50.0,
        };

        assert_eq!(r.stage_id, 3);
        assert_eq!(r.block_id, Some(2));
        assert_eq!(r.constraint_id, 15);
        assert_eq!(r.slack_value, 5.0);
        assert_eq!(r.slack_cost, 50.0);
    }

    #[test]
    fn stage_result_empty_optional_vecs() {
        let stage = SimulationStageResult {
            stage_id: 0,
            costs: vec![],
            hydros: vec![],
            thermals: vec![],
            exchanges: vec![],
            buses: vec![],
            pumping_stations: vec![],
            contracts: vec![],
            non_controllables: vec![],
            inflow_lags: vec![],
            generic_violations: vec![],
        };

        assert!(stage.pumping_stations.is_empty());
        assert!(stage.contracts.is_empty());
        assert!(stage.non_controllables.is_empty());
        assert!(stage.inflow_lags.is_empty());
        assert!(stage.generic_violations.is_empty());
    }

    #[test]
    fn scenario_result_is_send() {
        // Compile-time Send bound check.
        fn assert_send<T: Send>() {}
        assert_send::<SimulationScenarioResult>();
    }

    #[test]
    fn scenario_result_with_multiple_stages() {
        let stages: Vec<SimulationStageResult> = (0..12)
            .map(|i| SimulationStageResult {
                stage_id: i,
                costs: vec![],
                hydros: vec![],
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

        let result = SimulationScenarioResult {
            scenario_id: 42,
            total_cost: 1_000_000.0,
            per_category_costs: ScenarioCategoryCosts {
                resource_cost: 600_000.0,
                recourse_cost: 100_000.0,
                violation_cost: 50_000.0,
                regularization_cost: 200_000.0,
                imputed_cost: 50_000.0,
            },
            stages,
        };

        assert_eq!(result.scenario_id, 42);
        assert_eq!(result.stages.len(), 12);
    }

    #[test]
    fn category_costs_construction() {
        let c = ScenarioCategoryCosts {
            resource_cost: 1.0,
            recourse_cost: 2.0,
            violation_cost: 3.0,
            regularization_cost: 4.0,
            imputed_cost: 5.0,
        };

        assert_eq!(c.resource_cost, 1.0);
        assert_eq!(c.recourse_cost, 2.0);
        assert_eq!(c.violation_cost, 3.0);
        assert_eq!(c.regularization_cost, 4.0);
        assert_eq!(c.imputed_cost, 5.0);
    }

    #[test]
    fn category_cost_stats_construction() {
        let stats = CategoryCostStats {
            category: "recourse".to_string(),
            mean: 500.0,
            max: 2000.0,
            frequency: 0.15,
        };

        assert_eq!(stats.category, "recourse");
        assert_eq!(stats.mean, 500.0);
        assert_eq!(stats.max, 2000.0);
        assert_eq!(stats.frequency, 0.15);
    }

    #[test]
    fn simulation_summary_construction() {
        let category_stats: Vec<CategoryCostStats> = (0_i32..5)
            .map(|i| CategoryCostStats {
                category: format!("cat_{i}"),
                mean: f64::from(i) * 100.0,
                max: f64::from(i) * 500.0,
                frequency: 0.1 * f64::from(i),
            })
            .collect();

        let summary = SimulationSummary {
            mean_cost: 1_500_000.0,
            std_cost: 200_000.0,
            min_cost: 900_000.0,
            max_cost: 2_100_000.0,
            cvar: 1_900_000.0,
            cvar_alpha: 0.95,
            category_stats,
            deficit_frequency: 0.08,
            total_deficit_mwh: 12_500.0,
            total_spillage_mwh: 3_200.0,
            n_scenarios: 2000,
        };

        assert_eq!(summary.mean_cost, 1_500_000.0);
        assert_eq!(summary.std_cost, 200_000.0);
        assert_eq!(summary.min_cost, 900_000.0);
        assert_eq!(summary.max_cost, 2_100_000.0);
        assert_eq!(summary.cvar, 1_900_000.0);
        assert_eq!(summary.cvar_alpha, 0.95);
        assert_eq!(summary.category_stats.len(), 5);
        assert_eq!(summary.deficit_frequency, 0.08);
        assert_eq!(summary.total_deficit_mwh, 12_500.0);
        assert_eq!(summary.total_spillage_mwh, 3_200.0);
        assert_eq!(summary.n_scenarios, 2000);
    }
}
