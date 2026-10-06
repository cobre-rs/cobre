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

use chrono::NaiveDate;

use crate::setup::NodeId;

/// Cost breakdown for one (stage, block) pair.
///
/// Corresponds to one row in the costs output schema
/// (output-schemas.md SS5.1). Contains both aggregate totals and
/// per-category breakdowns used for cost statistics (SS4.2).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
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
    /// Cost attributed to thermal generation dispatch.
    pub thermal_cost: f64,
    /// Cost of anticipated (forward-committed) thermal generation, charged on the
    /// anticipated-decision column at the decision stage. The delivery-stage
    /// generation cost is zeroed to avoid double-counting; reporting this as its
    /// own category lets the breakdown reconcile to `immediate_cost`.
    pub anticipated_thermal_cost: f64,
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
    /// Regularization cost for turbining (applied to every hydro's turbine flow).
    pub turbined_cost: f64,
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
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
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
    /// Incremental (local) natural inflow in m³/s; the same value as
    /// [`Self::incremental_inflow_m3s`].
    pub inflow_m3s: f64,
    /// Reservoir storage at the start of the block in hm³.
    pub storage_initial_hm3: f64,
    /// Reservoir storage at the end of the block in hm³.
    pub storage_final_hm3: f64,
    /// Active power generation in MW.
    pub generation_mw: f64,
    /// Equivalent productivity `ρ_eq` \[MW/(m³/s)\]. For `ConstantProductivity` /
    /// `LinearizedHead` models this is the stored input scalar; for FPHA models
    /// it is derived from VHA geometry, `ρ_esp`, `V_ref`, and `Q_ref` by
    /// `EnergyConversionSet`. Always populated.
    pub equivalent_productivity_mw_per_m3s: f64,
    /// Accumulated productivity `ρ_acum` \[MW/(m³/s)\]: the sum of `ρ_eq` along
    /// the downstream cascade. Always populated.
    pub accumulated_productivity_mw_per_m3s: f64,
    /// Incremental inflow expressed in energy units
    /// (`ρ_acum · incremental_inflow_m3s`) \[MW\].
    pub incremental_inflow_energy_mw: f64,
    /// Stored energy at the start of the block
    /// (`(storage_initial_hm3 − V_min) · ρ_acum · 10^6 / 3600`) \[`MWh`\].
    pub stored_energy_initial_mwh: f64,
    /// Stored energy at the end of the block
    /// (`(storage_final_hm3 − V_min) · ρ_acum · 10^6 / 3600`) \[`MWh`\].
    pub stored_energy_final_mwh: f64,
    /// Regularization cost for spillage at this plant.
    pub spillage_cost: f64,
    /// Marginal water value: the dual of the row's own water-balance constraint
    /// (each block's own row in chronological stages; the stage row, repeated on
    /// every block row, in parallel stages).
    pub water_value_per_hm3: f64,
    /// Storage binding code: indicates which storage bound is active.
    /// Encoded as `i8` matching the Parquet schema.
    pub storage_binding_code: i8,
    /// Operative state code for this hydro plant at this block.
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
    /// Storage-range mean equivalent productivity `ρ_eq` \[MW/(m³/s)\] for this
    /// reservoir. Always populated.
    pub integrated_equivalent_productivity_mw_per_m3s: f64,
    /// Storage-range mean accumulated productivity `ρ_acum` \[MW/(m³/s)\] summed
    /// along the downstream cascade. Always populated.
    pub integrated_accumulated_productivity_mw_per_m3s: f64,
    /// Stored energy at the start of the block, averaged over the STAGE's total
    /// hours (`stored_energy_initial_mwh / Σ block_hours`) \[MW\].
    pub stored_energy_initial_mw: f64,
    /// Stored energy at the end of the block, averaged over the STAGE's total
    /// hours (`stored_energy_final_mwh / Σ block_hours`) \[MW\].
    pub stored_energy_final_mw: f64,
}

/// Per-cell hydro dispatch result for one (stage, block, hydro, bus) tuple —
/// one LP cell.
///
/// Corresponds to one row in the `hydro_bus_generation` output schema. Field
/// for field the mirror of `HydroBusWriteRecord`.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SimulationHydroBusResult {
    /// Stage index (0-based).
    pub stage_id: u32,
    /// Block index within the stage, or [`None`] for stage-level aggregates.
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

/// Thermal unit result for one (stage, block, thermal) tuple.
///
/// Corresponds to one row in the thermals output schema
/// (output-schemas.md SS5.3). The derived column `generation_mwh`
/// is computed by the output writer.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
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
    /// Whether this unit uses anticipated dispatch.
    pub is_anticipated: bool,
    /// Realised delivery commitment in MW, or `None` if not anticipated.
    pub anticipated_committed_mw: Option<f64>,
    /// New anticipated commitment decided at this stage in MW, or `None`.
    pub anticipated_decision_mw: Option<f64>,
    /// Operative state code for this thermal unit at this block.
    pub operative_state_code: i8,
}

/// Exchange (transmission line) result for one (stage, block, line) tuple.
///
/// Corresponds to one row in the exchanges output schema
/// (output-schemas.md SS5.4). Derived columns (`net_flow_mw`,
/// `losses_mw`, and all `MWh` energy columns) are computed by the output
/// writer.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
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
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
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
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
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
    /// Imputed pumping cost, fixed `0.0` (no per-station cost input exists).
    /// `build_pumping_batch` forwards this verbatim — do NOT recompute a cost in
    /// the output writer; with no authoritative source it would diverge.
    pub pumping_cost: f64,
    /// Operative state code for this station at this block.
    pub operative_state_code: i8,
}

/// Contract result for one (stage, block, contract) tuple.
///
/// Corresponds to one row in the contracts output schema
/// (output-schemas.md SS5.7). The derived column `energy_mwh` is
/// computed by the output writer.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
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
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
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
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
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

/// Travel-time in-transit water state for one (stage, downstream-plant, lag)
/// tuple.
///
/// Corresponds to one row in the `in_transit` output schema. Variable-depth,
/// mirroring [`SimulationInflowLagResult`]: a downstream plant fed by a declared
/// travel-time arc owns `L_j` rows, one per maturity bucket. Empty for a plant
/// with no incoming arc, so the whole table is absent for a non-travel-time
/// study (`n_buckets == 0`).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SimulationTransitBucketResult {
    /// Stage index (0-based).
    pub stage_id: u32,
    /// Downstream hydro plant entity ID the arc feeds.
    pub hydro_id: i32,
    /// Maturity bucket index (1-based): water in this bucket matures `lag`
    /// stages ahead of the outgoing state.
    pub lag: u32,
    /// Outgoing in-transit water volume at this maturity in hm³ — the bucket
    /// state carried to the next stage.
    pub in_transit_volume_hm3: f64,
    /// Delivered volume in hm³ that matured onto the downstream water balance
    /// this stage — the incoming lag-1 bucket `b_1^in`. Non-zero only at
    /// `lag == 1` (deeper maturities deliver in later stages).
    pub delayed_arrival_hm3: f64,
}

/// One rolling-seed release window for a declared travel-time arc's upstream
/// hydro, in the same shape [`cobre_core::HydroPastDefluence`] carries.
///
/// Corresponds to one row in the `transit_seed` output partition. Reconstructed
/// from the scenario's realized releases (never the LP's terminal bucket
/// state), covering the trailing `[study_end − travel_time_hours, study_end)`
/// span: one window per in-study stage overlapping that span, plus the
/// stitched pre-study windows sliced from the run's own `past_defluences` when
/// the arc's travel time exceeds the study horizon. Scenario-level (no
/// stage/node axis — a window's own date range anchors it). Empty for a study
/// with no declared travel-time arc.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SimulationTransitSeedResult {
    /// Upstream hydro plant entity ID whose release feeds the arc.
    pub hydro_id: i32,
    /// Start of the release window (inclusive).
    pub start_date: NaiveDate,
    /// End of the release window (exclusive).
    pub end_date: NaiveDate,
    /// Realized mean release rate over the window, in m³/s.
    pub value_m3s: f64,
}

/// Generic constraint violation for one (stage, block, constraint) tuple.
///
/// Corresponds to one row in the violations/generic output schema
/// (output-schemas.md SS5.11). Only entries with non-zero slack values
/// are included.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SimulationGenericViolationResult {
    /// Stage index (0-based).
    pub stage_id: u32,
    /// Block index within the stage.
    pub block_id: Option<u32>,
    /// Generic constraint entity ID.
    pub constraint_id: i32,
    /// Violation slack value: non-negative for a one-sided constraint, the signed net
    /// `s_plus - s_minus` (may be negative) for a two-sided constraint (both bounds finite).
    pub slack_value: f64,
    /// Cost incurred for this violation.
    pub slack_cost: f64,
}

/// Post-horizon commitment lane result for one resolved post-study lane, one
/// terminal scenario.
///
/// Corresponds to one row in the `anticipated_lanes` output schema, keyed
/// `(thermal_id, delivery_date)` — distinct from the per-plant
/// [`SimulationThermalResult::anticipated_committed_mw`]/
/// [`SimulationThermalResult::anticipated_decision_mw`] columns, which stay
/// unchanged. Emitted once per lane, at the lane's own in-study decider
/// stage — the only stage whose LP carries both the decision column and this
/// snapshot of the ring's carried state. Absent for a study with no
/// `post_study_stages`.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SimulationAnticipatedLaneResult {
    /// Stage index (0-based) — the lane's own in-study decider stage.
    pub stage_id: u32,
    /// Thermal unit entity ID owning this lane.
    pub thermal_id: i32,
    /// `YYYYMM01` anchor of the lane's resolved post-study delivery stage.
    pub delivery_date: i32,
    /// Deposited decision in MW, read from the post-horizon decision column.
    pub deposited_decision_mw: f64,
    /// Carried committed value in MW, read from the lane's `commit_out` ring slot.
    pub carried_committed_mw: f64,
}

/// All simulation results for a single stage within one scenario.
///
/// The simulation loop produces one [`SimulationStageResult`] per stage per
/// scenario. Each per-entity-type [`Vec`] holds one entry per (block, entity)
/// pair within the stage. Entity types that are absent from the system or that
/// produce no violations result in empty [`Vec`]s.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SimulationStageResult {
    /// Stage index (0-based).
    pub stage_id: u32,
    /// Declared id of the `NodeGraph` node this scenario's sampled walk visits at
    /// this stage (`node_graph.node_ids[node]`). On a chain this is the
    /// degenerate per-stage node id; it is never gated on whether `nodes[]` was
    /// declared. Broadcast onto every entity row written for the stage.
    pub node_id: NodeId,
    /// Cost breakdown results for this stage (one entry per block).
    pub costs: Vec<SimulationCostResult>,
    /// Hydro plant results for this stage.
    pub hydros: Vec<SimulationHydroResult>,
    /// Per-cell hydro dispatch results for this stage.
    pub hydro_bus_generation: Vec<SimulationHydroBusResult>,
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
    /// Travel-time in-transit bucket records for this stage.
    /// Empty if no travel-time arc is declared (`n_buckets == 0`).
    pub transit_buckets: Vec<SimulationTransitBucketResult>,
    /// Generic constraint violation records for this stage.
    /// Empty if no generic constraints exist or no violations occurred.
    /// Only non-zero violations are included.
    pub generic_violations: Vec<SimulationGenericViolationResult>,
    /// Post-horizon commitment lane records for this stage.
    /// Empty except at a declared window's own in-study decider stage.
    pub anticipated_lanes: Vec<SimulationAnticipatedLaneResult>,
}

/// Per-category cost totals for one scenario, summed across all stages.
///
/// Matches the category breakdown in SS4.2 and is retained in the compact
/// cost buffer even after per-stage detail is streamed to the output writer.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ScenarioCategoryCosts {
    /// Sum of thermal and contract costs: `thermal_cost + contract_cost`.
    pub resource_cost: f64,
    /// Sum of deficit and excess costs: `deficit_cost + excess_cost`.
    pub recourse_cost: f64,
    /// Sum of all violation costs: `storage_violation_cost +
    /// filling_target_cost + hydro_violation_cost + inflow_penalty_cost +
    /// generic_violation_cost`.
    pub violation_cost: f64,
    /// Sum of regularization costs: `spillage_cost + turbined_cost +
    /// curtailment_cost + exchange_cost`.
    pub regularization_cost: f64,
    /// Imputed pumping cost: `pumping_cost`.
    pub imputed_cost: f64,
}

/// Complete simulation result for one scenario.
///
/// Payload type of the bounded channel connecting simulation threads to the
/// background I/O thread (SS6.1); at most `channel_capacity` instances exist at
/// once (SS3.3). Must be `Send` — pinned by the `assert_send` static check below.
#[derive(Debug, PartialEq, serde::Serialize, serde::Deserialize)]
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

    /// Rolling-seed release windows for the next run's `past_defluences`
    /// input, one entry per declared arc's upstream hydro per emitted window.
    /// Empty when the study declares no travel-time arc.
    pub transit_seed: Vec<SimulationTransitSeedResult>,
}

/// Aggregate simulation statistics computed after all scenarios complete.
///
/// Produced by weighted MPI aggregation
/// ([`crate::simulation::aggregation::aggregate_simulation`]), identical on
/// every rank, and returned as the `Ok` value of `fn simulate()`.
#[derive(Debug)]
pub struct SimulationSummary {
    /// Weighted mean total cost across all scenarios:
    /// `mean_cost = Σ wᵢ·Cᵢ` under the caller-selected
    /// [`SimulationWeighting`](crate::simulation::aggregation::SimulationWeighting).
    pub mean_cost: f64,
    /// Sample standard deviation of total cost: `std_cost = sqrt((1/(S-1)) * sum((C_s - mean_cost)^2 for s in 1..=S))`.
    pub std_cost: f64,
    /// Number of scenarios simulated (across all ranks).
    pub n_scenarios: u32,
}

const _: fn() = || {
    fn assert_send<T: Send>() {}
    assert_send::<SimulationScenarioResult>();
};

#[cfg(test)]
mod tests {
    use super::{
        NodeId, ScenarioCategoryCosts, SimulationBusResult, SimulationContractResult,
        SimulationCostResult, SimulationExchangeResult, SimulationGenericViolationResult,
        SimulationHydroBusResult, SimulationHydroResult, SimulationInflowLagResult,
        SimulationNonControllableResult, SimulationPumpingResult, SimulationScenarioResult,
        SimulationStageResult, SimulationSummary, SimulationThermalResult,
        SimulationTransitBucketResult,
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
            anticipated_thermal_cost: 90.0,
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
            turbined_cost: 4.0,
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
        assert_eq!(r.anticipated_thermal_cost, 90.0);
        assert_eq!(r.contract_cost, 100.0);
        assert_eq!(r.deficit_cost, 50.0);
        assert_eq!(r.excess_cost, 10.0);
        assert_eq!(r.storage_violation_cost, 20.0);
        assert_eq!(r.filling_target_cost, 30.0);
        assert_eq!(r.hydro_violation_cost, 5.0);
        assert_eq!(r.inflow_penalty_cost, 3.0);
        assert_eq!(r.generic_violation_cost, 2.0);
        assert_eq!(r.spillage_cost, 1.0);
        assert_eq!(r.turbined_cost, 4.0);
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
            equivalent_productivity_mw_per_m3s: 0.5,
            accumulated_productivity_mw_per_m3s: 0.5,
            incremental_inflow_energy_mw: 100.0,
            stored_energy_initial_mwh: 250.0,
            stored_energy_final_mwh: 240.0,
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
            integrated_equivalent_productivity_mw_per_m3s: 0.7,
            integrated_accumulated_productivity_mw_per_m3s: 0.9,
            stored_energy_initial_mw: 25.0,
            stored_energy_final_mw: 24.0,
        };

        assert_eq!(r.hydro_id, 5);
        assert_eq!(r.turbined_m3s, 100.0);
        assert_eq!(r.evaporation_m3s, None);
        assert_eq!(r.diverted_inflow_m3s, None);
        assert_eq!(r.diverted_outflow_m3s, None);
        assert_eq!(r.equivalent_productivity_mw_per_m3s, 0.5);
        assert_eq!(r.accumulated_productivity_mw_per_m3s, 0.5);
        assert_eq!(r.incremental_inflow_energy_mw, 100.0);
        assert_eq!(r.stored_energy_initial_mwh, 250.0);
        assert_eq!(r.stored_energy_final_mwh, 240.0);
        assert_eq!(r.integrated_equivalent_productivity_mw_per_m3s, 0.7);
        assert_eq!(r.integrated_accumulated_productivity_mw_per_m3s, 0.9);
        assert_eq!(r.stored_energy_initial_mw, 25.0);
        assert_eq!(r.stored_energy_final_mw, 24.0);
    }

    #[test]
    fn simulation_hydro_result_serde_round_trip() {
        let original = SimulationHydroResult {
            stage_id: 3,
            block_id: Some(1),
            hydro_id: 7,
            turbined_m3s: 120.0,
            spillage_m3s: 5.0,
            evaporation_m3s: Some(0.3),
            diverted_inflow_m3s: Some(10.0),
            diverted_outflow_m3s: Some(8.0),
            incremental_inflow_m3s: 300.0,
            inflow_m3s: 310.0,
            storage_initial_hm3: 600.0,
            storage_final_hm3: 590.0,
            generation_mw: 65.0,
            equivalent_productivity_mw_per_m3s: 1.0,
            accumulated_productivity_mw_per_m3s: 2.0,
            incremental_inflow_energy_mw: 3.0,
            stored_energy_initial_mwh: 4.0,
            stored_energy_final_mwh: 5.0,
            spillage_cost: 0.1,
            water_value_per_hm3: 15.0,
            storage_binding_code: 1,
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
            integrated_equivalent_productivity_mw_per_m3s: 7.0,
            integrated_accumulated_productivity_mw_per_m3s: 9.0,
            stored_energy_initial_mw: 0.4,
            stored_energy_final_mw: 0.5,
        };

        let json = serde_json::to_string(&original).expect("serialize");
        let decoded: SimulationHydroResult = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(decoded.stage_id, original.stage_id);
        assert_eq!(decoded.block_id, original.block_id);
        assert_eq!(decoded.hydro_id, original.hydro_id);
        assert_eq!(
            decoded.equivalent_productivity_mw_per_m3s,
            original.equivalent_productivity_mw_per_m3s
        );
        assert_eq!(
            decoded.accumulated_productivity_mw_per_m3s,
            original.accumulated_productivity_mw_per_m3s
        );
        assert_eq!(
            decoded.incremental_inflow_energy_mw,
            original.incremental_inflow_energy_mw
        );
        assert_eq!(
            decoded.stored_energy_initial_mwh,
            original.stored_energy_initial_mwh
        );
        assert_eq!(
            decoded.stored_energy_final_mwh,
            original.stored_energy_final_mwh
        );
        assert_eq!(
            decoded.equivalent_productivity_mw_per_m3s.to_bits(),
            original.equivalent_productivity_mw_per_m3s.to_bits()
        );
        assert_eq!(
            decoded.accumulated_productivity_mw_per_m3s.to_bits(),
            original.accumulated_productivity_mw_per_m3s.to_bits()
        );
        assert_eq!(
            decoded.incremental_inflow_energy_mw.to_bits(),
            original.incremental_inflow_energy_mw.to_bits()
        );
        assert_eq!(
            decoded.stored_energy_initial_mwh.to_bits(),
            original.stored_energy_initial_mwh.to_bits()
        );
        assert_eq!(
            decoded.stored_energy_final_mwh.to_bits(),
            original.stored_energy_final_mwh.to_bits()
        );
        assert_eq!(
            decoded
                .integrated_equivalent_productivity_mw_per_m3s
                .to_bits(),
            original
                .integrated_equivalent_productivity_mw_per_m3s
                .to_bits()
        );
        assert_eq!(
            decoded
                .integrated_accumulated_productivity_mw_per_m3s
                .to_bits(),
            original
                .integrated_accumulated_productivity_mw_per_m3s
                .to_bits()
        );
        assert_eq!(
            decoded.stored_energy_initial_mw.to_bits(),
            original.stored_energy_initial_mw.to_bits()
        );
        assert_eq!(
            decoded.stored_energy_final_mw.to_bits(),
            original.stored_energy_final_mw.to_bits()
        );
    }

    #[test]
    fn thermal_result_anticipated_fields_nullable() {
        let anticipated = SimulationThermalResult {
            stage_id: 2,
            block_id: Some(1),
            thermal_id: 10,
            generation_mw: 200.0,
            generation_cost: 5000.0,
            is_anticipated: true,
            anticipated_committed_mw: Some(250.0),
            anticipated_decision_mw: Some(200.0),
            operative_state_code: 1,
        };
        assert!(anticipated.is_anticipated);
        assert_eq!(anticipated.anticipated_committed_mw, Some(250.0));
        assert_eq!(anticipated.anticipated_decision_mw, Some(200.0));

        let non_anticipated = SimulationThermalResult {
            stage_id: 2,
            block_id: Some(1),
            thermal_id: 11,
            generation_mw: 100.0,
            generation_cost: 3000.0,
            is_anticipated: false,
            anticipated_committed_mw: None,
            anticipated_decision_mw: None,
            operative_state_code: 1,
        };
        assert!(!non_anticipated.is_anticipated);
        assert_eq!(non_anticipated.anticipated_committed_mw, None);
        assert_eq!(non_anticipated.anticipated_decision_mw, None);
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
    fn transit_bucket_result_construction() {
        let r = SimulationTransitBucketResult {
            stage_id: 4,
            hydro_id: 7,
            lag: 1,
            in_transit_volume_hm3: 12.5,
            delayed_arrival_hm3: 3.0,
        };

        assert_eq!(r.stage_id, 4);
        assert_eq!(r.hydro_id, 7);
        assert_eq!(r.lag, 1);
        assert_eq!(r.in_transit_volume_hm3, 12.5);
        assert_eq!(r.delayed_arrival_hm3, 3.0);
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
    fn hydro_bus_result_construction() {
        let r = SimulationHydroBusResult {
            stage_id: 0,
            block_id: Some(1),
            hydro_id: 5,
            bus_id: 12,
            turbined_m3s: 30.0,
            generation_mw: 15.0,
        };

        assert_eq!(r.stage_id, 0);
        assert_eq!(r.block_id, Some(1));
        assert_eq!(r.hydro_id, 5);
        assert_eq!(r.bus_id, 12);
        assert_eq!(r.turbined_m3s, 30.0);
        assert_eq!(r.generation_mw, 15.0);
    }

    #[test]
    fn stage_result_empty_optional_vecs() {
        let stage = SimulationStageResult {
            stage_id: 0,
            node_id: NodeId(0),
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

        assert!(stage.pumping_stations.is_empty());
        assert!(stage.contracts.is_empty());
        assert!(stage.non_controllables.is_empty());
        assert!(stage.inflow_lags.is_empty());
        assert!(stage.transit_buckets.is_empty());
        assert!(stage.generic_violations.is_empty());
        assert!(stage.anticipated_lanes.is_empty());
    }

    #[test]
    fn scenario_result_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<SimulationScenarioResult>();
    }

    #[test]
    fn scenario_result_with_multiple_stages() {
        let stages: Vec<SimulationStageResult> = (0..12)
            .map(|i| SimulationStageResult {
                stage_id: i,
                node_id: NodeId(i as i32),
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
            transit_seed: vec![],
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
    fn simulation_summary_construction() {
        let summary = SimulationSummary {
            mean_cost: 1_500_000.0,
            std_cost: 200_000.0,
            n_scenarios: 2000,
        };

        assert_eq!(summary.mean_cost, 1_500_000.0);
        assert_eq!(summary.std_cost, 200_000.0);
        assert_eq!(summary.n_scenarios, 2000);
    }
}
