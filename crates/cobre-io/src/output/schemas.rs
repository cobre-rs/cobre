//! Arrow schema definitions for all Parquet output files per output-schemas spec
//! (SS5.1–5.11 and SS6.1–6.2).
//!
//! Every `*_schema()` function in this module has exactly one row in
//! [`OUTPUT_SCHEMAS`], the crate-internal registry that pairs each schema
//! constructor with its registry name and its optional `variables.csv` label.
//! [`OUTPUT_FILES`](super::file_registry::OUTPUT_FILES) owns output paths: it
//! maps every written Parquet file to the schema that backs it. The
//! axis-spelling gate test, `variables.csv` generation and that file table all
//! derive from `OUTPUT_SCHEMAS`.
//!
//! The per-scenario simulation entity schemas are cached behind a
//! process-lifetime [`LazyLock`] and returned as `Arc<Schema>`, since
//! `simulation_writer`'s `build_*_batch` functions build one per scenario;
//! `OUTPUT_SCHEMAS` still registers each one's uncached builder so the table's
//! `fn() -> Schema` entries stay uniform.

use std::sync::{Arc, LazyLock};

use arrow::datatypes::{DataType, Field, Schema};

/// The `(scenario_id, stage_id, node_id)` axis prefix shared by every simulation
/// entity row and by `paths.parquet`. `scenario_id` duplicates the Hive partition
/// as a column so joins don't parse directory names, and `node_id` is the visited
/// node's declared id (the degenerate per-stage id on a chain — never gated on
/// `nodes[]`).
fn simulation_row_prefix() -> Vec<Field> {
    vec![
        Field::new("scenario_id", DataType::Int32, false),
        Field::new("stage_id", DataType::Int32, false),
        Field::new("node_id", DataType::Int32, false),
    ]
}

fn build_costs_schema() -> Schema {
    let mut fields = simulation_row_prefix();
    fields.extend([
        Field::new("block_id", DataType::Int32, true),
        Field::new("total_cost", DataType::Float64, false),
        Field::new("immediate_cost", DataType::Float64, false),
        Field::new("future_cost", DataType::Float64, false),
        Field::new("discount_factor", DataType::Float64, false),
        Field::new("thermal_cost", DataType::Float64, false),
        Field::new("anticipated_thermal_cost", DataType::Float64, false),
        Field::new("contract_cost", DataType::Float64, false),
        Field::new("deficit_cost", DataType::Float64, false),
        Field::new("excess_cost", DataType::Float64, false),
        Field::new("storage_violation_cost", DataType::Float64, false),
        Field::new("filling_target_cost", DataType::Float64, false),
        Field::new("hydro_violation_cost", DataType::Float64, false),
        Field::new("outflow_violation_below_cost", DataType::Float64, false),
        Field::new("outflow_violation_above_cost", DataType::Float64, false),
        Field::new("turbined_violation_cost", DataType::Float64, false),
        Field::new("generation_violation_cost", DataType::Float64, false),
        Field::new("evaporation_violation_cost", DataType::Float64, false),
        Field::new("withdrawal_violation_cost", DataType::Float64, false),
        Field::new("inflow_penalty_cost", DataType::Float64, false),
        Field::new("generic_violation_cost", DataType::Float64, false),
        Field::new("spillage_cost", DataType::Float64, false),
        Field::new("turbined_cost", DataType::Float64, false),
        Field::new("curtailment_cost", DataType::Float64, false),
        Field::new("exchange_cost", DataType::Float64, false),
        Field::new("pumping_cost", DataType::Float64, false),
    ]);
    Schema::new(fields)
}

static COSTS_SCHEMA: LazyLock<Arc<Schema>> = LazyLock::new(|| Arc::new(build_costs_schema()));

/// Schema of the stage and block-level cost breakdown.
///
/// See output-schemas.md SS5.1.
pub(crate) fn costs_schema() -> Arc<Schema> {
    Arc::clone(&COSTS_SCHEMA)
}

fn build_hydros_schema() -> Schema {
    let mut fields = simulation_row_prefix();
    fields.extend([
        Field::new("block_id", DataType::Int32, true),
        Field::new("hydro_id", DataType::Int32, false),
        Field::new("turbined_m3s", DataType::Float64, false),
        Field::new("spillage_m3s", DataType::Float64, false),
        Field::new("outflow_m3s", DataType::Float64, false),
        Field::new("evaporation_m3s", DataType::Float64, true),
        Field::new("diverted_inflow_m3s", DataType::Float64, true),
        Field::new("diverted_outflow_m3s", DataType::Float64, true),
        Field::new("incremental_inflow_m3s", DataType::Float64, false),
        Field::new("inflow_m3s", DataType::Float64, false),
        Field::new("storage_initial_hm3", DataType::Float64, false),
        Field::new("storage_final_hm3", DataType::Float64, false),
        Field::new("generation_mw", DataType::Float64, false),
        Field::new("generation_mwh", DataType::Float64, false),
        Field::new(
            "equivalent_productivity_mw_per_m3s",
            DataType::Float64,
            false,
        ),
        Field::new(
            "accumulated_productivity_mw_per_m3s",
            DataType::Float64,
            false,
        ),
        Field::new("incremental_inflow_energy_mw", DataType::Float64, false),
        Field::new("stored_energy_initial_mwh", DataType::Float64, false),
        Field::new("stored_energy_final_mwh", DataType::Float64, false),
        Field::new("spillage_cost", DataType::Float64, false),
        Field::new("water_value_per_hm3", DataType::Float64, false),
        Field::new("storage_binding_code", DataType::Int8, false),
        Field::new("operative_state_code", DataType::Int8, false),
        Field::new("turbined_slack_m3s", DataType::Float64, false),
        Field::new("outflow_slack_below_m3s", DataType::Float64, false),
        Field::new("outflow_slack_above_m3s", DataType::Float64, false),
        Field::new("generation_slack_mw", DataType::Float64, false),
        Field::new("storage_violation_below_hm3", DataType::Float64, false),
        Field::new("filling_target_violation_hm3", DataType::Float64, false),
        Field::new("evaporation_violation_pos_m3s", DataType::Float64, false),
        Field::new("evaporation_violation_neg_m3s", DataType::Float64, false),
        Field::new("inflow_nonnegativity_slack_m3s", DataType::Float64, false),
        Field::new(
            "water_withdrawal_violation_pos_m3s",
            DataType::Float64,
            false,
        ),
        Field::new(
            "water_withdrawal_violation_neg_m3s",
            DataType::Float64,
            false,
        ),
        Field::new(
            "integrated_equivalent_productivity_mw_per_m3s",
            DataType::Float64,
            false,
        ),
        Field::new(
            "integrated_accumulated_productivity_mw_per_m3s",
            DataType::Float64,
            false,
        ),
        Field::new("stored_energy_initial_mw", DataType::Float64, false),
        Field::new("stored_energy_final_mw", DataType::Float64, false),
    ]);
    Schema::new(fields)
}

static HYDROS_SCHEMA: LazyLock<Arc<Schema>> = LazyLock::new(|| Arc::new(build_hydros_schema()));

/// Schema of the hydro plant dispatch results.
///
/// See output-schemas.md SS5.2.
pub(crate) fn hydros_schema() -> Arc<Schema> {
    Arc::clone(&HYDROS_SCHEMA)
}

fn build_hydro_bus_generation_schema() -> Schema {
    let mut fields = simulation_row_prefix();
    fields.extend([
        Field::new("block_id", DataType::Int32, true),
        Field::new("hydro_id", DataType::Int32, false),
        Field::new("bus_id", DataType::Int32, false),
        Field::new("turbined_m3s", DataType::Float64, false),
        Field::new("generation_mw", DataType::Float64, false),
        Field::new("generation_mwh", DataType::Float64, false),
    ]);
    Schema::new(fields)
}

static HYDRO_BUS_GENERATION_SCHEMA: LazyLock<Arc<Schema>> =
    LazyLock::new(|| Arc::new(build_hydro_bus_generation_schema()));

/// Schema of the per-cell hydro dispatch results.
///
/// One row per (stage, block, hydro, bus) — one LP cell.
pub(crate) fn hydro_bus_generation_schema() -> Arc<Schema> {
    Arc::clone(&HYDRO_BUS_GENERATION_SCHEMA)
}

fn build_thermals_schema() -> Schema {
    let mut fields = simulation_row_prefix();
    fields.extend([
        Field::new("block_id", DataType::Int32, true),
        Field::new("thermal_id", DataType::Int32, false),
        Field::new("generation_mw", DataType::Float64, false),
        Field::new("generation_mwh", DataType::Float64, false),
        Field::new("generation_cost", DataType::Float64, false),
        Field::new("is_anticipated", DataType::Boolean, false),
        Field::new("anticipated_committed_mw", DataType::Float64, true),
        Field::new("anticipated_decision_mw", DataType::Float64, true),
        Field::new("operative_state_code", DataType::Int8, false),
    ]);
    Schema::new(fields)
}

static THERMALS_SCHEMA: LazyLock<Arc<Schema>> = LazyLock::new(|| Arc::new(build_thermals_schema()));

/// Schema of the thermal unit dispatch results.
///
/// See output-schemas.md SS5.3.
pub(crate) fn thermals_schema() -> Arc<Schema> {
    Arc::clone(&THERMALS_SCHEMA)
}

fn build_exchanges_schema() -> Schema {
    let mut fields = simulation_row_prefix();
    fields.extend([
        Field::new("block_id", DataType::Int32, true),
        Field::new("line_id", DataType::Int32, false),
        Field::new("direct_flow_mw", DataType::Float64, false),
        Field::new("reverse_flow_mw", DataType::Float64, false),
        Field::new("net_flow_mw", DataType::Float64, false),
        Field::new("net_flow_mwh", DataType::Float64, false),
        Field::new("losses_mw", DataType::Float64, false),
        Field::new("losses_mwh", DataType::Float64, false),
        Field::new("exchange_cost", DataType::Float64, false),
        Field::new("operative_state_code", DataType::Int8, false),
    ]);
    Schema::new(fields)
}

static EXCHANGES_SCHEMA: LazyLock<Arc<Schema>> =
    LazyLock::new(|| Arc::new(build_exchanges_schema()));

/// Schema of the transmission line flow results.
///
/// See output-schemas.md SS5.4.
pub(crate) fn exchanges_schema() -> Arc<Schema> {
    Arc::clone(&EXCHANGES_SCHEMA)
}

fn build_buses_schema() -> Schema {
    let mut fields = simulation_row_prefix();
    fields.extend([
        Field::new("block_id", DataType::Int32, true),
        Field::new("bus_id", DataType::Int32, false),
        Field::new("load_mw", DataType::Float64, false),
        Field::new("load_mwh", DataType::Float64, false),
        Field::new("deficit_mw", DataType::Float64, false),
        Field::new("deficit_mwh", DataType::Float64, false),
        Field::new("excess_mw", DataType::Float64, false),
        Field::new("excess_mwh", DataType::Float64, false),
        Field::new("spot_price", DataType::Float64, false),
    ]);
    Schema::new(fields)
}

static BUSES_SCHEMA: LazyLock<Arc<Schema>> = LazyLock::new(|| Arc::new(build_buses_schema()));

/// Schema of the bus load balance results.
///
/// See output-schemas.md SS5.5.
pub(crate) fn buses_schema() -> Arc<Schema> {
    Arc::clone(&BUSES_SCHEMA)
}

fn build_pumping_stations_schema() -> Schema {
    let mut fields = simulation_row_prefix();
    fields.extend([
        Field::new("block_id", DataType::Int32, true),
        Field::new("pumping_station_id", DataType::Int32, false),
        Field::new("pumped_flow_m3s", DataType::Float64, false),
        Field::new("pumped_volume_hm3", DataType::Float64, false),
        Field::new("power_consumption_mw", DataType::Float64, false),
        Field::new("energy_consumption_mwh", DataType::Float64, false),
        Field::new("pumping_cost", DataType::Float64, false),
        Field::new("operative_state_code", DataType::Int8, false),
    ]);
    Schema::new(fields)
}

static PUMPING_STATIONS_SCHEMA: LazyLock<Arc<Schema>> =
    LazyLock::new(|| Arc::new(build_pumping_stations_schema()));

/// Schema of the pumping station results.
///
/// See output-schemas.md SS5.6.
pub(crate) fn pumping_stations_schema() -> Arc<Schema> {
    Arc::clone(&PUMPING_STATIONS_SCHEMA)
}

fn build_contracts_schema() -> Schema {
    let mut fields = simulation_row_prefix();
    fields.extend([
        Field::new("block_id", DataType::Int32, true),
        Field::new("contract_id", DataType::Int32, false),
        Field::new("power_mw", DataType::Float64, false),
        Field::new("energy_mwh", DataType::Float64, false),
        Field::new("price_per_mwh", DataType::Float64, false),
        Field::new("total_cost", DataType::Float64, false),
        Field::new("operative_state_code", DataType::Int8, false),
    ]);
    Schema::new(fields)
}

static CONTRACTS_SCHEMA: LazyLock<Arc<Schema>> =
    LazyLock::new(|| Arc::new(build_contracts_schema()));

/// Schema of the energy contract results.
///
/// See output-schemas.md SS5.7.
pub(crate) fn contracts_schema() -> Arc<Schema> {
    Arc::clone(&CONTRACTS_SCHEMA)
}

fn build_non_controllables_schema() -> Schema {
    let mut fields = simulation_row_prefix();
    fields.extend([
        Field::new("block_id", DataType::Int32, true),
        Field::new("non_controllable_id", DataType::Int32, false),
        Field::new("generation_mw", DataType::Float64, false),
        Field::new("generation_mwh", DataType::Float64, false),
        Field::new("available_mw", DataType::Float64, false),
        Field::new("curtailment_mw", DataType::Float64, false),
        Field::new("curtailment_mwh", DataType::Float64, false),
        Field::new("curtailment_cost", DataType::Float64, false),
        Field::new("operative_state_code", DataType::Int8, false),
    ]);
    Schema::new(fields)
}

static NON_CONTROLLABLES_SCHEMA: LazyLock<Arc<Schema>> =
    LazyLock::new(|| Arc::new(build_non_controllables_schema()));

/// Schema of the non-controllable source results.
///
/// See output-schemas.md SS5.8.
pub(crate) fn non_controllables_schema() -> Arc<Schema> {
    Arc::clone(&NON_CONTROLLABLES_SCHEMA)
}

fn build_inflow_lags_schema() -> Schema {
    let mut fields = simulation_row_prefix();
    fields.extend([
        Field::new("hydro_id", DataType::Int32, false),
        Field::new("lag_index", DataType::Int32, false),
        Field::new("inflow_m3s", DataType::Float64, false),
    ]);
    Schema::new(fields)
}

static INFLOW_LAGS_SCHEMA: LazyLock<Arc<Schema>> =
    LazyLock::new(|| Arc::new(build_inflow_lags_schema()));

/// Schema of the autoregressive inflow state variables.
///
/// See output-schemas.md SS5.10.
pub(crate) fn inflow_lags_schema() -> Arc<Schema> {
    Arc::clone(&INFLOW_LAGS_SCHEMA)
}

fn build_in_transit_schema() -> Schema {
    let mut fields = simulation_row_prefix();
    fields.extend([
        Field::new("hydro_id", DataType::Int32, false),
        Field::new("lag", DataType::Int32, false),
        Field::new("in_transit_volume_hm3", DataType::Float64, false),
        Field::new("delayed_arrival_hm3", DataType::Float64, false),
    ]);
    Schema::new(fields)
}

static IN_TRANSIT_SCHEMA: LazyLock<Arc<Schema>> =
    LazyLock::new(|| Arc::new(build_in_transit_schema()));

/// Schema of the travel-time in-transit water volumes.
///
/// One row per (stage, downstream plant, maturity lag). Written only when the
/// system declares a travel-time arc.
pub(crate) fn in_transit_schema() -> Arc<Schema> {
    Arc::clone(&IN_TRANSIT_SCHEMA)
}

fn build_transit_seed_schema() -> Schema {
    Schema::new(vec![
        Field::new("scenario_id", DataType::Int32, false),
        Field::new("hydro_id", DataType::Int32, false),
        Field::new("start_date", DataType::Date32, false),
        Field::new("end_date", DataType::Date32, false),
        Field::new("value_m3s", DataType::Float64, false),
    ])
}

static TRANSIT_SEED_SCHEMA: LazyLock<Arc<Schema>> =
    LazyLock::new(|| Arc::new(build_transit_seed_schema()));

/// Schema of the rolling release-window seed for a continuing run's own
/// upstream-release input.
///
/// Scenario-level: unlike every other simulation partition, a window's own
/// `[start_date, end_date)` span anchors the row, not a stage/node index, so
/// this schema carries `scenario_id` alone (no `stage_id`/`node_id`). Written
/// only when the system declares a travel-time arc.
pub(crate) fn transit_seed_schema() -> Arc<Schema> {
    Arc::clone(&TRANSIT_SEED_SCHEMA)
}

/// Schema of the run-level echo of declared fixed post-horizon commitment
/// windows.
///
/// One row per anticipated plant × fixed window, carrying the window's real
/// delivery dates (`start_date`/`end_date` as `Date32`) and its committed
/// `value_mw`. Unpartitioned and axis-free: the values are scenario- and
/// stage-independent constants, so there is no `simulation_row_prefix`. No cost
/// or energy column — a fixed commitment is never booked (the fuel was charged
/// at the revision that decided it) — and no source marker, since the file's
/// existence is the marker.
pub(crate) fn fixed_delivery_schema() -> Schema {
    Schema::new(vec![
        Field::new("thermal_id", DataType::Int32, false),
        Field::new("start_date", DataType::Date32, false),
        Field::new("end_date", DataType::Date32, false),
        Field::new("value_mw", DataType::Float64, false),
    ])
}

fn build_anticipated_lanes_schema() -> Schema {
    let mut fields = simulation_row_prefix();
    fields.extend([
        Field::new("thermal_id", DataType::Int32, false),
        Field::new("delivery_date", DataType::Int32, false),
        Field::new("deposited_decision_mw", DataType::Float64, false),
        Field::new("carried_committed_mw", DataType::Float64, false),
    ]);
    Schema::new(fields)
}

static ANTICIPATED_LANES_SCHEMA: LazyLock<Arc<Schema>> =
    LazyLock::new(|| Arc::new(build_anticipated_lanes_schema()));

/// Schema of the post-horizon commitment lane results, keyed
/// `(thermal_id, delivery_date)`.
///
/// One row per resolved post-study commitment lane per terminal scenario,
/// written only when the system declares `post_study_stages`.
pub(crate) fn anticipated_lanes_schema() -> Arc<Schema> {
    Arc::clone(&ANTICIPATED_LANES_SCHEMA)
}

fn build_generic_violations_schema() -> Schema {
    let mut fields = simulation_row_prefix();
    fields.extend([
        Field::new("block_id", DataType::Int32, true),
        Field::new("constraint_id", DataType::Int32, false),
        Field::new("slack_value", DataType::Float64, false),
        Field::new("slack_cost", DataType::Float64, false),
    ]);
    Schema::new(fields)
}

static GENERIC_VIOLATIONS_SCHEMA: LazyLock<Arc<Schema>> =
    LazyLock::new(|| Arc::new(build_generic_violations_schema()));

/// Schema of the generic constraint violations.
///
/// See output-schemas.md SS5.11.
pub(crate) fn generic_violations_schema() -> Arc<Schema> {
    Arc::clone(&GENERIC_VIOLATIONS_SCHEMA)
}

/// Schema of the per-scenario node-path trace.
///
/// Run-level and unpartitioned (never Hive-partitioned): exactly the
/// `(scenario_id, stage_id, node_id)` axis prefix, all non-null `Int32`. Joins to
/// any entity file on `(scenario_id, stage_id)`.
pub(crate) fn paths_schema() -> Schema {
    Schema::new(simulation_row_prefix())
}

/// Schema of the run-level, unpartitioned per-scenario summary.
///
/// `scenario_id` is the non-null `Int32` join key shared with every entity file
/// and `paths.parquet` (the `simulation_row_prefix` convention); a wider type
/// here would break the join on the primary key. `probability` is populated only
/// under a declared census (the per-scenario leaf-path weight) and is NULL on
/// every row under sampled selection.
pub(crate) fn scenario_summary_schema() -> Schema {
    Schema::new(vec![
        Field::new("scenario_id", DataType::Int32, false),
        Field::new("probability", DataType::Float64, true),
        Field::new("discounted_immediate_cost", DataType::Float64, false),
    ])
}

/// Schema of the iteration-level convergence log.
///
/// See output-schemas.md SS6.1.
pub(crate) fn convergence_schema() -> Schema {
    Schema::new(vec![
        Field::new("iteration", DataType::Int32, false),
        Field::new("lower_bound", DataType::Float64, false),
        Field::new("upper_bound", DataType::Float64, false),
        Field::new("upper_bound_std", DataType::Float64, true),
        Field::new("upper_bound_kind", DataType::Utf8, false),
        Field::new("gap_percent", DataType::Float64, true),
        Field::new("cuts_added", DataType::Int32, false),
        Field::new("cuts_removed", DataType::Int32, false),
        Field::new("cuts_active", DataType::Int64, false),
        Field::new("time_forward_ms", DataType::Int64, false),
        Field::new("time_backward_ms", DataType::Int64, false),
        Field::new("time_total_ms", DataType::Int64, false),
        Field::new("forward_passes", DataType::Int32, false),
        Field::new("lp_solves", DataType::Int64, false),
        Field::new("mean_rows_in_lp", DataType::Float64, false),
    ])
}

/// Schema of the per-iteration timing breakdown.
///
/// Row semantics: one row per `(iteration, rank)` for rank-only sequential values
/// (`worker_id = NULL`), and one row per `(iteration, rank, worker_id)` for
/// per-worker parallel-region values. `SUM(col) GROUP BY iteration` recovers the
/// single-row-per-iteration value for each of the 16 timing columns.
///
/// Only the top-level non-overlapping phases (`forward_wall_ms`,
/// `backward_wall_ms`, `cut_selection_ms`, `mpi_allreduce_ms`, `lower_bound_ms`)
/// and `overhead_ms` are addends of the iteration total; the `*_setup_ms`,
/// `*_load_imbalance_ms`, `*_scheduling_overhead_ms`, `cut_sync_ms`,
/// `state_exchange_ms`, `cut_batch_build_ms`, and `lazy_scoring_ms` columns are
/// sub-components nested under a pass, not top-level addends.
pub(crate) fn iteration_timing_schema() -> Schema {
    Schema::new(vec![
        Field::new("iteration", DataType::Int32, false),
        Field::new("rank", DataType::Int32, true),
        Field::new("worker_id", DataType::Int32, true),
        Field::new("forward_wall_ms", DataType::Int64, false),
        Field::new("backward_wall_ms", DataType::Int64, false),
        Field::new("cut_selection_ms", DataType::Int64, false),
        Field::new("mpi_allreduce_ms", DataType::Int64, false),
        Field::new("cut_sync_ms", DataType::Int64, false),
        Field::new("lower_bound_ms", DataType::Int64, false),
        Field::new("state_exchange_ms", DataType::Int64, false),
        Field::new("cut_batch_build_ms", DataType::Int64, false),
        Field::new("bwd_setup_ms", DataType::Int64, false),
        Field::new("bwd_load_imbalance_ms", DataType::Int64, false),
        Field::new("bwd_scheduling_overhead_ms", DataType::Int64, false),
        Field::new("fwd_setup_ms", DataType::Int64, false),
        Field::new("fwd_load_imbalance_ms", DataType::Int64, false),
        Field::new("fwd_scheduling_overhead_ms", DataType::Int64, false),
        Field::new("overhead_ms", DataType::Int64, false),
        Field::new("lazy_scoring_ms", DataType::Int64, false),
    ])
}

/// Schema of the per-iteration, per-phase solver statistics for diagnosing LP
/// conditioning and retry behavior.
///
/// One row per (iteration, phase, `stage_id`, `opening_index`) tuple for backward
/// rows; forward and `lower_bound` rows carry `opening_index = NULL`, and
/// `lower_bound` rows also carry `stage_id = NULL` (no stage). A training row
/// fills `iteration` (and leaves `scenario_id = NULL`); a simulation row fills
/// `scenario_id` (and leaves `iteration = NULL`). The nullable `rank` and
/// `worker_id` columns are `NULL` for rank-aggregated rows and otherwise carry
/// the producing rank and worker index.
pub(crate) fn solver_iterations_schema() -> Schema {
    Schema::new(vec![
        Field::new("iteration", DataType::Int32, true),
        Field::new("scenario_id", DataType::Int32, true),
        Field::new("phase", DataType::Utf8, false),
        Field::new("stage_id", DataType::Int32, true),
        Field::new("opening_index", DataType::Int32, true),
        Field::new("rank", DataType::Int32, true),
        Field::new("worker_id", DataType::Int32, true),
        Field::new("lp_solves", DataType::UInt32, false),
        Field::new("lp_successes", DataType::UInt32, false),
        Field::new("lp_retries", DataType::UInt32, false),
        Field::new("lp_failures", DataType::UInt32, false),
        Field::new("retry_attempts", DataType::UInt32, false),
        Field::new("basis_offered", DataType::UInt32, false),
        Field::new("basis_consistency_failures", DataType::UInt32, false),
        Field::new("simplex_iterations", DataType::UInt64, false),
        Field::new("solve_time_ms", DataType::Float64, false),
        Field::new("load_model_time_ms", DataType::Float64, false),
        Field::new("set_bounds_time_ms", DataType::Float64, false),
        Field::new("basis_set_time_ms", DataType::Float64, false),
    ])
}

/// Schema of the per-level retry success counts for training and simulation
/// rows, normalized from the solver iterations table.
///
/// Sparse: one row per (iteration, phase, `stage_id`, `retry_level`) tuple where
/// `count > 0`. `stage_id` is `NULL` for the forward, `lower_bound`, and
/// simulation rows that carry no per-stage attribution.
pub(crate) fn retry_histogram_schema() -> Schema {
    Schema::new(vec![
        Field::new("iteration", DataType::UInt32, false),
        Field::new("phase", DataType::Utf8, false),
        Field::new("stage_id", DataType::Int32, true),
        Field::new("retry_level", DataType::UInt32, false),
        Field::new("count", DataType::UInt64, false),
    ])
}

/// Schema of the per-stage row-selection statistics.
///
/// One row per (iteration, `stage_id`) pair. The nullable `budget_evicted` and
/// `active_after_budget` columns are `None` when budget enforcement is disabled.
pub(crate) fn row_selection_schema() -> Schema {
    Schema::new(vec![
        Field::new("iteration", DataType::Int32, false),
        Field::new("stage_id", DataType::Int32, false),
        Field::new("cuts_populated", DataType::Int32, false),
        Field::new("cuts_active_before", DataType::Int32, false),
        Field::new("cuts_deactivated", DataType::Int32, false),
        Field::new("cuts_reactivated", DataType::Int32, false),
        Field::new("cuts_active_after", DataType::Int32, false),
        Field::new("selection_time_ms", DataType::Float64, false),
        Field::new("budget_evicted", DataType::Int32, true),
        Field::new("active_after_budget", DataType::Int32, true),
    ])
}

/// Schema for the resolved generic-constraint echo — one row per
/// `(constraint, stage, block, term)`.
///
/// `bound_lower`/`bound_upper` are the resolved interval endpoints (min before
/// max), each `None` where unbounded on that side; `derived_shape` labels the
/// shape those endpoints imply. The per-term columns (`term_index`,
/// `variable_kind`, `variable`, `coefficient`) are `None` on a term-less
/// constraint's placeholder row, and `slack_penalty` is `None` when slack is off.
pub(crate) fn generic_constraint_echo_schema() -> Schema {
    Schema::new(vec![
        Field::new("stage_id", DataType::Int32, false),
        Field::new("block_id", DataType::Int32, true),
        Field::new("constraint_id", DataType::Int32, false),
        Field::new("constraint_name", DataType::Utf8, false),
        Field::new("term_index", DataType::Int32, true),
        Field::new("variable_kind", DataType::Utf8, true),
        Field::new("variable", DataType::Utf8, true),
        Field::new("coefficient", DataType::Float64, true),
        Field::new("bound_lower", DataType::Float64, true),
        Field::new("bound_upper", DataType::Float64, true),
        Field::new("derived_shape", DataType::Utf8, false),
        Field::new("slack_enabled", DataType::Boolean, false),
        Field::new("slack_penalty", DataType::Float64, true),
    ])
}

/// Schema of the fitted FPHA hyperplane coefficients, mirroring the
/// `system/fpha_hyperplanes.parquet` input schema.
pub(crate) fn fpha_hyperplanes_schema() -> Schema {
    Schema::new(vec![
        Field::new("hydro_id", DataType::Int32, false),
        Field::new("stage_id", DataType::Int32, true),
        Field::new("plane_id", DataType::Int32, false),
        Field::new("gamma_0", DataType::Float64, false),
        Field::new("gamma_v", DataType::Float64, false),
        Field::new("gamma_q", DataType::Float64, false),
        Field::new("gamma_s", DataType::Float64, false),
        Field::new("kappa", DataType::Float64, true),
        Field::new("valid_v_min_hm3", DataType::Float64, true),
        Field::new("valid_v_max_hm3", DataType::Float64, true),
        Field::new("valid_q_max_m3s", DataType::Float64, true),
    ])
}

/// Schema of the per-hydro evaporation model coefficients.
pub(crate) fn evaporation_models_schema() -> Schema {
    Schema::new(vec![
        Field::new("hydro_id", DataType::Int32, false),
        Field::new("stage_id", DataType::Int32, true),
        Field::new("intercept_m3s", DataType::Float64, false),
        Field::new("volume_slope_m3s_per_hm3", DataType::Float64, false),
        Field::new("reference_volume_hm3", DataType::Float64, false),
        Field::new("source", DataType::Utf8, false),
    ])
}

/// Schema of the per-(hydro, stage) FPHA fit deviation diagnostics.
pub(crate) fn fpha_deviation_points_schema() -> Schema {
    Schema::new(vec![
        Field::new("hydro_id", DataType::Int32, false),
        Field::new("stage_id", DataType::Int32, true),
        Field::new("v", DataType::Float64, false),
        Field::new("q", DataType::Float64, false),
        Field::new("fph_exact", DataType::Float64, false),
        Field::new("fpha_fitted", DataType::Float64, false),
        Field::new("deviation", DataType::Float64, false),
        Field::new("relative", DataType::Float64, false),
    ])
}

/// Schema of the per-(stage, opening, entity) noise realizations.
pub(crate) fn noise_openings_schema() -> Schema {
    Schema::new(vec![
        Field::new("stage_id", DataType::Int32, false),
        Field::new("opening_index", DataType::UInt32, false),
        Field::new("entity_index", DataType::UInt32, false),
        Field::new("value", DataType::Float64, false),
    ])
}

/// Schema of the fitted per-(hydro, stage) seasonal inflow statistics.
pub(crate) fn inflow_seasonal_stats_schema() -> Schema {
    Schema::new(vec![
        Field::new("hydro_id", DataType::Int32, false),
        Field::new("stage_id", DataType::Int32, false),
        Field::new("mean_m3s", DataType::Float64, false),
        Field::new("std_m3s", DataType::Float64, false),
    ])
}

/// Schema of the fitted per-(hydro, stage, lag) AR coefficients.
pub(crate) fn inflow_ar_coefficients_schema() -> Schema {
    Schema::new(vec![
        Field::new("hydro_id", DataType::Int32, false),
        Field::new("stage_id", DataType::Int32, false),
        Field::new("lag", DataType::Int32, false),
        Field::new("coefficient", DataType::Float64, false),
    ])
}

/// Schema of the fitted per-(hydro, stage) annual inflow component.
pub(crate) fn inflow_annual_component_schema() -> Schema {
    Schema::new(vec![
        Field::new("hydro_id", DataType::Int32, false),
        Field::new("stage_id", DataType::Int32, false),
        Field::new("annual_coefficient", DataType::Float64, false),
        Field::new("annual_mean_m3s", DataType::Float64, false),
        Field::new("annual_std_m3s", DataType::Float64, false),
    ])
}

/// Schema of the fitted per-(bus, stage) seasonal load statistics.
pub(crate) fn load_seasonal_stats_schema() -> Schema {
    Schema::new(vec![
        Field::new("bus_id", DataType::Int32, false),
        Field::new("stage_id", DataType::Int32, false),
        Field::new("mean_mw", DataType::Float64, false),
        Field::new("std_mw", DataType::Float64, false),
    ])
}

/// Schema of the per-entity, per-stage resolved bound values.
pub(crate) fn bounds_schema() -> Schema {
    Schema::new(vec![
        Field::new("entity_type_code", DataType::Int8, false),
        Field::new("entity_id", DataType::Int32, false),
        Field::new("hydro_id", DataType::Int32, true),
        Field::new("stage_id", DataType::Int32, false),
        Field::new("block_id", DataType::Int32, true),
        Field::new("bound_type_code", DataType::Int8, false),
        Field::new("bound_value", DataType::Float64, false),
    ])
}

/// One row of the crate-internal registry of every output-schema function: a
/// schema's constructor paired with its registry name and its optional
/// `training/dictionaries/variables.csv` grouping label. Output paths live in
/// [`OUTPUT_FILES`](super::file_registry::OUTPUT_FILES), whose rows name their
/// backing schema by `name`.
pub(crate) struct SchemaRegistryEntry {
    /// Registry name, unique across rows; keys per-column metadata such as
    /// units.
    pub(crate) name: &'static str,
    /// `file` column label in `variables.csv`, or `None` when this schema is
    /// not enumerated there.
    pub(crate) csv_label: Option<&'static str>,
    /// The schema constructor itself.
    pub(crate) schema_fn: fn() -> Schema,
}

/// The single owner of the output-schema family: every `*_schema()` function
/// in this module appears here exactly once, with its registry name and its
/// optional `variables.csv` label. The gate test
/// `one_spelling_per_axis_across_every_output_schema`,
/// `dictionary::variables_csv_schemas` and the output-path owner
/// [`OUTPUT_FILES`](super::file_registry::OUTPUT_FILES) all derive from this
/// table rather than keeping their own copies of the schema set.
pub(crate) const OUTPUT_SCHEMAS: &[SchemaRegistryEntry] = &[
    SchemaRegistryEntry {
        name: "costs",
        csv_label: Some("costs"),
        schema_fn: build_costs_schema,
    },
    SchemaRegistryEntry {
        name: "hydros",
        csv_label: Some("hydros"),
        schema_fn: build_hydros_schema,
    },
    SchemaRegistryEntry {
        name: "hydro_bus_generation",
        csv_label: Some("hydro_bus_generation"),
        schema_fn: build_hydro_bus_generation_schema,
    },
    SchemaRegistryEntry {
        name: "thermals",
        csv_label: Some("thermals"),
        schema_fn: build_thermals_schema,
    },
    SchemaRegistryEntry {
        name: "exchanges",
        csv_label: Some("exchanges"),
        schema_fn: build_exchanges_schema,
    },
    SchemaRegistryEntry {
        name: "buses",
        csv_label: Some("buses"),
        schema_fn: build_buses_schema,
    },
    SchemaRegistryEntry {
        name: "pumping_stations",
        csv_label: Some("pumping_stations"),
        schema_fn: build_pumping_stations_schema,
    },
    SchemaRegistryEntry {
        name: "contracts",
        csv_label: Some("contracts"),
        schema_fn: build_contracts_schema,
    },
    SchemaRegistryEntry {
        name: "non_controllables",
        csv_label: Some("non_controllables"),
        schema_fn: build_non_controllables_schema,
    },
    SchemaRegistryEntry {
        name: "inflow_lags",
        csv_label: Some("inflow_lags"),
        schema_fn: build_inflow_lags_schema,
    },
    SchemaRegistryEntry {
        name: "in_transit",
        csv_label: Some("in_transit"),
        schema_fn: build_in_transit_schema,
    },
    SchemaRegistryEntry {
        name: "transit_seed",
        csv_label: Some("transit_seed"),
        schema_fn: build_transit_seed_schema,
    },
    SchemaRegistryEntry {
        name: "generic_violations",
        csv_label: Some("generic_violations"),
        schema_fn: build_generic_violations_schema,
    },
    SchemaRegistryEntry {
        name: "paths",
        csv_label: Some("paths"),
        schema_fn: paths_schema,
    },
    SchemaRegistryEntry {
        name: "scenario_summary",
        csv_label: Some("scenario_summary"),
        schema_fn: scenario_summary_schema,
    },
    SchemaRegistryEntry {
        name: "convergence",
        csv_label: Some("convergence"),
        schema_fn: convergence_schema,
    },
    SchemaRegistryEntry {
        name: "iteration_timing",
        csv_label: Some("iteration_timing"),
        schema_fn: iteration_timing_schema,
    },
    SchemaRegistryEntry {
        name: "row_selection",
        csv_label: Some("cut_selection"),
        schema_fn: row_selection_schema,
    },
    SchemaRegistryEntry {
        name: "solver_iterations",
        csv_label: Some("solver_iterations"),
        schema_fn: solver_iterations_schema,
    },
    SchemaRegistryEntry {
        name: "retry_histogram",
        csv_label: Some("retry_histogram"),
        schema_fn: retry_histogram_schema,
    },
    SchemaRegistryEntry {
        name: "fixed_delivery",
        csv_label: None,
        schema_fn: fixed_delivery_schema,
    },
    SchemaRegistryEntry {
        name: "anticipated_lanes",
        csv_label: None,
        schema_fn: build_anticipated_lanes_schema,
    },
    SchemaRegistryEntry {
        name: "generic_constraint_echo",
        csv_label: None,
        schema_fn: generic_constraint_echo_schema,
    },
    SchemaRegistryEntry {
        name: "bounds",
        csv_label: None,
        schema_fn: bounds_schema,
    },
    SchemaRegistryEntry {
        name: "fpha_hyperplanes",
        csv_label: None,
        schema_fn: fpha_hyperplanes_schema,
    },
    SchemaRegistryEntry {
        name: "evaporation_models",
        csv_label: None,
        schema_fn: evaporation_models_schema,
    },
    SchemaRegistryEntry {
        name: "fpha_deviation_points",
        csv_label: None,
        schema_fn: fpha_deviation_points_schema,
    },
    SchemaRegistryEntry {
        name: "noise_openings",
        csv_label: None,
        schema_fn: noise_openings_schema,
    },
    SchemaRegistryEntry {
        name: "inflow_seasonal_stats",
        csv_label: None,
        schema_fn: inflow_seasonal_stats_schema,
    },
    SchemaRegistryEntry {
        name: "inflow_ar_coefficients",
        csv_label: None,
        schema_fn: inflow_ar_coefficients_schema,
    },
    SchemaRegistryEntry {
        name: "inflow_annual_component",
        csv_label: None,
        schema_fn: inflow_annual_component_schema,
    },
    SchemaRegistryEntry {
        name: "load_seasonal_stats",
        csv_label: None,
        schema_fn: load_seasonal_stats_schema,
    },
];

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::expect_used)]
mod tests {
    use super::*;
    use arrow::datatypes::DataType;

    fn field_names(schema: &Schema) -> Vec<&str> {
        schema.fields().iter().map(|f| f.name().as_str()).collect()
    }

    fn field_type(schema: &Schema, name: &str) -> DataType {
        schema
            .field_with_name(name)
            .unwrap_or_else(|_| panic!("field '{name}' not found in schema"))
            .data_type()
            .clone()
    }

    fn is_nullable(schema: &Schema, name: &str) -> bool {
        schema
            .field_with_name(name)
            .unwrap_or_else(|_| panic!("field '{name}' not found in schema"))
            .is_nullable()
    }

    #[test]
    fn costs_schema_field_count_and_names() {
        let schema = costs_schema();
        assert_eq!(
            schema.fields().len(),
            29,
            "costs schema must have 29 fields"
        );
        let names = field_names(&schema);
        assert_eq!(
            names,
            vec![
                "scenario_id",
                "stage_id",
                "node_id",
                "block_id",
                "total_cost",
                "immediate_cost",
                "future_cost",
                "discount_factor",
                "thermal_cost",
                "anticipated_thermal_cost",
                "contract_cost",
                "deficit_cost",
                "excess_cost",
                "storage_violation_cost",
                "filling_target_cost",
                "hydro_violation_cost",
                "outflow_violation_below_cost",
                "outflow_violation_above_cost",
                "turbined_violation_cost",
                "generation_violation_cost",
                "evaporation_violation_cost",
                "withdrawal_violation_cost",
                "inflow_penalty_cost",
                "generic_violation_cost",
                "spillage_cost",
                "turbined_cost",
                "curtailment_cost",
                "exchange_cost",
                "pumping_cost",
            ]
        );
    }

    #[test]
    fn costs_schema_types_and_nullability() {
        let schema = costs_schema();
        // scenario_id / stage_id / node_id: i32, not nullable (the axis prefix)
        for col in &["scenario_id", "stage_id", "node_id"] {
            assert_eq!(field_type(&schema, col), DataType::Int32);
            assert!(!is_nullable(&schema, col));
        }
        // block_id: i32, nullable
        assert_eq!(field_type(&schema, "block_id"), DataType::Int32);
        assert!(is_nullable(&schema, "block_id"));
        // all cost columns: f64, not nullable
        for col in &[
            "total_cost",
            "immediate_cost",
            "future_cost",
            "discount_factor",
            "thermal_cost",
            "anticipated_thermal_cost",
            "contract_cost",
            "deficit_cost",
            "excess_cost",
            "storage_violation_cost",
            "filling_target_cost",
            "hydro_violation_cost",
            "outflow_violation_below_cost",
            "outflow_violation_above_cost",
            "turbined_violation_cost",
            "generation_violation_cost",
            "evaporation_violation_cost",
            "withdrawal_violation_cost",
            "inflow_penalty_cost",
            "generic_violation_cost",
            "spillage_cost",
            "turbined_cost",
            "curtailment_cost",
            "exchange_cost",
            "pumping_cost",
        ] {
            assert_eq!(
                field_type(&schema, col),
                DataType::Float64,
                "column {col} must be Float64"
            );
            assert!(
                !is_nullable(&schema, col),
                "column {col} must not be nullable"
            );
        }
    }

    #[test]
    fn hydros_schema_field_count_and_names() {
        let schema = hydros_schema();
        assert_eq!(
            schema.fields().len(),
            41,
            "hydros schema must have 41 fields"
        );
        let names = field_names(&schema);
        assert_eq!(
            names,
            vec![
                "scenario_id",
                "stage_id",
                "node_id",
                "block_id",
                "hydro_id",
                "turbined_m3s",
                "spillage_m3s",
                "outflow_m3s",
                "evaporation_m3s",
                "diverted_inflow_m3s",
                "diverted_outflow_m3s",
                "incremental_inflow_m3s",
                "inflow_m3s",
                "storage_initial_hm3",
                "storage_final_hm3",
                "generation_mw",
                "generation_mwh",
                "equivalent_productivity_mw_per_m3s",
                "accumulated_productivity_mw_per_m3s",
                "incremental_inflow_energy_mw",
                "stored_energy_initial_mwh",
                "stored_energy_final_mwh",
                "spillage_cost",
                "water_value_per_hm3",
                "storage_binding_code",
                "operative_state_code",
                "turbined_slack_m3s",
                "outflow_slack_below_m3s",
                "outflow_slack_above_m3s",
                "generation_slack_mw",
                "storage_violation_below_hm3",
                "filling_target_violation_hm3",
                "evaporation_violation_pos_m3s",
                "evaporation_violation_neg_m3s",
                "inflow_nonnegativity_slack_m3s",
                "water_withdrawal_violation_pos_m3s",
                "water_withdrawal_violation_neg_m3s",
                "integrated_equivalent_productivity_mw_per_m3s",
                "integrated_accumulated_productivity_mw_per_m3s",
                "stored_energy_initial_mw",
                "stored_energy_final_mw",
            ]
        );
    }

    #[test]
    fn hydros_schema_nullable_fields() {
        let schema = hydros_schema();
        for col in &[
            "block_id",
            "evaporation_m3s",
            "diverted_inflow_m3s",
            "diverted_outflow_m3s",
        ] {
            assert!(is_nullable(&schema, col), "column {col} must be nullable");
        }
        for col in &[
            "scenario_id",
            "stage_id",
            "node_id",
            "hydro_id",
            "turbined_m3s",
            "spillage_m3s",
            "outflow_m3s",
            "incremental_inflow_m3s",
            "inflow_m3s",
            "storage_initial_hm3",
            "storage_final_hm3",
            "generation_mw",
            "generation_mwh",
            "equivalent_productivity_mw_per_m3s",
            "accumulated_productivity_mw_per_m3s",
            "incremental_inflow_energy_mw",
            "stored_energy_initial_mwh",
            "stored_energy_final_mwh",
            "spillage_cost",
            "water_value_per_hm3",
            "storage_binding_code",
            "operative_state_code",
            "turbined_slack_m3s",
            "outflow_slack_below_m3s",
            "outflow_slack_above_m3s",
            "generation_slack_mw",
            "storage_violation_below_hm3",
            "filling_target_violation_hm3",
            "evaporation_violation_pos_m3s",
            "evaporation_violation_neg_m3s",
            "inflow_nonnegativity_slack_m3s",
            "water_withdrawal_violation_pos_m3s",
            "water_withdrawal_violation_neg_m3s",
            "integrated_equivalent_productivity_mw_per_m3s",
            "integrated_accumulated_productivity_mw_per_m3s",
            "stored_energy_initial_mw",
            "stored_energy_final_mw",
        ] {
            assert!(
                !is_nullable(&schema, col),
                "column {col} must not be nullable"
            );
        }
    }

    #[test]
    fn hydro_bus_generation_schema_field_count_and_names() {
        let schema = hydro_bus_generation_schema();
        assert_eq!(
            schema.fields().len(),
            9,
            "hydro_bus_generation schema must have 9 fields"
        );
        let names = field_names(&schema);
        assert_eq!(
            names,
            vec![
                "scenario_id",
                "stage_id",
                "node_id",
                "block_id",
                "hydro_id",
                "bus_id",
                "turbined_m3s",
                "generation_mw",
                "generation_mwh",
            ]
        );
        assert!(
            is_nullable(&schema, "block_id"),
            "block_id must be nullable"
        );
        for col in &[
            "scenario_id",
            "stage_id",
            "node_id",
            "hydro_id",
            "bus_id",
            "turbined_m3s",
            "generation_mw",
            "generation_mwh",
        ] {
            assert!(
                !is_nullable(&schema, col),
                "column {col} must not be nullable"
            );
        }
        for col in &[
            "scenario_id",
            "stage_id",
            "node_id",
            "block_id",
            "hydro_id",
            "bus_id",
        ] {
            assert_eq!(field_type(&schema, col), DataType::Int32);
        }
        for col in &["turbined_m3s", "generation_mw", "generation_mwh"] {
            assert_eq!(field_type(&schema, col), DataType::Float64);
        }
    }

    #[test]
    fn thermals_schema_field_count() {
        let schema = thermals_schema();
        assert_eq!(
            schema.fields().len(),
            12,
            "thermals schema must have 12 fields"
        );
    }

    #[test]
    fn thermals_schema_anticipated_fields_nullable() {
        let schema = thermals_schema();
        assert!(is_nullable(&schema, "anticipated_committed_mw"));
        assert!(is_nullable(&schema, "anticipated_decision_mw"));
        assert!(!is_nullable(&schema, "is_anticipated"));
        assert_eq!(field_type(&schema, "is_anticipated"), DataType::Boolean);
        assert_eq!(field_type(&schema, "operative_state_code"), DataType::Int8);
    }

    #[test]
    fn exchanges_schema_field_count() {
        let schema = exchanges_schema();
        assert_eq!(
            schema.fields().len(),
            13,
            "exchanges schema must have 13 fields"
        );
    }

    #[test]
    fn buses_schema_field_count() {
        let schema = buses_schema();
        assert_eq!(
            schema.fields().len(),
            12,
            "buses schema must have 12 fields"
        );
    }

    #[test]
    fn pumping_stations_schema_field_count() {
        let schema = pumping_stations_schema();
        assert_eq!(
            schema.fields().len(),
            11,
            "pumping_stations schema must have 11 fields"
        );
    }

    #[test]
    fn contracts_schema_field_count() {
        let schema = contracts_schema();
        assert_eq!(
            schema.fields().len(),
            10,
            "contracts schema must have 10 fields"
        );
    }

    #[test]
    fn non_controllables_schema_field_count() {
        let schema = non_controllables_schema();
        assert_eq!(
            schema.fields().len(),
            12,
            "non_controllables schema must have 12 fields"
        );
    }

    #[test]
    fn inflow_lags_schema_field_count() {
        let schema = inflow_lags_schema();
        assert_eq!(
            schema.fields().len(),
            6,
            "inflow_lags schema must have 6 fields"
        );
    }

    #[test]
    fn inflow_lags_schema_all_non_nullable() {
        let schema = inflow_lags_schema();
        for field in schema.fields() {
            assert!(
                !field.is_nullable(),
                "inflow_lags field '{}' must not be nullable",
                field.name()
            );
        }
    }

    #[test]
    fn transit_seed_schema_field_count() {
        let schema = transit_seed_schema();
        assert_eq!(
            schema.fields().len(),
            5,
            "transit_seed schema must have 5 fields"
        );
    }

    #[test]
    fn transit_seed_schema_all_non_nullable() {
        let schema = transit_seed_schema();
        for field in schema.fields() {
            assert!(
                !field.is_nullable(),
                "transit_seed field '{}' must not be nullable",
                field.name()
            );
        }
    }

    #[test]
    fn transit_seed_schema_carries_no_stage_or_node_axis() {
        let schema = transit_seed_schema();
        assert!(
            schema.field_with_name("stage_id").is_err()
                && schema.field_with_name("node_id").is_err(),
            "transit_seed is scenario-level; it must not carry the stage/node row prefix"
        );
    }

    #[test]
    fn fixed_delivery_schema_field_names_and_types() {
        let schema = fixed_delivery_schema();
        assert_eq!(
            field_names(&schema),
            vec!["thermal_id", "start_date", "end_date", "value_mw"]
        );
        assert_eq!(field_type(&schema, "thermal_id"), DataType::Int32);
        assert_eq!(field_type(&schema, "start_date"), DataType::Date32);
        assert_eq!(field_type(&schema, "end_date"), DataType::Date32);
        assert_eq!(field_type(&schema, "value_mw"), DataType::Float64);
    }

    #[test]
    fn fixed_delivery_schema_all_non_nullable() {
        let schema = fixed_delivery_schema();
        for field in schema.fields() {
            assert!(
                !field.is_nullable(),
                "fixed_delivery field '{}' must not be nullable",
                field.name()
            );
        }
    }

    #[test]
    fn fixed_delivery_schema_carries_no_scenario_or_stage_axis() {
        let schema = fixed_delivery_schema();
        assert!(
            schema.field_with_name("scenario_id").is_err()
                && schema.field_with_name("stage_id").is_err(),
            "fixed_delivery is run-level; it must not carry the scenario/stage row prefix"
        );
    }

    #[test]
    fn fixed_delivery_schema_carries_no_cost_or_energy_column() {
        let schema = fixed_delivery_schema();
        for field in schema.fields() {
            assert!(
                !field.name().contains("cost"),
                "fixed_delivery must book no cost (§7): field '{}'",
                field.name()
            );
            assert!(
                !field.name().contains("energy"),
                "fixed_delivery carries MW only: field '{}'",
                field.name()
            );
        }
    }

    #[test]
    fn generic_violations_schema_field_count() {
        let schema = generic_violations_schema();
        assert_eq!(
            schema.fields().len(),
            7,
            "generic_violations schema must have 7 fields"
        );
    }

    #[test]
    fn paths_schema_is_three_non_null_int32_axis_columns() {
        let schema = paths_schema();
        let names = field_names(&schema);
        assert_eq!(
            names,
            vec!["scenario_id", "stage_id", "node_id"],
            "paths.parquet is exactly the (scenario_id, stage_id, node_id) axis prefix"
        );
        for col in &["scenario_id", "stage_id", "node_id"] {
            assert_eq!(field_type(&schema, col), DataType::Int32);
            assert!(!is_nullable(&schema, col), "{col} must be non-null");
        }
    }

    #[test]
    fn scenario_summary_schema_field_count_names_and_nullability() {
        let schema = scenario_summary_schema();
        assert_eq!(
            schema.fields().len(),
            3,
            "scenario_summary schema must have 3 fields"
        );
        let names = field_names(&schema);
        assert_eq!(
            names,
            vec!["scenario_id", "probability", "discounted_immediate_cost"]
        );
        assert!(
            schema.field_with_name("total_cost").is_err(),
            "no column may be named total_cost"
        );
        assert_eq!(field_type(&schema, "scenario_id"), DataType::Int32);
        assert!(!is_nullable(&schema, "scenario_id"));
        assert_eq!(field_type(&schema, "probability"), DataType::Float64);
        assert!(is_nullable(&schema, "probability"));
        assert_eq!(
            field_type(&schema, "discounted_immediate_cost"),
            DataType::Float64
        );
        assert!(!is_nullable(&schema, "discounted_immediate_cost"));
    }

    #[test]
    fn convergence_schema_field_count_and_types() {
        let schema = convergence_schema();
        assert_eq!(
            schema.fields().len(),
            15,
            "convergence schema must have 15 fields"
        );
        assert_eq!(field_type(&schema, "iteration"), DataType::Int32);
        assert_eq!(field_type(&schema, "lower_bound"), DataType::Float64);
        assert_eq!(field_type(&schema, "upper_bound"), DataType::Float64);
        assert_eq!(field_type(&schema, "upper_bound_kind"), DataType::Utf8);
        assert_eq!(field_type(&schema, "cuts_added"), DataType::Int32);
        assert_eq!(field_type(&schema, "cuts_active"), DataType::Int64);
        assert_eq!(field_type(&schema, "time_forward_ms"), DataType::Int64);
        assert_eq!(field_type(&schema, "lp_solves"), DataType::Int64);
        assert_eq!(field_type(&schema, "forward_passes"), DataType::Int32);
    }

    #[test]
    fn convergence_schema_nullable_fields() {
        let schema = convergence_schema();
        // gap_percent is nullable (None when LB <= 0); upper_bound_std is nullable
        // (NULL under an exact bound).
        assert!(is_nullable(&schema, "gap_percent"));
        assert!(is_nullable(&schema, "upper_bound_std"));
        for name in &[
            "iteration",
            "lower_bound",
            "upper_bound",
            "upper_bound_kind",
            "cuts_added",
            "cuts_removed",
            "cuts_active",
            "time_forward_ms",
            "time_backward_ms",
            "time_total_ms",
            "forward_passes",
            "lp_solves",
        ] {
            assert!(
                !is_nullable(&schema, name),
                "column {name} must not be nullable"
            );
        }
    }

    #[test]
    fn iteration_timing_schema_field_count() {
        let schema = iteration_timing_schema();
        assert_eq!(
            schema.fields().len(),
            19,
            "iteration_timing schema must have 19 fields"
        );
    }

    #[test]
    fn iteration_timing_schema_rank_worker_nullable() {
        let schema = iteration_timing_schema();
        let rank_field = schema
            .field_with_name("rank")
            .expect("rank field must exist");
        assert!(rank_field.is_nullable(), "rank must be nullable");
        let worker_id_field = schema
            .field_with_name("worker_id")
            .expect("worker_id field must exist");
        assert!(worker_id_field.is_nullable(), "worker_id must be nullable");
        for field in schema.fields() {
            if field.name() != "rank" && field.name() != "worker_id" {
                assert!(
                    !field.is_nullable(),
                    "iteration_timing field '{}' must not be nullable",
                    field.name()
                );
            }
        }
    }

    #[test]
    fn row_selection_schema_field_count_and_types() {
        let schema = row_selection_schema();
        assert_eq!(
            schema.fields().len(),
            10,
            "cut_selection schema must have 10 fields"
        );
        for field in &schema.fields()[..7] {
            assert_eq!(field.data_type(), &DataType::Int32);
            assert!(!field.is_nullable());
        }
        assert_eq!(schema.fields()[7].name(), "selection_time_ms");
        assert_eq!(schema.fields()[7].data_type(), &DataType::Float64);
        assert!(!schema.fields()[7].is_nullable());
        for &name in &["budget_evicted", "active_after_budget"] {
            let field = schema
                .field_with_name(name)
                .unwrap_or_else(|_| panic!("field '{name}' not found"));
            assert_eq!(
                field.data_type(),
                &DataType::Int32,
                "field '{name}' must be Int32"
            );
            assert!(field.is_nullable(), "field '{name}' must be nullable");
        }
        assert!(
            schema.field_with_name("cuts_in_lp").is_err(),
            "cuts_in_lp must not be present in schema"
        );
    }

    #[test]
    fn solver_iterations_schema_field_count_and_types() {
        let schema = solver_iterations_schema();
        assert_eq!(
            schema.fields().len(),
            19,
            "solver_iterations schema must have 19 fields"
        );
        let expected: &[(&str, DataType, bool)] = &[
            ("iteration", DataType::Int32, true),
            ("scenario_id", DataType::Int32, true),
            ("phase", DataType::Utf8, false),
            ("stage_id", DataType::Int32, true),
            ("opening_index", DataType::Int32, true),
            ("rank", DataType::Int32, true),
            ("worker_id", DataType::Int32, true),
            ("lp_solves", DataType::UInt32, false),
            ("lp_successes", DataType::UInt32, false),
            ("lp_retries", DataType::UInt32, false),
            ("lp_failures", DataType::UInt32, false),
            ("retry_attempts", DataType::UInt32, false),
            ("basis_offered", DataType::UInt32, false),
            ("basis_consistency_failures", DataType::UInt32, false),
            ("simplex_iterations", DataType::UInt64, false),
            ("solve_time_ms", DataType::Float64, false),
            ("load_model_time_ms", DataType::Float64, false),
            ("set_bounds_time_ms", DataType::Float64, false),
            ("basis_set_time_ms", DataType::Float64, false),
        ];
        for (i, (name, dtype, nullable)) in expected.iter().enumerate() {
            let field = &schema.fields()[i];
            assert_eq!(field.name(), name, "field {i} name mismatch");
            assert_eq!(field.data_type(), dtype, "field {i} ({name}) type mismatch");
            assert_eq!(
                field.is_nullable(),
                *nullable,
                "field {i} ({name}) nullability mismatch"
            );
        }
    }

    #[test]
    fn retry_histogram_schema_field_count_and_types() {
        let schema = retry_histogram_schema();
        assert_eq!(
            schema.fields().len(),
            5,
            "retry_histogram schema must have 5 fields"
        );
        let expected: &[(&str, DataType, bool)] = &[
            ("iteration", DataType::UInt32, false),
            ("phase", DataType::Utf8, false),
            ("stage_id", DataType::Int32, true),
            ("retry_level", DataType::UInt32, false),
            ("count", DataType::UInt64, false),
        ];
        for (i, (name, dtype, nullable)) in expected.iter().enumerate() {
            let field = &schema.fields()[i];
            assert_eq!(field.name(), name, "field {i} name mismatch");
            assert_eq!(field.data_type(), dtype, "field {i} ({name}) type mismatch");
            assert_eq!(
                field.is_nullable(),
                *nullable,
                "field {i} ({name}) nullability mismatch"
            );
        }
    }

    #[test]
    fn all_schema_functions_return_valid_schemas() {
        let schemas: Vec<(Arc<Schema>, &str)> = vec![
            (costs_schema(), "costs"),
            (hydros_schema(), "hydros"),
            (hydro_bus_generation_schema(), "hydro_bus_generation"),
            (thermals_schema(), "thermals"),
            (exchanges_schema(), "exchanges"),
            (buses_schema(), "buses"),
            (pumping_stations_schema(), "pumping_stations"),
            (contracts_schema(), "contracts"),
            (non_controllables_schema(), "non_controllables"),
            (inflow_lags_schema(), "inflow_lags"),
            (in_transit_schema(), "in_transit"),
            (generic_violations_schema(), "generic_violations"),
            (Arc::new(paths_schema()), "paths"),
            (Arc::new(convergence_schema()), "convergence"),
            (Arc::new(iteration_timing_schema()), "iteration_timing"),
            (Arc::new(row_selection_schema()), "cut_selection"),
            (Arc::new(solver_iterations_schema()), "solver_iterations"),
            (Arc::new(retry_histogram_schema()), "retry_histogram"),
            (
                Arc::new(generic_constraint_echo_schema()),
                "generic_constraint_echo",
            ),
        ];
        for (schema, name) in &schemas {
            assert!(
                !schema.fields().is_empty(),
                "schema '{name}' must have at least one field"
            );
        }
        let counts: Vec<(&str, usize)> = schemas
            .iter()
            .map(|(s, n)| (*n, s.fields().len()))
            .collect();
        let expected: &[(&str, usize)] = &[
            ("costs", 29),
            ("hydros", 41),
            ("hydro_bus_generation", 9),
            ("thermals", 12),
            ("exchanges", 13),
            ("buses", 12),
            ("pumping_stations", 11),
            ("contracts", 10),
            ("non_controllables", 12),
            ("inflow_lags", 6),
            ("in_transit", 7),
            ("generic_violations", 7),
            ("paths", 3),
            ("convergence", 15),
            ("iteration_timing", 19),
            ("cut_selection", 10),
            ("solver_iterations", 19),
            ("retry_histogram", 5),
            ("generic_constraint_echo", 13),
        ];
        for ((name, actual), (_, exp)) in counts.iter().zip(expected.iter()) {
            assert_eq!(
                actual, exp,
                "schema '{name}' field count: expected {exp}, got {actual}"
            );
        }
    }

    #[test]
    fn one_spelling_per_axis_across_every_output_schema() {
        // Every output parquet spells each axis with a single canonical name.
        // A renamed axis's OLD spelling must never reappear in any schema, and a
        // later file cannot reintroduce a variant without failing this one test.
        let names: Vec<String> = OUTPUT_SCHEMAS
            .iter()
            .flat_map(|entry| {
                (entry.schema_fn)()
                    .fields()
                    .iter()
                    .map(|f| f.name().clone())
                    .collect::<Vec<_>>()
            })
            .collect();

        // Forbidden variant spellings, each superseded by one canonical axis.
        let forbidden = [
            ("stage", "stage_id"),
            ("opening", "opening_index"),
            ("upper_bound_mean", "upper_bound"),
        ];
        for (variant, canonical) in forbidden {
            assert!(
                !names.iter().any(|n| n == variant),
                "forbidden axis spelling '{variant}' present; use '{canonical}'"
            );
        }

        // Each canonical axis must appear at least once so the gate has power.
        for canonical in [
            "iteration",
            "scenario_id",
            "stage_id",
            "node_id",
            "opening_index",
            "block_id",
        ] {
            assert!(
                names.iter().any(|n| n == canonical),
                "canonical axis '{canonical}' must be spelled somewhere in the family"
            );
        }
    }

    /// `OUTPUT_SCHEMAS` is the single owner of the output-schema family: every
    /// `*_schema()` function in this module must have exactly one row. This
    /// count is a manual pin, not a reflection-derived count (Rust has no
    /// runtime enumeration of a module's functions) — adding or removing a
    /// schema function must add or remove its row and update this count in
    /// the same change.
    #[test]
    fn output_schema_registry_has_no_duplicate_or_missing_rows() {
        const EXPECTED_SCHEMA_COUNT: usize = 32;
        assert_eq!(
            OUTPUT_SCHEMAS.len(),
            EXPECTED_SCHEMA_COUNT,
            "OUTPUT_SCHEMAS must list every *_schema() function in this module exactly once"
        );

        let mut seen_fns: Vec<usize> = Vec::with_capacity(OUTPUT_SCHEMAS.len());
        for entry in OUTPUT_SCHEMAS {
            let fn_ptr = entry.schema_fn as usize;
            assert!(
                !seen_fns.contains(&fn_ptr),
                "a schema fn appears more than once in OUTPUT_SCHEMAS"
            );
            seen_fns.push(fn_ptr);
        }

        let mut seen_names: Vec<&str> = Vec::with_capacity(OUTPUT_SCHEMAS.len());
        for entry in OUTPUT_SCHEMAS {
            assert!(
                !entry.name.is_empty(),
                "every OUTPUT_SCHEMAS row must carry a non-empty name"
            );
            assert!(
                !seen_names.contains(&entry.name),
                "schema name '{}' appears more than once in OUTPUT_SCHEMAS",
                entry.name
            );
            seen_names.push(entry.name);
        }
    }

    #[test]
    fn generic_constraint_echo_schema_field_count_and_names() {
        let schema = generic_constraint_echo_schema();
        assert_eq!(
            schema.fields().len(),
            13,
            "generic_constraint_echo schema must have 13 fields"
        );
        let expected: &[(&str, DataType, bool)] = &[
            ("stage_id", DataType::Int32, false),
            ("block_id", DataType::Int32, true),
            ("constraint_id", DataType::Int32, false),
            ("constraint_name", DataType::Utf8, false),
            ("term_index", DataType::Int32, true),
            ("variable_kind", DataType::Utf8, true),
            ("variable", DataType::Utf8, true),
            ("coefficient", DataType::Float64, true),
            ("bound_lower", DataType::Float64, true),
            ("bound_upper", DataType::Float64, true),
            ("derived_shape", DataType::Utf8, false),
            ("slack_enabled", DataType::Boolean, false),
            ("slack_penalty", DataType::Float64, true),
        ];
        for (i, (name, dtype, nullable)) in expected.iter().enumerate() {
            let field = &schema.fields()[i];
            assert_eq!(field.name(), name, "field {i} name mismatch");
            assert_eq!(field.data_type(), dtype, "field {i} ({name}) type mismatch");
            assert_eq!(
                field.is_nullable(),
                *nullable,
                "field {i} ({name}) nullability mismatch"
            );
        }
    }
}
