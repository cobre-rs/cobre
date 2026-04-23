//! Arrow schema definitions for all Parquet output files per output-schemas spec
//! (SS5.1–5.11 and SS6.1–6.3).

use arrow::datatypes::{DataType, Field, Schema};

/// Schema for `simulation/costs/` — stage and block-level cost breakdown.
///
/// 20 fields. `block_id` is nullable. See output-schemas.md SS5.1.
pub(crate) fn costs_schema() -> Schema {
    Schema::new(vec![
        Field::new("stage_id", DataType::Int32, false),
        Field::new("block_id", DataType::Int32, true),
        Field::new("total_cost", DataType::Float64, false),
        Field::new("immediate_cost", DataType::Float64, false),
        Field::new("future_cost", DataType::Float64, false),
        Field::new("discount_factor", DataType::Float64, false),
        Field::new("thermal_cost", DataType::Float64, false),
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
        Field::new("fpha_turbined_cost", DataType::Float64, false),
        Field::new("curtailment_cost", DataType::Float64, false),
        Field::new("exchange_cost", DataType::Float64, false),
        Field::new("pumping_cost", DataType::Float64, false),
    ])
}

/// Schema for `simulation/hydros/` — hydro plant dispatch results.
///
/// 31 fields. See output-schemas.md SS5.2.
pub(crate) fn hydros_schema() -> Schema {
    Schema::new(vec![
        Field::new("stage_id", DataType::Int32, false),
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
        Field::new("productivity_mw_per_m3s", DataType::Float64, true),
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
    ])
}

/// Schema for `simulation/thermals/` — thermal unit dispatch results.
///
/// 10 fields. See output-schemas.md SS5.3.
pub(crate) fn thermals_schema() -> Schema {
    Schema::new(vec![
        Field::new("stage_id", DataType::Int32, false),
        Field::new("block_id", DataType::Int32, true),
        Field::new("thermal_id", DataType::Int32, false),
        Field::new("generation_mw", DataType::Float64, false),
        Field::new("generation_mwh", DataType::Float64, false),
        Field::new("generation_cost", DataType::Float64, false),
        Field::new("is_gnl", DataType::Boolean, false),
        Field::new("gnl_committed_mw", DataType::Float64, true),
        Field::new("gnl_decision_mw", DataType::Float64, true),
        Field::new("operative_state_code", DataType::Int8, false),
    ])
}

/// Schema for `simulation/exchanges/` — transmission line flow results.
///
/// 11 fields. See output-schemas.md SS5.4.
pub(crate) fn exchanges_schema() -> Schema {
    Schema::new(vec![
        Field::new("stage_id", DataType::Int32, false),
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
    ])
}

/// Schema for `simulation/buses/` — bus load balance results.
///
/// 10 fields. See output-schemas.md SS5.5.
pub(crate) fn buses_schema() -> Schema {
    Schema::new(vec![
        Field::new("stage_id", DataType::Int32, false),
        Field::new("block_id", DataType::Int32, true),
        Field::new("bus_id", DataType::Int32, false),
        Field::new("load_mw", DataType::Float64, false),
        Field::new("load_mwh", DataType::Float64, false),
        Field::new("deficit_mw", DataType::Float64, false),
        Field::new("deficit_mwh", DataType::Float64, false),
        Field::new("excess_mw", DataType::Float64, false),
        Field::new("excess_mwh", DataType::Float64, false),
        Field::new("spot_price", DataType::Float64, false),
    ])
}

/// Schema for `simulation/pumping_stations/` — pumping station results.
///
/// 9 fields. See output-schemas.md SS5.6.
pub(crate) fn pumping_stations_schema() -> Schema {
    Schema::new(vec![
        Field::new("stage_id", DataType::Int32, false),
        Field::new("block_id", DataType::Int32, true),
        Field::new("pumping_station_id", DataType::Int32, false),
        Field::new("pumped_flow_m3s", DataType::Float64, false),
        Field::new("pumped_volume_hm3", DataType::Float64, false),
        Field::new("power_consumption_mw", DataType::Float64, false),
        Field::new("energy_consumption_mwh", DataType::Float64, false),
        Field::new("pumping_cost", DataType::Float64, false),
        Field::new("operative_state_code", DataType::Int8, false),
    ])
}

/// Schema for `simulation/contracts/` — energy contract results.
///
/// 8 fields. See output-schemas.md SS5.7.
pub(crate) fn contracts_schema() -> Schema {
    Schema::new(vec![
        Field::new("stage_id", DataType::Int32, false),
        Field::new("block_id", DataType::Int32, true),
        Field::new("contract_id", DataType::Int32, false),
        Field::new("power_mw", DataType::Float64, false),
        Field::new("energy_mwh", DataType::Float64, false),
        Field::new("price_per_mwh", DataType::Float64, false),
        Field::new("total_cost", DataType::Float64, false),
        Field::new("operative_state_code", DataType::Int8, false),
    ])
}

/// Schema for `simulation/non_controllables/` — non-controllable source results.
///
/// 10 fields. See output-schemas.md SS5.8.
pub(crate) fn non_controllables_schema() -> Schema {
    Schema::new(vec![
        Field::new("stage_id", DataType::Int32, false),
        Field::new("block_id", DataType::Int32, true),
        Field::new("non_controllable_id", DataType::Int32, false),
        Field::new("generation_mw", DataType::Float64, false),
        Field::new("generation_mwh", DataType::Float64, false),
        Field::new("available_mw", DataType::Float64, false),
        Field::new("curtailment_mw", DataType::Float64, false),
        Field::new("curtailment_mwh", DataType::Float64, false),
        Field::new("curtailment_cost", DataType::Float64, false),
        Field::new("operative_state_code", DataType::Int8, false),
    ])
}

/// Schema for `simulation/inflow_lags/` — autoregressive inflow state variables.
///
/// 4 fields. See output-schemas.md SS5.10.
pub(crate) fn inflow_lags_schema() -> Schema {
    Schema::new(vec![
        Field::new("stage_id", DataType::Int32, false),
        Field::new("hydro_id", DataType::Int32, false),
        Field::new("lag_index", DataType::Int32, false),
        Field::new("inflow_m3s", DataType::Float64, false),
    ])
}

/// Schema for `simulation/violations/generic/` — generic constraint violations.
///
/// 5 fields. See output-schemas.md SS5.11.
pub(crate) fn generic_violations_schema() -> Schema {
    Schema::new(vec![
        Field::new("stage_id", DataType::Int32, false),
        Field::new("block_id", DataType::Int32, true),
        Field::new("constraint_id", DataType::Int32, false),
        Field::new("slack_value", DataType::Float64, false),
        Field::new("slack_cost", DataType::Float64, false),
    ])
}

/// Schema for `training/convergence.parquet` — iteration-level convergence log.
///
/// 13 fields. See output-schemas.md SS6.1.
pub(crate) fn convergence_schema() -> Schema {
    Schema::new(vec![
        Field::new("iteration", DataType::Int32, false),
        Field::new("lower_bound", DataType::Float64, false),
        Field::new("upper_bound_mean", DataType::Float64, false),
        Field::new("upper_bound_std", DataType::Float64, false),
        Field::new("gap_percent", DataType::Float64, true),
        Field::new("cuts_added", DataType::Int32, false),
        Field::new("cuts_removed", DataType::Int32, false),
        Field::new("cuts_active", DataType::Int64, false),
        Field::new("time_forward_ms", DataType::Int64, false),
        Field::new("time_backward_ms", DataType::Int64, false),
        Field::new("time_total_ms", DataType::Int64, false),
        Field::new("forward_passes", DataType::Int32, false),
        Field::new("lp_solves", DataType::Int64, false),
    ])
}

/// Schema for `training/timing/iterations.parquet` — per-iteration timing breakdown.
///
/// 18 fields. Row semantics (T007): one row per `(iteration, rank)` for
/// rank-only sequential values (`worker_id = NULL`), and one row per
/// `(iteration, rank, worker_id)` for per-worker parallel-region values.
/// `SUM(col) GROUP BY iteration` recovers the pre-T007 single-row-per-iteration
/// value for each of the 16 timing columns.
///
/// Top-level non-overlapping phases: `forward_wall_ms`,
/// `backward_wall_ms`, `cut_selection_ms`, `mpi_allreduce_ms`,
/// `lower_bound_ms`. Sub-components of backward: `cut_sync_ms`,
/// `state_exchange_ms`, `cut_batch_build_ms`, `bwd_setup_ms`,
/// `bwd_load_imbalance_ms`, `bwd_scheduling_overhead_ms`.
/// Sub-components of forward: `fwd_setup_ms`, `fwd_load_imbalance_ms`,
/// `fwd_scheduling_overhead_ms`. Residual: `overhead_ms`.
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
    ])
}

/// Schema for `training/timing/mpi_ranks.parquet` — per-rank timing statistics.
///
/// 8 fields. See output-schemas.md SS6.3.
pub(crate) fn rank_timing_schema() -> Schema {
    Schema::new(vec![
        Field::new("iteration", DataType::Int32, false),
        Field::new("rank", DataType::Int32, false),
        Field::new("forward_time_ms", DataType::Int64, false),
        Field::new("backward_time_ms", DataType::Int64, false),
        Field::new("communication_time_ms", DataType::Int64, false),
        Field::new("idle_time_ms", DataType::Int64, false),
        Field::new("lp_solves", DataType::Int64, false),
        Field::new("scenarios_processed", DataType::Int32, false),
    ])
}

/// Schema for `training/solver/iterations.parquet` -- per-iteration, per-phase
/// solver statistics for diagnosing LP conditioning and retry behavior.
///
/// 19 columns. One row per (iteration, phase, stage, opening) tuple for
/// backward rows; forward, `lower_bound`, and simulation rows carry
/// `opening = NULL`. The `rank` and `worker_id` columns (positions 5 and 6,
/// 0-indexed) are `Int32 nullable`; they are `NULL` for rank-aggregated rows
/// and will carry real values once T005 (MPI allgatherv) is wired. Includes
/// one basis reconstruction column: `basis_reconstructions`.
pub(crate) fn solver_iterations_schema() -> Schema {
    Schema::new(vec![
        Field::new("iteration", DataType::UInt32, false),
        Field::new("phase", DataType::Utf8, false),
        Field::new("stage", DataType::Int32, false),
        Field::new("opening", DataType::Int32, true),
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
        Field::new("basis_reconstructions", DataType::UInt64, false),
    ])
}

/// Schema for `training/solver/retry_histogram.parquet` -- per-level retry
/// success counts, normalized from the solver iterations table.
///
/// 5 columns. One row per (iteration, phase, stage, `retry_level`) tuple where
/// `count > 0` (sparse encoding).
pub(crate) fn retry_histogram_schema() -> Schema {
    Schema::new(vec![
        Field::new("iteration", DataType::UInt32, false),
        Field::new("phase", DataType::Utf8, false),
        Field::new("stage", DataType::Int32, false),
        Field::new("retry_level", DataType::UInt32, false),
        Field::new("count", DataType::UInt64, false),
    ])
}

/// Schema for `training/cut_selection/iterations.parquet` — per-stage
/// row-selection statistics.
///
/// 9 fields. One row per (iteration, stage) pair. The two nullable Int32
/// columns (`budget_evicted`, `active_after_budget`) are `None` when
/// budget enforcement is disabled.
pub(crate) fn row_selection_schema() -> Schema {
    Schema::new(vec![
        Field::new("iteration", DataType::Int32, false),
        Field::new("stage", DataType::Int32, false),
        Field::new("cuts_populated", DataType::Int32, false),
        Field::new("cuts_active_before", DataType::Int32, false),
        Field::new("cuts_deactivated", DataType::Int32, false),
        Field::new("cuts_active_after", DataType::Int32, false),
        Field::new("selection_time_ms", DataType::Float64, false),
        Field::new("budget_evicted", DataType::Int32, true),
        Field::new("active_after_budget", DataType::Int32, true),
    ])
}

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
    fn parquet_writer_config_default_values() {
        use crate::output::parquet_config::ParquetWriterConfig;
        use parquet::basic::Compression;
        let cfg = ParquetWriterConfig::default();
        assert_eq!(cfg.row_group_size, 100_000);
        assert!(cfg.dictionary_encoding);
        assert!(matches!(cfg.compression, Compression::ZSTD(_)));
    }

    #[test]
    fn costs_schema_field_count_and_names() {
        let schema = costs_schema();
        assert_eq!(
            schema.fields().len(),
            26,
            "costs schema must have 26 fields"
        );
        let names = field_names(&schema);
        assert_eq!(
            names,
            vec![
                "stage_id",
                "block_id",
                "total_cost",
                "immediate_cost",
                "future_cost",
                "discount_factor",
                "thermal_cost",
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
                "fpha_turbined_cost",
                "curtailment_cost",
                "exchange_cost",
                "pumping_cost",
            ]
        );
    }

    #[test]
    fn costs_schema_types_and_nullability() {
        let schema = costs_schema();
        // stage_id: i32, not nullable
        assert_eq!(field_type(&schema, "stage_id"), DataType::Int32);
        assert!(!is_nullable(&schema, "stage_id"));
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
            "fpha_turbined_cost",
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
        // The spec (output-schemas.md SS5.2) defines 31 data columns.
        // Bidirectional withdrawal/evaporation slacks added pos/neg pairs.
        assert_eq!(
            schema.fields().len(),
            31,
            "hydros schema must have 31 fields"
        );
        let names = field_names(&schema);
        assert_eq!(
            names,
            vec![
                "stage_id",
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
                "productivity_mw_per_m3s",
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
            ]
        );
    }

    #[test]
    fn hydros_schema_nullable_fields() {
        let schema = hydros_schema();
        // block_id, evaporation_m3s, diverted_inflow_m3s, diverted_outflow_m3s,
        // productivity_mw_per_m3s are nullable per spec SS5.2
        for col in &[
            "block_id",
            "evaporation_m3s",
            "diverted_inflow_m3s",
            "diverted_outflow_m3s",
            "productivity_mw_per_m3s",
        ] {
            assert!(is_nullable(&schema, col), "column {col} must be nullable");
        }
        // All other non-nullable columns
        for col in &[
            "stage_id",
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
        ] {
            assert!(
                !is_nullable(&schema, col),
                "column {col} must not be nullable"
            );
        }
    }

    #[test]
    fn thermals_schema_field_count() {
        let schema = thermals_schema();
        assert_eq!(
            schema.fields().len(),
            10,
            "thermals schema must have 10 fields"
        );
    }

    #[test]
    fn thermals_schema_gnl_fields_nullable() {
        let schema = thermals_schema();
        assert!(is_nullable(&schema, "gnl_committed_mw"));
        assert!(is_nullable(&schema, "gnl_decision_mw"));
        assert!(!is_nullable(&schema, "is_gnl"));
        assert_eq!(field_type(&schema, "is_gnl"), DataType::Boolean);
        assert_eq!(field_type(&schema, "operative_state_code"), DataType::Int8);
    }

    #[test]
    fn exchanges_schema_field_count() {
        let schema = exchanges_schema();
        assert_eq!(
            schema.fields().len(),
            11,
            "exchanges schema must have 11 fields"
        );
    }

    #[test]
    fn buses_schema_field_count() {
        let schema = buses_schema();
        assert_eq!(
            schema.fields().len(),
            10,
            "buses schema must have 10 fields"
        );
    }

    #[test]
    fn pumping_stations_schema_field_count() {
        let schema = pumping_stations_schema();
        assert_eq!(
            schema.fields().len(),
            9,
            "pumping_stations schema must have 9 fields"
        );
    }

    #[test]
    fn contracts_schema_field_count() {
        let schema = contracts_schema();
        assert_eq!(
            schema.fields().len(),
            8,
            "contracts schema must have 8 fields"
        );
    }

    #[test]
    fn non_controllables_schema_field_count() {
        let schema = non_controllables_schema();
        assert_eq!(
            schema.fields().len(),
            10,
            "non_controllables schema must have 10 fields"
        );
    }

    #[test]
    fn inflow_lags_schema_field_count() {
        let schema = inflow_lags_schema();
        assert_eq!(
            schema.fields().len(),
            4,
            "inflow_lags schema must have 4 fields"
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
    fn generic_violations_schema_field_count() {
        let schema = generic_violations_schema();
        assert_eq!(
            schema.fields().len(),
            5,
            "generic_violations schema must have 5 fields"
        );
    }

    #[test]
    fn convergence_schema_field_count_and_types() {
        let schema = convergence_schema();
        assert_eq!(
            schema.fields().len(),
            13,
            "convergence schema must have 13 fields"
        );
        // spot-check types per spec SS6.1
        assert_eq!(field_type(&schema, "iteration"), DataType::Int32);
        assert_eq!(field_type(&schema, "lower_bound"), DataType::Float64);
        assert_eq!(field_type(&schema, "upper_bound_mean"), DataType::Float64);
        assert_eq!(field_type(&schema, "cuts_added"), DataType::Int32);
        assert_eq!(field_type(&schema, "cuts_active"), DataType::Int64);
        assert_eq!(field_type(&schema, "time_forward_ms"), DataType::Int64);
        assert_eq!(field_type(&schema, "lp_solves"), DataType::Int64);
        assert_eq!(field_type(&schema, "forward_passes"), DataType::Int32);
    }

    #[test]
    fn convergence_schema_nullable_fields() {
        let schema = convergence_schema();
        // gap_percent is nullable (None when LB <= 0)
        assert!(is_nullable(&schema, "gap_percent"));
        // all other fields must not be nullable
        for name in &[
            "iteration",
            "lower_bound",
            "upper_bound_mean",
            "upper_bound_std",
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
            18,
            "iteration_timing schema must have 18 fields"
        );
    }

    #[test]
    fn iteration_timing_schema_rank_worker_nullable() {
        let schema = iteration_timing_schema();
        // rank (position 1) and worker_id (position 2) must be nullable.
        let rank_field = schema
            .field_with_name("rank")
            .expect("rank field must exist");
        assert!(rank_field.is_nullable(), "rank must be nullable");
        let worker_id_field = schema
            .field_with_name("worker_id")
            .expect("worker_id field must exist");
        assert!(worker_id_field.is_nullable(), "worker_id must be nullable");
        // All other 16 timing columns must be non-nullable.
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
    fn rank_timing_schema_field_count() {
        let schema = rank_timing_schema();
        assert_eq!(
            schema.fields().len(),
            8,
            "rank_timing schema must have 8 fields"
        );
    }

    #[test]
    fn rank_timing_schema_all_non_nullable() {
        let schema = rank_timing_schema();
        for field in schema.fields() {
            assert!(
                !field.is_nullable(),
                "rank_timing field '{}' must not be nullable",
                field.name()
            );
        }
    }

    #[test]
    fn row_selection_schema_field_count_and_types() {
        let schema = row_selection_schema();
        assert_eq!(
            schema.fields().len(),
            9,
            "cut_selection schema must have 9 fields"
        );
        // First 6 fields are non-nullable Int32.
        for field in &schema.fields()[..6] {
            assert_eq!(field.data_type(), &DataType::Int32);
            assert!(!field.is_nullable());
        }
        // Field 7 (index 6): selection_time_ms, Float64, non-nullable.
        assert_eq!(schema.fields()[6].name(), "selection_time_ms");
        assert_eq!(schema.fields()[6].data_type(), &DataType::Float64);
        assert!(!schema.fields()[6].is_nullable());
        // Fields 8-9 (indices 7-8): nullable Int32.
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
            ("iteration", DataType::UInt32, false),
            ("phase", DataType::Utf8, false),
            ("stage", DataType::Int32, false),
            ("opening", DataType::Int32, true),
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
            ("basis_reconstructions", DataType::UInt64, false),
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
            ("stage", DataType::Int32, false),
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
        // Call all schema functions and verify they return non-empty schemas
        // without panicking.
        let schemas: Vec<(Schema, &str)> = vec![
            (costs_schema(), "costs"),
            (hydros_schema(), "hydros"),
            (thermals_schema(), "thermals"),
            (exchanges_schema(), "exchanges"),
            (buses_schema(), "buses"),
            (pumping_stations_schema(), "pumping_stations"),
            (contracts_schema(), "contracts"),
            (non_controllables_schema(), "non_controllables"),
            (inflow_lags_schema(), "inflow_lags"),
            (generic_violations_schema(), "generic_violations"),
            (convergence_schema(), "convergence"),
            (iteration_timing_schema(), "iteration_timing"),
            (rank_timing_schema(), "rank_timing"),
            (row_selection_schema(), "cut_selection"),
            (solver_iterations_schema(), "solver_iterations"),
            (retry_histogram_schema(), "retry_histogram"),
        ];
        for (schema, name) in &schemas {
            assert!(
                !schema.fields().is_empty(),
                "schema '{name}' must have at least one field"
            );
        }
        // Verify total counts per spec
        let counts: Vec<(&str, usize)> = schemas
            .iter()
            .map(|(s, n)| (*n, s.fields().len()))
            .collect();
        let expected: &[(&str, usize)] = &[
            ("costs", 26),
            ("hydros", 31),
            ("thermals", 10),
            ("exchanges", 11),
            ("buses", 10),
            ("pumping_stations", 9),
            ("contracts", 8),
            ("non_controllables", 10),
            ("inflow_lags", 4),
            ("generic_violations", 5),
            ("convergence", 13),
            ("iteration_timing", 18),
            ("rank_timing", 8),
            ("cut_selection", 9),
            ("solver_iterations", 19),
            ("retry_histogram", 5),
        ];
        for ((name, actual), (_, exp)) in counts.iter().zip(expected.iter()) {
            assert_eq!(
                actual, exp,
                "schema '{name}' field count: expected {exp}, got {actual}"
            );
        }
    }
}
