//! Arrow schema definitions for all Parquet output files per output-schemas spec
//! (SS5.1–5.11 and SS6.1–6.3).

#![allow(dead_code)]

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
/// 28 fields. See output-schemas.md SS5.2.
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
        Field::new("evaporation_violation_m3s", DataType::Float64, false),
        Field::new("inflow_nonnegativity_slack_m3s", DataType::Float64, false),
        Field::new("water_withdrawal_violation_m3s", DataType::Float64, false),
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
/// 10 fields. See output-schemas.md SS6.2.
pub(crate) fn iteration_timing_schema() -> Schema {
    Schema::new(vec![
        Field::new("iteration", DataType::Int32, false),
        Field::new("forward_solve_ms", DataType::Int64, false),
        Field::new("forward_sample_ms", DataType::Int64, false),
        Field::new("backward_solve_ms", DataType::Int64, false),
        Field::new("backward_cut_ms", DataType::Int64, false),
        Field::new("cut_selection_ms", DataType::Int64, false),
        Field::new("mpi_allreduce_ms", DataType::Int64, false),
        Field::new("mpi_broadcast_ms", DataType::Int64, false),
        Field::new("io_write_ms", DataType::Int64, false),
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
/// 12 columns. One row per (iteration, phase, stage) triple.
pub(crate) fn solver_iterations_schema() -> Schema {
    Schema::new(vec![
        Field::new("iteration", DataType::UInt32, false),
        Field::new("phase", DataType::Utf8, false),
        Field::new("stage", DataType::Int32, false),
        Field::new("lp_solves", DataType::UInt32, false),
        Field::new("lp_successes", DataType::UInt32, false),
        Field::new("lp_retries", DataType::UInt32, false),
        Field::new("lp_failures", DataType::UInt32, false),
        Field::new("retry_attempts", DataType::UInt32, false),
        Field::new("basis_offered", DataType::UInt32, false),
        Field::new("basis_rejections", DataType::UInt32, false),
        Field::new("simplex_iterations", DataType::UInt64, false),
        Field::new("solve_time_ms", DataType::Float64, false),
    ])
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
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
            20,
            "costs schema must have 20 fields"
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
        // The spec (output-schemas.md SS5.2) defines 29 data columns.
        // The ticket's acceptance criterion listed 26, which is a stale count;
        // the spec is the authoritative source.
        assert_eq!(
            schema.fields().len(),
            29,
            "hydros schema must have 29 fields"
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
                "evaporation_violation_m3s",
                "inflow_nonnegativity_slack_m3s",
                "water_withdrawal_violation_m3s",
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
            "evaporation_violation_m3s",
            "inflow_nonnegativity_slack_m3s",
            "water_withdrawal_violation_m3s",
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
            10,
            "iteration_timing schema must have 10 fields"
        );
    }

    #[test]
    fn iteration_timing_schema_all_non_nullable() {
        let schema = iteration_timing_schema();
        for field in schema.fields() {
            assert!(
                !field.is_nullable(),
                "iteration_timing field '{}' must not be nullable",
                field.name()
            );
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
    fn all_schema_functions_return_valid_schemas() {
        // Call all 14 schema functions and verify they return non-empty schemas
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
            ("costs", 20),
            ("hydros", 29),
            ("thermals", 10),
            ("exchanges", 11),
            ("buses", 10),
            ("pumping_stations", 9),
            ("contracts", 8),
            ("non_controllables", 10),
            ("inflow_lags", 4),
            ("generic_violations", 5),
            ("convergence", 13),
            ("iteration_timing", 10),
            ("rank_timing", 8),
        ];
        for ((name, actual), (_, exp)) in counts.iter().zip(expected.iter()) {
            assert_eq!(
                actual, exp,
                "schema '{name}' field count: expected {exp}, got {actual}"
            );
        }
    }
}
