//! Back-compatibility and forward-roundtrip tests for the output metadata structs.
//!
//! Contract: new fields on `training/metadata.json` and `simulation/metadata.json`
//! must be `#[serde(default)]` so older output directories still deserialize. The
//! legacy fixtures are hand-frozen JSON literals, NOT struct-serialized — a
//! struct-built fixture carries every field and so could never catch a field
//! accidentally made required.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]

use cobre_io::{
    DeviationSummary, DeviationWorstEntry, DistributionInfo, HostLayout, MetadataBounds,
    MetadataConfiguration, MetadataConvergence, MetadataCost, MetadataIterations,
    MetadataProblemDimensions, MetadataRowPool, MetadataScenarios, MetadataSimulationSolveStats,
    MetadataTrainingSolveStats, SimulationMetadata, TrainingMetadata, read_simulation_metadata,
    read_training_metadata, write_simulation_metadata, write_training_metadata,
};

// ── Frozen legacy fixtures ─────────────────────────────────────────────────────
//
// These literals must NOT contain the keys `bounds`, `solve_stats`, `cost`, or
// `hosts`; adding one defeats the back-compat guarantee. The `*_omits_new_keys`
// tests below guard this.

/// Legacy `training/metadata.json` without the `bounds`, `solve_stats`, and
/// `distribution.hosts` fields.
const LEGACY_TRAINING_JSON: &str = r#"{
  "cobre_version": "0.1.0",
  "hostname": "legacy-host",
  "solver": "highs",
  "solver_version": "1.8.0",
  "started_at": "2026-01-17T08:00:00Z",
  "completed_at": "2026-01-17T12:30:00Z",
  "duration_seconds": 16200.0,
  "status": "complete",
  "configuration": {
    "seed": 42,
    "max_iterations": 100,
    "forward_passes": 192,
    "stopping_mode": "any",
    "policy_mode": "fresh"
  },
  "problem_dimensions": {
    "num_stages": 12,
    "num_hydros": 160,
    "num_thermals": 200,
    "num_buses": 5,
    "num_lines": 8
  },
  "iterations": {
    "completed": 100,
    "converged_at": 95
  },
  "convergence": {
    "achieved": true,
    "final_gap_percent": 0.45,
    "termination_reason": "bound_stalling"
  },
  "row_pool": {
    "total_generated": 1250000,
    "total_active": 980000,
    "peak_active": 1100000
  },
  "distribution": {
    "backend": "local",
    "world_size": 1,
    "ranks_participated": 1,
    "num_nodes": 1,
    "threads_per_rank": 1
  }
}"#;

/// Legacy `simulation/metadata.json` without the `cost`, `solve_stats`, and
/// `distribution.hosts` fields.
const LEGACY_SIM_JSON: &str = r#"{
  "cobre_version": "0.1.0",
  "hostname": "legacy-host",
  "solver": "highs",
  "solver_version": "1.8.0",
  "started_at": "2026-01-17T13:00:00Z",
  "completed_at": "2026-01-17T13:15:00Z",
  "duration_seconds": 900.0,
  "status": "complete",
  "scenarios": {
    "total": 100,
    "completed": 100,
    "failed": 0
  },
  "distribution": {
    "backend": "local",
    "world_size": 1,
    "ranks_participated": 1,
    "num_nodes": 1,
    "threads_per_rank": 1
  }
}"#;

/// Frozen multi-node `DistributionInfo` pinning the on-disk per-host rank shape.
const MULTI_NODE_DISTRIBUTION_JSON: &str = r#"{
  "backend": "mpi",
  "world_size": 8,
  "ranks_participated": 8,
  "num_nodes": 2,
  "threads_per_rank": 4,
  "mpi_library": "Open MPI v4.1.6",
  "mpi_standard": "MPI 4.0",
  "thread_level": "Funneled",
  "hosts": [
    { "hostname": "node01", "ranks": [0, 1, 2, 3] },
    { "hostname": "node02", "ranks": [4, 5, 6, 7] }
  ]
}"#;

// ── Back-compat: legacy JSON deserializes with documented defaults ──────────────

#[test]
fn legacy_training_json_deserializes_with_defaults() {
    let decoded: TrainingMetadata = serde_json::from_str(LEGACY_TRAINING_JSON)
        .expect("legacy training metadata must still deserialize");

    assert_eq!(decoded.bounds.final_lower_bound, 0.0);
    assert_eq!(decoded.bounds.final_upper_bound, None);
    assert_eq!(decoded.bounds.final_upper_bound_std, None);

    assert_eq!(decoded.solve_stats.total_lp_solves, None);
    assert_eq!(decoded.solve_stats.first_try, None);
    assert_eq!(decoded.solve_stats.retried, None);
    assert_eq!(decoded.solve_stats.failed, None);
    assert_eq!(decoded.solve_stats.forward_solve_seconds, None);
    assert_eq!(decoded.solve_stats.backward_solve_seconds, None);
    assert_eq!(decoded.solve_stats.parallelism, None);
    assert_eq!(decoded.solve_stats.forward_phase_wall_seconds, None);
    assert_eq!(decoded.solve_stats.backward_phase_wall_seconds, None);
    assert_eq!(decoded.solve_stats.forward_wait_seconds, None);
    assert_eq!(decoded.solve_stats.backward_wait_seconds, None);
    assert_eq!(decoded.solve_stats.serial_lower_bound_seconds, None);
    assert_eq!(decoded.solve_stats.serial_row_selection_seconds, None);
    assert_eq!(decoded.solve_stats.serial_row_sync_seconds, None);
    assert_eq!(decoded.solve_stats.serial_allreduce_seconds, None);
    assert_eq!(decoded.solve_stats.serial_scheduling_seconds, None);

    assert!(decoded.distribution.hosts.is_empty());

    // Regression gate for an accidental `#[serde(default)]` removal on either optional section.
    assert!(decoded.setup.is_none());
    assert!(decoded.production_fit_deviation.is_none());

    assert_eq!(decoded.iterations.completed, 100);
    assert_eq!(decoded.row_pool.total_generated, 1_250_000);
}

#[test]
fn training_metadata_with_legacy_solve_stats_defaults_phase_timings() {
    let mut value = serde_json::to_value(fully_populated_training_metadata())
        .expect("metadata fixture must serialize");
    let solve_stats = value["solve_stats"]
        .as_object_mut()
        .expect("solve_stats must be an object");
    for field in [
        "forward_phase_wall_seconds",
        "backward_phase_wall_seconds",
        "forward_wait_seconds",
        "backward_wait_seconds",
        "serial_lower_bound_seconds",
        "serial_row_selection_seconds",
        "serial_row_sync_seconds",
        "serial_allreduce_seconds",
        "serial_scheduling_seconds",
    ] {
        solve_stats.remove(field);
    }

    let decoded: TrainingMetadata =
        serde_json::from_value(value).expect("legacy solve_stats must deserialize");
    assert_eq!(decoded.solve_stats.forward_phase_wall_seconds, None);
    assert_eq!(decoded.solve_stats.backward_phase_wall_seconds, None);
    assert_eq!(decoded.solve_stats.forward_wait_seconds, None);
    assert_eq!(decoded.solve_stats.backward_wait_seconds, None);
    assert_eq!(decoded.solve_stats.serial_lower_bound_seconds, None);
    assert_eq!(decoded.solve_stats.serial_row_selection_seconds, None);
    assert_eq!(decoded.solve_stats.serial_row_sync_seconds, None);
    assert_eq!(decoded.solve_stats.serial_allreduce_seconds, None);
    assert_eq!(decoded.solve_stats.serial_scheduling_seconds, None);
}

#[test]
fn legacy_simulation_json_deserializes_with_defaults() {
    let decoded: SimulationMetadata = serde_json::from_str(LEGACY_SIM_JSON)
        .expect("legacy simulation metadata must still deserialize");

    assert!(decoded.cost.is_none());

    assert_eq!(decoded.solve_stats.total_lp_solves, None);
    assert_eq!(decoded.solve_stats.first_try, None);
    assert_eq!(decoded.solve_stats.retried, None);
    assert_eq!(decoded.solve_stats.failed, None);
    assert_eq!(decoded.solve_stats.solve_seconds, None);
    assert_eq!(decoded.solve_stats.parallelism, None);

    assert!(decoded.distribution.hosts.is_empty());

    assert_eq!(decoded.scenarios.total, 100);
    assert_eq!(decoded.scenarios.completed, 100);
}

// ── Guard: the legacy fixtures genuinely omit the new keys ──────────────────────

#[test]
fn legacy_training_fixture_omits_new_keys() {
    assert!(
        !LEGACY_TRAINING_JSON.contains("bounds"),
        "legacy training fixture must omit the `bounds` key to exercise back-compat"
    );
    assert!(
        !LEGACY_TRAINING_JSON.contains("solve_stats"),
        "legacy training fixture must omit the `solve_stats` key to exercise back-compat"
    );
    assert!(
        !LEGACY_TRAINING_JSON.contains("hosts"),
        "legacy training fixture must omit the `hosts` key to exercise back-compat"
    );
    assert!(
        !LEGACY_TRAINING_JSON.contains("setup"),
        "legacy training fixture must omit the `setup` key to exercise back-compat"
    );
    assert!(
        !LEGACY_TRAINING_JSON.contains("production_fit_deviation"),
        "legacy training fixture must omit the `production_fit_deviation` key to exercise back-compat"
    );
}

#[test]
fn legacy_simulation_fixture_omits_new_keys() {
    assert!(
        !LEGACY_SIM_JSON.contains("cost"),
        "legacy simulation fixture must omit the `cost` key to exercise back-compat"
    );
    assert!(
        !LEGACY_SIM_JSON.contains("solve_stats"),
        "legacy simulation fixture must omit the `solve_stats` key to exercise back-compat"
    );
    assert!(
        !LEGACY_SIM_JSON.contains("hosts"),
        "legacy simulation fixture must omit the `hosts` key to exercise back-compat"
    );
}

// ── Multi-node hosts fixture ────────────────────────────────────────────────────

#[test]
fn multi_node_distribution_deserializes_with_correct_rank_lists() {
    let decoded: DistributionInfo = serde_json::from_str(MULTI_NODE_DISTRIBUTION_JSON)
        .expect("multi-node distribution fixture must deserialize");

    assert_eq!(decoded.backend, "mpi");
    assert_eq!(decoded.world_size, 8);
    assert_eq!(decoded.num_nodes, 2);
    assert_eq!(decoded.hosts.len(), 2);

    assert_eq!(decoded.hosts[0].hostname, "node01");
    assert_eq!(decoded.hosts[0].ranks, vec![0, 1, 2, 3]);
    assert_eq!(decoded.hosts[1].hostname, "node02");
    assert_eq!(decoded.hosts[1].ranks, vec![4, 5, 6, 7]);
}

// ── Forward roundtrip: fully-populated metadata survives write → read ────────────

fn fully_populated_distribution() -> DistributionInfo {
    DistributionInfo {
        backend: "mpi".to_string(),
        world_size: 8,
        ranks_participated: 8,
        num_nodes: 2,
        threads_per_rank: 4,
        mpi_library: Some("Open MPI v4.1.6".to_string()),
        mpi_standard: Some("MPI 4.0".to_string()),
        thread_level: Some("Funneled".to_string()),
        slurm_job_id: Some("123456".to_string()),
        hosts: vec![
            HostLayout {
                hostname: "node01".to_string(),
                ranks: vec![0, 1, 2, 3],
            },
            HostLayout {
                hostname: "node02".to_string(),
                ranks: vec![4, 5, 6, 7],
            },
        ],
        rank_affinity: Vec::new(),
    }
}

fn fully_populated_training_metadata() -> TrainingMetadata {
    TrainingMetadata {
        cobre_version: "0.1.6".to_string(),
        hostname: "node01".to_string(),
        solver: "highs".to_string(),
        solver_version: Some("1.8.0".to_string()),
        started_at: "2026-01-17T08:00:00Z".to_string(),
        completed_at: "2026-01-17T12:30:00Z".to_string(),
        duration_seconds: 16_200.0,
        status: "complete".to_string(),
        configuration: MetadataConfiguration {
            seed: Some(42),
            max_iterations: Some(100),
            forward_passes: Some(192),
            stopping_mode: "any".to_string(),
            policy_mode: "fresh".to_string(),
        },
        problem_dimensions: MetadataProblemDimensions {
            num_stages: 12,
            num_hydros: 160,
            num_thermals: 200,
            num_buses: 5,
            num_lines: 8,
        },
        iterations: MetadataIterations {
            completed: 100,
            converged_at: Some(95),
        },
        convergence: MetadataConvergence {
            achieved: true,
            final_gap_percent: Some(0.45),
            termination_reason: "bound_stalling".to_string(),
        },
        row_pool: MetadataRowPool {
            total_generated: 1_250_000,
            total_active: 980_000,
            peak_active: 1_100_000,
            cuts_active: 980_000,
            rows_in_lp_total: 0,
            rows_in_lp_solve_count: 0,
            rows_in_lp_max: 0,
        },
        bounds: MetadataBounds {
            final_lower_bound: 48_500.0,
            final_upper_bound: Some(49_000.0),
            final_upper_bound_std: Some(250.0),
        },
        solve_stats: MetadataTrainingSolveStats {
            total_lp_solves: Some(84_000),
            first_try: Some(80_000),
            retried: Some(3_800),
            failed: Some(200),
            forward_solve_seconds: Some(123.5),
            backward_solve_seconds: Some(456.75),
            parallelism: Some(8),
            forward_phase_wall_seconds: Some(150.0),
            backward_phase_wall_seconds: Some(500.0),
            forward_wait_seconds: Some(10.0),
            backward_wait_seconds: Some(20.0),
            serial_lower_bound_seconds: Some(30.0),
            serial_row_selection_seconds: Some(40.0),
            serial_row_sync_seconds: Some(50.0),
            serial_allreduce_seconds: Some(60.0),
            serial_scheduling_seconds: Some(70.0),
        },
        setup: None,
        production_fit_deviation: Some(DeviationSummary {
            n_entries: 3,
            mean_abs: 4.1,
            max_abs: 31.7,
            worst_relative: 0.062,
            worst_entry: Some(DeviationWorstEntry {
                entity_id: 12,
                stage_id: 4,
                relative: 0.062,
                mean_abs: 8.0,
                max_abs: 31.7,
            }),
        }),
        distribution: fully_populated_distribution(),
    }
}

fn fully_populated_simulation_metadata() -> SimulationMetadata {
    SimulationMetadata {
        cobre_version: "0.1.6".to_string(),
        hostname: "node01".to_string(),
        solver: "highs".to_string(),
        solver_version: Some("1.8.0".to_string()),
        started_at: "2026-01-17T13:00:00Z".to_string(),
        completed_at: "2026-01-17T13:15:00Z".to_string(),
        duration_seconds: 900.0,
        status: "complete".to_string(),
        scenarios: MetadataScenarios {
            total: 100,
            completed: 100,
            failed: 0,
        },
        cost: Some(MetadataCost {
            mean_cost: 12_345.6,
            std_cost: 200.0,
            cvar: 13_000.0,
            cvar_alpha: 0.95,
        }),
        solve_stats: MetadataSimulationSolveStats {
            total_lp_solves: Some(50_000),
            first_try: Some(48_000),
            retried: Some(1_900),
            failed: Some(100),
            solve_seconds: Some(321.0),
            parallelism: Some(8),
        },
        distribution: fully_populated_distribution(),
    }
}

#[test]
fn training_metadata_new_fields_survive_write_read_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("metadata.json");
    let original = fully_populated_training_metadata();

    write_training_metadata(&path, &original).expect("write must succeed");
    let decoded = read_training_metadata(&path).expect("read must succeed");

    assert_eq!(
        decoded.bounds.final_lower_bound,
        original.bounds.final_lower_bound
    );
    assert_eq!(
        decoded.bounds.final_upper_bound,
        original.bounds.final_upper_bound
    );
    assert_eq!(
        decoded.bounds.final_upper_bound_std,
        original.bounds.final_upper_bound_std
    );

    assert_eq!(
        decoded.solve_stats.total_lp_solves,
        original.solve_stats.total_lp_solves
    );
    assert_eq!(
        decoded.solve_stats.first_try,
        original.solve_stats.first_try
    );
    assert_eq!(decoded.solve_stats.retried, original.solve_stats.retried);
    assert_eq!(decoded.solve_stats.failed, original.solve_stats.failed);
    assert_eq!(
        decoded.solve_stats.forward_solve_seconds,
        original.solve_stats.forward_solve_seconds
    );
    assert_eq!(
        decoded.solve_stats.backward_solve_seconds,
        original.solve_stats.backward_solve_seconds
    );
    assert_eq!(
        decoded.solve_stats.parallelism,
        original.solve_stats.parallelism
    );
    assert_eq!(
        decoded.solve_stats.forward_phase_wall_seconds,
        original.solve_stats.forward_phase_wall_seconds
    );
    assert_eq!(
        decoded.solve_stats.backward_phase_wall_seconds,
        original.solve_stats.backward_phase_wall_seconds
    );
    assert_eq!(
        decoded.solve_stats.forward_wait_seconds,
        original.solve_stats.forward_wait_seconds
    );
    assert_eq!(
        decoded.solve_stats.backward_wait_seconds,
        original.solve_stats.backward_wait_seconds
    );
    assert_eq!(
        decoded.solve_stats.serial_lower_bound_seconds,
        original.solve_stats.serial_lower_bound_seconds
    );
    assert_eq!(
        decoded.solve_stats.serial_row_selection_seconds,
        original.solve_stats.serial_row_selection_seconds
    );
    assert_eq!(
        decoded.solve_stats.serial_row_sync_seconds,
        original.solve_stats.serial_row_sync_seconds
    );
    assert_eq!(
        decoded.solve_stats.serial_allreduce_seconds,
        original.solve_stats.serial_allreduce_seconds
    );
    assert_eq!(
        decoded.solve_stats.serial_scheduling_seconds,
        original.solve_stats.serial_scheduling_seconds
    );

    assert_eq!(decoded.distribution.hosts.len(), 2);
    assert_eq!(decoded.distribution.hosts[0].hostname, "node01");
    assert_eq!(decoded.distribution.hosts[0].ranks, vec![0, 1, 2, 3]);
    assert_eq!(decoded.distribution.hosts[1].hostname, "node02");
    assert_eq!(decoded.distribution.hosts[1].ranks, vec![4, 5, 6, 7]);

    let summary = decoded
        .production_fit_deviation
        .expect("deviation summary must survive round-trip");
    assert_eq!(summary.n_entries, 3);
    assert_eq!(summary.mean_abs, 4.1);
    assert_eq!(summary.max_abs, 31.7);
    assert_eq!(summary.worst_relative, 0.062);
    let worst = summary.worst_entry.expect("worst entry must survive");
    assert_eq!(worst.entity_id, 12);
    assert_eq!(worst.stage_id, 4);
}

#[test]
fn simulation_metadata_new_fields_survive_write_read_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("metadata.json");
    let original = fully_populated_simulation_metadata();

    write_simulation_metadata(&path, &original).expect("write must succeed");
    let decoded = read_simulation_metadata(&path).expect("read must succeed");

    let original_cost = original.cost.expect("fixture must populate cost");
    let decoded_cost = decoded.cost.expect("cost must survive roundtrip");
    assert_eq!(decoded_cost.mean_cost, original_cost.mean_cost);
    assert_eq!(decoded_cost.std_cost, original_cost.std_cost);
    assert_eq!(decoded_cost.cvar, original_cost.cvar);
    assert_eq!(decoded_cost.cvar_alpha, original_cost.cvar_alpha);

    assert_eq!(
        decoded.solve_stats.total_lp_solves,
        original.solve_stats.total_lp_solves
    );
    assert_eq!(
        decoded.solve_stats.first_try,
        original.solve_stats.first_try
    );
    assert_eq!(decoded.solve_stats.retried, original.solve_stats.retried);
    assert_eq!(decoded.solve_stats.failed, original.solve_stats.failed);
    assert_eq!(
        decoded.solve_stats.solve_seconds,
        original.solve_stats.solve_seconds
    );
    assert_eq!(
        decoded.solve_stats.parallelism,
        original.solve_stats.parallelism
    );

    assert_eq!(decoded.distribution.hosts.len(), 2);
    assert_eq!(decoded.distribution.hosts[0].hostname, "node01");
    assert_eq!(decoded.distribution.hosts[0].ranks, vec![0, 1, 2, 3]);
    assert_eq!(decoded.distribution.hosts[1].hostname, "node02");
    assert_eq!(decoded.distribution.hosts[1].ranks, vec![4, 5, 6, 7]);
}
