//! Integration tests for the intra-artifact date-consistency validation that
//! `read_policy_checkpoint` runs after decode.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::cast_possible_truncation
)]

use cobre_io::{
    CheckpointManifest, ENTITY_SLOT_DATE_SENTINEL, EntitySlot, FORMAT_VERSION, GraphManifest,
    ProducerBlock, STAGE_CUTS_PRICED_STATE_DATE_SENTINEL, SeasonManifest, StageCutsPayload,
    read_policy_checkpoint, write_policy_checkpoint,
};

fn metadata() -> CheckpointManifest {
    CheckpointManifest {
        format_version: FORMAT_VERSION,
        software: Some("cobre".to_string()),
        software_version: "0.13.0".to_string(),
        created_at: "2026-08-11T00:00:00Z".to_string(),
        num_stages: 1,
        graph_manifest: GraphManifest::default(),
        producer: ProducerBlock {
            completed_iterations: 1,
            final_lower_bound: 0.0,
            best_upper_bound: None,
            max_iterations: 1,
            forward_passes: 1,
            warm_start_cuts: 0,
            warm_start_counts: vec![],
            rng_seed: 0,
            total_visited_states: 0,
            training_block_mode: "parallel".to_string(),
            training_block_mode_per_stage: vec![],
            cost_scale_factor: None,
            lower_bound_history: Vec::new(),
        },
        season_manifest: SeasonManifest::default(),
    }
}

/// Write a one-pool checkpoint fixture whose only payload is `manifest`.
fn write_fixture(dir: &std::path::Path, pool_id: u32, manifest: &[EntitySlot]) {
    let payload = StageCutsPayload {
        stage_id: pool_id,
        state_dimension: manifest.len() as u32,
        capacity: 0,
        warm_start_count: 0,
        cuts: &[],
        active_cut_indices: &[],
        populated_count: 0,
        entity_manifest: manifest,
        cost_scale_factor: 1_000_000.0,
        node_id: -1,
        graph_stage_id: -1,
        priced_state_date: STAGE_CUTS_PRICED_STATE_DATE_SENTINEL,
    };
    write_policy_checkpoint(dir, &[payload], &[], &metadata(), &[]).expect("fixture must write");
}

#[test]
fn malformed_month_reference_date_rejected_naming_slot() {
    let dir = tempfile::tempdir().unwrap();
    let manifest = [EntitySlot::inflow_lag(7, 0, true).with_reference_date(20_261_305)];
    write_fixture(dir.path(), 0, &manifest);

    let err = read_policy_checkpoint(dir.path()).expect_err("month-13 date must be rejected");
    let msg = err.to_string();
    assert!(msg.contains("pool 0"), "must name the pool: {msg}");
    assert!(msg.contains("subindex 0"), "must name the subindex: {msg}");
    assert!(
        msg.contains("20261305"),
        "must name the malformed date: {msg}"
    );
}

#[test]
fn hydro_transit_bucket_decreasing_dates_rejected_naming_pool_and_subindex() {
    let dir = tempfile::tempdir().unwrap();
    let manifest = [
        EntitySlot::transit_bucket(42, 0, true).with_interval(20_260_601, 20_260_701),
        EntitySlot::transit_bucket(42, 1, true).with_interval(20_260_501, 20_260_601),
    ];
    write_fixture(dir.path(), 3, &manifest);

    let err = read_policy_checkpoint(dir.path())
        .expect_err("decreasing HydroTransitBucket dates must be rejected");
    let msg = err.to_string();
    assert!(msg.contains("pool 3"), "must name the pool: {msg}");
    assert!(
        msg.contains("subindex 1"),
        "must name the offending subindex: {msg}"
    );
    assert!(
        msg.contains("interval_start"),
        "must name interval_start: {msg}"
    );
}

#[test]
fn well_formed_monotone_checkpoint_accepted_manifest_unchanged() {
    let dir = tempfile::tempdir().unwrap();
    let manifest = [
        EntitySlot::storage(1, true),
        EntitySlot::transit_bucket(42, 0, true).with_interval(20_260_501, 20_260_601),
        EntitySlot::transit_bucket(42, 1, true).with_interval(20_260_601, 20_260_701),
    ];
    write_fixture(dir.path(), 1, &manifest);

    let checkpoint = read_policy_checkpoint(dir.path()).expect("well-formed checkpoint must read");
    assert_eq!(checkpoint.stage_cuts.len(), 1);
    let read_back = &checkpoint.stage_cuts[0].entity_manifest;
    assert_eq!(read_back.len(), manifest.len());
    for (got, want) in read_back.iter().zip(manifest.iter()) {
        assert_eq!(got.entity_type, want.entity_type);
        assert_eq!(got.entity_id, want.entity_id);
        assert_eq!(got.subindex, want.subindex);
        assert_eq!(got.interval_start, want.interval_start);
        assert_eq!(got.interval_end, want.interval_end);
    }
}

#[test]
fn fully_sentinel_legacy_checkpoint_accepted() {
    let dir = tempfile::tempdir().unwrap();
    let manifest = [
        EntitySlot::storage(1, true),
        EntitySlot::anticipated(7, 0, true),
        EntitySlot::anticipated(7, 1, true),
        EntitySlot::transit_bucket(42, 0, true),
    ];
    write_fixture(dir.path(), 0, &manifest);

    read_policy_checkpoint(dir.path())
        .expect("a fully sentinel-dated legacy checkpoint must be accepted");
}

#[test]
fn anticipated_thermal_state_non_monotone_dates_accepted() {
    let dir = tempfile::tempdir().unwrap();
    // At a stage where `t mod k_max != 0`, a correctly-produced modular-residue
    // subindex sequence decreases (subindex 0 delivers later than subindex 1).
    // Rejecting this would be a false positive on real data — see
    // `StateFamily::HydroTransitBucket`'s doc in `checkpoint.rs`.
    let manifest = [
        EntitySlot::anticipated(1, 0, true).with_interval(20_260_601, 20_260_701),
        EntitySlot::anticipated(1, 1, true).with_interval(20_260_501, 20_260_601),
    ];
    write_fixture(dir.path(), 1, &manifest);

    read_policy_checkpoint(dir.path())
        .expect("a modular-residue subindex family must be exempt from ordering checks");
}

#[test]
fn half_populated_interval_rejected_naming_slot_and_endpoint() {
    let dir = tempfile::tempdir().unwrap();
    let manifest = [EntitySlot::transit_bucket(42, 0, true)
        .with_interval(20_320_101, ENTITY_SLOT_DATE_SENTINEL)];
    write_fixture(dir.path(), 0, &manifest);

    let err =
        read_policy_checkpoint(dir.path()).expect_err("a half-populated interval must be rejected");
    let msg = err.to_string();
    assert!(msg.contains("pool 0"), "must name the pool: {msg}");
    assert!(msg.contains("entity 42"), "must name the entity: {msg}");
    assert!(msg.contains("subindex 0"), "must name the subindex: {msg}");
    assert!(
        msg.contains("interval_end"),
        "must name the missing endpoint: {msg}"
    );
}

#[test]
fn reversed_interval_rejected_naming_both_endpoints() {
    let dir = tempfile::tempdir().unwrap();
    let manifest = [EntitySlot::transit_bucket(42, 0, true).with_interval(20_320_101, 20_311_201)];
    write_fixture(dir.path(), 0, &manifest);

    let err = read_policy_checkpoint(dir.path()).expect_err("a reversed interval must be rejected");
    let msg = err.to_string();
    assert!(msg.contains("pool 0"), "must name the pool: {msg}");
    assert!(msg.contains("entity 42"), "must name the entity: {msg}");
    assert!(msg.contains("20320101"), "must name interval_start: {msg}");
    assert!(msg.contains("20311201"), "must name interval_end: {msg}");
}

#[test]
fn degenerate_zero_length_interval_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let manifest = [EntitySlot::transit_bucket(42, 0, true).with_interval(20_320_101, 20_320_101)];
    write_fixture(dir.path(), 0, &manifest);

    read_policy_checkpoint(dir.path())
        .expect_err("a zero-length interval (start == end) must be rejected");
}

#[test]
fn storage_slot_with_live_reference_date_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let manifest = [EntitySlot::storage(1, true).with_reference_date(20_311_201)];
    write_fixture(dir.path(), 0, &manifest);

    let err = read_policy_checkpoint(dir.path())
        .expect_err("a storage slot with a live reference_date must be rejected");
    let msg = err.to_string();
    assert!(msg.contains("pool 0"), "must name the pool: {msg}");
    assert!(msg.contains("entity 1"), "must name the entity: {msg}");
    assert!(
        msg.contains("no per-slot date"),
        "must state that a storage slot carries no per-slot date: {msg}"
    );
}

#[test]
fn transit_bucket_slot_with_live_reference_date_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let manifest = [EntitySlot::transit_bucket(42, 0, true).with_reference_date(20_311_201)];
    write_fixture(dir.path(), 0, &manifest);

    let err = read_policy_checkpoint(dir.path())
        .expect_err("a transit-bucket slot with a live reference_date must be rejected");
    let msg = err.to_string();
    assert!(msg.contains("pool 0"), "must name the pool: {msg}");
    assert!(msg.contains("entity 42"), "must name the entity: {msg}");
    assert!(msg.contains("subindex 0"), "must name the subindex: {msg}");
    assert!(
        msg.contains("reference_date"),
        "must mention reference_date: {msg}"
    );
}

#[test]
fn anticipated_slot_with_live_reference_date_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let manifest = [EntitySlot::anticipated(7, 0, true).with_reference_date(20_311_201)];
    write_fixture(dir.path(), 0, &manifest);

    let err = read_policy_checkpoint(dir.path()).expect_err(
        "an anticipated-thermal-state slot with a live reference_date must be rejected",
    );
    let msg = err.to_string();
    assert!(msg.contains("pool 0"), "must name the pool: {msg}");
    assert!(msg.contains("entity 7"), "must name the entity: {msg}");
    assert!(msg.contains("subindex 0"), "must name the subindex: {msg}");
    assert!(
        msg.contains("reference_date"),
        "must mention reference_date: {msg}"
    );
}

#[test]
fn inflow_lag_slot_with_live_interval_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let manifest = [EntitySlot::inflow_lag(5, 1, true).with_interval(20_320_101, 20_320_601)];
    write_fixture(dir.path(), 0, &manifest);

    let err = read_policy_checkpoint(dir.path())
        .expect_err("an inflow-lag slot with a live interval must be rejected");
    let msg = err.to_string();
    assert!(msg.contains("pool 0"), "must name the pool: {msg}");
    assert!(msg.contains("entity 5"), "must name the entity: {msg}");
}

#[test]
fn fully_sentinel_dated_checkpoint_still_accepted() {
    let dir = tempfile::tempdir().unwrap();
    let manifest = [
        EntitySlot::storage(1, true),
        EntitySlot::inflow_lag(1, 1, true),
        EntitySlot::transit_bucket(42, 0, true),
        EntitySlot::anticipated(7, 0, true),
    ];
    write_fixture(dir.path(), 0, &manifest);

    read_policy_checkpoint(dir.path()).expect(
        "a checkpoint whose reference_date/interval fields are all sentinel must be accepted",
    );
}

#[test]
fn first_reported_error_identical_across_slot_declaration_order() {
    let violation_10 =
        || EntitySlot::transit_bucket(10, 0, true).with_interval(20_320_101, 20_311_201);
    let violation_20 =
        || EntitySlot::transit_bucket(20, 0, true).with_interval(20_330_101, 20_320_101);

    let dir_a = tempfile::tempdir().unwrap();
    write_fixture(dir_a.path(), 0, &[violation_10(), violation_20()]);

    let dir_b = tempfile::tempdir().unwrap();
    write_fixture(dir_b.path(), 0, &[violation_20(), violation_10()]);

    let err_a =
        read_policy_checkpoint(dir_a.path()).expect_err("a reversed interval must be rejected");
    let err_b =
        read_policy_checkpoint(dir_b.path()).expect_err("a reversed interval must be rejected");
    assert_eq!(
        err_a.to_string(),
        err_b.to_string(),
        "declaration order must not change the first reported error"
    );
    assert!(
        err_a.to_string().contains("entity 10"),
        "the canonically-first violating slot (entity 10) must be the one reported: {err_a}"
    );
}
