//! Integration coverage for the `BoundaryInjection` load path's per-family
//! identity reconciliation: storage and inflow-lag reconcile by identity and
//! REJECT on a missing core slot; a `FullFcf`-typed manifest check stays on
//! the exact-match path, unaffected by wiring `reconcile` into
//! `load_boundary_cuts`.
//!
//! Also covers transit-bucket (an identical arrival interval blends at unit
//! weight, a miss defaults to `0.0`), sentinel-dated-anticipated default-`0.0`,
//! and the dated-anticipated
//! date-driven fan-out (`÷H_M` `Blend`, coverage-`Renormalize`) — including a
//! ring-sourced post-study slot reconciling bit-identically to the
//! pre-switchover shape, and an in-study ring slot (its own interval ending
//! at the boundary date) defaulting to `Zero`.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::doc_markdown,
    clippy::too_many_lines
)]

use chrono::NaiveDate;
use cobre_io::{
    EntitySlot, GraphManifest, PolicyCutRecord, ProducerBlock, StageCutsPayload, encode_slot_date,
    write_policy_checkpoint,
};
use cobre_sddp::test_support::{
    anticipated_slot, anticipated_slot_at, anticipated_slot_over, chain_graph_manifest,
    inflow_lag_slot, inflow_lag_slot_at, storage_slot, transit_bucket_slot_over, ymd,
};
use cobre_sddp::{
    BoundaryInjection, BoundaryLoadRequest, FullFcf, POLICY_COBRE_VERSION, PolicyStageManifest,
    load_boundary_cuts, validate_policy_load,
};

/// A single `HydroInflowLag` slot dated at the fixed `2031-01-01` reference —
/// distinct from the sentinel-dated shared [`inflow_lag_slot`], for the
/// tests in this suite that assert on the reference-date reconciliation gate.
fn dated_inflow_lag_slot(id: i32, lag_depth: u32) -> EntitySlot {
    inflow_lag_slot_at(id, lag_depth, 20_310_101)
}

/// Pool `pool`'s fixture `priced_state_date`: `2026-04-01` plus `pool`
/// months — at or before the April 2026 span every dated-anticipated
/// fixture in this suite prices, so a later date-driven boundary selector
/// never zeroes the coefficients those fixtures assert on.
fn fixture_priced_date(pool: u32) -> NaiveDate {
    cobre_sddp::test_support::fixture_priced_date(ymd(2026, 4, 1), pool)
}

#[test]
fn fixture_priced_date_pool_zero_is_april_2026() {
    assert_eq!(fixture_priced_date(0), ymd(2026, 4, 1));
}

fn producer_block() -> ProducerBlock {
    ProducerBlock {
        completed_iterations: 10,
        max_iterations: 50,
        forward_passes: 1,
        ..cobre_sddp::test_support::producer_block()
    }
}

/// Write a single-stage checkpoint whose one cut carries `coefficients`, one
/// per `manifest` slot in the same order.
fn write_checkpoint(dir: &std::path::Path, manifest: &[EntitySlot], coefficients: &[f64]) {
    let state_dimension = u32::try_from(coefficients.len()).expect("small coefficient count");
    let cut = PolicyCutRecord {
        cut_id: 0,
        slot_index: 0,
        iteration: 0,
        forward_pass_index: 0,
        intercept: 1.0,
        coefficients,
        is_active: true,
    };
    let cuts = vec![cut];
    let payload = StageCutsPayload {
        stage_id: 0,
        state_dimension,
        capacity: 1,
        warm_start_count: 0,
        cuts: &cuts,
        active_cut_indices: &[0],
        populated_count: 1,
        entity_manifest: manifest,
        cost_scale_factor: 1.0,
        node_id: 0,
        graph_stage_id: -1,
        priced_state_date: encode_slot_date(fixture_priced_date(0)),
    };
    let metadata =
        cobre_sddp::test_support::checkpoint_metadata(1, chain_graph_manifest(1), producer_block());
    write_policy_checkpoint(dir, &[payload], &[], &metadata, &[]).expect("write checkpoint");
}

/// Given a `BoundaryInjection` load whose source matches the current
/// terminal manifest's storage and inflow-lag slots by identity, when the
/// load runs, then it succeeds and the coefficients land at their
/// identity-matched positions.
#[test]
fn boundary_injection_storage_lag_identity_match_succeeds() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let manifest = vec![storage_slot(1), dated_inflow_lag_slot(1, 1)];
    write_checkpoint(tmp.path(), &manifest, &[10.0, 20.0]);

    let current = vec![storage_slot(1), dated_inflow_lag_slot(1, 1)];
    let cuts = load_boundary_cuts(&BoundaryLoadRequest::new(
        tmp.path(),
        fixture_priced_date(0),
        2,
        &current,
        1.0,
    ))
    .expect("an identity-matching boundary must load");

    assert_eq!(cuts.len(), 1);
    assert_eq!(
        cuts[0].coefficients,
        vec![10.0, 20.0],
        "coefficients must land at their identity-matched positions"
    );
}

/// Given a source lag slot and a target lag slot for the same hydro and lag
/// depth but with DIFFERENT non-sentinel `reference_date`s, when the
/// `BoundaryInjection` load runs, then it rejects naming the hydro, the lag
/// depth, and both dates.
#[test]
fn boundary_injection_differing_lag_reference_date_rejects() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let manifest = vec![inflow_lag_slot_at(1, 2, 20_300_301)];
    write_checkpoint(tmp.path(), &manifest, &[10.0]);

    let current = vec![inflow_lag_slot_at(1, 2, 20_300_401)];
    let result = load_boundary_cuts(&BoundaryLoadRequest::new(
        tmp.path(),
        fixture_priced_date(0),
        1,
        &current,
        1.0,
    ));

    let msg = result
        .expect_err("differing lag reference dates must reject")
        .to_string();
    assert!(msg.contains("hydro 1"), "must name hydro 1: {msg}");
    assert!(msg.contains("lag depth 2"), "must name lag depth 2: {msg}");
    assert!(
        msg.contains("2030-03-01"),
        "must name the source date: {msg}"
    );
    assert!(
        msg.contains("2030-04-01"),
        "must name the target date: {msg}"
    );
    assert!(
        !msg.contains("lag-depth-incompatible"),
        "a date mismatch is not a depth incompatibility: {msg}"
    );
}

/// Given source and target lag slots for hydro 1 at depth 2 that agree on a
/// non-sentinel `reference_date`, when the `BoundaryInjection` load runs,
/// then the coefficient copies and the reconciliation tally is identical to
/// the same load with both reference dates at the sentinel.
#[test]
fn boundary_injection_matching_lag_reference_date_tally_matches_sentinel() {
    let tmp_dated = tempfile::tempdir().expect("tempdir");
    let dated_manifest = vec![inflow_lag_slot_at(1, 2, 20_300_301)];
    write_checkpoint(tmp_dated.path(), &dated_manifest, &[10.0]);
    let dated_current = vec![inflow_lag_slot_at(1, 2, 20_300_301)];
    let dated_cuts = load_boundary_cuts(&BoundaryLoadRequest::new(
        tmp_dated.path(),
        fixture_priced_date(0),
        1,
        &dated_current,
        1.0,
    ))
    .expect("matching dated reference dates must load");
    assert_eq!(dated_cuts.len(), 1);
    assert_eq!(dated_cuts[0].coefficients, vec![10.0]);

    let tmp_sentinel = tempfile::tempdir().expect("tempdir");
    let sentinel_manifest = vec![inflow_lag_slot(1, 2)];
    write_checkpoint(tmp_sentinel.path(), &sentinel_manifest, &[10.0]);
    let sentinel_current = vec![inflow_lag_slot(1, 2)];
    let sentinel_cuts = load_boundary_cuts(&BoundaryLoadRequest::new(
        tmp_sentinel.path(),
        fixture_priced_date(0),
        1,
        &sentinel_current,
        1.0,
    ))
    .expect("sentinel reference dates must load");
    assert_eq!(sentinel_cuts.len(), 1);
    assert_eq!(sentinel_cuts[0].coefficients, vec![10.0]);

    assert_eq!(
        dated_cuts.report().inflow_lag.copy,
        sentinel_cuts.report().inflow_lag.copy,
        "a dated-equal match and a sentinel-both match must tally identically"
    );
}

/// Given a source lag slot at the sentinel `reference_date` and a target lag
/// slot dated — the shape `reserve_boundary_inflow_lag_slots` produces —
/// when the `BoundaryInjection` load runs, then the coefficient copies and
/// no warning is emitted; the symmetric case (dated source, sentinel target)
/// behaves identically.
#[test]
fn boundary_injection_undated_source_lag_still_copies() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let manifest = vec![inflow_lag_slot(1, 2)];
    write_checkpoint(tmp.path(), &manifest, &[10.0]);

    let current = vec![inflow_lag_slot_at(1, 2, 20_300_301)];
    let cuts = load_boundary_cuts(&BoundaryLoadRequest::new(
        tmp.path(),
        fixture_priced_date(0),
        1,
        &current,
        1.0,
    ))
    .expect("an undated source lag must still copy by identity");
    assert_eq!(cuts.len(), 1);
    assert_eq!(cuts[0].coefficients, vec![10.0]);

    let tmp_symmetric = tempfile::tempdir().expect("tempdir");
    let symmetric_manifest = vec![inflow_lag_slot_at(1, 2, 20_300_301)];
    write_checkpoint(tmp_symmetric.path(), &symmetric_manifest, &[10.0]);
    let symmetric_current = vec![inflow_lag_slot(1, 2)];
    let symmetric_cuts = load_boundary_cuts(&BoundaryLoadRequest::new(
        tmp_symmetric.path(),
        fixture_priced_date(0),
        1,
        &symmetric_current,
        1.0,
    ))
    .expect("an undated-target lag must still copy by identity");
    assert_eq!(symmetric_cuts.len(), 1);
    assert_eq!(symmetric_cuts[0].coefficients, vec![10.0]);
}

/// Given a current terminal storage slot for a hydro the boundary source
/// never prices, when the `BoundaryInjection` load runs, then it rejects,
/// naming the unpriced hydro.
#[test]
fn boundary_injection_different_hydro_source_rejects_naming_hydro() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let manifest = vec![storage_slot(7)];
    write_checkpoint(tmp.path(), &manifest, &[10.0]);

    let current = vec![storage_slot(42)];
    let result = load_boundary_cuts(&BoundaryLoadRequest::new(
        tmp.path(),
        fixture_priced_date(0),
        1,
        &current,
        1.0,
    ));

    let msg = result
        .expect_err("an unpriced hydro must reject")
        .to_string();
    assert!(
        msg.contains("42"),
        "error must name the unpriced hydro 42: {msg}"
    );
}

/// A `FullFcf`-typed manifest check over the same differently-shaped pair
/// stays on the exact-match path: `validate_policy_load::<FullFcf>` still
/// hard-rejects the per-slot mismatch that `BoundaryInjection` now reconciles
/// (and, for this exact case, still rejects, but with a different message) —
/// unaffected by wiring `reconcile` into `load_boundary_cuts`.
#[test]
fn full_fcf_manifest_check_unaffected_by_boundary_reconcile_wiring() {
    let source_slots = vec![storage_slot(7)];
    let current_slots = vec![storage_slot(42)];
    let empty_graph = GraphManifest::default();
    let source = PolicyStageManifest {
        state_dimension: 1,
        num_stages: 1,
        n_pools: 1,
        slots: &source_slots,
        graph: &empty_graph,
    };
    let current = PolicyStageManifest {
        state_dimension: 1,
        num_stages: 1,
        n_pools: 1,
        slots: &current_slots,
        graph: &empty_graph,
    };

    let full_fcf_result = validate_policy_load::<FullFcf>(POLICY_COBRE_VERSION, &source, &current);
    assert!(
        full_fcf_result.is_err(),
        "FullFcf's exact per-slot match must still hard-reject: {full_fcf_result:?}"
    );

    let boundary_result =
        validate_policy_load::<BoundaryInjection>(POLICY_COBRE_VERSION, &source, &current);
    assert!(
        boundary_result.is_ok(),
        "BoundaryInjection defers slot identity to reconcile::build_rebind, not this check: \
         {boundary_result:?}"
    );
}

/// Given a current terminal manifest with a DATED transit-bucket slot and a
/// source carrying no transit slots at all (the NEWAVE-shaped case), when the
/// `BoundaryInjection` load runs, then it succeeds and the transit
/// coefficient defaults to `0.0` — no source interval exists to overlap —
/// while the storage slot still loads its identity-matched coefficient.
#[test]
fn boundary_injection_transit_bucket_defaults_to_zero() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let manifest = vec![storage_slot(1), storage_slot(2)];
    write_checkpoint(tmp.path(), &manifest, &[10.0, 20.0]);

    let current = vec![
        storage_slot(1),
        transit_bucket_slot_over(
            2,
            1,
            encode_slot_date(ymd(2026, 4, 8)),
            encode_slot_date(ymd(2026, 4, 15)),
        ),
    ];
    let cuts = load_boundary_cuts(&BoundaryLoadRequest::new(
        tmp.path(),
        fixture_priced_date(0),
        2,
        &current,
        1.0,
    ))
    .expect("a transit-only-target load must succeed via the default-zero arm");

    assert_eq!(cuts.len(), 1);
    assert_eq!(
        cuts[0].coefficients,
        vec![10.0, 0.0],
        "the transit slot must default to 0.0"
    );
}

/// Given a boundary SOURCE that prices a transit-bucket arrival interval for
/// downstream hydro 2 the current study's own transit-bucket target does not
/// overlap in calendar time (same entity, same state dimension, disjoint
/// arrival intervals), when the `BoundaryInjection` load runs, then the
/// source coupling is dropped (its coefficient discarded) and the load
/// succeeds with no warning — the C17 source-drop is surfaced only in the
/// reconciliation report. The target's own transit slot still defaults to
/// `0.0`.
#[test]
fn boundary_injection_dropped_source_transit_coupling_loads_silently_and_is_reported() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let manifest = vec![
        storage_slot(1),
        transit_bucket_slot_over(
            2,
            1,
            encode_slot_date(ymd(2026, 4, 1)),
            encode_slot_date(ymd(2026, 5, 1)),
        ),
    ];
    write_checkpoint(tmp.path(), &manifest, &[10.0, 20.0]);

    let current = vec![
        storage_slot(1),
        transit_bucket_slot_over(
            2,
            9,
            encode_slot_date(ymd(2026, 5, 1)),
            encode_slot_date(ymd(2026, 6, 1)),
        ),
    ];
    let cuts = load_boundary_cuts(&BoundaryLoadRequest::new(
        tmp.path(),
        fixture_priced_date(0),
        2,
        &current,
        1.0,
    ))
    .expect("a dropped source coupling must load, never reject, by default");

    assert_eq!(cuts.len(), 1);
    assert_eq!(
        cuts[0].coefficients,
        vec![10.0, 0.0],
        "the target's own non-overlapping transit slot still defaults to 0.0"
    );

    let report = cuts.report();
    assert_eq!(
        report.dropped_source_slots.len(),
        1,
        "the report must still name the dropped transit bucket: {:?}",
        report.dropped_source_slots
    );
    let dropped = &report.dropped_source_slots[0];
    assert_eq!(dropped.family, "transit-bucket");
    assert_eq!(dropped.entity_id, 2);
    assert_eq!(
        dropped.interval,
        Some((ymd(2026, 4, 1), ymd(2026, 5, 1))),
        "the dropped slot's own arrival interval, not the target's non-overlapping one"
    );
}

/// Given the same boundary source and target as
/// [`boundary_injection_dropped_source_transit_coupling_loads_silently_and_is_reported`]
/// but a request built with `.with_strict(true)`, when the `BoundaryInjection`
/// load runs, then it rejects naming the boundary path, the dropped total and
/// the dropping family, and no cuts are returned.
#[test]
fn boundary_injection_dropped_source_transit_coupling_rejects_under_strict() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let manifest = vec![
        storage_slot(1),
        transit_bucket_slot_over(
            2,
            1,
            encode_slot_date(ymd(2026, 4, 1)),
            encode_slot_date(ymd(2026, 5, 1)),
        ),
    ];
    write_checkpoint(tmp.path(), &manifest, &[10.0, 20.0]);

    let current = vec![
        storage_slot(1),
        transit_bucket_slot_over(
            2,
            9,
            encode_slot_date(ymd(2026, 5, 1)),
            encode_slot_date(ymd(2026, 6, 1)),
        ),
    ];
    let result = load_boundary_cuts(
        &BoundaryLoadRequest::new(tmp.path(), fixture_priced_date(0), 2, &current, 1.0)
            .with_strict(true),
    );

    let err = result
        .expect_err("a dropped source coupling must reject under strict")
        .to_string();
    assert!(
        err.contains(&format!("boundary policy at {}", tmp.path().display())),
        "must name the boundary path: {err}"
    );
    assert!(
        err.contains(
            "is a superset: 1 source slot(s) price entities this study does not model \
             (transit-bucket: 1)"
        ),
        "must name the total and the dropping family: {err}"
    );
    assert!(
        err.contains("; see the reconciliation report or set policy.boundary.strict = false"),
        "must state the remedy: {err}"
    );
}

/// Given a current terminal manifest with a sentinel-dated anticipated slot
/// and a source carrying no counterpart, when the `BoundaryInjection` load
/// runs, then it succeeds and the anticipated coefficient defaults to `0.0`,
/// while the storage slot still loads its identity-matched coefficient.
#[test]
fn boundary_injection_sentinel_anticipated_defaults_to_zero() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let manifest = vec![storage_slot(1), storage_slot(2)];
    write_checkpoint(tmp.path(), &manifest, &[10.0, 20.0]);

    let current = vec![storage_slot(1), anticipated_slot(9, 0)];
    let cuts = load_boundary_cuts(&BoundaryLoadRequest::new(
        tmp.path(),
        fixture_priced_date(0),
        2,
        &current,
        1.0,
    ))
    .expect("a sentinel-anticipated-target load must succeed via the default-zero arm");

    assert_eq!(cuts.len(), 1);
    assert_eq!(
        cuts[0].coefficients,
        vec![10.0, 0.0],
        "the sentinel-dated anticipated slot must default to 0.0"
    );
}

/// Given a source already shaped identically to the current terminal
/// manifest (storage, inflow-lag, and a sentinel-dated anticipated slot),
/// when the `BoundaryInjection` load runs, then the injected coefficients
/// equal the source cut's own coefficients bit-for-bit (`f64::to_bits`) — the
/// superset property: reconciling a target-shaped source never regresses
/// today's exact-match load.
#[test]
fn boundary_injection_target_shaped_source_reconciles_bit_identically() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let manifest = vec![
        storage_slot(1),
        dated_inflow_lag_slot(1, 1),
        anticipated_slot(9, 0),
    ];
    let coefficients = vec![10.5, -3.25, 0.0];
    write_checkpoint(tmp.path(), &manifest, &coefficients);

    let current = manifest;
    let cuts = load_boundary_cuts(&BoundaryLoadRequest::new(
        tmp.path(),
        fixture_priced_date(0),
        3,
        &current,
        1.0,
    ))
    .expect("a target-shaped source must load");

    assert_eq!(cuts.len(), 1);
    for (actual, expected) in cuts[0].coefficients.iter().zip(coefficients.iter()) {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "reconciling a target-shaped source must reproduce its coefficients bit-for-bit: \
             {actual} != {expected}"
        );
    }
}

/// Given a current terminal manifest with a transit-bucket slot and a source
/// carrying a matching transit-bucket slot at the SAME arrival interval (a
/// Cobre-to-Cobre boundary with matching transit arcs), when the
/// `BoundaryInjection` load runs, then the transit coefficient blends at
/// unit weight from the source verbatim, never zeroed — distinct from
/// `boundary_injection_transit_bucket_defaults_to_zero`'s no-source-transit
/// (NEWAVE) case, which still defaults to `0.0` on a miss — and the report
/// tallies the slot as `fan_out`, not `copy`: an exact interval match now
/// routes through the date-driven join.
#[test]
fn boundary_injection_transit_bucket_blends_on_identical_arrival_interval() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let start = encode_slot_date(ymd(2026, 4, 1));
    let end = encode_slot_date(ymd(2026, 5, 1));
    let manifest = vec![storage_slot(1), transit_bucket_slot_over(2, 1, start, end)];
    write_checkpoint(tmp.path(), &manifest, &[10.0, 30.0]);

    let current = vec![storage_slot(1), transit_bucket_slot_over(2, 1, start, end)];
    let cuts = load_boundary_cuts(&BoundaryLoadRequest::new(
        tmp.path(),
        fixture_priced_date(0),
        2,
        &current,
        1.0,
    ))
    .expect("a matching transit-bucket boundary must load");

    assert_eq!(cuts.len(), 1);
    assert_eq!(
        cuts[0].coefficients,
        vec![10.0, 30.0],
        "the transit slot blends its identity-matched source coefficient at unit weight, not \
         zero it"
    );

    let report = cuts.report();
    assert_eq!(
        report.transit_bucket.fan_out, 1,
        "an interval-join match tallies as fan_out"
    );
    assert_eq!(
        report.transit_bucket.copy, 0,
        "never copy, even at unit weight"
    );
}

/// Given a source transit-bucket slot spanning one calendar month and a
/// current transit-bucket slot spanning one week strictly inside it — a
/// mismatched-calendar transit boundary, unlike
/// `boundary_injection_transit_bucket_blends_on_identical_arrival_interval`'s
/// matching-interval case — when the `BoundaryInjection` load runs, then the
/// loaded coefficient equals the source coefficient scaled by the target's
/// own span over the source's own span: the target draws the source's
/// density over its own overlapping hours. The report still tallies the
/// slot as a fully-covered `fan_out`, never `straddling`.
#[test]
fn boundary_injection_transit_bucket_blends_at_fractional_source_hours() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let source_start = ymd(2026, 4, 1);
    let source_end = ymd(2026, 5, 1);
    let manifest = vec![transit_bucket_slot_over(
        2,
        1,
        encode_slot_date(source_start),
        encode_slot_date(source_end),
    )];
    let source_coeff = 30.0;
    write_checkpoint(tmp.path(), &manifest, &[source_coeff]);

    let target_start = ymd(2026, 4, 8);
    let target_end = ymd(2026, 4, 15);
    let current = vec![transit_bucket_slot_over(
        2,
        1,
        encode_slot_date(target_start),
        encode_slot_date(target_end),
    )];
    let cuts = load_boundary_cuts(&BoundaryLoadRequest::new(
        tmp.path(),
        fixture_priced_date(0),
        1,
        &current,
        1.0,
    ))
    .expect("a strictly-inside transit-bucket target must load");

    assert_eq!(cuts.len(), 1);

    let source_span_hours = (source_end - source_start).num_days() as f64 * 24.0;
    let target_span_hours = (target_end - target_start).num_days() as f64 * 24.0;
    let expected = source_coeff * target_span_hours / source_span_hours;
    let actual = cuts[0].coefficients[0];
    assert!(
        (actual - expected).abs() < expected.abs() * 1e-9,
        "coefficient {actual} != expected {expected} (source_coeff * target_span_hours / \
         source_span_hours)"
    );

    let report = cuts.report();
    assert_eq!(
        report.transit_bucket.fan_out, 1,
        "a fully-covered target interval tallies as fan_out"
    );
    assert_eq!(
        report.transit_bucket.straddling, 0,
        "full coverage by one source interval is never straddling"
    );
}

/// One source month (April 2026, `π_M = 300.0`) and the four `[start, end)`
/// week-lane intervals that exactly tile it — shared by the fan-out matrix and
/// interior-weeks conservation tests below.
fn april_2026_weekly_intervals() -> [(NaiveDate, NaiveDate); 4] {
    [
        (ymd(2026, 4, 1), ymd(2026, 4, 8)),
        (ymd(2026, 4, 8), ymd(2026, 4, 15)),
        (ymd(2026, 4, 15), ymd(2026, 4, 22)),
        (ymd(2026, 4, 22), ymd(2026, 5, 1)),
    ]
}

/// Given a source manifest with one monthly anticipated slot for thermal `9`
/// (`π_M = 300.0`, April 2026) and a target manifest with 4 post-horizon lane
/// slots for thermal `9` whose `[start, end)` intervals tile that month
/// exactly, when `load_boundary_cuts` runs end-to-end, then each lane
/// coefficient lands at `π_M · overlap/H_M` — the `÷H_M` distribute.
#[test]
fn boundary_injection_dated_anticipated_fan_out_matrix() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let source_manifest = vec![
        storage_slot(1),
        anticipated_slot_at(9, 0, 20_260_401),
        anticipated_slot(9, 1),
        anticipated_slot(9, 2),
        anticipated_slot(9, 3),
    ];
    write_checkpoint(tmp.path(), &source_manifest, &[10.0, 300.0, 0.0, 0.0, 0.0]);

    let weeks = april_2026_weekly_intervals();
    let current = vec![
        storage_slot(1),
        anticipated_slot_over(
            9,
            100,
            encode_slot_date(weeks[0].0),
            encode_slot_date(weeks[0].1),
        ),
        anticipated_slot_over(
            9,
            101,
            encode_slot_date(weeks[1].0),
            encode_slot_date(weeks[1].1),
        ),
        anticipated_slot_over(
            9,
            102,
            encode_slot_date(weeks[2].0),
            encode_slot_date(weeks[2].1),
        ),
        anticipated_slot_over(
            9,
            103,
            encode_slot_date(weeks[3].0),
            encode_slot_date(weeks[3].1),
        ),
    ];

    let cuts = load_boundary_cuts(&BoundaryLoadRequest::new(
        tmp.path(),
        fixture_priced_date(0),
        5,
        &current,
        1.0,
    ))
    .expect("a fully-covered dated anticipated fan-out must load");

    assert_eq!(cuts.len(), 1);
    assert_eq!(
        cuts[0].coefficients[0], 10.0,
        "the storage slot still copies its identity-matched coefficient"
    );

    let h_m = 30.0 * 24.0;
    let expected_widths_days = [7.0, 7.0, 7.0, 9.0];
    for (lane, &width_days) in expected_widths_days.iter().enumerate() {
        let expected = 300.0 * (width_days * 24.0) / h_m;
        let actual = cuts[0].coefficients[lane + 1];
        assert!(
            (actual - expected).abs() < expected.abs() * 1e-9,
            "lane {lane}: coefficient {actual} != expected {expected} (π_M · overlap/H_M)"
        );
    }
}

/// Given the same one-source-month → 4-target-lane fan-out as
/// `boundary_injection_dated_anticipated_fan_out_matrix`, when the fanned lane
/// coefficients are summed, then the sum reproduces the source's own monthly
/// valuation `π_M` — the interior-weeks conservation invariant: a
/// constant per-week commitment `K` sees `Σ_w β_w · K = π_M · K` for any `K`,
/// so the coefficient-level identity `Σ_w β_w == π_M` holds with `K` divided
/// out.
#[test]
fn boundary_injection_dated_anticipated_interior_weeks_conservation() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let source_manifest = vec![
        anticipated_slot_at(9, 0, 20_260_401),
        anticipated_slot(9, 1),
        anticipated_slot(9, 2),
        anticipated_slot(9, 3),
    ];
    let source_coeff = 300.0;
    write_checkpoint(tmp.path(), &source_manifest, &[source_coeff, 0.0, 0.0, 0.0]);

    let weeks = april_2026_weekly_intervals();
    let current = vec![
        anticipated_slot_over(
            9,
            100,
            encode_slot_date(weeks[0].0),
            encode_slot_date(weeks[0].1),
        ),
        anticipated_slot_over(
            9,
            101,
            encode_slot_date(weeks[1].0),
            encode_slot_date(weeks[1].1),
        ),
        anticipated_slot_over(
            9,
            102,
            encode_slot_date(weeks[2].0),
            encode_slot_date(weeks[2].1),
        ),
        anticipated_slot_over(
            9,
            103,
            encode_slot_date(weeks[3].0),
            encode_slot_date(weeks[3].1),
        ),
    ];

    let cuts = load_boundary_cuts(&BoundaryLoadRequest::new(
        tmp.path(),
        fixture_priced_date(0),
        4,
        &current,
        1.0,
    ))
    .expect("a fully-covered dated anticipated fan-out must load");

    assert_eq!(cuts.len(), 1);
    let summed: f64 = cuts[0].coefficients.iter().sum();
    assert!(
        (summed - source_coeff).abs() < source_coeff.abs() * 1e-9,
        "Σ_w β_w = {summed} must reproduce the source monthly valuation π_M = {source_coeff}"
    );
}

/// Given the same one-source-month → 4-target-lane fan-out as
/// `boundary_injection_dated_anticipated_fan_out_matrix`, when `.report()` is
/// read, then the anticipated family's `fan_out`/`straddling`/`default_zero`
/// match the fully-covered fan-out and the coverage line renders the
/// expected shape.
#[test]
fn boundary_injection_report_fan_out_matrix_coverage() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let source_manifest = vec![
        storage_slot(1),
        anticipated_slot_at(9, 0, 20_260_401),
        anticipated_slot(9, 1),
        anticipated_slot(9, 2),
        anticipated_slot(9, 3),
    ];
    write_checkpoint(tmp.path(), &source_manifest, &[10.0, 300.0, 0.0, 0.0, 0.0]);

    let weeks = april_2026_weekly_intervals();
    let current = vec![
        storage_slot(1),
        anticipated_slot_over(
            9,
            100,
            encode_slot_date(weeks[0].0),
            encode_slot_date(weeks[0].1),
        ),
        anticipated_slot_over(
            9,
            101,
            encode_slot_date(weeks[1].0),
            encode_slot_date(weeks[1].1),
        ),
        anticipated_slot_over(
            9,
            102,
            encode_slot_date(weeks[2].0),
            encode_slot_date(weeks[2].1),
        ),
        anticipated_slot_over(
            9,
            103,
            encode_slot_date(weeks[3].0),
            encode_slot_date(weeks[3].1),
        ),
    ];

    let cuts = load_boundary_cuts(&BoundaryLoadRequest::new(
        tmp.path(),
        fixture_priced_date(0),
        5,
        &current,
        1.0,
    ))
    .expect("a fully-covered dated anticipated fan-out must load");

    let report = cuts.report();
    assert!(report.reconciled);
    assert_eq!(
        report.anticipated.fan_out, 4,
        "4 fully-covered target lanes"
    );
    assert_eq!(report.anticipated.straddling, 0);
    assert_eq!(report.anticipated.default_zero, 0);
    assert_eq!(
        report.anticipated_coverage.source_interval_count, 1,
        "one source delivery interval (April 2026)"
    );

    let lines = report.detail_lines();
    assert!(
        lines.iter().any(|l| l
            == "anticipated: 1 source delivery intervals fanned to 4 target slots (0 \
                straddling, overlap-blended), 0 months defaulted"),
        "coverage line must match the expected shape: {lines:?}"
    );

    assert!(
        report.straddling_slots.is_empty(),
        "a fully-covered fan-out straddles nothing: {:?}",
        report.straddling_slots
    );
    // The source's three undated ring-buffer pad slots (subindex 1..=3,
    // alongside the one priced month at subindex 0) are structural pads
    // (`is_structural_pad`): unreferenced by any op, like the fanned month's
    // slot, but excluded from the drop tally because they price nothing.
    assert!(
        report.dropped_source_slots.is_empty(),
        "a fully-covered fan-out's only unreferenced source slots are structural pads, which \
         are never reported dropped: {:?}",
        report.dropped_source_slots
    );
}

/// Given a target-shaped source (storage + lag + sentinel anticipated) that
/// loads bit-identically, when `.report()` is read, then every family
/// reports only `copy`, `fan_out == 0` everywhere — matching the superset
/// guarantee.
#[test]
fn boundary_injection_report_target_shaped_superset_is_copy_only() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let manifest = vec![
        storage_slot(1),
        dated_inflow_lag_slot(1, 1),
        anticipated_slot(9, 0),
    ];
    let coefficients = vec![10.5, -3.25, 0.0];
    write_checkpoint(tmp.path(), &manifest, &coefficients);

    let current = manifest;
    let cuts = load_boundary_cuts(&BoundaryLoadRequest::new(
        tmp.path(),
        fixture_priced_date(0),
        3,
        &current,
        1.0,
    ))
    .expect("a target-shaped source must load");

    let report = cuts.report();
    assert!(report.reconciled);
    assert_eq!(report.storage.copy, 1);
    assert_eq!(report.inflow_lag.copy, 1);
    assert_eq!(
        report.anticipated.copy, 0,
        "sentinel Zero is excluded, not copy"
    );
    assert_eq!(report.anticipated.fan_out, 0);
    assert_eq!(report.transit_bucket.fan_out, 0);
    assert_eq!(report.other_identity.fan_out, 0);

    assert!(
        report.straddling_slots.is_empty(),
        "an all-Copy/Zero reconciliation straddles nothing: {:?}",
        report.straddling_slots
    );
    // The sentinel anticipated slot resolves to `Zero` and is never
    // referenced by any op, but it is a structural pad
    // (`is_structural_pad`): it prices nothing, so it is excluded from the
    // drop tally exactly as it is excluded from `default_zero`.
    assert!(
        report.dropped_source_slots.is_empty(),
        "the target-shaped source's own sentinel pad is a structural pad, never reported \
         dropped: {:?}",
        report.dropped_source_slots
    );
}

/// Given a source manifest carrying a dated anticipated slot for thermal 33
/// at subindex 1 over `[2032-01-01, 2032-02-01)` that the current study does
/// not model at all, when `load_boundary_cuts` runs end-to-end, then
/// `report.dropped_source_slots` names that slot with its own delivery
/// interval.
#[test]
fn boundary_injection_report_lists_every_dropped_slot_with_its_interval() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let source_manifest = vec![storage_slot(1), anticipated_slot_at(33, 1, 20_320_101)];
    write_checkpoint(tmp.path(), &source_manifest, &[10.0, 300.0]);

    let current = vec![storage_slot(1)];
    let cuts = load_boundary_cuts(&BoundaryLoadRequest::new(
        tmp.path(),
        fixture_priced_date(0),
        1,
        &current,
        1.0,
    ))
    .expect("a superset source must load, never reject");

    let report = cuts.report();
    assert_eq!(report.anticipated.dropped_source, 1);
    assert_eq!(
        report.dropped_source_slots.len(),
        1,
        "{:?}",
        report.dropped_source_slots
    );
    let dropped = &report.dropped_source_slots[0];
    assert_eq!(dropped.family, "anticipated");
    assert_eq!(dropped.entity_id, 33);
    assert_eq!(dropped.subindex, 1);
    assert_eq!(dropped.interval, Some((ymd(2032, 1, 1), ymd(2032, 2, 1))));
    assert_eq!(dropped.reference_date, None);
}

/// Given a boundary checkpoint with an empty manifest, when `.report()` is
/// read, then `reconciled == false` and the render states a dimension-only
/// load.
#[test]
fn boundary_injection_report_empty_manifest_is_unreconciled() {
    let tmp = tempfile::tempdir().expect("tempdir");
    write_checkpoint(tmp.path(), &[], &[10.0, 20.0]);

    let current = vec![storage_slot(1), storage_slot(2)];
    let cuts = load_boundary_cuts(&BoundaryLoadRequest::new(
        tmp.path(),
        fixture_priced_date(0),
        2,
        &current,
        1.0,
    ))
    .expect("an absent manifest must still load cuts");

    let report = cuts.report();
    assert!(!report.reconciled);
    assert!(
        report.summary_line().contains("dimension-only"),
        "must state a dimension-only load: {}",
        report.summary_line()
    );
    assert!(report.detail_lines().is_empty());
}

/// Given a monthly-priced source (April 2026, `π_M = 300.0`) and a current
/// terminal manifest whose ring-residue `subindex` values (100/101) tag a
/// post-study-targeted provenance, when `load_boundary_cuts` fans one
/// fully-covered lane and one lane straddling into unpriced (post-study) time,
/// then the coefficients equal the `Blend` (`π_M · overlap/H_M`, the same value
/// `boundary_injection_dated_anticipated_fan_out_matrix` asserts for a 7-day
/// lane) and `Renormalize` (`π_M · H_w/H_M`) values — reconcile keys on the
/// slot's interval, never the ring `subindex`, so ring provenance reconciles
/// identically to the retired lane shape.
#[test]
fn boundary_injection_ring_sourced_post_study_fan_out_matches_pre_switchover() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let source_manifest = vec![
        storage_slot(1),
        anticipated_slot_at(9, 0, 20_260_401),
        anticipated_slot(9, 1),
    ];
    write_checkpoint(tmp.path(), &source_manifest, &[10.0, 300.0, 0.0]);

    let current = vec![
        storage_slot(1),
        anticipated_slot_over(
            9,
            100,
            encode_slot_date(ymd(2026, 4, 1)),
            encode_slot_date(ymd(2026, 4, 8)),
        ),
        anticipated_slot_over(
            9,
            101,
            encode_slot_date(ymd(2026, 4, 22)),
            encode_slot_date(ymd(2026, 5, 8)),
        ),
    ];

    let cuts = load_boundary_cuts(&BoundaryLoadRequest::new(
        tmp.path(),
        fixture_priced_date(0),
        3,
        &current,
        1.0,
    ))
    .expect("a ring-sourced post-study dated fan-out must load");

    assert_eq!(cuts.len(), 1);
    assert_eq!(
        cuts[0].coefficients[0], 10.0,
        "the storage slot still copies its identity-matched coefficient"
    );

    let h_m = 30.0 * 24.0;
    let expected_blend = 300.0 * (7.0 * 24.0) / h_m;
    assert!(
        (cuts[0].coefficients[1] - expected_blend).abs() < expected_blend.abs() * 1e-9,
        "fully-covered ring lane must Blend to π_M · overlap/H_M = {expected_blend}, got {}",
        cuts[0].coefficients[1]
    );
    let expected_renorm = 300.0 * (16.0 * 24.0) / h_m;
    assert!(
        (cuts[0].coefficients[2] - expected_renorm).abs() < expected_renorm.abs() * 1e-9,
        "straddling ring lane must Renormalize to π_M · H_w/H_M = {expected_renorm}, got {}",
        cuts[0].coefficients[2]
    );
}

/// Given a current terminal manifest carrying a live anticipated ring slot
/// whose interval ENDS at the boundary date — an in-study ring slot (a
/// within-horizon delivery, e.g. a `K = 0` self-delivery) — when
/// `load_boundary_cuts` runs, then that slot reconciles to `0.0`, not a reject
/// and not a blend, even though the source carries a month whose date would
/// overlap it if it were wrongly fanned out. The integration mirror of the unit
/// pin `anticipated_target_ending_at_the_boundary_date_zeroes`.
#[test]
fn boundary_injection_dated_target_interval_ending_at_the_boundary_date_yields_zero() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let source_manifest = vec![storage_slot(1), anticipated_slot_at(9, 0, 20_260_301)];
    write_checkpoint(tmp.path(), &source_manifest, &[10.0, 300.0]);

    let current = vec![
        storage_slot(1),
        anticipated_slot_over(
            9,
            0,
            encode_slot_date(ymd(2026, 3, 1)),
            encode_slot_date(ymd(2026, 4, 1)),
        ),
    ];

    let cuts = load_boundary_cuts(&BoundaryLoadRequest::new(
        tmp.path(),
        fixture_priced_date(0),
        2,
        &current,
        1.0,
    ))
    .expect("an in-study ring slot ending at the boundary date must load, defaulting to 0.0");

    assert_eq!(cuts.len(), 1);
    assert_eq!(
        cuts[0].coefficients,
        vec![10.0, 0.0],
        "a target interval ending at or before the boundary date is an in-study delivery: it \
         defaults to 0.0, never fans out against the overlapping source month"
    );
}

/// Given a boundary SOURCE that carries an inflow-lag depth for hydro 1 the
/// current study's own inflow-lag block does not carry (same hydro, one
/// deeper lag order), when the `BoundaryInjection` load runs, then the extra
/// depth is dropped (its coefficient discarded) and the load succeeds with
/// no warning — the source-drop is surfaced only in the reconciliation
/// report. The storage and matching lag-1 coefficients still land
/// identity-matched.
#[test]
fn boundary_injection_dropped_source_inflow_lag_loads_silently_and_is_reported() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let manifest = vec![
        storage_slot(1),
        dated_inflow_lag_slot(1, 1),
        dated_inflow_lag_slot(1, 2),
    ];
    write_checkpoint(tmp.path(), &manifest, &[10.0, 20.0, 30.0]);

    let current = vec![storage_slot(1), dated_inflow_lag_slot(1, 1)];
    let cuts = load_boundary_cuts(&BoundaryLoadRequest::new(
        tmp.path(),
        fixture_priced_date(0),
        2,
        &current,
        1.0,
    ))
    .expect("a dropped source lag depth must load, never reject, by default");

    assert_eq!(cuts.len(), 1);
    assert_eq!(
        cuts[0].coefficients,
        vec![10.0, 20.0],
        "the identity-matched storage and lag-1 coefficients still land"
    );

    let report = cuts.report();
    assert_eq!(
        report.dropped_source_slots.len(),
        1,
        "the report must still name the dropped lag depth: {:?}",
        report.dropped_source_slots
    );
    let dropped = &report.dropped_source_slots[0];
    assert_eq!(dropped.family, "inflow-lag");
    assert_eq!(dropped.entity_id, 1);
    assert_eq!(dropped.subindex, 2);
    assert_eq!(
        dropped.reference_date,
        Some(ymd(2031, 1, 1)),
        "the dropped slot's own reference date"
    );
}

/// Given the same boundary source and target as
/// [`boundary_injection_dropped_source_inflow_lag_loads_silently_and_is_reported`]
/// but a request built with `.with_strict(true)`, when the `BoundaryInjection`
/// load runs, then it rejects naming the boundary path, the dropped total and
/// the dropping family, and no cuts are returned.
#[test]
fn boundary_injection_dropped_source_inflow_lag_rejects_under_strict() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let manifest = vec![
        storage_slot(1),
        dated_inflow_lag_slot(1, 1),
        dated_inflow_lag_slot(1, 2),
    ];
    write_checkpoint(tmp.path(), &manifest, &[10.0, 20.0, 30.0]);

    let current = vec![storage_slot(1), dated_inflow_lag_slot(1, 1)];
    let result = load_boundary_cuts(
        &BoundaryLoadRequest::new(tmp.path(), fixture_priced_date(0), 2, &current, 1.0)
            .with_strict(true),
    );

    let err = result
        .expect_err("a dropped source lag depth must reject under strict")
        .to_string();
    assert!(
        err.contains(&format!("boundary policy at {}", tmp.path().display())),
        "must name the boundary path: {err}"
    );
    assert!(
        err.contains(
            "is a superset: 1 source slot(s) price entities this study does not model \
             (inflow-lag: 1)"
        ),
        "must name the total and the dropping family: {err}"
    );
    assert!(
        err.contains("; see the reconciliation report or set policy.boundary.strict = false"),
        "must state the remedy: {err}"
    );
}

/// Given a boundary SOURCE that prices a dated anticipated lane for thermal 7
/// the current study's own manifest does not model at all (no matching
/// entity id, not merely a different lane), when the `BoundaryInjection`
/// load runs, then the source lane is dropped (its coefficient discarded)
/// and the load succeeds with no warning — the source-drop is surfaced only
/// in the reconciliation report. The storage slot still loads its
/// identity-matched coefficient.
#[test]
fn boundary_injection_dropped_source_anticipated_lane_loads_silently_and_is_reported() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let manifest = vec![storage_slot(1), anticipated_slot_at(7, 0, 20_260_401)];
    write_checkpoint(tmp.path(), &manifest, &[10.0, 300.0]);

    let current = vec![storage_slot(1)];
    let cuts = load_boundary_cuts(&BoundaryLoadRequest::new(
        tmp.path(),
        fixture_priced_date(0),
        1,
        &current,
        1.0,
    ))
    .expect("a dropped source anticipated lane must load, never reject, by default");

    assert_eq!(cuts.len(), 1);
    assert_eq!(
        cuts[0].coefficients,
        vec![10.0],
        "the identity-matched storage coefficient still lands"
    );

    let report = cuts.report();
    assert_eq!(
        report.dropped_source_slots.len(),
        1,
        "the report must still name the dropped anticipated lane: {:?}",
        report.dropped_source_slots
    );
    let dropped = &report.dropped_source_slots[0];
    assert_eq!(dropped.family, "anticipated");
    assert_eq!(dropped.entity_id, 7);
    assert_eq!(dropped.subindex, 0);
    assert_eq!(
        dropped.interval,
        Some((ymd(2026, 4, 1), ymd(2026, 5, 1))),
        "the dropped slot's own delivery interval"
    );
}

/// Given the same boundary source and target as
/// [`boundary_injection_dropped_source_anticipated_lane_loads_silently_and_is_reported`]
/// but a request built with `.with_strict(true)`, when the `BoundaryInjection`
/// load runs, then it rejects naming the boundary path, the dropped total and
/// the dropping family, and no cuts are returned.
#[test]
fn boundary_injection_dropped_source_anticipated_lane_rejects_under_strict() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let manifest = vec![storage_slot(1), anticipated_slot_at(7, 0, 20_260_401)];
    write_checkpoint(tmp.path(), &manifest, &[10.0, 300.0]);

    let current = vec![storage_slot(1)];
    let result = load_boundary_cuts(
        &BoundaryLoadRequest::new(tmp.path(), fixture_priced_date(0), 1, &current, 1.0)
            .with_strict(true),
    );

    let err = result
        .expect_err("a dropped source anticipated lane must reject under strict")
        .to_string();
    assert!(
        err.contains(&format!("boundary policy at {}", tmp.path().display())),
        "must name the boundary path: {err}"
    );
    assert!(
        err.contains(
            "is a superset: 1 source slot(s) price entities this study does not model \
             (anticipated: 1)"
        ),
        "must name the total and the dropping family: {err}"
    );
    assert!(
        err.contains("; see the reconciliation report or set policy.boundary.strict = false"),
        "must state the remedy: {err}"
    );
}

/// Given the same current manifest as
/// [`boundary_injection_dropped_source_anticipated_lane_loads_silently_and_is_reported`]
/// but a source carrying a SENTINEL-dated anticipated slot in place of the
/// dated one, when the `BoundaryInjection` load runs, then the pad is
/// excluded from the dropped-source tally under the default request, and a
/// request built with `.with_strict(true)` still succeeds — an excluded pad
/// can never trip the strict reject.
#[test]
fn boundary_injection_sentinel_source_anticipated_pad_is_not_a_dropped_source() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let manifest = vec![storage_slot(1), anticipated_slot(7, 0)];
    write_checkpoint(tmp.path(), &manifest, &[10.0, 300.0]);

    let current = vec![storage_slot(1)];

    let cuts = load_boundary_cuts(&BoundaryLoadRequest::new(
        tmp.path(),
        fixture_priced_date(0),
        1,
        &current,
        1.0,
    ))
    .expect("a sentinel-dated source pad must never be treated as dropped");
    let report = cuts.report();
    assert!(
        report.dropped_source_slots.is_empty(),
        "a structural pad must not appear in dropped_source_slots: {:?}",
        report.dropped_source_slots
    );
    assert_eq!(
        report.anticipated.dropped_source, 0,
        "a structural pad must not increment the anticipated dropped_source tally"
    );

    load_boundary_cuts(
        &BoundaryLoadRequest::new(tmp.path(), fixture_priced_date(0), 1, &current, 1.0)
            .with_strict(true),
    )
    .expect("an excluded structural pad must never trip the strict reject");
}

/// Given a boundary SOURCE dropping one storage slot (an extra hydro 2
/// reservoir the current study does not model) AND one dated anticipated
/// lane (for thermal 7, likewise unmodeled), when the `BoundaryInjection`
/// load runs with `.with_strict(true)`, then it rejects naming both
/// dropping families in [`BoundaryReconciliationReport::families`] order;
/// without `strict` the same source loads, and both families' tallies read
/// their own single drop.
#[test]
fn boundary_injection_multi_family_superset_rejects_naming_families_in_order() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let manifest = vec![
        storage_slot(1),
        storage_slot(2),
        anticipated_slot_at(7, 0, 20_260_401),
    ];
    write_checkpoint(tmp.path(), &manifest, &[10.0, 20.0, 300.0]);

    let current = vec![storage_slot(1)];

    let cuts = load_boundary_cuts(&BoundaryLoadRequest::new(
        tmp.path(),
        fixture_priced_date(0),
        1,
        &current,
        1.0,
    ))
    .expect("a two-family superset source must load, never reject, by default");
    assert_eq!(cuts.len(), 1);
    assert_eq!(
        cuts[0].coefficients,
        vec![10.0],
        "the identity-matched storage coefficient still lands"
    );

    let report = cuts.report();
    assert_eq!(report.storage.dropped_source, 1);
    assert_eq!(report.anticipated.dropped_source, 1);

    let err = load_boundary_cuts(
        &BoundaryLoadRequest::new(tmp.path(), fixture_priced_date(0), 1, &current, 1.0)
            .with_strict(true),
    )
    .expect_err("a two-family superset source must reject under strict")
    .to_string();
    assert!(
        err.contains(
            "is a superset: 2 source slot(s) price entities this study does not model \
             (storage: 1, anticipated: 1)"
        ),
        "must name both dropping families in declaration order: {err}"
    );
}
