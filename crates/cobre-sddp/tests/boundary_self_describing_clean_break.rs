//! Behavioral regressions for the boundary load's clean-break rewire: pool
//! resolution and the source cost scale now read from a resolved
//! `cuts/<pool>.bin`'s own self-describing facts (`cost_scale_factor`,
//! `graph_stage_id`), never `metadata.producer`/`metadata.graph_manifest`; a
//! pre-change `.bin` (missing those facts) rejects instead of silently
//! defaulting.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::Path;
use std::process::Command;

use chrono::NaiveDate;
use cobre_io::{
    ENTITY_SLOT_DATE_SENTINEL, FORMAT_VERSION, GraphManifest, PolicyCutRecord, ProducerBlock,
    StageCutsPayload, encode_slot_date, write_policy_checkpoint,
};
use cobre_sddp::test_support::ymd;
use cobre_sddp::{BoundaryLoadRequest, SddpError, load_boundary_cuts};
use serde_json::json;

fn fixture_priced_date(pool: u32) -> NaiveDate {
    cobre_sddp::test_support::fixture_priced_date(ymd(2030, 1, 1), pool)
}

/// A minimal producer block for artifact-writing test helpers. Its own
/// `cost_scale_factor` is deliberately `None` in every fixture below: the
/// boundary path no longer reads it, so leaving it absent proves the load
/// result cannot be coming from this field.
fn producer_block() -> ProducerBlock {
    ProducerBlock {
        completed_iterations: 1,
        max_iterations: 1,
        forward_passes: 1,
        ..cobre_sddp::test_support::producer_block()
    }
}

/// The boundary load reads its source cost scale from the resolved pool's own
/// `cuts/<pool>.bin` (`cost_scale_factor`), never from `metadata.producer`: a
/// checkpoint with `metadata.producer.cost_scale_factor: None` (the legacy,
/// now-unread field) but a marked `.bin` (`Some(_)`) still takes the MARKED
/// rescale branch (plain division by the loading factor) rather than the
/// LEGACY branch (a `LEGACY_COST_SCALE_FACTOR / loading_factor` ratio
/// multiply) — the two disagree numerically at a non-default loading factor,
/// so this discriminates a regression that resurrects the
/// `metadata.producer` read.
#[test]
fn boundary_load_reads_cost_scale_from_bin() {
    let tmp = tempfile::tempdir().unwrap();
    let at_rest_intercept = 1_234_000.0;
    let at_rest_coefficients = [10_000.0_f64, -25_000.0];
    let cut = PolicyCutRecord {
        cut_id: 0,
        slot_index: 0,
        iteration: 0,
        forward_pass_index: 0,
        intercept: at_rest_intercept,
        coefficients: &at_rest_coefficients,
        is_active: true,
    };
    let payload = StageCutsPayload {
        stage_id: 0,
        state_dimension: 2,
        capacity: 1,
        warm_start_count: 0,
        cuts: &[cut],
        active_cut_indices: &[0],
        populated_count: 1,
        entity_manifest: &[],
        cost_scale_factor: 500_000.0,
        node_id: 0,
        graph_stage_id: 0,
        priced_state_date: encode_slot_date(fixture_priced_date(0)),
    };
    let metadata = cobre_sddp::test_support::checkpoint_metadata(
        1,
        GraphManifest::default(),
        producer_block(),
    );
    write_policy_checkpoint(tmp.path(), &[payload], &[], &metadata, &[]).unwrap();

    let loading_factor = 2_500_000.0;
    let cuts = load_boundary_cuts(&BoundaryLoadRequest::new(
        tmp.path(),
        fixture_priced_date(0),
        2,
        &[],
        loading_factor,
    ))
    .expect(
        "a self-describing checkpoint with metadata.producer.cost_scale_factor: None must \
         still load via the .bin's own marked cost_scale_factor",
    );

    let expected_intercept = at_rest_intercept / loading_factor;
    assert!(
        (cuts[0].intercept - expected_intercept).abs() < expected_intercept.abs() * 1e-9,
        "intercept {} must equal at_rest / loading_factor = {expected_intercept} (the MARKED \
         rescale branch); a LEGACY-branch result would prove the load still reads \
         metadata.producer.cost_scale_factor",
        cuts[0].intercept
    );
    for (c, &at_rest) in cuts[0].coefficients.iter().zip(&at_rest_coefficients) {
        let expected = at_rest / loading_factor;
        assert!(
            (c - expected).abs() < expected.abs() * 1e-9,
            "coefficient {c} != expected {expected}"
        );
    }
}

/// The schema shape written before the self-describing per-pool facts existed:
/// identical to the current `StageCuts`/`AffinePiece`/`EntitySlot` tables up to
/// `entity_manifest (id: 7)`, stopping there — matches the forward-compat
/// fixture in
/// `crates/cobre-io/tests/flatbuffers_schema_conformance.rs`
/// (`pre_self_describing_stage_cuts_reads_as_absent_and_sentinels`).
const PRE_SELF_DESCRIBING_SCHEMA: &str = r#"
namespace Cobre.IO.Policy;

file_identifier "CBVF";

enum EntityType : byte {
  HydroStorage = 0,
  HydroInflowLag = 1,
  AnticipatedThermalState = 2,
  HydroTransitBucket = 3,
}

table EntitySlot {
  entity_type:EntityType (id: 0);
  entity_id:int32 (id: 1);
  subindex:uint32 (id: 2);
  was_active:bool (id: 3);
  delivery_anchor:int32 (id: 4, deprecated);
  delivery_date:int32 (id: 5);
}

table AffinePiece {
  piece_id:uint64 (id: 0);
  slot_index:uint32 (id: 1);
  iteration:uint32 (id: 2);
  forward_pass_index:uint32 (id: 3);
  intercept:float64 (id: 4);
  coefficients:[float64] (id: 5);
  is_active:bool (id: 6);
  reserved_7:[float64] (id: 7, deprecated);
}

table StageCuts {
  stage_id:uint32 (id: 0);
  state_dimension:uint32 (id: 1);
  capacity:uint32 (id: 2);
  warm_start_count:uint32 (id: 3);
  cuts:[AffinePiece] (id: 4);
  active_cut_indices:[uint32] (id: 5);
  populated_count:uint32 (id: 6);
  entity_manifest:[EntitySlot] (id: 7);
}
"#;

/// Build a `StageCuts` buffer shaped by [`PRE_SELF_DESCRIBING_SCHEMA`] — no
/// `cost_scale_factor`/`node_id`/`graph_stage_id` fields at all, so the
/// current reader decodes `cost_scale_factor: None` and the sentinel ids —
/// via `flatc -b`, mirroring the fixture technique in
/// `flatbuffers_schema_conformance.rs`.
///
/// # Panics
/// When `flatc` is not found on `$PATH` (or via `FLATC`), or the encode fails.
fn build_pre_self_describing_stage_cuts_bin(
    stage_id: u32,
    intercept: f64,
    coefficient: f64,
) -> Vec<u8> {
    let dir = tempfile::tempdir().unwrap();
    let schema_path = dir.path().join("pre_self_describing.fbs");
    std::fs::write(&schema_path, PRE_SELF_DESCRIBING_SCHEMA).unwrap();

    let document = json!({
        "stage_id": stage_id,
        "state_dimension": 1,
        "capacity": 1,
        "warm_start_count": 0,
        "populated_count": 1,
        "active_cut_indices": [0],
        "cuts": [
            {
                "piece_id": 1,
                "slot_index": 0,
                "iteration": 0,
                "forward_pass_index": 0,
                "intercept": intercept,
                "coefficients": [coefficient],
                "is_active": true
            }
        ],
        "entity_manifest": [
            {
                "entity_type": "HydroStorage",
                "entity_id": 1,
                "subindex": 0,
                "was_active": true,
                "delivery_date": ENTITY_SLOT_DATE_SENTINEL
            }
        ]
    });
    let json_path = dir.path().join("doc.json");
    std::fs::write(&json_path, serde_json::to_vec(&document).unwrap()).unwrap();

    let flatc = std::env::var_os("FLATC").unwrap_or_else(|| "flatc".into());
    let status = Command::new(&flatc)
        .arg("-b")
        .arg("--root-type")
        .arg("Cobre.IO.Policy.StageCuts")
        .arg("-o")
        .arg(dir.path())
        .arg(&schema_path)
        .arg(&json_path)
        .status()
        .unwrap_or_else(|e| {
            panic!(
                "building a pre-self-describing StageCuts fixture requires `flatc` on PATH (or \
                 via FLATC); tried `{}`: {e}",
                Path::new(&flatc).display()
            )
        });
    assert!(
        status.success(),
        "flatc -b on the pre-self-describing schema failed"
    );
    std::fs::read(dir.path().join("doc.bin")).unwrap()
}

/// A boundary checkpoint whose `cuts/<pool>.bin` predates the self-describing
/// per-pool facts (`cost_scale_factor`/`node_id`/`graph_stage_id`) rejects with
/// a message naming the checkpoint path and ending with the shared remedy —
/// never a silent default, and never a fall-through to the generic not-found/state_dimension
/// messages.
#[test]
fn boundary_load_rejects_pre_self_describing_checkpoint() {
    let tmp = tempfile::tempdir().unwrap();

    // A normal, self-describing checkpoint first, so `metadata.json` and the
    // `basis/` directory exist — then overwrite `cuts/000.bin` with the
    // pre-self-describing buffer, reproducing an artifact this Cobre never
    // writes but must still detect on load.
    let placeholder_coeff = [1.0_f64];
    let cut = PolicyCutRecord {
        cut_id: 0,
        slot_index: 0,
        iteration: 0,
        forward_pass_index: 0,
        intercept: 1.0,
        coefficients: &placeholder_coeff,
        is_active: true,
    };
    let payload = StageCutsPayload {
        stage_id: 0,
        state_dimension: 1,
        capacity: 1,
        warm_start_count: 0,
        cuts: &[cut],
        active_cut_indices: &[0],
        populated_count: 1,
        entity_manifest: &[],
        cost_scale_factor: 1_000_000.0,
        node_id: 0,
        graph_stage_id: 0,
        priced_state_date: encode_slot_date(fixture_priced_date(0)),
    };
    let metadata = cobre_sddp::test_support::checkpoint_metadata(
        1,
        GraphManifest::default(),
        producer_block(),
    );
    write_policy_checkpoint(tmp.path(), &[payload], &[], &metadata, &[]).unwrap();

    let pre_change_buf = build_pre_self_describing_stage_cuts_bin(0, 7.0, 3.5);
    std::fs::write(tmp.path().join("cuts/000.bin"), &pre_change_buf).unwrap();

    let result = load_boundary_cuts(&BoundaryLoadRequest::new(
        tmp.path(),
        fixture_priced_date(0),
        1,
        &[],
        1_000_000.0,
    ));

    let err = result.expect_err("a pre-self-describing .bin must reject, never load silently");
    assert!(
        matches!(err, SddpError::Validation(_)),
        "must reject as SddpError::Validation: {err:?}"
    );
    let msg = err.to_string();
    assert!(
        msg.ends_with(&cobre_io::policy_checkpoint_remedy()),
        "message must end with the shared remedy: {msg}"
    );
    assert!(
        msg.contains(&tmp.path().display().to_string()),
        "message must name the checkpoint path: {msg}"
    );
}

/// A checkpoint whose `CheckpointManifest::format_version` predates
/// `FORMAT_VERSION` rejects as a boundary-path error naming the checkpoint
/// directory and the version field — never a fall-through to a generic
/// not-found or dimension message.
#[test]
fn boundary_load_rejects_pre_format_version_checkpoint() {
    let tmp = tempfile::tempdir().unwrap();

    let placeholder_coeff = [1.0_f64];
    let cut = PolicyCutRecord {
        cut_id: 0,
        slot_index: 0,
        iteration: 0,
        forward_pass_index: 0,
        intercept: 1.0,
        coefficients: &placeholder_coeff,
        is_active: true,
    };
    let payload = StageCutsPayload {
        stage_id: 0,
        state_dimension: 1,
        capacity: 1,
        warm_start_count: 0,
        cuts: &[cut],
        active_cut_indices: &[0],
        populated_count: 1,
        entity_manifest: &[],
        cost_scale_factor: 1_000_000.0,
        node_id: 0,
        graph_stage_id: 0,
        priced_state_date: encode_slot_date(fixture_priced_date(0)),
    };
    let mut metadata = cobre_sddp::test_support::checkpoint_metadata(
        1,
        GraphManifest::default(),
        producer_block(),
    );
    metadata.format_version = FORMAT_VERSION - 1;
    write_policy_checkpoint(tmp.path(), &[payload], &[], &metadata, &[]).unwrap();

    let result = load_boundary_cuts(&BoundaryLoadRequest::new(
        tmp.path(),
        fixture_priced_date(0),
        1,
        &[],
        1_000_000.0,
    ));

    let err = result.expect_err("a pre-format-version checkpoint must reject, never load silently");
    assert!(
        matches!(err, SddpError::Validation(_)),
        "must reject as SddpError::Validation: {err:?}"
    );
    let msg = err.to_string();
    assert!(
        msg.contains(&tmp.path().display().to_string()),
        "message must name the checkpoint directory: {msg}"
    );
    assert!(
        msg.contains("format_version"),
        "message must name the version field: {msg}"
    );
    assert!(
        msg.ends_with(&cobre_io::policy_checkpoint_remedy()),
        "message must end with the shared remedy: {msg}"
    );
}
