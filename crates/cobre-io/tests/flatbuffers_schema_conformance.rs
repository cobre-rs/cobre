//! Schema-conformance tests for `schemas/policy.fbs`, gated behind the
//! `flatc-conformance` feature (requires `flatc` on `$PATH` or via `FLATC`).
//!
//! The hand-rolled writer/reader encode field positions via the `*_FIELD_*: u16`
//! slot constants; the schema encodes them via `(id: N)` attributes. These two
//! views must agree exactly, or every schema consumer is corrupted. The tests
//! check both directions — writer→`flatc -t` (catches wrong/unknown slot writes)
//! and `flatc -b`→reader (catches a slot the reader expects at another offset);
//! changing one side without the other fails at least one check.
//!
//! Missing `flatc` panics rather than silently passing: invoking the feature is
//! an explicit request to run these checks.
//!
//! ```bash
//! cargo test -p cobre-io --features flatc-conformance --test flatbuffers_schema_conformance
//! ```

#![cfg(feature = "flatc-conformance")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreadable_literal
)]

use std::path::{Path, PathBuf};
use std::process::Command;

use cobre_io::{
    CheckpointManifest, ENTITY_SLOT_DATE_SENTINEL, EntitySlot, FORMAT_VERSION, GraphManifest,
    HydroSeasonOrders, ManifestEdge, ManifestNode, OwnedPolicyBasisRecord, OwnedPolicyCutRecord,
    PolicyBasisRecord, PolicyCutRecord, ProducerBlock, SEASON_CYCLE_CODE_MONTHLY,
    SEASON_CYCLE_CODE_WEEKLY, STAGE_CUTS_GRAPH_STAGE_ID_SENTINEL, STAGE_CUTS_NODE_ID_SENTINEL,
    STAGE_CUTS_PRICED_STATE_DATE_SENTINEL, STAGE_STATES_NODE_ID_SENTINEL, SeasonManifest,
    StageCutsPayload, StageCutsReadResult, StageStatesPayload, StageStatesReadResult,
    deserialize_checkpoint_manifest, deserialize_stage_basis, deserialize_stage_cuts,
    deserialize_stage_states, serialize_checkpoint_manifest, serialize_stage_basis,
    serialize_stage_cuts, serialize_stage_states,
};
use serde_json::{Value, json};
use tempfile::TempDir;

/// # Panics
/// When `flatc` is not found on `$PATH` or via the `FLATC` env var.
fn flatc_command() -> Command {
    let exe = std::env::var_os("FLATC").unwrap_or_else(|| "flatc".into());
    let mut probe = Command::new(&exe);
    probe.arg("--version");
    let probe_output = probe.output().unwrap_or_else(|err| {
        panic!(
            "the `flatc-conformance` feature requires `flatc` on PATH (or via the FLATC env var); \
             tried `{}`: {err}",
            Path::new(&exe).display()
        )
    });
    assert!(
        probe_output.status.success(),
        "`flatc --version` failed; stderr = {}",
        String::from_utf8_lossy(&probe_output.stderr)
    );
    Command::new(exe)
}

fn schema_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("schemas/policy.fbs")
}

/// `--root-type` must be namespace-qualified: flatc rejects the unqualified
/// short name (e.g. `StageCuts`) with `unknown root type`.
fn qualified(root_type: &str) -> String {
    format!("Cobre.IO.Policy.{root_type}")
}

fn flatc_decode(buf: &[u8], root_type: &str) -> Value {
    let dir = TempDir::new().expect("create tempdir");
    let bin_path = dir.path().join("buf.bin");
    std::fs::write(&bin_path, buf).expect("write bin");

    let status = flatc_command()
        .arg("-t")
        .arg("--strict-json")
        .arg("--raw-binary")
        .arg("--root-type")
        .arg(qualified(root_type))
        .arg("-o")
        .arg(dir.path())
        .arg(schema_path())
        .arg("--")
        .arg(&bin_path)
        .status()
        .expect("run flatc -t");
    assert!(
        status.success(),
        "flatc -t failed for root {root_type}; buffer contains a slot or layout the schema does \
         not match"
    );

    let json_path = dir.path().join("buf.json");
    let json_bytes = std::fs::read(&json_path).expect("read flatc JSON");
    serde_json::from_slice(&json_bytes).expect("parse flatc JSON")
}

fn flatc_encode(json: &Value, root_type: &str) -> Vec<u8> {
    let dir = TempDir::new().expect("create tempdir");
    let json_path = dir.path().join("doc.json");
    std::fs::write(&json_path, serde_json::to_vec(json).unwrap()).expect("write JSON");

    let status = flatc_command()
        .arg("-b")
        .arg("--root-type")
        .arg(qualified(root_type))
        .arg("-o")
        .arg(dir.path())
        .arg(schema_path())
        .arg(&json_path)
        .status()
        .expect("run flatc -b");
    assert!(
        status.success(),
        "flatc -b failed for root {root_type}; the JSON does not satisfy the schema"
    );

    let bin_path = dir.path().join("doc.bin");
    std::fs::read(&bin_path).expect("read flatc-produced bin")
}

fn get<'a>(v: &'a Value, key: &str) -> &'a Value {
    v.get(key)
        .unwrap_or_else(|| panic!("flatc JSON missing field `{key}`; got: {v}"))
}

fn as_u64(v: &Value, key: &str) -> u64 {
    get(v, key)
        .as_u64()
        .unwrap_or_else(|| panic!("field `{key}` is not a u64: {v}"))
}

fn as_f64(v: &Value, key: &str) -> f64 {
    get(v, key)
        .as_f64()
        .unwrap_or_else(|| panic!("field `{key}` is not a f64: {v}"))
}

fn as_bool(v: &Value, key: &str) -> bool {
    get(v, key)
        .as_bool()
        .unwrap_or_else(|| panic!("field `{key}` is not a bool: {v}"))
}

/// flatc `-t` omits scalar fields equal to their default, so manifest assertions
/// must treat an absent field as its default rather than panicking.
fn u64_or(v: &Value, key: &str, default: u64) -> u64 {
    v.get(key).map_or(default, |f| {
        f.as_u64()
            .unwrap_or_else(|| panic!("field `{key}` is not a u64: {v}"))
    })
}

fn i64_or(v: &Value, key: &str, default: i64) -> i64 {
    v.get(key).map_or(default, |f| {
        f.as_i64()
            .unwrap_or_else(|| panic!("field `{key}` is not an i64: {v}"))
    })
}

fn bool_or(v: &Value, key: &str, default: bool) -> bool {
    v.get(key).map_or(default, |f| {
        f.as_bool()
            .unwrap_or_else(|| panic!("field `{key}` is not a bool: {v}"))
    })
}

/// flatc renders the `EntityType` enum by name; map the raw byte to the name the
/// schema declares so writer→flatc assertions can compare against it. An absent
/// `entity_type` (the `0`/`HydroStorage` default) is treated as `HydroStorage`.
fn entity_type_name(byte: u8) -> &'static str {
    match byte {
        0 => "HydroStorage",
        1 => "HydroInflowLag",
        2 => "AnticipatedThermalState",
        3 => "HydroTransitBucket",
        other => panic!("unexpected entity_type byte {other}"),
    }
}

fn assert_manifest_json_matches(manifest_json: &Value, expected: &[EntitySlot]) {
    let arr = manifest_json
        .as_array()
        .expect("entity_manifest must be an array");
    assert_eq!(arr.len(), expected.len(), "entity_manifest length");
    for (i, (obj, slot)) in arr.iter().zip(expected).enumerate() {
        let ty = obj
            .get("entity_type")
            .and_then(Value::as_str)
            .unwrap_or("HydroStorage");
        assert_eq!(
            ty,
            entity_type_name(slot.entity_type),
            "slot {i} entity_type"
        );
        assert_eq!(
            i64_or(obj, "entity_id", 0),
            i64::from(slot.entity_id),
            "slot {i} entity_id"
        );
        assert_eq!(
            u64_or(obj, "subindex", 0),
            u64::from(slot.subindex),
            "slot {i} subindex"
        );
        assert_eq!(
            bool_or(obj, "was_active", false),
            slot.was_active,
            "slot {i} was_active"
        );
        assert_eq!(
            i64_or(obj, "reference_date", 0),
            i64::from(slot.reference_date),
            "slot {i} reference_date"
        );
        assert_eq!(
            i64_or(obj, "interval_start", 0),
            i64::from(slot.interval_start),
            "slot {i} interval_start"
        );
        assert_eq!(
            i64_or(obj, "interval_end", 0),
            i64::from(slot.interval_end),
            "slot {i} interval_end"
        );
    }
}

/// Manifest fixture exercising every `entity_type` byte and a `-1` id, so the
/// writer→flatc and flatc→reader paths cover every field including a signed id.
fn conformance_manifest() -> Vec<EntitySlot> {
    vec![
        EntitySlot::anticipated(7, 1, true).with_interval(20_240_501, 20_240_601),
        EntitySlot::inflow_lag(-1, 3, false),
        EntitySlot::transit_bucket(42, 2, true).with_interval(20_200_101, 20_200_201),
    ]
}

// ─── StageCuts ───────────────────────────────────────────────────────────────

/// Serialize a `StageCuts` buffer from positional fields, defaulting the
/// self-describing per-pool facts (their own conformance test exercises them).
fn ser_cuts(
    stage_id: u32,
    state_dimension: u32,
    capacity: u32,
    warm_start_count: u32,
    cuts: &[PolicyCutRecord<'_>],
    active_cut_indices: &[u32],
    populated_count: u32,
    entity_manifest: &[EntitySlot],
) -> Vec<u8> {
    serialize_stage_cuts(&StageCutsPayload {
        stage_id,
        state_dimension,
        capacity,
        warm_start_count,
        cuts,
        active_cut_indices,
        populated_count,
        entity_manifest,
        cost_scale_factor: 1_000_000.0,
        node_id: -1,
        graph_stage_id: -1,
        priced_state_date: STAGE_CUTS_PRICED_STATE_DATE_SENTINEL,
    })
}

#[test]
fn stage_cuts_writer_matches_schema() {
    let coeffs_a = [1.0, 2.0, 3.0, 4.0];
    let coeffs_b = [-0.5, 0.25, 0.125, 0.0625];
    let cuts = [
        PolicyCutRecord {
            cut_id: 7,
            slot_index: 0,
            iteration: 1,
            forward_pass_index: 0,
            intercept: 100.5,
            coefficients: &coeffs_a,
            is_active: true,
        },
        PolicyCutRecord {
            cut_id: 8,
            slot_index: 1,
            iteration: 1,
            forward_pass_index: 1,
            intercept: -42.0,
            coefficients: &coeffs_b,
            is_active: false,
        },
    ];
    let active = [0_u32];
    let manifest = conformance_manifest();
    let buf = ser_cuts(3, 4, 16, 0, &cuts, &active, 2, &manifest);

    let json = flatc_decode(&buf, "StageCuts");

    assert_eq!(as_u64(&json, "stage_id"), 3);
    assert_eq!(as_u64(&json, "state_dimension"), 4);
    assert_eq!(as_u64(&json, "capacity"), 16);
    assert_eq!(as_u64(&json, "warm_start_count"), 0);
    assert_eq!(as_u64(&json, "populated_count"), 2);
    assert_eq!(get(&json, "active_cut_indices"), &json!([0]));
    assert_manifest_json_matches(get(&json, "entity_manifest"), &manifest);

    let cuts_json = get(&json, "cuts").as_array().expect("cuts is an array");
    assert_eq!(cuts_json.len(), 2);

    let c0 = &cuts_json[0];
    assert_eq!(as_u64(c0, "piece_id"), 7);
    assert_eq!(as_u64(c0, "slot_index"), 0);
    assert_eq!(as_u64(c0, "iteration"), 1);
    assert_eq!(as_u64(c0, "forward_pass_index"), 0);
    assert!((as_f64(c0, "intercept") - 100.5).abs() < 1e-12);
    assert_eq!(get(c0, "coefficients"), &json!([1.0, 2.0, 3.0, 4.0]));
    assert!(as_bool(c0, "is_active"));

    let c1 = &cuts_json[1];
    assert_eq!(as_u64(c1, "piece_id"), 8);
    assert_eq!(as_u64(c1, "slot_index"), 1);
    assert!((as_f64(c1, "intercept") - (-42.0)).abs() < 1e-12);
    assert!(!as_bool(c1, "is_active"));
}

#[test]
fn stage_cuts_reader_consumes_flatc_buffer() {
    let document = json!({
        "stage_id": 5,
        "state_dimension": 3,
        "capacity": 8,
        "warm_start_count": 2,
        "populated_count": 1,
        "active_cut_indices": [0],
        "cuts": [
            {
                "piece_id": 99,
                "slot_index": 0,
                "iteration": 4,
                "forward_pass_index": 2,
                "intercept": 12.5,
                "coefficients": [0.1, 0.2, 0.3],
                "is_active": true
            }
        ],
        "entity_manifest": [
            {"entity_type": "AnticipatedThermalState", "entity_id": 7, "subindex": 1, "was_active": true},
            {"entity_type": "HydroInflowLag", "entity_id": -1, "subindex": 3, "was_active": false},
            {"entity_type": "HydroTransitBucket", "entity_id": 42, "subindex": 2, "was_active": true}
        ]
    });
    let buf = flatc_encode(&document, "StageCuts");
    let result: StageCutsReadResult =
        deserialize_stage_cuts(&buf).expect("hand-rolled reader must consume flatc-built buffer");

    assert_eq!(result.stage_id, 5);
    assert_eq!(result.state_dimension, 3);
    assert_eq!(result.capacity, 8);
    assert_eq!(result.warm_start_count, 2);
    assert_eq!(result.populated_count, 1);
    assert_eq!(result.cuts.len(), 1);

    let cut: &OwnedPolicyCutRecord = &result.cuts[0];
    assert_eq!(cut.cut_id, 99);
    assert_eq!(cut.slot_index, 0);
    assert_eq!(cut.iteration, 4);
    assert_eq!(cut.forward_pass_index, 2);
    assert!((cut.intercept - 12.5).abs() < 1e-12);
    assert_eq!(cut.coefficients, vec![0.1, 0.2, 0.3]);
    assert!(cut.is_active);

    assert_eq!(result.entity_manifest.len(), 3);
    let m0 = &result.entity_manifest[0];
    assert_eq!(m0.entity_type, 2, "AnticipatedThermalState byte");
    assert_eq!(m0.entity_id, 7);
    assert_eq!(m0.subindex, 1);
    assert!(m0.was_active);
    let m1 = &result.entity_manifest[1];
    assert_eq!(m1.entity_type, 1, "HydroInflowLag byte");
    assert_eq!(m1.entity_id, -1);
    assert_eq!(m1.subindex, 3);
    assert!(!m1.was_active);
    let m2 = &result.entity_manifest[2];
    assert_eq!(m2.entity_type, 3, "HydroTransitBucket byte");
    assert_eq!(m2.entity_id, 42);
    assert_eq!(m2.subindex, 2);
    assert!(m2.was_active);
}

/// Reduced-dimension manifest: storage-only (two `HydroStorage` slots, zero lag
/// slots), `state_dimension = 2` where the all-enabled analogue would be larger.
/// This is the reduced-per-stage case at the conformance level — a manifest
/// shorter than the all-enabled count must round-trip in both directions.
fn reduced_storage_manifest() -> Vec<EntitySlot> {
    vec![EntitySlot::storage(1, true), EntitySlot::storage(2, true)]
}

#[test]
fn stage_cuts_reduced_dimension_writer_matches_schema() {
    let coeffs = [3.5, -1.25];
    let cuts = [PolicyCutRecord {
        cut_id: 42,
        slot_index: 0,
        iteration: 2,
        forward_pass_index: 0,
        intercept: 7.0,
        coefficients: &coeffs,
        is_active: true,
    }];
    let active = [0_u32];
    let manifest = reduced_storage_manifest();
    let buf = ser_cuts(1, 2, 8, 0, &cuts, &active, 1, &manifest);

    let json = flatc_decode(&buf, "StageCuts");

    assert_eq!(as_u64(&json, "state_dimension"), 2);
    assert_eq!(as_u64(&json, "populated_count"), 1);

    let manifest_json = get(&json, "entity_manifest");
    assert_eq!(
        manifest_json
            .as_array()
            .expect("entity_manifest must be an array")
            .len(),
        2,
        "reduced manifest must have 2 storage slots"
    );
    assert_manifest_json_matches(manifest_json, &manifest);

    let decoded_cut = &get(&json, "cuts").as_array().expect("cuts array")[0];
    assert_eq!(get(decoded_cut, "coefficients"), &json!([3.5, -1.25]));
}

#[test]
fn stage_cuts_reduced_dimension_reader_consumes_flatc_buffer() {
    let document = json!({
        "stage_id": 1,
        "state_dimension": 2,
        "capacity": 8,
        "warm_start_count": 0,
        "populated_count": 1,
        "active_cut_indices": [0],
        "cuts": [
            {
                "piece_id": 42,
                "slot_index": 0,
                "iteration": 2,
                "forward_pass_index": 0,
                "intercept": 7.0,
                "coefficients": [3.5, -1.25],
                "is_active": true
            }
        ],
        "entity_manifest": [
            {"entity_type": "HydroStorage", "entity_id": 1, "subindex": 0, "was_active": true},
            {"entity_type": "HydroStorage", "entity_id": 2, "subindex": 0, "was_active": true}
        ]
    });
    let buf = flatc_encode(&document, "StageCuts");
    let result: StageCutsReadResult =
        deserialize_stage_cuts(&buf).expect("hand-rolled reader must consume flatc-built buffer");

    assert_eq!(result.state_dimension, 2);
    assert_eq!(result.entity_manifest.len(), 2);
    for (i, slot) in result.entity_manifest.iter().enumerate() {
        assert_eq!(slot.entity_type, 0, "slot {i} HydroStorage byte");
        assert_eq!(slot.subindex, 0, "slot {i} subindex");
        assert!(slot.was_active, "slot {i} was_active");
    }
    assert_eq!(result.entity_manifest[0].entity_id, 1);
    assert_eq!(result.entity_manifest[1].entity_id, 2);
}

/// Round-trip of the self-describing per-pool facts (`cost_scale_factor` id 8,
/// `node_id` id 9, `graph_stage_id` id 10, `priced_state_date` id 11) in both
/// directions: the hand-rolled writer→reader path preserves them, `flatc`
/// decodes the hand-rolled buffer with the four at their slots, and a
/// `flatc`-built buffer carrying them decodes identically through the
/// hand-rolled reader.
#[test]
fn stage_cuts_self_describing_facts_round_trip() {
    let coeffs = [1.0, 2.0, 3.0];
    let cuts = [PolicyCutRecord {
        cut_id: 5,
        slot_index: 0,
        iteration: 2,
        forward_pass_index: 0,
        intercept: 9.0,
        coefficients: &coeffs,
        is_active: true,
    }];
    let payload = StageCutsPayload {
        stage_id: 4,
        state_dimension: 3,
        capacity: 8,
        warm_start_count: 0,
        cuts: &cuts,
        active_cut_indices: &[0],
        populated_count: 1,
        entity_manifest: &[],
        cost_scale_factor: 1_000_000.0,
        node_id: 3,
        graph_stage_id: 7,
        priced_state_date: 20_311_201,
    };
    let buf = serialize_stage_cuts(&payload);

    let hand_rolled =
        deserialize_stage_cuts(&buf).expect("hand-rolled reader must consume its own buffer");
    assert_eq!(hand_rolled.cost_scale_factor, Some(1_000_000.0));
    assert_eq!(hand_rolled.node_id, 3);
    assert_eq!(hand_rolled.graph_stage_id, 7);
    assert_eq!(hand_rolled.priced_state_date, 20_311_201);

    let json = flatc_decode(&buf, "StageCuts");
    assert!((as_f64(&json, "cost_scale_factor") - 1_000_000.0).abs() < 1e-6);
    assert_eq!(i64_or(&json, "node_id", -1), 3);
    assert_eq!(i64_or(&json, "graph_stage_id", -1), 7);
    assert_eq!(i64_or(&json, "priced_state_date", -1), 20_311_201);

    let document = json!({
        "stage_id": 4,
        "state_dimension": 3,
        "capacity": 8,
        "warm_start_count": 0,
        "populated_count": 1,
        "active_cut_indices": [0],
        "cuts": [
            {
                "piece_id": 5,
                "slot_index": 0,
                "iteration": 2,
                "forward_pass_index": 0,
                "intercept": 9.0,
                "coefficients": [1.0, 2.0, 3.0],
                "is_active": true
            }
        ],
        "cost_scale_factor": 1_000_000.0,
        "node_id": 3,
        "graph_stage_id": 7,
        "priced_state_date": 20_311_201
    });
    let flatc_buf = flatc_encode(&document, "StageCuts");
    let from_flatc =
        deserialize_stage_cuts(&flatc_buf).expect("hand-rolled reader must consume flatc buffer");
    assert_eq!(from_flatc.cost_scale_factor, Some(1_000_000.0));
    assert_eq!(from_flatc.node_id, 3);
    assert_eq!(from_flatc.graph_stage_id, 7);
    assert_eq!(from_flatc.priced_state_date, 20_311_201);
}

/// Forward-compat: a buffer built from a pre-`id:8` `StageCuts` schema (no
/// self-describing facts) reads back `cost_scale_factor == None`, `node_id`,
/// `graph_stage_id` and `priced_state_date` at their sentinels — never a bare
/// `0`. Mirrors `pre_node_id_stage_states_reads_as_sentinel`: flatc cannot
/// emit a field the schema lacks, so the buffer is built from a schema that
/// stops at `entity_manifest (id: 7)`.
#[test]
fn pre_self_describing_stage_cuts_reads_as_absent_and_sentinels() {
    let schema_pre_self_describing = "
namespace Cobre.IO.Policy;

file_identifier \"CBVF\";

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
";
    let dir = TempDir::new().unwrap();
    let pre_schema = dir.path().join("pre_self_describing.fbs");
    std::fs::write(&pre_schema, schema_pre_self_describing).unwrap();

    let document = json!({
        "stage_id": 1,
        "state_dimension": 2,
        "capacity": 4,
        "warm_start_count": 0,
        "populated_count": 1,
        "active_cut_indices": [0],
        "cuts": [
            {
                "piece_id": 1,
                "slot_index": 0,
                "iteration": 1,
                "forward_pass_index": 0,
                "intercept": 1.0,
                "coefficients": [1.0, 2.0],
                "is_active": true
            }
        ],
        "entity_manifest": [
            {"entity_type": "HydroStorage", "entity_id": 1, "subindex": 0, "was_active": true}
        ]
    });
    let json_path = dir.path().join("doc.json");
    std::fs::write(&json_path, serde_json::to_vec(&document).unwrap()).unwrap();

    let status = flatc_command()
        .arg("-b")
        .arg("--root-type")
        .arg(qualified("StageCuts"))
        .arg("-o")
        .arg(dir.path())
        .arg(&pre_schema)
        .arg(&json_path)
        .status()
        .expect("run flatc -b on pre-self-describing schema");
    assert!(
        status.success(),
        "flatc -b on pre-self-describing schema failed"
    );
    let buf = std::fs::read(dir.path().join("doc.bin")).unwrap();

    let result = deserialize_stage_cuts(&buf)
        .expect("hand-rolled reader must accept pre-self-describing buffer");
    assert_eq!(
        result.cost_scale_factor, None,
        "a pre-id:8 buffer must read cost_scale_factor as None, not 0.0"
    );
    assert_eq!(
        result.node_id, STAGE_CUTS_NODE_ID_SENTINEL,
        "a pre-id:8 buffer must read node_id as the sentinel, not 0"
    );
    assert_eq!(
        result.graph_stage_id, STAGE_CUTS_GRAPH_STAGE_ID_SENTINEL,
        "a pre-id:8 buffer must read graph_stage_id as the sentinel, not 0"
    );
    assert_eq!(
        result.priced_state_date, STAGE_CUTS_PRICED_STATE_DATE_SENTINEL,
        "a pre-id:11 buffer must read priced_state_date as the sentinel, not 0"
    );
}

// ─── StageBasis ──────────────────────────────────────────────────────────────

#[test]
fn stage_basis_writer_matches_schema() {
    let cols = [0_u8, 1, 2, 3];
    let rows = [4_u8, 5, 6, 0, 1];
    let record = PolicyBasisRecord {
        stage_id: 11,
        iteration: 23,
        column_status: &cols,
        row_status: &rows,
        num_cut_rows: 2,
    };
    let buf = serialize_stage_basis(&record);

    let json = flatc_decode(&buf, "StageBasis");

    assert_eq!(as_u64(&json, "stage_id"), 11);
    assert_eq!(as_u64(&json, "iteration"), 23);
    assert_eq!(as_u64(&json, "num_columns"), 4);
    assert_eq!(as_u64(&json, "num_rows"), 5);
    assert_eq!(as_u64(&json, "num_cut_rows"), 2);
    assert_eq!(get(&json, "column_status"), &json!([0, 1, 2, 3]));
    assert_eq!(get(&json, "row_status"), &json!([4, 5, 6, 0, 1]));
}

#[test]
fn stage_basis_reader_consumes_flatc_buffer() {
    let document = json!({
        "stage_id": 7,
        "iteration": 3,
        "num_columns": 3,
        "num_rows": 4,
        "column_status": [1, 2, 3],
        "row_status": [0, 1, 1, 2],
        "num_cut_rows": 1
    });
    let buf = flatc_encode(&document, "StageBasis");
    let result: OwnedPolicyBasisRecord =
        deserialize_stage_basis(&buf).expect("hand-rolled reader must consume flatc-built buffer");

    assert_eq!(result.stage_id, 7);
    assert_eq!(result.iteration, 3);
    assert_eq!(result.column_status, vec![1, 2, 3]);
    assert_eq!(result.row_status, vec![0, 1, 1, 2]);
    assert_eq!(result.num_cut_rows, 1);
}

// ─── StageStates ─────────────────────────────────────────────────────────────

#[test]
fn stage_states_writer_matches_schema() {
    let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let manifest = conformance_manifest();
    let payload = StageStatesPayload {
        stage_id: 2,
        node_id: 5,
        state_dimension: 3,
        count: 2,
        data: &data,
        entity_manifest: &manifest,
    };
    let buf = serialize_stage_states(&payload);

    let json = flatc_decode(&buf, "StageStates");

    assert_eq!(as_u64(&json, "stage_id"), 2);
    assert_eq!(as_u64(&json, "state_dimension"), 3);
    assert_eq!(as_u64(&json, "count"), 2);
    assert_eq!(get(&json, "data"), &json!([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
    assert_manifest_json_matches(get(&json, "entity_manifest"), &manifest);
    assert_eq!(i64_or(&json, "node_id", -1), 5);
}

#[test]
fn stage_states_reader_consumes_flatc_buffer() {
    let document = json!({
        "stage_id": 9,
        "node_id": 12,
        "state_dimension": 2,
        "count": 3,
        "data": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
        "entity_manifest": [
            {"entity_type": "HydroStorage", "entity_id": 4, "subindex": 0, "was_active": true},
            {"entity_type": "HydroStorage", "entity_id": 5, "subindex": 0, "was_active": true}
        ]
    });
    let buf = flatc_encode(&document, "StageStates");
    let result: StageStatesReadResult =
        deserialize_stage_states(&buf).expect("hand-rolled reader must consume flatc-built buffer");

    assert_eq!(result.stage_id, 9);
    assert_eq!(result.node_id, 12);
    assert_eq!(result.state_dimension, 2);
    assert_eq!(result.count, 3);
    assert_eq!(result.data, vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0]);

    assert_eq!(result.entity_manifest.len(), 2);
    assert_eq!(result.entity_manifest[0].entity_type, 0);
    assert_eq!(result.entity_manifest[0].entity_id, 4);
    assert!(result.entity_manifest[0].was_active);
    assert_eq!(result.entity_manifest[1].entity_id, 5);
}

/// Forward-compat: a buffer written against a pre-`id:5` `StageStates` schema
/// (no `node_id` field) must deserialize with `node_id` at
/// [`STAGE_STATES_NODE_ID_SENTINEL`], never a bare `0` (a valid node id).
/// Mirrors `pre_interval_entity_slot_reads_as_sentinel`'s rewritten-schema
/// approach: flatc cannot emit a field the schema lacks, so the buffer is built
/// from a schema that stops at `entity_manifest (id: 4)`.
#[test]
fn pre_node_id_stage_states_reads_as_sentinel() {
    let schema_pre_node_id = "
namespace Cobre.IO.Policy;

file_identifier \"CBVF\";

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
}

table StageStates {
  stage_id:uint32 (id: 0);
  state_dimension:uint32 (id: 1);
  count:uint32 (id: 2);
  data:[float64] (id: 3);
  entity_manifest:[EntitySlot] (id: 4);
}
";
    let dir = TempDir::new().unwrap();
    let pre_node_id_schema = dir.path().join("pre_node_id.fbs");
    std::fs::write(&pre_node_id_schema, schema_pre_node_id).unwrap();

    let document = json!({
        "stage_id": 3,
        "state_dimension": 1,
        "count": 1,
        "data": [1.0],
        "entity_manifest": [
            {"entity_type": "HydroStorage", "entity_id": 1, "subindex": 0, "was_active": true}
        ]
    });
    let json_path = dir.path().join("doc.json");
    std::fs::write(&json_path, serde_json::to_vec(&document).unwrap()).unwrap();

    let status = flatc_command()
        .arg("-b")
        .arg("--root-type")
        .arg(qualified("StageStates"))
        .arg("-o")
        .arg(dir.path())
        .arg(&pre_node_id_schema)
        .arg(&json_path)
        .status()
        .expect("run flatc -b on pre-node_id schema");
    assert!(status.success(), "flatc -b on pre-node_id schema failed");
    let buf = std::fs::read(dir.path().join("doc.bin")).unwrap();

    let result =
        deserialize_stage_states(&buf).expect("hand-rolled reader must accept pre-node_id buffer");
    assert_eq!(result.stage_id, 3);
    assert_eq!(
        result.node_id, STAGE_STATES_NODE_ID_SENTINEL,
        "a pre-node_id buffer must read back node_id as the sentinel, not 0"
    );
}

// ─── CheckpointManifest ──────────────────────────────────────────────────────

/// A fully-populated manifest fixture: multi-node/multi-edge graph and a
/// non-default producer with both `Option<f64>` provenance fields set.
fn conformance_manifest_value() -> CheckpointManifest {
    CheckpointManifest {
        format_version: FORMAT_VERSION,
        software: Some("cobre".to_string()),
        software_version: "9.9.9".to_string(),
        created_at: "2026-08-23T12:00:00Z".to_string(),
        num_stages: 60,
        graph_manifest: GraphManifest {
            n_pools: 3,
            nodes: vec![
                ManifestNode {
                    id: 10,
                    stage_id: 1,
                    pool_id: 0,
                },
                ManifestNode {
                    id: 11,
                    stage_id: 2,
                    pool_id: 2,
                },
            ],
            edges: vec![ManifestEdge {
                source_id: 10,
                target_id: 11,
                probability: 0.75,
            }],
        },
        producer: ProducerBlock {
            completed_iterations: 50,
            final_lower_bound: 1234.5,
            best_upper_bound: Some(1300.0),
            max_iterations: 200,
            forward_passes: 4,
            warm_start_cuts: 7,
            warm_start_counts: vec![3, 4],
            rng_seed: 42,
            total_visited_states: 999,
            training_block_mode: "parallel".to_string(),
            training_block_mode_per_stage: vec![
                "parallel".to_string(),
                "chronological".to_string(),
            ],
            cost_scale_factor: Some(1_000_000.0),
            lower_bound_history: vec![1300.5, 1250.25, 1234.5],
        },
        season_manifest: SeasonManifest {
            cycle_code: SEASON_CYCLE_CODE_MONTHLY,
            n_seasons: 12,
            hydro_orders: vec![
                HydroSeasonOrders {
                    hydro_id: 3,
                    orders: vec![1, 2, 1, 1, 3, 2, 1, 1, 2, 2, 1, 1],
                },
                HydroSeasonOrders {
                    hydro_id: 9,
                    orders: vec![4, 4, 3, 3, 2, 2, 1, 1, 2, 2, 3, 3],
                },
            ],
        },
    }
}

fn assert_manifest_eq(actual: &CheckpointManifest, expected: &CheckpointManifest) {
    assert_eq!(actual.format_version, expected.format_version);
    assert_eq!(actual.software, expected.software);
    assert_eq!(actual.software_version, expected.software_version);
    assert_eq!(actual.created_at, expected.created_at);
    assert_eq!(actual.num_stages, expected.num_stages);

    assert_eq!(
        actual.graph_manifest.n_pools,
        expected.graph_manifest.n_pools
    );
    assert_eq!(
        actual.graph_manifest.nodes.len(),
        expected.graph_manifest.nodes.len()
    );
    for (a, e) in actual
        .graph_manifest
        .nodes
        .iter()
        .zip(&expected.graph_manifest.nodes)
    {
        assert_eq!(a.id, e.id);
        assert_eq!(a.stage_id, e.stage_id);
        assert_eq!(a.pool_id, e.pool_id);
    }
    assert_eq!(
        actual.graph_manifest.edges.len(),
        expected.graph_manifest.edges.len()
    );
    for (a, e) in actual
        .graph_manifest
        .edges
        .iter()
        .zip(&expected.graph_manifest.edges)
    {
        assert_eq!(a.source_id, e.source_id);
        assert_eq!(a.target_id, e.target_id);
        assert_eq!(a.probability.to_bits(), e.probability.to_bits());
    }

    let (ap, ep) = (&actual.producer, &expected.producer);
    assert_eq!(ap.completed_iterations, ep.completed_iterations);
    assert_eq!(
        ap.final_lower_bound.to_bits(),
        ep.final_lower_bound.to_bits()
    );
    assert_eq!(
        ap.best_upper_bound.map(f64::to_bits),
        ep.best_upper_bound.map(f64::to_bits)
    );
    assert_eq!(ap.max_iterations, ep.max_iterations);
    assert_eq!(ap.forward_passes, ep.forward_passes);
    assert_eq!(ap.warm_start_cuts, ep.warm_start_cuts);
    assert_eq!(ap.warm_start_counts, ep.warm_start_counts);
    assert_eq!(ap.rng_seed, ep.rng_seed);
    assert_eq!(ap.total_visited_states, ep.total_visited_states);
    assert_eq!(ap.training_block_mode, ep.training_block_mode);
    assert_eq!(
        ap.training_block_mode_per_stage,
        ep.training_block_mode_per_stage
    );
    assert_eq!(
        ap.cost_scale_factor.map(f64::to_bits),
        ep.cost_scale_factor.map(f64::to_bits)
    );
    assert_eq!(
        ap.lower_bound_history
            .iter()
            .copied()
            .map(f64::to_bits)
            .collect::<Vec<_>>(),
        ep.lower_bound_history
            .iter()
            .copied()
            .map(f64::to_bits)
            .collect::<Vec<_>>()
    );

    let (asm, esm) = (&actual.season_manifest, &expected.season_manifest);
    assert_eq!(asm.cycle_code, esm.cycle_code);
    assert_eq!(asm.n_seasons, esm.n_seasons);
    assert_eq!(asm.hydro_orders.len(), esm.hydro_orders.len());
    for (a, e) in asm.hydro_orders.iter().zip(&esm.hydro_orders) {
        assert_eq!(a.hydro_id, e.hydro_id);
        assert_eq!(a.orders, e.orders);
    }
}

/// Round-trip of the `CheckpointManifest` root in both directions: the
/// hand-rolled writer → reader path preserves every field, `flatc` decodes the
/// hand-rolled buffer with each field at its slot, and a `flatc`-built buffer
/// carrying the same document decodes identically through the hand-rolled reader.
#[test]
fn checkpoint_manifest_round_trip() {
    let manifest = conformance_manifest_value();
    let buf = serialize_checkpoint_manifest(&manifest);

    let hand_rolled = deserialize_checkpoint_manifest(&buf)
        .expect("hand-rolled reader must consume its own buffer");
    assert_manifest_eq(&hand_rolled, &manifest);

    let json = flatc_decode(&buf, "CheckpointManifest");
    assert_eq!(as_u64(&json, "format_version"), u64::from(FORMAT_VERSION));
    assert_eq!(get(&json, "software").as_str().unwrap(), "cobre");
    assert_eq!(get(&json, "software_version").as_str().unwrap(), "9.9.9");
    assert_eq!(
        get(&json, "created_at").as_str().unwrap(),
        "2026-08-23T12:00:00Z"
    );
    assert_eq!(as_u64(&json, "num_stages"), 60);
    assert_eq!(as_u64(&json, "n_pools"), 3);
    assert_eq!(as_u64(&json, "completed_iterations"), 50);
    assert!((as_f64(&json, "final_lower_bound") - 1234.5).abs() < 1e-9);
    assert!((as_f64(&json, "best_upper_bound") - 1300.0).abs() < 1e-9);
    assert_eq!(as_u64(&json, "max_iterations"), 200);
    assert_eq!(as_u64(&json, "forward_passes"), 4);
    assert_eq!(as_u64(&json, "warm_start_cuts"), 7);
    assert_eq!(get(&json, "warm_start_counts"), &json!([3, 4]));
    assert_eq!(as_u64(&json, "rng_seed"), 42);
    assert_eq!(as_u64(&json, "total_visited_states"), 999);
    assert_eq!(
        get(&json, "training_block_mode").as_str().unwrap(),
        "parallel"
    );
    assert_eq!(
        get(&json, "training_block_mode_per_stage"),
        &json!(["parallel", "chronological"])
    );
    assert!((as_f64(&json, "cost_scale_factor") - 1_000_000.0).abs() < 1e-6);
    let lower_bound_history: Vec<f64> = get(&json, "lower_bound_history")
        .as_array()
        .expect("lower_bound_history is an array")
        .iter()
        .map(|v| v.as_f64().expect("lower_bound_history entries are numbers"))
        .collect();
    assert_eq!(lower_bound_history, vec![1300.5, 1250.25, 1234.5]);

    let nodes = get(&json, "nodes").as_array().expect("nodes is an array");
    assert_eq!(nodes.len(), 2);
    assert_eq!(i64_or(&nodes[0], "id", 0), 10);
    assert_eq!(i64_or(&nodes[0], "stage_id", 0), 1);
    assert_eq!(u64_or(&nodes[0], "pool_id", 0), 0);
    assert_eq!(i64_or(&nodes[1], "id", 0), 11);
    assert_eq!(i64_or(&nodes[1], "stage_id", 0), 2);
    assert_eq!(u64_or(&nodes[1], "pool_id", 0), 2);

    let edges = get(&json, "edges").as_array().expect("edges is an array");
    assert_eq!(edges.len(), 1);
    assert_eq!(i64_or(&edges[0], "source_id", 0), 10);
    assert_eq!(i64_or(&edges[0], "target_id", 0), 11);
    assert!((as_f64(&edges[0], "probability") - 0.75).abs() < 1e-12);

    let season_manifest_json = get(&json, "season_manifest");
    assert_eq!(
        as_u64(season_manifest_json, "cycle_code"),
        u64::from(SEASON_CYCLE_CODE_MONTHLY)
    );
    assert_eq!(as_u64(season_manifest_json, "n_seasons"), 12);
    let hydro_orders = get(season_manifest_json, "hydro_orders")
        .as_array()
        .expect("hydro_orders is an array");
    assert_eq!(hydro_orders.len(), 2);
    assert_eq!(i64_or(&hydro_orders[0], "hydro_id", 0), 3);
    assert_eq!(
        get(&hydro_orders[0], "orders"),
        &json!([1, 2, 1, 1, 3, 2, 1, 1, 2, 2, 1, 1])
    );
    assert_eq!(i64_or(&hydro_orders[1], "hydro_id", 0), 9);
    assert_eq!(
        get(&hydro_orders[1], "orders"),
        &json!([4, 4, 3, 3, 2, 2, 1, 1, 2, 2, 3, 3])
    );

    let document = json!({
        "format_version": FORMAT_VERSION,
        "software": "cobre",
        "software_version": "9.9.9",
        "created_at": "2026-08-23T12:00:00Z",
        "num_stages": 60,
        "n_pools": 3,
        "nodes": [
            {"id": 10, "stage_id": 1, "pool_id": 0},
            {"id": 11, "stage_id": 2, "pool_id": 2}
        ],
        "edges": [
            {"source_id": 10, "target_id": 11, "probability": 0.75}
        ],
        "completed_iterations": 50,
        "final_lower_bound": 1234.5,
        "best_upper_bound": 1300.0,
        "max_iterations": 200,
        "forward_passes": 4,
        "warm_start_cuts": 7,
        "warm_start_counts": [3, 4],
        "rng_seed": 42,
        "total_visited_states": 999,
        "training_block_mode": "parallel",
        "training_block_mode_per_stage": ["parallel", "chronological"],
        "cost_scale_factor": 1_000_000.0,
        "lower_bound_history": [1300.5, 1250.25, 1234.5],
        "season_manifest": {
            "cycle_code": 0,
            "n_seasons": 12,
            "hydro_orders": [
                {"hydro_id": 3, "orders": [1, 2, 1, 1, 3, 2, 1, 1, 2, 2, 1, 1]},
                {"hydro_id": 9, "orders": [4, 4, 3, 3, 2, 2, 1, 1, 2, 2, 3, 3]}
            ]
        }
    });
    let flatc_buf = flatc_encode(&document, "CheckpointManifest");
    let from_flatc = deserialize_checkpoint_manifest(&flatc_buf)
        .expect("hand-rolled reader must consume flatc-built buffer");
    assert_manifest_eq(&from_flatc, &manifest);
}

/// Dedicated round-trip for `season_manifest`, independent of
/// [`checkpoint_manifest_round_trip`]'s fixture: hand-rolled writer → `flatc`
/// reader (checked field by field) → `flatc` writer → hand-rolled reader.
#[test]
fn season_manifest_round_trips_through_flatc() {
    let mut manifest = conformance_manifest_value();
    manifest.season_manifest = SeasonManifest {
        cycle_code: SEASON_CYCLE_CODE_WEEKLY,
        n_seasons: 4,
        hydro_orders: vec![
            HydroSeasonOrders {
                hydro_id: 1,
                orders: vec![2, 2, 3, 1],
            },
            HydroSeasonOrders {
                hydro_id: 5,
                orders: vec![0, 1, 1, 2],
            },
        ],
    };

    let buf = serialize_checkpoint_manifest(&manifest);
    let hand_rolled = deserialize_checkpoint_manifest(&buf)
        .expect("hand-rolled reader must consume its own buffer");
    assert_manifest_eq(&hand_rolled, &manifest);

    let json = flatc_decode(&buf, "CheckpointManifest");
    let season_manifest_json = get(&json, "season_manifest");
    assert_eq!(
        as_u64(season_manifest_json, "cycle_code"),
        u64::from(SEASON_CYCLE_CODE_WEEKLY)
    );
    assert_eq!(as_u64(season_manifest_json, "n_seasons"), 4);
    let hydro_orders = get(season_manifest_json, "hydro_orders")
        .as_array()
        .expect("hydro_orders is an array");
    assert_eq!(hydro_orders.len(), 2);
    assert_eq!(i64_or(&hydro_orders[0], "hydro_id", 0), 1);
    assert_eq!(get(&hydro_orders[0], "orders"), &json!([2, 2, 3, 1]));
    assert_eq!(i64_or(&hydro_orders[1], "hydro_id", 0), 5);
    assert_eq!(get(&hydro_orders[1], "orders"), &json!([0, 1, 1, 2]));

    let flatc_buf = flatc_encode(&json, "CheckpointManifest");
    let from_flatc = deserialize_checkpoint_manifest(&flatc_buf)
        .expect("hand-rolled reader must consume flatc-built buffer");
    assert_manifest_eq(&from_flatc, &manifest);
}

/// The `None` half of the `Option<f64>` producer contract that
/// [`checkpoint_manifest_round_trip`] exercises only in its `Some` form:
/// `best_upper_bound` and `cost_scale_factor` both `None` round-trip as `None`,
/// never `Some(0.0)` — the writer omits an absent slot, the reader keeps it `None`.
#[test]
fn checkpoint_manifest_producer_optionals_round_trip_none() {
    let mut manifest = conformance_manifest_value();
    manifest.producer.best_upper_bound = None;
    manifest.producer.cost_scale_factor = None;
    let buf = serialize_checkpoint_manifest(&manifest);

    let hand_rolled = deserialize_checkpoint_manifest(&buf)
        .expect("hand-rolled reader must consume its own buffer");
    assert_eq!(
        hand_rolled.producer.best_upper_bound, None,
        "an absent best_upper_bound must decode to None, never Some(0.0)"
    );
    assert_eq!(
        hand_rolled.producer.cost_scale_factor, None,
        "an absent cost_scale_factor must decode to None, never Some(0.0)"
    );
    assert_manifest_eq(&hand_rolled, &manifest);

    let json = flatc_decode(&buf, "CheckpointManifest");
    assert!(
        json.get("best_upper_bound").is_none(),
        "a None best_upper_bound must be absent from the buffer, not a 0.0 slot"
    );
    assert!(
        json.get("cost_scale_factor").is_none(),
        "a None cost_scale_factor must be absent from the buffer, not a 0.0 slot"
    );
}

/// The reject-stale-version half of the dual-owned obligation: a manifest buffer
/// whose `format_version` is not [`FORMAT_VERSION`] is rejected before any field
/// is trusted, so a stale-version manifest never reaches a consumer.
#[test]
fn checkpoint_manifest_rejects_stale_format_version() {
    let mut manifest = conformance_manifest_value();
    let stale = FORMAT_VERSION + 1;
    manifest.format_version = stale;
    let buf = serialize_checkpoint_manifest(&manifest);

    let err = deserialize_checkpoint_manifest(&buf)
        .expect_err("a manifest with a stale format_version must be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("format_version") && msg.contains(&stale.to_string()),
        "rejection must name the version mismatch: {err}"
    );
}

// ─── Schema neutrality (value-function artifact) ─────────────────────────────

/// The published schema speaks affine-piece vocabulary: no standalone `Cut`
/// table, no `state_at_generation` field, no live (non-deprecated)
/// `domination_count` slot, and the header names the `src/output/policy/`
/// directory module. A text inspection of `schemas/policy.fbs`.
#[test]
fn schema_carries_no_algorithm_vocabulary() {
    let schema = std::fs::read_to_string(schema_path()).expect("read policy.fbs");
    assert!(
        !schema.contains("table Cut "),
        "the schema must not declare a `Cut` table (renamed to AffinePiece)"
    );
    assert!(
        schema.contains("table AffinePiece"),
        "the schema must declare the neutral `AffinePiece` table"
    );
    assert!(
        !schema.contains("state_at_generation"),
        "the deleted `state_at_generation` field must not appear"
    );
    // A reclaimed id-4 field named `intercept` — not a live `domination_count`.
    assert!(
        !schema.contains("domination_count:uint32 (id: 4);"),
        "no live domination_count slot may survive the id-4 reclaim"
    );
    assert!(
        schema.contains("src/output/policy/"),
        "the header must name the src/output/policy/ directory module"
    );
}

/// `EntitySlot`: id 4 and id 5 are burned `deprecated` placeholders, and the
/// artifact declares the `CBVF` `file_identifier` plus a `file_extension`. A
/// text inspection of `schemas/policy.fbs`.
#[test]
fn schema_burns_delivery_anchor_and_declares_file_identifier() {
    let schema = std::fs::read_to_string(schema_path()).expect("read policy.fbs");
    assert!(
        schema.contains("delivery_anchor:int32 (id: 4, deprecated);"),
        "EntitySlot id 4 must be the burned, deprecated delivery_anchor placeholder"
    );
    assert!(
        schema.contains("delivery_date:int32 (id: 5, deprecated);"),
        "EntitySlot id 5 must be the burned, deprecated delivery_date placeholder"
    );
    assert!(
        schema.contains("file_identifier \"CBVF\";"),
        "the schema must declare the CBVF file_identifier"
    );
    assert!(
        schema.contains("file_extension"),
        "the schema must declare a file_extension"
    );
}

// ─── EntitySlot per-family dating (ids: 6/7/8) round-trip + forward-compat ────

/// Round-trip: an `EntitySlot` carrying `reference_date` (inflow-lag) or
/// `interval_start`/`interval_end` (transit bucket) survives the hand-rolled
/// writer -> hand-rolled reader path, and the same buffer decodes through
/// `flatc` with each field at its own slot id (6/7/8).
#[test]
fn entity_slot_per_family_dates_round_trip() {
    let coeffs = [1.0, 2.0];
    let cuts = [PolicyCutRecord {
        cut_id: 1,
        slot_index: 0,
        iteration: 1,
        forward_pass_index: 0,
        intercept: 3.0,
        coefficients: &coeffs,
        is_active: true,
    }];
    let manifest = vec![
        EntitySlot::inflow_lag(5, 3, true).with_reference_date(20_310_401),
        EntitySlot::transit_bucket(9, 2, true).with_interval(20_311_201, 20_320_101),
    ];
    let buf = ser_cuts(3, 2, 8, 0, &cuts, &[0], 1, &manifest);

    let result: StageCutsReadResult =
        deserialize_stage_cuts(&buf).expect("hand-rolled reader must consume its own buffer");
    assert_eq!(result.entity_manifest.len(), 2);
    assert_eq!(result.entity_manifest[0].reference_date, 20_310_401);
    assert_eq!(
        result.entity_manifest[0].interval_start,
        ENTITY_SLOT_DATE_SENTINEL
    );
    assert_eq!(
        result.entity_manifest[0].interval_end,
        ENTITY_SLOT_DATE_SENTINEL
    );
    assert_eq!(result.entity_manifest[1].interval_start, 20_311_201);
    assert_eq!(result.entity_manifest[1].interval_end, 20_320_101);
    assert_eq!(
        result.entity_manifest[1].reference_date,
        ENTITY_SLOT_DATE_SENTINEL
    );

    let json = flatc_decode(&buf, "StageCuts");
    let arr = get(&json, "entity_manifest")
        .as_array()
        .expect("entity_manifest is an array")
        .clone();
    assert_eq!(i64_or(&arr[0], "reference_date", 0), i64::from(20_310_401));
    assert_eq!(i64_or(&arr[1], "interval_start", 0), i64::from(20_311_201));
    assert_eq!(i64_or(&arr[1], "interval_end", 0), i64::from(20_320_101));
}

/// Forward-compat: a buffer written against a pre-`id:6` `EntitySlot` schema
/// (fields through the now-retired `delivery_date` at id 5 only) must
/// deserialize with every slot's `reference_date`, `interval_start` and
/// `interval_end` at the sentinel and no error. The rewritten-schema
/// technique this test uses (flatc cannot emit a field the schema lacks, so
/// the buffer is built from a schema stopping short of the field under test)
/// is mirrored by `pre_node_id_stage_states_reads_as_sentinel`.
#[test]
fn pre_interval_entity_slot_reads_as_sentinel() {
    let schema_pre_interval = "
namespace Cobre.IO.Policy;

file_identifier \"CBVF\";

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
";
    let dir = TempDir::new().unwrap();
    let pre_interval_schema = dir.path().join("pre_interval.fbs");
    std::fs::write(&pre_interval_schema, schema_pre_interval).unwrap();

    let document = json!({
        "stage_id": 1,
        "state_dimension": 2,
        "capacity": 4,
        "warm_start_count": 0,
        "populated_count": 1,
        "active_cut_indices": [0],
        "cuts": [
            {
                "piece_id": 1,
                "slot_index": 0,
                "iteration": 1,
                "forward_pass_index": 0,
                "intercept": 1.0,
                "coefficients": [1.0, 2.0],
                "is_active": true
            }
        ],
        "entity_manifest": [
            {"entity_type": "HydroStorage", "entity_id": 1, "subindex": 0, "was_active": true, "delivery_date": 20240101},
            {"entity_type": "HydroInflowLag", "entity_id": 7, "subindex": 1, "was_active": true}
        ]
    });
    let json_path = dir.path().join("doc.json");
    std::fs::write(&json_path, serde_json::to_vec(&document).unwrap()).unwrap();

    let status = flatc_command()
        .arg("-b")
        .arg("--root-type")
        .arg(qualified("StageCuts"))
        .arg("-o")
        .arg(dir.path())
        .arg(&pre_interval_schema)
        .arg(&json_path)
        .status()
        .expect("run flatc -b on pre-interval schema");
    assert!(status.success(), "flatc -b on pre-interval schema failed");
    let buf = std::fs::read(dir.path().join("doc.bin")).unwrap();

    let result =
        deserialize_stage_cuts(&buf).expect("hand-rolled reader must accept pre-interval buffer");
    assert_eq!(result.entity_manifest.len(), 2);
    for (i, slot) in result.entity_manifest.iter().enumerate() {
        assert_eq!(
            slot.reference_date, ENTITY_SLOT_DATE_SENTINEL,
            "pre-interval slot {i} reference_date must read back as the sentinel"
        );
        assert_eq!(
            slot.interval_start, ENTITY_SLOT_DATE_SENTINEL,
            "pre-interval slot {i} interval_start must read back as the sentinel"
        );
        assert_eq!(
            slot.interval_end, ENTITY_SLOT_DATE_SENTINEL,
            "pre-interval slot {i} interval_end must read back as the sentinel"
        );
    }
}
