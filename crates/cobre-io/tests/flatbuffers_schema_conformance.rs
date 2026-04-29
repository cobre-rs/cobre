//! Schema-conformance tests for `schemas/policy.fbs`.
//!
//! These tests are gated behind the `flatc-conformance` cargo feature and
//! require the `flatc` compiler on `$PATH` (or the `FLATC` environment
//! variable). They prove that the hand-rolled `FlatBuffers` writer/reader in
//! `src/output/policy.rs` produces and consumes buffers that are
//! bit-compatible with the canonical schema in `schemas/policy.fbs`.
//!
//! ## What "drift" looks like
//!
//! The hand-rolled writer encodes field positions via the `*_FIELD_*: u16`
//! slot constants in `policy.rs`. The schema encodes them via `(id: N)`
//! attributes on each field of each table. These two views must agree
//! exactly: a mismatch corrupts every consumer that uses the schema. The
//! tests detect a mismatch in **both directions**:
//!
//! 1. **Writer → flatc:** the hand-rolled writer emits a buffer; `flatc -t`
//!    decodes it to JSON using the schema. Assertions on the JSON catch
//!    "writer wrote a slot the schema doesn't know about" and "writer wrote
//!    a value at the wrong slot."
//! 2. **flatc → reader:** `flatc -b` encodes a JSON document into a buffer
//!    using the schema. The hand-rolled reader decodes it. Assertions on
//!    the decoded struct catch "schema declared a slot the reader expects
//!    at a different offset."
//!
//! Together the two directions form a tight loop: any single change to one
//! side without the other will fail at least one of the two checks.
//!
//! ## Running
//!
//! ```bash
//! cargo test -p cobre-io --features flatc-conformance --test flatbuffers_schema_conformance
//! ```
//!
//! When `flatc` is missing the tests panic with a clear message rather than
//! silently passing — invoking the feature is an explicit request to run
//! these checks.

#![cfg(feature = "flatc-conformance")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::{Path, PathBuf};
use std::process::Command;

use cobre_io::{
    OwnedPolicyBasisRecord, OwnedPolicyCutRecord, PolicyBasisRecord, PolicyCutRecord,
    StageCutsReadResult, StageStatesPayload, StageStatesReadResult, deserialize_stage_basis,
    deserialize_stage_cuts, deserialize_stage_states, serialize_stage_basis, serialize_stage_cuts,
    serialize_stage_states,
};
use serde_json::{Value, json};
use tempfile::TempDir;

/// Resolve the `flatc` command. Panics with a clear message when missing.
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

/// Run `flatc -t --strict-json --raw-binary --root-type T` and return the
/// decoded JSON value.
fn flatc_decode(buf: &[u8], root_type: &str) -> Value {
    let dir = TempDir::new().expect("create tempdir");
    let bin_path = dir.path().join("buf.bin");
    std::fs::write(&bin_path, buf).expect("write bin");

    let status = flatc_command()
        .arg("-t")
        .arg("--strict-json")
        .arg("--raw-binary")
        .arg("--root-type")
        .arg(root_type)
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

/// Run `flatc -b --root-type T` on the provided JSON document and return
/// the binary buffer.
fn flatc_encode(json: &Value, root_type: &str) -> Vec<u8> {
    let dir = TempDir::new().expect("create tempdir");
    let json_path = dir.path().join("doc.json");
    std::fs::write(&json_path, serde_json::to_vec(json).unwrap()).expect("write JSON");

    let status = flatc_command()
        .arg("-b")
        .arg("--root-type")
        .arg(root_type)
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

/// `serde_json::Value` field accessor that panics with a useful message.
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

// ─── StageCuts ───────────────────────────────────────────────────────────────

#[test]
fn stage_cuts_writer_matches_schema() {
    // Two cuts at different slot positions, both active, with non-trivial
    // coefficients. Three stage-level scalar fields exercised.
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
    let buf = serialize_stage_cuts(3, 4, 16, 0, &cuts, &active, 2);

    let json = flatc_decode(&buf, "StageCuts");

    assert_eq!(as_u64(&json, "stage_id"), 3);
    assert_eq!(as_u64(&json, "state_dimension"), 4);
    assert_eq!(as_u64(&json, "capacity"), 16);
    assert_eq!(as_u64(&json, "warm_start_count"), 0);
    assert_eq!(as_u64(&json, "populated_count"), 2);
    assert_eq!(get(&json, "active_cut_indices"), &json!([0]));

    let cuts_json = get(&json, "cuts").as_array().expect("cuts is an array");
    assert_eq!(cuts_json.len(), 2);

    let c0 = &cuts_json[0];
    assert_eq!(as_u64(c0, "cut_id"), 7);
    assert_eq!(as_u64(c0, "slot_index"), 0);
    assert_eq!(as_u64(c0, "iteration"), 1);
    assert_eq!(as_u64(c0, "forward_pass_index"), 0);
    assert!((as_f64(c0, "intercept") - 100.5).abs() < 1e-12);
    assert_eq!(get(c0, "coefficients"), &json!([1.0, 2.0, 3.0, 4.0]));
    assert!(as_bool(c0, "is_active"));
    // The deprecated `domination_count` slot must be absent from flatc's
    // JSON output. flatc's `deprecated` attribute strips the accessor on
    // both sides; presence here would mean either the schema forgot the
    // attribute or the writer regressed and started emitting it again.
    assert!(
        c0.get("domination_count").is_none(),
        "deprecated domination_count must not appear in flatc-decoded JSON: {c0}"
    );

    let c1 = &cuts_json[1];
    assert_eq!(as_u64(c1, "cut_id"), 8);
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
                "cut_id": 99,
                "slot_index": 0,
                "iteration": 4,
                "forward_pass_index": 2,
                "intercept": 12.5,
                "coefficients": [0.1, 0.2, 0.3],
                "is_active": true
            }
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
    let payload = StageStatesPayload {
        stage_id: 2,
        state_dimension: 3,
        count: 2,
        data: &data,
    };
    let buf = serialize_stage_states(&payload);

    let json = flatc_decode(&buf, "StageStates");

    assert_eq!(as_u64(&json, "stage_id"), 2);
    assert_eq!(as_u64(&json, "state_dimension"), 3);
    assert_eq!(as_u64(&json, "count"), 2);
    assert_eq!(get(&json, "data"), &json!([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
}

#[test]
fn stage_states_reader_consumes_flatc_buffer() {
    let document = json!({
        "stage_id": 9,
        "state_dimension": 2,
        "count": 3,
        "data": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
    });
    let buf = flatc_encode(&document, "StageStates");
    let result: StageStatesReadResult =
        deserialize_stage_states(&buf).expect("hand-rolled reader must consume flatc-built buffer");

    assert_eq!(result.stage_id, 9);
    assert_eq!(result.state_dimension, 2);
    assert_eq!(result.count, 3);
    assert_eq!(result.data, vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0]);
}

// ─── Legacy / deprecated-slot regression ─────────────────────────────────────

/// Pre-v0.5.0 policy files set the now-deprecated `domination_count` slot
/// (vtable slot 12) on every cut. The schema marks that slot
/// `(id: 4, deprecated)`, which:
///   • removes the accessor on the writer side, so flatc-generated writers
///     cannot emit it, and
///   • lets readers tolerate its presence in legacy files via the
///     graceful-absence rule.
///
/// This test forces a value into the slot via flatc by temporarily
/// rewriting the schema without `deprecated`, then verifies the
/// hand-rolled reader still decodes the buffer correctly.
#[test]
fn legacy_domination_count_slot_is_ignored_by_reader() {
    let schema_with_legacy = "
namespace Cobre.IO.Policy;

table Cut {
  cut_id:uint64 (id: 0);
  slot_index:uint32 (id: 1);
  iteration:uint32 (id: 2);
  forward_pass_index:uint32 (id: 3);
  domination_count:uint32 (id: 4);
  intercept:float64 (id: 5);
  coefficients:[float64] (id: 6);
  state_at_generation:[float64] (id: 7);
  is_active:bool (id: 8);
}

table StageCuts {
  stage_id:uint32 (id: 0);
  state_dimension:uint32 (id: 1);
  capacity:uint32 (id: 2);
  warm_start_count:uint32 (id: 3);
  cuts:[Cut] (id: 4);
  active_cut_indices:[uint32] (id: 5);
  populated_count:uint32 (id: 6);
}
";
    let dir = TempDir::new().unwrap();
    let legacy_schema = dir.path().join("legacy.fbs");
    std::fs::write(&legacy_schema, schema_with_legacy).unwrap();

    let document = json!({
        "stage_id": 1,
        "state_dimension": 2,
        "capacity": 4,
        "warm_start_count": 0,
        "populated_count": 1,
        "active_cut_indices": [0],
        "cuts": [
            {
                "cut_id": 1,
                "slot_index": 0,
                "iteration": 1,
                "forward_pass_index": 0,
                "domination_count": 99999,
                "intercept": 1.0,
                "coefficients": [1.0, 2.0],
                "is_active": true
            }
        ]
    });
    let json_path = dir.path().join("doc.json");
    std::fs::write(&json_path, serde_json::to_vec(&document).unwrap()).unwrap();

    let status = flatc_command()
        .arg("-b")
        .arg("--root-type")
        .arg("StageCuts")
        .arg("-o")
        .arg(dir.path())
        .arg(&legacy_schema)
        .arg(&json_path)
        .status()
        .expect("run flatc -b on legacy schema");
    assert!(status.success(), "flatc -b on legacy schema failed");
    let buf = std::fs::read(dir.path().join("doc.bin")).unwrap();

    let result =
        deserialize_stage_cuts(&buf).expect("hand-rolled reader must accept legacy buffer");
    assert_eq!(result.cuts.len(), 1);
    let cut = &result.cuts[0];
    assert_eq!(cut.cut_id, 1);
    assert_eq!(cut.coefficients, vec![1.0, 2.0]);
    assert!(cut.is_active);
}
