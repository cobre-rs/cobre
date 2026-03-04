//! End-to-end integration tests for `cobre_io::load_case`.
//!
//! Each test constructs a complete case directory in a [`TempDir`] using the
//! helpers in [`helpers`], calls [`load_case`], and verifies either the
//! returned [`System`] entity counts or the returned [`LoadError`] variant.
//!
//! These tests exercise the full five-layer validation pipeline and the
//! `SystemBuilder` assembly step in one shot.
#![allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::too_many_lines,
    clippy::doc_markdown
)]

mod helpers;

use cobre_core::EntityId;
use cobre_io::{deserialize_system, load_case, serialize_system};
use tempfile::TempDir;

// ── test_minimal_valid_case ────────────────────────────────────────────────────

/// Given the 8 required files for a minimal case, `load_case` must return an
/// `Ok(System)` with exactly 1 bus, 0 hydros, 0 thermals, 0 lines, and 1 stage.
#[test]
fn test_minimal_valid_case() {
    let dir = TempDir::new().unwrap();
    helpers::make_minimal_case(&dir);

    let result = load_case(dir.path());

    let system = match result {
        Ok(s) => s,
        Err(e) => panic!("expected Ok(System) for minimal case, got Err: {e}"),
    };

    assert_eq!(
        system.n_buses(),
        1,
        "minimal case should have exactly 1 bus"
    );
    assert_eq!(system.n_hydros(), 0, "minimal case should have 0 hydros");
    assert_eq!(
        system.n_thermals(),
        0,
        "minimal case should have 0 thermals"
    );
    assert_eq!(system.n_lines(), 0, "minimal case should have 0 lines");
    assert_eq!(
        system.n_stages(),
        1,
        "minimal case should have exactly 1 stage"
    );
    assert!(
        system.bus(EntityId(1)).is_some(),
        "bus with id=1 should be found by O(1) lookup"
    );
}

// ── test_multi_entity_case ────────────────────────────────────────────────────

/// Given a richer case with 2 buses, 1 hydro, 1 thermal, 1 line, and 2 stages,
/// `load_case` must return `Ok(System)` with matching entity counts.
#[test]
fn test_multi_entity_case() {
    let dir = TempDir::new().unwrap();
    helpers::make_multi_entity_case(&dir);

    let result = load_case(dir.path());

    let system = match result {
        Ok(s) => s,
        Err(e) => panic!("expected Ok(System) for multi-entity case, got Err: {e}"),
    };

    assert_eq!(
        system.n_buses(),
        2,
        "multi-entity case should have exactly 2 buses"
    );
    assert_eq!(
        system.n_hydros(),
        1,
        "multi-entity case should have exactly 1 hydro"
    );
    assert_eq!(
        system.n_thermals(),
        1,
        "multi-entity case should have exactly 1 thermal"
    );
    assert_eq!(
        system.n_lines(),
        1,
        "multi-entity case should have exactly 1 line"
    );
    assert_eq!(
        system.n_stages(),
        2,
        "multi-entity case should have exactly 2 stages"
    );
    assert!(
        system.bus(EntityId(1)).is_some(),
        "bus with id=1 should be accessible via O(1) lookup"
    );
    assert!(
        system.bus(EntityId(2)).is_some(),
        "bus with id=2 should be accessible via O(1) lookup"
    );
}

// ── test_missing_required_file ────────────────────────────────────────────────

/// Given a minimal case with `system/buses.json` removed, `load_case` must
/// return an `Err` whose display representation contains `"buses"`.
#[test]
fn test_missing_required_file() {
    let dir = TempDir::new().unwrap();
    helpers::make_minimal_case(&dir);

    // Remove the required buses file after populating the full case.
    std::fs::remove_file(dir.path().join("system/buses.json")).unwrap();

    let result = load_case(dir.path());

    match result {
        Err(err) => {
            let display = err.to_string();
            assert!(
                display.contains("buses"),
                "error display should mention 'buses', got: {display}"
            );
        }
        Ok(_) => panic!("expected Err when buses.json is missing, got Ok"),
    }
}

// ── test_malformed_json ───────────────────────────────────────────────────────

/// Given a minimal case with `system/hydros.json` containing invalid JSON,
/// `load_case` must return an `Err` (parse failure).
#[test]
fn test_malformed_json() {
    let dir = TempDir::new().unwrap();
    helpers::make_minimal_case(&dir);

    // Overwrite hydros.json with syntactically invalid content.
    helpers::write_file(
        dir.path(),
        "system/hydros.json",
        "{ this is not valid json }",
    );

    let result = load_case(dir.path());

    match result {
        Err(err) => {
            // The error must be a parse or constraint error — not an Ok.
            // We only assert it is an Err; the exact variant is implementation-
            // defined (ParseError or ConstraintError wrapping the parse failure).
            let display = err.to_string();
            assert!(
                !display.is_empty(),
                "error display should be non-empty for malformed JSON"
            );
        }
        Ok(_) => panic!("expected Err for malformed hydros.json, got Ok"),
    }
}

// ── test_referential_integrity_violation ─────────────────────────────────────

/// Given a case where a hydro references a non-existent bus (id=999),
/// `load_case` must return an `Err` whose display representation mentions
/// the invalid reference.
#[test]
fn test_referential_integrity_violation() {
    let dir = TempDir::new().unwrap();
    helpers::make_referential_violation_case(&dir);

    let result = load_case(dir.path());

    match result {
        Err(err) => {
            let display = err.to_string();
            // The error description must mention that a reference is missing —
            // the structural format from ValidationContext::into_result is:
            // "[InvalidReference] system/hydros.json (Hydro 1): ... 999 ..."
            assert!(
                display.contains("999") || display.contains("bus") || display.contains("Bus"),
                "error display should mention the invalid bus reference (999), got: {display}"
            );
        }
        Ok(_) => panic!("expected Err when hydro references non-existent bus_id=999, got Ok"),
    }
}

// ── test_postcard_round_trip ──────────────────────────────────────────────────

/// Given a System produced by `load_case`, serializing it with `serialize_system`
/// and deserializing with `deserialize_system` must produce a System with the
/// same entity counts and working O(1) lookups.
#[test]
fn test_postcard_round_trip() {
    let dir = TempDir::new().unwrap();
    helpers::make_minimal_case(&dir);

    let original = load_case(dir.path())
        .unwrap_or_else(|e| panic!("load_case should succeed for minimal case, got: {e}"));

    let bytes = serialize_system(&original)
        .unwrap_or_else(|e| panic!("serialize_system should succeed, got: {e}"));

    assert!(!bytes.is_empty(), "serialized bytes should be non-empty");

    let deserialized = deserialize_system(&bytes)
        .unwrap_or_else(|e| panic!("deserialize_system should succeed, got: {e}"));

    assert_eq!(
        deserialized.n_buses(),
        original.n_buses(),
        "bus count must match after postcard round-trip"
    );
    assert!(
        deserialized.bus(EntityId(1)).is_some(),
        "O(1) bus lookup must work after index rebuild on deserialized System"
    );

    // Verify no data was lost for other entity types.
    assert_eq!(
        deserialized.n_hydros(),
        original.n_hydros(),
        "hydro count must match after round-trip"
    );
    assert_eq!(
        deserialized.n_thermals(),
        original.n_thermals(),
        "thermal count must match after round-trip"
    );
    assert_eq!(
        deserialized.n_lines(),
        original.n_lines(),
        "line count must match after round-trip"
    );
    assert_eq!(
        deserialized.n_stages(),
        original.n_stages(),
        "stage count must match after round-trip"
    );
}
