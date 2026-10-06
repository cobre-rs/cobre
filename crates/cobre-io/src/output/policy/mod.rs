//! `FlatBuffers` builder and reader types for policy checkpoint serialization.
//!
//! Types use generic names to maintain infrastructure crate genericity; conversion
//! from algorithm-specific types is the calling crate's responsibility.
//!
//! The canonical wire-format description is `schemas/policy.fbs` in this crate
//! (namespace `Cobre.IO.Policy`, tables `StageCuts`, `AffinePiece`, `StageBasis`,
//! `StageStates`); the build hand-writes both the builder calls and the safe
//! raw-byte parser rather than consuming the schema. The `*_FIELD_*: u16` slot
//! constants in `codec` mirror the schema's `(id: N)` attributes via
//! `slot = (id + 2) * 2` and MUST stay in sync — the `flatc-conformance` feature
//! gates the round-trip test in `tests/flatbuffers_schema_conformance.rs` that
//! fails when they diverge. The wire layout itself is documented at `codec`.

pub mod checkpoint;
pub mod codec;
pub mod records;

pub use checkpoint::{
    ResolvedCheckpoint, read_policy_checkpoint, resolve_policy_checkpoint, write_policy_checkpoint,
};
pub(crate) use checkpoint::{check_checkpoint_replaceable, checkpoint_target};
pub use codec::{deserialize_stage_basis, deserialize_stage_cuts, deserialize_stage_states};
pub use codec::{serialize_stage_basis, serialize_stage_cuts, serialize_stage_states};
pub use records::{
    CheckpointManifest, ENTITY_SLOT_DATE_SENTINEL, EntitySlot, FORMAT_VERSION, GraphManifest,
    HydroSeasonOrders, ManifestEdge, ManifestNode, OwnedPolicyBasisRecord, OwnedPolicyCutRecord,
    PolicyBasisRecord, PolicyCheckpoint, PolicyCutRecord, ProducerBlock, SEASON_CYCLE_CODE_ABSENT,
    SEASON_CYCLE_CODE_CUSTOM, SEASON_CYCLE_CODE_MONTHLY, SEASON_CYCLE_CODE_WEEKLY,
    STAGE_CUTS_GRAPH_STAGE_ID_SENTINEL, STAGE_CUTS_NODE_ID_SENTINEL,
    STAGE_CUTS_PRICED_STATE_DATE_SENTINEL, STAGE_STATES_NODE_ID_SENTINEL, SeasonManifest,
    StageCutsPayload, StageCutsReadResult, StageStatesPayload, StageStatesReadResult, StateFamily,
    decode_slot_date, encode_slot_date,
};

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::unreadable_literal
)]
mod tests {
    use std::collections::BTreeMap;
    use std::path::{Path, PathBuf};

    use super::super::error::OutputError;
    use super::super::software::policy_checkpoint_remedy;
    use super::checkpoint::finish_interrupted_swap;
    use super::*;

    fn make_cut_record(
        cut_id: u64,
        slot_index: u32,
        iteration: u32,
        coefficients: &[f64],
    ) -> PolicyCutRecord<'_> {
        PolicyCutRecord {
            cut_id,
            slot_index,
            iteration,
            forward_pass_index: 0,
            intercept: 42.0,
            coefficients,
            is_active: true,
        }
    }

    /// Serialize a `StageCuts` buffer from positional fields, defaulting the
    /// self-describing per-pool facts the codec round-trip does not exercise.
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

    // ── serialize_stage_cuts tests ────────────────────────────────────────────

    #[test]
    fn serialize_stage_cuts_single_cut_round_trip() {
        let coefficients = [1.0_f64, 2.0, 3.0];
        let cut = PolicyCutRecord {
            cut_id: 7,
            slot_index: 5,
            iteration: 3,
            forward_pass_index: 0,
            intercept: 42.0,
            coefficients: &coefficients,
            is_active: true,
        };

        let buf = ser_cuts(0, 3, 100, 0, &[cut], &[0], 1, &[]);

        assert!(!buf.is_empty(), "buffer must not be empty");
        assert!(buf.len() >= 4, "buffer must have at least 4 bytes");
        let root_offset = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
        assert!(
            root_offset < buf.len(),
            "root offset must point inside the buffer"
        );
    }

    #[test]
    fn serialize_stage_cuts_empty_cuts_valid_buffer() {
        let buf = ser_cuts(0, 3, 100, 0, &[], &[], 0, &[]);

        assert!(!buf.is_empty(), "buffer must not be empty for empty cuts");
        assert!(
            buf.len() >= 4,
            "buffer must have at least 4 bytes even for empty cuts"
        );
        let root_offset = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
        assert!(
            root_offset < buf.len(),
            "root offset must point inside the buffer"
        );
    }

    #[test]
    fn serialize_stage_cuts_multiple_cuts_deterministic() {
        let c0 = [1.0_f64, 2.0, 3.0];
        let c1 = [4.0_f64, 5.0, 6.0];
        let c2 = [7.0_f64, 8.0, 9.0];

        let cuts = [
            make_cut_record(1, 0, 1, &c0),
            make_cut_record(2, 1, 1, &c1),
            make_cut_record(3, 2, 1, &c2),
        ];

        let buf_a = ser_cuts(5, 3, 50, 0, &cuts, &[0, 1, 2], 3, &[]);
        let buf_b = ser_cuts(5, 3, 50, 0, &cuts, &[0, 1, 2], 3, &[]);

        assert_eq!(buf_a, buf_b, "output must be byte-identical for same input");
    }

    #[test]
    fn serialize_stage_cuts_non_empty_for_varying_state_dimensions() {
        for &dim in &[1u32, 10, 100, 1000] {
            let coefs: Vec<f64> = (0..dim).map(f64::from).collect();
            let cut = PolicyCutRecord {
                cut_id: 0,
                slot_index: 0,
                iteration: 1,
                forward_pass_index: 0,
                intercept: 0.0,
                coefficients: &coefs,
                is_active: true,
            };
            let buf = ser_cuts(0, dim, 10, 0, &[cut], &[0], 1, &[]);
            assert!(
                !buf.is_empty(),
                "buffer must not be empty for state_dimension={dim}"
            );
        }
    }

    // ── serialize_stage_basis tests ───────────────────────────────────────────

    #[test]
    fn serialize_stage_basis_round_trip() {
        let record = PolicyBasisRecord {
            stage_id: 0,
            iteration: 5,
            column_status: &[0, 1, 2],
            row_status: &[1, 1, 0, 0],
            num_cut_rows: 2,
        };

        let buf = serialize_stage_basis(&record);

        assert!(!buf.is_empty(), "buffer must not be empty");
        assert!(buf.len() >= 4, "buffer must have at least 4 bytes");
        let root_offset = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
        assert!(
            root_offset < buf.len(),
            "root offset must point inside the buffer"
        );
    }

    #[test]
    fn serialize_stage_basis_empty_status_vectors() {
        let record = PolicyBasisRecord {
            stage_id: 1,
            iteration: 0,
            column_status: &[],
            row_status: &[],
            num_cut_rows: 0,
        };

        let buf = serialize_stage_basis(&record);

        assert!(
            !buf.is_empty(),
            "buffer must not be empty even with empty status vectors"
        );
        assert!(
            buf.len() >= 4,
            "buffer must have at least 4 bytes even with empty status vectors"
        );
    }

    #[test]
    fn serialize_stage_basis_deterministic() {
        let col = [0u8, 1, 2, 3];
        let row = [1u8, 0, 1, 0, 1];
        let record = PolicyBasisRecord {
            stage_id: 7,
            iteration: 12,
            column_status: &col,
            row_status: &row,
            num_cut_rows: 3,
        };

        let buf_a = serialize_stage_basis(&record);
        let buf_b = serialize_stage_basis(&record);

        assert_eq!(
            buf_a, buf_b,
            "basis output must be byte-identical for same input"
        );
    }

    // ── write_policy_checkpoint tests ─────────────────────────────────────────

    /// A trivial 1:1 chain graph manifest over `num_stages` nodes (node id ==
    /// stage id == pool id) — the shape a chain-degenerate study writes.
    fn chain_manifest(num_stages: u32) -> GraphManifest {
        let nodes = (0..num_stages)
            .map(|t| ManifestNode {
                id: i32::try_from(t).unwrap(),
                stage_id: i32::try_from(t).unwrap(),
                pool_id: t,
            })
            .collect();
        let edges = (0..num_stages.saturating_sub(1))
            .map(|t| ManifestEdge {
                source_id: i32::try_from(t).unwrap(),
                target_id: i32::try_from(t + 1).unwrap(),
                probability: 1.0,
            })
            .collect();
        GraphManifest {
            n_pools: num_stages,
            nodes,
            edges,
        }
    }

    /// Build a minimal [`CheckpointManifest`] for use in artifact tests.
    /// `state_dimension` is retained for call-site clarity; it now lives per-pool
    /// on the payloads, not in the manifest core.
    fn make_metadata(num_stages: u32, _state_dimension: u32) -> CheckpointManifest {
        CheckpointManifest {
            format_version: FORMAT_VERSION,
            software: Some("cobre".to_string()),
            software_version: "0.0.1".to_string(),
            created_at: "2026-03-08T00:00:00Z".to_string(),
            num_stages,
            graph_manifest: chain_manifest(num_stages),
            producer: ProducerBlock {
                completed_iterations: 10,
                final_lower_bound: 999.0,
                best_upper_bound: Some(1050.0),
                max_iterations: 100,
                forward_passes: 4,
                warm_start_cuts: 0,
                warm_start_counts: vec![0; num_stages as usize],
                rng_seed: 42,
                total_visited_states: 0,
                training_block_mode: "parallel".to_string(),
                training_block_mode_per_stage: vec![],
                cost_scale_factor: None,
                lower_bound_history: Vec::new(),
            },
            season_manifest: SeasonManifest::default(),
        }
    }

    /// Build a [`StageCutsPayload`] with `n_cuts` cuts, all using the supplied
    /// `coefficients` slice (shared across cuts for test simplicity).
    fn make_stage_cuts_payload<'a>(
        stage_id: u32,
        cuts: &'a [PolicyCutRecord<'a>],
        active_cut_indices: &'a [u32],
        state_dimension: u32,
    ) -> StageCutsPayload<'a> {
        StageCutsPayload {
            stage_id,
            state_dimension,
            capacity: 100,
            warm_start_count: 0,
            cuts,
            active_cut_indices,
            populated_count: u32::try_from(cuts.len()).unwrap(),
            entity_manifest: &[],
            cost_scale_factor: 1_000_000.0,
            node_id: -1,
            graph_stage_id: -1,
            priced_state_date: STAGE_CUTS_PRICED_STATE_DATE_SENTINEL,
        }
    }

    fn make_basis_record(stage_id: u32) -> PolicyBasisRecord<'static> {
        PolicyBasisRecord {
            stage_id,
            iteration: 10,
            column_status: &[0, 1, 2, 3],
            row_status: &[1, 0, 1, 0, 1],
            num_cut_rows: 2,
        }
    }

    #[test]
    fn write_policy_checkpoint_creates_directory_structure() {
        let tmp = tempfile::tempdir().unwrap();

        let c0 = [1.0_f64, 2.0, 3.0];
        let c1 = [4.0_f64, 5.0, 6.0];
        let cuts_s0 = [make_cut_record(1, 0, 1, &c0), make_cut_record(2, 1, 1, &c1)];
        let cuts_s1 = [make_cut_record(3, 0, 2, &c0)];
        let cuts_s2 = [make_cut_record(4, 0, 3, &c1)];

        let stage_cuts = [
            make_stage_cuts_payload(0, &cuts_s0, &[0, 1], 3),
            make_stage_cuts_payload(1, &cuts_s1, &[0], 3),
            make_stage_cuts_payload(2, &cuts_s2, &[0], 3),
        ];
        let basis_records = [
            make_basis_record(0),
            make_basis_record(1),
            make_basis_record(2),
        ];
        let metadata = make_metadata(3, 3);

        write_policy_checkpoint(tmp.path(), &stage_cuts, &basis_records, &metadata, &[])
            .expect("write_policy_checkpoint must succeed");

        assert!(tmp.path().join("cuts").is_dir(), "cuts/ must exist");
        assert!(tmp.path().join("basis").is_dir(), "basis/ must exist");

        for i in 0..3u32 {
            let p = tmp.path().join(format!("cuts/{i:03}.bin"));
            assert!(p.is_file(), "cuts/{i:03}.bin must exist");
        }

        for i in 0..3u32 {
            let p = tmp.path().join(format!("basis/{i:03}.bin"));
            assert!(p.is_file(), "basis/{i:03}.bin must exist");
        }

        assert!(
            tmp.path().join("manifest.bin").is_file(),
            "manifest.bin must exist"
        );
        assert!(
            !tmp.path().join("metadata.json").exists(),
            "metadata.json must NOT be written"
        );
    }

    #[test]
    fn write_policy_checkpoint_manifest_bin_carries_metadata() {
        let tmp = tempfile::tempdir().unwrap();

        let c0 = [1.0_f64, 2.0, 3.0];
        let cuts_s0 = [make_cut_record(1, 0, 1, &c0)];
        let stage_cuts = [make_stage_cuts_payload(0, &cuts_s0, &[0], 3)];
        let metadata = make_metadata(1, 3);

        write_policy_checkpoint(tmp.path(), &stage_cuts, &[], &metadata, &[])
            .expect("write_policy_checkpoint must succeed");

        assert!(
            tmp.path().join("manifest.bin").is_file(),
            "manifest.bin must exist"
        );

        let checkpoint = read_policy_checkpoint(tmp.path()).expect("read must succeed");
        assert_eq!(checkpoint.metadata.format_version, FORMAT_VERSION);
        assert_eq!(checkpoint.metadata.num_stages, 1, "num_stages must match");
        assert_eq!(
            checkpoint.metadata.producer.completed_iterations, 10,
            "completed_iterations must match under producer"
        );
    }

    #[test]
    fn write_policy_checkpoint_cut_files_non_empty() {
        let tmp = tempfile::tempdir().unwrap();

        let c0 = [1.0_f64, 2.0, 3.0];
        let c1 = [4.0_f64, 5.0, 6.0];
        let cuts_s0 = [make_cut_record(1, 0, 1, &c0), make_cut_record(2, 1, 1, &c1)];
        let cuts_s1 = [make_cut_record(3, 0, 2, &c0)];
        let cuts_s2 = [make_cut_record(4, 0, 3, &c1)];

        let stage_cuts = [
            make_stage_cuts_payload(0, &cuts_s0, &[0, 1], 3),
            make_stage_cuts_payload(1, &cuts_s1, &[0], 3),
            make_stage_cuts_payload(2, &cuts_s2, &[0], 3),
        ];
        let metadata = make_metadata(3, 3);

        write_policy_checkpoint(tmp.path(), &stage_cuts, &[], &metadata, &[])
            .expect("write_policy_checkpoint must succeed");

        for i in 0..3u32 {
            let p = tmp.path().join(format!("cuts/{i:03}.bin"));
            let bytes = std::fs::read(&p).unwrap();
            assert!(!bytes.is_empty(), "cuts/{i:03}.bin must not be empty");
            assert!(bytes.len() >= 4, "cuts/{i:03}.bin must have >= 4 bytes");
            let root_offset = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
            assert!(
                root_offset < bytes.len(),
                "cuts/{i:03}.bin root offset must be in-range"
            );
        }
    }

    #[test]
    fn write_policy_checkpoint_basis_files_non_empty() {
        let tmp = tempfile::tempdir().unwrap();

        let c0 = [1.0_f64, 2.0];
        let cuts_s0 = [make_cut_record(1, 0, 1, &c0)];
        let stage_cuts = [make_stage_cuts_payload(0, &cuts_s0, &[0], 2)];
        let basis_records = [make_basis_record(0)];
        let metadata = make_metadata(1, 2);

        write_policy_checkpoint(tmp.path(), &stage_cuts, &basis_records, &metadata, &[])
            .expect("write_policy_checkpoint must succeed");

        let p = tmp.path().join("basis/000.bin");
        let bytes = std::fs::read(&p).unwrap();
        assert!(!bytes.is_empty(), "basis/000.bin must not be empty");
        assert!(bytes.len() >= 4, "basis/000.bin must have >= 4 bytes");
        let root_offset = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
        assert!(
            root_offset < bytes.len(),
            "basis/000.bin root offset must be in-range"
        );
    }

    #[test]
    fn write_policy_checkpoint_empty_bases_no_basis_files() {
        let tmp = tempfile::tempdir().unwrap();

        let c0 = [1.0_f64, 2.0];
        let cuts_s0 = [make_cut_record(1, 0, 1, &c0)];
        let stage_cuts = [make_stage_cuts_payload(0, &cuts_s0, &[0], 2)];
        let metadata = make_metadata(1, 2);

        let result = write_policy_checkpoint(tmp.path(), &stage_cuts, &[], &metadata, &[]);

        assert!(
            result.is_ok(),
            "write_policy_checkpoint must return Ok(()) with empty stage_bases"
        );

        assert!(
            tmp.path().join("basis").is_dir(),
            "basis/ directory must exist even with empty stage_bases"
        );

        let entries: Vec<_> = std::fs::read_dir(tmp.path().join("basis"))
            .unwrap()
            .filter_map(std::result::Result::ok)
            .collect();
        assert!(
            entries.is_empty(),
            "basis/ must contain no files when stage_bases is empty"
        );
    }

    // Root's read-only permission enforcement is unreliable, so callers that
    // rely on read-only-directory failures skip the test in that case.
    #[cfg(unix)]
    fn is_root() -> bool {
        std::fs::read_to_string("/proc/self/status")
            .ok()
            .and_then(|s| {
                s.lines()
                    .find(|l| l.starts_with("Uid:"))
                    .and_then(|l| l.split_whitespace().nth(2))
                    .and_then(|uid| uid.parse::<u32>().ok())
            })
            == Some(0)
    }

    #[cfg(not(unix))]
    fn is_root() -> bool {
        false
    }

    #[test]
    fn write_policy_checkpoint_error_on_readonly_dir() {
        if is_root() {
            return;
        }

        let tmp = tempfile::tempdir().unwrap();

        // Make the temp directory itself read-only so create_dir_all fails.
        let mut perms = std::fs::metadata(tmp.path()).unwrap().permissions();
        std::os::unix::fs::PermissionsExt::set_mode(&mut perms, 0o555);
        std::fs::set_permissions(tmp.path(), perms).unwrap();

        let readonly_target = tmp.path().join("policy");

        let c0 = [1.0_f64];
        let cuts_s0 = [make_cut_record(1, 0, 1, &c0)];
        let stage_cuts = [make_stage_cuts_payload(0, &cuts_s0, &[0], 1)];
        let metadata = make_metadata(1, 1);

        let result = write_policy_checkpoint(&readonly_target, &stage_cuts, &[], &metadata, &[]);

        // Restore permissions so the tempdir can be cleaned up.
        let mut perms2 = std::fs::metadata(tmp.path()).unwrap().permissions();
        std::os::unix::fs::PermissionsExt::set_mode(&mut perms2, 0o755);
        std::fs::set_permissions(tmp.path(), perms2).unwrap();

        assert!(
            matches!(result, Err(OutputError::IoError { .. })),
            "write_policy_checkpoint must return Err(OutputError::IoError) on read-only dir, got: {result:?}"
        );
    }

    #[test]
    fn write_policy_checkpoint_stage_numbering_zero_padded() {
        let tmp = tempfile::tempdir().unwrap();

        let c0 = [1.0_f64, 2.0];
        let cuts_s0 = [make_cut_record(1, 0, 1, &c0)];
        let cuts_s1 = [make_cut_record(2, 0, 1, &c0)];
        let cuts_s59 = [make_cut_record(3, 0, 1, &c0)];

        let stage_cuts = [
            make_stage_cuts_payload(0, &cuts_s0, &[0], 2),
            make_stage_cuts_payload(1, &cuts_s1, &[0], 2),
            make_stage_cuts_payload(59, &cuts_s59, &[0], 2),
        ];
        let basis_records_0 = PolicyBasisRecord {
            stage_id: 0,
            iteration: 1,
            column_status: &[0u8],
            row_status: &[1u8],
            num_cut_rows: 0,
        };
        let basis_records_1 = PolicyBasisRecord {
            stage_id: 1,
            iteration: 1,
            column_status: &[0u8],
            row_status: &[1u8],
            num_cut_rows: 0,
        };
        let basis_records_59 = PolicyBasisRecord {
            stage_id: 59,
            iteration: 1,
            column_status: &[0u8],
            row_status: &[1u8],
            num_cut_rows: 0,
        };
        let stage_bases = [basis_records_0, basis_records_1, basis_records_59];
        let metadata = make_metadata(3, 2);

        write_policy_checkpoint(tmp.path(), &stage_cuts, &stage_bases, &metadata, &[])
            .expect("write_policy_checkpoint must succeed");

        assert!(
            tmp.path().join("cuts/000.bin").is_file(),
            "cuts/000.bin must exist"
        );
        assert!(
            tmp.path().join("cuts/001.bin").is_file(),
            "cuts/001.bin must exist"
        );
        assert!(
            tmp.path().join("cuts/059.bin").is_file(),
            "cuts/059.bin must exist"
        );
        assert!(
            tmp.path().join("basis/000.bin").is_file(),
            "basis/000.bin must exist"
        );
        assert!(
            tmp.path().join("basis/001.bin").is_file(),
            "basis/001.bin must exist"
        );
        assert!(
            tmp.path().join("basis/059.bin").is_file(),
            "basis/059.bin must exist"
        );
    }

    // ── deserialize_stage_cuts tests ──────────────────────────────────────────

    #[test]
    fn deserialize_stage_cuts_single_cut_all_fields() {
        let coefficients = [1.0_f64, 2.0, 3.0];
        let cut = PolicyCutRecord {
            cut_id: 7,
            slot_index: 5,
            iteration: 3,
            forward_pass_index: 2,
            intercept: 42.0,
            coefficients: &coefficients,
            is_active: true,
        };

        let buf = ser_cuts(0, 3, 100, 0, &[cut], &[0], 1, &[]);
        let result = deserialize_stage_cuts(&buf).expect("deserialization must succeed");

        assert_eq!(result.stage_id, 0, "stage_id must round-trip");
        assert_eq!(result.state_dimension, 3, "state_dimension must round-trip");
        assert_eq!(result.capacity, 100, "capacity must round-trip");
        assert_eq!(
            result.warm_start_count, 0,
            "warm_start_count must round-trip"
        );
        assert_eq!(result.populated_count, 1, "populated_count must round-trip");
        assert_eq!(result.cuts.len(), 1, "one cut must be deserialized");

        let c = &result.cuts[0];
        assert_eq!(c.cut_id, 7, "cut_id must round-trip");
        assert_eq!(c.slot_index, 5, "slot_index must round-trip");
        assert_eq!(c.iteration, 3, "iteration must round-trip");
        assert_eq!(
            c.forward_pass_index, 2,
            "forward_pass_index must round-trip"
        );
        assert_eq!(c.intercept, 42.0, "intercept must round-trip");
        assert_eq!(
            c.coefficients,
            &[1.0, 2.0, 3.0],
            "coefficients must round-trip"
        );
        assert!(c.is_active, "is_active must round-trip");
    }

    #[test]
    fn deserialize_stage_cuts_three_cuts_all_match() {
        let c0 = [1.0_f64, 0.5];
        let c1 = [2.0_f64, 1.5];
        let c2 = [3.0_f64, 2.5];
        let cuts = [
            PolicyCutRecord {
                cut_id: 10,
                slot_index: 0,
                iteration: 1,
                forward_pass_index: 0,
                intercept: 100.0,
                coefficients: &c0,
                is_active: true,
            },
            PolicyCutRecord {
                cut_id: 20,
                slot_index: 1,
                iteration: 2,
                forward_pass_index: 1,
                intercept: 200.0,
                coefficients: &c1,
                is_active: false,
            },
            PolicyCutRecord {
                cut_id: 30,
                slot_index: 2,
                iteration: 3,
                forward_pass_index: 2,
                intercept: 300.0,
                coefficients: &c2,
                is_active: true,
            },
        ];

        let buf = ser_cuts(5, 2, 50, 1, &cuts, &[0, 2], 3, &[]);
        let result = deserialize_stage_cuts(&buf).expect("deserialization must succeed");

        assert_eq!(result.stage_id, 5);
        assert_eq!(result.state_dimension, 2);
        assert_eq!(result.capacity, 50);
        assert_eq!(result.warm_start_count, 1);
        assert_eq!(result.populated_count, 3);
        assert_eq!(result.cuts.len(), 3);

        let expected_cut_ids = [10u64, 20, 30];
        let expected_intercepts = [100.0f64, 200.0, 300.0];
        let expected_coefficients = [&c0[..], &c1[..], &c2[..]];
        let expected_active = [true, false, true];

        for (i, cut) in result.cuts.iter().enumerate() {
            assert_eq!(cut.cut_id, expected_cut_ids[i], "cut {i} cut_id");
            assert_eq!(cut.intercept, expected_intercepts[i], "cut {i} intercept");
            assert_eq!(
                cut.coefficients, expected_coefficients[i],
                "cut {i} coefficients"
            );
            assert_eq!(cut.is_active, expected_active[i], "cut {i} is_active");
        }
    }

    #[test]
    fn deserialize_stage_cuts_empty_cut_pool() {
        let buf = ser_cuts(2, 10, 200, 0, &[], &[], 0, &[]);
        let result =
            deserialize_stage_cuts(&buf).expect("deserialization of empty cut pool must succeed");

        assert_eq!(result.stage_id, 2);
        assert_eq!(result.capacity, 200);
        assert_eq!(result.populated_count, 0);
        assert!(
            result.cuts.is_empty(),
            "empty cut pool must produce zero cuts"
        );
    }

    #[test]
    fn deserialize_stage_cuts_zero_length_coefficients() {
        let cut = PolicyCutRecord {
            cut_id: 1,
            slot_index: 0,
            iteration: 1,
            forward_pass_index: 0,
            intercept: 5.0,
            coefficients: &[],
            is_active: true,
        };
        let buf = ser_cuts(0, 0, 10, 0, &[cut], &[0], 1, &[]);
        let result =
            deserialize_stage_cuts(&buf).expect("zero-length coefficients must deserialize");
        assert_eq!(result.cuts.len(), 1);
        assert!(
            result.cuts[0].coefficients.is_empty(),
            "empty coefficients must round-trip"
        );
    }

    #[test]
    fn deserialize_stage_cuts_large_coefficient_vector() {
        let dim = 1000u32;
        let coefs: Vec<f64> = (0..dim).map(f64::from).collect();
        let cut = PolicyCutRecord {
            cut_id: 42,
            slot_index: 0,
            iteration: 1,
            forward_pass_index: 0,
            intercept: -99.0,
            coefficients: &coefs,
            is_active: false,
        };
        let buf = ser_cuts(3, dim, 10, 0, &[cut], &[0], 1, &[]);
        let result =
            deserialize_stage_cuts(&buf).expect("large coefficient vector must deserialize");
        assert_eq!(result.cuts[0].coefficients.len(), dim as usize);
        assert_eq!(result.cuts[0].coefficients[999], 999.0);
        assert_eq!(result.cuts[0].intercept, -99.0);
    }

    #[test]
    fn deserialize_stage_cuts_truncated_buffer_returns_error() {
        let coefs = [1.0_f64, 2.0];
        let cut = make_cut_record(1, 0, 1, &coefs);
        let full_buf = ser_cuts(0, 2, 10, 0, &[cut], &[0], 1, &[]);
        // Truncate to 2 bytes — root offset itself is incomplete.
        let truncated = &full_buf[..2];
        let result = deserialize_stage_cuts(truncated);
        assert!(result.is_err(), "truncated buffer must return an error");
    }

    #[test]
    fn deserialize_stage_cuts_stage_id_nonzero() {
        let buf = ser_cuts(59, 4, 50, 0, &[], &[], 0, &[]);
        let result = deserialize_stage_cuts(&buf).expect("stage_id=59 must deserialize");
        assert_eq!(result.stage_id, 59, "stage_id=59 must round-trip");
    }

    // ── entity_manifest round-trip tests ──────────────────────────────────────

    fn sample_manifest() -> Vec<EntitySlot> {
        vec![
            EntitySlot::storage(12, true),
            EntitySlot::inflow_lag(-1, 3, false),
            EntitySlot::anticipated(7, 1, true).with_interval(20_240_501, 20_240_601),
        ]
    }

    fn assert_manifest_eq(actual: &[EntitySlot], expected: &[EntitySlot]) {
        assert_eq!(actual.len(), expected.len(), "manifest length must match");
        for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(a.entity_type, e.entity_type, "slot {i} entity_type");
            assert_eq!(a.entity_id, e.entity_id, "slot {i} entity_id");
            assert_eq!(a.subindex, e.subindex, "slot {i} subindex");
            assert_eq!(a.was_active, e.was_active, "slot {i} was_active");
            assert_eq!(
                a.interval_start, e.interval_start,
                "slot {i} interval_start"
            );
        }
    }

    #[test]
    fn stage_cuts_entity_manifest_round_trip() {
        let coefficients = [1.0_f64, 2.0, 3.0];
        let cut = make_cut_record(7, 0, 1, &coefficients);
        let manifest = sample_manifest();

        let buf = ser_cuts(4, 3, 100, 0, &[cut], &[0], 1, &manifest);
        let result = deserialize_stage_cuts(&buf).expect("manifest round-trip must succeed");

        assert_eq!(result.cuts.len(), 1, "cuts must still round-trip");
        assert_manifest_eq(&result.entity_manifest, &manifest);
    }

    #[test]
    fn stage_states_entity_manifest_round_trip() {
        let data = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let manifest = sample_manifest();
        let payload = StageStatesPayload {
            stage_id: 2,
            node_id: 9,
            state_dimension: 3,
            count: 2,
            data: &data,
            entity_manifest: &manifest,
        };

        let buf = serialize_stage_states(&payload);
        let result = deserialize_stage_states(&buf).expect("manifest round-trip must succeed");

        assert_eq!(result.data, data.to_vec(), "data must still round-trip");
        assert_eq!(result.node_id, 9, "node_id must round-trip");
        assert_manifest_eq(&result.entity_manifest, &manifest);
    }

    #[test]
    fn stage_cuts_empty_manifest_deserializes_to_empty_vec() {
        let coefficients = [1.0_f64, 2.0];
        let cut = make_cut_record(1, 0, 1, &coefficients);
        let buf = ser_cuts(0, 2, 10, 0, &[cut], &[0], 1, &[]);
        let result = deserialize_stage_cuts(&buf).expect("empty manifest must deserialize");
        assert!(
            result.entity_manifest.is_empty(),
            "empty manifest must produce zero slots"
        );
    }

    // ── deserialize_stage_basis tests ─────────────────────────────────────────

    #[test]
    fn deserialize_stage_basis_all_fields() {
        let record = PolicyBasisRecord {
            stage_id: 3,
            iteration: 7,
            column_status: &[0, 1, 2, 3],
            row_status: &[1, 0, 1, 0, 1],
            num_cut_rows: 2,
        };

        let buf = serialize_stage_basis(&record);
        let owned = deserialize_stage_basis(&buf).expect("basis round-trip must succeed");

        assert_eq!(owned.stage_id, 3, "stage_id must round-trip");
        assert_eq!(owned.iteration, 7, "iteration must round-trip");
        assert_eq!(
            owned.column_status,
            &[0u8, 1, 2, 3],
            "column_status must round-trip"
        );
        assert_eq!(
            owned.row_status,
            &[1u8, 0, 1, 0, 1],
            "row_status must round-trip"
        );
        assert_eq!(owned.num_cut_rows, 2, "num_cut_rows must round-trip");
    }

    #[test]
    fn deserialize_stage_basis_empty_status_vectors() {
        let record = PolicyBasisRecord {
            stage_id: 0,
            iteration: 0,
            column_status: &[],
            row_status: &[],
            num_cut_rows: 0,
        };

        let buf = serialize_stage_basis(&record);
        let owned = deserialize_stage_basis(&buf).expect("empty basis must deserialize");

        assert!(
            owned.column_status.is_empty(),
            "empty column_status must round-trip"
        );
        assert!(
            owned.row_status.is_empty(),
            "empty row_status must round-trip"
        );
        assert_eq!(owned.num_cut_rows, 0);
    }

    #[test]
    fn deserialize_stage_basis_large_status_vectors() {
        let col: Vec<u8> = (0..200u8).collect();
        let row: Vec<u8> = (0..100u8).rev().collect();
        let record = PolicyBasisRecord {
            stage_id: 10,
            iteration: 99,
            column_status: &col,
            row_status: &row,
            num_cut_rows: 50,
        };

        let buf = serialize_stage_basis(&record);
        let owned = deserialize_stage_basis(&buf).expect("large basis must deserialize");

        assert_eq!(owned.column_status, col);
        assert_eq!(owned.row_status, row);
        assert_eq!(owned.num_cut_rows, 50);
    }

    #[test]
    fn deserialize_stage_basis_truncated_buffer_returns_error() {
        let record = PolicyBasisRecord {
            stage_id: 0,
            iteration: 1,
            column_status: &[0, 1],
            row_status: &[1, 0],
            num_cut_rows: 0,
        };
        let full_buf = serialize_stage_basis(&record);
        let truncated = &full_buf[..3];
        let result = deserialize_stage_basis(truncated);
        assert!(
            result.is_err(),
            "truncated basis buffer must return an error"
        );
    }

    // ── read_policy_checkpoint round-trip tests ───────────────────────────────

    fn assert_no_tmp_files_under(dir: &std::path::Path) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                assert_no_tmp_files_under(&path);
            } else {
                assert_ne!(
                    path.extension().and_then(std::ffi::OsStr::to_str),
                    Some("tmp"),
                    "no .tmp file must remain under {}: found {}",
                    dir.display(),
                    path.display()
                );
            }
        }
    }

    #[test]
    fn read_policy_checkpoint_full_round_trip() {
        let tmp = tempfile::tempdir().unwrap();

        let c0 = [1.0_f64, 2.0, 3.0];
        let c1 = [4.0_f64, 5.0, 6.0];
        let c2 = [7.0_f64, 8.0, 9.0];

        let cuts_s0 = [make_cut_record(1, 0, 1, &c0), make_cut_record(2, 1, 1, &c1)];
        let cuts_s1 = [make_cut_record(3, 0, 2, &c2)];

        let stage_cuts_payloads = [
            make_stage_cuts_payload(0, &cuts_s0, &[0, 1], 3),
            make_stage_cuts_payload(1, &cuts_s1, &[0], 3),
        ];
        let basis_records = [make_basis_record(0), make_basis_record(1)];
        let metadata = make_metadata(2, 3);

        write_policy_checkpoint(
            tmp.path(),
            &stage_cuts_payloads,
            &basis_records,
            &metadata,
            &[],
        )
        .expect("write must succeed");

        assert_no_tmp_files_under(tmp.path());

        let checkpoint = read_policy_checkpoint(tmp.path()).expect("read must succeed");

        assert_eq!(checkpoint.metadata.producer.completed_iterations, 10);
        assert_eq!(checkpoint.metadata.num_stages, 2);
        assert_eq!(checkpoint.metadata.producer.rng_seed, 42);

        assert_eq!(
            checkpoint.stage_cuts.len(),
            2,
            "must have two stage cut results"
        );
        assert_eq!(checkpoint.stage_cuts[0].stage_id, 0);
        assert_eq!(checkpoint.stage_cuts[1].stage_id, 1);
        assert_eq!(checkpoint.stage_cuts[0].cuts.len(), 2);
        assert_eq!(checkpoint.stage_cuts[1].cuts.len(), 1);

        let cut00 = &checkpoint.stage_cuts[0].cuts[0];
        assert_eq!(cut00.cut_id, 1);
        assert_eq!(cut00.coefficients, &[1.0f64, 2.0, 3.0]);
        assert_eq!(cut00.intercept, 42.0);
        assert!(cut00.is_active);

        let cut01 = &checkpoint.stage_cuts[0].cuts[1];
        assert_eq!(cut01.cut_id, 2);
        assert_eq!(cut01.coefficients, &[4.0f64, 5.0, 6.0]);

        let cut10 = &checkpoint.stage_cuts[1].cuts[0];
        assert_eq!(cut10.cut_id, 3);
        assert_eq!(cut10.coefficients, &[7.0f64, 8.0, 9.0]);

        assert_eq!(checkpoint.stage_bases.len(), 2, "must have two stage bases");
        assert_eq!(checkpoint.stage_bases[0].stage_id, 0);
        assert_eq!(checkpoint.stage_bases[1].stage_id, 1);
        assert_eq!(checkpoint.stage_bases[0].column_status, &[0u8, 1, 2, 3]);
        assert_eq!(checkpoint.stage_bases[0].row_status, &[1u8, 0, 1, 0, 1]);
        assert_eq!(checkpoint.stage_bases[0].num_cut_rows, 2);
    }

    #[test]
    fn read_policy_checkpoint_no_bases_empty_stage_bases() {
        let tmp = tempfile::tempdir().unwrap();

        let c0 = [1.0_f64];
        let cuts_s0 = [make_cut_record(1, 0, 1, &c0)];
        let stage_cuts_payloads = [make_stage_cuts_payload(0, &cuts_s0, &[0], 1)];
        let metadata = make_metadata(1, 1);

        write_policy_checkpoint(tmp.path(), &stage_cuts_payloads, &[], &metadata, &[])
            .expect("write must succeed");

        let checkpoint = read_policy_checkpoint(tmp.path()).expect("read must succeed");

        assert_eq!(checkpoint.stage_cuts.len(), 1);
        assert!(
            checkpoint.stage_bases.is_empty(),
            "no basis files must produce empty stage_bases"
        );
    }

    #[test]
    fn rewrite_over_existing_checkpoint_reads_back_the_new_artifact() {
        let tmp = tempfile::tempdir().unwrap();

        let a0 = [1.0_f64, 2.0, 3.0];
        let piece_a = PolicyCutRecord {
            intercept: 11.0,
            ..make_cut_record(101, 0, 1, &a0)
        };
        let cuts_a = [piece_a];
        let stage_cuts_a = [make_stage_cuts_payload(0, &cuts_a, &[0], 3)];
        let basis_a = [make_basis_record(0)];
        let metadata_a = make_metadata(1, 3);

        write_policy_checkpoint(tmp.path(), &stage_cuts_a, &basis_a, &metadata_a, &[])
            .expect("write of checkpoint A must succeed");

        let b0 = [40.0_f64, 50.0, 60.0];
        let piece_b = PolicyCutRecord {
            intercept: 99.0,
            ..make_cut_record(202, 0, 5, &b0)
        };
        let cuts_b = [piece_b];
        let stage_cuts_b = [make_stage_cuts_payload(0, &cuts_b, &[0], 3)];
        let basis_b = [make_basis_record(0)];
        let metadata_b = make_metadata(1, 3);

        write_policy_checkpoint(tmp.path(), &stage_cuts_b, &basis_b, &metadata_b, &[])
            .expect("rewrite with checkpoint B must succeed");

        assert_no_tmp_files_under(tmp.path());

        let checkpoint = read_policy_checkpoint(tmp.path())
            .expect("read of the rewritten checkpoint must succeed");

        assert_eq!(
            checkpoint.stage_cuts.len(),
            1,
            "rewrite must not accumulate stale pools from A"
        );
        assert_eq!(
            checkpoint.stage_cuts[0].stage_id, 0,
            "must read back B's stage id"
        );
        assert_eq!(checkpoint.stage_cuts[0].cuts.len(), 1);

        let cut = &checkpoint.stage_cuts[0].cuts[0];
        assert_eq!(cut.cut_id, 202, "must read back B's cut id, not A's");
        assert_ne!(cut.cut_id, 101, "A's cut id must not survive the rewrite");
        assert_eq!(
            cut.coefficients,
            &[40.0f64, 50.0, 60.0],
            "must read back B's coefficients, not A's"
        );
        assert_eq!(cut.intercept, 99.0, "must read back B's intercept, not A's");
        assert_ne!(
            cut.intercept, 11.0,
            "A's intercept must not survive the rewrite"
        );

        assert_eq!(checkpoint.stage_bases.len(), 1);
        assert_eq!(checkpoint.stage_bases[0].stage_id, 0);
    }

    /// Run A leaves two pools, two bases and exported states; run B, into the
    /// same directory, has one pool, one basis and no states. The reader lists
    /// the payload directories, so anything A left behind would be read back as
    /// B's — and B's terminal pool would then be A's stale pool 1.
    #[test]
    fn rewrite_with_fewer_pools_and_no_states_leaves_no_stale_payloads() {
        let tmp = tempfile::tempdir().unwrap();

        let a0 = [1.0_f64, 2.0, 3.0];
        let cuts_a = [make_cut_record(101, 0, 1, &a0)];
        let stage_cuts_a = [
            make_stage_cuts_payload(0, &cuts_a, &[0], 3),
            make_stage_cuts_payload(1, &cuts_a, &[0], 3),
        ];
        let basis_a = [make_basis_record(0), make_basis_record(1)];
        let states_data = [1.0_f64, 2.0, 3.0];
        let states_manifest = sample_manifest();
        let states_a = [StageStatesPayload {
            stage_id: 0,
            node_id: 0,
            state_dimension: 3,
            count: 1,
            data: &states_data,
            entity_manifest: &states_manifest,
        }];
        write_policy_checkpoint(
            tmp.path(),
            &stage_cuts_a,
            &basis_a,
            &make_metadata(2, 3),
            &states_a,
        )
        .expect("write of checkpoint A must succeed");
        assert!(tmp.path().join("states").is_dir());

        let b0 = [40.0_f64, 50.0, 60.0];
        let cuts_b = [make_cut_record(202, 0, 5, &b0)];
        let stage_cuts_b = [make_stage_cuts_payload(0, &cuts_b, &[0], 3)];
        let basis_b = [make_basis_record(0)];
        write_policy_checkpoint(
            tmp.path(),
            &stage_cuts_b,
            &basis_b,
            &make_metadata(1, 3),
            &[],
        )
        .expect("rewrite with checkpoint B must succeed");

        assert_no_tmp_files_under(tmp.path());
        assert!(
            !tmp.path().join("cuts").join("001.bin").exists(),
            "A's second pool must not survive the rewrite"
        );
        assert!(
            !tmp.path().join("basis").join("001.bin").exists(),
            "A's second basis must not survive the rewrite"
        );
        assert!(
            !tmp.path().join("states").exists(),
            "A's states directory must not survive a rewrite that exports no states"
        );

        let checkpoint = read_policy_checkpoint(tmp.path())
            .expect("read of the rewritten checkpoint must succeed");
        assert_eq!(checkpoint.stage_cuts.len(), 1);
        assert_eq!(checkpoint.stage_cuts[0].cuts[0].cut_id, 202);
        assert_eq!(checkpoint.stage_bases.len(), 1);
        assert!(checkpoint.stage_states.is_empty());
    }

    #[test]
    fn rewrite_with_fewer_states_payloads_leaves_no_stale_states() {
        let tmp = tempfile::tempdir().unwrap();
        let a0 = [1.0_f64, 2.0, 3.0];
        let cuts = [make_cut_record(101, 0, 1, &a0)];
        let stage_cuts = [make_stage_cuts_payload(0, &cuts, &[0], 3)];
        let states_data = [1.0_f64, 2.0, 3.0];
        let states_manifest = sample_manifest();
        let states_for = |stage_id: u32| StageStatesPayload {
            stage_id,
            node_id: i32::try_from(stage_id).unwrap(),
            state_dimension: 3,
            count: 1,
            data: &states_data,
            entity_manifest: &states_manifest,
        };

        let states_a = [states_for(0), states_for(1)];
        write_policy_checkpoint(
            tmp.path(),
            &stage_cuts,
            &[],
            &make_metadata(1, 3),
            &states_a,
        )
        .expect("write of checkpoint A must succeed");

        let states_b = [states_for(0)];
        write_policy_checkpoint(
            tmp.path(),
            &stage_cuts,
            &[],
            &make_metadata(1, 3),
            &states_b,
        )
        .expect("rewrite with checkpoint B must succeed");

        assert!(
            !tmp.path().join("states").join("001.bin").exists(),
            "A's second states payload must not survive the rewrite"
        );
        let checkpoint = read_policy_checkpoint(tmp.path()).expect("read must succeed");
        assert_eq!(checkpoint.stage_states.len(), 1);
        assert_eq!(checkpoint.stage_states[0].stage_id, 0);
    }

    #[cfg(unix)]
    #[test]
    fn interrupted_rewrite_keeps_a_loadable_checkpoint() {
        use std::os::unix::fs::PermissionsExt;
        if is_root() {
            return;
        }

        let tmp = tempfile::tempdir().unwrap();
        let policy = tmp.path().join("policy");
        write_copy(&policy, OLDER);
        // A read-only `cuts/` in A lets the swap finish but stops its removal.
        std::fs::set_permissions(policy.join("cuts"), std::fs::Permissions::from_mode(0o555))
            .unwrap();

        let result = try_write_copy(&policy, NEWER);

        let previous = tmp.path().join("policy.previous");
        std::fs::set_permissions(
            previous.join("cuts"),
            std::fs::Permissions::from_mode(0o755),
        )
        .unwrap();
        assert!(
            matches!(&result, Err(OutputError::IoError { path, .. }) if *path == previous),
            "the removal of the replaced copy must fail with IoError, got {result:?}"
        );
        assert_eq!(read_iterations(&policy).unwrap(), NEWER);
    }

    #[test]
    fn read_policy_checkpoint_missing_manifest_returns_error() {
        let tmp = tempfile::tempdir().unwrap();
        // A pre-manifest.bin artifact carries no manifest.bin — the clean-break
        // reader errors instead of falling back to any legacy carrier.
        let result = read_policy_checkpoint(tmp.path());
        assert!(result.is_err(), "missing manifest.bin must return an error");
        assert!(
            matches!(result, Err(OutputError::IoError { .. })),
            "error must be IoError for missing manifest.bin"
        );
    }

    #[test]
    fn read_policy_checkpoint_stages_sorted_by_id() {
        let tmp = tempfile::tempdir().unwrap();

        // Write stages in non-ascending order — reader must sort.
        let c = [1.0_f64, 2.0];
        let cuts2 = [make_cut_record(1, 0, 1, &c)];
        let cuts0 = [make_cut_record(2, 0, 1, &c)];
        let cuts1 = [make_cut_record(3, 0, 1, &c)];

        let stage_cuts_payloads = [
            make_stage_cuts_payload(2, &cuts2, &[0], 2),
            make_stage_cuts_payload(0, &cuts0, &[0], 2),
            make_stage_cuts_payload(1, &cuts1, &[0], 2),
        ];
        let metadata = make_metadata(3, 2);

        write_policy_checkpoint(tmp.path(), &stage_cuts_payloads, &[], &metadata, &[])
            .expect("write must succeed");

        let checkpoint = read_policy_checkpoint(tmp.path()).expect("read must succeed");

        assert_eq!(checkpoint.stage_cuts.len(), 3);
        assert_eq!(
            checkpoint.stage_cuts[0].stage_id, 0,
            "first result must be stage 0"
        );
        assert_eq!(
            checkpoint.stage_cuts[1].stage_id, 1,
            "second result must be stage 1"
        );
        assert_eq!(
            checkpoint.stage_cuts[2].stage_id, 2,
            "third result must be stage 2"
        );
    }

    #[test]
    fn read_policy_checkpoint_metadata_field_by_field() {
        let tmp = tempfile::tempdir().unwrap();

        let mut meta_in = make_metadata(4, 8);
        meta_in.created_at = "2026-01-01T00:00:00Z".to_string();
        meta_in.producer.completed_iterations = 77;
        meta_in.producer.final_lower_bound = 12345.678;
        meta_in.producer.best_upper_bound = Some(13000.0);
        meta_in.producer.max_iterations = 500;
        meta_in.producer.forward_passes = 8;
        meta_in.producer.warm_start_cuts = 20;
        meta_in.producer.warm_start_counts = vec![20; 4];
        meta_in.producer.rng_seed = 99999;

        let stage_cuts_payloads: [StageCutsPayload<'_>; 0] = [];
        write_policy_checkpoint(tmp.path(), &stage_cuts_payloads, &[], &meta_in, &[])
            .expect("write must succeed");

        let checkpoint = read_policy_checkpoint(tmp.path()).expect("read must succeed");
        let m = &checkpoint.metadata;

        assert_eq!(m.format_version, FORMAT_VERSION);
        assert_eq!(m.producer.completed_iterations, 77);
        assert_eq!(m.producer.final_lower_bound, 12345.678);
        assert_eq!(m.producer.best_upper_bound, Some(13000.0));
        assert_eq!(m.num_stages, 4);
        assert_eq!(m.producer.max_iterations, 500);
        assert_eq!(m.producer.forward_passes, 8);
        assert_eq!(m.producer.warm_start_cuts, 20);
        assert_eq!(m.producer.warm_start_counts, vec![20u32; 4]);
        assert_eq!(m.producer.rng_seed, 99999);
    }

    #[test]
    fn read_policy_checkpoint_warm_start_counts_in_metadata() {
        let tmp = tempfile::tempdir().unwrap();

        let c0 = [1.0_f64, 2.0];
        let c1 = [3.0_f64, 4.0];
        let c2 = [5.0_f64, 6.0];
        let cuts_s0 = [make_cut_record(1, 0, 1, &c0), make_cut_record(2, 1, 1, &c0)];
        let cuts_s1 = [
            make_cut_record(3, 0, 2, &c1),
            make_cut_record(4, 1, 2, &c1),
            make_cut_record(5, 2, 2, &c1),
        ];
        let cuts_s2 = [
            make_cut_record(6, 0, 3, &c2),
            make_cut_record(7, 1, 3, &c2),
            make_cut_record(8, 2, 3, &c2),
            make_cut_record(9, 3, 3, &c2),
        ];

        let stage_cuts_payloads = [
            StageCutsPayload {
                stage_id: 0,
                state_dimension: 2,
                capacity: 100,
                warm_start_count: 10,
                cuts: &cuts_s0,
                active_cut_indices: &[0, 1],
                populated_count: 2,
                entity_manifest: &[],
                cost_scale_factor: 1_000_000.0,
                node_id: -1,
                graph_stage_id: -1,
                priced_state_date: STAGE_CUTS_PRICED_STATE_DATE_SENTINEL,
            },
            StageCutsPayload {
                stage_id: 1,
                state_dimension: 2,
                capacity: 100,
                warm_start_count: 8,
                cuts: &cuts_s1,
                active_cut_indices: &[0, 1, 2],
                populated_count: 3,
                entity_manifest: &[],
                cost_scale_factor: 1_000_000.0,
                node_id: -1,
                graph_stage_id: -1,
                priced_state_date: STAGE_CUTS_PRICED_STATE_DATE_SENTINEL,
            },
            StageCutsPayload {
                stage_id: 2,
                state_dimension: 2,
                capacity: 100,
                warm_start_count: 6,
                cuts: &cuts_s2,
                active_cut_indices: &[0, 1, 2, 3],
                populated_count: 4,
                entity_manifest: &[],
                cost_scale_factor: 1_000_000.0,
                node_id: -1,
                graph_stage_id: -1,
                priced_state_date: STAGE_CUTS_PRICED_STATE_DATE_SENTINEL,
            },
        ];

        let mut metadata = make_metadata(3, 2);
        metadata.producer.warm_start_counts = vec![10, 8, 6];

        write_policy_checkpoint(tmp.path(), &stage_cuts_payloads, &[], &metadata, &[])
            .expect("write must succeed");

        let checkpoint = read_policy_checkpoint(tmp.path()).expect("read must succeed");

        assert_eq!(
            checkpoint.metadata.producer.warm_start_counts,
            vec![10u32, 8, 6],
            "warm_start_counts [10, 8, 6] must round-trip through manifest.bin"
        );
    }

    // ── 0.14 format-wave markers ──────────────────────────────────────────────

    /// A manifest whose `format_version` is not [`FORMAT_VERSION`] is rejected on
    /// the version marker BEFORE any payload is parsed: the reader names
    /// `format_version`, not the deliberately-corrupt `cuts/000.bin`.
    #[test]
    fn read_policy_checkpoint_rejects_stale_manifest_version_before_parsing_payloads() {
        let tmp = tempfile::tempdir().unwrap();
        let c0 = [1.0_f64];
        let cuts_s0 = [make_cut_record(1, 0, 1, &c0)];
        let stage_cuts = [make_stage_cuts_payload(0, &cuts_s0, &[0], 1)];
        write_policy_checkpoint(tmp.path(), &stage_cuts, &[], &make_metadata(1, 1), &[])
            .expect("write must succeed");

        let mut stale = make_metadata(1, 1);
        stale.format_version = FORMAT_VERSION + 1;
        std::fs::write(
            tmp.path().join("manifest.bin"),
            super::codec::serialize_checkpoint_manifest(&stale),
        )
        .unwrap();
        std::fs::write(tmp.path().join("cuts/000.bin"), b"garbage").unwrap();

        let err =
            read_policy_checkpoint(tmp.path()).expect_err("stale manifest version must reject");
        let msg = err.to_string();
        assert!(
            msg.contains("format_version"),
            "must reject on the version marker, not the corrupt payload: {msg}"
        );
    }

    /// The older-version mirror of the test above: a manifest predating
    /// [`FORMAT_VERSION`] is rejected on the same version marker BEFORE any
    /// payload is parsed, naming both versions and ending with the shared remedy.
    #[test]
    fn read_policy_checkpoint_rejects_older_manifest_version() {
        let tmp = tempfile::tempdir().unwrap();
        let c0 = [1.0_f64];
        let cuts_s0 = [make_cut_record(1, 0, 1, &c0)];
        let stage_cuts = [make_stage_cuts_payload(0, &cuts_s0, &[0], 1)];
        write_policy_checkpoint(tmp.path(), &stage_cuts, &[], &make_metadata(1, 1), &[])
            .expect("write must succeed");

        let mut older = make_metadata(1, 1);
        older.format_version = FORMAT_VERSION - 1;
        std::fs::write(
            tmp.path().join("manifest.bin"),
            super::codec::serialize_checkpoint_manifest(&older),
        )
        .unwrap();
        std::fs::write(tmp.path().join("cuts/000.bin"), b"garbage").unwrap();

        let err =
            read_policy_checkpoint(tmp.path()).expect_err("older manifest version must reject");
        let msg = err.to_string();
        assert!(
            msg.contains("format_version")
                && msg.contains(&(FORMAT_VERSION - 1).to_string())
                && msg.contains(&FORMAT_VERSION.to_string())
                && msg.ends_with(&policy_checkpoint_remedy()),
            "must name both versions and end with the shared remedy, not surface the corrupt payload: {msg}"
        );
    }

    /// The positional `stage_NNN` filename convention is gone: no
    /// `format!("stage_{...}.bin", ...)` writer literal survives, and the reader
    /// derives identity from inside each buffer via `read_sorted_bin_files`, not
    /// from a filename. A surviving positional reader is the exact path that
    /// would read a 0.13 file as pool 0.
    #[test]
    fn no_stage_positional_filename_convention_remains() {
        let checkpoint_src = include_str!("checkpoint.rs");
        assert!(
            !checkpoint_src.contains("stage_{"),
            "no `format!(\"stage_{{...}}.bin\", ...)` writer literal may remain"
        );
        assert!(
            checkpoint_src.contains("read_sorted_bin_files"),
            "the reader must stay filename-agnostic (sorts by the in-buffer id)"
        );
    }

    /// K-fan (K = 4) whose leaves share one pool: the writer emits ONE payload
    /// for the shared leaf pool (not K), and the graph manifest maps all K leaf
    /// node ids to that one pool id.
    #[test]
    fn k_fan_shared_leaf_pool_serialized_once_and_manifest_maps_leaves() {
        let tmp = tempfile::tempdir().unwrap();
        let k = 4u32;

        // One payload per pool: pool 0 (root, stage 0) and pool 1 (shared leaf pool).
        let c = [1.0_f64];
        let cuts_p0 = [make_cut_record(1, 0, 1, &c)];
        let cuts_p1 = [make_cut_record(2, 0, 1, &c)];
        let stage_cuts = [
            make_stage_cuts_payload(0, &cuts_p0, &[0], 1),
            make_stage_cuts_payload(1, &cuts_p1, &[0], 1),
        ];

        let mut nodes = vec![ManifestNode {
            id: 0,
            stage_id: 0,
            pool_id: 0,
        }];
        for leaf in 1..=k {
            nodes.push(ManifestNode {
                id: i32::try_from(leaf).unwrap(),
                stage_id: 1,
                pool_id: 1,
            });
        }
        let edges = (1..=k)
            .map(|leaf| ManifestEdge {
                source_id: 0,
                target_id: i32::try_from(leaf).unwrap(),
                probability: 1.0 / f64::from(k),
            })
            .collect();
        let mut metadata = make_metadata(2, 1);
        metadata.graph_manifest = GraphManifest {
            n_pools: 2,
            nodes,
            edges,
        };

        write_policy_checkpoint(tmp.path(), &stage_cuts, &[], &metadata, &[])
            .expect("write must succeed");

        let bin_count = std::fs::read_dir(tmp.path().join("cuts"))
            .unwrap()
            .filter_map(std::result::Result::ok)
            .filter(|e| e.path().extension().is_some_and(|x| x == "bin"))
            .count();
        assert_eq!(
            bin_count, 2,
            "the shared leaf pool is serialized once (2 pool files), not K times"
        );

        let checkpoint = read_policy_checkpoint(tmp.path()).expect("read must succeed");
        let leaf_pools: std::collections::BTreeSet<u32> = checkpoint
            .metadata
            .graph_manifest
            .nodes
            .iter()
            .filter(|n| n.stage_id == 1)
            .map(|n| n.pool_id)
            .collect();
        assert_eq!(
            leaf_pools.len(),
            1,
            "all K leaf node ids must map to one pool id: {leaf_pools:?}"
        );
        assert_eq!(*leaf_pools.iter().next().unwrap(), 1);
        assert_eq!(
            checkpoint
                .metadata
                .graph_manifest
                .nodes
                .iter()
                .filter(|n| n.stage_id == 1)
                .count(),
            k as usize,
            "all K leaf nodes must be present in the manifest"
        );
    }

    // ── checkpoint resolution tests ───────────────────────────────────────────

    const OLDER: u32 = 1;
    const NEWER: u32 = 2;
    const NEWEST: u32 = 3;

    fn try_write_copy(dir: &Path, completed_iterations: u32) -> Result<(), OutputError> {
        let coefficients = [1.0_f64];
        let cuts = [make_cut_record(1, 0, 1, &coefficients)];
        let stage_cuts = [make_stage_cuts_payload(0, &cuts, &[0], 1)];
        let mut metadata = make_metadata(1, 1);
        metadata.producer.completed_iterations = completed_iterations;
        write_policy_checkpoint(dir, &stage_cuts, &[make_basis_record(0)], &metadata, &[])
    }

    fn write_copy(dir: &Path, completed_iterations: u32) {
        try_write_copy(dir, completed_iterations).unwrap();
    }

    fn read_iterations(path: &Path) -> Result<u32, OutputError> {
        read_policy_checkpoint(path).map(|c| c.metadata.producer.completed_iterations)
    }

    /// What a run killed while committing a checkpoint can leave beside it.
    #[derive(Debug, Clone, Copy)]
    enum CrashState {
        StagingCutOff,
        StagingComplete,
        BetweenRenames,
        AfterRenames,
        PreviousPartlyRemoved,
        StagingWithoutManifest,
    }

    const CRASH_STATES: [CrashState; 6] = [
        CrashState::StagingCutOff,
        CrashState::StagingComplete,
        CrashState::BetweenRenames,
        CrashState::AfterRenames,
        CrashState::PreviousPartlyRemoved,
        CrashState::StagingWithoutManifest,
    ];

    impl CrashState {
        /// The directory a reader must resolve, and its `completed_iterations`.
        fn resolved_copy(self) -> (&'static str, u32) {
            match self {
                Self::StagingCutOff | Self::StagingComplete => ("policy", OLDER),
                Self::BetweenRenames => ("policy.staging", NEWER),
                Self::AfterRenames | Self::PreviousPartlyRemoved => ("policy", NEWER),
                Self::StagingWithoutManifest => ("policy.previous", OLDER),
            }
        }
    }

    /// Build `state` at `root/policy` from copies written under `root/scratch`
    /// and renamed into place, and return `root/policy`.
    fn build_crash_state(root: &Path, state: CrashState) -> PathBuf {
        let policy = root.join("policy");
        let staging = root.join("policy.staging");
        let previous = root.join("policy.previous");
        let older = root.join("scratch/older");
        let newer = root.join("scratch/newer");
        write_copy(&older, OLDER);
        write_copy(&newer, NEWER);
        let (older_at, newer_at) = match state {
            CrashState::StagingCutOff => {
                std::fs::rename(newer.join("manifest.bin"), newer.join("manifest.bin.tmp"))
                    .unwrap();
                (&policy, &staging)
            }
            CrashState::StagingComplete => (&policy, &staging),
            CrashState::BetweenRenames => (&previous, &staging),
            CrashState::AfterRenames => (&previous, &policy),
            CrashState::PreviousPartlyRemoved => {
                std::fs::remove_dir_all(older.join("cuts")).unwrap();
                (&previous, &policy)
            }
            CrashState::StagingWithoutManifest => {
                std::fs::remove_file(newer.join("manifest.bin")).unwrap();
                (&previous, &staging)
            }
        };
        std::fs::rename(&older, older_at).unwrap();
        std::fs::rename(&newer, newer_at).unwrap();
        policy
    }

    #[derive(Debug, PartialEq, Eq)]
    enum SnapshotEntry {
        Dir,
        File(Vec<u8>),
        Link(PathBuf),
    }

    /// Every entry under `root` by relative path, recording a link's target
    /// without following it.
    fn tree_snapshot(root: &Path) -> BTreeMap<PathBuf, SnapshotEntry> {
        let mut snapshot = BTreeMap::new();
        let mut pending = vec![root.to_path_buf()];
        while let Some(dir) = pending.pop() {
            for entry in std::fs::read_dir(&dir).unwrap() {
                let path = entry.unwrap().path();
                let file_type = std::fs::symlink_metadata(&path).unwrap().file_type();
                let recorded = if file_type.is_symlink() {
                    SnapshotEntry::Link(std::fs::read_link(&path).unwrap())
                } else if file_type.is_dir() {
                    pending.push(path.clone());
                    SnapshotEntry::Dir
                } else {
                    SnapshotEntry::File(std::fs::read(&path).unwrap())
                };
                snapshot.insert(path.strip_prefix(root).unwrap().to_path_buf(), recorded);
            }
        }
        snapshot
    }

    /// Resolve and read `policy` in `state`, and describe every way the answer
    /// differs from `state`'s copy (`is_copy` judges a resolved directory) or
    /// the disk under `root` changed.
    fn crash_state_failures(
        state: CrashState,
        root: &Path,
        policy: &Path,
        is_copy: impl Fn(&Path) -> bool,
    ) -> Vec<String> {
        let (_, completed_iterations) = state.resolved_copy();
        let before = tree_snapshot(root);
        let resolved = resolve_policy_checkpoint(policy);
        let read = read_policy_checkpoint(policy).map(|c| c.metadata.producer.completed_iterations);
        let mut failures = Vec::new();
        if !matches!(&resolved, Ok(ResolvedCheckpoint::Found(dir)) if is_copy(dir)) {
            failures.push(format!("{state:?}: resolved {resolved:?}"));
        }
        if !matches!(&read, Ok(n) if *n == completed_iterations) {
            failures.push(format!(
                "{state:?}: read {read:?}, expected {completed_iterations}"
            ));
        }
        if tree_snapshot(root) != before {
            failures.push(format!("{state:?}: the disk changed"));
        }
        failures
    }

    #[test]
    fn readers_resolve_one_copy_per_crash_state_without_changing_the_disk() {
        let failures: Vec<String> = CRASH_STATES
            .into_iter()
            .flat_map(|state| {
                let tmp = tempfile::tempdir().unwrap();
                let policy = build_crash_state(tmp.path(), state);
                let copy = tmp.path().join(state.resolved_copy().0);
                crash_state_failures(state, tmp.path(), &policy, |dir| dir == copy)
            })
            .collect();
        assert!(failures.is_empty(), "{failures:#?}");
    }

    /// Link `root/out/policy` to `../real/policy` and return the link.
    #[cfg(unix)]
    fn link_to_real_policy(root: &Path) -> PathBuf {
        std::fs::create_dir_all(root.join("real")).unwrap();
        std::fs::create_dir(root.join("out")).unwrap();
        let link = root.join("out/policy");
        std::os::unix::fs::symlink("../real/policy", &link).unwrap();
        link
    }

    #[cfg(unix)]
    #[test]
    fn readers_resolve_one_copy_per_crash_state_through_a_linked_policy_directory() {
        let failures: Vec<String> = CRASH_STATES
            .into_iter()
            .flat_map(|state| {
                let tmp = tempfile::tempdir().unwrap();
                build_crash_state(&tmp.path().join("real"), state);
                let link = link_to_real_policy(tmp.path());
                let copy =
                    std::fs::canonicalize(tmp.path().join("real").join(state.resolved_copy().0))
                        .unwrap();
                crash_state_failures(state, tmp.path(), &link, |dir| {
                    std::fs::canonicalize(dir).is_ok_and(|dir| dir == copy)
                })
            })
            .collect();
        assert!(failures.is_empty(), "{failures:#?}");

        let tmp = tempfile::tempdir().unwrap();
        let link = link_to_real_policy(tmp.path());
        let resolve_without_changing_the_disk = || {
            let before = tree_snapshot(tmp.path());
            let resolved = resolve_policy_checkpoint(&link).unwrap();
            assert_eq!(tree_snapshot(tmp.path()), before);
            resolved
        };
        assert_eq!(
            resolve_without_changing_the_disk(),
            ResolvedCheckpoint::NoDirectory
        );
        std::fs::create_dir(tmp.path().join("real/policy")).unwrap();
        assert_eq!(
            resolve_without_changing_the_disk(),
            ResolvedCheckpoint::NoManifest
        );
    }

    #[cfg(unix)]
    #[test]
    fn reads_a_staged_copy_from_a_read_only_directory() {
        use std::os::unix::fs::PermissionsExt;
        if is_root() {
            return;
        }

        let tmp = tempfile::tempdir().unwrap();
        let policy = build_crash_state(tmp.path(), CrashState::BetweenRenames);
        let read_only = [
            tmp.path().to_path_buf(),
            tmp.path().join("policy.staging"),
            tmp.path().join("policy.previous"),
        ];
        for dir in &read_only {
            std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o555)).unwrap();
        }

        let read =
            read_policy_checkpoint(&policy).map(|c| c.metadata.producer.completed_iterations);

        for dir in &read_only {
            std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        assert_eq!(read.unwrap(), NEWER);
    }

    #[test]
    fn resolver_tells_a_missing_directory_from_a_missing_manifest() {
        let tmp = tempfile::tempdir().unwrap();
        let policy = tmp.path().join("policy");
        assert_eq!(
            resolve_policy_checkpoint(&policy).unwrap(),
            ResolvedCheckpoint::NoDirectory
        );

        std::fs::create_dir(tmp.path().join("policy.staging")).unwrap();
        assert_eq!(
            resolve_policy_checkpoint(&policy).unwrap(),
            ResolvedCheckpoint::NoDirectory
        );

        std::fs::create_dir(&policy).unwrap();
        assert_eq!(
            resolve_policy_checkpoint(&policy).unwrap(),
            ResolvedCheckpoint::NoManifest
        );

        assert_eq!(
            resolve_policy_checkpoint(Path::new("/nonexistent-root-for-test/..")).unwrap(),
            ResolvedCheckpoint::NoDirectory
        );
    }

    #[cfg(unix)]
    #[test]
    fn resolver_reports_an_unreadable_candidate_as_an_io_error() {
        use std::os::unix::fs::PermissionsExt;
        if is_root() {
            return;
        }

        let tmp = tempfile::tempdir().unwrap();
        let policy = tmp.path().join("policy");
        std::fs::create_dir(&policy).unwrap();
        std::fs::set_permissions(&policy, std::fs::Permissions::from_mode(0o000)).unwrap();

        let resolved = resolve_policy_checkpoint(&policy);

        std::fs::set_permissions(&policy, std::fs::Permissions::from_mode(0o755)).unwrap();
        assert!(
            matches!(
                &resolved,
                Err(OutputError::IoError { path, source })
                    if path.ends_with("policy/manifest.bin")
                        && source.kind() == std::io::ErrorKind::PermissionDenied
            ),
            "expected a permission IoError on policy/manifest.bin, got {resolved:?}"
        );
    }

    // ── checkpoint commit tests ───────────────────────────────────────────────

    fn refused_entry(result: &Result<(), OutputError>) -> Option<(&Path, &Path)> {
        match result {
            Err(OutputError::ForeignEntry { dir, entry }) => Some((dir, entry)),
            _ => None,
        }
    }

    fn leftover_siblings(policy: &Path) -> Vec<PathBuf> {
        [".staging", ".previous"]
            .into_iter()
            .map(|suffix| {
                let mut name = policy.as_os_str().to_os_string();
                name.push(suffix);
                PathBuf::from(name)
            })
            .filter(|sibling| std::fs::symlink_metadata(sibling).is_ok())
            .collect()
    }

    #[test]
    fn interrupted_swap_completion_keeps_the_copy_readers_resolve() {
        let failures: Vec<String> = CRASH_STATES
            .into_iter()
            .flat_map(|state| {
                let tmp = tempfile::tempdir().unwrap();
                let policy = build_crash_state(tmp.path(), state);
                let resolved = resolve_policy_checkpoint(&policy);
                let Ok(ResolvedCheckpoint::Found(copy)) = resolved else {
                    return vec![format!("{state:?}: resolved {resolved:?}")];
                };
                let completed_iterations = read_iterations(&copy).unwrap();

                let mut failures = Vec::new();
                if let Err(e) = finish_interrupted_swap(&policy) {
                    failures.push(format!("{state:?}: finish {e:?}"));
                }
                match read_iterations(&policy) {
                    Ok(n) if n == completed_iterations => {}
                    read => failures.push(format!(
                        "{state:?}: read {read:?}, expected {completed_iterations}"
                    )),
                }
                match resolve_policy_checkpoint(&policy) {
                    Ok(ResolvedCheckpoint::Found(dir)) if dir == policy => {}
                    resolved => failures.push(format!("{state:?}: resolved {resolved:?}")),
                }
                let siblings = leftover_siblings(&policy);
                if !siblings.is_empty() {
                    failures.push(format!("{state:?}: finish left {siblings:?}"));
                }

                if let Err(e) = try_write_copy(&policy, NEWEST) {
                    failures.push(format!("{state:?}: write {e:?}"));
                }
                match read_iterations(&policy) {
                    Ok(NEWEST) => {}
                    read => failures.push(format!("{state:?}: read {read:?}, expected {NEWEST}")),
                }
                let siblings = leftover_siblings(&policy);
                if !siblings.is_empty() {
                    failures.push(format!("{state:?}: write left {siblings:?}"));
                }
                failures
            })
            .collect();
        assert!(failures.is_empty(), "{failures:#?}");
    }

    #[test]
    fn first_write_creates_the_directory_and_leaves_no_siblings() {
        let tmp = tempfile::tempdir().unwrap();
        let policy = tmp.path().join("out/policy");

        try_write_copy(&policy, OLDER).unwrap();

        assert_eq!(read_iterations(&policy).unwrap(), OLDER);
        assert_eq!(leftover_siblings(&policy), Vec::<PathBuf>::new());
        let mut entries: Vec<_> = std::fs::read_dir(&policy)
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect();
        entries.sort();
        assert_eq!(entries, ["basis", "cuts", "manifest.bin"]);
    }

    #[test]
    fn write_refuses_a_path_without_a_file_name() {
        let tmp = tempfile::tempdir().unwrap();
        let before = tree_snapshot(tmp.path());

        let result = try_write_copy(&tmp.path().join("a/.."), OLDER);

        assert!(
            matches!(
                &result,
                Err(OutputError::IoError { source, .. })
                    if source.kind() == std::io::ErrorKind::InvalidInput
            ),
            "a path without a file name must fail with InvalidInput, got {result:?}"
        );
        assert_eq!(tree_snapshot(tmp.path()), before);
    }

    #[test]
    fn rewrite_refuses_a_leftover_sibling_holding_an_unrecognized_entry() {
        let cases = [
            ("policy.staging", true, "notes.txt"),
            ("policy.previous", true, "cuts/README"),
            ("policy.previous", false, "notes.txt"),
        ];

        let failures: Vec<String> = cases
            .into_iter()
            .flat_map(|(sibling, with_policy, name)| {
                let tmp = tempfile::tempdir().unwrap();
                let policy = tmp.path().join("policy");
                if with_policy {
                    write_copy(&policy, OLDER);
                }
                let leftover = tmp.path().join(sibling);
                write_copy(&leftover, NEWER);
                std::fs::write(leftover.join(name), b"user data").unwrap();
                let before = tree_snapshot(tmp.path());

                let result = try_write_copy(&policy, NEWEST);

                let mut failures = Vec::new();
                if !refused_entry(&result)
                    .is_some_and(|(dir, entry)| dir == leftover && entry == leftover.join(name))
                {
                    failures.push(format!("{sibling}/{name}: {result:?}"));
                }
                if tree_snapshot(tmp.path()) != before {
                    failures.push(format!("{sibling}/{name}: the disk changed"));
                }
                failures
            })
            .collect();
        assert!(failures.is_empty(), "{failures:#?}");
    }

    #[cfg(unix)]
    #[test]
    fn rewrite_that_cannot_stage_keeps_the_previous_checkpoint() {
        use std::os::unix::fs::PermissionsExt;
        if is_root() {
            return;
        }

        let tmp = tempfile::tempdir().unwrap();
        let policy = tmp.path().join("policy");
        write_copy(&policy, OLDER);
        std::fs::set_permissions(tmp.path(), std::fs::Permissions::from_mode(0o555)).unwrap();

        let result = try_write_copy(&policy, NEWER);

        std::fs::set_permissions(tmp.path(), std::fs::Permissions::from_mode(0o755)).unwrap();
        assert!(
            matches!(result, Err(OutputError::IoError { .. })),
            "a rewrite that cannot stage must fail with IoError, got {result:?}"
        );
        assert_eq!(read_iterations(&policy).unwrap(), OLDER);
    }

    #[test]
    fn rewrite_refuses_a_directory_holding_an_unrecognized_entry() {
        type AddEntry = fn(&Path);
        let add_file = |path: &Path| std::fs::write(path, b"user data").unwrap();
        let add_dir = |path: &Path| std::fs::create_dir(path).unwrap();
        let cases: [(&str, AddEntry); 4] = [
            ("notes.txt", add_file),
            ("cuts/README", add_file),
            ("cuts/metadata.json", add_file),
            ("metadata.json", add_dir),
        ];

        let failures: Vec<String> = cases
            .into_iter()
            .flat_map(|(name, add)| {
                let tmp = tempfile::tempdir().unwrap();
                let policy = tmp.path().join("policy");
                write_copy(&policy, OLDER);
                add(&policy.join(name));
                let before = tree_snapshot(tmp.path());

                let result = try_write_copy(&policy, NEWER);

                let mut failures = Vec::new();
                if !refused_entry(&result)
                    .is_some_and(|(dir, entry)| dir == policy && entry.ends_with(name))
                {
                    failures.push(format!("{name}: {result:?}"));
                }
                if tree_snapshot(tmp.path()) != before {
                    failures.push(format!("{name}: the disk changed"));
                }
                failures
            })
            .collect();
        assert!(failures.is_empty(), "{failures:#?}");
    }

    #[cfg(unix)]
    #[test]
    fn rewrite_refuses_a_policy_path_that_is_not_a_directory() {
        use std::os::unix::fs::symlink;
        type BuildRefusedPath = fn(&Path) -> PathBuf;

        let file_at_policy = |root: &Path| {
            std::fs::write(root.join("policy"), b"not a directory").unwrap();
            root.join("policy")
        };
        let file_at_staging = |root: &Path| {
            write_copy(&root.join("policy"), OLDER);
            std::fs::write(root.join("policy.staging"), b"not a directory").unwrap();
            root.join("policy.staging")
        };
        let link_at_previous = |root: &Path| {
            write_copy(&root.join("policy"), OLDER);
            std::fs::create_dir(root.join("elsewhere")).unwrap();
            symlink("elsewhere", root.join("policy.previous")).unwrap();
            root.join("policy.previous")
        };
        let link_to_a_file = |root: &Path| {
            std::fs::write(root.join("file.txt"), b"not a directory").unwrap();
            symlink("file.txt", root.join("policy")).unwrap();
            root.join("policy")
        };
        let link_to_a_link = |root: &Path| {
            std::fs::create_dir(root.join("real")).unwrap();
            symlink("real", root.join("hop")).unwrap();
            symlink("hop", root.join("policy")).unwrap();
            root.join("policy")
        };
        let link_to_nothing = |root: &Path| {
            symlink("missing", root.join("policy")).unwrap();
            root.join("policy")
        };
        let cases: [(&str, BuildRefusedPath); 6] = [
            ("file at policy", file_at_policy),
            ("file at policy.staging", file_at_staging),
            ("link at policy.previous", link_at_previous),
            ("link to a file", link_to_a_file),
            ("link to a link", link_to_a_link),
            ("link to nothing", link_to_nothing),
        ];

        let failures: Vec<String> = cases
            .into_iter()
            .flat_map(|(case, build)| {
                let tmp = tempfile::tempdir().unwrap();
                let refused = build(tmp.path());
                let before = tree_snapshot(tmp.path());

                let result = try_write_copy(&tmp.path().join("policy"), NEWER);

                let mut failures = Vec::new();
                if !refused_entry(&result)
                    .is_some_and(|(dir, entry)| dir == tmp.path() && entry == refused)
                {
                    failures.push(format!("{case}: {result:?}"));
                }
                if tree_snapshot(tmp.path()) != before {
                    failures.push(format!("{case}: the disk changed"));
                }
                failures
            })
            .collect();
        assert!(failures.is_empty(), "{failures:#?}");
    }

    #[cfg(unix)]
    #[test]
    fn write_through_a_linked_policy_directory_replaces_the_target_and_keeps_the_link() {
        let write_failures = |label: String, root: &Path, link: &Path| {
            let result = try_write_copy(link, NEWEST);
            let real_policy = root.join("real/policy");
            let mut failures = Vec::new();
            if let Err(e) = result {
                failures.push(format!("{label}: write {e:?}"));
            }
            match read_iterations(link) {
                Ok(NEWEST) => {}
                read => failures.push(format!("{label}: read {read:?}, expected {NEWEST}")),
            }
            match std::fs::read_link(link) {
                Ok(target) if target == Path::new("../real/policy") => {}
                target => failures.push(format!("{label}: link reads {target:?}")),
            }
            if !std::fs::symlink_metadata(&real_policy).is_ok_and(|m| m.is_dir()) {
                failures.push(format!("{label}: the link's target is not a directory"));
            }
            let siblings = leftover_siblings(&real_policy);
            if !siblings.is_empty() {
                failures.push(format!("{label}: left {siblings:?}"));
            }
            let out: Vec<_> = std::fs::read_dir(root.join("out"))
                .unwrap()
                .map(|entry| entry.unwrap().file_name())
                .collect();
            if out != ["policy"] {
                failures.push(format!("{label}: out/ lists {out:?}"));
            }
            failures
        };

        let mut failures: Vec<String> = CRASH_STATES
            .into_iter()
            .flat_map(|state| {
                let tmp = tempfile::tempdir().unwrap();
                build_crash_state(&tmp.path().join("real"), state);
                let link = link_to_real_policy(tmp.path());
                write_failures(format!("{state:?}"), tmp.path(), &link)
            })
            .collect();

        let tmp = tempfile::tempdir().unwrap();
        let link = link_to_real_policy(tmp.path());
        std::fs::create_dir(tmp.path().join("real/policy")).unwrap();
        failures.extend(write_failures(
            "empty target".to_string(),
            tmp.path(),
            &link,
        ));

        assert!(failures.is_empty(), "{failures:#?}");
    }

    #[test]
    fn rewrite_discards_the_metadata_file_earlier_releases_left() {
        let tmp = tempfile::tempdir().unwrap();
        let policy = tmp.path().join("policy");
        write_copy(&policy, OLDER);
        std::fs::write(policy.join("metadata.json"), b"{}").unwrap();

        let result = try_write_copy(&policy, NEWER);

        assert!(result.is_ok(), "{result:?}");
        assert_eq!(read_iterations(&policy).unwrap(), NEWER);
        assert!(
            !policy.join("metadata.json").exists(),
            "the earlier release's metadata.json must go with the replaced copy"
        );
        assert_eq!(leftover_siblings(&policy), Vec::<PathBuf>::new());
    }
}
