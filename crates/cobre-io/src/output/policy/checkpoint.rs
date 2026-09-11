//! Filesystem write and read entry points for value-function artifacts.
//!
//! `manifest.bin` is written last so its presence is the commit signal of a
//! complete artifact, and it carries the `format_version` marker the reader
//! checks first — an artifact without it is cleanly rejected before any payload
//! is parsed.

use std::collections::BTreeMap;
use std::path::Path;

use chrono::NaiveDate;

use super::super::atomic::write_bytes_atomic;
use super::super::error::OutputError;
use super::codec::{
    deserialize_checkpoint_manifest, deserialize_stage_basis, deserialize_stage_cuts,
    deserialize_stage_states, read_sorted_bin_files, serialize_checkpoint_manifest,
    serialize_stage_basis, serialize_stage_cuts, serialize_stage_states,
};
use super::records::{
    CheckpointManifest, ENTITY_SLOT_DELIVERY_DATE_SENTINEL, OwnedPolicyBasisRecord,
    PolicyBasisRecord, PolicyCheckpoint, StageCutsPayload, StageCutsReadResult, StageStatesPayload,
    StageStatesReadResult, StateFamily,
};

/// Whether `delivery_date` is [`ENTITY_SLOT_DELIVERY_DATE_SENTINEL`] or decodes
/// as a valid `YYYYMMDD` date.
fn is_well_formed_delivery_date(delivery_date: i32) -> bool {
    if delivery_date == ENTITY_SLOT_DELIVERY_DATE_SENTINEL {
        return true;
    }
    let year = delivery_date / 10_000;
    let month = (delivery_date / 100) % 100;
    let day = delivery_date % 100;
    let (Ok(month), Ok(day)) = (u32::try_from(month), u32::try_from(day)) else {
        return false;
    };
    NaiveDate::from_ymd_opt(year, month, day).is_some()
}

/// Verify one pool's [`StateFamily::HydroTransitBucket`] slots (grouped by
/// `entity_id`) carry non-sentinel `delivery_date`s that are monotone
/// non-decreasing in `subindex` (the maturity-lag depth).
///
/// Only this family is checked: its `subindex` is a genuine delivery-ordered
/// maturity depth, whereas the other calendar-shaped family's modular
/// delivery-target-residue `subindex` wraps across the horizon, so enforcing
/// monotonicity there would reject correctly-produced, non-monotone dates.
///
/// # Errors
///
/// Returns [`OutputError::SerializationError`] naming the pool, the offending
/// subindex, and its `delivery_date`.
fn check_transit_bucket_monotonicity(pool: &StageCutsReadResult) -> Result<(), OutputError> {
    let mut by_entity: BTreeMap<i32, Vec<(u32, i32)>> = BTreeMap::new();
    for slot in &pool.entity_manifest {
        if slot.family() == Some(StateFamily::HydroTransitBucket)
            && slot.delivery_date != ENTITY_SLOT_DELIVERY_DATE_SENTINEL
        {
            by_entity
                .entry(slot.entity_id)
                .or_default()
                .push((slot.subindex, slot.delivery_date));
        }
    }
    for dates in by_entity.values_mut() {
        dates.sort_by_key(|&(subindex, _)| subindex);
        for pair in dates.windows(2) {
            let (prev_subindex, prev_date) = pair[0];
            let (subindex, date) = pair[1];
            if date < prev_date {
                let pool_id = pool.stage_id;
                return Err(OutputError::serialization(
                    "policy_checkpoint_dates",
                    format!(
                        "pool {pool_id} subindex {subindex} carries delivery_date {date}, \
                         earlier than subindex {prev_subindex}'s {prev_date}"
                    ),
                ));
            }
        }
    }
    Ok(())
}

/// Validate that `checkpoint` is internally date-consistent: every
/// [`EntitySlot`](super::records::EntitySlot)'s non-sentinel `delivery_date` is
/// a well-formed `YYYYMMDD` date, and every pool's `HydroTransitBucket` slots
/// are monotone non-decreasing in `subindex`
/// (see [`check_transit_bucket_monotonicity`]).
///
/// # Errors
///
/// Returns [`OutputError::SerializationError`] naming the offending pool,
/// subindex, and `delivery_date`.
fn validate_checkpoint_dates(checkpoint: &PolicyCheckpoint) -> Result<(), OutputError> {
    for pool in &checkpoint.stage_cuts {
        for slot in &pool.entity_manifest {
            if !is_well_formed_delivery_date(slot.delivery_date) {
                return Err(OutputError::serialization(
                    "policy_checkpoint_dates",
                    format!(
                        "pool {} subindex {} carries malformed delivery_date {}",
                        pool.stage_id, slot.subindex, slot.delivery_date
                    ),
                ));
            }
        }
        check_transit_bucket_monotonicity(pool)?;
    }
    Ok(())
}

/// One `.bin` payload file name, keyed by the payload's own id (the pool id for
/// `cuts/`, the stage id for `basis/`/`states/`). Zero-padded for a stable
/// on-disk sort; the reader derives identity from inside each buffer, never from
/// this name.
fn bin_file_name(id: u32) -> String {
    format!("{id:03}.bin")
}

/// Write a complete value-function artifact to `path`.
///
/// ## Directory layout produced
///
/// ```text
/// path/
///   manifest.bin
///   cuts/
///     000.bin        (one per pool, keyed by pool id; a shared leaf pool once)
///     001.bin
///     ...
///   basis/
///     000.bin        (only when stage_bases is non-empty)
///     001.bin
///     ...
/// ```
///
/// `manifest.bin` is written **last**, only after every `.bin` write succeeds:
/// its absence is how the caller detects an incomplete artifact. Partially
/// written files are not cleaned up. An empty `stage_bases` writes no basis files
/// (the `basis/` directory is still created).
///
/// Rewriting a directory that already holds a checkpoint removes its
/// `manifest.bin` before any payload write, so a crash partway through the
/// rewrite cannot leave that old manifest pointing at new or partially written
/// payloads.
///
/// # Errors
///
/// - [`OutputError::IoError`] — directory creation or file write failed.
///
/// # Examples
///
/// ```no_run
/// use cobre_io::{
///     write_policy_checkpoint, FORMAT_VERSION, GraphManifest, PolicyBasisRecord,
///     CheckpointManifest, PolicyCutRecord, ProducerBlock, StageCutsPayload,
/// };
/// use std::path::Path;
///
/// # fn main() -> Result<(), cobre_io::OutputError> {
/// let coefficients = [1.0_f64, 2.0, 3.0];
/// let piece = PolicyCutRecord {
///     cut_id: 1,
///     slot_index: 0,
///     iteration: 1,
///     forward_pass_index: 0,
///     intercept: 42.0,
///     coefficients: &coefficients,
///     is_active: true,
/// };
/// let stage_cuts = [StageCutsPayload {
///     stage_id: 0,
///     state_dimension: 3,
///     capacity: 100,
///     warm_start_count: 0,
///     cuts: &[piece],
///     active_cut_indices: &[0],
///     populated_count: 1,
///     entity_manifest: &[],
///     cost_scale_factor: 1_000_000.0,
///     node_id: 0,
///     graph_stage_id: 0,
/// }];
/// let metadata = CheckpointManifest {
///     format_version: FORMAT_VERSION,
///     cobre_version: env!("CARGO_PKG_VERSION").to_string(),
///     created_at: "2026-03-08T00:00:00Z".to_string(),
///     num_stages: 1,
///     graph_manifest: GraphManifest::default(),
///     producer: ProducerBlock {
///         completed_iterations: 1,
///         final_lower_bound: 42.0,
///         best_upper_bound: None,
///         max_iterations: 100,
///         forward_passes: 4,
///         warm_start_cuts: 0,
///         warm_start_counts: vec![0],
///         rng_seed: 0,
///         total_visited_states: 0,
///         training_block_mode: "parallel".to_string(),
///         training_block_mode_per_stage: vec![],
///         cost_scale_factor: None,
///     },
/// };
/// write_policy_checkpoint(Path::new("/tmp/policy"), &stage_cuts, &[], &metadata, &[])?;
/// # Ok(())
/// # }
/// ```
pub fn write_policy_checkpoint(
    path: &Path,
    stage_cuts: &[StageCutsPayload<'_>],
    stage_bases: &[PolicyBasisRecord<'_>],
    metadata: &CheckpointManifest,
    stage_states: &[StageStatesPayload<'_>],
) -> Result<(), OutputError> {
    let manifest_path = path.join("manifest.bin");

    let cuts_dir = path.join("cuts");
    std::fs::create_dir_all(&cuts_dir).map_err(|e| OutputError::io(&cuts_dir, e))?;

    let basis_dir = path.join("basis");
    std::fs::create_dir_all(&basis_dir).map_err(|e| OutputError::io(&basis_dir, e))?;

    match std::fs::remove_file(&manifest_path) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(OutputError::io(&manifest_path, e)),
    }

    for payload in stage_cuts {
        let file_path = cuts_dir.join(bin_file_name(payload.stage_id));
        let buf = serialize_stage_cuts(payload);
        write_bytes_atomic(&file_path, &buf)?;
    }

    for record in stage_bases {
        let file_path = basis_dir.join(bin_file_name(record.stage_id));
        let buf = serialize_stage_basis(record);
        write_bytes_atomic(&file_path, &buf)?;
    }

    if !stage_states.is_empty() {
        let states_dir = path.join("states");
        std::fs::create_dir_all(&states_dir).map_err(|e| OutputError::io(&states_dir, e))?;

        for payload in stage_states {
            let file_path = states_dir.join(bin_file_name(payload.stage_id));
            let buf = serialize_stage_states(payload);
            write_bytes_atomic(&file_path, &buf)?;
        }
    }

    // Write manifest.bin LAST — its presence is the commit signal.
    let manifest_buf = serialize_checkpoint_manifest(metadata);
    write_bytes_atomic(&manifest_path, &manifest_buf)?;

    Ok(())
}

/// Read a complete value-function artifact from `path`.
///
/// `manifest.bin` is read first and its `format_version` is checked
/// **before any `.bin` payload is parsed**: an absent `manifest.bin`, a missing
/// `CBVF` identifier, or a mismatched version is a named error, so an artifact
/// this build cannot read is cleanly rejected rather than read positionally.
///
/// Per-pool/-stage results are sorted by `stage_id` in the returned
/// [`PolicyCheckpoint`].
///
/// # Errors
///
/// - [`OutputError::IoError`] — directory or file read failed (a missing
///   `manifest.bin`, i.e. a pre-`manifest.bin` artifact, included).
/// - [`OutputError::SerializationError`] — a `FlatBuffers` parse failure, a
///   missing `CBVF` identifier or a `format_version` mismatch (both enforced by
///   [`deserialize_checkpoint_manifest`]), or a date-consistency violation
///   caught by [`validate_checkpoint_dates`].
///
/// # Examples
///
/// ```no_run
/// use cobre_io::read_policy_checkpoint;
/// use std::path::Path;
///
/// # fn main() -> Result<(), cobre_io::OutputError> {
/// let checkpoint = read_policy_checkpoint(Path::new("/tmp/policy"))?;
/// println!("metadata: {} stages", checkpoint.metadata.num_stages);
/// println!("stages loaded: {}", checkpoint.stage_cuts.len());
/// # Ok(())
/// # }
/// ```
pub fn read_policy_checkpoint(path: &Path) -> Result<PolicyCheckpoint, OutputError> {
    // Read manifest.bin FIRST: its CBVF-identifier and format_version gates
    // reject an unreadable artifact before any payload is parsed.
    let manifest_path = path.join("manifest.bin");
    let manifest_bytes =
        std::fs::read(&manifest_path).map_err(|e| OutputError::io(&manifest_path, e))?;
    let metadata = deserialize_checkpoint_manifest(&manifest_bytes)?;

    let cuts_dir = path.join("cuts");
    let mut stage_cuts: Vec<StageCutsReadResult> =
        read_sorted_bin_files(&cuts_dir, "stage_cuts", deserialize_stage_cuts)?;
    stage_cuts.sort_by_key(|r| r.stage_id);

    let basis_dir = path.join("basis");
    let mut stage_bases: Vec<OwnedPolicyBasisRecord> =
        read_sorted_bin_files(&basis_dir, "stage_basis", deserialize_stage_basis)?;
    stage_bases.sort_by_key(|r| r.stage_id);

    let states_dir = path.join("states");
    let stage_states: Vec<StageStatesReadResult> = if states_dir.is_dir() {
        let mut ss = read_sorted_bin_files(&states_dir, "stage_states", deserialize_stage_states)?;
        ss.sort_by_key(|r| r.stage_id);
        ss
    } else {
        Vec::new()
    };

    let checkpoint = PolicyCheckpoint {
        metadata,
        stage_cuts,
        stage_bases,
        stage_states,
    };
    validate_checkpoint_dates(&checkpoint)?;
    Ok(checkpoint)
}
