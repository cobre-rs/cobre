//! Filesystem write and read entry points for value-function artifacts.
//!
//! `manifest.bin` is written last so its presence is the commit signal of a
//! complete artifact, and it carries the `format_version` marker the reader
//! checks first — an artifact without it is cleanly rejected before any payload
//! is parsed.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use super::super::atomic::write_bytes_atomic;
use super::super::error::OutputError;
use super::codec::{
    build_checkpoint_manifest, build_stage_basis, build_stage_cuts, build_stage_states,
    deserialize_checkpoint_manifest, deserialize_stage_basis, deserialize_stage_cuts,
    deserialize_stage_states, read_sorted_bin_files,
};
use super::records::{
    CheckpointManifest, ENTITY_SLOT_DATE_SENTINEL, EntitySlot, OwnedPolicyBasisRecord,
    PolicyBasisRecord, PolicyCheckpoint, StageCutsPayload, StageCutsReadResult, StageStatesPayload,
    StageStatesReadResult, StateFamily, decode_slot_date,
};

fn is_well_formed_slot_date(value: i32) -> bool {
    value == ENTITY_SLOT_DATE_SENTINEL || decode_slot_date(value).is_some()
}

fn slot_date_error(pool_id: u32, slot: &EntitySlot, detail: &str) -> OutputError {
    OutputError::serialization(
        "policy_checkpoint_dates",
        format!(
            "pool {pool_id} entity {} subindex {} {detail}",
            slot.entity_id, slot.subindex
        ),
    )
}

fn check_well_formed_slot_dates(pool_id: u32, slot: &EntitySlot) -> Result<(), OutputError> {
    for (field_name, value) in [
        ("reference_date", slot.reference_date),
        ("interval_start", slot.interval_start),
        ("interval_end", slot.interval_end),
    ] {
        if !is_well_formed_slot_date(value) {
            return Err(slot_date_error(
                pool_id,
                slot,
                &format!("carries malformed {field_name} {value}"),
            ));
        }
    }
    Ok(())
}

fn check_interval_pairing(pool_id: u32, slot: &EntitySlot) -> Result<(), OutputError> {
    let start_live = slot.interval_start != ENTITY_SLOT_DATE_SENTINEL;
    let end_live = slot.interval_end != ENTITY_SLOT_DATE_SENTINEL;
    match (start_live, end_live) {
        (true, false) => Err(slot_date_error(
            pool_id,
            slot,
            "carries a live interval_start with no interval_end",
        )),
        (false, true) => Err(slot_date_error(
            pool_id,
            slot,
            "carries a live interval_end with no interval_start",
        )),
        _ => Ok(()),
    }
}

fn check_interval_ordering(pool_id: u32, slot: &EntitySlot) -> Result<(), OutputError> {
    let start = slot.interval_start;
    let end = slot.interval_end;
    if start != ENTITY_SLOT_DATE_SENTINEL && end != ENTITY_SLOT_DATE_SENTINEL && start >= end {
        return Err(slot_date_error(
            pool_id,
            slot,
            &format!("carries interval_start {start} not before interval_end {end}"),
        ));
    }
    Ok(())
}

fn check_family_applicability(pool_id: u32, slot: &EntitySlot) -> Result<(), OutputError> {
    let interval_live = slot.interval_start != ENTITY_SLOT_DATE_SENTINEL
        || slot.interval_end != ENTITY_SLOT_DATE_SENTINEL;
    let reference_live = slot.reference_date != ENTITY_SLOT_DATE_SENTINEL;
    match slot.family() {
        Some(StateFamily::HydroStorage) => {
            if reference_live {
                return Err(slot_date_error(
                    pool_id,
                    slot,
                    &format!(
                        "is a storage slot, which carries no per-slot date, but carries a live reference_date {}",
                        slot.reference_date
                    ),
                ));
            }
            if interval_live {
                return Err(slot_date_error(
                    pool_id,
                    slot,
                    "is a storage slot, which carries no per-slot date, but carries a live interval",
                ));
            }
        }
        Some(StateFamily::HydroInflowLag) => {
            if interval_live {
                return Err(slot_date_error(
                    pool_id,
                    slot,
                    "is an inflow-lag slot, which carries no interval, but carries a live interval",
                ));
            }
        }
        other => {
            if reference_live {
                let noun = match other {
                    Some(StateFamily::HydroTransitBucket) => "a transit-bucket slot",
                    Some(StateFamily::AnticipatedThermalState) => {
                        "an anticipated-thermal-state slot"
                    }
                    _ => "a slot with no recognized family",
                };
                return Err(slot_date_error(
                    pool_id,
                    slot,
                    &format!(
                        "is {noun}, which carries no reference_date (only inflow-lag slots do), but carries a live reference_date {}",
                        slot.reference_date
                    ),
                ));
            }
        }
    }
    Ok(())
}

/// Non-sentinel `interval_start`s must be monotone non-decreasing in `subindex`
/// for [`StateFamily::HydroTransitBucket`] slots. Only this family is checked:
/// the other calendar-shaped family's modular delivery-target-residue `subindex`
/// wraps across the horizon, so monotonicity there would reject valid dates.
///
/// # Errors
///
/// Returns [`OutputError::SerializationError`] naming the pool, the offending
/// subindex, and its `interval_start`.
fn check_transit_bucket_monotonicity(pool: &StageCutsReadResult) -> Result<(), OutputError> {
    let mut by_entity: BTreeMap<i32, Vec<(u32, i32)>> = BTreeMap::new();
    for slot in &pool.entity_manifest {
        if slot.family() == Some(StateFamily::HydroTransitBucket)
            && slot.interval_start != ENTITY_SLOT_DATE_SENTINEL
        {
            by_entity
                .entry(slot.entity_id)
                .or_default()
                .push((slot.subindex, slot.interval_start));
        }
    }
    for starts in by_entity.values_mut() {
        starts.sort_by_key(|&(subindex, _)| subindex);
        for pair in starts.windows(2) {
            let (prev_subindex, prev_start) = pair[0];
            let (subindex, start) = pair[1];
            if start < prev_start {
                let pool_id = pool.stage_id;
                return Err(OutputError::serialization(
                    "policy_checkpoint_dates",
                    format!(
                        "pool {pool_id} subindex {subindex} carries interval_start {start}, \
                         earlier than subindex {prev_subindex}'s {prev_start}"
                    ),
                ));
            }
        }
    }
    Ok(())
}

/// Validate that `checkpoint` is internally date-consistent, returning the
/// first violation found.
///
/// # Errors
///
/// Returns [`OutputError::SerializationError`] naming the offending pool and
/// slot.
fn validate_checkpoint_dates(checkpoint: &PolicyCheckpoint) -> Result<(), OutputError> {
    for pool in &checkpoint.stage_cuts {
        let pool_id = pool.stage_id;
        // Canonical order so the first reported error is declaration-order invariant.
        let mut slots: Vec<&EntitySlot> = pool.entity_manifest.iter().collect();
        slots.sort_by_key(|slot| (slot.entity_type, slot.entity_id, slot.subindex));
        for slot in slots {
            check_well_formed_slot_dates(pool_id, slot)?;
            check_interval_pairing(pool_id, slot)?;
            check_interval_ordering(pool_id, slot)?;
            check_family_applicability(pool_id, slot)?;
        }
        check_transit_bucket_monotonicity(pool)?;
    }
    Ok(())
}

/// Zero-padded `.bin` name for stable on-disk sort. Identity comes from the
/// buffer content, never this filename.
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
///   states/          (only when stage_states is non-empty)
///     000.bin
///     ...
/// ```
///
/// `manifest.bin` is written **last**, only after every `.bin` write succeeds:
/// its absence is how the caller detects an incomplete artifact. Partially
/// written files are not cleaned up. An empty `stage_bases` writes no basis files
/// (the `basis/` directory is still created).
///
/// Rewriting a directory that already holds a checkpoint removes its
/// `manifest.bin` and every previous payload file before any payload write:
/// an old manifest left in place would pair with new payloads, and a stale
/// payload would survive a rewrite with fewer pools or with states export off.
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
///     CheckpointManifest, PolicyCutRecord, ProducerBlock, SeasonManifest,
///     SOFTWARE_NAME, SOFTWARE_VERSION, STAGE_CUTS_PRICED_STATE_DATE_SENTINEL, StageCutsPayload,
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
///     priced_state_date: STAGE_CUTS_PRICED_STATE_DATE_SENTINEL,
/// }];
/// let metadata = CheckpointManifest {
///     format_version: FORMAT_VERSION,
///     software: Some(SOFTWARE_NAME.to_string()),
///     software_version: SOFTWARE_VERSION.to_string(),
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
///     season_manifest: SeasonManifest::default(),
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
    remove_bin_files(&cuts_dir)?;
    remove_bin_files(&basis_dir)?;

    let states_dir = path.join("states");
    if stage_states.is_empty() {
        match std::fs::remove_dir_all(&states_dir) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(OutputError::io(&states_dir, e)),
        }
    } else {
        std::fs::create_dir_all(&states_dir).map_err(|e| OutputError::io(&states_dir, e))?;
        remove_bin_files(&states_dir)?;
    }

    for payload in stage_cuts {
        let file_path = cuts_dir.join(bin_file_name(payload.stage_id));
        let builder = build_stage_cuts(payload);
        write_bytes_atomic(&file_path, builder.finished_data())?;
    }

    for record in stage_bases {
        let file_path = basis_dir.join(bin_file_name(record.stage_id));
        let builder = build_stage_basis(record);
        write_bytes_atomic(&file_path, builder.finished_data())?;
    }

    for payload in stage_states {
        let file_path = states_dir.join(bin_file_name(payload.stage_id));
        let builder = build_stage_states(payload);
        write_bytes_atomic(&file_path, builder.finished_data())?;
    }

    let manifest_builder = build_checkpoint_manifest(metadata);
    write_bytes_atomic(&manifest_path, manifest_builder.finished_data())?;

    Ok(())
}

/// Removes stale `.bin` files so [`read_policy_checkpoint`] does not mistake
/// them for live pools.
fn remove_bin_files(dir: &Path) -> Result<(), OutputError> {
    for entry in std::fs::read_dir(dir).map_err(|e| OutputError::io(dir, e))? {
        let file_path = entry.map_err(|e| OutputError::io(dir, e))?.path();
        if file_path.extension().is_some_and(|ext| ext == "bin") {
            std::fs::remove_file(&file_path).map_err(|e| OutputError::io(&file_path, e))?;
        }
    }
    Ok(())
}

fn sibling(path: &Path, suffix: &str) -> Option<PathBuf> {
    let mut name = path.file_name()?.to_os_string();
    name.push(suffix);
    Some(path.with_file_name(name))
}

/// The directory a checkpoint at `path` lives in: the recorded target of a
/// symbolic link at `path`, else `path` itself.
fn checkpoint_target(path: &Path) -> Result<PathBuf, OutputError> {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() => {
            // `read_link`, not `canonicalize`: a link left pointing at nothing by
            // an interrupted swap still names the target's siblings.
            let target = std::fs::read_link(path).map_err(|e| OutputError::io(path, e))?;
            Ok(match path.parent() {
                Some(parent) => parent.join(target),
                None => target,
            })
        }
        Ok(_) => Ok(path.to_path_buf()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(path.to_path_buf()),
        Err(e) => Err(OutputError::io(path, e)),
    }
}

/// Which copy of a checkpoint a read uses, as [`resolve_policy_checkpoint`]
/// finds it. The target is the checkpoint path itself, or the recorded target
/// of a symbolic link there.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResolvedCheckpoint {
    /// The directory a read uses: the target, its `.staging` sibling or its
    /// `.previous` sibling.
    Found(PathBuf),
    /// The target exists, and no candidate holds a `manifest.bin`.
    NoManifest,
    /// The target is absent, and no candidate holds a `manifest.bin`.
    NoDirectory,
}

/// Resolve which copy of the checkpoint at `path` a read uses.
///
/// The candidates, in order, are `path`, its `.staging` sibling and its
/// `.previous` sibling; the first that holds a `manifest.bin` is the copy. A
/// present `manifest.bin` is the only completeness signal, because a writer
/// puts it in place last. A symbolic link at `path` is read once, and its
/// recorded target and that target's siblings are the candidates.
///
/// The call changes nothing on disk, so any number of processes may resolve
/// the same `path` at once.
///
/// # Errors
///
/// [`OutputError::IoError`] naming the probed path when a probe fails for any
/// reason other than absence, or naming `path` when a link there cannot be
/// inspected or read.
pub fn resolve_policy_checkpoint(path: &Path) -> Result<ResolvedCheckpoint, OutputError> {
    let target = checkpoint_target(path)?;
    let candidates = [
        Some(target.clone()),
        sibling(&target, ".staging"),
        sibling(&target, ".previous"),
    ];
    for dir in candidates.into_iter().flatten() {
        let manifest_path = dir.join("manifest.bin");
        if manifest_path
            .try_exists()
            .map_err(|e| OutputError::io(&manifest_path, e))?
        {
            return Ok(ResolvedCheckpoint::Found(dir));
        }
    }
    if target
        .try_exists()
        .map_err(|e| OutputError::io(&target, e))?
    {
        Ok(ResolvedCheckpoint::NoManifest)
    } else {
        Ok(ResolvedCheckpoint::NoDirectory)
    }
}

/// Read a complete value-function artifact from the copy of `path` that
/// [`resolve_policy_checkpoint`] finds.
///
/// That copy is `path` (for a symbolic link, its target), else its `.staging`
/// sibling, else its `.previous` sibling, whichever first holds a
/// `manifest.bin`; with none, the read fails on `<path>/manifest.bin`. The
/// read changes nothing on disk.
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
/// - [`OutputError::IoError`] — a probe, directory or file read failed (a
///   missing `manifest.bin`, i.e. a pre-`manifest.bin` artifact, included).
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
    match resolve_policy_checkpoint(path)? {
        ResolvedCheckpoint::Found(dir) => read_checkpoint_dir(&dir),
        ResolvedCheckpoint::NoManifest | ResolvedCheckpoint::NoDirectory => {
            read_checkpoint_dir(path)
        }
    }
}

fn read_checkpoint_dir(path: &Path) -> Result<PolicyCheckpoint, OutputError> {
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
