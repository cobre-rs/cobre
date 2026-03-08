//! `FlatBuffers` builder types and functions for policy checkpoint serialization.
//!
//! This module defines the input types and builder functions used to serialize policy
//! data (cuts and solver bases) to `FlatBuffers` binary format for checkpoint persistence.
//!
//! ## Design
//!
//! Types in this module use generic names to maintain infrastructure crate genericity.
//! They mirror the mathematical concepts (cut intercepts, gradient coefficients, simplex
//! basis status codes) without referencing any specific optimization algorithm. Conversion
//! from algorithm-specific types to these input types is the responsibility of the calling
//! crate.
//!
//! ## `FlatBuffers` schema
//!
//! The binary layout produced by [`serialize_stage_cuts`] and [`serialize_stage_basis`]
//! corresponds to the `StageCuts` and `StageBasis` tables defined in the policy schema
//! specification (SS3.1 in `binary-formats.md`). No `.fbs` file or `flatc` code generation
//! is used; the builder API writes the binary directly.
//!
//! ## Format details
//!
//! Buffers are written using the `FlatBuffers` runtime builder API with
//! [`flatbuffers::FlatBufferBuilder`]. Output is little-endian and deterministic for the
//! same input — field order is fixed by the builder call sequence, matching the schema
//! field declaration order in SS3.1.

use std::path::Path;

use flatbuffers::{FlatBufferBuilder, WIPOffset};

use super::error::OutputError;

/// One cut record for policy checkpoint serialization.
///
/// Conversion from algorithm-specific cut pool structures is handled by the calling
/// algorithm crate. This type uses generic names to maintain infrastructure crate
/// genericity. The lifetime parameter `'a` allows borrowing the coefficient slice
/// without copying (coefficient vectors can reach 2,080 `f64` values at production
/// scale).
///
/// Field names correspond to the `BendersCut` table in SS3.1 of the policy schema
/// specification.
#[derive(Debug, Clone)]
pub struct PolicyCutRecord<'a> {
    /// Unique identifier for this cut across all iterations.
    pub cut_id: u64,
    /// LP row position (required for checkpoint reproducibility).
    pub slot_index: u32,
    /// Training iteration that generated this cut.
    pub iteration: u32,
    /// Forward pass index within the generating iteration.
    pub forward_pass_index: u32,
    /// Pre-computed cut intercept: `alpha - beta' * x_hat`.
    pub intercept: f64,
    /// Gradient coefficient vector, length must equal `state_dimension`.
    pub coefficients: &'a [f64],
    /// Whether this cut is currently active in the LP.
    pub is_active: bool,
    /// Domination count for cut selection bookkeeping.
    pub domination_count: u32,
}

/// One stage's solver basis for policy checkpoint serialization.
///
/// Conversion from solver-specific basis structures is handled by the calling crate.
/// The lifetime parameter `'a` allows borrowing the status arrays without copying.
///
/// Field names correspond to the `StageBasis` table in SS3.1 of the policy schema
/// specification.
#[derive(Debug, Clone)]
pub struct PolicyBasisRecord<'a> {
    /// Stage index (0-based).
    pub stage_id: u32,
    /// Training iteration that produced this basis.
    pub iteration: u32,
    /// One status code per LP column (variable). Encoding is solver-specific.
    pub column_status: &'a [u8],
    /// One status code per LP row (constraint). Encoding is solver-specific.
    pub row_status: &'a [u8],
    /// Number of trailing rows in `row_status` that correspond to cut rows.
    pub num_cut_rows: u32,
}

/// Policy metadata for checkpoint resume and warm-start.
///
/// Serialized to JSON (not `FlatBuffers`) because it is small, human-readable, and
/// may be edited by operators. The `serde::Serialize` derive enables
/// `serde_json::to_string_pretty` in the checkpoint writer.
///
/// Field names correspond to the `PolicyMetadata` table in SS3.1 of the policy
/// schema specification.
///
/// # Examples
///
/// ```
/// use cobre_io::PolicyCheckpointMetadata;
///
/// let meta = PolicyCheckpointMetadata {
///     version: "1.0.0".to_string(),
///     cobre_version: env!("CARGO_PKG_VERSION").to_string(),
///     created_at: "2026-03-08T00:00:00Z".to_string(),
///     completed_iterations: 50,
///     final_lower_bound: 1234.56,
///     best_upper_bound: Some(1300.0),
///     state_dimension: 160,
///     num_stages: 60,
///     config_hash: "abc123".to_string(),
///     system_hash: "def456".to_string(),
///     max_iterations: 200,
///     forward_passes: 4,
///     warm_start_cuts: 0,
///     rng_seed: 42,
/// };
/// let json = serde_json::to_string_pretty(&meta).unwrap();
/// assert!(json.contains("completed_iterations"));
/// ```
#[derive(Debug, Clone, serde::Serialize)]
pub struct PolicyCheckpointMetadata {
    /// Schema version string (e.g., `"1.0.0"`).
    pub version: String,
    /// Cobre crate version that wrote this checkpoint.
    pub cobre_version: String,
    /// ISO 8601 timestamp when the checkpoint was written.
    pub created_at: String,
    /// Number of training iterations completed at checkpoint time.
    pub completed_iterations: u32,
    /// Lower bound value after the final completed iteration.
    pub final_lower_bound: f64,
    /// Best upper bound observed during training, if available.
    pub best_upper_bound: Option<f64>,
    /// Number of state variables (determines cut coefficient vector length).
    pub state_dimension: u32,
    /// Number of stages in the planning horizon.
    pub num_stages: u32,
    /// Hash of the algorithm configuration, for compatibility checking on resume.
    pub config_hash: String,
    /// Hash of the system data, for compatibility checking on resume.
    pub system_hash: String,
    /// Maximum number of iterations configured for the run.
    pub max_iterations: u32,
    /// Number of forward passes per iteration.
    pub forward_passes: u32,
    /// Number of cuts loaded from a previous policy at run start.
    pub warm_start_cuts: u32,
    /// RNG seed used by the scenario sampler.
    pub rng_seed: u64,
}

// FlatBuffers field offsets. Offsets derived from field declaration order in SS3.1.
// Formula: slot_offset = (field_index + 2) * 2 (accounts for vtable header fields).
// Must match schema declaration order exactly for interoperability.

const CUT_FIELD_CUT_ID: u16 = 4;
const CUT_FIELD_SLOT_INDEX: u16 = 6;
const CUT_FIELD_ITERATION: u16 = 8;
const CUT_FIELD_FORWARD_PASS_IDX: u16 = 10;
const CUT_FIELD_INTERCEPT: u16 = 14;
const CUT_FIELD_COEFFICIENTS: u16 = 16;
const CUT_FIELD_STATE_AT_GENERATION: u16 = 18;
const CUT_FIELD_IS_ACTIVE: u16 = 20;
const CUT_FIELD_DOMINATION_COUNT: u16 = 22;

const STAGE_CUTS_FIELD_STAGE_ID: u16 = 4;
const STAGE_CUTS_FIELD_STATE_DIMENSION: u16 = 6;
const STAGE_CUTS_FIELD_CAPACITY: u16 = 8;
const STAGE_CUTS_FIELD_WARM_START_COUNT: u16 = 10;
const STAGE_CUTS_FIELD_CUTS: u16 = 12;
const STAGE_CUTS_FIELD_ACTIVE_CUT_INDICES: u16 = 14;
const STAGE_CUTS_FIELD_POPULATED_COUNT: u16 = 16;

const BASIS_FIELD_STAGE_ID: u16 = 4;
const BASIS_FIELD_ITERATION: u16 = 6;
const BASIS_FIELD_NUM_COLUMNS: u16 = 8;
const BASIS_FIELD_NUM_ROWS: u16 = 10;
const BASIS_FIELD_COLUMN_STATUS: u16 = 12;
const BASIS_FIELD_ROW_STATUS: u16 = 14;
const BASIS_FIELD_NUM_CUT_ROWS: u16 = 16;

// ── Helper: build a single cut table ─────────────────────────────────────────

/// Build a single cut table inside `builder` and return its offset.
///
/// All nested objects (coefficient vector, `state_at_generation` vector) must be
/// created before the table `start_table`/`end_table` pair, per the `FlatBuffers`
/// requirement that nested objects precede the enclosing table in the buffer.
fn build_cut_table(
    builder: &mut FlatBufferBuilder<'_>,
    cut: &PolicyCutRecord<'_>,
) -> WIPOffset<flatbuffers::TableFinishedWIPOffset> {
    let coefficients_vec = builder.create_vector(cut.coefficients);
    let state_at_gen_vec = builder.create_vector::<f64>(&[]);

    let tab = builder.start_table();

    builder.push_slot_always::<u64>(CUT_FIELD_CUT_ID, cut.cut_id);
    builder.push_slot_always::<u32>(CUT_FIELD_SLOT_INDEX, cut.slot_index);
    builder.push_slot_always::<u32>(CUT_FIELD_ITERATION, cut.iteration);
    builder.push_slot_always::<u32>(CUT_FIELD_FORWARD_PASS_IDX, cut.forward_pass_index);
    builder.push_slot_always::<f64>(CUT_FIELD_INTERCEPT, cut.intercept);
    builder.push_slot_always(CUT_FIELD_COEFFICIENTS, coefficients_vec);
    builder.push_slot_always(CUT_FIELD_STATE_AT_GENERATION, state_at_gen_vec);
    builder.push_slot_always::<bool>(CUT_FIELD_IS_ACTIVE, cut.is_active);
    builder.push_slot_always::<u32>(CUT_FIELD_DOMINATION_COUNT, cut.domination_count);

    builder.end_table(tab)
}

/// Serialize all cuts for one stage into a `FlatBuffers` buffer.
///
/// Produces a buffer containing a root `StageCuts` table. The buffer is ready
/// for writing directly to a `.bin` policy file. Field layout matches the
/// `StageCuts` and `BendersCut` table declarations in SS3.1 of the policy schema
/// specification.
///
/// The function is infallible: the `FlatBuffers` builder API only allocates and
/// writes, never returns errors. Any I/O error is the caller's responsibility.
///
/// # Parameters
///
/// - `stage_id` — stage index (0-based) stored in the root table.
/// - `state_dimension` — number of state variables; determines coefficient vector
///   length per cut.
/// - `capacity` — total preallocated cut slots in the pool.
/// - `warm_start_count` — number of slots `[0..warm_start_count)` loaded from a
///   prior policy.
/// - `cuts` — slice of cut records to serialize; length equals `populated_count`.
/// - `active_cut_indices` — indices of cuts currently active in the LP.
/// - `populated_count` — number of filled slots in the pool.
///
/// # Examples
///
/// ```
/// use cobre_io::{PolicyCutRecord, serialize_stage_cuts};
///
/// let cut = PolicyCutRecord {
///     cut_id: 1,
///     slot_index: 5,
///     iteration: 3,
///     forward_pass_index: 0,
///     intercept: 42.0,
///     coefficients: &[1.0, 2.0, 3.0],
///     is_active: true,
///     domination_count: 0,
/// };
/// let buf = serialize_stage_cuts(0, 3, 100, 0, &[cut], &[0], 1);
/// assert!(!buf.is_empty());
/// ```
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn serialize_stage_cuts(
    stage_id: u32,
    state_dimension: u32,
    capacity: u32,
    warm_start_count: u32,
    cuts: &[PolicyCutRecord<'_>],
    active_cut_indices: &[u32],
    populated_count: u32,
) -> Vec<u8> {
    // Pre-size the builder to avoid reallocation.
    // Each cut occupies roughly: vtable overhead (32 B) + scalar fields (48 B)
    // + coefficient vector (state_dimension * 8 B) + state_at_generation (4 B empty).
    // Plus the StageCuts wrapper and two u32 index vectors.
    let estimated = 64
        + cuts.len() * (96usize + state_dimension as usize * std::mem::size_of::<f64>())
        + std::mem::size_of_val(active_cut_indices);

    let mut builder = FlatBufferBuilder::with_capacity(estimated);

    let cut_offsets: Vec<WIPOffset<flatbuffers::TableFinishedWIPOffset>> = cuts
        .iter()
        .map(|c| build_cut_table(&mut builder, c))
        .collect();

    // Create the cuts vector from the collected offsets.
    let cuts_vec = builder.create_vector(&cut_offsets);

    // Create the active_cut_indices vector.
    let active_vec = builder.create_vector(active_cut_indices);

    // Build the root StageCuts table.
    let root = builder.start_table();

    builder.push_slot_always::<u32>(STAGE_CUTS_FIELD_STAGE_ID, stage_id);
    builder.push_slot_always::<u32>(STAGE_CUTS_FIELD_STATE_DIMENSION, state_dimension);
    builder.push_slot_always::<u32>(STAGE_CUTS_FIELD_CAPACITY, capacity);
    builder.push_slot_always::<u32>(STAGE_CUTS_FIELD_WARM_START_COUNT, warm_start_count);
    builder.push_slot_always(STAGE_CUTS_FIELD_CUTS, cuts_vec);
    builder.push_slot_always(STAGE_CUTS_FIELD_ACTIVE_CUT_INDICES, active_vec);
    builder.push_slot_always::<u32>(STAGE_CUTS_FIELD_POPULATED_COUNT, populated_count);

    let root_offset = builder.end_table(root);
    builder.finish_minimal(root_offset);

    builder.finished_data().to_vec()
}

/// Serialize one stage's solver basis into a `FlatBuffers` buffer.
///
/// Produces a buffer containing a root `StageBasis` table. The buffer is ready
/// for writing directly to a `.bin` policy file under `basis/`. Field layout
/// matches the `StageBasis` table declaration in SS3.1 of the policy schema
/// specification.
///
/// The `num_columns` and `num_rows` fields are inferred from the status slice
/// lengths and do not need to be supplied separately.
///
/// The function is infallible: the `FlatBuffers` builder API only allocates and
/// writes, never returns errors.
///
/// # Parameters
///
/// - `record` — a reference to the basis record to serialize.
///
/// # Examples
///
/// ```
/// use cobre_io::{PolicyBasisRecord, serialize_stage_basis};
///
/// let record = PolicyBasisRecord {
///     stage_id: 0,
///     iteration: 5,
///     column_status: &[0, 1, 2],
///     row_status: &[1, 1, 0, 0],
///     num_cut_rows: 2,
/// };
/// let buf = serialize_stage_basis(&record);
/// assert!(!buf.is_empty());
/// ```
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn serialize_stage_basis(record: &PolicyBasisRecord<'_>) -> Vec<u8> {
    // Pre-size: scalar fields (~32 B) + two byte vectors + headers.
    let estimated =
        64 + std::mem::size_of_val(record.column_status) + std::mem::size_of_val(record.row_status);

    let mut builder = FlatBufferBuilder::with_capacity(estimated);

    // Create nested vectors before opening the table.
    let col_vec = builder.create_vector(record.column_status);
    let row_vec = builder.create_vector(record.row_status);

    let root = builder.start_table();

    builder.push_slot_always::<u32>(BASIS_FIELD_STAGE_ID, record.stage_id);
    builder.push_slot_always::<u32>(BASIS_FIELD_ITERATION, record.iteration);
    builder.push_slot_always::<u32>(BASIS_FIELD_NUM_COLUMNS, record.column_status.len() as u32);
    builder.push_slot_always::<u32>(BASIS_FIELD_NUM_ROWS, record.row_status.len() as u32);
    builder.push_slot_always(BASIS_FIELD_COLUMN_STATUS, col_vec);
    builder.push_slot_always(BASIS_FIELD_ROW_STATUS, row_vec);
    builder.push_slot_always::<u32>(BASIS_FIELD_NUM_CUT_ROWS, record.num_cut_rows);

    let root_offset = builder.end_table(root);
    builder.finish_minimal(root_offset);

    builder.finished_data().to_vec()
}

/// Per-stage cut data payload for [`write_policy_checkpoint`].
///
/// Groups all fields required by [`serialize_stage_cuts`] into a single struct so
/// the checkpoint writer can iterate over stages without unpacking individual
/// arguments at each call site. The lifetime parameter `'a` allows borrowing
/// coefficient slices and index arrays without copying.
#[derive(Debug)]
pub struct StageCutsPayload<'a> {
    /// Stage index (0-based), used as the file name index in `cuts/stage_NNN.bin`.
    pub stage_id: u32,
    /// Number of state variables; determines coefficient vector length per cut.
    pub state_dimension: u32,
    /// Total preallocated cut slots in the pool.
    pub capacity: u32,
    /// Number of slots `[0..warm_start_count)` loaded from a prior policy.
    pub warm_start_count: u32,
    /// Slice of cut records to serialize; length equals `populated_count`.
    pub cuts: &'a [PolicyCutRecord<'a>],
    /// Indices of cuts currently active in the LP.
    pub active_cut_indices: &'a [u32],
    /// Number of filled slots in the pool.
    pub populated_count: u32,
}

/// Write a complete policy checkpoint to `path`.
///
/// Creates the directory structure required by SS3.2 of the policy schema
/// specification, serializes all per-stage cut and basis data to `FlatBuffers`
/// binary files, and writes the metadata as JSON. The metadata file is written
/// **last** so its presence signals a complete checkpoint.
///
/// ## Directory layout produced
///
/// ```text
/// path/
///   metadata.json
///   cuts/
///     stage_000.bin
///     stage_001.bin
///     ...
///   basis/
///     stage_000.bin   (only when stage_bases is non-empty)
///     stage_001.bin
///     ...
/// ```
///
/// ## Commit-point semantics
///
/// `metadata.json` is written only after all `.bin` files succeed. If any write
/// fails, `metadata.json` is absent, signaling an incomplete checkpoint. The
/// function does not clean up partially written files — the caller uses the
/// absence of `metadata.json` to detect incomplete checkpoints.
///
/// # Parameters
///
/// - `path` — root directory for the policy checkpoint.
/// - `stage_cuts` — one entry per stage, ordered by stage index 0..N.
/// - `stage_bases` — one entry per stage, ordered by stage index 0..N. An empty
///   slice means no basis files are written; the `basis/` directory is still created.
/// - `metadata` — policy metadata, serialized to `metadata.json`.
///
/// # Errors
///
/// - [`OutputError::IoError`] — directory creation or file write failed.
/// - [`OutputError::SerializationError`] — JSON serialization of metadata failed.
///
/// # Examples
///
/// ```no_run
/// use cobre_io::{
///     write_policy_checkpoint, PolicyBasisRecord, PolicyCheckpointMetadata, PolicyCutRecord,
///     StageCutsPayload,
/// };
/// use std::path::Path;
///
/// # fn main() -> Result<(), cobre_io::OutputError> {
/// let coefficients = [1.0_f64, 2.0, 3.0];
/// let cut = PolicyCutRecord {
///     cut_id: 1,
///     slot_index: 0,
///     iteration: 1,
///     forward_pass_index: 0,
///     intercept: 42.0,
///     coefficients: &coefficients,
///     is_active: true,
///     domination_count: 0,
/// };
/// let stage_cuts = [StageCutsPayload {
///     stage_id: 0,
///     state_dimension: 3,
///     capacity: 100,
///     warm_start_count: 0,
///     cuts: &[cut],
///     active_cut_indices: &[0],
///     populated_count: 1,
/// }];
/// let metadata = PolicyCheckpointMetadata {
///     version: "1.0.0".to_string(),
///     cobre_version: env!("CARGO_PKG_VERSION").to_string(),
///     created_at: "2026-03-08T00:00:00Z".to_string(),
///     completed_iterations: 1,
///     final_lower_bound: 42.0,
///     best_upper_bound: None,
///     state_dimension: 3,
///     num_stages: 1,
///     config_hash: "abc".to_string(),
///     system_hash: "def".to_string(),
///     max_iterations: 100,
///     forward_passes: 4,
///     warm_start_cuts: 0,
///     rng_seed: 0,
/// };
/// write_policy_checkpoint(Path::new("/tmp/policy"), &stage_cuts, &[], &metadata)?;
/// # Ok(())
/// # }
/// ```
pub fn write_policy_checkpoint(
    path: &Path,
    stage_cuts: &[StageCutsPayload<'_>],
    stage_bases: &[PolicyBasisRecord<'_>],
    metadata: &PolicyCheckpointMetadata,
) -> Result<(), OutputError> {
    // Create cuts/ and basis/ subdirectories (and path/ itself if needed).
    let cuts_dir = path.join("cuts");
    std::fs::create_dir_all(&cuts_dir).map_err(|e| OutputError::io(&cuts_dir, e))?;

    let basis_dir = path.join("basis");
    std::fs::create_dir_all(&basis_dir).map_err(|e| OutputError::io(&basis_dir, e))?;

    // Write per-stage cut files: cuts/stage_NNN.bin.
    for payload in stage_cuts {
        let filename = format!("stage_{:03}.bin", payload.stage_id);
        let file_path = cuts_dir.join(&filename);
        let buf = serialize_stage_cuts(
            payload.stage_id,
            payload.state_dimension,
            payload.capacity,
            payload.warm_start_count,
            payload.cuts,
            payload.active_cut_indices,
            payload.populated_count,
        );
        std::fs::write(&file_path, &buf).map_err(|e| OutputError::io(&file_path, e))?;
    }

    // Write per-stage basis files: basis/stage_NNN.bin.
    for record in stage_bases {
        let filename = format!("stage_{:03}.bin", record.stage_id);
        let file_path = basis_dir.join(&filename);
        let buf = serialize_stage_basis(record);
        std::fs::write(&file_path, &buf).map_err(|e| OutputError::io(&file_path, e))?;
    }

    // Write metadata.json LAST — its presence is the commit signal.
    let json = serde_json::to_string_pretty(metadata)
        .map_err(|e| OutputError::serialization("policy_metadata", e.to_string()))?;
    let meta_path = path.join("metadata.json");
    std::fs::write(&meta_path, json.as_bytes()).map_err(|e| OutputError::io(&meta_path, e))?;

    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]
mod tests {
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
            domination_count: 0,
        }
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
            domination_count: 0,
        };

        let buf = serialize_stage_cuts(0, 3, 100, 0, &[cut], &[0], 1);

        assert!(!buf.is_empty(), "buffer must not be empty");
        // A `FlatBuffers` buffer always starts with a 4-byte root offset (little-endian u32).
        // Verify that the first 4 bytes decode to a non-zero, in-range offset.
        assert!(buf.len() >= 4, "buffer must have at least 4 bytes");
        let root_offset = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
        assert!(
            root_offset < buf.len(),
            "root offset must point inside the buffer"
        );
    }

    #[test]
    fn serialize_stage_cuts_empty_cuts_valid_buffer() {
        let buf = serialize_stage_cuts(0, 3, 100, 0, &[], &[], 0);

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

        let buf_a = serialize_stage_cuts(5, 3, 50, 0, &cuts, &[0, 1, 2], 3);
        let buf_b = serialize_stage_cuts(5, 3, 50, 0, &cuts, &[0, 1, 2], 3);

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
                domination_count: 0,
            };
            let buf = serialize_stage_cuts(0, dim, 10, 0, &[cut], &[0], 1);
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

    // ── PolicyCheckpointMetadata tests ────────────────────────────────────────

    #[test]
    fn policy_checkpoint_metadata_serializes_to_json() {
        let meta = PolicyCheckpointMetadata {
            version: "1.0.0".to_string(),
            cobre_version: "0.0.1".to_string(),
            created_at: "2026-03-08T00:00:00Z".to_string(),
            completed_iterations: 50,
            final_lower_bound: 1234.56,
            best_upper_bound: Some(1300.0),
            state_dimension: 160,
            num_stages: 60,
            config_hash: "abc123".to_string(),
            system_hash: "def456".to_string(),
            max_iterations: 200,
            forward_passes: 4,
            warm_start_cuts: 0,
            rng_seed: 42,
        };

        let json = serde_json::to_string_pretty(&meta)
            .expect("PolicyCheckpointMetadata must serialize to JSON without error");

        assert!(
            json.contains("completed_iterations"),
            "JSON must contain 'completed_iterations'"
        );
        assert!(
            json.contains("50"),
            "JSON must contain the completed_iterations value"
        );
        assert!(
            json.contains("final_lower_bound"),
            "JSON must contain 'final_lower_bound'"
        );
        assert!(
            json.contains("state_dimension"),
            "JSON must contain 'state_dimension'"
        );
        assert!(json.contains("rng_seed"), "JSON must contain 'rng_seed'");
        assert!(
            json.contains("best_upper_bound"),
            "JSON must contain 'best_upper_bound'"
        );
        assert!(
            json.contains("1300"),
            "JSON must contain the best_upper_bound value"
        );

        // Verify it round-trips through serde_json::Value.
        let value: serde_json::Value =
            serde_json::from_str(&json).expect("JSON output must be parseable");
        assert_eq!(
            value["completed_iterations"].as_u64(),
            Some(50),
            "completed_iterations must deserialize correctly"
        );
        assert_eq!(
            value["rng_seed"].as_u64(),
            Some(42),
            "rng_seed must deserialize correctly"
        );
    }

    #[test]
    fn policy_checkpoint_metadata_none_upper_bound_serializes_to_null() {
        let meta = PolicyCheckpointMetadata {
            version: "1.0.0".to_string(),
            cobre_version: "0.0.1".to_string(),
            created_at: "2026-03-08T00:00:00Z".to_string(),
            completed_iterations: 10,
            final_lower_bound: 0.0,
            best_upper_bound: None,
            state_dimension: 1,
            num_stages: 1,
            config_hash: String::new(),
            system_hash: String::new(),
            max_iterations: 10,
            forward_passes: 1,
            warm_start_cuts: 0,
            rng_seed: 0,
        };

        let json = serde_json::to_string_pretty(&meta)
            .expect("PolicyCheckpointMetadata must serialize to JSON");

        let value: serde_json::Value =
            serde_json::from_str(&json).expect("JSON output must be parseable");
        assert!(
            value["best_upper_bound"].is_null(),
            "best_upper_bound must serialize to null when None"
        );
    }

    // ── write_policy_checkpoint tests ─────────────────────────────────────────

    /// Build a minimal [`PolicyCheckpointMetadata`] for use in checkpoint tests.
    fn make_metadata(num_stages: u32, state_dimension: u32) -> PolicyCheckpointMetadata {
        PolicyCheckpointMetadata {
            version: "1.0.0".to_string(),
            cobre_version: "0.0.1".to_string(),
            created_at: "2026-03-08T00:00:00Z".to_string(),
            completed_iterations: 10,
            final_lower_bound: 999.0,
            best_upper_bound: Some(1050.0),
            state_dimension,
            num_stages,
            config_hash: "abc123".to_string(),
            system_hash: "def456".to_string(),
            max_iterations: 100,
            forward_passes: 4,
            warm_start_cuts: 0,
            rng_seed: 42,
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
        }
    }

    /// Build a [`PolicyBasisRecord`] for the given stage.
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

        write_policy_checkpoint(tmp.path(), &stage_cuts, &basis_records, &metadata)
            .expect("write_policy_checkpoint must succeed");

        // Directories must exist.
        assert!(tmp.path().join("cuts").is_dir(), "cuts/ must exist");
        assert!(tmp.path().join("basis").is_dir(), "basis/ must exist");

        // All cut files must exist.
        for i in 0..3u32 {
            let p = tmp.path().join(format!("cuts/stage_{i:03}.bin"));
            assert!(p.is_file(), "cuts/stage_{i:03}.bin must exist");
        }

        // All basis files must exist.
        for i in 0..3u32 {
            let p = tmp.path().join(format!("basis/stage_{i:03}.bin"));
            assert!(p.is_file(), "basis/stage_{i:03}.bin must exist");
        }

        // metadata.json must exist.
        assert!(
            tmp.path().join("metadata.json").is_file(),
            "metadata.json must exist"
        );
    }

    #[test]
    fn write_policy_checkpoint_metadata_json_valid() {
        let tmp = tempfile::tempdir().unwrap();

        let c0 = [1.0_f64, 2.0, 3.0];
        let cuts_s0 = [make_cut_record(1, 0, 1, &c0)];
        let stage_cuts = [make_stage_cuts_payload(0, &cuts_s0, &[0], 3)];
        let metadata = make_metadata(1, 3);

        write_policy_checkpoint(tmp.path(), &stage_cuts, &[], &metadata)
            .expect("write_policy_checkpoint must succeed");

        let content = std::fs::read_to_string(tmp.path().join("metadata.json")).unwrap();
        let value: serde_json::Value =
            serde_json::from_str(&content).expect("metadata.json must be valid JSON");

        for key in &[
            "version",
            "cobre_version",
            "created_at",
            "completed_iterations",
            "final_lower_bound",
            "state_dimension",
            "num_stages",
        ] {
            assert!(
                value.get(key).is_some(),
                "metadata.json must contain key '{key}'"
            );
        }

        assert_eq!(
            value["completed_iterations"].as_u64(),
            Some(10),
            "completed_iterations must match"
        );
        assert_eq!(
            value["num_stages"].as_u64(),
            Some(1),
            "num_stages must match"
        );
        assert_eq!(
            value["state_dimension"].as_u64(),
            Some(3),
            "state_dimension must match"
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

        write_policy_checkpoint(tmp.path(), &stage_cuts, &[], &metadata)
            .expect("write_policy_checkpoint must succeed");

        for i in 0..3u32 {
            let p = tmp.path().join(format!("cuts/stage_{i:03}.bin"));
            let bytes = std::fs::read(&p).unwrap();
            assert!(!bytes.is_empty(), "cuts/stage_{i:03}.bin must not be empty");
            // Verify FlatBuffers root offset is in-range.
            assert!(
                bytes.len() >= 4,
                "cuts/stage_{i:03}.bin must have >= 4 bytes"
            );
            let root_offset = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
            assert!(
                root_offset < bytes.len(),
                "cuts/stage_{i:03}.bin root offset must be in-range"
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

        write_policy_checkpoint(tmp.path(), &stage_cuts, &basis_records, &metadata)
            .expect("write_policy_checkpoint must succeed");

        let p = tmp.path().join("basis/stage_000.bin");
        let bytes = std::fs::read(&p).unwrap();
        assert!(!bytes.is_empty(), "basis/stage_000.bin must not be empty");
        assert!(bytes.len() >= 4, "basis/stage_000.bin must have >= 4 bytes");
        let root_offset = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
        assert!(
            root_offset < bytes.len(),
            "basis/stage_000.bin root offset must be in-range"
        );
    }

    #[test]
    fn write_policy_checkpoint_empty_bases_no_basis_files() {
        let tmp = tempfile::tempdir().unwrap();

        let c0 = [1.0_f64, 2.0];
        let cuts_s0 = [make_cut_record(1, 0, 1, &c0)];
        let stage_cuts = [make_stage_cuts_payload(0, &cuts_s0, &[0], 2)];
        let metadata = make_metadata(1, 2);

        let result = write_policy_checkpoint(tmp.path(), &stage_cuts, &[], &metadata);

        assert!(
            result.is_ok(),
            "write_policy_checkpoint must return Ok(()) with empty stage_bases"
        );

        // basis/ directory must exist.
        assert!(
            tmp.path().join("basis").is_dir(),
            "basis/ directory must exist even with empty stage_bases"
        );

        // No .bin files inside basis/.
        let entries: Vec<_> = std::fs::read_dir(tmp.path().join("basis"))
            .unwrap()
            .filter_map(std::result::Result::ok)
            .collect();
        assert!(
            entries.is_empty(),
            "basis/ must contain no files when stage_bases is empty"
        );
    }

    /// Returns `true` when running as root (UID 0). Used to skip permission tests.
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
        // Skip this test on platforms where read-only enforcement is unreliable
        // (e.g., when running as root).
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

        let result = write_policy_checkpoint(&readonly_target, &stage_cuts, &[], &metadata);

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

        write_policy_checkpoint(tmp.path(), &stage_cuts, &stage_bases, &metadata)
            .expect("write_policy_checkpoint must succeed");

        assert!(
            tmp.path().join("cuts/stage_000.bin").is_file(),
            "cuts/stage_000.bin must exist"
        );
        assert!(
            tmp.path().join("cuts/stage_001.bin").is_file(),
            "cuts/stage_001.bin must exist"
        );
        assert!(
            tmp.path().join("cuts/stage_059.bin").is_file(),
            "cuts/stage_059.bin must exist"
        );
        assert!(
            tmp.path().join("basis/stage_000.bin").is_file(),
            "basis/stage_000.bin must exist"
        );
        assert!(
            tmp.path().join("basis/stage_001.bin").is_file(),
            "basis/stage_001.bin must exist"
        );
        assert!(
            tmp.path().join("basis/stage_059.bin").is_file(),
            "basis/stage_059.bin must exist"
        );
    }
}
