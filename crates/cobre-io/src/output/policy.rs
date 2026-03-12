//! `FlatBuffers` builder and reader types for policy checkpoint serialization.
//!
//! This module defines the input types and builder functions used to serialize policy
//! data (cuts and solver bases) to `FlatBuffers` binary format for checkpoint persistence,
//! as well as owned output types and reader functions for deserializing those buffers.
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
//!
//! ## Reading policy checkpoints
//!
//! The reader functions ([`deserialize_stage_cuts`], [`deserialize_stage_basis`],
//! [`read_policy_checkpoint`]) use **safe raw byte parsing** of the `FlatBuffers` wire
//! format instead of the generated `Table::get` API (which is `unsafe fn`). This is
//! required because the workspace forbids `unsafe_code`.
//!
//! ### Wire format summary
//!
//! A finished `FlatBuffers` buffer layout (produced by [`flatbuffers::FlatBufferBuilder::finish_minimal`]):
//!
//! ```text
//! offset 0: u32 root_offset — byte offset from position 0 to the root table
//! ...data written right-to-left by the builder...
//! vtable:
//!   u16 vtable_bytesize
//!   u16 table_data_bytesize
//!   u16 field_0_data_offset   (or 0 if field absent)
//!   u16 field_1_data_offset   ...
//! root_table at root_offset:
//!   i32 soffset_to_vtable     (= table_pos - vtable_pos; positive = vtable before table)
//!   ...inline field data...
//! ```
//!
//! Nested table fields and vector fields store a `u32` forward offset from the field's
//! own buffer position to the nested object. Vectors are: `u32 length`, then
//! `length × element_size` bytes of element data.

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
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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

// ── Owned output types for deserialization ───────────────────────────────────

/// Owned version of [`PolicyCutRecord`] returned by [`deserialize_stage_cuts`].
///
/// Unlike [`PolicyCutRecord<'a>`], this type owns its `coefficients` vector so it
/// can be returned from a deserialization function that does not borrow from the
/// input buffer.
#[derive(Debug, Clone)]
pub struct OwnedPolicyCutRecord {
    /// Unique identifier for this cut across all iterations.
    pub cut_id: u64,
    /// LP row position (required for checkpoint reproducibility).
    pub slot_index: u32,
    /// Training iteration that generated this cut.
    pub iteration: u32,
    /// Forward pass index within the generating iteration.
    pub forward_pass_index: u32,
    /// Pre-computed cut intercept.
    pub intercept: f64,
    /// Gradient coefficient vector, length equals `state_dimension` of the stage.
    pub coefficients: Vec<f64>,
    /// Whether this cut is currently active in the LP.
    pub is_active: bool,
    /// Domination count for cut selection bookkeeping.
    pub domination_count: u32,
}

/// Owned version of [`PolicyBasisRecord`] returned by [`deserialize_stage_basis`].
///
/// Unlike [`PolicyBasisRecord<'a>`], this type owns its status byte vectors so it
/// can be returned from a deserialization function.
#[derive(Debug, Clone)]
pub struct OwnedPolicyBasisRecord {
    /// Stage index (0-based).
    pub stage_id: u32,
    /// Training iteration that produced this basis.
    pub iteration: u32,
    /// One status code per LP column (variable). Encoding is solver-specific.
    pub column_status: Vec<u8>,
    /// One status code per LP row (constraint). Encoding is solver-specific.
    pub row_status: Vec<u8>,
    /// Number of trailing rows in `row_status` that correspond to cut rows.
    pub num_cut_rows: u32,
}

/// Stage-level metadata and cut records returned by [`deserialize_stage_cuts`].
///
/// Contains the stage-level fields stored in the `StageCuts` root table plus the
/// vector of deserialized cut records.
#[derive(Debug, Clone)]
pub struct StageCutsReadResult {
    /// Stage index (0-based).
    pub stage_id: u32,
    /// Number of state variables; equals the length of each cut's `coefficients` vector.
    pub state_dimension: u32,
    /// Total preallocated cut slots in the pool.
    pub capacity: u32,
    /// Number of slots loaded from a prior policy.
    pub warm_start_count: u32,
    /// Number of filled slots in the pool.
    pub populated_count: u32,
    /// Deserialized cut records.
    pub cuts: Vec<OwnedPolicyCutRecord>,
}

/// Complete deserialized policy checkpoint returned by [`read_policy_checkpoint`].
#[derive(Debug, Clone)]
pub struct PolicyCheckpoint {
    /// Policy metadata read from `metadata.json`.
    pub metadata: PolicyCheckpointMetadata,
    /// Per-stage cut pools, sorted by `stage_id`.
    pub stage_cuts: Vec<StageCutsReadResult>,
    /// Per-stage solver bases, sorted by `stage_id`.
    pub stage_bases: Vec<OwnedPolicyBasisRecord>,
}

// ── Safe FlatBuffers wire-format helpers ─────────────────────────────────────
//
// All helpers return `Option` so callers can propagate truncation / corruption
// errors without panicking. The `resolve_*` functions follow the FlatBuffers
// specification exactly:
//
//   Buffer layout (finish_minimal):
//     bytes[0..4]  = u32 LE root_offset — byte offset from position 0 to root table
//     ...builder data (written right-to-left)...
//     vtable  = [u16 vtable_size][u16 table_size][u16 field0][u16 field1]...
//     table   = [i32 soffset_to_vtable][...inline field data...]
//
//   soffset_to_vtable at table_pos:
//     vtable_pos = table_pos - (i32 at table_pos)
//
//   Field data for field with vtable slot `slot`:
//     field_data_offset_from_table_start = u16 at vtable[slot]
//     (0 means field absent)
//     actual data at: table_pos + field_data_offset_from_table_start
//
//   Nested table / vector fields store a u32 forward uoffset at their data position:
//     nested_pos = field_data_pos + u32_at(field_data_pos)

#[inline]
fn read_u16_le(buf: &[u8], offset: usize) -> Option<u16> {
    let bytes = buf.get(offset..offset.checked_add(2)?)?;
    Some(u16::from_le_bytes([bytes[0], bytes[1]]))
}

#[inline]
fn read_i32_le(buf: &[u8], offset: usize) -> Option<i32> {
    let bytes = buf.get(offset..offset.checked_add(4)?)?;
    Some(i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

#[inline]
fn read_u32_le(buf: &[u8], offset: usize) -> Option<u32> {
    let bytes = buf.get(offset..offset.checked_add(4)?)?;
    Some(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

#[inline]
fn read_u64_le(buf: &[u8], offset: usize) -> Option<u64> {
    let bytes = buf.get(offset..offset.checked_add(8)?)?;
    Some(u64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ]))
}

#[inline]
fn read_f64_le(buf: &[u8], offset: usize) -> Option<f64> {
    read_u64_le(buf, offset).map(f64::from_bits)
}

#[inline]
fn read_bool_byte(buf: &[u8], offset: usize) -> Option<bool> {
    buf.get(offset).map(|&b| b != 0)
}

/// Resolve the root table position from a finished `FlatBuffers` buffer.
///
/// Returns the byte offset of the root table within `buf`.
fn resolve_root(buf: &[u8]) -> Option<usize> {
    let offset = read_u32_le(buf, 0)? as usize;
    // The root offset must point inside the buffer (at minimum for the soffset).
    if offset.checked_add(4)? > buf.len() {
        return None;
    }
    Some(offset)
}

/// Resolve the vtable position for the table at `table_pos`.
///
/// Returns the byte offset of the vtable within `buf`.
fn resolve_vtable_pos(buf: &[u8], table_pos: usize) -> Option<usize> {
    let soffset = read_i32_le(buf, table_pos)?;
    // vtable_pos = table_pos - soffset (soffset is signed; positive = vtable before table).
    // Avoid lossy `as i64` casts (clippy::cast_possible_wrap / cast_possible_truncation).
    let vtable_pos = if soffset >= 0 {
        // Vtable precedes the table: table_pos - soffset (as a non-negative offset).
        table_pos.checked_sub(u32::try_from(soffset).ok()? as usize)?
    } else {
        // Vtable follows the table: table_pos + |soffset|.
        let abs = u32::try_from(soffset.wrapping_neg()).ok()? as usize;
        table_pos.checked_add(abs)?
    };
    if vtable_pos.checked_add(4)? > buf.len() {
        return None;
    }
    Some(vtable_pos)
}

/// Read the data offset for field slot `slot` from the vtable at `vtable_pos`.
///
/// Returns `None` if the slot is beyond the vtable, or `Some(0)` if the field
/// is absent (the `FlatBuffers` convention for optional fields).
fn field_data_offset(buf: &[u8], vtable_pos: usize, slot: u16) -> Option<u16> {
    let vtable_size = read_u16_le(buf, vtable_pos)?;
    let slot_pos = vtable_pos.checked_add(slot as usize)?;
    if slot_pos.checked_add(2)? > vtable_pos.checked_add(vtable_size as usize)? {
        // Slot is past end of vtable — field was added in a later schema version.
        return Some(0);
    }
    read_u16_le(buf, slot_pos)
}

/// Resolve the absolute position of field `slot` data in a table at `table_pos`.
///
/// Returns `None` if the field is absent (vtable offset is 0) or if the buffer
/// is truncated.
fn field_pos(buf: &[u8], table_pos: usize, vtable_pos: usize, slot: u16) -> Option<usize> {
    let data_off = field_data_offset(buf, vtable_pos, slot)?;
    if data_off == 0 {
        return None; // field absent
    }
    table_pos.checked_add(data_off as usize)
}

/// Follow a `uoffset` stored at `pos` to reach a nested table or vector.
///
/// `FlatBuffers` stores forward offsets: the referenced object is at
/// `pos + u32_at(pos)`. The offset is relative to the position of the u32 itself.
fn follow_uoffset(buf: &[u8], pos: usize) -> Option<usize> {
    let off = read_u32_le(buf, pos)?;
    pos.checked_add(off as usize)
}

/// Read a `f32` vector stored at `vec_pos` and return its elements as `f64`.
///
/// `FlatBuffers` vector layout: `u32 length` followed by `length × 4` bytes.
/// This function is not used currently but kept for completeness.
#[allow(dead_code)]
fn read_f32_vector_as_f64(buf: &[u8], vec_pos: usize) -> Option<Vec<f64>> {
    let len = read_u32_le(buf, vec_pos)? as usize;
    let data_start = vec_pos.checked_add(4)?;
    let data_end = data_start.checked_add(len.checked_mul(4)?)?;
    if data_end > buf.len() {
        return None;
    }
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        let pos = data_start + i * 4;
        let bits = u32::from_le_bytes([buf[pos], buf[pos + 1], buf[pos + 2], buf[pos + 3]]);
        out.push(f64::from(f32::from_bits(bits)));
    }
    Some(out)
}

/// Read a `f64` vector stored at `vec_pos`.
///
/// `FlatBuffers` vector layout: `u32 length` followed by `length × 8` bytes.
fn read_f64_vector(buf: &[u8], vec_pos: usize) -> Option<Vec<f64>> {
    let len = read_u32_le(buf, vec_pos)? as usize;
    let data_start = vec_pos.checked_add(4)?;
    let data_end = data_start.checked_add(len.checked_mul(8)?)?;
    if data_end > buf.len() {
        return None;
    }
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        let pos = data_start + i * 8;
        out.push(read_f64_le(buf, pos)?);
    }
    Some(out)
}

/// Read a `u8` vector stored at `vec_pos`.
///
/// `FlatBuffers` vector layout: `u32 length` followed by `length × 1` bytes.
fn read_u8_vector(buf: &[u8], vec_pos: usize) -> Option<Vec<u8>> {
    let len = read_u32_le(buf, vec_pos)? as usize;
    let data_start = vec_pos.checked_add(4)?;
    let data_end = data_start.checked_add(len)?;
    if data_end > buf.len() {
        return None;
    }
    Some(buf[data_start..data_end].to_vec())
}

/// Read a vector of nested tables stored at `vec_pos`.
///
/// Returns a `Vec` of absolute buffer positions, one per element. Each element
/// stores a `u32` uoffset from its own position to the nested table.
fn read_table_vector_positions(buf: &[u8], vec_pos: usize) -> Option<Vec<usize>> {
    let len = read_u32_le(buf, vec_pos)? as usize;
    let data_start = vec_pos.checked_add(4)?;
    let data_end = data_start.checked_add(len.checked_mul(4)?)?;
    if data_end > buf.len() {
        return None;
    }
    let mut positions = Vec::with_capacity(len);
    for i in 0..len {
        let elem_pos = data_start + i * 4;
        let nested_pos = follow_uoffset(buf, elem_pos)?;
        positions.push(nested_pos);
    }
    Some(positions)
}

// ── Deserializers ─────────────────────────────────────────────────────────────

/// Deserialize a `StageCuts` `FlatBuffers` buffer into an owned [`StageCutsReadResult`].
///
/// Reads the root `StageCuts` table and each nested `BendersCut` table using safe
/// raw byte parsing of the `FlatBuffers` wire format. No `unsafe` code is used.
///
/// # Errors
///
/// Returns [`OutputError::SerializationError`] if the buffer is truncated, corrupted,
/// or otherwise does not conform to the expected layout.
///
/// # Examples
///
/// ```
/// use cobre_io::{PolicyCutRecord, serialize_stage_cuts, deserialize_stage_cuts};
///
/// let cut = PolicyCutRecord {
///     cut_id: 7,
///     slot_index: 5,
///     iteration: 3,
///     forward_pass_index: 1,
///     intercept: 42.0,
///     coefficients: &[1.0, 2.0, 3.0],
///     is_active: true,
///     domination_count: 0,
/// };
/// let buf = serialize_stage_cuts(2, 3, 100, 0, &[cut], &[0], 1);
/// let result = deserialize_stage_cuts(&buf).expect("round-trip must succeed");
/// assert_eq!(result.stage_id, 2);
/// assert_eq!(result.cuts.len(), 1);
/// assert_eq!(result.cuts[0].cut_id, 7);
/// assert_eq!(result.cuts[0].coefficients, &[1.0, 2.0, 3.0]);
/// ```
pub fn deserialize_stage_cuts(buf: &[u8]) -> Result<StageCutsReadResult, OutputError> {
    let ctx = "stage_cuts";

    let table_pos = resolve_root(buf)
        .ok_or_else(|| OutputError::serialization(ctx, "buffer too short for root offset"))?;

    let vtable_pos = resolve_vtable_pos(buf, table_pos)
        .ok_or_else(|| OutputError::serialization(ctx, "invalid soffset_to_vtable"))?;

    // Read scalar fields from StageCuts root table.
    let stage_id = field_pos(buf, table_pos, vtable_pos, STAGE_CUTS_FIELD_STAGE_ID)
        .and_then(|p| read_u32_le(buf, p))
        .unwrap_or(0);

    let state_dimension = field_pos(buf, table_pos, vtable_pos, STAGE_CUTS_FIELD_STATE_DIMENSION)
        .and_then(|p| read_u32_le(buf, p))
        .unwrap_or(0);

    let capacity = field_pos(buf, table_pos, vtable_pos, STAGE_CUTS_FIELD_CAPACITY)
        .and_then(|p| read_u32_le(buf, p))
        .unwrap_or(0);

    let warm_start_count = field_pos(
        buf,
        table_pos,
        vtable_pos,
        STAGE_CUTS_FIELD_WARM_START_COUNT,
    )
    .and_then(|p| read_u32_le(buf, p))
    .unwrap_or(0);

    let populated_count = field_pos(buf, table_pos, vtable_pos, STAGE_CUTS_FIELD_POPULATED_COUNT)
        .and_then(|p| read_u32_le(buf, p))
        .unwrap_or(0);

    // Read the cuts vector of nested tables.
    let cuts = if let Some(cuts_field_pos) =
        field_pos(buf, table_pos, vtable_pos, STAGE_CUTS_FIELD_CUTS)
    {
        let vec_pos = follow_uoffset(buf, cuts_field_pos)
            .ok_or_else(|| OutputError::serialization(ctx, "invalid uoffset for cuts vector"))?;

        let nested_positions = read_table_vector_positions(buf, vec_pos).ok_or_else(|| {
            OutputError::serialization(ctx, "cuts vector header truncated or corrupt")
        })?;

        let mut out = Vec::with_capacity(nested_positions.len());
        for (idx, &cut_table_pos) in nested_positions.iter().enumerate() {
            let cut = deserialize_cut_table(buf, cut_table_pos).ok_or_else(|| {
                OutputError::serialization(ctx, format!("cut table {idx} truncated or corrupt"))
            })?;
            out.push(cut);
        }
        out
    } else {
        Vec::new()
    };

    Ok(StageCutsReadResult {
        stage_id,
        state_dimension,
        capacity,
        warm_start_count,
        populated_count,
        cuts,
    })
}

/// Deserialize a single `BendersCut` nested table at `cut_table_pos`.
fn deserialize_cut_table(buf: &[u8], cut_table_pos: usize) -> Option<OwnedPolicyCutRecord> {
    let vtable_pos = resolve_vtable_pos(buf, cut_table_pos)?;

    let cut_id = field_pos(buf, cut_table_pos, vtable_pos, CUT_FIELD_CUT_ID)
        .and_then(|p| read_u64_le(buf, p))
        .unwrap_or(0);

    let slot_index = field_pos(buf, cut_table_pos, vtable_pos, CUT_FIELD_SLOT_INDEX)
        .and_then(|p| read_u32_le(buf, p))
        .unwrap_or(0);

    let iteration = field_pos(buf, cut_table_pos, vtable_pos, CUT_FIELD_ITERATION)
        .and_then(|p| read_u32_le(buf, p))
        .unwrap_or(0);

    let forward_pass_index = field_pos(buf, cut_table_pos, vtable_pos, CUT_FIELD_FORWARD_PASS_IDX)
        .and_then(|p| read_u32_le(buf, p))
        .unwrap_or(0);

    let intercept = field_pos(buf, cut_table_pos, vtable_pos, CUT_FIELD_INTERCEPT)
        .and_then(|p| read_f64_le(buf, p))
        .unwrap_or(0.0);

    let coefficients = if let Some(coeff_field_pos) =
        field_pos(buf, cut_table_pos, vtable_pos, CUT_FIELD_COEFFICIENTS)
    {
        let vec_pos = follow_uoffset(buf, coeff_field_pos)?;
        read_f64_vector(buf, vec_pos)?
    } else {
        Vec::new()
    };

    let is_active = field_pos(buf, cut_table_pos, vtable_pos, CUT_FIELD_IS_ACTIVE)
        .and_then(|p| read_bool_byte(buf, p))
        .unwrap_or(false);

    let domination_count = field_pos(buf, cut_table_pos, vtable_pos, CUT_FIELD_DOMINATION_COUNT)
        .and_then(|p| read_u32_le(buf, p))
        .unwrap_or(0);

    Some(OwnedPolicyCutRecord {
        cut_id,
        slot_index,
        iteration,
        forward_pass_index,
        intercept,
        coefficients,
        is_active,
        domination_count,
    })
}

/// Deserialize a `StageBasis` `FlatBuffers` buffer into an owned [`OwnedPolicyBasisRecord`].
///
/// Reads the root `StageBasis` table using safe raw byte parsing. No `unsafe` code is used.
///
/// # Errors
///
/// Returns [`OutputError::SerializationError`] if the buffer is truncated, corrupted,
/// or otherwise does not conform to the expected layout.
///
/// # Examples
///
/// ```
/// use cobre_io::{PolicyBasisRecord, serialize_stage_basis, deserialize_stage_basis};
///
/// let record = PolicyBasisRecord {
///     stage_id: 0,
///     iteration: 5,
///     column_status: &[0, 1, 2],
///     row_status: &[1, 1, 0, 0],
///     num_cut_rows: 2,
/// };
/// let buf = serialize_stage_basis(&record);
/// let owned = deserialize_stage_basis(&buf).expect("round-trip must succeed");
/// assert_eq!(owned.stage_id, 0);
/// assert_eq!(owned.column_status, &[0, 1, 2]);
/// assert_eq!(owned.row_status, &[1, 1, 0, 0]);
/// ```
pub fn deserialize_stage_basis(buf: &[u8]) -> Result<OwnedPolicyBasisRecord, OutputError> {
    let ctx = "stage_basis";

    let table_pos = resolve_root(buf)
        .ok_or_else(|| OutputError::serialization(ctx, "buffer too short for root offset"))?;

    let vtable_pos = resolve_vtable_pos(buf, table_pos)
        .ok_or_else(|| OutputError::serialization(ctx, "invalid soffset_to_vtable"))?;

    let stage_id = field_pos(buf, table_pos, vtable_pos, BASIS_FIELD_STAGE_ID)
        .and_then(|p| read_u32_le(buf, p))
        .unwrap_or(0);

    let iteration = field_pos(buf, table_pos, vtable_pos, BASIS_FIELD_ITERATION)
        .and_then(|p| read_u32_le(buf, p))
        .unwrap_or(0);

    let column_status = if let Some(col_field_pos) =
        field_pos(buf, table_pos, vtable_pos, BASIS_FIELD_COLUMN_STATUS)
    {
        let vec_pos = follow_uoffset(buf, col_field_pos).ok_or_else(|| {
            OutputError::serialization(ctx, "invalid uoffset for column_status vector")
        })?;
        read_u8_vector(buf, vec_pos)
            .ok_or_else(|| OutputError::serialization(ctx, "column_status vector truncated"))?
    } else {
        Vec::new()
    };

    let row_status = if let Some(row_field_pos) =
        field_pos(buf, table_pos, vtable_pos, BASIS_FIELD_ROW_STATUS)
    {
        let vec_pos = follow_uoffset(buf, row_field_pos).ok_or_else(|| {
            OutputError::serialization(ctx, "invalid uoffset for row_status vector")
        })?;
        read_u8_vector(buf, vec_pos)
            .ok_or_else(|| OutputError::serialization(ctx, "row_status vector truncated"))?
    } else {
        Vec::new()
    };

    let num_cut_rows = field_pos(buf, table_pos, vtable_pos, BASIS_FIELD_NUM_CUT_ROWS)
        .and_then(|p| read_u32_le(buf, p))
        .unwrap_or(0);

    Ok(OwnedPolicyBasisRecord {
        stage_id,
        iteration,
        column_status,
        row_status,
        num_cut_rows,
    })
}

/// Read a complete policy checkpoint from `path`.
///
/// Reads `metadata.json`, all `cuts/stage_NNN.bin` files, and all
/// `basis/stage_NNN.bin` files. Results are sorted by `stage_id` in the returned
/// [`PolicyCheckpoint`].
///
/// # Errors
///
/// - [`OutputError::IoError`] — directory or file read failed.
/// - [`OutputError::SerializationError`] — JSON or `FlatBuffers` parse failure.
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
    // Read and deserialize metadata.json.
    let meta_path = path.join("metadata.json");
    let meta_bytes = std::fs::read(&meta_path).map_err(|e| OutputError::io(&meta_path, e))?;
    let metadata: PolicyCheckpointMetadata = serde_json::from_slice(&meta_bytes)
        .map_err(|e| OutputError::serialization("policy_metadata", e.to_string()))?;

    // Read all cuts/stage_NNN.bin files.
    let cuts_dir = path.join("cuts");
    let mut stage_cuts = read_sorted_bin_files(&cuts_dir, "stage_cuts", deserialize_stage_cuts)?;
    stage_cuts.sort_by_key(|r| r.stage_id);

    // Read all basis/stage_NNN.bin files (may be empty if no bases were written).
    let basis_dir = path.join("basis");
    let mut stage_bases =
        read_sorted_bin_files(&basis_dir, "stage_basis", deserialize_stage_basis)?;
    stage_bases.sort_by_key(|r| r.stage_id);

    Ok(PolicyCheckpoint {
        metadata,
        stage_cuts,
        stage_bases,
    })
}

/// Read all `*.bin` files from `dir`, deserialize each with `deser_fn`, and return a `Vec`.
///
/// Files are enumerated via [`std::fs::read_dir`]. The returned `Vec` is unsorted —
/// callers should sort by the appropriate `stage_id` field after this call.
///
/// If `dir` exists but contains no `.bin` files, an empty `Vec` is returned.
fn read_sorted_bin_files<T, F>(dir: &Path, ctx: &str, deser_fn: F) -> Result<Vec<T>, OutputError>
where
    F: Fn(&[u8]) -> Result<T, OutputError>,
{
    let entries = std::fs::read_dir(dir).map_err(|e| OutputError::io(dir, e))?;

    let mut results = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|e| OutputError::io(dir, e))?;
        let file_name = entry.file_name();
        let name = file_name.to_string_lossy();
        if !name.ends_with(".bin") {
            continue;
        }
        let file_path = entry.path();
        let bytes = std::fs::read(&file_path).map_err(|e| OutputError::io(&file_path, e))?;
        let record = deser_fn(&bytes).map_err(|e| {
            // Re-wrap with file context for better diagnostics.
            OutputError::serialization(
                ctx,
                format!("failed to deserialize {}: {e}", file_path.display()),
            )
        })?;
        results.push(record);
    }
    Ok(results)
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
            domination_count: 11,
        };

        let buf = serialize_stage_cuts(0, 3, 100, 0, &[cut], &[0], 1);
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
        assert_eq!(c.domination_count, 11, "domination_count must round-trip");
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
                domination_count: 0,
            },
            PolicyCutRecord {
                cut_id: 20,
                slot_index: 1,
                iteration: 2,
                forward_pass_index: 1,
                intercept: 200.0,
                coefficients: &c1,
                is_active: false,
                domination_count: 3,
            },
            PolicyCutRecord {
                cut_id: 30,
                slot_index: 2,
                iteration: 3,
                forward_pass_index: 2,
                intercept: 300.0,
                coefficients: &c2,
                is_active: true,
                domination_count: 7,
            },
        ];

        let buf = serialize_stage_cuts(5, 2, 50, 1, &cuts, &[0, 2], 3);
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
        let expected_domination = [0u32, 3, 7];

        for (i, cut) in result.cuts.iter().enumerate() {
            assert_eq!(cut.cut_id, expected_cut_ids[i], "cut {i} cut_id");
            assert_eq!(cut.intercept, expected_intercepts[i], "cut {i} intercept");
            assert_eq!(
                cut.coefficients, expected_coefficients[i],
                "cut {i} coefficients"
            );
            assert_eq!(cut.is_active, expected_active[i], "cut {i} is_active");
            assert_eq!(
                cut.domination_count, expected_domination[i],
                "cut {i} domination_count"
            );
        }
    }

    #[test]
    fn deserialize_stage_cuts_empty_cut_pool() {
        let buf = serialize_stage_cuts(2, 10, 200, 0, &[], &[], 0);
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
            domination_count: 0,
        };
        let buf = serialize_stage_cuts(0, 0, 10, 0, &[cut], &[0], 1);
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
            domination_count: 5,
        };
        let buf = serialize_stage_cuts(3, dim, 10, 0, &[cut], &[0], 1);
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
        let full_buf = serialize_stage_cuts(0, 2, 10, 0, &[cut], &[0], 1);
        // Truncate to 2 bytes — root offset itself is incomplete.
        let truncated = &full_buf[..2];
        let result = deserialize_stage_cuts(truncated);
        assert!(result.is_err(), "truncated buffer must return an error");
    }

    #[test]
    fn deserialize_stage_cuts_stage_id_nonzero() {
        let buf = serialize_stage_cuts(59, 4, 50, 0, &[], &[], 0);
        let result = deserialize_stage_cuts(&buf).expect("stage_id=59 must deserialize");
        assert_eq!(result.stage_id, 59, "stage_id=59 must round-trip");
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

    // ── PolicyCheckpointMetadata deserialization tests ────────────────────────

    #[test]
    fn policy_checkpoint_metadata_deserializes_from_json() {
        let meta = PolicyCheckpointMetadata {
            version: "1.0.0".to_string(),
            cobre_version: "0.0.1".to_string(),
            created_at: "2026-03-08T00:00:00Z".to_string(),
            completed_iterations: 42,
            final_lower_bound: 9999.0,
            best_upper_bound: Some(10100.0),
            state_dimension: 5,
            num_stages: 3,
            config_hash: "cfghash".to_string(),
            system_hash: "syshash".to_string(),
            max_iterations: 100,
            forward_passes: 4,
            warm_start_cuts: 10,
            rng_seed: 12345,
        };

        let json = serde_json::to_string(&meta).expect("serialize must succeed");
        let back: PolicyCheckpointMetadata =
            serde_json::from_str(&json).expect("deserialize must succeed");

        assert_eq!(back.version, meta.version);
        assert_eq!(back.cobre_version, meta.cobre_version);
        assert_eq!(back.completed_iterations, meta.completed_iterations);
        assert_eq!(back.final_lower_bound, meta.final_lower_bound);
        assert_eq!(back.best_upper_bound, meta.best_upper_bound);
        assert_eq!(back.state_dimension, meta.state_dimension);
        assert_eq!(back.num_stages, meta.num_stages);
        assert_eq!(back.config_hash, meta.config_hash);
        assert_eq!(back.system_hash, meta.system_hash);
        assert_eq!(back.max_iterations, meta.max_iterations);
        assert_eq!(back.forward_passes, meta.forward_passes);
        assert_eq!(back.warm_start_cuts, meta.warm_start_cuts);
        assert_eq!(back.rng_seed, meta.rng_seed);
    }

    #[test]
    fn policy_checkpoint_metadata_deserializes_none_upper_bound() {
        let meta = PolicyCheckpointMetadata {
            version: "1.0.0".to_string(),
            cobre_version: "0.0.1".to_string(),
            created_at: "2026-03-08T00:00:00Z".to_string(),
            completed_iterations: 1,
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

        let json = serde_json::to_string(&meta).expect("serialize must succeed");
        let back: PolicyCheckpointMetadata =
            serde_json::from_str(&json).expect("deserialize must succeed");

        assert!(
            back.best_upper_bound.is_none(),
            "None upper bound must round-trip"
        );
    }

    // ── read_policy_checkpoint round-trip tests ───────────────────────────────

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

        write_policy_checkpoint(tmp.path(), &stage_cuts_payloads, &basis_records, &metadata)
            .expect("write must succeed");

        let checkpoint = read_policy_checkpoint(tmp.path()).expect("read must succeed");

        // Metadata fields.
        assert_eq!(checkpoint.metadata.completed_iterations, 10);
        assert_eq!(checkpoint.metadata.num_stages, 2);
        assert_eq!(checkpoint.metadata.state_dimension, 3);
        assert_eq!(checkpoint.metadata.rng_seed, 42);

        // Cuts: two stages, sorted by stage_id.
        assert_eq!(
            checkpoint.stage_cuts.len(),
            2,
            "must have two stage cut results"
        );
        assert_eq!(checkpoint.stage_cuts[0].stage_id, 0);
        assert_eq!(checkpoint.stage_cuts[1].stage_id, 1);
        assert_eq!(checkpoint.stage_cuts[0].cuts.len(), 2);
        assert_eq!(checkpoint.stage_cuts[1].cuts.len(), 1);

        // Stage 0 cut fields.
        let cut00 = &checkpoint.stage_cuts[0].cuts[0];
        assert_eq!(cut00.cut_id, 1);
        assert_eq!(cut00.coefficients, &[1.0f64, 2.0, 3.0]);
        assert_eq!(cut00.intercept, 42.0);
        assert!(cut00.is_active);

        let cut01 = &checkpoint.stage_cuts[0].cuts[1];
        assert_eq!(cut01.cut_id, 2);
        assert_eq!(cut01.coefficients, &[4.0f64, 5.0, 6.0]);

        // Stage 1 cut fields.
        let cut10 = &checkpoint.stage_cuts[1].cuts[0];
        assert_eq!(cut10.cut_id, 3);
        assert_eq!(cut10.coefficients, &[7.0f64, 8.0, 9.0]);

        // Bases: two stages, sorted by stage_id.
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

        write_policy_checkpoint(tmp.path(), &stage_cuts_payloads, &[], &metadata)
            .expect("write must succeed");

        let checkpoint = read_policy_checkpoint(tmp.path()).expect("read must succeed");

        assert_eq!(checkpoint.stage_cuts.len(), 1);
        assert!(
            checkpoint.stage_bases.is_empty(),
            "no basis files must produce empty stage_bases"
        );
    }

    #[test]
    fn read_policy_checkpoint_missing_metadata_returns_error() {
        let tmp = tempfile::tempdir().unwrap();
        // Intentionally do NOT write metadata.json.
        let result = read_policy_checkpoint(tmp.path());
        assert!(
            result.is_err(),
            "missing metadata.json must return an error"
        );
        assert!(
            matches!(result, Err(OutputError::IoError { .. })),
            "error must be IoError for missing metadata.json"
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

        write_policy_checkpoint(tmp.path(), &stage_cuts_payloads, &[], &metadata)
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
    fn read_policy_checkpoint_metadata_json_field_by_field() {
        let tmp = tempfile::tempdir().unwrap();

        let meta_in = PolicyCheckpointMetadata {
            version: "1.0.0".to_string(),
            cobre_version: "0.0.1".to_string(),
            created_at: "2026-01-01T00:00:00Z".to_string(),
            completed_iterations: 77,
            final_lower_bound: 12345.678,
            best_upper_bound: Some(13000.0),
            state_dimension: 8,
            num_stages: 4,
            config_hash: "cfghash123".to_string(),
            system_hash: "syshash456".to_string(),
            max_iterations: 500,
            forward_passes: 8,
            warm_start_cuts: 20,
            rng_seed: 99999,
        };

        let stage_cuts_payloads: [StageCutsPayload<'_>; 0] = [];
        write_policy_checkpoint(tmp.path(), &stage_cuts_payloads, &[], &meta_in)
            .expect("write must succeed");

        let checkpoint = read_policy_checkpoint(tmp.path()).expect("read must succeed");
        let m = &checkpoint.metadata;

        assert_eq!(m.version, "1.0.0");
        assert_eq!(m.completed_iterations, 77);
        assert_eq!(m.final_lower_bound, 12345.678);
        assert_eq!(m.best_upper_bound, Some(13000.0));
        assert_eq!(m.state_dimension, 8);
        assert_eq!(m.num_stages, 4);
        assert_eq!(m.config_hash, "cfghash123");
        assert_eq!(m.system_hash, "syshash456");
        assert_eq!(m.max_iterations, 500);
        assert_eq!(m.forward_passes, 8);
        assert_eq!(m.warm_start_cuts, 20);
        assert_eq!(m.rng_seed, 99999);
    }
}
