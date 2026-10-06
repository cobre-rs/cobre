//! `FlatBuffers` serializers, wire-format helpers, and deserializers for policy data.
//!
//! Wire layout follows the policy schema specification (spec SS3.2).
//!
//! This module is the sole owner of the policy `FlatBuffers` byte layout. The
//! `*_FIELD_*: u16` slot constants below mirror the `(id: N)` attributes in
//! `schemas/policy.fbs` via `slot = (id + 2) * 2` and MUST stay in sync; the
//! `flatc-conformance` feature gates the round-trip test in
//! `tests/flatbuffers_schema_conformance.rs` that fails when they diverge.
//!
//! Reader functions parse raw bytes rather than the generated `Table::get` API
//! (an `unsafe fn`) because the workspace forbids `unsafe_code`.

use flatbuffers::{FlatBufferBuilder, WIPOffset};

use super::super::error::OutputError;
use super::super::software::policy_checkpoint_remedy;
use super::records::{
    CheckpointManifest, ENTITY_SLOT_DATE_SENTINEL, EntitySlot, FORMAT_VERSION, GraphManifest,
    HydroSeasonOrders, ManifestEdge, ManifestNode, OwnedPolicyBasisRecord, OwnedPolicyCutRecord,
    PolicyBasisRecord, PolicyCutRecord, ProducerBlock, SEASON_CYCLE_CODE_ABSENT,
    STAGE_CUTS_GRAPH_STAGE_ID_SENTINEL, STAGE_CUTS_NODE_ID_SENTINEL,
    STAGE_CUTS_PRICED_STATE_DATE_SENTINEL, STAGE_STATES_NODE_ID_SENTINEL, SeasonManifest,
    StageCutsPayload, StageCutsReadResult, StageStatesPayload, StageStatesReadResult,
};

use std::path::Path;

/// `FlatBuffers` `file_identifier` for every policy artifact (`schemas/policy.fbs`).
/// Written by `finish(root, Some(POLICY_FILE_IDENTIFIER))` and required on read:
/// a buffer lacking it (a pre-0.14 `finish_minimal` artifact) is rejected.
const POLICY_FILE_IDENTIFIER: &str = "CBVF";

// ── FlatBuffers vtable slot offsets ──────────────────────────────────────────
//
// slot = (id + 2) * 2. Three slots are permanently burned and never read or
// written here — reusing any would diverge the hand-written layout from the
// schema's `deprecated` placeholder: `AffinePiece` id 7 (slot 18), the former
// `state_at_generation`; `EntitySlot` id 4 (slot 12), the former month-integer
// `delivery_anchor`; and `EntitySlot` id 5 (slot 14), the former single-field
// `delivery_date` anchor replaced by `reference_date`/`interval_start`/
// `interval_end` at ids 6/7/8.

const CUT_FIELD_CUT_ID: u16 = 4;
const CUT_FIELD_SLOT_INDEX: u16 = 6;
const CUT_FIELD_ITERATION: u16 = 8;
const CUT_FIELD_FORWARD_PASS_IDX: u16 = 10;
const CUT_FIELD_INTERCEPT: u16 = 12;
const CUT_FIELD_COEFFICIENTS: u16 = 14;
const CUT_FIELD_IS_ACTIVE: u16 = 16;

const ENTITY_SLOT_FIELD_ENTITY_TYPE: u16 = 4;
const ENTITY_SLOT_FIELD_ENTITY_ID: u16 = 6;
const ENTITY_SLOT_FIELD_SUBINDEX: u16 = 8;
const ENTITY_SLOT_FIELD_WAS_ACTIVE: u16 = 10;
const ENTITY_SLOT_FIELD_REFERENCE_DATE: u16 = 16;
const ENTITY_SLOT_FIELD_INTERVAL_START: u16 = 18;
const ENTITY_SLOT_FIELD_INTERVAL_END: u16 = 20;

const STAGE_CUTS_FIELD_STAGE_ID: u16 = 4;
const STAGE_CUTS_FIELD_STATE_DIMENSION: u16 = 6;
const STAGE_CUTS_FIELD_CAPACITY: u16 = 8;
const STAGE_CUTS_FIELD_WARM_START_COUNT: u16 = 10;
const STAGE_CUTS_FIELD_CUTS: u16 = 12;
const STAGE_CUTS_FIELD_ACTIVE_CUT_INDICES: u16 = 14;
const STAGE_CUTS_FIELD_POPULATED_COUNT: u16 = 16;
const STAGE_CUTS_FIELD_ENTITY_MANIFEST: u16 = 18;
const STAGE_CUTS_FIELD_COST_SCALE_FACTOR: u16 = 20;
const STAGE_CUTS_FIELD_NODE_ID: u16 = 22;
const STAGE_CUTS_FIELD_GRAPH_STAGE_ID: u16 = 24;
const STAGE_CUTS_FIELD_PRICED_STATE_DATE: u16 = 26;

const BASIS_FIELD_STAGE_ID: u16 = 4;
const BASIS_FIELD_ITERATION: u16 = 6;
const BASIS_FIELD_NUM_COLUMNS: u16 = 8;
const BASIS_FIELD_NUM_ROWS: u16 = 10;
const BASIS_FIELD_COLUMN_STATUS: u16 = 12;
const BASIS_FIELD_ROW_STATUS: u16 = 14;
const BASIS_FIELD_NUM_CUT_ROWS: u16 = 16;

const STATES_FIELD_STAGE_ID: u16 = 4;
const STATES_FIELD_STATE_DIMENSION: u16 = 6;
const STATES_FIELD_COUNT: u16 = 8;
const STATES_FIELD_DATA: u16 = 10;
const STATES_FIELD_ENTITY_MANIFEST: u16 = 12;
const STATES_FIELD_NODE_ID: u16 = 14;

// CheckpointManifest ids start fresh at 0; its own vtable is distinct from the
// AffinePiece/EntitySlot vtables, so its id 4 / id 7 collide with no burned slot.
const MANIFEST_FIELD_FORMAT_VERSION: u16 = 4;
const MANIFEST_FIELD_SOFTWARE_VERSION: u16 = 6;
const MANIFEST_FIELD_CREATED_AT: u16 = 8;
const MANIFEST_FIELD_NUM_STAGES: u16 = 10;
const MANIFEST_FIELD_N_POOLS: u16 = 12;
const MANIFEST_FIELD_NODES: u16 = 14;
const MANIFEST_FIELD_EDGES: u16 = 16;
const MANIFEST_FIELD_COMPLETED_ITERATIONS: u16 = 18;
const MANIFEST_FIELD_FINAL_LOWER_BOUND: u16 = 20;
const MANIFEST_FIELD_BEST_UPPER_BOUND: u16 = 22;
const MANIFEST_FIELD_MAX_ITERATIONS: u16 = 24;
const MANIFEST_FIELD_FORWARD_PASSES: u16 = 26;
const MANIFEST_FIELD_WARM_START_CUTS: u16 = 28;
const MANIFEST_FIELD_WARM_START_COUNTS: u16 = 30;
const MANIFEST_FIELD_RNG_SEED: u16 = 32;
const MANIFEST_FIELD_TOTAL_VISITED_STATES: u16 = 34;
const MANIFEST_FIELD_TRAINING_BLOCK_MODE: u16 = 36;
const MANIFEST_FIELD_TRAINING_BLOCK_MODE_PER_STAGE: u16 = 38;
const MANIFEST_FIELD_COST_SCALE_FACTOR: u16 = 40;
const MANIFEST_FIELD_SEASON_MANIFEST: u16 = 42;
const MANIFEST_FIELD_SOFTWARE: u16 = 44;
const MANIFEST_FIELD_LOWER_BOUND_HISTORY: u16 = 46;

const MANIFEST_NODE_FIELD_ID: u16 = 4;
const MANIFEST_NODE_FIELD_STAGE_ID: u16 = 6;
const MANIFEST_NODE_FIELD_POOL_ID: u16 = 8;

const MANIFEST_EDGE_FIELD_SOURCE_ID: u16 = 4;
const MANIFEST_EDGE_FIELD_TARGET_ID: u16 = 6;
const MANIFEST_EDGE_FIELD_PROBABILITY: u16 = 8;

// SeasonManifest/HydroSeasonOrders ids start fresh at 0, distinct from
// CheckpointManifest's own vtable; they collide with no burned slot.
const SEASON_MANIFEST_FIELD_CYCLE_CODE: u16 = 4;
const SEASON_MANIFEST_FIELD_N_SEASONS: u16 = 6;
const SEASON_MANIFEST_FIELD_HYDRO_ORDERS: u16 = 8;

const HYDRO_SEASON_ORDERS_FIELD_HYDRO_ID: u16 = 4;
const HYDRO_SEASON_ORDERS_FIELD_ORDERS: u16 = 6;

/// The coefficient vector must be created before the `start_table`/`end_table`
/// pair — `FlatBuffers` requires nested objects to precede the enclosing table
/// in the buffer.
fn build_cut_table(
    builder: &mut FlatBufferBuilder<'_>,
    piece: &PolicyCutRecord<'_>,
) -> WIPOffset<flatbuffers::TableFinishedWIPOffset> {
    let coefficients_vec = builder.create_vector(piece.coefficients);

    let tab = builder.start_table();

    builder.push_slot_always::<u64>(CUT_FIELD_CUT_ID, piece.cut_id);
    builder.push_slot_always::<u32>(CUT_FIELD_SLOT_INDEX, piece.slot_index);
    builder.push_slot_always::<u32>(CUT_FIELD_ITERATION, piece.iteration);
    builder.push_slot_always::<u32>(CUT_FIELD_FORWARD_PASS_IDX, piece.forward_pass_index);
    builder.push_slot_always::<f64>(CUT_FIELD_INTERCEPT, piece.intercept);
    builder.push_slot_always(CUT_FIELD_COEFFICIENTS, coefficients_vec);
    builder.push_slot_always::<bool>(CUT_FIELD_IS_ACTIVE, piece.is_active);

    builder.end_table(tab)
}

fn build_entity_slot_table(
    builder: &mut FlatBufferBuilder<'_>,
    slot: &EntitySlot,
) -> WIPOffset<flatbuffers::TableFinishedWIPOffset> {
    let tab = builder.start_table();

    builder.push_slot_always::<u8>(ENTITY_SLOT_FIELD_ENTITY_TYPE, slot.entity_type);
    builder.push_slot_always::<i32>(ENTITY_SLOT_FIELD_ENTITY_ID, slot.entity_id);
    builder.push_slot_always::<u32>(ENTITY_SLOT_FIELD_SUBINDEX, slot.subindex);
    builder.push_slot_always::<bool>(ENTITY_SLOT_FIELD_WAS_ACTIVE, slot.was_active);
    builder.push_slot_always::<i32>(ENTITY_SLOT_FIELD_REFERENCE_DATE, slot.reference_date);
    builder.push_slot_always::<i32>(ENTITY_SLOT_FIELD_INTERVAL_START, slot.interval_start);
    builder.push_slot_always::<i32>(ENTITY_SLOT_FIELD_INTERVAL_END, slot.interval_end);

    builder.end_table(tab)
}

fn build_manifest_node_table(
    builder: &mut FlatBufferBuilder<'_>,
    node: &ManifestNode,
) -> WIPOffset<flatbuffers::TableFinishedWIPOffset> {
    let tab = builder.start_table();

    builder.push_slot_always::<i32>(MANIFEST_NODE_FIELD_ID, node.id);
    builder.push_slot_always::<i32>(MANIFEST_NODE_FIELD_STAGE_ID, node.stage_id);
    builder.push_slot_always::<u32>(MANIFEST_NODE_FIELD_POOL_ID, node.pool_id);

    builder.end_table(tab)
}

fn build_manifest_edge_table(
    builder: &mut FlatBufferBuilder<'_>,
    edge: &ManifestEdge,
) -> WIPOffset<flatbuffers::TableFinishedWIPOffset> {
    let tab = builder.start_table();

    builder.push_slot_always::<i32>(MANIFEST_EDGE_FIELD_SOURCE_ID, edge.source_id);
    builder.push_slot_always::<i32>(MANIFEST_EDGE_FIELD_TARGET_ID, edge.target_id);
    builder.push_slot_always::<f64>(MANIFEST_EDGE_FIELD_PROBABILITY, edge.probability);

    builder.end_table(tab)
}

fn build_hydro_season_orders_table(
    builder: &mut FlatBufferBuilder<'_>,
    entry: &HydroSeasonOrders,
) -> WIPOffset<flatbuffers::TableFinishedWIPOffset> {
    let orders_vec = builder.create_vector(&entry.orders);

    let tab = builder.start_table();

    builder.push_slot_always::<i32>(HYDRO_SEASON_ORDERS_FIELD_HYDRO_ID, entry.hydro_id);
    builder.push_slot_always(HYDRO_SEASON_ORDERS_FIELD_ORDERS, orders_vec);

    builder.end_table(tab)
}

fn build_season_manifest_table(
    builder: &mut FlatBufferBuilder<'_>,
    season_manifest: &SeasonManifest,
) -> WIPOffset<flatbuffers::TableFinishedWIPOffset> {
    let hydro_order_offsets: Vec<WIPOffset<flatbuffers::TableFinishedWIPOffset>> = season_manifest
        .hydro_orders
        .iter()
        .map(|h| build_hydro_season_orders_table(builder, h))
        .collect();
    let hydro_orders_vec = builder.create_vector(&hydro_order_offsets);

    let tab = builder.start_table();

    builder.push_slot_always::<u8>(SEASON_MANIFEST_FIELD_CYCLE_CODE, season_manifest.cycle_code);
    builder.push_slot_always::<u32>(SEASON_MANIFEST_FIELD_N_SEASONS, season_manifest.n_seasons);
    builder.push_slot_always(SEASON_MANIFEST_FIELD_HYDRO_ORDERS, hydro_orders_vec);

    builder.end_table(tab)
}

/// Reject a buffer whose leading `file_identifier` is not [`POLICY_FILE_IDENTIFIER`].
///
/// `finish(root, Some(id))` writes the 4-byte identifier at bytes `4..8` (right
/// after the 4-byte root uoffset), so a pre-0.14 `finish_minimal` buffer carries
/// none and is rejected here before any field is decoded.
///
/// # Errors
///
/// Returns [`OutputError::SerializationError`] when the identifier is absent or wrong.
fn check_file_identifier(buf: &[u8], ctx: &str) -> Result<(), OutputError> {
    if buf.get(4..8) == Some(POLICY_FILE_IDENTIFIER.as_bytes()) {
        Ok(())
    } else {
        Err(OutputError::serialization(
            ctx,
            format!(
                "missing FlatBuffers file_identifier {POLICY_FILE_IDENTIFIER:?}; not a 0.14+ policy artifact"
            ),
        ))
    }
}

// ── Serializers ───────────────────────────────────────────────────────────────

/// Serialize all cuts for one stage into a root `StageCuts` `FlatBuffers` buffer,
/// ready to write directly to a `.bin` policy file.
///
/// Infallible: the builder only allocates and writes. Any I/O error is the
/// caller's responsibility.
///
/// # Examples
///
/// ```
/// use cobre_io::{
///     PolicyCutRecord, STAGE_CUTS_PRICED_STATE_DATE_SENTINEL, StageCutsPayload,
///     serialize_stage_cuts,
/// };
///
/// let piece = PolicyCutRecord {
///     cut_id: 1,
///     slot_index: 5,
///     iteration: 3,
///     forward_pass_index: 0,
///     intercept: 42.0,
///     coefficients: &[1.0, 2.0, 3.0],
///     is_active: true,
/// };
/// let buf = serialize_stage_cuts(&StageCutsPayload {
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
/// });
/// assert!(!buf.is_empty());
/// ```
#[must_use]
pub fn serialize_stage_cuts(payload: &StageCutsPayload<'_>) -> Vec<u8> {
    build_stage_cuts(payload).finished_data().to_vec()
}

#[allow(clippy::cast_possible_truncation)]
pub(super) fn build_stage_cuts(payload: &StageCutsPayload<'_>) -> FlatBufferBuilder<'static> {
    let estimated = 64
        + payload.cuts.len()
            * (96usize + payload.state_dimension as usize * std::mem::size_of::<f64>())
        + std::mem::size_of_val(payload.active_cut_indices)
        + payload.entity_manifest.len() * 32usize;

    let mut builder = FlatBufferBuilder::with_capacity(estimated);

    let cut_offsets: Vec<WIPOffset<flatbuffers::TableFinishedWIPOffset>> = payload
        .cuts
        .iter()
        .map(|c| build_cut_table(&mut builder, c))
        .collect();
    let manifest_offsets: Vec<WIPOffset<flatbuffers::TableFinishedWIPOffset>> = payload
        .entity_manifest
        .iter()
        .map(|s| build_entity_slot_table(&mut builder, s))
        .collect();

    let cuts_vec = builder.create_vector(&cut_offsets);
    let active_vec = builder.create_vector(payload.active_cut_indices);
    let manifest_vec = builder.create_vector(&manifest_offsets);

    let root = builder.start_table();

    builder.push_slot_always::<u32>(STAGE_CUTS_FIELD_STAGE_ID, payload.stage_id);
    builder.push_slot_always::<u32>(STAGE_CUTS_FIELD_STATE_DIMENSION, payload.state_dimension);
    builder.push_slot_always::<u32>(STAGE_CUTS_FIELD_CAPACITY, payload.capacity);
    builder.push_slot_always::<u32>(STAGE_CUTS_FIELD_WARM_START_COUNT, payload.warm_start_count);
    builder.push_slot_always(STAGE_CUTS_FIELD_CUTS, cuts_vec);
    builder.push_slot_always(STAGE_CUTS_FIELD_ACTIVE_CUT_INDICES, active_vec);
    builder.push_slot_always::<u32>(STAGE_CUTS_FIELD_POPULATED_COUNT, payload.populated_count);
    builder.push_slot_always(STAGE_CUTS_FIELD_ENTITY_MANIFEST, manifest_vec);
    builder.push_slot_always::<f64>(
        STAGE_CUTS_FIELD_COST_SCALE_FACTOR,
        payload.cost_scale_factor,
    );
    builder.push_slot_always::<i32>(STAGE_CUTS_FIELD_NODE_ID, payload.node_id);
    builder.push_slot_always::<i32>(STAGE_CUTS_FIELD_GRAPH_STAGE_ID, payload.graph_stage_id);
    builder.push_slot_always::<i32>(
        STAGE_CUTS_FIELD_PRICED_STATE_DATE,
        payload.priced_state_date,
    );

    let root_offset = builder.end_table(root);
    builder.finish(root_offset, Some(POLICY_FILE_IDENTIFIER));

    builder
}

/// Serialize one stage's solver basis into a root `StageBasis` `FlatBuffers`
/// buffer, ready to write directly to a `.bin` policy file under `basis/`.
///
/// `num_columns` and `num_rows` are inferred from the status slice lengths, not
/// supplied separately. Infallible: the builder only allocates and writes.
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
pub fn serialize_stage_basis(record: &PolicyBasisRecord<'_>) -> Vec<u8> {
    build_stage_basis(record).finished_data().to_vec()
}

#[allow(clippy::cast_possible_truncation)]
pub(super) fn build_stage_basis(record: &PolicyBasisRecord<'_>) -> FlatBufferBuilder<'static> {
    let estimated =
        64 + std::mem::size_of_val(record.column_status) + std::mem::size_of_val(record.row_status);

    let mut builder = FlatBufferBuilder::with_capacity(estimated);

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
    builder.finish(root_offset, Some(POLICY_FILE_IDENTIFIER));

    builder
}

/// Serialize one stage's visited states into a root `StageStates` `FlatBuffers`
/// buffer, ready to write directly to a `.bin` policy file under `states/`.
#[must_use]
pub fn serialize_stage_states(payload: &StageStatesPayload<'_>) -> Vec<u8> {
    build_stage_states(payload).finished_data().to_vec()
}

#[allow(clippy::cast_possible_truncation)]
pub(super) fn build_stage_states(payload: &StageStatesPayload<'_>) -> FlatBufferBuilder<'static> {
    let estimated =
        64 + std::mem::size_of_val(payload.data) + payload.entity_manifest.len() * 32usize;
    let mut builder = FlatBufferBuilder::with_capacity(estimated);

    let manifest_offsets: Vec<WIPOffset<flatbuffers::TableFinishedWIPOffset>> = payload
        .entity_manifest
        .iter()
        .map(|s| build_entity_slot_table(&mut builder, s))
        .collect();

    let data_vec = builder.create_vector(payload.data);
    let manifest_vec = builder.create_vector(&manifest_offsets);

    let root = builder.start_table();
    builder.push_slot_always::<u32>(STATES_FIELD_STAGE_ID, payload.stage_id);
    builder.push_slot_always::<u32>(STATES_FIELD_STATE_DIMENSION, payload.state_dimension);
    builder.push_slot_always::<u32>(STATES_FIELD_COUNT, payload.count);
    builder.push_slot_always(STATES_FIELD_DATA, data_vec);
    builder.push_slot_always(STATES_FIELD_ENTITY_MANIFEST, manifest_vec);
    builder.push_slot_always::<i32>(STATES_FIELD_NODE_ID, payload.node_id);

    let root_offset = builder.end_table(root);
    builder.finish(root_offset, Some(POLICY_FILE_IDENTIFIER));

    builder
}

/// Serialize a [`CheckpointManifest`] into a root `CheckpointManifest`
/// `FlatBuffers` buffer.
///
/// The two `Option<f64>` provenance fields (`best_upper_bound`,
/// `cost_scale_factor`) are written only when `Some`, so absence round-trips as
/// `None` rather than a spurious `0.0`. Infallible: the builder only allocates
/// and writes.
#[must_use]
pub fn serialize_checkpoint_manifest(manifest: &CheckpointManifest) -> Vec<u8> {
    build_checkpoint_manifest(manifest).finished_data().to_vec()
}

pub(super) fn build_checkpoint_manifest(
    manifest: &CheckpointManifest,
) -> FlatBufferBuilder<'static> {
    let graph = &manifest.graph_manifest;
    let producer = &manifest.producer;

    let season_manifest = &manifest.season_manifest;

    let estimated = 128
        + graph.nodes.len() * 32
        + graph.edges.len() * 40
        + manifest.software.as_ref().map_or(0, String::len)
        + manifest.software_version.len()
        + manifest.created_at.len()
        + producer.training_block_mode.len()
        + producer.warm_start_counts.len() * std::mem::size_of::<u32>()
        + producer.lower_bound_history.len() * std::mem::size_of::<f64>()
        + producer
            .training_block_mode_per_stage
            .iter()
            .map(|s| s.len() + 8)
            .sum::<usize>()
        + season_manifest
            .hydro_orders
            .iter()
            .map(|h| 16 + h.orders.len() * std::mem::size_of::<u32>())
            .sum::<usize>();

    let mut builder = FlatBufferBuilder::with_capacity(estimated);

    let node_offsets: Vec<WIPOffset<flatbuffers::TableFinishedWIPOffset>> = graph
        .nodes
        .iter()
        .map(|n| build_manifest_node_table(&mut builder, n))
        .collect();
    let edge_offsets: Vec<WIPOffset<flatbuffers::TableFinishedWIPOffset>> = graph
        .edges
        .iter()
        .map(|e| build_manifest_edge_table(&mut builder, e))
        .collect();
    let season_manifest_offset = build_season_manifest_table(&mut builder, season_manifest);

    let software = manifest
        .software
        .as_deref()
        .map(|name| builder.create_string(name));
    let software_version = builder.create_string(&manifest.software_version);
    let created_at = builder.create_string(&manifest.created_at);
    let training_block_mode = builder.create_string(&producer.training_block_mode);
    let per_stage_offsets: Vec<WIPOffset<&str>> = producer
        .training_block_mode_per_stage
        .iter()
        .map(|s| builder.create_string(s))
        .collect();

    let nodes_vec = builder.create_vector(&node_offsets);
    let edges_vec = builder.create_vector(&edge_offsets);
    let warm_start_counts_vec = builder.create_vector(producer.warm_start_counts.as_slice());
    let per_stage_vec = builder.create_vector(&per_stage_offsets);
    let lower_bound_history_vec = builder.create_vector(producer.lower_bound_history.as_slice());

    let root = builder.start_table();

    builder.push_slot_always::<u32>(MANIFEST_FIELD_FORMAT_VERSION, manifest.format_version);
    builder.push_slot_always(MANIFEST_FIELD_SOFTWARE_VERSION, software_version);
    builder.push_slot_always(MANIFEST_FIELD_CREATED_AT, created_at);
    builder.push_slot_always::<u32>(MANIFEST_FIELD_NUM_STAGES, manifest.num_stages);
    builder.push_slot_always::<u32>(MANIFEST_FIELD_N_POOLS, graph.n_pools);
    builder.push_slot_always(MANIFEST_FIELD_NODES, nodes_vec);
    builder.push_slot_always(MANIFEST_FIELD_EDGES, edges_vec);
    builder.push_slot_always::<u32>(
        MANIFEST_FIELD_COMPLETED_ITERATIONS,
        producer.completed_iterations,
    );
    builder.push_slot_always::<f64>(MANIFEST_FIELD_FINAL_LOWER_BOUND, producer.final_lower_bound);
    if let Some(best) = producer.best_upper_bound {
        builder.push_slot_always::<f64>(MANIFEST_FIELD_BEST_UPPER_BOUND, best);
    }
    builder.push_slot_always::<u32>(MANIFEST_FIELD_MAX_ITERATIONS, producer.max_iterations);
    builder.push_slot_always::<u32>(MANIFEST_FIELD_FORWARD_PASSES, producer.forward_passes);
    builder.push_slot_always::<u32>(MANIFEST_FIELD_WARM_START_CUTS, producer.warm_start_cuts);
    builder.push_slot_always(MANIFEST_FIELD_WARM_START_COUNTS, warm_start_counts_vec);
    builder.push_slot_always::<u64>(MANIFEST_FIELD_RNG_SEED, producer.rng_seed);
    builder.push_slot_always::<u64>(
        MANIFEST_FIELD_TOTAL_VISITED_STATES,
        producer.total_visited_states,
    );
    builder.push_slot_always(MANIFEST_FIELD_TRAINING_BLOCK_MODE, training_block_mode);
    builder.push_slot_always(MANIFEST_FIELD_TRAINING_BLOCK_MODE_PER_STAGE, per_stage_vec);
    if let Some(csf) = producer.cost_scale_factor {
        builder.push_slot_always::<f64>(MANIFEST_FIELD_COST_SCALE_FACTOR, csf);
    }
    builder.push_slot_always(MANIFEST_FIELD_SEASON_MANIFEST, season_manifest_offset);
    if let Some(software) = software {
        builder.push_slot_always(MANIFEST_FIELD_SOFTWARE, software);
    }
    builder.push_slot_always(MANIFEST_FIELD_LOWER_BOUND_HISTORY, lower_bound_history_vec);

    let root_offset = builder.end_table(root);
    builder.finish(root_offset, Some(POLICY_FILE_IDENTIFIER));

    builder
}

// ── Safe FlatBuffers wire-format helpers ─────────────────────────────────────
//
// All helpers return `Option` so callers can propagate truncation / corruption
// errors without panicking. The `resolve_*` functions follow the FlatBuffers
// specification exactly:
//
//   Buffer layout (finish with file_identifier):
//     bytes[0..4]  = u32 LE root_offset — byte offset from position 0 to root table
//     bytes[4..8]  = 4-byte file_identifier ("CBVF"), checked before any decode
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
//
//   Vector at vec_pos: [u32 length][length × element_size bytes of element data].

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

fn resolve_root(buf: &[u8]) -> Option<usize> {
    let offset = read_u32_le(buf, 0)? as usize;
    if offset.checked_add(4)? > buf.len() {
        return None;
    }
    Some(offset)
}

fn resolve_vtable_pos(buf: &[u8], table_pos: usize) -> Option<usize> {
    let soffset = read_i32_le(buf, table_pos)?;
    // soffset is signed: positive = vtable precedes the table, negative = follows.
    let vtable_pos = if soffset >= 0 {
        table_pos.checked_sub(u32::try_from(soffset).ok()? as usize)?
    } else {
        let abs = u32::try_from(soffset.wrapping_neg()).ok()? as usize;
        table_pos.checked_add(abs)?
    };
    if vtable_pos.checked_add(4)? > buf.len() {
        return None;
    }
    Some(vtable_pos)
}

/// `Some(0)` means the field is absent — the `FlatBuffers` optional-field
/// convention. A slot past the vtable end is a field added in a later schema
/// version (forward compatibility): treat it as absent, not as an error.
fn field_data_offset(buf: &[u8], vtable_pos: usize, slot: u16) -> Option<u16> {
    let vtable_size = read_u16_le(buf, vtable_pos)?;
    let slot_pos = vtable_pos.checked_add(slot as usize)?;
    if slot_pos.checked_add(2)? > vtable_pos.checked_add(vtable_size as usize)? {
        return Some(0);
    }
    read_u16_le(buf, slot_pos)
}

fn field_pos(buf: &[u8], table_pos: usize, vtable_pos: usize, slot: u16) -> Option<usize> {
    let data_off = field_data_offset(buf, vtable_pos, slot)?;
    if data_off == 0 {
        return None; // field absent
    }
    table_pos.checked_add(data_off as usize)
}

/// `FlatBuffers` uoffsets are forward and self-relative: the referenced nested
/// table or vector is at `pos + u32_at(pos)`, not `0 + u32_at(pos)`.
fn follow_uoffset(buf: &[u8], pos: usize) -> Option<usize> {
    let off = read_u32_le(buf, pos)?;
    pos.checked_add(off as usize)
}

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

fn read_u8_vector(buf: &[u8], vec_pos: usize) -> Option<Vec<u8>> {
    let len = read_u32_le(buf, vec_pos)? as usize;
    let data_start = vec_pos.checked_add(4)?;
    let data_end = data_start.checked_add(len)?;
    if data_end > buf.len() {
        return None;
    }
    Some(buf[data_start..data_end].to_vec())
}

fn read_u32_vector(buf: &[u8], vec_pos: usize) -> Option<Vec<u32>> {
    let len = read_u32_le(buf, vec_pos)? as usize;
    let data_start = vec_pos.checked_add(4)?;
    let data_end = data_start.checked_add(len.checked_mul(4)?)?;
    if data_end > buf.len() {
        return None;
    }
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        out.push(read_u32_le(buf, data_start + i * 4)?);
    }
    Some(out)
}

/// Read a length-prefixed `FlatBuffers` UTF-8 string at `str_pos`.
fn read_string(buf: &[u8], str_pos: usize) -> Option<String> {
    let len = read_u32_le(buf, str_pos)? as usize;
    let data_start = str_pos.checked_add(4)?;
    let data_end = data_start.checked_add(len)?;
    let bytes = buf.get(data_start..data_end)?;
    std::str::from_utf8(bytes).ok().map(str::to_owned)
}

/// Returns one absolute buffer position per element; each element stores a `u32`
/// uoffset from its own position to the nested table (self-relative, not from 0).
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

/// Read an `entity_manifest` table-vector at vtable `slot`, mirroring the cuts
/// read block. An absent field yields an empty `Vec` (graceful absence).
fn read_entity_manifest(
    buf: &[u8],
    table_pos: usize,
    vtable_pos: usize,
    slot: u16,
    ctx: &str,
) -> Result<Vec<EntitySlot>, OutputError> {
    let Some(field_pos) = field_pos(buf, table_pos, vtable_pos, slot) else {
        return Ok(Vec::new());
    };
    let vec_pos = follow_uoffset(buf, field_pos).ok_or_else(|| {
        OutputError::serialization(ctx, "invalid uoffset for entity_manifest vector")
    })?;
    let nested_positions = read_table_vector_positions(buf, vec_pos).ok_or_else(|| {
        OutputError::serialization(ctx, "entity_manifest vector header truncated or corrupt")
    })?;

    let mut out = Vec::with_capacity(nested_positions.len());
    for (idx, &slot_table_pos) in nested_positions.iter().enumerate() {
        let entry = deserialize_entity_slot_table(buf, slot_table_pos).ok_or_else(|| {
            OutputError::serialization(ctx, format!("entity_slot table {idx} truncated or corrupt"))
        })?;
        out.push(entry);
    }
    Ok(out)
}

fn deserialize_entity_slot_table(buf: &[u8], slot_table_pos: usize) -> Option<EntitySlot> {
    let vtable_pos = resolve_vtable_pos(buf, slot_table_pos)?;

    let entity_type = field_pos(
        buf,
        slot_table_pos,
        vtable_pos,
        ENTITY_SLOT_FIELD_ENTITY_TYPE,
    )
    .and_then(|p| buf.get(p).copied())
    .unwrap_or(0);

    let entity_id = field_pos(buf, slot_table_pos, vtable_pos, ENTITY_SLOT_FIELD_ENTITY_ID)
        .and_then(|p| read_i32_le(buf, p))
        .unwrap_or(0);

    let subindex = field_pos(buf, slot_table_pos, vtable_pos, ENTITY_SLOT_FIELD_SUBINDEX)
        .and_then(|p| read_u32_le(buf, p))
        .unwrap_or(0);

    let was_active = field_pos(
        buf,
        slot_table_pos,
        vtable_pos,
        ENTITY_SLOT_FIELD_WAS_ACTIVE,
    )
    .and_then(|p| read_bool_byte(buf, p))
    .unwrap_or(false);

    // Absent in a buffer written before these ids existed (FlatBuffers graceful
    // absence): default each to the sentinel, not zero — zero is a valid
    // calendar value.
    let reference_date = field_pos(
        buf,
        slot_table_pos,
        vtable_pos,
        ENTITY_SLOT_FIELD_REFERENCE_DATE,
    )
    .and_then(|p| read_i32_le(buf, p))
    .unwrap_or(ENTITY_SLOT_DATE_SENTINEL);

    let interval_start = field_pos(
        buf,
        slot_table_pos,
        vtable_pos,
        ENTITY_SLOT_FIELD_INTERVAL_START,
    )
    .and_then(|p| read_i32_le(buf, p))
    .unwrap_or(ENTITY_SLOT_DATE_SENTINEL);

    let interval_end = field_pos(
        buf,
        slot_table_pos,
        vtable_pos,
        ENTITY_SLOT_FIELD_INTERVAL_END,
    )
    .and_then(|p| read_i32_le(buf, p))
    .unwrap_or(ENTITY_SLOT_DATE_SENTINEL);

    Some(EntitySlot {
        entity_type,
        entity_id,
        subindex,
        was_active,
        reference_date,
        interval_start,
        interval_end,
    })
}

// The next five helpers (`read_string_field` through `read_manifest_edges`)
// share one absence contract: a field missing from the vtable yields an empty
// `String`/`Vec`, never an error (graceful absence).

fn read_string_field(
    buf: &[u8],
    table_pos: usize,
    vtable_pos: usize,
    slot: u16,
    ctx: &str,
) -> Result<String, OutputError> {
    let Some(field_pos) = field_pos(buf, table_pos, vtable_pos, slot) else {
        return Ok(String::new());
    };
    let str_pos = follow_uoffset(buf, field_pos)
        .ok_or_else(|| OutputError::serialization(ctx, "invalid uoffset for string field"))?;
    read_string(buf, str_pos)
        .ok_or_else(|| OutputError::serialization(ctx, "string field truncated or not UTF-8"))
}

fn read_u32_vector_field(
    buf: &[u8],
    table_pos: usize,
    vtable_pos: usize,
    slot: u16,
    ctx: &str,
) -> Result<Vec<u32>, OutputError> {
    let Some(field_pos) = field_pos(buf, table_pos, vtable_pos, slot) else {
        return Ok(Vec::new());
    };
    let vec_pos = follow_uoffset(buf, field_pos)
        .ok_or_else(|| OutputError::serialization(ctx, "invalid uoffset for uint32 vector"))?;
    read_u32_vector(buf, vec_pos)
        .ok_or_else(|| OutputError::serialization(ctx, "uint32 vector truncated or corrupt"))
}

fn read_f64_vector_field(
    buf: &[u8],
    table_pos: usize,
    vtable_pos: usize,
    slot: u16,
    ctx: &str,
) -> Result<Vec<f64>, OutputError> {
    let Some(field_pos) = field_pos(buf, table_pos, vtable_pos, slot) else {
        return Ok(Vec::new());
    };
    let vec_pos = follow_uoffset(buf, field_pos)
        .ok_or_else(|| OutputError::serialization(ctx, "invalid uoffset for float64 vector"))?;
    read_f64_vector(buf, vec_pos)
        .ok_or_else(|| OutputError::serialization(ctx, "float64 vector truncated or corrupt"))
}

fn read_string_vector_field(
    buf: &[u8],
    table_pos: usize,
    vtable_pos: usize,
    slot: u16,
    ctx: &str,
) -> Result<Vec<String>, OutputError> {
    let Some(field_pos) = field_pos(buf, table_pos, vtable_pos, slot) else {
        return Ok(Vec::new());
    };
    let vec_pos = follow_uoffset(buf, field_pos)
        .ok_or_else(|| OutputError::serialization(ctx, "invalid uoffset for string vector"))?;
    let positions = read_table_vector_positions(buf, vec_pos)
        .ok_or_else(|| OutputError::serialization(ctx, "string vector header truncated"))?;
    let mut out = Vec::with_capacity(positions.len());
    for (idx, &str_pos) in positions.iter().enumerate() {
        let entry = read_string(buf, str_pos).ok_or_else(|| {
            OutputError::serialization(ctx, format!("string vector element {idx} truncated"))
        })?;
        out.push(entry);
    }
    Ok(out)
}

fn read_manifest_nodes(
    buf: &[u8],
    table_pos: usize,
    vtable_pos: usize,
    slot: u16,
    ctx: &str,
) -> Result<Vec<ManifestNode>, OutputError> {
    let Some(field_pos) = field_pos(buf, table_pos, vtable_pos, slot) else {
        return Ok(Vec::new());
    };
    let vec_pos = follow_uoffset(buf, field_pos)
        .ok_or_else(|| OutputError::serialization(ctx, "invalid uoffset for nodes vector"))?;
    let positions = read_table_vector_positions(buf, vec_pos)
        .ok_or_else(|| OutputError::serialization(ctx, "nodes vector header truncated"))?;
    let mut out = Vec::with_capacity(positions.len());
    for (idx, &node_pos) in positions.iter().enumerate() {
        let node = deserialize_manifest_node_table(buf, node_pos).ok_or_else(|| {
            OutputError::serialization(ctx, format!("manifest node table {idx} truncated"))
        })?;
        out.push(node);
    }
    Ok(out)
}

fn read_manifest_edges(
    buf: &[u8],
    table_pos: usize,
    vtable_pos: usize,
    slot: u16,
    ctx: &str,
) -> Result<Vec<ManifestEdge>, OutputError> {
    let Some(field_pos) = field_pos(buf, table_pos, vtable_pos, slot) else {
        return Ok(Vec::new());
    };
    let vec_pos = follow_uoffset(buf, field_pos)
        .ok_or_else(|| OutputError::serialization(ctx, "invalid uoffset for edges vector"))?;
    let positions = read_table_vector_positions(buf, vec_pos)
        .ok_or_else(|| OutputError::serialization(ctx, "edges vector header truncated"))?;
    let mut out = Vec::with_capacity(positions.len());
    for (idx, &edge_pos) in positions.iter().enumerate() {
        let edge = deserialize_manifest_edge_table(buf, edge_pos).ok_or_else(|| {
            OutputError::serialization(ctx, format!("manifest edge table {idx} truncated"))
        })?;
        out.push(edge);
    }
    Ok(out)
}

/// Unlike [`read_string_field`], an absent field reads as `None`, not `""`.
fn read_optional_string_field(
    buf: &[u8],
    table_pos: usize,
    vtable_pos: usize,
    slot: u16,
    ctx: &str,
) -> Result<Option<String>, OutputError> {
    if field_pos(buf, table_pos, vtable_pos, slot).is_none() {
        return Ok(None);
    }
    read_string_field(buf, table_pos, vtable_pos, slot, ctx).map(Some)
}

fn deserialize_manifest_node_table(buf: &[u8], node_table_pos: usize) -> Option<ManifestNode> {
    let vtable_pos = resolve_vtable_pos(buf, node_table_pos)?;

    let id = field_pos(buf, node_table_pos, vtable_pos, MANIFEST_NODE_FIELD_ID)
        .and_then(|p| read_i32_le(buf, p))
        .unwrap_or(0);
    let stage_id = field_pos(
        buf,
        node_table_pos,
        vtable_pos,
        MANIFEST_NODE_FIELD_STAGE_ID,
    )
    .and_then(|p| read_i32_le(buf, p))
    .unwrap_or(0);
    let pool_id = field_pos(buf, node_table_pos, vtable_pos, MANIFEST_NODE_FIELD_POOL_ID)
        .and_then(|p| read_u32_le(buf, p))
        .unwrap_or(0);

    Some(ManifestNode {
        id,
        stage_id,
        pool_id,
    })
}

fn deserialize_manifest_edge_table(buf: &[u8], edge_table_pos: usize) -> Option<ManifestEdge> {
    let vtable_pos = resolve_vtable_pos(buf, edge_table_pos)?;

    let source_id = field_pos(
        buf,
        edge_table_pos,
        vtable_pos,
        MANIFEST_EDGE_FIELD_SOURCE_ID,
    )
    .and_then(|p| read_i32_le(buf, p))
    .unwrap_or(0);
    let target_id = field_pos(
        buf,
        edge_table_pos,
        vtable_pos,
        MANIFEST_EDGE_FIELD_TARGET_ID,
    )
    .and_then(|p| read_i32_le(buf, p))
    .unwrap_or(0);
    let probability = field_pos(
        buf,
        edge_table_pos,
        vtable_pos,
        MANIFEST_EDGE_FIELD_PROBABILITY,
    )
    .and_then(|p| read_f64_le(buf, p))
    .unwrap_or(0.0);

    Some(ManifestEdge {
        source_id,
        target_id,
        probability,
    })
}

/// Read the `season_manifest` nested table at vtable `slot`. An absent field
/// yields [`SeasonManifest::default`] (graceful absence), never an error —
/// the same contract [`read_manifest_nodes`] follows.
fn read_season_manifest(
    buf: &[u8],
    table_pos: usize,
    vtable_pos: usize,
    slot: u16,
    ctx: &str,
) -> Result<SeasonManifest, OutputError> {
    let Some(field_pos) = field_pos(buf, table_pos, vtable_pos, slot) else {
        return Ok(SeasonManifest::default());
    };
    let season_table_pos = follow_uoffset(buf, field_pos).ok_or_else(|| {
        OutputError::serialization(ctx, "invalid uoffset for season_manifest table")
    })?;
    let season_manifest =
        deserialize_season_manifest_table(buf, season_table_pos).ok_or_else(|| {
            OutputError::serialization(ctx, "season_manifest table truncated or corrupt")
        })?;
    validate_season_manifest_shape(&season_manifest, ctx)?;
    Ok(season_manifest)
}

/// Reject a decoded `season_manifest` whose `hydro_orders` is not canonically
/// ascending by `hydro_id`, or whose per-hydro `orders` length disagrees with
/// `n_seasons` — the two shape invariants a positional season-identity
/// comparison assumes without re-checking. A writer-only bug (never observed
/// from this crate's own writer, which builds `hydro_orders` from a
/// `BTreeMap`) would otherwise reach a consumer as silent misalignment
/// instead of a decode-time reject.
fn validate_season_manifest_shape(manifest: &SeasonManifest, ctx: &str) -> Result<(), OutputError> {
    if let Some(pair) = manifest
        .hydro_orders
        .windows(2)
        .find(|pair| pair[0].hydro_id >= pair[1].hydro_id)
    {
        return Err(OutputError::serialization(
            ctx,
            format!(
                "season_manifest hydro_orders not ascending by hydro_id: {} then {}",
                pair[0].hydro_id, pair[1].hydro_id
            ),
        ));
    }
    if let Some(bad) = manifest
        .hydro_orders
        .iter()
        .find(|h| h.orders.len() != manifest.n_seasons as usize)
    {
        return Err(OutputError::serialization(
            ctx,
            format!(
                "season_manifest hydro_id {} has {} orders, expected n_seasons={}",
                bad.hydro_id,
                bad.orders.len(),
                manifest.n_seasons
            ),
        ));
    }
    Ok(())
}

fn deserialize_season_manifest_table(buf: &[u8], table_pos: usize) -> Option<SeasonManifest> {
    let vtable_pos = resolve_vtable_pos(buf, table_pos)?;

    let cycle_code = field_pos(buf, table_pos, vtable_pos, SEASON_MANIFEST_FIELD_CYCLE_CODE)
        .and_then(|p| buf.get(p).copied())
        .unwrap_or(SEASON_CYCLE_CODE_ABSENT);

    let n_seasons = field_pos(buf, table_pos, vtable_pos, SEASON_MANIFEST_FIELD_N_SEASONS)
        .and_then(|p| read_u32_le(buf, p))
        .unwrap_or(0);

    let hydro_orders = match field_pos(
        buf,
        table_pos,
        vtable_pos,
        SEASON_MANIFEST_FIELD_HYDRO_ORDERS,
    ) {
        None => Vec::new(),
        Some(field_pos) => {
            let vec_pos = follow_uoffset(buf, field_pos)?;
            let positions = read_table_vector_positions(buf, vec_pos)?;
            let mut out = Vec::with_capacity(positions.len());
            for &pos in &positions {
                out.push(deserialize_hydro_season_orders_table(buf, pos)?);
            }
            out
        }
    };

    Some(SeasonManifest {
        cycle_code,
        n_seasons,
        hydro_orders,
    })
}

fn deserialize_hydro_season_orders_table(
    buf: &[u8],
    table_pos: usize,
) -> Option<HydroSeasonOrders> {
    let vtable_pos = resolve_vtable_pos(buf, table_pos)?;

    let hydro_id = field_pos(
        buf,
        table_pos,
        vtable_pos,
        HYDRO_SEASON_ORDERS_FIELD_HYDRO_ID,
    )
    .and_then(|p| read_i32_le(buf, p))
    .unwrap_or(0);

    let orders = match field_pos(buf, table_pos, vtable_pos, HYDRO_SEASON_ORDERS_FIELD_ORDERS) {
        None => Vec::new(),
        Some(field_pos) => {
            let vec_pos = follow_uoffset(buf, field_pos)?;
            read_u32_vector(buf, vec_pos)?
        }
    };

    Some(HydroSeasonOrders { hydro_id, orders })
}

// ── Deserializers ─────────────────────────────────────────────────────────────

/// Deserialize a `StageCuts` `FlatBuffers` buffer into an owned [`StageCutsReadResult`].
///
/// # Errors
///
/// Returns [`OutputError::SerializationError`] if the buffer is truncated, corrupted,
/// or otherwise does not conform to the expected layout.
///
/// # Examples
///
/// ```
/// use cobre_io::{
///     PolicyCutRecord, STAGE_CUTS_PRICED_STATE_DATE_SENTINEL, StageCutsPayload,
///     deserialize_stage_cuts, serialize_stage_cuts,
/// };
///
/// let piece = PolicyCutRecord {
///     cut_id: 7,
///     slot_index: 5,
///     iteration: 3,
///     forward_pass_index: 1,
///     intercept: 42.0,
///     coefficients: &[1.0, 2.0, 3.0],
///     is_active: true,
/// };
/// let buf = serialize_stage_cuts(&StageCutsPayload {
///     stage_id: 2,
///     state_dimension: 3,
///     capacity: 100,
///     warm_start_count: 0,
///     cuts: &[piece],
///     active_cut_indices: &[0],
///     populated_count: 1,
///     entity_manifest: &[],
///     cost_scale_factor: 1_000_000.0,
///     node_id: 2,
///     graph_stage_id: 2,
///     priced_state_date: STAGE_CUTS_PRICED_STATE_DATE_SENTINEL,
/// });
/// let result = deserialize_stage_cuts(&buf).expect("round-trip must succeed");
/// assert_eq!(result.stage_id, 2);
/// assert_eq!(result.cuts.len(), 1);
/// assert_eq!(result.cuts[0].cut_id, 7);
/// assert_eq!(result.cuts[0].coefficients, &[1.0, 2.0, 3.0]);
/// assert_eq!(result.cost_scale_factor, Some(1_000_000.0));
/// assert_eq!(result.priced_state_date, STAGE_CUTS_PRICED_STATE_DATE_SENTINEL);
/// ```
pub fn deserialize_stage_cuts(buf: &[u8]) -> Result<StageCutsReadResult, OutputError> {
    let ctx = "stage_cuts";
    check_file_identifier(buf, ctx)?;

    let table_pos = resolve_root(buf)
        .ok_or_else(|| OutputError::serialization(ctx, "buffer too short for root offset"))?;

    let vtable_pos = resolve_vtable_pos(buf, table_pos)
        .ok_or_else(|| OutputError::serialization(ctx, "invalid soffset_to_vtable"))?;

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

    let cuts = if let Some(cuts_field_pos) =
        field_pos(buf, table_pos, vtable_pos, STAGE_CUTS_FIELD_CUTS)
    {
        let vec_pos = follow_uoffset(buf, cuts_field_pos)
            .ok_or_else(|| OutputError::serialization(ctx, "invalid uoffset for cuts vector"))?;

        let nested_positions = read_table_vector_positions(buf, vec_pos).ok_or_else(|| {
            OutputError::serialization(ctx, "cuts vector header truncated or corrupt")
        })?;

        let mut out = Vec::with_capacity(nested_positions.len());
        for (idx, &piece_table_pos) in nested_positions.iter().enumerate() {
            let piece = deserialize_cut_table(buf, piece_table_pos).ok_or_else(|| {
                OutputError::serialization(
                    ctx,
                    format!("affine-piece table {idx} truncated or corrupt"),
                )
            })?;
            out.push(piece);
        }
        out
    } else {
        Vec::new()
    };

    let entity_manifest = read_entity_manifest(
        buf,
        table_pos,
        vtable_pos,
        STAGE_CUTS_FIELD_ENTITY_MANIFEST,
        ctx,
    )?;

    // Absent in a pre-`id:8` buffer (FlatBuffers graceful absence): cost_scale_factor
    // stays `None` (distinct from a real `0.0`), the two ids default to the sentinel
    // (never a bare `0` — `0` is a valid node/stage id).
    let cost_scale_factor = field_pos(
        buf,
        table_pos,
        vtable_pos,
        STAGE_CUTS_FIELD_COST_SCALE_FACTOR,
    )
    .and_then(|p| read_f64_le(buf, p));

    let node_id = field_pos(buf, table_pos, vtable_pos, STAGE_CUTS_FIELD_NODE_ID)
        .and_then(|p| read_i32_le(buf, p))
        .unwrap_or(STAGE_CUTS_NODE_ID_SENTINEL);

    let graph_stage_id = field_pos(buf, table_pos, vtable_pos, STAGE_CUTS_FIELD_GRAPH_STAGE_ID)
        .and_then(|p| read_i32_le(buf, p))
        .unwrap_or(STAGE_CUTS_GRAPH_STAGE_ID_SENTINEL);

    let priced_state_date = field_pos(
        buf,
        table_pos,
        vtable_pos,
        STAGE_CUTS_FIELD_PRICED_STATE_DATE,
    )
    .and_then(|p| read_i32_le(buf, p))
    .unwrap_or(STAGE_CUTS_PRICED_STATE_DATE_SENTINEL);

    Ok(StageCutsReadResult {
        stage_id,
        state_dimension,
        capacity,
        warm_start_count,
        populated_count,
        cuts,
        entity_manifest,
        cost_scale_factor,
        node_id,
        graph_stage_id,
        priced_state_date,
    })
}

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

    Some(OwnedPolicyCutRecord {
        cut_id,
        slot_index,
        iteration,
        forward_pass_index,
        intercept,
        coefficients,
        is_active,
    })
}

/// Deserialize a `StageBasis` `FlatBuffers` buffer into an owned [`OwnedPolicyBasisRecord`].
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
    check_file_identifier(buf, ctx)?;

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

/// Deserialize one stage's visited states from a `StageStates` `FlatBuffers` buffer.
///
/// # Errors
///
/// Returns [`OutputError::SerializationError`] if the buffer is truncated or
/// has an invalid wire format.
pub fn deserialize_stage_states(buf: &[u8]) -> Result<StageStatesReadResult, OutputError> {
    let ctx = "stage_states";
    check_file_identifier(buf, ctx)?;

    let table_pos = resolve_root(buf)
        .ok_or_else(|| OutputError::serialization(ctx, "buffer too short for root offset"))?;

    let vtable_pos = resolve_vtable_pos(buf, table_pos)
        .ok_or_else(|| OutputError::serialization(ctx, "invalid soffset_to_vtable"))?;

    let stage_id = field_pos(buf, table_pos, vtable_pos, STATES_FIELD_STAGE_ID)
        .and_then(|p| read_u32_le(buf, p))
        .unwrap_or(0);

    let state_dimension = field_pos(buf, table_pos, vtable_pos, STATES_FIELD_STATE_DIMENSION)
        .and_then(|p| read_u32_le(buf, p))
        .unwrap_or(0);

    let count = field_pos(buf, table_pos, vtable_pos, STATES_FIELD_COUNT)
        .and_then(|p| read_u32_le(buf, p))
        .unwrap_or(0);

    let data = if let Some(data_field_pos) =
        field_pos(buf, table_pos, vtable_pos, STATES_FIELD_DATA)
    {
        let vec_pos = follow_uoffset(buf, data_field_pos)
            .ok_or_else(|| OutputError::serialization(ctx, "invalid uoffset for data vector"))?;
        read_f64_vector(buf, vec_pos)
            .ok_or_else(|| OutputError::serialization(ctx, "data vector truncated"))?
    } else {
        Vec::new()
    };

    let entity_manifest = read_entity_manifest(
        buf,
        table_pos,
        vtable_pos,
        STATES_FIELD_ENTITY_MANIFEST,
        ctx,
    )?;

    // Absent in a pre-`id:5` buffer (FlatBuffers graceful absence): default to
    // the sentinel, never a bare 0 — 0 is a valid node id.
    let node_id = field_pos(buf, table_pos, vtable_pos, STATES_FIELD_NODE_ID)
        .and_then(|p| read_i32_le(buf, p))
        .unwrap_or(STAGE_STATES_NODE_ID_SENTINEL);

    Ok(StageStatesReadResult {
        stage_id,
        node_id,
        state_dimension,
        count,
        data,
        entity_manifest,
    })
}

/// Deserialize a `CheckpointManifest` `FlatBuffers` buffer into an owned
/// [`CheckpointManifest`].
///
/// # Errors
///
/// Returns [`OutputError::SerializationError`] when the buffer lacks the `CBVF`
/// identifier, is truncated or corrupt, or carries a `format_version` other than
/// [`FORMAT_VERSION`] (an absent version field reads as `0` and is likewise
/// rejected), so a stale-version manifest is refused before any consumer reads it.
pub fn deserialize_checkpoint_manifest(buf: &[u8]) -> Result<CheckpointManifest, OutputError> {
    let ctx = "checkpoint_manifest";
    check_file_identifier(buf, ctx)?;

    let table_pos = resolve_root(buf)
        .ok_or_else(|| OutputError::serialization(ctx, "buffer too short for root offset"))?;

    let vtable_pos = resolve_vtable_pos(buf, table_pos)
        .ok_or_else(|| OutputError::serialization(ctx, "invalid soffset_to_vtable"))?;

    let format_version = field_pos(buf, table_pos, vtable_pos, MANIFEST_FIELD_FORMAT_VERSION)
        .and_then(|p| read_u32_le(buf, p))
        .unwrap_or(0);
    if format_version != FORMAT_VERSION {
        return Err(OutputError::serialization(
            ctx,
            format!(
                "unsupported checkpoint manifest format_version {format_version}; expected \
                 {FORMAT_VERSION}; {remedy}",
                remedy = policy_checkpoint_remedy()
            ),
        ));
    }

    let software =
        read_optional_string_field(buf, table_pos, vtable_pos, MANIFEST_FIELD_SOFTWARE, ctx)?;
    let software_version = read_string_field(
        buf,
        table_pos,
        vtable_pos,
        MANIFEST_FIELD_SOFTWARE_VERSION,
        ctx,
    )?;
    let created_at = read_string_field(buf, table_pos, vtable_pos, MANIFEST_FIELD_CREATED_AT, ctx)?;

    let num_stages = field_pos(buf, table_pos, vtable_pos, MANIFEST_FIELD_NUM_STAGES)
        .and_then(|p| read_u32_le(buf, p))
        .unwrap_or(0);
    let n_pools = field_pos(buf, table_pos, vtable_pos, MANIFEST_FIELD_N_POOLS)
        .and_then(|p| read_u32_le(buf, p))
        .unwrap_or(0);

    let nodes = read_manifest_nodes(buf, table_pos, vtable_pos, MANIFEST_FIELD_NODES, ctx)?;
    let edges = read_manifest_edges(buf, table_pos, vtable_pos, MANIFEST_FIELD_EDGES, ctx)?;

    let completed_iterations = field_pos(
        buf,
        table_pos,
        vtable_pos,
        MANIFEST_FIELD_COMPLETED_ITERATIONS,
    )
    .and_then(|p| read_u32_le(buf, p))
    .unwrap_or(0);

    // A plain-f64 provenance field: absent reads as 0.0, matching the intercept
    // read. The two Option<f64> fields below stay None when absent, never 0.0.
    let final_lower_bound = field_pos(buf, table_pos, vtable_pos, MANIFEST_FIELD_FINAL_LOWER_BOUND)
        .and_then(|p| read_f64_le(buf, p))
        .unwrap_or(0.0);
    let best_upper_bound = field_pos(buf, table_pos, vtable_pos, MANIFEST_FIELD_BEST_UPPER_BOUND)
        .and_then(|p| read_f64_le(buf, p));

    let max_iterations = field_pos(buf, table_pos, vtable_pos, MANIFEST_FIELD_MAX_ITERATIONS)
        .and_then(|p| read_u32_le(buf, p))
        .unwrap_or(0);
    let forward_passes = field_pos(buf, table_pos, vtable_pos, MANIFEST_FIELD_FORWARD_PASSES)
        .and_then(|p| read_u32_le(buf, p))
        .unwrap_or(0);
    let warm_start_cuts = field_pos(buf, table_pos, vtable_pos, MANIFEST_FIELD_WARM_START_CUTS)
        .and_then(|p| read_u32_le(buf, p))
        .unwrap_or(0);
    let warm_start_counts = read_u32_vector_field(
        buf,
        table_pos,
        vtable_pos,
        MANIFEST_FIELD_WARM_START_COUNTS,
        ctx,
    )?;

    let rng_seed = field_pos(buf, table_pos, vtable_pos, MANIFEST_FIELD_RNG_SEED)
        .and_then(|p| read_u64_le(buf, p))
        .unwrap_or(0);
    let total_visited_states = field_pos(
        buf,
        table_pos,
        vtable_pos,
        MANIFEST_FIELD_TOTAL_VISITED_STATES,
    )
    .and_then(|p| read_u64_le(buf, p))
    .unwrap_or(0);

    let training_block_mode = read_string_field(
        buf,
        table_pos,
        vtable_pos,
        MANIFEST_FIELD_TRAINING_BLOCK_MODE,
        ctx,
    )?;
    let training_block_mode_per_stage = read_string_vector_field(
        buf,
        table_pos,
        vtable_pos,
        MANIFEST_FIELD_TRAINING_BLOCK_MODE_PER_STAGE,
        ctx,
    )?;

    let cost_scale_factor = field_pos(buf, table_pos, vtable_pos, MANIFEST_FIELD_COST_SCALE_FACTOR)
        .and_then(|p| read_f64_le(buf, p));

    let season_manifest = read_season_manifest(
        buf,
        table_pos,
        vtable_pos,
        MANIFEST_FIELD_SEASON_MANIFEST,
        ctx,
    )?;

    let lower_bound_history = read_f64_vector_field(
        buf,
        table_pos,
        vtable_pos,
        MANIFEST_FIELD_LOWER_BOUND_HISTORY,
        ctx,
    )?;

    Ok(CheckpointManifest {
        format_version,
        software,
        software_version,
        created_at,
        num_stages,
        graph_manifest: GraphManifest {
            n_pools,
            nodes,
            edges,
        },
        producer: ProducerBlock {
            completed_iterations,
            final_lower_bound,
            best_upper_bound,
            max_iterations,
            forward_passes,
            warm_start_cuts,
            warm_start_counts,
            rng_seed,
            total_visited_states,
            training_block_mode,
            training_block_mode_per_stage,
            cost_scale_factor,
            lower_bound_history,
        },
        season_manifest,
    })
}

/// Read all `*.bin` files from `dir`, deserialize each with `deser_fn`, and return a `Vec`.
///
/// The returned `Vec` is unsorted — callers must sort by `stage_id` after this call
/// (`read_dir` order is not guaranteed; sorting upholds declaration-order invariance).
pub(super) fn read_sorted_bin_files<T, F>(
    dir: &Path,
    ctx: &str,
    deser_fn: F,
) -> Result<Vec<T>, OutputError>
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
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::super::records::SEASON_CYCLE_CODE_MONTHLY;
    use super::*;

    /// The reject-old-version half of the dual-owned wire-format checklist: a
    /// buffer without the `CBVF` `file_identifier` (a pre-0.14 `finish_minimal`
    /// artifact) is rejected by the read path before any field is decoded, so no
    /// stale month-integer anchor is ever decoded as a `YYYYMMDD` date.
    #[test]
    fn deserialize_rejects_buffer_without_cbvf_identifier() {
        let coeffs = [1.0_f64, 2.0];
        let cut = PolicyCutRecord {
            cut_id: 1,
            slot_index: 0,
            iteration: 1,
            forward_pass_index: 0,
            intercept: 3.0,
            coefficients: &coeffs,
            is_active: true,
        };
        let mut buf = serialize_stage_cuts(&StageCutsPayload {
            stage_id: 0,
            state_dimension: 2,
            capacity: 8,
            warm_start_count: 0,
            cuts: &[cut],
            active_cut_indices: &[0],
            populated_count: 1,
            entity_manifest: &[],
            cost_scale_factor: 1_000_000.0,
            node_id: -1,
            graph_stage_id: -1,
            priced_state_date: STAGE_CUTS_PRICED_STATE_DATE_SENTINEL,
        });
        assert_eq!(
            buf.get(4..8),
            Some(POLICY_FILE_IDENTIFIER.as_bytes()),
            "a freshly written buffer must carry the CBVF identifier"
        );

        // Strip the identifier to mimic a pre-0.14 `finish_minimal` buffer.
        buf[4..8].copy_from_slice(&[0, 0, 0, 0]);
        let err = deserialize_stage_cuts(&buf)
            .expect_err("a buffer without the CBVF identifier must be rejected");
        assert!(
            err.to_string().contains(POLICY_FILE_IDENTIFIER),
            "rejection must name the expected identifier: {err}"
        );
    }

    /// `EntitySlot`'s `reference_date`/`interval_start`/`interval_end` (ids
    /// 6/7/8) round-trip through the writer/reader pair, each independently of
    /// the others.
    #[test]
    fn entity_slot_new_date_fields_round_trip() {
        let coeffs = [1.0_f64, 2.0];
        let cut = PolicyCutRecord {
            cut_id: 1,
            slot_index: 0,
            iteration: 1,
            forward_pass_index: 0,
            intercept: 3.0,
            coefficients: &coeffs,
            is_active: true,
        };
        let manifest = [
            EntitySlot::inflow_lag(5, 3, true).with_reference_date(20_310_401),
            EntitySlot::transit_bucket(9, 2, true).with_interval(20_311_201, 20_320_101),
        ];
        let buf = serialize_stage_cuts(&StageCutsPayload {
            stage_id: 0,
            state_dimension: 2,
            capacity: 8,
            warm_start_count: 0,
            cuts: &[cut],
            active_cut_indices: &[0],
            populated_count: 1,
            entity_manifest: &manifest,
            cost_scale_factor: 1_000_000.0,
            node_id: -1,
            graph_stage_id: -1,
            priced_state_date: STAGE_CUTS_PRICED_STATE_DATE_SENTINEL,
        });

        let result = deserialize_stage_cuts(&buf).expect("round-trip must succeed");
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
    }

    #[test]
    fn stage_cuts_priced_state_date_round_trips() {
        let coeffs = [1.0_f64, 2.0];
        let cut = PolicyCutRecord {
            cut_id: 1,
            slot_index: 0,
            iteration: 1,
            forward_pass_index: 0,
            intercept: 3.0,
            coefficients: &coeffs,
            is_active: true,
        };
        let buf = serialize_stage_cuts(&StageCutsPayload {
            stage_id: 0,
            state_dimension: 2,
            capacity: 8,
            warm_start_count: 0,
            cuts: &[cut],
            active_cut_indices: &[0],
            populated_count: 1,
            entity_manifest: &[],
            cost_scale_factor: 1_000_000.0,
            node_id: -1,
            graph_stage_id: -1,
            priced_state_date: 20_311_201,
        });

        let result = deserialize_stage_cuts(&buf).expect("round-trip must succeed");
        assert_eq!(result.priced_state_date, 20_311_201);
    }

    /// A buffer built with only ids 0–10 present (the id-11 slot never
    /// pushed) decodes `priced_state_date` as the sentinel, never `0`.
    #[test]
    fn stage_cuts_absent_priced_state_date_reads_as_sentinel() {
        let mut builder = FlatBufferBuilder::with_capacity(64);
        let cuts_vec = builder.create_vector::<WIPOffset<flatbuffers::TableFinishedWIPOffset>>(&[]);
        let active_vec = builder.create_vector::<u32>(&[]);
        let manifest_vec =
            builder.create_vector::<WIPOffset<flatbuffers::TableFinishedWIPOffset>>(&[]);

        let root = builder.start_table();
        builder.push_slot_always::<u32>(STAGE_CUTS_FIELD_STAGE_ID, 3);
        builder.push_slot_always::<u32>(STAGE_CUTS_FIELD_STATE_DIMENSION, 0);
        builder.push_slot_always::<u32>(STAGE_CUTS_FIELD_CAPACITY, 0);
        builder.push_slot_always::<u32>(STAGE_CUTS_FIELD_WARM_START_COUNT, 0);
        builder.push_slot_always(STAGE_CUTS_FIELD_CUTS, cuts_vec);
        builder.push_slot_always(STAGE_CUTS_FIELD_ACTIVE_CUT_INDICES, active_vec);
        builder.push_slot_always::<u32>(STAGE_CUTS_FIELD_POPULATED_COUNT, 0);
        builder.push_slot_always(STAGE_CUTS_FIELD_ENTITY_MANIFEST, manifest_vec);
        builder.push_slot_always::<f64>(STAGE_CUTS_FIELD_COST_SCALE_FACTOR, 1_000_000.0);
        builder.push_slot_always::<i32>(STAGE_CUTS_FIELD_NODE_ID, -1);
        builder.push_slot_always::<i32>(STAGE_CUTS_FIELD_GRAPH_STAGE_ID, -1);
        // STAGE_CUTS_FIELD_PRICED_STATE_DATE (id 11) deliberately omitted.
        let root_offset = builder.end_table(root);
        builder.finish(root_offset, Some(POLICY_FILE_IDENTIFIER));

        let buf = builder.finished_data().to_vec();
        let result = deserialize_stage_cuts(&buf)
            .expect("a buffer with only ids 0-10 present must still decode");
        assert_eq!(
            result.priced_state_date,
            STAGE_CUTS_PRICED_STATE_DATE_SENTINEL
        );
    }

    fn minimal_manifest_producer() -> ProducerBlock {
        ProducerBlock {
            completed_iterations: 0,
            final_lower_bound: 0.0,
            best_upper_bound: None,
            max_iterations: 0,
            forward_passes: 0,
            warm_start_cuts: 0,
            warm_start_counts: vec![],
            rng_seed: 0,
            total_visited_states: 0,
            training_block_mode: "parallel".to_string(),
            training_block_mode_per_stage: vec![],
            cost_scale_factor: None,
            lower_bound_history: Vec::new(),
        }
    }

    #[test]
    fn checkpoint_manifest_software_round_trips_and_absent_reads_as_none() {
        for software in [Some("cobre".to_string()), None] {
            let manifest = CheckpointManifest {
                format_version: FORMAT_VERSION,
                software: software.clone(),
                software_version: "0.14.0".to_string(),
                created_at: "2026-09-15T00:00:00Z".to_string(),
                num_stages: 1,
                graph_manifest: GraphManifest::default(),
                producer: minimal_manifest_producer(),
                season_manifest: SeasonManifest::default(),
            };

            let buf = serialize_checkpoint_manifest(&manifest);
            let decoded = deserialize_checkpoint_manifest(&buf).expect("round-trip must succeed");

            assert_eq!(decoded.software, software);
            assert_eq!(decoded.software_version, "0.14.0");
        }
    }

    #[test]
    fn checkpoint_manifest_round_trips_the_lower_bound_history_bitwise() {
        let history = [1.5, -0.0, f64::MIN_POSITIVE, 1.0e300];
        let manifest = CheckpointManifest {
            format_version: FORMAT_VERSION,
            software: Some("cobre".to_string()),
            software_version: "0.18.0".to_string(),
            created_at: "2026-10-06T00:00:00Z".to_string(),
            num_stages: 1,
            graph_manifest: GraphManifest::default(),
            producer: ProducerBlock {
                lower_bound_history: history.to_vec(),
                ..minimal_manifest_producer()
            },
            season_manifest: SeasonManifest::default(),
        };

        let buf = serialize_checkpoint_manifest(&manifest);
        let decoded = deserialize_checkpoint_manifest(&buf).expect("round-trip must succeed");

        let decoded_bits: Vec<u64> = decoded
            .producer
            .lower_bound_history
            .iter()
            .copied()
            .map(f64::to_bits)
            .collect();
        let expected_bits: Vec<u64> = history.iter().copied().map(f64::to_bits).collect();
        assert_eq!(decoded_bits, expected_bits);
    }

    #[test]
    fn checkpoint_manifest_without_the_lower_bound_history_reads_an_empty_series() {
        let mut builder = FlatBufferBuilder::with_capacity(128);

        let season_manifest_offset =
            build_season_manifest_table(&mut builder, &SeasonManifest::default());
        let software = builder.create_string("cobre");
        let software_version = builder.create_string("0.18.0");
        let created_at = builder.create_string("2026-10-06T00:00:00Z");
        let training_block_mode = builder.create_string("parallel");
        let nodes_vec =
            builder.create_vector::<WIPOffset<flatbuffers::TableFinishedWIPOffset>>(&[]);
        let edges_vec =
            builder.create_vector::<WIPOffset<flatbuffers::TableFinishedWIPOffset>>(&[]);
        let warm_start_counts_vec = builder.create_vector::<u32>(&[]);
        let per_stage_offsets: Vec<WIPOffset<&str>> = Vec::new();
        let per_stage_vec = builder.create_vector(&per_stage_offsets);

        let root = builder.start_table();
        builder.push_slot_always::<u32>(MANIFEST_FIELD_FORMAT_VERSION, FORMAT_VERSION);
        builder.push_slot_always(MANIFEST_FIELD_SOFTWARE_VERSION, software_version);
        builder.push_slot_always(MANIFEST_FIELD_CREATED_AT, created_at);
        builder.push_slot_always::<u32>(MANIFEST_FIELD_NUM_STAGES, 1);
        builder.push_slot_always::<u32>(MANIFEST_FIELD_N_POOLS, 0);
        builder.push_slot_always(MANIFEST_FIELD_NODES, nodes_vec);
        builder.push_slot_always(MANIFEST_FIELD_EDGES, edges_vec);
        builder.push_slot_always::<u32>(MANIFEST_FIELD_COMPLETED_ITERATIONS, 3);
        builder.push_slot_always::<f64>(MANIFEST_FIELD_FINAL_LOWER_BOUND, 42.0);
        builder.push_slot_always::<u32>(MANIFEST_FIELD_MAX_ITERATIONS, 10);
        builder.push_slot_always::<u32>(MANIFEST_FIELD_FORWARD_PASSES, 1);
        builder.push_slot_always::<u32>(MANIFEST_FIELD_WARM_START_CUTS, 0);
        builder.push_slot_always(MANIFEST_FIELD_WARM_START_COUNTS, warm_start_counts_vec);
        builder.push_slot_always::<u64>(MANIFEST_FIELD_RNG_SEED, 0);
        builder.push_slot_always::<u64>(MANIFEST_FIELD_TOTAL_VISITED_STATES, 0);
        builder.push_slot_always(MANIFEST_FIELD_TRAINING_BLOCK_MODE, training_block_mode);
        builder.push_slot_always(MANIFEST_FIELD_TRAINING_BLOCK_MODE_PER_STAGE, per_stage_vec);
        builder.push_slot_always(MANIFEST_FIELD_SEASON_MANIFEST, season_manifest_offset);
        builder.push_slot_always(MANIFEST_FIELD_SOFTWARE, software);
        // MANIFEST_FIELD_LOWER_BOUND_HISTORY (id 21) deliberately omitted.
        let root_offset = builder.end_table(root);
        builder.finish(root_offset, Some(POLICY_FILE_IDENTIFIER));

        let buf = builder.finished_data().to_vec();
        let decoded = deserialize_checkpoint_manifest(&buf)
            .expect("a buffer without the lower-bound history must still decode");

        assert_eq!(decoded.producer.completed_iterations, 3);
        assert!(decoded.producer.lower_bound_history.is_empty());
    }

    /// Two hydros with differing order vectors round-trip field for field, in
    /// the same order.
    #[test]
    fn checkpoint_manifest_season_descriptor_round_trips() {
        let manifest = CheckpointManifest {
            format_version: FORMAT_VERSION,
            software: Some("cobre".to_string()),
            software_version: "0.14.0".to_string(),
            created_at: "2026-09-15T00:00:00Z".to_string(),
            num_stages: 2,
            graph_manifest: GraphManifest::default(),
            producer: minimal_manifest_producer(),
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
        };

        let buf = serialize_checkpoint_manifest(&manifest);
        let decoded = deserialize_checkpoint_manifest(&buf).expect("round-trip must succeed");

        assert_eq!(
            decoded.season_manifest.cycle_code,
            SEASON_CYCLE_CODE_MONTHLY
        );
        assert_eq!(decoded.season_manifest.n_seasons, 12);
        assert_eq!(decoded.season_manifest.hydro_orders.len(), 2);
        assert_eq!(decoded.season_manifest.hydro_orders[0].hydro_id, 3);
        assert_eq!(
            decoded.season_manifest.hydro_orders[0].orders,
            vec![1, 2, 1, 1, 3, 2, 1, 1, 2, 2, 1, 1]
        );
        assert_eq!(decoded.season_manifest.hydro_orders[1].hydro_id, 9);
        assert_eq!(
            decoded.season_manifest.hydro_orders[1].orders,
            vec![4, 4, 3, 3, 2, 2, 1, 1, 2, 2, 3, 3]
        );
    }

    /// A `season_manifest` whose `hydro_orders` is not canonically ascending
    /// by `hydro_id` is rejected before any consumer reads it — this crate's
    /// own writer always emits `BTreeMap` order, so this guards only a
    /// hand-built or corrupt buffer.
    #[test]
    fn checkpoint_manifest_rejects_unsorted_season_hydro_orders() {
        let manifest = CheckpointManifest {
            format_version: FORMAT_VERSION,
            software: Some("cobre".to_string()),
            software_version: "0.14.0".to_string(),
            created_at: "2026-09-16T00:00:00Z".to_string(),
            num_stages: 1,
            graph_manifest: GraphManifest::default(),
            producer: minimal_manifest_producer(),
            season_manifest: SeasonManifest {
                cycle_code: SEASON_CYCLE_CODE_MONTHLY,
                n_seasons: 2,
                hydro_orders: vec![
                    HydroSeasonOrders {
                        hydro_id: 9,
                        orders: vec![1, 1],
                    },
                    HydroSeasonOrders {
                        hydro_id: 3,
                        orders: vec![1, 1],
                    },
                ],
            },
        };
        let buf = serialize_checkpoint_manifest(&manifest);

        let err = deserialize_checkpoint_manifest(&buf)
            .expect_err("hydro_orders not ascending by hydro_id must be rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("hydro_orders") && msg.contains("ascending"),
            "rejection must name the ordering violation: {err}"
        );
    }

    /// A `season_manifest` hydro whose `orders` length disagrees with
    /// `n_seasons` is rejected before any consumer positionally compares it
    /// against a study's own season-ordinal vector.
    #[test]
    fn checkpoint_manifest_rejects_season_orders_length_mismatch() {
        let manifest = CheckpointManifest {
            format_version: FORMAT_VERSION,
            software: Some("cobre".to_string()),
            software_version: "0.14.0".to_string(),
            created_at: "2026-09-16T00:00:00Z".to_string(),
            num_stages: 1,
            graph_manifest: GraphManifest::default(),
            producer: minimal_manifest_producer(),
            season_manifest: SeasonManifest {
                cycle_code: SEASON_CYCLE_CODE_MONTHLY,
                n_seasons: 4,
                hydro_orders: vec![HydroSeasonOrders {
                    hydro_id: 3,
                    orders: vec![1, 1, 1],
                }],
            },
        };
        let buf = serialize_checkpoint_manifest(&manifest);

        let err = deserialize_checkpoint_manifest(&buf)
            .expect_err("orders.len() != n_seasons must be rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("hydro_id 3") && msg.contains("n_seasons=4"),
            "rejection must name the hydro and the expected season count: {err}"
        );
    }

    /// A buffer built with only ids 0–18 present (the id-19 slot never
    /// pushed) decodes `season_manifest` as the absent descriptor: cycle code
    /// [`SEASON_CYCLE_CODE_ABSENT`], zero seasons, no hydros — never a
    /// spurious `0`-cycle "monthly" default and never an error.
    #[test]
    fn checkpoint_manifest_absent_season_descriptor_reads_as_absent_cycle_code() {
        let mut builder = FlatBufferBuilder::with_capacity(128);

        let software_version = builder.create_string("0.13.0");
        let created_at = builder.create_string("2026-01-01T00:00:00Z");
        let training_block_mode = builder.create_string("parallel");
        let nodes_vec =
            builder.create_vector::<WIPOffset<flatbuffers::TableFinishedWIPOffset>>(&[]);
        let edges_vec =
            builder.create_vector::<WIPOffset<flatbuffers::TableFinishedWIPOffset>>(&[]);
        let warm_start_counts_vec = builder.create_vector::<u32>(&[]);
        let per_stage_offsets: Vec<WIPOffset<&str>> = Vec::new();
        let per_stage_vec = builder.create_vector(&per_stage_offsets);

        let root = builder.start_table();
        builder.push_slot_always::<u32>(MANIFEST_FIELD_FORMAT_VERSION, FORMAT_VERSION);
        builder.push_slot_always(MANIFEST_FIELD_SOFTWARE_VERSION, software_version);
        builder.push_slot_always(MANIFEST_FIELD_CREATED_AT, created_at);
        builder.push_slot_always::<u32>(MANIFEST_FIELD_NUM_STAGES, 1);
        builder.push_slot_always::<u32>(MANIFEST_FIELD_N_POOLS, 0);
        builder.push_slot_always(MANIFEST_FIELD_NODES, nodes_vec);
        builder.push_slot_always(MANIFEST_FIELD_EDGES, edges_vec);
        builder.push_slot_always::<u32>(MANIFEST_FIELD_COMPLETED_ITERATIONS, 0);
        builder.push_slot_always::<f64>(MANIFEST_FIELD_FINAL_LOWER_BOUND, 0.0);
        builder.push_slot_always::<u32>(MANIFEST_FIELD_MAX_ITERATIONS, 0);
        builder.push_slot_always::<u32>(MANIFEST_FIELD_FORWARD_PASSES, 0);
        builder.push_slot_always::<u32>(MANIFEST_FIELD_WARM_START_CUTS, 0);
        builder.push_slot_always(MANIFEST_FIELD_WARM_START_COUNTS, warm_start_counts_vec);
        builder.push_slot_always::<u64>(MANIFEST_FIELD_RNG_SEED, 0);
        builder.push_slot_always::<u64>(MANIFEST_FIELD_TOTAL_VISITED_STATES, 0);
        builder.push_slot_always(MANIFEST_FIELD_TRAINING_BLOCK_MODE, training_block_mode);
        builder.push_slot_always(MANIFEST_FIELD_TRAINING_BLOCK_MODE_PER_STAGE, per_stage_vec);
        // MANIFEST_FIELD_SEASON_MANIFEST (id 19) deliberately omitted.
        let root_offset = builder.end_table(root);
        builder.finish(root_offset, Some(POLICY_FILE_IDENTIFIER));

        let buf = builder.finished_data().to_vec();
        let decoded = deserialize_checkpoint_manifest(&buf)
            .expect("a buffer with only ids 0-18 present must still decode");

        assert_eq!(decoded.season_manifest.cycle_code, SEASON_CYCLE_CODE_ABSENT);
        assert_eq!(decoded.season_manifest.n_seasons, 0);
        assert!(decoded.season_manifest.hydro_orders.is_empty());
    }
}
