//! `cobre.write_policy_checkpoint` — writes a policy checkpoint from plain
//! Python dicts/sequences, single-sourcing the `FlatBuffers` byte layout in
//! `cobre_io`.
//!
//! Input dict shapes mirror what [`crate::results::load_policy`] emits, so a
//! loaded checkpoint round-trips: load -> edit -> write. `season_manifest` and
//! `graph_manifest` round-trip; `active_cut_indices` is written but not returned
//! by `load_policy`, so a load → write cycle resets it (cut activity round-trips
//! through each cut's `is_active`).

use std::borrow::Cow;
use std::collections::HashMap;
use std::path::PathBuf;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use cobre_io::{
    CheckpointManifest, ENTITY_SLOT_DATE_SENTINEL, EntitySlot, FORMAT_VERSION, GraphManifest,
    HydroSeasonOrders, ManifestEdge, ManifestNode, PolicyBasisRecord, PolicyCutRecord,
    ProducerBlock, STAGE_CUTS_GRAPH_STAGE_ID_SENTINEL, STAGE_CUTS_NODE_ID_SENTINEL,
    STAGE_CUTS_PRICED_STATE_DATE_SENTINEL, STAGE_STATES_NODE_ID_SENTINEL, SeasonManifest,
    StageCutsPayload, StageStatesPayload, StateFamily,
};
use cobre_sddp::{POLICY_COBRE_VERSION, SddpError, reserve_boundary_inflow_lag_slots};

use crate::errors::{ErrorSource, convert_error};

#[derive(Debug, FromPyObject)]
#[pyo3(from_item_all)]
pub(crate) struct PyEntitySlot {
    entity_type: u8,
    entity_id: i32,
    subindex: u32,
    was_active: bool,
    #[pyo3(default = ENTITY_SLOT_DATE_SENTINEL)]
    reference_date: i32,
    #[pyo3(default = ENTITY_SLOT_DATE_SENTINEL)]
    interval_start: i32,
    #[pyo3(default = ENTITY_SLOT_DATE_SENTINEL)]
    interval_end: i32,
}

impl From<&PyEntitySlot> for EntitySlot {
    fn from(slot: &PyEntitySlot) -> Self {
        match StateFamily::from_code(slot.entity_type) {
            Some(StateFamily::HydroStorage) => Self::storage(slot.entity_id, slot.was_active),
            Some(StateFamily::HydroInflowLag) => {
                Self::inflow_lag(slot.entity_id, slot.subindex, slot.was_active)
            }
            Some(StateFamily::HydroTransitBucket) => {
                Self::transit_bucket(slot.entity_id, slot.subindex, slot.was_active)
            }
            Some(StateFamily::AnticipatedThermalState) => {
                Self::anticipated(slot.entity_id, slot.subindex, slot.was_active)
            }
            None => Self {
                entity_type: slot.entity_type,
                entity_id: slot.entity_id,
                subindex: slot.subindex,
                was_active: slot.was_active,
                reference_date: ENTITY_SLOT_DATE_SENTINEL,
                interval_start: ENTITY_SLOT_DATE_SENTINEL,
                interval_end: ENTITY_SLOT_DATE_SENTINEL,
            },
        }
        .with_reference_date(slot.reference_date)
        .with_interval(slot.interval_start, slot.interval_end)
    }
}

#[derive(Debug, FromPyObject)]
#[pyo3(from_item_all)]
pub(crate) struct PyCutRecord {
    cut_id: u64,
    slot_index: u32,
    iteration: u32,
    forward_pass_index: u32,
    intercept: f64,
    coefficients: Vec<f64>,
    is_active: bool,
    /// Inflow-lag gradient terms keyed by hydro id (`hydro_id -> [coef_by_depth]`,
    /// index `0` = lag depth 1). Requires the top-level `inflow_lag_depth`; empty
    /// (the default) leaves the checkpoint byte-identical.
    #[pyo3(default)]
    inflow_lag_coefficients: HashMap<i32, Vec<f64>>,
}

#[derive(Debug, FromPyObject)]
#[pyo3(from_item_all)]
pub(crate) struct PyStageCutsPayload {
    stage_id: u32,
    state_dimension: u32,
    capacity: u32,
    #[pyo3(default)]
    warm_start_count: u32,
    cuts: Vec<PyCutRecord>,
    #[pyo3(default)]
    active_cut_indices: Vec<u32>,
    #[pyo3(default)]
    populated_count: Option<u32>,
    #[pyo3(default)]
    entity_manifest: Vec<PyEntitySlot>,
    /// Defaults to the metadata producer block's factor when omitted (see
    /// [`write_policy_checkpoint`]).
    #[pyo3(default)]
    cost_scale_factor: Option<f64>,
    #[pyo3(default = STAGE_CUTS_NODE_ID_SENTINEL)]
    node_id: i32,
    #[pyo3(default = STAGE_CUTS_GRAPH_STAGE_ID_SENTINEL)]
    graph_stage_id: i32,
    #[pyo3(default = STAGE_CUTS_PRICED_STATE_DATE_SENTINEL)]
    priced_state_date: i32,
}

#[derive(Debug, FromPyObject)]
#[pyo3(from_item_all)]
pub(crate) struct PyBasisRecord {
    stage_id: u32,
    iteration: u32,
    column_status: Vec<u8>,
    row_status: Vec<u8>,
    num_cut_rows: u32,
}

#[derive(Debug, FromPyObject)]
#[pyo3(from_item_all)]
pub(crate) struct PyStageStatesPayload {
    stage_id: u32,
    #[pyo3(default = STAGE_STATES_NODE_ID_SENTINEL)]
    node_id: i32,
    state_dimension: u32,
    count: u32,
    data: Vec<f64>,
    #[pyo3(default)]
    entity_manifest: Vec<PyEntitySlot>,
}

#[derive(Debug, FromPyObject)]
#[pyo3(from_item_all)]
pub(crate) struct PyManifestNode {
    id: i32,
    stage_id: i32,
    pool_id: u32,
}

#[derive(Debug, FromPyObject)]
#[pyo3(from_item_all)]
pub(crate) struct PyManifestEdge {
    source_id: i32,
    target_id: i32,
    probability: f64,
}

#[derive(Debug, FromPyObject)]
#[pyo3(from_item_all)]
pub(crate) struct PyGraphManifest {
    n_pools: u32,
    nodes: Vec<PyManifestNode>,
    edges: Vec<PyManifestEdge>,
}

impl From<PyGraphManifest> for GraphManifest {
    fn from(g: PyGraphManifest) -> Self {
        Self {
            n_pools: g.n_pools,
            nodes: g
                .nodes
                .into_iter()
                .map(|n| ManifestNode {
                    id: n.id,
                    stage_id: n.stage_id,
                    pool_id: n.pool_id,
                })
                .collect(),
            edges: g
                .edges
                .into_iter()
                .map(|e| ManifestEdge {
                    source_id: e.source_id,
                    target_id: e.target_id,
                    probability: e.probability,
                })
                .collect(),
        }
    }
}

#[derive(Debug, FromPyObject)]
#[pyo3(from_item_all)]
pub(crate) struct PyHydroSeasonOrders {
    hydro_id: i32,
    orders: Vec<u32>,
}

#[derive(Debug, FromPyObject)]
#[pyo3(from_item_all)]
pub(crate) struct PySeasonManifest {
    cycle_code: u8,
    n_seasons: u32,
    hydro_orders: Vec<PyHydroSeasonOrders>,
}

impl From<PySeasonManifest> for SeasonManifest {
    fn from(s: PySeasonManifest) -> Self {
        Self {
            cycle_code: s.cycle_code,
            n_seasons: s.n_seasons,
            hydro_orders: s
                .hydro_orders
                .into_iter()
                .map(|h| HydroSeasonOrders {
                    hydro_id: h.hydro_id,
                    orders: h.orders,
                })
                .collect(),
        }
    }
}

/// The producer-namespaced metadata block, mirroring what
/// [`crate::results::load_policy`] emits under `metadata["producer"]`.
#[derive(Debug, FromPyObject)]
#[pyo3(from_item_all)]
pub(crate) struct PyProducerBlock {
    completed_iterations: u32,
    final_lower_bound: f64,
    #[pyo3(default)]
    best_upper_bound: Option<f64>,
    max_iterations: u32,
    forward_passes: u32,
    warm_start_cuts: u32,
    #[pyo3(default)]
    warm_start_counts: Vec<u32>,
    rng_seed: u64,
    #[pyo3(default)]
    total_visited_states: u64,
    #[pyo3(default)]
    training_block_mode: String,
    #[pyo3(default)]
    training_block_mode_per_stage: Vec<String>,
    #[pyo3(default)]
    cost_scale_factor: Option<f64>,
}

impl From<PyProducerBlock> for ProducerBlock {
    fn from(p: PyProducerBlock) -> Self {
        Self {
            completed_iterations: p.completed_iterations,
            final_lower_bound: p.final_lower_bound,
            best_upper_bound: p.best_upper_bound,
            max_iterations: p.max_iterations,
            forward_passes: p.forward_passes,
            warm_start_cuts: p.warm_start_cuts,
            warm_start_counts: p.warm_start_counts,
            rng_seed: p.rng_seed,
            total_visited_states: p.total_visited_states,
            training_block_mode: p.training_block_mode,
            training_block_mode_per_stage: p.training_block_mode_per_stage,
            cost_scale_factor: p.cost_scale_factor,
        }
    }
}

#[derive(Debug, FromPyObject)]
#[pyo3(from_item_all)]
pub(crate) struct PyPolicyCheckpointMetadata {
    #[pyo3(default = FORMAT_VERSION)]
    format_version: u32,
    created_at: String,
    num_stages: u32,
    #[pyo3(default)]
    graph_manifest: Option<PyGraphManifest>,
    #[pyo3(default)]
    season_manifest: Option<PySeasonManifest>,
    producer: PyProducerBlock,
}

impl From<PyPolicyCheckpointMetadata> for CheckpointManifest {
    fn from(m: PyPolicyCheckpointMetadata) -> Self {
        Self {
            format_version: m.format_version,
            cobre_version: POLICY_COBRE_VERSION.to_string(),
            created_at: m.created_at,
            num_stages: m.num_stages,
            graph_manifest: m
                .graph_manifest
                .map(GraphManifest::from)
                .unwrap_or_default(),
            season_manifest: m
                .season_manifest
                .map(SeasonManifest::from)
                .unwrap_or_default(),
            producer: m.producer.into(),
        }
    }
}

fn checked_u32_len(len: usize, what: &str) -> PyResult<u32> {
    u32::try_from(len)
        .map_err(|_| PyValueError::new_err(format!("{what} has {len} entries, exceeding u32::MAX")))
}

/// Validates `coefficients.len() == state_dimension` per cut; resolves
/// `populated_count` default to `cuts.len()`.
fn resolve_populated_counts(stage_cuts: &[PyStageCutsPayload]) -> PyResult<Vec<u32>> {
    let mut resolved = Vec::with_capacity(stage_cuts.len());
    for sc in stage_cuts {
        for cut in &sc.cuts {
            if cut.coefficients.len() != sc.state_dimension as usize {
                return Err(PyValueError::new_err(format!(
                    "stage {} cut {}: coefficients has {} entries, expected \
                     state_dimension={}",
                    sc.stage_id,
                    cut.cut_id,
                    cut.coefficients.len(),
                    sc.state_dimension
                )));
            }
        }
        resolved.push(match sc.populated_count {
            Some(p) => p,
            None => checked_u32_len(sc.cuts.len(), "cuts")?,
        });
    }
    Ok(resolved)
}

/// Validates `data.len() == count * state_dimension` per stage.
fn validate_stage_states(stage_states: &[PyStageStatesPayload]) -> PyResult<()> {
    for ss in stage_states {
        let expected = ss.count as usize * ss.state_dimension as usize;
        if ss.data.len() != expected {
            return Err(PyValueError::new_err(format!(
                "stage {}: states data has {} entries, expected count*state_dimension={}",
                ss.stage_id,
                ss.data.len(),
                expected
            )));
        }
    }
    Ok(())
}

/// Reject `inflow_lag_coefficients` supplied without a positive
/// `inflow_lag_depth` — they would be silently dropped.
fn reject_unreserved_lag_coefficients(
    stage_cuts: &[PyStageCutsPayload],
    reserve_depth: Option<u32>,
) -> PyResult<()> {
    if reserve_depth.is_some() {
        return Ok(());
    }
    for sc in stage_cuts {
        for cut in &sc.cuts {
            if !cut.inflow_lag_coefficients.is_empty() {
                return Err(PyValueError::new_err(format!(
                    "stage {} cut {}: inflow_lag_coefficients supplied without \
                     inflow_lag_depth; pass inflow_lag_depth=N to reserve the lag slots",
                    sc.stage_id, cut.cut_id
                )));
            }
        }
    }
    Ok(())
}

/// Owns the manifest and coefficients so borrowed [`StageCutsPayload`]/[`PolicyCutRecord`]
/// views can reference them (reservation widens coefficients, requiring owned storage;
/// no-reservation borrows `sc.cuts[*].coefficients` directly).
struct StageCutsData<'a> {
    manifest: Vec<EntitySlot>,
    coefficients: Vec<Cow<'a, [f64]>>,
    state_dimension: u32,
}

/// Convert one stage's manifest and coefficients to owned form, reserving the
/// canonical `HydroInflowLag` slots when `reserve_depth` is set.
fn build_stage_cuts_data(
    sc: &PyStageCutsPayload,
    reserve_depth: Option<u32>,
) -> PyResult<StageCutsData<'_>> {
    let manifest: Vec<EntitySlot> = sc.entity_manifest.iter().map(EntitySlot::from).collect();
    match reserve_depth {
        Some(depth) => {
            let coefficients: Vec<Vec<f64>> =
                sc.cuts.iter().map(|c| c.coefficients.clone()).collect();
            let cut_lag: Vec<HashMap<i32, Vec<f64>>> = sc
                .cuts
                .iter()
                .map(|c| c.inflow_lag_coefficients.clone())
                .collect();
            let reserved =
                reserve_boundary_inflow_lag_slots(&manifest, &coefficients, &cut_lag, depth)
                    .map_err(|e| match e {
                        SddpError::Validation(msg) => PyValueError::new_err(msg),
                        other => PyValueError::new_err(other.to_string()),
                    })?;
            Ok(StageCutsData {
                manifest: reserved.manifest,
                coefficients: reserved.coefficients.into_iter().map(Cow::Owned).collect(),
                state_dimension: reserved.state_dimension,
            })
        }
        None => Ok(StageCutsData {
            manifest,
            coefficients: sc
                .cuts
                .iter()
                .map(|c| Cow::Borrowed(c.coefficients.as_slice()))
                .collect(),
            state_dimension: sc.state_dimension,
        }),
    }
}

/// Write a policy checkpoint from plain Python dicts/sequences to `path`.
///
/// `stage_cuts` and `metadata` mirror the `"stage_cuts"`/`"metadata"` keys
/// [`crate::results::load_policy`] returns; `stage_bases`/`stage_states`
/// default to empty when omitted (a checkpoint authored from raw cut data
/// carries neither).
///
/// `inflow_lag_depth`, when set to `N > 0`, has cobre reserve `N` canonical
/// `HydroInflowLag` state slots per storage hydro in every stage's manifest
/// (via [`reserve_boundary_inflow_lag_slots`]), placing each cut's
/// `inflow_lag_coefficients` at their `(hydro, depth)` positions. This is the
/// authoring path for a boundary policy of a case with no PAR model to infer
/// the depth from (a DECOMP-bridge bootstrap). Absent or `0`, the checkpoint is
/// byte-identical to one written without the argument.
///
/// The checkpoint always records the running cobre version
/// ([`cobre_sddp::POLICY_COBRE_VERSION`]); a `cobre_version` in `metadata` is
/// ignored.
///
/// # Errors
///
/// `ValueError` when a cut's `coefficients` length does not match its stage's
/// `state_dimension`, a stage's state data length does not match
/// `count * state_dimension`, a cut carries `inflow_lag_coefficients` without a
/// positive `inflow_lag_depth`, or (under `inflow_lag_depth`) a manifest lacks a
/// leading storage block or an inflow-lag coefficient is unplaceable. A
/// `season_manifest` whose `hydro_orders` are not ascending by `hydro_id` or
/// whose `orders` lengths disagree with `n_seasons` is written as given and
/// rejected by [`crate::results::load_policy`] with `OutputError`. Otherwise
/// the `cobre.errors` leaf mapped from the underlying [`cobre_io::OutputError`].
#[pyfunction]
#[pyo3(signature = (path, stage_cuts, metadata, stage_bases=None, stage_states=None, inflow_lag_depth=None))]
#[allow(clippy::needless_pass_by_value)]
pub fn write_policy_checkpoint(
    py: Python<'_>,
    path: PathBuf,
    stage_cuts: Vec<PyStageCutsPayload>,
    metadata: PyPolicyCheckpointMetadata,
    stage_bases: Option<Vec<PyBasisRecord>>,
    stage_states: Option<Vec<PyStageStatesPayload>>,
    inflow_lag_depth: Option<u32>,
) -> PyResult<()> {
    let stage_bases = stage_bases.unwrap_or_default();
    let stage_states = stage_states.unwrap_or_default();

    let populated_counts = resolve_populated_counts(&stage_cuts)?;
    validate_stage_states(&stage_states)?;

    let reserve_depth = inflow_lag_depth.filter(|&n| n > 0);
    reject_unreserved_lag_coefficients(&stage_cuts, reserve_depth)?;
    let stage_data: Vec<StageCutsData<'_>> = stage_cuts
        .iter()
        .map(|sc| build_stage_cuts_data(sc, reserve_depth))
        .collect::<PyResult<_>>()?;

    let metadata: CheckpointManifest = metadata.into();

    py.detach(|| {
        let cut_records: Vec<Vec<PolicyCutRecord<'_>>> = stage_cuts
            .iter()
            .zip(&stage_data)
            .map(|(sc, data)| {
                sc.cuts
                    .iter()
                    .zip(&data.coefficients)
                    .map(|(c, coefficients)| PolicyCutRecord {
                        cut_id: c.cut_id,
                        slot_index: c.slot_index,
                        iteration: c.iteration,
                        forward_pass_index: c.forward_pass_index,
                        intercept: c.intercept,
                        coefficients,
                        is_active: c.is_active,
                    })
                    .collect()
            })
            .collect();

        let stage_cuts_payloads: Vec<StageCutsPayload<'_>> = stage_cuts
            .iter()
            .zip(&stage_data)
            .enumerate()
            .map(|(i, (sc, data))| StageCutsPayload {
                stage_id: sc.stage_id,
                state_dimension: data.state_dimension,
                capacity: sc.capacity,
                warm_start_count: sc.warm_start_count,
                cuts: &cut_records[i],
                active_cut_indices: &sc.active_cut_indices,
                populated_count: populated_counts[i],
                entity_manifest: &data.manifest,
                cost_scale_factor: sc
                    .cost_scale_factor
                    .or(metadata.producer.cost_scale_factor)
                    .unwrap_or(1_000_000.0),
                node_id: sc.node_id,
                graph_stage_id: sc.graph_stage_id,
                priced_state_date: sc.priced_state_date,
            })
            .collect();

        let basis_records: Vec<PolicyBasisRecord<'_>> = stage_bases
            .iter()
            .map(|b| PolicyBasisRecord {
                stage_id: b.stage_id,
                iteration: b.iteration,
                column_status: &b.column_status,
                row_status: &b.row_status,
                num_cut_rows: b.num_cut_rows,
            })
            .collect();

        let state_manifests: Vec<Vec<EntitySlot>> = stage_states
            .iter()
            .map(|ss| ss.entity_manifest.iter().map(EntitySlot::from).collect())
            .collect();

        let state_payloads: Vec<StageStatesPayload<'_>> = stage_states
            .iter()
            .enumerate()
            .map(|(i, ss)| StageStatesPayload {
                stage_id: ss.stage_id,
                node_id: ss.node_id,
                state_dimension: ss.state_dimension,
                count: ss.count,
                data: &ss.data,
                entity_manifest: &state_manifests[i],
            })
            .collect();

        cobre_io::write_policy_checkpoint(
            &path,
            &stage_cuts_payloads,
            &basis_records,
            &metadata,
            &state_payloads,
        )
    })
    .map_err(|e| convert_error(ErrorSource::Output(&e)))
}

#[cfg(test)]
mod tests {
    use super::{EntitySlot, PyEntitySlot};

    #[test]
    fn py_entity_slot_unclassified_entity_type_round_trips() {
        let slot = PyEntitySlot {
            entity_type: 200,
            entity_id: 7,
            subindex: 3,
            was_active: true,
            reference_date: 20_260_102,
            interval_start: 20_260_103,
            interval_end: 20_260_104,
        };

        let converted = EntitySlot::from(&slot);

        assert_eq!(converted.entity_type, 200);
        assert_eq!(converted.entity_id, 7);
        assert_eq!(converted.subindex, 3);
        assert!(converted.was_active);
        assert_eq!(converted.reference_date, 20_260_102);
        assert_eq!(converted.interval_start, 20_260_103);
        assert_eq!(converted.interval_end, 20_260_104);
    }
}
