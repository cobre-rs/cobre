//! Policy checkpoint export helpers.
//!
//! Shared conversion logic for extracting active cuts and basis data from a
//! trained [`FutureCostFunction`] and [`TrainingResult`] into the `cobre-io`
//! policy types needed by [`cobre_io::write_policy_checkpoint`].

// Rationale: harvested counts/indices are small non-negative values bounded far below FlatBuffers field widths; narrowing casts are pervasive.
#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]

use std::collections::{HashMap, HashSet};

use crate::visited_states::VisitedStatesArchive;
use chrono::NaiveDate;
use cobre_core::Stage;
use cobre_core::System;
use cobre_core::Thermal;
use cobre_core::commissioning::{commissioning_active, hydro_operating_active};
use cobre_io::output::policy::{
    ENTITY_SLOT_DATE_SENTINEL, EntitySlot, GraphManifest, ManifestEdge, ManifestNode,
    OwnedPolicyCutRecord, PolicyBasisRecord, PolicyCutRecord, STAGE_CUTS_GRAPH_STAGE_ID_SENTINEL,
    STAGE_CUTS_NODE_ID_SENTINEL, STAGE_CUTS_PRICED_STATE_DATE_SENTINEL, StageCutsPayload,
    StageStatesPayload, StateFamily, encode_slot_date,
};

use crate::SddpError;
use crate::cut::FutureCostFunction;
use crate::lp::builder::delivery_ring::DeliveryRing;
use crate::lp::indexer::{
    AnticipatedPlants, CutSlot, CutStateProjection, StateRegion, StateSpace,
    for_each_live_commitment_slot,
};
use crate::setup::{NodeGraph, NodePos, extended_delivery_stages};
use crate::time_value::post_study_delivery_calendar;
use crate::training::TrainingResult;

/// The sentinel-or-dated `(interval_start, interval_end)` pair for a resolved
/// delivery `stage`: sentinel when `None`; otherwise the day-accurate
/// `YYYYMMDD` interval endpoints ([`encode_slot_date`]) of `start_date`/`end_date`.
fn slot_interval(stage: Option<&Stage>) -> (i32, i32) {
    stage.map_or(
        (ENTITY_SLOT_DATE_SENTINEL, ENTITY_SLOT_DATE_SENTINEL),
        |stage| {
            (
                encode_slot_date(stage.start_date),
                encode_slot_date(stage.end_date),
            )
        },
    )
}

/// The inflow-lag slot's reference stage: pool `p` prices the state leaving
/// stage `p` (`training::backward`'s stage convention, the same outgoing state
/// `cut::row` prices into each cut row), so 1-based lag `1` (`lag == 0` here) is
/// `p`'s own stage and each deeper lag steps one further stage back, walked over
/// the FULL `system.stages()` ordering (pre-study stages included); the sentinel
/// when `pool_pos_in_all` is `None` or the walk reaches before the earliest
/// declared stage. The returned anchor is [`encode_slot_date`] of the referenced
/// stage's own `start_date` — its full date, not the enclosing calendar month.
fn lag_reference_anchor(all_stages: &[Stage], pool_pos_in_all: Option<usize>, lag: usize) -> i32 {
    pool_pos_in_all
        .and_then(|t| t.checked_sub(lag))
        .and_then(|idx| all_stages.get(idx))
        .map_or(ENTITY_SLOT_DATE_SENTINEL, |s| {
            encode_slot_date(s.start_date)
        })
}

/// Build the per-slot entity-identity manifest for one stage's cut pool: one
/// [`EntitySlot`] per enabled cut-state dimension of `projection`.
///
/// Slots are emitted in `projection`'s storage → lag → buckets →
/// commitment-hold order, so slot `j` describes the entity owning positional
/// coefficient `j` — the order a consumer matches the manifest against the cut
/// coefficients. Each slot is classified by the global [`StateSpace`] region
/// containing its incoming-state column ([`CutStateProjection::incoming_column`]),
/// never by re-deriving column arithmetic.
///
/// Each hydro-region slot's `was_active` comes from [`hydro_operating_active`].
/// The commitment-hold region is the in-study anticipated ring alone: every
/// post-study-targeted delivery is carried by a ring slot, never a separate
/// post-horizon lane.
///
/// A ring slot's `interval_start`/`interval_end` are the day-accurate
/// `YYYYMMDD` anchors of the physical delivery it holds at the pool's own
/// stage — live exactly when [`for_each_live_commitment_slot`] visits it
/// (the LP's own latch set: carry plus deposit), over `study_stages` extended
/// by [`post_study_delivery_calendar`]. A slot the sweep does not latch this
/// stage, or whose target lands past that calendar, carries the sentinel for
/// both fields. With no post-study stages the walk is byte-identical to a
/// study-only one.
///
/// An inflow-lag slot's `reference_date` is the referenced past stage's own
/// full `start_date`, [`encode_slot_date`]-encoded, via
/// [`lag_reference_anchor`]: 1-based lag `1` is the pool's own stage and each
/// deeper lag steps one further stage back over the full `system.stages()`
/// ordering (pre-study stages included), or the sentinel when the pool's own
/// stage or the referenced stage is unresolvable.
///
/// # Panics (debug builds only)
///
/// Panics if the built manifest length differs from `projection.n_slots()`.
#[must_use]
pub fn build_stage_entity_manifest(
    system: &System,
    global_layout: &StateSpace,
    anticipated_plants: &AnticipatedPlants,
    projection: &CutStateProjection,
    stage_id: i32,
) -> Vec<EntitySlot> {
    let n = global_layout.hydro_count;
    let hydros = system.hydros();
    let thermals = system.thermals();
    let anticipated_thermals: Vec<&Thermal> = anticipated_plants
        .thermals()
        .map(|t| &thermals[t.get()])
        .collect();
    // Only `slot_lane_at`'s reverse decomposition is read here — the manifest
    // never emits ring rows/columns.
    let anticipated_ring = DeliveryRing::anticipated(global_layout);

    // Study stages in canonical index order — the space `AnticipatedResolution`'s
    // decider/depth and the bucket topology both index.
    let study_stages: Vec<&Stage> = system.stages().iter().filter(|s| s.id >= 0).collect();
    let current_stage_idx = study_stages.iter().position(|s| s.id == stage_id);
    // Full stage ordering (pre-study stages included) — the basis
    // `lag_reference_anchor` walks, since a deep lag can reach into the
    // pre-study window `study_stages` excludes.
    let pool_pos_in_all = system.stages().iter().position(|s| s.id == stage_id);
    let post_study_calendar = post_study_delivery_calendar(system);
    let delivery_stages = extended_delivery_stages(&study_stages, &post_study_calendar);
    // `b_d^out(t)` arrives at stage `t + d`, never `t + 1 + d`.
    let bucket_arrival_stage =
        |lag: usize| current_stage_idx.and_then(|t| delivery_stages.get(t + lag).copied());

    let n_anticipated = global_layout.n_anticipated;
    let mut live_target = vec![None; n_anticipated * global_layout.k_max];
    if let Some(t) = current_stage_idx {
        for_each_live_commitment_slot(global_layout, t, |res, _| {
            live_target[res.slot * n_anticipated + res.plant] = Some(res.target);
        });
    }

    let anticipated_slot = |offset: usize| -> EntitySlot {
        let (slot_idx, plant_pos) = anticipated_ring.slot_lane_at(offset);
        let plant = anticipated_thermals[plant_pos];
        let resolved_stage = live_target[slot_idx * n_anticipated + plant_pos]
            .and_then(|m| delivery_stages.get(m).copied());
        let (interval_start, interval_end) = slot_interval(resolved_stage);
        EntitySlot::anticipated(
            plant.id.0,
            slot_idx as u32,
            commissioning_active(plant.entry_stage_id, plant.exit_stage_id, stage_id),
        )
        .with_interval(interval_start, interval_end)
    };

    let mut manifest = Vec::with_capacity(projection.n_slots());
    for j in 0..projection.n_slots() {
        let (region, offset) =
            global_layout.classify_incoming_column(projection.incoming_column(CutSlot::new(j)));
        let slot = match region {
            StateRegion::Storage => {
                let hydro = &hydros[offset];
                EntitySlot::storage(
                    hydro.id.0,
                    hydro_operating_active(
                        hydro.filling.as_ref(),
                        hydro.entry_stage_id,
                        hydro.exit_stage_id,
                        stage_id,
                    ),
                )
            }
            StateRegion::Lag => {
                let lag = offset / n;
                let h = offset % n;
                let hydro = &hydros[h];
                EntitySlot::inflow_lag(
                    hydro.id.0,
                    (lag + 1) as u32,
                    hydro_operating_active(
                        hydro.filling.as_ref(),
                        hydro.entry_stage_id,
                        hydro.exit_stage_id,
                        stage_id,
                    ),
                )
                .with_reference_date(lag_reference_anchor(
                    system.stages(),
                    pool_pos_in_all,
                    lag,
                ))
            }
            StateRegion::Buckets => {
                let (plant_idx, lag) = global_layout.transit_bucket_column_order[offset];
                let hydro = &hydros[plant_idx.get()];
                let (interval_start, interval_end) = slot_interval(bucket_arrival_stage(lag));
                EntitySlot::transit_bucket(
                    hydro.id.0,
                    lag as u32,
                    hydro_operating_active(
                        hydro.filling.as_ref(),
                        hydro.entry_stage_id,
                        hydro.exit_stage_id,
                        stage_id,
                    ),
                )
                .with_interval(interval_start, interval_end)
            }
            StateRegion::CommitmentHold => anticipated_slot(offset),
        };
        manifest.push(slot);
    }

    debug_assert_eq!(
        manifest.len(),
        projection.n_slots(),
        "manifest length must equal projection.n_slots()"
    );
    manifest
}

/// A boundary checkpoint manifest widened with canonical `HydroInflowLag`
/// slots, each cut's coefficient vector extended to match. The three fields are
/// mutually consistent: `state_dimension == manifest.len()`, and every
/// `coefficients` row has that same length.
#[derive(Debug)]
pub struct ReservedInflowLagLayout {
    /// Widened per-slot manifest: the original leading storage block, then the
    /// reserved lag block, then the original tail (buckets / commitment-hold).
    pub manifest: Vec<EntitySlot>,
    /// One extended coefficient vector per input cut, positionally aligned to
    /// [`Self::manifest`].
    pub coefficients: Vec<Vec<f64>>,
    /// `manifest.len()` — the widened state dimension the writer stamps on the
    /// pool.
    pub state_dimension: u32,
}

/// Reserve the canonical `HydroInflowLag` state slots in a boundary checkpoint
/// authored outside cobre (the DECOMP-bridge bootstrap), so cobre — never the
/// caller — owns the lag-block layout.
///
/// The caller supplies a `manifest` whose leading contiguous block is one
/// `HydroStorage` slot per hydro (the shape [`build_stage_entity_manifest`]
/// emits), each cut's storage-aligned `cut_coefficients`, and — separately —
/// each cut's inflow-lag coefficients keyed by hydro id
/// (`hydro_id -> [coef_by_depth]`, index `0` = lag depth 1). For
/// `inflow_lag_depth == N`, this inserts `N` `HydroInflowLag` slots per storage
/// hydro directly after the storage block, in the SAME lag-major
/// (`for lag { for hydro }`) order and 1-based `subindex` convention
/// [`build_stage_entity_manifest`]'s [`StateRegion::Lag`] arm produces, and
/// extends every cut's coefficient vector at the same positions — each keyed
/// value at its `(hydro_id, depth)` slot, `0.0` elsewhere. The widened manifest
/// then self-describes depth `N`, so
/// [`boundary_policy_required_lag_depth`](crate::boundary_policy_required_lag_depth)
/// reads `N` and the load path reserves the matching forward lag state. Each
/// lag slot inherits its owning storage slot's `was_active`, matching the
/// per-hydro `hydro_operating_active` flag stamped on both families.
///
/// # Errors
///
/// Returns [`SddpError::Validation`] if `inflow_lag_depth == 0`, the two
/// per-cut slices differ in length, the manifest's `HydroStorage` slots are not
/// a non-empty leading contiguous block, a cut's coefficient length does not
/// equal the manifest length, or a keyed lag coefficient names a hydro with no
/// storage slot or a depth beyond `N` (an unplaceable term — never silently
/// dropped, so every source `pi_qafl` term is placed or the write fails).
pub fn reserve_boundary_inflow_lag_slots<S: std::hash::BuildHasher>(
    manifest: &[EntitySlot],
    cut_coefficients: &[Vec<f64>],
    cut_inflow_lag_coefficients: &[HashMap<i32, Vec<f64>, S>],
    inflow_lag_depth: u32,
) -> Result<ReservedInflowLagLayout, SddpError> {
    if inflow_lag_depth == 0 {
        return Err(SddpError::Validation(
            "reserve_boundary_inflow_lag_slots called with inflow_lag_depth == 0; reserve lag \
             slots only for a positive declared depth"
                .to_string(),
        ));
    }
    if cut_coefficients.len() != cut_inflow_lag_coefficients.len() {
        return Err(SddpError::Validation(format!(
            "cut coefficient count {} does not match keyed inflow-lag count {}",
            cut_coefficients.len(),
            cut_inflow_lag_coefficients.len()
        )));
    }
    let n = inflow_lag_depth as usize;

    // The lag block spans exactly the storage hydros and inserts after them, so
    // they must be the non-empty leading contiguous block.
    let storage_count = manifest
        .iter()
        .take_while(|s| s.entity_type == StateFamily::HydroStorage.code())
        .count();
    if storage_count == 0 {
        return Err(SddpError::Validation(
            "boundary manifest carries no leading HydroStorage slot; cannot reserve inflow-lag \
             slots without the storage hydros to key them on"
                .to_string(),
        ));
    }
    if manifest[storage_count..]
        .iter()
        .any(|s| s.entity_type == StateFamily::HydroStorage.code())
    {
        return Err(SddpError::Validation(
            "boundary manifest interleaves HydroStorage slots with other entity types; the \
             canonical layout emits storage as a leading contiguous block"
                .to_string(),
        ));
    }
    let storage_slots = &manifest[..storage_count];
    let storage_ids: HashSet<i32> = storage_slots.iter().map(|s| s.entity_id).collect();

    // Reject an unplaceable keyed term BEFORE building, so it fails the write
    // rather than dropping.
    for (c, keyed) in cut_inflow_lag_coefficients.iter().enumerate() {
        for (&hydro_id, coeffs) in keyed {
            if !storage_ids.contains(&hydro_id) {
                return Err(SddpError::Validation(format!(
                    "cut {c} carries an inflow-lag coefficient for hydro {hydro_id}, which has no \
                     HydroStorage slot in the boundary manifest"
                )));
            }
            if coeffs.len() > n {
                return Err(SddpError::Validation(format!(
                    "cut {c} inflow-lag coefficient for hydro {hydro_id} has depth {} exceeding \
                     the reserved inflow_lag_depth {n}",
                    coeffs.len()
                )));
            }
        }
    }

    // Lag-major (`for lag { for hydro }`), matching `build_stage_entity_manifest`'s
    // `StateRegion::Lag` arm — `reserved_cut_coefficients` below must iterate
    // identically, or a coefficient lands on the wrong slot.
    let reserved_slots: Vec<EntitySlot> = (0..n)
        .flat_map(|lag| {
            storage_slots.iter().map(move |storage| {
                EntitySlot::inflow_lag(storage.entity_id, (lag + 1) as u32, storage.was_active)
            })
        })
        .collect();

    // Same lag-major order as `reserved_slots` above, matching
    // `build_stage_entity_manifest`'s `StateRegion::Lag` arm.
    let reserved_cut_coefficients: Vec<Vec<f64>> = cut_inflow_lag_coefficients
        .iter()
        .map(|keyed| {
            (0..n)
                .flat_map(|lag| {
                    storage_slots.iter().map(move |storage| {
                        keyed
                            .get(&storage.entity_id)
                            .and_then(|per_depth| per_depth.get(lag))
                            .copied()
                            .unwrap_or(0.0)
                    })
                })
                .collect()
        })
        .collect();

    let (manifest, coefficients) = splice_reserved_state_block(
        manifest,
        storage_count,
        &reserved_slots,
        cut_coefficients,
        &reserved_cut_coefficients,
    )?;
    let state_dimension = manifest.len() as u32;
    Ok(ReservedInflowLagLayout {
        manifest,
        coefficients,
        state_dimension,
    })
}

/// Splice a reserved state block into a boundary manifest and every cut's
/// coefficient vector at `anchor_count` — after the leading anchor block, before
/// the tail — keeping both positionally aligned. Family-independent: the caller
/// builds `reserved_slots` and the per-cut `reserved_cut_coefficients` for its own
/// family, and this owns only the splice and the coefficient-alignment guard, so
/// a second boundary state family reuses it unchanged.
///
/// # Errors
///
/// Returns [`SddpError::Validation`] if a cut's coefficient length does not equal
/// the input manifest length (the two must be positionally aligned before
/// reserving).
fn splice_reserved_state_block(
    manifest: &[EntitySlot],
    anchor_count: usize,
    reserved_slots: &[EntitySlot],
    cut_coefficients: &[Vec<f64>],
    reserved_cut_coefficients: &[Vec<f64>],
) -> Result<(Vec<EntitySlot>, Vec<Vec<f64>>), SddpError> {
    let mut new_manifest = Vec::with_capacity(manifest.len() + reserved_slots.len());
    new_manifest.extend_from_slice(&manifest[..anchor_count]);
    new_manifest.extend_from_slice(reserved_slots);
    new_manifest.extend_from_slice(&manifest[anchor_count..]);

    let mut new_coefficients = Vec::with_capacity(cut_coefficients.len());
    for (coeffs, reserved) in cut_coefficients.iter().zip(reserved_cut_coefficients) {
        if coeffs.len() != manifest.len() {
            return Err(SddpError::Validation(format!(
                "cut has {} coefficients but the boundary manifest has {} slots; the two must be \
                 positionally aligned before reserving state slots",
                coeffs.len(),
                manifest.len()
            )));
        }
        let mut extended = Vec::with_capacity(coeffs.len() + reserved_slots.len());
        extended.extend_from_slice(&coeffs[..anchor_count]);
        extended.extend_from_slice(reserved);
        extended.extend_from_slice(&coeffs[anchor_count..]);
        new_coefficients.push(extended);
    }
    Ok((new_manifest, new_coefficients))
}

/// Build per-stage vectors of **all** populated [`PolicyCutRecord`]s from the FCF pools.
///
/// Both active and inactive cuts are included so the checkpoint preserves the
/// full training history. Use [`build_active_indices`] for the active subset.
#[must_use]
pub fn build_stage_cut_records(fcf: &FutureCostFunction) -> Vec<Vec<PolicyCutRecord<'_>>> {
    fcf.pools
        .iter()
        .map(|pool| {
            (0..pool.populated())
                .map(|i| {
                    let meta = pool.metadata(i);
                    PolicyCutRecord {
                        cut_id: meta.iteration_generated * u64::from(pool.visit_stride)
                            + u64::from(meta.forward_pass_index),
                        slot_index: i as u32,
                        iteration: meta.iteration_generated as u32,
                        forward_pass_index: meta.forward_pass_index,
                        intercept: pool.intercept(i),
                        coefficients: pool.coefficient_row(i),
                        is_active: pool.is_active(i),
                    }
                })
                .collect()
        })
        .collect()
}

/// Rescale [`build_stage_cut_records`]'s output from the writing study's
/// internal scaled cost space to canonical currency units at rest: every
/// `coefficients` entry and `intercept`, multiplied by
/// `cost_scale_factor`.
///
/// Returns owned records because [`PolicyCutRecord::coefficients`] borrows —
/// the writing study's [`FutureCostFunction`] pool storage cannot be mutated
/// in place (it stays live for further training/warm-start), so the scaled
/// values need a fresh, owned home. Pair with [`borrow_cut_records`] to
/// rebuild the `PolicyCutRecord` views [`build_stage_cuts_payloads`] needs.
#[must_use]
pub fn scale_cut_records_for_export(
    stage_records: &[Vec<PolicyCutRecord<'_>>],
    cost_scale_factor: f64,
) -> Vec<Vec<OwnedPolicyCutRecord>> {
    stage_records
        .iter()
        .map(|records| {
            records
                .iter()
                .map(|r| OwnedPolicyCutRecord {
                    cut_id: r.cut_id,
                    slot_index: r.slot_index,
                    iteration: r.iteration,
                    forward_pass_index: r.forward_pass_index,
                    intercept: r.intercept * cost_scale_factor,
                    coefficients: r
                        .coefficients
                        .iter()
                        .map(|c| c * cost_scale_factor)
                        .collect(),
                    is_active: r.is_active,
                })
                .collect()
        })
        .collect()
}

/// Rebuild borrowed [`PolicyCutRecord`] views over
/// [`scale_cut_records_for_export`]'s owned output, for
/// [`build_stage_cuts_payloads`] and [`build_active_indices`].
#[must_use]
pub fn borrow_cut_records(owned: &[Vec<OwnedPolicyCutRecord>]) -> Vec<Vec<PolicyCutRecord<'_>>> {
    owned
        .iter()
        .map(|records| {
            records
                .iter()
                .map(|r| PolicyCutRecord {
                    cut_id: r.cut_id,
                    slot_index: r.slot_index,
                    iteration: r.iteration,
                    forward_pass_index: r.forward_pass_index,
                    intercept: r.intercept,
                    coefficients: &r.coefficients,
                    is_active: r.is_active,
                })
                .collect()
        })
        .collect()
}

/// Build per-stage active cut index lists from the stage cut records.
#[must_use]
pub fn build_active_indices(stage_records: &[Vec<PolicyCutRecord<'_>>]) -> Vec<Vec<u32>> {
    stage_records
        .iter()
        .map(|records| {
            records
                .iter()
                .filter(|r| r.is_active)
                .map(|r| r.slot_index)
                .collect()
        })
        .collect()
}

/// The declared id of `pool`'s sole owning node, or [`STAGE_CUTS_NODE_ID_SENTINEL`]
/// when more than one node shares the pool. A shared pool is never a boundary
/// source, so its provenance node id is deliberately undefined; a single-owner
/// pool (every non-leaf pool, and a boundary source) yields the owner's id.
fn sole_pool_owner_node_id(node_graph: &NodeGraph, pool: usize) -> i32 {
    let mut owner = None;
    for (pos, node) in node_graph.nodes.iter_indexed() {
        if node.pool_id == pool {
            if owner.is_some() {
                return STAGE_CUTS_NODE_ID_SENTINEL;
            }
            owner = Some(node_graph.node_ids[pos].0);
        }
    }
    owner.unwrap_or(STAGE_CUTS_NODE_ID_SENTINEL)
}

/// Build [`StageCutsPayload`] references from pre-built records, indices, and
/// per-stage entity manifests, stamping the self-describing per-pool facts.
///
/// `stage_records`, `stage_active_indices`, and `stage_manifests` must have been
/// built from the same `fcf` (via [`build_stage_cut_records`],
/// [`build_active_indices`], and [`build_stage_entity_manifest`] per pool), so
/// each is indexed by the same pool index. `stage_manifests[t]` carries one slot
/// per cut-state dimension of pool `t`. Each pool's `cost_scale_factor` is the
/// study's single resolved factor, `graph_stage_id` is
/// `study_stage_ids[node_graph.pool_stage[pool]]`, `node_id` is the pool's
/// sole owning node id (see [`sole_pool_owner_node_id`]), and
/// `priced_state_date` is [`encode_slot_date`] of
/// `study_stage_end_dates[node_graph.pool_stage[pool]]` — the pool's owning
/// stage's exclusive `end_date`, day-accurate: the instant the pool's priced
/// state leaves that stage, not the enclosing calendar month.
/// `study_stage_ids` and `study_stage_end_dates` share the same
/// `pool_stage` indirection and the same out-of-range sentinel fallback, so a
/// pool with a sentinel `graph_stage_id` also carries a sentinel
/// `priced_state_date`.
#[must_use]
pub fn build_stage_cuts_payloads<'a>(
    fcf: &FutureCostFunction,
    node_graph: &NodeGraph,
    study_stage_ids: &[i32],
    study_stage_end_dates: &[NaiveDate],
    cost_scale_factor: f64,
    stage_records: &'a [Vec<PolicyCutRecord<'a>>],
    stage_active_indices: &'a [Vec<u32>],
    stage_manifests: &'a [Vec<EntitySlot>],
) -> Vec<StageCutsPayload<'a>> {
    debug_assert_eq!(
        study_stage_ids.len(),
        study_stage_end_dates.len(),
        "study_stage_ids and study_stage_end_dates must be threaded 1:1"
    );
    fcf.pools
        .iter()
        .enumerate()
        .map(|(pool, pool_data)| {
            let pool_stage = node_graph.pool_stage[pool].0;
            StageCutsPayload {
                stage_id: pool as u32,
                state_dimension: fcf.state_dimension as u32,
                capacity: pool_data.capacity as u32,
                warm_start_count: pool_data.warm_start_count,
                cuts: &stage_records[pool],
                active_cut_indices: &stage_active_indices[pool],
                populated_count: pool_data.populated() as u32,
                entity_manifest: &stage_manifests[pool],
                cost_scale_factor,
                node_id: sole_pool_owner_node_id(node_graph, pool),
                graph_stage_id: study_stage_ids
                    .get(pool_stage)
                    .copied()
                    .unwrap_or(STAGE_CUTS_GRAPH_STAGE_ID_SENTINEL),
                priced_state_date: study_stage_end_dates
                    .get(pool_stage)
                    .copied()
                    .map_or(STAGE_CUTS_PRICED_STATE_DATE_SENTINEL, encode_slot_date),
            }
        })
        .collect()
}

/// Convert the solver basis cache to u8 byte vectors via `to_discriminant_code`.
///
/// The canonical discriminant space (`0..=6`) is a strict superset of the `HiGHS`
/// code space this format previously stored, so pre-existing checkpoints (bytes
/// `0..=4`) decode identically — while a CLP-captured `Superbasic`/`Fixed`, which
/// `to_highs_code` would fold, now survives reload. Mirrored on load by
/// `build_basis_cache_from_checkpoint`'s `from_discriminant_code`.
#[must_use]
pub fn convert_basis_cache(training_result: &TrainingResult) -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
    training_result
        .basis_cache
        .iter()
        .map(|opt| {
            opt.as_ref().map_or_else(
                || (Vec::new(), Vec::new()),
                |cb| {
                    (
                        cb.basis
                            .col_status
                            .iter()
                            .map(|status| status.to_discriminant_code())
                            .collect(),
                        cb.basis
                            .row_status
                            .iter()
                            .map(|status| status.to_discriminant_code())
                            .collect(),
                    )
                },
            )
        })
        .unzip()
}

/// Build per-node [`PolicyBasisRecord`] references from pre-converted basis data.
///
/// `training_result.basis_cache` is node-indexed (one slot per `node_graph`
/// position), so the enumeration index is the node ordinal, carried into
/// `stage_id`. `num_cut_rows` counts the node's OWN pool's affine rows —
/// resolved through `node_graph.nodes[node].pool_id`, never `fcf.pools[node]`,
/// which reads the wrong pool once `n_pools != n_nodes` on a branching graph.
#[must_use]
pub fn build_stage_basis_records<'a>(
    fcf: &FutureCostFunction,
    training_result: &TrainingResult,
    node_graph: &NodeGraph,
    basis_col_u8: &'a [Vec<u8>],
    basis_row_u8: &'a [Vec<u8>],
) -> Vec<PolicyBasisRecord<'a>> {
    training_result
        .basis_cache
        .iter()
        .enumerate()
        .filter_map(|(node, opt)| {
            opt.as_ref().map(|_| {
                let pool = node_graph.nodes[NodePos(node)].pool_id;
                let num_cut_rows = fcf
                    .pools
                    .get(pool)
                    .map_or(0, |pool| pool.populated().min(pool.capacity) as u32);
                PolicyBasisRecord {
                    stage_id: node as u32,
                    iteration: training_result.iterations as u32,
                    column_status: &basis_col_u8[node],
                    row_status: &basis_row_u8[node],
                    num_cut_rows,
                }
            })
        })
        .collect()
}

/// Build the value-function artifact's graph manifest from the runtime node
/// graph: one manifest node per canonical position carrying its declared id, its
/// stage id (resolved through `study_stage_ids`), and its pool id, plus the
/// flattened edge list. `pool_id` IS the node → pool map; leaf nodes sharing a
/// pool all name the same `pool_id`.
///
/// Single owner shared by the checkpoint writer and the full-FCF load-path
/// graph-identity check, so the two cannot derive a divergent manifest for the
/// same study.
#[must_use]
pub fn build_graph_manifest(node_graph: &NodeGraph, study_stage_ids: &[i32]) -> GraphManifest {
    let nodes = node_graph
        .nodes
        .iter_indexed()
        .map(|(pos, node)| ManifestNode {
            id: node_graph.node_ids[pos].0,
            stage_id: study_stage_ids.get(node.stage.0).copied().unwrap_or(-1),
            pool_id: node.pool_id as u32,
        })
        .collect();
    let edges = node_graph
        .successors
        .iter_indexed()
        .flat_map(|(pos, succs)| {
            let source_id = node_graph.node_ids[pos].0;
            succs.iter().map(move |succ| ManifestEdge {
                source_id,
                target_id: node_graph.node_ids[succ.child].0,
                probability: succ.probability,
            })
        })
        .collect();
    GraphManifest {
        n_pools: node_graph.n_pools as u32,
        nodes,
        edges,
    }
}

/// Build per-stage [`StageStatesPayload`]s from the visited states archive.
///
/// Returns an empty `Vec` if the archive is `None` (non-Dominated strategies).
/// `archive` is indexed per NODE (one `NodeStates` per `node_graph` position);
/// `stage_manifests` is indexed per POOL, one entry per `fcf.pools` slot —
/// the same array [`build_stage_cuts_payloads`] indexes. A branching graph has
/// strictly more nodes than pools (sibling leaves share one pool), so node
/// position `pos`'s manifest is `stage_manifests[node_graph.nodes[pos].pool_id]`
/// — never `stage_manifests[pos.0]`, which is only safe when `pool_id == pos.0`
/// (pool id is not itself a `NodePos`, so this substitution still type-checks).
///
/// `stage_id` carries the node's STUDY STAGE (`node_graph.nodes[pos].stage`);
/// `node_id` carries the node's own declared id (`node_graph.node_ids[pos]`) —
/// both distinct from the node's position `pos`, so a branching artifact's
/// per-node states are attributable to their stage, their declared id, and
/// their position independently.
#[must_use]
pub fn build_stage_states_payloads<'a>(
    archive: Option<&'a VisitedStatesArchive>,
    stage_manifests: &'a [Vec<EntitySlot>],
    node_graph: &NodeGraph,
) -> Vec<StageStatesPayload<'a>> {
    let Some(archive) = archive else {
        return Vec::new();
    };
    (0..archive.num_nodes())
        .map(|t| {
            let pos = NodePos(t);
            let node = archive.node(pos);
            let pool_id = node_graph.nodes[pos].pool_id;
            StageStatesPayload {
                stage_id: node_graph.nodes[pos].stage.0 as u32,
                node_id: node_graph.node_ids[pos].0,
                state_dimension: node.state_dimension() as u32,
                count: node.count() as u32,
                data: node.states(),
                entity_manifest: &stage_manifests[pool_id],
            }
        })
        .collect()
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::unreadable_literal
)]
mod tests {
    use super::{
        EntitySlot, HashMap, StateFamily, build_stage_entity_manifest, build_stage_states_payloads,
        reserve_boundary_inflow_lag_slots,
    };
    use crate::lead_time::{AnticipatedResolution, DeliveryAxis, LeadTime};
    use crate::lp::indexer::{AnticipatedPlants, CutStateProjection, HydroSys, StateSpace};
    use crate::setup::{
        NodeGraph, NodeId, NodeOpenings, NodePos, NodeRuntime, NodeSuccessor, OpeningSource,
        StageIdx, extended_delivery_stages, year_month_day_anchor,
    };
    use crate::test_support::{self, anticipated_slot};
    use crate::time_value::post_study_delivery_calendar;
    use crate::visited_states::VisitedStatesArchive;
    use cobre_core::commissioning::hydro_operating_active;
    use cobre_core::temporal::StageStateConfig;
    use cobre_core::{
        AnticipatedConfig, Block, BlockMode, Bus, DeficitSegment, EntityId, Hydro,
        HydroGenerationModel, HydroPenalties, NoiseMethod, PostStudyStage, PostStudyStages,
        ScenarioSourceConfig, Stage, StageRiskConfig, System, SystemBuilder, Thermal,
        resolved::{
            BoundsCountsSpec, BoundsDefaults, ContractBlockBounds, HydroBlockBounds,
            HydroStageBounds, LineBlockBounds, PumpingBlockBounds, ResolvedBounds,
            ThermalBlockBounds, ThermalStageBounds,
        },
    };
    use cobre_io::{ENTITY_SLOT_DATE_SENTINEL, encode_slot_date};

    const ALL_ENABLED: StageStateConfig = StageStateConfig {
        storage: true,
        inflow_lags: true,
    };
    const STORAGE_ONLY: StageStateConfig = StageStateConfig {
        storage: true,
        inflow_lags: false,
    };

    fn penalties_zero() -> HydroPenalties {
        HydroPenalties {
            spillage_cost: 0.0,
            diversion_cost: 0.0,
            turbined_cost: 0.0,
            storage_violation_below_cost: 0.0,
            filling_target_violation_cost: 0.0,
            turbined_violation_below_cost: 0.0,
            outflow_violation_below_cost: 0.0,
            outflow_violation_above_cost: 0.0,
            generation_violation_below_cost: 0.0,
            evaporation_violation_cost: 0.0,
            water_withdrawal_violation_cost: 0.0,
            water_withdrawal_violation_pos_cost: 0.0,
            water_withdrawal_violation_neg_cost: 0.0,
            evaporation_violation_pos_cost: 0.0,
            evaporation_violation_neg_cost: 0.0,
            inflow_nonnegativity_cost: 1000.0,
        }
    }

    fn bounds_defaults() -> BoundsDefaults {
        BoundsDefaults {
            hydro: HydroStageBounds {
                min_storage_hm3: 0.0,
                max_storage_hm3: 100.0,
                filling_min_rate_m3s: 0.0,
                water_withdrawal_m3s: 0.0,
            },
            hydro_block: HydroBlockBounds {
                max_turbined_m3s: 50.0,
                max_generation_mw: 45.0,
                ..Default::default()
            },
            thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
            thermal_block: ThermalBlockBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 100.0,
            },
            line_block: LineBlockBounds {
                direct_mw: 500.0,
                reverse_mw: 500.0,
            },
            pumping_block: PumpingBlockBounds {
                min_flow_m3s: 0.0,
                max_flow_m3s: 0.0,
            },
            contract_block: ContractBlockBounds {
                min_mw: 0.0,
                max_mw: 0.0,
                price_per_mwh: 0.0,
            },
        }
    }

    fn make_hydro(id: i32, entry: Option<i32>, exit: Option<i32>) -> Hydro {
        let mut hydro = Hydro {
            unit_groups: Vec::new(),
            id: EntityId(id),
            name: format!("Hydro{id}"),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            downstream_id: None,
            travel_time_hours: None,
            entry_stage_id: entry,
            exit_stage_id: exit,
            min_storage_hm3: 0.0,
            max_storage_hm3: 100.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 50.0,
            specific_productivity_mw_per_m3s_per_m: None,
            min_generation_mw: 0.0,
            max_generation_mw: 45.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: penalties_zero(),
        };
        hydro.declare_mirror_unit_group(EntityId(1));
        hydro
    }

    fn anticipated_thermal(id: i32, lead_stages: u32) -> Thermal {
        anticipated_thermal_cfg(id, AnticipatedConfig::LeadStages(lead_stages))
    }

    fn anticipated_thermal_cfg(id: i32, cfg: AnticipatedConfig) -> Thermal {
        Thermal {
            id: EntityId(id),
            name: format!("Thermal{id}"),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(1),
            entry_stage_id: None,
            exit_stage_id: None,
            cost_per_mwh: 50.0,
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            anticipated_config: Some(cfg),
        }
    }

    fn make_bus() -> Bus {
        Bus {
            id: EntityId(1),
            name: "Bus1".to_string(),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 1000.0,
            }],
            excess_cost: 0.0,
        }
    }

    fn make_stage() -> Stage {
        Stage {
            index: 0,
            id: 0,
            start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: chrono::NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: Some(0),
            blocks: vec![Block {
                index: 0,
                name: "SINGLE".to_string(),
                duration_hours: 720.0,
            }],
            block_mode: BlockMode::Parallel,
            state_config: ALL_ENABLED,
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        }
    }

    /// `System` with 2 hydros and 1 anticipated thermal (`lead_stages = 2`),
    /// matching the `N=2, L=2, A=1, k_max=2` layout fixture. `hydros` carry the
    /// supplied commissioning windows so `was_active` can be exercised.
    /// `post_study_stages` threads straight to the builder — `None` for a
    /// fixture with no post-horizon lane.
    fn system_2h_1ant(
        h1_window: (Option<i32>, Option<i32>),
        h2_window: (Option<i32>, Option<i32>),
        post_study_stages: Option<PostStudyStages>,
    ) -> System {
        let bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 2,
                n_thermals: 1,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: 1,
                k_max: 2,
            },
            &bounds_defaults(),
        );
        SystemBuilder::new()
            .buses(vec![make_bus()])
            .hydros(vec![
                make_hydro(1, h1_window.0, h1_window.1),
                make_hydro(2, h2_window.0, h2_window.1),
            ])
            .thermals(vec![anticipated_thermal(1, 2)])
            .stages(vec![make_stage()])
            .bounds(bounds)
            .post_study_stages(post_study_stages)
            .build()
            .expect("valid system")
    }

    /// A single post-study stage starting `start_date`, covering exactly one
    /// post-study delivery target on the anticipated ring.
    fn post_study_stages_from(start_date: chrono::NaiveDate) -> PostStudyStages {
        PostStudyStages {
            stages: vec![PostStudyStage {
                start_date,
                duration_hours: 720.0,
            }],
            thermal_bounds: Vec::new(),
        }
    }

    /// `system_2h_1ant`'s single anticipated plant (`LeadStages(2)`) resolved
    /// against its one-stage delivery axis (`n_decision = n_delivery = 1`) —
    /// `g = 0`, so attaching it is byte-neutral for every non-dating
    /// assertion, but [`for_each_live_commitment_slot`]'s ring sweep needs a
    /// real per-plant [`PointResolution`] to index into.
    fn single_plant_lead2_one_stage_resolution() -> AnticipatedResolution {
        AnticipatedResolution::resolve(
            &[LeadTime::Stages(2)],
            DeliveryAxis {
                study_stage_hours: &[720.0],
                post_study_stage_hours: &[],
            },
        )
    }

    /// The `N=2, L=2, A=1, k_max=2` global layout the fixture system maps onto.
    fn layout_2h_1ant() -> StateSpace {
        test_support::state_layout_with_transit_buckets_and_resolution(
            2,
            2,
            Vec::new(),
            vec![2],
            single_plant_lead2_one_stage_resolution(),
        )
    }

    /// The `N=2, L=2, B=2, A=1, k_max=2` global layout with two travel-time
    /// buckets, sharing [`layout_2h_1ant`]'s attached single-plant resolution.
    fn layout_2h_2buckets_1ant() -> StateSpace {
        test_support::state_layout_with_transit_buckets_and_resolution(
            2,
            2,
            vec![(HydroSys::new(0), 1), (HydroSys::new(1), 2)],
            vec![2],
            single_plant_lead2_one_stage_resolution(),
        )
    }

    /// All-enabled projection: length 8 (2 storage + 4 lag + 2 anticipated), with
    /// storage slots typed 0 (subindex 0), lag slots typed 1 in lag-major order
    /// (hydro-interleaved: subindex 1,1,2,2 for hydros 1,2,1,2), and anticipated
    /// slots typed 2 (plant id 1, ring subindex 0,1).
    #[test]
    fn all_enabled_classification_identity_and_subindex() {
        let system = system_2h_1ant((None, None), (None, None), None);
        let global = layout_2h_1ant();
        let projection = CutStateProjection::new(&global, ALL_ENABLED);

        let manifest = build_stage_entity_manifest(
            &system,
            &global,
            &AnticipatedPlants::build(system.thermals()),
            &projection,
            0,
        );

        assert_eq!(manifest.len(), projection.n_slots());
        assert_eq!(manifest.len(), 8);

        assert_eq!(manifest[0].entity_type, StateFamily::HydroStorage.code());
        assert_eq!(manifest[0].entity_id, 1);
        assert_eq!(manifest[0].subindex, 0);
        assert_eq!(manifest[1].entity_type, StateFamily::HydroStorage.code());
        assert_eq!(manifest[1].entity_id, 2);
        assert_eq!(manifest[1].subindex, 0);

        // Lag block, lag-major: (lag0,h0),(lag0,h1),(lag1,h0),(lag1,h1).
        for (slot, (expected_id, expected_lag)) in
            [(2, (1, 1)), (3, (2, 1)), (4, (1, 2)), (5, (2, 2))]
        {
            assert_eq!(
                manifest[slot].entity_type,
                StateFamily::HydroInflowLag.code(),
                "slot {slot} must be an inflow-lag slot"
            );
            assert_eq!(
                manifest[slot].entity_id, expected_id,
                "slot {slot} hydro id"
            );
            assert_eq!(
                manifest[slot].subindex, expected_lag,
                "slot {slot} 1-based lag"
            );
        }

        // Anticipated block, slot-major (single plant, ring slots 0 and 1).
        assert_eq!(
            manifest[6].entity_type,
            StateFamily::AnticipatedThermalState.code()
        );
        assert_eq!(manifest[6].entity_id, 1);
        assert_eq!(manifest[6].subindex, 0);
        assert_eq!(
            manifest[7].entity_type,
            StateFamily::AnticipatedThermalState.code()
        );
        assert_eq!(manifest[7].entity_id, 1);
        assert_eq!(manifest[7].subindex, 1);
    }

    /// Bucket block classification (`N=2, L=2, B=2, A=1, k_max=2`): the two
    /// travel-time bucket slots sit between the lag block and the anticipated block,
    /// each carrying `entity_type == HydroTransitBucket`, `entity_id ==` the
    /// downstream hydro id (`transit_bucket_column_order[b].0` into `system.hydros()`), and
    /// `subindex ==` the maturity lag `d` (`transit_bucket_column_order[b].1`).
    #[test]
    fn bucket_slots_classify_as_transit_bucket_with_downstream_id_and_lag() {
        let system = system_2h_1ant((None, None), (None, None), None);
        let global = layout_2h_2buckets_1ant();
        let projection = CutStateProjection::new(&global, ALL_ENABLED);

        let manifest = build_stage_entity_manifest(
            &system,
            &global,
            &AnticipatedPlants::build(system.thermals()),
            &projection,
            0,
        );

        assert_eq!(manifest.len(), projection.n_slots());
        assert_eq!(
            manifest.len(),
            10,
            "2 storage + 4 lag + 2 buckets + 2 anticipated"
        );

        for (slot, (expected_id, expected_lag)) in [(6, (1, 1)), (7, (2, 2))] {
            assert_eq!(
                manifest[slot].entity_type,
                StateFamily::HydroTransitBucket.code(),
                "slot {slot} must be a transit-bucket slot"
            );
            assert_eq!(
                manifest[slot].entity_id, expected_id,
                "slot {slot} downstream hydro id"
            );
            assert_eq!(
                manifest[slot].subindex, expected_lag,
                "slot {slot} maturity lag d"
            );
        }

        assert_eq!(
            manifest[5].entity_type,
            StateFamily::HydroInflowLag.code(),
            "buckets must follow the lag block"
        );
        assert_eq!(
            manifest[8].entity_type,
            StateFamily::AnticipatedThermalState.code(),
            "buckets must precede the anticipated block"
        );
    }

    /// A transit-bucket lag reaching a declared post-study stage now dates onto
    /// that stage's own `[start, end)` interval, exactly as the anticipated
    /// ring already does; a lag past the extended calendar still stays
    /// sentinel in every date field. On this single-study-stage fixture lag 1
    /// (`t + lag == 1`) resolves to the declared post-study stage while lag 2
    /// (`t + lag == 2`) lands past it.
    #[test]
    fn transit_bucket_dates_onto_the_post_study_calendar_and_sentinels_past_it() {
        let post_study_start = chrono::NaiveDate::from_ymd_opt(2024, 2, 1).unwrap();
        let system = system_2h_1ant(
            (None, None),
            (None, None),
            Some(post_study_stages_from(post_study_start)),
        );
        let global = layout_2h_2buckets_1ant();
        let projection = CutStateProjection::new(&global, ALL_ENABLED);

        let manifest = build_stage_entity_manifest(
            &system,
            &global,
            &AnticipatedPlants::build(system.thermals()),
            &projection,
            0,
        );
        let post_study_end = post_study_delivery_calendar(&system)[0].end_date;

        assert_eq!(
            manifest[6].entity_type,
            StateFamily::HydroTransitBucket.code(),
            "slot 6 must be a transit-bucket slot"
        );
        assert_eq!(
            manifest[6].interval_start, 20_240_201,
            "lag 1 dates onto the declared post-study stage's own interval start"
        );
        assert_eq!(manifest[6].interval_end, encode_slot_date(post_study_end));

        assert_eq!(
            manifest[7].entity_type,
            StateFamily::HydroTransitBucket.code(),
            "slot 7 must be a transit-bucket slot"
        );
        for (field, name) in [
            (manifest[7].interval_start, "interval_start"),
            (manifest[7].interval_end, "interval_end"),
        ] {
            assert_eq!(
                field, ENTITY_SLOT_DATE_SENTINEL,
                "lag 2 lands past the extended calendar, so {name} must stay sentinel"
            );
        }
    }

    /// Storage-only projection (`inflow_lags: false`): the lag block is dropped, so
    /// the manifest is length 4 (2 storage + 2 anticipated) and carries NO type-1
    /// (`HydroInflowLag`) slot. Anticipated state is always included.
    #[test]
    fn storage_only_drops_lag_slots() {
        let system = system_2h_1ant((None, None), (None, None), None);
        let global = layout_2h_1ant();
        let projection = CutStateProjection::new(&global, STORAGE_ONLY);

        let manifest = build_stage_entity_manifest(
            &system,
            &global,
            &AnticipatedPlants::build(system.thermals()),
            &projection,
            0,
        );

        assert_eq!(manifest.len(), projection.n_slots());
        assert_eq!(manifest.len(), 4);
        assert!(
            manifest
                .iter()
                .all(|s| s.entity_type != StateFamily::HydroInflowLag.code()),
            "storage-only manifest must contain no HydroInflowLag slot"
        );
        assert_eq!(manifest[0].entity_type, StateFamily::HydroStorage.code());
        assert_eq!(manifest[1].entity_type, StateFamily::HydroStorage.code());
        assert_eq!(
            manifest[2].entity_type,
            StateFamily::AnticipatedThermalState.code()
        );
        assert_eq!(manifest[2].subindex, 0);
        assert_eq!(
            manifest[3].entity_type,
            StateFamily::AnticipatedThermalState.code()
        );
        assert_eq!(manifest[3].subindex, 1);
    }

    /// A hydro dormant at the slot's stage (commissioning window `[2, 5)` queried at
    /// stage 1) yields `was_active == false` on every slot it owns, and that value
    /// equals the single-owner `hydro_operating_active` predicate. The second hydro
    /// (no window) stays active, isolating the per-entity flag.
    #[test]
    fn was_active_matches_hydro_operating_active_for_dormant_window() {
        let h1_window = (Some(2), Some(5));
        let system = system_2h_1ant(h1_window, (None, None), None);
        let global = layout_2h_1ant();
        let projection = CutStateProjection::new(&global, ALL_ENABLED);
        let stage_id = 1;

        let manifest = build_stage_entity_manifest(
            &system,
            &global,
            &AnticipatedPlants::build(system.thermals()),
            &projection,
            stage_id,
        );

        let expected_h1 = hydro_operating_active(None, h1_window.0, h1_window.1, stage_id);
        assert!(!expected_h1, "hydro 1 must be dormant at stage 1");

        // Hydro 1 owns storage slot 0 and lag slots 2, 4 (lag-major, h index 0).
        for slot in [0, 2, 4] {
            assert_eq!(manifest[slot].entity_id, 1, "slot {slot} must be hydro 1");
            assert_eq!(
                manifest[slot].was_active, expected_h1,
                "slot {slot} was_active must equal hydro_operating_active"
            );
        }
        // Hydro 2 has no window: active.
        for slot in [1, 3, 5] {
            assert_eq!(manifest[slot].entity_id, 2, "slot {slot} must be hydro 2");
            assert!(manifest[slot].was_active, "slot {slot} hydro 2 is active");
        }
    }

    // -- inflow-lag `reference_date` dating --

    /// A stage covering `[start, end)` at domain id `id`; `index`/`season_id`
    /// are overwritten by `SystemBuilder::build`'s canonical-order sort, so a
    /// placeholder value is fine. `id` may be negative (a pre-study stage).
    fn make_stage_dated(id: i32, start: chrono::NaiveDate, end: chrono::NaiveDate) -> Stage {
        Stage {
            index: 0,
            id,
            start_date: start,
            end_date: end,
            season_id: None,
            blocks: vec![Block {
                index: 0,
                name: "SINGLE".to_string(),
                duration_hours: 720.0,
            }],
            block_mode: BlockMode::Parallel,
            state_config: ALL_ENABLED,
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        }
    }

    /// A storage+lag-only `System` (no anticipated thermals): `n_hydro` hydros
    /// (ids `1..=n_hydro`) over `stages`, already carrying their own ids/dates
    /// (negative ids for pre-study stages included).
    fn system_lag_only(n_hydro: usize, stages: Vec<Stage>) -> System {
        let n_stages = stages.len();
        let bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: n_hydro,
                n_thermals: 0,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages,
                k_max: 0,
            },
            &bounds_defaults(),
        );
        SystemBuilder::new()
            .buses(vec![make_bus()])
            .hydros(
                (1..=n_hydro as i32)
                    .map(|id| make_hydro(id, None, None))
                    .collect(),
            )
            .stages(stages)
            .bounds(bounds)
            .build()
            .expect("valid lag-only system")
    }

    /// Three consecutive monthly study stages (ids 0, 1, 2) ending on the pool's
    /// own stage, 2031-11-01.
    fn three_monthly_stages_ending_nov_2031() -> Vec<Stage> {
        let d = |y, m| chrono::NaiveDate::from_ymd_opt(y, m, 1).unwrap();
        vec![
            make_stage_dated(0, d(2031, 9), d(2031, 10)),
            make_stage_dated(1, d(2031, 10), d(2031, 11)),
            make_stage_dated(2, d(2031, 11), d(2031, 12)),
        ]
    }

    #[test]
    fn inflow_lag_slot_lag_one_references_the_pools_own_stage() {
        let system = system_lag_only(1, three_monthly_stages_ending_nov_2031());
        let global = test_support::state_layout(1, 3);
        let projection = CutStateProjection::new(&global, ALL_ENABLED);

        let manifest = build_stage_entity_manifest(
            &system,
            &global,
            &AnticipatedPlants::build(system.thermals()),
            &projection,
            2,
        );

        let lag1 = manifest
            .iter()
            .find(|s| s.entity_type == StateFamily::HydroInflowLag.code() && s.subindex == 1)
            .expect("a subindex-1 inflow-lag slot must exist");
        assert_eq!(lag1.reference_date, 20_311_101);
    }

    #[test]
    fn inflow_lag_slot_deeper_lags_step_back_one_stage_each() {
        let system = system_lag_only(1, three_monthly_stages_ending_nov_2031());
        let global = test_support::state_layout(1, 3);
        let projection = CutStateProjection::new(&global, ALL_ENABLED);

        let manifest = build_stage_entity_manifest(
            &system,
            &global,
            &AnticipatedPlants::build(system.thermals()),
            &projection,
            2,
        );

        let lag1 = manifest
            .iter()
            .find(|s| s.entity_type == StateFamily::HydroInflowLag.code() && s.subindex == 1)
            .expect("a subindex-1 inflow-lag slot must exist");
        let lag3 = manifest
            .iter()
            .find(|s| s.entity_type == StateFamily::HydroInflowLag.code() && s.subindex == 3)
            .expect("a subindex-3 inflow-lag slot must exist");
        assert_eq!(lag1.reference_date, 20_311_101);
        assert_eq!(
            lag3.reference_date, 20_310_901,
            "subindex 3 is two stages earlier than the subindex-1 slot's stage"
        );
    }

    #[test]
    fn inflow_lag_slot_reaching_pre_study_window_dates_onto_a_negative_id_stage() {
        let d = |y, m| chrono::NaiveDate::from_ymd_opt(y, m, 1).unwrap();
        let stages = vec![
            make_stage_dated(-3, d(2031, 8), d(2031, 9)),
            make_stage_dated(-2, d(2031, 9), d(2031, 10)),
            make_stage_dated(-1, d(2031, 10), d(2031, 11)),
            make_stage_dated(0, d(2031, 11), d(2031, 12)),
        ];
        let system = system_lag_only(1, stages);
        let global = test_support::state_layout(1, 2);
        let projection = CutStateProjection::new(&global, ALL_ENABLED);

        let manifest = build_stage_entity_manifest(
            &system,
            &global,
            &AnticipatedPlants::build(system.thermals()),
            &projection,
            0,
        );

        let lag2 = manifest
            .iter()
            .find(|s| s.entity_type == StateFamily::HydroInflowLag.code() && s.subindex == 2)
            .expect("a subindex-2 inflow-lag slot must exist");
        assert_eq!(
            lag2.reference_date, 20_311_001,
            "must date onto the pre-study stage id -1, not the sentinel"
        );
    }

    #[test]
    fn inflow_lag_slot_beyond_the_earliest_stage_stays_sentinel() {
        let d = |y, m| chrono::NaiveDate::from_ymd_opt(y, m, 1).unwrap();
        let stages = vec![make_stage_dated(0, d(2031, 11), d(2031, 12))];
        let system = system_lag_only(1, stages);
        let global = test_support::state_layout(1, 3);
        let projection = CutStateProjection::new(&global, ALL_ENABLED);

        let manifest = build_stage_entity_manifest(
            &system,
            &global,
            &AnticipatedPlants::build(system.thermals()),
            &projection,
            0,
        );

        let lag1 = manifest
            .iter()
            .find(|s| s.entity_type == StateFamily::HydroInflowLag.code() && s.subindex == 1)
            .expect("a subindex-1 inflow-lag slot must exist");
        assert_eq!(
            lag1.reference_date, 20_311_101,
            "the pool's own stage stays dated"
        );

        for subindex in [2, 3] {
            let slot = manifest
                .iter()
                .find(|s| {
                    s.entity_type == StateFamily::HydroInflowLag.code() && s.subindex == subindex
                })
                .unwrap_or_else(|| panic!("a subindex-{subindex} inflow-lag slot must exist"));
            assert_eq!(
                slot.reference_date, ENTITY_SLOT_DATE_SENTINEL,
                "subindex {subindex} reaches before the earliest declared stage"
            );
        }
    }

    #[test]
    fn inflow_lag_reference_date_is_independent_of_hydro_declaration_order() {
        let system = system_lag_only(2, three_monthly_stages_ending_nov_2031());
        let global = test_support::state_layout(2, 2);
        let projection = CutStateProjection::new(&global, ALL_ENABLED);

        let manifest = build_stage_entity_manifest(
            &system,
            &global,
            &AnticipatedPlants::build(system.thermals()),
            &projection,
            2,
        );

        for subindex in [1, 2] {
            let dates: Vec<i32> = manifest
                .iter()
                .filter(|s| {
                    s.entity_type == StateFamily::HydroInflowLag.code() && s.subindex == subindex
                })
                .map(|s| s.reference_date)
                .collect();
            assert_eq!(
                dates.len(),
                2,
                "both hydros must own a slot at subindex {subindex}"
            );
            assert_eq!(
                dates[0], dates[1],
                "subindex {subindex}'s reference_date must not depend on which hydro owns the slot"
            );
        }
    }

    /// A stage starting mid-week (2031-11-03, not day-01) must stamp
    /// `reference_date` at its own day — `year_month_day_anchor` would
    /// truncate it to `20311101`, colliding it with any sibling stage
    /// starting earlier in the same month.
    #[test]
    fn inflow_lag_reference_date_keeps_the_stage_start_day() {
        let stages = vec![make_stage_dated(
            0,
            chrono::NaiveDate::from_ymd_opt(2031, 11, 3).unwrap(),
            chrono::NaiveDate::from_ymd_opt(2031, 11, 10).unwrap(),
        )];
        let system = system_lag_only(1, stages);
        let global = test_support::state_layout(1, 1);
        let projection = CutStateProjection::new(&global, ALL_ENABLED);

        let manifest = build_stage_entity_manifest(
            &system,
            &global,
            &AnticipatedPlants::build(system.thermals()),
            &projection,
            0,
        );

        let lag1 = manifest
            .iter()
            .find(|s| s.entity_type == StateFamily::HydroInflowLag.code() && s.subindex == 1)
            .expect("a subindex-1 inflow-lag slot must exist");
        assert_eq!(
            lag1.reference_date, 20_311_103,
            "the reference date must keep the stage's own start day, not truncate to day 01"
        );
    }

    /// A monthly stage (`index`/`id` shared) starting on the first of `year`/`month`.
    fn make_stage_ym(index: usize, id: i32, year: i32, month: u32) -> Stage {
        let start = chrono::NaiveDate::from_ymd_opt(year, month, 1).unwrap();
        let (end_year, end_month) = if month == 12 {
            (year + 1, 1)
        } else {
            (year, month + 1)
        };
        Stage {
            index,
            id,
            start_date: start,
            end_date: chrono::NaiveDate::from_ymd_opt(end_year, end_month, 1).unwrap(),
            season_id: Some((month - 1) as usize),
            blocks: vec![Block {
                index: 0,
                name: "SINGLE".to_string(),
                duration_hours: 720.0,
            }],
            block_mode: BlockMode::Parallel,
            state_config: ALL_ENABLED,
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        }
    }

    /// `System` with 1 hydro and 1 anticipated thermal carrying `cfg` over three
    /// consecutive monthly stages 2024-04, 2024-05, 2024-06 (ids 0, 1, 2).
    fn system_1h_1ant_3monthly(cfg: AnticipatedConfig) -> System {
        let bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 1,
                n_thermals: 1,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: 3,
                k_max: 2,
            },
            &bounds_defaults(),
        );
        SystemBuilder::new()
            .buses(vec![make_bus()])
            .hydros(vec![make_hydro(1, None, None)])
            .thermals(vec![anticipated_thermal_cfg(1, cfg)])
            .stages(vec![
                make_stage_ym(0, 0, 2024, 4),
                make_stage_ym(1, 1, 2024, 5),
                make_stage_ym(2, 2, 2024, 6),
            ])
            .bounds(bounds)
            .build()
            .expect("valid 3-stage system")
    }

    /// `System` with 1 hydro and two anticipated thermals of DIFFERENT
    /// `LeadStages` (id 1: ℓ=1, id 2: ℓ=2) over `n_stages` consecutive monthly
    /// stages starting 2024-04 (the first three match
    /// [`system_1h_1ant_3monthly`]'s own; stage 3, when present, starts
    /// 2024-07-01), sharing one `k_max=2` ring.
    fn system_1h_2ant_monthly(n_stages: usize) -> System {
        let bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 1,
                n_thermals: 2,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages,
                k_max: 2,
            },
            &bounds_defaults(),
        );
        let stages = (0..n_stages)
            .map(|i| make_stage_ym(i, i as i32, 2024, 4 + i as u32))
            .collect();
        SystemBuilder::new()
            .buses(vec![make_bus()])
            .hydros(vec![make_hydro(1, None, None)])
            .thermals(vec![anticipated_thermal(1, 1), anticipated_thermal(2, 2)])
            .stages(stages)
            .bounds(bounds)
            .build()
            .expect("valid multi-stage 2-anticipated-plant system")
    }

    /// `System` with 1 hydro and 1 anticipated thermal (`LeadStages(3)`, so the
    /// ring is `k_max = 3`) over the same three monthly stages as
    /// [`system_1h_1ant_3monthly`], carrying `post_study`. The extended-calendar
    /// fixture: at the terminal stage the three ring slots' modular delivery
    /// targets fan across in-study, post-study, and past-extended.
    fn system_1h_1ant_3monthly_lead3(post_study: PostStudyStages) -> System {
        let bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 1,
                n_thermals: 1,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: 3,
                k_max: 3,
            },
            &bounds_defaults(),
        );
        SystemBuilder::new()
            .buses(vec![make_bus()])
            .hydros(vec![make_hydro(1, None, None)])
            .thermals(vec![anticipated_thermal(1, 3)])
            .stages(vec![
                make_stage_ym(0, 0, 2024, 4),
                make_stage_ym(1, 1, 2024, 5),
                make_stage_ym(2, 2, 2024, 6),
            ])
            .bounds(bounds)
            .post_study_stages(Some(post_study))
            .build()
            .expect("valid 3-stage lead-3 system with post-study calendar")
    }

    /// A post-study calendar of consecutive `(year, month)` monthly stages.
    fn post_study_stages_from_months(months: &[(i32, u32)]) -> PostStudyStages {
        PostStudyStages {
            stages: months
                .iter()
                .map(|&(year, month)| PostStudyStage {
                    start_date: chrono::NaiveDate::from_ymd_opt(year, month, 1).unwrap(),
                    duration_hours: 720.0,
                })
                .collect(),
            thermal_bounds: Vec::new(),
        }
    }

    /// `System` with 1 hydro and 1 anticipated thermal (`LeadStages(7)`) over
    /// four monthly study stages, carrying `post_study`: over a `DeliveryAxis`
    /// of `n_decision = 4`, [`AnticipatedResolution::resolve`]'s own ring
    /// depth (`k_max`) is bounded to 4, excising ring-axis targets `4..7`
    /// ([`PointResolution::ring_index`]) even though the plant's own lead
    /// reaches 7.
    fn system_1h_1ant_4monthly_lead7(post_study: PostStudyStages) -> System {
        let bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 1,
                n_thermals: 1,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: 4,
                k_max: 7,
            },
            &bounds_defaults(),
        );
        SystemBuilder::new()
            .buses(vec![make_bus()])
            .hydros(vec![make_hydro(1, None, None)])
            .thermals(vec![anticipated_thermal(1, 7)])
            .stages(vec![
                make_stage_ym(0, 0, 2024, 1),
                make_stage_ym(1, 1, 2024, 2),
                make_stage_ym(2, 2, 2024, 3),
                make_stage_ym(3, 3, 2024, 4),
            ])
            .bounds(bounds)
            .post_study_stages(Some(post_study))
            .build()
            .expect("valid 4-stage lead-7 system with post-study calendar")
    }

    /// `System` with 1 hydro and 1 anticipated thermal (`LeadStages(1)`, so
    /// `k_max = 1`) over two study stages: a 14-day first stage, then a
    /// non-month-aligned 35-day (five-week) terminal stage starting
    /// 2026-01-15 — a real, day-accurate delivery span for a slot targeting
    /// it, not the enclosing calendar month.
    fn system_1h_1ant_short_then_five_week_terminal() -> System {
        let bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 1,
                n_thermals: 1,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: 2,
                k_max: 1,
            },
            &bounds_defaults(),
        );
        let stage0_start = chrono::NaiveDate::from_ymd_opt(2026, 1, 1).unwrap();
        let stage0_end = chrono::NaiveDate::from_ymd_opt(2026, 1, 15).unwrap();
        let stage1_end = chrono::NaiveDate::from_ymd_opt(2026, 2, 19).unwrap();
        let stage = |index, id, start, end, season_id, duration_hours| Stage {
            index,
            id,
            start_date: start,
            end_date: end,
            season_id: Some(season_id),
            blocks: vec![Block {
                index: 0,
                name: "SINGLE".to_string(),
                duration_hours,
            }],
            block_mode: BlockMode::Parallel,
            state_config: ALL_ENABLED,
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        };
        SystemBuilder::new()
            .buses(vec![make_bus()])
            .hydros(vec![make_hydro(1, None, None)])
            .thermals(vec![anticipated_thermal(1, 1)])
            .stages(vec![
                stage(0, 0, stage0_start, stage0_end, 0, 336.0),
                stage(1, 1, stage0_end, stage1_end, 1, 840.0),
            ])
            .bounds(bounds)
            .build()
            .expect("valid 2-stage system with a non-month-aligned five-week terminal stage")
    }

    /// `n` consecutive monthly study stages (ids `0..n`), the last one ending
    /// `end_year`/`end_month` — the many-stage generalization of
    /// [`three_monthly_stages_ending_nov_2031`].
    fn monthly_stages_ending(n: usize, end_year: i32, end_month: u32) -> Vec<Stage> {
        let end_abs = end_year * 12 + (end_month as i32 - 1);
        (0..n)
            .map(|i| {
                let abs = end_abs - (n - 1 - i) as i32;
                let year = abs.div_euclid(12);
                let month = (abs.rem_euclid(12) + 1) as u32;
                make_stage_ym(i, i as i32, year, month)
            })
            .collect()
    }

    /// 64 consecutive monthly study stages ending 2031-11-01 (ids 0..63), 1
    /// hydro, and 1 anticipated thermal with `lead_stages = 2` (so
    /// `k_max = 2`), carrying `post_study` — the terminal-maturing-residue
    /// fixture.
    fn system_1h_1ant_64monthly_lead2(post_study: Option<PostStudyStages>) -> System {
        let bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 1,
                n_thermals: 1,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: 64,
                k_max: 2,
            },
            &bounds_defaults(),
        );
        SystemBuilder::new()
            .buses(vec![make_bus()])
            .hydros(vec![make_hydro(1, None, None)])
            .thermals(vec![anticipated_thermal(1, 2)])
            .stages(monthly_stages_ending(64, 2031, 11))
            .bounds(bounds)
            .post_study_stages(post_study)
            .build()
            .expect("valid 64-stage lead-2 system")
    }

    /// An `AnticipatedThermalState` ring slot's `interval_start` is the
    /// `YYYYMMDD` start date of its delivery stage — the next stage
    /// `m >= t_out` whose residue `m mod k_max` equals the slot, where
    /// `t_out` is the OUTGOING anchor (`current_stage_idx + 1`) — resolved
    /// through the attached `AnticipatedResolution`; storage/lag slots stay
    /// at the sentinel. At stage index 1 (`t_out = 2`) with `k_max = 2`, ring
    /// slot 0 (residue 0) matures at the outgoing instant itself (index 2,
    /// 2024-06); ring slot 1 (residue 1, the class matching `t_out`'s own
    /// predecessor) next recurs a full `k_max` stages out, at index 3 — past
    /// the 3-stage horizon, so it stays sentinel.
    #[test]
    fn anticipated_slot_delivery_anchor_matches_delivery_stage_year_month() {
        let system = system_1h_1ant_3monthly(AnticipatedConfig::LeadStages(2));
        let global = test_support::state_layout_with_transit_buckets_and_resolution(
            1,
            1,
            Vec::new(),
            vec![2],
            AnticipatedResolution::resolve(
                &[LeadTime::Stages(2)],
                DeliveryAxis {
                    study_stage_hours: &[720.0; 3],
                    post_study_stage_hours: &[],
                },
            ),
        );
        let projection = CutStateProjection::new(&global, ALL_ENABLED);

        // stage_id 1 is the middle stage (2024-05), study index 1.
        let manifest = build_stage_entity_manifest(
            &system,
            &global,
            &AnticipatedPlants::build(system.thermals()),
            &projection,
            1,
        );

        // Layout N=1, L=1, A=1, k_max=2: storage j=0, lag j=1, anticipated j=2,3.
        assert_eq!(manifest.len(), 4);
        assert_eq!(manifest[0].entity_type, StateFamily::HydroStorage.code());
        assert_eq!(
            manifest[0].interval_start, ENTITY_SLOT_DATE_SENTINEL,
            "storage slot carries no delivery date"
        );
        assert_eq!(manifest[1].entity_type, StateFamily::HydroInflowLag.code());
        assert_eq!(
            manifest[1].interval_start, ENTITY_SLOT_DATE_SENTINEL,
            "inflow-lag slot carries no delivery date"
        );

        assert_eq!(
            manifest[2].entity_type,
            StateFamily::AnticipatedThermalState.code()
        );
        assert_eq!(manifest[2].subindex, 0);
        assert_eq!(
            manifest[2].interval_start, 20240601,
            "ring slot 0 (residue 0) matures at the outgoing anchor itself (index 2, 2024-06)"
        );

        assert_eq!(
            manifest[3].entity_type,
            StateFamily::AnticipatedThermalState.code()
        );
        assert_eq!(manifest[3].subindex, 1);
        assert_eq!(
            manifest[3].interval_start, ENTITY_SLOT_DATE_SENTINEL,
            "ring slot 1 (residue 1) next recurs at index 3, past the 3-stage horizon"
        );
    }

    /// A ring slot whose delivery stage lands past the horizon reads the
    /// sentinel: at the terminal stage (index 2, outgoing anchor `t_out = 3`),
    /// ring slot 0 (residue 0, the class matching `t_out`'s own predecessor)
    /// next recurs at index 4 and ring slot 1 (residue 1) at index 3 — both
    /// past the 3-stage horizon, so both read the sentinel. Anchoring on the
    /// entering `current_stage_idx = 2` instead would wrongly mature ring
    /// slot 0 in-study at index 2 (2024-06), the terminal maturing-residue
    /// pitfall [`build_stage_entity_manifest`]'s rustdoc names.
    #[test]
    fn anticipated_slot_delivery_anchor_past_horizon_is_sentinel() {
        let system = system_1h_1ant_3monthly(AnticipatedConfig::LeadStages(2));
        let global = test_support::state_layout_with_transit_buckets_and_resolution(
            1,
            1,
            Vec::new(),
            vec![2],
            AnticipatedResolution::resolve(
                &[LeadTime::Stages(2)],
                DeliveryAxis {
                    study_stage_hours: &[720.0; 3],
                    post_study_stage_hours: &[],
                },
            ),
        );
        let projection = CutStateProjection::new(&global, ALL_ENABLED);

        // stage_id 2 is the terminal stage (2024-06), study index 2.
        let manifest = build_stage_entity_manifest(
            &system,
            &global,
            &AnticipatedPlants::build(system.thermals()),
            &projection,
            2,
        );

        // Both ring slots' next occurrence (index 4 and index 3) lands past
        // the 3-stage horizon.
        assert_eq!(manifest[2].subindex, 0);
        assert_eq!(
            manifest[2].interval_start, ENTITY_SLOT_DATE_SENTINEL,
            "ring slot 0 next recurs at index 4, past the horizon"
        );
        assert_eq!(manifest[3].subindex, 1);
        assert_eq!(
            manifest[3].interval_start, ENTITY_SLOT_DATE_SENTINEL,
            "ring slot 1 next recurs at index 3, past the horizon"
        );
    }

    /// A `LeadTime`-mode anticipated plant resolves its ring slots exactly
    /// like `LeadStages` (compare
    /// `anticipated_slot_delivery_anchor_matches_delivery_stage_year_month`):
    /// lead mode does not change which anchor gates the ring, only
    /// reachability (`anticipated_lead_stages`) and horizon truncation do.
    #[test]
    fn anticipated_slot_leadtime_mode_yields_real_anchor() {
        let system = system_1h_1ant_3monthly(AnticipatedConfig::LeadTime(720.0));
        let global = test_support::state_layout_with_transit_buckets_and_resolution(
            1,
            1,
            Vec::new(),
            vec![2],
            AnticipatedResolution::resolve(
                &[LeadTime::Time(720.0)],
                DeliveryAxis {
                    study_stage_hours: &[720.0; 3],
                    post_study_stage_hours: &[],
                },
            ),
        );
        let projection = CutStateProjection::new(&global, ALL_ENABLED);

        let manifest = build_stage_entity_manifest(
            &system,
            &global,
            &AnticipatedPlants::build(system.thermals()),
            &projection,
            1,
        );

        assert_eq!(manifest[2].subindex, 0);
        assert_eq!(
            manifest[2].interval_start, 20240601,
            "ring slot 0 (residue 0) matures at the outgoing anchor itself (index 2, 2024-06)"
        );
        assert_eq!(manifest[3].subindex, 1);
        assert_eq!(
            manifest[3].interval_start, ENTITY_SLOT_DATE_SENTINEL,
            "ring slot 1 (residue 1) next recurs at index 3, past the 3-stage horizon"
        );
    }

    /// Leads `(1, 3)`, `k_max = 3`, 4 study stages, no post-study calendar:
    /// every anticipated slot's date must match the LP's own latch set
    /// (carry plus deposit), never the retired per-plant-lead-bounded rule,
    /// under which the ℓ=1 plant's slot 1 (a fresh deposit at stage 0) would
    /// read the sentinel instead of its real date.
    #[test]
    fn mixed_lead_manifest_dates_exactly_the_slots_the_lp_latches() {
        let system = system_1h_2ant_monthly(4);
        let global = test_support::state_layout_with_transit_buckets_and_resolution(
            1,
            1,
            Vec::new(),
            vec![1, 3],
            AnticipatedResolution::resolve(
                &[LeadTime::Stages(1), LeadTime::Stages(3)],
                DeliveryAxis {
                    study_stage_hours: &[720.0; 4],
                    post_study_stage_hours: &[],
                },
            ),
        );
        let projection = CutStateProjection::new(&global, ALL_ENABLED);

        // (stage_id, entity_id, subindex) -> expected interval_start; every
        // other anticipated slot must read the sentinel.
        let dated: [(i32, i32, u32, i32); 9] = [
            (0, 1, 1, 20240501),
            (0, 2, 0, 20240701),
            (0, 2, 1, 20240501),
            (0, 2, 2, 20240601),
            (1, 1, 2, 20240601),
            (1, 2, 0, 20240701),
            (1, 2, 2, 20240601),
            (2, 1, 0, 20240701),
            (2, 2, 0, 20240701),
        ];

        let anticipated_plants = AnticipatedPlants::build(system.thermals());
        for stage_id in 0..4i32 {
            let manifest = build_stage_entity_manifest(
                &system,
                &global,
                &anticipated_plants,
                &projection,
                stage_id,
            );
            for entity_id in [1, 2] {
                for subindex in 0..3u32 {
                    let expected = dated
                        .iter()
                        .find(|&&(s, e, i, _)| s == stage_id && e == entity_id && i == subindex)
                        .map_or(ENTITY_SLOT_DATE_SENTINEL, |&(_, _, _, date)| date);
                    let slot = manifest
                        .iter()
                        .find(|s| {
                            s.entity_type == StateFamily::AnticipatedThermalState.code()
                                && s.entity_id == entity_id
                                && s.subindex == subindex
                        })
                        .expect("every (entity_id, subindex) must own a manifest entry");
                    assert_eq!(
                        slot.interval_start, expected,
                        "stage {stage_id} entity {entity_id} slot {subindex}"
                    );
                }
            }
        }
    }

    /// A ring slot's liveness is derived from the LP's own reachability
    /// sweep ([`for_each_live_commitment_slot`]), never from the plant's own
    /// lead: with plants ℓ=1 and ℓ=2 sharing one `k_max=2` ring, at stage 0
    /// the ℓ=1 plant's only live residue is its own deposit (slot 1) — slot
    /// 0 stays sentinel even though it is inside the plant's nominal lead —
    /// while the ℓ=2 plant is live on both slots (its own deposit at slot 0,
    /// a carry at slot 1).
    #[test]
    fn anticipated_short_lead_slot_dates_the_residue_it_latches() {
        let system = system_1h_2ant_monthly(3);
        let global = test_support::state_layout_with_transit_buckets_and_resolution(
            1,
            1,
            Vec::new(),
            vec![1, 2],
            AnticipatedResolution::resolve(
                &[LeadTime::Stages(1), LeadTime::Stages(2)],
                DeliveryAxis {
                    study_stage_hours: &[720.0; 3],
                    post_study_stage_hours: &[],
                },
            ),
        );
        let projection = CutStateProjection::new(&global, ALL_ENABLED);

        let manifest = build_stage_entity_manifest(
            &system,
            &global,
            &AnticipatedPlants::build(system.thermals()),
            &projection,
            0,
        );

        // Layout N=1, L=1, A=2, k_max=2: storage j=0, lag j=1, anticipated j=2..6
        // (slot-major, plant-minor: [slot0,plant0][slot0,plant1][slot1,plant0][slot1,plant1]).
        assert_eq!(manifest[2].entity_id, 1, "slot0/plant0 = ℓ=1 plant");
        assert_eq!(manifest[2].subindex, 0);
        assert_eq!(
            manifest[2].interval_start, ENTITY_SLOT_DATE_SENTINEL,
            "ℓ=1 plant slot 0 is not latched by the LP at stage 0"
        );

        assert_eq!(manifest[3].entity_id, 2, "slot0/plant1 = ℓ=2 plant");
        assert_eq!(manifest[3].subindex, 0);
        assert_eq!(
            manifest[3].interval_start, 20240601,
            "ℓ=2 plant slot 0 is its own deposit, maturing at index 2 (2024-06)"
        );

        assert_eq!(manifest[4].entity_id, 1, "slot1/plant0 = ℓ=1 plant");
        assert_eq!(manifest[4].subindex, 1);
        assert_eq!(
            manifest[4].interval_start, 20240501,
            "ℓ=1 plant slot 1 is its own deposit, maturing at index 1 (2024-05)"
        );

        assert_eq!(manifest[5].entity_id, 2, "slot1/plant1 = ℓ=2 plant");
        assert_eq!(manifest[5].subindex, 1);
        assert_eq!(
            manifest[5].interval_start, 20240501,
            "ℓ=2 plant slot 1 is a carry, maturing at index 1 (2024-05)"
        );
    }

    /// A `LeadTime::Time` plant's ring slot whose target is inside the
    /// delivery window but not yet decided (`is_ready_at` false, a future
    /// `decider`) stays sentinel — distinct from horizon truncation
    /// (`anticipated_slot_leadtime_mode_yields_real_anchor` covers
    /// `target >= n_delivery`, filtered before `is_ready_at` ever runs).
    #[test]
    fn anticipated_leadtime_undecided_in_window_slot_stays_sentinel() {
        let system = system_1h_2ant_monthly(3);
        let resolution = AnticipatedResolution::resolve(
            &[LeadTime::Time(720.0), LeadTime::Stages(2)],
            DeliveryAxis {
                study_stage_hours: &[720.0; 3],
                post_study_stage_hours: &[],
            },
        );

        // Independent oracle over the Time-mode plant's own decider table
        // (never `for_each_live_commitment_slot`): target 2 sits inside the
        // delivery window but is decided only at stage 1, a future stage
        // relative to the queried stage 0; target 1 is a fresh deposit
        // exactly at stage 0.
        let point = &resolution.per_plant[0];
        assert_eq!(point.decider[2], Some(1));
        assert!(
            !point.is_ready_at(2, 0),
            "target 2 is not yet decided at stage 0"
        );
        assert_eq!(point.decider[1], Some(0));
        assert!(point.is_ready_at(1, 0), "target 1 is a deposit at stage 0");

        let global = test_support::state_layout_with_transit_buckets_and_resolution(
            1,
            1,
            Vec::new(),
            vec![1, 2],
            resolution,
        );
        let projection = CutStateProjection::new(&global, ALL_ENABLED);
        let manifest = build_stage_entity_manifest(
            &system,
            &global,
            &AnticipatedPlants::build(system.thermals()),
            &projection,
            0,
        );

        // Layout N=1, L=1, A=2, k_max=2: anticipated j=2..6, slot-major/plant-minor.
        assert_eq!(manifest[2].entity_id, 1, "slot0/plant0 targets delivery 2");
        assert_eq!(
            manifest[2].interval_start, ENTITY_SLOT_DATE_SENTINEL,
            "delivery 2 is undecided at stage 0, not past the horizon"
        );
        assert_eq!(manifest[4].entity_id, 1, "slot1/plant0 targets delivery 1");
        assert_eq!(
            manifest[4].interval_start, 20240501,
            "delivery 1 is the plant's own deposit at stage 0"
        );
    }

    // -- Extended delivery calendar: ring slots targeting post-study stages --

    /// With a post-study calendar declared, an in-study ring slot whose modular
    /// delivery target lands on a post-study stage carries that stage's real
    /// interval — not the sentinel — while a target past the extended
    /// calendar stays sentinel. Layout `N=1, L=1, A=1, k_max=3` at the
    /// terminal stage index 2 (outgoing anchor `t_out = 3`): ring slot 0
    /// delivers at `m=3` (the post-study stage 2024-07), slot 1 at `m=4`
    /// (past the extended calendar), and slot 2 — the residue matching the
    /// terminal stage's own class, the "maturing residue" — recurs at `m=5`,
    /// also past the 1-stage-deep extended calendar: at the terminal stage
    /// the outgoing anchor is itself already one past the horizon, so no
    /// residue can resolve in-study any more.
    #[test]
    fn ring_slot_targeting_post_study_carries_a_real_anchor() {
        let start = chrono::NaiveDate::from_ymd_opt(2024, 7, 1).unwrap();
        let system = system_1h_1ant_3monthly_lead3(post_study_stages_from(start));
        let global = test_support::state_layout_with_transit_buckets_and_resolution(
            1,
            1,
            Vec::new(),
            vec![3],
            AnticipatedResolution::resolve(
                &[LeadTime::Stages(3)],
                DeliveryAxis {
                    study_stage_hours: &[720.0; 3],
                    post_study_stage_hours: &[720.0; 1],
                },
            ),
        );
        let projection = CutStateProjection::new(&global, ALL_ENABLED);

        // Terminal stage index 2 (2024-06).
        let manifest = build_stage_entity_manifest(
            &system,
            &global,
            &AnticipatedPlants::build(system.thermals()),
            &projection,
            2,
        );

        // storage j=0, lag j=1, anticipated ring slots j=2,3,4 (slot-major).
        assert_eq!(manifest.len(), 5);
        assert_eq!(manifest[2].subindex, 0);
        assert_eq!(
            manifest[2].interval_start, 20240701,
            "ring slot 0 (m=3) targets the post-study stage 2024-07"
        );
        assert_eq!(manifest[3].subindex, 1);
        assert_eq!(
            manifest[3].interval_start, ENTITY_SLOT_DATE_SENTINEL,
            "ring slot 1 (m=4) lands past the extended calendar"
        );
        assert_eq!(manifest[4].subindex, 2);
        assert_eq!(
            manifest[4].interval_start, ENTITY_SLOT_DATE_SENTINEL,
            "ring slot 2 (m=5), the terminal maturing residue, lands past the extended calendar"
        );
    }

    // -- Manifest-carried intervals --

    /// A live anticipated slot's `interval_start`/`interval_end` are the
    /// resolved delivery stage's own `start_date`/`end_date` anchors, over
    /// the same fixture as
    /// [`anticipated_slot_delivery_anchor_matches_delivery_stage_year_month`].
    #[test]
    fn anticipated_slot_interval_matches_its_delivery_stage_span() {
        let system = system_1h_1ant_3monthly(AnticipatedConfig::LeadStages(2));
        let global = test_support::state_layout_with_transit_buckets_and_resolution(
            1,
            1,
            Vec::new(),
            vec![2],
            AnticipatedResolution::resolve(
                &[LeadTime::Stages(2)],
                DeliveryAxis {
                    study_stage_hours: &[720.0; 3],
                    post_study_stage_hours: &[],
                },
            ),
        );
        let projection = CutStateProjection::new(&global, ALL_ENABLED);

        // stage_id 1 is the middle stage (2024-05), study index 1; ring slot 0
        // (residue 0) matures at the outgoing anchor itself (index 2, 2024-06).
        let manifest = build_stage_entity_manifest(
            &system,
            &global,
            &AnticipatedPlants::build(system.thermals()),
            &projection,
            1,
        );

        assert_eq!(manifest[2].subindex, 0);
        assert_eq!(
            manifest[2].interval_start, 20240601,
            "interval_start is the delivery stage's own start_date anchor"
        );
        assert_eq!(
            manifest[2].interval_end, 20240701,
            "interval_end is the delivery stage's own exclusive end_date anchor"
        );
    }

    /// A non-month-aligned five-week (35-day) delivery stage's interval spans
    /// its own real 35 days, not the enclosing calendar month —
    /// `year_month_day_anchor`'s day-01 pin would collapse both endpoints to
    /// the first of their months instead.
    #[test]
    fn anticipated_slot_interval_on_a_five_week_stage_spans_thirty_five_days() {
        let system = system_1h_1ant_short_then_five_week_terminal();
        let global = test_support::state_layout_with_transit_buckets_and_resolution(
            1,
            1,
            Vec::new(),
            vec![1],
            AnticipatedResolution::resolve(
                &[LeadTime::Stages(1)],
                DeliveryAxis {
                    study_stage_hours: &[336.0, 840.0],
                    post_study_stage_hours: &[],
                },
            ),
        );
        let projection = CutStateProjection::new(&global, ALL_ENABLED);

        // stage_id 0's single ring slot matures at the outgoing anchor
        // (index 1), the five-week terminal stage.
        let manifest = build_stage_entity_manifest(
            &system,
            &global,
            &AnticipatedPlants::build(system.thermals()),
            &projection,
            0,
        );

        // Layout N=1, L=1, A=1, k_max=1: storage j=0, lag j=1, anticipated j=2.
        assert_eq!(manifest.len(), 3);
        assert_eq!(
            manifest[2].interval_start, 20260115,
            "interval_start is the terminal stage's real start day, not day-01"
        );
        assert_eq!(
            manifest[2].interval_end, 20260219,
            "interval_end is the terminal stage's real exclusive end day"
        );

        let start = chrono::NaiveDate::from_ymd_opt(2026, 1, 15).unwrap();
        let end = chrono::NaiveDate::from_ymd_opt(2026, 2, 19).unwrap();
        assert_eq!(
            (end - start).num_days(),
            35,
            "a non-month-aligned five-week stage's interval must span its real 35 days, not the \
             enclosing calendar month"
        );
    }

    /// Over a fixture whose ring reaches both in-study and post-study
    /// targets, every slot's `interval_start`/`interval_end` are either both
    /// live or both sentinel — `with_interval` always sets the pair together,
    /// and in-study deliveries carry a live interval too, not only
    /// post-study ones.
    #[test]
    fn anticipated_slot_dated_iff_intervalled() {
        let start = chrono::NaiveDate::from_ymd_opt(2024, 7, 1).unwrap();
        let system = system_1h_1ant_3monthly_lead3(post_study_stages_from(start));
        let global = test_support::state_layout_with_transit_buckets_and_resolution(
            1,
            1,
            Vec::new(),
            vec![3],
            AnticipatedResolution::resolve(
                &[LeadTime::Stages(3)],
                DeliveryAxis {
                    study_stage_hours: &[720.0; 3],
                    post_study_stage_hours: &[720.0; 1],
                },
            ),
        );
        let projection = CutStateProjection::new(&global, ALL_ENABLED);

        let mut saw_in_study_live = false;
        let mut saw_post_study_live = false;
        let anticipated_plants = AnticipatedPlants::build(system.thermals());
        for stage_id in 0..3i32 {
            let manifest = build_stage_entity_manifest(
                &system,
                &global,
                &anticipated_plants,
                &projection,
                stage_id,
            );
            for slot in &manifest {
                let start_live = slot.interval_start != ENTITY_SLOT_DATE_SENTINEL;
                let end_live = slot.interval_end != ENTITY_SLOT_DATE_SENTINEL;
                assert_eq!(
                    start_live, end_live,
                    "stage {stage_id} subindex {}: interval_start != SENTINEL must hold iff \
                     interval_end != SENTINEL",
                    slot.subindex
                );
                if start_live && slot.entity_type == StateFamily::AnticipatedThermalState.code() {
                    if slot.interval_start < 20_240_700 {
                        saw_in_study_live = true;
                    } else {
                        saw_post_study_live = true;
                    }
                }
            }
        }
        assert!(
            saw_in_study_live,
            "fixture must exercise a live in-study delivery"
        );
        assert!(
            saw_post_study_live,
            "fixture must exercise a live post-study delivery"
        );
    }

    // -- Re-anchoring onto the outgoing state: terminal maturing residue --

    /// The terminal pool's maturing residue — the ring slot whose subindex
    /// matches the terminal stage's own class — re-anchors onto its real
    /// post-study delivery instead of the in-study terminal month: over 64
    /// monthly stages ending 2031-11-01 with `lead_stages = 2` and a 2-stage
    /// post-study calendar, the terminal pool's subindex-1 slot dates onto
    /// 2032-01-01, not `20311101`.
    #[test]
    fn terminal_maturing_residue_dates_onto_its_post_study_delivery() {
        let post_study = post_study_stages_from_months(&[(2031, 12), (2032, 1)]);
        let system = system_1h_1ant_64monthly_lead2(Some(post_study));
        let global = test_support::state_layout_with_transit_buckets_and_resolution(
            1,
            1,
            Vec::new(),
            vec![2],
            AnticipatedResolution::resolve(
                &[LeadTime::Stages(2)],
                DeliveryAxis {
                    study_stage_hours: &[720.0; 64],
                    post_study_stage_hours: &[720.0; 2],
                },
            ),
        );
        let projection = CutStateProjection::new(&global, ALL_ENABLED);

        // Terminal stage id 63 (2031-11), study index 63.
        let manifest = build_stage_entity_manifest(
            &system,
            &global,
            &AnticipatedPlants::build(system.thermals()),
            &projection,
            63,
        );

        // storage j=0, lag j=1, anticipated ring slots j=2 (subindex 0), j=3 (subindex 1).
        assert_eq!(manifest.len(), 4);
        assert_eq!(manifest[2].subindex, 0);
        assert_eq!(
            manifest[2].interval_start, 20311201,
            "ring slot 0 matures at the outgoing anchor itself (2031-12)"
        );

        assert_eq!(manifest[3].subindex, 1);
        assert_ne!(
            manifest[3].interval_start, 20311101,
            "the maturing residue must not date onto the in-study terminal month"
        );
        assert_eq!(
            manifest[3].interval_start, 20320101,
            "the maturing residue re-anchors a full k_max stages past the horizon (2032-01)"
        );
    }

    /// Without a declared post-study calendar, the terminal pool's maturing
    /// residue reads the sentinel rather than the in-study terminal month:
    /// its re-anchored target has no stage in the un-extended calendar.
    #[test]
    fn terminal_maturing_residue_stays_sentinel_without_a_post_study_calendar() {
        let system = system_1h_1ant_64monthly_lead2(None);
        let global = test_support::state_layout_with_transit_buckets_and_resolution(
            1,
            1,
            Vec::new(),
            vec![2],
            AnticipatedResolution::resolve(
                &[LeadTime::Stages(2)],
                DeliveryAxis {
                    study_stage_hours: &[720.0; 64],
                    post_study_stage_hours: &[],
                },
            ),
        );
        let projection = CutStateProjection::new(&global, ALL_ENABLED);

        let manifest = build_stage_entity_manifest(
            &system,
            &global,
            &AnticipatedPlants::build(system.thermals()),
            &projection,
            63,
        );

        assert_eq!(manifest[3].subindex, 1);
        assert_eq!(
            manifest[3].interval_start, ENTITY_SLOT_DATE_SENTINEL,
            "the maturing residue's re-anchored target has no stage in the un-extended calendar"
        );
    }

    // -- General correspondence and reachability invariance --

    /// For every `(stage_id, slot_idx)` combination over a fixture whose ring
    /// reaches both in-study and post-study targets, a live anticipated
    /// slot's `interval_start` equals `encode_slot_date` of the `start_date`
    /// of the unique delivery `m` the LP itself would latch into that slot at
    /// that stage — an oracle independent of the code under test: the unique
    /// `m > t` with `m < n_delivery`, `resolution.is_ready_at(m, t)`, and
    /// `global.commitment_hold_in_study_offset(0, m)` landing on `slot_idx`,
    /// or the sentinel when no such `m` exists.
    #[test]
    fn anticipated_slot_date_matches_the_resolved_physical_delivery_stage() {
        let start = chrono::NaiveDate::from_ymd_opt(2024, 7, 1).unwrap();
        let system = system_1h_1ant_3monthly_lead3(post_study_stages_from(start));
        let global = test_support::state_layout_with_transit_buckets_and_resolution(
            1,
            1,
            Vec::new(),
            vec![3],
            AnticipatedResolution::resolve(
                &[LeadTime::Stages(3)],
                DeliveryAxis {
                    study_stage_hours: &[720.0; 3],
                    post_study_stage_hours: &[],
                },
            ),
        );
        let projection = CutStateProjection::new(&global, ALL_ENABLED);
        let study_stages: Vec<&Stage> = system.stages().iter().filter(|s| s.id >= 0).collect();
        let post_study_calendar = post_study_delivery_calendar(&system);
        let delivery_stages = extended_delivery_stages(&study_stages, &post_study_calendar);
        let n_delivery = global.n_delivery();
        let resolution = &global.anticipated_resolution.per_plant[0];

        let anticipated_plants = AnticipatedPlants::build(system.thermals());
        for stage_id in 0..3i32 {
            let manifest = build_stage_entity_manifest(
                &system,
                &global,
                &anticipated_plants,
                &projection,
                stage_id,
            );
            let current_stage_idx = study_stages
                .iter()
                .position(|s| s.id == stage_id)
                .expect("stage_id must resolve to a study stage");

            for slot_idx in 0..global.k_max {
                let slot = manifest
                    .iter()
                    .find(|s| {
                        s.entity_type == StateFamily::AnticipatedThermalState.code()
                            && s.subindex == slot_idx as u32
                    })
                    .expect("every ring slot_idx must own a manifest entry");
                let expected = (current_stage_idx + 1..n_delivery)
                    .find(|&m| {
                        resolution.is_ready_at(m, current_stage_idx)
                            && global.commitment_hold_in_study_offset(0, m) == slot_idx
                    })
                    .map_or(ENTITY_SLOT_DATE_SENTINEL, |m| {
                        encode_slot_date(delivery_stages[m].start_date)
                    });
                assert_eq!(
                    slot.interval_start, expected,
                    "stage {stage_id} slot {slot_idx} must match the independent latch-set oracle"
                );
            }
        }
    }

    /// A slot the LP does not latch at a given stage stays sentinel-dated at
    /// that stage — and only that stage: for the ℓ=1 plant, slot 0 is
    /// sentinel at stage 0 (not yet decided) but dated at stage 1 (its own
    /// deposit), and slot 1 is dated at stage 0 (its own deposit) but
    /// sentinel from stage 1 on (past `n_delivery`) — never a fixed
    /// `slot_idx >= k_i` bound independent of the stage.
    #[test]
    fn anticipated_slots_the_lp_does_not_latch_stay_sentinel() {
        let system = system_1h_2ant_monthly(3);
        let global = test_support::state_layout_with_transit_buckets_and_resolution(
            1,
            1,
            Vec::new(),
            vec![1, 2],
            AnticipatedResolution::resolve(
                &[LeadTime::Stages(1), LeadTime::Stages(2)],
                DeliveryAxis {
                    study_stage_hours: &[720.0; 3],
                    post_study_stage_hours: &[],
                },
            ),
        );
        let projection = CutStateProjection::new(&global, ALL_ENABLED);

        // (stage_id, subindex) -> expected interval_start for the ℓ=1 plant
        // (entity id 1); every other combination stays sentinel.
        let dated: [(i32, u32, i32); 2] = [(0, 1, 20240501), (1, 0, 20240601)];

        let anticipated_plants = AnticipatedPlants::build(system.thermals());
        for stage_id in 0..3i32 {
            let manifest = build_stage_entity_manifest(
                &system,
                &global,
                &anticipated_plants,
                &projection,
                stage_id,
            );
            for subindex in 0..2u32 {
                let expected = dated
                    .iter()
                    .find(|&&(s, i, _)| s == stage_id && i == subindex)
                    .map_or(ENTITY_SLOT_DATE_SENTINEL, |&(_, _, date)| date);
                let slot = manifest
                    .iter()
                    .find(|s| {
                        s.entity_type == StateFamily::AnticipatedThermalState.code()
                            && s.entity_id == 1
                            && s.subindex == subindex
                    })
                    .expect("the ℓ=1 plant must own this subindex slot");
                assert_eq!(
                    slot.interval_start, expected,
                    "stage {stage_id} slot {subindex}"
                );
            }
        }
    }

    // -- Ring-axis excision: dating maps through `physical_target` --

    /// With a fixed post-horizon window (a `LeadStages(7)` lead over
    /// `n_decision = 4` study stages derives a resolved ring depth `k_max =
    /// 4`, excising ring-axis targets `4..7`), the terminal-stage ring slots
    /// whose ring-axis residue search lands in the excised window date at
    /// their REAL physical post-study stage — never the excised stub the raw
    /// ring-axis index alone would name.
    #[test]
    fn date_ring_slots_in_excised_space_maps_through_physical_target() {
        let post_study = post_study_stages_from_months(&[
            (2024, 5),
            (2024, 6),
            (2024, 7),
            (2024, 8),
            (2024, 9),
            (2024, 10),
        ]);
        let system = system_1h_1ant_4monthly_lead7(post_study);

        let resolution = AnticipatedResolution::resolve(
            &[LeadTime::Stages(7)],
            DeliveryAxis {
                study_stage_hours: &[720.0; 10][..4],
                post_study_stage_hours: &[720.0; 10][4..],
            },
        );
        assert_eq!(
            resolution.anchored_depth(),
            4,
            "a lead-7 plant over 4 study stages must derive ring depth k_max=4"
        );
        let point = &resolution.per_plant[0];
        assert_eq!(
            point.ring_index(4),
            None,
            "m=4 sits inside the excised window"
        );
        assert_eq!(
            point.ring_index(5),
            None,
            "m=5 sits inside the excised window"
        );
        assert_eq!(
            point.ring_index(6),
            None,
            "m=6 sits inside the excised window"
        );
        assert_eq!(
            point.physical_target(4),
            7,
            "ring index 4 maps past the window to m=7"
        );
        assert_eq!(point.physical_target(5), 8);
        assert_eq!(point.physical_target(6), 9);

        let k_max = resolution.anchored_depth();
        let global = test_support::state_layout_with_transit_buckets_and_resolution(
            1,
            1,
            Vec::new(),
            vec![k_max],
            resolution,
        );
        let projection = CutStateProjection::new(&global, ALL_ENABLED);

        // Terminal study stage (index 3, 2024-04).
        let manifest = build_stage_entity_manifest(
            &system,
            &global,
            &AnticipatedPlants::build(system.thermals()),
            &projection,
            3,
        );
        let study_stages: Vec<&Stage> = system.stages().iter().filter(|s| s.id >= 0).collect();
        let post_study_calendar = post_study_delivery_calendar(&system);
        let delivery_stages = extended_delivery_stages(&study_stages, &post_study_calendar);

        // Layout N=1, L=1, A=1, k_max=4: storage j=0, lag j=1, anticipated j=2..6.
        assert_eq!(manifest.len(), 6);

        assert_eq!(manifest[2].subindex, 0);
        assert_eq!(
            manifest[2].interval_start,
            encode_slot_date(delivery_stages[7].start_date),
            "ring slot 0 (ring index 4) dates at the real physical target m=7 (2024-08), not \
             the excised 2024-05 stub"
        );
        assert_eq!(
            manifest[2].interval_end,
            encode_slot_date(delivery_stages[7].end_date)
        );

        assert_eq!(manifest[3].subindex, 1);
        assert_eq!(
            manifest[3].interval_start,
            encode_slot_date(delivery_stages[8].start_date),
            "ring slot 1 (ring index 5) dates at the real physical target m=8 (2024-09), not \
             the excised 2024-06 stub"
        );
        assert_eq!(
            manifest[3].interval_end,
            encode_slot_date(delivery_stages[8].end_date)
        );

        assert_eq!(manifest[4].subindex, 2);
        assert_eq!(
            manifest[4].interval_start,
            encode_slot_date(delivery_stages[9].start_date),
            "ring slot 2 (ring index 6) dates at the real physical target m=9 (2024-10), not \
             the excised 2024-07 stub"
        );
        assert_eq!(
            manifest[4].interval_end,
            encode_slot_date(delivery_stages[9].end_date)
        );

        // Ring slot 3's ring-axis target r=7 (slot_idx 3 at outgoing anchor
        // t_out=4, k_max=4) sits at n_decision+g=7, past the excised window's
        // far edge, so physical_target shifts it to m=10 — one past the
        // 10-stage extended calendar (n_delivery=10) — sentinel, not the
        // in-study terminal anchor a raw ring-axis reading might suggest.
        assert_eq!(manifest[5].subindex, 3);
        assert_eq!(
            manifest[5].interval_start, ENTITY_SLOT_DATE_SENTINEL,
            "ring slot 3 (ring index 7) maps past the extended calendar (m=10 >= n_delivery=10)"
        );
    }

    // -- build_stage_states_payloads: node-vs-pool manifest indexing --

    /// A 7-node binary tree mirroring `setup::tests::
    /// node_native_binary_tree_loads_and_constructs_node_graph`'s fixture:
    /// nodes 0/1/2 are internal and each own their own pool (0/1/2); leaves
    /// 3/4/5/6 share pool 3. `n_pools == 4 < nodes.len() == 7`.
    fn binary_tree_node_graph() -> NodeGraph {
        let opening = NodeOpenings {
            source: OpeningSource::Generated,
            offset: 0,
            len: 1,
            q: 1.0,
        };
        let node = |stage: usize, pool_id: usize| NodeRuntime {
            stage: StageIdx(stage),
            pool_id,
            openings: opening,
        };
        let edge = |child: usize| NodeSuccessor {
            child: NodePos(child),
            probability: 0.5,
        };
        NodeGraph {
            node_ids: vec![
                NodeId(0),
                NodeId(1),
                NodeId(2),
                NodeId(3),
                NodeId(4),
                NodeId(5),
                NodeId(6),
            ]
            .into(),
            nodes: vec![
                node(0, 0),
                node(1, 1),
                node(1, 2),
                node(2, 3),
                node(2, 3),
                node(2, 3),
                node(2, 3),
            ]
            .into(),
            successors: vec![
                vec![edge(1), edge(2)],
                vec![edge(3), edge(4)],
                vec![edge(5), edge(6)],
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
            ]
            .into(),
            n_pools: 4,
            pool_stage: vec![StageIdx(0), StageIdx(1), StageIdx(1), StageIdx(2)],
        }
    }

    /// A degenerate one-node-per-stage chain: `pool_stage[t] == StageIdx(t)`.
    fn chain_node_graph(n_stages: usize) -> NodeGraph {
        let opening = NodeOpenings {
            source: OpeningSource::Generated,
            offset: 0,
            len: 1,
            q: 1.0,
        };
        NodeGraph {
            node_ids: (0..n_stages as i32).map(NodeId).collect(),
            nodes: (0..n_stages)
                .map(|t| NodeRuntime {
                    stage: StageIdx(t),
                    pool_id: t,
                    openings: opening,
                })
                .collect(),
            successors: (0..n_stages)
                .map(|t| {
                    if t + 1 < n_stages {
                        vec![NodeSuccessor {
                            child: NodePos(t + 1),
                            probability: 1.0,
                        }]
                    } else {
                        Vec::new()
                    }
                })
                .collect(),
            n_pools: n_stages,
            pool_stage: (0..n_stages).map(StageIdx).collect(),
        }
    }

    /// A one-slot manifest tagging `pool_id` into `entity_id`, so a payload's
    /// manifest can be traced back to the pool it came from.
    fn manifest_for_pool(pool_id: usize) -> Vec<EntitySlot> {
        vec![EntitySlot::storage(
            100 + i32::try_from(pool_id).unwrap(),
            true,
        )]
    }

    /// On a branching graph, `archive.num_nodes() (7) > stage_manifests.len()
    /// (4)`: indexing the manifest directly by node position `t` is wrong — it
    /// panics out of bounds the moment `t >= 4`. Indexing through
    /// `node_graph.nodes[t].pool_id` stays in range and resolves every one of
    /// the 4 leaves (t = 3..=6) to the SAME shared pool-3 manifest.
    ///
    /// Also pins U13: `stage_id` carries `node_graph.nodes[t].stage` (the
    /// binary tree's stage 2 for every leaf, not the leaves' distinct node
    /// positions 3..=6) and `node_id` carries `node_graph.node_ids[t]`.
    #[test]
    fn tree_states_payload_resolves_manifest_through_node_pool_id() {
        let node_graph = binary_tree_node_graph();
        let stage_manifests: Vec<Vec<EntitySlot>> =
            (0..node_graph.n_pools).map(manifest_for_pool).collect();

        let mut archive = VisitedStatesArchive::new(node_graph.nodes.len(), 1, 1, 1);
        for t in 0..node_graph.nodes.len() {
            archive.archive_gathered_states(NodePos(t), &[0.0], 1);
        }

        let payloads = build_stage_states_payloads(Some(&archive), &stage_manifests, &node_graph);

        assert_eq!(payloads.len(), 7, "one payload per node, not per pool");
        for (t, payload) in payloads.iter().enumerate() {
            let pos = NodePos(t);
            let expected_pool = node_graph.nodes[pos].pool_id;
            assert_eq!(
                payload.entity_manifest.len(),
                1,
                "node {t} manifest must not be empty"
            );
            assert_eq!(
                payload.entity_manifest[0].entity_id,
                100 + i32::try_from(expected_pool).unwrap(),
                "node {t} must carry pool {expected_pool}'s manifest, not stage_manifests[{t}]"
            );
            assert_eq!(
                payload.stage_id, node_graph.nodes[pos].stage.0 as u32,
                "node {t}'s stage_id must be its STUDY STAGE, not its node position"
            );
            assert_eq!(
                payload.node_id, node_graph.node_ids[pos].0,
                "node {t}'s node_id must be its declared node id"
            );
        }

        // The 4 leaves all resolve to the one shared pool-3 manifest and the
        // one shared terminal stage, despite carrying 4 distinct node ids.
        for (t, payload) in payloads[3..=6].iter().enumerate() {
            assert_eq!(payload.entity_manifest[0].entity_id, 103);
            assert_eq!(payload.stage_id, 2, "every leaf sits at stage 2");
            assert_eq!(
                payload.node_id,
                i32::try_from(3 + t).unwrap(),
                "leaves keep their own distinct node ids"
            );
        }
    }

    /// On a branching (K-fan) graph, `build_stage_basis_records` is node-indexed
    /// (one record per `basis_cache` node slot, `stage_id == node ordinal`) and
    /// resolves each record's `num_cut_rows` through the node's OWN pool
    /// (`node_graph.nodes[node].pool_id`), never `fcf.pools[node_ordinal]` —
    /// which for leaves 4/5/6 (ordinal > `n_pools - 1`) reads no pool at all.
    /// The 4 leaves all report the shared pool-3 count.
    #[test]
    fn build_stage_basis_records_resolves_num_cut_rows_via_node_pool() {
        use super::{build_stage_basis_records, convert_basis_cache};
        use crate::TrainingResult;
        use crate::cut::FutureCostFunction;
        use crate::workspace::CapturedBasis;
        use cobre_solver::Basis;

        let node_graph = binary_tree_node_graph(); // 7 nodes; pools 0/1/2 + shared 3

        // 4 pools; pool 3 (shared leaf pool) gets 2 cuts, pools 0/1/2 get 1 each.
        let mut fcf = FutureCostFunction::new(4, 1, 1, 10, &[0; 4]);
        fcf.add_cut(NodeId(0), 0, 0, 0, 0.0, &[0.0]);
        fcf.add_cut(NodeId(0), 1, 0, 0, 0.0, &[0.0]);
        fcf.add_cut(NodeId(0), 2, 0, 0, 0.0, &[0.0]);
        fcf.add_cut(NodeId(0), 3, 0, 0, 0.0, &[0.0]);
        fcf.add_cut(NodeId(0), 3, 1, 0, 0.0, &[0.0]);

        let empty_basis = || CapturedBasis {
            basis: Basis {
                col_status: Vec::new(),
                row_status: Vec::new(),
            },
            base_row_count: 0,
            cut_row_slots: Vec::new(),
            state_at_capture: Vec::new(),
            node_id: NodeId(0),
        };
        let basis_cache: Vec<Option<CapturedBasis>> = (0..7).map(|_| Some(empty_basis())).collect();
        let training_result = TrainingResult::new(
            0.0,
            0.0,
            0.0,
            0.0,
            1,
            "t".to_string(),
            0,
            basis_cache,
            Vec::new(),
            None,
            None,
        );

        let (col, row) = convert_basis_cache(&training_result);
        let records = build_stage_basis_records(&fcf, &training_result, &node_graph, &col, &row);

        assert_eq!(records.len(), 7, "one basis record per node, not per pool");
        for (node, rec) in records.iter().enumerate() {
            assert_eq!(
                rec.stage_id as usize, node,
                "stage_id must carry the node ordinal"
            );
            let pool = node_graph.nodes[NodePos(node)].pool_id;
            let expected = if pool == 3 { 2 } else { 1 };
            assert_eq!(
                rec.num_cut_rows, expected,
                "node {node} num_cut_rows must reflect its own pool {pool}, not fcf.pools[{node}]"
            );
        }
    }

    // -- build_stage_cuts_payloads: priced_state_date stamping --

    #[test]
    fn stage_cuts_payload_stamps_owning_stage_end_date() {
        use super::{build_active_indices, build_stage_cut_records, build_stage_cuts_payloads};
        use crate::cut::FutureCostFunction;

        let node_graph = chain_node_graph(3);
        let fcf = FutureCostFunction::new(3, 1, 1, 10, &[0; 3]);
        let stage_records = build_stage_cut_records(&fcf);
        let stage_active_indices = build_active_indices(&stage_records);
        let stage_manifests: Vec<Vec<EntitySlot>> = vec![Vec::new(); 3];
        let study_stage_ids = vec![0, 1, 2];
        let study_stage_end_dates = vec![
            chrono::NaiveDate::from_ymd_opt(2031, 10, 1).unwrap(),
            chrono::NaiveDate::from_ymd_opt(2031, 12, 1).unwrap(),
            chrono::NaiveDate::from_ymd_opt(2032, 1, 1).unwrap(),
        ];

        let payloads = build_stage_cuts_payloads(
            &fcf,
            &node_graph,
            &study_stage_ids,
            &study_stage_end_dates,
            1_000_000.0,
            &stage_records,
            &stage_active_indices,
            &stage_manifests,
        );

        assert_eq!(payloads[0].priced_state_date, 20_311_001);
        assert_eq!(
            payloads[1].priced_state_date, 20_311_201,
            "pool 1's priced_state_date must be its owning stage's exclusive end_date, \
             YYYYMMDD-encoded"
        );
        assert_eq!(payloads[2].priced_state_date, 20_320_101);
    }

    /// On a branching graph, pool 2's own ordinal is stage-shaped `2`, but its
    /// OWNING stage is 1 (`pool_stage[2] == StageIdx(1)`, shared with pool 1's
    /// owner). Reading `study_stage_end_dates[2]` instead of
    /// `study_stage_end_dates[pool_stage[2]]` is the forbidden pool-ordinal path
    /// this test distinguishes.
    #[test]
    fn stage_cuts_payload_priced_date_resolves_through_pool_stage_on_branching_graph() {
        use super::{build_active_indices, build_stage_cut_records, build_stage_cuts_payloads};
        use crate::cut::FutureCostFunction;

        let node_graph = binary_tree_node_graph(); // 4 pools; pool_stage = [0, 1, 1, 2]
        let fcf = FutureCostFunction::new(4, 1, 1, 10, &[0; 4]);
        let stage_records = build_stage_cut_records(&fcf);
        let stage_active_indices = build_active_indices(&stage_records);
        let stage_manifests: Vec<Vec<EntitySlot>> = vec![Vec::new(); 4];
        let study_stage_ids = vec![0, 1, 2];
        let study_stage_end_dates = vec![
            chrono::NaiveDate::from_ymd_opt(2030, 1, 1).unwrap(),
            chrono::NaiveDate::from_ymd_opt(2030, 2, 1).unwrap(),
            chrono::NaiveDate::from_ymd_opt(2030, 3, 1).unwrap(),
        ];

        let payloads = build_stage_cuts_payloads(
            &fcf,
            &node_graph,
            &study_stage_ids,
            &study_stage_end_dates,
            1_000_000.0,
            &stage_records,
            &stage_active_indices,
            &stage_manifests,
        );

        assert_eq!(
            payloads.len(),
            4,
            "one payload per pool; n_pools (4) > n_stages (3)"
        );
        assert_eq!(
            payloads[0].priced_state_date, 20_300_101,
            "pool 0 owns stage 0"
        );
        assert_eq!(
            payloads[1].priced_state_date, 20_300_201,
            "pool 1 owns stage 1"
        );
        assert_eq!(
            payloads[2].priced_state_date, 20_300_201,
            "pool 2 owns stage 1, not stage 2 — its own pool ordinal must not be read"
        );
        assert_eq!(
            payloads[3].priced_state_date, 20_300_301,
            "the shared leaf pool 3 owns the terminal stage 2"
        );
    }

    #[test]
    fn stage_cuts_payload_unresolvable_pool_stage_keeps_both_sentinels() {
        use super::{
            STAGE_CUTS_GRAPH_STAGE_ID_SENTINEL, STAGE_CUTS_PRICED_STATE_DATE_SENTINEL,
            build_active_indices, build_stage_cut_records, build_stage_cuts_payloads,
        };
        use crate::cut::FutureCostFunction;

        let node_graph = NodeGraph {
            node_ids: vec![NodeId(0)].into(),
            nodes: vec![NodeRuntime {
                stage: StageIdx(5),
                pool_id: 0,
                openings: NodeOpenings {
                    source: OpeningSource::Generated,
                    offset: 0,
                    len: 1,
                    q: 1.0,
                },
            }]
            .into(),
            successors: vec![Vec::new()].into(),
            n_pools: 1,
            pool_stage: vec![StageIdx(5)],
        };
        let fcf = FutureCostFunction::new(1, 1, 1, 10, &[0]);
        let stage_records = build_stage_cut_records(&fcf);
        let stage_active_indices = build_active_indices(&stage_records);
        let stage_manifests: Vec<Vec<EntitySlot>> = vec![Vec::new(); 1];
        let study_stage_ids = vec![0, 1]; // len 2; pool_stage[0] == 5 is out of range
        let study_stage_end_dates = vec![
            chrono::NaiveDate::from_ymd_opt(2030, 1, 1).unwrap(),
            chrono::NaiveDate::from_ymd_opt(2030, 2, 1).unwrap(),
        ];

        let payloads = build_stage_cuts_payloads(
            &fcf,
            &node_graph,
            &study_stage_ids,
            &study_stage_end_dates,
            1_000_000.0,
            &stage_records,
            &stage_active_indices,
            &stage_manifests,
        );

        assert_eq!(
            payloads[0].graph_stage_id, STAGE_CUTS_GRAPH_STAGE_ID_SENTINEL,
            "an out-of-range pool_stage must fall back to the graph_stage_id sentinel"
        );
        assert_eq!(
            payloads[0].priced_state_date, STAGE_CUTS_PRICED_STATE_DATE_SENTINEL,
            "the two sentinel fallbacks must agree: an out-of-range pool_stage keeps \
             priced_state_date at its sentinel too"
        );
    }

    /// A weekly stage's exclusive `end_date` (2031-11-10, not month-aligned)
    /// must stamp `priced_state_date` at its own day — `year_month_day_anchor`
    /// would truncate it to `20311101`, colliding it with a sibling weekly
    /// pool ending earlier in the same month.
    #[test]
    fn stage_cuts_payload_priced_state_date_keeps_the_stage_end_day() {
        use super::{build_active_indices, build_stage_cut_records, build_stage_cuts_payloads};
        use crate::cut::FutureCostFunction;

        let node_graph = chain_node_graph(1);
        let fcf = FutureCostFunction::new(1, 1, 1, 10, &[0; 1]);
        let stage_records = build_stage_cut_records(&fcf);
        let stage_active_indices = build_active_indices(&stage_records);
        let stage_manifests: Vec<Vec<EntitySlot>> = vec![Vec::new(); 1];
        let study_stage_ids = vec![0];
        let study_stage_end_dates = vec![chrono::NaiveDate::from_ymd_opt(2031, 11, 10).unwrap()];

        let payloads = build_stage_cuts_payloads(
            &fcf,
            &node_graph,
            &study_stage_ids,
            &study_stage_end_dates,
            1_000_000.0,
            &stage_records,
            &stage_active_indices,
            &stage_manifests,
        );

        assert_eq!(
            payloads[0].priced_state_date, 20_311_110,
            "a weekly stage's exclusive end_date must keep its own day, not truncate to day 01"
        );
    }

    #[test]
    fn year_month_day_anchor_same_month_dates_are_equal() {
        let weekly_stage_start = chrono::NaiveDate::from_ymd_opt(2026, 9, 5).unwrap();
        let monthly_stage_start = chrono::NaiveDate::from_ymd_opt(2026, 9, 1).unwrap();

        assert_eq!(year_month_day_anchor(weekly_stage_start), 20260901);
        assert_eq!(year_month_day_anchor(monthly_stage_start), 20260901);
    }

    #[test]
    fn year_month_day_anchor_always_normalizes_to_day_01() {
        let dates = [
            chrono::NaiveDate::from_ymd_opt(2024, 1, 15).unwrap(),
            chrono::NaiveDate::from_ymd_opt(2025, 6, 30).unwrap(),
            chrono::NaiveDate::from_ymd_opt(2026, 12, 1).unwrap(),
            chrono::NaiveDate::from_ymd_opt(2027, 2, 28).unwrap(),
        ];

        for date in dates {
            assert_eq!(year_month_day_anchor(date) % 100, 1);
        }
    }

    // ── reserve_boundary_inflow_lag_slots ────────────────────────────────────

    /// A depth-2 reservation over a 2-storage manifest inserts the lag block in
    /// canonical lag-major / 1-based-subindex order immediately after storage,
    /// preserving the trailing (anticipated) slots, and places each keyed
    /// `pi_qafl` at its `(hydro, depth)` position with `0.0` elsewhere.
    #[test]
    fn reserve_inserts_canonical_lag_block_after_storage_and_places_coefficients() {
        // storage(h1), storage(h2), anticipated(t9)
        let manifest = vec![
            EntitySlot::storage(1, true),
            EntitySlot::storage(2, false),
            anticipated_slot(9, 0),
        ];
        // One cut: storage coeffs [s1, s2, a9]; lag coeffs keyed by hydro.
        let cut_coefficients = vec![vec![10.0, 20.0, 99.0]];
        let mut keyed: HashMap<i32, Vec<f64>> = HashMap::new();
        keyed.insert(1, vec![1.1, 1.2]); // hydro 1: depth1=1.1, depth2=1.2
        keyed.insert(2, vec![2.1]); // hydro 2: depth1=2.1, depth2 defaults 0.0
        let cut_lag = vec![keyed];

        let out = reserve_boundary_inflow_lag_slots(&manifest, &cut_coefficients, &cut_lag, 2)
            .expect("reservation succeeds");

        // Manifest: 2 storage + (2 hydros × 2 depths) lag + 1 anticipated = 7.
        assert_eq!(out.state_dimension, 7);
        assert_eq!(out.manifest.len(), 7);
        assert_eq!(
            out.manifest[0].entity_type,
            StateFamily::HydroStorage.code()
        );
        assert_eq!(
            out.manifest[1].entity_type,
            StateFamily::HydroStorage.code()
        );

        // Lag block, lag-major: (h1,d1),(h2,d1),(h1,d2),(h2,d2); subindex = depth.
        let expected_lag = [(1, 1u32, true), (2, 1, false), (1, 2, true), (2, 2, false)];
        for (i, (id, subindex, active)) in expected_lag.into_iter().enumerate() {
            let slot = &out.manifest[2 + i];
            assert_eq!(slot.entity_type, StateFamily::HydroInflowLag.code());
            assert_eq!(slot.entity_id, id, "lag slot {i} hydro id");
            assert_eq!(slot.subindex, subindex, "lag slot {i} 1-based depth");
            assert_eq!(
                slot.was_active, active,
                "lag slot {i} inherits storage was_active"
            );
            assert_eq!(slot.interval_start, ENTITY_SLOT_DATE_SENTINEL);
        }

        // Trailing anticipated slot survives unchanged, now at index 6.
        assert_eq!(
            out.manifest[6].entity_type,
            StateFamily::AnticipatedThermalState.code()
        );
        assert_eq!(out.manifest[6].entity_id, 9);

        // Coefficients: storage[..2] ++ lag_block ++ tail; lag block in the same
        // lag-major order: (h1,d1)=1.1,(h2,d1)=2.1,(h1,d2)=1.2,(h2,d2)=0.0.
        assert_eq!(out.coefficients.len(), 1);
        assert_eq!(
            out.coefficients[0],
            vec![10.0, 20.0, 1.1, 2.1, 1.2, 0.0, 99.0]
        );
    }

    /// The written lag slots' `(entity_type, entity_id, subindex)` identity is
    /// exactly what `boundary_cut_lag_depth`/reconciliation match on: the
    /// deepest `HydroInflowLag` subindex equals the declared depth.
    #[test]
    fn reserve_self_describes_the_declared_depth() {
        let manifest = vec![EntitySlot::storage(1, true), EntitySlot::storage(2, true)];
        let cut_coefficients = vec![vec![1.0, 2.0]];
        let cut_lag = vec![HashMap::new()];

        let out = reserve_boundary_inflow_lag_slots(&manifest, &cut_coefficients, &cut_lag, 12)
            .expect("reservation succeeds");

        let max_lag_subindex = out
            .manifest
            .iter()
            .filter(|s| s.entity_type == StateFamily::HydroInflowLag.code())
            .map(|s| s.subindex)
            .max()
            .expect("lag slots exist");
        assert_eq!(max_lag_subindex, 12);
        // 2 storage + 2 hydros × 12 depths = 26.
        assert_eq!(out.state_dimension, 26);
    }

    /// A keyed `pi_qafl` for a hydro with no storage slot is unplaceable — the
    /// write fails rather than silently dropping the term.
    #[test]
    fn reserve_rejects_coefficient_for_unknown_hydro() {
        let manifest = vec![EntitySlot::storage(1, true)];
        let cut_coefficients = vec![vec![1.0]];
        let mut keyed: HashMap<i32, Vec<f64>> = HashMap::new();
        keyed.insert(7, vec![0.5]); // hydro 7 has no storage slot
        let cut_lag = vec![keyed];

        let err = reserve_boundary_inflow_lag_slots(&manifest, &cut_coefficients, &cut_lag, 2)
            .expect_err("unplaceable term must reject");
        assert!(
            format!("{err}").contains("hydro 7"),
            "error names the unplaceable hydro: {err}"
        );
    }

    /// A keyed coefficient vector deeper than the declared depth is unplaceable.
    #[test]
    fn reserve_rejects_depth_beyond_declared() {
        let manifest = vec![EntitySlot::storage(1, true)];
        let cut_coefficients = vec![vec![1.0]];
        let mut keyed: HashMap<i32, Vec<f64>> = HashMap::new();
        keyed.insert(1, vec![0.1, 0.2, 0.3]); // depth 3 > N=2
        let cut_lag = vec![keyed];

        let err = reserve_boundary_inflow_lag_slots(&manifest, &cut_coefficients, &cut_lag, 2)
            .expect_err("depth beyond N must reject");
        assert!(format!("{err}").contains("exceeding"), "{err}");
    }

    /// A manifest with no leading storage block cannot key lag slots.
    #[test]
    fn reserve_rejects_manifest_without_leading_storage() {
        let manifest = vec![anticipated_slot(9, 0)];
        let cut_coefficients = vec![vec![1.0]];
        let cut_lag = vec![HashMap::new()];

        let err = reserve_boundary_inflow_lag_slots(&manifest, &cut_coefficients, &cut_lag, 1)
            .expect_err("no leading storage must reject");
        assert!(format!("{err}").contains("HydroStorage"), "{err}");
    }

    /// A coefficient vector whose length disagrees with the manifest cannot be
    /// positionally aligned for insertion.
    #[test]
    fn reserve_rejects_coefficient_manifest_length_mismatch() {
        let manifest = vec![EntitySlot::storage(1, true), EntitySlot::storage(2, true)];
        let cut_coefficients = vec![vec![1.0]]; // len 1, manifest len 2
        let cut_lag = vec![HashMap::new()];

        let err = reserve_boundary_inflow_lag_slots(&manifest, &cut_coefficients, &cut_lag, 1)
            .expect_err("length mismatch must reject");
        assert!(format!("{err}").contains("positionally aligned"), "{err}");
    }

    /// `inflow_lag_depth == 0` is a misuse: the caller reserves only for a
    /// positive depth (the absent/zero case is the byte-identical no-op path).
    #[test]
    fn reserve_rejects_zero_depth() {
        let manifest = vec![EntitySlot::storage(1, true)];
        let cut_coefficients = vec![vec![1.0]];
        let cut_lag = vec![HashMap::new()];

        assert!(
            reserve_boundary_inflow_lag_slots(&manifest, &cut_coefficients, &cut_lag, 0).is_err()
        );
    }
}
