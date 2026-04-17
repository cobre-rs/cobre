//! Policy checkpoint export helpers.
//!
//! Shared conversion logic for extracting active cuts and basis data from a
//! trained [`FutureCostFunction`] and [`TrainingResult`] into the `cobre-io`
//! policy types needed by [`cobre_io::write_policy_checkpoint`].
//!
//! This module eliminates the duplicated conversion that previously existed
//! independently in `cobre-cli` and `cobre-python`.

#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]

use cobre_io::output::policy::{
    PolicyBasisRecord, PolicyCutRecord, StageCutsPayload, StageStatesPayload,
};

use crate::cut::FutureCostFunction;
use crate::training::TrainingResult;

/// Build per-stage vectors of **all** populated [`PolicyCutRecord`]s from the FCF pools.
///
/// Both active and inactive cuts are included so the checkpoint preserves
/// the full training history. Use [`build_active_indices`] to obtain the
/// subset that is currently active in the LP.
///
/// Each record borrows its `coefficients` slice from the FCF, so the returned
/// vectors are valid as long as `fcf` is alive.
#[must_use]
pub fn build_stage_cut_records(fcf: &FutureCostFunction) -> Vec<Vec<PolicyCutRecord<'_>>> {
    fcf.pools
        .iter()
        .map(|pool| {
            (0..pool.populated_count)
                .map(|i| {
                    let meta = &pool.metadata[i];
                    PolicyCutRecord {
                        cut_id: meta.iteration_generated * u64::from(pool.forward_passes)
                            + u64::from(meta.forward_pass_index),
                        slot_index: i as u32,
                        iteration: meta.iteration_generated as u32,
                        forward_pass_index: meta.forward_pass_index,
                        intercept: pool.intercepts[i],
                        coefficients: &pool.coefficients
                            [i * pool.state_dimension..(i + 1) * pool.state_dimension],
                        is_active: pool.active[i],
                        domination_count: meta.active_count as u32,
                    }
                })
                .collect()
        })
        .collect()
}

/// Build per-stage active cut index lists from the stage cut records.
///
/// Returns only the `slot_index` values of records where `is_active` is `true`.
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

/// Build [`StageCutsPayload`] references from pre-built records and indices.
///
/// `stage_records` and `stage_active_indices` must have been built from the
/// same `fcf` via [`build_stage_cut_records`] and [`build_active_indices`].
#[must_use]
pub fn build_stage_cuts_payloads<'a>(
    fcf: &FutureCostFunction,
    stage_records: &'a [Vec<PolicyCutRecord<'a>>],
    stage_active_indices: &'a [Vec<u32>],
) -> Vec<StageCutsPayload<'a>> {
    fcf.pools
        .iter()
        .enumerate()
        .map(|(stage_idx, pool)| StageCutsPayload {
            stage_id: stage_idx as u32,
            state_dimension: fcf.state_dimension as u32,
            capacity: pool.capacity as u32,
            warm_start_count: pool.warm_start_count,
            cuts: &stage_records[stage_idx],
            active_cut_indices: &stage_active_indices[stage_idx],
            populated_count: pool.populated_count as u32,
        })
        .collect()
}

/// Convert the solver basis cache from i32 status codes to u8 byte vectors.
///
/// `HiGHS` status codes are in the range 0..=4, so the truncation is safe.
/// Returns `(col_status_bytes, row_status_bytes)`.
#[must_use]
pub fn convert_basis_cache(training_result: &TrainingResult) -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
    let col = training_result
        .basis_cache
        .iter()
        .map(|opt| {
            opt.as_ref()
                .map(|cb| cb.basis.col_status.iter().map(|&v| v as u8).collect())
                .unwrap_or_default()
        })
        .collect();
    let row = training_result
        .basis_cache
        .iter()
        .map(|opt| {
            opt.as_ref()
                .map(|cb| cb.basis.row_status.iter().map(|&v| v as u8).collect())
                .unwrap_or_default()
        })
        .collect();
    (col, row)
}

/// Build per-stage [`PolicyBasisRecord`] references from pre-converted basis data.
#[must_use]
pub fn build_stage_basis_records<'a>(
    fcf: &FutureCostFunction,
    training_result: &TrainingResult,
    basis_col_u8: &'a [Vec<u8>],
    basis_row_u8: &'a [Vec<u8>],
) -> Vec<PolicyBasisRecord<'a>> {
    training_result
        .basis_cache
        .iter()
        .enumerate()
        .filter_map(|(stage_idx, opt)| {
            opt.as_ref().map(|_| {
                let num_cut_rows = fcf
                    .pools
                    .get(stage_idx)
                    .map_or(0, |pool| pool.populated_count.min(pool.capacity) as u32);
                PolicyBasisRecord {
                    stage_id: stage_idx as u32,
                    iteration: training_result.iterations as u32,
                    column_status: &basis_col_u8[stage_idx],
                    row_status: &basis_row_u8[stage_idx],
                    num_cut_rows,
                }
            })
        })
        .collect()
}

/// Build per-stage [`StageStatesPayload`]s from the visited states archive.
///
/// Returns an empty `Vec` if the archive is `None` (non-Dominated strategies).
#[must_use]
pub fn build_stage_states_payloads(
    archive: Option<&crate::visited_states::VisitedStatesArchive>,
) -> Vec<StageStatesPayload<'_>> {
    let Some(archive) = archive else {
        return Vec::new();
    };
    (0..archive.num_stages())
        .map(|t| {
            let stage = archive.stage(t);
            StageStatesPayload {
                stage_id: t as u32,
                state_dimension: stage.state_dimension() as u32,
                count: stage.count() as u32,
                data: stage.states(),
            }
        })
        .collect()
}
