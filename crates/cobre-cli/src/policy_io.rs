//! Policy checkpoint writer: extracts cuts and basis from the trained
//! [`FutureCostFunction`] and writes them to `FlatBuffers` files via
//! [`write_policy_checkpoint`].
//!
//! This module bridges the gap between the solver-layer FCF (cobre-sddp)
//! and the I/O-layer policy writer (cobre-io). The two crates cannot depend
//! on each other, so conversion happens here in the CLI.

use std::path::Path;

use cobre_io::output::policy::{
    PolicyBasisRecord, PolicyCheckpointMetadata, PolicyCutRecord, StageCutsPayload,
    write_policy_checkpoint,
};
use cobre_sddp::{FutureCostFunction, TrainingResult};

use crate::error::CliError;

/// Parameters for writing a policy checkpoint.
pub struct CheckpointParams {
    pub max_iterations: u64,
    pub forward_passes: u32,
    pub seed: u64,
}

/// Write a policy checkpoint from the trained FCF and basis cache.
///
/// Extracts all active cuts from each stage pool, converts the per-stage
/// basis from i32 status codes to u8, and serializes everything to
/// `FlatBuffers` binary files under `policy_dir/`.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn write_checkpoint(
    policy_dir: &Path,
    fcf: &FutureCostFunction,
    training_result: &TrainingResult,
    params: &CheckpointParams,
) -> Result<(), CliError> {
    let n_stages = fcf.pools.len();
    let state_dimension = fcf.state_dimension;

    // Build per-stage PolicyCutRecord vectors from the FCF pools.
    let stage_records: Vec<Vec<PolicyCutRecord<'_>>> = fcf
        .pools
        .iter()
        .map(|pool| {
            (0..pool.populated_count)
                .filter(|&i| pool.active[i])
                .map(|i| {
                    let meta = &pool.metadata[i];
                    PolicyCutRecord {
                        cut_id: meta.iteration_generated * u64::from(pool.forward_passes)
                            + u64::from(meta.forward_pass_index),
                        slot_index: i as u32,
                        iteration: meta.iteration_generated as u32,
                        forward_pass_index: meta.forward_pass_index,
                        intercept: pool.intercepts[i],
                        coefficients: &pool.coefficients[i],
                        is_active: true,
                        domination_count: meta.active_count as u32,
                    }
                })
                .collect()
        })
        .collect();

    // Build active cut index lists and StageCutsPayload references.
    let stage_active_indices: Vec<Vec<u32>> = stage_records
        .iter()
        .map(|records| records.iter().map(|r| r.slot_index).collect())
        .collect();

    let stage_cuts: Vec<StageCutsPayload<'_>> = fcf
        .pools
        .iter()
        .enumerate()
        .map(|(stage_idx, pool)| StageCutsPayload {
            stage_id: stage_idx as u32,
            state_dimension: state_dimension as u32,
            capacity: pool.capacity as u32,
            warm_start_count: pool.warm_start_count,
            cuts: &stage_records[stage_idx],
            active_cut_indices: &stage_active_indices[stage_idx],
            populated_count: pool.populated_count as u32,
        })
        .collect();

    // Convert basis cache: solver stores i32 status codes, policy writer expects u8.
    // HiGHS status codes are 0-4, so the truncation is safe.
    let basis_col_u8: Vec<Vec<u8>> = training_result
        .basis_cache
        .iter()
        .map(|opt| {
            opt.as_ref()
                .map(|b| b.col_status.iter().map(|&v| v as u8).collect())
                .unwrap_or_default()
        })
        .collect();

    let basis_row_u8: Vec<Vec<u8>> = training_result
        .basis_cache
        .iter()
        .map(|opt| {
            opt.as_ref()
                .map(|b| b.row_status.iter().map(|&v| v as u8).collect())
                .unwrap_or_default()
        })
        .collect();

    let stage_bases: Vec<PolicyBasisRecord<'_>> = training_result
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
        .collect();

    let metadata = PolicyCheckpointMetadata {
        version: "1.0.0".to_string(),
        cobre_version: env!("CARGO_PKG_VERSION").to_string(),
        created_at: epoch_timestamp(),
        completed_iterations: training_result.iterations as u32,
        final_lower_bound: training_result.final_lb,
        best_upper_bound: Some(training_result.final_ub),
        state_dimension: state_dimension as u32,
        num_stages: n_stages as u32,
        config_hash: String::new(),
        system_hash: String::new(),
        max_iterations: params.max_iterations as u32,
        forward_passes: params.forward_passes,
        warm_start_cuts: 0,
        rng_seed: params.seed,
    };

    write_policy_checkpoint(policy_dir, &stage_cuts, &stage_bases, &metadata)
        .map_err(CliError::from)
}

/// Produce a timestamp string from the system clock.
fn epoch_timestamp() -> String {
    use std::time::SystemTime;
    match SystemTime::now().duration_since(SystemTime::UNIX_EPOCH) {
        Ok(d) => format!("{}s-since-epoch", d.as_secs()),
        Err(_) => "unknown".to_string(),
    }
}
