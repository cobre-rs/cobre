//! Policy checkpoint writer: extracts cuts and basis from the trained
//! [`FutureCostFunction`] and writes them to `FlatBuffers` files via
//! [`write_policy_checkpoint`].
//!
//! Cut and basis extraction is delegated to the shared helpers in
//! [`cobre_sddp::policy_export`]. This module adds only the CLI-specific
//! metadata assembly and error mapping.

use std::path::Path;

use cobre_io::output::policy::{PolicyCheckpointMetadata, write_policy_checkpoint};
use cobre_sddp::policy_export::{
    build_active_indices, build_stage_basis_records, build_stage_cut_records,
    build_stage_cuts_payloads, convert_basis_cache,
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

    let stage_records = build_stage_cut_records(fcf);
    let stage_active_indices = build_active_indices(&stage_records);
    let stage_cuts = build_stage_cuts_payloads(fcf, &stage_records, &stage_active_indices);

    let (basis_col_u8, basis_row_u8) = convert_basis_cache(training_result);
    let stage_bases = build_stage_basis_records(fcf, training_result, &basis_col_u8, &basis_row_u8);

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
