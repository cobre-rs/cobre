//! Policy loading and compatibility validation.
//!
//! This module contains the validation logic for checking whether a loaded
//! policy checkpoint is compatible with the current system configuration,
//! and helpers for reconstructing solver state from deserialized checkpoint
//! data.
//!
//! [`FutureCostFunction`]: crate::FutureCostFunction

use cobre_io::PolicyCheckpointMetadata;
use cobre_solver::Basis;

use crate::SddpError;

/// Validate that a loaded policy checkpoint is compatible with the current
/// system configuration.
///
/// Returns [`SddpError::Validation`] if either of these conditions hold:
/// - `metadata.state_dimension != current_state_dimension`
/// - `metadata.num_stages != current_num_stages`
///
/// # Examples
///
/// ```
/// use cobre_io::PolicyCheckpointMetadata;
/// use cobre_sddp::validate_policy_compatibility;
///
/// let meta = PolicyCheckpointMetadata {
///     cobre_version: "0.2.2".to_string(),
///     created_at: "2026-03-29T00:00:00Z".to_string(),
///     completed_iterations: 50,
///     final_lower_bound: 1234.56,
///     best_upper_bound: Some(1300.0),
///     state_dimension: 10,
///     num_stages: 12,
///     max_iterations: 200,
///     forward_passes: 4,
///     warm_start_cuts: 0,
///     rng_seed: 42,
///     total_visited_states: 0,
/// };
///
/// // Compatible metadata passes validation.
/// assert!(validate_policy_compatibility(&meta, 10, 12).is_ok());
///
/// // Mismatched state_dimension returns an error.
/// assert!(validate_policy_compatibility(&meta, 8, 12).is_err());
/// ```
///
/// # Errors
///
/// Returns `SddpError::Validation` if `state_dimension` or `num_stages`
/// do not match.
pub fn validate_policy_compatibility(
    metadata: &PolicyCheckpointMetadata,
    current_state_dimension: u32,
    current_num_stages: u32,
) -> Result<(), SddpError> {
    if metadata.state_dimension != current_state_dimension {
        return Err(SddpError::Validation(format!(
            "policy state_dimension mismatch: policy has {}, current system has {}",
            metadata.state_dimension, current_state_dimension
        )));
    }

    if metadata.num_stages != current_num_stages {
        return Err(SddpError::Validation(format!(
            "policy num_stages mismatch: policy has {}, current system has {}",
            metadata.num_stages, current_num_stages
        )));
    }

    Ok(())
}

/// Build a basis cache from deserialized checkpoint basis records.
///
/// Returns a `Vec<Option<Basis>>` with one entry per stage (0-based). Stages
/// that have a matching record in `stage_bases` get `Some(Basis)` with the
/// `u8` status codes widened to `i32`. Stages without a record get `None`.
///
/// # Parameters
///
/// - `num_stages`: total number of stages in the study.
/// - `stage_bases`: deserialized basis records from the policy checkpoint.
#[must_use]
pub fn build_basis_cache_from_checkpoint(
    num_stages: usize,
    stage_bases: &[cobre_io::OwnedPolicyBasisRecord],
) -> Vec<Option<Basis>> {
    let mut cache: Vec<Option<Basis>> = vec![None; num_stages];
    for record in stage_bases {
        let stage = record.stage_id as usize;
        if stage < num_stages {
            let col_status: Vec<i32> = record.column_status.iter().map(|&c| i32::from(c)).collect();
            let row_status: Vec<i32> = record.row_status.iter().map(|&r| i32::from(r)).collect();
            cache[stage] = Some(Basis {
                col_status,
                row_status,
            });
        }
    }
    cache
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use cobre_io::PolicyCheckpointMetadata;

    use super::validate_policy_compatibility;

    fn sample_metadata() -> PolicyCheckpointMetadata {
        PolicyCheckpointMetadata {
            cobre_version: "0.2.2".to_string(),
            created_at: "2026-03-29T00:00:00Z".to_string(),
            completed_iterations: 50,
            final_lower_bound: 1234.56,
            best_upper_bound: Some(1300.0),
            state_dimension: 10,
            num_stages: 12,
            max_iterations: 200,
            forward_passes: 4,
            warm_start_cuts: 0,
            rng_seed: 42,
            total_visited_states: 0,
        }
    }

    #[test]
    fn compatible_metadata_passes() {
        let meta = sample_metadata();
        assert!(validate_policy_compatibility(&meta, 10, 12).is_ok());
    }

    #[test]
    fn state_dimension_mismatch_fails() {
        let meta = sample_metadata();
        let result = validate_policy_compatibility(&meta, 8, 12);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("state_dimension"), "{msg}");
        assert!(msg.contains("10"), "should include policy value: {msg}");
        assert!(msg.contains('8'), "should include current value: {msg}");
    }

    #[test]
    fn num_stages_mismatch_fails() {
        let meta = sample_metadata();
        let result = validate_policy_compatibility(&meta, 10, 24);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("num_stages"), "{msg}");
        assert!(msg.contains("12"), "should include policy value: {msg}");
        assert!(msg.contains("24"), "should include current value: {msg}");
    }

    #[test]
    fn both_dimensions_mismatched_returns_err() {
        let meta = sample_metadata();
        // Both state_dimension (10 vs 8) and num_stages (12 vs 24) mismatch.
        // The function should return an error on the first mismatch (state_dimension).
        let result = validate_policy_compatibility(&meta, 8, 24);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("state_dimension"),
            "should report state_dimension mismatch first: {msg}"
        );
    }
}
