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

/// Resolve the per-stage warm-start cut counts from a loaded policy checkpoint.
///
/// Returns a `Vec<u32>` of length `num_stages` for [`FutureCostFunction::new`].
///
/// - If `metadata.warm_start_counts` is non-empty, it is returned after length validation.
/// - If `metadata.warm_start_counts` is empty (old checkpoint format), the scalar
///   `warm_start_cuts` is broadcast to all stages.
///
/// # Errors
///
/// Returns [`SddpError::Validation`] if `warm_start_counts.len() != num_stages`.
///
/// [`FutureCostFunction::new`]: crate::FutureCostFunction::new
pub fn resolve_warm_start_counts(
    metadata: &PolicyCheckpointMetadata,
    num_stages: usize,
) -> Result<Vec<u32>, SddpError> {
    if metadata.warm_start_counts.is_empty() {
        // Old checkpoint: broadcast scalar to all stages.
        Ok(vec![metadata.warm_start_cuts; num_stages])
    } else if metadata.warm_start_counts.len() != num_stages {
        Err(SddpError::Validation(format!(
            "warm_start_counts length mismatch: checkpoint has {}, current system has {} stages",
            metadata.warm_start_counts.len(),
            num_stages,
        )))
    } else {
        Ok(metadata.warm_start_counts.clone())
    }
}

/// Validate that a loaded policy checkpoint is compatible with the current
/// system configuration.
///
/// # Errors
///
/// Returns [`SddpError::Validation`] if `state_dimension` or `num_stages`
/// do not match the current system configuration.
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
/// Returns a `Vec<Option<Basis>>` with one entry per stage. Stages with a
/// matching record get `Some(Basis)` (with `u8` status codes widened to `i32`);
/// stages without a record get `None`.
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

    use super::{resolve_warm_start_counts, validate_policy_compatibility};

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
            warm_start_counts: vec![],
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

    // ── resolve_warm_start_counts tests ───────────────────────────────────────

    fn meta_with_counts(
        warm_start_cuts: u32,
        warm_start_counts: Vec<u32>,
    ) -> PolicyCheckpointMetadata {
        #[allow(clippy::cast_possible_truncation)]
        let num_stages: u32 = if warm_start_counts.is_empty() {
            3
        } else {
            warm_start_counts.len() as u32
        };
        PolicyCheckpointMetadata {
            cobre_version: "0.4.0".to_string(),
            created_at: "2026-04-01T00:00:00Z".to_string(),
            completed_iterations: 10,
            final_lower_bound: 0.0,
            best_upper_bound: None,
            state_dimension: 2,
            num_stages,
            max_iterations: 50,
            forward_passes: 1,
            warm_start_cuts,
            warm_start_counts,
            rng_seed: 0,
            total_visited_states: 0,
        }
    }

    #[test]
    fn resolve_warm_start_counts_new_format_returns_per_stage_counts() {
        let meta = meta_with_counts(10, vec![10, 8, 6]);
        let counts = resolve_warm_start_counts(&meta, 3).unwrap();
        assert_eq!(counts, vec![10u32, 8, 6]);
    }

    #[test]
    fn resolve_warm_start_counts_old_format_broadcasts_scalar() {
        // Empty warm_start_counts: fall back to warm_start_cuts broadcast.
        let meta = meta_with_counts(5, vec![]);
        let counts = resolve_warm_start_counts(&meta, 3).unwrap();
        assert_eq!(counts, vec![5u32, 5, 5]);
    }

    #[test]
    fn resolve_warm_start_counts_old_format_zero_scalar_broadcasts_zeros() {
        let meta = meta_with_counts(0, vec![]);
        let counts = resolve_warm_start_counts(&meta, 3).unwrap();
        assert_eq!(counts, vec![0u32, 0, 0]);
    }

    #[test]
    fn resolve_warm_start_counts_wrong_length_returns_validation_error() {
        // warm_start_counts has 2 entries but num_stages is 3 — corrupted checkpoint.
        let meta = meta_with_counts(5, vec![5, 5]);
        let result = resolve_warm_start_counts(&meta, 3);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("warm_start_counts length mismatch"),
            "error message should mention length mismatch: {msg}"
        );
        assert!(msg.contains('2'), "should include vector length: {msg}");
        assert!(msg.contains('3'), "should include num_stages: {msg}");
    }

    #[test]
    fn resolve_warm_start_counts_single_stage_new_format() {
        let meta = meta_with_counts(7, vec![7]);
        let counts = resolve_warm_start_counts(&meta, 1).unwrap();
        assert_eq!(counts, vec![7u32]);
    }

    #[test]
    fn resolve_warm_start_counts_zero_stages_old_format_returns_empty() {
        let meta = meta_with_counts(5, vec![]);
        let counts = resolve_warm_start_counts(&meta, 0).unwrap();
        assert!(counts.is_empty());
    }
}
