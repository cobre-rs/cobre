//! Policy loading and compatibility validation.
//!
//! This module contains the validation logic for checking whether a loaded
//! policy checkpoint is compatible with the current system configuration.
//! The reconstruction logic (building a [`FutureCostFunction`] from deserialized
//! cuts) will be added in subsequent tickets.
//!
//! [`FutureCostFunction`]: crate::FutureCostFunction

use cobre_io::PolicyCheckpointMetadata;

use crate::SddpError;

/// Validate that a loaded policy checkpoint is compatible with the current
/// system configuration.
///
/// # Hard failures
///
/// Returns [`SddpError::Validation`] if either of these conditions hold:
/// - `metadata.state_dimension != current_state_dimension`
/// - `metadata.num_stages != current_num_stages`
///
/// # Soft warnings
///
/// When `config_hash` or `system_hash` differ between the metadata and the
/// current system, a `tracing::warn!` is emitted but the function returns
/// `Ok(())`. Hash comparison only fires when the provided hash reference is
/// `Some` and the metadata hash is non-empty, since hashes are currently
/// written as empty strings.
///
/// # Examples
///
/// ```
/// use cobre_io::PolicyCheckpointMetadata;
/// use cobre_sddp::validate_policy_compatibility;
///
/// let meta = PolicyCheckpointMetadata {
///     version: "1.0.0".to_string(),
///     cobre_version: "0.2.2".to_string(),
///     created_at: "2026-03-29T00:00:00Z".to_string(),
///     completed_iterations: 50,
///     final_lower_bound: 1234.56,
///     best_upper_bound: Some(1300.0),
///     state_dimension: 10,
///     num_stages: 12,
///     config_hash: String::new(),
///     system_hash: String::new(),
///     max_iterations: 200,
///     forward_passes: 4,
///     warm_start_cuts: 0,
///     rng_seed: 42,
/// };
///
/// // Compatible metadata passes validation.
/// assert!(validate_policy_compatibility(&meta, 10, 12, None, None).is_ok());
///
/// // Mismatched state_dimension returns an error.
/// assert!(validate_policy_compatibility(&meta, 8, 12, None, None).is_err());
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
    current_config_hash: Option<&str>,
    current_system_hash: Option<&str>,
) -> Result<(), SddpError> {
    // Hard failures: dimensional mismatches.
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

    // Soft warnings: hash mismatches (only when both sides have non-empty values).
    if let Some(current) = current_config_hash {
        if !metadata.config_hash.is_empty() && metadata.config_hash != current {
            eprintln!(
                "warning: policy config_hash differs from current configuration \
                 (policy={}, current={})",
                metadata.config_hash, current
            );
        }
    }

    if let Some(current) = current_system_hash {
        if !metadata.system_hash.is_empty() && metadata.system_hash != current {
            eprintln!(
                "warning: policy system_hash differs from current system \
                 (policy={}, current={})",
                metadata.system_hash, current
            );
        }
    }

    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use cobre_io::PolicyCheckpointMetadata;

    use super::validate_policy_compatibility;

    fn sample_metadata() -> PolicyCheckpointMetadata {
        PolicyCheckpointMetadata {
            version: "1.0.0".to_string(),
            cobre_version: "0.2.2".to_string(),
            created_at: "2026-03-29T00:00:00Z".to_string(),
            completed_iterations: 50,
            final_lower_bound: 1234.56,
            best_upper_bound: Some(1300.0),
            state_dimension: 10,
            num_stages: 12,
            config_hash: String::new(),
            system_hash: String::new(),
            max_iterations: 200,
            forward_passes: 4,
            warm_start_cuts: 0,
            rng_seed: 42,
        }
    }

    #[test]
    fn compatible_metadata_passes() {
        let meta = sample_metadata();
        assert!(validate_policy_compatibility(&meta, 10, 12, None, None).is_ok());
    }

    #[test]
    fn state_dimension_mismatch_fails() {
        let meta = sample_metadata();
        let result = validate_policy_compatibility(&meta, 8, 12, None, None);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("state_dimension"), "{msg}");
        assert!(msg.contains("10"), "should include policy value: {msg}");
        assert!(msg.contains('8'), "should include current value: {msg}");
    }

    #[test]
    fn num_stages_mismatch_fails() {
        let meta = sample_metadata();
        let result = validate_policy_compatibility(&meta, 10, 24, None, None);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("num_stages"), "{msg}");
        assert!(msg.contains("12"), "should include policy value: {msg}");
        assert!(msg.contains("24"), "should include current value: {msg}");
    }

    #[test]
    fn config_hash_mismatch_emits_warning_not_error() {
        let mut meta = sample_metadata();
        meta.config_hash = "abc123".to_string();
        let result = validate_policy_compatibility(&meta, 10, 12, Some("def456"), None);
        assert!(
            result.is_ok(),
            "hash mismatch should be a warning, not an error"
        );
    }

    #[test]
    fn system_hash_mismatch_emits_warning_not_error() {
        let mut meta = sample_metadata();
        meta.system_hash = "sys_old".to_string();
        let result = validate_policy_compatibility(&meta, 10, 12, None, Some("sys_new"));
        assert!(
            result.is_ok(),
            "hash mismatch should be a warning, not an error"
        );
    }

    #[test]
    fn empty_policy_hash_skips_comparison() {
        let meta = sample_metadata(); // config_hash and system_hash are empty
        let result = validate_policy_compatibility(&meta, 10, 12, Some("abc"), Some("def"));
        assert!(result.is_ok(), "empty policy hash should skip comparison");
    }

    #[test]
    fn matching_hashes_pass_silently() {
        let mut meta = sample_metadata();
        meta.config_hash = "same".to_string();
        meta.system_hash = "same".to_string();
        let result = validate_policy_compatibility(&meta, 10, 12, Some("same"), Some("same"));
        assert!(result.is_ok());
    }

    #[test]
    fn both_dimensions_mismatched_returns_err() {
        let meta = sample_metadata();
        // Both state_dimension (10 vs 8) and num_stages (12 vs 24) mismatch.
        // The function should return an error on the first mismatch (state_dimension).
        let result = validate_policy_compatibility(&meta, 8, 24, None, None);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("state_dimension"),
            "should report state_dimension mismatch first: {msg}"
        );
    }
}
