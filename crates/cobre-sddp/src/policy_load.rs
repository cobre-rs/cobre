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

use crate::cut::pool::CutPool;
use crate::setup::StudySetup;
use crate::workspace::CapturedBasis;
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
/// Returns a `Vec<Option<CapturedBasis>>` with one entry per stage. Stages with
/// a matching record get `Some(CapturedBasis)` (with `u8` status codes widened
/// to `i32`); stages without a record get `None`.
///
/// The returned `CapturedBasis` entries have empty metadata (`cut_row_slots`,
/// `state_at_capture`, `base_row_count = 0`). Checkpoint files do not store
/// slot-tracking metadata; on the simulation warm-start path, `reconstruct_basis`
/// degrades gracefully when `cut_row_slots` is empty (all rows treated as new).
#[must_use]
pub fn build_basis_cache_from_checkpoint(
    num_stages: usize,
    stage_bases: &[cobre_io::OwnedPolicyBasisRecord],
) -> Vec<Option<CapturedBasis>> {
    let mut cache: Vec<Option<CapturedBasis>> = vec![None; num_stages];
    for record in stage_bases {
        let stage = record.stage_id as usize;
        if stage < num_stages {
            let col_status: Vec<i32> = record.column_status.iter().map(|&c| i32::from(c)).collect();
            let row_status: Vec<i32> = record.row_status.iter().map(|&r| i32::from(r)).collect();
            cache[stage] = Some(CapturedBasis {
                basis: Basis {
                    col_status,
                    row_status,
                },
                base_row_count: 0,
                cut_row_slots: Vec::new(),
                state_at_capture: Vec::new(),
            });
        }
    }
    cache
}

/// Load boundary cuts from a source Cobre policy checkpoint.
///
/// Reads the checkpoint at `boundary_path`, extracts cuts from the stage
/// identified by `source_stage`, and validates that the source state dimension
/// matches `current_state_dimension`.
///
/// Only `state_dimension` must match between the source and current study —
/// `num_stages` may differ (e.g., a monthly source checkpoint vs. a
/// weekly+monthly current study).
///
/// # Errors
///
/// Returns [`SddpError::Validation`] if:
/// - The checkpoint cannot be read
/// - `source_stage` does not exist in the checkpoint
/// - The source stage's state dimension does not match `current_state_dimension`
pub fn load_boundary_cuts(
    boundary_path: &std::path::Path,
    source_stage: u32,
    current_state_dimension: u32,
) -> Result<Vec<cobre_io::OwnedPolicyCutRecord>, SddpError> {
    let checkpoint =
        cobre_io::output::policy::read_policy_checkpoint(boundary_path).map_err(|e| {
            SddpError::Validation(format!(
                "failed to read boundary policy checkpoint at {}: {e}",
                boundary_path.display()
            ))
        })?;

    let stage_result = checkpoint
        .stage_cuts
        .iter()
        .find(|sr| sr.stage_id == source_stage)
        .ok_or_else(|| {
            SddpError::Validation(format!(
                "boundary policy: source_stage {} not found in checkpoint \
                 (available stages: {:?})",
                source_stage,
                checkpoint
                    .stage_cuts
                    .iter()
                    .map(|sr| sr.stage_id)
                    .collect::<Vec<_>>()
            ))
        })?;

    if stage_result.state_dimension != current_state_dimension {
        return Err(SddpError::Validation(format!(
            "boundary policy state_dimension mismatch: source stage {} has {}, \
             current study has {}",
            source_stage, stage_result.state_dimension, current_state_dimension
        )));
    }

    Ok(stage_result.cuts.clone())
}

/// Inject boundary cuts into the terminal stage of the study's FCF.
///
/// Replaces the terminal stage's [`CutPool`] with one pre-populated from
/// `boundary_records`. The pool retains capacity for new training cuts;
/// only the terminal pool is modified.
///
/// After this call, `setup.fcf().pools[num_stages - 1].warm_start_count`
/// equals `boundary_records.len()`, which causes the forward pass to set
/// `terminal_has_boundary_cuts = true` and prevents theta zeroing at the
/// terminal stage.
pub fn inject_boundary_cuts(
    setup: &mut StudySetup,
    boundary_records: &[cobre_io::OwnedPolicyCutRecord],
) {
    let fcf = setup.fcf_mut();
    let terminal_idx = fcf.pools.len() - 1;
    let state_dimension = fcf.state_dimension;
    let forward_passes = fcf.forward_passes;
    let existing_capacity = fcf.pools[terminal_idx].capacity;
    let existing_warm_start = fcf.pools[terminal_idx].warm_start_count as usize;
    let training_capacity = existing_capacity.saturating_sub(existing_warm_start);
    let max_iterations = if forward_passes > 0 {
        #[allow(clippy::cast_possible_truncation)]
        let m = (training_capacity / forward_passes as usize) as u64;
        m
    } else {
        0
    };
    fcf.pools[terminal_idx] = CutPool::new_with_warm_start(
        state_dimension,
        forward_passes,
        max_iterations,
        boundary_records,
    );
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::cast_possible_truncation)]
mod tests {
    use cobre_io::{PolicyCheckpointMetadata, StageCutsPayload};

    use super::{load_boundary_cuts, resolve_warm_start_counts, validate_policy_compatibility};

    // ── helpers ───────────────────────────────────────────────────────────────

    /// Write a minimal policy checkpoint to `dir` with `n_stages` stages each
    /// having `state_dimension` state variables and the supplied cut intercepts.
    ///
    /// Each stage gets `cuts.len()` cuts with coefficients all set to 1.0.
    fn write_minimal_checkpoint(
        dir: &std::path::Path,
        n_stages: u32,
        state_dimension: u32,
        cut_intercepts: &[f64],
    ) {
        let state_dim = state_dimension as usize;
        let coefficients = vec![1.0_f64; state_dim];
        let n_cuts = cut_intercepts.len();

        // Build cut records for each stage.
        let cut_records: Vec<Vec<cobre_io::PolicyCutRecord<'_>>> = (0..n_stages)
            .map(|_| {
                cut_intercepts
                    .iter()
                    .enumerate()
                    .map(|(i, &intercept)| cobre_io::PolicyCutRecord {
                        cut_id: i as u64,
                        slot_index: i as u32,
                        iteration: i as u32,
                        forward_pass_index: 0,
                        intercept,
                        coefficients: &coefficients,
                        is_active: true,
                        domination_count: 0,
                    })
                    .collect()
            })
            .collect();

        let active_indices: Vec<Vec<u32>> = (0..n_stages)
            .map(|_| (0..n_cuts as u32).collect())
            .collect();

        let payloads: Vec<StageCutsPayload<'_>> = (0..n_stages as usize)
            .map(|s| StageCutsPayload {
                stage_id: s as u32,
                state_dimension,
                capacity: n_cuts as u32,
                warm_start_count: 0,
                cuts: &cut_records[s],
                active_cut_indices: &active_indices[s],
                populated_count: n_cuts as u32,
            })
            .collect();

        let metadata = PolicyCheckpointMetadata {
            cobre_version: "0.4.0".to_string(),
            created_at: "2026-04-14T00:00:00Z".to_string(),
            completed_iterations: 10,
            final_lower_bound: 0.0,
            best_upper_bound: None,
            state_dimension,
            num_stages: n_stages,
            max_iterations: 50,
            forward_passes: 1,
            warm_start_cuts: 0,
            warm_start_counts: vec![],
            rng_seed: 0,
            total_visited_states: 0,
        };

        cobre_io::write_policy_checkpoint(dir, &payloads, &[], &metadata, &[]).unwrap();
    }

    // ── load_boundary_cuts tests ──────────────────────────────────────────────

    /// Given a valid checkpoint with 12 stages and `state_dimension=10`, when
    /// `load_boundary_cuts` is called for stage 2 with matching dimension,
    /// then it returns `Ok` with the cuts from that stage.
    #[test]
    fn load_boundary_cuts_valid_stage() {
        let tmp = tempfile::tempdir().unwrap();
        let intercepts = vec![10.0, 20.0, 30.0];
        write_minimal_checkpoint(tmp.path(), 12, 10, &intercepts);

        let cuts = load_boundary_cuts(tmp.path(), 2, 10).unwrap();

        assert_eq!(cuts.len(), 3, "should return all 3 cuts from stage 2");
        let returned_intercepts: Vec<f64> = cuts.iter().map(|c| c.intercept).collect();
        assert_eq!(
            returned_intercepts, intercepts,
            "intercepts should match written values"
        );
        for cut in &cuts {
            assert_eq!(
                cut.coefficients.len(),
                10,
                "each cut should have state_dimension=10 coefficients"
            );
        }
    }

    /// Given a checkpoint without stage 99, when `load_boundary_cuts` is called
    /// for stage 99, then it returns `Err(SddpError::Validation)` with a message
    /// containing `"source_stage"` and `"99"`.
    #[test]
    fn load_boundary_cuts_missing_stage_returns_error() {
        let tmp = tempfile::tempdir().unwrap();
        write_minimal_checkpoint(tmp.path(), 5, 10, &[1.0]);

        let result = load_boundary_cuts(tmp.path(), 99, 10);

        assert!(result.is_err(), "should fail for missing stage");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("source_stage"),
            "error should mention 'source_stage': {msg}"
        );
        assert!(
            msg.contains("99"),
            "error should include the missing stage index: {msg}"
        );
    }

    /// Given a checkpoint with `state_dimension=10`, when `load_boundary_cuts` is
    /// called with `current_state_dimension=5`, then it returns
    /// `Err(SddpError::Validation)` with a message containing `"state_dimension"`.
    #[test]
    fn load_boundary_cuts_state_dimension_mismatch_returns_error() {
        let tmp = tempfile::tempdir().unwrap();
        write_minimal_checkpoint(tmp.path(), 5, 10, &[1.0]);

        let result = load_boundary_cuts(tmp.path(), 0, 5);

        assert!(result.is_err(), "should fail for dimension mismatch");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("state_dimension"),
            "error should mention 'state_dimension': {msg}"
        );
    }

    /// Given a non-existent path, when `load_boundary_cuts` is called, then it
    /// returns `Err(SddpError::Validation)` with a message describing the failure.
    #[test]
    fn load_boundary_cuts_nonexistent_path_returns_error() {
        let result = load_boundary_cuts(std::path::Path::new("/nonexistent/path/to/policy"), 0, 10);

        assert!(result.is_err(), "should fail for non-existent path");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("failed to read boundary policy checkpoint"),
            "error should describe the IO failure: {msg}"
        );
    }

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
