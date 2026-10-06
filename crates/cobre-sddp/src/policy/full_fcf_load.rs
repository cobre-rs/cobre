//! The single full-FCF policy load shared by the CLI and the Python bindings.
//!
//! A warm-start, resume or simulation-only run loads a stored checkpoint in
//! two steps: [`check_full_fcf_load`] reads, validates and decodes it without
//! touching the study, and the returned [`CheckedFullFcfLoad`] is then applied.
//! Every MPI rank runs the load itself; no step is collective.

use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use cobre_core::System;
use cobre_io::output::policy::{
    ResolvedCheckpoint, read_policy_checkpoint, resolve_policy_checkpoint,
};
use cobre_io::{EntitySlot, OutputError, ProducerBlock};

use crate::cut::fcf::FutureCostFunction;
use crate::error::{ErrorClass, SddpError};
use crate::policy::policy_load::{
    FullFcf, PolicyStageManifest, StoredBasisLoad, UnusedStoredBases,
    build_basis_cache_from_checkpoint, checkpoint_terminal_cost_scale_factor,
    rescale_checkpoint_cuts_for_load, validate_policy_load,
};
use crate::setup::StudySetup;
use crate::training::training::TrainingResult;
use crate::workspace::workspace::CapturedBasis;

/// Which run consumes the loaded policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FullFcfLoadKind {
    /// Train on top of a prior policy's cuts.
    WarmStart,
    /// Continue a prior training run from its checkpoint.
    Resume,
    /// Simulate from a trained policy without training.
    SimulationOnly,
}

impl FullFcfLoadKind {
    fn unmet_requirement(self) -> &'static str {
        match self {
            Self::WarmStart => "Cannot warm-start without a prior policy.",
            Self::Resume => "Cannot resume without a prior checkpoint.",
            Self::SimulationOnly => "Cannot run simulation-only mode without a trained policy.",
        }
    }
}

/// The step of the load that failed.
#[derive(Debug, thiserror::Error)]
pub enum FullFcfLoadError {
    /// The policy directory does not exist.
    #[error("Policy directory not found: {}. {}", path.display(), kind.unmet_requirement())]
    MissingPolicyDirectory {
        /// The load that needed the directory.
        kind: FullFcfLoadKind,
        /// The directory that was looked up.
        path: PathBuf,
    },
    /// The checkpoint could not be read.
    #[error("failed to read policy checkpoint: {source}")]
    Read {
        /// The underlying read failure.
        source: OutputError,
    },
    /// The checkpoint is incompatible with the study: cost scale, software or
    /// manifests.
    #[error(transparent)]
    Refused(SddpError),
    /// The future-cost function could not be built from the cuts.
    #[error("{source}")]
    FcfConstruction {
        /// The load that was building the FCF.
        kind: FullFcfLoadKind,
        /// The underlying construction failure.
        source: SddpError,
    },
}

impl FullFcfLoadError {
    /// The [`ErrorClass`] of this failure.
    #[must_use]
    pub fn class(&self) -> ErrorClass {
        match self {
            Self::MissingPolicyDirectory { .. } => ErrorClass::InvalidInput,
            Self::Read { source } => match source {
                OutputError::IoError { source, .. } if source.kind() != ErrorKind::NotFound => {
                    ErrorClass::Io
                }
                OutputError::IoError { .. }
                | OutputError::SerializationError { .. }
                | OutputError::SchemaError { .. }
                | OutputError::ManifestError { .. } => ErrorClass::IncompatiblePolicy,
                OutputError::ForeignEntry { .. } => ErrorClass::InvalidInput,
            },
            // The inner `SddpError::Validation` classifies as `InvalidInput`, but a
            // refused checkpoint is an incompatible policy.
            Self::Refused(_) | Self::FcfConstruction { .. } => ErrorClass::IncompatiblePolicy,
        }
    }
}

/// Locate the policy directory of `setup` under `output_dir`.
///
/// The directory counts as present when [`resolve_policy_checkpoint`] finds it
/// or a committed sibling copy, beside a link's target when it is a symbolic
/// link.
///
/// # Errors
///
/// - [`FullFcfLoadError::MissingPolicyDirectory`] when the directory is absent
///   and no sibling copy is committed.
/// - [`FullFcfLoadError::Read`] when a probe fails for a reason other than
///   absence.
pub fn locate_policy_dir(
    kind: FullFcfLoadKind,
    output_dir: &Path,
    setup: &StudySetup,
) -> Result<PathBuf, FullFcfLoadError> {
    let path = output_dir.join(&setup.policy_path);
    match resolve_policy_checkpoint(&path) {
        Ok(ResolvedCheckpoint::NoDirectory) => {
            Err(FullFcfLoadError::MissingPolicyDirectory { kind, path })
        }
        Ok(ResolvedCheckpoint::NoManifest | ResolvedCheckpoint::Found(_)) => Ok(path),
        Err(source) => Err(FullFcfLoadError::Read { source }),
    }
}

/// A checkpoint that passed every check, ready to be applied.
#[derive(Debug)]
pub struct CheckedFullFcfLoad {
    kind: FullFcfLoadKind,
    fcf: FutureCostFunction,
    basis_cache: Vec<Option<CapturedBasis>>,
    unused_stored_bases: Option<UnusedStoredBases>,
    stored_basis_records: usize,
    producer: ProducerBlock,
}

/// Read, validate and decode the checkpoint in `policy_dir` against `setup`.
///
/// `on_warning` receives each compatibility warning right after validation,
/// then, once the basis cache is built, one warning when stored bases do not fit
/// the study. `setup` is not mutated.
///
/// # Errors
///
/// The [`FullFcfLoadError`] variant naming the failing step.
pub fn check_full_fcf_load(
    kind: FullFcfLoadKind,
    policy_dir: &Path,
    system: &System,
    setup: &StudySetup,
    on_warning: &mut dyn FnMut(&str),
) -> Result<CheckedFullFcfLoad, FullFcfLoadError> {
    let mut checkpoint =
        read_policy_checkpoint(policy_dir).map_err(|source| FullFcfLoadError::Read { source })?;
    let source_cost_scale_factor =
        checkpoint_terminal_cost_scale_factor(&checkpoint).map_err(FullFcfLoadError::Refused)?;
    rescale_checkpoint_cuts_for_load(
        &mut checkpoint.stage_cuts,
        Some(source_cost_scale_factor),
        setup.inputs.stage_data.stage_templates.cost_scale_factor,
    );

    // Rationale: the cast cannot truncate — `n_stages` is the validated study
    // horizon (a `u16`-scale stage count), far below `u32::MAX`.
    #[allow(clippy::cast_possible_truncation)]
    let n_stages = system.stages().iter().filter(|s| s.id >= 0).count() as u32;
    // Rationale: the cast cannot truncate — `state_dimension` counts FCF state
    // variables (one per reservoir/lag), bounded by the validated study
    // dimensions and far below `u32::MAX`.
    #[allow(clippy::cast_possible_truncation)]
    let state_dim = setup.fcf.state_dimension as u32;

    // The terminal pool is always full-config, so its manifest witnesses every
    // state family's slot identity — a terminal-only comparison covers all stages.
    let current_manifest = setup.build_terminal_entity_manifest(system);
    let checkpoint_terminal_manifest: &[EntitySlot] = checkpoint
        .stage_cuts
        .last()
        .map_or(&[], |s| s.entity_manifest.as_slice());
    let source_state_dim = checkpoint
        .stage_cuts
        .last()
        .map_or(0, |s| s.state_dimension);
    let source_graph = &checkpoint.metadata.graph_manifest;
    let current_graph = setup.build_graph_manifest();

    let source = PolicyStageManifest {
        state_dimension: source_state_dim,
        num_stages: checkpoint.metadata.num_stages,
        n_pools: source_graph.n_pools,
        slots: checkpoint_terminal_manifest,
        graph: source_graph,
    };
    let current = PolicyStageManifest {
        state_dimension: state_dim,
        num_stages: n_stages,
        n_pools: current_graph.n_pools,
        slots: &current_manifest,
        graph: &current_graph,
    };
    let proof =
        validate_policy_load::<FullFcf>(checkpoint.metadata.written_by(), &source, &current)
            .map_err(FullFcfLoadError::Refused)?;
    for msg in &proof.warnings {
        on_warning(msg);
    }

    // Reuse per-pool dimensions from the current study's FCF, not the checkpoint's.
    let pool_state_dimensions: Vec<usize> =
        setup.fcf.pools.iter().map(|p| p.state_dimension).collect();
    let fcf = match kind {
        FullFcfLoadKind::WarmStart | FullFcfLoadKind::Resume => {
            let visit_bounds: Vec<u64> = setup
                .fcf
                .pools
                .iter()
                .map(|p| u64::from(p.visit_stride))
                .collect();
            // Reserve one extra slot for cuts added in the final iteration.
            FutureCostFunction::new_with_warm_start(
                &proof,
                &checkpoint.stage_cuts,
                &pool_state_dimensions,
                &visit_bounds,
                setup.loop_params.forward_passes,
                setup.loop_params.max_iterations.saturating_add(1),
            )
        }
        FullFcfLoadKind::SimulationOnly => FutureCostFunction::from_deserialized(
            &proof,
            &checkpoint.stage_cuts,
            &pool_state_dimensions,
        ),
    }
    .map_err(|source| FullFcfLoadError::FcfConstruction { kind, source })?;

    let StoredBasisLoad {
        cache: basis_cache,
        unused: unused_stored_bases,
    } = build_basis_cache_from_checkpoint(&checkpoint.stage_bases, &checkpoint.stage_cuts, setup);
    if let Some(unused) = &unused_stored_bases {
        on_warning(&unused.to_string());
    }

    Ok(CheckedFullFcfLoad {
        kind,
        fcf,
        basis_cache,
        unused_stored_bases,
        stored_basis_records: checkpoint.stage_bases.len(),
        producer: checkpoint.metadata.producer,
    })
}

impl CheckedFullFcfLoad {
    /// Training iterations the checkpoint records as completed.
    #[must_use]
    pub fn completed_iterations(&self) -> u64 {
        u64::from(self.producer.completed_iterations)
    }

    /// The stored bases the load left out, when any did not fit the study.
    #[must_use]
    pub fn unused_stored_bases(&self) -> Option<&UnusedStoredBases> {
        self.unused_stored_bases.as_ref()
    }

    /// Install the loaded FCF, the stored basis cache (when the checkpoint
    /// holds bases) and, for resume, the completed iterations and their recorded
    /// lower bounds into `setup`.
    pub fn apply_to_training(self, setup: &mut StudySetup) {
        debug_assert!(self.kind != FullFcfLoadKind::SimulationOnly);
        let completed = self.completed_iterations();
        setup.replace_fcf(self.fcf);
        // No stored bases (the training run captured none) → iteration 1
        // cold-starts.
        if self.stored_basis_records > 0 {
            setup.set_warm_start_basis_cache(self.basis_cache);
        }
        if self.kind == FullFcfLoadKind::Resume {
            setup.set_resume_point(completed, self.producer.lower_bound_history);
        }
    }

    /// The loaded FCF and the synthetic training result that stands in for a
    /// training run in simulation-only mode.
    #[must_use]
    pub fn into_simulation_policy(self) -> (FutureCostFunction, TrainingResult) {
        debug_assert_eq!(self.kind, FullFcfLoadKind::SimulationOnly);
        let result = TrainingResult::new(
            self.producer.final_lower_bound,
            self.producer.best_upper_bound.unwrap_or(f64::INFINITY),
            0.0,
            0.0,
            self.producer.completed_iterations.into(),
            "loaded from checkpoint".to_string(),
            0,
            self.basis_cache,
            Vec::new(),
            None,
            // Checkpoints store no frozen templates; `simulate()` re-freezes from the
            // FCF row pool when this is None.
            None,
        );
        (self.fcf, result)
    }
}

#[cfg(test)]
mod tests {
    use std::io;

    use super::*;

    const KINDS: [FullFcfLoadKind; 3] = [
        FullFcfLoadKind::WarmStart,
        FullFcfLoadKind::Resume,
        FullFcfLoadKind::SimulationOnly,
    ];

    fn read_io_error(kind: ErrorKind) -> FullFcfLoadError {
        FullFcfLoadError::Read {
            source: OutputError::IoError {
                path: PathBuf::from("policy/manifest.bin"),
                source: io::Error::from(kind),
            },
        }
    }

    #[test]
    fn missing_policy_directory_classifies_as_invalid_input() {
        for kind in KINDS {
            let err = FullFcfLoadError::MissingPolicyDirectory {
                kind,
                path: PathBuf::from("out/policy"),
            };
            assert_eq!(err.class(), ErrorClass::InvalidInput, "{kind:?}");
        }
    }

    #[test]
    fn policy_file_not_found_classifies_as_incompatible_policy() {
        assert_eq!(
            read_io_error(ErrorKind::NotFound).class(),
            ErrorClass::IncompatiblePolicy
        );
    }

    #[test]
    fn unparseable_policy_file_classifies_as_incompatible_policy() {
        let sources = [
            OutputError::SerializationError {
                entity: "checkpoint_manifest".to_string(),
                message: "missing file identifier".to_string(),
            },
            OutputError::SchemaError {
                file: "cuts/0.bin".to_string(),
                column: "coefficients".to_string(),
                message: "unexpected type".to_string(),
            },
            OutputError::ManifestError {
                manifest_type: "checkpoint".to_string(),
                message: "inconsistent dates".to_string(),
            },
        ];
        for source in sources {
            let err = FullFcfLoadError::Read { source };
            assert_eq!(err.class(), ErrorClass::IncompatiblePolicy, "{err}");
        }
    }

    #[test]
    fn os_refused_policy_read_classifies_as_io() {
        assert_eq!(
            read_io_error(ErrorKind::PermissionDenied).class(),
            ErrorClass::Io
        );
    }

    #[test]
    fn refusals_and_fcf_construction_classify_as_incompatible_policy() {
        let mut errors = vec![
            FullFcfLoadError::Refused(SddpError::Validation("state_dimension mismatch".into())),
            FullFcfLoadError::Refused(SddpError::PolicySoftwareMismatch {
                policy_software: None,
                policy_version: "0.0.1".to_string(),
            }),
        ];
        errors.extend(KINDS.map(|kind| FullFcfLoadError::FcfConstruction {
            kind,
            source: SddpError::Validation("stage_results is empty".into()),
        }));
        for err in errors {
            assert_eq!(err.class(), ErrorClass::IncompatiblePolicy, "{err}");
        }
    }
}
