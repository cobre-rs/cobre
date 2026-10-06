//! Policy directory and checkpointing configuration types for `config.json → policy`.

use serde::{Deserialize, Serialize};

use std::fmt;
use std::num::NonZeroU64;
use std::path::{Component, Path, PathBuf};

use crate::LoadError;
use crate::output::policy::{check_checkpoint_replaceable, checkpoint_target};
use crate::output::{Clearing, OutputError, cleared_output_dirs};

/// Policy initialization mode (`config.json → policy.mode`).
///
/// Controls whether the training phase starts from scratch, warm-starts from
/// a prior policy's rows, or resumes a checkpointed training run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub enum PolicyMode {
    /// Start training from an empty future-cost function.
    Fresh,
    /// Load rows from a prior policy checkpoint and continue training.
    WarmStart,
    /// Resume a previously interrupted training run from its checkpoint.
    Resume,
}

impl std::fmt::Display for PolicyMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PolicyMode::Fresh => f.write_str("fresh"),
            PolicyMode::WarmStart => f.write_str("warm_start"),
            PolicyMode::Resume => f.write_str("resume"),
        }
    }
}

/// What a run does with its policy directory, which decides what
/// [`PolicyConfig::check_dir`] refuses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolicyDirIntent {
    /// The run only reads the policy directory.
    Read,
    /// The run writes a checkpoint there, replacing the directory.
    Replace,
}

/// Boundary-row configuration for terminal-stage FCF coupling.
///
/// When present, the solver loads rows from a source Cobre policy
/// checkpoint and injects them as fixed boundary conditions at the
/// terminal stage of the current study. The loader selects the source pool
/// whose priced state date equals this study's last stage `end_date`.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct BoundaryPolicy {
    /// Path to the source policy checkpoint directory (a different study's
    /// output). Resolved relative to the case (input) directory, like every
    /// other input; an absolute path is used as-is. NOT relative to this run's
    /// output directory.
    pub path: String,

    /// A source slot pricing an entity or commitment this study does not
    /// model is dropped during reconciliation either way. Left `false`, the
    /// drop is recorded in the reconciliation report and the load proceeds;
    /// set `true`, the load is rejected, naming every dropping family and
    /// its count.
    #[serde(default)]
    pub strict: bool,
}

impl BoundaryPolicy {
    /// The source checkpoint directory, resolved against the CASE (input) dir —
    /// an external source checkpoint, never the current run's output dir; an
    /// absolute [`Self::path`] passes through unchanged.
    #[must_use]
    pub fn checkpoint_path(&self, case_dir: &Path) -> PathBuf {
        case_dir.join(&self.path)
    }
}

/// Policy directory settings (`config.json → policy`).
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct PolicyConfig {
    /// Policy directory, resolved against the output directory unless absolute. A checkpoint write replaces the whole directory, so an empty path, `.`, `..`, the output directory and its ancestors are refused. So is a directory that names or contains one a run clears before writing its outputs, such as `simulation/solver` or `training/solver`, or that lies inside one a run removes whole, such as `simulation/costs`.
    pub path: String,

    /// Initialization mode: `"fresh"`, `"warm_start"`, or `"resume"`.
    pub mode: PolicyMode,

    /// Checkpoint settings.
    pub checkpointing: CheckpointingConfig,

    /// Optional boundary-row policy for terminal-stage coupling.
    #[serde(default)]
    pub boundary: Option<BoundaryPolicy>,
}

impl Default for PolicyConfig {
    fn default() -> Self {
        Self {
            path: "./policy".to_string(),
            mode: PolicyMode::Fresh,
            checkpointing: CheckpointingConfig::default(),
            boundary: None,
        }
    }
}

impl PolicyConfig {
    /// Refuses a [`Self::path`] that names the output directory or one of its
    /// ancestors wherever the output directory is.
    pub(crate) fn check_path(&self, config_path: &Path) -> Result<(), LoadError> {
        let names_a_subdirectory = normalize_lexically(Path::new(&self.path))
            .components()
            .any(|component| matches!(component, Component::Normal(_)));
        if names_a_subdirectory {
            Ok(())
        } else {
            Err(output_dir_or_ancestor_refusal(config_path, &self.path))
        }
    }

    /// Refuses a [`Self::path`] that, resolved against `output_dir`, reaches the
    /// output directory or one of its ancestors, names or contains a directory a
    /// run clears before writing its outputs, or lies inside one a run removes
    /// whole. A symbolic link at the policy path is checked both where it sits
    /// and where it points. For [`PolicyDirIntent::Replace`], the policy
    /// directory must also be one a checkpoint write may replace.
    ///
    /// The comparisons are lexical, so nothing is created or canonicalised; the
    /// only link read is the one at the policy path.
    ///
    /// # Errors
    ///
    /// - [`LoadError::SchemaError`] on `policy.path`, labelled with
    ///   `config_path`, for each refusal above.
    /// - [`LoadError::IoError`] when a path cannot be made absolute, or the
    ///   policy path or its entries cannot be inspected.
    pub fn check_dir(
        &self,
        config_path: &Path,
        output_dir: &Path,
        intent: PolicyDirIntent,
    ) -> Result<(), LoadError> {
        self.check_path(config_path)?;

        let absolute = |path: &Path, reported: &Path| {
            std::path::absolute(path)
                .map(|absolute| normalize_lexically(&absolute))
                .map_err(|source| LoadError::IoError {
                    path: reported.to_path_buf(),
                    source,
                })
        };
        let to_load_error = |err: OutputError| match err {
            OutputError::IoError { path, source } => LoadError::IoError { path, source },
            other => LoadError::SchemaError {
                path: config_path.to_path_buf(),
                field: "policy.path".to_string(),
                message: other.to_string(),
            },
        };

        let out = absolute(output_dir, output_dir)?;
        let link = output_dir.join(&self.path);
        check_against_output_dir(config_path, &self.path, &absolute(&link, output_dir)?, &out)?;

        let target = checkpoint_target(&link).map_err(to_load_error)?;
        if target != link {
            let target_dir = absolute(&target, &target)?;
            let value = format!("{} -> {}", self.path, target_dir.display());
            check_against_output_dir(config_path, &value, &target_dir, &out)?;
        }

        if intent == PolicyDirIntent::Replace {
            check_checkpoint_replaceable(&link).map_err(to_load_error)?;
        }
        Ok(())
    }
}

/// `dir` and `out` must be absolute and lexically normalised.
fn check_against_output_dir(
    config_path: &Path,
    value: &str,
    dir: &Path,
    out: &Path,
) -> Result<(), LoadError> {
    if out.starts_with(dir) {
        return Err(output_dir_or_ancestor_refusal(config_path, value));
    }
    for (cleared_dir, clearing) in cleared_output_dirs() {
        let cleared = normalize_lexically(&out.join(&cleared_dir));
        let relation = if dir == cleared {
            "names"
        } else if cleared.starts_with(dir) {
            "contains"
        } else if clearing == Clearing::WholeTree && dir.starts_with(&cleared) {
            "lies inside"
        } else {
            continue;
        };
        return Err(cleared_dir_refusal(
            config_path,
            value,
            relation,
            &cleared_dir,
            clearing,
        ));
    }
    Ok(())
}

fn normalize_lexically(path: &Path) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir
                if matches!(
                    normalized.components().next_back(),
                    Some(Component::Normal(_))
                ) =>
            {
                normalized.pop();
            }
            Component::ParentDir
            | Component::Normal(_)
            | Component::RootDir
            | Component::Prefix(_) => normalized.push(component),
        }
    }
    normalized
}

fn output_dir_or_ancestor_refusal(config_path: &Path, value: &str) -> LoadError {
    LoadError::SchemaError {
        path: config_path.to_path_buf(),
        field: "policy.path".to_string(),
        message: format!(
            "{value:?} names the output directory or one of its ancestors, which a \
             checkpoint write would replace; choose another directory, such as \"./policy\""
        ),
    }
}

fn cleared_dir_refusal(
    config_path: &Path,
    value: &str,
    relation: &str,
    cleared_dir: &Path,
    clearing: Clearing,
) -> LoadError {
    // Output-relative, so spelled with `/` on every platform like the other cleared directories.
    let cleared_dir = cleared_dir
        .components()
        .map(|c| c.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/");
    let message = match clearing {
        Clearing::WholeTree => format!(
            "{value:?} {relation} {cleared_dir}, which a run removes whole before writing its \
             outputs; choose a directory that neither contains nor lies inside it, such as \
             \"./policy\""
        ),
        Clearing::NamedFiles => format!(
            "{value:?} {relation} {cleared_dir}, which holds files a run writes and a checkpoint \
             write would replace; choose a directory that neither names nor contains it, such \
             as \"./policy\""
        ),
    };
    LoadError::SchemaError {
        path: config_path.to_path_buf(),
        field: "policy.path".to_string(),
        message,
    }
}

/// Periodic checkpoint settings (`config.json → policy.checkpointing`). Each periodic checkpoint replaces the previous one in the policy directory, so only the latest is kept.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(default, deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct CheckpointingConfig {
    /// Write periodic checkpoints during training. Off when absent. When true, `interval_iterations` must be at least 1.
    #[serde(default)]
    pub enabled: Option<bool>,

    /// Iteration that writes the first periodic checkpoint. Defaults to `interval_iterations`. Iteration numbers are absolute: a resumed run continues the numbering of the run it resumes.
    #[serde(default)]
    pub initial_iteration: Option<u32>,

    /// Iterations between periodic checkpoints, counted from `initial_iteration`. Required, and at least 1, when `enabled` is true.
    #[serde(default)]
    pub interval_iterations: Option<u32>,

    /// Include LP basis in checkpoints for warm-start.
    #[serde(default)]
    pub store_basis: Option<bool>,

    /// Compress checkpoint files.
    #[serde(default)]
    pub compress: Option<bool>,
}

/// Resolved periodic checkpoint schedule; [`Config::checkpoint_schedule`](super::Config::checkpoint_schedule)
/// is its only producer.
///
/// A periodic checkpoint is written at [`Self::first_iteration`] and every
/// [`Self::interval`] iterations after it. Iteration numbers are absolute, so a
/// resumed run keeps the schedule of the run it resumes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointSchedule {
    pub(super) first_iteration: u64,
    pub(super) interval: NonZeroU64,
}

impl CheckpointSchedule {
    /// Absolute iteration that writes the first periodic checkpoint.
    #[must_use]
    pub fn first_iteration(self) -> u64 {
        self.first_iteration
    }

    /// Iterations between periodic checkpoints.
    #[must_use]
    pub fn interval(self) -> NonZeroU64 {
        self.interval
    }

    /// Whether the absolute `iteration` writes a periodic checkpoint.
    #[must_use]
    pub fn fires_at(self, iteration: u64) -> bool {
        iteration >= self.first_iteration && (iteration - self.first_iteration) % self.interval == 0
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::CheckpointSchedule;
    use std::num::NonZeroU64;

    #[test]
    fn schedule_fires_at_the_first_iteration_and_every_interval_after() {
        let schedule = CheckpointSchedule {
            first_iteration: 3,
            interval: NonZeroU64::new(2).unwrap(),
        };
        for iteration in [3, 5, 7] {
            assert!(schedule.fires_at(iteration), "must fire at {iteration}");
        }
        for iteration in [1, 2, 4, 6] {
            assert!(
                !schedule.fires_at(iteration),
                "must not fire at {iteration}"
            );
        }
    }
}
