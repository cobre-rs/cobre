//! Policy directory and checkpointing configuration types for `config.json → policy`.

use serde::{Deserialize, Serialize};

use std::fmt;
use std::num::NonZeroU64;
use std::path::{Path, PathBuf};

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
    /// Directory for policy data (rows, states, vertices, basis).
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
