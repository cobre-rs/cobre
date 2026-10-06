//! Policy load/warm-start/resume phase for `cobre run`.

use std::path::Path;

use cobre_comm::Communicator;
use cobre_core::System;
use cobre_io::Config;
use cobre_io::OwnedPolicyCutRecord;
use cobre_io::PolicyMode;
use cobre_io::PolicyMode::Fresh;
use cobre_io::PolicyMode::Resume;
use cobre_io::PolicyMode::WarmStart;
use cobre_sddp::StudySetup;
use cobre_sddp::TrainingResult;
use cobre_sddp::ValidatedBoundaryCuts;
use cobre_sddp::inject_boundary_cuts;
use cobre_sddp::policy::full_fcf_load::CheckedFullFcfLoad;
use cobre_sddp::policy::full_fcf_load::FullFcfLoadKind;
use cobre_sddp::policy::full_fcf_load::check_full_fcf_load;
use cobre_sddp::policy::full_fcf_load::locate_policy_dir;
use cobre_sddp::reconcile_boundary_policy;

use crate::commands::broadcast::broadcast_value;
use crate::error::CliError;
use crate::summary::print_boundary_summary;

use super::RunContext;

fn check_policy_load(
    ctx: &RunContext<impl Communicator>,
    kind: FullFcfLoadKind,
    policy_dir: &Path,
    system: &System,
    setup: &StudySetup,
) -> Result<CheckedFullFcfLoad, CliError> {
    check_full_fcf_load(kind, policy_dir, system, setup, &mut |msg| {
        if ctx.is_root && !ctx.quiet {
            let _ = ctx.stderr.write_line(&format!("warning: {msg}"));
        }
    })
    .map_err(CliError::from)
}

/// Apply warm-start or resume policy before training, if requested.
pub(super) fn apply_training_policy(
    ctx: &RunContext<impl Communicator>,
    system: &System,
    setup: &mut StudySetup,
    root_config: Option<&Config>,
    policy_mode: PolicyMode,
) -> Result<(), CliError> {
    match policy_mode {
        WarmStart => {
            let policy_dir = locate_policy_dir(FullFcfLoadKind::WarmStart, &ctx.output_dir, setup)?;
            if ctx.is_root && !ctx.quiet {
                let _ = ctx
                    .stderr
                    .write_line("Loading prior policy for warm-start training...");
            }
            let checked =
                check_policy_load(ctx, FullFcfLoadKind::WarmStart, &policy_dir, system, setup)?;
            checked.apply_to_training(setup);
            if ctx.is_root && !ctx.quiet {
                // pools[0] as representative; this is not a per-pool count.
                let warm_count = setup.fcf.pools[0].warm_start_count;
                let _ = ctx.stderr.write_line(&format!(
                    "Warm-start: loaded {warm_count} cuts per stage from prior policy."
                ));
            }
        }
        Resume => {
            let policy_dir = locate_policy_dir(FullFcfLoadKind::Resume, &ctx.output_dir, setup)?;
            if ctx.is_root && !ctx.quiet {
                let _ = ctx
                    .stderr
                    .write_line("Loading prior checkpoint for resume training...");
            }
            let checked =
                check_policy_load(ctx, FullFcfLoadKind::Resume, &policy_dir, system, setup)?;
            let completed = checked.completed_iterations();
            if completed >= setup.loop_params.max_iterations && ctx.is_root && !ctx.quiet {
                let _ = ctx.stderr.write_line(&format!(
                    "WARNING: Checkpoint already completed {completed} iterations \
                     (max_iterations = {}). No additional training will occur.",
                    setup.loop_params.max_iterations
                ));
            }
            checked.apply_to_training(setup);
            if ctx.is_root && !ctx.quiet {
                let warm_count = setup.fcf.pools[0].warm_start_count;
                let _ = ctx.stderr.write_line(&format!(
                    "Resume: loaded {warm_count} cuts per stage, \
                     resuming from iteration {completed}."
                ));
            }
        }
        Fresh => {}
    }

    // Must run after the match: warm-start replaces the whole FCF first, then
    // boundary cuts overwrite only the terminal pool.
    //
    // boundary_requirements() is identical on all ranks; rank 0 reads and
    // broadcasts the reconciled cuts, then every rank injects the identical
    // terminal pool. Gating on root_config would leave non-root ranks with an
    // empty terminal pool — a rank-count-dependent wrong bound.
    if setup.boundary_requirements().is_present() {
        let boundary_records: Option<Vec<OwnedPolicyCutRecord>> = if ctx.is_root {
            let bp = root_config
                .and_then(|c| c.policy.boundary.as_ref())
                .ok_or_else(|| CliError::Internal {
                    message: "rank 0 missing policy.boundary while boundary_requirements \
                              reports present — internal invariant violated"
                        .to_string(),
                })?;
            let reconciled = reconcile_boundary_policy(setup, system, bp, &ctx.case_dir)
                .map_err(CliError::from)?;
            if !ctx.quiet {
                print_boundary_summary(
                    &ctx.stderr,
                    reconciled.cuts.len(),
                    reconciled.boundary_date,
                    &reconciled.checkpoint_path,
                    reconciled.cuts.report(),
                );
            }
            for line in reconciled.cuts.report().detail_lines() {
                tracing::debug!("{line}");
            }
            Some(reconciled.cuts.to_vec())
        } else {
            None
        };

        // Collective: rank 0 sends the reconciled records, every rank receives
        // and injects the same terminal pool.
        let boundary_records = broadcast_value(boundary_records, &ctx.comm)?;
        let validated = ValidatedBoundaryCuts::from_broadcast_records(boundary_records);
        inject_boundary_cuts(setup, &validated)?;
    }

    Ok(())
}

/// Load a policy checkpoint and build a synthetic `TrainingResult` for simulation-only mode.
pub(super) fn load_policy_for_simulation(
    ctx: &RunContext<impl Communicator>,
    system: &System,
    setup: &mut StudySetup,
) -> Result<TrainingResult, CliError> {
    if ctx.is_root && !ctx.quiet {
        let _ = ctx
            .stderr
            .write_line("Training disabled. Loading policy for simulation-only mode...");
    }

    let policy_dir = locate_policy_dir(FullFcfLoadKind::SimulationOnly, &ctx.output_dir, setup)?;

    let checked = check_policy_load(
        ctx,
        FullFcfLoadKind::SimulationOnly,
        &policy_dir,
        system,
        setup,
    )?;
    let (loaded_fcf, result) = checked.into_simulation_policy();
    setup.replace_fcf(loaded_fcf);
    Ok(result)
}
