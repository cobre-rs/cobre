//! Policy load/warm-start/resume phase for `cobre run`.

use std::path::Path;

use cobre_comm::Communicator;
use cobre_core::System;
use cobre_io::Config;
use cobre_io::EntitySlot;
use cobre_io::OwnedPolicyCutRecord;
use cobre_io::PolicyMode;
use cobre_io::PolicyMode::Fresh;
use cobre_io::PolicyMode::Resume;
use cobre_io::PolicyMode::WarmStart;
use cobre_io::output::policy::read_policy_checkpoint;
use cobre_sddp::FullFcf;
use cobre_sddp::FutureCostFunction;
use cobre_sddp::PolicyLoadProof;
use cobre_sddp::PolicyStageManifest;
use cobre_sddp::StudySetup;
use cobre_sddp::TrainingResult;
use cobre_sddp::ValidatedBoundaryCuts;
use cobre_sddp::build_basis_cache_from_checkpoint;
use cobre_sddp::checkpoint_terminal_cost_scale_factor;
use cobre_sddp::inject_boundary_cuts;
use cobre_sddp::load_boundary_cuts;
use cobre_sddp::rescale_checkpoint_cuts_for_load;
use cobre_sddp::resolve_boundary_source_stage;
use cobre_sddp::validate_policy_load;

use crate::commands::broadcast::broadcast_value;
use crate::error::CliError;
use crate::summary::print_boundary_summary;

use super::RunContext;

/// Load a policy checkpoint from disk, rescale its cuts into the current
/// study's cost-scale space, and validate compatibility.
///
/// Every warm-start, resume, and simulation-only load shares this one function,
/// so [`cobre_sddp::rescale_checkpoint_cuts_for_load`] and
/// validation both run exactly once per load, unconditionally. Validation
/// routes through [`cobre_sddp::validate_policy_load`] typed to [`FullFcf`],
/// checking `state_dimension` and `num_stages`, then the checkpoint terminal
/// manifest against the current study's terminal manifest — rejecting a
/// same-dimension-different-entity policy the dims check alone would pass.
/// Returns the checkpoint alongside the resulting [`PolicyLoadProof<FullFcf>`],
/// the sole credential
/// [`FutureCostFunction::new_with_warm_start`](cobre_sddp::FutureCostFunction::new_with_warm_start)
/// and
/// [`FutureCostFunction::from_deserialized`](cobre_sddp::FutureCostFunction::from_deserialized)
/// accept.
fn load_and_validate_checkpoint(
    ctx: &RunContext<impl Communicator>,
    policy_dir: &Path,
    system: &System,
    setup: &StudySetup,
) -> Result<(cobre_io::PolicyCheckpoint, PolicyLoadProof<FullFcf>), CliError> {
    let mut checkpoint = read_policy_checkpoint(policy_dir).map_err(|e| CliError::Internal {
        message: format!("failed to read policy checkpoint: {e}"),
    })?;
    let source_cost_scale_factor =
        checkpoint_terminal_cost_scale_factor(&checkpoint).map_err(CliError::from)?;
    rescale_checkpoint_cuts_for_load(
        &mut checkpoint.stage_cuts,
        Some(source_cost_scale_factor),
        setup.stage_data.stage_templates.cost_scale_factor,
    );

    // Rationale: the cast cannot truncate — `n_stages` is the validated study
    // horizon (a `u16`-scale stage count), far below `u32::MAX`.
    #[allow(clippy::cast_possible_truncation)]
    let n_stages = system.stages().iter().filter(|s| s.id >= 0).count() as u32;
    let state_dim = u32::try_from(setup.fcf.state_dimension).map_err(|e| CliError::Internal {
        message: format!("state_dimension overflows u32: {e}"),
    })?;

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
    let proof = validate_policy_load::<FullFcf>(&source, &current).map_err(CliError::from)?;

    if ctx.is_root && !ctx.quiet {
        for msg in &proof.warnings {
            let _ = ctx.stderr.write_line(&format!("warning: {msg}"));
        }
    }

    Ok((checkpoint, proof))
}

/// Build the warm-start FCF from a loaded checkpoint and seed the basis cache.
/// Shared by the warm-start and resume paths. `proof` is the credential
/// [`load_and_validate_checkpoint`] produced for this same `checkpoint`.
fn load_checkpoint_into_setup(
    checkpoint: &cobre_io::PolicyCheckpoint,
    proof: &PolicyLoadProof<FullFcf>,
    setup: &mut StudySetup,
) -> Result<(), CliError> {
    // The pre-replacement (cold-path) FCF already carries the per-pool arrays
    // `new_per_pool` derived for this study; the resume path reuses them
    // verbatim rather than substituting a scalar.
    let pool_state_dimensions: Vec<usize> =
        setup.fcf.pools.iter().map(|p| p.state_dimension).collect();
    let visit_bounds: Vec<u64> = setup
        .fcf
        .pools
        .iter()
        .map(|p| u64::from(p.visit_stride))
        .collect();
    // Reserve one extra slot for cuts added in the final iteration.
    let warm_fcf = FutureCostFunction::new_with_warm_start(
        proof,
        &checkpoint.stage_cuts,
        &pool_state_dimensions,
        &visit_bounds,
        setup.loop_params.forward_passes,
        setup.loop_params.max_iterations.saturating_add(1),
    )
    .map_err(CliError::from)?;
    setup.replace_fcf(warm_fcf);
    // No stored bases (checkpoint written without `store_basis`) → iteration 1
    // cold-starts.
    if !checkpoint.stage_bases.is_empty() {
        let basis_cache = build_basis_cache_from_checkpoint(
            &checkpoint.stage_bases,
            &checkpoint.stage_cuts,
            &setup.node_graph.node_ids,
            &setup.node_graph.node_pool_ids(),
        );
        setup.set_warm_start_basis_cache(basis_cache);
    }
    Ok(())
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
            let policy_dir = ctx.output_dir.join(&setup.policy_path);
            if !policy_dir.exists() {
                return Err(CliError::Internal {
                    message: format!(
                        "Policy directory not found: {}. Cannot warm-start \
                         without a prior policy.",
                        policy_dir.display()
                    ),
                });
            }
            if ctx.is_root && !ctx.quiet {
                let _ = ctx
                    .stderr
                    .write_line("Loading prior policy for warm-start training...");
            }
            let (checkpoint, proof) =
                load_and_validate_checkpoint(ctx, &policy_dir, system, setup)?;
            load_checkpoint_into_setup(&checkpoint, &proof, setup)?;
            if ctx.is_root && !ctx.quiet {
                // Pool 0 (the chain's stage-0 / lowest-id pool) stands in as a
                // representative sample for this advisory message, not a
                // per-pool count.
                let warm_count = setup.fcf.pools[0].warm_start_count;
                let _ = ctx.stderr.write_line(&format!(
                    "Warm-start: loaded {warm_count} cuts per stage from prior policy."
                ));
            }
        }
        Resume => {
            let policy_dir = ctx.output_dir.join(&setup.policy_path);
            if !policy_dir.exists() {
                return Err(CliError::Internal {
                    message: format!(
                        "Policy directory not found: {}. Cannot resume \
                         without a prior checkpoint.",
                        policy_dir.display()
                    ),
                });
            }
            if ctx.is_root && !ctx.quiet {
                let _ = ctx
                    .stderr
                    .write_line("Loading prior checkpoint for resume training...");
            }
            let (checkpoint, proof) =
                load_and_validate_checkpoint(ctx, &policy_dir, system, setup)?;
            let completed = u64::from(checkpoint.metadata.producer.completed_iterations);
            if completed >= setup.loop_params.max_iterations && ctx.is_root && !ctx.quiet {
                let _ = ctx.stderr.write_line(&format!(
                    "WARNING: Checkpoint already completed {completed} iterations \
                     (max_iterations = {}). No additional training will occur.",
                    setup.loop_params.max_iterations
                ));
            }
            load_checkpoint_into_setup(&checkpoint, &proof, setup)?;
            setup.set_start_iteration(completed);
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
    // Boundary PRESENCE is a broadcast fact (`boundary_requirements`), identical
    // on every rank; the reconciled cuts are read and reconciled once on rank 0
    // (the single disk reader, the only rank carrying `root_config`) and
    // broadcast, so every rank injects the identical terminal pool. Gating on the
    // rank-0-only `root_config` instead leaves non-root ranks with an empty
    // terminal pool, so their forward/backward/simulation terminal solves drop
    // the post-horizon value-to-go — a rank-count-dependent wrong bound.
    if setup.boundary_requirements().is_present() {
        let boundary_records: Option<Vec<OwnedPolicyCutRecord>> = if ctx.is_root {
            let bp = root_config
                .and_then(|c| c.policy.boundary.as_ref())
                .ok_or_else(|| CliError::Internal {
                    message: "rank 0 missing policy.boundary while boundary_requirements \
                              reports present — internal invariant violated"
                        .to_string(),
                })?;
            let boundary_path = bp.checkpoint_path(&ctx.case_dir);
            // Rationale: the cast cannot truncate — `state_dimension` counts FCF
            // state variables (one per reservoir/lag), bounded by the validated
            // study dimensions and far below `u32::MAX`.
            #[allow(clippy::cast_possible_truncation)]
            let state_dim = setup.fcf.state_dimension as u32;
            let current_manifest = setup.build_terminal_entity_manifest(system);
            let target_delivery_intervals =
                setup.build_terminal_anticipated_delivery_intervals(system);
            let fixed_windows = setup.build_terminal_fixed_post_horizon_windows(system);
            let source_stage = if let Some(idx) = bp.source_stage {
                idx
            } else {
                let resolved =
                    resolve_boundary_source_stage(&boundary_path, &target_delivery_intervals)
                        .map_err(CliError::from)?;
                if !ctx.quiet {
                    let _ = ctx.stderr.write_line(&format!(
                        "Boundary source_stage resolved to {resolved} (no explicit \
                         policy.boundary.source_stage configured)."
                    ));
                }
                resolved
            };
            let stderr = &ctx.stderr;
            let quiet = ctx.quiet;
            let mut on_warning = |msg: &str| {
                if !quiet {
                    let _ = stderr.write_line(&format!("warning: {msg}"));
                }
            };
            // The depth the state layout already reserved (read off the constructed
            // setup, not re-inferred from the checkpoint), so the load-time depth
            // guard is a defensive check, never a user error.
            let effective_inflow_lag_depth = setup.boundary_requirements().inflow_lag_depth();
            let validated = load_boundary_cuts(
                &boundary_path,
                source_stage,
                state_dim,
                &current_manifest,
                &target_delivery_intervals,
                &fixed_windows,
                effective_inflow_lag_depth,
                setup.stage_data.stage_templates.cost_scale_factor,
                &mut on_warning,
            )
            .map_err(CliError::from)?;
            if !ctx.quiet {
                print_boundary_summary(
                    &ctx.stderr,
                    validated.len(),
                    source_stage,
                    &boundary_path,
                    validated.report(),
                );
            }
            for line in validated.report().detail_lines() {
                tracing::debug!("{line}");
            }
            Some(validated.to_vec())
        } else {
            None
        };

        // Collective: rank 0 sends the reconciled records, every rank receives
        // and injects the same terminal pool.
        let boundary_records = broadcast_value(boundary_records, &ctx.comm)?;
        let validated = ValidatedBoundaryCuts::from_broadcast_records(boundary_records);
        inject_boundary_cuts(setup, &validated);
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

    let policy_dir = ctx.output_dir.join(&setup.policy_path);
    if !policy_dir.exists() {
        return Err(CliError::Internal {
            message: format!(
                "Policy directory not found: {}. Cannot run simulation-only \
                 mode without a trained policy.",
                policy_dir.display()
            ),
        });
    }

    let (checkpoint, proof) = load_and_validate_checkpoint(ctx, &policy_dir, system, setup)?;

    let pool_state_dimensions: Vec<usize> =
        setup.fcf.pools.iter().map(|p| p.state_dimension).collect();
    let loaded_fcf = FutureCostFunction::from_deserialized(
        &proof,
        &checkpoint.stage_cuts,
        &pool_state_dimensions,
    )
    .map_err(CliError::from)?;
    setup.replace_fcf(loaded_fcf);

    let basis_cache = build_basis_cache_from_checkpoint(
        &checkpoint.stage_bases,
        &checkpoint.stage_cuts,
        &setup.node_graph.node_ids,
        &setup.node_graph.node_pool_ids(),
    );

    Ok(TrainingResult::new(
        checkpoint.metadata.producer.final_lower_bound,
        checkpoint
            .metadata
            .producer
            .best_upper_bound
            .unwrap_or(f64::INFINITY),
        0.0,
        0.0,
        checkpoint.metadata.producer.completed_iterations.into(),
        "loaded from checkpoint".to_string(),
        0,
        basis_cache,
        Vec::new(),
        None,
        // Checkpoints store no frozen templates; `simulate()` re-freezes from the FCF
        // row pool when this is None.
        None,
    ))
}
