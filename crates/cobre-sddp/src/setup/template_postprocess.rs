//! Template post-processing: discount factors, LP scaling, and noise pre-scaling.

use cobre_core::System;

use crate::scaling_report::{
    build_scaling_report, compute_coefficient_range, summarize_scale_factors, LpDimensions,
    StageScalingReport,
};
use crate::{lp_builder, StageIndexer, StageTemplates};

/// Apply discount factors, LP scaling, and noise pre-scaling to stage templates.
///
/// Returns a [`crate::scaling_report::ScalingReport`] with pre/post coefficient ranges.
pub(crate) fn postprocess_templates(
    stage_templates: &mut StageTemplates,
    system: &System,
) -> crate::scaling_report::ScalingReport {
    // Compute per-stage one-step discount factors from the PolicyGraph
    // and store in StageTemplates. This is done here (not inside
    // build_stage_templates) to avoid threading PolicyGraph through
    // the template builder's signature.
    {
        let pg = system.policy_graph();
        let study_stages: Vec<_> = system.stages().iter().filter(|s| s.id >= 0).collect();
        stage_templates.discount_factors = study_stages
            .iter()
            .map(|stage| {
                let rate = pg
                    .transitions
                    .iter()
                    .find(|tr| tr.source_id == stage.id)
                    .and_then(|tr| tr.annual_discount_rate_override)
                    .unwrap_or(pg.annual_discount_rate);
                if rate == 0.0 {
                    1.0
                } else {
                    let dt_days = f64::from(
                        i32::try_from((stage.end_date - stage.start_date).num_days())
                            .unwrap_or(i32::MAX),
                    );
                    1.0 / (1.0 + rate).powf(dt_days / 365.25)
                }
            })
            .collect();
    }

    // D_0 = 1.0, D_t = D_{t-1} * d_{t-1} for t >= 1.
    // Used by the simulation extraction layer for reporting only.
    {
        let n = stage_templates.discount_factors.len();
        let mut cumulative = vec![1.0; n];
        for t in 1..n {
            cumulative[t] = cumulative[t - 1] * stage_templates.discount_factors[t - 1];
        }
        stage_templates.cumulative_discount_factors = cumulative;
    }

    // Apply discount factors to theta objective coefficients before
    // column/row scaling. The discount factor d_t converts
    // `1.0 * theta` to `d_t * theta` in the objective, correctly
    // valuing discounted future cost. This is orthogonal to cost
    // scaling (which divides c_i by K but leaves theta untouched);
    // the discount factor multiplies that untouched 1.0 to d_t.
    // When annual_discount_rate == 0.0, d_t == 1.0 and this is a no-op.
    if let Some(first) = stage_templates.templates.first() {
        let theta_col = StageIndexer::new(stage_templates.n_hydros, first.max_par_order).theta;
        for (s_idx, tmpl) in stage_templates.templates.iter_mut().enumerate() {
            tmpl.objective[theta_col] *= stage_templates.discount_factors[s_idx];
        }
    }

    // Compute and apply column scaling, then row scaling for numerical
    // conditioning (D_r * A * D_c form). Scale factors are stored in the
    // template for unscaling primal/dual solutions in the forward and
    // backward passes.
    //
    // Scaling report: capture pre/post coefficient ranges for diagnostics.

    let mut stage_scaling_reports = Vec::with_capacity(stage_templates.templates.len());

    for (stage_id, tmpl) in stage_templates.templates.iter_mut().enumerate() {
        // Pre-scaling snapshot (before col/row scaling; cost scaling is
        // already baked into the objective during template construction).
        let pre_scaling = compute_coefficient_range(tmpl);

        let col_scale =
            lp_builder::compute_col_scale(tmpl.num_cols, &tmpl.col_starts, &tmpl.values);
        lp_builder::apply_col_scale(tmpl, &col_scale);
        tmpl.col_scale.clone_from(&col_scale);
        // Row scaling is applied to the already column-scaled matrix.
        let row_scale = lp_builder::compute_row_scale(
            tmpl.num_rows,
            tmpl.num_cols,
            &tmpl.col_starts,
            &tmpl.row_indices,
            &tmpl.values,
        );
        lp_builder::apply_row_scale(tmpl, &row_scale);
        tmpl.row_scale.clone_from(&row_scale);

        // Post-scaling snapshot (after col + row scaling).
        let post_scaling = compute_coefficient_range(tmpl);

        stage_scaling_reports.push(StageScalingReport {
            stage_id,
            dimensions: LpDimensions {
                num_cols: tmpl.num_cols,
                num_rows: tmpl.num_rows,
                num_nz: tmpl.num_nz,
            },
            pre_scaling,
            post_scaling,
            col_scale: summarize_scale_factors(&col_scale),
            row_scale: summarize_scale_factors(&row_scale),
        });
    }

    let scaling_report = build_scaling_report(lp_builder::COST_SCALE_FACTOR, stage_scaling_reports);

    // Pre-scale noise_scale by row_scale so that the inflow noise
    // perturbation (noise_scale * eta) is in the same scaled units as
    // the template row bounds (which were already row-scaled above).
    // Without this, transform_inflow_noise would produce a mixed-scale
    // RHS: scaled base + unscaled perturbation.
    let n_hydros_noise = stage_templates.n_hydros;
    for (s_idx, tmpl) in stage_templates.templates.iter().enumerate() {
        if !tmpl.row_scale.is_empty() {
            let base_row = stage_templates.base_rows[s_idx];
            for h in 0..n_hydros_noise {
                stage_templates.noise_scale[s_idx * n_hydros_noise + h] *=
                    tmpl.row_scale[base_row + h];
            }
        }
    }

    scaling_report
}
