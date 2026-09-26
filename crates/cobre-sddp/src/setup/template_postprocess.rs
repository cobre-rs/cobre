//! Template post-processing: discount factors and LP scaling.

use cobre_core::{EntityId, System};

use crate::block_clock::BlockClock;
use crate::lp::builder::{self, StageTemplates};
use crate::lp::indexer::StateSpace;
use crate::scaling_report::{
    LpDimensions, ScalingReport, StageScalingReport, build_scaling_report,
    compute_coefficient_range, summarize_scale_factors,
};
use crate::time_value::TimeValue;

/// Apply discount factors and LP scaling to stage templates.
///
/// Returns a [`ScalingReport`] with pre/post coefficient ranges.
pub(crate) fn postprocess_templates(
    stage_templates: &mut StageTemplates,
    system: &System,
    state_layout: &StateSpace,
    cost_scale_factor: f64,
) -> ScalingReport {
    let study_stages: Vec<_> = system.stages().iter().filter(|s| s.id >= 0).collect();

    // Commitment-hold resolution context the box builder's one hand-written
    // special case needs — resolved once here and threaded
    // through every stage's `build_state_box` call, mirroring the same
    // `TimeValue::resolve` inputs `build_stage_templates` uses. Runs
    // BEFORE column scaling below: the storage/transit-bucket identity families
    // read `template.col_lower`/`col_upper` verbatim, which must still be the
    // PHYSICAL bounds `apply_col_scale` has not yet divided in place — the same
    // physical units as the unscaled trial state (`fill_unscaled` in
    // `training/forward/stage_solve.rs`) and the raw commitment-hold bound.
    let bounds = system.bounds();
    let mut anticipated_thermal_indices: Vec<usize> = Vec::new();
    let mut anticipated_windows: Vec<(Option<i32>, Option<i32>)> = Vec::new();
    let mut anticipated_thermal_ids: Vec<EntityId> = Vec::new();
    for (t_idx, thermal) in system.thermals().iter().enumerate() {
        if thermal.anticipated_config.is_some() {
            anticipated_thermal_indices.push(t_idx);
            anticipated_windows.push((thermal.entry_stage_id, thermal.exit_stage_id));
            anticipated_thermal_ids.push(thermal.id);
        }
    }

    let study_total_hours: Vec<f64> = study_stages
        .iter()
        .map(|s| BlockClock::new(s).total_hours())
        .collect();
    let time_value = TimeValue::resolve(
        &study_stages,
        &study_total_hours,
        system.post_study_stages(),
        &anticipated_thermal_ids,
        system.policy_graph(),
    );
    // The setter derives cumulative factors in the same call, so the two slices
    // cannot drift.
    stage_templates.set_discount_factors(time_value.discount_factors().to_vec());

    debug_assert_eq!(
        stage_templates.cumulative_discount_factors().len(),
        stage_templates.templates.len(),
        "cumulative_discount_factors must have length n_stages after postprocess"
    );

    // Discount theta before column/row scaling: cost scaling divides c_i by K but
    // leaves theta untouched, so the two must not be folded together.
    //
    // Use the per-stage `StageGeometry::theta_col` (= authoritative
    // `StageLayout::col_theta()`), NOT a re-derivation from `n_state`/`n_hydros`:
    // that hand arithmetic omits the commitment-hold region's `commit_out`/
    // `commit_in` blocks and, with anticipated thermals, lands on a
    // zero-cost `storage_in` column, silently disabling discounting (`0 * d = 0`).
    {
        let theta_cols: Vec<usize> = stage_templates
            .geometry_per_stage
            .iter()
            .map(|g| g.theta_col)
            .collect();
        let discount_factors = stage_templates.discount_factors().to_vec();
        debug_assert_eq!(
            theta_cols.len(),
            stage_templates.templates.len(),
            "geometry_per_stage must be populated and aligned with templates",
        );
        for (s_idx, tmpl) in stage_templates.templates.iter_mut().enumerate() {
            tmpl.objective[theta_cols[s_idx]] *= discount_factors[s_idx];
        }
    }

    for stage_idx in 0..stage_templates.templates.len() {
        let state_box = builder::build_state_box(
            &stage_templates.templates[stage_idx],
            state_layout,
            stage_idx,
            bounds,
            &anticipated_thermal_indices,
            &anticipated_windows,
            time_value.delivery_stage_ids(),
            time_value.post_study(),
        );
        stage_templates.state_boxes.push(state_box);
    }

    // Column scaling then row scaling (D_r * A * D_c). Scale factors are stored on
    // the template for unscaling primal/dual solutions in the forward/backward passes.
    let mut stage_scaling_reports = Vec::with_capacity(stage_templates.templates.len());

    for (stage_id, tmpl) in stage_templates.templates.iter_mut().enumerate() {
        let pre_scaling = compute_coefficient_range(tmpl);

        let mut col_scale =
            builder::compute_col_scale(tmpl.num_cols, &tmpl.col_starts, &tmpl.values);
        builder::apply_commitment_hold_col_scale_unscale(&mut col_scale, state_layout);
        builder::apply_col_scale(tmpl, &col_scale);
        tmpl.col_scale.clone_from(&col_scale);
        let row_scale = builder::compute_row_scale(
            tmpl.num_rows,
            tmpl.num_cols,
            &tmpl.col_starts,
            &tmpl.row_indices,
            &tmpl.values,
        );
        builder::apply_row_scale(tmpl, &row_scale);
        tmpl.row_scale.clone_from(&row_scale);

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

    build_scaling_report(cost_scale_factor, stage_scaling_reports)
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp
)]
mod tests {
    use super::postprocess_templates;
    use crate::lp::builder::{StageGeometry, StageTemplates};
    use crate::lp::indexer::StateSpace;
    use crate::test_support::state_layout_full;
    use chrono::NaiveDate;
    use cobre_core::temporal::{
        BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig, StageStateConfig,
    };
    use cobre_core::{ResolvedBounds, SystemBuilder};
    use cobre_solver::StageTemplate;

    fn one_year_stage(id: i32) -> Stage {
        Stage {
            index: 0,
            id,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(),
            season_id: None,
            blocks: vec![],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: true,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        }
    }

    /// A minimal 4-column template (`storage`, then 3 filler columns padding
    /// out to `theta`'s index — `apply_commitment_hold_col_scale_unscale`
    /// requires `col_scale` to cover every state column through `theta`)
    /// whose storage column carries two matrix entries of different
    /// magnitude (`1.0`, `4.0`), forcing `compute_col_scale` to a non-unit
    /// factor (`1/sqrt(4*1) = 0.5`).
    fn scaled_storage_template(physical_upper: f64) -> StageTemplate {
        StageTemplate {
            num_cols: 4,
            num_rows: 2,
            num_nz: 2,
            col_starts: vec![0, 2, 2, 2, 2],
            row_indices: vec![0, 1],
            values: vec![1.0, 4.0],
            col_lower: vec![0.0, f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY],
            col_upper: vec![physical_upper, f64::INFINITY, f64::INFINITY, f64::INFINITY],
            objective: vec![0.0; 4],
            row_lower: vec![0.0, 0.0],
            row_upper: vec![0.0, 0.0],
            n_state: 1,
            n_transfer: 0,
            n_dual_relevant: 0,
            n_hydro: 0,
            max_par_order: 0,
            col_scale: Vec::new(),
            row_scale: Vec::new(),
        }
    }

    /// Regression: `state_boxes`'s storage bound must be the PHYSICAL
    /// `col_upper` the template started with, never `col_upper / col_scale` —
    /// `build_state_box` must read the identity families before
    /// `apply_col_scale` divides them in place.
    #[test]
    fn postprocess_templates_storage_box_is_physical_not_scaled() {
        const PHYSICAL_UPPER: f64 = 200.0;

        let mut stage_templates = StageTemplates::empty(0, 1.0);
        stage_templates
            .templates
            .push(scaled_storage_template(PHYSICAL_UPPER));
        stage_templates
            .geometry_per_stage
            .push(StageGeometry::default());

        let state_layout: StateSpace = state_layout_full(1, 0, 0, 0, Vec::new());
        let system = SystemBuilder::new()
            .stages(vec![one_year_stage(0)])
            .bounds(ResolvedBounds::empty())
            .build()
            .expect("minimal system must build");

        postprocess_templates(&mut stage_templates, &system, &state_layout, 1.0);

        assert_ne!(
            stage_templates.templates[0].col_scale[0], 1.0,
            "storage's col_scale must be != 1.0 for this regression to be meaningful"
        );
        let storage_j = state_layout.storage.start;
        assert_eq!(
            stage_templates.state_boxes[0].upper[storage_j], PHYSICAL_UPPER,
            "the storage box's upper bound must be the physical max_storage, \
             not col_upper / col_scale"
        );
    }
}
