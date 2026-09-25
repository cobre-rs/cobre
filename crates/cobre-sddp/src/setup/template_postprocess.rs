//! Template post-processing: discount factors and LP scaling.

use cobre_core::{EntityId, HorizonGraph, Stage, System};

use crate::lp::builder::{self, StageTemplates};
use crate::lp::indexer::StateSpace;
use crate::scaling_report::ScalingReport;
use crate::scaling_report::{
    LpDimensions, StageScalingReport, build_scaling_report, compute_coefficient_range,
    summarize_scale_factors,
};

/// Compute per-stage one-step discount factors from study stages and a policy graph.
///
/// `discount_factors[t] = 1 / (1 + r_t)^(Dt / 365.25)` where `r_t` is the annual
/// discount rate for stage `t` (`HorizonGraph::stage_discount_rate_overrides` keyed
/// by `Stage::id`, else the global `annual_discount_rate`) and `Dt` is the stage
/// duration in days. When `rate == 0.0`, the factor is `1.0` (no discounting).
pub(crate) fn compute_per_stage_discount_factors(
    study_stages: &[&Stage],
    pg: &HorizonGraph,
) -> Vec<f64> {
    study_stages
        .iter()
        .map(|stage| {
            let rate = pg
                .stage_discount_rate_overrides
                .get(&stage.id)
                .copied()
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
        .collect()
}

/// Compute cumulative discount factors from per-stage one-step factors.
///
/// Length is `per_stage.len()` exactly: the strict anticipated-decision predicate
/// (`stage_idx + K_i < n_stages`) keeps every delivery lookup within
/// `[0, n_stages)`, so no boundary-stage entry is needed.
pub(crate) fn compute_cumulative_discount_factors(per_stage: &[f64]) -> Vec<f64> {
    let n = per_stage.len();
    let mut cumulative = vec![1.0; n];
    for t in 1..n {
        cumulative[t] = cumulative[t - 1] * per_stage[t - 1];
    }
    cumulative
}

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

    // The setter derives cumulative factors in the same call, so the two slices
    // cannot drift.
    stage_templates.set_discount_factors(compute_per_stage_discount_factors(
        &study_stages,
        system.policy_graph(),
    ));

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

    // Commitment-hold resolution context the box builder's one hand-written
    // special case needs — resolved once here and threaded
    // through every stage's `build_state_box` call, mirroring the same
    // `resolve_post_study_artifacts` inputs `build_stage_templates` uses. Runs
    // BEFORE column scaling below: the storage/transit-bucket identity families
    // read `template.col_lower`/`col_upper` verbatim, which must still be the
    // PHYSICAL bounds `apply_col_scale` has not yet divided in place — the same
    // physical units as the unscaled trial state (`fill_unscaled` in
    // `training/forward/stage_solve.rs`) and the raw commitment-hold bound.
    let bounds = system.bounds();
    let mut anticipated_thermal_indices: Vec<usize> = Vec::new();
    let mut anticipated_windows: Vec<(Option<i32>, Option<i32>)> = Vec::new();
    for (t_idx, thermal) in system.thermals().iter().enumerate() {
        if thermal.anticipated_config.is_some() {
            anticipated_thermal_indices.push(t_idx);
            anticipated_windows.push((thermal.entry_stage_id, thermal.exit_stage_id));
        }
    }
    let anticipated_thermal_ids: Vec<EntityId> = anticipated_thermal_indices
        .iter()
        .map(|&idx| system.thermals()[idx].id)
        .collect();
    let last_real_cumulative = stage_templates
        .cumulative_discount_factors()
        .last()
        .copied()
        .unwrap_or(1.0);
    let last_real_per_stage = stage_templates
        .discount_factors()
        .last()
        .copied()
        .unwrap_or(1.0);
    let post_study_resolved = super::resolve_post_study_artifacts(
        system.post_study_stages(),
        &anticipated_thermal_ids,
        system.policy_graph(),
        last_real_cumulative,
        last_real_per_stage,
    );
    let study_stage_ids: Vec<i32> = study_stages.iter().map(|s| s.id).collect();
    let n_post = post_study_resolved.total_hours.len();
    let next_delivery_id = study_stage_ids.last().map_or(0, |&last| last + 1);
    let end_delivery_id =
        next_delivery_id.saturating_add(i32::try_from(n_post).unwrap_or(i32::MAX));
    let delivery_stage_ids: Vec<i32> = study_stage_ids
        .iter()
        .copied()
        .chain(next_delivery_id..end_delivery_id)
        .collect();

    for stage_idx in 0..stage_templates.templates.len() {
        let state_box = builder::build_state_box(
            &stage_templates.templates[stage_idx],
            state_layout,
            stage_idx,
            bounds,
            &anticipated_thermal_indices,
            &anticipated_windows,
            &delivery_stage_ids,
            &post_study_resolved,
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
    use super::{
        compute_cumulative_discount_factors, compute_per_stage_discount_factors,
        postprocess_templates,
    };
    use crate::lp::builder::{StageGeometry, StageTemplates};
    use crate::lp::indexer::StateSpace;
    use crate::test_support::state_layout_full;
    use chrono::NaiveDate;
    use cobre_core::temporal::{
        BlockMode, NoiseMethod, PolicyGraphType, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };
    use cobre_core::{HorizonGraph, ResolvedBounds, SystemBuilder};
    use cobre_solver::StageTemplate;
    use std::collections::BTreeMap;

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

    /// A `stages[].annual_discount_rate_override` (carried on
    /// `HorizonGraph::stage_discount_rate_overrides`) sets that stage's rate,
    /// overriding the global `annual_discount_rate` (B2).
    #[test]
    fn stage_discount_override_is_read_off_the_stage() {
        let stage = one_year_stage(0);
        let days = f64::from((stage.end_date - stage.start_date).num_days() as i32);
        let mut overrides = BTreeMap::new();
        overrides.insert(0, 0.10);
        let pg = HorizonGraph {
            graph_type: PolicyGraphType::FiniteHorizon,
            annual_discount_rate: 0.06,
            transitions: vec![],
            nodes: vec![],
            stage_discount_rate_overrides: overrides,
            season_map: None,
        };

        let factors = compute_per_stage_discount_factors(&[&stage], &pg);
        let expected = 1.0 / (1.0_f64 + 0.10).powf(days / 365.25);
        assert!(
            (factors[0] - expected).abs() < 1e-12,
            "stage override 0.10 must set the rate, got {}",
            factors[0]
        );
        let global = 1.0 / (1.0_f64 + 0.06).powf(days / 365.25);
        assert!(
            (factors[0] - global).abs() > 1e-6,
            "override must differ from the global-rate factor"
        );
    }

    /// A stage with no override falls back to the global `annual_discount_rate`.
    #[test]
    fn stage_without_override_uses_global_rate() {
        let stage = one_year_stage(0);
        let days = f64::from((stage.end_date - stage.start_date).num_days() as i32);
        let pg = HorizonGraph {
            graph_type: PolicyGraphType::FiniteHorizon,
            annual_discount_rate: 0.06,
            transitions: vec![],
            nodes: vec![],
            stage_discount_rate_overrides: BTreeMap::new(),
            season_map: None,
        };

        let factors = compute_per_stage_discount_factors(&[&stage], &pg);
        let expected = 1.0 / (1.0_f64 + 0.06).powf(days / 365.25);
        assert!(
            (factors[0] - expected).abs() < 1e-12,
            "absent override must fall back to the global rate, got {}",
            factors[0]
        );
    }

    #[test]
    fn cumulative_discount_factors_length_matches_n_stages() {
        let n_stages = 4_usize;
        let per_stage = vec![0.95_f64; n_stages];
        let cumulative = compute_cumulative_discount_factors(&per_stage);

        assert_eq!(
            cumulative.len(),
            n_stages,
            "cumulative_discount_factors length must equal n_stages = {n_stages}"
        );

        assert_eq!(cumulative[0], 1.0, "cumulative[0] == 1.0 (present value)");
        assert_eq!(
            cumulative[1], 0.95,
            "cumulative[1] == 0.95 = 1.0 * per_stage[0]"
        );
        // Approximate: repeated multiplication may differ from powi(3) by a ULP
        // (floating-point associativity).
        assert!(
            (cumulative[n_stages - 1] - 0.95_f64.powi(3)).abs() < 1e-15,
            "cumulative[n_stages-1] must be within 1e-15 of 0.95^(n_stages-1) (got {})",
            cumulative[n_stages - 1]
        );
    }

    #[test]
    fn cumulative_discount_factors_all_ones_when_rate_zero() {
        let per_stage = vec![1.0_f64; 3];
        let cumulative = compute_cumulative_discount_factors(&per_stage);
        assert_eq!(cumulative.len(), 3);
        for (i, &v) in cumulative.iter().enumerate() {
            assert_eq!(
                v, 1.0,
                "cumulative[{i}] must be 1.0 when per-stage factor is 1.0"
            );
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
