//! Template post-processing: discount factors and LP scaling.

use cobre_core::System;

use crate::lp::builder::{self, StageTemplates};
use crate::lp::indexer::{AnticipatedPlants, StateSpace};
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
    anticipated_plants: &AnticipatedPlants,
    cost_scale_factor: f64,
    time_value: &TimeValue,
) -> ScalingReport {
    // Commitment-hold resolution context the box builder's one hand-written
    // special case needs — resolved once here and threaded through every
    // stage's `build_state_box` call. Runs BEFORE column scaling below: the
    // storage/transit-bucket identity families read `template.col_lower`/
    // `col_upper` verbatim, which must still be the PHYSICAL bounds
    // `apply_col_scale` has not yet divided in place — the same physical units
    // as the unscaled trial state (`fill_unscaled` in
    // `training/forward/stage_solve.rs`) and the raw commitment-hold bound.
    let bounds = system.bounds();
    let anticipated_windows = super::build_anticipated_windows(system, anticipated_plants);

    debug_assert_eq!(
        time_value.discount_factors().len(),
        stage_templates.templates.len(),
        "time_value.discount_factors must have length n_stages"
    );

    // Discount theta before column/row scaling: cost scaling divides c_i by K but
    // leaves theta untouched, so the two must not be folded together.
    //
    // Use `state_layout.theta` (its single owner), NOT a re-derivation from
    // `n_state`/`n_hydros`: that hand arithmetic omits the commitment-hold
    // region's `commit_out`/`commit_in` blocks and, with anticipated thermals,
    // lands on a zero-cost `storage_in` column, silently disabling discounting
    // (`0 * d = 0`).
    {
        let discount_factors = time_value.discount_factors();
        for (s_idx, tmpl) in stage_templates.templates.iter_mut().enumerate() {
            tmpl.objective[state_layout.theta] *= discount_factors[s_idx];
        }
    }

    for stage_idx in 0..stage_templates.templates.len() {
        let state_box = builder::build_state_box(
            &stage_templates.templates[stage_idx],
            state_layout,
            stage_idx,
            bounds,
            anticipated_plants,
            &anticipated_windows,
            time_value,
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
    use crate::lp::indexer::{AnticipatedPlants, StateSpace};
    use crate::test_support::state_layout_full;
    use crate::time_value::{PostStudyResolved, TimeValue};
    use chrono::NaiveDate;
    use cobre_core::temporal::{
        BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig, StageStateConfig,
    };
    use cobre_core::{AnticipatedConfig, Bus, EntityId, ResolvedBounds, SystemBuilder, Thermal};
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

        let state_layout: StateSpace = state_layout_full(1, 0, 0, Vec::new());
        let system = SystemBuilder::new()
            .stages(vec![one_year_stage(0)])
            .bounds(ResolvedBounds::empty())
            .build()
            .expect("minimal system must build");
        let time_value = TimeValue::from_parts(
            vec![1.0],
            vec![1.0],
            vec![0.0],
            vec![0],
            PostStudyResolved::default(),
        );

        let anticipated_plants = AnticipatedPlants::build(system.thermals());
        postprocess_templates(
            &mut stage_templates,
            &system,
            &state_layout,
            &anticipated_plants,
            1.0,
            &time_value,
        );

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

    /// θ's discount must land on `state_layout.theta`, never a hand
    /// re-derivation from `n_state`/`n_hydros`: this fixture's commitment-hold
    /// region (one anticipated thermal, `k_max = 1`) shifts `theta` off both
    /// (`n_state == 1`, `n_hydros == 0`, `theta == 2`), so a wrong
    /// re-derivation would silently discount the wrong column.
    #[test]
    fn theta_discount_lands_on_the_state_theta_column_with_anticipated_thermals() {
        let state_layout: StateSpace = state_layout_full(0, 0, 1, vec![1]);
        assert_eq!(
            state_layout.theta, 2,
            "fixture sanity: theta must sit past commit_out/commit_in"
        );

        let bus = Bus {
            id: EntityId(1),
            name: String::new(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            deficit_segments: Vec::new(),
            excess_cost: 0.0,
        };
        let thermal = Thermal {
            id: EntityId(2),
            name: String::new(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(1),
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            cost_per_mwh: 50.0,
            anticipated_config: Some(AnticipatedConfig::LeadStages(1)),
            entry_stage_id: None,
            exit_stage_id: None,
        };
        let system = SystemBuilder::new()
            .buses(vec![bus])
            .thermals(vec![thermal])
            .stages(vec![one_year_stage(0), one_year_stage(1)])
            .bounds(ResolvedBounds::empty())
            .build()
            .expect("minimal anticipated system must build");
        let anticipated_plants = AnticipatedPlants::build(system.thermals());

        let num_cols = state_layout.theta + 2;
        let build = |discount_factors: Vec<f64>| -> StageTemplates {
            let mut stage_templates = StageTemplates::empty(0, 1.0);
            for _ in 0..2 {
                stage_templates.templates.push(StageTemplate {
                    num_cols,
                    num_rows: 0,
                    num_nz: 0,
                    col_starts: vec![0; num_cols + 1],
                    row_indices: Vec::new(),
                    values: Vec::new(),
                    col_lower: vec![f64::NEG_INFINITY; num_cols],
                    col_upper: vec![f64::INFINITY; num_cols],
                    objective: vec![10.0, 20.0, 30.0, 40.0],
                    row_lower: Vec::new(),
                    row_upper: Vec::new(),
                    n_state: state_layout.n_state,
                    n_transfer: 0,
                    n_dual_relevant: 0,
                    n_hydro: 0,
                    max_par_order: 0,
                    col_scale: Vec::new(),
                    row_scale: Vec::new(),
                });
            }
            let time_value = TimeValue::from_parts(
                discount_factors,
                vec![1.0, 1.0],
                vec![0.0, 0.0],
                vec![0, 1],
                PostStudyResolved::default(),
            );
            postprocess_templates(
                &mut stage_templates,
                &system,
                &state_layout,
                &anticipated_plants,
                1.0,
                &time_value,
            );
            stage_templates
        };

        let discount_factors = vec![0.6_f64, 0.3_f64];
        let undiscounted = build(vec![1.0, 1.0]);
        let discounted = build(discount_factors.clone());

        for (t, &d) in discount_factors.iter().enumerate() {
            let base = undiscounted.templates[t].objective[state_layout.theta];
            let got = discounted.templates[t].objective[state_layout.theta];
            assert_eq!(
                got.to_bits(),
                (base * d).to_bits(),
                "stage {t}: theta's discount must land on state_layout.theta bit-for-bit"
            );
            for j in 0..num_cols {
                if j == state_layout.theta {
                    continue;
                }
                assert_eq!(
                    discounted.templates[t].objective[j].to_bits(),
                    undiscounted.templates[t].objective[j].to_bits(),
                    "stage {t} col {j}: only theta may move with the discount rate"
                );
            }
        }
    }
}
