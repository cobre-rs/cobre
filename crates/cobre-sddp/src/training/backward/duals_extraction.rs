//! Benders subgradient extraction from a solved backward LP.
//!
//! The subgradient is `rc_scaled / col_scale[col]` — **divided, not multiplied** —
//! because the incoming-state column pin sets `v_scaled = v_orig / col_scale`
//! (sddp.md "Benders cut sign & subgradient extraction").

use cobre_solver::SolutionView;

use crate::indexer::{CutSlot, CutStateProjection};

use super::SuccessorChild;

fn fill_state_duals(
    view: &SolutionView<'_>,
    cut_state: &CutStateProjection,
    col_scale: &[f64],
    state_duals: &mut Vec<f64>,
) {
    // Divided, not multiplied (module-level contract). Empty col_scale ⇒ raw rc.
    state_duals.clear();
    for j in 0..cut_state.n_slots() {
        let col = cut_state.incoming_column(CutSlot::new(j)).get();
        let rc = view.reduced_costs[col];
        let unscaled = if col_scale.is_empty() {
            rc
        } else {
            rc / col_scale[col]
        };
        state_duals.push(unscaled);
    }
    debug_assert_eq!(
        state_duals.len(),
        cut_state.n_slots(),
        "state_duals must contain exactly cut_state.n_slots() entries after fill"
    );
}

/// Extract state and cut duals from the live solver view into pre-warmed scratch
/// buffers, returning the LP objective. `state_duals` holds the unscaled
/// incoming-state reduced costs of the stage's enabled cut-state dimensions (the
/// module-level `col_scale`-division contract), iterated over `cut_state`; a
/// disabled dimension's reduced cost is computed in the LP but not extracted.
/// `cut_duals` holds the cut-row slice
/// `[template_num_rows, template_num_rows + num_cuts)`.
pub(crate) fn extract_duals_from_view(
    view: &SolutionView<'_>,
    cut_state: &CutStateProjection,
    col_scale: &[f64],
    succ: &SuccessorChild<'_>,
    state_duals: &mut Vec<f64>,
    cut_duals: &mut Vec<f64>,
) -> f64 {
    let objective = view.objective;

    fill_state_duals(view, cut_state, col_scale, state_duals);

    cut_duals.clear();
    if succ.num_cuts_at_successor > 0 {
        cut_duals.extend_from_slice(
            &view.dual[succ.template_num_rows..succ.template_num_rows + succ.num_cuts_at_successor],
        );
    }

    objective
}

/// State-dual half of [`extract_duals_from_view`] for the lazy-solve path: fills
/// `state_duals` with the unscaled incoming-state reduced costs of the enabled
/// cut-state dimensions, iterated over `cut_state`.
///
/// Unlike [`extract_duals_from_view`] it does NOT read cut-row duals — under
/// lazy-solve the resident cut rows are an insertion-order subset, so the
/// cut-row→slot mapping does not apply.
pub(crate) fn extract_state_duals_only(
    view: &SolutionView<'_>,
    cut_state: &CutStateProjection,
    col_scale: &[f64],
    state_duals: &mut Vec<f64>,
) -> f64 {
    let objective = view.objective;

    fill_state_duals(view, cut_state, col_scale, state_duals);

    objective
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss)]
mod tests {
    use cobre_solver::SolutionView;

    use super::extract_state_duals_only;
    use crate::indexer::{CutStateProjection, StateDim, StateSpace};
    use crate::lead_time::AnticipatedResolution;
    use cobre_core::temporal::StageStateConfig;

    const ALL_ENABLED: StageStateConfig = StageStateConfig {
        storage: true,
        inflow_lags: true,
    };
    const STORAGE_ONLY: StageStateConfig = StageStateConfig {
        storage: true,
        inflow_lags: false,
    };

    fn state_layout(hydro_count: usize, max_par_order: usize) -> StateSpace {
        let lag_counts = vec![max_par_order; hydro_count];
        StateSpace::new(
            hydro_count,
            max_par_order,
            Vec::new(),
            vec![],
            AnticipatedResolution::default(),
            &lag_counts,
        )
    }

    /// Build a `SolutionView` whose `reduced_costs[col]` equals `col as f64`, so
    /// an extracted subgradient at LP column `col` is exactly `col as f64`
    /// (`col_scale` empty ⇒ raw rc).
    fn indexed_view(reduced_costs: &[f64]) -> SolutionView<'_> {
        SolutionView {
            objective: 0.0,
            primal: &[],
            dual: &[],
            reduced_costs,
            iterations: 0,
            solve_time_seconds: 0.0,
        }
    }

    /// Storage-only `CutStateProjection` (`N=3, L=2`): extraction yields an `N`-length
    /// subgradient reading exactly the storage columns `[0, N)`, no lag entries.
    #[test]
    fn storage_only_yields_n_length_subgradient_with_storage_columns() {
        let global = state_layout(3, 2);
        let cut_state = CutStateProjection::new(&global, STORAGE_ONLY);
        assert_eq!(cut_state.n_slots(), 3);

        let rc: Vec<f64> = (0..=global.theta).map(|c| c as f64).collect();
        let view = indexed_view(&rc);
        let mut state_duals = Vec::new();

        let _ = extract_state_duals_only(&view, &cut_state, &[], &mut state_duals);

        assert_eq!(state_duals.len(), 3);
        for (j, &dual) in state_duals.iter().enumerate() {
            assert_eq!(
                dual,
                global.state_to_lp_incoming_column(StateDim::new(j)).get() as f64,
                "storage slot {j} reads its storage_in column reduced cost"
            );
        }
    }

    /// All-enabled `CutStateProjection` reproduces the pre-change global-loop result:
    /// `n_slots()` equals the global `n_state` and every slot reads the same LP
    /// column the global `StateSpace` resolver would.
    #[test]
    fn all_enabled_reproduces_global_loop_result() {
        let global = state_layout(3, 2);
        let cut_state = CutStateProjection::new(&global, ALL_ENABLED);
        assert_eq!(cut_state.n_slots(), global.n_state);

        let rc: Vec<f64> = (0..=global.theta).map(|c| c as f64).collect();
        let view = indexed_view(&rc);

        let mut via_cut_state = Vec::new();
        let _ = extract_state_duals_only(&view, &cut_state, &[], &mut via_cut_state);

        let global_loop: Vec<f64> = (0..global.n_state)
            .map(|j| rc[global.state_to_lp_incoming_column(StateDim::new(j)).get()])
            .collect();

        assert_eq!(via_cut_state, global_loop);
    }
}
