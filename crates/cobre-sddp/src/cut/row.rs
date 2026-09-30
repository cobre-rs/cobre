//! Benders cut-row construction for the SDDP LP.
//!
//! Owns the cut-sign convention: `push_scaled_coefficient` negates the stored
//! raw subgradient so each row reads `−∇·x + θ ≥ intercept`, the Benders cut
//! `θ ≥ Q(x̂) + π'(x − x̂)`. The subgradient-extraction side lives in
//! `training::backward::duals_extraction::extract_duals_from_view`.
//!
//! The forward training loop uses pre-frozen templates and does not call these
//! builders; the backward pass, simulation, lower-bound evaluation, and DCS do.

use cobre_solver::{RowBatch, SolverInterface};

use crate::cut::CutPool;
use crate::cut::CutRowMap;
use crate::cut::FutureCostFunction;
use crate::indexer::{CutStateProjection, OutCol, StateSpace};

/// Push one cut-row coefficient: `-coeff * col_scale[j]` (sign negation per the
/// module-doc Benders contract). Sole owner of the negate-and-scale rule, shared
/// by all three cut-row builders so it cannot drift apart.
#[inline]
pub(crate) fn push_scaled_coefficient(
    batch: &mut RowBatch,
    col: OutCol,
    coeff: f64,
    col_scale: &[f64],
) {
    let j = col.get();
    debug_assert!(
        i32::try_from(j).is_ok(),
        "column index j={j} exceeds i32::MAX"
    );
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    batch.col_indices.push(j as i32);
    let d = if col_scale.is_empty() {
        1.0
    } else {
        col_scale[j]
    };
    batch.values.push(-coeff * d);
}

/// Append one Benders cut row to `batch` in CSR form, matching the layout of
/// [`build_cut_row_batch_into`].
///
/// The caller pushes this row's `row_starts` offset before calling and the
/// terminator / `num_rows` / `add_rows` afterward; this helper appends only the
/// non-zeros and bounds. Shared by [`build_cut_row_batch_into`],
/// [`append_new_cuts_to_lp`], and the DCS [`append_slots_to_lp`] so the three
/// cannot drift apart.
///
/// `coefficients` has length `cut_state.n_slots()` (the pool's enabled cut-state
/// dimensions); the row places each enabled non-padding coefficient onto the
/// outgoing column [`CutStateProjection::render_pairs`] yields. `theta` is the global
/// scalar column (stage-invariant).
#[inline]
pub(crate) fn push_cut_row(
    batch: &mut RowBatch,
    intercept: f64,
    coefficients: &[f64],
    cut_state: &CutStateProjection,
    theta_col: usize,
    col_scale: &[f64],
) {
    for (j, lp_col) in cut_state.render_pairs() {
        push_scaled_coefficient(batch, lp_col, coefficients[j.get()], col_scale);
    }

    debug_assert!(
        i32::try_from(theta_col).is_ok(),
        "theta_col={theta_col} exceeds i32::MAX"
    );
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    batch.col_indices.push(theta_col as i32);
    let d_theta = if col_scale.is_empty() {
        1.0
    } else {
        col_scale[theta_col]
    };
    batch.values.push(d_theta);

    batch.row_lower.push(intercept);
    batch.row_upper.push(f64::INFINITY);
}

/// Fill a pre-allocated [`RowBatch`] with Benders cut rows from the FCF.
///
/// Clears `batch` and repopulates it with active cuts from `fcf` at the
/// given `pool` id. The buffers inside `batch` retain their allocated capacity
/// across calls, eliminating heap allocation on the hot path.
///
/// # Panics
///
/// Panics if the total number of non-zeros exceeds `i32::MAX` (the `HiGHS`
/// API limit for CSR indices).
pub fn build_cut_row_batch_into(
    batch: &mut RowBatch,
    fcf: &FutureCostFunction,
    pool: usize,
    state: &StateSpace,
    cut_state: &CutStateProjection,
    col_scale: &[f64],
) {
    batch.clear();

    let n_cut_state = cut_state.n_slots();
    let theta_col = state.theta;

    let num_cuts = fcf.pools[pool].active_count();

    if num_cuts == 0 {
        batch.row_starts.push(0_i32);
        return;
    }

    let nnz_per_cut = cut_state.render_len() + 1;
    let total_nnz = num_cuts * nnz_per_cut;

    let mut nz_offset = 0;

    for (_slot, intercept, coefficients) in fcf.active_cuts(pool) {
        debug_assert_eq!(
            coefficients.len(),
            n_cut_state,
            "cut coefficients length {got} != pool n_state {expected}",
            got = coefficients.len(),
            expected = n_cut_state,
        );

        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        batch.row_starts.push(nz_offset as i32);

        push_cut_row(
            batch,
            intercept,
            coefficients,
            cut_state,
            theta_col,
            col_scale,
        );

        nz_offset += nnz_per_cut;
    }

    #[allow(clippy::expect_used)]
    batch.row_starts.push(
        i32::try_from(total_nnz).expect("total_nnz exceeds i32::MAX; LP exceeds HiGHS API limit"),
    );

    batch.num_rows = num_cuts;
}

/// Build a fresh [`RowBatch`] of Benders cut rows from the FCF.
///
/// Convenience wrapper around [`build_cut_row_batch_into`] that allocates a
/// new `RowBatch`. For allocation-free usage on the hot path, prefer calling
/// [`build_cut_row_batch_into`] with a pre-allocated batch.
#[must_use]
pub fn build_cut_row_batch(
    fcf: &FutureCostFunction,
    pool: usize,
    state: &StateSpace,
    cut_state: &CutStateProjection,
    col_scale: &[f64],
) -> RowBatch {
    let mut batch = RowBatch {
        num_rows: 0,
        row_starts: Vec::new(),
        col_indices: Vec::new(),
        values: Vec::new(),
        row_lower: Vec::new(),
        row_upper: Vec::new(),
    };
    build_cut_row_batch_into(&mut batch, fcf, pool, state, cut_state, col_scale);
    batch
}

/// Append only the newly active cuts (not yet in the LP) to a live solver,
/// updating `row_map` with the new LP row indices and returning the count
/// appended. Rows use the same transformation as [`build_cut_row_batch_into`].
///
/// # Design invariant
///
/// The lower-bound LP grows monotonically (cuts appended, never removed);
/// re-freezing its template each iteration would raise cumulative setup cost from
/// `O(n_iters)` to `O(n_iters^2)`, so the append-only design is intentional.
///
/// # Arguments
///
/// - `col_scale`: column scaling factors (empty slice if no scaling).
/// - `row_map`: per-pool [`CutRowMap`] to update.
/// - `batch_buf`: reusable [`RowBatch`] buffer for constructing the new cut rows.
///
/// # Panics
///
/// Panics if `total_nnz` exceeds `i32::MAX` (LP exceeds the `HiGHS` API limit).
/// In debug builds, also panics if `pool >= fcf.pools.len()`.
///
/// [`CutRowMap`]: CutRowMap
pub fn append_new_cuts_to_lp<S: SolverInterface>(
    solver: &mut S,
    fcf: &FutureCostFunction,
    pool: usize,
    state: &StateSpace,
    cut_state: &CutStateProjection,
    col_scale: &[f64],
    row_map: &mut CutRowMap,
    batch_buf: &mut RowBatch,
) -> usize {
    batch_buf.clear();

    let n_cut_state = cut_state.n_slots();
    let theta_col = state.theta;
    let nnz_per_cut = cut_state.render_len() + 1;

    let mut new_count = 0usize;
    let mut nz_offset = 0usize;

    for (slot, intercept, coefficients) in fcf.active_cuts(pool) {
        if row_map.lp_row_for_slot(slot).is_some() {
            continue;
        }

        debug_assert_eq!(
            coefficients.len(),
            n_cut_state,
            "cut coefficients length {got} != pool n_state {expected}",
            got = coefficients.len(),
            expected = n_cut_state,
        );

        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        batch_buf.row_starts.push(nz_offset as i32);

        push_cut_row(
            batch_buf,
            intercept,
            coefficients,
            cut_state,
            theta_col,
            col_scale,
        );

        row_map.insert(slot);
        new_count += 1;
        nz_offset += nnz_per_cut;
    }

    if new_count > 0 {
        let total_nnz = new_count * nnz_per_cut;
        #[allow(clippy::expect_used)]
        batch_buf.row_starts.push(
            i32::try_from(total_nnz)
                .expect("total_nnz exceeds i32::MAX; LP exceeds HiGHS API limit"),
        );
        batch_buf.num_rows = new_count;
        solver.add_rows(batch_buf);
    }

    new_count
}

/// Append an explicit set of cut slots from a [`CutPool`] to a live solver.
///
/// The DCS analogue of [`append_new_cuts_to_lp`]: it adds exactly the active,
/// not-yet-resident slots in `slots` (identical row layout), recording each in
/// `row_map` and returning the count appended (`0` makes no `add_rows` call).
///
/// # Parameters
///
/// - `slots`: the slot ids to append, in caller order (the appended LP rows
///   follow this order).
/// - `col_scale`: column scaling factors (empty slice ⇒ no scaling).
/// - `row_map`: per-(stage, solve) [`CutRowMap`] to update.
///
/// # Panics
///
/// Panics if the total non-zero count exceeds `i32::MAX` (the `HiGHS` API
/// limit), matching [`append_new_cuts_to_lp`].
///
/// [`CutPool`]: CutPool
pub fn append_slots_to_lp<S: SolverInterface>(
    solver: &mut S,
    pool: &CutPool,
    slots: &[u32],
    state: &StateSpace,
    cut_state: &CutStateProjection,
    col_scale: &[f64],
    row_map: &mut CutRowMap,
    batch_buf: &mut RowBatch,
) -> usize {
    batch_buf.clear();

    // The pool stores coefficients at its own (possibly reduced) dimension; slice
    // by that, render via the matching per-pool projection.
    debug_assert_eq!(
        pool.state_dimension,
        cut_state.n_slots(),
        "append_slots_to_lp: pool.state_dimension {} != cut_state.n_slots() {}",
        pool.state_dimension,
        cut_state.n_slots(),
    );
    let theta_col = state.theta;
    let nnz_per_cut = cut_state.render_len() + 1;

    let mut new_count = 0usize;
    let mut nz_offset = 0usize;

    for &slot in slots {
        let slot_usize = slot as usize;

        if slot_usize >= pool.populated()
            || !pool.is_active(slot_usize)
            || row_map.lp_row_for_slot(slot_usize).is_some()
        {
            continue;
        }

        let intercept = pool.intercept(slot_usize);
        let coefficients = pool.coefficient_row(slot_usize);

        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        batch_buf.row_starts.push(nz_offset as i32);

        push_cut_row(
            batch_buf,
            intercept,
            coefficients,
            cut_state,
            theta_col,
            col_scale,
        );

        row_map.insert(slot_usize);
        new_count += 1;
        nz_offset += nnz_per_cut;
    }

    if new_count > 0 {
        let total_nnz = new_count * nnz_per_cut;
        #[allow(clippy::expect_used)]
        batch_buf.row_starts.push(
            i32::try_from(total_nnz)
                .expect("total_nnz exceeds i32::MAX; LP exceeds HiGHS API limit"),
        );
        batch_buf.num_rows = new_count;
        solver.add_rows(batch_buf);
    }

    new_count
}

#[cfg(test)]
mod tests {
    use cobre_core::temporal::StageStateConfig;
    use cobre_solver::{
        Basis, RowBatch, SolverError, SolverInterface, SolverStatistics, StageTemplate,
    };

    use super::{append_new_cuts_to_lp, build_cut_row_batch, build_cut_row_batch_into};
    use crate::cut::FutureCostFunction;
    use crate::indexer::{CutStateProjection, StateSpace};
    use crate::lead_time::AnticipatedResolution;
    use crate::setup::NodeId;

    /// Build a finalized storage+lag [`StateSpace`] (no anticipated thermals)
    /// with the full `max_par_order` lag stride for every hydro — the dense
    /// coverage production `resolve_state_layout` finalizes for a study with no
    /// per-hydro AR-order truncation.
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

    /// All-enabled per-pool projection of `state` — these builder tests use
    /// full-dimension pools, so the render reproduces the global nonzero mask.
    fn cut_state(state: &StateSpace) -> CutStateProjection {
        CutStateProjection::new(
            state,
            StageStateConfig {
                storage: true,
                inflow_lags: true,
            },
        )
    }

    // ── Unit tests: build_cut_row_batch ──────────────────────────────────────

    #[test]
    fn build_cut_row_batch_empty_cuts_returns_empty_batch() {
        let fcf = FutureCostFunction::new(2, 1, 1, 10, &[0; 2]);
        let state = state_layout(1, 0);
        let batch = build_cut_row_batch(&fcf, 0, &state, &cut_state(&state), &[]);

        assert_eq!(batch.num_rows, 0);
        assert_eq!(batch.row_starts, vec![0]);
        assert!(batch.col_indices.is_empty());
        assert!(batch.values.is_empty());
        assert!(batch.row_lower.is_empty());
        assert!(batch.row_upper.is_empty());
    }

    #[test]
    fn build_cut_row_batch_one_cut_correct_structure() {
        let mut fcf = FutureCostFunction::new(2, 1, 1, 10, &[0; 2]);
        fcf.add_cut(NodeId(0), 0, 0, 0, 5.0, &[2.0]);
        let state = state_layout(1, 0);
        let batch = build_cut_row_batch(&fcf, 0, &state, &cut_state(&state), &[]);

        assert_eq!(batch.num_rows, 1);
        assert_eq!(batch.row_starts, vec![0, 2]);
        assert_eq!(batch.col_indices, vec![0, 3]); // theta at col N*(3+L) = 3
        assert_eq!(batch.values, vec![-2.0, 1.0]);
        assert_eq!(batch.row_lower, vec![5.0]);
        assert!(batch.row_upper[0].is_infinite() && batch.row_upper[0] > 0.0);
    }

    #[test]
    fn build_cut_row_batch_two_cuts_correct_row_starts() {
        let mut fcf = FutureCostFunction::new(2, 2, 1, 10, &[0; 2]);
        fcf.add_cut(NodeId(0), 1, 0, 0, 10.0, &[1.0, 3.0]);
        fcf.add_cut(NodeId(0), 1, 1, 0, 20.0, &[2.0, 4.0]);
        let state = state_layout(1, 1);
        let batch = build_cut_row_batch(&fcf, 1, &state, &cut_state(&state), &[]);

        assert_eq!(batch.num_rows, 2);
        assert_eq!(batch.row_starts, vec![0, 3, 6]);
        assert_eq!(batch.col_indices[0], 0); // storage col 0
        assert_eq!(batch.col_indices[1], 2); // lag 0 → z_inflow col N*(1+L)=2
        assert_eq!(batch.col_indices[2], 4); // theta at N*(3+L) = 1*(3+1) = 4
        assert_eq!(batch.values[0], -1.0);
        assert_eq!(batch.values[1], -3.0);
        assert_eq!(batch.values[2], 1.0);
        assert_eq!(batch.col_indices[3], 0); // storage col 0
        assert_eq!(batch.col_indices[4], 2); // lag 0 → z_inflow col 2
        assert_eq!(batch.col_indices[5], 4); // theta at N*(3+L) = 4
        assert_eq!(batch.values[3], -2.0);
        assert_eq!(batch.values[4], -4.0);
        assert_eq!(batch.values[5], 1.0);
        assert_eq!(batch.row_lower, vec![10.0, 20.0]);
        assert!(batch.row_upper[0].is_infinite() && batch.row_upper[0] > 0.0);
        assert!(batch.row_upper[1].is_infinite() && batch.row_upper[1] > 0.0);
    }

    #[test]
    fn build_cut_row_batch_zero_coefficient_state_variable() {
        let mut fcf = FutureCostFunction::new(1, 2, 1, 5, &[0; 1]);
        fcf.add_cut(NodeId(0), 0, 0, 0, 3.0, &[0.0, 7.0]);
        let state = state_layout(1, 1);
        let batch = build_cut_row_batch(&fcf, 0, &state, &cut_state(&state), &[]);

        assert_eq!(batch.num_rows, 1);
        assert_eq!(batch.col_indices, vec![0, 2, 4]); // lag 0 → z_inflow col 2; theta at 4
        assert_eq!(batch.values, vec![0.0, -7.0, 1.0]);
        assert_eq!(batch.row_lower, vec![3.0]);
    }

    // ── Tests for append_new_cuts_to_lp ─────────────────────────────────

    /// Mock solver that records the last `add_rows` call for verification.
    struct RecordingMockSolver {
        last_batch: Option<RowBatch>,
        add_rows_count: usize,
    }

    impl RecordingMockSolver {
        fn new() -> Self {
            Self {
                last_batch: None,
                add_rows_count: 0,
            }
        }
    }

    impl SolverInterface for RecordingMockSolver {
        type Profile = cobre_solver::ActiveProfile;

        fn apply_profile(&mut self, _profile: &cobre_solver::ActiveProfile) {}

        fn solver_name_version(&self) -> String {
            "MockSolver 0.0.0".to_string()
        }
        fn load_model(&mut self, _template: &StageTemplate) {}

        fn add_rows(&mut self, cuts: &RowBatch) {
            self.last_batch = Some(RowBatch {
                num_rows: cuts.num_rows,
                row_starts: cuts.row_starts.clone(),
                col_indices: cuts.col_indices.clone(),
                values: cuts.values.clone(),
                row_lower: cuts.row_lower.clone(),
                row_upper: cuts.row_upper.clone(),
            });
            self.add_rows_count += 1;
        }

        fn set_row_bounds(&mut self, _indices: &[usize], _lower: &[f64], _upper: &[f64]) {}

        fn set_col_bounds(&mut self, _indices: &[usize], _lower: &[f64], _upper: &[f64]) {}

        fn solve(
            &mut self,
            _basis: Option<&Basis>,
        ) -> Result<cobre_solver::SolutionView<'_>, SolverError> {
            Err(SolverError::InternalError {
                message: "not implemented for test".to_string(),
                error_code: None,
            })
        }

        fn get_basis(&mut self, out: &mut Basis) {
            crate::test_support::fill_consistent_basis(out);
        }

        fn statistics(&self) -> SolverStatistics {
            SolverStatistics::default()
        }

        fn statistics_into(&self, out: &mut SolverStatistics) {
            out.copy_from(&SolverStatistics::default());
        }

        fn name(&self) -> &'static str {
            "RecordingMock"
        }
    }

    fn empty_row_batch() -> RowBatch {
        RowBatch {
            num_rows: 0,
            row_starts: Vec::new(),
            col_indices: Vec::new(),
            values: Vec::new(),
            row_lower: Vec::new(),
            row_upper: Vec::new(),
        }
    }

    #[test]
    fn append_new_cuts_returns_zero_when_no_new_cuts() {
        use crate::cut::CutRowMap;

        let fcf = FutureCostFunction::new(2, 1, 1, 10, &[0; 2]);
        let state = state_layout(1, 0);
        let mut row_map = CutRowMap::new(10, 5);
        let mut batch_buf = empty_row_batch();
        let mut solver = RecordingMockSolver::new();

        let count = append_new_cuts_to_lp(
            &mut solver,
            &fcf,
            0,
            &state,
            &cut_state(&state),
            &[],
            &mut row_map,
            &mut batch_buf,
        );
        assert_eq!(count, 0);
        assert_eq!(solver.add_rows_count, 0);
    }

    #[test]
    fn append_new_cuts_appends_all_on_empty_row_map() {
        use crate::cut::CutRowMap;

        let mut fcf = FutureCostFunction::new(2, 1, 1, 10, &[0; 2]);
        fcf.add_cut(NodeId(0), 0, 0, 0, 10.0, &[1.0]); // slot 0
        fcf.add_cut(NodeId(0), 0, 1, 0, 20.0, &[3.0]); // slot 1

        let state = state_layout(1, 0);
        let mut row_map = CutRowMap::new(10, 5);
        let mut batch_buf = empty_row_batch();
        let mut solver = RecordingMockSolver::new();

        let count = append_new_cuts_to_lp(
            &mut solver,
            &fcf,
            0,
            &state,
            &cut_state(&state),
            &[],
            &mut row_map,
            &mut batch_buf,
        );

        assert_eq!(count, 2);
        assert_eq!(solver.add_rows_count, 1);
        assert_eq!(row_map.total_cut_rows(), 2);
        assert_eq!(row_map.lp_row_for_slot(0), Some(5));
        assert_eq!(row_map.lp_row_for_slot(1), Some(6));
    }

    #[test]
    fn append_new_cuts_skips_already_mapped_cuts() {
        use crate::cut::CutRowMap;

        let mut fcf = FutureCostFunction::new(2, 1, 1, 10, &[0; 2]);
        fcf.add_cut(NodeId(0), 0, 0, 0, 10.0, &[1.0]); // slot 0
        fcf.add_cut(NodeId(0), 0, 1, 0, 20.0, &[3.0]); // slot 1

        let state = state_layout(1, 0);
        let mut row_map = CutRowMap::new(10, 5);
        // Pre-insert slot 0 as if it was already in the LP.
        row_map.insert(0);

        let mut batch_buf = empty_row_batch();
        let mut solver = RecordingMockSolver::new();

        let count = append_new_cuts_to_lp(
            &mut solver,
            &fcf,
            0,
            &state,
            &cut_state(&state),
            &[],
            &mut row_map,
            &mut batch_buf,
        );

        // Only slot 1 should be appended (slot 0 was already mapped).
        assert_eq!(count, 1);
        assert_eq!(solver.add_rows_count, 1);
        assert_eq!(row_map.total_cut_rows(), 2);
        assert!(solver.last_batch.as_ref().is_some_and(|b| b.num_rows == 1));
    }

    #[test]
    fn append_new_cuts_matches_build_cut_row_batch_into() {
        use crate::cut::CutRowMap;

        let mut fcf = FutureCostFunction::new(2, 1, 1, 10, &[0; 2]);
        fcf.add_cut(NodeId(0), 0, 0, 0, 10.0, &[1.0]); // slot 0
        fcf.add_cut(NodeId(0), 0, 1, 0, 20.0, &[3.0]); // slot 1

        let state = state_layout(1, 0);

        let mut expected_batch = empty_row_batch();
        build_cut_row_batch_into(
            &mut expected_batch,
            &fcf,
            0,
            &state,
            &cut_state(&state),
            &[],
        );

        // Empty row_map, so append_new_cuts_to_lp treats all cuts as new.
        let mut row_map = CutRowMap::new(10, 5);
        let mut actual_batch = empty_row_batch();
        let mut solver = RecordingMockSolver::new();
        append_new_cuts_to_lp(
            &mut solver,
            &fcf,
            0,
            &state,
            &cut_state(&state),
            &[],
            &mut row_map,
            &mut actual_batch,
        );

        assert_eq!(actual_batch.num_rows, expected_batch.num_rows);
        assert_eq!(actual_batch.row_starts, expected_batch.row_starts);
        assert_eq!(actual_batch.col_indices, expected_batch.col_indices);
        assert_eq!(actual_batch.values, expected_batch.values);
        assert_eq!(actual_batch.row_lower, expected_batch.row_lower);
        assert_eq!(actual_batch.row_upper, expected_batch.row_upper);
    }

    #[test]
    fn append_new_cuts_with_scaling_matches_build() {
        use crate::cut::CutRowMap;

        let mut fcf = FutureCostFunction::new(2, 1, 1, 10, &[0; 2]);
        fcf.add_cut(NodeId(0), 0, 0, 0, 10.0, &[1.0]);

        let state = state_layout(1, 0);
        // col_scale must have at least theta+1 = 4 entries.
        let col_scale = vec![0.5, 2.0, 1.0, 0.1];

        let mut expected = empty_row_batch();
        build_cut_row_batch_into(
            &mut expected,
            &fcf,
            0,
            &state,
            &cut_state(&state),
            &col_scale,
        );

        let mut row_map = CutRowMap::new(10, 5);
        let mut actual = empty_row_batch();
        let mut solver = RecordingMockSolver::new();
        append_new_cuts_to_lp(
            &mut solver,
            &fcf,
            0,
            &state,
            &cut_state(&state),
            &col_scale,
            &mut row_map,
            &mut actual,
        );

        assert_eq!(actual.values, expected.values);
        assert_eq!(actual.col_indices, expected.col_indices);
    }
}
