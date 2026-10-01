//! CSR-to-CSC template freeze for reusable LP stage templates.
//!
//! Provides [`freeze_rows_into_template`], which merges a CSC base template with a
//! CSR row batch into a larger CSC template loadable directly via
//! [`crate::SolverInterface::load_model`] without a subsequent `add_rows` call.
//!
//! Within each column the base rows appear first (original CSC order), then the
//! appended rows in ascending CSR row order — a reproducibility convention;
//! `HiGHS` does not require sorted per-column row indices.
//!
//! See [Solver Abstraction SS11.1](../../../cobre-docs/src/specs/architecture/solver-abstraction.md).

use crate::types::{RowBatch, StageTemplate};

/// Caller-owned reusable scratch buffers for [`freeze_rows_into_template`].
///
/// Each buffer is `clear()`-ed and re-grown per call without `shrink_to_fit`, so
/// passing one `FreezeScratch` across freezes allocates no temporaries at steady
/// state (once the largest seen template/batch has been processed).
#[derive(Debug, Default)]
pub struct FreezeScratch {
    csc_nz_per_col: Vec<u32>,
    col_list_start: Vec<u32>,
    col_list_row: Vec<i32>,
    col_list_val: Vec<f64>,
    write_cursor: Vec<u32>,
}

impl FreezeScratch {
    /// Construct an empty scratch; buffers grow on first freeze.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

/// Merge a CSC base template with a CSR row batch into an output CSC template.
///
/// After return, `out` is a valid [`StageTemplate`] in CSC form with
/// `out.num_rows == base.num_rows + rows.num_rows`. Both `out` and `scratch`
/// are cleared and refilled per call without `shrink_to_fit`; reuse them across
/// calls for zero allocation at steady state (see [`FreezeScratch`]).
///
/// # Preconditions
///
/// Checked via `debug_assert!` in debug builds:
/// - `base.col_starts.len() == base.num_cols + 1` and sentinel `== base.num_nz`
/// - `base.row_indices.len() == base.values.len() == base.num_nz`
/// - `base` column/row bound arrays and objective each have the correct length
/// - `rows.row_starts[0] == 0` (when `rows.num_rows > 0`)
/// - Every `col_indices[k] < base.num_cols`
///
/// # Panics
///
/// Panics if `base.num_nz + rows_nnz` exceeds `i32::MAX` (`HiGHS` API limit).
#[allow(clippy::too_many_lines)] // complex data-structure merge; extracting sub-functions would obscure the algorithm
pub fn freeze_rows_into_template(
    base: &StageTemplate,
    rows: &RowBatch,
    out: &mut StageTemplate,
    scratch: &mut FreezeScratch,
) {
    #[allow(clippy::cast_sign_loss)]
    {
        debug_assert_eq!(
            base.col_starts.len(),
            base.num_cols + 1,
            "base.col_starts.len()={} but num_cols+1={}",
            base.col_starts.len(),
            base.num_cols + 1
        );
        debug_assert_eq!(
            base.col_starts.last().copied().unwrap_or(0) as usize,
            base.num_nz,
            "base.col_starts[num_cols] != base.num_nz"
        );
        debug_assert_eq!(base.row_indices.len(), base.num_nz);
        debug_assert_eq!(base.values.len(), base.num_nz);
        debug_assert_eq!(base.col_lower.len(), base.num_cols);
        debug_assert_eq!(base.col_upper.len(), base.num_cols);
        debug_assert_eq!(base.objective.len(), base.num_cols);
        debug_assert_eq!(base.row_lower.len(), base.num_rows);
        debug_assert_eq!(base.row_upper.len(), base.num_rows);
        debug_assert!(
            base.col_scale.is_empty() || base.col_scale.len() == base.num_cols,
            "base.col_scale must be empty or length num_cols"
        );
        debug_assert!(
            base.row_scale.is_empty() || base.row_scale.len() == base.num_rows,
            "base.row_scale must be empty or length num_rows"
        );

        if rows.num_rows > 0 {
            debug_assert_eq!(
                rows.row_starts.len(),
                rows.num_rows + 1,
                "rows.row_starts.len()={} but num_rows+1={}",
                rows.row_starts.len(),
                rows.num_rows + 1
            );
            debug_assert_eq!(
                rows.row_starts[0], 0,
                "RowBatch invariant: row_starts[0] must be 0"
            );
            debug_assert_eq!(rows.row_lower.len(), rows.num_rows);
            debug_assert_eq!(rows.row_upper.len(), rows.num_rows);

            let rows_nnz = rows.row_starts[rows.num_rows] as usize;
            debug_assert_eq!(rows.col_indices.len(), rows_nnz);
            debug_assert_eq!(rows.values.len(), rows_nnz);

            #[cfg(debug_assertions)]
            for &col in &rows.col_indices {
                debug_assert!(
                    (col as usize) < base.num_cols,
                    "col_indices[k]={col} >= base.num_cols={}",
                    base.num_cols
                );
            }
        }
    }

    #[allow(clippy::cast_sign_loss)]
    let rows_nnz = if rows.num_rows > 0 {
        rows.row_starts[rows.num_rows] as usize
    } else {
        0
    };
    let total_nnz = base.num_nz + rows_nnz;

    #[allow(clippy::expect_used)]
    let total_nnz_i32 = i32::try_from(total_nnz).expect("total nnz exceeds i32::MAX");

    let num_cols = base.num_cols;
    let num_rows = base.num_rows + rows.num_rows;

    scratch.csc_nz_per_col.clear();
    scratch.csc_nz_per_col.resize(num_cols, 0u32);
    #[allow(clippy::cast_sign_loss)]
    for &col in &rows.col_indices {
        scratch.csc_nz_per_col[col as usize] += 1;
    }

    out.col_starts.clear();
    out.row_indices.clear();
    out.values.clear();
    out.col_lower.clear();
    out.col_upper.clear();
    out.objective.clear();
    out.col_scale.clear();
    out.row_lower.clear();
    out.row_upper.clear();
    out.row_scale.clear();

    out.num_cols = num_cols;
    out.num_rows = num_rows;
    out.num_nz = total_nnz;
    out.n_state = base.n_state;

    out.col_lower.extend_from_slice(&base.col_lower);
    out.col_upper.extend_from_slice(&base.col_upper);
    out.objective.extend_from_slice(&base.objective);
    out.col_scale.extend_from_slice(&base.col_scale);

    out.row_lower.extend_from_slice(&base.row_lower);
    out.row_lower.extend_from_slice(&rows.row_lower);
    out.row_upper.extend_from_slice(&base.row_upper);
    out.row_upper.extend_from_slice(&rows.row_upper);

    // StageTemplate invariant: a non-empty row_scale has len == num_rows. When the
    // base has none but rows are appended, materialise the full vector as 1.0 — not
    // empty — so the appended rows do not leave it length-mismatched.
    if !base.row_scale.is_empty() {
        out.row_scale.extend_from_slice(&base.row_scale);
        out.row_scale
            .resize(out.row_scale.len() + rows.num_rows, 1.0_f64);
    } else if rows.num_rows > 0 {
        out.row_scale.resize(base.num_rows + rows.num_rows, 1.0_f64);
    }

    scratch.col_list_start.clear();
    scratch.col_list_start.reserve(num_cols + 1);
    let mut running = 0u32;
    for &count in &scratch.csc_nz_per_col {
        scratch.col_list_start.push(running);
        running += count;
    }
    scratch.col_list_start.push(running);

    scratch.col_list_row.clear();
    scratch.col_list_row.resize(rows_nnz, 0i32);
    scratch.col_list_val.clear();
    scratch.col_list_val.resize(rows_nnz, 0.0f64);
    scratch.write_cursor.clear();
    scratch.write_cursor.resize(num_cols, 0u32);

    #[allow(clippy::cast_sign_loss)]
    for r in 0..rows.num_rows {
        let start = rows.row_starts[r] as usize;
        let end = rows.row_starts[r + 1] as usize;
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let row_i32 = (base.num_rows + r) as i32;
        for k in start..end {
            let j = rows.col_indices[k] as usize;
            let pos = (scratch.col_list_start[j] + scratch.write_cursor[j]) as usize;
            scratch.col_list_row[pos] = row_i32;
            scratch.col_list_val[pos] = rows.values[k];
            scratch.write_cursor[j] += 1;
        }
    }

    let mut nz_cursor: i32 = 0;
    #[allow(clippy::cast_sign_loss)]
    for j in 0..num_cols {
        out.col_starts.push(nz_cursor);

        let base_start = base.col_starts[j] as usize;
        let base_end = base.col_starts[j + 1] as usize;
        out.row_indices
            .extend_from_slice(&base.row_indices[base_start..base_end]);
        out.values
            .extend_from_slice(&base.values[base_start..base_end]);

        let list_start = scratch.col_list_start[j] as usize;
        let list_end = scratch.col_list_start[j + 1] as usize;
        out.row_indices
            .extend_from_slice(&scratch.col_list_row[list_start..list_end]);
        out.values
            .extend_from_slice(&scratch.col_list_val[list_start..list_end]);

        let col_len = (base_end - base_start) + (list_end - list_start);
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        {
            nz_cursor += col_len as i32;
        }
    }
    out.col_starts.push(total_nnz_i32);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SolverInterface;
    use crate::profile::MockProfile;
    use crate::types::{
        Basis, RowBatch, SolutionView, SolverError, SolverStatistics, StageTemplate,
    };

    /// Builds the canonical 3-col, 2-row fixture from `types.rs::make_fixture_stage_template`.
    ///
    /// ```
    /// col_starts = [0, 2, 2, 3]
    /// row_indices = [0, 1, 1]
    /// values = [1.0, 2.0, 1.0]
    /// row_lower = [6.0, 14.0]
    /// row_upper = [6.0, 14.0]
    /// ```
    fn make_fixture_stage_template() -> StageTemplate {
        StageTemplate {
            num_cols: 3,
            num_rows: 2,
            num_nz: 3,
            col_starts: vec![0_i32, 2, 2, 3],
            row_indices: vec![0_i32, 1, 1],
            values: vec![1.0, 2.0, 1.0],
            col_lower: vec![0.0, 0.0, 0.0],
            col_upper: vec![10.0, f64::INFINITY, 8.0],
            objective: vec![0.0, 1.0, 50.0],
            row_lower: vec![6.0, 14.0],
            row_upper: vec![6.0, 14.0],
            n_state: 1,
            col_scale: Vec::new(),
            row_scale: Vec::new(),
        }
    }

    /// Builds an empty [`RowBatch`] with zero rows (but valid `row_starts` sentinel).
    fn make_empty_row_batch() -> RowBatch {
        RowBatch {
            num_rows: 0,
            row_starts: vec![0_i32],
            col_indices: vec![],
            values: vec![],
            row_lower: vec![],
            row_upper: vec![],
        }
    }

    // -----------------------------------------------------------------------
    // Test 1: empty rows → structural copy of base
    // -----------------------------------------------------------------------

    #[test]
    fn test_freeze_empty_rows_copies_base() {
        let base = make_fixture_stage_template();
        let rows = make_empty_row_batch();
        let mut out = StageTemplate::empty();

        freeze_rows_into_template(&base, &rows, &mut out, &mut FreezeScratch::default());

        assert_eq!(out.num_cols, base.num_cols);
        assert_eq!(out.num_rows, base.num_rows);
        assert_eq!(out.num_nz, base.num_nz);
        assert_eq!(out.col_starts, base.col_starts);
        assert_eq!(out.row_indices, base.row_indices);
        assert_eq!(out.values, base.values);
        assert_eq!(out.col_lower, base.col_lower);
        assert_eq!(out.col_upper, base.col_upper);
        assert_eq!(out.objective, base.objective);
        assert_eq!(out.row_lower, base.row_lower);
        assert_eq!(out.row_upper, base.row_upper);
        assert_eq!(out.n_state, base.n_state);
        // empty row_scale stays empty
        assert!(out.row_scale.is_empty());
    }

    // -----------------------------------------------------------------------
    // Test 2: single appended row — exact CSC column layout
    // -----------------------------------------------------------------------

    #[test]
    fn test_freeze_single_row_appends_correct_column_entries() {
        // Fixture: 3-col, 2-row base as described.
        // RowBatch: one row touching cols 0 and 2.
        let base = make_fixture_stage_template();
        let rows = RowBatch {
            num_rows: 1,
            row_starts: vec![0_i32, 2],
            col_indices: vec![0_i32, 2],
            values: vec![-1.5, 1.0],
            row_lower: vec![10.0],
            row_upper: vec![f64::INFINITY],
        };
        let mut out = StageTemplate::empty();

        freeze_rows_into_template(&base, &rows, &mut out, &mut FreezeScratch::default());

        assert_eq!(out.num_rows, 3);
        assert_eq!(out.num_nz, 5);

        // Corrected col_starts: col 1 has 0 entries.
        assert_eq!(out.col_starts, vec![0_i32, 3, 3, 5]);

        // Column 0: base rows [0,1] then cut row [2]
        assert_eq!(&out.row_indices[0..3], &[0_i32, 1, 2]);
        assert_eq!(&out.values[0..3], &[1.0_f64, 2.0, -1.5]);

        // Column 1: no entries (col_starts[1]==3, col_starts[2]==3)

        // Column 2: base row [1] then cut row [2]  (positions 3..5)
        assert_eq!(&out.row_indices[3..5], &[1_i32, 2]);
        assert_eq!(&out.values[3..5], &[1.0_f64, 1.0]);

        // Row bounds
        assert_eq!(out.row_lower, vec![6.0_f64, 14.0, 10.0]);
        assert!(out.row_upper[2].is_infinite() && out.row_upper[2] > 0.0);
    }

    // -----------------------------------------------------------------------
    // Test 3: non-empty row_scale — appended cut rows default to 1.0
    // -----------------------------------------------------------------------

    #[test]
    fn test_freeze_preserves_row_scale_and_defaults_cut_rows_to_one() {
        let mut base = make_fixture_stage_template();
        base.row_scale = vec![1.0, 2.0];

        let rows = RowBatch {
            num_rows: 1,
            row_starts: vec![0_i32, 1],
            col_indices: vec![0_i32],
            values: vec![-1.0],
            row_lower: vec![5.0],
            row_upper: vec![f64::INFINITY],
        };
        let mut out = StageTemplate::empty();

        freeze_rows_into_template(&base, &rows, &mut out, &mut FreezeScratch::default());

        assert_eq!(out.row_scale.len(), 3);
        assert_eq!(out.row_scale[0], 1.0);
        assert_eq!(out.row_scale[1], 2.0);
        assert_eq!(out.row_scale[2], 1.0); // cut row implicit scale
    }

    // -----------------------------------------------------------------------
    // Test 4: empty row_scale + zero rows → out.row_scale stays empty
    // -----------------------------------------------------------------------

    #[test]
    fn test_freeze_preserves_empty_row_scale_when_no_rows() {
        let base = make_fixture_stage_template(); // row_scale is empty
        let rows = make_empty_row_batch();
        let mut out = StageTemplate::empty();

        freeze_rows_into_template(&base, &rows, &mut out, &mut FreezeScratch::default());

        assert!(out.row_scale.is_empty());
        assert_eq!(out.num_rows, base.num_rows);
    }

    // -----------------------------------------------------------------------
    // Test 5: reuse out buffer — capacity must not decrease
    // -----------------------------------------------------------------------

    #[test]
    fn test_freeze_reuses_out_buffer_capacity() {
        // First call: 5 rows, 10 nz (simulated by a larger base).
        let big_base = StageTemplate {
            num_cols: 2,
            num_rows: 5,
            num_nz: 10,
            col_starts: vec![0_i32, 5, 10],
            row_indices: vec![0_i32, 1, 2, 3, 4, 0, 1, 2, 3, 4],
            values: vec![1.0; 10],
            col_lower: vec![0.0, 0.0],
            col_upper: vec![f64::INFINITY, f64::INFINITY],
            objective: vec![1.0, 1.0],
            row_lower: vec![0.0; 5],
            row_upper: vec![f64::INFINITY; 5],
            n_state: 0,
            col_scale: Vec::new(),
            row_scale: Vec::new(),
        };
        let empty_rows = make_empty_row_batch();
        let mut out = StageTemplate::empty();

        freeze_rows_into_template(
            &big_base,
            &empty_rows,
            &mut out,
            &mut FreezeScratch::default(),
        );

        // Capture capacities after the first (larger) call.
        let cap_col_starts = out.col_starts.capacity();
        let cap_row_indices = out.row_indices.capacity();
        let cap_values = out.values.capacity();
        let cap_row_lower = out.row_lower.capacity();
        let cap_row_upper = out.row_upper.capacity();

        // Second call: 4 rows, 8 nz (smaller — must not reallocate downward).
        let small_base = StageTemplate {
            num_cols: 2,
            num_rows: 4,
            num_nz: 8,
            col_starts: vec![0_i32, 4, 8],
            row_indices: vec![0_i32, 1, 2, 3, 0, 1, 2, 3],
            values: vec![1.0; 8],
            col_lower: vec![0.0, 0.0],
            col_upper: vec![f64::INFINITY, f64::INFINITY],
            objective: vec![1.0, 1.0],
            row_lower: vec![0.0; 4],
            row_upper: vec![f64::INFINITY; 4],
            n_state: 0,
            col_scale: Vec::new(),
            row_scale: Vec::new(),
        };

        freeze_rows_into_template(
            &small_base,
            &empty_rows,
            &mut out,
            &mut FreezeScratch::default(),
        );

        assert_eq!(out.num_rows, 4);
        assert_eq!(out.num_nz, 8);

        // Capacities must not have decreased.
        assert!(out.col_starts.capacity() >= cap_col_starts);
        assert!(out.row_indices.capacity() >= cap_row_indices);
        assert!(out.values.capacity() >= cap_values);
        assert!(out.row_lower.capacity() >= cap_row_lower);
        assert!(out.row_upper.capacity() >= cap_row_upper);
    }

    // -----------------------------------------------------------------------
    // Test 6: determinism — two calls produce identical output
    // -----------------------------------------------------------------------

    #[test]
    fn test_freeze_determinism() {
        let base = make_fixture_stage_template();
        let rows = RowBatch {
            num_rows: 2,
            row_starts: vec![0_i32, 2, 3],
            col_indices: vec![0_i32, 2, 1],
            values: vec![-1.0, 0.5, 3.0],
            row_lower: vec![8.0, 12.0],
            row_upper: vec![f64::INFINITY, f64::INFINITY],
        };

        let mut out1 = StageTemplate::empty();
        let mut out2 = StageTemplate::empty();

        freeze_rows_into_template(&base, &rows, &mut out1, &mut FreezeScratch::default());
        freeze_rows_into_template(&base, &rows, &mut out2, &mut FreezeScratch::default());

        assert_eq!(out1.col_starts, out2.col_starts);
        assert_eq!(out1.row_indices, out2.row_indices);
        assert_eq!(out1.values, out2.values);
        assert_eq!(out1.row_lower, out2.row_lower);
        assert_eq!(out1.row_upper, out2.row_upper);
    }

    // -----------------------------------------------------------------------
    // Test 7: multi-column distribution — 4-column base, 3 CSR rows
    // -----------------------------------------------------------------------

    #[test]
    fn test_freeze_multi_column_distribution() {
        // 4-column, 3-row base:
        //   col 0: rows [0,1]         values [1.0, 2.0]
        //   col 1: row  [2]           value  [3.0]
        //   col 2: rows [0,1,2]       values [4.0, 5.0, 6.0]
        //   col 3: (empty)
        // col_starts = [0, 2, 3, 6, 6]
        let base = StageTemplate {
            num_cols: 4,
            num_rows: 3,
            num_nz: 6,
            col_starts: vec![0_i32, 2, 3, 6, 6],
            row_indices: vec![0_i32, 1, 2, 0, 1, 2],
            values: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            col_lower: vec![0.0; 4],
            col_upper: vec![f64::INFINITY; 4],
            objective: vec![0.0; 4],
            row_lower: vec![0.0; 3],
            row_upper: vec![f64::INFINITY; 3],
            n_state: 2,
            col_scale: Vec::new(),
            row_scale: Vec::new(),
        };

        // 3 CSR rows:
        //   row 3 (cut 0): cols [0, 3]
        //   row 4 (cut 1): cols [1, 2]
        //   row 5 (cut 2): cols [0, 2, 3]
        let rows = RowBatch {
            num_rows: 3,
            row_starts: vec![0_i32, 2, 4, 7],
            col_indices: vec![0_i32, 3, 1, 2, 0, 2, 3],
            values: vec![-1.0, 1.0, -2.0, 2.0, -3.0, 3.0, -4.0],
            row_lower: vec![10.0, 20.0, 30.0],
            row_upper: vec![f64::INFINITY; 3],
        };

        let mut out = StageTemplate::empty();
        freeze_rows_into_template(&base, &rows, &mut out, &mut FreezeScratch::default());

        assert_eq!(out.num_rows, 6);
        // col 0: 2 base + 2 cut (rows 3, 5) = 4
        // col 1: 1 base + 1 cut (row 4) = 2
        // col 2: 3 base + 2 cut (rows 4, 5) = 5
        // col 3: 0 base + 2 cut (rows 3, 5) = 2
        // total = 13
        assert_eq!(out.num_nz, 13);
        assert_eq!(out.col_starts, vec![0_i32, 4, 6, 11, 13]);

        // Column 0 entries: base rows [0,1], then cut rows [3,5]
        assert_eq!(&out.row_indices[0..4], &[0_i32, 1, 3, 5]);
        assert_eq!(&out.values[0..4], &[1.0_f64, 2.0, -1.0, -3.0]);

        // Column 1 entries: base row [2], then cut row [4]
        assert_eq!(&out.row_indices[4..6], &[2_i32, 4]);
        assert_eq!(&out.values[4..6], &[3.0_f64, -2.0]);

        // Column 2 entries: base rows [0,1,2], then cut rows [4,5]
        assert_eq!(&out.row_indices[6..11], &[0_i32, 1, 2, 4, 5]);
        assert_eq!(&out.values[6..11], &[4.0_f64, 5.0, 6.0, 2.0, 3.0]);

        // Column 3 entries: base (empty), then cut rows [3,5]
        assert_eq!(&out.row_indices[11..13], &[3_i32, 5]);
        assert_eq!(&out.values[11..13], &[1.0_f64, -4.0]);

        // Row bounds
        assert_eq!(&out.row_lower[3..6], &[10.0_f64, 20.0, 30.0]);
    }

    // -----------------------------------------------------------------------
    // Test 8: MockSolver records num_rows from load_model
    // -----------------------------------------------------------------------

    /// A minimal [`crate::SolverInterface`] implementation that records the
    /// `num_rows` value from the most recent `load_model` call.
    struct MockSolver {
        last_loaded_num_rows: usize,
        stats: SolverStatistics,
    }

    impl MockSolver {
        fn new() -> Self {
            Self {
                last_loaded_num_rows: 0,
                stats: SolverStatistics::default(),
            }
        }
    }

    impl SolverInterface for MockSolver {
        type Profile = MockProfile;

        fn apply_profile(&mut self, _profile: &MockProfile) {}

        fn load_model(&mut self, template: &StageTemplate) {
            self.last_loaded_num_rows = template.num_rows;
            self.stats.load_model_count += 1;
        }

        fn add_rows(&mut self, _rows: &RowBatch) {}

        fn set_row_bounds(&mut self, _indices: &[usize], _lower: &[f64], _upper: &[f64]) {}

        fn set_col_bounds(&mut self, _indices: &[usize], _lower: &[f64], _upper: &[f64]) {}

        fn solve(&mut self, _basis: Option<&Basis>) -> Result<SolutionView<'_>, SolverError> {
            Err(SolverError::InternalError {
                message: "mock".to_string(),
                error_code: None,
            })
        }

        fn get_basis(&mut self, _out: &mut Basis) {}

        fn statistics(&self) -> SolverStatistics {
            self.stats.clone()
        }

        fn statistics_into(&self, out: &mut SolverStatistics) {
            out.copy_from(&self.stats);
        }

        fn name(&self) -> &'static str {
            "Mock"
        }

        fn solver_name_version(&self) -> String {
            "MockSolver 0.0.0".to_string()
        }
    }

    #[test]
    fn test_freeze_load_model_row_count() {
        let base = make_fixture_stage_template();
        let rows = RowBatch {
            num_rows: 3,
            row_starts: vec![0_i32, 1, 2, 3],
            col_indices: vec![0_i32, 1, 2],
            values: vec![-1.0, -1.0, -1.0],
            row_lower: vec![5.0, 6.0, 7.0],
            row_upper: vec![f64::INFINITY; 3],
        };

        let mut out = StageTemplate::empty();
        freeze_rows_into_template(&base, &rows, &mut out, &mut FreezeScratch::default());

        let expected_rows = base.num_rows + rows.num_rows; // 2 + 3 = 5

        let mut solver = MockSolver::new();
        let before = solver.statistics().load_model_count;
        solver.load_model(&out);
        let after = solver.statistics().load_model_count;

        assert_eq!(after - before, 1);
        assert_eq!(solver.last_loaded_num_rows, expected_rows);
    }

    // -----------------------------------------------------------------------
    // Test (extra): empty row_scale + non-zero rows → appended 1.0 entries only
    // -----------------------------------------------------------------------

    #[test]
    fn test_freeze_empty_base_row_scale_with_cut_rows_appends_ones() {
        let base = make_fixture_stage_template(); // row_scale is empty
        let rows = RowBatch {
            num_rows: 2,
            row_starts: vec![0_i32, 1, 2],
            col_indices: vec![0_i32, 0],
            values: vec![-1.0, -2.0],
            row_lower: vec![5.0, 6.0],
            row_upper: vec![f64::INFINITY; 2],
        };
        let mut out = StageTemplate::empty();

        freeze_rows_into_template(&base, &rows, &mut out, &mut FreezeScratch::default());

        // StageTemplate invariant: when non-empty, row_scale.len() == num_rows.
        // base.row_scale was empty but rows.num_rows == 2, so the frozen template
        // materialises a full row_scale of 1.0 (base.num_rows + rows.num_rows == 4).
        assert_eq!(out.row_scale.len(), base.num_rows + rows.num_rows);
        assert!(out.row_scale.iter().all(|&s| s == 1.0));
    }

    // -----------------------------------------------------------------------
    // Test: i32::MAX overflow panics
    // -----------------------------------------------------------------------

    /// Verifies that `freeze_rows_into_template` panics with the expected message
    /// when `base.num_nz + rows_nnz` exceeds `i32::MAX`.
    ///
    /// Release-only: debug asserts on fabricated sizes fire first in debug builds.
    /// Run `cargo test --release` to exercise the `i32::try_from` guard.
    #[test]
    #[cfg(not(debug_assertions))]
    #[should_panic(expected = "total nnz exceeds i32::MAX")]
    fn test_freeze_panics_on_nnz_overflow() {
        // base.num_nz = i32::MAX; rows adds 1 more to trigger the overflow.
        let large_num_nz = usize::try_from(i32::MAX).unwrap();
        let base = StageTemplate {
            num_cols: 0,
            num_rows: 0,
            num_nz: large_num_nz,
            col_starts: vec![0_i32],
            row_indices: vec![],
            values: vec![],
            col_lower: vec![],
            col_upper: vec![],
            objective: vec![],
            row_lower: vec![],
            row_upper: vec![],
            n_state: 0,
            col_scale: Vec::new(),
            row_scale: Vec::new(),
        };
        let rows = RowBatch {
            num_rows: 1,
            row_starts: vec![0_i32, 1],
            col_indices: vec![0_i32],
            values: vec![1.0],
            row_lower: vec![0.0],
            row_upper: vec![f64::INFINITY],
        };
        let mut out = StageTemplate::empty();
        freeze_rows_into_template(&base, &rows, &mut out, &mut FreezeScratch::default());
    }

    // -----------------------------------------------------------------------
    // Test: reusing one scratch across two freezes is bit-identical to a fresh
    // scratch, and the scratch buffers never realloc downward on reuse.
    // -----------------------------------------------------------------------

    #[test]
    fn freeze_twice_same_scratch_is_bit_identical() {
        // Test 7 multi-column fixture: 4-col base, 3 CSR rows.
        let base = StageTemplate {
            num_cols: 4,
            num_rows: 3,
            num_nz: 6,
            col_starts: vec![0_i32, 2, 3, 6, 6],
            row_indices: vec![0_i32, 1, 2, 0, 1, 2],
            values: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            col_lower: vec![0.0; 4],
            col_upper: vec![f64::INFINITY; 4],
            objective: vec![0.0; 4],
            row_lower: vec![0.0; 3],
            row_upper: vec![f64::INFINITY; 3],
            n_state: 2,
            col_scale: Vec::new(),
            row_scale: Vec::new(),
        };
        let rows = RowBatch {
            num_rows: 3,
            row_starts: vec![0_i32, 2, 4, 7],
            col_indices: vec![0_i32, 3, 1, 2, 0, 2, 3],
            values: vec![-1.0, 1.0, -2.0, 2.0, -3.0, 3.0, -4.0],
            row_lower: vec![10.0, 20.0, 30.0],
            row_upper: vec![f64::INFINITY; 3],
        };

        let mut scratch = FreezeScratch::default();

        // First reused-scratch freeze.
        let mut out1 = StageTemplate::empty();
        freeze_rows_into_template(&base, &rows, &mut out1, &mut scratch);

        // Capacities after the first freeze (must not shrink on reuse).
        let cap_cut_nz = scratch.csc_nz_per_col.capacity();
        let cap_col_start = scratch.col_list_start.capacity();
        let cap_col_row = scratch.col_list_row.capacity();
        let cap_col_val = scratch.col_list_val.capacity();
        let cap_write_cursor = scratch.write_cursor.capacity();

        // Second freeze of the SAME inputs reusing the SAME scratch.
        let mut out2 = StageTemplate::empty();
        freeze_rows_into_template(&base, &rows, &mut out2, &mut scratch);

        // Reference freeze with a fresh scratch.
        let mut out_fresh = StageTemplate::empty();
        freeze_rows_into_template(&base, &rows, &mut out_fresh, &mut FreezeScratch::default());

        // Bit-identical outputs across reuse and fresh scratch.
        assert_eq!(out1.col_starts, out2.col_starts);
        assert_eq!(out1.col_starts, out_fresh.col_starts);
        assert_eq!(out1.row_indices, out2.row_indices);
        assert_eq!(out1.row_indices, out_fresh.row_indices);
        assert_eq!(out1.values, out2.values);
        assert_eq!(out1.values, out_fresh.values);
        assert_eq!(out1.row_lower, out2.row_lower);
        assert_eq!(out1.row_lower, out_fresh.row_lower);
        assert_eq!(out1.row_upper, out2.row_upper);
        assert_eq!(out1.row_upper, out_fresh.row_upper);
        assert_eq!(out1.row_scale, out2.row_scale);
        assert_eq!(out1.row_scale, out_fresh.row_scale);

        // No downward realloc on the second (reused) freeze.
        assert!(scratch.csc_nz_per_col.capacity() >= cap_cut_nz);
        assert!(scratch.col_list_start.capacity() >= cap_col_start);
        assert!(scratch.col_list_row.capacity() >= cap_col_row);
        assert!(scratch.col_list_val.capacity() >= cap_col_val);
        assert!(scratch.write_cursor.capacity() >= cap_write_cursor);
    }
}
