//! Per-thread solver workspace, workspace pool, and per-scenario basis store.
//!
//! [`SolverWorkspace`] bundles all mutable per-thread resources needed for one LP solve sequence.
//! [`WorkspacePool`] allocates one workspace per worker thread.
//! [`BasisStore`] provides per-scenario, per-stage basis storage for warm-starting LP solves.

use cobre_solver::{Basis, SolverInterface};

use crate::lp_builder::PatchBuffer;

/// Pre-allocated scratch buffers for noise transformation and simulation.
///
/// Grouped here for readability; individual fields are passed by `&mut`
/// reference to noise transformation functions in `noise.rs`.
#[allow(clippy::struct_field_names)]
pub(crate) struct ScratchBuffers {
    /// Scratch buffer for transformed noise per hydro.
    pub(crate) noise_buf: Vec<f64>,
    /// Scratch buffer for inflow in m³/s (used by simulation pipeline).
    pub(crate) inflow_m3s_buf: Vec<f64>,
    /// Scratch buffer for lag state in lag-major layout (used by inflow truncation).
    pub(crate) lag_matrix_buf: Vec<f64>,
    /// Scratch buffer for evaluated PAR inflows (used by inflow truncation).
    pub(crate) par_inflow_buf: Vec<f64>,
    /// Scratch buffer for solved noise floor (used by inflow truncation).
    pub(crate) eta_floor_buf: Vec<f64>,
    /// Zero-filled scratch buffer for `solve_par_noises` targets (inflow truncation).
    pub(crate) zero_targets_buf: Vec<f64>,
    /// Scratch buffer for NCS column upper bounds (forward/backward pass).
    ///
    /// Sized to `n_stochastic_ncs * max_blocks`. Built per stage by
    /// `transform_ncs_noise` before calling `solver.set_col_bounds`.
    pub(crate) ncs_col_upper_buf: Vec<f64>,
    /// Scratch buffer for NCS column lower bounds (all zeros).
    pub(crate) ncs_col_lower_buf: Vec<f64>,
    /// Pre-computed NCS column indices for `set_col_bounds`.
    pub(crate) ncs_col_indices_buf: Vec<usize>,
    /// Scratch buffer for stochastic load RHS values (forward pass).
    ///
    /// Sized to `n_load_buses * max_blocks`. Built per stage by the forward pass
    /// before calling [`PatchBuffer::fill_load_patches`].
    pub(crate) load_rhs_buf: Vec<f64>,
    /// Scratch buffer for patched `row_lower` values (simulation extraction).
    ///
    /// Sized to the maximum template `row_lower.len()` across stages.
    /// The simulation pipeline clones the template's `row_lower` into this
    /// buffer and overwrites load balance rows with stochastic realizations
    /// before passing to [`extract_stage_result`].
    pub(crate) row_lower_buf: Vec<f64>,
    /// Scratch buffer for z-inflow definition row RHS values (m3/s).
    ///
    /// Filled by `transform_inflow_noise` with `base_h + sigma_h * eta_effective_h`
    /// for each hydro. Used by `PatchBuffer::fill_z_inflow_patches` (Category 5).
    pub(crate) z_inflow_rhs_buf: Vec<f64>,
    /// Scratch buffer for effective (possibly clamped) eta values per hydro.
    ///
    /// Filled by [`compute_effective_eta`] with either raw eta (no truncation)
    /// or clamped eta (truncation active). Sized to `hydro_count`.
    ///
    /// [`compute_effective_eta`]: crate::noise::compute_effective_eta
    pub(crate) effective_eta_buf: Vec<f64>,
    /// Scratch buffer for unscaled primal values.
    ///
    /// When column scaling is active, solver primal values are in scaled
    /// coordinates. This buffer holds the unscaled values after applying
    /// `x_original[j] = col_scale[j] * x_scaled[j]`. Reused across solves
    /// to avoid per-solve allocation.
    pub(crate) unscaled_primal: Vec<f64>,
    /// Scratch buffer for unscaled dual values.
    ///
    /// When row scaling is active, solver dual values are in scaled
    /// coordinates. This buffer holds the unscaled values after applying
    /// `dual_original[i] = row_scale[i] * dual_scaled[i]`. Reused across
    /// solves to avoid per-solve allocation.
    pub(crate) unscaled_dual: Vec<f64>,
}

/// All per-thread mutable resources required for one LP solve sequence.
///
/// Each field is exclusively owned by the thread — there is no shared state
/// between workspaces. Distributed to worker threads via mutable references
/// from a [`WorkspacePool`].
pub struct SolverWorkspace<S: SolverInterface> {
    /// LP solver instance owned exclusively by this workspace.
    pub solver: S,
    /// Pre-allocated row-bound patch buffer.
    pub patch_buf: PatchBuffer,
    /// Scratch buffer for the current state vector.
    pub current_state: Vec<f64>,
    /// Pre-allocated scratch buffers for noise transformation and simulation.
    pub(crate) scratch: ScratchBuffers,
}

impl<S: SolverInterface> SolverWorkspace<S> {
    /// Construct a workspace with the given solver, patch buffer, and state capacity.
    ///
    /// `hydro_count` and `max_par_order` determine the capacities of internal
    /// scratch buffers for the inflow truncation path.  `n_load_buses` and
    /// `max_blocks` determine the capacity of the load RHS scratch buffer.
    #[must_use]
    pub fn new(
        solver: S,
        patch_buf: PatchBuffer,
        n_state: usize,
        hydro_count: usize,
        max_par_order: usize,
        n_load_buses: usize,
        max_blocks: usize,
    ) -> Self {
        Self {
            solver,
            patch_buf,
            current_state: Vec::with_capacity(n_state),
            scratch: ScratchBuffers {
                noise_buf: Vec::with_capacity(hydro_count),
                inflow_m3s_buf: Vec::with_capacity(hydro_count),
                lag_matrix_buf: Vec::with_capacity(max_par_order * hydro_count),
                par_inflow_buf: Vec::with_capacity(hydro_count),
                eta_floor_buf: Vec::with_capacity(hydro_count),
                zero_targets_buf: vec![0.0_f64; hydro_count],
                ncs_col_upper_buf: Vec::new(),
                ncs_col_lower_buf: Vec::new(),
                ncs_col_indices_buf: Vec::new(),
                load_rhs_buf: Vec::with_capacity(n_load_buses * max_blocks),
                row_lower_buf: Vec::new(),
                z_inflow_rhs_buf: Vec::with_capacity(hydro_count),
                effective_eta_buf: Vec::with_capacity(hydro_count),
                unscaled_primal: Vec::new(),
                unscaled_dual: Vec::new(),
            },
        }
    }
}

/// A pool of [`SolverWorkspace`] instances, one per worker thread.
///
/// Create once at algorithm startup via [`WorkspacePool::new`] and distribute
/// workspaces to threads by indexing into [`workspaces`](WorkspacePool::workspaces).
///
/// The pool size equals the number of worker threads. Each workspace is
/// independently allocated and does not share any mutable state with the others.
pub struct WorkspacePool<S: SolverInterface> {
    /// The individual workspaces, indexed by thread number.
    pub workspaces: Vec<SolverWorkspace<S>>,
}

impl<S: SolverInterface> WorkspacePool<S> {
    /// Construct a pool of `n_threads` independently allocated workspaces.
    ///
    /// Each workspace receives a fresh solver instance, patch buffer, and state buffer.
    /// `solver_factory` is called once per thread.
    ///
    /// `n_load_buses` and `max_blocks` size the load RHS scratch buffer and the
    /// Category 4 region of each workspace's [`PatchBuffer`].  Pass `0` for both
    /// when there is no stochastic load.
    #[must_use]
    pub fn new(
        n_threads: usize,
        hydro_count: usize,
        max_par_order: usize,
        n_state: usize,
        n_load_buses: usize,
        max_blocks: usize,
        solver_factory: impl Fn() -> S,
    ) -> Self {
        let workspaces = (0..n_threads)
            .map(|_| SolverWorkspace {
                solver: solver_factory(),
                patch_buf: PatchBuffer::new(hydro_count, max_par_order, n_load_buses, max_blocks),
                current_state: Vec::with_capacity(n_state),
                scratch: ScratchBuffers {
                    noise_buf: Vec::with_capacity(hydro_count),
                    inflow_m3s_buf: Vec::with_capacity(hydro_count),
                    lag_matrix_buf: Vec::with_capacity(max_par_order * hydro_count),
                    par_inflow_buf: Vec::with_capacity(hydro_count),
                    eta_floor_buf: Vec::with_capacity(hydro_count),
                    zero_targets_buf: vec![0.0_f64; hydro_count],
                    ncs_col_upper_buf: Vec::new(),
                    ncs_col_lower_buf: Vec::new(),
                    ncs_col_indices_buf: Vec::new(),
                    load_rhs_buf: Vec::with_capacity(n_load_buses * max_blocks),
                    row_lower_buf: Vec::new(),
                    z_inflow_rhs_buf: Vec::with_capacity(hydro_count),
                    effective_eta_buf: Vec::with_capacity(hydro_count),
                    unscaled_primal: Vec::new(),
                    unscaled_dual: Vec::new(),
                },
            })
            .collect();
        Self { workspaces }
    }

    /// Construct a pool of `n_threads` independently allocated workspaces using
    /// a fallible factory.
    ///
    /// Identical to [`WorkspacePool::new`] except that `solver_factory` returns
    /// `Result<S, E>`. The first error from any factory call is returned
    /// immediately and no partial pool is produced.
    ///
    /// # Arguments
    ///
    /// - `n_threads`: number of worker threads (determines pool size).
    /// - `hydro_count`: number of operating hydro plants (N in the LP layout).
    /// - `max_par_order`: maximum PAR order across all hydros (L in the layout).
    /// - `n_state`: capacity for the `current_state` scratch buffer.
    /// - `n_load_buses`: number of buses with stochastic load noise (M).
    ///   Pass `0` when there is no stochastic load.
    /// - `max_blocks`: maximum block count across all stages.
    ///   Pass `0` when there is no stochastic load.
    /// - `solver_factory`: called once per thread; returns `Result<S, E>`.
    ///
    /// # Errors
    ///
    /// Returns `Err(E)` if any call to `solver_factory` fails.
    pub fn try_new<E>(
        n_threads: usize,
        hydro_count: usize,
        max_par_order: usize,
        n_state: usize,
        n_load_buses: usize,
        max_blocks: usize,
        solver_factory: impl Fn() -> Result<S, E>,
    ) -> Result<Self, E> {
        let mut workspaces = Vec::with_capacity(n_threads);
        for _ in 0..n_threads {
            workspaces.push(SolverWorkspace {
                solver: solver_factory()?,
                patch_buf: PatchBuffer::new(hydro_count, max_par_order, n_load_buses, max_blocks),
                current_state: Vec::with_capacity(n_state),
                scratch: ScratchBuffers {
                    noise_buf: Vec::with_capacity(hydro_count),
                    inflow_m3s_buf: Vec::with_capacity(hydro_count),
                    lag_matrix_buf: Vec::with_capacity(max_par_order * hydro_count),
                    par_inflow_buf: Vec::with_capacity(hydro_count),
                    eta_floor_buf: Vec::with_capacity(hydro_count),
                    zero_targets_buf: vec![0.0_f64; hydro_count],
                    ncs_col_upper_buf: Vec::new(),
                    ncs_col_lower_buf: Vec::new(),
                    ncs_col_indices_buf: Vec::new(),
                    load_rhs_buf: Vec::with_capacity(n_load_buses * max_blocks),
                    row_lower_buf: Vec::new(),
                    z_inflow_rhs_buf: Vec::with_capacity(hydro_count),
                    effective_eta_buf: Vec::with_capacity(hydro_count),
                    unscaled_primal: Vec::new(),
                    unscaled_dual: Vec::new(),
                },
            });
        }
        Ok(Self { workspaces })
    }
}

// ---------------------------------------------------------------------------
// BasisStore
// ---------------------------------------------------------------------------

/// Per-scenario, per-stage basis storage for warm-starting LP solves.
///
/// The store is indexed as `store[scenario_index][stage_index]`. During the
/// forward pass, each worker writes to its disjoint scenario range. During
/// the backward pass, all workers read from any scenario's basis.
///
/// Internally, data is stored flat as
/// `bases[scenario * num_stages + stage]` for cache-friendly sequential
/// access within a single scenario's stage loop.
///
/// # Cut selection interaction
///
/// Basis row statuses are positional: `row_status[i]` corresponds to LP
/// row `i`. When cut selection changes the active cut set between
/// iterations, the number of cut rows in the LP changes and the stored
/// basis row statuses become stale — they no longer align with the
/// current LP row layout.
///
/// **Current behavior (option 1):** We accept the degraded warm-start.
/// `HiGHS` detects the dimension mismatch when `solve_with_basis` is called
/// with a basis whose row count differs from the current LP row count and
/// falls back to a crash start. This is tracked as a `basis_rejection` in
/// [`SolverStatistics`]. The template (non-cut) row statuses remain valid;
/// only the cut row portion becomes meaningless.
///
/// **If degradation is problematic (option 3):** After cut selection runs,
/// discard the cut row statuses from all stored bases, retaining only the
/// template row portion. This gives a clean partial warm-start at zero
/// implementation cost beyond a single truncation. See
/// `docs/specs/backward-pass-performance-spec.md` section 6.1 for the full
/// design discussion.
///
/// [`SolverStatistics`]: cobre_solver::SolverStatistics
pub struct BasisStore {
    /// Flat storage: `bases[scenario * num_stages + stage]`.
    bases: Vec<Option<Basis>>,
    /// Number of stages per scenario.
    num_stages: usize,
}

impl BasisStore {
    /// Allocate a new store for `num_scenarios` scenarios and `num_stages`
    /// stages, with every slot initialised to `None`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cobre_sddp::workspace::BasisStore;
    ///
    /// let store = BasisStore::new(4, 10);
    /// assert_eq!(store.num_scenarios(), 4);
    /// assert_eq!(store.num_stages(), 10);
    /// assert!(store.get(0, 0).is_none());
    /// ```
    #[must_use]
    pub fn new(num_scenarios: usize, num_stages: usize) -> Self {
        Self {
            bases: vec![None; num_scenarios * num_stages],
            num_stages,
        }
    }

    /// Return the number of scenarios this store was allocated for.
    #[must_use]
    pub fn num_scenarios(&self) -> usize {
        if self.num_stages == 0 {
            0
        } else {
            self.bases.len() / self.num_stages
        }
    }

    /// Return the number of stages.
    #[must_use]
    pub fn num_stages(&self) -> usize {
        self.num_stages
    }

    /// Get an immutable reference to the basis at `[scenario][stage]`.
    ///
    /// Returns `None` if the slot has not yet been populated.
    #[must_use]
    pub fn get(&self, scenario: usize, stage: usize) -> Option<&Basis> {
        self.bases[scenario * self.num_stages + stage].as_ref()
    }

    /// Get a mutable reference to the basis slot at `[scenario][stage]`.
    pub fn get_mut(&mut self, scenario: usize, stage: usize) -> &mut Option<Basis> {
        &mut self.bases[scenario * self.num_stages + stage]
    }

    /// Split the store into `n_workers` disjoint mutable sub-views by scenario
    /// range, one per worker.
    ///
    /// Worker `w` receives the scenarios in the range produced by
    /// `partition(num_scenarios, n_workers, w)`. Each [`BasisStoreSliceMut`]
    /// carries its scenario offset so that callers can index using absolute
    /// scenario indices.
    ///
    /// When `n_workers` exceeds `num_scenarios`, some workers receive empty
    /// slices (`start == end`), which is valid — their slice covers zero
    /// scenarios.
    ///
    /// # Panics (debug only)
    ///
    /// Panics if `n_workers == 0`.
    #[must_use]
    pub fn split_workers_mut(&mut self, n_workers: usize) -> Vec<BasisStoreSliceMut<'_>> {
        debug_assert!(n_workers > 0, "n_workers must be > 0");
        let total_scenarios = self.num_scenarios();
        let mut slices = Vec::with_capacity(n_workers);
        let mut remainder = self.bases.as_mut_slice();
        let mut offset = 0usize;

        for w in 0..n_workers {
            let (start, end) = crate::forward::partition(total_scenarios, n_workers, w);
            let count = end - start;
            let (left, rest) = remainder.split_at_mut(count * self.num_stages);
            remainder = rest;
            slices.push(BasisStoreSliceMut {
                bases: left,
                scenario_offset: offset,
                num_stages: self.num_stages,
            });
            offset += count;
        }
        slices
    }
}

/// A mutable sub-view of a [`BasisStore`] covering a contiguous range of
/// scenarios.
///
/// Produced by [`BasisStore::split_workers_mut`]. Each slice is exclusive
/// to one worker thread; multiple slices can coexist because they cover
/// disjoint memory regions.
pub struct BasisStoreSliceMut<'a> {
    /// Sub-slice of the flat basis array for this worker's scenario range.
    bases: &'a mut [Option<Basis>],
    /// Absolute scenario index of the first scenario in this slice.
    scenario_offset: usize,
    /// Number of stages per scenario.
    num_stages: usize,
}

impl BasisStoreSliceMut<'_> {
    /// Get an immutable reference to the basis at absolute scenario index
    /// `scenario` and stage `stage`.
    ///
    /// Returns `None` if the slot has not yet been populated.
    ///
    /// # Panics
    ///
    /// Panics if `scenario < self.scenario_offset` (scenario not in this slice).
    #[must_use]
    pub fn get(&self, scenario: usize, stage: usize) -> Option<&Basis> {
        let local = scenario - self.scenario_offset;
        self.bases[local * self.num_stages + stage].as_ref()
    }

    /// Get a mutable reference to the basis slot at absolute scenario index
    /// `scenario` and stage `stage`.
    ///
    /// # Panics
    ///
    /// Panics if `scenario < self.scenario_offset` (scenario not in this slice).
    pub fn get_mut(&mut self, scenario: usize, stage: usize) -> &mut Option<Basis> {
        let local = scenario - self.scenario_offset;
        &mut self.bases[local * self.num_stages + stage]
    }
}

#[cfg(test)]
mod tests {
    use super::{BasisStore, SolverWorkspace, WorkspacePool};
    use cobre_solver::{
        Basis, SolutionView, SolverError, SolverInterface, SolverStatistics,
        types::{RowBatch, StageTemplate},
    };

    /// Minimal no-op solver for workspace tests.
    struct MockSolver;

    impl SolverInterface for MockSolver {
        fn load_model(&mut self, _t: &StageTemplate) {}
        fn add_rows(&mut self, _r: &RowBatch) {}
        fn set_row_bounds(&mut self, _i: &[usize], _l: &[f64], _u: &[f64]) {}
        fn set_col_bounds(&mut self, _i: &[usize], _l: &[f64], _u: &[f64]) {}
        fn solve(&mut self) -> Result<SolutionView<'_>, SolverError> {
            Err(SolverError::InternalError {
                message: "mock".into(),
                error_code: None,
            })
        }
        fn reset(&mut self) {}
        fn get_basis(&mut self, _out: &mut Basis) {}
        fn solve_with_basis(&mut self, _b: &Basis) -> Result<SolutionView<'_>, SolverError> {
            Err(SolverError::InternalError {
                message: "mock".into(),
                error_code: None,
            })
        }
        fn statistics(&self) -> SolverStatistics {
            SolverStatistics::default()
        }
        fn name(&self) -> &'static str {
            "Mock"
        }
    }

    /// Compile-time assertion that `SolverWorkspace<MockSolver>` is `Send`.
    fn assert_send<T: Send>() {}

    #[test]
    fn test_workspace_send_bound() {
        assert_send::<SolverWorkspace<MockSolver>>();
    }

    #[test]
    fn test_workspace_pool_size() {
        let pool = WorkspacePool::new(4, 3, 2, 9, 0, 0, || MockSolver);
        assert_eq!(pool.workspaces.len(), 4);
    }

    #[test]
    fn test_workspace_buffer_dimensions() {
        // N=3, L=2 → patch_buf length = 3*(2+2) + 3 (z-inflow) = 15
        // n_state=9 → current_state capacity = 9
        let pool = WorkspacePool::new(4, 3, 2, 9, 0, 0, || MockSolver);
        for ws in &pool.workspaces {
            assert_eq!(ws.patch_buf.indices.len(), 15, "patch_buf length");
            assert_eq!(ws.current_state.capacity(), 9, "current_state capacity");
            assert_eq!(ws.current_state.len(), 0, "current_state starts empty");
        }
    }

    #[test]
    fn test_workspace_pool_zero_threads() {
        let pool = WorkspacePool::new(0, 3, 2, 9, 0, 0, || MockSolver);
        assert_eq!(pool.workspaces.len(), 0);
    }

    #[test]
    fn test_workspace_pool_single_thread() {
        let pool = WorkspacePool::new(1, 0, 0, 0, 0, 0, || MockSolver);
        assert_eq!(pool.workspaces.len(), 1);
        assert_eq!(pool.workspaces[0].patch_buf.indices.len(), 0);
    }

    #[test]
    fn test_workspace_pool_each_solver_independent() {
        // Factory is called n_threads times; each workspace gets its own instance.
        // Verify by checking pool size matches factory call expectation.
        let n = 6;
        let pool = WorkspacePool::new(n, 1, 0, 1, 0, 0, || MockSolver);
        assert_eq!(pool.workspaces.len(), n);
    }

    // ---------------------------------------------------------------------------
    // BasisStore tests
    // ---------------------------------------------------------------------------

    #[test]
    fn basis_store_new_all_none() {
        let store = BasisStore::new(3, 5);
        assert_eq!(store.num_scenarios(), 3);
        assert_eq!(store.num_stages(), 5);
        for s in 0..3 {
            for t in 0..5 {
                assert!(
                    store.get(s, t).is_none(),
                    "slot [{s}][{t}] must start as None"
                );
            }
        }
    }

    #[test]
    fn basis_store_get_mut_set_and_retrieve() {
        let mut store = BasisStore::new(2, 3);
        let basis = Basis::new(4, 2);
        *store.get_mut(1, 2) = Some(basis);
        assert!(store.get(1, 2).is_some());
        assert!(store.get(0, 0).is_none());
        assert!(store.get(1, 0).is_none());
    }

    #[test]
    fn basis_store_zero_scenarios() {
        let store = BasisStore::new(0, 5);
        assert_eq!(store.num_scenarios(), 0);
        assert_eq!(store.num_stages(), 5);
    }

    #[test]
    fn basis_store_zero_stages() {
        let store = BasisStore::new(3, 0);
        assert_eq!(store.num_scenarios(), 0);
        assert_eq!(store.num_stages(), 0);
    }

    #[test]
    fn basis_store_split_workers_mut_disjoint_writes() {
        // 4 scenarios, 3 stages, 2 workers.
        // Worker 0 covers scenarios 0..2; worker 1 covers scenarios 2..4.
        let mut store = BasisStore::new(4, 3);
        let mut slices = store.split_workers_mut(2);

        // Worker 0 writes to scenario 0 stage 1.
        *slices[0].get_mut(0, 1) = Some(Basis::new(2, 1));
        // Worker 1 writes to scenario 3 stage 2.
        *slices[1].get_mut(3, 2) = Some(Basis::new(2, 1));

        // Drop slices to release the borrow on store.
        drop(slices);

        assert!(
            store.get(0, 1).is_some(),
            "scenario 0 stage 1 must be populated"
        );
        assert!(
            store.get(3, 2).is_some(),
            "scenario 3 stage 2 must be populated"
        );
        assert!(store.get(0, 0).is_none());
        assert!(store.get(3, 0).is_none());
    }

    #[test]
    fn basis_store_split_single_worker() {
        let mut store = BasisStore::new(3, 2);
        let mut slices = store.split_workers_mut(1);
        *slices[0].get_mut(2, 1) = Some(Basis::new(1, 0));
        drop(slices);
        assert!(store.get(2, 1).is_some());
    }

    #[test]
    fn basis_store_split_more_workers_than_scenarios() {
        // 2 scenarios, 4 workers → workers 2 and 3 receive empty slices.
        let mut store = BasisStore::new(2, 3);
        let slices = store.split_workers_mut(4);
        assert_eq!(slices.len(), 4);
        // Workers 0 and 1 cover 1 scenario each; workers 2 and 3 cover 0.
        assert_eq!(slices[0].bases.len(), 3); // 1 scenario × 3 stages
        assert_eq!(slices[1].bases.len(), 3);
        assert_eq!(slices[2].bases.len(), 0);
        assert_eq!(slices[3].bases.len(), 0);
    }

    #[test]
    fn basis_store_slice_offset_correct() {
        // 6 scenarios, 2 stages, 3 workers → 2 scenarios each.
        let mut store = BasisStore::new(6, 2);
        let mut slices = store.split_workers_mut(3);

        // Worker 1 covers absolute scenarios 2..4.
        *slices[1].get_mut(2, 0) = Some(Basis::new(1, 0));
        *slices[1].get_mut(3, 1) = Some(Basis::new(1, 0));
        drop(slices);

        assert!(store.get(2, 0).is_some());
        assert!(store.get(3, 1).is_some());
        assert!(store.get(0, 0).is_none());
        assert!(store.get(4, 0).is_none());
    }
}
