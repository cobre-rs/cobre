//! Per-thread solver workspace, workspace pool, and per-scenario basis store.
//!
//! This module provides three facilities:
//!
//! - [`SolverWorkspace`]: bundles all mutable per-thread resources needed for
//!   one LP solve sequence — a solver instance, a patch buffer, and a
//!   current-state buffer. Each worker thread owns exactly one workspace
//!   for the duration of a parallel loop, eliminating contention on shared
//!   mutable state.
//!
//! - [`WorkspacePool`]: a collection of `N` independently allocated workspaces,
//!   one per worker thread. Created once at algorithm startup and reused across
//!   iterations.
//!
//! - [`BasisStore`]: per-scenario, per-stage basis storage for warm-starting LP
//!   solves. Indexed as `store[scenario][stage]`. The forward pass writes each
//!   scenario's optimal basis at each stage; the backward pass reads the basis
//!   for the matching forward scenario to warm-start the LP at that stage.
//!
//! ## Design
//!
//! LP solvers backed by C libraries (`HiGHS`, CLP) maintain mutable internal
//! state — factorization workspace, working arrays — that is not thread-safe.
//! Concurrent access to a single solver instance causes data corruption. The
//! standard pattern for parallelizing solver loops is to give each thread its
//! own solver instance and patch buffer. This module codifies that pattern.
//!
//! The `BasisStore` replaces the previous per-worker per-stage basis cache,
//! which was incorrect: when a worker processes multiple forward scenarios it
//! overwrote the basis at each stage, losing prior scenarios' bases. The
//! backward pass trial point `m` must warm-start from the exact basis that was
//! optimal for forward scenario `m` at that stage, which requires a separate
//! slot per `(scenario, stage)` pair.
//!
//! ## Thread safety
//!
//! [`SolverWorkspace<S>`] is `Send` whenever `S: Send`. The `Send` bound is
//! required on [`SolverInterface`] itself, so all concrete solver backends
//! satisfy it. No `Arc` or `Mutex` is used: the pool owns all workspaces
//! outright, and callers distribute them via exclusive mutable references.
//!
//! [`BasisStoreSliceMut`] slices are produced by [`BasisStore::split_workers_mut`]
//! and cover disjoint scenario ranges. Each slice carries only its own
//! sub-slice of the underlying storage, so parallel workers can hold multiple
//! slices simultaneously without violating Rust's aliasing rules.
//!
//! ## Example
//!
//! ```rust
//! use cobre_sddp::workspace::{SolverWorkspace, WorkspacePool};
//! use cobre_solver::{Basis, SolverInterface, SolutionView, SolverError, SolverStatistics};
//! use cobre_solver::types::{RowBatch, StageTemplate};
//!
//! struct NullSolver;
//!
//! impl SolverInterface for NullSolver {
//!     fn load_model(&mut self, _t: &StageTemplate) {}
//!     fn add_rows(&mut self, _r: &RowBatch) {}
//!     fn set_row_bounds(&mut self, _i: &[usize], _l: &[f64], _u: &[f64]) {}
//!     fn set_col_bounds(&mut self, _i: &[usize], _l: &[f64], _u: &[f64]) {}
//!     fn solve(&mut self) -> Result<SolutionView<'_>, SolverError> {
//!         Err(SolverError::InternalError { message: "null".into(), error_code: None })
//!     }
//!     fn reset(&mut self) {}
//!     fn get_basis(&mut self, _out: &mut Basis) {}
//!     fn solve_with_basis(&mut self, _b: &Basis) -> Result<SolutionView<'_>, SolverError> {
//!         Err(SolverError::InternalError { message: "null".into(), error_code: None })
//!     }
//!     fn statistics(&self) -> SolverStatistics { SolverStatistics::default() }
//!     fn name(&self) -> &'static str { "Null" }
//! }
//!
//! // 4 threads, 3 hydros, AR(2), 9-element state
//! let pool = WorkspacePool::new(4, 3, 2, 9, || NullSolver);
//! assert_eq!(pool.workspaces.len(), 4);
//! assert_eq!(pool.workspaces[0].patch_buf.indices.len(), 12); // 3*(2+2)
//! assert_eq!(pool.workspaces[0].current_state.capacity(), 9);
//! ```

use cobre_solver::{Basis, SolverInterface};

use crate::lp_builder::PatchBuffer;

/// All per-thread mutable resources required for one LP solve sequence.
///
/// Bundle one of these per worker thread. Each field is exclusively owned by
/// the thread — there is no shared state between workspaces. When a parallel
/// loop distributes work across `N` threads, each thread is given a mutable
/// reference to its own workspace from a [`WorkspacePool`].
///
/// # Fields
///
/// - `solver`: the LP solver instance for this thread. Not thread-safe; must
///   not be shared between threads.
/// - `patch_buf`: pre-allocated row-bound patch buffer. Reused across all
///   solves in the thread to avoid per-solve allocation on the hot path.
/// - `current_state`: scratch buffer for the current state vector. Pre-allocated
///   to avoid hot-path allocation.
/// - `noise_buf`: scratch buffer for transformed noise values `ζ*base + ζ*σ*η`.
///   Pre-allocated to hydro count; cleared and refilled at each stage solve.
/// - `inflow_m3s_buf`: scratch buffer for inflow in m³/s (`noise_buf` / ζ).
///   Used by the simulation pipeline to pass inflow values to result extraction.
pub struct SolverWorkspace<S: SolverInterface> {
    /// LP solver instance owned exclusively by this workspace.
    pub solver: S,
    /// Pre-allocated row-bound patch buffer for scenario-dependent RHS updates.
    pub patch_buf: PatchBuffer,
    /// Scratch buffer for the current state vector. Capacity equals `n_state`.
    pub current_state: Vec<f64>,
    /// Scratch buffer for transformed noise: `ζ*base + ζ*σ*η` per hydro.
    /// Capacity equals `hydro_count`.
    pub(crate) noise_buf: Vec<f64>,
    /// Scratch buffer for inflow in m³/s, derived from `noise_buf / ζ`.
    /// Used by the simulation pipeline for output reporting.
    pub(crate) inflow_m3s_buf: Vec<f64>,
}

impl<S: SolverInterface> SolverWorkspace<S> {
    /// Construct a workspace with the given solver, patch buffer, and state capacity.
    ///
    /// `hydro_count` determines the pre-allocated capacity of the internal
    /// noise scratch buffer. Set to the number of hydro plants in the LP model.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cobre_sddp::workspace::SolverWorkspace;
    /// use cobre_sddp::lp_builder::PatchBuffer;
    /// use cobre_solver::{Basis, SolverInterface, SolutionView, SolverError, SolverStatistics};
    /// use cobre_solver::types::{RowBatch, StageTemplate};
    ///
    /// struct NullSolver;
    ///
    /// impl SolverInterface for NullSolver {
    ///     fn load_model(&mut self, _t: &StageTemplate) {}
    ///     fn add_rows(&mut self, _r: &RowBatch) {}
    ///     fn set_row_bounds(&mut self, _i: &[usize], _l: &[f64], _u: &[f64]) {}
    ///     fn set_col_bounds(&mut self, _i: &[usize], _l: &[f64], _u: &[f64]) {}
    ///     fn solve(&mut self) -> Result<SolutionView<'_>, SolverError> {
    ///         Err(SolverError::InternalError { message: "null".into(), error_code: None })
    ///     }
    ///     fn reset(&mut self) {}
    ///     fn get_basis(&mut self, _out: &mut Basis) {}
    ///     fn solve_with_basis(&mut self, _b: &Basis) -> Result<SolutionView<'_>, SolverError> {
    ///         Err(SolverError::InternalError { message: "null".into(), error_code: None })
    ///     }
    ///     fn statistics(&self) -> SolverStatistics { SolverStatistics::default() }
    ///     fn name(&self) -> &'static str { "Null" }
    /// }
    ///
    /// let ws = SolverWorkspace::new(NullSolver, PatchBuffer::new(3, 2), 9, 3);
    /// assert_eq!(ws.current_state.capacity(), 9);
    /// ```
    #[must_use]
    pub fn new(solver: S, patch_buf: PatchBuffer, n_state: usize, hydro_count: usize) -> Self {
        Self {
            solver,
            patch_buf,
            current_state: Vec::with_capacity(n_state),
            noise_buf: Vec::with_capacity(hydro_count),
            inflow_m3s_buf: Vec::with_capacity(hydro_count),
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
    /// Each workspace receives:
    ///
    /// - A fresh `S` instance from `solver_factory`.
    /// - A [`PatchBuffer`] sized for `hydro_count` hydros and `max_par_order`
    ///   AR lag order: `hydro_count * (2 + max_par_order)` patch slots.
    /// - A `Vec<f64>` with capacity `n_state` and length zero.
    ///
    /// # Arguments
    ///
    /// - `n_threads`: number of worker threads (determines pool size).
    /// - `hydro_count`: number of operating hydro plants (N in the LP layout).
    /// - `max_par_order`: maximum PAR order across all hydros (L in the layout).
    /// - `n_state`: capacity for the `current_state` scratch buffer.
    /// - `solver_factory`: called once per thread to create an independent solver
    ///   instance. Must not share mutable state between calls.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cobre_sddp::workspace::WorkspacePool;
    /// use cobre_solver::{Basis, SolverInterface, SolutionView, SolverError, SolverStatistics};
    /// use cobre_solver::types::{RowBatch, StageTemplate};
    ///
    /// struct NullSolver;
    ///
    /// impl SolverInterface for NullSolver {
    ///     fn load_model(&mut self, _t: &StageTemplate) {}
    ///     fn add_rows(&mut self, _r: &RowBatch) {}
    ///     fn set_row_bounds(&mut self, _i: &[usize], _l: &[f64], _u: &[f64]) {}
    ///     fn set_col_bounds(&mut self, _i: &[usize], _l: &[f64], _u: &[f64]) {}
    ///     fn solve(&mut self) -> Result<SolutionView<'_>, SolverError> {
    ///         Err(SolverError::InternalError { message: "null".into(), error_code: None })
    ///     }
    ///     fn reset(&mut self) {}
    ///     fn get_basis(&mut self, _out: &mut Basis) {}
    ///     fn solve_with_basis(&mut self, _b: &Basis) -> Result<SolutionView<'_>, SolverError> {
    ///         Err(SolverError::InternalError { message: "null".into(), error_code: None })
    ///     }
    ///     fn statistics(&self) -> SolverStatistics { SolverStatistics::default() }
    ///     fn name(&self) -> &'static str { "Null" }
    /// }
    ///
    /// let pool = WorkspacePool::new(2, 3, 2, 9, || NullSolver);
    /// assert_eq!(pool.workspaces.len(), 2);
    /// ```
    #[must_use]
    pub fn new(
        n_threads: usize,
        hydro_count: usize,
        max_par_order: usize,
        n_state: usize,
        solver_factory: impl Fn() -> S,
    ) -> Self {
        let workspaces = (0..n_threads)
            .map(|_| SolverWorkspace {
                solver: solver_factory(),
                patch_buf: PatchBuffer::new(hydro_count, max_par_order),
                current_state: Vec::with_capacity(n_state),
                noise_buf: Vec::with_capacity(hydro_count),
                inflow_m3s_buf: Vec::with_capacity(hydro_count),
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
        solver_factory: impl Fn() -> Result<S, E>,
    ) -> Result<Self, E> {
        let mut workspaces = Vec::with_capacity(n_threads);
        for _ in 0..n_threads {
            workspaces.push(SolverWorkspace {
                solver: solver_factory()?,
                patch_buf: PatchBuffer::new(hydro_count, max_par_order),
                current_state: Vec::with_capacity(n_state),
                noise_buf: Vec::with_capacity(hydro_count),
                inflow_m3s_buf: Vec::with_capacity(hydro_count),
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
        let pool = WorkspacePool::new(4, 3, 2, 9, || MockSolver);
        assert_eq!(pool.workspaces.len(), 4);
    }

    #[test]
    fn test_workspace_buffer_dimensions() {
        // N=3, L=2 → patch_buf length = 3*(2+2) = 12
        // n_state=9 → current_state capacity = 9
        let pool = WorkspacePool::new(4, 3, 2, 9, || MockSolver);
        for ws in &pool.workspaces {
            assert_eq!(ws.patch_buf.indices.len(), 12, "patch_buf length");
            assert_eq!(ws.current_state.capacity(), 9, "current_state capacity");
            assert_eq!(ws.current_state.len(), 0, "current_state starts empty");
        }
    }

    #[test]
    fn test_workspace_pool_zero_threads() {
        let pool = WorkspacePool::new(0, 3, 2, 9, || MockSolver);
        assert_eq!(pool.workspaces.len(), 0);
    }

    #[test]
    fn test_workspace_pool_single_thread() {
        let pool = WorkspacePool::new(1, 0, 0, 0, || MockSolver);
        assert_eq!(pool.workspaces.len(), 1);
        assert_eq!(pool.workspaces[0].patch_buf.indices.len(), 0);
    }

    #[test]
    fn test_workspace_pool_each_solver_independent() {
        // Factory is called n_threads times; each workspace gets its own instance.
        // Verify by checking pool size matches factory call expectation.
        let n = 6;
        let pool = WorkspacePool::new(n, 1, 0, 1, || MockSolver);
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
