//! Per-thread solver workspace, workspace pool, and per-scenario basis store.
//!
//! [`SolverWorkspace`] bundles all mutable per-thread resources needed for one LP solve sequence.
//! [`WorkspacePool`] allocates one workspace per worker thread.
//! [`BasisStore`] provides per-scenario, per-stage basis storage for warm-starting LP solves.

use cobre_solver::{Basis, SolverInterface};

// ---------------------------------------------------------------------------
// CapturedBasis
// ---------------------------------------------------------------------------

/// A solver basis augmented with slot-tracking metadata for cut-set-aware
/// warm-start reconstruction.
///
/// `CapturedBasis` wraps a raw [`Basis`] and attaches two pieces of metadata
/// that the reconstruction algorithm (tickets 002–004) needs:
///
/// - `base_row_count`: the number of template (non-cut) rows in the LP when
///   the basis was captured.  Row statuses at indices `0..base_row_count`
///   belong to template rows and are always valid.
///
/// - `cut_row_slots`: maps each cut row in the captured basis to the cut pool
///   slot it occupied at capture time.  Entry `i` corresponds to LP row
///   `base_row_count + i`.  Length must equal
///   `basis.row_status.len() - base_row_count` when both are populated;
///   tickets 003 and 004 enforce this invariant — this ticket only declares it.
///
/// - `state_at_capture`: the state vector `x_hat` at which the basis was
///   captured.  Used by the backward warm-start to evaluate newly added cuts
///   at the correct operating point.
///
/// # Zero-allocation design
///
/// `cut_row_slots` and `state_at_capture` are sized via explicit capacity
/// parameters in [`CapturedBasis::new`] so the forward capture site can
/// pre-allocate once and reuse the same `CapturedBasis` on subsequent
/// iterations without heap reallocation.
#[derive(Clone, Debug)]
pub struct CapturedBasis {
    /// The underlying solver basis (row and column statuses).
    pub basis: Basis,
    /// Number of template (non-cut) LP rows at capture time.
    pub base_row_count: usize,
    /// Cut pool slot for each cut row in `basis.row_status[base_row_count..]`.
    /// Empty until ticket 003 populates it during forward capture.
    pub cut_row_slots: Vec<u32>,
    /// State vector `x_hat` at which this basis was captured.
    /// Empty until ticket 003 populates it during forward capture.
    pub state_at_capture: Vec<f64>,
}

impl CapturedBasis {
    /// Construct an empty `CapturedBasis` with the given capacities.
    ///
    /// - `num_cols` / `num_rows`: forwarded to [`Basis::new`].
    /// - `base_row_count`: stored as-is; typically `ctx.templates[t].num_rows`.
    /// - `cut_slot_capacity`: pre-allocated capacity for `cut_row_slots`
    ///   (`basis_row_capacity - base_row_count` in the forward pass).
    /// - `n_state`: pre-allocated capacity for `state_at_capture`.
    ///
    /// `cut_row_slots` and `state_at_capture` start empty (length 0);
    /// tickets 003/004 push into them on the hot path without allocation.
    #[must_use]
    pub fn new(
        num_cols: usize,
        num_rows: usize,
        base_row_count: usize,
        cut_slot_capacity: usize,
        n_state: usize,
    ) -> Self {
        Self {
            basis: Basis::new(num_cols, num_rows),
            base_row_count,
            cut_row_slots: Vec::with_capacity(cut_slot_capacity),
            state_at_capture: Vec::with_capacity(n_state),
        }
    }

    /// Clear slot and state metadata in place.
    ///
    /// Does **not** touch `basis` — the solver's `get_basis` call overwrites
    /// that on the next capture.  Keeps the allocated capacity of both vectors
    /// so subsequent pushes are allocation-free.
    pub fn clear(&mut self) {
        self.cut_row_slots.clear();
        self.state_at_capture.clear();
    }
}

use crate::lp_builder::PatchBuffer;
use crate::risk_measure::BackwardOutcome;

/// Sizing parameters shared by [`SolverWorkspace`], [`WorkspacePool`], and
/// `ScratchBuffers` constructors.
///
/// Grouping these eight dimensions into one struct keeps constructor argument
/// counts within the clippy budget while making the sizing relationship
/// explicit: every workspace allocates a [`PatchBuffer`], `ScratchBuffers`,
/// and `BackwardAccumulators` from exactly these values.
///
/// Set `max_openings`, `initial_pool_capacity`, and `n_state` to `0` for
/// simulation-only workspaces that do not participate in the backward pass.
/// The `BackwardAccumulators` buffers will then start empty and grow
/// on-demand via the growth-only resize semantics in the backward pass.
#[derive(Clone, Copy, Debug, Default)]
pub struct WorkspaceSizing {
    /// Number of hydro plants in the study.
    pub hydro_count: usize,
    /// Maximum PAR order across all hydro plants.
    pub max_par_order: usize,
    /// Number of load buses (0 if no stochastic load).
    pub n_load_buses: usize,
    /// Maximum number of blocks per stage (0 if no stochastic load).
    pub max_blocks: usize,
    /// PAR order of the downstream (coarser) resolution for multi-resolution
    /// studies. Pass `0` for uniform-resolution studies (no downstream
    /// transition).
    pub downstream_par_order: usize,
    /// Maximum number of openings across all successor stages. Used to
    /// pre-size `BackwardAccumulators::outcomes`. Pass `0` for
    /// simulation-only workspaces.
    pub max_openings: usize,
    /// Initial cut pool capacity for pre-sizing
    /// `BackwardAccumulators::slot_increments`. Pass `0` for
    /// simulation-only workspaces.
    pub initial_pool_capacity: usize,
    /// State dimension `n_state` for pre-sizing
    /// `BackwardAccumulators::agg_coefficients`. Pass `0` for
    /// simulation-only workspaces.
    pub n_state: usize,
}

/// Pre-allocated accumulators for the backward pass trial-point loop.
///
/// Survives across stages and trial points without per-call allocation.
/// Buffers grow monotonically (never shrink) using growth-only resize
/// semantics — excess capacity from earlier stages is retained and reused.
///
/// Each rayon worker owns an exclusive [`SolverWorkspace`] and therefore an
/// exclusive `BackwardAccumulators` instance; no synchronisation is needed.
#[derive(Default)]
pub(crate) struct BackwardAccumulators {
    /// Per-opening backward outcomes. Grown monotonically to the maximum
    /// `n_openings` seen so far via `push`.
    pub(crate) outcomes: Vec<BackwardOutcome>,
    /// Per-slot binding count, indexed by cut pool slot. Grown via
    /// `.resize(pop, 0)` and zeroed per trial point via `.fill(0)`.
    pub(crate) slot_increments: Vec<u64>,
    /// Scratch buffer for aggregated cut coefficients (`n_state` entries).
    /// Written by `aggregate_weighted_into` and then copied into the owned
    /// `Vec<f64>` stored in each [`StagedCut`].
    pub(crate) agg_coefficients: Vec<f64>,
    /// Per-worker metadata sync contribution, indexed by cut pool slot.
    ///
    /// Accumulates binding increments across all trial points processed by
    /// this worker for a given stage. Grown via `.resize(pop, 0)` when the
    /// pool grows, and zeroed once per stage (not per trial point) via
    /// `.fill(0)`. After the parallel region the sequential merge phase sums
    /// contributions across all workers into `metadata_sync_buf`, replacing
    /// the old per-`StagedCut` `binding_increments` Vec iteration.
    pub(crate) metadata_sync_contribution: Vec<u64>,
}

impl BackwardAccumulators {
    /// Allocate accumulators pre-sized from the given workspace dimensions.
    ///
    /// `max_openings`, `initial_pool_capacity`, and `n_state` may all be
    /// `0` for simulation-only workspaces; buffers will then start empty
    /// and grow lazily on the first backward pass stage.
    pub(crate) fn new(max_openings: usize, initial_pool_capacity: usize, n_state: usize) -> Self {
        let outcomes = (0..max_openings)
            .map(|_| BackwardOutcome {
                intercept: 0.0,
                coefficients: vec![0.0_f64; n_state],
                objective_value: 0.0,
            })
            .collect();
        Self {
            outcomes,
            slot_increments: vec![0u64; initial_pool_capacity],
            agg_coefficients: vec![0.0_f64; n_state],
            metadata_sync_contribution: vec![0u64; initial_pool_capacity],
        }
    }
}

/// Pre-allocated scratch buffers for noise transformation and simulation.
///
/// Grouped here for readability; individual fields are passed by `&mut`
/// reference to noise transformation functions in `noise.rs`.
#[allow(clippy::struct_field_names)]
pub(crate) struct ScratchBuffers {
    pub(crate) noise_buf: Vec<f64>,
    pub(crate) inflow_m3s_buf: Vec<f64>,
    pub(crate) lag_matrix_buf: Vec<f64>,
    pub(crate) par_inflow_buf: Vec<f64>,
    pub(crate) eta_floor_buf: Vec<f64>,
    pub(crate) zero_targets_buf: Vec<f64>,
    pub(crate) ncs_col_upper_buf: Vec<f64>,
    pub(crate) ncs_col_lower_buf: Vec<f64>,
    pub(crate) ncs_col_indices_buf: Vec<usize>,
    pub(crate) load_rhs_buf: Vec<f64>,
    pub(crate) row_lower_buf: Vec<f64>,
    pub(crate) z_inflow_rhs_buf: Vec<f64>,
    pub(crate) effective_eta_buf: Vec<f64>,
    pub(crate) unscaled_primal: Vec<f64>,
    pub(crate) unscaled_dual: Vec<f64>,
    // Used by accumulate_and_shift_lag_state (ticket-004 / ticket-010).
    pub(crate) lag_accumulator: Vec<f64>,
    pub(crate) lag_weight_accum: f64,
    // Downstream ring buffer for multi-resolution lag accumulation (ticket-010).
    pub(crate) downstream_accumulator: Vec<f64>,
    pub(crate) downstream_weight_accum: f64,
    // Slot-major: `completed_lags[slot * hydro_count + hydro]`.
    // Slot 0 = oldest completed quarter, slot n-1 = most recent.
    pub(crate) downstream_completed_lags: Vec<f64>,
    pub(crate) downstream_n_completed: usize,
    /// Scratch lookup table for basis reconstruction (ticket-002+).
    ///
    /// Maps each cut pool slot to its position in the stored
    /// `CapturedBasis::cut_row_slots`, so the reconstruction algorithm can
    /// locate the row status for any active cut in O(1) without allocation.
    /// Pre-filled with `None` to `initial_pool_capacity` entries so the
    /// first call can index up to that bound without resize.
    ///
    /// When `initial_pool_capacity == 0` (simulation-only workspaces), this
    /// vec starts empty; tickets 003/004 grow it in-place if needed.
    pub(crate) recon_slot_lookup: Vec<Option<u32>>,
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
    /// Pre-allocated scratch basis for backward-pass padding (P03).
    ///
    /// Used to copy-then-pad a read-only basis from `BasisStore` before
    /// passing it to `solve_with_basis`. Sized after construction via
    /// [`WorkspacePool::resize_scratch_bases`] to the maximum LP dimensions
    /// so that `Basis::clone_from` never reallocates on the hot path.
    pub(crate) scratch_basis: Basis,
    /// Pre-allocated accumulators for the backward pass trial-point loop.
    ///
    /// Survives across stages without reallocation. Buffers grow
    /// monotonically (never shrink) as larger stages are encountered.
    /// Simulation-only workspaces (constructed with `max_openings = 0`)
    /// start with empty buffers; the backward pass will never touch them.
    pub(crate) backward_accum: BackwardAccumulators,
}

impl<S: SolverInterface> SolverWorkspace<S> {
    /// Construct a workspace with the given solver, patch buffer, and state capacity.
    ///
    /// `sizing` provides the buffer-dimension parameters shared between the
    /// [`PatchBuffer`], the internal `ScratchBuffers`, and the
    /// `BackwardAccumulators` allocation. Pass `max_openings = 0`,
    /// `initial_pool_capacity = 0`, and `n_state = 0` in `sizing` for
    /// simulation-only workspaces that do not participate in the backward pass.
    ///
    /// The `scratch_basis` starts empty. Call `WorkspacePool::resize_scratch_bases`
    /// after construction to pre-allocate for backward-pass padding.
    #[must_use]
    pub fn new(solver: S, patch_buf: PatchBuffer, n_state: usize, sizing: WorkspaceSizing) -> Self {
        Self {
            solver,
            patch_buf,
            current_state: Vec::with_capacity(n_state),
            scratch: ScratchBuffers::new(sizing),
            scratch_basis: Basis::new(0, 0),
            backward_accum: BackwardAccumulators::new(
                sizing.max_openings,
                sizing.initial_pool_capacity,
                sizing.n_state,
            ),
        }
    }
}

impl ScratchBuffers {
    /// Allocate scratch buffers sized for the given per-worker parameters.
    ///
    /// Extracted from the three `SolverWorkspace` construction sites
    /// (`SolverWorkspace::new`, `WorkspacePool::new`, `WorkspacePool::try_new`)
    /// to keep them in sync (F1-008 fix).
    pub(crate) fn new(s: WorkspaceSizing) -> Self {
        let WorkspaceSizing {
            hydro_count,
            max_par_order,
            n_load_buses,
            max_blocks,
            downstream_par_order,
            initial_pool_capacity,
            // max_openings, n_state used by BackwardAccumulators only
            ..
        } = s;
        Self {
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
            lag_accumulator: vec![0.0_f64; hydro_count],
            lag_weight_accum: 0.0,
            downstream_accumulator: if downstream_par_order > 0 {
                vec![0.0_f64; hydro_count]
            } else {
                Vec::new()
            },
            downstream_weight_accum: 0.0,
            downstream_completed_lags: if downstream_par_order > 0 {
                vec![0.0_f64; hydro_count * downstream_par_order]
            } else {
                Vec::new()
            },
            downstream_n_completed: 0,
            recon_slot_lookup: vec![None; initial_pool_capacity],
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
    /// `sizing` provides all buffer-dimension parameters; pass
    /// `WorkspaceSizing { n_load_buses: 0, max_blocks: 0, .. }` when there is
    /// no stochastic load.
    #[must_use]
    pub fn new(
        n_threads: usize,
        n_state: usize,
        sizing: WorkspaceSizing,
        solver_factory: impl Fn() -> S,
    ) -> Self {
        let workspaces = (0..n_threads)
            .map(|_| SolverWorkspace {
                solver: solver_factory(),
                patch_buf: PatchBuffer::new(
                    sizing.hydro_count,
                    sizing.max_par_order,
                    sizing.n_load_buses,
                    sizing.max_blocks,
                ),
                current_state: Vec::with_capacity(n_state),
                scratch: ScratchBuffers::new(sizing),
                scratch_basis: Basis::new(0, 0),
                backward_accum: BackwardAccumulators::new(
                    sizing.max_openings,
                    sizing.initial_pool_capacity,
                    sizing.n_state,
                ),
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
    /// # Errors
    ///
    /// Returns `Err(E)` if any call to `solver_factory` fails.
    pub fn try_new<E>(
        n_threads: usize,
        n_state: usize,
        sizing: WorkspaceSizing,
        solver_factory: impl Fn() -> Result<S, E>,
    ) -> Result<Self, E> {
        let mut workspaces = Vec::with_capacity(n_threads);
        for _ in 0..n_threads {
            workspaces.push(SolverWorkspace {
                solver: solver_factory()?,
                patch_buf: PatchBuffer::new(
                    sizing.hydro_count,
                    sizing.max_par_order,
                    sizing.n_load_buses,
                    sizing.max_blocks,
                ),
                current_state: Vec::with_capacity(n_state),
                scratch: ScratchBuffers::new(sizing),
                scratch_basis: Basis::new(0, 0),
                backward_accum: BackwardAccumulators::new(
                    sizing.max_openings,
                    sizing.initial_pool_capacity,
                    sizing.n_state,
                ),
            });
        }
        Ok(Self { workspaces })
    }

    /// Pre-allocate each workspace's `scratch_basis` to the given LP dimensions.
    ///
    /// Call after construction when backward-pass basis padding is enabled.
    /// The allocation happens once during setup; `Basis::clone_from` on the
    /// hot path then reuses the existing capacity without reallocating.
    pub(crate) fn resize_scratch_bases(&mut self, max_cols: usize, max_rows: usize) {
        for ws in &mut self.workspaces {
            ws.scratch_basis = Basis::new(max_cols, max_rows);
        }
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
    bases: Vec<Option<CapturedBasis>>,
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
    pub fn get(&self, scenario: usize, stage: usize) -> Option<&CapturedBasis> {
        self.bases[scenario * self.num_stages + stage].as_ref()
    }

    /// Get a mutable reference to the basis slot at `[scenario][stage]`.
    pub fn get_mut(&mut self, scenario: usize, stage: usize) -> &mut Option<CapturedBasis> {
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
    bases: &'a mut [Option<CapturedBasis>],
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
    pub fn get(&self, scenario: usize, stage: usize) -> Option<&CapturedBasis> {
        let local = scenario - self.scenario_offset;
        self.bases[local * self.num_stages + stage].as_ref()
    }

    /// Get a mutable reference to the basis slot at absolute scenario index
    /// `scenario` and stage `stage`.
    ///
    /// # Panics
    ///
    /// Panics if `scenario < self.scenario_offset` (scenario not in this slice).
    pub fn get_mut(&mut self, scenario: usize, stage: usize) -> &mut Option<CapturedBasis> {
        let local = scenario - self.scenario_offset;
        &mut self.bases[local * self.num_stages + stage]
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BasisStore, CapturedBasis, ScratchBuffers, SolverWorkspace, WorkspacePool, WorkspaceSizing,
    };
    use cobre_solver::{
        Basis, SolutionView, SolverError, SolverInterface, SolverStatistics,
        types::{RowBatch, StageTemplate},
    };

    /// Minimal no-op solver for workspace tests.
    struct MockSolver;

    impl SolverInterface for MockSolver {
        fn solver_name_version(&self) -> String {
            "MockSolver 0.0.0".to_string()
        }
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

    fn sizing(
        hydro_count: usize,
        max_par_order: usize,
        downstream_par_order: usize,
    ) -> WorkspaceSizing {
        WorkspaceSizing {
            hydro_count,
            max_par_order,
            n_load_buses: 0,
            max_blocks: 0,
            downstream_par_order,
            ..WorkspaceSizing::default()
        }
    }

    #[test]
    fn test_workspace_pool_size() {
        let pool = WorkspacePool::new(4, 9, sizing(3, 2, 0), || MockSolver);
        assert_eq!(pool.workspaces.len(), 4);
    }

    #[test]
    fn test_workspace_buffer_dimensions() {
        // N=3, L=2 → patch_buf length = 3*(2+2) + 3 (z-inflow) = 15
        // n_state=9 → current_state capacity = 9
        let pool = WorkspacePool::new(4, 9, sizing(3, 2, 0), || MockSolver);
        for ws in &pool.workspaces {
            assert_eq!(ws.patch_buf.indices.len(), 15, "patch_buf length");
            assert_eq!(ws.current_state.capacity(), 9, "current_state capacity");
            assert_eq!(ws.current_state.len(), 0, "current_state starts empty");
        }
    }

    #[test]
    fn test_workspace_pool_zero_threads() {
        let pool = WorkspacePool::new(0, 9, sizing(3, 2, 0), || MockSolver);
        assert_eq!(pool.workspaces.len(), 0);
    }

    #[test]
    fn test_workspace_pool_single_thread() {
        let pool = WorkspacePool::new(1, 0, sizing(0, 0, 0), || MockSolver);
        assert_eq!(pool.workspaces.len(), 1);
        assert_eq!(pool.workspaces[0].patch_buf.indices.len(), 0);
    }

    #[test]
    fn test_workspace_pool_each_solver_independent() {
        // Factory is called n_threads times; each workspace gets its own instance.
        // Verify by checking pool size matches factory call expectation.
        let n = 6;
        let pool = WorkspacePool::new(n, 1, sizing(1, 0, 0), || MockSolver);
        assert_eq!(pool.workspaces.len(), n);
    }

    #[test]
    fn test_scratch_buffers_zero_downstream_par_order_empty_buffers() {
        // AC: downstream_par_order=0 → all downstream fields are zero/empty.
        let scratch = ScratchBuffers::new(WorkspaceSizing {
            hydro_count: 5,
            max_par_order: 2,
            n_load_buses: 0,
            max_blocks: 1,
            downstream_par_order: 0,
            ..WorkspaceSizing::default()
        });
        assert!(
            scratch.downstream_accumulator.is_empty(),
            "downstream_accumulator must be empty when downstream_par_order=0"
        );
        assert!(
            scratch.downstream_completed_lags.is_empty(),
            "downstream_completed_lags must be empty when downstream_par_order=0"
        );
        assert_eq!(
            scratch.downstream_weight_accum, 0.0,
            "downstream_weight_accum must be 0.0"
        );
        assert_eq!(
            scratch.downstream_n_completed, 0,
            "downstream_n_completed must be 0"
        );
    }

    #[test]
    fn test_scratch_buffers_nonzero_downstream_par_order_allocates_correctly() {
        // AC: downstream_par_order=2, hydro_count=3 → lengths 3 and 6, all 0.0.
        let scratch = ScratchBuffers::new(WorkspaceSizing {
            hydro_count: 3,
            max_par_order: 2,
            n_load_buses: 0,
            max_blocks: 1,
            downstream_par_order: 2,
            ..WorkspaceSizing::default()
        });
        assert_eq!(
            scratch.downstream_accumulator.len(),
            3,
            "downstream_accumulator.len() must equal hydro_count"
        );
        assert_eq!(
            scratch.downstream_completed_lags.len(),
            6,
            "downstream_completed_lags.len() must equal hydro_count * downstream_par_order"
        );
        assert!(
            scratch.downstream_accumulator.iter().all(|&v| v == 0.0),
            "downstream_accumulator must be initialized to 0.0"
        );
        assert!(
            scratch.downstream_completed_lags.iter().all(|&v| v == 0.0),
            "downstream_completed_lags must be initialized to 0.0"
        );
        assert_eq!(scratch.downstream_weight_accum, 0.0);
        assert_eq!(scratch.downstream_n_completed, 0);
    }

    #[test]
    fn test_workspace_pool_propagates_downstream_par_order() {
        // AC: WorkspacePool propagates downstream_par_order=2, hydro_count=3.
        let pool = WorkspacePool::new(
            2,
            6,
            WorkspaceSizing {
                hydro_count: 3,
                max_par_order: 2,
                n_load_buses: 0,
                max_blocks: 1,
                downstream_par_order: 2,
                ..WorkspaceSizing::default()
            },
            || MockSolver,
        );
        for ws in &pool.workspaces {
            assert_eq!(
                ws.scratch.downstream_accumulator.len(),
                3,
                "downstream_accumulator.len() per workspace"
            );
            assert_eq!(
                ws.scratch.downstream_completed_lags.len(),
                6,
                "downstream_completed_lags.len() per workspace"
            );
            assert_eq!(ws.scratch.downstream_weight_accum, 0.0);
            assert_eq!(ws.scratch.downstream_n_completed, 0);
        }
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
        // test shim: zero metadata is acceptable for tests exercising the length path
        *store.get_mut(1, 2) = Some(CapturedBasis::new(4, 2, 0, 0, 0));
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
        // test shim: zero metadata is acceptable for tests exercising the length path
        *slices[0].get_mut(0, 1) = Some(CapturedBasis::new(2, 1, 0, 0, 0));
        // Worker 1 writes to scenario 3 stage 2.
        // test shim: zero metadata is acceptable for tests exercising the length path
        *slices[1].get_mut(3, 2) = Some(CapturedBasis::new(2, 1, 0, 0, 0));

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
        // test shim: zero metadata is acceptable for tests exercising the length path
        *slices[0].get_mut(2, 1) = Some(CapturedBasis::new(1, 0, 0, 0, 0));
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
        // test shim: zero metadata is acceptable for tests exercising the length path
        *slices[1].get_mut(2, 0) = Some(CapturedBasis::new(1, 0, 0, 0, 0));
        // test shim: zero metadata is acceptable for tests exercising the length path
        *slices[1].get_mut(3, 1) = Some(CapturedBasis::new(1, 0, 0, 0, 0));
        drop(slices);

        assert!(store.get(2, 0).is_some());
        assert!(store.get(3, 1).is_some());
        assert!(store.get(0, 0).is_none());
        assert!(store.get(4, 0).is_none());
    }

    // ---------------------------------------------------------------------------
    // New unit tests for ticket-001
    // ---------------------------------------------------------------------------

    #[test]
    fn test_captured_basis_new_capacities() {
        // AC: CapturedBasis::new(4, 6, 3, 10, 2) must produce:
        //   basis.row_status.len() == 6, base_row_count == 3,
        //   cut_row_slots.capacity() >= 10, cut_row_slots.len() == 0,
        //   state_at_capture.capacity() >= 2, state_at_capture.len() == 0.
        let cb = CapturedBasis::new(4, 6, 3, 10, 2);
        assert_eq!(cb.basis.row_status.len(), 6, "row_status length");
        assert_eq!(cb.base_row_count, 3, "base_row_count");
        assert!(
            cb.cut_row_slots.capacity() >= 10,
            "cut_row_slots capacity must be >= 10 (got {})",
            cb.cut_row_slots.capacity()
        );
        assert_eq!(cb.cut_row_slots.len(), 0, "cut_row_slots starts empty");
        assert!(
            cb.state_at_capture.capacity() >= 2,
            "state_at_capture capacity must be >= 2 (got {})",
            cb.state_at_capture.capacity()
        );
        assert_eq!(
            cb.state_at_capture.len(),
            0,
            "state_at_capture starts empty"
        );
    }

    #[test]
    fn test_basis_store_holds_captured_basis() {
        // AC: BasisStore after migration holds Option<CapturedBasis>, not Option<Basis>.
        // slot set, slot read, default None holds for all 15 cells.
        let mut store = BasisStore::new(3, 5);
        // All 15 slots start as None.
        for s in 0..3 {
            for t in 0..5 {
                assert!(
                    store.get(s, t).is_none(),
                    "slot [{s}][{t}] must be None before any write"
                );
            }
        }
        // Write a CapturedBasis at [1][3]; read it back.
        *store.get_mut(1, 3) = Some(CapturedBasis::new(4, 6, 3, 10, 2));
        let retrieved = store.get(1, 3);
        assert!(retrieved.is_some(), "slot [1][3] must be Some after write");
        let cb = retrieved.expect("just checked is_some");
        assert_eq!(cb.base_row_count, 3);
        // All other slots remain None.
        for s in 0..3 {
            for t in 0..5 {
                if s == 1 && t == 3 {
                    continue;
                }
                assert!(
                    store.get(s, t).is_none(),
                    "slot [{s}][{t}] must remain None"
                );
            }
        }
    }

    #[test]
    fn test_recon_slot_lookup_presized() {
        // AC: every workspace in a freshly constructed WorkspacePool with
        //   WorkspaceSizing { initial_pool_capacity: 50, .. }
        //   must have recon_slot_lookup.len() == 50 and every entry is None.
        let pool = WorkspacePool::new(
            4,
            0,
            WorkspaceSizing {
                initial_pool_capacity: 50,
                ..WorkspaceSizing::default()
            },
            || MockSolver,
        );
        for (i, ws) in pool.workspaces.iter().enumerate() {
            assert_eq!(
                ws.scratch.recon_slot_lookup.len(),
                50,
                "workspace {i}: recon_slot_lookup.len() must equal initial_pool_capacity (50)"
            );
            assert!(
                ws.scratch.recon_slot_lookup.iter().all(Option::is_none),
                "workspace {i}: all recon_slot_lookup entries must be None"
            );
        }
        // Verify zero initial_pool_capacity produces an empty vec.
        let pool_empty = WorkspacePool::new(1, 0, WorkspaceSizing::default(), || MockSolver);
        assert_eq!(
            pool_empty.workspaces[0].scratch.recon_slot_lookup.len(),
            0,
            "initial_pool_capacity=0 must produce empty recon_slot_lookup"
        );
    }
}
