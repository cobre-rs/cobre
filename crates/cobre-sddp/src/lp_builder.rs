//! Stage LP patch buffer for SDDP forward and backward pass solves.
//!
//! [`PatchBuffer`] pre-allocates the three parallel arrays consumed by
//! [`SolverInterface::set_row_bounds`] and fills them with scenario-dependent
//! values before each LP solve.  Allocating once at training start and reusing
//! the same buffer across all iterations and stages is critical for hot-path
//! performance: the training loop calls `fill_forward_patches` or
//! `fill_state_patches` millions of times.
//!
//! ## Patch sequence (Training Loop SS4.2a)
//!
//! Each forward-pass solve requires exactly `N*(2+L)` row-bound patches:
//!
//! ```text
//! Category 1 — storage fixing    rows [0, N)
//!     patch row h = state[h]   for h in [0, N)
//!
//! Category 2 — AR lag fixing     rows [N, N*(1+L))
//!     patch row N + ℓ·N + h = state[N + ℓ·N + h]
//!     for h in [0, N), ℓ in [0, L)
//!
//! Category 3 — noise innovation   N rows in the static non-dual region
//!     patch ar_dynamics_row(base_row, h) = noise[h]   for h in [0, N)
//! ```
//!
//! The backward pass uses only categories 1 and 2 (`N*(1+L)` patches); noise
//! innovations are drawn from the fixed opening tree by the caller.
//!
//! ## Row index types
//!
//! All indices are stored as `usize` to match the `set_row_bounds` interface
//! signature directly; no casting is required at the call site.
//!
//! ## Worked example (SS4.2a): N = 3, L = 2
//!
//! ```text
//! Patch  Category       Row index              Value
//!     0  storage-fix    0                      state[0]  (H0)
//!     1  storage-fix    1                      state[1]  (H1)
//!     2  storage-fix    2                      state[2]  (H2)
//!     3  lag-fix        N + 0·N + 0 = 3        state[3]  (H0 lag 0)
//!     4  lag-fix        N + 0·N + 1 = 4        state[4]  (H1 lag 0)
//!     5  lag-fix        N + 0·N + 2 = 5        state[5]  (H2 lag 0)
//!     6  lag-fix        N + 1·N + 0 = 6        state[6]  (H0 lag 1)
//!     7  lag-fix        N + 1·N + 1 = 7        state[7]  (H1 lag 1)
//!     8  lag-fix        N + 1·N + 2 = 8        state[8]  (H2 lag 1)
//!     9  noise-fix      ar_dynamics_row(br, 0) noise[0]  (H0)
//!    10  noise-fix      ar_dynamics_row(br, 1) noise[1]  (H1)
//!    11  noise-fix      ar_dynamics_row(br, 2) noise[2]  (H2)
//! ```
//!
//! Total: 12 = 3*(2+2) patches.

use crate::indexer::StageIndexer;

/// Pre-allocated row-bound patch arrays for one SDDP stage LP solve.
///
/// Holds three parallel `Vec`s of equal length ready for a single
/// [`SolverInterface::set_row_bounds`] call.  The buffer is sized for
/// `N*(2+L)` patches at construction and reused across all iterations.
///
/// # Memory layout
///
/// The `N*(2+L)` entries are written in category order:
///
/// | Entry range           | Category                           |
/// | --------------------- | ---------------------------------- |
/// | `[0, N)`              | Storage-fixing (Category 1)        |
/// | `[N, N*(1+L))`        | AR lag-fixing (Category 2)         |
/// | `[N*(1+L), N*(2+L))`  | AR dynamics / noise (Category 3)   |
///
/// [`fill_state_patches`](PatchBuffer::fill_state_patches) writes only
/// `[0, N*(1+L))` (Categories 1 and 2).  The Category 3 slice is left as
/// the previous iteration's values, which is safe because the caller passes
/// only `&self.indices[..active_len]` to `set_row_bounds`.
#[derive(Debug, Clone)]
pub struct PatchBuffer {
    /// Row indices to patch.
    ///
    /// Length `N*(2+L)`.  Entries are `usize` to match the
    /// `set_row_bounds(&[usize], ...)` interface directly.
    pub indices: Vec<usize>,

    /// New lower bounds for each patched row.
    ///
    /// Length `N*(2+L)`.  For equality constraints, `lower[i] == upper[i]`.
    pub lower: Vec<f64>,

    /// New upper bounds for each patched row.
    ///
    /// Length `N*(2+L)`.  For equality constraints, `upper[i] == lower[i]`.
    pub upper: Vec<f64>,

    /// Number of operating hydro plants (N).
    hydro_count: usize,

    /// Maximum PAR order across all operating hydros (L).
    max_par_order: usize,
}

impl PatchBuffer {
    /// Construct a [`PatchBuffer`] pre-allocated for `N*(2+L)` patches.
    ///
    /// `hydro_count` is the number of operating hydro plants (N).
    /// `max_par_order` is the maximum PAR order across all operating hydros (L).
    ///
    /// The buffer's `indices`, `lower`, and `upper` vectors are sized to
    /// `N*(2+L)` and zero-initialised.  Call [`fill_forward_patches`] or
    /// [`fill_state_patches`] to populate them before each LP solve.
    ///
    /// # Examples
    ///
    /// ```
    /// use cobre_sddp::PatchBuffer;
    ///
    /// // 3-hydro AR(2) system from spec SS4.2a worked example
    /// let buf = PatchBuffer::new(3, 2);
    /// assert_eq!(buf.indices.len(), 12); // 3*(2+2)
    ///
    /// // Production scale: N = 160, L = 12
    /// let big = PatchBuffer::new(160, 12);
    /// assert_eq!(big.indices.len(), 2240); // 160*(2+12)
    ///
    /// // Edge case: no lags (L = 0) — only storage + noise patches
    /// let no_lag = PatchBuffer::new(5, 0);
    /// assert_eq!(no_lag.indices.len(), 10); // 5*(2+0)
    /// ```
    ///
    /// [`fill_forward_patches`]: PatchBuffer::fill_forward_patches
    /// [`fill_state_patches`]: PatchBuffer::fill_state_patches
    #[must_use]
    pub fn new(hydro_count: usize, max_par_order: usize) -> Self {
        let capacity = hydro_count * (2 + max_par_order);
        Self {
            indices: vec![0; capacity],
            lower: vec![0.0; capacity],
            upper: vec![0.0; capacity],
            hydro_count,
            max_par_order,
        }
    }

    /// Fill all `N*(2+L)` patches for a forward-pass solve.
    ///
    /// Populates Categories 1, 2, and 3 in sequence:
    ///
    /// - **Category 1** — `N` storage-fixing patches: row `h` ← `state[h]`
    ///   for `h ∈ [0, N)`.
    /// - **Category 2** — `N*L` AR lag-fixing patches: row `N + ℓ·N + h` ←
    ///   `state[N + ℓ·N + h]` for `h ∈ [0, N)`, `ℓ ∈ [0, L)`.
    /// - **Category 3** — `N` noise-fixing patches: row
    ///   `ar_dynamics_row_offset(base_row, h)` ← `noise[h]` for `h ∈ [0, N)`.
    ///
    /// All patches are equality constraints: `lower[i] == upper[i] == value`.
    ///
    /// After this call, pass `&buf.indices`, `&buf.lower`, `&buf.upper` to
    /// [`SolverInterface::set_row_bounds`].
    ///
    /// # Arguments
    ///
    /// - `indexer` — LP layout map for this stage (provides `hydro_count`,
    ///   `max_par_order`, and fixing-constraint row ranges).
    /// - `state` — incoming state vector of length `n_state = N*(1+L)`.
    ///   Prefix `[0, N)` is storage; `[N, N*(1+L))` is AR lags.
    /// - `noise` — stochastic noise innovations of length `N`, one per hydro.
    /// - `base_row` — first row index of the AR dynamics constraints in the
    ///   static non-dual region of the LP ([Solver Abstraction SS2.2]).
    ///   Computed during stage template construction and stored alongside
    ///   `indexer`.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `state.len() != indexer.n_state` or
    /// `noise.len() != indexer.hydro_count`.
    pub fn fill_forward_patches(
        &mut self,
        indexer: &StageIndexer,
        state: &[f64],
        noise: &[f64],
        base_row: usize,
    ) {
        debug_assert_eq!(
            state.len(),
            indexer.n_state,
            "state slice length {got} != n_state {expected}",
            got = state.len(),
            expected = indexer.n_state,
        );
        debug_assert_eq!(
            noise.len(),
            indexer.hydro_count,
            "noise slice length {got} != hydro_count {expected}",
            got = noise.len(),
            expected = indexer.hydro_count,
        );

        let n = self.hydro_count;
        let l = self.max_par_order;

        // Category 1: storage-fixing rows [0, N)
        // patch(row = h, value = state[h])
        for (h, &sv) in state[..n].iter().enumerate() {
            self.indices[h] = h;
            self.lower[h] = sv;
            self.upper[h] = sv;
        }

        // Category 2: AR lag-fixing rows [N, N*(1+L))
        // patch(row = N + ℓ·N + h, value = state[N + ℓ·N + h])
        for lag in 0..l {
            for h in 0..n {
                let slot = n + lag * n + h;
                self.indices[slot] = slot;
                self.lower[slot] = state[slot];
                self.upper[slot] = state[slot];
            }
        }

        // Category 3: AR dynamics rows in the static non-dual region
        // patch(row = ar_dynamics_row_offset(base_row, h), value = noise[h])
        let cat3_start = n * (1 + l);
        for (h, &nv) in noise.iter().enumerate() {
            let slot = cat3_start + h;
            self.indices[slot] = ar_dynamics_row_offset(base_row, h);
            self.lower[slot] = nv;
            self.upper[slot] = nv;
        }
    }

    /// Fill `N*(1+L)` patches for a backward-pass (state-only) solve.
    ///
    /// Populates Categories 1 and 2 only.  Category 3 (noise innovations)
    /// is omitted because noise values for the backward pass come from the
    /// fixed opening tree and are applied separately by the caller.
    ///
    /// - **Category 1** — `N` storage-fixing patches: row `h` ← `state[h]`
    ///   for `h ∈ [0, N)`.
    /// - **Category 2** — `N*L` AR lag-fixing patches: row `N + ℓ·N + h` ←
    ///   `state[N + ℓ·N + h]` for `h ∈ [0, N)`, `ℓ ∈ [0, L)`.
    ///
    /// Pass `&buf.indices[..active_len()]`, `&buf.lower[..active_len()]`, and
    /// `&buf.upper[..active_len()]` to [`SolverInterface::set_row_bounds`],
    /// where `active_len` is `N*(1+L)`.  Use [`state_patch_count`] to obtain
    /// this length.
    ///
    /// # Arguments
    ///
    /// - `indexer` — LP layout map for this stage.
    /// - `state` — incoming state vector of length `n_state = N*(1+L)`.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `state.len() != indexer.n_state`.
    ///
    /// [`state_patch_count`]: PatchBuffer::state_patch_count
    pub fn fill_state_patches(&mut self, indexer: &StageIndexer, state: &[f64]) {
        debug_assert_eq!(
            state.len(),
            indexer.n_state,
            "state slice length {got} != n_state {expected}",
            got = state.len(),
            expected = indexer.n_state,
        );

        let n = self.hydro_count;
        let l = self.max_par_order;

        // Category 1: storage-fixing rows [0, N)
        for (h, &sv) in state[..n].iter().enumerate() {
            self.indices[h] = h;
            self.lower[h] = sv;
            self.upper[h] = sv;
        }

        // Category 2: AR lag-fixing rows [N, N*(1+L))
        for lag in 0..l {
            for h in 0..n {
                let slot = n + lag * n + h;
                self.indices[slot] = slot;
                self.lower[slot] = state[slot];
                self.upper[slot] = state[slot];
            }
        }
        // Category 3 is intentionally not written; the caller slices
        // [0..state_patch_count()] before passing to set_row_bounds.
    }

    /// Number of active patches after [`fill_forward_patches`]: `N*(2+L)`.
    ///
    /// Use this to pass the full buffer to `set_row_bounds`.
    ///
    /// [`fill_forward_patches`]: PatchBuffer::fill_forward_patches
    #[must_use]
    #[inline]
    pub fn forward_patch_count(&self) -> usize {
        self.hydro_count * (2 + self.max_par_order)
    }

    /// Number of active patches after [`fill_state_patches`]: `N*(1+L)`.
    ///
    /// Slice the buffer to this length before passing to `set_row_bounds`
    /// when using the state-only (backward-pass) fill.
    ///
    /// [`fill_state_patches`]: PatchBuffer::fill_state_patches
    #[must_use]
    #[inline]
    pub fn state_patch_count(&self) -> usize {
        self.hydro_count * (1 + self.max_par_order)
    }
}

/// Compute the row index of hydro `h`'s AR dynamics constraint.
///
/// AR dynamics constraints live in the static non-dual region of the LP row
/// layout ([Solver Abstraction SS2.2]).  Their exact position depends on the
/// system's bus and block counts, which are not known to this crate.  The
/// `base_row` parameter is the first row index of the AR dynamics block,
/// computed during stage template construction and stored alongside the
/// [`StageIndexer`].
///
/// The formula is:
///
/// ```text
/// ar_dynamics_row(base_row, h) = base_row + h
/// ```
///
/// Hydros are laid out sequentially in the AR dynamics block, matching the
/// hydro-major order used throughout the column and row layout
/// ([Solver Abstraction SS2.1–SS2.2]).
///
/// # Pitfall
///
/// This row is in the **static non-dual region**, not in the fixing constraint
/// region `[0, n_state)`.  Do not confuse with the lag-fixing row
/// `N + ℓ·N + h` from Category 2.
#[must_use]
#[inline]
pub fn ar_dynamics_row_offset(base_row: usize, hydro_index: usize) -> usize {
    base_row + hydro_index
}

#[cfg(test)]
mod tests {
    use super::{ar_dynamics_row_offset, PatchBuffer};
    use crate::indexer::StageIndexer;

    /// Convenience: make an indexer without repeating N/L everywhere.
    fn idx(n: usize, l: usize) -> StageIndexer {
        StageIndexer::new(n, l)
    }

    #[test]
    fn new_3_2_sizes_to_12() {
        // Spec acceptance criterion: PatchBuffer::new(3, 2) → 12 = 3*(2+2)
        let buf = PatchBuffer::new(3, 2);
        assert_eq!(buf.indices.len(), 12);
        assert_eq!(buf.lower.len(), 12);
        assert_eq!(buf.upper.len(), 12);
    }

    #[test]
    fn new_160_12_sizes_to_2240() {
        // Spec acceptance criterion: PatchBuffer::new(160, 12) → 2240 = 160*(2+12)
        let buf = PatchBuffer::new(160, 12);
        assert_eq!(buf.indices.len(), 2240);
        assert_eq!(buf.lower.len(), 2240);
        assert_eq!(buf.upper.len(), 2240);
    }

    #[test]
    fn new_zero_lags_sizes_to_2n() {
        // Edge case: L = 0 → N*(2+0) = 2*N patches
        let buf = PatchBuffer::new(5, 0);
        assert_eq!(buf.indices.len(), 10); // 5*(2+0)
    }

    #[test]
    fn new_zero_hydros_sizes_to_zero() {
        let buf = PatchBuffer::new(0, 0);
        assert_eq!(buf.indices.len(), 0);
    }

    #[test]
    fn forward_patch_count_matches_buffer_len() {
        let buf = PatchBuffer::new(3, 2);
        assert_eq!(buf.forward_patch_count(), buf.indices.len());
        assert_eq!(buf.forward_patch_count(), 12);
    }

    #[test]
    fn state_patch_count_is_n_times_one_plus_l() {
        // N*(1+L) = 3*(1+2) = 9 for the spec worked example
        let buf = PatchBuffer::new(3, 2);
        assert_eq!(buf.state_patch_count(), 9);
    }

    #[test]
    fn state_patch_count_zero_lags() {
        // L = 0 → N*(1+0) = N = 4
        let buf = PatchBuffer::new(4, 0);
        assert_eq!(buf.state_patch_count(), 4);
    }

    #[test]
    fn ar_dynamics_row_offset_adds_base_plus_hydro() {
        assert_eq!(ar_dynamics_row_offset(100, 0), 100);
        assert_eq!(ar_dynamics_row_offset(100, 1), 101);
        assert_eq!(ar_dynamics_row_offset(100, 2), 102);
    }

    #[test]
    fn ar_dynamics_row_offset_zero_base() {
        assert_eq!(ar_dynamics_row_offset(0, 7), 7);
    }

    #[test]
    fn fill_forward_patches_category1_indices() {
        // Spec AC: first 3 patches correspond to storage fixing rows 0, 1, 2
        let mut buf = PatchBuffer::new(3, 2);
        let state = [10.0, 20.0, 30.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let noise = [0.1, 0.2, 0.3];
        buf.fill_forward_patches(&idx(3, 2), &state, &noise, 50);

        assert_eq!(buf.indices[0], 0);
        assert_eq!(buf.indices[1], 1);
        assert_eq!(buf.indices[2], 2);
    }

    #[test]
    fn fill_forward_patches_category2_indices() {
        // Spec AC: patches 3-8 correspond to AR lag fixing rows 3..=8
        // Row index formula: N + ℓ·N + h
        // ℓ=0: 3+0=3, 3+1=4, 3+2=5
        // ℓ=1: 6+0=6, 6+1=7, 6+2=8
        let mut buf = PatchBuffer::new(3, 2);
        let state = [10.0, 20.0, 30.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let noise = [0.1, 0.2, 0.3];
        buf.fill_forward_patches(&idx(3, 2), &state, &noise, 50);

        assert_eq!(buf.indices[3], 3); // N + 0·N + 0
        assert_eq!(buf.indices[4], 4); // N + 0·N + 1
        assert_eq!(buf.indices[5], 5); // N + 0·N + 2
        assert_eq!(buf.indices[6], 6); // N + 1·N + 0
        assert_eq!(buf.indices[7], 7); // N + 1·N + 1
        assert_eq!(buf.indices[8], 8); // N + 1·N + 2
    }

    #[test]
    fn fill_forward_patches_category3_indices() {
        // Spec AC: last 3 patches correspond to AR dynamics rows
        // base_row = 50 → rows 50, 51, 52
        let mut buf = PatchBuffer::new(3, 2);
        let state = [10.0, 20.0, 30.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let noise = [0.1, 0.2, 0.3];
        buf.fill_forward_patches(&idx(3, 2), &state, &noise, 50);

        assert_eq!(buf.indices[9], 50); // ar_dynamics_row_offset(50, 0)
        assert_eq!(buf.indices[10], 51); // ar_dynamics_row_offset(50, 1)
        assert_eq!(buf.indices[11], 52); // ar_dynamics_row_offset(50, 2)
    }

    #[test]
    fn fill_forward_patches_category1_values() {
        // Category 1: lower == upper == state[h]
        let mut buf = PatchBuffer::new(3, 2);
        let state = [10.0, 20.0, 30.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let noise = [0.1, 0.2, 0.3];
        buf.fill_forward_patches(&idx(3, 2), &state, &noise, 50);

        assert_eq!(buf.lower[0], 10.0);
        assert_eq!(buf.upper[0], 10.0);
        assert_eq!(buf.lower[1], 20.0);
        assert_eq!(buf.upper[1], 20.0);
        assert_eq!(buf.lower[2], 30.0);
        assert_eq!(buf.upper[2], 30.0);
    }

    #[test]
    fn fill_forward_patches_category2_values() {
        // Category 2: lower == upper == state[N + ℓ·N + h]
        let mut buf = PatchBuffer::new(3, 2);
        let state = [10.0, 20.0, 30.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let noise = [0.1, 0.2, 0.3];
        buf.fill_forward_patches(&idx(3, 2), &state, &noise, 50);

        // ℓ=0: state[3]=1, state[4]=2, state[5]=3
        assert_eq!(buf.lower[3], 1.0);
        assert_eq!(buf.upper[3], 1.0);
        assert_eq!(buf.lower[4], 2.0);
        assert_eq!(buf.upper[4], 2.0);
        assert_eq!(buf.lower[5], 3.0);
        assert_eq!(buf.upper[5], 3.0);
        // ℓ=1: state[6]=4, state[7]=5, state[8]=6
        assert_eq!(buf.lower[6], 4.0);
        assert_eq!(buf.upper[6], 4.0);
        assert_eq!(buf.lower[7], 5.0);
        assert_eq!(buf.upper[7], 5.0);
        assert_eq!(buf.lower[8], 6.0);
        assert_eq!(buf.upper[8], 6.0);
    }

    #[test]
    fn fill_forward_patches_category3_values() {
        // Category 3: lower == upper == noise[h]
        let mut buf = PatchBuffer::new(3, 2);
        let state = [10.0, 20.0, 30.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let noise = [0.1, 0.2, 0.3];
        buf.fill_forward_patches(&idx(3, 2), &state, &noise, 50);

        assert_eq!(buf.lower[9], 0.1);
        assert_eq!(buf.upper[9], 0.1);
        assert_eq!(buf.lower[10], 0.2);
        assert_eq!(buf.upper[10], 0.2);
        assert_eq!(buf.lower[11], 0.3);
        assert_eq!(buf.upper[11], 0.3);
    }

    #[test]
    fn fill_forward_patches_all_equality_constraints() {
        // Every patch must satisfy lower == upper (equality constraint)
        let mut buf = PatchBuffer::new(3, 2);
        let state = [10.0, 20.0, 30.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let noise = [0.1, 0.2, 0.3];
        buf.fill_forward_patches(&idx(3, 2), &state, &noise, 50);

        for i in 0..buf.forward_patch_count() {
            assert_eq!(
                buf.lower[i],
                buf.upper[i],
                "patch {i}: lower {lo} != upper {up}",
                lo = buf.lower[i],
                up = buf.upper[i],
            );
        }
    }

    #[test]
    fn fill_state_patches_count_is_n_times_one_plus_l() {
        // Backward pass: N*(1+L) patches, no noise
        let mut buf = PatchBuffer::new(3, 2);
        let state = [10.0, 20.0, 30.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        buf.fill_state_patches(&idx(3, 2), &state);

        // Active slice is [0, 9) = [0, N*(1+L))
        assert_eq!(buf.state_patch_count(), 9);
    }

    #[test]
    fn fill_state_patches_category1_correct() {
        let mut buf = PatchBuffer::new(3, 2);
        let state = [10.0, 20.0, 30.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        buf.fill_state_patches(&idx(3, 2), &state);

        assert_eq!(buf.indices[0], 0);
        assert_eq!(buf.lower[0], 10.0);
        assert_eq!(buf.upper[0], 10.0);
        assert_eq!(buf.indices[1], 1);
        assert_eq!(buf.lower[1], 20.0);
        assert_eq!(buf.upper[1], 20.0);
        assert_eq!(buf.indices[2], 2);
        assert_eq!(buf.lower[2], 30.0);
        assert_eq!(buf.upper[2], 30.0);
    }

    #[test]
    fn fill_state_patches_category2_correct() {
        let mut buf = PatchBuffer::new(3, 2);
        let state = [10.0, 20.0, 30.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        buf.fill_state_patches(&idx(3, 2), &state);

        // Same index/value expectations as fill_forward_patches for cat 2
        assert_eq!(buf.indices[3], 3);
        assert_eq!(buf.lower[3], 1.0);
        assert_eq!(buf.upper[3], 1.0);
        assert_eq!(buf.indices[8], 8);
        assert_eq!(buf.lower[8], 6.0);
        assert_eq!(buf.upper[8], 6.0);
    }

    #[test]
    fn fill_state_patches_equality_constraints_in_active_range() {
        let mut buf = PatchBuffer::new(3, 2);
        let state = [10.0, 20.0, 30.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        buf.fill_state_patches(&idx(3, 2), &state);

        let active = buf.state_patch_count();
        for i in 0..active {
            assert_eq!(
                buf.lower[i],
                buf.upper[i],
                "state patch {i}: lower {lo} != upper {up}",
                lo = buf.lower[i],
                up = buf.upper[i],
            );
        }
    }

    #[test]
    fn forward_patches_zero_lags_only_storage_and_noise() {
        // L=0: N*(2+0) = 2*N patches: N storage + N noise, zero lag
        let n = 2;
        let mut buf = PatchBuffer::new(n, 0);
        let state = [5.0, 7.0]; // n_state = 2*(1+0) = 2
        let noise = [0.5, 0.6];
        buf.fill_forward_patches(&idx(n, 0), &state, &noise, 10);

        // 4 patches total
        assert_eq!(buf.forward_patch_count(), 4);

        // Category 1: rows 0, 1
        assert_eq!(buf.indices[0], 0);
        assert_eq!(buf.lower[0], 5.0);
        assert_eq!(buf.indices[1], 1);
        assert_eq!(buf.lower[1], 7.0);

        // Category 2: empty (L=0, zero iterations)

        // Category 3 starts at slot N*(1+0) = N = 2
        assert_eq!(buf.indices[2], 10); // ar_dynamics_row_offset(10, 0)
        assert_eq!(buf.lower[2], 0.5);
        assert_eq!(buf.indices[3], 11); // ar_dynamics_row_offset(10, 1)
        assert_eq!(buf.lower[3], 0.6);
    }

    #[test]
    fn state_patches_zero_lags_only_storage() {
        // L=0: N*(1+0) = N patches (storage only)
        let n = 3;
        let mut buf = PatchBuffer::new(n, 0);
        let state = [1.0, 2.0, 3.0]; // n_state = 3
        buf.fill_state_patches(&idx(n, 0), &state);

        assert_eq!(buf.state_patch_count(), 3);
        assert_eq!(buf.indices[0], 0);
        assert_eq!(buf.lower[0], 1.0);
        assert_eq!(buf.upper[0], 1.0);
        assert_eq!(buf.indices[1], 1);
        assert_eq!(buf.lower[1], 2.0);
        assert_eq!(buf.indices[2], 2);
        assert_eq!(buf.lower[2], 3.0);
    }

    #[test]
    fn production_scale_forward_patch_count() {
        // Spec acceptance criterion: 160*(2+12) = 2240
        let buf = PatchBuffer::new(160, 12);
        assert_eq!(buf.forward_patch_count(), 2240);
        assert_eq!(buf.indices.len(), 2240);
    }

    #[test]
    #[allow(clippy::cast_precision_loss)] // fixture: values are small integers, no precision lost
    fn production_scale_fill_forward_patches_smoke() {
        let n = 160;
        let l = 12;
        let mut buf = PatchBuffer::new(n, l);
        let n_state = n * (1 + l);
        let state: Vec<f64> = (0..n_state).map(|i| i as f64).collect();
        let noise: Vec<f64> = (0..n).map(|h| h as f64 * 0.01).collect();
        buf.fill_forward_patches(&StageIndexer::new(n, l), &state, &noise, 500);

        // Spot-check category 1
        assert_eq!(buf.indices[0], 0);
        assert_eq!(buf.lower[0], 0.0);

        // Spot-check category 3 start at slot N*(1+L) = 160*13 = 2080
        assert_eq!(buf.indices[2080], 500); // ar_dynamics_row_offset(500, 0)
        assert_eq!(buf.lower[2080], 0.0); // noise[0]
        assert_eq!(buf.indices[2239], 659); // ar_dynamics_row_offset(500, 159)
        assert_eq!(buf.lower[2239], 159.0 * 0.01);

        // All patches must be equality constraints
        for i in 0..buf.forward_patch_count() {
            assert_eq!(buf.lower[i], buf.upper[i], "patch {i} not equality");
        }
    }

    #[test]
    fn clone_and_debug() {
        let buf = PatchBuffer::new(3, 2);
        let cloned = buf.clone();
        assert_eq!(cloned.indices.len(), buf.indices.len());

        let s = format!("{buf:?}");
        assert!(s.contains("PatchBuffer"));
    }
}
