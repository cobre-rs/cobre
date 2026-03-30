//! Stage LP patch buffer and stage template builder for SDDP subproblems.
//!
//! This module provides two public facilities:
//!
//! - [`PatchBuffer`]: pre-allocates the parallel arrays consumed by
//!   `SolverInterface::set_row_bounds` and fills them with scenario-dependent
//!   values before each LP solve.  Allocating once at training start and
//!   reusing the same buffer across all iterations and stages is critical for
//!   hot-path performance: the training loop calls `fill_forward_patches`,
//!   `fill_load_patches`, or `fill_state_patches` millions of times.
//!
//! - [`build_stage_templates`]: constructs a `Vec<StageTemplate>` from a
//!   loaded `System` — one template per study stage.  The template encodes
//!   the full structural LP in CSC format, column/row bounds, and objective
//!   coefficients.  It is built once at startup and shared read-only across
//!   all threads.
//!
//! ## LP layout (Solver Abstraction SS2)
//!
//! ### Column layout (contiguous regions)
//!
//! ```text
//! [0,  N)              outgoing storage      (N = n_hydros)
//! [N,  N*(1+L))        AR lag variables      (N*L lags, hydro-major)
//! [N*(1+L), N*(2+L))   z_inflow              (realized inflow, free, not state)
//! [N*(2+L), N*(3+L))   incoming storage      (fixed by storage-fixing rows)
//! N*(3+L)              theta                 (future cost, scalar)
//! N*(3+L)+1 ..         decision variables:
//!   hydro turbine:   N*K columns
//!   hydro spillage:  N*K columns
//!   thermal gen:     T*K columns
//!   line fwd flow:   Lines*K columns
//!   line rev flow:   Lines*K columns
//!   bus deficit:     B*S*K columns  (S = max_deficit_segments across all buses)
//!   bus excess:      B*K columns
//!   inflow slack:    N columns (sigma_inf_h, only when penalty method is active)
//!   FPHA generation: N_fpha*K columns (one per FPHA hydro per block)
//!   evaporation:     N_evap*3 columns (Q_ev, f_evap_plus, f_evap_minus per evap hydro)
//!   withdrawal slack: N columns (sigma^r_h, one per hydro when hydro_count > 0)
//!   NCS generation:  n_active_ncs*K columns  (VARIES per stage)
//!   generic slack:   n_generic_slack columns  (VARIES per stage)
//! ```
//!
//! ### Row layout (contiguous regions)
//!
//! ```text
//! [0,  N)              storage-fixing constraints
//! [N,  N*(1+L))        AR lag-fixing constraints
//! [N*(1+L), N*(2+L))   z_inflow definition constraints (structural, equality)
//! N*(2+L) ..           structural constraints (non-dual region):
//!   water balance:   N rows   (one per hydro)
//!   load balance:    B*K rows (one per bus per block)
//!   FPHA:            n_fpha_rows
//!   evaporation:     n_evap rows
//!   generic:         n_generic_rows  (VARIES per stage)
//! ```
//!
//! The AR dynamics (noise patch target) rows are the water balance constraints
//! beginning at row `base_rows[stage]` (= `N*(2+L)` for stages without extra
//! variable-offset rows). The `base_rows` value returned alongside the templates
//! encodes this offset for each stage so that [`PatchBuffer`] can update the
//! correct RHS during forward-pass solves.
//!
//! ## Patch sequence (Training Loop SS4.2a)
//!
//! Each forward-pass solve requires up to `N*(2+L) + N + M*K` row-bound patches
//! (where M is the number of stochastic load buses and K is the block count):
//!
//! ```text
//! Category 1 -- storage fixing    rows [0, N)
//!     patch row h = state[h]   for h in [0, N)
//!
//! Category 2 -- AR lag fixing     rows [N, N*(1+L))
//!     patch row N + l*N + h = state[N + l*N + h]
//!     for h in [0, N), l in [0, L)
//!
//! Category 3 -- noise innovation   N rows at base_rows[stage] (= N*(2+L))
//!     patch water_balance_row(base_row, h) = noise[h]   for h in [0, N)
//!
//! Category 4 -- load balance patches   M*K rows (optional, stochastic load)
//!     patch load_row_start + bus_pos[i]*K + blk = load_rhs[i*K + blk]
//!     for i in [0, M), blk in [0, K)
//!
//! Category 5 -- z-inflow RHS      N rows at N*(1+L)
//!     patch z_inflow_row(h) = base_h + noise_scale_h * eta_h   for h in [0, N)
//! ```
//!
//! The backward pass uses only categories 1 and 2 (`N*(1+L)` patches); noise
//! innovations are drawn from the fixed opening tree by the caller.
//! When `n_load_buses == 0`, Category 4 is empty and has no effect.
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
//!     3  lag-fix        N + 0*N + 0 = 3        state[3]  (H0 lag 0)
//!     4  lag-fix        N + 0*N + 1 = 4        state[4]  (H1 lag 0)
//!     5  lag-fix        N + 0*N + 2 = 5        state[5]  (H2 lag 0)
//!     6  lag-fix        N + 1*N + 0 = 6        state[6]  (H0 lag 1)
//!     7  lag-fix        N + 1*N + 1 = 7        state[7]  (H1 lag 1)
//!     8  lag-fix        N + 1*N + 2 = 8        state[8]  (H2 lag 1)
//!     9  noise-fix      N*(2+L) + 0 = 12       noise[0]  (H0)
//!    10  noise-fix      N*(2+L) + 1 = 13       noise[1]  (H1)
//!    11  noise-fix      N*(2+L) + 2 = 14       noise[2]  (H2)
//!    12  z-inflow       N*(1+L) + 0 = 9        z_rhs[0]  (H0)
//!    13  z-inflow       N*(1+L) + 1 = 10       z_rhs[1]  (H1)
//!    14  z-inflow       N*(1+L) + 2 = 11       z_rhs[2]  (H2)
//! ```
//!
//! Total: 15 = 3*(2+2) + 3 patches.

use cobre_core::ConstraintSense;

mod layout;
mod matrix;
mod patch;
mod scaling;
mod template;

// --- Public re-exports (stable API) ---
pub use patch::{PatchBuffer, ar_dynamics_row_offset};
pub use template::{StageTemplates, build_stage_templates};

// --- Crate-internal re-exports ---
pub(crate) use scaling::{apply_col_scale, apply_row_scale, compute_col_scale, compute_row_scale};

// ---------------------------------------------------------------------------
// Shared constants
// ---------------------------------------------------------------------------

/// Per-hour conversion factor from m³/s to hm³.
///
/// `M3S_TO_HM3 = seconds_per_hour / m³_per_hm³ = 3600 / 1_000_000 = 0.0036`
///
/// Callers multiply by `Block::duration_hours` to get the full block
/// conversion: `volume_hm3 = flow_m3s * M3S_TO_HM3 * duration_hours`.
/// For a 30-day month (720 h): `0.0036 * 720 = 2.592`.
pub(crate) const M3S_TO_HM3: f64 = 3_600.0 / 1_000_000.0; // multiply by hours to get hm³

/// Divisor applied to all objective-function cost coefficients.
///
/// Dividing monetary costs by this factor reduces objective magnitudes
/// (e.g., from ~8.8e11 to ~8.8e8), improving LP solver numerical
/// conditioning without changing the optimization argmin. All cost-domain
/// outputs (objective values, duals, cost breakdowns) are multiplied by
/// this factor at the reporting boundary to recover original units.
pub(crate) const COST_SCALE_FACTOR: f64 = 1_000.0;

/// Safety margin applied to the physical upper bound on the evaporation flow
/// variable `Q_ev`.  The bound is `(k_evap0 + k_evap_v * v_max).max(0) * margin`.
/// A 2x margin accounts for linearization approximation error (the actual
/// area-volume curve may exceed the linear estimate near `v_max`).
pub(crate) const Q_EV_SAFETY_MARGIN: f64 = 2.0;

/// Historical multiplier for over-evaporation penalty (100x).
/// No longer applied in the LP builder — the 100x is now embedded in the
/// default `evaporation_violation_pos_cost` during penalties.json parsing.
/// Retained for reference documentation only.
#[allow(dead_code)]
pub(crate) const OVER_EVAPORATION_COST_MULTIPLIER: f64 = 100.0;

// ---------------------------------------------------------------------------
// Shared types
// ---------------------------------------------------------------------------

/// Per-row metadata for one active generic constraint row at a single stage.
///
/// Stores the information needed by the LP builder to fill CSC matrix entries,
/// row bounds, and objective coefficients for generic constraint rows and their
/// associated slack columns.  Also used by the simulation extraction pipeline
/// to map LP row/column indices back to constraint identity and block.
///
/// One entry is created per active `(constraint, block)` pair. A constraint
/// with `block_id = None` in its bounds generates one entry per block;
/// a constraint with `block_id = Some(k)` generates exactly one entry.
#[derive(Debug, Clone)]
pub struct GenericConstraintRowEntry {
    /// Index into `System::generic_constraints()` for the parent constraint.
    pub constraint_idx: usize,
    /// Entity ID of the parent constraint (copied from `GenericConstraint::id`).
    ///
    /// Stored here so that the simulation extraction pipeline can report the
    /// constraint identity without needing a reference to the full constraint list.
    pub entity_id: i32,
    /// Block index within the stage (0-indexed).
    pub block_idx: usize,
    /// The right-hand-side bound value for this row.
    pub bound: f64,
    /// Comparison sense of the constraint (`>=`, `<=`, or `==`).
    pub sense: ConstraintSense,
    /// Whether slack is enabled for this constraint.
    pub slack_enabled: bool,
    /// Penalty cost per unit of slack violation (`None` when slack is disabled).
    pub slack_penalty: f64,
    /// Column index of the positive-violation slack (`slack_plus`) when
    /// `slack.enabled = true`.  `None` when slack is disabled.
    pub slack_plus_col: Option<usize>,
    /// Column index of the negative-violation slack (`slack_minus`) when
    /// `slack.enabled = true` and `sense == Equal`.  `None` otherwise.
    pub slack_minus_col: Option<usize>,
}
