//! Context structs for reducing parameter count in hot-path functions.

use cobre_solver::StageTemplate;

/// Immutable per-stage LP layout and noise scaling parameters.
///
/// Groups the read-only parameters shared by the forward pass, backward pass,
/// and simulation pipeline. Constructed once at algorithm startup and passed
/// by reference to all hot-path functions.
pub struct StageContext<'a> {
    /// Stage LP templates (one per study stage).
    pub templates: &'a [StageTemplate],
    /// Row index of the first water-balance row in each stage template.
    pub base_rows: &'a [usize],
    /// Noise scaling factors, layout: `[stage * n_hydros + hydro]`.
    pub noise_scale: &'a [f64],
    /// Number of hydro plants with LP variables.
    pub n_hydros: usize,
    /// Number of buses with stochastic load noise.
    pub n_load_buses: usize,
    /// Row index of the first load-balance row in each stage template.
    pub load_balance_row_starts: &'a [usize],
    /// Bus indices for stochastic load mapping.
    pub load_bus_indices: &'a [usize],
    /// Number of blocks per stage.
    pub block_counts_per_stage: &'a [usize],
}
