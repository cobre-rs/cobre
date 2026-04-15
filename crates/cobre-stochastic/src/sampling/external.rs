//! `External` scenario sampling scheme — library type and eta standardization.
//!
//! [`ExternalScenarioLibrary`] stores pre-standardized eta values loaded from
//! externally provided scenario files. During the forward pass, the
//! `ClassSampler::External` variant indexes into this library to retrieve
//! noise vectors for a given (stage, scenario) pair.
//!
//! The library is per entity class — each class that uses the `External`
//! sampling scheme gets its own library instance.
//!
//! ## Eta storage layout
//!
//! The `eta` buffer uses **stage-major** layout:
//! `eta[stage * n_scenarios * n_entities + scenario * n_entities + entity]`.
//!
//! This is optimal for accessing all entities of a given (stage, scenario) pair,
//! matching the forward-pass access pattern.
//!
//! ## Standardization functions
//!
//! Three public functions populate the library from raw external scenario rows:
//!
//! - [`standardize_external_inflow`] — full PAR(p) standardization via [`solve_par_noise`]
//! - [`standardize_external_load`] — simple `(value - mean) / std` per (bus, stage)
//! - [`standardize_external_ncs`] — simple `(value - mean) / std` per (ncs, stage)
//!
//! [`solve_par_noise`]: crate::par::evaluate::solve_par_noise

use std::collections::HashSet;

use cobre_core::{
    scenario::{ExternalLoadRow, ExternalNcsRow, ExternalScenarioRow, LoadModel, NcsModel},
    temporal::{Stage, StageLagTransition},
    EntityId, HydroPastInflows,
};

use crate::StochasticError;

use crate::par::{evaluate::solve_par_noise, precompute::PrecomputedPar};

// ---------------------------------------------------------------------------
// ExternalScenarioLibrary
// ---------------------------------------------------------------------------

/// Pre-standardized eta store for external scenario files.
///
/// A pure data container — no sampling or selection logic is included.
/// Population is performed by the external-file parsing pass after
/// construction; selection is performed by the `ClassSampler::External`
/// variant during the forward pass.
///
/// # Construction
///
/// Use [`ExternalScenarioLibrary::new`], which allocates zero-filled buffers.
///
/// # Examples
///
/// ```
/// use cobre_stochastic::ExternalScenarioLibrary;
///
/// let raw = vec![50usize; 12];
/// let mut lib = ExternalScenarioLibrary::new(12, 50, 5, "inflow", raw);
/// assert_eq!(lib.n_stages(), 12);
/// assert_eq!(lib.n_scenarios(), 50);
/// assert_eq!(lib.n_entities(), 5);
/// assert_eq!(lib.entity_class(), "inflow");
///
/// // Write and read eta values.
/// lib.eta_slice_mut(0, 1).copy_from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
/// assert_eq!(lib.eta_slice(0, 1), &[1.0, 2.0, 3.0, 4.0, 5.0]);
/// ```
#[derive(Debug, Clone)]
pub struct ExternalScenarioLibrary {
    /// Flat eta buffer in stage-major layout.
    eta: Box<[f64]>,
    /// Number of study stages.
    n_stages: usize,
    /// Number of scenarios per stage (the padded, uniform count used for the eta buffer).
    n_scenarios: usize,
    /// Number of entities in the eta vector width.
    n_entities: usize,
    /// Entity class label for diagnostic messages (e.g., `"inflow"`, `"load"`, `"ncs"`).
    entity_class: &'static str,
    /// Pre-padding scenario count per stage.
    ///
    /// When no padding is needed, all entries equal `n_scenarios`. When
    /// padding was applied (non-uniform input counts), stages with fewer raw
    /// scenarios retain their original count here so that downstream code
    /// (e.g., opening-tree clamping) can distinguish padded from unpadded
    /// stages.
    raw_scenarios_per_stage: Vec<usize>,
}

impl ExternalScenarioLibrary {
    /// Construct a new library with zero-filled buffers.
    ///
    /// `raw_scenarios_per_stage` records the pre-padding scenario count for
    /// each stage. When all stages share the same count (no padding), every
    /// entry equals `n_scenarios`. When padding is applied by
    /// [`pad_library_to_uniform`], the padded stages retain their original
    /// smaller count in this vector.
    ///
    /// # Parameters
    ///
    /// - `n_stages` — number of study stages
    /// - `n_scenarios` — number of scenarios per stage in the eta buffer (max across all stages)
    /// - `n_entities` — number of entities in the eta vector (e.g., hydros, buses, NCS units)
    /// - `entity_class` — label for diagnostic messages (e.g., `"inflow"`, `"load"`, `"ncs"`)
    /// - `raw_scenarios_per_stage` — pre-padding scenario count per stage (length must equal `n_stages`)
    #[must_use]
    pub fn new(
        n_stages: usize,
        n_scenarios: usize,
        n_entities: usize,
        entity_class: &'static str,
        raw_scenarios_per_stage: Vec<usize>,
    ) -> Self {
        debug_assert_eq!(
            raw_scenarios_per_stage.len(),
            n_stages,
            "raw_scenarios_per_stage.len() ({}) must equal n_stages ({})",
            raw_scenarios_per_stage.len(),
            n_stages,
        );
        Self {
            eta: vec![0.0_f64; n_stages * n_scenarios * n_entities].into_boxed_slice(),
            n_stages,
            n_scenarios,
            n_entities,
            entity_class,
            raw_scenarios_per_stage,
        }
    }

    // -----------------------------------------------------------------------
    // Dimension accessors
    // -----------------------------------------------------------------------

    /// Returns the number of study stages.
    #[must_use]
    #[inline]
    pub fn n_stages(&self) -> usize {
        self.n_stages
    }

    /// Returns the number of scenarios per stage.
    #[must_use]
    #[inline]
    pub fn n_scenarios(&self) -> usize {
        self.n_scenarios
    }

    /// Returns the number of entities in the eta vector.
    #[must_use]
    #[inline]
    pub fn n_entities(&self) -> usize {
        self.n_entities
    }

    /// Returns the entity class label for diagnostic messages.
    #[must_use]
    #[inline]
    pub fn entity_class(&self) -> &str {
        self.entity_class
    }

    /// Returns the pre-padding scenario count per stage.
    ///
    /// When no padding was applied, every entry equals `n_scenarios()`. When
    /// padding was applied, stages with fewer raw scenarios carry their
    /// original (pre-padding) count here.
    #[must_use]
    #[inline]
    pub fn raw_scenarios_per_stage(&self) -> &[usize] {
        &self.raw_scenarios_per_stage
    }

    // -----------------------------------------------------------------------
    // Eta accessors
    // -----------------------------------------------------------------------

    /// Returns the `n_entities`-length slice of eta values for `(stage, scenario)`.
    ///
    /// Layout: `eta[stage * n_scenarios * n_entities + scenario * n_entities + entity]`.
    ///
    /// # Panics
    ///
    /// Panics if `stage >= n_stages` or `scenario >= n_scenarios`.
    #[must_use]
    #[inline]
    pub fn eta_slice(&self, stage: usize, scenario: usize) -> &[f64] {
        assert!(
            stage < self.n_stages,
            "stage ({stage}) must be < n_stages ({})",
            self.n_stages
        );
        assert!(
            scenario < self.n_scenarios,
            "scenario ({scenario}) must be < n_scenarios ({})",
            self.n_scenarios
        );
        let offset = (stage * self.n_scenarios + scenario) * self.n_entities;
        &self.eta[offset..offset + self.n_entities]
    }

    /// Returns a mutable `n_entities`-length slice of eta values for `(stage, scenario)`.
    ///
    /// Used by the external-file parsing pass to populate the library.
    ///
    /// # Panics
    ///
    /// Panics if `stage >= n_stages` or `scenario >= n_scenarios`.
    #[must_use]
    #[inline]
    pub fn eta_slice_mut(&mut self, stage: usize, scenario: usize) -> &mut [f64] {
        assert!(
            stage < self.n_stages,
            "stage ({stage}) must be < n_stages ({})",
            self.n_stages
        );
        assert!(
            scenario < self.n_scenarios,
            "scenario ({scenario}) must be < n_scenarios ({})",
            self.n_scenarios
        );
        let offset = (stage * self.n_scenarios + scenario) * self.n_entities;
        &mut self.eta[offset..offset + self.n_entities]
    }
}

// ---------------------------------------------------------------------------
// standardize_external_inflow
// ---------------------------------------------------------------------------

/// Populate `library` with standardized eta values from external inflow rows.
///
/// For each (stage, scenario, hydro), inverts the PAR(p) model equation via
/// [`solve_par_noise`] to produce the standardized noise `η` that, when fed
/// through the forward PAR pass, would reproduce the raw external value.
///
/// ## Lag initialization and advancement
///
/// Lag state is initialized from `past_inflows` and advanced using the
/// frozen-lag + accumulation pattern encoded in `stage_lag_transitions`.
/// Within a lag period (`finalize_period == false`), the lag buffer is frozen
/// at the previous period's values. At a period boundary (`finalize_period == true`),
/// the lag buffer is shifted with the weighted average of the raw external values
/// accumulated during the finalized period.
///
/// For uniform monthly studies (each stage is one lag period), this produces
/// bit-for-bit identical eta values to the simple per-stage advancement used
/// previously, because each stage has `accumulate_weight=1.0`, `spillover_weight=0.0`,
/// and `finalize_period=true`.
///
/// ## `NEG_INFINITY` values
///
/// If `solve_par_noise` returns `f64::NEG_INFINITY` (sigma=0, non-matching target),
/// the value is stored as-is and will be caught by Tier 3 validation (V3.7).
///
/// # Parameters
///
/// - `library` — destination library, must have `n_entities() == hydro_ids.len()`
/// - `external_rows` — sorted raw rows (sorted by `(stage_id, scenario_id, hydro_id)`)
/// - `hydro_ids` — canonical-order hydro entity IDs
/// - `stages` — study stages (must match `library.n_stages()`)
/// - `par` — precomputed PAR model parameters
/// - `past_inflows` — pre-study inflow history sorted by `hydro_id`; used for
///   lag initialization at stage 0
/// - `stage_lag_transitions` — pre-computed lag transition config, one per stage,
///   same length as `stages`
///
/// # Panics
///
/// Panics in debug builds if dimension mismatches are detected.
#[allow(clippy::too_many_lines)]
pub fn standardize_external_inflow(
    library: &mut ExternalScenarioLibrary,
    external_rows: &[ExternalScenarioRow],
    hydro_ids: &[EntityId],
    stages: &[Stage],
    par: &PrecomputedPar,
    past_inflows: &[HydroPastInflows],
    stage_lag_transitions: &[StageLagTransition],
) {
    let n_stages = library.n_stages();
    let n_scenarios = library.n_scenarios();
    let n_hydros = hydro_ids.len();
    let max_order = par.max_order();

    debug_assert_eq!(
        library.n_entities(),
        n_hydros,
        "library.n_entities() ({}) must equal hydro_ids.len() ({})",
        library.n_entities(),
        n_hydros,
    );
    debug_assert_eq!(
        n_stages,
        stages.len(),
        "library.n_stages() ({}) must equal stages.len() ({})",
        n_stages,
        stages.len(),
    );
    debug_assert_eq!(
        stage_lag_transitions.len(),
        n_stages,
        "stage_lag_transitions.len() ({}) must equal n_stages ({})",
        stage_lag_transitions.len(),
        n_stages,
    );

    if n_hydros == 0 || n_stages == 0 || n_scenarios == 0 {
        return;
    }

    // -----------------------------------------------------------------------
    // Build hydro ID → canonical index map.
    // -----------------------------------------------------------------------
    let hydro_index: std::collections::HashMap<EntityId, usize> = hydro_ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect();

    // -----------------------------------------------------------------------
    // Build raw value lookup: flat array indexed as
    //   raw[stage * n_scenarios * n_hydros + scenario * n_hydros + h_idx]
    //
    // External rows are sorted by (stage_id, scenario_id, hydro_id) per
    // ticket-022 contract, but we use a HashMap for robustness.
    // -----------------------------------------------------------------------
    let mut raw_values = vec![0.0_f64; n_stages * n_scenarios * n_hydros].into_boxed_slice();

    #[allow(clippy::cast_sign_loss)]
    for row in external_rows {
        let stage_idx = row.stage_id as usize;
        let scenario_idx = row.scenario_id as usize;
        if let Some(&h_idx) = hydro_index.get(&row.hydro_id) {
            debug_assert!(
                stage_idx < n_stages,
                "row stage_id ({stage_idx}) >= n_stages ({n_stages})",
            );
            debug_assert!(
                scenario_idx < n_scenarios,
                "row scenario_id ({scenario_idx}) >= n_scenarios ({n_scenarios})",
            );
            raw_values[stage_idx * n_scenarios * n_hydros + scenario_idx * n_hydros + h_idx] =
                row.value_m3s;
        }
    }

    // -----------------------------------------------------------------------
    // Build past-inflows lookup: past_lag[h_idx][lag] = values_m3s[lag].
    // Indexed as past_lag_buf[h_idx * max_order + lag].
    // Hydros absent from past_inflows default to 0.0 for all lags.
    // -----------------------------------------------------------------------
    let safe_max_order = max_order.max(1);
    let mut past_lag_buf = vec![0.0_f64; n_hydros * safe_max_order];
    for pi in past_inflows {
        if let Some(&h_idx) = hydro_index.get(&pi.hydro_id) {
            let order_h = par.order(h_idx);
            for (lag, &value) in pi.values_m3s.iter().enumerate().take(order_h) {
                past_lag_buf[h_idx * safe_max_order + lag] = value;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Pre-allocate reusable buffers. All reused across iterations.
    //
    // lag_state[h * safe_max_order + l] = lag-l value for hydro h.
    // Initialized from past_lag_buf at the start of each scenario.
    //
    // lag_buf: scratch buffer of length safe_max_order passed to solve_par_noise.
    //
    // lag_accum[h]: accumulator of weighted raw values for hydro h, used to
    // compute the weighted-average monthly inflow when finalizing a lag period.
    //
    // lag_weight_accum: sum of accumulate_weight values accumulated so far in
    // the current lag period.
    // -----------------------------------------------------------------------
    let mut lag_state = vec![0.0_f64; n_hydros * safe_max_order];
    let mut lag_buf = vec![0.0_f64; safe_max_order];
    let mut lag_accum = vec![0.0_f64; n_hydros];

    // -----------------------------------------------------------------------
    // March forward through (scenario, stage, hydro) and compute eta.
    // Each scenario is independent: reset lag state and accumulators at the
    // start of each scenario.
    // -----------------------------------------------------------------------
    for scenario in 0..n_scenarios {
        // Reset lag state from past_lag_buf (same initial conditions for all scenarios).
        lag_state.copy_from_slice(&past_lag_buf);
        lag_accum.fill(0.0);
        let mut lag_weight_accum = 0.0_f64;

        for t in 0..n_stages {
            let stage_lag = &stage_lag_transitions[t];

            // ── Compute eta for each hydro using the current (frozen) lag state ──
            for h in 0..n_hydros {
                let target = raw_values[t * n_scenarios * n_hydros + scenario * n_hydros + h];
                let order_h = par.order(h);

                // Build the lag_buf for solve_par_noise from the per-hydro lag state.
                // lag_state uses lag-major layout: lag_state[h * safe_max_order + l].
                for (l, slot) in lag_buf.iter_mut().enumerate().take(order_h) {
                    *slot = lag_state[h * safe_max_order + l];
                }

                let det_base = par.deterministic_base(t, h);
                let psi = par.psi_slice(t, h);
                let sigma = par.sigma(t, h);

                let eta = solve_par_noise(det_base, psi, order_h, &lag_buf, sigma, target);

                library.eta_slice_mut(t, scenario)[h] = eta;
            }

            // ── Step 1: Accumulate this stage's raw values × accumulate_weight ──
            // Must happen unconditionally before the finalize check so that this
            // stage's contribution is always included in the period average.
            let w = stage_lag.accumulate_weight;
            for h in 0..n_hydros {
                let val = raw_values[t * n_scenarios * n_hydros + scenario * n_hydros + h];
                lag_accum[h] += val * w;
            }
            lag_weight_accum += w;

            // ── Step 2: Finalize — shift lag state at period boundary ────────────
            if stage_lag.finalize_period && lag_weight_accum > 0.0 {
                // Compute per-hydro weighted average and shift the lag state:
                //   lag_state[h, l] <- lag_state[h, l-1]  for l in (1..max_order).rev()
                //   lag_state[h, 0] <- weighted_avg[h]
                let inv = 1.0 / lag_weight_accum;
                for h in 0..n_hydros {
                    let avg = lag_accum[h] * inv;
                    // Shift older lags down (from highest lag to lag-1 to avoid overwrite).
                    for l in (1..safe_max_order).rev() {
                        lag_state[h * safe_max_order + l] = lag_state[h * safe_max_order + l - 1];
                    }
                    // Newest lag slot gets the weighted average for this period.
                    lag_state[h * safe_max_order] = avg;
                }

                // ── Reset accumulator; seed spillover if required ────────────────
                // Spillover uses the RAW inflow value (not the averaged value),
                // consistent with lag-state accumulation patterns.
                let sw = stage_lag.spillover_weight;
                if sw > 0.0 {
                    for h in 0..n_hydros {
                        let raw_val =
                            raw_values[t * n_scenarios * n_hydros + scenario * n_hydros + h];
                        lag_accum[h] = raw_val * sw;
                    }
                    lag_weight_accum = sw;
                } else {
                    lag_accum.fill(0.0);
                    lag_weight_accum = 0.0;
                }
            }
            // Non-finalizing stages: lag_state is left untouched (lags frozen).
        }
    }
}

// ---------------------------------------------------------------------------
// standardize_external_load
// ---------------------------------------------------------------------------

/// Populate `library` with standardized eta values from external load rows.
///
/// For each (stage, scenario, bus), computes `η = (value_mw - mean_mw) / std_mw`
/// using the [`LoadModel`] for that (bus, stage). When `std_mw == 0.0`, stores
/// `η = 0.0` (deterministic entity, consistent with the sigma=0 convention).
///
/// # Parameters
///
/// - `library` — destination library, must have `n_entities() == bus_ids.len()`
/// - `external_rows` — sorted raw rows (sorted by `(stage_id, scenario_id, bus_id)`)
/// - `bus_ids` — canonical-order bus entity IDs
/// - `load_models` — per-(bus, stage) load models sorted by `(bus_id, stage_id)`
/// - `n_stages` — number of study stages
///
/// # Panics
///
/// Panics in debug builds if dimension mismatches are detected.
pub fn standardize_external_load(
    library: &mut ExternalScenarioLibrary,
    external_rows: &[ExternalLoadRow],
    bus_ids: &[EntityId],
    load_models: &[LoadModel],
    n_stages: usize,
) {
    let n_buses = bus_ids.len();
    let n_scenarios = library.n_scenarios();

    debug_assert_eq!(
        library.n_entities(),
        n_buses,
        "library.n_entities() ({}) must equal bus_ids.len() ({})",
        library.n_entities(),
        n_buses,
    );
    debug_assert_eq!(
        library.n_stages(),
        n_stages,
        "library.n_stages() ({}) must equal n_stages ({})",
        library.n_stages(),
        n_stages,
    );

    if n_buses == 0 || n_stages == 0 || n_scenarios == 0 {
        return;
    }

    // -----------------------------------------------------------------------
    // Build bus ID → canonical index map.
    // -----------------------------------------------------------------------
    let bus_index: std::collections::HashMap<EntityId, usize> =
        bus_ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();

    // -----------------------------------------------------------------------
    // Build (bus_idx, stage_idx) → (mean_mw, std_mw) lookup.
    // Indexed as mean_std[stage * n_buses + bus_idx] = (mean, std).
    // -----------------------------------------------------------------------
    let mut mean_std = vec![(0.0_f64, 0.0_f64); n_stages * n_buses];
    #[allow(clippy::cast_sign_loss)]
    for model in load_models {
        if let Some(&b_idx) = bus_index.get(&model.bus_id) {
            let stage_idx = model.stage_id as usize;
            if stage_idx < n_stages {
                mean_std[stage_idx * n_buses + b_idx] = (model.mean_mw, model.std_mw);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Apply (value - mean) / std for each row.
    // -----------------------------------------------------------------------
    #[allow(clippy::cast_sign_loss)]
    for row in external_rows {
        let stage_idx = row.stage_id as usize;
        let scenario_idx = row.scenario_id as usize;
        if let Some(&b_idx) = bus_index.get(&row.bus_id) {
            debug_assert!(
                stage_idx < n_stages,
                "row stage_id ({stage_idx}) >= n_stages ({n_stages})",
            );
            debug_assert!(
                scenario_idx < n_scenarios,
                "row scenario_id ({scenario_idx}) >= n_scenarios ({n_scenarios})",
            );
            let (mean, std) = mean_std[stage_idx * n_buses + b_idx];
            let eta = if std == 0.0 {
                0.0
            } else {
                (row.value_mw - mean) / std
            };
            library.eta_slice_mut(stage_idx, scenario_idx)[b_idx] = eta;
        }
    }
}

// ---------------------------------------------------------------------------
// standardize_external_ncs
// ---------------------------------------------------------------------------

/// Populate `library` with standardized eta values from external NCS rows.
///
/// For each (stage, scenario, ncs), computes `η = (value - mean) / std`
/// using the [`NcsModel`] for that (ncs, stage). When `std == 0.0`, stores
/// `η = 0.0` (deterministic entity, consistent with the sigma=0 convention).
///
/// # Parameters
///
/// - `library` — destination library, must have `n_entities() == ncs_ids.len()`
/// - `external_rows` — sorted raw rows (sorted by `(stage_id, scenario_id, ncs_id)`)
/// - `ncs_ids` — canonical-order NCS entity IDs
/// - `ncs_models` — per-(ncs, stage) NCS models sorted by `(ncs_id, stage_id)`
/// - `n_stages` — number of study stages
///
/// # Panics
///
/// Panics in debug builds if dimension mismatches are detected.
pub fn standardize_external_ncs(
    library: &mut ExternalScenarioLibrary,
    external_rows: &[ExternalNcsRow],
    ncs_ids: &[EntityId],
    ncs_models: &[NcsModel],
    n_stages: usize,
) {
    let n_ncs = ncs_ids.len();
    let n_scenarios = library.n_scenarios();

    debug_assert_eq!(
        library.n_entities(),
        n_ncs,
        "library.n_entities() ({}) must equal ncs_ids.len() ({})",
        library.n_entities(),
        n_ncs,
    );
    debug_assert_eq!(
        library.n_stages(),
        n_stages,
        "library.n_stages() ({}) must equal n_stages ({})",
        library.n_stages(),
        n_stages,
    );

    if n_ncs == 0 || n_stages == 0 || n_scenarios == 0 {
        return;
    }

    // -----------------------------------------------------------------------
    // Build ncs ID → canonical index map.
    // -----------------------------------------------------------------------
    let ncs_index: std::collections::HashMap<EntityId, usize> =
        ncs_ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();

    // -----------------------------------------------------------------------
    // Build (ncs_idx, stage_idx) → (mean, std) lookup.
    // Indexed as mean_std[stage * n_ncs + ncs_idx] = (mean, std).
    // -----------------------------------------------------------------------
    let mut mean_std = vec![(0.0_f64, 0.0_f64); n_stages * n_ncs];
    #[allow(clippy::cast_sign_loss)]
    for model in ncs_models {
        if let Some(&n_idx) = ncs_index.get(&model.ncs_id) {
            let stage_idx = model.stage_id as usize;
            if stage_idx < n_stages {
                mean_std[stage_idx * n_ncs + n_idx] = (model.mean, model.std);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Apply (value - mean) / std for each row.
    // -----------------------------------------------------------------------
    #[allow(clippy::cast_sign_loss)]
    for row in external_rows {
        let stage_idx = row.stage_id as usize;
        let scenario_idx = row.scenario_id as usize;
        if let Some(&n_idx) = ncs_index.get(&row.ncs_id) {
            debug_assert!(
                stage_idx < n_stages,
                "row stage_id ({stage_idx}) >= n_stages ({n_stages})",
            );
            debug_assert!(
                scenario_idx < n_scenarios,
                "row scenario_id ({scenario_idx}) >= n_scenarios ({n_scenarios})",
            );
            let (mean, std) = mean_std[stage_idx * n_ncs + n_idx];
            let eta = if std == 0.0 {
                0.0
            } else {
                (row.value - mean) / std
            };
            library.eta_slice_mut(stage_idx, scenario_idx)[n_idx] = eta;
        }
    }
}

// ---------------------------------------------------------------------------
// validate_external_library
// ---------------------------------------------------------------------------

/// Validate a populated [`ExternalScenarioLibrary`] against construction inputs.
///
/// This is the Tier 3 validation gate for external scenario libraries.
/// It runs after per-class file parsing and eta standardization, confirming
/// that the library is well-formed before it is stored on `StudySetup`.
///
/// Validation uses **fail-fast** semantics: the first failed check immediately
/// returns `Err`. The scenario-count warning (V3.8) is emitted via
/// `tracing::warn!` and does not abort construction.
///
/// ## Checks performed
///
/// | ID   | Kind    | Description                                                              |
/// |------|---------|--------------------------------------------------------------------------|
/// | V3.2 | Error   | Every entity in `entity_ids` must have data in `row_entity_ids`.         |
/// | V3.3 | Error   | Every study stage must have at least one row in `rows_per_stage`.        |
/// | V3.4 | Error   | The number of scenarios per stage must be uniform across all stages.     |
/// | V3.5 | Error   | Every entity ID in `row_entity_ids` must exist in `entity_ids`.          |
/// | V3.6 | Assert  | All values in raw rows are finite (parser invariant — `debug_assert`).   |
/// | V3.7 | Error   | No eta value in the library may be `f64::NEG_INFINITY` or `NaN`.        |
/// | V3.8 | Warning | `library.n_scenarios() < forward_passes` — log a warning.               |
///
/// ## Pre-extracted metadata
///
/// The caller (ticket-025) is responsible for extracting `row_entity_ids` and
/// `rows_per_stage` from the raw parsed rows before calling this function.
///
/// - `row_entity_ids` — the set of entity IDs that appear in the raw rows
/// - `rows_per_stage` — number of raw rows for each study stage (length must
///   equal `n_stages`); `rows_per_stage[s]` is the total row count for stage `s`
///   across all entities and scenarios
///
/// # Errors
///
/// Returns [`StochasticError::InsufficientData`] with a message prefixed by the
/// check ID (e.g., `"V3.2: ..."`) for the first failed error check.
///
/// # Examples
///
/// ```
/// use std::collections::HashSet;
/// use cobre_core::EntityId;
/// use cobre_stochastic::{ExternalScenarioLibrary, sampling::external::validate_external_library};
///
/// let lib = ExternalScenarioLibrary::new(3, 50, 2, "inflow", vec![50usize; 3]);
/// let entity_ids = [EntityId(1), EntityId(2)];
/// let row_entity_ids: HashSet<EntityId> = entity_ids.iter().copied().collect();
/// // 50 scenarios × 2 entities = 100 rows per stage.
/// let rows_per_stage = vec![100usize; 3];
/// let result = validate_external_library(
///     &lib, &entity_ids, &row_entity_ids, &rows_per_stage, 3, 50,
/// );
/// assert!(result.is_ok());
/// ```
pub fn validate_external_library<S: std::hash::BuildHasher>(
    library: &ExternalScenarioLibrary,
    entity_ids: &[EntityId],
    row_entity_ids: &HashSet<EntityId, S>,
    rows_per_stage: &[usize],
    n_stages: usize,
    forward_passes: u32,
) -> Result<(), StochasticError> {
    let n_entities = entity_ids.len();
    let class = library.entity_class();

    // -----------------------------------------------------------------------
    // V3.2 — Entity coverage: every entity in `entity_ids` must appear in
    // `row_entity_ids`.
    // -----------------------------------------------------------------------
    for &id in entity_ids {
        if !row_entity_ids.contains(&id) {
            return Err(StochasticError::InsufficientData {
                context: format!(
                    "V3.2: external {class} library missing data for {class} {id}; \
                     entity has zero rows in the external file",
                    id = id.0,
                ),
            });
        }
    }

    // -----------------------------------------------------------------------
    // V3.3 — Stage coverage: every study stage must have at least one row.
    // -----------------------------------------------------------------------
    for (stage_idx, &count) in rows_per_stage.iter().enumerate().take(n_stages) {
        if count == 0 {
            return Err(StochasticError::InsufficientData {
                context: format!(
                    "V3.3: external {class} library has no rows for stage {stage_idx}; \
                     every study stage must have at least one row",
                ),
            });
        }
    }

    // -----------------------------------------------------------------------
    // V3.4 — Exact divisibility: rows_per_stage[s] must be exactly divisible
    // by n_entities (no partial rows). Non-uniform counts across stages are
    // now accepted — padding is applied after standardization in the setup
    // blocks. See `pad_library_to_uniform`.
    //
    // Guard against zero entities to avoid division by zero; this situation
    // is benign (empty library) so we skip the check.
    // -----------------------------------------------------------------------
    if n_entities > 0 && n_stages > 0 {
        for (stage_idx, &count) in rows_per_stage.iter().enumerate().take(n_stages) {
            if count % n_entities != 0 {
                return Err(StochasticError::InsufficientData {
                    context: format!(
                        "V3.4: external {class} library has {count} rows for stage \
                         {stage_idx} which is not exactly divisible by {n_entities} \
                         entities; each stage must have a whole number of scenarios",
                    ),
                });
            }
        }
    }

    // -----------------------------------------------------------------------
    // V3.5 — Entity ID existence: every ID in `row_entity_ids` must be a
    // known entity in `entity_ids`.
    // -----------------------------------------------------------------------
    let entity_id_set: HashSet<EntityId> = entity_ids.iter().copied().collect();
    for &id in row_entity_ids {
        if !entity_id_set.contains(&id) {
            return Err(StochasticError::InsufficientData {
                context: format!(
                    "V3.5: external {class} library contains unknown entity ID {id}; \
                     the ID does not exist in the canonical {class} entity list",
                    id = id.0,
                ),
            });
        }
    }

    // -----------------------------------------------------------------------
    // V3.6 — Finite values (parser invariant — debug_assert only).
    // The parser enforces this; we do not repeat the full scan here.
    // -----------------------------------------------------------------------
    debug_assert!(
        row_entity_ids.iter().all(|id| entity_id_set.contains(id)),
        "V3.6: row_entity_ids contains IDs not in entity_id_set (parser invariant violated)",
    );

    // -----------------------------------------------------------------------
    // V3.7 — PAR compatibility: no eta value may be NEG_INFINITY or NaN.
    //
    // NEG_INFINITY is the sentinel written by standardize_external_inflow when
    // sigma=0 but the external value does not match the deterministic base
    // (data quality issue). NaN indicates a numerical failure.
    // -----------------------------------------------------------------------
    for stage in 0..library.n_stages() {
        for scenario in 0..library.n_scenarios() {
            let eta = library.eta_slice(stage, scenario);
            for (entity_idx, &value) in eta.iter().enumerate() {
                if value == f64::NEG_INFINITY || value.is_nan() {
                    return Err(StochasticError::InsufficientData {
                        context: format!(
                            "V3.7: external {class} library contains non-finite eta at \
                             stage {stage}, scenario {scenario}, entity {entity_idx} \
                             (value = {value}) — sigma=0 with non-matching external value \
                             or numerical failure",
                        ),
                    });
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // V3.8 — Scenario count warning: fewer scenarios than forward passes.
    // -----------------------------------------------------------------------
    if library.n_scenarios() < forward_passes as usize {
        tracing::warn!(
            n_scenarios = library.n_scenarios(),
            forward_passes = forward_passes,
            entity_class = class,
            "external {class} library has fewer scenarios ({n_scenarios}) than forward \
             passes ({forward_passes}); scenarios will be reused across forward passes",
            n_scenarios = library.n_scenarios(),
            forward_passes = forward_passes,
        );
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// pad_library_to_uniform
// ---------------------------------------------------------------------------

/// Replicate eta values in stages that have fewer raw scenarios than
/// `library.n_scenarios()`, filling the library to a uniform count.
///
/// For each stage `s` where `raw_scenarios_per_stage[s] < n_scenarios`,
/// the raw scenario slots `0..raw_count` are already populated by the
/// preceding standardization call. This function copies those values into
/// the remaining slots `raw_count..n_scenarios` using wrap-around indexing
/// (`k % raw_count`), so that every scenario index in `0..n_scenarios` holds
/// a valid (possibly replicated) eta vector.
///
/// The function is a no-op when all stages already have `n_scenarios` raw
/// scenarios (uniform input). A single `tracing::info!` is emitted only when
/// at least one stage is actually padded.
///
/// # Panics
///
/// Does not panic — all indices are derived from library dimensions.
pub fn pad_library_to_uniform(library: &mut ExternalScenarioLibrary) {
    let n_scenarios = library.n_scenarios();
    let n_stages = library.n_stages();
    let n_entities = library.n_entities();
    // Copy the class name before the mutable borrow loop so the
    // tracing macro can reference it after eta mutation.
    let class = library.entity_class().to_owned();

    // Collect which stages need padding so we can emit a single info log.
    let mut padded_stages: Vec<(usize, usize)> = Vec::new();

    for s in 0..n_stages {
        let raw_count = library.raw_scenarios_per_stage[s];
        if raw_count == 0 || raw_count >= n_scenarios {
            // Nothing to do for this stage.
            continue;
        }

        padded_stages.push((s, raw_count));

        // For each slot that needs to be filled, copy from the wrap-around
        // raw slot. We work directly on the flat eta buffer to avoid borrow
        // issues with `eta_slice` / `eta_slice_mut`.
        for k in raw_count..n_scenarios {
            let src_k = k % raw_count;
            let src_offset = (s * n_scenarios + src_k) * n_entities;
            let dst_offset = (s * n_scenarios + k) * n_entities;
            library
                .eta
                .copy_within(src_offset..src_offset + n_entities, dst_offset);
        }
    }

    if !padded_stages.is_empty() {
        let stage_list: Vec<String> = padded_stages
            .iter()
            .map(|(s, raw)| format!("stage {s} ({raw}→{n_scenarios})"))
            .collect();
        tracing::info!(
            entity_class = class,
            padded_to = n_scenarios,
            "external {class} library padded to {n_scenarios} scenarios: {}",
            stage_list.join(", "),
        );
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp
)]
mod tests {
    use chrono::NaiveDate;
    use cobre_core::{
        scenario::{
            ExternalLoadRow, ExternalNcsRow, ExternalScenarioRow, InflowModel, LoadModel, NcsModel,
        },
        temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageLagTransition,
            StageRiskConfig, StageStateConfig,
        },
        EntityId, HydroPastInflows,
    };

    use super::{
        standardize_external_inflow, standardize_external_load, standardize_external_ncs,
        ExternalScenarioLibrary,
    };
    use crate::par::{evaluate::evaluate_par, precompute::PrecomputedPar};

    /// Build `n_stages` uniform-monthly transitions: each stage finalizes its own
    /// period with full weight and no spillover. Passing these to
    /// `standardize_external_inflow` produces bit-for-bit identical results to the
    /// old per-stage advancement logic for uniform monthly studies.
    fn uniform_monthly_transitions(n_stages: usize) -> Vec<StageLagTransition> {
        vec![
            StageLagTransition {
                accumulate_weight: 1.0,
                spillover_weight: 0.0,
                finalize_period: true,
                accumulate_downstream: false,
                downstream_accumulate_weight: 0.0,
                downstream_spillover_weight: 0.0,
                downstream_finalize: false,
                rebuild_from_downstream: false,
            };
            n_stages
        ]
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn make_stage(index: usize, id: i32, season_id: usize) -> Stage {
        let date = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
        Stage {
            index,
            id,
            start_date: date,
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: Some(season_id),
            blocks: vec![Block {
                index: 0,
                name: "SINGLE".to_string(),
                duration_hours: 744.0,
            }],
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

    fn make_inflow_model(
        hydro_id: i32,
        stage_id: i32,
        mean: f64,
        std: f64,
        ar: Vec<f64>,
    ) -> InflowModel {
        InflowModel {
            hydro_id: EntityId(hydro_id),
            stage_id,
            mean_m3s: mean,
            std_m3s: std,
            ar_coefficients: ar,
            residual_std_ratio: 1.0,
        }
    }

    // -----------------------------------------------------------------------
    // Inflow standardization tests
    // -----------------------------------------------------------------------

    /// AR(0): 1 hydro, 2 stages, 1 scenario. Values [120.0, 90.0].
    /// Expected: eta[stage=0] = (120-100)/30 = 0.6667, eta[stage=1] = (90-100)/30 = -0.3333.
    #[test]
    fn test_inflow_ar0_standardization() {
        let hydro_id = EntityId(1);
        let hydro_ids = vec![hydro_id];

        // Two stages, both season 0 (single-season system).
        let stages = vec![make_stage(0, 0, 0), make_stage(1, 1, 0)];

        // AR(0): mean=100, std=30.
        let models = vec![
            make_inflow_model(1, 0, 100.0, 30.0, vec![]),
            make_inflow_model(1, 1, 100.0, 30.0, vec![]),
        ];
        let par = PrecomputedPar::build(&models, &stages, &hydro_ids).unwrap();

        // 2 stages, 1 scenario, 1 hydro.
        let mut lib = ExternalScenarioLibrary::new(2, 1, 1, "inflow", vec![1, 1]);

        let rows = vec![
            ExternalScenarioRow {
                stage_id: 0,
                scenario_id: 0,
                hydro_id,
                value_m3s: 120.0,
            },
            ExternalScenarioRow {
                stage_id: 1,
                scenario_id: 0,
                hydro_id,
                value_m3s: 90.0,
            },
        ];
        // AR(0) has no lags; past_inflows is irrelevant but must be provided.
        let transitions = uniform_monthly_transitions(stages.len());
        standardize_external_inflow(
            &mut lib,
            &rows,
            &hydro_ids,
            &stages,
            &par,
            &[],
            &transitions,
        );

        let eta_0 = lib.eta_slice(0, 0)[0];
        let eta_1 = lib.eta_slice(1, 0)[0];

        assert!(
            (eta_0 - (120.0_f64 - 100.0) / 30.0).abs() < 1e-10,
            "eta[stage=0] = {eta_0}"
        );
        assert!(
            (eta_1 - (90.0_f64 - 100.0) / 30.0).abs() < 1e-10,
            "eta[stage=1] = {eta_1}"
        );
    }

    /// AR(1): stage 0 must use `past_inflows` (110.0) as lag-1,
    /// stage 1 must use the raw external value from stage 0 (130.0) as lag-1.
    ///
    /// Parameters: base=80, psi=[0.5], sigma=25.
    /// `past_inflows`: `values_m3s`=[110.0] for hydro 1.
    #[test]
    fn test_inflow_ar1_uses_external_lags() {
        let hydro_id = EntityId(1);
        let hydro_ids = vec![hydro_id];

        let stages = vec![make_stage(0, 0, 0), make_stage(1, 1, 0)];

        // AR(1): mean=160, std=25, psi*=0.5.
        // PrecomputedPar will compute: psi_val=0.5, base=80.0, sigma=25.0.
        let models = vec![
            make_inflow_model(1, 0, 160.0, 25.0, vec![0.5]),
            make_inflow_model(1, 1, 160.0, 25.0, vec![0.5]),
        ];
        let par = PrecomputedPar::build(&models, &stages, &hydro_ids).unwrap();

        // Sanity-check precomputed values.
        assert!((par.deterministic_base(0, 0) - 80.0).abs() < 1e-10);
        assert!((par.sigma(0, 0) - 25.0).abs() < 1e-10);
        assert!((par.psi_slice(0, 0)[0] - 0.5).abs() < 1e-10);

        let mut lib = ExternalScenarioLibrary::new(2, 1, 1, "inflow", vec![1, 1]);
        let rows = vec![
            ExternalScenarioRow {
                stage_id: 0,
                scenario_id: 0,
                hydro_id,
                value_m3s: 130.0,
            },
            ExternalScenarioRow {
                stage_id: 1,
                scenario_id: 0,
                hydro_id,
                value_m3s: 95.0,
            },
        ];

        // past_inflows provides lag-1 = 110.0 for stage 0.
        let past_inflows = vec![HydroPastInflows {
            hydro_id,
            values_m3s: vec![110.0],
        }];
        let transitions = uniform_monthly_transitions(stages.len());
        standardize_external_inflow(
            &mut lib,
            &rows,
            &hydro_ids,
            &stages,
            &par,
            &past_inflows,
            &transitions,
        );

        // Stage 0: lag-1 = 110.0 (from past_inflows).
        // eta_0 = (130.0 - 80.0 - 0.5 * 110.0) / 25.0 = (130 - 80 - 55) / 25 = -5/25 = -0.2
        let det_base_0 = par.deterministic_base(0, 0);
        let psi_0 = par.psi_slice(0, 0)[0];
        let sigma_0 = par.sigma(0, 0);
        let expected_eta_0 = (130.0 - det_base_0 - psi_0 * 110.0) / sigma_0;
        let eta_0 = lib.eta_slice(0, 0)[0];
        assert!(
            (eta_0 - expected_eta_0).abs() < 1e-10,
            "eta[stage=0] = {eta_0}, expected {expected_eta_0}"
        );

        // Stage 1: lag-1 = raw external value at stage 0 = 130.0.
        // eta_1 = (95.0 - 80.0 - 0.5 * 130.0) / 25.0 = (95 - 80 - 65) / 25 = -50/25 = -2.0
        let det_base_1 = par.deterministic_base(1, 0);
        let psi_1 = par.psi_slice(1, 0)[0];
        let sigma_1 = par.sigma(1, 0);
        let expected_eta_1 = (95.0 - det_base_1 - psi_1 * 130.0) / sigma_1;
        let eta_1 = lib.eta_slice(1, 0)[0];
        assert!(
            (eta_1 - expected_eta_1).abs() < 1e-10,
            "eta[stage=1] = {eta_1}, expected {expected_eta_1}"
        );
    }

    /// AR(1) with 3 weekly stages all within the same lag period:
    ///   stage 0: `accumulate_weight`=0.4, `finalize_period`=false
    ///   stage 1: `accumulate_weight`=0.4, `finalize_period`=false
    ///   stage 2: `accumulate_weight`=0.2, `finalize_period`=true
    ///
    /// Parameters: base=80, psi=\[0.5\], sigma=25.
    /// `past_inflows` provides lag-1 = 110.0.
    ///
    /// Stages 0 and 1 must use the frozen `past_inflows` lag (110.0), NOT the
    /// previous stage's raw value. Stage 2 also uses 110.0 (still frozen during
    /// that stage's `solve_par_noise` call, since the shift happens after). The
    /// weighted average computed at finalize is: (200*0.4 + 160*0.4 + 120*0.2) = 168.0.
    #[test]
    #[allow(clippy::too_many_lines)]
    fn test_inflow_ar1_weekly_frozen_lags() {
        let hydro_id = EntityId(1);
        let hydro_ids = vec![hydro_id];

        // 3 stages, all season 0 — same PAR parameters apply.
        let stages = vec![
            make_stage(0, 0, 0),
            make_stage(1, 1, 0),
            make_stage(2, 2, 0),
        ];

        // AR(1): mean=160, std=25, psi*=0.5 → base=80, sigma=25.
        let models = vec![
            make_inflow_model(1, 0, 160.0, 25.0, vec![0.5]),
            make_inflow_model(1, 1, 160.0, 25.0, vec![0.5]),
            make_inflow_model(1, 2, 160.0, 25.0, vec![0.5]),
        ];
        let par = PrecomputedPar::build(&models, &stages, &hydro_ids).unwrap();

        // 3 weekly stages within one monthly lag period:
        //   stages 0 and 1: accumulate but do not finalize
        //   stage 2: accumulate and finalize
        let transitions = vec![
            StageLagTransition {
                accumulate_weight: 0.4,
                spillover_weight: 0.0,
                finalize_period: false,
                accumulate_downstream: false,
                downstream_accumulate_weight: 0.0,
                downstream_spillover_weight: 0.0,
                downstream_finalize: false,
                rebuild_from_downstream: false,
            },
            StageLagTransition {
                accumulate_weight: 0.4,
                spillover_weight: 0.0,
                finalize_period: false,
                accumulate_downstream: false,
                downstream_accumulate_weight: 0.0,
                downstream_spillover_weight: 0.0,
                downstream_finalize: false,
                rebuild_from_downstream: false,
            },
            StageLagTransition {
                accumulate_weight: 0.2,
                spillover_weight: 0.0,
                finalize_period: true,
                accumulate_downstream: false,
                downstream_accumulate_weight: 0.0,
                downstream_spillover_weight: 0.0,
                downstream_finalize: false,
                rebuild_from_downstream: false,
            },
        ];

        let mut lib = ExternalScenarioLibrary::new(3, 1, 1, "inflow", vec![1, 1, 1]);
        let rows = vec![
            ExternalScenarioRow {
                stage_id: 0,
                scenario_id: 0,
                hydro_id,
                value_m3s: 200.0,
            },
            ExternalScenarioRow {
                stage_id: 1,
                scenario_id: 0,
                hydro_id,
                value_m3s: 160.0,
            },
            ExternalScenarioRow {
                stage_id: 2,
                scenario_id: 0,
                hydro_id,
                value_m3s: 120.0,
            },
        ];

        // past_inflows lag-1 = 110.0 for hydro 1.
        let past_inflows = vec![HydroPastInflows {
            hydro_id,
            values_m3s: vec![110.0],
        }];

        standardize_external_inflow(
            &mut lib,
            &rows,
            &hydro_ids,
            &stages,
            &par,
            &past_inflows,
            &transitions,
        );

        let det_base = par.deterministic_base(0, 0);
        let psi = par.psi_slice(0, 0)[0];
        let sigma = par.sigma(0, 0);

        // All three stages use the frozen lag-1 = 110.0 (from past_inflows).
        // The lag state is NOT shifted until stage 2 finalizes the period —
        // and even at stage 2 the shift happens AFTER solve_par_noise.
        let frozen_lag = 110.0_f64;
        let expected_eta_0 = (200.0 - det_base - psi * frozen_lag) / sigma;
        let expected_eta_1 = (160.0 - det_base - psi * frozen_lag) / sigma;
        let expected_eta_2 = (120.0 - det_base - psi * frozen_lag) / sigma;

        let eta_0 = lib.eta_slice(0, 0)[0];
        let eta_1 = lib.eta_slice(1, 0)[0];
        let eta_2 = lib.eta_slice(2, 0)[0];

        assert!(
            (eta_0 - expected_eta_0).abs() < 1e-10,
            "eta[stage=0] = {eta_0}, expected {expected_eta_0} (frozen lag)"
        );
        assert!(
            (eta_1 - expected_eta_1).abs() < 1e-10,
            "eta[stage=1] = {eta_1}, expected {expected_eta_1} (frozen lag, not stage-0 raw)"
        );
        assert!(
            (eta_2 - expected_eta_2).abs() < 1e-10,
            "eta[stage=2] = {eta_2}, expected {expected_eta_2} (frozen lag before finalize)"
        );

        // Also verify these differ from what naive per-stage advancement would produce.
        // With naive advancement, stage 1 would use raw value at stage 0 = 200.0.
        let naive_eta_1 = (160.0 - det_base - psi * 200.0) / sigma;
        assert!(
            (eta_1 - naive_eta_1).abs() > 1e-6,
            "eta[stage=1] must differ from naive per-stage value; got {eta_1} == naive {naive_eta_1}"
        );
    }

    /// AR(1): 2 stages where stage 0 has `spillover_weight > 0` and finalizes.
    ///
    /// stage 0: `accumulate_weight`=0.7, `spillover_weight`=0.3, `finalize_period`=true
    /// stage 1: `accumulate_weight`=1.0, `spillover_weight`=0.0, `finalize_period`=true
    ///
    /// Parameters: base=80, psi=\[0.5\], sigma=25. `past_inflows` lag-1 = 110.0.
    /// raw values: stage 0 = 150.0, stage 1 = 130.0.
    ///
    /// Stage 0 computation:
    ///   lag for `solve_par_noise` = 110.0 (frozen from `past_inflows`)
    ///   accumulate: `lag_accum[0]` = 150.0 * 0.7 = 105.0, `lag_weight` = 0.7
    ///   finalize: avg = 105.0 / 0.7 = 150.0; `lag_state[0]` shifts to 150.0
    ///   spillover seed: `lag_accum[0]` = 150.0 * 0.3 = 45.0, `lag_weight` = 0.3
    ///
    /// Stage 1 computation:
    ///   lag for `solve_par_noise` = 150.0 (shifted in at stage 0 finalize)
    ///   accumulate: `lag_accum[0]` += 130.0 * 1.0 → 45.0 + 130.0 = 175.0, `lag_weight` = 1.3
    ///   finalize: avg = 175.0 / 1.3 ≈ 134.615...
    #[test]
    fn test_inflow_ar1_spillover_accumulation() {
        let hydro_id = EntityId(1);
        let hydro_ids = vec![hydro_id];

        let stages = vec![make_stage(0, 0, 0), make_stage(1, 1, 0)];

        let models = vec![
            make_inflow_model(1, 0, 160.0, 25.0, vec![0.5]),
            make_inflow_model(1, 1, 160.0, 25.0, vec![0.5]),
        ];
        let par = PrecomputedPar::build(&models, &stages, &hydro_ids).unwrap();

        let transitions = vec![
            StageLagTransition {
                accumulate_weight: 0.7,
                spillover_weight: 0.3,
                finalize_period: true,
                accumulate_downstream: false,
                downstream_accumulate_weight: 0.0,
                downstream_spillover_weight: 0.0,
                downstream_finalize: false,
                rebuild_from_downstream: false,
            },
            StageLagTransition {
                accumulate_weight: 1.0,
                spillover_weight: 0.0,
                finalize_period: true,
                accumulate_downstream: false,
                downstream_accumulate_weight: 0.0,
                downstream_spillover_weight: 0.0,
                downstream_finalize: false,
                rebuild_from_downstream: false,
            },
        ];

        let mut lib = ExternalScenarioLibrary::new(2, 1, 1, "inflow", vec![1, 1]);
        let rows = vec![
            ExternalScenarioRow {
                stage_id: 0,
                scenario_id: 0,
                hydro_id,
                value_m3s: 150.0,
            },
            ExternalScenarioRow {
                stage_id: 1,
                scenario_id: 0,
                hydro_id,
                value_m3s: 130.0,
            },
        ];

        let past_inflows = vec![HydroPastInflows {
            hydro_id,
            values_m3s: vec![110.0],
        }];

        standardize_external_inflow(
            &mut lib,
            &rows,
            &hydro_ids,
            &stages,
            &par,
            &past_inflows,
            &transitions,
        );

        let det_base = par.deterministic_base(0, 0);
        let psi = par.psi_slice(0, 0)[0];
        let sigma = par.sigma(0, 0);

        // Stage 0: lag-1 = 110.0 (frozen from past_inflows before any finalize).
        let expected_eta_0 = (150.0 - det_base - psi * 110.0) / sigma;
        let eta_0 = lib.eta_slice(0, 0)[0];
        assert!(
            (eta_0 - expected_eta_0).abs() < 1e-10,
            "eta[stage=0] = {eta_0}, expected {expected_eta_0}"
        );

        // Stage 1: lag-1 = 150.0 (shifted in at stage 0 finalize: avg = 150*0.7/0.7 = 150.0).
        // The spillover seeds the next accumulator with 150.0*0.3=45.0, weight=0.3.
        // Stage 1 then adds 130.0*1.0=130.0 → accum=175.0, weight=1.3 (finalized at end).
        // But the lag used for eta is the one shifted AT stage 0, which is 150.0.
        let expected_eta_1 = (130.0 - det_base - psi * 150.0) / sigma;
        let eta_1 = lib.eta_slice(1, 0)[0];
        assert!(
            (eta_1 - expected_eta_1).abs() < 1e-10,
            "eta[stage=1] = {eta_1}, expected {expected_eta_1} (lag = 150.0 from spillover period)"
        );
    }

    // -----------------------------------------------------------------------
    // Load standardization tests
    // -----------------------------------------------------------------------

    /// 1 bus, 1 stage, 1 scenario. `value_mw`=240, mean=200, std=40 → eta=1.0.
    #[test]
    fn test_load_standardization() {
        let bus_id = EntityId(3);
        let bus_ids = vec![bus_id];

        let load_models = vec![LoadModel {
            bus_id,
            stage_id: 0,
            mean_mw: 200.0,
            std_mw: 40.0,
        }];

        let mut lib = ExternalScenarioLibrary::new(1, 1, 1, "load", vec![1]);
        let rows = vec![ExternalLoadRow {
            stage_id: 0,
            scenario_id: 0,
            bus_id,
            value_mw: 240.0,
        }];
        standardize_external_load(&mut lib, &rows, &bus_ids, &load_models, 1);

        let eta = lib.eta_slice(0, 0)[0];
        assert!((eta - 1.0).abs() < 1e-10, "eta = {eta}");
    }

    // -----------------------------------------------------------------------
    // NCS standardization tests
    // -----------------------------------------------------------------------

    /// 1 NCS, 1 stage, 1 scenario. value=0.7, mean=0.5, std=0.2 → eta=1.0.
    #[test]
    fn test_ncs_standardization() {
        let ncs_id = EntityId(7);
        let ncs_ids = vec![ncs_id];

        let ncs_models = vec![NcsModel {
            ncs_id,
            stage_id: 0,
            mean: 0.5,
            std: 0.2,
        }];

        let mut lib = ExternalScenarioLibrary::new(1, 1, 1, "ncs", vec![1]);
        let rows = vec![ExternalNcsRow {
            stage_id: 0,
            scenario_id: 0,
            ncs_id,
            value: 0.7,
        }];
        standardize_external_ncs(&mut lib, &rows, &ncs_ids, &ncs_models, 1);

        let eta = lib.eta_slice(0, 0)[0];
        assert!((eta - 1.0).abs() < 1e-10, "eta = {eta}");
    }

    // -----------------------------------------------------------------------
    // std=0 guard test
    // -----------------------------------------------------------------------

    /// When `std_mw`=0.0, eta must be 0.0 (not NaN or infinity).
    #[test]
    fn test_std_zero_returns_zero() {
        let bus_id = EntityId(5);
        let bus_ids = vec![bus_id];

        let load_models = vec![LoadModel {
            bus_id,
            stage_id: 0,
            mean_mw: 0.5,
            std_mw: 0.0,
        }];

        let mut lib = ExternalScenarioLibrary::new(1, 1, 1, "load", vec![1]);
        let rows = vec![ExternalLoadRow {
            stage_id: 0,
            scenario_id: 0,
            bus_id,
            value_mw: 0.5,
        }];
        standardize_external_load(&mut lib, &rows, &bus_ids, &load_models, 1);

        let eta = lib.eta_slice(0, 0)[0];
        assert_eq!(eta, 0.0, "eta must be 0.0 when std=0.0, got {eta}");
    }

    #[test]
    fn test_new_allocates_correct_sizes() {
        let lib = ExternalScenarioLibrary::new(12, 50, 5, "inflow", vec![50usize; 12]);
        assert_eq!(lib.n_stages(), 12);
        assert_eq!(lib.n_scenarios(), 50);
        assert_eq!(lib.n_entities(), 5);
        // Verify each accessor slice has the correct length.
        assert_eq!(lib.eta_slice(0, 0).len(), 5);
        assert_eq!(lib.eta_slice(11, 49).len(), 5);
    }

    #[test]
    fn test_eta_roundtrip() {
        let mut lib = ExternalScenarioLibrary::new(3, 2, 4, "load", vec![2, 2, 2]);
        let written = [1.0_f64, 2.0, 3.0, 4.0];
        lib.eta_slice_mut(1, 0).copy_from_slice(&written);
        assert_eq!(lib.eta_slice(1, 0), &written);
    }

    #[test]
    fn test_entity_class_metadata() {
        let lib = ExternalScenarioLibrary::new(1, 1, 1, "ncs", vec![1]);
        assert_eq!(lib.entity_class(), "ncs");

        let lib2 = ExternalScenarioLibrary::new(1, 1, 1, "inflow", vec![1]);
        assert_eq!(lib2.entity_class(), "inflow");
    }

    #[test]
    fn test_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ExternalScenarioLibrary>();
    }

    #[test]
    fn test_zero_initialized() {
        let lib = ExternalScenarioLibrary::new(2, 3, 4, "inflow", vec![3, 3]);
        for stage in 0..2 {
            for scenario in 0..3 {
                for &v in lib.eta_slice(stage, scenario) {
                    assert_eq!(v, 0.0_f64);
                }
            }
        }
    }

    #[test]
    fn test_eta_roundtrip_multiple_cells() {
        let mut lib = ExternalScenarioLibrary::new(3, 2, 4, "inflow", vec![2, 2, 2]);
        // Write to (0, 0)
        lib.eta_slice_mut(0, 0)
            .copy_from_slice(&[0.1, 0.2, 0.3, 0.4]);
        // Write to (2, 1)
        lib.eta_slice_mut(2, 1)
            .copy_from_slice(&[9.0, 8.0, 7.0, 6.0]);

        // Verify (0, 0) and (2, 1) are independent.
        assert_eq!(lib.eta_slice(0, 0), &[0.1, 0.2, 0.3, 0.4]);
        assert_eq!(lib.eta_slice(2, 1), &[9.0, 8.0, 7.0, 6.0]);
        // (1, 0) was not written and must still be zero.
        assert_eq!(lib.eta_slice(1, 0), &[0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_clone_is_independent() {
        let mut lib = ExternalScenarioLibrary::new(2, 2, 2, "ncs", vec![2, 2]);
        lib.eta_slice_mut(0, 0).copy_from_slice(&[1.0, 2.0]);

        let mut cloned = lib.clone();
        cloned.eta_slice_mut(0, 0).copy_from_slice(&[99.0, 99.0]);

        // Original must be unaffected.
        assert_eq!(lib.eta_slice(0, 0), &[1.0, 2.0]);
        assert_eq!(cloned.eta_slice(0, 0), &[99.0, 99.0]);
    }

    // -----------------------------------------------------------------------
    // validate_external_library tests
    // -----------------------------------------------------------------------

    use std::collections::HashSet;

    use super::validate_external_library;
    use crate::StochasticError;

    /// Build a valid `ExternalScenarioLibrary` with all-finite eta values.
    fn make_valid_library(
        n_stages: usize,
        n_scenarios: usize,
        n_entities: usize,
        class: &'static str,
    ) -> ExternalScenarioLibrary {
        let raw = vec![n_scenarios; n_stages];
        let mut lib = ExternalScenarioLibrary::new(n_stages, n_scenarios, n_entities, class, raw);
        // Fill with a known finite value so V3.7 passes.
        for stage in 0..n_stages {
            for scenario in 0..n_scenarios {
                for entity in 0..n_entities {
                    lib.eta_slice_mut(stage, scenario)[entity] = 0.5;
                }
            }
        }
        lib
    }

    /// Build a `HashSet` of `EntityId`s from a range of i32 values.
    fn entity_id_set(ids: impl IntoIterator<Item = i32>) -> HashSet<EntityId> {
        ids.into_iter().map(EntityId).collect()
    }

    /// Build a `rows_per_stage` vector where each stage has `n_scenarios * n_entities` rows.
    fn uniform_rows_per_stage(
        n_stages: usize,
        n_scenarios: usize,
        n_entities: usize,
    ) -> Vec<usize> {
        vec![n_scenarios * n_entities; n_stages]
    }

    /// Given a valid external library with 50 scenarios, 12 stages, 5 entities,
    /// all finite eta values, `validate_external_library` returns `Ok(())`.
    #[test]
    fn test_valid_library_passes() {
        let n_stages = 12;
        let n_scenarios = 50;
        let n_entities = 5;
        let lib = make_valid_library(n_stages, n_scenarios, n_entities, "inflow");
        let entity_ids: Vec<EntityId> = (1..=5).map(EntityId).collect();
        let row_entity_ids = entity_id_set(1..=5);
        let rows_per_stage = uniform_rows_per_stage(n_stages, n_scenarios, n_entities);

        let result = validate_external_library(
            &lib,
            &entity_ids,
            &row_entity_ids,
            &rows_per_stage,
            n_stages,
            50,
        );
        assert!(result.is_ok(), "expected Ok(()), got: {result:?}");
    }

    /// Given raw rows missing data for entity ID 7, `validate_external_library`
    /// returns `Err` with a message containing "V3.2" and "7".
    #[test]
    fn test_missing_entity_fails_v3_2() {
        let n_stages = 3;
        let n_scenarios = 10;
        let n_entities = 3;
        let lib = make_valid_library(n_stages, n_scenarios, n_entities, "inflow");
        // Entity IDs include 7, but row_entity_ids omits it.
        let entity_ids = vec![EntityId(5), EntityId(7), EntityId(9)];
        let row_entity_ids = entity_id_set([5, 9]); // 7 is missing
        let rows_per_stage = uniform_rows_per_stage(n_stages, n_scenarios, n_entities);

        let result = validate_external_library(
            &lib,
            &entity_ids,
            &row_entity_ids,
            &rows_per_stage,
            n_stages,
            10,
        );
        match result {
            Err(StochasticError::InsufficientData { context }) => {
                assert!(
                    context.contains("V3.2"),
                    "expected message to contain 'V3.2', got: {context}",
                );
                assert!(
                    context.contains('7'),
                    "expected message to contain entity ID '7', got: {context}",
                );
            }
            other => panic!("expected Err(InsufficientData), got: {other:?}"),
        }
    }

    /// Given raw rows where stage counts differ but are all exactly divisible by
    /// `n_entities`, `validate_external_library` now returns `Ok(())` because V3.4
    /// only enforces exact divisibility — non-uniform counts are accepted and
    /// handled by `pad_library_to_uniform`.
    #[test]
    fn test_nonuniform_divisible_counts_accepted_v3_4() {
        let n_stages = 3;
        let n_scenarios = 50;
        let n_entities = 2;
        let lib = make_valid_library(n_stages, n_scenarios, n_entities, "load");
        let entity_ids = vec![EntityId(1), EntityId(2)];
        let row_entity_ids = entity_id_set([1, 2]);
        // Stage 0: 50*2=100, Stage 1: 49*2=98 (non-uniform but divisible), Stage 2: 50*2=100.
        let rows_per_stage = vec![100usize, 98, 100];

        let result = validate_external_library(
            &lib,
            &entity_ids,
            &row_entity_ids,
            &rows_per_stage,
            n_stages,
            50,
        );
        assert!(
            result.is_ok(),
            "expected Ok(()) for non-uniform but divisible counts, got: {result:?}",
        );
    }

    /// Given a library where `eta_slice(3, 10)[2]` is `NaN`,
    /// `validate_external_library` returns `Err` with "V3.7".
    #[test]
    fn test_nan_eta_fails_v3_7() {
        let n_stages = 5;
        let n_scenarios = 20;
        let n_entities = 4;
        let mut lib = make_valid_library(n_stages, n_scenarios, n_entities, "ncs");
        // Inject NaN at stage=3, scenario=10, entity=2.
        lib.eta_slice_mut(3, 10)[2] = f64::NAN;

        let entity_ids: Vec<EntityId> = (1..=4).map(EntityId).collect();
        let row_entity_ids = entity_id_set(1..=4);
        let rows_per_stage = uniform_rows_per_stage(n_stages, n_scenarios, n_entities);

        let result = validate_external_library(
            &lib,
            &entity_ids,
            &row_entity_ids,
            &rows_per_stage,
            n_stages,
            20,
        );
        match result {
            Err(StochasticError::InsufficientData { context }) => {
                assert!(
                    context.contains("V3.7"),
                    "expected message to contain 'V3.7', got: {context}",
                );
            }
            other => panic!("expected Err(InsufficientData), got: {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Round-trip standardization tests
    // -----------------------------------------------------------------------

    /// Build the fixture for the weekly+monthly AR(1) round-trip test.
    ///
    /// Returns `(stages, par, stage_lag_transitions, targets, past_lag, hydro_ids)` for
    /// a 4-weekly + 1-monthly layout matching the PMO\_APR\_2026 excerpt in the design doc.
    ///
    /// Stage layout (`season_id=3` for April, `season_id=4` for May):
    /// - W1 `[2026-03-28, 2026-04-04)` — 3 April days
    /// - W2 `[2026-04-04, 2026-04-11)` — 7 April days
    /// - W3 `[2026-04-11, 2026-04-18)` — 7 April days
    /// - W4 `[2026-04-18, 2026-04-25)` — 7 April days, finalizes April
    /// - M2 `[2026-05-02, 2026-06-01)` — 30 May days, finalizes May
    ///
    /// `StageLagTransition` weights: April = 720 h, May = 744 h.
    /// `psi=[0.3]`, `mean=500`, `std=50`, past lag-1 = 450.
    #[allow(clippy::type_complexity)]
    fn make_round_trip_fixture() -> (
        Vec<Stage>,
        PrecomputedPar,
        Vec<StageLagTransition>,
        [[f64; 5]; 2],
        f64,
        Vec<EntityId>,
    ) {
        const N_STAGES: usize = 5;
        let hydro_ids = vec![EntityId(1)];

        // season_id=3 for April stages, season_id=4 for the May stage.
        let stages = vec![
            make_stage(0, 0, 3), // W1 — April
            make_stage(1, 1, 3), // W2 — April
            make_stage(2, 2, 3), // W3 — April
            make_stage(3, 3, 3), // W4 — April (finalizes April period)
            make_stage(4, 4, 4), // M2 — May
        ];

        // AR(1): mean=500, std=50, psi=[0.3], residual_std_ratio=1.0 → sigma=50.
        let models: Vec<_> = (0..i32::try_from(N_STAGES).unwrap())
            .map(|stage_id| make_inflow_model(1, stage_id, 500.0, 50.0, vec![0.3]))
            .collect();
        let par = PrecomputedPar::build(&models, &stages, &hydro_ids).unwrap();

        // StageLagTransition weights (hand-computed from date boundaries).
        // April = 30 days = 720 h.  W1 covers only 3 April days; W2/W3/W4 cover 7 each.
        // May = 31 days = 744 h. M2 covers 30 May days.
        let weight_w1 = 3.0 * 24.0 / 720.0;
        let weight_weekly = 7.0 * 24.0 / 720.0;
        let weight_may = 30.0 * 24.0 / 744.0;
        let stage_lag_transitions = vec![
            StageLagTransition {
                accumulate_weight: weight_w1,
                spillover_weight: 0.0,
                finalize_period: false,
                accumulate_downstream: false,
                downstream_accumulate_weight: 0.0,
                downstream_spillover_weight: 0.0,
                downstream_finalize: false,
                rebuild_from_downstream: false,
            },
            StageLagTransition {
                accumulate_weight: weight_weekly,
                spillover_weight: 0.0,
                finalize_period: false,
                accumulate_downstream: false,
                downstream_accumulate_weight: 0.0,
                downstream_spillover_weight: 0.0,
                downstream_finalize: false,
                rebuild_from_downstream: false,
            },
            StageLagTransition {
                accumulate_weight: weight_weekly,
                spillover_weight: 0.0,
                finalize_period: false,
                accumulate_downstream: false,
                downstream_accumulate_weight: 0.0,
                downstream_spillover_weight: 0.0,
                downstream_finalize: false,
                rebuild_from_downstream: false,
            },
            StageLagTransition {
                accumulate_weight: weight_weekly,
                spillover_weight: 0.0,
                finalize_period: true, // last April stage → finalize the April period
                accumulate_downstream: false,
                downstream_accumulate_weight: 0.0,
                downstream_spillover_weight: 0.0,
                downstream_finalize: false,
                rebuild_from_downstream: false,
            },
            StageLagTransition {
                accumulate_weight: weight_may,
                spillover_weight: 0.0,
                finalize_period: true, // only May stage → finalizes itself
                accumulate_downstream: false,
                downstream_accumulate_weight: 0.0,
                downstream_spillover_weight: 0.0,
                downstream_finalize: false,
                rebuild_from_downstream: false,
            },
        ];

        // External targets: 2 scenarios × 5 stages, 1 hydro.
        let targets = [
            [480.0_f64, 520.0, 490.0, 510.0, 530.0], // scenario 0
            [550.0_f64, 470.0, 500.0, 540.0, 460.0], // scenario 1
        ];

        // Past inflow lag-1 = 450.0 (December monthly average before the study).
        let past_lag = 450.0_f64;

        (
            stages,
            par,
            stage_lag_transitions,
            targets,
            past_lag,
            hydro_ids,
        )
    }

    /// Round-trip consistency: [`standardize_external_inflow`] followed by
    /// [`evaluate_par`] must reconstruct the original external targets for a
    /// mixed 4-weekly + 1-monthly layout with AR(1) lags.
    ///
    /// The lag state used during standardization (frozen within each lag period,
    /// advanced by weighted average at period boundaries) is replicated in the
    /// reconstruction loop. Any divergence between the two paths would cause the
    /// assertion to fail.
    ///
    /// See `make_round_trip_fixture` for the full stage layout and parameter set.
    #[test]
    fn test_round_trip_weekly_monthly_ar1() {
        let (stages, par, stage_lag_transitions, targets, past_lag, hydro_ids) =
            make_round_trip_fixture();
        let hydro_id = hydro_ids[0];
        let n_stages = stages.len();
        let n_scenarios = targets.len();

        // Build ExternalScenarioRow entries from the targets array.
        let mut rows = Vec::with_capacity(n_stages * n_scenarios);
        for (scenario, scenario_targets) in targets.iter().enumerate() {
            for (stage, &value) in scenario_targets.iter().enumerate() {
                rows.push(ExternalScenarioRow {
                    stage_id: i32::try_from(stage).unwrap(),
                    scenario_id: i32::try_from(scenario).unwrap(),
                    hydro_id,
                    value_m3s: value,
                });
            }
        }

        let past_inflows = vec![HydroPastInflows {
            hydro_id,
            values_m3s: vec![past_lag],
        }];

        // Standardize: compute eta values for all (stage, scenario) pairs.
        let raw = vec![n_scenarios; n_stages];
        let mut lib = ExternalScenarioLibrary::new(n_stages, n_scenarios, 1, "inflow", raw);
        standardize_external_inflow(
            &mut lib,
            &rows,
            &hydro_ids,
            &stages,
            &par,
            &past_inflows,
            &stage_lag_transitions,
        );

        // Forward reconstruction: mirror the frozen-lag + accumulation logic from
        // `standardize_external_inflow` and assert that `evaluate_par` reproduces
        // the original target within 1e-10 at every (stage, scenario).
        for (scenario, scenario_targets) in targets.iter().enumerate() {
            let mut lag_buf = vec![past_lag]; // lag-1 initialized from past_inflows
            let mut accum = 0.0_f64;
            let mut weight_accum = 0.0_f64;

            for (t, (&target, slt)) in scenario_targets
                .iter()
                .zip(&stage_lag_transitions)
                .enumerate()
            {
                let eta = lib.eta_slice(t, scenario)[0];
                let det_base = par.deterministic_base(t, 0);
                let psi = par.psi_slice(t, 0);
                let order = par.order(0);
                let sigma = par.sigma(t, 0);

                // evaluate_par with the frozen lag state must reproduce the target.
                let reconstructed = evaluate_par(det_base, psi, order, &lag_buf, sigma, eta);
                assert!(
                    (reconstructed - target).abs() < 1e-10,
                    "stage={t}, scenario={scenario}: reconstructed={reconstructed:.15}, \
                     target={target:.15}, diff={:.2e}",
                    (reconstructed - target).abs()
                );

                // Accumulate this stage's contribution to the lag period average.
                accum += target * slt.accumulate_weight;
                weight_accum += slt.accumulate_weight;

                // At a period boundary: shift lag state, reset accumulators.
                if slt.finalize_period && weight_accum > 0.0 {
                    lag_buf[0] = accum / weight_accum;
                    accum = 0.0;
                    weight_accum = 0.0;
                }
                // Non-finalizing stages: lag_buf stays frozen (unchanged).
            }
        }
    }

    /// Given `library.n_scenarios() = 10` and `forward_passes = 50`,
    /// `validate_external_library` returns `Ok(())` (the V3.8 warning is emitted
    /// via tracing but does not abort construction).
    #[test]
    fn test_scenario_count_warning_returns_ok() {
        let n_stages = 2;
        let n_scenarios = 10;
        let n_entities = 2;
        let lib = make_valid_library(n_stages, n_scenarios, n_entities, "inflow");
        let entity_ids = vec![EntityId(1), EntityId(2)];
        let row_entity_ids = entity_id_set([1, 2]);
        let rows_per_stage = uniform_rows_per_stage(n_stages, n_scenarios, n_entities);

        // 10 scenarios < 50 forward passes — must warn but not error.
        let result = validate_external_library(
            &lib,
            &entity_ids,
            &row_entity_ids,
            &rows_per_stage,
            n_stages,
            50,
        );
        assert!(
            result.is_ok(),
            "V3.8 warning path must return Ok(()), got: {result:?}",
        );
    }

    // -----------------------------------------------------------------------
    // ticket-010: relaxed V3.4, raw_scenarios_per_stage, pad_library_to_uniform
    // -----------------------------------------------------------------------

    use super::pad_library_to_uniform;

    /// V3.4 accepts non-uniform scenario counts as long as every stage is
    /// exactly divisible by `n_entities` (`rows_per_stage` = [2,2,2,2,100] with
    /// `n_entities=2` → scenario counts [1,1,1,1,50]).
    #[test]
    fn test_v34_accepts_nonuniform_scenario_counts() {
        // 5 stages: 4 with 1 scenario (2 rows each) and 1 with 50 scenarios (100 rows).
        let n_entities = 2;
        let n_stages = 5;
        // Library must be big enough to hold V3.7 pass (50 scenarios max).
        let lib = make_valid_library(n_stages, 50, n_entities, "inflow");
        let entity_ids = vec![EntityId(1), EntityId(2)];
        let row_entity_ids = entity_id_set([1, 2]);
        let rows_per_stage = vec![2usize, 2, 2, 2, 100];

        let result = validate_external_library(
            &lib,
            &entity_ids,
            &row_entity_ids,
            &rows_per_stage,
            n_stages,
            50,
        );
        assert!(
            result.is_ok(),
            "V3.4 must accept non-uniform but divisible counts, got: {result:?}",
        );
    }

    /// V3.4 still rejects `rows_per_stage` where any stage has a row count not
    /// exactly divisible by `n_entities`.
    #[test]
    fn test_v34_still_rejects_indivisible_rows() {
        let n_entities = 2;
        let n_stages = 2;
        let lib = make_valid_library(n_stages, 2, n_entities, "inflow");
        let entity_ids = vec![EntityId(1), EntityId(2)];
        let row_entity_ids = entity_id_set([1, 2]);
        // Stage 0: 3 rows (not divisible by 2), Stage 1: 2 rows (ok).
        let rows_per_stage = vec![3usize, 2];

        let result = validate_external_library(
            &lib,
            &entity_ids,
            &row_entity_ids,
            &rows_per_stage,
            n_stages,
            1,
        );
        match result {
            Err(StochasticError::InsufficientData { context }) => {
                assert!(
                    context.contains("V3.4"),
                    "expected error to contain 'V3.4', got: {context}",
                );
            }
            other => panic!("expected Err(InsufficientData) with V3.4, got: {other:?}"),
        }
    }

    /// When all stages have the same scenario count (uniform), `raw_scenarios_per_stage`
    /// must equal `n_scenarios` for every entry.
    #[test]
    fn test_raw_scenarios_per_stage_uniform() {
        let n_stages = 4;
        let n_scenarios = 10;
        let raw = vec![n_scenarios; n_stages];
        let lib = ExternalScenarioLibrary::new(n_stages, n_scenarios, 2, "inflow", raw);
        assert_eq!(lib.raw_scenarios_per_stage(), &[10, 10, 10, 10]);
    }

    /// When the library is created with non-uniform raw counts, `raw_scenarios_per_stage`
    /// returns exactly what was passed in.
    #[test]
    fn test_raw_scenarios_per_stage_nonuniform() {
        let n_stages = 3;
        let n_scenarios = 50; // max (padded-to) count
        let raw = vec![1usize, 1, 50];
        let lib = ExternalScenarioLibrary::new(n_stages, n_scenarios, 1, "inflow", raw);
        assert_eq!(lib.raw_scenarios_per_stage(), &[1, 1, 50]);
        assert_eq!(lib.n_scenarios(), 50);
    }

    /// `pad_library_to_uniform` replicates stage 0's single eta value into
    /// all `n_scenarios` slots so that `eta_slice(0, k)` is identical for all k.
    #[test]
    fn test_pad_library_replicates_eta() {
        // 2 stages, 1 entity, raw counts [1, 3], padded to n_scenarios=3.
        let raw = vec![1usize, 3];
        let mut lib = ExternalScenarioLibrary::new(2, 3, 1, "inflow", raw);

        // Write known values only to the raw slots.
        // Stage 0 raw slot: scenario 0
        lib.eta_slice_mut(0, 0).copy_from_slice(&[7.0]);
        // Stage 1 raw slots: scenarios 0..3
        lib.eta_slice_mut(1, 0).copy_from_slice(&[1.0]);
        lib.eta_slice_mut(1, 1).copy_from_slice(&[2.0]);
        lib.eta_slice_mut(1, 2).copy_from_slice(&[3.0]);

        pad_library_to_uniform(&mut lib);

        // Stage 0: all three scenario slots must equal the single raw value.
        assert_eq!(lib.eta_slice(0, 0), &[7.0], "stage 0 scenario 0");
        assert_eq!(lib.eta_slice(0, 1), &[7.0], "stage 0 scenario 1 (padded)");
        assert_eq!(lib.eta_slice(0, 2), &[7.0], "stage 0 scenario 2 (padded)");

        // Stage 1: unchanged (already had 3 raw scenarios == n_scenarios).
        assert_eq!(lib.eta_slice(1, 0), &[1.0], "stage 1 scenario 0");
        assert_eq!(lib.eta_slice(1, 1), &[2.0], "stage 1 scenario 1");
        assert_eq!(lib.eta_slice(1, 2), &[3.0], "stage 1 scenario 2");
    }

    /// `pad_library_to_uniform` is a no-op when all stages already have the
    /// maximum scenario count — eta values must not change.
    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn test_pad_library_noop_when_uniform() {
        let n_stages = 3;
        let n_scenarios = 5;
        let raw = vec![n_scenarios; n_stages];
        let mut lib = ExternalScenarioLibrary::new(n_stages, n_scenarios, 2, "load", raw);

        // Fill with recognizable values.
        for s in 0..n_stages {
            for k in 0..n_scenarios {
                lib.eta_slice_mut(s, k)
                    .copy_from_slice(&[s as f64, k as f64]);
            }
        }

        pad_library_to_uniform(&mut lib);

        // Values must be identical to what was written.
        for s in 0..n_stages {
            for k in 0..n_scenarios {
                assert_eq!(
                    lib.eta_slice(s, k),
                    &[s as f64, k as f64],
                    "stage {s} scenario {k} must be unchanged",
                );
            }
        }
    }
}
