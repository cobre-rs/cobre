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
    EntityId, HydroPastInflows,
    scenario::{ExternalLoadRow, ExternalNcsRow, ExternalScenarioRow, LoadModel, NcsModel},
    temporal::Stage,
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
/// let mut lib = ExternalScenarioLibrary::new(12, 50, 5, "inflow");
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
    /// Number of scenarios per stage (uniform across all stages — V3.4).
    n_scenarios: usize,
    /// Number of entities in the eta vector width.
    n_entities: usize,
    /// Entity class label for diagnostic messages (e.g., `"inflow"`, `"load"`, `"ncs"`).
    entity_class: &'static str,
}

impl ExternalScenarioLibrary {
    /// Construct a new library with zero-filled buffers.
    ///
    /// # Parameters
    ///
    /// - `n_stages` — number of study stages
    /// - `n_scenarios` — number of scenarios per stage (uniform across all stages)
    /// - `n_entities` — number of entities in the eta vector (e.g., hydros, buses, NCS units)
    /// - `entity_class` — label for diagnostic messages (e.g., `"inflow"`, `"load"`, `"ncs"`)
    #[must_use]
    pub fn new(
        n_stages: usize,
        n_scenarios: usize,
        n_entities: usize,
        entity_class: &'static str,
    ) -> Self {
        Self {
            eta: vec![0.0_f64; n_stages * n_scenarios * n_entities].into_boxed_slice(),
            n_stages,
            n_scenarios,
            n_entities,
            entity_class,
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
/// ## Lag initialization
///
/// Within a single scenario, lags at stage `t` are the raw external values at
/// stages `t-1, t-2, ..., t-order` from the same scenario (marching forward
/// sequentially). For stage 0, there are no pre-study external values, so lags
/// come from `past_inflows` (the initial conditions). `past_inflows[i].values_m3s[0]`
/// is lag-1 (most recent), `past_inflows[i].values_m3s[1]` is lag-2, etc.
/// The same `past_inflows` values are used for all scenarios at stage 0.
/// If a hydro has no entry in `past_inflows`, the lag defaults to `0.0`.
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
///
/// # Panics
///
/// Panics in debug builds if dimension mismatches are detected.
pub fn standardize_external_inflow(
    library: &mut ExternalScenarioLibrary,
    external_rows: &[ExternalScenarioRow],
    hydro_ids: &[EntityId],
    stages: &[Stage],
    par: &PrecomputedPar,
    past_inflows: &[HydroPastInflows],
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
    // Pre-allocate a reusable lag buffer. Reused across all iterations.
    // -----------------------------------------------------------------------
    let mut lag_buf = vec![0.0_f64; safe_max_order];

    // -----------------------------------------------------------------------
    // March forward through (scenario, stage, hydro) and compute eta.
    // -----------------------------------------------------------------------
    for scenario in 0..n_scenarios {
        for t in 0..n_stages {
            for h in 0..n_hydros {
                let target = raw_values[t * n_scenarios * n_hydros + scenario * n_hydros + h];

                let order_h = par.order(h);

                // Build lag buffer: lag_buf[l] = raw value at stage (t - l - 1)
                // for the same scenario. When t <= l (no pre-study external data
                // available at that lag depth), use past_inflows values instead.
                for (l, slot) in lag_buf.iter_mut().enumerate().take(order_h) {
                    *slot = if t > l {
                        // Use raw external value from the same scenario.
                        let lag_stage = t - l - 1;
                        raw_values[lag_stage * n_scenarios * n_hydros + scenario * n_hydros + h]
                    } else {
                        // No pre-study external data: use past_inflows for this hydro.
                        // past_inflows[h].values_m3s[lag] where lag = l - t (adjusted
                        // for how many study stages have elapsed).
                        let past_lag = l - t;
                        past_lag_buf[h * safe_max_order + past_lag]
                    };
                }

                let det_base = par.deterministic_base(t, h);
                let psi = par.psi_slice(t, h);
                let sigma = par.sigma(t, h);

                let eta = solve_par_noise(det_base, psi, order_h, &lag_buf, sigma, target);

                library.eta_slice_mut(t, scenario)[h] = eta;
            }
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
/// let lib = ExternalScenarioLibrary::new(3, 50, 2, "inflow");
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
    // V3.4 — Consistent scenario count: rows_per_stage[s] / n_entities must
    // be the same for every stage, and rows must be exactly divisible by
    // n_entities (no partial rows).
    //
    // Guard against zero entities to avoid division by zero; this situation
    // is benign (empty library) so we skip the check.
    // -----------------------------------------------------------------------
    if n_entities > 0 && n_stages > 0 {
        // Check exact divisibility for every stage first.
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
        let first_count = rows_per_stage[0] / n_entities;
        for (stage_idx, &count) in rows_per_stage.iter().enumerate().take(n_stages).skip(1) {
            let stage_count = count / n_entities;
            if stage_count != first_count {
                return Err(StochasticError::InsufficientData {
                    context: format!(
                        "V3.4: external {class} library has inconsistent scenario counts: \
                         stage 0 has {first_count} scenarios but stage {stage_idx} has \
                         {stage_count} scenarios; all stages must have the same count",
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
        EntityId, HydroPastInflows,
        scenario::{
            ExternalLoadRow, ExternalNcsRow, ExternalScenarioRow, InflowModel, LoadModel, NcsModel,
        },
        temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
            StageStateConfig,
        },
    };

    use super::{
        ExternalScenarioLibrary, standardize_external_inflow, standardize_external_load,
        standardize_external_ncs,
    };
    use crate::par::precompute::PrecomputedPar;

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
        let mut lib = ExternalScenarioLibrary::new(2, 1, 1, "inflow");

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
        standardize_external_inflow(&mut lib, &rows, &hydro_ids, &stages, &par, &[]);

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

        let mut lib = ExternalScenarioLibrary::new(2, 1, 1, "inflow");
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
        standardize_external_inflow(&mut lib, &rows, &hydro_ids, &stages, &par, &past_inflows);

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

        let mut lib = ExternalScenarioLibrary::new(1, 1, 1, "load");
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

        let mut lib = ExternalScenarioLibrary::new(1, 1, 1, "ncs");
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

        let mut lib = ExternalScenarioLibrary::new(1, 1, 1, "load");
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
        let lib = ExternalScenarioLibrary::new(12, 50, 5, "inflow");
        assert_eq!(lib.n_stages(), 12);
        assert_eq!(lib.n_scenarios(), 50);
        assert_eq!(lib.n_entities(), 5);
        // Verify each accessor slice has the correct length.
        assert_eq!(lib.eta_slice(0, 0).len(), 5);
        assert_eq!(lib.eta_slice(11, 49).len(), 5);
    }

    #[test]
    fn test_eta_roundtrip() {
        let mut lib = ExternalScenarioLibrary::new(3, 2, 4, "load");
        let written = [1.0_f64, 2.0, 3.0, 4.0];
        lib.eta_slice_mut(1, 0).copy_from_slice(&written);
        assert_eq!(lib.eta_slice(1, 0), &written);
    }

    #[test]
    fn test_entity_class_metadata() {
        let lib = ExternalScenarioLibrary::new(1, 1, 1, "ncs");
        assert_eq!(lib.entity_class(), "ncs");

        let lib2 = ExternalScenarioLibrary::new(1, 1, 1, "inflow");
        assert_eq!(lib2.entity_class(), "inflow");
    }

    #[test]
    fn test_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ExternalScenarioLibrary>();
    }

    #[test]
    fn test_zero_initialized() {
        let lib = ExternalScenarioLibrary::new(2, 3, 4, "inflow");
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
        let mut lib = ExternalScenarioLibrary::new(3, 2, 4, "inflow");
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
        let mut lib = ExternalScenarioLibrary::new(2, 2, 2, "ncs");
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
        let mut lib = ExternalScenarioLibrary::new(n_stages, n_scenarios, n_entities, class);
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

    /// Given raw rows with 50 scenarios for stage 0 but 49 for stage 1,
    /// `validate_external_library` returns `Err` with "V3.4" and the counts.
    #[test]
    fn test_inconsistent_scenario_count_fails_v3_4() {
        let n_stages = 3;
        let n_scenarios = 50;
        let n_entities = 2;
        let lib = make_valid_library(n_stages, n_scenarios, n_entities, "load");
        let entity_ids = vec![EntityId(1), EntityId(2)];
        let row_entity_ids = entity_id_set([1, 2]);
        // Stage 0: 50*2=100, Stage 1: 49*2=98 (inconsistent), Stage 2: 50*2=100.
        let rows_per_stage = vec![100usize, 98, 100];

        let result = validate_external_library(
            &lib,
            &entity_ids,
            &row_entity_ids,
            &rows_per_stage,
            n_stages,
            50,
        );
        match result {
            Err(StochasticError::InsufficientData { context }) => {
                assert!(
                    context.contains("V3.4"),
                    "expected message to contain 'V3.4', got: {context}",
                );
                // The message must mention both the expected count (50) and the
                // observed count (49) for the inconsistent stage.
                assert!(
                    context.contains("50") && context.contains("49"),
                    "expected message to contain counts '50' and '49', got: {context}",
                );
            }
            other => panic!("expected Err(InsufficientData), got: {other:?}"),
        }
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
}
