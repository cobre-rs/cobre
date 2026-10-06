//! Pre-computation of PAR model coefficient arrays for LP RHS patching.
//!
//! [`PrecomputedPar`] is the cache built once from raw [`InflowModel`]
//! parameters and consumed read-only during optimization.
//!
//! ## Array layout
//!
//! Two-dimensional arrays are **stage-major** `array[stage * n_series + series_element]`;
//! the three-dimensional `psi` is **stage-major, series-minor, lag-innermost**
//! `psi[stage * n_series * max_order + series_element * max_order + lag]`. This keeps
//! a scenario trajectory's per-stage data contiguous for sequential per-stage traversal.
//!
//! ## Coefficient conversion
//!
//! `ar_coefficients` are stored standardized (ψ\*, the Yule-Walker output) and
//! converted to original units at build time:
//!
//! ```text
//! ψ_{m,ℓ} = ψ*_{m,ℓ} · s_m / s_{m-ℓ}
//! ```
//!
//! where `s_m` is `std_m3s` for the current season and `s_{m-ℓ}` for the season
//! `ℓ` stages prior.
//!
//! ## PAR(p)-A annual component
//!
//! When any [`InflowModel`] in the input set has `annual: Some(_)`, the
//! materialized `psi` stride is widened to 12 (one slot per month of the annual
//! lag polynomial) regardless of the classical AR order. The effective
//! length-12 polynomial is:
//!
//! ```text
//! ψ̂ = ψ · s_m / σ^A          (annual unit conversion; σ^A = annual-average std)
//! φ̂_j = φ_j · s_m / s_{m-j}  (classical AR lag conversion)
//!
//! psi[stage,h,j] = φ̂_{j+1} + ψ̂/12   for j ∈ [0, ar_order)
//! psi[stage,h,j] =           ψ̂/12   for j ∈ [ar_order, 12)
//!
//! deterministic_base = μ_m − Σ_{j=0..11} psi[stage,h,j] · μ_{m-j-1}
//! ```
//!
//! Hydros with `annual: None` inside a study where some hydro has `Some` use
//! `ψ̂ = 0`, preserving classical-PAR behavior on the wider stride. When every
//! model has `annual: None`, the classical materialization is preserved bit-for-bit.

use std::collections::{HashMap, HashSet};

use cobre_core::{
    EntityId,
    scenario::InflowModel,
    temporal::{SeasonCycles, SeasonMap, Stage},
};

use crate::{StochasticError, season_cast::StitchedSeasonMap};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Resolve a stage ID to a season ID by modular arithmetic on the cycle length.
///
/// `season_offset` is the season ID of stage 0 (e.g. 2 for a March-start study),
/// so that negative pre-study `stage_id`s map to the correct season rather than
/// assuming stage 0 ≡ season 0 (true only for January-start studies).
fn resolve_season_id(stage_id: i32, n_seasons: usize, season_offset: usize) -> usize {
    debug_assert!(n_seasons > 0, "n_seasons must be positive");
    // n_seasons ≤ ~52 and the debug_assert excludes 0, so these casts neither
    // wrap nor truncate.
    #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
    let n = n_seasons as i32;
    #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
    let offset = season_offset as i32;
    #[allow(clippy::cast_sign_loss)]
    let result = (((stage_id + offset) % n + n) % n) as usize;
    result
}

fn laid_out_stages(stages: &[Stage]) -> impl Iterator<Item = &Stage> {
    stages.iter().filter(|s| s.id >= 0)
}

fn cross_resolution_lag_season(
    cycles: &SeasonCycles,
    stage_season: usize,
    lag_stage_season: usize,
    steps: usize,
) -> Option<usize> {
    let level = cycles.group_of(stage_season)?;
    if cycles.group_of(lag_stage_season)? == level {
        return None;
    }
    (0..steps).try_fold(stage_season, |season, _| cycles.predecessor(season))
}

// ---------------------------------------------------------------------------
// PrecomputedPar
// ---------------------------------------------------------------------------

/// Cache-friendly PAR(p) model data for LP RHS patching.
///
/// The series-element dimension is ordered by canonical entity ID (the
/// declaration-order-invariant index). See the [module documentation](self) for
/// the array layout and coefficient-conversion formulas.
///
/// # Examples
///
/// ```
/// use cobre_core::{EntityId, scenario::InflowModel, temporal::{Stage, Block, BlockMode, StageStateConfig, StageRiskConfig, ScenarioSourceConfig, NoiseMethod}};
/// use cobre_stochastic::par::precompute::PrecomputedPar;
/// use chrono::NaiveDate;
///
/// let date = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
/// let stage = Stage {
///     index: 0,
///     id: 0,
///     start_date: date,
///     end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
///     season_id: Some(0),
///     blocks: vec![Block { index: 0, name: "SINGLE".to_string(), duration_hours: 744.0 }],
///     block_mode: BlockMode::Parallel,
///     state_config: StageStateConfig { storage: true, inflow_lags: false },
///     risk_config: StageRiskConfig::Expectation,
///     scenario_config: ScenarioSourceConfig { branching_factor: 10, noise_method: NoiseMethod::Saa },
/// };
///
/// let model = InflowModel {
///     hydro_id: EntityId(1),
///     stage_id: 0,
///     mean_m3s: 100.0,
///     std_m3s: 30.0,
///     ar_coefficients: vec![],
///     residual_std_ratio: 1.0,
///     annual: None,
/// };
///
/// let lp = PrecomputedPar::build(&[model], &[stage], &[EntityId(1)], None).unwrap();
/// assert_eq!(lp.n_stages(), 1);
/// assert_eq!(lp.n_hydros(), 1);
/// assert!((lp.deterministic_base(0, 0) - 100.0).abs() < f64::EPSILON);
/// ```
#[derive(Debug)]
pub struct PrecomputedPar {
    /// Deterministic base `b_{h,m(t)} = μ_{m(t)} - Σ_ℓ ψ_{m(t),ℓ} · μ_{m(t-ℓ)}`,
    /// stage-major.
    deterministic_base: Box<[f64]>,

    /// Residual standard deviation `σ = s_m · residual_std_ratio`, stage-major.
    sigma: Box<[f64]>,

    /// AR lag coefficients `ψ_{m(t),ℓ}` in original units, zero-padded for series
    /// with `ar_order < max_order`.
    psi: Box<[f64]>,

    /// Classical AR order per series element (count of meaningful lags in `psi`).
    orders: Box<[usize]>,

    /// Per series element, whether any [`InflowModel`] carries `annual: Some(_)`.
    /// Gates whether lag slots beyond `orders[h]` carry `ψ̂/12` (see
    /// [`Self::effective_lag_count`]).
    has_annual: Box<[bool]>,

    /// Number of study stages.
    n_stages: usize,

    /// Number of series elements (entities tracked by the PAR model).
    n_hydros: usize,

    /// Maximum AR order across all hydros and stages.
    max_order: usize,
}

impl PrecomputedPar {
    // -----------------------------------------------------------------------
    // Builder
    // -----------------------------------------------------------------------

    /// Build a [`PrecomputedPar`] from raw PAR model parameters.
    ///
    /// # Parameters
    ///
    /// - `inflow_models`: raw PAR parameters sorted by `(hydro_id, stage_id)`
    ///   from the system. May include pre-study stage models (negative `stage_id`)
    ///   used for lag initialization.
    /// - `stages`: the system's stages. Those with `id >= 0` are laid out in the
    ///   order given; those with `id < 0` (declared pre-study stages) are not laid
    ///   out and only supply season statistics. A lag stage with no model of its
    ///   own takes the statistics of the lowest-id stage in `stages` that carries
    ///   its season.
    /// - `hydro_ids`: canonical sorted entity IDs (determines series element array index order).
    /// - `season_map`: resolves a lag stage's season through [`StitchedSeasonMap`].
    ///   `None` falls back to modular arithmetic on the distinct study `season_id`
    ///   count, correct only for complete, consecutively numbered cycles.
    ///
    /// # Errors
    ///
    /// Returns [`StochasticError::InvalidParParameters`] when a required lag stage
    /// does not have a `season_id`, which prevents coefficient unit conversion.
    pub fn build(
        inflow_models: &[InflowModel],
        stages: &[Stage],
        hydro_ids: &[EntityId],
        season_map: Option<&SeasonMap>,
    ) -> Result<Self, StochasticError> {
        let n_stages = laid_out_stages(stages).count();
        let n_hydros = hydro_ids.len();

        let hydro_index: HashMap<EntityId, usize> = hydro_ids
            .iter()
            .enumerate()
            .map(|(i, &id)| (id, i))
            .collect();

        let model_map: HashMap<(i32, i32), &InflowModel> = inflow_models
            .iter()
            .map(|m| ((m.hydro_id.0, m.stage_id), m))
            .collect();

        let classical_max_order = inflow_models
            .iter()
            .map(InflowModel::ar_order)
            .max()
            .unwrap_or(0);

        // When any model carries an annual component, widen the psi stride to
        // 12 so that the ψ̂/12 term fills all 12 monthly lag positions. The
        // classical AR order may already exceed 12 in unusual configurations;
        // take the larger of the two.
        let any_annual = inflow_models.iter().any(|m| m.annual.is_some());
        let max_order = if any_annual {
            12_usize.max(classical_max_order)
        } else {
            classical_max_order
        };

        // Per-hydro AR order (maximum across all stages for that hydro).
        // This reports the original AR order p, not the widened stride.
        // Also record which hydros have at least one stage with an annual component.
        let mut orders = vec![0usize; n_hydros];
        let mut has_annual = vec![false; n_hydros];
        for model in inflow_models {
            if let Some(&h_idx) = hydro_index.get(&model.hydro_id) {
                if model.ar_order() > orders[h_idx] {
                    orders[h_idx] = model.ar_order();
                }
                if model.annual.is_some() {
                    has_annual[h_idx] = true;
                }
            }
        }

        let n2 = n_stages * n_hydros;
        let n3 = n_stages * n_hydros * max_order;
        let mut deterministic_base = vec![0.0f64; n2];
        let mut sigma = vec![0.0f64; n2];
        let mut psi = vec![0.0f64; n3];

        let mut bufs = StageArrayBuffers {
            deterministic_base: &mut deterministic_base,
            sigma: &mut sigma,
            psi: &mut psi,
            n_hydros,
            max_order,
        };
        fill_stage_arrays(
            stages,
            hydro_ids,
            &hydro_index,
            &model_map,
            inflow_models,
            season_map,
            &mut bufs,
        )?;

        Ok(Self {
            deterministic_base: deterministic_base.into_boxed_slice(),
            sigma: sigma.into_boxed_slice(),
            psi: psi.into_boxed_slice(),
            orders: orders.into_boxed_slice(),
            has_annual: has_annual.into_boxed_slice(),
            n_stages,
            n_hydros,
            max_order,
        })
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    fn debug_assert_stage_hydro(&self, stage: usize, hydro: usize) {
        debug_assert!(
            stage < self.n_stages,
            "stage index {stage} is out of bounds (n_stages = {})",
            self.n_stages
        );
        debug_assert!(
            hydro < self.n_hydros,
            "hydro index {hydro} is out of bounds (n_hydros = {})",
            self.n_hydros
        );
    }

    fn debug_assert_hydro(&self, hydro: usize) {
        debug_assert!(
            hydro < self.n_hydros,
            "hydro index {hydro} is out of bounds (n_hydros = {})",
            self.n_hydros
        );
    }

    /// Deterministic base `b_{h,m(t)}` for the given stage and series element indices.
    ///
    /// # Panics
    ///
    /// Panics if `stage >= n_stages` or `hydro >= n_hydros`.
    #[must_use]
    pub fn deterministic_base(&self, stage: usize, hydro: usize) -> f64 {
        self.debug_assert_stage_hydro(stage, hydro);
        self.deterministic_base[stage * self.n_hydros + hydro]
    }

    /// Residual standard deviation `σ_{m(t)}` for the given stage and series element indices.
    ///
    /// # Panics
    ///
    /// Panics if `stage >= n_stages` or `hydro >= n_hydros`.
    #[must_use]
    pub fn sigma(&self, stage: usize, hydro: usize) -> f64 {
        self.debug_assert_stage_hydro(stage, hydro);
        self.sigma[stage * self.n_hydros + hydro]
    }

    /// Slice of AR lag coefficients `ψ_{m(t),ℓ}` (original units), length
    /// `max_order`. The meaningful-slot count (which trailing slots are
    /// structurally zero vs. carry `ψ̂/12`) is [`Self::effective_lag_count`].
    ///
    /// # Panics
    ///
    /// Panics if `stage >= n_stages` or `hydro >= n_hydros`.
    #[must_use]
    pub fn psi_slice(&self, stage: usize, hydro: usize) -> &[f64] {
        self.debug_assert_stage_hydro(stage, hydro);
        if self.max_order == 0 {
            return &[];
        }
        let start = stage * self.n_hydros * self.max_order + hydro * self.max_order;
        &self.psi[start..start + self.max_order]
    }

    /// Classical AR order `p` for the series element (max across stages) — the lag
    /// positions holding explicit `φ̂_j`. For the PAR(p)-A meaningful-slot count
    /// including `ψ̂/12` slots, use [`Self::effective_lag_count`].
    ///
    /// # Panics
    ///
    /// Panics if `hydro >= n_hydros`.
    #[must_use]
    pub fn order(&self, hydro: usize) -> usize {
        self.debug_assert_hydro(hydro);
        self.orders[hydro]
    }

    /// Authoritative count of lag slots that may carry non-zero `ψ` for the series
    /// element — what downstream consumers building sparse lag layouts must track:
    ///
    /// - **Classical PAR(p)** (no annual component): `order(hydro)`; slots beyond
    ///   the AR order are structurally zero.
    /// - **PAR(p)-A** (annual component present): `max_order` (= 12 for a monthly
    ///   study); all slots carry the `ψ̂/12` annual term.
    ///
    /// # Panics
    ///
    /// Panics if `hydro >= n_hydros`.
    #[must_use]
    pub fn effective_lag_count(&self, hydro: usize) -> usize {
        self.debug_assert_hydro(hydro);
        if self.has_annual[hydro] {
            self.max_order
        } else {
            self.orders[hydro]
        }
    }

    /// Whether any [`InflowModel`] for this series element was built with
    /// `annual: Some(_)` (see [`Self::effective_lag_count`]).
    ///
    /// # Panics
    ///
    /// Panics if `hydro >= n_hydros`.
    #[must_use]
    pub fn has_annual(&self, hydro: usize) -> bool {
        self.debug_assert_hydro(hydro);
        self.has_annual[hydro]
    }

    /// Number of study stages.
    #[must_use]
    pub fn n_stages(&self) -> usize {
        self.n_stages
    }

    /// Number of series elements tracked by the PAR model.
    #[must_use]
    pub fn n_hydros(&self) -> usize {
        self.n_hydros
    }

    /// Maximum AR order across all hydros and stages.
    #[must_use]
    pub fn max_order(&self) -> usize {
        self.max_order
    }
}

impl Default for PrecomputedPar {
    /// Empty [`PrecomputedPar`] sentinel for callers with no PAR models.
    fn default() -> Self {
        Self {
            deterministic_base: Box::new([]),
            sigma: Box::new([]),
            psi: Box::new([]),
            orders: Box::new([]),
            has_annual: Box::new([]),
            n_stages: 0,
            n_hydros: 0,
            max_order: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Module-level helpers
// ---------------------------------------------------------------------------

/// Mutable output buffers and dimension metadata for [`fill_stage_arrays`].
///
/// Bundles the three flat arrays and their two governing size scalars so that
/// [`fill_stage_arrays`] stays within the allowed parameter count.
struct StageArrayBuffers<'a> {
    /// See [`PrecomputedPar::deterministic_base`].
    deterministic_base: &'a mut Vec<f64>,
    /// See [`PrecomputedPar::sigma`].
    sigma: &'a mut Vec<f64>,
    /// See [`PrecomputedPar::psi`].
    psi: &'a mut Vec<f64>,
    /// Number of series elements (hydros).
    n_hydros: usize,
    /// Maximum AR order across all hydros and stages.
    max_order: usize,
}

/// Fill `bufs` for every laid-out (stage, hydro) pair.
///
/// A lag stage whose exact `inflow_models` lookup misses takes its season's
/// statistics under the rule [`PrecomputedPar::build`] documents. A lag that
/// reaches back across a change of resolution instead reads the statistics of
/// the season its stage's own [`SeasonCycles`] level steps back to (the quarter
/// a quarterly stage's lag aggregates), when some stage carries that season.
///
/// # Errors
///
/// Returns [`StochasticError::InvalidParParameters`] when a study stage has
/// AR order > 0 or an annual component but no `season_id`.
fn fill_stage_arrays(
    stages: &[Stage],
    hydro_ids: &[EntityId],
    hydro_index: &HashMap<EntityId, usize>,
    model_map: &HashMap<(i32, i32), &InflowModel>,
    inflow_models: &[InflowModel],
    season_map: Option<&SeasonMap>,
    bufs: &mut StageArrayBuffers<'_>,
) -> Result<(), StochasticError> {
    let n_hydros = bufs.n_hydros;
    let max_order = bufs.max_order;

    let model_stats: HashMap<(usize, i32), (f64, f64)> = inflow_models
        .iter()
        .filter_map(|m| {
            hydro_index
                .get(&m.hydro_id)
                .map(|&h_idx| ((h_idx, m.stage_id), (m.mean_m3s, m.std_m3s)))
        })
        .collect();

    let n_seasons = laid_out_stages(stages)
        .filter_map(|s| s.season_id)
        .collect::<HashSet<_>>()
        .len();

    // season_id of stage 0 — the offset resolve_season_id needs for non-January
    // starts.
    let season_offset = laid_out_stages(stages)
        .find_map(|s| s.season_id)
        .unwrap_or(0);

    let stitched = season_map.map(|sm| StitchedSeasonMap::build(stages, sm, max_order));
    let cycles = season_map.map(SeasonCycles::new);

    let stage_to_season: HashMap<i32, usize> = stages
        .iter()
        .filter_map(|s| s.season_id.map(|sid| (s.id, sid)))
        .collect();

    let mut season_stats: HashMap<(usize, usize), (i32, (f64, f64))> = HashMap::new();
    for (&(h_idx, stage_id), &stats) in &model_stats {
        let Some(&sid) = stage_to_season.get(&stage_id) else {
            continue;
        };
        season_stats
            .entry((h_idx, sid))
            .and_modify(|kept| {
                if stage_id < kept.0 {
                    *kept = (stage_id, stats);
                }
            })
            .or_insert((stage_id, stats));
    }

    let lag_stats = |h_idx: usize, stage_season: Option<usize>, lag_stage_id: i32, steps: usize| {
        let cross_resolution_season = cycles
            .as_ref()
            .zip(stage_season)
            .filter(|_| lag_stage_id >= 0)
            .and_then(|(cycles, season)| {
                let &lag_stage_season = stage_to_season.get(&lag_stage_id)?;
                cross_resolution_lag_season(cycles, season, lag_stage_season, steps)
            });
        if let Some(&(_, stats)) =
            cross_resolution_season.and_then(|sid| season_stats.get(&(h_idx, sid)))
        {
            return stats;
        }
        if let Some(&stats) = model_stats.get(&(h_idx, lag_stage_id)) {
            return stats;
        }
        let season = match &stitched {
            Some(map) => map.season_of(lag_stage_id),
            None if n_seasons > 0 => {
                Some(resolve_season_id(lag_stage_id, n_seasons, season_offset))
            }
            None => None,
        };
        season
            .and_then(|sid| season_stats.get(&(h_idx, sid)))
            .map_or((0.0, 0.0), |&(_, stats)| stats)
    };

    for (s_idx, stage) in laid_out_stages(stages).enumerate() {
        let stage_id = stage.id;

        for (h_idx, &hydro_id) in hydro_ids.iter().enumerate() {
            let flat2 = s_idx * n_hydros + h_idx;

            let model = model_map.get(&(hydro_id.0, stage_id));

            match model {
                None => {
                    bufs.deterministic_base[flat2] = 0.0;
                    bufs.sigma[flat2] = 0.0;
                    // psi stays 0.0 (already initialized)
                }
                Some(m) => {
                    let s_m = m.std_m3s;
                    let mu_m = m.mean_m3s;
                    let order = m.ar_order();

                    bufs.sigma[flat2] = s_m * m.residual_std_ratio;

                    // Annual ψ̂ (see module docs). ψ̂ = 0 when PAR-A is off or the
                    // stride is classical (max_order < 12) or σ^A == 0, degenerating
                    // to the classical path.
                    let psi_hat = if bufs.max_order < 12 {
                        0.0
                    } else {
                        match m.annual.as_ref() {
                            None => 0.0,
                            Some(ann) if ann.std_m3s == 0.0 => 0.0,
                            Some(ann) => ann.coefficient * s_m / ann.std_m3s,
                        }
                    };

                    // Convert ψ* → ψ and fold the deterministic base (see module
                    // docs for the classical and PAR-A formulas).
                    let mut base = mu_m;

                    for lag in 0..max_order {
                        // lag is 0-based: index 0 is lag ℓ=1.
                        let lag_stage_id = stage_id - i32::try_from(lag + 1).unwrap_or(i32::MAX);

                        let (mu_lag, s_lag) =
                            lag_stats(h_idx, stage.season_id, lag_stage_id, lag + 1);

                        // φ̂ = 0 beyond the AR order; s_lag == 0 also yields 0 (the
                        // caller must validate that AR-order-bearing lag stages have
                        // positive std).
                        let ar_contrib = if lag >= order || s_lag == 0.0 {
                            0.0
                        } else {
                            m.ar_coefficients[lag] * s_m / s_lag
                        };

                        let psi_val = ar_contrib + psi_hat / 12.0;

                        let flat3 = s_idx * n_hydros * max_order + h_idx * max_order + lag;
                        bufs.psi[flat3] = psi_val;

                        base -= psi_val * mu_lag;
                    }

                    if (order > 0 || m.annual.is_some()) && stage.season_id.is_none() {
                        return Err(StochasticError::InvalidParParameters {
                            hydro_id: hydro_id.0,
                            stage_id,
                            reason: "stage has AR order > 0 or an annual component \
                                     but no season_id; cannot perform lag-stage \
                                     statistics lookup"
                                .to_string(),
                        });
                    }

                    bufs.deterministic_base[flat2] = base;
                }
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use chrono::{Months, NaiveDate};
    use cobre_core::{
        EntityId,
        scenario::InflowModel,
        temporal::{NoiseMethod, ScenarioSourceConfig, Stage},
        test_support::{StageSpec, date, f64_bits_eq, single_block},
    };

    use super::{PrecomputedPar, resolve_season_id};
    use crate::test_support::{
        InflowModelSpec, MonthlyLabels, monthly_quarterly_season_map, monthly_season_map,
        sparse_ring_season_map,
    };

    fn make_stage(index: usize, id: i32, season_id: Option<usize>) -> Stage {
        cobre_core::test_support::make_stage(StageSpec {
            id,
            index: Some(index),
            season_id,
            blocks: single_block("SINGLE", 744.0),
            scenario_config: ScenarioSourceConfig {
                branching_factor: 10,
                noise_method: NoiseMethod::Saa,
            },
            ..Default::default()
        })
    }

    fn make_model(
        hydro_id: i32,
        stage_id: i32,
        mean: f64,
        std: f64,
        coeffs: Vec<f64>,
        residual_ratio: f64,
    ) -> InflowModel {
        crate::test_support::make_inflow_model(InflowModelSpec {
            hydro_id,
            stage_id,
            mean_m3s: mean,
            std_m3s: std,
            ar_coefficients: coeffs,
            residual_std_ratio: residual_ratio,
            ..Default::default()
        })
    }

    #[test]
    fn ar_order_zero_deterministic_base_equals_mean() {
        let stage = make_stage(0, 0, Some(0));
        let model = make_model(1, 0, 100.0, 30.0, vec![], 1.0);
        let lp = PrecomputedPar::build(&[model], &[stage], &[EntityId(1)], None).unwrap();

        assert_eq!(lp.n_stages(), 1);
        assert_eq!(lp.n_hydros(), 1);
        assert_eq!(lp.max_order(), 0);
        assert!(
            (lp.deterministic_base(0, 0) - 100.0).abs() < f64::EPSILON,
            "expected 100.0, got {}",
            lp.deterministic_base(0, 0)
        );
        assert!(
            (lp.sigma(0, 0) - 30.0).abs() < f64::EPSILON,
            "expected 30.0, got {}",
            lp.sigma(0, 0)
        );
        assert_eq!(lp.psi_slice(0, 0).len(), 0);
        assert_eq!(lp.order(0), 0);
    }

    #[test]
    fn ar_order_zero_std_zero_gives_zero_sigma() {
        let stage = make_stage(0, 0, Some(0));
        let model = make_model(1, 0, 50.0, 0.0, vec![], 1.0);
        let lp = PrecomputedPar::build(&[model], &[stage], &[EntityId(1)], None).unwrap();

        assert!((lp.sigma(0, 0)).abs() < f64::EPSILON);
        assert!((lp.deterministic_base(0, 0) - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn ar_order_1_acceptance_criterion() {
        // Stage 0 (study stage, id=0, season=0) — the stage being computed.
        let study_stage = make_stage(0, 0, Some(0));

        // Pre-study stage (id=-1, season=0) — provides lag-1 stats.
        // This is NOT in the `stages` slice (which only has study stages),
        // but IS in inflow_models so the builder can find lag stats.
        let pre_study_model = make_model(1, -1, 100.0, 30.0, vec![], 1.0);
        let study_model = make_model(1, 0, 100.0, 30.0, vec![0.3], 0.954);

        let lp = PrecomputedPar::build(
            &[pre_study_model, study_model],
            &[study_stage],
            &[EntityId(1)],
            None,
        )
        .unwrap();

        let expected_base = 70.0_f64;
        assert!(
            (lp.deterministic_base(0, 0) - expected_base).abs() < 1e-10,
            "deterministic_base: expected {expected_base}, got {}",
            lp.deterministic_base(0, 0)
        );

        let expected_sigma = 30.0 * 0.954;
        assert!(
            (lp.sigma(0, 0) - expected_sigma).abs() < 1e-10,
            "sigma: expected {expected_sigma}, got {}",
            lp.sigma(0, 0)
        );
    }

    #[test]
    fn two_hydros_three_stages_varying_orders() {
        let hydro_ids = [EntityId(3), EntityId(5)];

        let stages: Vec<Stage> = (0..3)
            .map(|i| make_stage(i, i32::try_from(i).unwrap(), Some(0)))
            .collect();

        let prestudy_models = vec![
            make_model(3, -2, 80.0, 20.0, vec![], 1.0),
            make_model(3, -1, 90.0, 25.0, vec![], 1.0),
            make_model(5, -1, 60.0, 15.0, vec![], 1.0),
        ];

        let study_models = vec![
            make_model(3, 0, 100.0, 30.0, vec![0.4, 0.2], 0.9),
            make_model(3, 1, 110.0, 28.0, vec![0.35], 0.94),
            make_model(3, 2, 95.0, 25.0, vec![], 1.0),
            make_model(5, 0, 70.0, 18.0, vec![0.5], 0.87),
            make_model(5, 1, 75.0, 20.0, vec![0.45], 0.89),
            make_model(5, 2, 68.0, 17.0, vec![0.3], 0.95),
        ];

        let mut all_models: Vec<InflowModel> = prestudy_models;
        all_models.extend(study_models);

        let lp = PrecomputedPar::build(&all_models, &stages, &hydro_ids, None).unwrap();

        assert_eq!(lp.n_stages(), 3);
        assert_eq!(lp.n_hydros(), 2);
        assert_eq!(lp.max_order(), 2);

        assert_eq!(lp.order(0), 2);
        assert_eq!(lp.order(1), 1);

        let expected_base_h3_s0 = 100.0 - (0.4 * 30.0 / 25.0) * 90.0 - (0.2 * 30.0 / 20.0) * 80.0;
        assert!(
            (lp.deterministic_base(0, 0) - expected_base_h3_s0).abs() < 1e-10,
            "h3 s0 base: expected {expected_base_h3_s0}, got {}",
            lp.deterministic_base(0, 0)
        );
        assert!((lp.sigma(0, 0) - 27.0).abs() < 1e-10);

        // psi_slice for h3/s0 should have length 2 (max_order=2)
        let psi_h3_s0 = lp.psi_slice(0, 0);
        assert_eq!(psi_h3_s0.len(), 2);
        assert!((psi_h3_s0[0] - 0.4 * 30.0 / 25.0).abs() < 1e-10);
        assert!((psi_h3_s0[1] - 0.2 * 30.0 / 20.0).abs() < 1e-10);

        assert!((lp.deterministic_base(2, 0) - 95.0).abs() < 1e-10);
        assert!((lp.sigma(2, 0) - 25.0).abs() < 1e-10);
        let psi_h3_s2 = lp.psi_slice(2, 0);
        assert_eq!(psi_h3_s2.len(), 2);
        assert!((psi_h3_s2[0]).abs() < 1e-10);
        assert!((psi_h3_s2[1]).abs() < 1e-10);

        let expected_base_h5_s0 = 70.0 - (0.5 * 18.0 / 15.0) * 60.0;
        assert!(
            (lp.deterministic_base(0, 1) - expected_base_h5_s0).abs() < 1e-10,
            "h5 s0 base: expected {expected_base_h5_s0}, got {}",
            lp.deterministic_base(0, 1)
        );
        assert!((lp.sigma(0, 1) - 18.0 * 0.87).abs() < 1e-10);

        let psi_h5_s0 = lp.psi_slice(0, 1);
        assert_eq!(psi_h5_s0.len(), 2);
        assert!((psi_h5_s0[0] - 0.5 * 18.0 / 15.0).abs() < 1e-10);
        assert!((psi_h5_s0[1]).abs() < 1e-10);
    }

    #[test]
    fn declaration_order_invariance() {
        let stage = make_stage(0, 0, Some(0));

        let models = vec![
            make_model(5, 0, 200.0, 40.0, vec![], 1.0),
            make_model(3, 0, 100.0, 30.0, vec![], 1.0),
        ];

        let hydro_ids = [EntityId(3), EntityId(5)];

        let lp = PrecomputedPar::build(&models, &[stage], &hydro_ids, None).unwrap();

        assert!(
            (lp.deterministic_base(0, 0) - 100.0).abs() < f64::EPSILON,
            "h_idx=0 should be EntityId(3) with mean=100, got {}",
            lp.deterministic_base(0, 0)
        );
        assert!(
            (lp.deterministic_base(0, 1) - 200.0).abs() < f64::EPSILON,
            "h_idx=1 should be EntityId(5) with mean=200, got {}",
            lp.deterministic_base(0, 1)
        );
    }

    #[test]
    fn psi_slice_padded_for_shorter_order() {
        let stages = vec![make_stage(0, 0, Some(0))];
        let hydro_ids = [EntityId(1), EntityId(2)];

        let pre_models = vec![
            make_model(1, -3, 100.0, 30.0, vec![], 1.0),
            make_model(1, -2, 100.0, 30.0, vec![], 1.0),
            make_model(1, -1, 100.0, 30.0, vec![], 1.0),
            make_model(2, -2, 100.0, 30.0, vec![], 1.0),
            make_model(2, -1, 100.0, 30.0, vec![], 1.0),
        ];
        let study_models = vec![
            make_model(1, 0, 100.0, 30.0, vec![0.3, 0.2, 0.1], 0.9),
            make_model(2, 0, 100.0, 30.0, vec![0.4, 0.2], 0.9),
        ];

        let mut all_models = pre_models;
        all_models.extend(study_models);

        let lp = PrecomputedPar::build(&all_models, &stages, &hydro_ids, None).unwrap();

        assert_eq!(lp.max_order(), 3);

        // Hydro 2 (h_idx=1) has AR(2) but psi_slice returns length 3.
        let psi = lp.psi_slice(0, 1);
        assert_eq!(
            psi.len(),
            3,
            "expected length 3 (max_order), got {}",
            psi.len()
        );
        assert!(
            (psi[0] - 0.4).abs() < 1e-10,
            "psi[0]: expected 0.4, got {}",
            psi[0]
        );
        assert!(
            (psi[1] - 0.2).abs() < 1e-10,
            "psi[1]: expected 0.2, got {}",
            psi[1]
        );
        assert!((psi[2]).abs() < 1e-10);
        assert_eq!(lp.order(1), 2);
    }

    #[test]
    fn missing_model_fills_zero_defaults() {
        let stages = [make_stage(0, 0, Some(0))];
        let hydro_ids = [EntityId(1), EntityId(2)];

        let models = [make_model(1, 0, 100.0, 30.0, vec![], 1.0)];

        let lp = PrecomputedPar::build(&models, &[stages[0].clone()], &hydro_ids, None).unwrap();

        assert!((lp.deterministic_base(0, 1)).abs() < f64::EPSILON);
        assert!((lp.sigma(0, 1)).abs() < f64::EPSILON);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "stage index 1 is out of bounds")]
    fn deterministic_base_out_of_bounds_panics() {
        let stage = make_stage(0, 0, Some(0));
        let model = make_model(1, 0, 100.0, 30.0, vec![], 1.0);
        let lp = PrecomputedPar::build(&[model], &[stage], &[EntityId(1)], None).unwrap();
        let _ = lp.deterministic_base(1, 0);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "stage index 1 is out of bounds")]
    fn sigma_out_of_bounds_panics() {
        let stage = make_stage(0, 0, Some(0));
        let model = make_model(1, 0, 100.0, 30.0, vec![], 1.0);
        let lp = PrecomputedPar::build(&[model], &[stage], &[EntityId(1)], None).unwrap();
        let _ = lp.sigma(1, 0);
    }

    #[test]
    fn acceptance_criterion_ar_order_zero_std_zero() {
        let stage = make_stage(0, 0, Some(0));
        let model = make_model(1, 0, 50.0, 0.0, vec![], 1.0);
        let lp = PrecomputedPar::build(&[model], &[stage], &[EntityId(1)], None).unwrap();

        assert!((lp.sigma(0, 0)).abs() < f64::EPSILON);
        assert!((lp.deterministic_base(0, 0) - 50.0).abs() < f64::EPSILON);
    }

    // -----------------------------------------------------------------------
    // Pre-study lag season fallback tests
    // -----------------------------------------------------------------------

    /// Build 12 monthly study stages (id `0..11`, `season_id` `0..11`).
    fn make_monthly_stages() -> Vec<Stage> {
        (0..12_usize)
            .map(|i| make_stage(i, i32::try_from(i).unwrap(), Some(i)))
            .collect()
    }

    /// Build study-only inflow models for 12 monthly seasons.
    ///
    /// Each season `s` gets `mean = 100 + s*10`, `std = 20 + s*2`.
    /// The `ar_stage` parameter specifies which stage gets the given AR
    /// coefficients; all other stages are AR(0).
    fn make_monthly_models(
        hydro_id: i32,
        ar_stage: i32,
        ar_coeffs: &[f64],
        residual: f64,
    ) -> Vec<InflowModel> {
        (0..12_i32)
            .map(|i| {
                let fi = f64::from(i);
                let mean = 100.0 + fi * 10.0;
                let std = 20.0 + fi * 2.0;
                let (c, r) = if i == ar_stage {
                    (ar_coeffs.to_vec(), residual)
                } else {
                    (vec![], 1.0)
                };
                make_model(hydro_id, i, mean, std, c, r)
            })
            .collect()
    }

    #[test]
    fn pre_study_lag_resolves_via_season_fallback() {
        // 12 study stages, 1 hydro with AR(1). No pre-study entries.
        // At stage 0, lag-1 has id = -1, which should fall back to season 11.
        let stages = make_monthly_stages();
        let models = make_monthly_models(1, 0, &[0.5], 0.9);

        let lp = PrecomputedPar::build(&models, &stages, &[EntityId(1)], None).unwrap();

        // At stage 0 (season 0): lag-1 -> stage_id = -1 -> season 11.
        // Season 11 stats: mean = 100 + 11*10 = 210, std = 20 + 11*2 = 42.
        let mu_0 = 100.0;
        let s_0 = 20.0;
        let mu_11 = 210.0;
        let s_11 = 42.0;
        let psi_star = 0.5;

        let psi_val = psi_star * s_0 / s_11;
        let expected_base = mu_0 - psi_val * mu_11;

        // psi should be non-zero (the core fix assertion).
        assert!(
            lp.psi_slice(0, 0)[0].abs() > 1e-10,
            "psi should be non-zero when season fallback resolves; got {}",
            lp.psi_slice(0, 0)[0]
        );
        assert!(
            (lp.psi_slice(0, 0)[0] - psi_val).abs() < 1e-10,
            "psi[0]: expected {psi_val}, got {}",
            lp.psi_slice(0, 0)[0]
        );
        assert!(
            (lp.deterministic_base(0, 0) - expected_base).abs() < 1e-10,
            "base: expected {expected_base}, got {}",
            lp.deterministic_base(0, 0)
        );
    }

    #[test]
    fn pre_study_lag_ar6_all_lags_resolve() {
        // 12 study stages, 1 hydro with AR(6). No pre-study entries.
        // At stage 0, lags 1-6 map to stage_ids -1..-6, which should resolve
        // to season_ids 11, 10, 9, 8, 7, 6.
        let stages = make_monthly_stages();
        let models = make_monthly_models(1, 0, &[0.3, 0.2, 0.15, 0.1, 0.08, 0.05], 0.85);

        let lp = PrecomputedPar::build(&models, &stages, &[EntityId(1)], None).unwrap();

        // All 6 AR coefficients at stage 0 should be non-zero.
        let psi = lp.psi_slice(0, 0);
        for (lag, psi_lag) in psi.iter().enumerate().take(6) {
            assert!(
                psi_lag.abs() > 1e-10,
                "psi[{lag}] should be non-zero; got {psi_lag}",
            );
        }

        // Verify specific values for lag 0 (season 11) and lag 5 (season 6).
        let s_0 = 20.0;
        let s_11 = 42.0;
        let s_6 = 32.0; // 20 + 6*2
        assert!(
            (psi[0] - 0.3 * s_0 / s_11).abs() < 1e-10,
            "psi[0]: expected {}, got {}",
            0.3 * s_0 / s_11,
            psi[0]
        );
        assert!(
            (psi[5] - 0.05 * s_0 / s_6).abs() < 1e-10,
            "psi[5]: expected {}, got {}",
            0.05 * s_0 / s_6,
            psi[5]
        );
    }

    #[test]
    fn pre_study_lag_partial_resolution() {
        // 12 study stages, 1 hydro with AR(2). Target: stage 1 (not 0).
        // Lag-1 hits stage 0 (exists in study, exact match).
        // Lag-2 hits stage -1 (needs season fallback -> season 11).
        let stages = make_monthly_stages();
        let models = make_monthly_models(1, 1, &[0.4, 0.25], 0.9);

        let lp = PrecomputedPar::build(&models, &stages, &[EntityId(1)], None).unwrap();

        // Stage 1 (season 1): s_1 = 22.0
        let s_1 = 22.0;

        // Lag-1: exact match to stage 0 (season 0), s_0 = 20.0.
        let expected_psi_0 = 0.4 * s_1 / 20.0;
        assert!(
            (lp.psi_slice(1, 0)[0] - expected_psi_0).abs() < 1e-10,
            "psi[0] (exact match): expected {expected_psi_0}, got {}",
            lp.psi_slice(1, 0)[0]
        );

        // Lag-2: stage_id = -1 -> season 11 (fallback), s_11 = 42.0.
        let expected_psi_1 = 0.25 * s_1 / 42.0;
        assert!(
            (lp.psi_slice(1, 0)[1] - expected_psi_1).abs() < 1e-10,
            "psi[1] (season fallback): expected {expected_psi_1}, got {}",
            lp.psi_slice(1, 0)[1]
        );
    }

    #[test]
    fn pre_study_lag_with_explicit_prestudy_model() {
        // 12 study stages + 1 explicit pre-study InflowModel for stage_id = -1.
        // AR(1) at stage 0. The exact match path should fire, NOT season fallback.
        let stages = make_monthly_stages();
        let mut models = make_monthly_models(1, 0, &[0.3], 0.954);

        // Add explicit pre-study model for stage_id = -1.
        // Use DIFFERENT stats than season 11 to prove exact match is used.
        models.push(make_model(1, -1, 100.0, 30.0, vec![], 1.0));

        let lp = PrecomputedPar::build(&models, &stages, &[EntityId(1)], None).unwrap();

        // Stage 0: AR(1), lag-1 -> stage_id = -1 -> EXACT match to prestudy
        // (mean=100, std=30), NOT season 11 (mean=210, std=42).
        let s_0 = 20.0;
        let psi_val = 0.3 * s_0 / 30.0; // Using prestudy std=30
        let expected_base = 100.0 - psi_val * 100.0; // Using prestudy mean=100

        assert!(
            (lp.psi_slice(0, 0)[0] - psi_val).abs() < 1e-10,
            "psi[0] should use exact match (std=30), not fallback (std=42); got {}",
            lp.psi_slice(0, 0)[0]
        );
        assert!(
            (lp.deterministic_base(0, 0) - expected_base).abs() < 1e-10,
            "base should use exact match; expected {expected_base}, got {}",
            lp.deterministic_base(0, 0)
        );
    }

    #[test]
    fn season_fallback_deep_negative_wraps_correctly() {
        // January-start: offset=0, same as bare modular arithmetic.
        assert_eq!(resolve_season_id(-1, 12, 0), 11);
        assert_eq!(resolve_season_id(-6, 12, 0), 6);
        assert_eq!(resolve_season_id(-12, 12, 0), 0);
        assert_eq!(resolve_season_id(-13, 12, 0), 11);
        assert_eq!(resolve_season_id(0, 12, 0), 0);
        assert_eq!(resolve_season_id(5, 12, 0), 5);

        // Additional edge cases.
        assert_eq!(resolve_season_id(-24, 12, 0), 0);
        assert_eq!(resolve_season_id(-25, 12, 0), 11);
        assert_eq!(resolve_season_id(12, 12, 0), 0);
        assert_eq!(resolve_season_id(13, 12, 0), 1);

        // Non-12 cycle lengths.
        assert_eq!(resolve_season_id(-1, 4, 0), 3);
        assert_eq!(resolve_season_id(-5, 4, 0), 3);
        assert_eq!(resolve_season_id(-4, 4, 0), 0);
        assert_eq!(resolve_season_id(-1, 1, 0), 0);
    }

    #[test]
    fn season_fallback_with_nonzero_offset() {
        // March-start study: stage 0 = season 2, offset = 2.
        // stage -1 should be February (season 1), not December (season 11).
        assert_eq!(resolve_season_id(-1, 12, 2), 1);
        assert_eq!(resolve_season_id(-2, 12, 2), 0);
        assert_eq!(resolve_season_id(-3, 12, 2), 11);
        assert_eq!(resolve_season_id(0, 12, 2), 2);
        assert_eq!(resolve_season_id(9, 12, 2), 11);

        // April-start: offset = 3.
        assert_eq!(resolve_season_id(-1, 12, 3), 2);
        assert_eq!(resolve_season_id(-4, 12, 3), 11);
        assert_eq!(resolve_season_id(0, 12, 3), 3);

        // Deep negative with offset.
        assert_eq!(resolve_season_id(-13, 12, 2), 1);
        assert_eq!(resolve_season_id(-25, 12, 2), 1);
    }

    // -----------------------------------------------------------------------
    // Integration tests for AR conditioning at stage 0
    // -----------------------------------------------------------------------

    /// Season stats helper: `mean = 100 + season*10`, `std = 20 + season*2`.
    fn season_mean(season: usize) -> f64 {
        #[allow(clippy::cast_precision_loss)]
        let s = season as f64;
        100.0 + s * 10.0
    }

    fn season_std(season: usize) -> f64 {
        #[allow(clippy::cast_precision_loss)]
        let s = season as f64;
        20.0 + s * 2.0
    }

    #[test]
    fn integration_ar_conditioning_at_stage_zero() {
        // 12 monthly stages, 3 hydros with AR(1), AR(3), AR(6).
        // NO pre-study entries -- simulates external-file code path.
        let stages = make_monthly_stages();

        let mut all_models = Vec::new();

        // Hydro 1 (EntityId(1)): AR(1) at stage 0 only.
        all_models.extend(make_monthly_models(1, 0, &[0.5], 0.9));

        // Hydro 2 (EntityId(2)): AR(3) at stage 0 only.
        let models_h2: Vec<InflowModel> = (0..12_i32)
            .map(|i| {
                let fi = f64::from(i);
                let mean = 100.0 + fi * 10.0;
                let std = 20.0 + fi * 2.0;
                let (c, r) = if i == 0 {
                    (vec![0.4, 0.25, 0.15], 0.85)
                } else {
                    (vec![], 1.0)
                };
                make_model(2, i, mean, std, c, r)
            })
            .collect();
        all_models.extend(models_h2);

        // Hydro 3 (EntityId(3)): AR(6) at stage 0 only.
        let models_h3: Vec<InflowModel> = (0..12_i32)
            .map(|i| {
                let fi = f64::from(i);
                let mean = 100.0 + fi * 10.0;
                let std = 20.0 + fi * 2.0;
                let (c, r) = if i == 0 {
                    (vec![0.3, 0.2, 0.15, 0.1, 0.08, 0.05], 0.8)
                } else {
                    (vec![], 1.0)
                };
                make_model(3, i, mean, std, c, r)
            })
            .collect();
        all_models.extend(models_h3);

        let hydro_ids = [EntityId(1), EntityId(2), EntityId(3)];
        let lp = PrecomputedPar::build(&all_models, &stages, &hydro_ids, None).unwrap();

        // 1. Non-zero coefficients at stage 0 for all hydros.
        for (h_idx, order) in [(0, 1), (1, 3), (2, 6)] {
            let psi = lp.psi_slice(0, h_idx);
            for (lag, psi_lag) in psi.iter().enumerate().take(order) {
                assert!(
                    psi_lag.abs() > 1e-10,
                    "hydro {h_idx} psi[{lag}] at stage 0 should be non-zero; got {psi_lag}",
                );
            }
        }

        // 2. Correct coefficient values for hydro 3 (AR(6)).
        let s_0 = season_std(0);
        let psi_star_h3 = [0.3, 0.2, 0.15, 0.1, 0.08, 0.05];
        let psi_h3 = lp.psi_slice(0, 2);
        for (lag, (&psi_val, &psi_star)) in psi_h3.iter().zip(&psi_star_h3).enumerate() {
            let lag_i32 = i32::try_from(lag).unwrap();
            let lag_season = resolve_season_id(-(lag_i32 + 1), 12, 0);
            let expected = psi_star * s_0 / season_std(lag_season);
            assert!(
                (psi_val - expected).abs() < 1e-10,
                "h3 psi[{lag}]: expected {expected}, got {psi_val}",
            );
        }

        // 3. Deterministic base correctness for hydro 3.
        let mu_0 = season_mean(0);
        let mut expected_base = mu_0;
        for (lag, &psi_val) in psi_h3.iter().enumerate().take(6) {
            let lag_i32 = i32::try_from(lag).unwrap();
            let lag_season = resolve_season_id(-(lag_i32 + 1), 12, 0);
            expected_base -= psi_val * season_mean(lag_season);
        }
        assert!(
            (lp.deterministic_base(0, 2) - expected_base).abs() < 1e-10,
            "h3 base at stage 0: expected {expected_base}, got {}",
            lp.deterministic_base(0, 2)
        );

        // 4. No regression at stage 6 (all lags hit study stages via exact match).
        let psi_h3_s6 = lp.psi_slice(6, 2);
        // Hydro 3 has AR(6) only at stage 0; stage 6 is AR(0), so psi should be zero.
        for (lag, psi_lag) in psi_h3_s6.iter().enumerate().take(6) {
            assert!(
                psi_lag.abs() < 1e-10,
                "h3 stage 6 psi[{lag}] should be zero (AR only at stage 0); got {psi_lag}",
            );
        }
    }

    #[test]
    fn integration_multiyear_ar_conditioning() {
        // 24 monthly stages (2 years), 2 hydros with AR(2).
        // No pre-study entries.
        let stages: Vec<Stage> = (0..24_usize)
            .map(|i| make_stage(i, i32::try_from(i).unwrap(), Some(i % 12)))
            .collect();

        // Build models for both hydros across 24 stages.
        let mut all_models = Vec::new();
        for h_id in [1, 2] {
            for i in 0..24_i32 {
                #[allow(clippy::cast_sign_loss)] // i is always in 0..24
                let season = (i % 12) as usize;
                let mean = season_mean(season);
                let std = season_std(season);
                let (c, r) = if i == 0 {
                    (vec![0.4, 0.25], 0.9)
                } else {
                    (vec![], 1.0)
                };
                all_models.push(make_model(h_id, i, mean, std, c, r));
            }
        }

        let hydro_ids = [EntityId(1), EntityId(2)];
        let lp = PrecomputedPar::build(&all_models, &stages, &hydro_ids, None).unwrap();

        // Stage 0: lag-1 (id=-1, season 11) and lag-2 (id=-2, season 10)
        // resolve via season fallback.
        let s_0 = season_std(0);
        for h_idx in 0..2 {
            let psi_s0 = lp.psi_slice(0, h_idx);
            // lag 0: season 11 fallback
            let expected_0 = 0.4 * s_0 / season_std(11);
            assert!(
                (psi_s0[0] - expected_0).abs() < 1e-10,
                "h{h_idx} stage 0 psi[0]: expected {expected_0}, got {}",
                psi_s0[0]
            );
            // lag 1: season 10 fallback
            let expected_1 = 0.25 * s_0 / season_std(10);
            assert!(
                (psi_s0[1] - expected_1).abs() < 1e-10,
                "h{h_idx} stage 0 psi[1]: expected {expected_1}, got {}",
                psi_s0[1]
            );
        }

        let mut models_v2 = Vec::new();
        for h_id in [1, 2] {
            for i in 0..24_i32 {
                #[allow(clippy::cast_sign_loss)] // i is always in 0..24
                let season = (i % 12) as usize;
                let mean = season_mean(season);
                let std = season_std(season);
                // AR(2) at stages 0 and 12 (both season 0).
                let (c, r) = if i == 0 || i == 12 {
                    (vec![0.4, 0.25], 0.9)
                } else {
                    (vec![], 1.0)
                };
                models_v2.push(make_model(h_id, i, mean, std, c, r));
            }
        }

        let lp2 = PrecomputedPar::build(&models_v2, &stages, &hydro_ids, None).unwrap();

        // Stage 12: lag-1 = stage 11 (exact match, season 11).
        // Stage 0: lag-1 = stage -1 (season fallback, season 11).
        // Both should produce the same psi[0] since they reference
        // the same season's std.
        for h_idx in 0..2 {
            let psi_s0 = lp2.psi_slice(0, h_idx);
            let psi_s12 = lp2.psi_slice(12, h_idx);

            assert!(
                (psi_s0[0] - psi_s12[0]).abs() < 1e-10,
                "h{h_idx}: stage 0 psi[0] ({}) should equal stage 12 psi[0] ({})",
                psi_s0[0],
                psi_s12[0]
            );
            assert!(
                (psi_s0[1] - psi_s12[1]).abs() < 1e-10,
                "h{h_idx}: stage 0 psi[1] ({}) should equal stage 12 psi[1] ({})",
                psi_s0[1],
                psi_s12[1]
            );
        }
    }

    // -----------------------------------------------------------------------
    // Non-January start: season offset regression test
    // -----------------------------------------------------------------------

    /// Build 12 monthly stages starting at March (`season_id`=2).
    ///
    /// Stage 0 → season 2 (Mar), stage 1 → season 3 (Apr), ...,
    /// stage 9 → season 11 (Dec), stage 10 → season 0 (Jan),
    /// stage 11 → season 1 (Feb).
    fn make_march_start_stages() -> Vec<Stage> {
        (0..12_usize)
            .map(|i| make_stage(i, i32::try_from(i).unwrap(), Some((i + 2) % 12)))
            .collect()
    }

    /// Build inflow models matching March-start stages.
    ///
    /// Stats follow the same `mean = 100 + season*10, std = 20 + season*2`
    /// formula but are keyed by `stage_id` (not `season_id`). The underlying
    /// season stats are resolved via the stage's `season_id`.
    fn make_march_start_models(
        hydro_id: i32,
        ar_stage: i32,
        ar_coeffs: &[f64],
        residual: f64,
    ) -> Vec<InflowModel> {
        (0..12_i32)
            .map(|i| {
                #[allow(clippy::cast_sign_loss)]
                let season = ((i + 2) % 12) as usize;
                let mean = season_mean(season);
                let std = season_std(season);
                let (c, r) = if i == ar_stage {
                    (ar_coeffs.to_vec(), residual)
                } else {
                    (vec![], 1.0)
                };
                make_model(hydro_id, i, mean, std, c, r)
            })
            .collect()
    }

    #[test]
    fn march_start_ar1_resolves_to_february() {
        // Stage 0 = March (season 2), AR(1). Lag-1 → stage -1 → season 1 (Feb).
        // Without the season offset fix, stage -1 resolves to season 11 (Dec).
        let stages = make_march_start_stages();
        let models = make_march_start_models(1, 0, &[0.5], 0.9);

        let lp = PrecomputedPar::build(&models, &stages, &[EntityId(1)], None).unwrap();

        // Stage 0 is season 2 (March): mean=120, std=24.
        let s_mar = season_std(2); // 24.0
        let mu_mar = season_mean(2); // 120.0

        // Lag-1 should be February (season 1): mean=110, std=22.
        let s_feb = season_std(1); // 22.0
        let mu_feb = season_mean(1); // 110.0

        let expected_psi = 0.5 * s_mar / s_feb;
        let expected_base = mu_mar - expected_psi * mu_feb;

        assert!(
            (lp.psi_slice(0, 0)[0] - expected_psi).abs() < 1e-10,
            "psi[0] should use Feb (season 1, std=22), got {} (expected {})",
            lp.psi_slice(0, 0)[0],
            expected_psi
        );
        assert!(
            (lp.deterministic_base(0, 0) - expected_base).abs() < 1e-10,
            "base should use Feb mean; expected {expected_base}, got {}",
            lp.deterministic_base(0, 0)
        );

        // Verify it does NOT match the old (wrong) December resolution.
        let s_dec = season_std(11); // 42.0
        let wrong_psi = 0.5 * s_mar / s_dec;
        assert!(
            (lp.psi_slice(0, 0)[0] - wrong_psi).abs() > 0.1,
            "psi should NOT match December fallback ({wrong_psi})"
        );
    }

    #[test]
    fn march_start_ar6_all_lags_use_correct_seasons() {
        // Stage 0 = March (season 2), AR(6). Lags 1-6 → stages -1..-6.
        // Correct seasons: 1 (Feb), 0 (Jan), 11 (Dec), 10 (Nov), 9 (Oct), 8 (Sep).
        let stages = make_march_start_stages();
        let psi_star = [0.3, 0.2, 0.15, 0.1, 0.08, 0.05];
        let models = make_march_start_models(1, 0, &psi_star, 0.85);

        let lp = PrecomputedPar::build(&models, &stages, &[EntityId(1)], None).unwrap();

        let s_mar = season_std(2);
        let expected_lag_seasons = [1, 0, 11, 10, 9, 8]; // Feb, Jan, Dec, Nov, Oct, Sep

        let psi = lp.psi_slice(0, 0);
        for (lag, &expected_season) in expected_lag_seasons.iter().enumerate() {
            let expected_psi = psi_star[lag] * s_mar / season_std(expected_season);
            assert!(
                (psi[lag] - expected_psi).abs() < 1e-10,
                "lag {}: expected season {expected_season} (psi={expected_psi:.6}), got psi={:.6}",
                lag + 1,
                psi[lag]
            );
        }

        // Deterministic base.
        let mu_mar = season_mean(2);
        let mut expected_base = mu_mar;
        for (lag, &expected_season) in expected_lag_seasons.iter().enumerate() {
            expected_base -= psi[lag] * season_mean(expected_season);
        }
        assert!(
            (lp.deterministic_base(0, 0) - expected_base).abs() < 1e-10,
            "base: expected {expected_base}, got {}",
            lp.deterministic_base(0, 0)
        );
    }

    #[test]
    fn march_start_parity_with_january_start_at_later_stage() {
        // When all lags are within the study period, the season offset should
        // not matter because the exact-match path (tier 1) is used.
        // Stage 6 (September in March-start) with AR(2): lags hit stages 5, 4
        // (both in study). This should match a January-start study at the
        // same season.
        let march_stages = make_march_start_stages();
        let march_models = make_march_start_models(1, 6, &[0.4, 0.25], 0.9);

        let jan_stages = make_monthly_stages();
        // Stage 8 in January-start is also September (season 8).
        // Give it the same AR coefficients at the equivalent stage.
        let jan_models = make_monthly_models(1, 8, &[0.4, 0.25], 0.9);

        let lp_mar =
            PrecomputedPar::build(&march_models, &march_stages, &[EntityId(1)], None).unwrap();
        let lp_jan = PrecomputedPar::build(&jan_models, &jan_stages, &[EntityId(1)], None).unwrap();

        // Both should have identical psi and base for their September stage
        // since all lags are resolved via exact match.
        assert!(
            (lp_mar.psi_slice(6, 0)[0] - lp_jan.psi_slice(8, 0)[0]).abs() < 1e-10,
            "September psi[0] should match: march={}, jan={}",
            lp_mar.psi_slice(6, 0)[0],
            lp_jan.psi_slice(8, 0)[0]
        );
        assert!(
            (lp_mar.deterministic_base(6, 0) - lp_jan.deterministic_base(8, 0)).abs() < 1e-10,
            "September base should match: march={}, jan={}",
            lp_mar.deterministic_base(6, 0),
            lp_jan.deterministic_base(8, 0)
        );
    }

    // -----------------------------------------------------------------------
    // PAR(p)-A annual component tests
    // -----------------------------------------------------------------------

    use cobre_core::scenario::AnnualComponent;

    /// Build an `InflowModel` carrying a `Some(AnnualComponent)`.
    fn make_model_with_annual(
        hydro_id: i32,
        stage_id: i32,
        mean: f64,
        std: f64,
        coeffs: Vec<f64>,
        residual_ratio: f64,
        annual: AnnualComponent,
    ) -> InflowModel {
        crate::test_support::make_inflow_model(InflowModelSpec {
            hydro_id,
            stage_id,
            mean_m3s: mean,
            std_m3s: std,
            ar_coefficients: coeffs,
            residual_std_ratio: residual_ratio,
            annual: Some(annual),
        })
    }

    #[test]
    fn par_a_off_preserves_classical_max_order() {
        // Single hydro AR(2), all annual: None. Verifies that the classical
        // path is preserved bit-for-bit when no annual component is present.
        let stages = vec![make_stage(0, 0, Some(0))];
        let pre_models = vec![
            make_model(1, -2, 100.0, 30.0, vec![], 1.0),
            make_model(1, -1, 100.0, 30.0, vec![], 1.0),
        ];
        let study_model = make_model(1, 0, 100.0, 30.0, vec![0.3, 0.2], 0.9);
        let mut all_models = pre_models;
        all_models.push(study_model);

        let lp = PrecomputedPar::build(&all_models, &stages, &[EntityId(1)], None).unwrap();

        assert_eq!(lp.max_order(), 2, "classical max_order should be 2");
        assert_eq!(
            lp.psi_slice(0, 0).len(),
            2,
            "psi_slice length should equal max_order"
        );
    }

    #[test]
    fn par_a_on_widens_max_order_to_12() {
        // Same fixture but with one hydro having annual: Some(_).
        // max_order must widen to 12 and every (s, h) psi_slice has len 12.
        let stages = vec![make_stage(0, 0, Some(0))];
        let pre_models = vec![
            make_model(1, -2, 100.0, 30.0, vec![], 1.0),
            make_model(1, -1, 100.0, 30.0, vec![], 1.0),
        ];
        let ann = AnnualComponent {
            coefficient: 0.5,
            mean_m3s: 100.0,
            std_m3s: 30.0,
        };
        let study_model = make_model_with_annual(1, 0, 100.0, 30.0, vec![0.3, 0.2], 0.9, ann);
        let mut all_models = pre_models;
        all_models.push(study_model);

        let lp = PrecomputedPar::build(&all_models, &stages, &[EntityId(1)], None).unwrap();

        assert_eq!(
            lp.max_order(),
            12,
            "annual component must widen max_order to 12"
        );
        assert_eq!(lp.psi_slice(0, 0).len(), 12);
    }

    #[test]
    fn par_a_on_hand_computed_uniform_seasons() {
        // Single stage, single hydro. All seasonal stats are equal:
        //   μ_m = s_m = σ^A = 30, ψ = 0.6, φ_1 = 0.3.
        //
        // Expected:
        //   ψ̂ = 0.6 * 30 / 30 = 0.6
        //   psi[0] = 0.3 * 30/30 + 0.6/12 = 0.3 + 0.05 = 0.35
        //   psi[1..12] = 0.6/12 = 0.05
        //   base = 100 - Σ_{j=0..11} psi[j] * 100
        //        = 100 - 100 * (0.35 + 11*0.05)
        //        = 100 - 100 * 0.9 = 10
        let study_stage = make_stage(0, 0, Some(0));
        let pre_study = make_model(1, -1, 100.0, 30.0, vec![], 1.0);
        let ann = AnnualComponent {
            coefficient: 0.6,
            mean_m3s: 100.0,
            std_m3s: 30.0,
        };
        let study_model = make_model_with_annual(1, 0, 100.0, 30.0, vec![0.3], 0.954, ann);

        let lp = PrecomputedPar::build(
            &[pre_study, study_model],
            &[study_stage],
            &[EntityId(1)],
            None,
        )
        .unwrap();

        let psi = lp.psi_slice(0, 0);
        assert_eq!(psi.len(), 12);
        assert!(
            (psi[0] - 0.35).abs() < 1e-10,
            "psi[0]: expected 0.35, got {}",
            psi[0]
        );
        for (j, &v) in psi[1..].iter().enumerate() {
            assert!(
                (v - 0.05).abs() < 1e-10,
                "psi[{}]: expected 0.05, got {}",
                j + 1,
                v
            );
        }
        let expected_base = 10.0_f64;
        assert!(
            (lp.deterministic_base(0, 0) - expected_base).abs() < 1e-10,
            "deterministic_base: expected {expected_base}, got {}",
            lp.deterministic_base(0, 0)
        );
    }

    #[test]
    fn par_a_on_12_seasons_with_season_fallback() {
        // 12 monthly stages, AR(1) at stage 0 with annual: Some.
        // Stage 0 has σ_0 = 20, σ^A = 18, ψ = 0.4, φ_1 = 0.5.
        // Lag-1 resolves to season 11 via fallback: σ_{11} = 42.
        //
        // psi[0] = 0.5 * 20/42 + (0.4 * 20/18) / 12
        // psi[1..12] = (0.4 * 20/18) / 12
        let stages = make_monthly_stages();

        let ann = AnnualComponent {
            coefficient: 0.4,
            mean_m3s: 100.0,
            std_m3s: 18.0,
        };

        // Build monthly models with the special stage 0 carrying annual.
        let models: Vec<InflowModel> = (0..12_i32)
            .map(|i| {
                let fi = f64::from(i);
                let mean = 100.0 + fi * 10.0;
                let std = 20.0 + fi * 2.0;
                if i == 0 {
                    make_model_with_annual(1, i, mean, std, vec![0.5], 0.9, ann.clone())
                } else {
                    make_model(1, i, mean, std, vec![], 1.0)
                }
            })
            .collect();

        let lp = PrecomputedPar::build(&models, &stages, &[EntityId(1)], None).unwrap();

        let s_0 = 20.0_f64;
        let s_11 = 42.0_f64;
        let psi_hat = 0.4 * s_0 / 18.0;
        let phi_hat_0 = 0.5 * s_0 / s_11;
        let expected_psi_0 = phi_hat_0 + psi_hat / 12.0;
        let expected_psi_rest = psi_hat / 12.0;

        let psi = lp.psi_slice(0, 0);
        assert_eq!(psi.len(), 12);
        assert!(
            (psi[0] - expected_psi_0).abs() < 1e-10,
            "psi[0]: expected {expected_psi_0}, got {}",
            psi[0]
        );
        for (j, &v) in psi[1..].iter().enumerate() {
            assert!(
                (v - expected_psi_rest).abs() < 1e-10,
                "psi[{}]: expected {expected_psi_rest}, got {}",
                j + 1,
                v
            );
        }
    }

    #[test]
    fn par_a_mixed_some_and_none_hydros() {
        // Two hydros: hydro 1 has annual: Some, hydro 2 has annual: None.
        // Stride 12 applies to both. For hydro 2: psi[0..ar_order] are
        // classical φ̂ values; psi[ar_order..12] are 0.0.
        let stages = vec![make_stage(0, 0, Some(0))];
        let pre_models = vec![
            make_model(1, -1, 100.0, 30.0, vec![], 1.0),
            make_model(2, -1, 80.0, 20.0, vec![], 1.0),
        ];
        let ann = AnnualComponent {
            coefficient: 0.5,
            mean_m3s: 100.0,
            std_m3s: 30.0,
        };
        // hydro 1: AR(1) + annual
        let model_h1 = make_model_with_annual(1, 0, 100.0, 30.0, vec![0.3], 0.9, ann);
        // hydro 2: AR(1), no annual
        let model_h2 = make_model(2, 0, 80.0, 20.0, vec![0.4], 0.9);
        let mut all_models = pre_models;
        all_models.push(model_h1);
        all_models.push(model_h2);

        let hydro_ids = [EntityId(1), EntityId(2)];
        let lp = PrecomputedPar::build(&all_models, &stages, &hydro_ids, None).unwrap();

        assert_eq!(lp.max_order(), 12);

        // Hydro 1 (h_idx=0): psi[0] = φ̂_1 + ψ̂/12, psi[1..12] = ψ̂/12.
        let psi_h1 = lp.psi_slice(0, 0);
        assert_eq!(psi_h1.len(), 12);
        // ψ̂ = ψ · s_m / σ^A = 0.5 * 30/30 = 0.5
        let annual_coeff_h1 = 0.5_f64 * 30.0 / 30.0;
        // φ̂_1 = φ_1 · s_m / s_{m-1} = 0.3 * 30/30 = 0.3
        let ar_coeff_h1 = 0.3_f64 * 30.0 / 30.0;
        assert!(
            (psi_h1[0] - (ar_coeff_h1 + annual_coeff_h1 / 12.0)).abs() < 1e-10,
            "h1 psi[0]: expected {}, got {}",
            ar_coeff_h1 + annual_coeff_h1 / 12.0,
            psi_h1[0]
        );
        for (j, &v) in psi_h1[1..].iter().enumerate() {
            assert!(
                (v - annual_coeff_h1 / 12.0).abs() < 1e-10,
                "h1 psi[{}]: expected {}, got {}",
                j + 1,
                annual_coeff_h1 / 12.0,
                v
            );
        }

        // Hydro 2 (h_idx=1): no annual, classical φ̂ at lag 0, zeros at [1..12].
        let psi_h2 = lp.psi_slice(0, 1);
        assert_eq!(psi_h2.len(), 12);
        // φ̂_1 = φ_1 · s_m / s_{m-1} = 0.4 * 20/20 = 0.4
        let ar_coeff_h2 = 0.4_f64 * 20.0 / 20.0;
        assert!(
            (psi_h2[0] - ar_coeff_h2).abs() < 1e-10,
            "h2 psi[0]: expected {ar_coeff_h2}, got {}",
            psi_h2[0]
        );
        for (j, &v) in psi_h2[1..].iter().enumerate() {
            assert!(
                v.abs() < 1e-12,
                "h2 psi[{}]: expected 0.0, got {}",
                j + 1,
                v
            );
        }

        // deterministic_base for hydro 2 must equal the classical formula
        // summed over the same wider psi_slice (trailing zeros contribute zero).
        let expected_base_h2 = 80.0 - ar_coeff_h2 * 80.0;
        assert!(
            (lp.deterministic_base(0, 1) - expected_base_h2).abs() < 1e-10,
            "h2 base: expected {expected_base_h2}, got {}",
            lp.deterministic_base(0, 1)
        );
    }

    #[test]
    fn par_a_zero_annual_std_yields_zero_contribution() {
        // annual.std_m3s == 0.0: ψ̂ contribution must be 0.0 for all 12 lags.
        // No error should be raised.
        let study_stage = make_stage(0, 0, Some(0));
        let pre_study = make_model(1, -1, 100.0, 30.0, vec![], 1.0);
        let ann = AnnualComponent {
            coefficient: 0.5,
            mean_m3s: 100.0,
            std_m3s: 0.0, // zero std: ψ̂ contribution is zero
        };
        let study_model = make_model_with_annual(1, 0, 100.0, 30.0, vec![0.3], 0.9, ann);

        let lp = PrecomputedPar::build(
            &[pre_study, study_model],
            &[study_stage],
            &[EntityId(1)],
            None,
        )
        .unwrap();

        // psi[0] = φ̂_1 + 0/12 = 0.3 * 30/30 = 0.3 (no annual contribution).
        // psi[1..12] = 0/12 = 0.0.
        let psi = lp.psi_slice(0, 0);
        assert_eq!(psi.len(), 12);
        assert!(
            (psi[0] - 0.3).abs() < 1e-12,
            "psi[0]: expected 0.3, got {}",
            psi[0]
        );
        for (j, &v) in psi[1..].iter().enumerate() {
            assert!(
                v.abs() < 1e-12,
                "psi[{}]: annual contribution with zero std must be 0.0, got {}",
                j + 1,
                v
            );
        }
    }

    #[test]
    fn par_a_declaration_order_invariance() {
        // Two hydros with PAR-A active. Result must be identical regardless
        // of the order in which inflow models appear in the input slice.
        let stage = make_stage(0, 0, Some(0));
        let ann1 = AnnualComponent {
            coefficient: 0.4,
            mean_m3s: 100.0,
            std_m3s: 25.0,
        };
        let ann2 = AnnualComponent {
            coefficient: 0.6,
            mean_m3s: 200.0,
            std_m3s: 40.0,
        };

        let model_h3 = make_model_with_annual(3, 0, 100.0, 30.0, vec![], 1.0, ann1);
        let model_h5 = make_model_with_annual(5, 0, 200.0, 40.0, vec![], 1.0, ann2);

        let hydro_ids = [EntityId(3), EntityId(5)];
        let stages = [stage];

        // Forward order.
        let lp_fwd = PrecomputedPar::build(
            &[model_h3.clone(), model_h5.clone()],
            &stages,
            &hydro_ids,
            None,
        )
        .unwrap();
        // Reversed order.
        let lp_rev =
            PrecomputedPar::build(&[model_h5, model_h3], &stages, &hydro_ids, None).unwrap();

        assert_eq!(lp_fwd.max_order(), 12);
        assert_eq!(lp_rev.max_order(), 12);

        for h in 0..2 {
            let psi_fwd = lp_fwd.psi_slice(0, h);
            let psi_rev = lp_rev.psi_slice(0, h);
            for (j, (&vf, &vr)) in psi_fwd.iter().zip(psi_rev).enumerate() {
                assert!(
                    (vf - vr).abs() < 1e-12,
                    "h={h} psi[{j}] differs: fwd={vf}, rev={vr}"
                );
            }
            assert!(
                (lp_fwd.deterministic_base(0, h) - lp_rev.deterministic_base(0, h)).abs() < 1e-12,
                "h={h} base differs"
            );
        }
    }

    #[test]
    fn par_a_cross_year_mean_fallback_for_lags_past_p() {
        // 12 monthly stages, AR(2) at stage 0 with annual: Some.
        // Lags j ∈ [2, 12) must use season-fallback statistics for μ_{m-j-1}
        // and σ_{m-j-1}. The deterministic_base must correctly subtract
        // psi[j] * μ_{m-j-1} for all 12 lags.
        let stages = make_monthly_stages();

        let ann = AnnualComponent {
            coefficient: 0.5,
            mean_m3s: 100.0,
            std_m3s: 30.0,
        };
        // Stage 0: AR(2) + annual. All other stages: AR(0).
        let models: Vec<InflowModel> = (0..12_i32)
            .map(|i| {
                let fi = f64::from(i);
                let mean = 100.0 + fi * 10.0;
                let std = 20.0 + fi * 2.0;
                if i == 0 {
                    make_model_with_annual(1, i, mean, std, vec![0.4, 0.25], 0.9, ann.clone())
                } else {
                    make_model(1, i, mean, std, vec![], 1.0)
                }
            })
            .collect();

        let lp = PrecomputedPar::build(&models, &stages, &[EntityId(1)], None).unwrap();

        let s_0 = season_std(0); // 20.0
        let annual_coeff = 0.5 * s_0 / 30.0;

        // j=0: lag-1 → season 11 (fallback)
        let ar_contrib_0 = 0.4 * s_0 / season_std(11);
        let expected_psi_0 = ar_contrib_0 + annual_coeff / 12.0;

        // j=1: lag-2 → season 10 (fallback)
        let ar_contrib_1 = 0.25 * s_0 / season_std(10);
        let expected_psi_1 = ar_contrib_1 + annual_coeff / 12.0;

        let psi = lp.psi_slice(0, 0);
        assert_eq!(psi.len(), 12);
        assert!(
            (psi[0] - expected_psi_0).abs() < 1e-10,
            "psi[0]: expected {expected_psi_0}, got {}",
            psi[0]
        );
        assert!(
            (psi[1] - expected_psi_1).abs() < 1e-10,
            "psi[1]: expected {expected_psi_1}, got {}",
            psi[1]
        );

        // j ∈ [2, 12): only ψ̂/12, season-fallback lag stats.
        for (j, &v) in psi[2..].iter().enumerate() {
            let expected = annual_coeff / 12.0;
            assert!(
                (v - expected).abs() < 1e-10,
                "psi[{}]: expected {expected}, got {}",
                j + 2,
                v
            );
        }

        // Verify deterministic_base accounts for all 12 lags correctly.
        let mu_0 = season_mean(0);
        let mut expected_base = mu_0;
        for (j, &psi_j) in psi.iter().enumerate() {
            // Lag j+1 corresponds to stage_id = -1 - j, which resolves
            // to season 11 - j (wrapping) via the fallback path.
            let lag_stage_id = -(i32::try_from(j).unwrap() + 1);
            let lag_season = resolve_season_id(lag_stage_id, 12, 0);
            expected_base -= psi_j * season_mean(lag_season);
        }
        assert!(
            (lp.deterministic_base(0, 0) - expected_base).abs() < 1e-10,
            "base: expected {expected_base}, got {}",
            lp.deterministic_base(0, 0)
        );
    }

    fn dated_stage(id: i32, start_date: NaiveDate, end_date: NaiveDate, season: usize) -> Stage {
        cobre_core::test_support::make_stage(StageSpec {
            id,
            start_date,
            end_date,
            season_id: Some(season),
            ..Default::default()
        })
    }

    fn sparse_ring_study_stages() -> Vec<Stage> {
        vec![
            dated_stage(0, date(2026, 1, 1), date(2026, 2, 1), 0),
            dated_stage(1, date(2026, 2, 1), date(2026, 3, 1), 1),
            dated_stage(2, date(2026, 3, 1), date(2026, 4, 1), 2),
            dated_stage(3, date(2026, 4, 1), date(2026, 7, 1), 12),
            dated_stage(4, date(2026, 7, 1), date(2026, 10, 1), 13),
        ]
    }

    #[test]
    fn pre_study_lag_statistics_come_from_the_lowest_stage_id_of_the_season() {
        let monthly = monthly_season_map(MonthlyLabels::ZeroBased);
        let stages: Vec<Stage> = (0..96_i32)
            .map(|id| {
                let start = date(2026 + id / 12, u32::try_from(id % 12 + 1).unwrap(), 1);
                let end = start.checked_add_months(Months::new(1)).unwrap();
                dated_stage(id, start, end, usize::try_from(id % 12).unwrap())
            })
            .collect();
        let mut models = vec![make_model(1, 0, 100.0, 10.0, vec![0.5], 1.0)];
        for (year, std) in (0..8).zip([40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0]) {
            models.push(make_model(1, 11 + 12 * year, 100.0, std, vec![], 1.0));
        }

        for build in 0..16 {
            let lp =
                PrecomputedPar::build(&models, &stages, &[EntityId(1)], Some(&monthly)).unwrap();
            let psi = lp.psi_slice(0, 0)[0];
            assert!(
                f64_bits_eq(psi, 0.125),
                "build {build}: lag 1 of stage 0 must use the std of stage 11, got psi = {psi}"
            );
        }
    }

    #[test]
    fn declared_pre_study_stage_supplies_lag_statistics_for_its_season() {
        let ring = sparse_ring_season_map();
        let mut stages = vec![dated_stage(-1, date(2025, 7, 1), date(2025, 10, 1), 13)];
        stages.extend(sparse_ring_study_stages());
        let models = vec![
            make_model(1, -1, 100.0, 10.0, vec![], 1.0),
            make_model(1, 0, 100.0, 20.0, vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.5], 1.0),
            make_model(1, 1, 100.0, 30.0, vec![], 1.0),
            make_model(1, 2, 100.0, 30.0, vec![], 1.0),
            make_model(1, 3, 100.0, 30.0, vec![], 1.0),
            make_model(1, 4, 100.0, 40.0, vec![], 1.0),
        ];

        let lp = PrecomputedPar::build(&models, &stages, &[EntityId(1)], Some(&ring)).unwrap();

        let psi = lp.psi_slice(0, 0)[5];
        assert!(
            f64_bits_eq(psi, 1.0),
            "lag 6 of stage 0 (season 13) must use the std of declared stage -1, got psi = {psi}"
        );
        assert_eq!(lp.n_stages(), 5);
    }

    fn monthly_then_quarterly_stages() -> Vec<Stage> {
        vec![
            dated_stage(0, date(2024, 4, 1), date(2024, 5, 1), 3),
            dated_stage(1, date(2024, 5, 1), date(2024, 6, 1), 4),
            dated_stage(2, date(2024, 6, 1), date(2024, 7, 1), 5),
            dated_stage(3, date(2024, 7, 1), date(2024, 10, 1), 12),
            dated_stage(4, date(2024, 10, 1), date(2025, 1, 1), 13),
            dated_stage(5, date(2025, 1, 1), date(2025, 4, 1), 14),
            dated_stage(6, date(2025, 4, 1), date(2025, 7, 1), 15),
        ]
    }

    fn monthly_then_quarterly_models() -> Vec<InflowModel> {
        vec![
            make_model(1, 0, 100.0, 20.0, vec![], 1.0),
            make_model(1, 1, 100.0, 20.0, vec![], 1.0),
            make_model(1, 2, 50.0, 10.0, vec![], 1.0),
            make_model(1, 3, 100.0, 30.0, vec![0.5], 1.0),
            make_model(1, 4, 100.0, 30.0, vec![0.5, 0.5], 1.0),
            make_model(1, 5, 100.0, 20.0, vec![], 1.0),
            make_model(1, 6, 200.0, 60.0, vec![], 1.0),
        ]
    }

    #[test]
    fn cross_resolution_lags_read_the_coarse_season_statistics() {
        let layered = monthly_quarterly_season_map();
        let lp = PrecomputedPar::build(
            &monthly_then_quarterly_models(),
            &monthly_then_quarterly_stages(),
            &[EntityId(1)],
            Some(&layered),
        )
        .unwrap();

        let stage_3 = (lp.psi_slice(3, 0)[0], lp.deterministic_base(3, 0));
        let stage_4 = (lp.psi_slice(4, 0), lp.deterministic_base(4, 0));
        assert!(
            f64_bits_eq(stage_3.0, 0.25)
                && f64_bits_eq(stage_3.1, 50.0)
                && matches!(stage_4.0, [lag_1, lag_2]
                    if f64_bits_eq(*lag_1, 0.5) && f64_bits_eq(*lag_2, 0.25))
                && f64_bits_eq(stage_4.1, 0.0),
            "quarterly lags reaching back into the monthly stages must use the statistics of \
             season 15: got stage 3 (lag-1 psi, base) = {stage_3:?}, \
             stage 4 (psi, base) = {stage_4:?}"
        );
    }

    #[test]
    fn single_group_maps_keep_the_lag_stage_statistics() {
        let ring = sparse_ring_season_map();
        let ring_models = vec![
            make_model(1, 0, 100.0, 10.0, vec![], 1.0),
            make_model(1, 1, 100.0, 20.0, vec![], 1.0),
            make_model(1, 2, 100.0, 40.0, vec![], 1.0),
            make_model(1, 3, 100.0, 30.0, vec![0.5], 1.0),
            make_model(1, 4, 100.0, 50.0, vec![], 1.0),
        ];
        let lp = PrecomputedPar::build(
            &ring_models,
            &sparse_ring_study_stages(),
            &[EntityId(1)],
            Some(&ring),
        )
        .unwrap();
        let psi = lp.psi_slice(3, 0)[0];
        assert!(
            f64_bits_eq(psi, 0.5 * 30.0 / 40.0),
            "lag 1 of the Apr-Jun stage must use March's std, got psi = {psi}"
        );

        let monthly = monthly_season_map(MonthlyLabels::ZeroBased);
        let stages: Vec<Stage> = (0..14_i32)
            .map(|id| {
                let start = date(2024 + id / 12, u32::try_from(id % 12 + 1).unwrap(), 1);
                let end = start.checked_add_months(Months::new(1)).unwrap();
                dated_stage(id, start, end, usize::try_from(id % 12).unwrap())
            })
            .collect();
        let monthly_models = vec![
            make_model(1, 0, 100.0, 10.0, vec![], 1.0),
            make_model(1, 12, 100.0, 40.0, vec![], 1.0),
            make_model(1, 13, 100.0, 30.0, vec![0.5], 1.0),
        ];
        let lp = PrecomputedPar::build(&monthly_models, &stages, &[EntityId(1)], Some(&monthly))
            .unwrap();
        let psi = lp.psi_slice(13, 0)[0];
        assert!(
            f64_bits_eq(psi, 0.5 * 30.0 / 40.0),
            "lag 1 of February 2025 must use January 2025's std, got psi = {psi}"
        );
    }

    #[test]
    fn cross_resolution_lag_without_coarse_statistics_keeps_the_lag_stage_statistics() {
        let layered = monthly_quarterly_season_map();
        let mut stages = monthly_then_quarterly_stages();
        stages.truncate(4);
        let mut models = monthly_then_quarterly_models();
        models.truncate(4);
        let lp = PrecomputedPar::build(&models, &stages, &[EntityId(1)], Some(&layered)).unwrap();

        let stage_3 = (lp.psi_slice(3, 0)[0], lp.deterministic_base(3, 0));
        assert!(
            f64_bits_eq(stage_3.0, 1.5) && f64_bits_eq(stage_3.1, 25.0),
            "with no stage carrying season 15, lag 1 of stage 3 must use June's statistics: \
             got (psi, base) = {stage_3:?}"
        );
    }

    #[test]
    fn pre_study_lags_keep_their_stage_statistics_across_levels() {
        let layered = monthly_quarterly_season_map();
        let stages = vec![
            dated_stage(-1, date(2024, 6, 1), date(2024, 7, 1), 5),
            dated_stage(0, date(2024, 7, 1), date(2024, 10, 1), 12),
            dated_stage(1, date(2024, 10, 1), date(2025, 1, 1), 13),
            dated_stage(2, date(2025, 1, 1), date(2025, 4, 1), 14),
            dated_stage(3, date(2025, 4, 1), date(2025, 7, 1), 15),
        ];
        let models = vec![
            make_model(1, -1, 50.0, 10.0, vec![], 1.0),
            make_model(1, 0, 100.0, 30.0, vec![0.5], 1.0),
            make_model(1, 1, 100.0, 30.0, vec![], 1.0),
            make_model(1, 2, 100.0, 30.0, vec![], 1.0),
            make_model(1, 3, 200.0, 60.0, vec![], 1.0),
        ];
        let lp = PrecomputedPar::build(&models, &stages, &[EntityId(1)], Some(&layered)).unwrap();

        let stage_0 = (lp.psi_slice(0, 0)[0], lp.deterministic_base(0, 0));
        assert!(
            f64_bits_eq(stage_0.0, 1.5) && f64_bits_eq(stage_0.1, 25.0),
            "lag 1 of stage 0 must use the statistics of declared stage -1, not season 15's: \
             got (psi, base) = {stage_0:?}"
        );
    }
}
