//! Pre-computation of PAR model coefficient arrays for LP RHS patching.
//!
//! This module provides [`PrecomputedPar`], the performance-adapted cache built
//! once during initialization from raw [`InflowModel`] parameters. It exposes flat,
//! contiguous arrays in stage-major layout so the calling algorithm can patch LP
//! right-hand sides without per-scenario recomputation.
//!
//! ## Array layout
//!
//! All two-dimensional arrays use **stage-major** layout:
//! `array[stage * n_series + series_element]`.
//!
//! The three-dimensional `psi` array uses **stage-major, series-minor, lag-innermost**:
//! `psi[stage * n_series * max_order + series_element * max_order + lag]`.
//!
//! This layout is optimal for sequential stage iteration within a scenario trajectory:
//! all per-stage data for every series element is contiguous in memory, maximizing
//! cache utilization during forward/backward LP passes.
//!
//! ## Coefficient conversion
//!
//! Input `ar_coefficients` in [`InflowModel`] are stored in **standardized form**
//! (ψ\*, the direct Yule-Walker output). This module converts them to
//! **original-unit** form at build time:
//!
//! ```text
//! ψ_{m,ℓ} = ψ*_{m,ℓ} · s_m / s_{m-ℓ}
//! ```
//!
//! where `s_m` is `std_m3s` for the current stage's season and `s_{m-ℓ}` is
//! `std_m3s` for the season `ℓ` stages prior.

use std::collections::{HashMap, HashSet};

use cobre_core::{EntityId, scenario::InflowModel, temporal::Stage};

use crate::StochasticError;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Resolve a stage ID to a season ID using modular arithmetic on the season
/// cycle length. Works for both positive and negative stage IDs.
///
/// For a monthly system (`n_seasons = 12`):
/// - `stage_id = -1`  -> season 11 (December)
/// - `stage_id = -12` -> season 0  (January)
/// - `stage_id = -13` -> season 11 (December, wraps)
/// - `stage_id = 5`   -> season 5
fn resolve_season_id(stage_id: i32, n_seasons: usize) -> usize {
    debug_assert!(n_seasons > 0, "n_seasons must be positive");
    // n_seasons is always small (12 for monthly, 52 for weekly) so truncation
    // from usize to i32 is safe in practice. The debug_assert above guards
    // against zero; values > i32::MAX are not realistic for season counts.
    #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
    let n = n_seasons as i32;
    #[allow(clippy::cast_sign_loss)]
    let result = ((stage_id % n + n) % n) as usize;
    result
}

// ---------------------------------------------------------------------------
// PrecomputedPar
// ---------------------------------------------------------------------------

/// Cache-friendly PAR(p) model data for LP RHS patching.
///
/// Built once during initialization from raw [`InflowModel`] parameters.
/// Consumed read-only during iterative optimization.
///
/// All arrays use stage-major layout: outer dimension is stage index,
/// inner dimension is series element index (sorted by canonical entity ID order).
/// This layout is optimal for sequential stage iteration within a
/// scenario trajectory.
///
/// See the [module documentation](self) for the derivation of each cached
/// component and the coefficient conversion formula.
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
/// };
///
/// let lp = PrecomputedPar::build(&[model], &[stage], &[EntityId(1)]).unwrap();
/// assert_eq!(lp.n_stages(), 1);
/// assert_eq!(lp.n_hydros(), 1);
/// assert!((lp.deterministic_base(0, 0) - 100.0).abs() < f64::EPSILON);
/// ```
#[derive(Debug)]
pub struct PrecomputedPar {
    /// Deterministic base `b_{h,m(t)} = μ_{m(t)} - Σ_ℓ ψ_{m(t),ℓ} · μ_{m(t-ℓ)}`.
    /// Flat array indexed as `[stage * n_series + series_element]`.
    /// Length: `n_stages * n_series`.
    deterministic_base: Box<[f64]>,

    /// Residual standard deviation `σ_{m(t)}` per (stage, series element).
    /// Derived as `σ = s_m · residual_std_ratio`.
    /// Flat array indexed as `[stage * n_series + series_element]`.
    /// Length: `n_stages * n_series`.
    sigma: Box<[f64]>,

    /// AR lag coefficients `ψ_{m(t),ℓ}` in original units per (stage, series element, lag).
    /// Flat array indexed as `[stage * n_series * max_order + series_element * max_order + lag]`.
    /// Length: `n_stages * n_series * max_order`.
    /// Padded with `0.0` for series elements with `ar_order < max_order`.
    psi: Box<[f64]>,

    /// AR order per series element. Length: `n_series`.
    /// `orders[h]` gives the number of meaningful lags in `psi` for series element `h`.
    orders: Box<[usize]>,

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
    /// - `stages`: study stages sorted by `index` from the system (non-negative IDs).
    /// - `hydro_ids`: canonical sorted entity IDs (determines series element array index order).
    ///
    /// # Errors
    ///
    /// Returns [`StochasticError::InvalidParParameters`] when a required lag stage
    /// does not have a `season_id`, which prevents coefficient unit conversion.
    ///
    /// # Panics
    ///
    /// Does not panic for valid inputs. All indexing is bounds-checked during build.
    pub fn build(
        inflow_models: &[InflowModel],
        stages: &[Stage],
        hydro_ids: &[EntityId],
    ) -> Result<Self, StochasticError> {
        let n_stages = stages.len();
        let n_hydros = hydro_ids.len();

        // Map hydro EntityId → canonical index (0-based, canonical sorted order).
        let hydro_index: HashMap<EntityId, usize> = hydro_ids
            .iter()
            .enumerate()
            .map(|(i, &id)| (id, i))
            .collect();

        // Key inflow models by (hydro_id.0, stage_id) for O(1) lookup.
        let model_map: HashMap<(i32, i32), &InflowModel> = inflow_models
            .iter()
            .map(|m| ((m.hydro_id.0, m.stage_id), m))
            .collect();

        // Determine max AR order across all inflow models.
        let max_order = inflow_models
            .iter()
            .map(InflowModel::ar_order)
            .max()
            .unwrap_or(0);

        // Per-hydro AR order (maximum across all stages for that hydro).
        let mut orders = vec![0usize; n_hydros];
        for model in inflow_models {
            if let Some(&h_idx) = hydro_index.get(&model.hydro_id) {
                if model.ar_order() > orders[h_idx] {
                    orders[h_idx] = model.ar_order();
                }
            }
        }

        // Allocate flat output arrays.
        let n2 = n_stages * n_hydros;
        let n3 = n_stages * n_hydros * max_order;
        let mut deterministic_base = vec![0.0f64; n2];
        let mut sigma = vec![0.0f64; n2];
        let mut psi = vec![0.0f64; n3];

        // For lag lookups we also need stages before the study horizon
        // (pre-study stages have negative IDs and are included in `inflow_models`
        // but NOT in `stages`). We carry all models, so we look them up via the
        // model map using the stage_id value directly.
        //
        // We need a mapping from stage_id → std_m3s for lag computation.
        // Build a map (hydro_index, stage_id) → (mean_m3s, std_m3s).
        let model_stats: HashMap<(usize, i32), (f64, f64)> = inflow_models
            .iter()
            .filter_map(|m| {
                hydro_index
                    .get(&m.hydro_id)
                    .map(|&h_idx| ((h_idx, m.stage_id), (m.mean_m3s, m.std_m3s)))
            })
            .collect();

        // Season-based fallback for pre-study lag stages (negative stage_id).
        //
        // When the exact (h_idx, lag_stage_id) is not in model_stats (because
        // the estimation pipeline or external files did not emit pre-study
        // entries), we resolve lag_stage_id to a season_id via modular
        // arithmetic on the season cycle length, then look up (h_idx, season_id)
        // in season_stats.
        let n_seasons = stages
            .iter()
            .filter_map(|s| s.season_id)
            .collect::<HashSet<_>>()
            .len();

        let stage_to_season: HashMap<i32, usize> = stages
            .iter()
            .filter_map(|s| s.season_id.map(|sid| (s.id, sid)))
            .collect();

        let season_stats: HashMap<(usize, usize), (f64, f64)> = model_stats
            .iter()
            .filter_map(|(&(h_idx, stage_id), &stats)| {
                stage_to_season
                    .get(&stage_id)
                    .map(|&sid| ((h_idx, sid), stats))
            })
            .collect();

        // For converting ψ* → ψ we need std_m3s for the lag stage's season.
        // The lag stage's stage_id is: current_stage.id - l (for lag l).
        // We look up (h_idx, stage_id - l) in model_stats. If that misses,
        // fall back to season_stats via modular arithmetic on n_seasons.

        for (s_idx, stage) in stages.iter().enumerate() {
            let stage_id = stage.id;

            for (h_idx, &hydro_id) in hydro_ids.iter().enumerate() {
                let flat2 = s_idx * n_hydros + h_idx;

                // Look up the InflowModel for this (stage, hydro) pair.
                let model = model_map.get(&(hydro_id.0, stage_id));

                match model {
                    None => {
                        // No model for this pair: deterministic zero inflow.
                        deterministic_base[flat2] = 0.0;
                        sigma[flat2] = 0.0;
                        // psi stays 0.0 (already initialized)
                    }
                    Some(m) => {
                        let s_m = m.std_m3s;
                        let mu_m = m.mean_m3s;
                        let order = m.ar_order();

                        // sigma = s_m * residual_std_ratio
                        sigma[flat2] = s_m * m.residual_std_ratio;

                        // Convert ψ* → ψ and compute the deterministic base.
                        //
                        // ψ_{m,ℓ} = ψ*_{m,ℓ} · s_m / s_{m-ℓ}
                        //
                        // deterministic_base = μ_m - Σ_ℓ ψ_{m,ℓ} · μ_{m-ℓ}
                        let mut base = mu_m;

                        for lag in 0..order {
                            // lag is 0-based: coefficient index 0 corresponds to lag ℓ=1.
                            let lag_stage_id =
                                stage_id - i32::try_from(lag + 1).unwrap_or(i32::MAX);

                            // Two-tier lookup for lag stage statistics:
                            //
                            // Tier 1: exact stage_id match (covers study-stage lags
                            // and any pre-study entries explicitly present in
                            // inflow_models).
                            //
                            // Tier 2: season-based fallback using modular arithmetic.
                            // For negative lag_stage_id, resolve to season_id via:
                            //   season_id = ((lag_stage_id % n) + n) % n
                            // where n = n_seasons (i32 cast).
                            let (mu_lag, s_lag) =
                                if let Some(&stats) = model_stats.get(&(h_idx, lag_stage_id)) {
                                    stats
                                } else if n_seasons > 0 {
                                    let season_id = resolve_season_id(lag_stage_id, n_seasons);
                                    season_stats
                                        .get(&(h_idx, season_id))
                                        .copied()
                                        .unwrap_or((0.0, 0.0))
                                } else {
                                    // No seasons defined -- cannot resolve; zero fallback.
                                    (0.0, 0.0)
                                };

                            // Avoid divide-by-zero: if s_lag == 0.0, the ratio is
                            // undefined. Treat psi as zero (no AR contribution from
                            // that lag). The caller is responsible for validating that
                            // lag stages with AR order > 0 have positive std.
                            let psi_star = m.ar_coefficients[lag];
                            let psi_val = if s_lag == 0.0 {
                                0.0
                            } else {
                                psi_star * s_m / s_lag
                            };

                            // Store in flat 3-D array.
                            let flat3 = s_idx * n_hydros * max_order + h_idx * max_order + lag;
                            psi[flat3] = psi_val;

                            base -= psi_val * mu_lag;
                        }

                        // Verify that we have a valid season_id when AR order > 0.
                        // (Season is needed for upstream validation; here we confirm
                        //  the stage is properly configured.)
                        if order > 0 && stage.season_id.is_none() {
                            return Err(StochasticError::InvalidParParameters {
                                hydro_id: hydro_id.0,
                                stage_id,
                                reason: "stage has AR order > 0 but no season_id; \
                                         cannot perform coefficient unit conversion"
                                    .to_string(),
                            });
                        }

                        deterministic_base[flat2] = base;
                    }
                }
            }
        }

        Ok(Self {
            deterministic_base: deterministic_base.into_boxed_slice(),
            sigma: sigma.into_boxed_slice(),
            psi: psi.into_boxed_slice(),
            orders: orders.into_boxed_slice(),
            n_stages,
            n_hydros,
            max_order,
        })
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Deterministic base `b_{h,m(t)}` for the given stage and series element indices.
    ///
    /// # Panics
    ///
    /// Panics if `stage >= n_stages` or `hydro >= n_hydros`.
    #[must_use]
    pub fn deterministic_base(&self, stage: usize, hydro: usize) -> f64 {
        assert!(
            stage < self.n_stages,
            "stage index {stage} is out of bounds (n_stages = {})",
            self.n_stages
        );
        assert!(
            hydro < self.n_hydros,
            "hydro index {hydro} is out of bounds (n_hydros = {})",
            self.n_hydros
        );
        self.deterministic_base[stage * self.n_hydros + hydro]
    }

    /// Residual standard deviation `σ_{m(t)}` for the given stage and series element indices.
    ///
    /// # Panics
    ///
    /// Panics if `stage >= n_stages` or `hydro >= n_hydros`.
    #[must_use]
    pub fn sigma(&self, stage: usize, hydro: usize) -> f64 {
        assert!(
            stage < self.n_stages,
            "stage index {stage} is out of bounds (n_stages = {})",
            self.n_stages
        );
        assert!(
            hydro < self.n_hydros,
            "hydro index {hydro} is out of bounds (n_hydros = {})",
            self.n_hydros
        );
        self.sigma[stage * self.n_hydros + hydro]
    }

    /// Slice of AR lag coefficients `ψ_{m(t),ℓ}` (original units) for the given
    /// stage and series element indices.
    ///
    /// The returned slice has length `max_order`. Positions `0..orders[hydro]` contain
    /// the meaningful coefficients; positions `orders[hydro]..max_order` are `0.0`.
    /// Use [`Self::order`] to determine how many entries are meaningful.
    ///
    /// # Panics
    ///
    /// Panics if `stage >= n_stages` or `hydro >= n_hydros`.
    #[must_use]
    pub fn psi_slice(&self, stage: usize, hydro: usize) -> &[f64] {
        assert!(
            stage < self.n_stages,
            "stage index {stage} is out of bounds (n_stages = {})",
            self.n_stages
        );
        assert!(
            hydro < self.n_hydros,
            "hydro index {hydro} is out of bounds (n_hydros = {})",
            self.n_hydros
        );
        if self.max_order == 0 {
            return &[];
        }
        let start = stage * self.n_hydros * self.max_order + hydro * self.max_order;
        &self.psi[start..start + self.max_order]
    }

    /// AR order for the given series element (maximum across all stages).
    ///
    /// Returns the number of meaningful lag entries in `psi_slice` for this series element.
    ///
    /// # Panics
    ///
    /// Panics if `hydro >= n_hydros`.
    #[must_use]
    pub fn order(&self, hydro: usize) -> usize {
        assert!(
            hydro < self.n_hydros,
            "hydro index {hydro} is out of bounds (n_hydros = {})",
            self.n_hydros
        );
        self.orders[hydro]
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
    /// Returns an empty [`PrecomputedPar`] with zero stages and zero series elements.
    ///
    /// Useful as a sentinel value for callers that do not use PAR models
    /// (e.g., test fixtures for systems with no series elements).
    fn default() -> Self {
        Self {
            deterministic_base: Box::new([]),
            sigma: Box::new([]),
            psi: Box::new([]),
            orders: Box::new([]),
            n_stages: 0,
            n_hydros: 0,
            max_order: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use chrono::NaiveDate;
    use cobre_core::{
        EntityId,
        scenario::InflowModel,
        temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
            StageStateConfig,
        },
    };

    use super::{PrecomputedPar, resolve_season_id};

    fn dummy_date(year: i32, month: u32, day: u32) -> chrono::NaiveDate {
        NaiveDate::from_ymd_opt(year, month, day).unwrap()
    }

    fn make_stage(index: usize, id: i32, season_id: Option<usize>) -> Stage {
        Stage {
            index,
            id,
            start_date: dummy_date(2024, 1, 1),
            end_date: dummy_date(2024, 2, 1),
            season_id,
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
                branching_factor: 10,
                noise_method: NoiseMethod::Saa,
            },
        }
    }

    fn make_model(
        hydro_id: i32,
        stage_id: i32,
        mean: f64,
        std: f64,
        coeffs: Vec<f64>,
        residual_ratio: f64,
    ) -> InflowModel {
        InflowModel {
            hydro_id: EntityId(hydro_id),
            stage_id,
            mean_m3s: mean,
            std_m3s: std,
            ar_coefficients: coeffs,
            residual_std_ratio: residual_ratio,
        }
    }

    #[test]
    fn ar_order_zero_deterministic_base_equals_mean() {
        let stage = make_stage(0, 0, Some(0));
        let model = make_model(1, 0, 100.0, 30.0, vec![], 1.0);
        let lp = PrecomputedPar::build(&[model], &[stage], &[EntityId(1)]).unwrap();

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
        let lp = PrecomputedPar::build(&[model], &[stage], &[EntityId(1)]).unwrap();

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

        let lp = PrecomputedPar::build(&all_models, &stages, &hydro_ids).unwrap();

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

        let lp = PrecomputedPar::build(&models, &[stage], &hydro_ids).unwrap();

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

        let lp = PrecomputedPar::build(&all_models, &stages, &hydro_ids).unwrap();

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

        let lp = PrecomputedPar::build(&models, &[stages[0].clone()], &hydro_ids).unwrap();

        assert!((lp.deterministic_base(0, 1)).abs() < f64::EPSILON);
        assert!((lp.sigma(0, 1)).abs() < f64::EPSILON);
    }

    #[test]
    #[should_panic(expected = "stage index 1 is out of bounds")]
    fn deterministic_base_out_of_bounds_panics() {
        let stage = make_stage(0, 0, Some(0));
        let model = make_model(1, 0, 100.0, 30.0, vec![], 1.0);
        let lp = PrecomputedPar::build(&[model], &[stage], &[EntityId(1)]).unwrap();
        let _ = lp.deterministic_base(1, 0);
    }

    #[test]
    #[should_panic(expected = "stage index 1 is out of bounds")]
    fn sigma_out_of_bounds_panics() {
        let stage = make_stage(0, 0, Some(0));
        let model = make_model(1, 0, 100.0, 30.0, vec![], 1.0);
        let lp = PrecomputedPar::build(&[model], &[stage], &[EntityId(1)]).unwrap();
        let _ = lp.sigma(1, 0);
    }

    #[test]
    fn acceptance_criterion_ar_order_zero_std_zero() {
        let stage = make_stage(0, 0, Some(0));
        let model = make_model(1, 0, 50.0, 0.0, vec![], 1.0);
        let lp = PrecomputedPar::build(&[model], &[stage], &[EntityId(1)]).unwrap();

        assert!((lp.sigma(0, 0)).abs() < f64::EPSILON);
        assert!((lp.deterministic_base(0, 0) - 50.0).abs() < f64::EPSILON);
    }

    // -----------------------------------------------------------------------
    // Pre-study lag season fallback tests (T003)
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

        let lp = PrecomputedPar::build(&models, &stages, &[EntityId(1)]).unwrap();

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

        let lp = PrecomputedPar::build(&models, &stages, &[EntityId(1)]).unwrap();

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

        let lp = PrecomputedPar::build(&models, &stages, &[EntityId(1)]).unwrap();

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

        let lp = PrecomputedPar::build(&models, &stages, &[EntityId(1)]).unwrap();

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
        // Property test of the modular arithmetic formula.
        assert_eq!(resolve_season_id(-1, 12), 11);
        assert_eq!(resolve_season_id(-6, 12), 6);
        assert_eq!(resolve_season_id(-12, 12), 0);
        assert_eq!(resolve_season_id(-13, 12), 11);
        assert_eq!(resolve_season_id(0, 12), 0);
        assert_eq!(resolve_season_id(5, 12), 5);

        // Additional edge cases.
        assert_eq!(resolve_season_id(-24, 12), 0);
        assert_eq!(resolve_season_id(-25, 12), 11);
        assert_eq!(resolve_season_id(12, 12), 0);
        assert_eq!(resolve_season_id(13, 12), 1);

        // Non-12 cycle lengths.
        assert_eq!(resolve_season_id(-1, 4), 3);
        assert_eq!(resolve_season_id(-5, 4), 3);
        assert_eq!(resolve_season_id(-4, 4), 0);
        assert_eq!(resolve_season_id(-1, 1), 0);
    }
}
