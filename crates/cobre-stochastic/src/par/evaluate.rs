//! PAR(p) inflow evaluation functions.
//!
//! This module provides functions for evaluating the Periodic Autoregressive
//! model equation in the forward direction: computing the inflow value at a
//! given stage from the current lag state and a noise realisation.
//!
//! ## PAR inflow equation
//!
//! Given precomputed parameters, the inflow for hydro `h` at stage `t` is:
//!
//! ```text
//! a_h = deterministic_base + sum_{l=0}^{order-1} psi[l] * lag[l] + sigma * eta
//! ```
//!
//! where:
//! - `deterministic_base` encodes `mu_m - sum psi_{m,l} * mu_{m-l}` (built by
//!   [`PrecomputedParLp`])
//! - `psi[l]` are the AR coefficients in original units (lag 1 is at index 0)
//! - `lag[l]` is the observed inflow value at lag `l+1`
//! - `sigma` is the residual standard deviation
//! - `eta` is the standardised noise realisation
//!
//! The returned value may be negative; truncation to a physical minimum is the
//! caller's responsibility.
//!
//! [`PrecomputedParLp`]: super::precompute::PrecomputedParLp

use super::precompute::PrecomputedParLp;

/// Evaluate the PAR(p) inflow equation for a single hydro at a single stage.
///
/// Computes:
/// ```text
/// a_h = deterministic_base + sum_{l=0}^{order-1} psi[l] * lags[l] + sigma * eta
/// ```
///
/// The returned value may be negative (truncation is the caller's
/// responsibility).
///
/// # Parameters
///
/// - `deterministic_base` — the precomputed `b_{h,m(t)} = mu_m - sum psi_{m,l} * mu_{m-l}`
/// - `psi` — AR coefficients in original units for this (stage, hydro) pair;
///   only `psi[0..order]` are used
/// - `order` — number of meaningful entries in `psi` (the AR order)
/// - `lags` — observed inflow values at lags 1..p; `lags[0]` = lag-1 inflow,
///   `lags[1]` = lag-2 inflow, etc.; must have `lags.len() >= order`
/// - `sigma` — residual standard deviation
/// - `eta` — standardised noise realisation (post-correlation)
///
/// # Examples
///
/// AR(0) — mean plus noise:
///
/// ```
/// use cobre_stochastic::evaluate_par_inflow;
///
/// let a_h = evaluate_par_inflow(100.0, &[], 0, &[], 30.0, 1.5);
/// assert!((a_h - 145.0).abs() < 1e-10);
/// ```
///
/// AR(1) — one lag:
///
/// ```
/// use cobre_stochastic::evaluate_par_inflow;
///
/// // a_h = 70.0 + 0.48 * 90.0 + 28.62 * 0.5 = 127.51
/// let a_h = evaluate_par_inflow(70.0, &[0.48], 1, &[90.0], 28.62, 0.5);
/// assert!((a_h - 127.51).abs() < 1e-10);
/// ```
#[must_use]
pub fn evaluate_par_inflow(
    deterministic_base: f64,
    psi: &[f64],
    order: usize,
    lags: &[f64],
    sigma: f64,
    eta: f64,
) -> f64 {
    debug_assert!(
        lags.len() >= order,
        "lags.len() ({}) must be >= order ({})",
        lags.len(),
        order
    );
    let mut a_h = deterministic_base;
    for l in 0..order {
        a_h += psi[l] * lags[l];
    }
    a_h + sigma * eta
}

/// Evaluate the PAR(p) inflow equation for all hydros at a given stage.
///
/// Writes the computed inflow for each hydro into `output`. Does not allocate.
/// Values may be negative (truncation is the caller's responsibility).
///
/// The `lag_matrix` is a flat array indexed as `[lag * n_hydros + hydro]`:
/// lag 0 (most recent, i.e. lag-1 inflow) for all hydros is contiguous,
/// followed by lag 1 for all hydros, and so on. This layout is optimal for
/// sequential hydro iteration at a fixed lag depth.
///
/// # Parameters
///
/// - `par_lp` — precomputed PAR cache built by [`PrecomputedParLp::build`]
/// - `stage` — 0-based stage index (must be `< par_lp.n_stages()`)
/// - `lag_matrix` — flat lag array, length `max_order * n_hydros`, indexed
///   as `[lag * n_hydros + hydro]`
/// - `noise` — standardised noise vector, length `n_hydros`
/// - `output` — output buffer; filled with computed inflows, length `n_hydros`
///
/// # Examples
///
/// ```
/// use cobre_core::{EntityId, scenario::InflowModel, temporal::{Stage, Block, BlockMode, StageStateConfig, StageRiskConfig, ScenarioSourceConfig, NoiseMethod}};
/// use cobre_stochastic::par::precompute::PrecomputedParLp;
/// use cobre_stochastic::evaluate_par_inflows;
/// use chrono::NaiveDate;
///
/// let date = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
/// let stage = Stage {
///     index: 0, id: 0,
///     start_date: date,
///     end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
///     season_id: Some(0),
///     blocks: vec![Block { index: 0, name: "SINGLE".to_string(), duration_hours: 744.0 }],
///     block_mode: BlockMode::Parallel,
///     state_config: StageStateConfig { storage: true, inflow_lags: false },
///     risk_config: StageRiskConfig::Expectation,
///     scenario_config: ScenarioSourceConfig { branching_factor: 10, noise_method: NoiseMethod::Saa },
/// };
/// let model = InflowModel {
///     hydro_id: EntityId(1), stage_id: 0,
///     mean_m3s: 100.0, std_m3s: 30.0,
///     ar_coefficients: vec![],
///     residual_std_ratio: 1.0,
/// };
/// let par_lp = PrecomputedParLp::build(&[model], &[stage], &[EntityId(1)]).unwrap();
///
/// let lag_matrix: Vec<f64> = vec![]; // no lags for AR(0)
/// let noise = vec![0.5];
/// let mut output = vec![0.0];
/// evaluate_par_inflows(&par_lp, 0, &lag_matrix, &noise, &mut output);
/// assert!((output[0] - 115.0).abs() < 1e-10); // 100.0 + 30.0 * 0.5
/// ```
pub fn evaluate_par_inflows(
    par_lp: &PrecomputedParLp,
    stage: usize,
    lag_matrix: &[f64],
    noise: &[f64],
    output: &mut [f64],
) {
    let n_hydros = par_lp.n_hydros();
    debug_assert!(
        noise.len() == n_hydros,
        "noise.len() ({}) must equal n_hydros ({})",
        noise.len(),
        n_hydros
    );
    debug_assert!(
        output.len() == n_hydros,
        "output.len() ({}) must equal n_hydros ({})",
        output.len(),
        n_hydros
    );

    for h in 0..n_hydros {
        let base = par_lp.deterministic_base(stage, h);
        let sigma = par_lp.sigma(stage, h);
        let psi = par_lp.psi_slice(stage, h);
        let order = par_lp.order(h);

        let mut a_h = base;
        for l in 0..order {
            a_h += psi[l] * lag_matrix[l * n_hydros + h];
        }
        a_h += sigma * noise[h];
        output[h] = a_h;
    }
}

/// Compute the noise value `eta_min` that makes the PAR(p) inflow exactly zero
/// for a single hydro.
///
/// Derived by setting `a_h = 0` in the PAR equation and solving for `eta`:
///
/// ```text
/// eta_min = -(deterministic_base + sum_{l=0}^{order-1} psi[l] * lags[l]) / sigma
/// ```
///
/// When `sigma == 0.0`, noise has no effect on the inflow. Any noise value is
/// equally valid (or invalid), so `f64::NEG_INFINITY` is returned to indicate
/// that the caller may use any noise without worrying about the truncation bound.
///
/// # Parameters
///
/// - `deterministic_base` — the precomputed `b_{h,m(t)} = mu_m - sum psi_{m,l} * mu_{m-l}`
/// - `psi` — AR coefficients in original units; only `psi[0..order]` are used
/// - `order` — number of meaningful entries in `psi` (the AR order)
/// - `lags` — observed inflow values at lags 1..p; must have `lags.len() >= order`
/// - `sigma` — residual standard deviation
///
/// # Examples
///
/// AR(1) — known truncation noise:
///
/// ```
/// use cobre_stochastic::compute_truncation_noise;
///
/// // eta_min = -(70.0 + 0.48 * 90.0) / 28.62 = -(70.0 + 43.2) / 28.62
/// let eta_min = compute_truncation_noise(70.0, &[0.48], 1, &[90.0], 28.62);
/// let expected = -(70.0 + 0.48 * 90.0) / 28.62;
/// assert!((eta_min - expected).abs() < 1e-10);
/// ```
///
/// Zero sigma returns `f64::NEG_INFINITY`:
///
/// ```
/// use cobre_stochastic::compute_truncation_noise;
///
/// let eta_min = compute_truncation_noise(100.0, &[0.5], 1, &[50.0], 0.0);
/// assert_eq!(eta_min, f64::NEG_INFINITY);
/// ```
#[must_use]
pub fn compute_truncation_noise(
    deterministic_base: f64,
    psi: &[f64],
    order: usize,
    lags: &[f64],
    sigma: f64,
) -> f64 {
    if sigma == 0.0 {
        return f64::NEG_INFINITY;
    }
    debug_assert!(
        lags.len() >= order,
        "lags.len() ({}) must be >= order ({})",
        lags.len(),
        order
    );
    let mut deterministic_inflow = deterministic_base;
    for l in 0..order {
        deterministic_inflow += psi[l] * lags[l];
    }
    -deterministic_inflow / sigma
}

/// Compute noise truncation thresholds for all hydros at a given stage.
///
/// Writes `eta_min` for each hydro into `output[0..n_hydros]`. Does not
/// allocate. The `lag_matrix` uses the same layout as [`evaluate_par_inflows`]:
/// indexed as `[lag * n_hydros + hydro]`.
///
/// # Parameters
///
/// - `par_lp` — precomputed PAR cache built by [`PrecomputedParLp::build`]
/// - `stage` — 0-based stage index (must be `< par_lp.n_stages()`)
/// - `lag_matrix` — flat lag array, length `max_order * n_hydros`, indexed
///   as `[lag * n_hydros + hydro]`
/// - `output` — output buffer; filled with `eta_min` values, length `n_hydros`
///
/// # Examples
///
/// ```
/// use cobre_core::{EntityId, scenario::InflowModel, temporal::{Stage, Block, BlockMode, StageStateConfig, StageRiskConfig, ScenarioSourceConfig, NoiseMethod}};
/// use cobre_stochastic::par::precompute::PrecomputedParLp;
/// use cobre_stochastic::{compute_truncation_noise, compute_truncation_noises};
/// use chrono::NaiveDate;
///
/// let date = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
/// let stage = Stage {
///     index: 0, id: 0,
///     start_date: date,
///     end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
///     season_id: Some(0),
///     blocks: vec![Block { index: 0, name: "SINGLE".to_string(), duration_hours: 744.0 }],
///     block_mode: BlockMode::Parallel,
///     state_config: StageStateConfig { storage: true, inflow_lags: false },
///     risk_config: StageRiskConfig::Expectation,
///     scenario_config: ScenarioSourceConfig { branching_factor: 10, noise_method: NoiseMethod::Saa },
/// };
/// let model = InflowModel {
///     hydro_id: EntityId(1), stage_id: 0,
///     mean_m3s: 100.0, std_m3s: 30.0,
///     ar_coefficients: vec![],
///     residual_std_ratio: 1.0,
/// };
/// let par_lp = PrecomputedParLp::build(&[model], &[stage], &[EntityId(1)]).unwrap();
///
/// let lag_matrix: Vec<f64> = vec![]; // no lags for AR(0)
/// let mut output = vec![0.0];
/// compute_truncation_noises(&par_lp, 0, &lag_matrix, &mut output);
/// // eta_min = -100.0 / 30.0
/// assert!((output[0] - (-100.0 / 30.0)).abs() < 1e-10);
/// ```
pub fn compute_truncation_noises(
    par_lp: &PrecomputedParLp,
    stage: usize,
    lag_matrix: &[f64],
    output: &mut [f64],
) {
    let n_hydros = par_lp.n_hydros();
    debug_assert!(
        output.len() == n_hydros,
        "output.len() ({}) must equal n_hydros ({})",
        output.len(),
        n_hydros
    );

    for h in 0..n_hydros {
        let base = par_lp.deterministic_base(stage, h);
        let sigma = par_lp.sigma(stage, h);
        let psi = par_lp.psi_slice(stage, h);
        let order = par_lp.order(h);

        if sigma == 0.0 {
            output[h] = f64::NEG_INFINITY;
            continue;
        }

        let mut deterministic_inflow = base;
        for l in 0..order {
            deterministic_inflow += psi[l] * lag_matrix[l * n_hydros + h];
        }
        output[h] = -deterministic_inflow / sigma;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use chrono::NaiveDate;
    use cobre_core::{
        scenario::InflowModel,
        temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
            StageStateConfig,
        },
        EntityId,
    };

    use super::{
        compute_truncation_noise, compute_truncation_noises, evaluate_par_inflow,
        evaluate_par_inflows,
    };
    use crate::par::precompute::PrecomputedParLp;

    fn dummy_date(year: i32, month: u32, day: u32) -> NaiveDate {
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
    fn ar0_produces_mean_plus_noise() {
        // a_h = 100.0 + 0 (no lags) + 30.0 * 1.5 = 145.0
        let a_h = evaluate_par_inflow(100.0, &[], 0, &[], 30.0, 1.5);
        assert!(
            (a_h - 145.0).abs() < 1e-10,
            "AR(0): expected 145.0, got {a_h}"
        );
    }

    #[test]
    fn ar1_acceptance_criterion() {
        // a_h = 70.0 + 0.48 * 90.0 + 28.62 * 0.5 = 70.0 + 43.2 + 14.31 = 127.51
        let a_h = evaluate_par_inflow(70.0, &[0.48], 1, &[90.0], 28.62, 0.5);
        assert!(
            (a_h - 127.51).abs() < 1e-10,
            "AR(1): expected 127.51, got {a_h}"
        );
    }

    #[test]
    fn ar2_known_values() {
        // a_h = 50.0 + 0.4 * 80.0 + 0.2 * 60.0 + 20.0 * (-0.5)
        //     = 50.0 + 32.0 + 12.0 - 10.0
        //     = 84.0
        let a_h = evaluate_par_inflow(50.0, &[0.4, 0.2], 2, &[80.0, 60.0], 20.0, -0.5);
        assert!(
            (a_h - 84.0).abs() < 1e-10,
            "AR(2): expected 84.0, got {a_h}"
        );
    }

    #[test]
    fn zero_sigma_returns_deterministic_value() {
        // a_h = 100.0 + 0.3 * 90.0 + 0.0 * anything = 100.0 + 27.0 = 127.0
        let a_h = evaluate_par_inflow(100.0, &[0.3], 1, &[90.0], 0.0, 999.0);
        assert!(
            (a_h - 127.0).abs() < 1e-10,
            "zero sigma: expected 127.0, got {a_h}"
        );
    }

    #[test]
    fn negative_noise_can_produce_negative_inflow() {
        // a_h = 10.0 + 0.0 (no lags) + 100.0 * (-1.0) = -90.0
        // Truncation is not this function's responsibility.
        let a_h = evaluate_par_inflow(10.0, &[], 0, &[], 100.0, -1.0);
        assert!(
            a_h < 0.0,
            "expected negative inflow for large negative noise, got {a_h}"
        );
        assert!((a_h - (-90.0)).abs() < 1e-10, "expected -90.0, got {a_h}");
    }

    #[test]
    fn order_respected_from_psi_slice_longer_than_order() {
        // psi has 3 entries but order=1: only psi[0] should be used.
        // a_h = 50.0 + 0.4 * 80.0 + 20.0 * 0.0 = 50.0 + 32.0 = 82.0
        let a_h = evaluate_par_inflow(50.0, &[0.4, 0.9, 0.9], 1, &[80.0, 60.0, 40.0], 20.0, 0.0);
        assert!(
            (a_h - 82.0).abs() < 1e-10,
            "order truncation: expected 82.0, got {a_h}"
        );
    }

    fn make_two_hydro_three_stage_par() -> PrecomputedParLp {
        let hydro_ids = [EntityId(3), EntityId(5)];

        let stages: Vec<Stage> = (0..3)
            .map(|i| make_stage(i, i32::try_from(i).unwrap(), Some(0)))
            .collect();

        // Pre-study models needed for lag coefficient conversion.
        let pre_models = vec![
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

        let mut all_models = pre_models;
        all_models.extend(study_models);

        PrecomputedParLp::build(&all_models, &stages, &hydro_ids).unwrap()
    }

    #[test]
    fn batch_matches_single_hydro_at_stage_1() {
        let par_lp = make_two_hydro_three_stage_par();

        // Stage 1: hydro 3 (h_idx=0) has AR(1), hydro 5 (h_idx=1) has AR(1).
        // lag_matrix layout: [lag * n_hydros + hydro]
        // For max_order=2, n_hydros=2: [lag0_h0, lag0_h1, lag1_h0, lag1_h1]
        let lag_matrix = vec![
            85.0, // lag0, hydro 3
            55.0, // lag0, hydro 5
            0.0,  // lag1, hydro 3 (unused for AR(1))
            0.0,  // lag1, hydro 5 (unused for AR(1))
        ];
        let noise = vec![0.3_f64, -0.4_f64];
        let mut output = vec![0.0_f64; 2];

        evaluate_par_inflows(&par_lp, 1, &lag_matrix, &noise, &mut output);

        let n_hydros = par_lp.n_hydros();
        let order_h0 = par_lp.order(0);
        let lags_h0: Vec<f64> = (0..order_h0).map(|l| lag_matrix[l * n_hydros]).collect();
        let expected_h0 = evaluate_par_inflow(
            par_lp.deterministic_base(1, 0),
            par_lp.psi_slice(1, 0),
            order_h0,
            &lags_h0,
            par_lp.sigma(1, 0),
            noise[0],
        );
        let order_h1 = par_lp.order(1);
        let lags_h1: Vec<f64> = (0..order_h1)
            .map(|l| lag_matrix[l * n_hydros + 1])
            .collect();
        let expected_h1 = evaluate_par_inflow(
            par_lp.deterministic_base(1, 1),
            par_lp.psi_slice(1, 1),
            order_h1,
            &lags_h1,
            par_lp.sigma(1, 1),
            noise[1],
        );

        assert!(
            (output[0] - expected_h0).abs() < 1e-10,
            "batch h0 mismatch: expected {expected_h0}, got {}",
            output[0]
        );
        assert!(
            (output[1] - expected_h1).abs() < 1e-10,
            "batch h1 mismatch: expected {expected_h1}, got {}",
            output[1]
        );
    }

    #[test]
    fn batch_matches_single_hydro_at_stage_0_ar2() {
        let par_lp = make_two_hydro_three_stage_par();

        // Stage 0: hydro 3 (h_idx=0) has AR(2), hydro 5 (h_idx=1) has AR(1).
        // lag_matrix layout: [lag0_h0, lag0_h1, lag1_h0, lag1_h1]
        let lag0_h0 = 90.0_f64;
        let lag0_h1 = 58.0_f64;
        let lag1_h0 = 78.0_f64;
        let lag_matrix = vec![lag0_h0, lag0_h1, lag1_h0, 0.0];
        let noise = vec![1.0_f64, -0.2_f64];
        let mut output = vec![0.0_f64; 2];

        evaluate_par_inflows(&par_lp, 0, &lag_matrix, &noise, &mut output);

        let expected_h0 = evaluate_par_inflow(
            par_lp.deterministic_base(0, 0),
            par_lp.psi_slice(0, 0),
            par_lp.order(0),
            &[lag0_h0, lag1_h0],
            par_lp.sigma(0, 0),
            noise[0],
        );
        let expected_h1 = evaluate_par_inflow(
            par_lp.deterministic_base(0, 1),
            par_lp.psi_slice(0, 1),
            par_lp.order(1),
            &[lag0_h1],
            par_lp.sigma(0, 1),
            noise[1],
        );

        assert!(
            (output[0] - expected_h0).abs() < 1e-10,
            "batch h0 stage0 AR2 mismatch: expected {expected_h0}, got {}",
            output[0]
        );
        assert!(
            (output[1] - expected_h1).abs() < 1e-10,
            "batch h1 stage0 AR1 mismatch: expected {expected_h1}, got {}",
            output[1]
        );
    }

    #[test]
    fn declaration_order_invariance_batch() {
        // Build par_lp with hydros in canonical order [3, 5].
        let hydro_ids_canonical = [EntityId(3), EntityId(5)];
        let stages = vec![make_stage(0, 0, Some(0))];

        let models = vec![
            make_model(3, 0, 100.0, 30.0, vec![], 1.0),
            make_model(5, 0, 200.0, 40.0, vec![], 1.0),
        ];
        // Also build with reversed model declaration order.
        let models_reversed = vec![
            make_model(5, 0, 200.0, 40.0, vec![], 1.0),
            make_model(3, 0, 100.0, 30.0, vec![], 1.0),
        ];

        let par_canonical =
            PrecomputedParLp::build(&models, &stages, &hydro_ids_canonical).unwrap();
        let par_reversed =
            PrecomputedParLp::build(&models_reversed, &[stages[0].clone()], &hydro_ids_canonical)
                .unwrap();

        // AR(0) for both, so lag_matrix is empty (max_order=0 → no lag entries).
        let lag_matrix: Vec<f64> = vec![];
        let noise = vec![0.5_f64, -0.3_f64];
        let mut output_canonical = vec![0.0_f64; 2];
        let mut output_reversed = vec![0.0_f64; 2];

        evaluate_par_inflows(
            &par_canonical,
            0,
            &lag_matrix,
            &noise,
            &mut output_canonical,
        );
        evaluate_par_inflows(&par_reversed, 0, &lag_matrix, &noise, &mut output_reversed);

        // Both must produce bit-for-bit identical results.
        assert_eq!(
            output_canonical[0].to_bits(),
            output_reversed[0].to_bits(),
            "h_idx=0 output differs between canonical and reversed model order"
        );
        assert_eq!(
            output_canonical[1].to_bits(),
            output_reversed[1].to_bits(),
            "h_idx=1 output differs between canonical and reversed model order"
        );

        // Cross-check expected values: h_idx=0 is EntityId(3) (mean=100, sigma=30).
        // a_h = 100.0 + 30.0 * 0.5 = 115.0
        assert!(
            (output_canonical[0] - 115.0).abs() < 1e-10,
            "h_idx=0 (EntityId 3): expected 115.0, got {}",
            output_canonical[0]
        );
        // h_idx=1 is EntityId(5) (mean=200, sigma=40).
        // a_h = 200.0 + 40.0 * (-0.3) = 188.0
        assert!(
            (output_canonical[1] - 188.0).abs() < 1e-10,
            "h_idx=1 (EntityId 5): expected 188.0, got {}",
            output_canonical[1]
        );
    }

    #[test]
    fn truncation_ar1_acceptance_criterion() {
        // eta_min = -(70.0 + 0.48 * 90.0) / 28.62 = -113.2 / 28.62
        let expected = -(70.0_f64 + 0.48 * 90.0) / 28.62;
        let eta_min = compute_truncation_noise(70.0, &[0.48], 1, &[90.0], 28.62);
        assert!(
            (eta_min - expected).abs() < 1e-10,
            "AR(1) acceptance criterion: expected {expected}, got {eta_min}"
        );
    }

    #[test]
    fn truncation_roundtrip_makes_inflow_zero() {
        // Given eta_min, evaluate_par_inflow(..., eta_min) must be 0.0 within 1e-10.
        let det_base = 70.0_f64;
        let psi = [0.48_f64];
        let lags = [90.0_f64];
        let sigma = 28.62_f64;

        let eta_min = compute_truncation_noise(det_base, &psi, 1, &lags, sigma);
        let inflow = evaluate_par_inflow(det_base, &psi, 1, &lags, sigma, eta_min);
        assert!(
            inflow.abs() < 1e-10,
            "roundtrip: inflow with eta_min must be 0.0, got {inflow}"
        );
    }

    #[test]
    fn truncation_roundtrip_ar2() {
        let det_base = 50.0_f64;
        let psi = [0.4_f64, 0.2];
        let lags = [80.0_f64, 60.0];
        let sigma = 20.0_f64;

        let eta_min = compute_truncation_noise(det_base, &psi, 2, &lags, sigma);
        let inflow = evaluate_par_inflow(det_base, &psi, 2, &lags, sigma, eta_min);
        assert!(
            inflow.abs() < 1e-10,
            "AR(2) roundtrip: expected 0.0, got {inflow}"
        );
    }

    #[test]
    fn truncation_ar0_case() {
        // eta_min = -deterministic_base / sigma
        let det_base = 120.0_f64;
        let sigma = 40.0_f64;
        let expected = -det_base / sigma;
        let eta_min = compute_truncation_noise(det_base, &[], 0, &[], sigma);
        assert!(
            (eta_min - expected).abs() < 1e-10,
            "AR(0): expected {expected}, got {eta_min}"
        );
    }

    #[test]
    fn truncation_zero_sigma_returns_neg_infinity() {
        let eta_min = compute_truncation_noise(100.0, &[0.5], 1, &[50.0], 0.0);
        assert!(
            eta_min.is_infinite() && eta_min.is_sign_negative(),
            "zero sigma: expected NEG_INFINITY, got {eta_min}"
        );
    }

    #[test]
    fn truncation_positive_deterministic_inflow_gives_negative_eta_min() {
        // Positive deterministic inflow → need large negative noise to reach zero.
        let det_base = 100.0_f64;
        let sigma = 30.0_f64;
        let eta_min = compute_truncation_noise(det_base, &[], 0, &[], sigma);
        assert!(
            eta_min < 0.0,
            "positive deterministic inflow: eta_min must be negative, got {eta_min}"
        );
    }

    #[test]
    fn truncation_negative_deterministic_inflow_gives_positive_eta_min() {
        // Negative deterministic inflow → need positive noise to push inflow up to zero.
        // det_base = -50.0, no lags, sigma = 20.0 → eta_min = -(-50.0) / 20.0 = 2.5
        let eta_min = compute_truncation_noise(-50.0, &[], 0, &[], 20.0);
        assert!(
            eta_min > 0.0,
            "negative deterministic inflow: eta_min must be positive, got {eta_min}"
        );
        assert!((eta_min - 2.5).abs() < 1e-10, "expected 2.5, got {eta_min}");
    }

    // -----------------------------------------------------------------------
    // compute_truncation_noises batch unit tests
    // -----------------------------------------------------------------------

    #[test]
    fn batch_truncation_matches_single_hydro_at_stage_0() {
        let par_lp = make_two_hydro_three_stage_par();
        let n_hydros = par_lp.n_hydros();

        // Stage 0: hydro 3 (h_idx=0) has AR(2), hydro 5 (h_idx=1) has AR(1).
        let lag0_h0 = 90.0_f64;
        let lag0_h1 = 58.0_f64;
        let lag1_h0 = 78.0_f64;
        let lag_matrix = vec![lag0_h0, lag0_h1, lag1_h0, 0.0];

        let mut output = vec![0.0_f64; n_hydros];
        compute_truncation_noises(&par_lp, 0, &lag_matrix, &mut output);

        // Per-hydro single-call references.
        let order_h0 = par_lp.order(0);
        let lags_for_h0: Vec<f64> = (0..order_h0).map(|l| lag_matrix[l * n_hydros]).collect();
        let expected_h0 = compute_truncation_noise(
            par_lp.deterministic_base(0, 0),
            par_lp.psi_slice(0, 0),
            order_h0,
            &lags_for_h0,
            par_lp.sigma(0, 0),
        );
        let order_h1 = par_lp.order(1);
        let lags_for_h1: Vec<f64> = (0..order_h1)
            .map(|l| lag_matrix[l * n_hydros + 1])
            .collect();
        let expected_h1 = compute_truncation_noise(
            par_lp.deterministic_base(0, 1),
            par_lp.psi_slice(0, 1),
            order_h1,
            &lags_for_h1,
            par_lp.sigma(0, 1),
        );

        assert!(
            (output[0] - expected_h0).abs() < 1e-10,
            "batch h0 stage0: expected {expected_h0}, got {}",
            output[0]
        );
        assert!(
            (output[1] - expected_h1).abs() < 1e-10,
            "batch h1 stage0: expected {expected_h1}, got {}",
            output[1]
        );
    }

    #[test]
    fn batch_truncation_roundtrip_makes_all_inflows_zero() {
        let par_lp = make_two_hydro_three_stage_par();
        let n_hydros = par_lp.n_hydros();

        let lag_matrix = vec![
            85.0, // lag0, hydro 3
            55.0, // lag0, hydro 5
            0.0,  // lag1, hydro 3 (unused for AR(1))
            0.0,  // lag1, hydro 5 (unused for AR(1))
        ];

        let mut eta_min = vec![0.0_f64; n_hydros];
        compute_truncation_noises(&par_lp, 1, &lag_matrix, &mut eta_min);

        let mut inflows = vec![0.0_f64; n_hydros];
        evaluate_par_inflows(&par_lp, 1, &lag_matrix, &eta_min, &mut inflows);

        for (h, &inflow) in inflows.iter().enumerate() {
            assert!(
                inflow.abs() < 1e-10,
                "batch roundtrip: h={h} inflow must be 0.0, got {inflow}"
            );
        }
    }
}
