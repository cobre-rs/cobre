//! Integration tests validating Halton QMC statistical properties and correctness.
//!
//! Exercises `NoiseMethod::QmcHalton` through the full `generate_opening_tree`
//! pipeline and verifies five properties that cannot be tested in unit tests:
//! star discrepancy bounds, normal marginal statistics, correlation application,
//! declaration-order invariance, and point-wise cross-path consistency.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss
)]

use std::collections::BTreeMap;

use chrono::NaiveDate;
use cobre_core::{
    Bus, DeficitSegment, EntityId, SystemBuilder,
    entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties},
    scenario::{
        CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile, InflowModel,
        SamplingScheme,
    },
    temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    },
};
use cobre_stochastic::{
    ClassDimensions, ClassSchemes, build_stochastic_context,
    correlation::resolve::DecomposedCorrelation,
    generate_opening_tree,
    tree::qmc_halton::{HaltonPointSpec, scrambled_halton_point},
};

// ---------------------------------------------------------------------------
// Helpers shared across tests
// ---------------------------------------------------------------------------

/// Approximate `erf(x)` using the Horner-form rational approximation
/// (Abramowitz & Stegun 7.1.26, max error 1.5e-7).
fn approx_erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0_f64 } else { 1.0_f64 };
    let t = 1.0 / (1.0 + 0.327_591_1 * x.abs());
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    sign * (1.0 - poly * (-x * x).exp())
}

/// Standard normal CDF approximation: `Φ(z) = 0.5 * (1 + erf(z / √2))`.
fn norm_cdf(z: f64) -> f64 {
    0.5 * (1.0 + approx_erf(z / std::f64::consts::SQRT_2))
}

/// Build a `Stage` with `NoiseMethod::QmcHalton` and no blocks, suitable for
/// direct use with `generate_opening_tree`.
fn make_stage_halton(index: usize, id: i32, branching_factor: usize) -> Stage {
    Stage {
        index,
        id,
        start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
        season_id: Some(0),
        blocks: vec![],
        block_mode: BlockMode::Parallel,
        state_config: StageStateConfig {
            storage: true,
            inflow_lags: false,
        },
        risk_config: StageRiskConfig::Expectation,
        scenario_config: ScenarioSourceConfig {
            branching_factor,
            noise_method: NoiseMethod::QmcHalton,
        },
    }
}

/// Build a `Stage` with `NoiseMethod::QmcHalton` with one block, suitable for
/// use with `build_stochastic_context`.
fn make_stage_halton_with_block(index: usize, id: i32, branching_factor: usize) -> Stage {
    Stage {
        index,
        id,
        start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
        season_id: Some(0),
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
            branching_factor,
            noise_method: NoiseMethod::QmcHalton,
        },
    }
}

fn identity_correlation(entity_ids: &[i32]) -> DecomposedCorrelation {
    let n = entity_ids.len();
    let matrix: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
        .collect();
    let mut profiles = BTreeMap::new();
    profiles.insert(
        "default".to_string(),
        CorrelationProfile {
            groups: vec![CorrelationGroup {
                name: "g1".to_string(),
                entities: entity_ids
                    .iter()
                    .map(|&id| CorrelationEntity {
                        entity_type: "inflow".to_string(),
                        id: EntityId(id),
                    })
                    .collect(),
                matrix,
            }],
        },
    );
    let model = CorrelationModel {
        method: "spectral".to_string(),
        profiles,
        schedule: vec![],
    };
    DecomposedCorrelation::build(&model).unwrap()
}

fn correlated_correlation(entity_ids: &[i32], rho: f64) -> DecomposedCorrelation {
    let n = entity_ids.len();
    let matrix: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| if i == j { 1.0 } else { rho }).collect())
        .collect();
    let mut profiles = BTreeMap::new();
    profiles.insert(
        "default".to_string(),
        CorrelationProfile {
            groups: vec![CorrelationGroup {
                name: "g1".to_string(),
                entities: entity_ids
                    .iter()
                    .map(|&id| CorrelationEntity {
                        entity_type: "inflow".to_string(),
                        id: EntityId(id),
                    })
                    .collect(),
                matrix,
            }],
        },
    );
    let model = CorrelationModel {
        method: "spectral".to_string(),
        profiles,
        schedule: vec![],
    };
    DecomposedCorrelation::build(&model).unwrap()
}

fn identity_correlation_model(entity_ids: &[i32]) -> CorrelationModel {
    let n = entity_ids.len();
    let matrix: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
        .collect();
    let mut profiles = BTreeMap::new();
    profiles.insert(
        "default".to_string(),
        CorrelationProfile {
            groups: vec![CorrelationGroup {
                name: "g1".to_string(),
                entities: entity_ids
                    .iter()
                    .map(|&id| CorrelationEntity {
                        entity_type: "inflow".to_string(),
                        id: EntityId(id),
                    })
                    .collect(),
                matrix,
            }],
        },
    );
    CorrelationModel {
        method: "spectral".to_string(),
        profiles,
        schedule: vec![],
    }
}

fn make_bus(id: i32) -> Bus {
    Bus {
        id: EntityId(id),
        name: format!("Bus{id}"),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 1000.0,
        }],
        excess_cost: 0.0,
    }
}

fn make_hydro(id: i32) -> Hydro {
    Hydro {
        id: EntityId(id),
        name: format!("H{id}"),
        bus_id: EntityId(0),
        downstream_id: None,
        entry_stage_id: None,
        exit_stage_id: None,
        min_storage_hm3: 0.0,
        max_storage_hm3: 100.0,
        min_outflow_m3s: 0.0,
        max_outflow_m3s: None,
        generation_model: HydroGenerationModel::ConstantProductivity {
            productivity_mw_per_m3s: 1.0,
        },
        min_turbined_m3s: 0.0,
        max_turbined_m3s: 100.0,
        min_generation_mw: 0.0,
        max_generation_mw: 100.0,
        tailrace: None,
        hydraulic_losses: None,
        efficiency: None,
        evaporation_coefficients_mm: None,
        evaporation_reference_volumes_hm3: None,
        diversion: None,
        filling: None,
        penalties: HydroPenalties {
            spillage_cost: 0.0,
            diversion_cost: 0.0,
            fpha_turbined_cost: 0.0,
            storage_violation_below_cost: 0.0,
            filling_target_violation_cost: 0.0,
            turbined_violation_below_cost: 0.0,
            outflow_violation_below_cost: 0.0,
            outflow_violation_above_cost: 0.0,
            generation_violation_below_cost: 0.0,
            evaporation_violation_cost: 0.0,
            water_withdrawal_violation_cost: 0.0,
            water_withdrawal_violation_pos_cost: 0.0,
            water_withdrawal_violation_neg_cost: 0.0,
            evaporation_violation_pos_cost: 0.0,
            evaporation_violation_neg_cost: 0.0,
            inflow_nonnegativity_cost: 1000.0,
        },
    }
}

fn make_inflow_model(hydro_id: i32, stage_id: i32) -> InflowModel {
    InflowModel {
        hydro_id: EntityId(hydro_id),
        stage_id,
        mean_m3s: 100.0,
        std_m3s: 30.0,
        ar_coefficients: vec![],
        residual_std_ratio: 1.0,
    }
}

/// Build a `StochasticContext` with `QmcHalton` stages and the given hydro list.
fn build_halton_context(
    hydros: Vec<Hydro>,
    n_openings: usize,
    base_seed: u64,
) -> cobre_stochastic::StochasticContext {
    let hydro_ids: Vec<i32> = {
        let mut ids: Vec<i32> = hydros.iter().map(|h| h.id.0).collect();
        ids.sort_unstable();
        ids
    };

    let stages = vec![
        make_stage_halton_with_block(0, 0, n_openings),
        make_stage_halton_with_block(1, 1, n_openings),
        make_stage_halton_with_block(2, 2, n_openings),
    ];

    let mut inflow_models: Vec<InflowModel> = Vec::new();
    for &hid in &hydro_ids {
        for &sid in &[0_i32, 1, 2] {
            inflow_models.push(make_inflow_model(hid, sid));
        }
    }

    let system = SystemBuilder::new()
        .buses(vec![make_bus(0)])
        .hydros(hydros)
        .stages(stages)
        .inflow_models(inflow_models)
        .correlation(identity_correlation_model(&hydro_ids))
        .build()
        .expect("build_halton_context: system build must succeed");

    build_stochastic_context(
        &system,
        base_seed,
        None,
        &[],
        &[],
        None,
        ClassSchemes {
            inflow: Some(SamplingScheme::InSample),
            load: Some(SamplingScheme::InSample),
            ncs: Some(SamplingScheme::InSample),
        },
    )
    .expect("build_halton_context: build_stochastic_context must succeed")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Verify the 2D star discrepancy of N=64 scrambled Halton points is below 0.15.
///
/// The star discrepancy D* measures the maximum deviation between the empirical
/// distribution of the point set and the uniform distribution over all
/// axis-aligned rectangles anchored at the origin. For 64 Halton points in 2D
/// this bound is generously set at 0.15; the actual discrepancy is typically
/// much lower, confirming the low-discrepancy property of the sequence.
///
/// The N(0,1) output from `generate_opening_tree` is mapped back to uniform
/// [0,1) via the normal CDF `Φ`. Because scrambled Halton is still a valid
/// low-discrepancy sequence modulo the CDF transform, this verifies the
/// end-to-end discrepancy bound. The threshold is 0.15 (more generous than
/// Sobol's 0.1) because the prime bases (2, 3) can produce slightly higher
/// discrepancy than the Gray-code Sobol sequence for small N.
#[test]
fn halton_2d_star_discrepancy() {
    let n = 64_usize;
    let stages = vec![make_stage_halton(0, 0, n)];
    let mut corr = identity_correlation(&[1, 2]);
    let entity_order = vec![EntityId(1), EntityId(2)];

    let dims = ClassDimensions {
        n_hydros: 2,
        n_load_buses: 0,
        n_ncs: 0,
    };
    let tree = generate_opening_tree(42, &stages, 2, &mut corr, &entity_order, dims)
        .expect("generate_opening_tree must succeed");

    assert_eq!(tree.n_stages(), 1);
    assert_eq!(tree.n_openings(0), n);

    // Map N(0,1) values back to uniform [0,1) via the normal CDF.
    let points: Vec<(f64, f64)> = (0..n)
        .map(|k| {
            let s = tree.opening(0, k);
            (norm_cdf(s[0]), norm_cdf(s[1]))
        })
        .collect();

    // Compute 2D star discrepancy via O(N^2) brute force.
    // D* = max over all axis-aligned rectangles [0,u) x [0,v) of
    //      |#{points in [0,u) x [0,v)} / N - u*v|
    // We evaluate at each point (x_i, y_i) as the upper corner.
    let n_f = n as f64;
    let mut d_star = 0.0_f64;

    for &(ux, uy) in &points {
        let count = points
            .iter()
            .filter(|&&(px, py)| px < ux && py < uy)
            .count();
        let empirical = count as f64 / n_f;
        let discrepancy = (empirical - ux * uy).abs();
        d_star = d_star.max(discrepancy);
    }

    assert!(
        d_star < 0.15,
        "2D star discrepancy D*={d_star:.4} exceeds 0.15; \
         expected Halton QMC to achieve low discrepancy for N={n} in 2D"
    );
}

/// Verify that Halton QMC produces standard-normal marginal statistics for N=256.
///
/// For N=256 openings and dim=1, the sample mean must be within 0.15 of 0.0
/// and the sample standard deviation within 0.15 of 1.0. Scrambled Halton
/// improves on pure Monte Carlo by ensuring better coverage of the probability
/// space, typically tightening convergence to the target distribution.
#[test]
fn halton_normal_statistics() {
    let n = 1000_usize;
    let dim = 1_usize;
    let stages = vec![make_stage_halton(0, 0, n)];
    let mut corr = identity_correlation(&[1]);
    let entity_order = vec![EntityId(1)];

    let dims = ClassDimensions {
        n_hydros: dim,
        n_load_buses: 0,
        n_ncs: 0,
    };
    let tree = generate_opening_tree(42, &stages, dim, &mut corr, &entity_order, dims)
        .expect("generate_opening_tree must succeed");

    let values: Vec<f64> = (0..n).map(|o| tree.opening(0, o)[0]).collect();

    let n_f = n as f64;
    let mean = values.iter().sum::<f64>() / n_f;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n_f - 1.0);
    let std = variance.sqrt();

    assert!(
        mean.abs() < 0.15,
        "mean {mean:.4} too far from 0.0 (tolerance 0.15); \
         expected Halton QMC N(0,1) marginal"
    );
    assert!(
        (std - 1.0).abs() < 0.15,
        "std {std:.4} too far from 1.0 (tolerance 0.15); \
         expected Halton QMC N(0,1) marginal"
    );
}

/// Verify that spatial correlation is correctly applied to Halton QMC noise.
///
/// With a 2×2 correlation matrix with off-diagonal rho=0.8 and N=256 openings,
/// the sample Pearson correlation between the two dimensions must be within 0.15
/// of the target 0.8. This exercises the spectral correlation transform applied
/// after Halton sampling inside `generate_opening_tree`.
#[test]
fn halton_correlation_applied() {
    let n = 256_usize;
    let rho = 0.8_f64;
    let stages = vec![make_stage_halton(0, 0, n)];
    let mut corr = correlated_correlation(&[1, 2], rho);
    let entity_order = vec![EntityId(1), EntityId(2)];

    let dims = ClassDimensions {
        n_hydros: 2,
        n_load_buses: 0,
        n_ncs: 0,
    };
    let tree = generate_opening_tree(54321, &stages, 2, &mut corr, &entity_order, dims)
        .expect("generate_opening_tree must succeed");

    let pairs: Vec<(f64, f64)> = (0..n)
        .map(|o| {
            let s = tree.opening(0, o);
            (s[0], s[1])
        })
        .collect();

    let n_f = n as f64;
    let mean_x = pairs.iter().map(|(x, _)| x).sum::<f64>() / n_f;
    let mean_y = pairs.iter().map(|(_, y)| y).sum::<f64>() / n_f;

    let cov_xy = pairs
        .iter()
        .map(|(x, y)| (x - mean_x) * (y - mean_y))
        .sum::<f64>()
        / (n_f - 1.0);
    let var_x = pairs.iter().map(|(x, _)| (x - mean_x).powi(2)).sum::<f64>() / (n_f - 1.0);
    let var_y = pairs.iter().map(|(_, y)| (y - mean_y).powi(2)).sum::<f64>() / (n_f - 1.0);

    let sample_corr = cov_xy / (var_x.sqrt() * var_y.sqrt());

    assert!(
        (sample_corr - rho).abs() < 0.15,
        "sample correlation {sample_corr:.4} too far from target {rho} (tolerance 0.15); \
         spectral correlation transform may not be applied correctly for Halton QMC"
    );
}

/// Verify declaration-order invariance: reversing the entity insertion order
/// in the system produces a bitwise identical opening tree.
///
/// The pipeline (`build_stochastic_context`) sorts entities by `EntityId`
/// internally, so the order in which hydros are supplied to `SystemBuilder`
/// must not affect the generated opening tree. This guarantees that
/// case results are reproducible regardless of the order entities appear in
/// input files.
#[test]
fn halton_declaration_order_invariant() {
    let n_openings = 30_usize;

    // Forward order: EntityId(1) before EntityId(2).
    let hydros_fwd = vec![make_hydro(1), make_hydro(2)];
    // Reversed order: EntityId(2) before EntityId(1).
    let hydros_rev = vec![make_hydro(2), make_hydro(1)];

    let ctx_fwd = build_halton_context(hydros_fwd, n_openings, 42);
    let ctx_rev = build_halton_context(hydros_rev, n_openings, 42);

    let tree_fwd = ctx_fwd.opening_tree();
    let tree_rev = ctx_rev.opening_tree();

    assert_eq!(
        tree_fwd.n_stages(),
        tree_rev.n_stages(),
        "n_stages must be identical regardless of entity insertion order"
    );

    for stage in 0..tree_fwd.n_stages() {
        assert_eq!(
            tree_fwd.n_openings(stage),
            tree_rev.n_openings(stage),
            "n_openings at stage={stage} must be identical regardless of insertion order"
        );
        for opening in 0..tree_fwd.n_openings(stage) {
            assert_eq!(
                tree_fwd.opening(stage, opening),
                tree_rev.opening(stage, opening),
                "opening tree data at stage={stage} opening={opening} must be bitwise \
                 identical regardless of entity insertion order"
            );
        }
    }
}

/// Verify point-wise consistency of the Halton forward-pass path.
///
/// For all scenarios 0..N, `scrambled_halton_point` must produce finite N(0,1)
/// values and different scenarios must produce different noise vectors. This
/// confirms that the communication-free point-wise path (used during the
/// forward pass) is self-consistent: each scenario generates a valid,
/// distinct, and finite noise vector without any inter-worker coordination.
#[test]
fn halton_point_wise_consistency() {
    let n = 64_usize;
    let dim = 3_usize;

    let mut outputs: Vec<Vec<f64>> = Vec::with_capacity(n);

    for scenario in 0..n {
        let spec = HaltonPointSpec {
            sampling_seed: 77,
            iteration: 3,
            scenario: scenario as u32,
            stage_id: 1,
            total_scenarios: n as u32,
            dim,
        };
        let mut output = vec![0.0_f64; dim];
        scrambled_halton_point(&spec, &mut output);

        // All values must be finite N(0,1).
        for (d, &v) in output.iter().enumerate() {
            assert!(
                v.is_finite(),
                "non-finite value at scenario={scenario} dim={d}: {v}"
            );
        }

        outputs.push(output);
    }

    // Different scenarios must produce different noise vectors.
    // Check that no two scenarios share an identical output (highly improbable
    // for any reasonable RNG, but required for correctness).
    for i in 0..n {
        for j in (i + 1)..n {
            assert_ne!(
                outputs[i], outputs[j],
                "scenarios {i} and {j} produced identical noise vectors — \
                 point-wise Halton is not scenario-distinct"
            );
        }
    }
}
