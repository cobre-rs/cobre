//! Integration tests validating Halton QMC statistical properties through the
//! full `generate_opening_tree` pipeline (`NoiseMethod::QmcHalton`).

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss
)]

use cobre_core::{
    EntityId, Hydro, SystemBuilder,
    scenario::{InflowModel, SamplingScheme},
    temporal::{NoiseMethod, ScenarioSourceConfig, Stage},
};
use cobre_stochastic::tree::generate::OpeningTreeGenerationInputs;
use cobre_stochastic::{
    ClassDimensions, ClassSchemes, NoisePointSpec, OpeningTreeInputs, build_stochastic_context,
    generate_opening_tree,
    tree::qmc_halton::{HaltonPrecomputed, scrambled_halton_point},
};

mod common;
use common::{
    StageSpec, correlated_correlation, default_inflow_model, deficit_bus, identity_correlation,
    identity_correlation_model, make_stage, norm_cdf, single_block, sized_hydro,
};

// ---------------------------------------------------------------------------
// Helpers shared across tests
// ---------------------------------------------------------------------------

/// No blocks — for direct use with `generate_opening_tree`.
fn make_stage_halton(index: usize, id: i32, branching_factor: usize) -> Stage {
    make_stage(StageSpec {
        id,
        index: Some(index),
        season_id: Some(0),
        blocks: Vec::new(),
        scenario_config: ScenarioSourceConfig {
            branching_factor,
            noise_method: NoiseMethod::QmcHalton,
        },
        ..Default::default()
    })
}

/// One block — for use with `build_stochastic_context`.
fn make_stage_halton_with_block(index: usize, id: i32, branching_factor: usize) -> Stage {
    make_stage(StageSpec {
        id,
        index: Some(index),
        season_id: Some(0),
        blocks: single_block("SINGLE", 744.0),
        scenario_config: ScenarioSourceConfig {
            branching_factor,
            noise_method: NoiseMethod::QmcHalton,
        },
        ..Default::default()
    })
}

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
            inflow_models.push(default_inflow_model(hid, sid));
        }
    }

    let system = SystemBuilder::new()
        .buses(vec![deficit_bus(0)])
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
        OpeningTreeInputs::default(),
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

/// The 0.15 threshold is more generous than Sobol's 0.1 because the prime bases
/// (2, 3) produce slightly higher discrepancy than the Gray-code Sobol sequence
/// for small N.
#[test]
fn halton_2d_star_discrepancy() {
    let n = 64_usize;
    let stages = vec![make_stage_halton(0, 0, n)];
    let corr = identity_correlation(&[1, 2]);
    let entity_order = vec![EntityId(1), EntityId(2)];

    let dims = ClassDimensions {
        n_hydros: 2,
        n_load_buses: 0,
        n_ncs: 0,
    };
    let tree = generate_opening_tree(
        42,
        &stages,
        &corr,
        &entity_order,
        dims,
        &OpeningTreeGenerationInputs::default(),
    )
    .expect("generate_opening_tree must succeed");

    assert_eq!(tree.n_stages(), 1);
    assert_eq!(tree.n_openings(0), n);

    let points: Vec<(f64, f64)> = (0..n)
        .map(|k| {
            let s = tree.opening(0, k);
            (norm_cdf(s[0]), norm_cdf(s[1]))
        })
        .collect();

    // D* = max over axis-aligned rectangles [0,u) x [0,v), evaluated at each point
    //      as the upper corner, of |#{points in [0,u) x [0,v)} / N - u*v|.
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

#[test]
fn halton_normal_statistics() {
    let n = 1000_usize;
    let dim = 1_usize;
    let stages = vec![make_stage_halton(0, 0, n)];
    let corr = identity_correlation(&[1]);
    let entity_order = vec![EntityId(1)];

    let dims = ClassDimensions {
        n_hydros: dim,
        n_load_buses: 0,
        n_ncs: 0,
    };
    let tree = generate_opening_tree(
        42,
        &stages,
        &corr,
        &entity_order,
        dims,
        &OpeningTreeGenerationInputs::default(),
    )
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

#[test]
fn halton_correlation_applied() {
    let n = 256_usize;
    let rho = 0.8_f64;
    let stages = vec![make_stage_halton(0, 0, n)];
    let corr = correlated_correlation(&[1, 2], rho);
    let entity_order = vec![EntityId(1), EntityId(2)];

    let dims = ClassDimensions {
        n_hydros: 2,
        n_load_buses: 0,
        n_ncs: 0,
    };
    let tree = generate_opening_tree(
        54321,
        &stages,
        &corr,
        &entity_order,
        dims,
        &OpeningTreeGenerationInputs::default(),
    )
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

/// `build_stochastic_context` sorts entities by `EntityId` internally, so the
/// order hydros are supplied to `SystemBuilder` must not change the opening tree.
#[test]
fn halton_declaration_order_invariant() {
    let n_openings = 30_usize;

    let hydros_fwd = vec![sized_hydro(1), sized_hydro(2)];
    let hydros_rev = vec![sized_hydro(2), sized_hydro(1)];

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

#[test]
fn halton_point_wise_consistency() {
    let n = 64_usize;
    let dim = 3_usize;

    let ctx = HaltonPrecomputed::new(77, 3, 1, dim, n as u32);
    let mut outputs: Vec<Vec<f64>> = Vec::with_capacity(n);

    for scenario in 0..n {
        let spec = NoisePointSpec {
            sampling_seed: 77,
            iteration: 3,
            scenario: scenario as u32,
            stream_id: 1,
            total_scenarios: n as u32,
            dim,
        };
        let mut output = vec![0.0_f64; dim];
        scrambled_halton_point(&spec, &ctx, &mut output);

        for (d, &v) in output.iter().enumerate() {
            assert!(
                v.is_finite(),
                "non-finite value at scenario={scenario} dim={d}: {v}"
            );
        }

        outputs.push(output);
    }

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
