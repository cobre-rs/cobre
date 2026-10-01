//! Integration tests validating LHS statistical properties through the full
//! `generate_opening_tree` pipeline (`NoiseMethod::Lhs`).

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
    tree::lhs::{LhsPrecomputed, sample_lhs_point},
};

mod common;
use common::{
    StageSpec, correlated_correlation, default_inflow_model, deficit_bus, identity_correlation,
    identity_correlation_model, make_stage, norm_cdf, single_block, sized_hydro,
};

// ---------------------------------------------------------------------------
// Helpers shared across tests
// ---------------------------------------------------------------------------

fn cdf_floor_stratum(z: f64, n: usize, n_f: f64) -> usize {
    ((norm_cdf(z) * n_f).floor() as usize).min(n - 1)
}

fn make_stage_lhs(index: usize, id: i32, branching_factor: usize) -> Stage {
    make_stage(StageSpec {
        id,
        index: Some(index),
        season_id: Some(0),
        blocks: single_block("SINGLE", 744.0),
        scenario_config: ScenarioSourceConfig {
            branching_factor,
            noise_method: NoiseMethod::Lhs,
        },
        ..Default::default()
    })
}

/// For direct use with `generate_opening_tree`.
fn make_stage_lhs_no_block(index: usize, id: i32, branching_factor: usize) -> Stage {
    make_stage(StageSpec {
        id,
        index: Some(index),
        season_id: Some(0),
        blocks: Vec::new(),
        scenario_config: ScenarioSourceConfig {
            branching_factor,
            noise_method: NoiseMethod::Lhs,
        },
        ..Default::default()
    })
}

fn build_lhs_context(
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
        make_stage_lhs(0, 0, n_openings),
        make_stage_lhs(1, 1, n_openings),
        make_stage_lhs(2, 2, n_openings),
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
        .expect("build_lhs_context: system build must succeed");

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
    .expect("build_lhs_context: build_stochastic_context must succeed")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn lhs_marginal_uniformity() {
    let n = 100_usize;
    let dim = 5_usize;
    let stages = vec![make_stage_lhs_no_block(0, 0, n)];
    let corr = identity_correlation(&[1, 2, 3, 4, 5]);
    let entity_order = vec![
        EntityId(1),
        EntityId(2),
        EntityId(3),
        EntityId(4),
        EntityId(5),
    ];

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

    assert_eq!(tree.n_stages(), 1);
    assert_eq!(tree.n_openings(0), n);

    let n_f = n as f64;
    for d in 0..dim {
        let mut strata: Vec<usize> = (0..n)
            .map(|k| cdf_floor_stratum(tree.opening(0, k)[d], n, n_f))
            .collect();
        strata.sort_unstable();
        let expected: Vec<usize> = (0..n).collect();
        assert_eq!(
            strata, expected,
            "dim {d}: CDF-floor indices are not a permutation of 0..{n} — \
             marginal uniformity violated"
        );
    }
}

#[test]
fn lhs_no_stratum_collision() {
    let n = 80_usize;
    let dim = 4_usize;
    let stages = vec![make_stage_lhs_no_block(0, 0, n)];
    let corr = identity_correlation(&[1, 2, 3, 4]);
    let entity_order = vec![EntityId(1), EntityId(2), EntityId(3), EntityId(4)];

    let dims = ClassDimensions {
        n_hydros: dim,
        n_load_buses: 0,
        n_ncs: 0,
    };
    let tree = generate_opening_tree(
        99,
        &stages,
        &corr,
        &entity_order,
        dims,
        &OpeningTreeGenerationInputs::default(),
    )
    .expect("generate_opening_tree must succeed");

    let n_f = n as f64;
    for d in 0..dim {
        let mut strata: Vec<usize> = (0..n)
            .map(|k| cdf_floor_stratum(tree.opening(0, k)[d], n, n_f))
            .collect();
        let original_len = strata.len();
        strata.sort_unstable();
        strata.dedup();
        assert_eq!(
            strata.len(),
            original_len,
            "dim {d}: stratum collision detected — two openings share the same stratum"
        );
    }
}

#[test]
fn lhs_normal_statistics() {
    let n = 1000_usize;
    let dim = 1_usize;
    let stages = vec![make_stage_lhs_no_block(0, 0, n)];
    let corr = identity_correlation(&[1]);
    let entity_order = vec![EntityId(1)];

    let dims = ClassDimensions {
        n_hydros: dim,
        n_load_buses: 0,
        n_ncs: 0,
    };
    let tree = generate_opening_tree(
        12345,
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
        mean.abs() < 0.1,
        "mean {mean:.4} too far from 0.0 (tolerance 0.1); \
         expected LHS N(0,1) marginal"
    );
    assert!(
        (std - 1.0).abs() < 0.1,
        "std {std:.4} too far from 1.0 (tolerance 0.1); \
         expected LHS N(0,1) marginal"
    );
}

#[test]
fn lhs_correlation_applied() {
    let n = 2000_usize;
    let rho = 0.8_f64;
    let stages = vec![make_stage_lhs_no_block(0, 0, n)];
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
        (sample_corr - rho).abs() < 0.1,
        "sample correlation {sample_corr:.4} too far from target {rho} (tolerance 0.1); \
         spectral correlation transform may not be applied correctly for LHS"
    );
}

/// `build_stochastic_context` sorts entities by `EntityId` internally, so the
/// order hydros are supplied to `SystemBuilder` must not change the opening tree.
#[test]
fn lhs_declaration_order_invariant() {
    let n_openings = 30_usize;

    let hydros_fwd = vec![sized_hydro(1), sized_hydro(2)];
    let hydros_rev = vec![sized_hydro(2), sized_hydro(1)];

    let ctx_fwd = build_lhs_context(hydros_fwd, n_openings, 42);
    let ctx_rev = build_lhs_context(hydros_rev, n_openings, 42);

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
fn lhs_point_wise_stratum_consistency() {
    let n = 60_usize;
    let dim = 3_usize;
    let n_f = n as f64;

    let ctx = LhsPrecomputed::new(77, 3, 1, dim, n as u32);
    let mut strata_by_dim: Vec<Vec<usize>> = (0..dim).map(|_| Vec::with_capacity(n)).collect();

    for scenario in 0..n {
        let mut output = vec![0.0_f64; dim];
        let spec = NoisePointSpec {
            sampling_seed: 77,
            iteration: 3,
            scenario: scenario as u32,
            stream_id: 1,
            total_scenarios: n as u32,
            dim,
        };
        sample_lhs_point(&spec, &ctx, &mut output);

        for (d, &v) in output.iter().enumerate() {
            strata_by_dim[d].push(cdf_floor_stratum(v, n, n_f));
        }
    }

    for (d, strata) in strata_by_dim.iter_mut().enumerate() {
        strata.sort_unstable();
        let expected: Vec<usize> = (0..n).collect();
        assert_eq!(
            *strata, expected,
            "dim {d}: point-wise strata across all scenarios are not a permutation of \
             0..{n} — LHS design property violated"
        );
    }
}
