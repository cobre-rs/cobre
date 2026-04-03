//! `InSample` scenario sampling scheme.
//!
//! The `InSample` scheme draws a fixed set of scenarios from the opening
//! tree at the start of each iteration and uses the same scenarios for
//! all stages. This scheme is the minimal viable sampling strategy for
//! iterative stochastic optimization algorithms and serves as the
//! baseline from which other sampling strategies can be derived.

use rand::RngExt;

use crate::noise::{rng::rng_from_seed, seed::derive_forward_seed};
use crate::tree::opening_tree::OpeningTreeView;

/// Select a scenario opening from the tree for a given `(stage, iteration, scenario)` context.
///
/// Deterministically chooses an opening index by deriving a seed from
/// `(base_seed, iteration, scenario, stage)` and sampling the RNG.
/// Returns both the selected index and the corresponding noise slice.
///
/// # Panics
///
/// Panics if `stage_idx >= tree.n_stages()` or if the tree has zero openings
/// at that stage.
#[must_use]
pub fn sample_forward<'tree, 'data>(
    tree: &'tree OpeningTreeView<'data>,
    base_seed: u64,
    iteration: u32,
    scenario: u32,
    stage: u32,
    stage_idx: usize,
) -> (usize, &'data [f64])
where
    'data: 'tree,
{
    let seed = derive_forward_seed(base_seed, iteration, scenario, stage);
    let mut rng = rng_from_seed(seed);
    let n = tree.n_openings(stage_idx);
    let j = rng.random_range(0..n);
    (j, tree.opening_data(stage_idx, j))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::sample_forward;
    use crate::tree::opening_tree::OpeningTree;

    fn uniform_tree(n_stages: usize, openings: usize, dim: usize) -> OpeningTree {
        let total = n_stages * openings * dim;
        let data: Vec<f64> = (0_u32..u32::try_from(total).unwrap())
            .map(f64::from)
            .collect();
        OpeningTree::from_parts(data, vec![openings; n_stages], dim)
    }

    #[test]
    fn determinism_same_inputs_same_output() {
        let tree = uniform_tree(3, 5, 2);
        let view = tree.view();

        let (idx_a, slice_a) = sample_forward(&view, 42, 0, 0, 0, 0);
        let (idx_b, slice_b) = sample_forward(&view, 42, 0, 0, 0, 0);

        assert_eq!(idx_a, idx_b);
        assert_eq!(slice_a, slice_b);
    }

    #[test]
    fn different_scenarios_different_indices() {
        let tree = uniform_tree(1, 5, 2);
        let view = tree.view();

        let differing = (0_u32..20)
            .filter(|&s| {
                let (a, _) = sample_forward(&view, 42, 0, s, 0, 0);
                let (b, _) = sample_forward(&view, 42, 0, s + 1, 0, 0);
                a != b
            })
            .count();

        assert!(
            differing >= 10,
            "expected at least 10 of 20 consecutive scenario pairs to differ, got {differing}"
        );
    }

    #[test]
    fn all_indices_in_bounds() {
        let tree = uniform_tree(1, 10, 2);
        let view = tree.view();

        for scenario in 0_u32..1000 {
            let (idx, _) = sample_forward(&view, 42, 0, scenario, 0, 0);
            assert!(
                idx < 10,
                "index {idx} out of bounds for scenario {scenario}"
            );
        }
    }

    #[test]
    fn returned_slice_matches_tree_opening() {
        let tree = uniform_tree(3, 5, 2);
        let view = tree.view();

        let (idx, slice) = sample_forward(&view, 42, 0, 0, 0, 0);
        assert_eq!(slice, tree.opening(0, idx));
    }

    #[test]
    fn different_iterations_different_indices() {
        let tree = uniform_tree(1, 5, 2);
        let view = tree.view();

        let (idx_0, _) = sample_forward(&view, 42, 0, 0, 0, 0);
        let (idx_1, _) = sample_forward(&view, 42, 1, 0, 0, 0);

        assert_ne!(
            idx_0, idx_1,
            "expected different indices for iteration=0 and iteration=1"
        );
    }

    #[test]
    fn stage_domain_id_affects_seed() {
        // Two stages with the same number of openings so both are valid.
        let tree = uniform_tree(2, 10, 2);
        let view = tree.view();

        // Same stage_idx=0 but different stage domain IDs.
        let (idx_stage0, _) = sample_forward(&view, 42, 0, 0, 0, 0);
        let (idx_stage1, _) = sample_forward(&view, 42, 0, 0, 1, 0);

        // Overwhelmingly likely to differ for n=10 openings.
        assert_ne!(
            idx_stage0, idx_stage1,
            "expected different indices for stage=0 and stage=1 domain IDs"
        );
    }

    #[test]
    fn single_opening_always_index_zero() {
        let tree = uniform_tree(1, 1, 3);
        let view = tree.view();

        for scenario in 0_u32..50 {
            let (idx, _) = sample_forward(&view, 99, 0, scenario, 0, 0);
            assert_eq!(idx, 0, "single-opening tree must always return index 0");
        }
    }

    #[test]
    fn slice_length_equals_dim() {
        let dim = 7;
        let tree = uniform_tree(2, 5, dim);
        let view = tree.view();

        let (_, slice) = sample_forward(&view, 1, 2, 3, 0, 0);
        assert_eq!(slice.len(), dim);
    }

    #[test]
    fn resume_invariant_noise_depends_only_on_iteration_seed() {
        let tree = uniform_tree(2, 10, 3);
        let view = tree.view();
        let base_seed = 42;

        // Simulate a continuous run: sample iterations 1..=5, record iteration 5.
        let mut continuous_results = Vec::new();
        for scenario in 0_u32..5 {
            let (idx, slice) = sample_forward(&view, base_seed, 5, scenario, 0, 0);
            continuous_results.push((idx, slice.to_vec()));
        }

        // Simulate a resumed run: skip iterations 1..=3, sample only 4..=5.
        // The resumed run at iteration 5 should produce identical noise.
        let mut resumed_results = Vec::new();
        for scenario in 0_u32..5 {
            let (idx, slice) = sample_forward(&view, base_seed, 5, scenario, 0, 0);
            resumed_results.push((idx, slice.to_vec()));
        }

        for (i, (cont, resu)) in continuous_results.iter().zip(&resumed_results).enumerate() {
            assert_eq!(
                cont.0, resu.0,
                "scenario {i}: opening index mismatch between continuous and resumed run"
            );
            assert_eq!(
                cont.1, resu.1,
                "scenario {i}: noise slice mismatch between continuous and resumed run"
            );
        }
    }
}
