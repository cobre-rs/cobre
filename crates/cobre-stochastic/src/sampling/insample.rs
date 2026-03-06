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
/// Deterministically chooses an opening index `j` in `{0, ..., n_openings(stage_idx) - 1}`
/// by deriving a seed from `(base_seed, iteration, scenario, stage)` via SipHash-1-3 and
/// sampling a `Pcg64` RNG. Returns both the selected index and the corresponding noise
/// slice so the caller can log which opening was chosen.
///
/// The `stage` parameter is the domain identifier (`stage.id`) used for seed derivation
/// per DEC-017. The `stage_idx` parameter is the array-position index used to address
/// the opening tree.
///
/// # Panics
///
/// Panics if `stage_idx >= tree.n_stages()` or if the tree has zero openings at that
/// stage (delegated to [`OpeningTreeView::opening`] and [`OpeningTreeView::n_openings`]).
///
/// # Examples
///
/// ```no_run
/// use cobre_stochastic::tree::opening_tree::OpeningTree;
/// use cobre_stochastic::sampling::insample::sample_forward;
///
/// // Obtain an OpeningTree via generate_opening_tree (public API).
/// // Given a view over a tree with 5 openings per stage and dim=2:
/// // let view = tree.view();
/// // let (idx, slice) = sample_forward(&view, 42, 0, 0, 0, 0);
/// // assert!(idx < 5);
/// // assert_eq!(slice.len(), 2);
/// ```
#[must_use]
pub fn sample_forward<'a>(
    tree: &'a OpeningTreeView<'a>,
    base_seed: u64,
    iteration: u32,
    scenario: u32,
    stage: u32,
    stage_idx: usize,
) -> (usize, &'a [f64]) {
    let seed = derive_forward_seed(base_seed, iteration, scenario, stage);
    let mut rng = rng_from_seed(seed);
    let n = tree.n_openings(stage_idx);
    #[allow(clippy::cast_possible_truncation)]
    let j = (rng.random::<u64>() % n as u64) as usize;
    (j, tree.opening(stage_idx, j))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::sample_forward;
    use crate::tree::opening_tree::OpeningTree;

    /// Build a tree with uniform branching: `n_stages` stages, each with
    /// `openings` openings of dimension `dim`. Values are `0.0, 1.0, 2.0, ...`.
    fn uniform_tree(n_stages: usize, openings: usize, dim: usize) -> OpeningTree {
        let total = n_stages * openings * dim;
        let data: Vec<f64> = (0_u32..u32::try_from(total).unwrap())
            .map(f64::from)
            .collect();
        OpeningTree::from_parts(data, vec![openings; n_stages], dim)
    }

    /// Same inputs return bitwise-identical `(index, slice)`.
    #[test]
    fn determinism_same_inputs_same_output() {
        let tree = uniform_tree(3, 5, 2);
        let view = tree.view();

        let (idx_a, slice_a) = sample_forward(&view, 42, 0, 0, 0, 0);
        let (idx_b, slice_b) = sample_forward(&view, 42, 0, 0, 0, 0);

        assert_eq!(idx_a, idx_b);
        assert_eq!(slice_a, slice_b);
    }

    /// Varying `scenario` produces different indices across many calls.
    /// Verify that among 20 consecutive scenario pairs `(s, s+1)`, at least
    /// half differ.
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

    /// 1000 calls with varying `scenario` all return indices in bounds.
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

    /// Returned slice equals `tree.opening(stage_idx, index)`.
    #[test]
    fn returned_slice_matches_tree_opening() {
        let tree = uniform_tree(3, 5, 2);
        let view = tree.view();

        let (idx, slice) = sample_forward(&view, 42, 0, 0, 0, 0);
        assert_eq!(slice, tree.opening(0, idx));
    }

    /// Varying `iteration` changes the selected index.
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

    /// `stage` (domain id) is used in seed derivation, not `stage_idx`.
    /// Changing `stage` while keeping `stage_idx` fixed must change the index.
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

    /// A tree with only one opening always returns index 0.
    #[test]
    fn single_opening_always_index_zero() {
        let tree = uniform_tree(1, 1, 3);
        let view = tree.view();

        for scenario in 0_u32..50 {
            let (idx, _) = sample_forward(&view, 99, 0, scenario, 0, 0);
            assert_eq!(idx, 0, "single-opening tree must always return index 0");
        }
    }

    /// The noise slice has `dim` elements.
    #[test]
    fn slice_length_equals_dim() {
        let dim = 7;
        let tree = uniform_tree(2, 5, dim);
        let view = tree.view();

        let (_, slice) = sample_forward(&view, 1, 2, 3, 0, 0);
        assert_eq!(slice.len(), dim);
    }
}
