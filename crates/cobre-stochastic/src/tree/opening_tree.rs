//! Opening scenario tree data structure.
//!
//! Defines the in-memory representation of the pre-generated noise realisations
//! used during the backward pass of iterative optimisation algorithms. The tree
//! is constructed once before the optimisation loop and remains read-only
//! throughout.

/// Fixed opening tree holding pre-generated noise realisations for
/// the backward pass of iterative optimisation algorithms.
///
/// All noise values are stored in a flat contiguous array with
/// stage-major ordering: all openings for stage 0, then all openings
/// for stage 1, etc. Within each stage, openings are contiguous blocks
/// of `dim` f64 values.
///
/// Access pattern: `data[stage_offsets[stage] + opening_idx * dim .. + dim]`
///
/// The sentinel value `stage_offsets[n_stages]` equals `data.len()`,
/// so bounds checks are always exact without special-casing the last stage.
///
/// # Memory layout
///
/// ```text
/// stage 0: [opening_0_entity_0 .. opening_0_entity_{dim-1}]
///                   [opening_1_entity_0 .. opening_1_entity_{dim-1}]
///                   ...
/// stage 1: [opening_0_entity_0 .. opening_0_entity_{dim-1}]
///                   ...
/// ```
///
/// See [Scenario Generation SS2.3a](scenario-generation.md) for the
/// full type specification and memory layout rationale.
#[derive(Debug)]
pub struct OpeningTree {
    data: Box<[f64]>,
    stage_offsets: Box<[usize]>,
    openings_per_stage: Box<[usize]>,
    n_stages: usize,
    dim: usize,
}

impl OpeningTree {
    /// Construct an `OpeningTree` from its raw parts.
    ///
    /// `stage_offsets` is computed internally from `openings_per_stage` and
    /// `dim`. The sentinel entry `stage_offsets[n_stages]` is set to `data.len()`.
    ///
    /// # Panics
    ///
    /// Panics if `data.len()` does not equal
    /// `openings_per_stage.iter().sum::<usize>() * dim`.
    #[must_use]
    pub fn from_parts(data: Vec<f64>, openings_per_stage: Vec<usize>, dim: usize) -> Self {
        let n_stages = openings_per_stage.len();

        let mut stage_offsets = Vec::with_capacity(n_stages + 1);
        let mut offset = 0usize;
        stage_offsets.push(offset);
        for &n_openings in &openings_per_stage {
            offset += n_openings * dim;
            stage_offsets.push(offset);
        }

        assert!(
            data.len() == stage_offsets[n_stages],
            "OpeningTree::from_parts: data.len() ({}) does not match expected total ({} = sum(openings_per_stage) * dim)",
            data.len(),
            stage_offsets[n_stages],
        );

        Self {
            data: data.into_boxed_slice(),
            stage_offsets: stage_offsets.into_boxed_slice(),
            openings_per_stage: openings_per_stage.into_boxed_slice(),
            n_stages,
            dim,
        }
    }

    /// Return a read-only borrowed view over this tree.
    #[must_use]
    pub fn view(&self) -> OpeningTreeView<'_> {
        OpeningTreeView {
            data: &self.data,
            stage_offsets: &self.stage_offsets,
            openings_per_stage: &self.openings_per_stage,
            n_stages: self.n_stages,
            dim: self.dim,
        }
    }

    /// Return the noise slice for a specific `(stage, opening_idx)` pair.
    ///
    /// # Panics
    ///
    /// Panics if `stage >= n_stages` or `opening_idx >= n_openings(stage)`.
    #[must_use]
    pub fn opening(&self, stage: usize, opening_idx: usize) -> &[f64] {
        assert!(
            stage < self.n_stages,
            "OpeningTree::opening: stage {stage} out of bounds (n_stages = {})",
            self.n_stages
        );
        assert!(
            opening_idx < self.openings_per_stage[stage],
            "OpeningTree::opening: opening_idx {opening_idx} out of bounds for stage {stage} (n_openings = {})",
            self.openings_per_stage[stage]
        );
        let start = self.stage_offsets[stage] + opening_idx * self.dim;
        &self.data[start..start + self.dim]
    }

    /// Return the number of openings (branching factor) at the given stage.
    ///
    /// # Panics
    ///
    /// Panics if `stage >= n_stages`.
    #[must_use]
    pub fn n_openings(&self, stage: usize) -> usize {
        assert!(
            stage < self.n_stages,
            "OpeningTree::n_openings: stage {stage} out of bounds (n_stages = {})",
            self.n_stages
        );
        self.openings_per_stage[stage]
    }

    /// Return the number of stages in the tree.
    #[must_use]
    pub fn n_stages(&self) -> usize {
        self.n_stages
    }

    /// Return the number of entities (noise dimension) per opening vector.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Return the total number of f64 elements in the backing array.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Return `true` if the backing array is empty.
    ///
    /// Required by Clippy when [`len`](Self::len) is defined.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Return the total size in bytes of the backing array.
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        std::mem::size_of_val(&*self.data)
    }

    /// Return a reference to the flat contiguous backing array.
    ///
    /// The layout is stage-major: all openings for stage 0, then stage 1, etc.
    /// Within each stage, openings are contiguous blocks of `dim` f64 values.
    #[must_use]
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    /// Return a reference to the per-stage branching factor array.
    #[must_use]
    pub fn openings_per_stage_slice(&self) -> &[usize] {
        &self.openings_per_stage
    }
}

/// Borrowed read-only view over opening tree data.
///
/// Provides the same access API as [`OpeningTree`] but borrows
/// the underlying storage. Used by downstream crates that consume
/// the tree without owning it.
pub struct OpeningTreeView<'a> {
    data: &'a [f64],
    stage_offsets: &'a [usize],
    openings_per_stage: &'a [usize],
    n_stages: usize,
    dim: usize,
}

impl OpeningTreeView<'_> {
    /// Return the noise slice for a specific `(stage, opening_idx)` pair.
    ///
    /// # Panics
    ///
    /// Panics if `stage >= n_stages` or `opening_idx >= n_openings(stage)`.
    #[must_use]
    pub fn opening(&self, stage: usize, opening_idx: usize) -> &[f64] {
        assert!(
            stage < self.n_stages,
            "OpeningTreeView::opening: stage {stage} out of bounds (n_stages = {})",
            self.n_stages
        );
        assert!(
            opening_idx < self.openings_per_stage[stage],
            "OpeningTreeView::opening: opening_idx {opening_idx} out of bounds for stage {stage} (n_openings = {})",
            self.openings_per_stage[stage]
        );
        let start = self.stage_offsets[stage] + opening_idx * self.dim;
        &self.data[start..start + self.dim]
    }

    /// Return the number of openings (branching factor) at the given stage.
    ///
    /// # Panics
    ///
    /// Panics if `stage >= n_stages`.
    #[must_use]
    pub fn n_openings(&self, stage: usize) -> usize {
        assert!(
            stage < self.n_stages,
            "OpeningTreeView::n_openings: stage {stage} out of bounds (n_stages = {})",
            self.n_stages
        );
        self.openings_per_stage[stage]
    }

    /// Return the number of stages in the tree.
    #[must_use]
    pub fn n_stages(&self) -> usize {
        self.n_stages
    }

    /// Return the number of entities (noise dimension) per opening vector.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Return the total number of f64 elements in the backing slice.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Return `true` if the backing slice is empty.
    ///
    /// Required by Clippy when [`len`](Self::len) is defined.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Return the total size in bytes of the backing slice.
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        std::mem::size_of_val(self.data)
    }

    /// Return a reference to the flat contiguous backing slice.
    ///
    /// The layout is stage-major: all openings for stage 0, then stage 1, etc.
    /// Within each stage, openings are contiguous blocks of `dim` f64 values.
    #[must_use]
    pub fn data(&self) -> &[f64] {
        self.data
    }

    /// Return a reference to the per-stage branching factor slice.
    #[must_use]
    pub fn openings_per_stage_slice(&self) -> &[usize] {
        self.openings_per_stage
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a tree with uniform branching: `n_stages` stages, each with
    /// `openings` openings, each opening of dimension `dim`.
    /// Values are `0.0, 1.0, 2.0, ...` in row-major order.
    fn uniform_tree(n_stages: usize, openings: usize, dim: usize) -> OpeningTree {
        let total = n_stages * openings * dim;
        // Use u32 range to avoid cast_precision_loss on usize->f64 conversion.
        let data: Vec<f64> = (0_u32..u32::try_from(total).expect("total fits in u32"))
            .map(f64::from)
            .collect();
        let ops = vec![openings; n_stages];
        OpeningTree::from_parts(data, ops, dim)
    }

    #[test]
    fn opening_stage0_opening0_returns_first_dim_elements() {
        // AC1: from_parts([1,2,3, 4,5,6, 7,8,9], [1,2], 3)
        //      opening(0,0) == [1,2,3]
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let tree = OpeningTree::from_parts(data, vec![1, 2], 3);
        assert_eq!(tree.opening(0, 0), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn opening_stage1_returns_correct_slices() {
        // AC2: opening(1,0) == [4,5,6], opening(1,1) == [7,8,9]
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let tree = OpeningTree::from_parts(data, vec![1, 2], 3);
        assert_eq!(tree.opening(1, 0), &[4.0, 5.0, 6.0]);
        assert_eq!(tree.opening(1, 1), &[7.0, 8.0, 9.0]);
    }

    #[test]
    fn n_openings_matches_branching_factors() {
        // AC3: n_openings(0) == 1, n_openings(1) == 2
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let tree = OpeningTree::from_parts(data, vec![1, 2], 3);
        assert_eq!(tree.n_openings(0), 1);
        assert_eq!(tree.n_openings(1), 2);
    }

    #[test]
    fn view_returns_identical_data_to_owned() {
        // AC4: view().opening(0,0) == tree.opening(0,0)
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let tree = OpeningTree::from_parts(data, vec![1, 2], 3);
        let view = tree.view();
        assert_eq!(view.opening(0, 0), tree.opening(0, 0));
        assert_eq!(view.opening(1, 0), tree.opening(1, 0));
        assert_eq!(view.opening(1, 1), tree.opening(1, 1));
    }

    #[test]
    fn len_and_size_bytes_uniform_branching() {
        // AC5: from_parts([1.0; 30], [5,5,5], 2): len==30, size_bytes==240
        let tree = OpeningTree::from_parts(vec![1.0; 30], vec![5, 5, 5], 2);
        assert_eq!(tree.len(), 30);
        assert_eq!(tree.size_bytes(), 240);
    }

    // -----------------------------------------------------------------------
    // Additional unit tests
    // -----------------------------------------------------------------------

    #[test]
    fn uniform_branching_3_stages_5_openings_2_entities() {
        let tree = uniform_tree(3, 5, 2);
        assert_eq!(tree.n_stages(), 3);
        assert_eq!(tree.dim(), 2);
        assert_eq!(tree.len(), 30);

        // Stage 0, opening 0: elements 0,1
        assert_eq!(tree.opening(0, 0), &[0.0_f64, 1.0]);
        // Stage 0, opening 4: elements 8,9
        assert_eq!(tree.opening(0, 4), &[8.0_f64, 9.0]);
        // Stage 1, opening 0: elements 10,11
        assert_eq!(tree.opening(1, 0), &[10.0_f64, 11.0]);
        // Stage 2, opening 4: elements 28,29
        assert_eq!(tree.opening(2, 4), &[28.0_f64, 29.0]);
    }

    #[test]
    fn variable_branching_access() {
        // stages with different branching: [3, 1, 4], dim=2
        // total = (3+1+4)*2 = 16 elements
        let data: Vec<f64> = (0_i32..16).map(f64::from).collect();
        let tree = OpeningTree::from_parts(data, vec![3, 1, 4], 2);

        assert_eq!(tree.n_stages(), 3);
        assert_eq!(tree.n_openings(0), 3);
        assert_eq!(tree.n_openings(1), 1);
        assert_eq!(tree.n_openings(2), 4);

        // stage 0: offsets 0..6 (3 openings * 2)
        assert_eq!(tree.opening(0, 0), &[0.0_f64, 1.0]);
        assert_eq!(tree.opening(0, 2), &[4.0_f64, 5.0]);

        // stage 1: offsets 6..8 (1 opening * 2)
        assert_eq!(tree.opening(1, 0), &[6.0_f64, 7.0]);

        // stage 2: offsets 8..16 (4 openings * 2)
        assert_eq!(tree.opening(2, 0), &[8.0_f64, 9.0]);
        assert_eq!(tree.opening(2, 3), &[14.0_f64, 15.0]);
    }

    #[test]
    fn single_stage_single_opening() {
        let tree = OpeningTree::from_parts(vec![42.0, 43.0], vec![1], 2);
        assert_eq!(tree.n_stages(), 1);
        assert_eq!(tree.n_openings(0), 1);
        assert_eq!(tree.opening(0, 0), &[42.0, 43.0]);
        assert!(!tree.is_empty());
    }

    #[test]
    #[should_panic(expected = "data.len()")]
    fn from_parts_panics_on_wrong_data_length() {
        // Expected 6 elements (2 openings * 3 dim), provide 5
        let _tree = OpeningTree::from_parts(vec![1.0; 5], vec![2], 3);
    }

    #[test]
    #[should_panic(expected = "stage 3 out of bounds")]
    fn opening_panics_on_out_of_bounds_stage() {
        let tree = uniform_tree(3, 2, 2);
        let _ = tree.opening(3, 0);
    }

    #[test]
    #[should_panic(expected = "opening_idx 5 out of bounds")]
    fn opening_panics_on_out_of_bounds_opening_idx() {
        let tree = uniform_tree(3, 5, 2);
        let _ = tree.opening(0, 5);
    }

    #[test]
    fn view_accessors_match_owned_for_variable_branching() {
        let data: Vec<f64> = (0_i32..16).map(f64::from).collect();
        let tree = OpeningTree::from_parts(data, vec![3, 1, 4], 2);
        let view = tree.view();

        assert_eq!(view.n_stages(), tree.n_stages());
        assert_eq!(view.dim(), tree.dim());
        assert_eq!(view.len(), tree.len());
        assert_eq!(view.size_bytes(), tree.size_bytes());

        for stage in 0..tree.n_stages() {
            assert_eq!(view.n_openings(stage), tree.n_openings(stage));
            for idx in 0..tree.n_openings(stage) {
                assert_eq!(view.opening(stage, idx), tree.opening(stage, idx));
            }
        }
    }

    #[test]
    fn is_empty_false_for_non_empty_tree() {
        let tree = uniform_tree(2, 3, 4);
        assert!(!tree.is_empty());
    }

    #[test]
    fn size_bytes_is_8_times_len() {
        let tree = uniform_tree(4, 3, 5);
        assert_eq!(tree.size_bytes(), tree.len() * 8);
    }

    #[test]
    fn view_is_empty_matches_owned() {
        let tree = uniform_tree(2, 3, 4);
        let view = tree.view();
        assert_eq!(view.is_empty(), tree.is_empty());
    }

    #[test]
    fn data_returns_full_backing_array() {
        // openings_per_stage=[1,2], dim=2: total = (1+2)*2 = 6 elements
        let raw = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tree = OpeningTree::from_parts(raw.clone(), vec![1, 2], 2);
        assert_eq!(tree.data(), raw.as_slice());
    }

    #[test]
    fn openings_per_stage_slice_matches_input() {
        // openings_per_stage=[1,2], dim=2: total = (1+2)*2 = 6 elements
        let tree = OpeningTree::from_parts(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 2], 2);
        assert_eq!(tree.openings_per_stage_slice(), &[1_usize, 2]);
    }

    #[test]
    fn view_data_matches_owned_data() {
        // openings_per_stage=[1,2], dim=2: total = (1+2)*2 = 6 elements
        let raw = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tree = OpeningTree::from_parts(raw.clone(), vec![1, 2], 2);
        let view = tree.view();
        assert_eq!(view.data(), raw.as_slice());
    }
}
