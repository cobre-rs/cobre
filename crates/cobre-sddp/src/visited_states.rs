//! Visited forward-pass states for dominated cut selection.
//!
//! When [`CutSelectionStrategy::Dominated`](crate::CutSelectionStrategy::Dominated)
//! is active, the training loop archives the trial-point state vectors produced
//! by each forward pass so that the domination test can evaluate every cut at
//! every visited point.
//!
//! The archive is organised as one [`StageStates`] per stage.  Each
//! `StageStates` stores its state vectors in a single flat `Vec<f64>` for
//! cache-friendly iteration during the domination sweep.

/// Single-stage visited-states buffer.
///
/// Stores forward-pass trial points as a flat contiguous `Vec<f64>`.
/// Entry `i * state_dimension .. (i + 1) * state_dimension` holds state `i`.
pub struct StageStates {
    /// Flat buffer of accumulated state vectors.
    data: Vec<f64>,
    /// Number of states currently stored.
    count: usize,
    /// Length of each state vector.
    state_dimension: usize,
}

impl StageStates {
    /// Creates a new single-stage buffer, pre-allocating space for
    /// `capacity_states` state vectors of length `state_dimension`.
    #[must_use]
    pub fn new(state_dimension: usize, capacity_states: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity_states * state_dimension),
            count: 0,
            state_dimension,
        }
    }

    /// Returns the number of states currently stored.
    #[must_use]
    pub fn count(&self) -> usize {
        self.count
    }

    /// Returns the dimension of each state vector.
    #[must_use]
    pub fn state_dimension(&self) -> usize {
        self.state_dimension
    }

    /// Append `total_fwd` state vectors from `gathered` into this stage's
    /// buffer.
    ///
    /// `gathered` is a flat slice of length `total_fwd * state_dimension`,
    /// produced by `ExchangeBuffers::gathered_states()`.
    ///
    /// # Panics (debug only)
    ///
    /// Panics if `gathered.len() != total_fwd * self.state_dimension`.
    pub fn append(&mut self, gathered: &[f64], total_fwd: usize) {
        debug_assert_eq!(gathered.len(), total_fwd * self.state_dimension);
        self.data.extend_from_slice(gathered);
        self.count += total_fwd;
    }

    /// Return the flat slice of all accumulated states.
    ///
    /// Length is `self.count * self.state_dimension`.
    #[must_use]
    pub fn states(&self) -> &[f64] {
        &self.data[..self.count * self.state_dimension]
    }
}

/// Multi-stage archive of visited forward-pass states.
///
/// One [`StageStates`] per stage.  Only created when
/// [`CutSelectionStrategy::Dominated`](crate::CutSelectionStrategy::Dominated)
/// is active.
pub struct VisitedStatesArchive {
    stages: Vec<StageStates>,
}

impl VisitedStatesArchive {
    /// Creates a new archive with one [`StageStates`] per stage.
    ///
    /// Each stage buffer is pre-allocated for
    /// `max_iterations * total_forward_passes` state vectors.
    #[must_use]
    pub fn new(
        num_stages: usize,
        state_dimension: usize,
        max_iterations: u64,
        total_forward_passes: usize,
    ) -> Self {
        let capacity_per_stage = usize::try_from(max_iterations)
            .unwrap_or(usize::MAX)
            .saturating_mul(total_forward_passes);
        let stages = (0..num_stages)
            .map(|_| StageStates::new(state_dimension, capacity_per_stage))
            .collect();
        Self { stages }
    }

    /// Returns the number of stages in the archive.
    #[must_use]
    pub fn num_stages(&self) -> usize {
        self.stages.len()
    }

    /// Returns a shared reference to the [`StageStates`] for `stage`.
    ///
    /// # Panics
    ///
    /// Panics if `stage >= self.num_stages()`.
    #[must_use]
    pub fn stage(&self, stage: usize) -> &StageStates {
        &self.stages[stage]
    }

    /// Returns a mutable reference to the [`StageStates`] for `stage`.
    ///
    /// # Panics
    ///
    /// Panics if `stage >= self.num_stages()`.
    pub fn stage_mut(&mut self, stage: usize) -> &mut StageStates {
        &mut self.stages[stage]
    }

    /// Archive one iteration's gathered states for a specific stage.
    ///
    /// Called in the backward pass after `exchange.exchange()` produces
    /// the gathered buffer for stage `t`.
    pub fn archive_gathered_states(&mut self, stage: usize, gathered: &[f64], total_fwd: usize) {
        self.stages[stage].append(gathered, total_fwd);
    }

    /// Return the flat state slice for a stage.
    ///
    /// Used by `select_for_stage` during cut selection.
    #[must_use]
    pub fn states_for_stage(&self, stage: usize) -> &[f64] {
        self.stages[stage].states()
    }

    /// Number of states accumulated at a given stage.
    #[must_use]
    pub fn count(&self, stage: usize) -> usize {
        self.stages[stage].count()
    }
}

#[cfg(test)]
mod tests {
    use super::{StageStates, VisitedStatesArchive};

    /// Build a synthetic gathered buffer: `base, base+1, ..., base + total_fwd*state_dim - 1`.
    #[allow(clippy::cast_precision_loss)]
    fn make_gathered(state_dim: usize, total_fwd: usize, base: f64) -> Vec<f64> {
        (0..total_fwd * state_dim)
            .map(|i| base + i as f64)
            .collect()
    }

    #[test]
    fn stage_states_new_preallocates() {
        let s = StageStates::new(4, 100);
        assert!(s.states().is_empty());
        assert_eq!(s.count(), 0);
        assert_eq!(s.state_dimension(), 4);
        // Vec capacity is at least what we asked for.
        assert!(s.data.capacity() >= 400);
    }

    #[test]
    fn stage_states_append_single_batch() {
        let mut s = StageStates::new(2, 10);
        let gathered = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        s.append(&gathered, 3);
        assert_eq!(s.count(), 3);
        assert_eq!(s.states(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn stage_states_append_multiple_batches() {
        let mut s = StageStates::new(2, 10);
        // Batch 1: 3 states.
        let g1 = make_gathered(2, 3, 0.0);
        s.append(&g1, 3);
        // Batch 2: 2 states.
        let g2 = make_gathered(2, 2, 100.0);
        s.append(&g2, 2);
        assert_eq!(s.count(), 5);
        assert_eq!(
            s.states(),
            &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 100.0, 101.0, 102.0, 103.0]
        );
    }

    #[test]
    fn stage_states_empty_states() {
        let s = StageStates::new(3, 50);
        assert_eq!(s.states(), &[] as &[f64]);
        assert_eq!(s.count(), 0);
    }

    #[test]
    fn archive_new_creates_correct_stages() {
        let a = VisitedStatesArchive::new(5, 4, 10, 20);
        assert_eq!(a.num_stages(), 5);
        for t in 0..5 {
            assert_eq!(a.count(t), 0);
            assert!(a.states_for_stage(t).is_empty());
        }
    }

    #[test]
    fn archive_gathered_states_delegates() {
        let mut a = VisitedStatesArchive::new(4, 3, 10, 10);
        let gathered = make_gathered(3, 3, 1.0);
        a.archive_gathered_states(2, &gathered, 3);
        assert_eq!(a.count(2), 3);
        assert_eq!(a.count(0), 0);
        assert_eq!(a.count(1), 0);
        assert_eq!(a.count(3), 0);
    }

    #[test]
    fn archive_accumulates_across_iterations() {
        let mut a = VisitedStatesArchive::new(3, 2, 10, 5);
        let g1 = make_gathered(2, 5, 0.0);
        a.archive_gathered_states(1, &g1, 5);
        let g2 = make_gathered(2, 5, 100.0);
        a.archive_gathered_states(1, &g2, 5);
        assert_eq!(a.count(1), 10);
    }

    #[test]
    fn archive_states_for_stage_returns_flat_slice() {
        let mut a = VisitedStatesArchive::new(3, 2, 10, 10);
        let gathered = vec![10.0, 20.0, 30.0, 40.0];
        a.archive_gathered_states(1, &gathered, 2);
        assert_eq!(a.states_for_stage(1), &[10.0, 20.0, 30.0, 40.0]);
        assert!(a.states_for_stage(0).is_empty());
        assert!(a.states_for_stage(2).is_empty());
    }
}
