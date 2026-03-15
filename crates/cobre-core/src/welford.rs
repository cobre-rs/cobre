//! Online accumulator for running mean and variance using Welford's algorithm.
//!
//! Accumulates one value at a time with O(1) updates and O(1) statistics
//! queries, with no re-scanning of previous data. Useful for streaming statistics
//! in iterative algorithms where values arrive one at a time and the full dataset
//! is not stored.

/// Online accumulator for running mean and variance using Welford's algorithm.
#[derive(Debug)]
pub struct WelfordAccumulator {
    count: u64,
    mean: f64,
    /// Sum of squared deviations from the running mean.
    m2: f64,
}

impl WelfordAccumulator {
    /// Create a new accumulator with no observations.
    #[must_use]
    pub fn new() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
        }
    }

    /// Incorporate a new observation into the running statistics.
    pub fn update(&mut self, value: f64) {
        self.count += 1;
        let delta = value - self.mean;
        #[allow(clippy::cast_precision_loss)] // count is bounded by scenario counts (u32-range)
        let count_f64 = self.count as f64;
        self.mean += delta / count_f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
    }

    /// Number of observations accumulated so far.
    #[must_use]
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Running mean of all observed values, or `0.0` if no observations.
    #[must_use]
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Population variance (`m2 / n`), or `0.0` if fewer than 2 observations.
    #[must_use]
    pub fn variance(&self) -> f64 {
        if self.count < 2 {
            0.0
        } else {
            #[allow(clippy::cast_precision_loss)]
            let count_f64 = self.count as f64;
            self.m2 / count_f64
        }
    }

    /// Population standard deviation, or `0.0` if fewer than 2 observations.
    #[must_use]
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Half-width of the 95% confidence interval (`1.96 * std / sqrt(n)`).
    ///
    /// Returns `0.0` when fewer than 2 observations are available.
    #[must_use]
    pub fn ci_95_half_width(&self) -> f64 {
        if self.count < 2 {
            0.0
        } else {
            #[allow(clippy::cast_precision_loss)]
            let count_f64 = self.count as f64;
            1.96 * self.std_dev() / count_f64.sqrt()
        }
    }
}

impl Default for WelfordAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::WelfordAccumulator;

    /// Known dataset: `[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]`
    /// Expected: mean=5.0, variance=4.0, `std_dev`=2.0.
    #[test]
    fn welford_known_dataset_mean_variance_std() {
        let values = [2.0_f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let mut acc = WelfordAccumulator::new();
        for &v in &values {
            acc.update(v);
        }
        assert!(
            (acc.mean() - 5.0).abs() < 1e-10,
            "mean: expected 5.0, got {}",
            acc.mean()
        );
        assert!(
            (acc.variance() - 4.0).abs() < 1e-10,
            "variance: expected 4.0, got {}",
            acc.variance()
        );
        assert!(
            (acc.std_dev() - 2.0).abs() < 1e-10,
            "std_dev: expected 2.0, got {}",
            acc.std_dev()
        );
    }

    /// Single value: mean equals that value, `std_dev`=0.0, CI half-width=0.0.
    #[test]
    fn welford_single_value_no_variance() {
        let mut acc = WelfordAccumulator::new();
        acc.update(42.0);
        assert!(
            (acc.mean() - 42.0).abs() < 1e-10,
            "mean: expected 42.0, got {}",
            acc.mean()
        );
        assert_eq!(
            acc.std_dev(),
            0.0,
            "std_dev must be 0.0 with one observation"
        );
        assert_eq!(
            acc.ci_95_half_width(),
            0.0,
            "ci_95_half_width must be 0.0 with one observation"
        );
    }

    /// Zero updates: mean=0.0, `std_dev`=0.0.
    #[test]
    fn welford_zero_updates() {
        let acc = WelfordAccumulator::new();
        assert_eq!(acc.mean(), 0.0, "mean must be 0.0 with no observations");
        assert_eq!(
            acc.std_dev(),
            0.0,
            "std_dev must be 0.0 with no observations"
        );
    }

    /// `count()` returns 0 before any updates and tracks each update correctly.
    #[test]
    fn welford_count_tracks_updates() {
        let mut acc = WelfordAccumulator::new();
        assert_eq!(acc.count(), 0, "count must be 0 before any updates");
        acc.update(1.0);
        acc.update(2.0);
        acc.update(3.0);
        assert_eq!(acc.count(), 3, "count must be 3 after 3 updates");
    }
}
