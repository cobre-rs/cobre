//! Present-value discounting of stage costs.

/// The cumulative discount of every delivery stage (study stages, then
/// post-study stages), `D(0) == 1.0`.
#[derive(Debug, Clone)]
pub(crate) struct TimeValue {
    delivery_cumulative_discount_factors: Vec<f64>,
}

impl TimeValue {
    #[allow(clippy::float_cmp)]
    pub(crate) fn new(delivery_cumulative_discount_factors: Vec<f64>) -> Self {
        debug_assert!(
            delivery_cumulative_discount_factors
                .first()
                .is_none_or(|&d| d == 1.0),
            "TimeValue::new: the first delivery stage's cumulative discount must be 1.0"
        );
        Self {
            delivery_cumulative_discount_factors,
        }
    }

    /// The discount of a cost delivered at `delivery`, in the units of stage
    /// `decision`: `D(delivery) / D(decision)`.
    pub(crate) fn relative_delivery_discount(&self, decision: usize, delivery: usize) -> f64 {
        self.delivery_cumulative_discount_factors[delivery]
            / self.delivery_cumulative_discount_factors[decision]
    }
}

#[cfg(test)]
mod tests {
    use super::TimeValue;

    fn fixture() -> Vec<f64> {
        vec![1.0, 0.9, 0.81, 0.729]
    }

    #[test]
    fn relative_delivery_discount_from_stage_zero_is_bit_exact_to_the_absolute_factor() {
        let d = fixture();
        let tv = TimeValue::new(d.clone());
        for (m, &expected) in d.iter().enumerate() {
            assert_eq!(tv.relative_delivery_discount(0, m), expected);
        }
    }

    #[test]
    fn relative_delivery_discount_divides_the_two_absolute_factors() {
        let d = fixture();
        let tv = TimeValue::new(d.clone());
        assert_eq!(tv.relative_delivery_discount(1, 3), d[3] / d[1]);
    }
}
