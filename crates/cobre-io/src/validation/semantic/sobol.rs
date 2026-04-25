//! Layer 5b — Sobol-noise-method semantic validation.
//!
//! Verifies that stages declaring the Sobol noise method have a
//! `branching_factor` that is a power of 2.

use super::super::{ErrorKind, ValidationContext, schema::ParsedData};

/// Warns when a stage uses `QmcSobol` with a non-power-of-2 `branching_factor`.
///
/// Sobol sequences achieve optimal low-discrepancy uniformity only when the
/// number of sample points is a power of 2. A non-power-of-2 value produces
/// valid noise but loses the stratification guarantee of the Gray-code
/// recurrence. This emits a `ModelQuality` warning (not an error) because the
/// configuration is valid but suboptimal.
pub(super) fn check_sobol_power_of_2(data: &ParsedData, ctx: &mut ValidationContext) {
    use cobre_core::temporal::NoiseMethod;

    for stage in &data.stages.stages {
        if stage.id < 0 {
            continue; // skip pre-study stages
        }
        let bf = stage.scenario_config.branching_factor;
        if stage.scenario_config.noise_method == NoiseMethod::QmcSobol && !bf.is_power_of_two() {
            // bf == 0 is unreachable after parsing validation, but guard
            // defensively to prevent overflow in leading_zeros arithmetic.
            let suggestion = if bf > 0 {
                let lower = 1usize << (usize::BITS - bf.leading_zeros() - 1);
                let upper = lower << 1;
                format!("consider {lower} or {upper}")
            } else {
                "consider a positive power of 2".to_string()
            };
            ctx.add_warning(
                ErrorKind::ModelQuality,
                "stages.json",
                Some(format!("Stage {}", stage.id)),
                format!(
                    "Stage {}: qmc_sobol with num_scenarios={bf} which is not a \
                     power of 2; Sobol sequences have optimal uniformity at powers \
                     of 2 ({suggestion})",
                    stage.id,
                ),
            );
        }
    }
}
