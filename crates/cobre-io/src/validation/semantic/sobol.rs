//! Layer 5b — Sobol-noise-method semantic validation.
//!
//! Verifies that stages declaring the Sobol noise method have a
//! `branching_factor` that is a power of 2.

use super::super::{ValidationContext, rules, schema::ParsedData};

/// Rule 25: warns when a stage uses `QmcSobol` with a non-power-of-2 `branching_factor`.
///
/// Sobol sequences achieve optimal low-discrepancy uniformity only when the
/// number of sample points is a power of 2. A non-power-of-2 value produces
/// valid noise but loses the stratification guarantee of the Gray-code
/// recurrence. It warns rather than rejects, because the configuration is valid
/// but suboptimal.
pub(super) fn check_sobol_power_of_2(data: &ParsedData, ctx: &mut ValidationContext) {
    use cobre_core::temporal::NoiseMethod;

    for stage in &data.stages.stages {
        if stage.id < 0 {
            continue; // skip pre-study stages
        }
        let bf = stage.scenario_config.branching_factor;
        if stage.scenario_config.noise_method == NoiseMethod::QmcSobol && !bf.is_power_of_two() {
            // bf == 0 (unreachable post-parse) would overflow the leading_zeros
            // arithmetic below — guard it.
            let suggestion = if bf > 0 {
                let lower = 1usize << (usize::BITS - bf.leading_zeros() - 1);
                let upper = lower << 1;
                format!("consider {lower} or {upper}")
            } else {
                "consider a positive power of 2".to_string()
            };
            ctx.emit(
                &rules::SEMANTIC_SOBOL_OPENING_COUNT,
                "stages.json",
                Some(format!("Stage {}", stage.id)),
                format!(
                    "Stage {}: qmc_sobol with num_openings={bf} which is not a \
                     power of 2; Sobol sequences have optimal uniformity at powers \
                     of 2 ({suggestion})",
                    stage.id,
                ),
            );
        }
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::too_many_lines,
    clippy::doc_markdown,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]
mod tests {
    use super::super::validate_semantic_stages_penalties_scenarios;
    use crate::test_support::*;
    use cobre_core::temporal::{NoiseMethod, ScenarioSourceConfig};

    use crate::validation::{ErrorKind, ValidationContext};

    /// Stage with `QmcSobol` and `branching_factor: 50` (not a power of 2)
    /// emits exactly one `ModelQuality` warning mentioning the branching factor.
    #[test]
    fn test_sobol_non_power_of_2_emits_warning() {
        let mut stages = make_stages_5b(vec![0]);
        stages.stages[0].scenario_config = ScenarioSourceConfig {
            branching_factor: 50,
            noise_method: NoiseMethod::QmcSobol,
        };
        let data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            stages,
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);

        let all_warnings = ctx.warnings();
        let quality_warnings: Vec<_> = all_warnings
            .iter()
            .filter(|w| w.kind == ErrorKind::ModelQuality && w.message.contains("qmc_sobol"))
            .collect();
        assert_eq!(
            quality_warnings.len(),
            1,
            "expected exactly 1 ModelQuality warning, got: {:?}",
            ctx.warnings()
        );
        let msg = &quality_warnings[0].message;
        assert!(
            msg.contains("50"),
            "warning message should contain the branching factor '50', got: {msg}"
        );
        assert!(
            msg.contains("Stage "),
            "warning message should contain 'Stage ', got: {msg}"
        );
    }

    /// Stage with `QmcSobol` and `branching_factor: 64` (a power of 2)
    /// produces no warnings.
    #[test]
    fn test_sobol_power_of_2_no_warning() {
        let mut stages = make_stages_5b(vec![0]);
        stages.stages[0].scenario_config = ScenarioSourceConfig {
            branching_factor: 64,
            noise_method: NoiseMethod::QmcSobol,
        };
        let data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            stages,
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);

        let all_warnings = ctx.warnings();
        let quality_warnings: Vec<_> = all_warnings
            .iter()
            .filter(|w| w.kind == ErrorKind::ModelQuality && w.message.contains("qmc_sobol"))
            .collect();
        assert!(
            quality_warnings.is_empty(),
            "branching_factor=64 (power of 2) should produce no ModelQuality warnings, \
             got: {quality_warnings:?}"
        );
    }

    /// Stage with `Saa` and `branching_factor: 50` (not a power of 2)
    /// produces no warnings — the check only applies to `QmcSobol`.
    #[test]
    fn test_saa_non_power_of_2_no_warning() {
        let mut stages = make_stages_5b(vec![0]);
        stages.stages[0].scenario_config = ScenarioSourceConfig {
            branching_factor: 50,
            noise_method: NoiseMethod::Saa,
        };
        let data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            stages,
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);

        let all_warnings = ctx.warnings();
        let quality_warnings: Vec<_> = all_warnings
            .iter()
            .filter(|w| w.kind == ErrorKind::ModelQuality && w.message.contains("qmc_sobol"))
            .collect();
        assert!(
            quality_warnings.is_empty(),
            "SAA with non-power-of-2 branching factor should produce no ModelQuality warnings, \
             got: {quality_warnings:?}"
        );
    }

    /// Two stages: stage 0 uses `QmcSobol` with `branching_factor: 100` (not a
    /// power of 2), stage 1 uses `QmcSobol` with `branching_factor: 128` (power
    /// of 2). Exactly 1 `ModelQuality` warning should be emitted, for stage 0.
    #[test]
    fn test_sobol_mixed_stages_only_warns_non_power() {
        let mut stages = make_stages_5b(vec![0, 1]);
        stages.stages[0].scenario_config = ScenarioSourceConfig {
            branching_factor: 100,
            noise_method: NoiseMethod::QmcSobol,
        };
        stages.stages[1].scenario_config = ScenarioSourceConfig {
            branching_factor: 128,
            noise_method: NoiseMethod::QmcSobol,
        };
        let data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            stages,
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);

        let all_warnings = ctx.warnings();
        let quality_warnings: Vec<_> = all_warnings
            .iter()
            .filter(|w| w.kind == ErrorKind::ModelQuality && w.message.contains("qmc_sobol"))
            .collect();
        assert_eq!(
            quality_warnings.len(),
            1,
            "expected exactly 1 ModelQuality warning (for stage 0 only), got: {:?}",
            ctx.warnings()
        );
        let msg = &quality_warnings[0].message;
        assert!(
            msg.contains("Stage 0"),
            "warning should be for stage 0, got: {msg}"
        );
        assert!(
            msg.contains("100"),
            "warning should mention branching_factor 100, got: {msg}"
        );
    }
}
