//! Layer 5b — correlation-domain semantic validation.

use super::super::{ValidationContext, rules, schema::ParsedData};
use super::CORR_TOLERANCE;

// ── Rules 14-16: Correlation matrix validation ────────────────────────────────

/// Rules 14-16: validates correlation matrix symmetry, diagonal, and
/// off-diagonal range for all groups in all profiles of the correlation model.
///
/// Only runs when `data.correlation` is `Some`.
pub(super) fn check_correlation_matrices(data: &ParsedData, ctx: &mut ValidationContext) {
    let Some(correlation) = &data.correlation else {
        return;
    };

    for profile in correlation.profiles.values() {
        for group in &profile.groups {
            let n = group.entities.len();
            let group_name = &group.name;

            // Rules 14-16 require a square matrix; the matrix row count is guaranteed
            // to match entity count by Layer 4 (dimensional check 4). Be defensive.
            if group.matrix.len() != n {
                continue;
            }

            for i in 0..n {
                if group.matrix[i].len() != n {
                    continue;
                }
                for j in 0..n {
                    let val = group.matrix[i][j];

                    if i == j && (val - 1.0).abs() > CORR_TOLERANCE {
                        ctx.emit(
                            &rules::SEMANTIC_CORRELATION_DIAGONAL,
                            "scenarios/correlation.json",
                            Some(format!("CorrelationGroup {group_name}")),
                            format!(
                                "CorrelationGroup '{group_name}': diagonal entry matrix[{i}][{i}] \
                                 is {val}, expected 1.0 (±{CORR_TOLERANCE}); \
                                 correlation matrix diagonal must be 1.0"
                            ),
                        );
                    }

                    if i != j && !((-1.0_f64)..=1.0).contains(&val) {
                        ctx.emit(
                            &rules::SEMANTIC_CORRELATION_OFF_DIAGONAL_RANGE,
                            "scenarios/correlation.json",
                            Some(format!("CorrelationGroup {group_name}")),
                            format!(
                                "CorrelationGroup '{group_name}': off-diagonal entry \
                                 matrix[{i}][{j}] is {val}, outside valid range [-1.0, 1.0]; \
                                 correlation coefficients must be in [-1.0, 1.0]"
                            ),
                        );
                    }

                    // Upper triangle only — avoids reporting each asymmetry twice.
                    if i < j {
                        let symmetric = group.matrix[j][i];
                        if (val - symmetric).abs() > CORR_TOLERANCE {
                            ctx.emit(
                                &rules::SEMANTIC_CORRELATION_ASYMMETRIC,
                                "scenarios/correlation.json",
                                Some(format!("CorrelationGroup {group_name}")),
                                format!(
                                    "CorrelationGroup '{group_name}': correlation matrix is not \
                                     symmetric at ({i},{j}): matrix[{i}][{j}]={val} but \
                                     matrix[{j}][{i}]={symmetric}; tolerance is {CORR_TOLERANCE}"
                                ),
                            );
                        }
                    }
                }
            }
        }
    }
}

// ── M4: Same-type enforcement within correlation groups ──────────────────────

/// Rule 16a: validates that all entities within each correlation group share the same
/// `entity_type` value. Mixed groups produce incorrect covariance matrices.
pub(super) fn check_correlation_same_type(data: &ParsedData, ctx: &mut ValidationContext) {
    let Some(correlation) = &data.correlation else {
        return;
    };

    for profile in correlation.profiles.values() {
        for group in &profile.groups {
            if group.entities.is_empty() {
                continue;
            }
            let first_type = &group.entities[0].entity_type;
            for entity in &group.entities[1..] {
                if entity.entity_type != *first_type {
                    ctx.emit(
                        &rules::SEMANTIC_CORRELATION_MIXED_ENTITY_TYPES,
                        "scenarios/correlation.json",
                        Some(format!("CorrelationGroup '{}'", group.name)),
                        format!(
                            "CorrelationGroup '{}': entity {} has type '{}' but entity {} has \
                             type '{}'; all entities in a group must share the same entity_type",
                            group.name,
                            group.entities[0].id.0,
                            first_type,
                            entity.id.0,
                            entity.entity_type,
                        ),
                    );
                    break;
                }
            }
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

    use crate::validation::{ErrorKind, ValidationContext};

    // ── Rule 14: Correlation matrix symmetry ──────────────────────────────────

    /// Asymmetric matrix (matrix[0][1] != matrix[1][0]) produces a
    /// `BusinessRuleViolation` with "symmetric" in the message.
    #[test]
    fn test_5b_correlation_asymmetric() {
        let group = make_corr_group(
            "Asymmetric",
            vec![
                vec![1.0, 0.8],
                vec![0.5, 1.0], // asymmetric: should be 0.8
            ],
        );
        let corr = make_correlation(group);
        let data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            make_stages_5b(vec![0]),
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            Some(corr),
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        assert!(ctx.has_errors());
        let errors = ctx.errors();
        let relevant: Vec<_> = errors
            .iter()
            .filter(|e| e.kind == ErrorKind::BusinessRuleViolation)
            .collect();
        assert!(
            !relevant.is_empty(),
            "asymmetric matrix should produce BusinessRuleViolation"
        );
        let msg = &relevant[0].message;
        assert!(
            msg.contains("symmetric"),
            "message should contain 'symmetric', got: {msg}"
        );
    }

    // ── Rule 15: Correlation matrix diagonal ──────────────────────────────────

    /// Diagonal entry not equal to 1.0 produces a `BusinessRuleViolation`.
    #[test]
    fn test_5b_correlation_diagonal_not_one() {
        let group = make_corr_group(
            "BadDiag",
            vec![
                vec![0.9, 0.0], // diagonal entry 0.9 != 1.0
                vec![0.0, 1.0],
            ],
        );
        let corr = make_correlation(group);
        let data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            make_stages_5b(vec![0]),
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            Some(corr),
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        assert!(ctx.has_errors());
        assert!(
            ctx.errors()
                .iter()
                .any(|e| e.kind == ErrorKind::BusinessRuleViolation),
            "diagonal != 1.0 should produce BusinessRuleViolation"
        );
    }

    // ── Rule 16: Correlation coefficient range ────────────────────────────────

    /// Off-diagonal entry > 1.0 produces a `BusinessRuleViolation`.
    #[test]
    fn test_5b_correlation_off_diagonal_out_of_range() {
        let group = make_corr_group(
            "BadRange",
            vec![
                vec![1.0, 1.5], // 1.5 > 1.0 — out of range
                vec![1.5, 1.0],
            ],
        );
        let corr = make_correlation(group);
        let data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            make_stages_5b(vec![0]),
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            Some(corr),
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        assert!(ctx.has_errors());
        assert!(
            ctx.errors()
                .iter()
                .any(|e| e.kind == ErrorKind::BusinessRuleViolation),
            "off-diagonal > 1.0 should produce BusinessRuleViolation"
        );
    }

    // ── Rule 16a: Correlation group entity-type coherence ─────────────────────

    /// A group mixing `inflow` and `load` entities produces exactly one
    /// `BusinessRuleViolation` error and no warning.
    #[test]
    fn test_5b_correlation_mixed_entity_types_rejected() {
        let mut group = make_corr_group("Mixed", vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
        group.entities[1].entity_type = "load".to_string();
        let corr = make_correlation(group);
        let data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            make_stages_5b(vec![0]),
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            Some(corr),
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        let errors = ctx.errors();
        let relevant: Vec<_> = errors
            .iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation
                    && e.message.contains("must share the same entity_type")
            })
            .collect();
        assert_eq!(
            relevant.len(),
            1,
            "expected exactly one same-entity_type BusinessRuleViolation, got: {errors:?}"
        );
        assert!(
            !ctx.warnings()
                .iter()
                .any(|w| w.message.contains("entity_type")),
            "mixed entity types must not warn, got: {:?}",
            ctx.warnings()
        );
    }

    /// Valid symmetric correlation matrix produces no errors.
    #[test]
    fn test_5b_correlation_valid_symmetric() {
        let group = make_corr_group("Valid", vec![vec![1.0, 0.6], vec![0.6, 1.0]]);
        let corr = make_correlation(group);
        let data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            make_stages_5b(vec![0]),
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            Some(corr),
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "valid symmetric matrix should produce no errors, got: {:?}",
            ctx.errors()
        );
    }

    // ── Edge case: no correlation data ────────────────────────────────────────

    /// `correlation = None` produces no false-positive errors.
    #[test]
    fn test_5b_no_correlation_no_inflow_no_false_positives() {
        let data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            make_stages_5b(vec![0]),
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            None, // no correlation
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "empty correlation and inflow should produce no errors, got: {:?}",
            ctx.errors()
        );
    }
}
