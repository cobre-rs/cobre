//! Layer 5b — correlation-domain semantic validation.
//!
//! Correlation matrix symmetry, diagonal entries equal to 1.0,
//! off-diagonal values within [-1.0, 1.0], and inflow-vs-load
//! same-type matrix-shape compatibility.

use super::super::{ErrorKind, ValidationContext, schema::ParsedData};
use super::CORR_TOLERANCE;

// ── Rules 14-16: Correlation matrix validation ────────────────────────────────

/// Validates correlation matrix symmetry, diagonal, and off-diagonal range for
/// all groups in all profiles of the correlation model.
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

                    // Rule 15: Diagonal entries must be 1.0 (±CORR_TOLERANCE).
                    if i == j && (val - 1.0).abs() > CORR_TOLERANCE {
                        ctx.add_error(
                            ErrorKind::BusinessRuleViolation,
                            "scenarios/correlation.json",
                            Some(format!("CorrelationGroup {group_name}")),
                            format!(
                                "CorrelationGroup '{group_name}': diagonal entry matrix[{i}][{i}] \
                                 is {val}, expected 1.0 (±{CORR_TOLERANCE}); \
                                 correlation matrix diagonal must be 1.0"
                            ),
                        );
                    }

                    // Rule 16: Off-diagonal entries must be in [-1.0, 1.0].
                    if i != j && !((-1.0_f64)..=1.0).contains(&val) {
                        ctx.add_error(
                            ErrorKind::BusinessRuleViolation,
                            "scenarios/correlation.json",
                            Some(format!("CorrelationGroup {group_name}")),
                            format!(
                                "CorrelationGroup '{group_name}': off-diagonal entry \
                                 matrix[{i}][{j}] is {val}, outside valid range [-1.0, 1.0]; \
                                 correlation coefficients must be in [-1.0, 1.0]"
                            ),
                        );
                    }

                    // Rule 14: Symmetry check (only check upper triangle to avoid duplicates).
                    if i < j {
                        let symmetric = group.matrix[j][i];
                        if (val - symmetric).abs() > CORR_TOLERANCE {
                            ctx.add_error(
                                ErrorKind::BusinessRuleViolation,
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

/// Validates that all entities within each correlation group share the same
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
                    ctx.add_error(
                        ErrorKind::BusinessRuleViolation,
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
