//! Resolution of raw generic constraint bound rows into an indexed lookup table.
//!
//! [`resolve_generic_constraint_bounds`] converts the parsed
//! `Vec<GenericConstraintBoundsRow>` and the sorted `Vec<GenericConstraint>`
//! (which provides the `constraint_id → positional_index` mapping) into a
//! [`ResolvedGenericConstraintBounds`] that supports O(1) lookup by
//! `(constraint_idx, stage_id)`.
//!
//! This function is infallible: the validation pipeline has already confirmed
//! that every `constraint_id` in the bounds rows refers to a known constraint.
//! Any residual unknown IDs (from rows that somehow slipped through or from
//! future Epic 06 relaxation) are silently skipped.

use std::collections::HashMap;

use cobre_core::{GenericConstraint, ResolvedGenericConstraintBounds};

use crate::constraints::GenericConstraintBoundsRow;

/// Build the resolved generic constraint bound table from sorted parsed inputs.
///
/// Converts the positionally-sorted `constraints` slice (which must already be
/// sorted by ID, as produced by `parse_generic_constraints`) and the flat
/// `raw_bounds` rows (sorted by `(constraint_id, stage_id, block_id)`, as
/// produced by `parse_generic_constraint_bounds`) into a
/// [`ResolvedGenericConstraintBounds`] indexed lookup table.
///
/// Rows whose `constraint_id` is not found in the constraint registry are
/// silently skipped — this is consistent with how other bound resolvers handle
/// unknown entity IDs.
///
/// # Arguments
///
/// * `constraints` — generic constraint definitions sorted by ID (canonical order)
/// * `raw_bounds` — sorted bound rows from
///   `constraints/generic_constraint_bounds.parquet`
///
/// # Examples
///
/// ```
/// use cobre_core::GenericConstraint;
/// use cobre_core::generic_constraint::{ConstraintExpression, ConstraintSense, SlackConfig};
/// use cobre_core::EntityId;
/// use cobre_io::constraints::GenericConstraintBoundsRow;
/// use cobre_io::resolution::resolve_generic_constraint_bounds;
///
/// let constraint = GenericConstraint {
///     id: EntityId(0),
///     name: "c0".to_string(),
///     description: None,
///     sense: ConstraintSense::LessEqual,
///     expression: ConstraintExpression { terms: vec![] },
///     slack: SlackConfig { enabled: false, penalty: None },
/// };
///
/// let row = GenericConstraintBoundsRow {
///     constraint_id: 0,
///     stage_id: 1,
///     block_id: None,
///     bound: 500.0,
/// };
///
/// let table = resolve_generic_constraint_bounds(&[constraint], &[row]);
///
/// assert!(table.is_active(0, 1));
/// assert!(!table.is_active(0, 0));
///
/// let slice = table.bounds_for_stage(0, 1);
/// assert_eq!(slice.len(), 1);
/// assert_eq!(slice[0], (None, 500.0));
/// ```
#[must_use]
pub fn resolve_generic_constraint_bounds(
    constraints: &[GenericConstraint],
    raw_bounds: &[GenericConstraintBoundsRow],
) -> ResolvedGenericConstraintBounds {
    // Build a mapping from domain-level constraint_id (i32) to positional index.
    // Constraints are already sorted by ID (canonical order from SystemBuilder).
    let id_to_idx: HashMap<i32, usize> = constraints
        .iter()
        .enumerate()
        .map(|(idx, c)| (c.id.0, idx))
        .collect();

    // Feed the sorted rows as a (constraint_id, stage_id, block_id, bound) iterator.
    ResolvedGenericConstraintBounds::new(
        &id_to_idx,
        raw_bounds
            .iter()
            .map(|r| (r.constraint_id, r.stage_id, r.block_id, r.bound)),
    )
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::too_many_lines,
    clippy::float_cmp
)]
mod tests {
    use super::*;
    use cobre_core::generic_constraint::{ConstraintExpression, ConstraintSense};
    use cobre_core::EntityId;

    fn make_constraint(id: i32) -> GenericConstraint {
        use cobre_core::generic_constraint::SlackConfig;
        GenericConstraint {
            id: EntityId(id),
            name: format!("c{id}"),
            description: None,
            sense: ConstraintSense::LessEqual,
            expression: ConstraintExpression { terms: vec![] },
            slack: SlackConfig {
                enabled: false,
                penalty: None,
            },
        }
    }

    fn make_row(
        constraint_id: i32,
        stage_id: i32,
        block_id: Option<i32>,
        bound: f64,
    ) -> GenericConstraintBoundsRow {
        GenericConstraintBoundsRow {
            constraint_id,
            stage_id,
            block_id,
            bound,
        }
    }

    /// Empty constraints and empty bounds produce an empty table.
    #[test]
    fn test_empty_constraints_empty_bounds() {
        let table = resolve_generic_constraint_bounds(&[], &[]);
        assert!(!table.is_active(0, 0));
        assert!(table.bounds_for_stage(0, 0).is_empty());
    }

    /// Two constraints with no bound rows: `is_active` always false.
    #[test]
    fn test_constraints_no_bounds() {
        let constraints = vec![make_constraint(0), make_constraint(1)];
        let table = resolve_generic_constraint_bounds(&constraints, &[]);
        assert!(!table.is_active(0, 0));
        assert!(!table.is_active(1, 0));
    }

    /// Two constraints, bounds for constraint 0 at stages 0 and 1.
    /// Constraint 1 has no bounds.
    #[test]
    fn test_two_constraints_sparse_bounds() {
        let constraints = vec![make_constraint(0), make_constraint(1)];
        let rows = vec![make_row(0, 0, None, 100.0), make_row(0, 1, None, 200.0)];
        let table = resolve_generic_constraint_bounds(&constraints, &rows);

        assert!(table.is_active(0, 0));
        assert!(table.is_active(0, 1));
        assert!(!table.is_active(1, 0));
        assert!(!table.is_active(1, 1));

        let s0 = table.bounds_for_stage(0, 0);
        assert_eq!(s0.len(), 1);
        assert!((s0[0].1 - 100.0).abs() < f64::EPSILON);
        assert!(s0[0].0.is_none());

        let s1 = table.bounds_for_stage(0, 1);
        assert_eq!(s1.len(), 1);
        assert!((s1[0].1 - 200.0).abs() < f64::EPSILON);
    }

    /// Block-specific bounds: multiple (`block_id`, bound) pairs for one (constraint, stage).
    #[test]
    fn test_block_specific_bounds() {
        let constraints = vec![make_constraint(0)];
        let rows = vec![
            make_row(0, 0, None, 50.0),
            make_row(0, 0, Some(0), 60.0),
            make_row(0, 0, Some(1), 70.0),
        ];
        let table = resolve_generic_constraint_bounds(&constraints, &rows);

        assert!(table.is_active(0, 0));
        let slice = table.bounds_for_stage(0, 0);
        assert_eq!(slice.len(), 3);
        assert_eq!(slice[0], (None, 50.0));
        assert_eq!(slice[1], (Some(0), 60.0));
        assert_eq!(slice[2], (Some(1), 70.0));
    }

    /// Rows referencing unknown constraint IDs are silently skipped.
    #[test]
    fn test_unknown_constraint_id_skipped() {
        let constraints = vec![make_constraint(0)];
        let rows = vec![
            make_row(0, 0, None, 100.0),
            make_row(99, 0, None, 9999.0), // Unknown ID.
        ];
        let table = resolve_generic_constraint_bounds(&constraints, &rows);

        assert!(table.is_active(0, 0));
        // No entry for the unknown constraint (idx 0 is constraint_id=0, not 99).
        let slice = table.bounds_for_stage(0, 0);
        assert_eq!(slice.len(), 1);
        assert!((slice[0].1 - 100.0).abs() < f64::EPSILON);
    }

    /// Acceptance criterion: constraint 0 at stage 0 `is_active` returns true.
    #[test]
    fn test_ac_is_active_true() {
        let constraints = vec![make_constraint(0), make_constraint(1)];
        let rows = vec![make_row(0, 0, None, 100.0), make_row(0, 1, None, 150.0)];
        let table = resolve_generic_constraint_bounds(&constraints, &rows);
        assert!(table.is_active(0, 0));
    }

    /// Acceptance criterion: constraint 1 at stage 0 `is_active` returns false when no bounds.
    #[test]
    fn test_ac_is_active_false() {
        let constraints = vec![make_constraint(0), make_constraint(1)];
        let rows = vec![make_row(0, 0, None, 100.0)];
        let table = resolve_generic_constraint_bounds(&constraints, &rows);
        assert!(!table.is_active(1, 0));
    }

    /// Acceptance criterion: `bounds_for_stage` returns (None, 100.0) for single bound row.
    #[test]
    fn test_ac_bounds_for_stage() {
        let constraints = vec![make_constraint(0)];
        let rows = vec![make_row(0, 0, None, 100.0)];
        let table = resolve_generic_constraint_bounds(&constraints, &rows);
        let slice = table.bounds_for_stage(0, 0);
        assert_eq!(slice.len(), 1);
        assert_eq!(slice[0], (None, 100.0));
    }
}
