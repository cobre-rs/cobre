//! Resolution of parsed NCS bounds rows into a dense lookup table.
//!
//! [`resolve_ncs_bounds`] converts `Vec<NcsBoundsRow>` (from
//! `constraints/ncs_bounds.parquet`) into a [`ResolvedNcsBounds`] indexed by
//! `(ncs_index, stage_index)` for O(1) lookup during LP construction.
//!
//! Resolution is infallible: unknown NCS or stage IDs in the override rows
//! are silently skipped (they would have been caught by upstream validation).
//! The default available generation is the entity's installed capacity
//! (`NonControllableSource::max_generation_mw`).

use std::collections::HashMap;

use cobre_core::entities::NonControllableSource;
use cobre_core::resolved::ResolvedNcsBounds;

use crate::constraints::NcsBoundsRow;

/// Build a resolved NCS bounds table from parsed override rows.
///
/// Maps domain-level `ncs_id` and `stage_id` values to 0-based positional
/// indices using the provided sorted entity slices. Rows referencing
/// unknown NCS IDs or stages are silently skipped.
///
/// When `overrides` is empty, the table is still populated with the installed
/// capacity defaults from each NCS entity.
///
/// # Arguments
///
/// * `overrides` — parsed NCS bounds rows from `constraints/ncs_bounds.parquet`
/// * `non_controllable_sources` — sorted NCS collection (for ID to index mapping)
/// * `n_stages` — total number of study stages
/// * `stage_index` — mapping from domain-level `stage_id` to positional 0-based index
#[must_use]
#[allow(clippy::implicit_hasher)]
pub fn resolve_ncs_bounds(
    overrides: &[NcsBoundsRow],
    non_controllable_sources: &[NonControllableSource],
    n_stages: usize,
    stage_index: &HashMap<i32, usize>,
) -> ResolvedNcsBounds {
    if non_controllable_sources.is_empty() || n_stages == 0 {
        return ResolvedNcsBounds::empty();
    }

    // Build ncs_id -> ncs_idx mapping.
    let ncs_id_to_idx: HashMap<i32, usize> = non_controllable_sources
        .iter()
        .enumerate()
        .map(|(idx, ncs)| (ncs.id.0, idx))
        .collect();

    // Collect installed capacities as defaults.
    let default_mw: Vec<f64> = non_controllable_sources
        .iter()
        .map(|ncs| ncs.max_generation_mw)
        .collect();

    let mut table = ResolvedNcsBounds::new(non_controllable_sources.len(), n_stages, &default_mw);

    for row in overrides {
        let Some(&ncs_idx) = ncs_id_to_idx.get(&row.ncs_id.0) else {
            continue; // Unknown NCS — skip.
        };
        let Some(&stage_idx) = stage_index.get(&row.stage_id) else {
            continue; // Unknown stage — skip.
        };
        table.set(ncs_idx, stage_idx, row.available_generation_mw);
    }

    table
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::float_cmp)]
mod tests {
    use super::*;
    use cobre_core::EntityId;

    fn make_ncs(id: i32, max_mw: f64) -> NonControllableSource {
        NonControllableSource {
            id: EntityId(id),
            name: format!("NCS{id}"),
            bus_id: EntityId(0),
            entry_stage_id: None,
            exit_stage_id: None,
            max_generation_mw: max_mw,
            curtailment_cost: 5.0,
        }
    }

    fn make_stage_index(ids: &[i32]) -> HashMap<i32, usize> {
        ids.iter().enumerate().map(|(idx, &id)| (id, idx)).collect()
    }

    #[test]
    fn test_empty_overrides_uses_defaults() {
        let ncs = vec![make_ncs(0, 100.0), make_ncs(1, 200.0)];
        let si = make_stage_index(&[0, 1]);
        let table = resolve_ncs_bounds(&[], &ncs, 2, &si);
        assert!((table.available_generation(0, 0) - 100.0).abs() < f64::EPSILON);
        assert!((table.available_generation(0, 1) - 100.0).abs() < f64::EPSILON);
        assert!((table.available_generation(1, 0) - 200.0).abs() < f64::EPSILON);
        assert!((table.available_generation(1, 1) - 200.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_overrides_applied() {
        let ncs = vec![make_ncs(0, 100.0)];
        let si = make_stage_index(&[0, 1]);
        let overrides = vec![NcsBoundsRow {
            ncs_id: EntityId(0),
            stage_id: 1,
            available_generation_mw: 50.0,
        }];
        let table = resolve_ncs_bounds(&overrides, &ncs, 2, &si);
        // Stage 0: default.
        assert!((table.available_generation(0, 0) - 100.0).abs() < f64::EPSILON);
        // Stage 1: overridden.
        assert!((table.available_generation(0, 1) - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_unknown_ncs_id_skipped() {
        let ncs = vec![make_ncs(0, 100.0)];
        let si = make_stage_index(&[0]);
        let overrides = vec![NcsBoundsRow {
            ncs_id: EntityId(99),
            stage_id: 0,
            available_generation_mw: 999.0,
        }];
        let table = resolve_ncs_bounds(&overrides, &ncs, 1, &si);
        assert!((table.available_generation(0, 0) - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_unknown_stage_id_skipped() {
        let ncs = vec![make_ncs(0, 100.0)];
        let si = make_stage_index(&[0]);
        let overrides = vec![NcsBoundsRow {
            ncs_id: EntityId(0),
            stage_id: 99,
            available_generation_mw: 999.0,
        }];
        let table = resolve_ncs_bounds(&overrides, &ncs, 1, &si);
        assert!((table.available_generation(0, 0) - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_empty_ncs_returns_empty() {
        let si = make_stage_index(&[0]);
        let table = resolve_ncs_bounds(&[], &[], 1, &si);
        assert!(table.is_empty());
    }

    #[test]
    fn test_zero_stages_returns_empty() {
        let ncs = vec![make_ncs(0, 100.0)];
        let si = make_stage_index(&[]);
        let table = resolve_ncs_bounds(&[], &ncs, 0, &si);
        assert!(table.is_empty());
    }
}
