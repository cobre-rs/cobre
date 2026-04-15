//! Resolution of parsed NCS factor entries into a dense lookup table.
//!
//! [`resolve_ncs_factors`] converts `Vec<NcsFactorEntry>` (from
//! `scenarios/non_controllable_factors.json`) into a [`ResolvedNcsFactors`]
//! indexed by `(ncs_index, stage_index, block_index)` for O(1) lookup during
//! LP construction.
//!
//! Resolution is infallible: unknown NCS or stage IDs in the factor entries
//! are silently skipped (they would have been caught by upstream validation).
//! The default factor is `1.0` (no scaling).

use std::collections::HashMap;

use cobre_core::entities::NonControllableSource;
use cobre_core::resolved::ResolvedNcsFactors;
use cobre_core::temporal::Stage;

use crate::scenarios::NcsFactorEntry;

/// Build a resolved NCS factor table from parsed entries.
///
/// Maps domain-level `ncs_id` and `stage_id` values to 0-based positional
/// indices using the provided sorted entity slices. Entries referencing
/// unknown NCS IDs or stages are silently skipped.
///
/// # Arguments
///
/// * `entries` — parsed NCS factor entries from `scenarios/non_controllable_factors.json`
/// * `non_controllable_sources` — sorted NCS collection (for `ncs_id` to index mapping)
/// * `stages` — sorted stage collection (for `stage_id` to index mapping and max block count)
#[must_use]
pub fn resolve_ncs_factors(
    entries: &[NcsFactorEntry],
    non_controllable_sources: &[NonControllableSource],
    stages: &[Stage],
) -> ResolvedNcsFactors {
    if entries.is_empty() || non_controllable_sources.is_empty() || stages.is_empty() {
        return ResolvedNcsFactors::empty();
    }

    // Build ncs_id -> ncs_idx mapping.
    let ncs_id_to_idx: HashMap<i32, usize> = non_controllable_sources
        .iter()
        .enumerate()
        .map(|(idx, ncs)| (ncs.id.0, idx))
        .collect();

    // Build stage_id -> stage_idx mapping (study stages only: id >= 0).
    let stage_id_to_idx: HashMap<i32, usize> = stages
        .iter()
        .filter(|s| s.id >= 0)
        .enumerate()
        .map(|(idx, s)| (s.id, idx))
        .collect();

    let n_ncs = non_controllable_sources.len();
    let n_stages = stage_id_to_idx.len();
    let max_blocks = stages
        .iter()
        .filter(|s| s.id >= 0)
        .map(|s| s.blocks.len())
        .max()
        .unwrap_or(0);

    let mut table = ResolvedNcsFactors::new(n_ncs, n_stages, max_blocks);

    for entry in entries {
        let Some(&ncs_idx) = ncs_id_to_idx.get(&entry.ncs_id.0) else {
            continue; // Unknown NCS — skip.
        };
        let Some(&stage_idx) = stage_id_to_idx.get(&entry.stage_id) else {
            continue; // Unknown stage — skip.
        };
        for bf in &entry.block_factors {
            let Some(block_idx) = usize::try_from(bf.block_id).ok() else {
                continue;
            };
            if block_idx < max_blocks {
                table.set(ncs_idx, stage_idx, block_idx, bf.factor);
            }
        }
    }

    table
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::float_cmp)]
mod tests {
    use super::*;
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, StageRiskConfig, StageStateConfig,
    };
    use cobre_core::EntityId;

    use crate::scenarios::BlockFactor;

    fn make_ncs(id: i32) -> NonControllableSource {
        NonControllableSource {
            id: EntityId(id),
            name: format!("NCS{id}"),
            bus_id: EntityId(0),
            entry_stage_id: None,
            exit_stage_id: None,
            max_generation_mw: 100.0,
            curtailment_cost: 5.0,
        }
    }

    fn make_stage(id: i32, n_blocks: usize) -> Stage {
        Stage {
            index: 0,
            id,
            start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: chrono::NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: None,
            blocks: (0..n_blocks)
                .map(|b| Block {
                    index: b,
                    name: format!("B{b}"),
                    duration_hours: 100.0,
                })
                .collect(),
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: true,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        }
    }

    fn make_entry(ncs_id: i32, stage_id: i32, factors: &[(i32, f64)]) -> NcsFactorEntry {
        NcsFactorEntry {
            ncs_id: EntityId(ncs_id),
            stage_id,
            block_factors: factors
                .iter()
                .map(|&(block_id, factor)| BlockFactor { block_id, factor })
                .collect(),
        }
    }

    #[test]
    fn test_empty_entries_returns_empty() {
        let ncs = vec![make_ncs(0)];
        let stages = vec![make_stage(0, 2)];
        let table = resolve_ncs_factors(&[], &ncs, &stages);
        assert!((table.factor(0, 0, 0) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_basic_resolution() {
        let ncs = vec![make_ncs(0), make_ncs(1)];
        let stages = vec![make_stage(0, 3)];
        let entries = vec![make_entry(0, 0, &[(0, 0.6), (1, 0.8)])];

        let table = resolve_ncs_factors(&entries, &ncs, &stages);
        assert!((table.factor(0, 0, 0) - 0.6).abs() < 1e-10);
        assert!((table.factor(0, 0, 1) - 0.8).abs() < 1e-10);
        // Block 2 has no entry — default 1.0.
        assert!((table.factor(0, 0, 2) - 1.0).abs() < f64::EPSILON);
        // NCS 1 has no entry — default 1.0.
        assert!((table.factor(1, 0, 0) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_unknown_ncs_id_skipped() {
        let ncs = vec![make_ncs(0)];
        let stages = vec![make_stage(0, 1)];
        let entries = vec![make_entry(99, 0, &[(0, 0.5)])];

        let table = resolve_ncs_factors(&entries, &ncs, &stages);
        assert!((table.factor(0, 0, 0) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_unknown_stage_id_skipped() {
        let ncs = vec![make_ncs(0)];
        let stages = vec![make_stage(0, 1)];
        let entries = vec![make_entry(0, 99, &[(0, 0.5)])];

        let table = resolve_ncs_factors(&entries, &ncs, &stages);
        assert!((table.factor(0, 0, 0) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_pre_study_stages_excluded() {
        let ncs = vec![make_ncs(0)];
        let stages = vec![make_stage(-1, 1), make_stage(0, 2)];
        let entries = vec![make_entry(0, 0, &[(0, 0.7)])];

        let table = resolve_ncs_factors(&entries, &ncs, &stages);
        assert!((table.factor(0, 0, 0) - 0.7).abs() < 1e-10);
    }
}
