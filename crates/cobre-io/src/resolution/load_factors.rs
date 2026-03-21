//! Resolution of parsed load factor entries into a dense lookup table.
//!
//! [`resolve_load_factors`] converts `Vec<LoadFactorEntry>` (from
//! `scenarios/load_factors.json`) into a [`ResolvedLoadFactors`] indexed by
//! `(bus_index, stage_index, block_index)` for O(1) lookup during LP
//! construction.
//!
//! Resolution is infallible: unknown bus or stage IDs in the factor entries
//! are silently skipped (they would have been caught by upstream validation).
//! The default factor is `1.0` (no scaling).

use std::collections::HashMap;

use cobre_core::{Bus, ResolvedLoadFactors, Stage};

use crate::scenarios::LoadFactorEntry;

/// Build a resolved load factor table from parsed entries.
///
/// Maps domain-level `bus_id` and `stage_id` values to 0-based positional
/// indices using the provided sorted entity slices. Entries referencing
/// unknown buses or stages are silently skipped.
///
/// # Arguments
///
/// * `entries` — parsed load factor entries from `scenarios/load_factors.json`
/// * `buses` — sorted bus collection (for `bus_id` to index mapping)
/// * `stages` — sorted stage collection (for `stage_id` to index mapping and max block count)
#[must_use]
pub fn resolve_load_factors(
    entries: &[LoadFactorEntry],
    buses: &[Bus],
    stages: &[Stage],
) -> ResolvedLoadFactors {
    if entries.is_empty() || buses.is_empty() || stages.is_empty() {
        return ResolvedLoadFactors::empty();
    }

    // Build bus_id -> bus_index mapping.
    let bus_id_to_idx: HashMap<i32, usize> = buses
        .iter()
        .enumerate()
        .map(|(idx, b)| (b.id.0, idx))
        .collect();

    // Build stage_id -> stage_index mapping (study stages only: id >= 0).
    let stage_id_to_idx: HashMap<i32, usize> = stages
        .iter()
        .filter(|s| s.id >= 0)
        .enumerate()
        .map(|(idx, s)| (s.id, idx))
        .collect();

    let n_buses = buses.len();
    let n_stages = stage_id_to_idx.len();
    let max_blocks = stages
        .iter()
        .filter(|s| s.id >= 0)
        .map(|s| s.blocks.len())
        .max()
        .unwrap_or(0);

    let mut table = ResolvedLoadFactors::new(n_buses, n_stages, max_blocks);

    for entry in entries {
        let Some(&bus_idx) = bus_id_to_idx.get(&entry.bus_id.0) else {
            continue; // Unknown bus — skip.
        };
        let Some(&stage_idx) = stage_id_to_idx.get(&entry.stage_id) else {
            continue; // Unknown stage — skip.
        };
        for bf in &entry.block_factors {
            let Some(block_idx) = usize::try_from(bf.block_id).ok() else {
                continue;
            };
            if block_idx < max_blocks {
                table.set(bus_idx, stage_idx, block_idx, bf.factor);
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
    use cobre_core::EntityId;
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, StageRiskConfig, StageStateConfig,
    };

    fn make_bus(id: i32) -> Bus {
        Bus {
            id: EntityId(id),
            name: format!("B{id}"),
            deficit_segments: vec![],
            excess_cost: 0.0,
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

    fn make_entry(bus_id: i32, stage_id: i32, factors: &[(i32, f64)]) -> LoadFactorEntry {
        use crate::scenarios::BlockFactor;
        LoadFactorEntry {
            bus_id: EntityId(bus_id),
            stage_id,
            block_factors: factors
                .iter()
                .map(|&(block_id, factor)| BlockFactor { block_id, factor })
                .collect(),
        }
    }

    #[test]
    fn test_empty_entries_returns_empty() {
        let buses = vec![make_bus(0)];
        let stages = vec![make_stage(0, 2)];
        let table = resolve_load_factors(&[], &buses, &stages);
        assert!((table.factor(0, 0, 0) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_basic_resolution() {
        let buses = vec![make_bus(0), make_bus(1)];
        let stages = vec![make_stage(0, 3)];
        let entries = vec![make_entry(0, 0, &[(0, 0.85), (1, 1.15)])];

        let table = resolve_load_factors(&entries, &buses, &stages);
        assert!((table.factor(0, 0, 0) - 0.85).abs() < 1e-10);
        assert!((table.factor(0, 0, 1) - 1.15).abs() < 1e-10);
        assert!((table.factor(0, 0, 2) - 1.0).abs() < f64::EPSILON);
        // Bus 1 has no entry.
        assert!((table.factor(1, 0, 0) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_unknown_bus_id_skipped() {
        let buses = vec![make_bus(0)];
        let stages = vec![make_stage(0, 1)];
        let entries = vec![make_entry(99, 0, &[(0, 0.5)])];

        let table = resolve_load_factors(&entries, &buses, &stages);
        assert!((table.factor(0, 0, 0) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_unknown_stage_id_skipped() {
        let buses = vec![make_bus(0)];
        let stages = vec![make_stage(0, 1)];
        let entries = vec![make_entry(0, 99, &[(0, 0.5)])];

        let table = resolve_load_factors(&entries, &buses, &stages);
        assert!((table.factor(0, 0, 0) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_pre_study_stages_excluded() {
        let buses = vec![make_bus(0)];
        let stages = vec![make_stage(-1, 1), make_stage(0, 2)];
        let entries = vec![make_entry(0, 0, &[(0, 0.9)])];

        let table = resolve_load_factors(&entries, &buses, &stages);
        // stage_id=0 maps to stage_idx=0 (only study stages counted).
        assert!((table.factor(0, 0, 0) - 0.9).abs() < 1e-10);
    }
}
