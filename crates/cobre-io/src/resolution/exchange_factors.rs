//! Resolution of parsed exchange factor entries into a dense lookup table.
//!
//! [`resolve_exchange_factors`] converts `Vec<ExchangeFactorEntry>` (from
//! `constraints/exchange_factors.json`) into a [`ResolvedExchangeFactors`]
//! indexed by `(line_index, stage_index, block_index)` for O(1) lookup
//! during LP construction.
//!
//! Resolution is infallible: unknown line or stage IDs in the factor entries
//! are silently skipped. The default factor pair is `(1.0, 1.0)`.

use std::collections::HashMap;

use cobre_core::{Line, ResolvedExchangeFactors, Stage};

use crate::constraints::ExchangeFactorEntry;

/// Build a resolved exchange factor table from parsed entries.
///
/// Maps domain-level `line_id` and `stage_id` values to 0-based positional
/// indices using the provided sorted entity slices. Entries referencing
/// unknown lines or stages are silently skipped.
///
/// # Arguments
///
/// * `entries` — parsed exchange factor entries from `constraints/exchange_factors.json`
/// * `lines` — sorted line collection (for `line_id` to index mapping)
/// * `stages` — sorted stage collection (for `stage_id` to index mapping and max block count)
#[must_use]
pub fn resolve_exchange_factors(
    entries: &[ExchangeFactorEntry],
    lines: &[Line],
    stages: &[Stage],
) -> ResolvedExchangeFactors {
    if entries.is_empty() || lines.is_empty() || stages.is_empty() {
        return ResolvedExchangeFactors::empty();
    }

    let line_id_to_idx: HashMap<i32, usize> = lines
        .iter()
        .enumerate()
        .map(|(idx, l)| (l.id.0, idx))
        .collect();

    let stage_id_to_idx: HashMap<i32, usize> = stages
        .iter()
        .filter(|s| s.id >= 0)
        .enumerate()
        .map(|(idx, s)| (s.id, idx))
        .collect();

    let n_lines = lines.len();
    let n_stages = stage_id_to_idx.len();
    let max_blocks = stages
        .iter()
        .filter(|s| s.id >= 0)
        .map(|s| s.blocks.len())
        .max()
        .unwrap_or(0);

    let mut table = ResolvedExchangeFactors::new(n_lines, n_stages, max_blocks);

    for entry in entries {
        let Some(&line_idx) = line_id_to_idx.get(&entry.line_id) else {
            continue;
        };
        let Some(&stage_idx) = stage_id_to_idx.get(&entry.stage_id) else {
            continue;
        };
        for bf in &entry.block_factors {
            let Some(block_idx) = usize::try_from(bf.block_id).ok() else {
                continue;
            };
            if block_idx < max_blocks {
                table.set(
                    line_idx,
                    stage_idx,
                    block_idx,
                    bf.direct_factor,
                    bf.reverse_factor,
                );
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

    fn make_line(id: i32) -> Line {
        Line {
            id: EntityId(id),
            name: format!("L{id}"),
            source_bus_id: EntityId(0),
            target_bus_id: EntityId(1),
            entry_stage_id: None,
            exit_stage_id: None,
            direct_capacity_mw: 1000.0,
            reverse_capacity_mw: 800.0,
            losses_percent: 0.0,
            exchange_cost: 0.0,
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

    fn make_entry(line_id: i32, stage_id: i32, factors: &[(i32, f64, f64)]) -> ExchangeFactorEntry {
        use crate::constraints::BlockExchangeFactor;
        ExchangeFactorEntry {
            line_id,
            stage_id,
            block_factors: factors
                .iter()
                .map(|&(block_id, df, rf)| BlockExchangeFactor {
                    block_id,
                    direct_factor: df,
                    reverse_factor: rf,
                })
                .collect(),
        }
    }

    #[test]
    fn test_empty_entries_returns_empty() {
        let lines = vec![make_line(0)];
        let stages = vec![make_stage(0, 2)];
        let table = resolve_exchange_factors(&[], &lines, &stages);
        assert_eq!(table.factors(0, 0, 0), (1.0, 1.0));
    }

    #[test]
    fn test_basic_resolution() {
        let lines = vec![make_line(0)];
        let stages = vec![make_stage(0, 2)];
        let entries = vec![make_entry(0, 0, &[(0, 0.9, 0.85)])];

        let table = resolve_exchange_factors(&entries, &lines, &stages);
        assert_eq!(table.factors(0, 0, 0), (0.9, 0.85));
        assert_eq!(table.factors(0, 0, 1), (1.0, 1.0));
    }

    #[test]
    fn test_unknown_line_id_skipped() {
        let lines = vec![make_line(0)];
        let stages = vec![make_stage(0, 1)];
        let entries = vec![make_entry(99, 0, &[(0, 0.5, 0.5)])];

        let table = resolve_exchange_factors(&entries, &lines, &stages);
        assert_eq!(table.factors(0, 0, 0), (1.0, 1.0));
    }

    #[test]
    fn test_unknown_stage_id_skipped() {
        let lines = vec![make_line(0)];
        let stages = vec![make_stage(0, 1)];
        let entries = vec![make_entry(0, 99, &[(0, 0.5, 0.5)])];

        let table = resolve_exchange_factors(&entries, &lines, &stages);
        assert_eq!(table.factors(0, 0, 0), (1.0, 1.0));
    }
}
