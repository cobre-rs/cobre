//! Layer 6 — cross-file productivity resolution.
//!
//! Enforces the authoring contract that for every non-FPHA
//! `(hydro, stage)` pair, equivalent productivity (`ρ_eq`) is supplied
//! by exactly one of:
//!
//! - `system/hydro_production_models.json` -> `productivity_mw_per_m3s`
//!   on the matching `stage_range` or `seasonal` entry, OR
//! - `system/hydro_energy_productivity.parquet` ->
//!   `equivalent_productivity_mw_per_m3s` for a stage-specific or
//!   per-hydro-default row.
//!
//! Supplying a value from both sources for the same `(hydro, stage)`
//! is a `ErrorKind::SchemaViolation`; supplying neither is a
//! `ErrorKind::DimensionMismatch`.
//! FPHA hydros are skipped: their `ρ_eq` derives from `VHA` + `ρ_esp`,
//! with the parquet override winning when present, and the JSON
//! parser already rejects `productivity_mw_per_m3s` for FPHA.

use std::collections::HashMap;

use cobre_core::{EntityId, entities::HydroGenerationModel, temporal::Stage};

use crate::{
    extensions::{HydroEnergyProductivityRow, ProductionModelConfig, SelectionMode},
    validation::{ValidationContext, rules},
};

use super::schema::ParsedData;

/// Validate that every non-FPHA (hydro, stage) pair has exactly one source
/// of `equivalent_productivity_mw_per_m3s`.
///
/// Called as Layer 6 in the validation pipeline, after Layers 3-5 have
/// established referential and dimensional consistency. All conflicts and
/// coverage gaps are collected before returning — no short-circuiting.
pub(crate) fn validate_productivity_resolution(data: &ParsedData, ctx: &mut ValidationContext) {
    let parquet_map = build_parquet_map(&data.hydro_energy_productivity_rows);

    let json_map: HashMap<EntityId, &ProductionModelConfig> = data
        .production_models
        .iter()
        .map(|c| (c.hydro_id, c))
        .collect();

    let study_stages: Vec<&Stage> = data.stages.stages.iter().filter(|s| s.id >= 0).collect();

    for hydro in &data.hydros {
        if hydro.generation_model == HydroGenerationModel::Fpha {
            continue;
        }

        for stage in &study_stages {
            let parquet_value = parquet_lookup(&parquet_map, hydro.id, stage.id);
            let json_value = json_map
                .get(&hydro.id)
                .and_then(|config| find_productivity_for_stage(config, stage));

            match (parquet_value, json_value) {
                (Some(p), Some(j)) => {
                    ctx.emit(
                        &rules::PRODUCTIVITY_SUPPLIED_TWICE,
                        "system/hydro_energy_productivity.parquet",
                        Some(format!("hydro_id={}, stage_id={}", hydro.id.0, stage.id)),
                        format!(
                            "conflict for (hydro_id={}, stage_id={}): \
                             value {} from system/hydro_energy_productivity.parquet \
                             conflicts with value {} from \
                             system/hydro_production_models.json; \
                             supply the value from exactly one source",
                            hydro.id.0, stage.id, p, j,
                        ),
                    );
                }
                (None, None) => {
                    ctx.emit(
                        &rules::PRODUCTIVITY_MISSING,
                        "system/hydro_production_models.json",
                        Some(format!("hydro_id={}, stage_id={}", hydro.id.0, stage.id)),
                        format!(
                            "no productivity_mw_per_m3s available for \
                             (hydro_id={}, stage_id={}): not supplied by \
                             system/hydro_production_models.json and not present in \
                             system/hydro_energy_productivity.parquet",
                            hydro.id.0, stage.id,
                        ),
                    );
                }
                (Some(_), None) | (None, Some(_)) => {}
            }
        }
    }
}

// ── Parquet lookup ────────────────────────────────────────────────────────────

type ParquetKey = (EntityId, Option<i32>);

/// Build a `HashMap` from `(hydro_id, stage_id)` to `rho_eq` value.
///
/// Rows where `equivalent_productivity_mw_per_m3s` is `None` are excluded —
/// only rows with a concrete value count as "supplied".
fn build_parquet_map(rows: &[HydroEnergyProductivityRow]) -> HashMap<ParquetKey, f64> {
    rows.iter()
        .filter_map(|row| {
            row.equivalent_productivity_mw_per_m3s
                .map(|v| ((row.hydro_id, row.stage_id), v))
        })
        .collect()
}

/// Look up the parquet value for a given `(hydro_id, stage_id)`.
///
/// Prefers the stage-specific row (`stage_id == Some(stage_id)`);
/// falls back to the per-hydro default row (`stage_id == None`).
fn parquet_lookup(
    map: &HashMap<ParquetKey, f64>,
    hydro_id: EntityId,
    stage_id: i32,
) -> Option<f64> {
    map.get(&(hydro_id, Some(stage_id)))
        .or_else(|| map.get(&(hydro_id, None)))
        .copied()
}

// ── JSON model lookup ─────────────────────────────────────────────────────────

/// Find the `productivity_mw_per_m3s` from the JSON config for a given stage.
///
/// The matching logic is reimplemented locally so this crate carries no
/// dependency on downstream solver crates and no algorithm-specific identifiers
/// — a mirror of the solver crate's `resolve_stage`/`selection_entries`
/// resolver's season-miss shape, kept in agreement rather than called.
///
/// For `Seasonal`, a stage with no matching season entry falls back to the
/// `default_model`, which carries no explicit productivity (returns `None`).
fn find_productivity_for_stage(config: &ProductionModelConfig, stage: &Stage) -> Option<f64> {
    match &config.selection_mode {
        SelectionMode::StageRanges { ranges } => ranges
            .iter()
            .find(|range| {
                stage.id >= range.start_stage_id
                    && range.end_stage_id.is_none_or(|end| stage.id <= end)
            })
            .and_then(|range| range.productivity_mw_per_m3s),
        SelectionMode::Seasonal { seasons, .. } => {
            let season_id = stage.season_id?;
            let sid = i32::try_from(season_id).ok()?;
            seasons
                .iter()
                .find(|season| season.season_id == sid)
                .and_then(|season| season.productivity_mw_per_m3s)
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::too_many_lines
)]
mod tests {
    use chrono::NaiveDate;
    use cobre_core::{
        DeficitSegment, EntityId, HorizonGraph, Hydro,
        entities::{Bus, HydroGenerationModel},
        temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
            StageStateConfig,
        },
    };

    use super::{find_productivity_for_stage, validate_productivity_resolution};
    use crate::{
        extensions::{
            HydroEnergyProductivityRow, ProductionModelConfig, SeasonConfig, SelectionMode,
            StageRange,
        },
        stages::StagesData,
        validation::{ErrorKind, ValidationContext},
    };

    // ── helpers ───────────────────────────────────────────────────────────────

    fn make_stage(id: i32) -> Stage {
        let index = usize::try_from(id).expect("stage id is non-negative in tests");
        Stage {
            index,
            id,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: None,
            blocks: vec![Block {
                index: 0,
                name: "FLAT".to_string(),
                duration_hours: 744.0,
            }],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: true,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 50,
                noise_method: NoiseMethod::Saa,
            },
        }
    }

    fn make_hydro(id: i32, model: HydroGenerationModel) -> Hydro {
        Hydro {
            unit_groups: Vec::new(),
            id: EntityId(id),
            name: format!("Hydro {id}"),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            downstream_id: None,
            travel_time_hours: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 1000.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: model,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 100.0,
            specific_productivity_mw_per_m3s_per_m: None,
            min_generation_mw: 0.0,
            max_generation_mw: 500.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: crate::test_support::penalties_all(1.0),
        }
    }

    fn parquet_row(
        hydro_id: i32,
        stage_id: Option<i32>,
        rho_eq: Option<f64>,
    ) -> HydroEnergyProductivityRow {
        HydroEnergyProductivityRow {
            hydro_id: EntityId(hydro_id),
            stage_id,
            equivalent_productivity_mw_per_m3s: rho_eq,
            reference_outflow_m3s: None,
            specific_productivity_mw_per_m3s_per_m: None,
        }
    }

    fn stage_range_config(hydro_id: i32, productivity: Option<f64>) -> ProductionModelConfig {
        ProductionModelConfig {
            hydro_id: EntityId(hydro_id),
            selection_mode: SelectionMode::StageRanges {
                ranges: vec![StageRange {
                    start_stage_id: 0,
                    end_stage_id: None,
                    model: "constant_productivity".to_string(),
                    fpha_config: None,
                    reference_volume: None,
                    productivity_mw_per_m3s: productivity,
                }],
            },
        }
    }

    fn stages() -> StagesData {
        StagesData {
            openings_declared: std::collections::HashSet::new(),
            stages: vec![make_stage(0)],
            policy_graph: HorizonGraph::default(),
        }
    }

    fn buses() -> Vec<Bus> {
        vec![Bus {
            id: EntityId(1),
            name: "BUS_1".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 1000.0,
            }],
            excess_cost: 100.0,
        }]
    }

    // ── Unit tests ─────────────────────────────────────────────────────────────

    #[test]
    fn test_no_error_when_only_parquet_supplies_value() {
        let mut data = crate::test_support::base_parsed_data(stages(), buses());
        data.hydros = vec![make_hydro(0, HydroGenerationModel::ConstantProductivity)];
        data.production_models = vec![];
        data.hydro_energy_productivity_rows = vec![parquet_row(0, Some(0), Some(0.9))];

        let mut ctx = ValidationContext::new();
        validate_productivity_resolution(&data, &mut ctx);

        assert!(
            !ctx.has_errors(),
            "expected no errors when parquet alone supplies the value"
        );
    }

    #[test]
    fn test_no_error_when_only_json_supplies_value() {
        let mut data = crate::test_support::base_parsed_data(stages(), buses());
        data.hydros = vec![make_hydro(0, HydroGenerationModel::ConstantProductivity)];
        data.production_models = vec![stage_range_config(0, Some(0.9))];
        data.hydro_energy_productivity_rows = vec![];

        let mut ctx = ValidationContext::new();
        validate_productivity_resolution(&data, &mut ctx);

        assert!(
            !ctx.has_errors(),
            "expected no errors when JSON alone supplies the value"
        );
    }

    #[test]
    fn test_conflict_when_both_supply_value() {
        let mut data = crate::test_support::base_parsed_data(stages(), buses());
        data.hydros = vec![make_hydro(0, HydroGenerationModel::ConstantProductivity)];
        data.production_models = vec![stage_range_config(0, Some(0.9))];
        data.hydro_energy_productivity_rows = vec![parquet_row(0, Some(0), Some(1.1))];

        let mut ctx = ValidationContext::new();
        validate_productivity_resolution(&data, &mut ctx);

        assert_eq!(ctx.error_count(), 1, "expected exactly one conflict error");
        let err = ctx.errors()[0];
        assert_eq!(err.kind, ErrorKind::SchemaViolation);
        let msg = &err.message;
        assert!(
            msg.contains("conflict for (hydro_id=0, stage_id=0)"),
            "message should identify the pair, got: {msg}"
        );
        assert!(
            msg.contains("system/hydro_energy_productivity.parquet"),
            "message should reference parquet file, got: {msg}"
        );
        assert!(
            msg.contains("system/hydro_production_models.json"),
            "message should reference JSON file, got: {msg}"
        );
    }

    #[test]
    fn test_gap_when_neither_supplies_value() {
        let mut data = crate::test_support::base_parsed_data(stages(), buses());
        data.hydros = vec![make_hydro(0, HydroGenerationModel::ConstantProductivity)];
        // JSON has an entry but productivity is None.
        data.production_models = vec![stage_range_config(0, None)];
        data.hydro_energy_productivity_rows = vec![];

        let mut ctx = ValidationContext::new();
        validate_productivity_resolution(&data, &mut ctx);

        assert_eq!(ctx.error_count(), 1, "expected exactly one gap error");
        let err = ctx.errors()[0];
        assert_eq!(err.kind, ErrorKind::DimensionMismatch);
        let msg = &err.message;
        assert!(
            msg.contains("no productivity_mw_per_m3s available for (hydro_id=0, stage_id=0)"),
            "message should describe the gap, got: {msg}"
        );
    }

    #[test]
    fn test_per_hydro_default_covers_when_stage_specific_absent() {
        let mut data = crate::test_support::base_parsed_data(stages(), buses());
        data.hydros = vec![make_hydro(0, HydroGenerationModel::ConstantProductivity)];
        data.production_models = vec![stage_range_config(0, None)];
        // Per-hydro default row (stage_id = None).
        data.hydro_energy_productivity_rows = vec![parquet_row(0, None, Some(0.7))];

        let mut ctx = ValidationContext::new();
        validate_productivity_resolution(&data, &mut ctx);

        assert!(
            !ctx.has_errors(),
            "expected no errors: per-hydro default row should cover the gap"
        );
    }

    /// JSON has `None`; parquet has both a per-hydro default (`stage_id = None`,
    /// `0.7`) and a stage-specific row (`stage_id = Some(0)`, `0.9`).
    /// Both parquet rows are accepted; stage-specific wins for the lookup.
    /// No conflict between two parquet rows — no error expected.
    #[test]
    fn test_stage_specific_wins_over_per_hydro_default() {
        let mut data = crate::test_support::base_parsed_data(stages(), buses());
        data.hydros = vec![make_hydro(0, HydroGenerationModel::ConstantProductivity)];
        data.production_models = vec![stage_range_config(0, None)];
        data.hydro_energy_productivity_rows = vec![
            parquet_row(0, None, Some(0.7)),
            parquet_row(0, Some(0), Some(0.9)),
        ];

        let mut ctx = ValidationContext::new();
        validate_productivity_resolution(&data, &mut ctx);

        assert!(
            !ctx.has_errors(),
            "expected no errors: stage-specific parquet row satisfies the requirement"
        );
    }

    #[test]
    fn test_fpha_hydros_are_not_validated() {
        let mut data = crate::test_support::base_parsed_data(stages(), buses());
        data.hydros = vec![make_hydro(0, HydroGenerationModel::Fpha)];
        // No production model config for hydro 0 (FPHA parser rejects productivity).
        data.production_models = vec![];
        data.hydro_energy_productivity_rows = vec![];

        let mut ctx = ValidationContext::new();
        validate_productivity_resolution(&data, &mut ctx);

        assert!(
            !ctx.has_errors(),
            "expected no errors: FPHA hydros must not be validated"
        );
    }

    #[test]
    fn test_multiple_issues_collected_in_one_pass() {
        let mut data = crate::test_support::base_parsed_data(stages(), buses());
        data.hydros = vec![
            make_hydro(0, HydroGenerationModel::ConstantProductivity),
            make_hydro(1, HydroGenerationModel::LinearizedHead),
        ];
        // Hydro 0: conflict — JSON Some(0.9), parquet Some(1.1).
        data.production_models = vec![
            stage_range_config(0, Some(0.9)),
            stage_range_config(1, None),
        ];
        data.hydro_energy_productivity_rows = vec![
            parquet_row(0, Some(0), Some(1.1)),
            // Hydro 1: no parquet row → gap.
        ];

        let mut ctx = ValidationContext::new();
        validate_productivity_resolution(&data, &mut ctx);

        assert_eq!(
            ctx.error_count(),
            2,
            "expected two errors: one conflict and one gap"
        );
        let kinds: Vec<ErrorKind> = ctx.errors().iter().map(|e| e.kind).collect();
        assert!(
            kinds.contains(&ErrorKind::SchemaViolation),
            "expected a SchemaViolation for the conflict"
        );
        assert!(
            kinds.contains(&ErrorKind::DimensionMismatch),
            "expected a DimensionMismatch for the gap"
        );
    }

    // ── Helper unit tests for find_productivity_for_stage ─────────────────────

    #[test]
    fn test_find_productivity_stage_ranges_match() {
        let config = stage_range_config(0, Some(0.5));
        let stage = make_stage(0);
        let result = find_productivity_for_stage(&config, &stage);
        assert_eq!(result, Some(0.5));
    }

    #[test]
    fn test_find_productivity_stage_ranges_no_match() {
        let config = ProductionModelConfig {
            hydro_id: EntityId(0),
            selection_mode: SelectionMode::StageRanges {
                ranges: vec![StageRange {
                    start_stage_id: 0,
                    end_stage_id: Some(5),
                    model: "constant_productivity".to_string(),
                    fpha_config: None,
                    reference_volume: None,
                    productivity_mw_per_m3s: Some(0.5),
                }],
            },
        };
        let stage = make_stage(99);
        let result = find_productivity_for_stage(&config, &stage);
        assert_eq!(result, None);
    }

    #[test]
    fn test_find_productivity_seasonal_match() {
        let config = ProductionModelConfig {
            hydro_id: EntityId(0),
            selection_mode: SelectionMode::Seasonal {
                default_model: "constant_productivity".to_string(),
                seasons: vec![SeasonConfig {
                    season_id: 0,
                    model: "constant_productivity".to_string(),
                    fpha_config: None,
                    reference_volume: None,
                    productivity_mw_per_m3s: Some(0.8),
                }],
            },
        };
        let mut stage = make_stage(0);
        stage.season_id = Some(0);
        let result = find_productivity_for_stage(&config, &stage);
        assert_eq!(result, Some(0.8));
    }

    #[test]
    fn test_find_productivity_seasonal_fallback_to_default_is_none() {
        let config = ProductionModelConfig {
            hydro_id: EntityId(0),
            selection_mode: SelectionMode::Seasonal {
                default_model: "constant_productivity".to_string(),
                seasons: vec![SeasonConfig {
                    season_id: 5,
                    model: "constant_productivity".to_string(),
                    fpha_config: None,
                    reference_volume: None,
                    productivity_mw_per_m3s: Some(0.8),
                }],
            },
        };
        // Stage has season_id=0, which is not in the seasons list.
        let mut stage = make_stage(0);
        stage.season_id = Some(0);
        let result = find_productivity_for_stage(&config, &stage);
        assert_eq!(
            result, None,
            "default model falls back to None productivity"
        );
    }
}
