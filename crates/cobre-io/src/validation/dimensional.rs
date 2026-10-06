//! Layer 4 — Dimensional consistency validation.
//!
//! Verifies that every entity which requires companion data actually has it,
//! and that companion data arrays have the expected dimensions. Coverage checks
//! are only performed when the optional data is present (non-empty `Vec` or
//! `Some`).
//!
//! The primary entry point is `validate_dimensional_consistency`. Its rules are the
//! `dimensional.*` entries of [`RULES`](super::rules::RULES).

use std::collections::{HashMap, HashSet};

use cobre_core::Hydro;
use cobre_core::entities::HydroGenerationModel;

use super::{ValidationContext, rules, schema::ParsedData};
use crate::extensions::{ProductionModelConfig, SelectionMode};

// ── validate_dimensional_consistency ─────────────────────────────────────────

/// Performs Layer 4 dimensional consistency validation on the parsed data.
///
/// Every coverage rule runs unconditionally — earlier failures do not
/// short-circuit later rules. A rule whose optional data is absent (empty `Vec`
/// or `None`) is silently skipped. Each failure adds one
/// `ErrorKind::DimensionMismatch` entry to `ctx`; the function is infallible.
// Rationale: all rules share one pass over the same parsed data and must run
// unconditionally so the caller receives a complete diagnostic set; one function per
// rule would force separate traversals or an intermediary structure with no benefit.
#[allow(clippy::too_many_lines)]
pub(crate) fn validate_dimensional_consistency(data: &ParsedData, ctx: &mut ValidationContext) {
    let study_stage_ids: Vec<i32> = data
        .stages
        .stages
        .iter()
        .filter(|s| s.id >= 0)
        .map(|s| s.id)
        .collect();

    if !data.inflow_seasonal_stats.is_empty() {
        let inflow_pairs: HashSet<(i32, i32)> = data
            .inflow_seasonal_stats
            .iter()
            .map(|row| (row.hydro_id.0, row.stage_id))
            .collect();

        for hydro in &data.hydros {
            for &stage_id in &study_stage_ids {
                if let Some(entry) = hydro.entry_stage_id
                    && stage_id < entry
                {
                    continue;
                }
                if let Some(exit) = hydro.exit_stage_id
                    && stage_id >= exit
                {
                    continue;
                }

                if !inflow_pairs.contains(&(hydro.id.0, stage_id)) {
                    ctx.emit(
                        &rules::DIMENSIONAL_INFLOW_STATS_COVERAGE,
                        "scenarios/inflow_seasonal_stats.parquet",
                        Some(format!("Hydro {}", hydro.id.0)),
                        format!(
                            "Hydro {} missing inflow seasonal stats for stage {}",
                            hydro.id.0, stage_id
                        ),
                    );
                }
            }
        }
    }

    if !data.load_seasonal_stats.is_empty() {
        let load_pairs: HashSet<(i32, i32)> = data
            .load_seasonal_stats
            .iter()
            .map(|row| (row.bus_id.0, row.stage_id))
            .collect();

        for bus in &data.buses {
            for &stage_id in &study_stage_ids {
                if !load_pairs.contains(&(bus.id.0, stage_id)) {
                    ctx.emit(
                        &rules::DIMENSIONAL_LOAD_STATS_COVERAGE,
                        "scenarios/load_seasonal_stats.parquet",
                        Some(format!("Bus {}", bus.id.0)),
                        format!(
                            "Bus {} missing load seasonal stats for stage {}",
                            bus.id.0, stage_id
                        ),
                    );
                }
            }
        }
    }

    if let Some(correlation) = &data.correlation {
        for (profile_name, profile) in &correlation.profiles {
            for group in &profile.groups {
                let n_entities = group.entities.len();
                let n_rows = group.matrix.len();

                if n_rows != n_entities {
                    ctx.emit(
                        &rules::DIMENSIONAL_CORRELATION_ROW_COUNT,
                        "scenarios/correlation.json",
                        Some(format!("group '{}' in profile '{}'", group.name, profile_name)),
                        format!(
                            "Correlation group '{}' in profile '{}': matrix has {} rows but {} entities",
                            group.name, profile_name, n_rows, n_entities
                        ),
                    );
                    continue;
                }

                for (i, row) in group.matrix.iter().enumerate() {
                    if row.len() != n_entities {
                        ctx.emit(
                            &rules::DIMENSIONAL_CORRELATION_ROW_LENGTH,
                            "scenarios/correlation.json",
                            Some(format!("group '{}' in profile '{}'", group.name, profile_name)),
                            format!(
                                "Correlation group '{}' in profile '{}': matrix row {} has {} columns but {} entities",
                                group.name, profile_name, i, row.len(), n_entities
                            ),
                        );
                    }
                }
            }
        }

        for entry in &correlation.schedule {
            if !correlation.profiles.contains_key(&entry.profile_name) {
                ctx.emit(
                    &rules::DIMENSIONAL_CORRELATION_PROFILE,
                    "scenarios/correlation.json",
                    Some(format!("schedule stage_id={}", entry.stage_id)),
                    format!(
                        "Correlation schedule references profile '{}' which does not exist in profiles",
                        entry.profile_name
                    ),
                );
            }
        }
    }

    if !data.fpha_hyperplanes.is_empty() {
        let hydros_with_hyperplanes: HashSet<i32> = data
            .fpha_hyperplanes
            .iter()
            .map(|row| row.hydro_id.0)
            .collect();

        let fpha_hydro_ids = collect_fpha_hydro_ids(&data.hydros, &data.production_models);
        let capacity_less: HashSet<i32> = data
            .hydros
            .iter()
            .filter(|h| !h.has_turbine_capacity())
            .map(|h| h.id.0)
            .collect();

        for &hydro_id in &fpha_hydro_ids {
            if capacity_less.contains(&hydro_id) {
                continue;
            }
            if !hydros_with_hyperplanes.contains(&hydro_id) {
                ctx.emit(
                    &rules::DIMENSIONAL_FPHA_HYPERPLANES,
                    "system/fpha_hyperplanes.parquet",
                    Some(format!("Hydro {hydro_id}")),
                    format!(
                        "Hydro {hydro_id} is configured with FPHA model but has no FPHA hyperplanes"
                    ),
                );
            }
        }
    }

    if !data.hydro_geometry.is_empty() {
        let mut geometry_row_counts: HashMap<i32, usize> = HashMap::new();
        for row in &data.hydro_geometry {
            *geometry_row_counts.entry(row.hydro_id.0).or_insert(0) += 1;
        }

        let head_hydro_ids =
            collect_head_dependent_hydro_ids(&data.hydros, &data.production_models);
        let fpha_hydro_ids = collect_fpha_hydro_ids(&data.hydros, &data.production_models);

        for &hydro_id in &head_hydro_ids {
            let count = geometry_row_counts.get(&hydro_id).copied().unwrap_or(0);
            // FPHA accepts a single geometry row (constant run-of-river forebay,
            // γ_V = 0); `linearized_head` needs ≥ 2 because it fits a head slope in
            // volume, which one point cannot define.
            let min_required = if fpha_hydro_ids.contains(&hydro_id) {
                1
            } else {
                2
            };
            if count < min_required {
                ctx.emit(
                    &rules::DIMENSIONAL_GEOMETRY_ROWS,
                    "system/hydro_geometry.parquet",
                    Some(format!("Hydro {hydro_id}")),
                    format!(
                        "Hydro {hydro_id} requires head-dependent model but has {count} hydro geometry row(s) (minimum {min_required} required)"
                    ),
                );
            }
        }
    }
}

// ── Private helpers ───────────────────────────────────────────────────────────

fn collect_fpha_hydro_ids(
    hydros: &[Hydro],
    production_models: &[ProductionModelConfig],
) -> HashSet<i32> {
    let mut ids = HashSet::new();

    for hydro in hydros {
        if matches!(hydro.generation_model, HydroGenerationModel::Fpha) {
            ids.insert(hydro.id.0);
        }
    }

    for config in production_models {
        if production_model_uses_fpha(config) {
            ids.insert(config.hydro_id.0);
        }
    }

    ids
}

fn collect_head_dependent_hydro_ids(
    hydros: &[Hydro],
    production_models: &[ProductionModelConfig],
) -> HashSet<i32> {
    let mut ids = HashSet::new();

    for hydro in hydros {
        if matches!(
            hydro.generation_model,
            HydroGenerationModel::Fpha | HydroGenerationModel::LinearizedHead
        ) {
            ids.insert(hydro.id.0);
        }
    }

    for config in production_models {
        if production_model_uses_head_dependent(config) {
            ids.insert(config.hydro_id.0);
        }
    }

    ids
}

// `production_model_uses_fpha` and `production_model_uses_head_dependent`
// mirror the solver crate's `resolve_stage`/`selection_entries` resolver: a
// Seasonal season miss falls back to `default_model`, so each fold reads
// `default_model` too — keep in agreement with that resolver's season-miss
// semantics, not just the declared seasons.
fn production_model_uses_fpha(config: &ProductionModelConfig) -> bool {
    match &config.selection_mode {
        SelectionMode::StageRanges { ranges } => ranges.iter().any(|r| r.model == "fpha"),
        SelectionMode::Seasonal {
            default_model,
            seasons,
        } => default_model == "fpha" || seasons.iter().any(|s| s.model == "fpha"),
    }
}

// Same mirror contract as `production_model_uses_fpha` above, extended to
// `linearized_head`.
fn production_model_uses_head_dependent(config: &ProductionModelConfig) -> bool {
    match &config.selection_mode {
        SelectionMode::StageRanges { ranges } => ranges
            .iter()
            .any(|r| r.model == "fpha" || r.model == "linearized_head"),
        SelectionMode::Seasonal {
            default_model,
            seasons,
        } => {
            default_model == "fpha"
                || default_model == "linearized_head"
                || seasons
                    .iter()
                    .any(|s| s.model == "fpha" || s.model == "linearized_head")
        }
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::too_many_lines,
    clippy::doc_markdown,
    clippy::cast_sign_loss,
    clippy::uninlined_format_args
)]
mod tests {
    use std::collections::BTreeMap;

    use cobre_core::{
        EntityId, HorizonGraph,
        entities::{Bus, HydroGenerationModel},
        scenario::{
            CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile,
            CorrelationScheduleEntry,
        },
        temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
            StageStateConfig,
        },
    };

    use crate::{
        extensions::{FphaHyperplaneRow, HydroGeometryRow, SeasonConfig},
        scenarios::{InflowSeasonalStatsRow, LoadSeasonalStatsRow},
        validation::{ErrorKind, ValidationContext},
    };

    use super::*;
    use chrono::NaiveDate;

    fn make_hydro(
        id: i32,
        generation_model: HydroGenerationModel,
        entry_stage_id: Option<i32>,
        exit_stage_id: Option<i32>,
    ) -> Hydro {
        Hydro {
            unit_groups: Vec::new(),
            id: EntityId(id),
            name: format!("Hydro {id}"),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            downstream_id: None,
            travel_time_hours: None,
            entry_stage_id,
            exit_stage_id,
            min_storage_hm3: 0.0,
            max_storage_hm3: 1000.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model,
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

    fn make_bus(id: i32) -> Bus {
        Bus {
            id: EntityId(id),
            name: format!("Bus {id}"),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            deficit_segments: vec![],
            excess_cost: 0.0,
        }
    }

    fn make_stage(id: i32) -> Stage {
        Stage {
            index: id as usize,
            id,
            start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: chrono::NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
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

    fn make_pre_study_stage(id: i32) -> Stage {
        // Pre-study stages have negative IDs.
        Stage {
            index: 0,
            id,
            start_date: chrono::NaiveDate::from_ymd_opt(2023, 12, 1).unwrap(),
            end_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            season_id: None,
            blocks: vec![],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: false,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        }
    }

    fn inflow_stats_row(hydro_id: i32, stage_id: i32) -> InflowSeasonalStatsRow {
        InflowSeasonalStatsRow {
            hydro_id: EntityId(hydro_id),
            stage_id,
            mean_m3s: 100.0,
            std_m3s: 10.0,
        }
    }

    fn load_stats_row(bus_id: i32, stage_id: i32) -> LoadSeasonalStatsRow {
        LoadSeasonalStatsRow {
            bus_id: EntityId(bus_id),
            stage_id,
            mean_mw: 500.0,
            std_mw: 50.0,
        }
    }

    fn fpha_row(hydro_id: i32) -> FphaHyperplaneRow {
        FphaHyperplaneRow {
            hydro_id: EntityId(hydro_id),
            stage_id: None,
            plane_id: 0,
            gamma_0: 100.0,
            gamma_v: 0.001,
            gamma_q: 0.9,
            gamma_s: -0.01,
            kappa: 1.0,
            valid_v_min_hm3: None,
            valid_v_max_hm3: None,
            valid_q_max_m3s: None,
        }
    }

    fn geometry_row(hydro_id: i32, volume: f64) -> HydroGeometryRow {
        HydroGeometryRow {
            hydro_id: EntityId(hydro_id),
            volume_hm3: volume,
            height_m: volume * 0.1,
            area_km2: volume * 0.001,
        }
    }

    fn make_correlation_model(
        groups: Vec<CorrelationGroup>,
        schedule: Vec<CorrelationScheduleEntry>,
    ) -> CorrelationModel {
        let mut profiles = BTreeMap::new();
        profiles.insert("default".to_string(), CorrelationProfile { groups });
        CorrelationModel {
            method: "spectral".to_string(),
            profiles,
            schedule,
        }
    }

    fn stages() -> crate::stages::StagesData {
        crate::stages::StagesData {
            openings_declared: std::collections::HashSet::new(),
            stages: vec![make_stage(0), make_stage(1)],
            policy_graph: HorizonGraph::default(),
        }
    }

    // ── AC 1: Valid coverage — no errors ──────────────────────────────────────

    /// Given ParsedData where every hydro has InflowModel entries for every
    /// study stage and all other coverage rules are satisfied,
    /// `validate_dimensional_consistency` produces no errors.
    #[test]
    fn test_valid_coverage_no_errors() {
        let mut data = crate::test_support::base_parsed_data(stages(), vec![]);

        // 2 hydros, 2 study stages — all 4 (hydro, stage) pairs present.
        data.hydros = vec![
            make_hydro(1, HydroGenerationModel::ConstantProductivity, None, None),
            make_hydro(2, HydroGenerationModel::ConstantProductivity, None, None),
        ];
        data.inflow_seasonal_stats = vec![
            inflow_stats_row(1, 0),
            inflow_stats_row(1, 1),
            inflow_stats_row(2, 0),
            inflow_stats_row(2, 1),
        ];

        let mut ctx = ValidationContext::new();
        validate_dimensional_consistency(&data, &mut ctx);

        assert!(
            !ctx.has_errors(),
            "expected no errors for valid coverage, got: {:?}",
            ctx.errors()
        );
    }

    // ── AC 2: Missing inflow stats for one hydro at one stage → 1 error ───────

    /// Given ParsedData with 3 hydros and 2 study stages where hydro id=2 is
    /// missing inflow stats for stage id=1, exactly 1 DimensionMismatch error
    /// is produced with message containing "Hydro 2" and "stage 1".
    #[test]
    fn test_missing_inflow_stats_one_hydro_one_stage() {
        let mut data = crate::test_support::base_parsed_data(stages(), vec![]);

        data.hydros = vec![
            make_hydro(1, HydroGenerationModel::ConstantProductivity, None, None),
            make_hydro(2, HydroGenerationModel::ConstantProductivity, None, None),
            make_hydro(3, HydroGenerationModel::ConstantProductivity, None, None),
        ];

        data.inflow_seasonal_stats = vec![
            inflow_stats_row(1, 0),
            inflow_stats_row(1, 1),
            inflow_stats_row(2, 0), // stage 1 missing for hydro 2
            inflow_stats_row(3, 0),
            inflow_stats_row(3, 1),
        ];

        let mut ctx = ValidationContext::new();
        validate_dimensional_consistency(&data, &mut ctx);

        let errors = ctx.errors();
        assert_eq!(
            errors.len(),
            1,
            "expected exactly 1 error, got {}: {:?}",
            errors.len(),
            errors
        );
        assert_eq!(errors[0].kind, ErrorKind::DimensionMismatch);
        assert!(
            errors[0].message.contains("Hydro 2"),
            "error message should contain 'Hydro 2', got: {}",
            errors[0].message
        );
        assert!(
            errors[0].message.contains("stage 1"),
            "error message should contain 'stage 1', got: {}",
            errors[0].message
        );
    }

    // ── AC 3: Correlation matrix row count mismatch → 1 error ─────────────────

    /// Given a ParsedData where a CorrelationGroup named "Southeast" has
    /// 3 entities but a 2x2 matrix, one DimensionMismatch error is produced
    /// with message containing "Southeast" and "3 entities".
    #[test]
    fn test_correlation_matrix_row_count_mismatch() {
        let mut data = crate::test_support::base_parsed_data(stages(), vec![]);

        let group = CorrelationGroup {
            name: "Southeast".to_string(),
            entities: vec![
                CorrelationEntity {
                    entity_type: "inflow".to_string(),
                    id: EntityId(1),
                },
                CorrelationEntity {
                    entity_type: "inflow".to_string(),
                    id: EntityId(2),
                },
                CorrelationEntity {
                    entity_type: "inflow".to_string(),
                    id: EntityId(3),
                },
            ],
            // Only 2 rows — mismatch: entities.len() == 3 but matrix.len() == 2.
            matrix: vec![vec![1.0, 0.8], vec![0.8, 1.0]],
        };

        data.correlation = Some(make_correlation_model(vec![group], vec![]));

        let mut ctx = ValidationContext::new();
        validate_dimensional_consistency(&data, &mut ctx);

        let errors = ctx.errors();
        assert!(
            !errors.is_empty(),
            "expected at least 1 error for row count mismatch"
        );
        assert_eq!(errors[0].kind, ErrorKind::DimensionMismatch);
        assert!(
            errors[0].message.contains("Southeast"),
            "error should mention group name 'Southeast', got: {}",
            errors[0].message
        );
        assert!(
            errors[0].message.contains("3 entities"),
            "error should mention '3 entities', got: {}",
            errors[0].message
        );
    }

    // ── AC 4: Correlation matrix non-square → 1 error ─────────────────────────

    /// Given a CorrelationGroup with 2 entities but one row has only 1 column,
    /// one DimensionMismatch error is produced.
    #[test]
    fn test_correlation_matrix_non_square_row() {
        let mut data = crate::test_support::base_parsed_data(stages(), vec![]);

        let group = CorrelationGroup {
            name: "North".to_string(),
            entities: vec![
                CorrelationEntity {
                    entity_type: "inflow".to_string(),
                    id: EntityId(1),
                },
                CorrelationEntity {
                    entity_type: "inflow".to_string(),
                    id: EntityId(2),
                },
            ],
            // 2 rows, but row 1 has only 1 column → non-square.
            matrix: vec![vec![1.0, 0.5], vec![0.5]],
        };

        data.correlation = Some(make_correlation_model(vec![group], vec![]));

        let mut ctx = ValidationContext::new();
        validate_dimensional_consistency(&data, &mut ctx);

        let errors = ctx.errors();
        assert_eq!(
            errors.len(),
            1,
            "expected exactly 1 column mismatch error, got: {:?}",
            errors
        );
        assert_eq!(errors[0].kind, ErrorKind::DimensionMismatch);
        assert!(
            errors[0].message.contains("North"),
            "error should mention group name 'North', got: {}",
            errors[0].message
        );
    }

    // ── AC 5: Empty optional data — no false positives ────────────────────────

    /// Given ParsedData with empty inflow_seasonal_stats, empty
    /// load_seasonal_stats, and empty fpha_hyperplanes, no DimensionMismatch
    /// errors are produced.
    #[test]
    fn test_empty_optional_data_no_false_positives() {
        let mut data = crate::test_support::base_parsed_data(stages(), vec![]);
        data.hydros = vec![make_hydro(1, HydroGenerationModel::Fpha, None, None)];
        data.buses = vec![make_bus(1)];

        let mut ctx = ValidationContext::new();
        validate_dimensional_consistency(&data, &mut ctx);

        assert!(
            !ctx.has_errors(),
            "expected no errors for empty optional data, got: {:?}",
            ctx.errors()
        );
    }

    // ── AC 6: FPHA hydro missing hyperplane rows → 1 error ───────────────────

    /// Given a ParsedData where a hydro uses HydroGenerationModel::Fpha but
    /// fpha_hyperplanes has no rows for that hydro, one DimensionMismatch error
    /// is produced mentioning the hydro ID and "FPHA hyperplanes".
    #[test]
    fn test_fpha_hydro_missing_hyperplane_rows() {
        let mut data = crate::test_support::base_parsed_data(stages(), vec![]);

        data.hydros = vec![
            make_hydro(1, HydroGenerationModel::Fpha, None, None),
            make_hydro(2, HydroGenerationModel::ConstantProductivity, None, None),
        ];

        data.fpha_hyperplanes = vec![fpha_row(2)];

        let mut ctx = ValidationContext::new();
        validate_dimensional_consistency(&data, &mut ctx);

        let errors = ctx.errors();
        assert_eq!(
            errors.len(),
            1,
            "expected exactly 1 error for missing FPHA hyperplanes, got: {:?}",
            errors
        );
        assert_eq!(errors[0].kind, ErrorKind::DimensionMismatch);
        assert!(
            errors[0].message.contains('1'),
            "error should mention hydro ID 1, got: {}",
            errors[0].message
        );
        assert!(
            errors[0]
                .message
                .to_lowercase()
                .contains("fpha hyperplanes"),
            "error should mention 'FPHA hyperplanes', got: {}",
            errors[0].message
        );
    }

    #[test]
    fn zero_capacity_fpha_hydro_needs_no_hyperplane_rows() {
        let mut data = crate::test_support::base_parsed_data(stages(), vec![]);

        let mut capacity_less = make_hydro(1, HydroGenerationModel::Fpha, None, None);
        capacity_less.max_turbined_m3s = 0.0;
        let with_capacity = make_hydro(2, HydroGenerationModel::Fpha, None, None);
        assert!(with_capacity.has_turbine_capacity());
        data.hydros = vec![
            capacity_less,
            with_capacity,
            make_hydro(3, HydroGenerationModel::Fpha, None, None),
        ];

        data.fpha_hyperplanes = vec![fpha_row(3)];

        let mut ctx = ValidationContext::new();
        validate_dimensional_consistency(&data, &mut ctx);

        let errors = ctx.errors();
        assert_eq!(
            errors.len(),
            1,
            "only the plant with turbine capacity needs rows, got: {errors:?}"
        );
        assert_eq!(errors[0].kind, ErrorKind::DimensionMismatch);
        assert!(
            errors[0].message.contains("Hydro 2"),
            "error should name hydro 2, got: {}",
            errors[0].message
        );
    }

    // ── AC 7: Hydro with entry_stage_id skips earlier stages ─────────────────

    /// Given a hydro with entry_stage_id=1, inflow stats are only required
    /// for stage 1, not stage 0. Providing only stage 1 produces no errors.
    #[test]
    fn test_hydro_lifecycle_entry_stage_id_skips_earlier_stages() {
        let mut data = crate::test_support::base_parsed_data(stages(), vec![]);
        data.hydros = vec![make_hydro(
            1,
            HydroGenerationModel::ConstantProductivity,
            Some(1), // enters service at stage 1
            None,
        )];

        data.inflow_seasonal_stats = vec![inflow_stats_row(1, 1)];

        let mut ctx = ValidationContext::new();
        validate_dimensional_consistency(&data, &mut ctx);

        assert!(
            !ctx.has_errors(),
            "expected no errors when stage 0 is before hydro entry, got: {:?}",
            ctx.errors()
        );
    }

    // ── AC 8: Hydro with exit_stage_id skips later stages ────────────────────

    /// Given a hydro with exit_stage_id=1 (exclusive), inflow stats are
    /// only required for stage 0. Providing only stage 0 produces no errors.
    #[test]
    fn test_hydro_lifecycle_exit_stage_id_skips_later_stages() {
        let mut data = crate::test_support::base_parsed_data(stages(), vec![]);
        data.hydros = vec![make_hydro(
            1,
            HydroGenerationModel::ConstantProductivity,
            None,
            Some(1), // decommissioned at stage 1 (exclusive)
        )];

        data.inflow_seasonal_stats = vec![inflow_stats_row(1, 0)];

        let mut ctx = ValidationContext::new();
        validate_dimensional_consistency(&data, &mut ctx);

        assert!(
            !ctx.has_errors(),
            "expected no errors when stage 1 is after hydro exit, got: {:?}",
            ctx.errors()
        );
    }

    // ── AC 9: Pre-study stages are not checked ────────────────────────────────

    /// Pre-study stages (negative IDs) must not appear in the coverage check.
    #[test]
    fn test_pre_study_stages_not_checked() {
        use crate::stages::StagesData;
        let mut data = crate::test_support::base_parsed_data(stages(), vec![]);
        // Include one pre-study stage (id = -1) alongside the study stages.
        data.stages = StagesData {
            openings_declared: std::collections::HashSet::new(),
            stages: vec![make_pre_study_stage(-1), make_stage(0), make_stage(1)],
            policy_graph: HorizonGraph::default(),
        };

        data.hydros = vec![make_hydro(
            1,
            HydroGenerationModel::ConstantProductivity,
            None,
            None,
        )];

        data.inflow_seasonal_stats = vec![inflow_stats_row(1, 0), inflow_stats_row(1, 1)];

        let mut ctx = ValidationContext::new();
        validate_dimensional_consistency(&data, &mut ctx);

        assert!(
            !ctx.has_errors(),
            "pre-study stages should not require inflow stats, got: {:?}",
            ctx.errors()
        );
    }

    // ── AC 10: Load stats coverage missing ────────────────────────────────────

    /// Given a bus that has no load stats for one stage, exactly 1 error is
    /// produced.
    #[test]
    fn test_load_stats_missing_for_one_stage() {
        let mut data = crate::test_support::base_parsed_data(stages(), vec![]);
        data.buses = vec![make_bus(1), make_bus(2)];

        data.load_seasonal_stats = vec![
            load_stats_row(1, 0),
            load_stats_row(1, 1),
            load_stats_row(2, 0), // stage 1 missing for bus 2
        ];

        let mut ctx = ValidationContext::new();
        validate_dimensional_consistency(&data, &mut ctx);

        let errors = ctx.errors();
        assert_eq!(
            errors.len(),
            1,
            "expected 1 error for missing load stats, got: {:?}",
            errors
        );
        assert_eq!(errors[0].kind, ErrorKind::DimensionMismatch);
        assert!(
            errors[0].message.contains("Bus 2"),
            "error should mention 'Bus 2', got: {}",
            errors[0].message
        );
        assert!(
            errors[0].message.contains("stage 1"),
            "error should mention 'stage 1', got: {}",
            errors[0].message
        );
    }

    // ── AC 12: Correlation schedule references non-existent profile ───────────

    /// A schedule entry referencing a profile name that does not exist in
    /// `profiles` produces one DimensionMismatch error.
    #[test]
    fn test_correlation_schedule_missing_profile() {
        let mut data = crate::test_support::base_parsed_data(stages(), vec![]);

        data.correlation = Some(make_correlation_model(
            vec![],
            vec![CorrelationScheduleEntry {
                stage_id: 0,
                profile_name: "nonexistent_profile".to_string(),
            }],
        ));

        let mut ctx = ValidationContext::new();
        validate_dimensional_consistency(&data, &mut ctx);

        let errors = ctx.errors();
        assert!(
            !errors.is_empty(),
            "expected error for missing profile reference"
        );
        assert_eq!(errors[0].kind, ErrorKind::DimensionMismatch);
        assert!(
            errors[0].message.contains("nonexistent_profile"),
            "error should mention the missing profile name, got: {}",
            errors[0].message
        );
    }

    // ── AC 13: LinearizedHead hydro missing geometry rows ────────────────────

    /// A hydro using LinearizedHead model must have at least 2 geometry rows.
    /// Missing rows produce a DimensionMismatch error.
    #[test]
    fn test_linearized_head_hydro_missing_geometry() {
        let mut data = crate::test_support::base_parsed_data(stages(), vec![]);

        data.hydros = vec![make_hydro(
            1,
            HydroGenerationModel::LinearizedHead,
            None,
            None,
        )];

        data.hydro_geometry = vec![geometry_row(1, 100.0)];

        let mut ctx = ValidationContext::new();
        validate_dimensional_consistency(&data, &mut ctx);

        let errors = ctx.errors();
        assert_eq!(
            errors.len(),
            1,
            "expected 1 error for insufficient geometry rows, got: {:?}",
            errors
        );
        assert_eq!(errors[0].kind, ErrorKind::DimensionMismatch);
    }

    /// An FPHA hydro with a SINGLE geometry row is accepted: a run-of-river plant
    /// has one operating volume → a constant forebay → a valid single-volume FPHA
    /// fit. This is the relaxation that `linearized_head` does NOT share.
    #[test]
    fn test_fpha_hydro_single_geometry_row_accepted() {
        let mut data = crate::test_support::base_parsed_data(stages(), vec![]);
        data.hydros = vec![make_hydro(1, HydroGenerationModel::Fpha, None, None)];
        data.hydro_geometry = vec![geometry_row(1, 100.0)];

        let mut ctx = ValidationContext::new();
        validate_dimensional_consistency(&data, &mut ctx);

        assert!(
            ctx.errors().is_empty(),
            "a single geometry row must be valid for an FPHA plant, got: {:?}",
            ctx.errors()
        );
    }

    /// An FPHA hydro with ZERO geometry rows (while other plants have geometry) is
    /// still rejected — there is no forebay to evaluate at all.
    #[test]
    fn test_fpha_hydro_zero_geometry_rows_rejected() {
        let mut data = crate::test_support::base_parsed_data(stages(), vec![]);
        data.hydros = vec![
            make_hydro(1, HydroGenerationModel::Fpha, None, None),
            make_hydro(2, HydroGenerationModel::ConstantProductivity, None, None),
        ];
        data.hydro_geometry = vec![geometry_row(2, 100.0), geometry_row(2, 200.0)];

        let mut ctx = ValidationContext::new();
        validate_dimensional_consistency(&data, &mut ctx);

        assert!(
            ctx.errors().iter().any(|e| {
                e.kind == ErrorKind::DimensionMismatch
                    && e.message.contains("Hydro 1")
                    && e.message.contains("minimum 1 required")
            }),
            "FPHA hydro with zero geometry rows must fail with minimum-1, got: {:?}",
            ctx.errors()
        );
    }

    // ── AC 14: All 7 rules checked independently ─────────────────────────────

    /// Errors in one rule do not suppress checking of subsequent rules.
    #[test]
    fn test_all_rules_checked_independently() {
        let mut data = crate::test_support::base_parsed_data(stages(), vec![]);

        // Trigger rule 1 violation: hydro 1 missing stage 1.
        data.hydros = vec![make_hydro(
            1,
            HydroGenerationModel::ConstantProductivity,
            None,
            None,
        )];
        data.inflow_seasonal_stats = vec![inflow_stats_row(1, 0)]; // missing stage 1

        // Trigger rule 2 violation: bus 1 missing stage 1.
        data.buses = vec![make_bus(1)];
        data.load_seasonal_stats = vec![load_stats_row(1, 0)]; // missing stage 1

        let mut ctx = ValidationContext::new();
        validate_dimensional_consistency(&data, &mut ctx);

        let errors = ctx.errors();
        assert!(
            errors.len() >= 2,
            "expected at least 2 errors (rules 1 and 2), got {}: {:?}",
            errors.len(),
            errors
        );
        assert!(
            errors
                .iter()
                .all(|e| e.kind == ErrorKind::DimensionMismatch),
            "all errors should be DimensionMismatch"
        );
    }

    // ── Cross-crate agreement with the solver crate's resolve_stage/selection_entries ──

    /// `production_model_uses_fpha` agrees with the solver crate's
    /// `resolve_stage`/`selection_entries` classification: a `Seasonal` config
    /// with `default_model == "fpha"` and every listed season non-FPHA still
    /// classifies as using FPHA.
    #[test]
    fn test_production_model_uses_fpha_agrees_with_resolve_stage_on_seasonal_default() {
        let config = ProductionModelConfig {
            hydro_id: EntityId(0),
            selection_mode: SelectionMode::Seasonal {
                default_model: "fpha".to_string(),
                seasons: vec![SeasonConfig {
                    season_id: 1,
                    model: "constant_productivity".to_string(),
                    fpha_config: None,
                    reference_volume: None,
                    productivity_mw_per_m3s: Some(0.5),
                }],
            },
        };

        assert!(
            production_model_uses_fpha(&config),
            "production_model_uses_fpha must agree with the solver crate's \
             resolve_stage/selection_entries: default_model == \"fpha\" classifies \
             as FPHA even when every listed season is non-FPHA"
        );
    }
}
