//! Layer 4 — Dimensional consistency validation.
//!
//! Verifies that every entity which requires companion data actually has it,
//! and that companion data arrays have the expected dimensions. Coverage checks
//! are only performed when the optional data is present (non-empty `Vec` or
//! `Some`).
//!
//! The primary entry point is `validate_dimensional_consistency`.
//!
//! ## Rules implemented
//!
//! | # | Rule | Source File |
//! |---|------|-------------|
//! | 1 | Every active hydro must have an `InflowModel` for every study stage. | `scenarios/inflow_seasonal_stats.parquet` |
//! | 2 | Every bus must have a `LoadModel` for every study stage. | `scenarios/load_seasonal_stats.parquet` |
//! | 3 | For each `CorrelationGroup`, `matrix.len() == entities.len()`. | `scenarios/correlation.json` |
//! | 4 | For each `CorrelationGroup`, every row `matrix[i].len() == entities.len()`. | `scenarios/correlation.json` |
//! | 5 | Every `profile_name` in the correlation schedule exists in `profiles`. | `scenarios/correlation.json` |
//! | 6 | Every FPHA-configured hydro must have at least 1 row in `fpha_hyperplanes`. | `system/fpha_hyperplanes.parquet` |
//! | 7 | Every FPHA- or `LinearizedHead`-configured hydro must have ≥ 2 rows in `hydro_geometry`. | `system/hydro_geometry.parquet` |

use std::collections::{HashMap, HashSet};

use cobre_core::entities::HydroGenerationModel;

use super::{schema::ParsedData, ErrorKind, ValidationContext};
use crate::extensions::{ProductionModelConfig, SelectionMode};

// ── validate_dimensional_consistency ─────────────────────────────────────────

/// Performs Layer 4 dimensional consistency validation on the parsed data.
///
/// Checks all 8 coverage rules. When a rule's optional data is absent (empty
/// `Vec` or `None`), that rule is silently skipped. All 8 rules are always
/// checked — earlier failures do not short-circuit later rules.
///
/// Any coverage failure adds one [`ErrorKind::DimensionMismatch`] entry to
/// `ctx`. This function is infallible — it never returns a `Result`.
///
/// # Arguments
///
/// * `data` — fully parsed case data produced by [`super::schema::validate_schema`].
/// * `ctx`  — mutable validation context that accumulates diagnostics.
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
                if let Some(entry) = hydro.entry_stage_id {
                    if stage_id < entry {
                        continue;
                    }
                }
                if let Some(exit) = hydro.exit_stage_id {
                    if stage_id >= exit {
                        continue;
                    }
                }

                if !inflow_pairs.contains(&(hydro.id.0, stage_id)) {
                    ctx.add_error(
                        ErrorKind::DimensionMismatch,
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
                    ctx.add_error(
                        ErrorKind::DimensionMismatch,
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

                // Rule 4: row count must equal entity count.
                if n_rows != n_entities {
                    ctx.add_error(
                        ErrorKind::DimensionMismatch,
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
                        ctx.add_error(
                            ErrorKind::DimensionMismatch,
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
    }

    if let Some(correlation) = &data.correlation {
        for entry in &correlation.schedule {
            if !correlation.profiles.contains_key(&entry.profile_name) {
                ctx.add_error(
                    ErrorKind::DimensionMismatch,
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

        for &hydro_id in &fpha_hydro_ids {
            if !hydros_with_hyperplanes.contains(&hydro_id) {
                ctx.add_error(
                    ErrorKind::DimensionMismatch,
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

        for &hydro_id in &head_hydro_ids {
            let count = geometry_row_counts.get(&hydro_id).copied().unwrap_or(0);
            if count < 2 {
                ctx.add_error(
                    ErrorKind::DimensionMismatch,
                    "system/hydro_geometry.parquet",
                    Some(format!("Hydro {hydro_id}")),
                    format!(
                        "Hydro {hydro_id} requires head-dependent model but has {count} hydro geometry row(s) (minimum 2 required)"
                    ),
                );
            }
        }
    }
}

// ── Private helpers ───────────────────────────────────────────────────────────

fn collect_fpha_hydro_ids(
    hydros: &[cobre_core::entities::Hydro],
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
    hydros: &[cobre_core::entities::Hydro],
    production_models: &[ProductionModelConfig],
) -> HashSet<i32> {
    let mut ids = HashSet::new();

    for hydro in hydros {
        if matches!(
            hydro.generation_model,
            HydroGenerationModel::Fpha | HydroGenerationModel::LinearizedHead { .. }
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

fn production_model_uses_fpha(config: &ProductionModelConfig) -> bool {
    match &config.selection_mode {
        SelectionMode::StageRanges { ranges } => ranges.iter().any(|r| r.model == "fpha"),
        SelectionMode::Seasonal {
            default_model,
            seasons,
        } => default_model == "fpha" || seasons.iter().any(|s| s.model == "fpha"),
    }
}

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
        entities::{Bus, HydroGenerationModel, HydroPenalties},
        scenario::{
            CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile,
            CorrelationScheduleEntry,
        },
        temporal::{
            Block, BlockMode, NoiseMethod, PolicyGraph, ScenarioSourceConfig, Stage,
            StageRiskConfig, StageStateConfig,
        },
        EntityId,
    };

    use crate::{
        extensions::{FphaHyperplaneRow, HydroGeometryRow},
        scenarios::{InflowSeasonalStatsRow, LoadSeasonalStatsRow},
        validation::{ErrorKind, ValidationContext},
    };

    use super::*;

    fn make_hydro(
        id: i32,
        generation_model: HydroGenerationModel,
        entry_stage_id: Option<i32>,
        exit_stage_id: Option<i32>,
    ) -> cobre_core::entities::Hydro {
        cobre_core::entities::Hydro {
            id: EntityId(id),
            name: format!("Hydro {id}"),
            bus_id: EntityId(1),
            downstream_id: None,
            entry_stage_id,
            exit_stage_id,
            min_storage_hm3: 0.0,
            max_storage_hm3: 1000.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 100.0,
            min_generation_mw: 0.0,
            max_generation_mw: 500.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            diversion: None,
            filling: None,
            penalties: penalties_default(),
        }
    }

    fn make_bus(id: i32) -> Bus {
        Bus {
            id: EntityId(id),
            name: format!("Bus {id}"),
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

    fn penalties_default() -> HydroPenalties {
        HydroPenalties {
            spillage_cost: 1.0,
            diversion_cost: 1.0,
            fpha_turbined_cost: 1.0,
            storage_violation_below_cost: 1.0,
            filling_target_violation_cost: 1.0,
            turbined_violation_below_cost: 1.0,
            outflow_violation_below_cost: 1.0,
            outflow_violation_above_cost: 1.0,
            generation_violation_below_cost: 1.0,
            evaporation_violation_cost: 1.0,
            water_withdrawal_violation_cost: 1.0,
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
            method: "cholesky".to_string(),
            profiles,
            schedule,
        }
    }

    // Minimal ParsedData builder — only sets the fields needed for each test.
    fn base_parsed_data() -> ParsedData {
        use crate::{
            config::{
                Config, CutSelectionConfig, ExportsConfig, ModelingConfig, PolicyConfig,
                SimulationConfig, StoppingRuleConfig, TrainingConfig, TrainingSolverConfig,
                UpperBoundEvaluationConfig,
            },
            stages::StagesData,
        };
        use cobre_core::{
            entities::DeficitSegment, initial_conditions::InitialConditions,
            penalty::GlobalPenaltyDefaults, scenario::ScenarioSource,
        };

        let config = Config {
            schema: None,
            version: None,
            modeling: ModelingConfig::default(),
            training: TrainingConfig {
                enabled: true,
                seed: None,
                forward_passes: Some(10),
                stopping_rules: Some(vec![StoppingRuleConfig::IterationLimit { limit: 100 }]),
                stopping_mode: "any".to_string(),
                cut_formulation: None,
                forward_pass: None,
                cut_selection: CutSelectionConfig::default(),
                solver: TrainingSolverConfig::default(),
            },
            upper_bound_evaluation: UpperBoundEvaluationConfig::default(),
            policy: PolicyConfig::default(),
            simulation: SimulationConfig::default(),
            exports: ExportsConfig::default(),
        };

        ParsedData {
            config,
            penalties: GlobalPenaltyDefaults {
                bus_deficit_segments: vec![DeficitSegment {
                    depth_mw: None,
                    cost_per_mwh: 1.0,
                }],
                bus_excess_cost: 1.0,
                line_exchange_cost: 1.0,
                hydro: HydroPenalties {
                    spillage_cost: 1.0,
                    diversion_cost: 1.0,
                    fpha_turbined_cost: 1.0,
                    storage_violation_below_cost: 1.0,
                    filling_target_violation_cost: 1.0,
                    turbined_violation_below_cost: 1.0,
                    outflow_violation_below_cost: 1.0,
                    outflow_violation_above_cost: 1.0,
                    generation_violation_below_cost: 1.0,
                    evaporation_violation_cost: 1.0,
                    water_withdrawal_violation_cost: 1.0,
                },
                ncs_curtailment_cost: 1.0,
            },
            stages: StagesData {
                stages: vec![make_stage(0), make_stage(1)],
                policy_graph: PolicyGraph::default(),
                scenario_source: ScenarioSource::default(),
            },
            initial_conditions: InitialConditions {
                storage: vec![],
                filling_storage: vec![],
            },
            buses: vec![],
            thermals: vec![],
            hydros: vec![],
            lines: vec![],
            non_controllable_sources: vec![],
            pumping_stations: vec![],
            energy_contracts: vec![],
            hydro_geometry: vec![],
            production_models: vec![],
            fpha_hyperplanes: vec![],
            inflow_history: vec![],
            inflow_seasonal_stats: vec![],
            inflow_ar_coefficients: vec![],
            external_scenarios: vec![],
            load_seasonal_stats: vec![],
            load_factors: vec![],
            correlation: None,
            thermal_bounds: vec![],
            hydro_bounds: vec![],
            line_bounds: vec![],
            pumping_bounds: vec![],
            contract_bounds: vec![],
            exchange_factors: vec![],
            generic_constraints: vec![],
            generic_constraint_bounds: vec![],
            penalty_overrides_bus: vec![],
            penalty_overrides_line: vec![],
            penalty_overrides_hydro: vec![],
            penalty_overrides_ncs: vec![],
        }
    }

    // ── AC 1: Valid coverage — no errors ──────────────────────────────────────

    /// Given ParsedData where every hydro has InflowModel entries for every
    /// study stage and all other coverage rules are satisfied,
    /// `validate_dimensional_consistency` produces no errors.
    #[test]
    fn test_valid_coverage_no_errors() {
        let mut data = base_parsed_data();

        // 2 hydros, 2 study stages — all 4 (hydro, stage) pairs present.
        data.hydros = vec![
            make_hydro(
                1,
                HydroGenerationModel::ConstantProductivity {
                    productivity_mw_per_m3s: 0.9,
                },
                None,
                None,
            ),
            make_hydro(
                2,
                HydroGenerationModel::ConstantProductivity {
                    productivity_mw_per_m3s: 0.9,
                },
                None,
                None,
            ),
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
        let mut data = base_parsed_data();

        data.hydros = vec![
            make_hydro(
                1,
                HydroGenerationModel::ConstantProductivity {
                    productivity_mw_per_m3s: 0.9,
                },
                None,
                None,
            ),
            make_hydro(
                2,
                HydroGenerationModel::ConstantProductivity {
                    productivity_mw_per_m3s: 0.9,
                },
                None,
                None,
            ),
            make_hydro(
                3,
                HydroGenerationModel::ConstantProductivity {
                    productivity_mw_per_m3s: 0.9,
                },
                None,
                None,
            ),
        ];

        // Hydro 2 is missing stage 1.
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
        let mut data = base_parsed_data();

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
        let mut data = base_parsed_data();

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
        let mut data = base_parsed_data();
        data.hydros = vec![make_hydro(1, HydroGenerationModel::Fpha, None, None)];
        data.buses = vec![make_bus(1)];
        // All optional data is empty — no rules 1, 2, 3, 7, 8 should fire.
        // data.inflow_seasonal_stats == vec![] → skip rule 1
        // data.load_seasonal_stats == vec![] → skip rule 3
        // data.fpha_hyperplanes == vec![] → skip rule 7
        // data.hydro_geometry == vec![] → skip rule 8

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
        let mut data = base_parsed_data();

        data.hydros = vec![
            make_hydro(1, HydroGenerationModel::Fpha, None, None),
            make_hydro(
                2,
                HydroGenerationModel::ConstantProductivity {
                    productivity_mw_per_m3s: 0.9,
                },
                None,
                None,
            ),
        ];

        // Only hydro 2 has a hyperplane row — hydro 1 (FPHA) is missing.
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

    // ── AC 7: Hydro with entry_stage_id skips earlier stages ─────────────────

    /// Given a hydro with entry_stage_id=1, inflow stats are only required
    /// for stage 1, not stage 0. Providing only stage 1 produces no errors.
    #[test]
    fn test_hydro_lifecycle_entry_stage_id_skips_earlier_stages() {
        let mut data = base_parsed_data();
        // Stages 0 and 1 are study stages.
        // Hydro enters service at stage 1: only stage 1 requires inflow stats.
        data.hydros = vec![make_hydro(
            1,
            HydroGenerationModel::ConstantProductivity {
                productivity_mw_per_m3s: 0.9,
            },
            Some(1), // enters service at stage 1
            None,
        )];

        // Only provide inflow stats for stage 1.
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
        let mut data = base_parsed_data();
        // Stages 0 and 1 are study stages.
        // Hydro exits at stage 1 (exclusive): only stage 0 requires inflow stats.
        data.hydros = vec![make_hydro(
            1,
            HydroGenerationModel::ConstantProductivity {
                productivity_mw_per_m3s: 0.9,
            },
            None,
            Some(1), // decommissioned at stage 1 (exclusive)
        )];

        // Only provide inflow stats for stage 0.
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
        use cobre_core::scenario::ScenarioSource;

        let mut data = base_parsed_data();
        // Include one pre-study stage (id = -1) alongside the study stages.
        data.stages = StagesData {
            stages: vec![make_pre_study_stage(-1), make_stage(0), make_stage(1)],
            policy_graph: PolicyGraph::default(),
            scenario_source: ScenarioSource::default(),
        };

        data.hydros = vec![make_hydro(
            1,
            HydroGenerationModel::ConstantProductivity {
                productivity_mw_per_m3s: 0.9,
            },
            None,
            None,
        )];

        // Provide inflow stats for study stages 0 and 1 only — not pre-study.
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
        let mut data = base_parsed_data();
        data.buses = vec![make_bus(1), make_bus(2)];

        // Bus 2 is missing stage 1.
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

    // ── AC 11 (renumbered 12): Correlation schedule references non-existent profile ───────────

    /// A schedule entry referencing a profile name that does not exist in
    /// `profiles` produces one DimensionMismatch error.
    #[test]
    fn test_correlation_schedule_missing_profile() {
        let mut data = base_parsed_data();

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
        let mut data = base_parsed_data();

        data.hydros = vec![make_hydro(
            1,
            HydroGenerationModel::LinearizedHead {
                productivity_mw_per_m3s: 0.9,
            },
            None,
            None,
        )];

        // Only 1 geometry row — minimum is 2.
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

    // ── AC 14: All 8 rules checked independently ─────────────────────────────

    /// Errors in one rule do not suppress checking of subsequent rules.
    #[test]
    fn test_all_rules_checked_independently() {
        let mut data = base_parsed_data();

        // Trigger rule 1 violation: hydro 1 missing stage 1.
        data.hydros = vec![make_hydro(
            1,
            HydroGenerationModel::ConstantProductivity {
                productivity_mw_per_m3s: 0.9,
            },
            None,
            None,
        )];
        data.inflow_seasonal_stats = vec![inflow_stats_row(1, 0)]; // missing stage 1

        // Trigger rule 3 violation: bus 1 missing stage 1.
        data.buses = vec![make_bus(1)];
        data.load_seasonal_stats = vec![load_stats_row(1, 0)]; // missing stage 1

        let mut ctx = ValidationContext::new();
        validate_dimensional_consistency(&data, &mut ctx);

        // Both rules fire — at least 2 errors.
        let errors = ctx.errors();
        assert!(
            errors.len() >= 2,
            "expected at least 2 errors (rules 1 and 3), got {}: {:?}",
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
}
